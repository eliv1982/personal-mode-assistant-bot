from __future__ import annotations

import glob
import json
import os

import pytest

import usage_tracker


@pytest.fixture
def usage_path(tmp_path):
    return str(tmp_path / "usage.json")


def _save(usage_path, chat_id):
    return usage_tracker.save_usage(
        path=usage_path,
        chat_id=chat_id,
        mode="daily_planner",
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        total_usd=0.001,
        usd_rub_rate=100.0,
        rate_date="01.01.2026",
        total_rub=0.1,
        is_fallback_rate=False,
    )


class TestPerChatOwnership:
    def test_lookup_cannot_return_another_chats_request(self, usage_path):
        request_id = _save(usage_path, chat_id=111)

        assert usage_tracker.get_usage_record(usage_path, request_id, chat_id=111) is not None
        assert usage_tracker.get_usage_record(usage_path, request_id, chat_id=222) is None

    def test_lookup_with_correct_chat_returns_record(self, usage_path):
        request_id = _save(usage_path, chat_id=111)

        record = usage_tracker.get_usage_record(usage_path, request_id, chat_id=111)

        assert record is not None
        assert record["id"] == request_id
        assert record["chat_id"] == "111"

    def test_unknown_request_id_returns_none(self, usage_path):
        _save(usage_path, chat_id=111)

        assert usage_tracker.get_usage_record(usage_path, "doesnotexist", chat_id=111) is None

    def test_stats_scoped_per_chat(self, usage_path):
        _save(usage_path, chat_id=1)
        _save(usage_path, chat_id=1)
        _save(usage_path, chat_id=2)

        stats1 = usage_tracker.get_stats(usage_path, 1)
        stats2 = usage_tracker.get_stats(usage_path, 2)

        assert stats1["total_requests"] == 2
        assert stats2["total_requests"] == 1


class TestCorruptJson:
    def test_corrupt_json_quarantined_not_destroyed(self, usage_path):
        with open(usage_path, "w", encoding="utf-8") as fh:
            fh.write("not json at all")

        record = usage_tracker.get_usage_record(usage_path, "any", chat_id=1)

        assert record is None
        quarantined = glob.glob(usage_path + ".corrupt.*")
        assert len(quarantined) == 1
        with open(quarantined[0], encoding="utf-8") as fh:
            assert fh.read() == "not json at all"
        assert not os.path.exists(usage_path)

    def test_wrong_top_level_type_quarantined(self, usage_path):
        with open(usage_path, "w", encoding="utf-8") as fh:
            json.dump({"not": "a list"}, fh)  # usage.json root must be an array

        record = usage_tracker.get_usage_record(usage_path, "any", chat_id=1)

        assert record is None
        assert len(glob.glob(usage_path + ".corrupt.*")) == 1

    def test_save_after_quarantine_produces_fresh_valid_file(self, usage_path):
        with open(usage_path, "w", encoding="utf-8") as fh:
            fh.write("garbage")

        _save(usage_path, chat_id=1)

        with open(usage_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == 1


class TestAtomicWrite:
    def test_write_produces_valid_parseable_json(self, usage_path):
        _save(usage_path, chat_id=1)

        with open(usage_path, encoding="utf-8") as fh:
            data = json.load(fh)  # must not raise
        assert isinstance(data, list)
        assert len(data) == 1

    def test_no_leftover_temp_files_after_write(self, usage_path):
        _save(usage_path, chat_id=1)

        directory = os.path.dirname(usage_path)
        leftovers = [f for f in os.listdir(directory) if f.startswith(".tmp-")]
        assert leftovers == []
