from __future__ import annotations

import glob
import json
import os

import pytest

import memory


@pytest.fixture
def mem_path(tmp_path):
    return str(tmp_path / "memory.json")


class TestChatIsolation:
    def test_messages_isolated_per_chat(self, mem_path):
        memory.append_messages(mem_path, chat_id=1, user_text="hi", assistant_text="hello", limit=20)
        memory.append_messages(mem_path, chat_id=2, user_text="yo", assistant_text="hey", limit=20)

        chat1 = memory.get_messages(mem_path, 1)
        chat2 = memory.get_messages(mem_path, 2)

        assert [m["content"] for m in chat1] == ["hi", "hello"]
        assert [m["content"] for m in chat2] == ["yo", "hey"]

    def test_mode_isolated_per_chat(self, mem_path):
        memory.set_mode(mem_path, chat_id=1, mode="idea_lab")
        memory.set_mode(mem_path, chat_id=2, mode="learning_coach")

        assert memory.get_mode(mem_path, 1, "daily_planner") == "idea_lab"
        assert memory.get_mode(mem_path, 2, "daily_planner") == "learning_coach"

    def test_reset_only_affects_target_chat(self, mem_path):
        memory.append_messages(mem_path, chat_id=1, user_text="a", assistant_text="b", limit=20)
        memory.append_messages(mem_path, chat_id=2, user_text="c", assistant_text="d", limit=20)

        memory.reset_messages(mem_path, chat_id=1)

        assert memory.get_messages(mem_path, 1) == []
        assert len(memory.get_messages(mem_path, 2)) == 2


class TestResetKeepsMode:
    def test_reset_messages_preserves_mode(self, mem_path):
        memory.set_mode(mem_path, chat_id=1, mode="idea_lab")
        memory.append_messages(mem_path, chat_id=1, user_text="a", assistant_text="b", limit=20)

        memory.reset_messages(mem_path, chat_id=1)

        assert memory.get_mode(mem_path, 1, "daily_planner") == "idea_lab"
        assert memory.get_messages(mem_path, 1) == []


class TestLimitEnforcement:
    def test_configured_limit_is_enforced(self, mem_path):
        limit = 4  # 2 individual messages per exchange => keep last 2 exchanges
        for i in range(5):
            memory.append_messages(
                mem_path, chat_id=1, user_text=f"u{i}", assistant_text=f"a{i}", limit=limit,
            )

        messages = memory.get_messages(mem_path, 1)
        assert len(messages) == limit
        assert [m["content"] for m in messages] == ["u3", "a3", "u4", "a4"]

    def test_history_never_starts_with_orphan_assistant_message(self, mem_path):
        for i in range(10):
            memory.append_messages(
                mem_path, chat_id=1, user_text=f"u{i}", assistant_text=f"a{i}", limit=6,
            )

        messages = memory.get_messages(mem_path, 1)
        assert messages[0]["role"] == "user"


class TestCorruptJson:
    def test_corrupt_json_is_quarantined_not_destroyed(self, mem_path):
        with open(mem_path, "w", encoding="utf-8") as fh:
            fh.write("{not valid json!!")

        result = memory.get_messages(mem_path, 1)

        assert result == []
        quarantined = glob.glob(mem_path + ".corrupt.*")
        assert len(quarantined) == 1
        with open(quarantined[0], encoding="utf-8") as fh:
            assert fh.read() == "{not valid json!!"
        # Original path no longer holds the damaged content.
        assert not os.path.exists(mem_path)

    def test_wrong_top_level_type_is_quarantined(self, mem_path):
        with open(mem_path, "w", encoding="utf-8") as fh:
            json.dump([1, 2, 3], fh)  # memory.json root must be an object, not an array

        result = memory.get_messages(mem_path, 1)

        assert result == []
        assert len(glob.glob(mem_path + ".corrupt.*")) == 1

    def test_writing_after_quarantine_produces_fresh_valid_file(self, mem_path):
        with open(mem_path, "w", encoding="utf-8") as fh:
            fh.write("garbage")

        memory.append_messages(mem_path, chat_id=1, user_text="hi", assistant_text="hello", limit=20)

        with open(mem_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["1"]["messages"][0]["content"] == "hi"


class TestAtomicWrite:
    def test_write_produces_valid_parseable_json(self, mem_path):
        memory.append_messages(mem_path, chat_id=1, user_text="hi", assistant_text="hello", limit=20)

        with open(mem_path, encoding="utf-8") as fh:
            data = json.load(fh)  # must not raise
        assert data["1"]["messages"][0]["content"] == "hi"

    def test_no_leftover_temp_files_after_write(self, mem_path):
        memory.append_messages(mem_path, chat_id=1, user_text="hi", assistant_text="hello", limit=20)

        directory = os.path.dirname(mem_path)
        leftovers = [f for f in os.listdir(directory) if f.startswith(".tmp-")]
        assert leftovers == []
