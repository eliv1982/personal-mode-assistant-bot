from __future__ import annotations

import json

import pytest

from prompt_manager import PromptConfigError, PromptManager

VALID_DATA = {
    "format_suffix": "Plain text only.",
    "modes": {
        "daily_planner": {
            "label": "Planner",
            "description": "Plans your day.",
            "system_prompt": "You are a planner.",
        },
        "idea_lab": {
            "label": "Ideas",
            "description": "Generates ideas.",
            "system_prompt": "You are an idea generator.",
        },
    },
    "default_mode": "daily_planner",
}


def _write(tmp_path, data) -> str:
    path = str(tmp_path / "prompts.json")
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(data, str):
            fh.write(data)
        else:
            json.dump(data, fh)
    return path


class TestValidConfigLoads:
    def test_valid_configuration_loads(self, tmp_path):
        pm = PromptManager(_write(tmp_path, VALID_DATA))
        assert pm.default_mode == "daily_planner"
        assert set(pm.mode_keys()) == {"daily_planner", "idea_lab"}
        assert "planner" in pm.get_system_prompt("daily_planner").lower()

    def test_format_suffix_appended_to_system_prompt(self, tmp_path):
        pm = PromptManager(_write(tmp_path, VALID_DATA))
        assert pm.get_system_prompt("daily_planner").endswith("Plain text only.")

    def test_missing_format_suffix_defaults_to_empty(self, tmp_path):
        data = dict(VALID_DATA)
        del data["format_suffix"]
        pm = PromptManager(_write(tmp_path, data))
        assert pm.format_suffix == ""


class TestMissingOrMalformedFile:
    def test_missing_file_fails_startup(self, tmp_path):
        with pytest.raises(PromptConfigError):
            PromptManager(str(tmp_path / "does_not_exist.json"))

    def test_malformed_json_fails_startup(self, tmp_path):
        with pytest.raises(PromptConfigError):
            PromptManager(_write(tmp_path, "{not valid json!!"))

    def test_non_object_top_level_fails_startup(self, tmp_path):
        with pytest.raises(PromptConfigError):
            PromptManager(_write(tmp_path, [1, 2, 3]))


class TestMissingRequiredStructure:
    def test_absent_modes_key_fails_startup(self, tmp_path):
        data = {"default_mode": "x"}
        with pytest.raises(PromptConfigError):
            PromptManager(_write(tmp_path, data))

    def test_empty_modes_fails_startup(self, tmp_path):
        data = {"modes": {}, "default_mode": "daily_planner"}
        with pytest.raises(PromptConfigError):
            PromptManager(_write(tmp_path, data))

    def test_mode_missing_system_prompt_fails_startup(self, tmp_path):
        data = {
            "modes": {"daily_planner": {"label": "Planner"}},
            "default_mode": "daily_planner",
        }
        with pytest.raises(PromptConfigError):
            PromptManager(_write(tmp_path, data))

    def test_mode_with_blank_system_prompt_fails_startup(self, tmp_path):
        data = {
            "modes": {"daily_planner": {"system_prompt": "   "}},
            "default_mode": "daily_planner",
        }
        with pytest.raises(PromptConfigError):
            PromptManager(_write(tmp_path, data))


class TestDefaultModeValidation:
    def test_default_mode_not_in_modes_fails_startup(self, tmp_path):
        data = {
            "modes": {"daily_planner": {"system_prompt": "You are a planner."}},
            "default_mode": "nonexistent_mode",
        }
        with pytest.raises(PromptConfigError):
            PromptManager(_write(tmp_path, data))

    def test_missing_default_mode_key_fails_startup(self, tmp_path):
        data = {"modes": {"daily_planner": {"system_prompt": "You are a planner."}}}
        with pytest.raises(PromptConfigError):
            PromptManager(_write(tmp_path, data))
