from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PromptManager:
    def __init__(self, prompts_file: str) -> None:
        self._file = prompts_file
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            with open(self._file, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.error("Failed to load prompts file '%s': %s", self._file, exc)
            return {"modes": {}, "default_mode": "daily_planner", "format_suffix": ""}

    @property
    def modes(self) -> dict[str, dict[str, str]]:
        return self._data.get("modes", {})

    @property
    def default_mode(self) -> str:
        return self._data.get("default_mode", "daily_planner")

    @property
    def format_suffix(self) -> str:
        return self._data.get("format_suffix", "")

    def get_system_prompt(self, mode_key: str) -> str:
        mode = self.modes.get(mode_key)
        if mode is None:
            logger.warning("Unknown mode '%s', falling back to default.", mode_key)
            mode = self.modes.get(self.default_mode, {})
        base = mode.get("system_prompt", "Ты — полезный ассистент. Отвечай по-русски.")
        suffix = self.format_suffix
        if suffix:
            return f"{base}\n\n{suffix}"
        return base

    def get_label(self, mode_key: str) -> str:
        return self.modes.get(mode_key, {}).get("label", mode_key)

    def get_description(self, mode_key: str) -> str:
        return self.modes.get(mode_key, {}).get("description", "")

    def mode_keys(self) -> list[str]:
        return list(self.modes.keys())
