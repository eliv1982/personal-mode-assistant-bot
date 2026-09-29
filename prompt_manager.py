from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PromptConfigError(RuntimeError):
    """Raised when the prompts file is missing, malformed, or fails validation.

    Intentionally not caught anywhere -- a broken prompt configuration must
    fail bot startup loudly rather than let the bot run with no usable mode.
    """


class PromptManager:
    def __init__(self, prompts_file: str) -> None:
        self._file = prompts_file
        self._data: dict[str, Any] = self._load()
        self._validate()

    def _load(self) -> dict[str, Any]:
        try:
            with open(self._file, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError as exc:
            raise PromptConfigError(f"Prompts file not found: '{self._file}'.") from exc
        except json.JSONDecodeError as exc:
            raise PromptConfigError(f"Prompts file '{self._file}' is not valid JSON: {exc}") from exc

    def _validate(self) -> None:
        if not isinstance(self._data, dict):
            raise PromptConfigError(
                f"Prompts file '{self._file}' must contain a JSON object at the top level."
            )

        modes = self._data.get("modes")
        if not isinstance(modes, dict) or not modes:
            raise PromptConfigError(
                f"Prompts file '{self._file}' must define at least one mode under 'modes'."
            )

        for mode_key, mode in modes.items():
            system_prompt = mode.get("system_prompt") if isinstance(mode, dict) else None
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                raise PromptConfigError(
                    f"Mode '{mode_key}' in '{self._file}' is missing a non-empty 'system_prompt'."
                )

        default_mode = self._data.get("default_mode")
        if not isinstance(default_mode, str) or default_mode not in modes:
            raise PromptConfigError(
                f"'default_mode' in '{self._file}' must reference one of the defined modes: "
                f"{sorted(modes)}."
            )

    @property
    def modes(self) -> dict[str, dict[str, str]]:
        return self._data["modes"]

    @property
    def default_mode(self) -> str:
        return self._data["default_mode"]

    @property
    def format_suffix(self) -> str:
        return self._data.get("format_suffix", "")

    def get_system_prompt(self, mode_key: str) -> str:
        mode = self.modes.get(mode_key)
        if mode is None:
            logger.warning("Unknown mode '%s', falling back to default.", mode_key)
            mode = self.modes[self.default_mode]
        base = mode["system_prompt"]
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
