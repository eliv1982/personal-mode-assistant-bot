from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODE = "daily_planner"


def _empty_record(mode: str = _DEFAULT_MODE) -> dict[str, Any]:
    return {"mode": mode, "messages": []}


def _load_all(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Memory file root must be a JSON object.")
        return data
    except Exception as exc:
        logger.error("Failed to read memory file '%s' (%s: %s). Starting with empty memory.", path, type(exc).__name__, exc)
        return {}


def _save_all(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("Failed to write memory file '%s' (%s: %s).", path, type(exc).__name__, exc)


def get_mode(path: str, chat_id: int | str, default_mode: str = _DEFAULT_MODE) -> str:
    data = _load_all(path)
    return data.get(str(chat_id), {}).get("mode", default_mode)


def set_mode(path: str, chat_id: int | str, mode: str) -> None:
    data = _load_all(path)
    key = str(chat_id)
    record = data.get(key, _empty_record(mode))
    record["mode"] = mode
    data[key] = record
    _save_all(path, data)


def get_messages(path: str, chat_id: int | str) -> list[dict[str, str]]:
    data = _load_all(path)
    return list(data.get(str(chat_id), {}).get("messages", []))


def append_messages(
    path: str,
    chat_id: int | str,
    user_text: str,
    assistant_text: str,
    limit: int,
) -> None:
    data = _load_all(path)
    key = str(chat_id)
    record = data.get(key, _empty_record())
    messages: list[dict[str, str]] = record.get("messages", [])
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": assistant_text})
    # Keep only the last `limit` messages (pairs count toward the limit individually)
    record["messages"] = messages[-limit:]
    data[key] = record
    _save_all(path, data)


def reset_messages(path: str, chat_id: int | str) -> None:
    data = _load_all(path)
    key = str(chat_id)
    record = data.get(key, _empty_record())
    record["messages"] = []
    data[key] = record
    _save_all(path, data)
