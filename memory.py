from __future__ import annotations

import logging
from typing import Any

from json_store import atomic_write_json, load_json_with_quarantine

logger = logging.getLogger(__name__)

_DEFAULT_MODE = "daily_planner"


def _empty_record(mode: str = _DEFAULT_MODE) -> dict[str, Any]:
    return {"mode": mode, "messages": []}


def _load_all(path: str) -> dict[str, Any]:
    return load_json_with_quarantine(path, dict, {})


def _save_all(path: str, data: dict[str, Any]) -> None:
    atomic_write_json(path, data)


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
    # `limit` counts individual messages, not user/assistant pairs.
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
