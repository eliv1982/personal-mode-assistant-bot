from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from json_store import atomic_write_json, load_json_with_quarantine

logger = logging.getLogger(__name__)


def _new_request_id() -> str:
    return uuid4().hex[:10]


def _load_all(path: str) -> list[dict[str, Any]]:
    return load_json_with_quarantine(path, list, [])


def _save_all(path: str, records: list[dict[str, Any]]) -> None:
    atomic_write_json(path, records)


def save_usage(
    path: str,
    chat_id: int | str,
    mode: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    total_usd: float,
    usd_rub_rate: float,
    rate_date: str,
    total_rub: float,
    is_fallback_rate: bool,
) -> str:
    request_id = _new_request_id()
    record: dict[str, Any] = {
        "id": request_id,
        "chat_id": str(chat_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_usd": total_usd,
        "usd_rub_rate": usd_rub_rate,
        "rate_date": rate_date,
        "total_rub": total_rub,
        "is_fallback_rate": is_fallback_rate,
    }
    records = _load_all(path)
    records.append(record)
    _save_all(path, records)
    logger.debug("Saved usage record id=%s for chat_id=%s", request_id, chat_id)
    return request_id


def get_usage_record(path: str, request_id: str, chat_id: int | str) -> dict[str, Any] | None:
    """Return the usage record for `request_id`, scoped to `chat_id`.

    A record is returned only if it also belongs to `chat_id`, so a
    guessed/leaked request id from another chat cannot be used to read
    someone else's usage details.
    """
    for record in _load_all(path):
        if record.get("id") == request_id and record.get("chat_id") == str(chat_id):
            return record
    return None


def get_stats(path: str, chat_id: int | str) -> dict[str, Any]:
    records = _load_all(path)
    chat_records = [r for r in records if r.get("chat_id") == str(chat_id)]

    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")
    month_str = now.strftime("%Y-%m")

    counters: dict[str, int | float] = {
        "total_requests": 0, "today_requests": 0, "month_requests": 0,
        "total_tokens": 0,   "today_tokens": 0,   "month_tokens": 0,
        "total_usd": 0.0,    "today_usd": 0.0,    "month_usd": 0.0,
    }

    for r in chat_records:
        created = r.get("created_at", "")
        tokens = r.get("total_tokens", 0)
        usd = r.get("total_usd", 0.0)

        counters["total_requests"] += 1
        counters["total_tokens"] += tokens
        counters["total_usd"] += usd

        if created.startswith(today_str):
            counters["today_requests"] += 1
            counters["today_tokens"] += tokens
            counters["today_usd"] += usd

        if created.startswith(month_str):
            counters["month_requests"] += 1
            counters["month_tokens"] += tokens
            counters["month_usd"] += usd

    return counters


def reset_stats(path: str, chat_id: int | str) -> int:
    records = _load_all(path)
    kept = [r for r in records if r.get("chat_id") != str(chat_id)]
    removed = len(records) - len(kept)
    _save_all(path, kept)
    return removed
