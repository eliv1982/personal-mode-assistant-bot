from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Required environment variable '{key}' is not set. Check your .env file.")
    return value


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise EnvironmentError(f"Environment variable '{key}' must be a float, got: {raw!r}")


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise EnvironmentError(f"Environment variable '{key}' must be an integer, got: {raw!r}")


def _non_empty_str_env(key: str, default: str) -> str:
    raw = os.getenv(key)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        raise EnvironmentError(f"Environment variable '{key}' must not be empty.")
    return value


def _require_non_negative(key: str, value: float) -> float:
    if value < 0:
        raise EnvironmentError(f"Environment variable '{key}' must not be negative (got {value}).")
    return value


def _require_positive(key: str, value: float) -> float:
    if value <= 0:
        raise EnvironmentError(f"Environment variable '{key}' must be a positive number (got {value}).")
    return value


def _validate_memory_limit(value: int) -> int:
    """MEMORY_LIMIT = maximum number of individual messages retained per chat.

    Must be an even number >= 2, so retained history always starts with a
    user message and never begins with an orphan assistant reply.
    """
    if value < 2:
        raise EnvironmentError(
            f"MEMORY_LIMIT must be at least 2 (got {value}). "
            "It is the number of individual messages retained per chat, not pairs."
        )
    if value % 2 != 0:
        raise EnvironmentError(
            f"MEMORY_LIMIT must be an even number (got {value}), "
            "so retained history cannot start with an orphan assistant message."
        )
    return value


def _parse_allowed_chat_ids(key: str = "ALLOWED_CHAT_IDS") -> frozenset[int] | None:
    """Parse a comma-separated allowlist of Telegram chat ids.

    Returns None when unset/blank, meaning unrestricted access (current
    default behavior). Any non-integer entry fails configuration loading
    clearly rather than being silently ignored.
    """
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return None

    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            raise EnvironmentError(
                f"Environment variable '{key}' contains an invalid chat id: {part!r}. "
                "Expected a comma-separated list of integers."
            )
    return frozenset(ids) if ids else None


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    openai_api_key: str
    openai_model: str
    input_price_per_1m: float
    output_price_per_1m: float
    memory_limit: int
    usd_rub_fallback: float
    allowed_chat_ids: frozenset[int] | None = field(default=None)
    data_dir: str = field(default="data")
    memory_file: str = field(default="data/memory.json")
    usage_file: str = field(default="data/usage.json")
    prompts_file: str = field(default="prompts.json")


def load_config() -> Config:
    return Config(
        telegram_bot_token=_require("TELEGRAM_BOT_TOKEN"),
        openai_api_key=_require("OPENAI_API_KEY"),
        openai_model=_non_empty_str_env("OPENAI_MODEL", "gpt-4o-mini"),
        input_price_per_1m=_require_non_negative(
            "OPENAI_INPUT_PRICE_PER_1M", _float_env("OPENAI_INPUT_PRICE_PER_1M", 0.15)
        ),
        output_price_per_1m=_require_non_negative(
            "OPENAI_OUTPUT_PRICE_PER_1M", _float_env("OPENAI_OUTPUT_PRICE_PER_1M", 0.60)
        ),
        memory_limit=_validate_memory_limit(_int_env("MEMORY_LIMIT", 20)),
        usd_rub_fallback=_require_positive("USD_RUB_FALLBACK", _float_env("USD_RUB_FALLBACK", 100.0)),
        allowed_chat_ids=_parse_allowed_chat_ids(),
    )
