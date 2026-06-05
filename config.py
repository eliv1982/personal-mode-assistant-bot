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


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    openai_api_key: str
    openai_model: str
    input_price_per_1m: float
    output_price_per_1m: float
    memory_limit: int
    usd_rub_fallback: float
    data_dir: str = field(default="data")
    memory_file: str = field(default="data/memory.json")
    usage_file: str = field(default="data/usage.json")
    prompts_file: str = field(default="prompts.json")


def load_config() -> Config:
    return Config(
        telegram_bot_token=_require("TELEGRAM_BOT_TOKEN"),
        openai_api_key=_require("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        input_price_per_1m=_float_env("OPENAI_INPUT_PRICE_PER_1M", 0.15),
        output_price_per_1m=_float_env("OPENAI_OUTPUT_PRICE_PER_1M", 0.60),
        memory_limit=_int_env("MEMORY_LIMIT", 20),
        usd_rub_fallback=_float_env("USD_RUB_FALLBACK", 100.0),
    )
