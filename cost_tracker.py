from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostReport:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_usd: float
    total_rub: float
    usd_rub_rate: float
    rate_date: str
    rate_is_fallback: bool


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
    usd_rub_rate: float,
    rate_date: str,
    rate_is_fallback: bool,
) -> CostReport:
    input_cost = input_tokens / 1_000_000 * input_price_per_1m
    output_cost = output_tokens / 1_000_000 * output_price_per_1m
    total_usd = input_cost + output_cost
    total_rub = total_usd * usd_rub_rate

    return CostReport(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        total_usd=total_usd,
        total_rub=total_rub,
        usd_rub_rate=usd_rub_rate,
        rate_date=rate_date,
        rate_is_fallback=rate_is_fallback,
    )
