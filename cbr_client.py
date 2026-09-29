from __future__ import annotations

import logging
import math
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import aiohttp

logger = logging.getLogger(__name__)

CBR_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
_TIMEOUT = aiohttp.ClientTimeout(total=5)

# How long a successfully fetched rate stays valid before a refresh is
# attempted again. The rate only changes once a day on CBR's side, so an
# hour keeps the bot responsive without hammering the endpoint.
_CACHE_TTL_SECONDS = 3600


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%d.%m.%Y")


class CbrParseError(ValueError):
    """Raised when the CBR XML response cannot be parsed into a USD/RUB rate."""


def parse_usd_rub_rate(content: bytes) -> tuple[float, str]:
    """Pure parse of a CBR ``XML_daily.asp`` response body.

    Returns (rate, rate_date). Raises CbrParseError on malformed XML, a
    missing USD entry, or an invalid/non-numeric Value or Nominal.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise CbrParseError(f"Malformed CBR XML: {exc}") from exc

    rate_date = root.get("Date", _today_str())

    for valute in root.findall("Valute"):
        if valute.findtext("CharCode") != "USD":
            continue

        value_str = valute.findtext("Value", "")
        nominal_str = valute.findtext("Nominal", "1")
        try:
            value = float(value_str.replace(",", "."))
            nominal = float(nominal_str.replace(",", "."))
        except ValueError as exc:
            raise CbrParseError(f"Invalid numeric value in CBR USD entry: {exc}") from exc

        if not math.isfinite(value) or value <= 0:
            raise CbrParseError(f"Invalid CBR USD Value: {value_str!r}")
        if not math.isfinite(nominal) or nominal <= 0:
            raise CbrParseError(f"Invalid CBR USD Nominal: {nominal_str!r}")

        rate = value / nominal
        if not math.isfinite(rate) or rate <= 0:
            raise CbrParseError(f"Invalid calculated CBR USD/RUB rate: {rate!r}")

        return rate, rate_date

    raise CbrParseError("USD entry not found in CBR response")


class _RateCache:
    """Process-local, in-memory cache of the last successfully fetched rate.

    Holds a single (rate, rate_date) pair -- there is only ever one "current"
    USD/RUB rate for this bot. `fresh()` is None once the TTL has elapsed,
    but `last_good()` keeps returning the last successful value indefinitely
    so a temporary CBR outage can still be bridged with real (if stale) data.
    """

    def __init__(self, ttl_seconds: float = _CACHE_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._rate: float | None = None
        self._rate_date: str | None = None
        self._fetched_at: float | None = None

    def fresh(self) -> tuple[float, str] | None:
        if self._rate is None or self._rate_date is None or self._fetched_at is None:
            return None
        if time.monotonic() - self._fetched_at > self.ttl_seconds:
            return None
        return self._rate, self._rate_date

    def last_good(self) -> tuple[float, str] | None:
        if self._rate is None or self._rate_date is None:
            return None
        return self._rate, self._rate_date

    def store(self, rate: float, rate_date: str) -> None:
        self._rate = rate
        self._rate_date = rate_date
        self._fetched_at = time.monotonic()

    def reset(self) -> None:
        self._rate = None
        self._rate_date = None
        self._fetched_at = None


_cache = _RateCache()


async def _fetch_raw() -> bytes:
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        async with session.get(CBR_URL) as response:
            response.raise_for_status()
            return await response.read()


async def get_usd_rub_rate(fallback: float) -> tuple[float, bool, str]:
    """Return (rate, is_fallback, rate_date).

    - A cached rate younger than ~1h is returned without any network call.
    - Otherwise a live CBR fetch is attempted; on success the cache is
      refreshed and the fresh value returned.
    - If the fetch fails but a previously successful value is cached (even if
      stale), that last known good rate is returned as real CBR data
      (is_fallback=False) rather than falling through to the configured
      fallback.
    - Only when no successful value has ever been obtained does this fall
      back to the configured `fallback` value, with is_fallback=True and
      today's date -- matching prior behavior.
    """
    cached = _cache.fresh()
    if cached is not None:
        rate, rate_date = cached
        logger.debug("Using cached CBR USD/RUB rate: %.4f (date: %s)", rate, rate_date)
        return rate, False, rate_date

    try:
        content = await _fetch_raw()
        rate, rate_date = parse_usd_rub_rate(content)
    except Exception as exc:
        last_good = _cache.last_good()
        if last_good is not None:
            rate, rate_date = last_good
            logger.warning(
                "Failed to refresh CBR rate (%s: %s); using last known good rate %.4f (date: %s)",
                type(exc).__name__, exc, rate, rate_date,
            )
            return rate, False, rate_date

        logger.warning(
            "Failed to fetch CBR rate (%s: %s); using fallback %.2f",
            type(exc).__name__, exc, fallback,
        )
        return fallback, True, _today_str()

    logger.debug("CBR USD/RUB rate fetched: %.4f (date: %s)", rate, rate_date)
    _cache.store(rate, rate_date)
    return rate, False, rate_date
