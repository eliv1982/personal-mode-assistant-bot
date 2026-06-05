from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import aiohttp

logger = logging.getLogger(__name__)

CBR_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
_TIMEOUT = aiohttp.ClientTimeout(total=5)


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%d.%m.%Y")


async def get_usd_rub_rate(fallback: float) -> tuple[float, bool, str]:
    """Return (rate, is_fallback, rate_date).

    rate_date is the date string from the CBR XML (e.g. "05.06.2026").
    On failure returns (fallback, True, today's date).
    """
    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            async with session.get(CBR_URL) as response:
                response.raise_for_status()
                content = await response.read()

        root = ET.fromstring(content)
        rate_date = root.get("Date", _today_str())

        for valute in root.findall("Valute"):
            char_code = valute.findtext("CharCode")
            if char_code == "USD":
                value_str = valute.findtext("Value", "")
                nominal_str = valute.findtext("Nominal", "1")
                rate = float(value_str.replace(",", ".")) / float(nominal_str.replace(",", "."))
                logger.debug("CBR USD/RUB rate fetched: %.4f (date: %s)", rate, rate_date)
                return rate, False, rate_date

        logger.warning("USD not found in CBR response; using fallback rate %.2f", fallback)
        return fallback, True, _today_str()

    except Exception as exc:
        logger.warning(
            "Failed to fetch CBR rate (%s: %s); using fallback %.2f",
            type(exc).__name__, exc, fallback,
        )
        return fallback, True, _today_str()
