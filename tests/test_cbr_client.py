from __future__ import annotations

import pytest

import cbr_client
from cbr_client import CbrParseError, get_usd_rub_rate, parse_usd_rub_rate

SAMPLE_XML = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b'<ValCurs Date="29.09.2026" name="Foreign Currency Market">\n'
    b'<Valute ID="R01235">'
    b"<NumCode>840</NumCode>"
    b"<CharCode>USD</CharCode>"
    b"<Nominal>1</Nominal>"
    b"<Name>US Dollar</Name>"
    b"<Value>95,1234</Value>"
    b"</Valute>\n"
    b"</ValCurs>"
)


def _with_usd_field(old: str, new: str) -> bytes:
    text = SAMPLE_XML.decode("utf-8")
    assert old in text
    return text.replace(old, new).encode("utf-8")


@pytest.fixture(autouse=True)
def _reset_cache():
    """Every test starts and ends with a clean process-local cache."""
    cbr_client._cache.reset()
    yield
    cbr_client._cache.reset()


class TestParseUsdRubRate:
    def test_normal_usd_entry(self):
        rate, rate_date = parse_usd_rub_rate(SAMPLE_XML)
        assert rate == pytest.approx(95.1234)
        assert rate_date == "29.09.2026"

    def test_comma_decimal_format(self):
        # CBR always uses a comma as the decimal separator; this pins that
        # the parser actually normalises it rather than accidentally
        # relying on float() tolerating commas.
        rate, _ = parse_usd_rub_rate(SAMPLE_XML)
        assert rate == pytest.approx(95.1234)

    def test_nominal_greater_than_one(self):
        xml = _with_usd_field("<Nominal>1</Nominal>", "<Nominal>10</Nominal>")
        xml = xml.replace(b"95,1234", b"951,234")
        rate, _ = parse_usd_rub_rate(xml)
        assert rate == pytest.approx(95.1234)

    def test_missing_usd_raises(self):
        xml = _with_usd_field("<CharCode>USD</CharCode>", "<CharCode>EUR</CharCode>")
        with pytest.raises(CbrParseError):
            parse_usd_rub_rate(xml)

    def test_malformed_xml_raises(self):
        with pytest.raises(CbrParseError):
            parse_usd_rub_rate(b"<not-valid-xml")

    def test_invalid_numeric_value_raises(self):
        xml = _with_usd_field("<Value>95,1234</Value>", "<Value>not-a-number</Value>")
        with pytest.raises(CbrParseError):
            parse_usd_rub_rate(xml)


class TestParseUsdRubRateInvalidNumerics:
    """Regression tests: a parseable-but-invalid Value/Nominal must raise
    rather than produce a rate that could be cached as trusted CBR data."""

    @pytest.mark.parametrize(
        "bad_value", ["0", "-95,1234", "-95.1234", "NaN", "inf", "-inf", "Infinity"]
    )
    def test_invalid_value_raises(self, bad_value):
        xml = _with_usd_field("<Value>95,1234</Value>", f"<Value>{bad_value}</Value>")
        with pytest.raises(CbrParseError):
            parse_usd_rub_rate(xml)

    @pytest.mark.parametrize("bad_nominal", ["0", "-1", "NaN", "inf", "Infinity"])
    def test_invalid_nominal_raises(self, bad_nominal):
        xml = _with_usd_field("<Nominal>1</Nominal>", f"<Nominal>{bad_nominal}</Nominal>")
        with pytest.raises(CbrParseError):
            parse_usd_rub_rate(xml)

    def test_calculated_rate_must_be_finite(self):
        # Value and Nominal are each individually finite and positive, but
        # the division overflows to +inf -- the rate-level check must catch
        # what the per-field checks cannot.
        xml = _with_usd_field("<Value>95,1234</Value>", "<Value>1e308</Value>")
        xml = xml.replace(b"<Nominal>1</Nominal>", b"<Nominal>1e-300</Nominal>")
        with pytest.raises(CbrParseError):
            parse_usd_rub_rate(xml)


class TestRateCacheAndFallback:
    """These tests monkeypatch `_fetch_raw`, the only network seam, so no
    test in this class makes a real HTTP call."""

    async def test_fresh_fetch_populates_cache_and_returns_real_rate(self, monkeypatch):
        calls = 0

        async def fake_fetch() -> bytes:
            nonlocal calls
            calls += 1
            return SAMPLE_XML

        monkeypatch.setattr(cbr_client, "_fetch_raw", fake_fetch)

        rate, is_fallback, rate_date = await get_usd_rub_rate(fallback=100.0)

        assert rate == pytest.approx(95.1234)
        assert is_fallback is False
        assert rate_date == "29.09.2026"
        assert calls == 1

    async def test_cache_hit_avoids_second_fetch(self, monkeypatch):
        calls = 0

        async def fake_fetch() -> bytes:
            nonlocal calls
            calls += 1
            return SAMPLE_XML

        monkeypatch.setattr(cbr_client, "_fetch_raw", fake_fetch)

        first = await get_usd_rub_rate(fallback=100.0)
        second = await get_usd_rub_rate(fallback=100.0)

        assert first == second
        assert calls == 1  # second call served entirely from cache

    async def test_expired_cache_triggers_refresh(self, monkeypatch):
        calls = 0

        async def fake_fetch() -> bytes:
            nonlocal calls
            calls += 1
            return SAMPLE_XML

        monkeypatch.setattr(cbr_client, "_fetch_raw", fake_fetch)

        fake_now = [1_000.0]
        monkeypatch.setattr(cbr_client.time, "monotonic", lambda: fake_now[0])

        await get_usd_rub_rate(fallback=100.0)
        fake_now[0] += cbr_client._CACHE_TTL_SECONDS + 1
        await get_usd_rub_rate(fallback=100.0)

        assert calls == 2

    async def test_refresh_failure_uses_last_good_cached_value(self, monkeypatch):
        attempt = 0

        async def fake_fetch() -> bytes:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                return SAMPLE_XML
            raise ConnectionError("network down")

        monkeypatch.setattr(cbr_client, "_fetch_raw", fake_fetch)

        fake_now = [1_000.0]
        monkeypatch.setattr(cbr_client.time, "monotonic", lambda: fake_now[0])

        rate1, is_fallback1, date1 = await get_usd_rub_rate(fallback=100.0)
        fake_now[0] += cbr_client._CACHE_TTL_SECONDS + 1
        rate2, is_fallback2, date2 = await get_usd_rub_rate(fallback=100.0)

        assert is_fallback1 is False
        assert rate2 == rate1
        assert date2 == date1
        # Stale-but-real CBR data is not the same thing as the configured
        # fallback, so is_fallback must stay False here.
        assert is_fallback2 is False
        assert attempt == 2

    async def test_initial_failure_uses_configured_fallback(self, monkeypatch):
        async def fake_fetch() -> bytes:
            raise ConnectionError("network down")

        monkeypatch.setattr(cbr_client, "_fetch_raw", fake_fetch)

        rate, is_fallback, rate_date = await get_usd_rub_rate(fallback=77.5)

        assert rate == 77.5
        assert is_fallback is True

    async def test_invalid_refresh_response_keeps_last_good_cached_rate(self, monkeypatch):
        """A parseable-but-invalid refresh response (e.g. Value=0) must be
        treated as a refresh failure, not silently replace the existing
        valid cached rate."""
        attempt = 0

        async def fake_fetch() -> bytes:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                return SAMPLE_XML
            return _with_usd_field("<Value>95,1234</Value>", "<Value>0</Value>")

        monkeypatch.setattr(cbr_client, "_fetch_raw", fake_fetch)

        fake_now = [1_000.0]
        monkeypatch.setattr(cbr_client.time, "monotonic", lambda: fake_now[0])

        rate1, is_fallback1, date1 = await get_usd_rub_rate(fallback=100.0)
        fake_now[0] += cbr_client._CACHE_TTL_SECONDS + 1
        rate2, is_fallback2, date2 = await get_usd_rub_rate(fallback=100.0)

        assert is_fallback1 is False
        assert rate2 == rate1
        assert date2 == date1
        assert is_fallback2 is False
        assert attempt == 2

    async def test_initial_invalid_response_uses_fallback_and_is_not_cached(self, monkeypatch):
        """An invalid response with no prior good rate must fall through to
        the configured fallback, and that fallback must never itself become
        the cached "last good" rate."""

        async def fake_fetch() -> bytes:
            return _with_usd_field("<Value>95,1234</Value>", "<Value>NaN</Value>")

        monkeypatch.setattr(cbr_client, "_fetch_raw", fake_fetch)

        rate, is_fallback, rate_date = await get_usd_rub_rate(fallback=77.5)

        assert rate == 77.5
        assert is_fallback is True
        assert cbr_client._cache.last_good() is None
