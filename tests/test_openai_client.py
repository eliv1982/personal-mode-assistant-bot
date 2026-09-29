from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError

from openai_client import OpenAIClient, OpenAIUserError, _REQUEST_TIMEOUT_SECONDS


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _fake_status_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code=status_code, request=_fake_request())


class _UnforeseenRateLimitSubclass(RateLimitError):
    """Stand-in for a hypothetical future SDK subclass of RateLimitError.

    Regression guard for the old exact-type `dict[type(exc)]` lookup, which
    raised an unhandled KeyError for any exception not identical in type to
    one of the dict's keys -- even a direct subclass of one.
    """


def _client() -> OpenAIClient:
    return OpenAIClient(api_key="test-key", model="gpt-4o-mini")


async def _run_chat_expecting_user_error(monkeypatch, raise_exc=None, fake_response=None) -> OpenAIUserError:
    client = _client()

    async def fake_create(*args, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        return fake_response

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    with pytest.raises(OpenAIUserError) as exc_info:
        await client.chat(system_prompt="sys", history=[], user_message="hi")
    return exc_info.value


class TestSubclassSafeErrorMapping:
    @pytest.mark.asyncio
    async def test_rate_limit_error_maps_to_friendly_message(self, monkeypatch):
        exc = RateLimitError("rate limited", response=_fake_status_response(429), body=None)
        err = await _run_chat_expecting_user_error(monkeypatch, raise_exc=exc)
        assert "лимит" in str(err).lower()

    @pytest.mark.asyncio
    async def test_subclass_of_rate_limit_error_is_still_handled(self, monkeypatch):
        # This is the exact scenario the old `_USER_ERROR_MESSAGES[type(exc)]`
        # dict lookup got wrong: isinstance(exc, RateLimitError) is True, but
        # type(exc) is not RateLimitError, so the old code raised KeyError.
        exc = _UnforeseenRateLimitSubclass("rate limited", response=_fake_status_response(429), body=None)
        err = await _run_chat_expecting_user_error(monkeypatch, raise_exc=exc)
        assert "лимит" in str(err).lower()

    @pytest.mark.asyncio
    async def test_timeout_error_maps_to_friendly_message(self, monkeypatch):
        exc = APITimeoutError(request=_fake_request())
        err = await _run_chat_expecting_user_error(monkeypatch, raise_exc=exc)
        assert "не ответил" in str(err).lower()

    @pytest.mark.asyncio
    async def test_connection_error_maps_to_friendly_message(self, monkeypatch):
        exc = APIConnectionError(message="boom", request=_fake_request())
        err = await _run_chat_expecting_user_error(monkeypatch, raise_exc=exc)
        assert "подключиться" in str(err).lower()

    @pytest.mark.asyncio
    async def test_generic_api_error_does_not_leak_raw_sdk_text(self, monkeypatch):
        secret_detail = "internal-trace-id-abc123-do-not-leak"
        exc = RateLimitError(secret_detail, response=_fake_status_response(500), body=None)
        # Force it through the generic-error branch by using a status error
        # subtype not covered by name-specific messages (500 InternalServerError).
        from openai import InternalServerError

        exc = InternalServerError(secret_detail, response=_fake_status_response(500), body=None)
        err = await _run_chat_expecting_user_error(monkeypatch, raise_exc=exc)
        assert secret_detail not in str(err)


class TestMalformedResponseHandling:
    @pytest.mark.asyncio
    async def test_empty_choices_does_not_raise_raw_index_error(self, monkeypatch):
        fake_response = SimpleNamespace(choices=[], usage=None)
        err = await _run_chat_expecting_user_error(monkeypatch, fake_response=fake_response)
        assert isinstance(err, OpenAIUserError)
        assert "IndexError" not in str(err)


class TestExplicitTimeout:
    def test_client_configured_with_finite_explicit_timeout(self):
        client = _client()
        assert client._client.timeout == _REQUEST_TIMEOUT_SECONDS
        assert _REQUEST_TIMEOUT_SECONDS > 0
        assert _REQUEST_TIMEOUT_SECONDS < 600  # strictly tighter than the SDK's 600s default
