from __future__ import annotations

import pytest

from delivery import EMPTY_REPLY_FALLBACK, TELEGRAM_MESSAGE_LIMIT, send_llm_reply


class FakeSender:
    """Records calls the way aiogram's Message.answer(...) would receive them."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, text, parse_mode=..., reply_markup=None):
        self.calls.append({"text": text, "parse_mode": parse_mode, "reply_markup": reply_markup})
        return object()


@pytest.mark.asyncio
async def test_short_reply_sends_one_message():
    sender = FakeSender()
    delivered = await send_llm_reply(sender, "Hello, world!", reply_markup="kb")

    assert delivered is True
    assert len(sender.calls) == 1
    assert sender.calls[0]["text"] == "Hello, world!"
    assert sender.calls[0]["reply_markup"] == "kb"


@pytest.mark.asyncio
async def test_long_reply_splits_into_multiple_ordered_messages():
    sender = FakeSender()
    long_text = "word " * 2000  # exceeds TELEGRAM_MESSAGE_LIMIT

    delivered = await send_llm_reply(sender, long_text, reply_markup="kb")

    assert delivered is True
    assert len(sender.calls) > 1
    assert all(len(call["text"]) <= TELEGRAM_MESSAGE_LIMIT for call in sender.calls)
    reconstructed = "".join(call["text"] for call in sender.calls)
    assert reconstructed == long_text


@pytest.mark.asyncio
async def test_keyboard_attached_only_to_final_chunk():
    sender = FakeSender()
    long_text = "word " * 2000

    await send_llm_reply(sender, long_text, reply_markup="kb")

    assert len(sender.calls) > 1
    for call in sender.calls[:-1]:
        assert call["reply_markup"] is None
    assert sender.calls[-1]["reply_markup"] == "kb"


@pytest.mark.asyncio
async def test_empty_reply_sends_fallback():
    sender = FakeSender()
    delivered = await send_llm_reply(sender, "", reply_markup="kb")

    assert delivered is False
    assert len(sender.calls) == 1
    assert sender.calls[0]["text"] == EMPTY_REPLY_FALLBACK


@pytest.mark.asyncio
async def test_blank_whitespace_reply_sends_fallback():
    sender = FakeSender()
    delivered = await send_llm_reply(sender, "   \n\t  ", reply_markup=None)

    assert delivered is False
    assert sender.calls[0]["text"] == EMPTY_REPLY_FALLBACK


@pytest.mark.asyncio
async def test_llm_text_always_sent_with_parse_mode_none():
    sender = FakeSender()
    await send_llm_reply(sender, "<b>x</b> a < b, x & y <div>", reply_markup=None)

    assert len(sender.calls) == 1
    assert sender.calls[0]["parse_mode"] is None
    # Content must reach the sender literally -- no HTML-escaping performed here.
    assert sender.calls[0]["text"] == "<b>x</b> a < b, x & y <div>"


@pytest.mark.asyncio
async def test_fallback_also_sent_with_parse_mode_none():
    sender = FakeSender()
    await send_llm_reply(sender, "", reply_markup=None)

    assert sender.calls[0]["parse_mode"] is None
