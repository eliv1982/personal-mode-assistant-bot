from __future__ import annotations

import json
import logging
import os

import pytest

import main
import memory
import usage_tracker
from config import Config
from json_store import JsonWriteError
from openai_client import LLMResponse

pytestmark = pytest.mark.asyncio


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeSentMessage:
    """Duck-types the Message returned by aiogram's Message.answer(...)."""

    def __init__(self, text, parse_mode, reply_markup) -> None:
        self.text = text
        self.parse_mode = parse_mode
        self.reply_markup = reply_markup
        self.deleted = False

    async def delete(self) -> None:
        self.deleted = True


class FakeChat:
    def __init__(self, chat_id) -> None:
        self.id = chat_id


class FakeMessage:
    """Duck-types aiogram's Message enough for process_text_message()."""

    def __init__(self, chat_id, text) -> None:
        self.chat = FakeChat(chat_id)
        self.text = text
        self.sent: list[FakeSentMessage] = []

    async def answer(self, text, parse_mode=None, reply_markup=None) -> FakeSentMessage:
        sent = FakeSentMessage(text, parse_mode, reply_markup)
        self.sent.append(sent)
        return sent


class FakeOpenAI:
    def __init__(self, response: LLMResponse | None = None, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc
        self.calls: list[dict] = []

    async def chat(self, system_prompt, history, user_message) -> LLMResponse:
        self.calls.append({
            "system_prompt": system_prompt,
            "history": history,
            "user_message": user_message,
        })
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response


class FakePromptManager:
    default_mode = "daily_planner"

    def get_system_prompt(self, mode_key: str) -> str:
        return "SYSTEM PROMPT"


async def _fake_rate(fallback: float):
    return (100.0, False, "01.01.2026")


def _make_cfg(tmp_path, allowed_chat_ids=None) -> Config:
    return Config(
        telegram_bot_token="test-token",
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        memory_limit=20,
        usd_rub_fallback=100.0,
        allowed_chat_ids=allowed_chat_ids,
        data_dir=str(tmp_path),
        memory_file=str(tmp_path / "memory.json"),
        usage_file=str(tmp_path / "usage.json"),
        prompts_file=str(tmp_path / "prompts.json"),
    )


def _log() -> logging.Logger:
    return logging.getLogger("test_handle_text")


# ── A. Normal successful request ────────────────────────────────────────────

async def test_normal_successful_request(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "get_usd_rub_rate", _fake_rate)

    cfg = _make_cfg(tmp_path)
    pm = FakePromptManager()
    response = LLMResponse(text="Привет!", input_tokens=10, output_tokens=5, total_tokens=15)
    oai = FakeOpenAI(response=response)
    message = FakeMessage(chat_id=1, text="hi")

    await main.process_text_message(message, pm, oai, cfg, _log())

    assert len(oai.calls) == 1

    # thinking placeholder sent first, then cleaned up
    assert message.sent[0].text == "⏳ Думаю…"
    assert message.sent[0].deleted is True

    # reply delivered with the request-details keyboard
    reply = message.sent[-1]
    assert reply.text == "Привет!"
    assert reply.parse_mode is None
    assert reply.reply_markup is not None

    # usage persisted
    with open(cfg.usage_file, encoding="utf-8") as fh:
        records = json.load(fh)
    assert len(records) == 1

    # assistant response appended to memory only after delivery
    msgs = memory.get_messages(cfg.memory_file, 1)
    assert [m["content"] for m in msgs] == ["hi", "Привет!"]


# ── B. Usage persistence failure ────────────────────────────────────────────

async def test_usage_persistence_failure_still_delivers_answer(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(main, "get_usd_rub_rate", _fake_rate)

    def _raise_write_error(*args, **kwargs):
        raise JsonWriteError("disk full")

    monkeypatch.setattr(usage_tracker, "save_usage", _raise_write_error)

    cfg = _make_cfg(tmp_path)
    pm = FakePromptManager()
    response = LLMResponse(text="Ответ готов", input_tokens=10, output_tokens=5, total_tokens=15)
    oai = FakeOpenAI(response=response)
    message = FakeMessage(chat_id=1, text="hi")

    assert len(oai.calls) == 0  # sanity: not called yet

    with caplog.at_level(logging.ERROR):
        await main.process_text_message(message, pm, oai, cfg, _log())

    # OpenAI succeeded despite the downstream persistence failure
    assert len(oai.calls) == 1

    # error is contained -- handler does not raise, no usage.json was written
    assert not os.path.exists(cfg.usage_file)

    # the answer is still delivered, without a request-details keyboard
    reply = message.sent[-1]
    assert reply.text == "Ответ готов"
    assert reply.reply_markup is None

    # thinking placeholder cleaned up
    assert message.sent[0].deleted is True

    # memory appended after successful delivery of the real content
    msgs = memory.get_messages(cfg.memory_file, 1)
    assert [m["content"] for m in msgs] == ["hi", "Ответ готов"]

    # the failure is logged clearly ...
    assert any("persist" in rec.message.lower() for rec in caplog.records)
    # ... but the raw exception text never reaches Telegram
    assert all("disk full" not in (s.text or "") for s in message.sent)


# ── C. Telegram delivery failure ────────────────────────────────────────────

async def test_delivery_failure_leaves_memory_unappended(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "get_usd_rub_rate", _fake_rate)

    async def _raise_delivery_error(send, text, reply_markup=None):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(main, "send_llm_reply", _raise_delivery_error)

    cfg = _make_cfg(tmp_path)
    pm = FakePromptManager()
    response = LLMResponse(text="Ответ", input_tokens=10, output_tokens=5, total_tokens=15)
    oai = FakeOpenAI(response=response)
    message = FakeMessage(chat_id=1, text="hi")

    with pytest.raises(RuntimeError, match="telegram down"):
        await main.process_text_message(message, pm, oai, cfg, _log())

    # usage/cost was attempted and persisted before delivery was attempted
    with open(cfg.usage_file, encoding="utf-8") as fh:
        records = json.load(fh)
    assert len(records) == 1

    # assistant content was never appended -- delivery never completed
    assert memory.get_messages(cfg.memory_file, 1) == []

    # thinking placeholder still cleaned up despite the raised exception
    assert message.sent[0].deleted is True


# ── D. Memory-after-delivery contract ───────────────────────────────────────

async def test_memory_appended_only_after_delivery_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "get_usd_rub_rate", _fake_rate)

    order: list[str] = []
    real_send_llm_reply = main.send_llm_reply

    async def _tracking_send(send, text, reply_markup=None):
        order.append("delivery")
        return await real_send_llm_reply(send, text, reply_markup=reply_markup)

    monkeypatch.setattr(main, "send_llm_reply", _tracking_send)

    real_append = memory.append_messages

    def _tracking_append(*args, **kwargs):
        order.append("memory_append")
        return real_append(*args, **kwargs)

    monkeypatch.setattr(memory, "append_messages", _tracking_append)

    cfg = _make_cfg(tmp_path)
    pm = FakePromptManager()
    response = LLMResponse(text="Ответ", input_tokens=10, output_tokens=5, total_tokens=15)
    oai = FakeOpenAI(response=response)
    message = FakeMessage(chat_id=1, text="hi")

    await main.process_text_message(message, pm, oai, cfg, _log())

    assert order == ["delivery", "memory_append"]


# ── E. Empty OpenAI content ──────────────────────────────────────────────────

async def test_empty_openai_content_sends_fallback_not_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "get_usd_rub_rate", _fake_rate)

    cfg = _make_cfg(tmp_path)
    pm = FakePromptManager()
    response = LLMResponse(text="", input_tokens=5, output_tokens=0, total_tokens=5)
    oai = FakeOpenAI(response=response)
    message = FakeMessage(chat_id=1, text="hi")

    await main.process_text_message(message, pm, oai, cfg, _log())

    from delivery import EMPTY_REPLY_FALLBACK

    reply = message.sent[-1]
    assert reply.text == EMPTY_REPLY_FALLBACK

    # the fallback is not stored as assistant conversation history
    assert memory.get_messages(cfg.memory_file, 1) == []

    # usage accounting still happened -- the OpenAI call itself succeeded
    with open(cfg.usage_file, encoding="utf-8") as fh:
        records = json.load(fh)
    assert len(records) == 1

    assert message.sent[0].deleted is True


# ── F. Allowlist before paid work ───────────────────────────────────────────

async def test_unauthorized_chat_rejected_before_openai(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "get_usd_rub_rate", _fake_rate)

    cfg = _make_cfg(tmp_path, allowed_chat_ids=frozenset({999}))
    pm = FakePromptManager()
    oai = FakeOpenAI(response=LLMResponse(text="unused", input_tokens=1, output_tokens=1, total_tokens=2))
    message = FakeMessage(chat_id=1, text="hi")  # not in allowlist

    await main.process_text_message(message, pm, oai, cfg, _log())

    assert oai.calls == []  # OpenAI never called
    assert not os.path.exists(cfg.usage_file)  # usage persistence never called
    assert memory.get_messages(cfg.memory_file, 1) == []

    assert len(message.sent) == 1
    assert message.sent[0].text == main._ACCESS_DENIED_TEXT


# ── G. Plain-text delivery wiring ───────────────────────────────────────────

async def test_routes_through_safe_delivery_helper(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "get_usd_rub_rate", _fake_rate)

    calls: list[dict] = []

    async def _spy_send_llm_reply(send, text, reply_markup=None):
        calls.append({"text": text, "reply_markup": reply_markup})
        await send(text, parse_mode=None, reply_markup=reply_markup)
        return True

    monkeypatch.setattr(main, "send_llm_reply", _spy_send_llm_reply)

    cfg = _make_cfg(tmp_path)
    pm = FakePromptManager()
    response = LLMResponse(text="**bold** text", input_tokens=1, output_tokens=1, total_tokens=2)
    oai = FakeOpenAI(response=response)
    message = FakeMessage(chat_id=1, text="hi")

    await main.process_text_message(message, pm, oai, cfg, _log())

    # the handler routes (sanitized) LLM output through the delivery helper,
    # not directly through message.answer(..., parse_mode=HTML)
    assert len(calls) == 1
    assert calls[0]["text"] == "bold text"


# ── H. Thinking-message cleanup does not mask a propagating error ──────────

async def test_thinking_delete_failure_does_not_mask_propagating_error(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "get_usd_rub_rate", _fake_rate)

    async def _raise_delivery_error(send, text, reply_markup=None):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(main, "send_llm_reply", _raise_delivery_error)

    cfg = _make_cfg(tmp_path)
    pm = FakePromptManager()
    response = LLMResponse(text="Ответ", input_tokens=10, output_tokens=5, total_tokens=15)
    oai = FakeOpenAI(response=response)
    message = FakeMessage(chat_id=1, text="hi")

    original_answer = message.answer

    async def _answer_with_failing_delete(text, parse_mode=None, reply_markup=None):
        sent = await original_answer(text, parse_mode=parse_mode, reply_markup=reply_markup)
        if len(message.sent) == 1:  # the "thinking" placeholder

            async def _failing_delete():
                raise RuntimeError("delete failed too")

            sent.delete = _failing_delete
        return sent

    message.answer = _answer_with_failing_delete

    # the original delivery error propagates -- not masked by the delete failure
    with pytest.raises(RuntimeError, match="telegram down"):
        await main.process_text_message(message, pm, oai, cfg, _log())
