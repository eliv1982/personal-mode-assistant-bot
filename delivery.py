from __future__ import annotations

from typing import Any, Awaitable, Callable

from text_utils import split_into_chunks

# Telegram's hard limit on a single message's text length.
TELEGRAM_MESSAGE_LIMIT = 4096

EMPTY_REPLY_FALLBACK = "🤖 Модель не вернула содержательный ответ. Попробуйте переформулировать запрос."

SendFunc = Callable[..., Awaitable[Any]]


async def send_llm_reply(
    send: SendFunc,
    text: str,
    reply_markup: Any = None,
) -> bool:
    """Deliver LLM-generated `text` as plain-text Telegram message(s).

    - Always sent with `parse_mode=None`, since LLM output is user-controlled
      and must never be interpreted as HTML/Markdown by Telegram.
    - Longer than Telegram's message limit -> split into ordered chunks that
      together reproduce the full text.
    - `reply_markup` (the request-details keyboard) is attached only to the
      last chunk sent.
    - Empty/blank `text` -> a short static fallback is sent instead of an
      (invalid) empty Telegram message.

    `send` is an async callable compatible with aiogram's
    `Message.answer(text, *, parse_mode=None, reply_markup=...)`.

    Returns True if `text` itself was delivered, False if the fallback was
    sent in its place (so callers can avoid recording undelivered content
    as if the user had received it).
    """
    if text and text.strip():
        chunks = split_into_chunks(text, TELEGRAM_MESSAGE_LIMIT)
        delivered = True
    else:
        chunks = [EMPTY_REPLY_FALLBACK]
        delivered = False

    last_index = len(chunks) - 1
    for i, chunk in enumerate(chunks):
        await send(
            chunk,
            parse_mode=None,
            reply_markup=reply_markup if i == last_index else None,
        )
    return delivered
