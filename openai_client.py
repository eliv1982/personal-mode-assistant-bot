from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)

_USER_ERROR_MESSAGES = {
    RateLimitError: "⚠️ Превышен лимит запросов к OpenAI. Подождите немного и попробуйте снова.",
    APITimeoutError: "⏱️ OpenAI не ответил вовремя. Попробуйте ещё раз.",
    APIConnectionError: "🌐 Не удалось подключиться к OpenAI. Проверьте интернет-соединение.",
}
_GENERIC_ERROR = "😔 Что-то пошло не так при обращении к AI. Попробуйте позже."


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class OpenAIClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def chat(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        user_message: str,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
            )
        except tuple(_USER_ERROR_MESSAGES.keys()) as exc:
            friendly = _USER_ERROR_MESSAGES[type(exc)]
            logger.error("OpenAI %s: %s", type(exc).__name__, exc)
            raise OpenAIUserError(friendly) from exc
        except APIError as exc:
            logger.error("OpenAI APIError (status=%s): %s", getattr(exc, "status_code", "?"), exc)
            raise OpenAIUserError(_GENERIC_ERROR) from exc
        except Exception as exc:
            logger.error("Unexpected error calling OpenAI: %s", exc, exc_info=True)
            raise OpenAIUserError(_GENERIC_ERROR) from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )


class OpenAIUserError(Exception):
    """Raised when an OpenAI call fails; carries a user-friendly message."""
