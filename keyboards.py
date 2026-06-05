from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from prompt_manager import PromptManager

CALLBACK_MODE_PREFIX = "mode:"
CALLBACK_USAGE_PREFIX = "usage:"
CALLBACK_RESETSTATS_CONFIRM = "resetstats:confirm"
CALLBACK_RESETSTATS_CANCEL = "resetstats:cancel"


def modes_keyboard(pm: PromptManager, current_mode: str | None = None) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for key in pm.mode_keys():
        label = pm.get_label(key)
        if key == current_mode:
            label = "✅ " + label
        builder.button(text=label, callback_data=f"{CALLBACK_MODE_PREFIX}{key}")
    builder.adjust(1)
    return builder.as_markup()


def usage_keyboard(request_id: str) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="📊 Детали запроса", callback_data=f"{CALLBACK_USAGE_PREFIX}{request_id}")
    return builder.as_markup()


def resetstats_confirm_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Да, очистить статистику", callback_data=CALLBACK_RESETSTATS_CONFIRM)
    builder.button(text="❌ Отмена", callback_data=CALLBACK_RESETSTATS_CANCEL)
    builder.adjust(1)
    return builder.as_markup()


def parse_mode_callback(data: str) -> str | None:
    if data.startswith(CALLBACK_MODE_PREFIX):
        return data[len(CALLBACK_MODE_PREFIX):]
    return None


def parse_usage_callback(data: str) -> str | None:
    if data.startswith(CALLBACK_USAGE_PREFIX):
        return data[len(CALLBACK_USAGE_PREFIX):]
    return None
