from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery
from aiogram.client.default import DefaultBotProperties

import memory
import keyboards
import usage_tracker
from cbr_client import get_usd_rub_rate
from config import load_config
from cost_tracker import calculate_cost
from delivery import send_llm_reply
from json_store import JsonWriteError
from keyboards import parse_mode_callback, parse_usage_callback
from openai_client import OpenAIClient, OpenAIUserError
from prompt_manager import PromptManager
from text_utils import sanitize_ai_text

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    for noisy in ("httpx", "httpcore", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Static bot messages (HTML-safe, code-generated — no raw LLM text) ─────────

_START_TEXT = (
    "👋 Привет! Я — <b>ModeMate AI</b>.\n\n"
    "Личный AI-помощник с режимами для ежедневных задач:\n\n"
    "  📅 Планирование дня\n"
    "  ⚖️ Принятие решений\n"
    "  ✍️ Редактирование текстов\n"
    "  🎓 Обучение и объяснения\n"
    "  💡 Генерация идей\n\n"
    "Команды:\n"
    "  /mode — выбрать режим работы\n"
    "  /stats — посмотреть статистику расходов\n"
    "  /reset — очистить память диалога\n"
    "  /resetstats — очистить статистику расходов\n"
    "  /help — справка\n\n"
    "Начни с /mode, выбери подходящий режим и напиши задачу.\n"
    "Я запомню выбранный режим для следующих сообщений 🤖"
)

_HELP_TEXT = (
    "📖 <b>ModeMate AI — справка</b>\n\n"
    "<b>Команды:</b>\n"
    "/start — запустить бота\n"
    "/mode — выбрать режим работы (inline-кнопки)\n"
    "/stats — статистика расходов токенов и стоимости\n"
    "/reset — очистить память диалога\n"
    "/resetstats — очистить статистику расходов (с подтверждением)\n"
    "/help — эта справка\n\n"
    "<b>Режимы:</b>\n"
    "  📅 <b>Планировщик дня</b> — реалистичный план без токсичной продуктивности\n"
    "  ⚖️ <b>Помощник по решениям</b> — плюсы, минусы, риски и нейтральный вывод\n"
    "  ✍️ <b>Редактор текста</b> — яснее, мягче, увереннее или короче\n"
    "  🎓 <b>Учебный помощник</b> — объяснения по шагам с примерами\n"
    "  💡 <b>Лаборатория идей</b> — идеи для проектов, названий, UX и фич\n\n"
    "После каждого ответа появляется кнопка <b>📊 Детали запроса</b> — "
    "нажми, чтобы увидеть расход токенов и стоимость.\n\n"
    "<i>Если режим уже выбран, следующие сообщения будут обрабатываться в этом режиме, "
    "пока ты не переключишь его через /mode.</i>\n\n"
    "<i>/reset очищает только память диалога. "
    "Статистика расходов сохраняется отдельно и сбрасывается через /resetstats.</i>"
)

_ACCESS_DENIED_TEXT = "⛔ Доступ к этому боту ограничен для данного чата."


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_usage_detail(record: dict, pm: PromptManager) -> str:
    mode_label = pm.get_label(record.get("mode", ""))
    created_raw = record.get("created_at", "")
    try:
        dt = datetime.fromisoformat(created_raw).astimezone()
        created_fmt = dt.strftime("%d.%m.%Y %H:%M:%S")
    except Exception:
        created_fmt = created_raw

    fallback_note = "\n⚠️ Использован fallback-курс (ЦБ РФ был недоступен)" if record.get("is_fallback_rate") else ""

    return (
        f"📊 Детали запроса\n\n"
        f"Режим: {mode_label}\n"
        f"Модель: {record.get('model', '—')}\n"
        f"Дата/время: {created_fmt}\n\n"
        f"Токены:\n"
        f"  • Входящие: {record.get('input_tokens', 0):,}\n"
        f"  • Исходящие: {record.get('output_tokens', 0):,}\n"
        f"  • Итого: {record.get('total_tokens', 0):,}\n\n"
        f"Стоимость:\n"
        f"  • {record.get('total_usd', 0):.6f} USD\n"
        f"  • {record.get('total_rub', 0):.4f} ₽\n\n"
        f"Курс ЦБ РФ: {record.get('usd_rub_rate', 0):.2f} ₽/USD "
        f"(на {record.get('rate_date', '—')})"
        f"{fallback_note}"
    )


def _format_stats(stats: dict, usd_rub: float, rate_date: str, is_fallback: bool) -> str:
    fallback_note = " ⚠️ fallback" if is_fallback else ""

    def rub(usd_val: float) -> str:
        return f"{usd_val * usd_rub:.4f}"

    return (
        f"📈 <b>Статистика расходов</b>\n\n"
        f"<b>Запросы:</b>\n"
        f"  • Всего: {stats['total_requests']}\n"
        f"  • Сегодня: {stats['today_requests']}\n"
        f"  • За месяц: {stats['month_requests']}\n\n"
        f"<b>Токены:</b>\n"
        f"  • Всего: {stats['total_tokens']:,}\n"
        f"  • Сегодня: {stats['today_tokens']:,}\n"
        f"  • За месяц: {stats['month_tokens']:,}\n\n"
        f"<b>Стоимость (USD):</b>\n"
        f"  • Всего: ${stats['total_usd']:.6f}\n"
        f"  • Сегодня: ${stats['today_usd']:.6f}\n"
        f"  • За месяц: ${stats['month_usd']:.6f}\n\n"
        f"<b>Стоимость (₽):</b>\n"
        f"  • Всего: {rub(stats['total_usd'])} ₽\n"
        f"  • Сегодня: {rub(stats['today_usd'])} ₽\n"
        f"  • За месяц: {rub(stats['month_usd'])} ₽\n\n"
        f"<i>Курс ЦБ РФ на {rate_date}: 1 USD = {usd_rub:.2f} ₽{fallback_note}</i>\n"
        f"<i>Стоимость примерная. Цены берутся из .env.</i>"
    )


# ── Text-message orchestration (extracted for handler-level testing) ──────────

async def process_text_message(
    message: Message,
    pm: PromptManager,
    oai: OpenAIClient,
    cfg,
    log: logging.Logger,
) -> None:
    chat_id = message.chat.id
    user_text = message.text or ""

    if cfg.allowed_chat_ids is not None and chat_id not in cfg.allowed_chat_ids:
        log.warning("Rejected LLM request from unauthorized chat_id=%s", chat_id)
        await message.answer(_ACCESS_DENIED_TEXT)
        return

    mode_key = memory.get_mode(cfg.memory_file, chat_id, pm.default_mode)
    system_prompt = pm.get_system_prompt(mode_key)
    history = memory.get_messages(cfg.memory_file, chat_id)

    thinking_msg = await message.answer("⏳ Думаю…")

    try:
        try:
            llm_response = await oai.chat(
                system_prompt=system_prompt,
                history=history,
                user_message=user_text,
            )
        except OpenAIUserError as exc:
            await message.answer(str(exc))
            return

        # OpenAI completed the request and returned usage data, so the cost
        # was incurred regardless of whether usage persistence or Telegram
        # delivery below succeed. Compute it now, independent of both.
        usd_rub, is_fallback, rate_date = await get_usd_rub_rate(cfg.usd_rub_fallback)
        report = calculate_cost(
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            total_tokens=llm_response.total_tokens,
            input_price_per_1m=cfg.input_price_per_1m,
            output_price_per_1m=cfg.output_price_per_1m,
            usd_rub_rate=usd_rub,
            rate_date=rate_date,
            rate_is_fallback=is_fallback,
        )

        # Usage persistence is auxiliary analytics -- the OpenAI cost is
        # already incurred, so a write failure here must not discard the
        # already-generated answer. On failure we still deliver the answer,
        # just without a details button, since there is no valid persisted
        # request record for it to look up.
        try:
            request_id = usage_tracker.save_usage(
                path=cfg.usage_file,
                chat_id=chat_id,
                mode=mode_key,
                model=cfg.openai_model,
                input_tokens=report.input_tokens,
                output_tokens=report.output_tokens,
                total_tokens=report.total_tokens,
                total_usd=report.total_usd,
                usd_rub_rate=report.usd_rub_rate,
                rate_date=report.rate_date,
                total_rub=report.total_rub,
                is_fallback_rate=report.rate_is_fallback,
            )
            kb = keyboards.usage_keyboard(request_id)
        except JsonWriteError:
            log.error(
                "Failed to persist usage record for chat_id=%s; delivering answer without details button.",
                chat_id, exc_info=True,
            )
            kb = None

        clean_text = sanitize_ai_text(llm_response.text)

        # Deliver as plain text (parse_mode=None), chunked/fallback-safe.
        # Only record the exchange as delivered conversation history once
        # Telegram has actually accepted it.
        delivered = await send_llm_reply(message.answer, clean_text, reply_markup=kb)
        if delivered:
            memory.append_messages(
                path=cfg.memory_file,
                chat_id=chat_id,
                user_text=user_text,
                assistant_text=llm_response.text,
                limit=cfg.memory_limit,
            )
    finally:
        try:
            await thinking_msg.delete()
        except Exception:
            log.warning(
                "Failed to delete 'thinking' placeholder for chat_id=%s", chat_id, exc_info=True,
            )


# ── Handlers ──────────────────────────────────────────────────────────────────

def register_handlers(
    dp: Dispatcher,
    pm: PromptManager,
    oai: OpenAIClient,
    cfg,
) -> None:
    log = logging.getLogger(__name__)

    # /start
    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        await message.answer(_START_TEXT, parse_mode=ParseMode.HTML)

    # /help
    @dp.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        await message.answer(_HELP_TEXT, parse_mode=ParseMode.HTML)

    # /mode
    @dp.message(Command("mode"))
    async def cmd_mode(message: Message) -> None:
        chat_id = message.chat.id
        current = memory.get_mode(cfg.memory_file, chat_id, pm.default_mode)
        kb = keyboards.modes_keyboard(pm, current_mode=current)
        await message.answer("Выбери режим работы:", reply_markup=kb)

    # callback: mode selected
    @dp.callback_query(F.data.startswith(keyboards.CALLBACK_MODE_PREFIX))
    async def cb_mode_selected(callback: CallbackQuery) -> None:
        mode_key = parse_mode_callback(callback.data or "")
        if mode_key not in pm.mode_keys():
            await callback.answer("Неизвестный режим.", show_alert=True)
            return

        chat_id = callback.message.chat.id  # type: ignore[union-attr]
        memory.set_mode(cfg.memory_file, chat_id, mode_key)
        label = pm.get_label(mode_key)
        description = pm.get_description(mode_key)
        log.info("chat_id=%s selected mode '%s'", chat_id, mode_key)

        await callback.message.edit_text(  # type: ignore[union-attr]
            f"✅ Режим установлен: <b>{label}</b>\n\n<i>{description}</i>",
            parse_mode=ParseMode.HTML,
        )
        await callback.answer()

    # /reset  (memory only, stats untouched)
    @dp.message(Command("reset"))
    async def cmd_reset(message: Message) -> None:
        chat_id = message.chat.id
        memory.reset_messages(cfg.memory_file, chat_id)
        log.info("chat_id=%s reset message history.", chat_id)
        await message.answer(
            "🗑️ История диалога очищена. Режим и статистика расходов сохранены.",
        )

    # /stats
    @dp.message(Command("stats"))
    async def cmd_stats(message: Message) -> None:
        chat_id = message.chat.id
        stats = usage_tracker.get_stats(cfg.usage_file, chat_id)
        usd_rub, is_fallback, rate_date = await get_usd_rub_rate(cfg.usd_rub_fallback)

        if stats["total_requests"] == 0:
            await message.answer(
                "📊 Статистика пуста — вы ещё не делали запросов.",
                parse_mode=ParseMode.HTML,
            )
            return

        text = _format_stats(stats, usd_rub, rate_date, is_fallback)
        await message.answer(text, parse_mode=ParseMode.HTML)

    # /resetstats  (shows confirm keyboard)
    @dp.message(Command("resetstats"))
    async def cmd_resetstats(message: Message) -> None:
        kb = keyboards.resetstats_confirm_keyboard()
        await message.answer(
            "⚠️ Вы уверены, что хотите очистить статистику расходов?\n"
            "Память диалога при этом не затронется.",
            reply_markup=kb,
        )

    # callback: resetstats confirm
    @dp.callback_query(F.data == keyboards.CALLBACK_RESETSTATS_CONFIRM)
    async def cb_resetstats_confirm(callback: CallbackQuery) -> None:
        chat_id = callback.message.chat.id  # type: ignore[union-attr]
        removed = usage_tracker.reset_stats(cfg.usage_file, chat_id)
        log.info("chat_id=%s reset stats (%d records removed).", chat_id, removed)
        await callback.message.edit_text(  # type: ignore[union-attr]
            f"✅ Статистика расходов очищена ({removed} записей удалено).",
        )
        await callback.answer()

    # callback: resetstats cancel
    @dp.callback_query(F.data == keyboards.CALLBACK_RESETSTATS_CANCEL)
    async def cb_resetstats_cancel(callback: CallbackQuery) -> None:
        await callback.message.edit_text("❌ Отменено. Статистика не изменена.")  # type: ignore[union-attr]
        await callback.answer()

    # callback: show usage details for a specific request
    @dp.callback_query(F.data.startswith(keyboards.CALLBACK_USAGE_PREFIX))
    async def cb_usage_detail(callback: CallbackQuery) -> None:
        request_id = parse_usage_callback(callback.data or "")
        if not request_id:
            await callback.answer("Неверный идентификатор запроса.", show_alert=True)
            return

        if callback.message is None:
            await callback.answer("Не удалось определить чат для запроса.", show_alert=True)
            return

        chat_id = callback.message.chat.id
        record = usage_tracker.get_usage_record(cfg.usage_file, request_id, chat_id=chat_id)
        if record is None:
            await callback.answer("Данные запроса не найдены.", show_alert=True)
            return

        text = _format_usage_detail(record, pm)
        await callback.message.answer(text)
        await callback.answer()

    # plain text → LLM
    @dp.message(F.text)
    async def handle_text(message: Message) -> None:
        await process_text_message(message, pm, oai, cfg, log)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config()
    pm = PromptManager(cfg.prompts_file)
    oai = OpenAIClient(api_key=cfg.openai_api_key, model=cfg.openai_model)

    os.makedirs(cfg.data_dir, exist_ok=True)

    bot = Bot(
        token=cfg.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    register_handlers(dp, pm, oai, cfg)

    logger.info("Starting ModeMate AI (model=%s)", cfg.openai_model)
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
