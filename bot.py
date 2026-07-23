from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import requests
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from database import (
    delete_user,
    get_state,
    get_user_code,
    init_database,
    list_notification_users,
    save_user,
    set_state,
)
from etu_rank import (
    LISTS,
    MODES,
    SCENARIOS,
    Applicant,
    exact_metrics,
    fetch_html,
    find_target,
    parse_applicants,
    parse_budget_places,
    stay_probability,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("etu-bot")

BOT_TOKEN = os.environ["BOT_TOKEN"]
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0"))
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
CRON_SECRET = os.environ["CRON_SECRET"]
DEFAULT_APPLICANT_CODE = os.getenv("APPLICANT_CODE", "1203485")
FORECAST_SIMULATIONS = int(os.getenv("FORECAST_SIMULATIONS", "300000"))
FORECAST_LIST = int(os.getenv("FORECAST_LIST", "2"))
SEED = int(os.getenv("SEED", "20260723"))

# Понятные названия направлений в сообщениях Telegram.
# При необходимости можно поменять только эти две строки.
DIRECTION_NAMES = {
    "019ee53c-e6ac-7791-af10-3a9842257c11": "Электроника, фотоника, приборостроение и связь",
    "019ee53c-e6ac-7971-af10-3a9844da95ee": "Информатика и вычислительная техника",
}

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
app = FastAPI(title="ETU admission Telegram bot")
forecast_lock = asyncio.Lock()


@dataclass
class ListSnapshot:
    name: str
    list_id: str
    updated_at: str
    budget_places: int | None
    modes: dict[str, list[Applicant]]


def telegram_call(method: str, payload: dict) -> dict:
    response = requests.post(f"{TELEGRAM_API}/{method}", json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data


def send_message(text: str, chat_id: int) -> None:
    chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)] or [""]
    for chunk in chunks:
        telegram_call(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
        )


def parse_update_time(page_html: str) -> str:
    text = re.sub(r"\s+", " ", page_html.replace("&nbsp;", " "))
    patterns = (
        r"Список\s+обновлен\s*:?\s*(\d{2}\.\d{2}\.\d{4}\s*,?\s*\d{2}:\d{2}:\d{2})",
        r"Список\s+обновл[её]н\s*:?\s*(\d{2}\.\d{2}\.\d{4}\s*,?\s*\d{2}:\d{2}:\d{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return re.sub(r"\s*,\s*", ", ", match.group(1))
    raise RuntimeError("На странице не найдено время «Список обновлен».")


def display_name(list_name: str, list_id: str) -> str:
    return DIRECTION_NAMES.get(list_id, list_name)


def load_snapshot(list_name: str, list_id: str) -> ListSnapshot:
    with requests.Session() as session:
        full_html = fetch_html(session, list_id, [], body_only=False)
        updated_at = parse_update_time(full_html)
        budget_places = parse_budget_places(full_html)
        modes: dict[str, list[Applicant]] = {}
        for mode_name, filters in MODES:
            body = fetch_html(session, list_id, filters, body_only=True)
            modes[mode_name] = parse_applicants(body)
    return ListSnapshot(display_name(list_name, list_id), list_id, updated_at, budget_places, modes)


def load_all_snapshots() -> list[ListSnapshot]:
    return [load_snapshot(name, list_id) for name, list_id in LISTS.items()]


def code_exists(snapshots: list[ListSnapshot], applicant_code: str) -> bool:
    return any(
        find_target(snapshot.modes.get("Все приоритеты", []), applicant_code) is not None
        for snapshot in snapshots
    )


def format_mode(
    mode_name: str,
    applicants: list[Applicant],
    applicant_code: str,
    budget_places: int | None,
) -> str:
    metrics = exact_metrics(applicants, applicant_code, budget_places)
    if metrics is None:
        return f"<b>{html.escape(mode_name)}</b>\nАбитуриент отсутствует (строк: {len(applicants)})"

    target = metrics["target"]
    assert isinstance(target, Applicant)
    lines = [
        f"<b>{html.escape(mode_name)}</b>",
        f"Место: <b>{metrics['position']}</b> из {len(applicants)}",
        f"Балл: <b>{target.real_score}</b> "
        f"({target.special_score} + {target.language_score} + {target.achievements_score})",
        f"Приоритет: {target.priority}; согласие: {'да' if target.has_consent else 'нет'}",
        f"Выше с согласием: {metrics['above_with_consent']}",
    ]
    if budget_places:
        lines.append(f"Бюджетных мест: {budget_places}")
        if metrics["position"] <= budget_places:
            lines.append(f"Запас до границы: {metrics['reserve']} мест")
        else:
            lines.append(f"До бюджетной зоны: {metrics['reserve']} мест")
        if metrics["cutoff_score"] is not None:
            lines.append(f"Текущий проходной балл: {metrics['cutoff_score']}")
        lines.append(f"Неизвестных баллов: {metrics['unknown_count']}")
    return "\n".join(lines)


def format_snapshot(snapshot: ListSnapshot, applicant_code: str) -> str:
    parts = [
        f"📋 <b>{html.escape(snapshot.name)}</b>",
        f"Обновлено: <b>{html.escape(snapshot.updated_at)}</b>",
    ]
    for mode_name, _ in MODES:
        applicants = snapshot.modes.get(mode_name)
        if applicants:
            parts.append(format_mode(mode_name, applicants, applicant_code, snapshot.budget_places))
    return "\n\n".join(parts)


def status_text(snapshots: list[ListSnapshot], applicant_code: str) -> str:
    header = f"Код поступающего: <code>{html.escape(applicant_code)}</code>"
    body = "\n\n━━━━━━━━━━━━━━\n\n".join(
        format_snapshot(snapshot, applicant_code) for snapshot in snapshots
    )
    return f"{header}\n\n{body}"


def vectorized_forecast(
    applicants: list[Applicant],
    applicant_code: str,
    budget_places: int,
    simulations: int,
    seed: int,
) -> list[dict[str, object]]:
    target = find_target(applicants, applicant_code)
    if target is None:
        raise RuntimeError("Абитуриент отсутствует в выбранном направлении.")

    competitors = [a for a in applicants if a.code != applicant_code]
    results: list[dict[str, object]] = []
    target_code = int(target.code)

    for scenario_index, scenario in enumerate(SCENARIOS):
        rng = np.random.default_rng(seed + scenario_index)
        above = np.zeros(simulations, dtype=np.int16)

        for applicant in competitors:
            if applicant.has_known_score:
                outranks = (
                    applicant.real_score > target.real_score
                    or (applicant.real_score == target.real_score and int(applicant.code) < target_code)
                )
                if outranks:
                    p = stay_probability(applicant, scenario, unknown=False)
                    above += rng.random(simulations) <= p
                continue

            active = (
                (rng.random(simulations) <= scenario.unknown_gets_score)
                & (rng.random(simulations) <= stay_probability(applicant, scenario, unknown=True))
            )
            if not np.any(active):
                continue

            band_prob = np.array([band[2] for band in scenario.unknown_score_bands], dtype=float)
            band_prob /= band_prob.sum()
            bands = rng.choice(len(band_prob), size=simulations, p=band_prob)
            scores = np.zeros(simulations, dtype=np.int16)
            for band_index, (low, high, _) in enumerate(scenario.unknown_score_bands):
                mask = bands == band_index
                count = int(mask.sum())
                if count:
                    scores[mask] = np.rint(
                        rng.triangular(low, (low + high) / 2, high, count)
                    ).astype(np.int16)

            outranks = (scores > target.real_score) | (
                (scores == target.real_score) & (int(applicant.code) < target_code)
            )
            above += active & outranks

        positions = above.astype(np.int32) + 1
        results.append(
            {
                "scenario": scenario.name,
                "probability": float(np.mean(positions <= budget_places)),
                "median": int(np.rint(np.median(positions))),
                "p10": int(np.percentile(positions, 10, method="higher")),
                "p90": int(np.percentile(positions, 90, method="higher")),
            }
        )
    return results


def build_forecast_message(list_number: int, applicant_code: str) -> str:
    entries = list(LISTS.items())
    if list_number < 1 or list_number > len(entries):
        raise ValueError(f"Номер направления должен быть от 1 до {len(entries)}")
    name, list_id = entries[list_number - 1]
    snapshot = load_snapshot(name, list_id)
    applicants = snapshot.modes["Все приоритеты"]
    if not snapshot.budget_places:
        raise RuntimeError("Не удалось определить число бюджетных мест.")

    results = vectorized_forecast(
        applicants,
        applicant_code,
        snapshot.budget_places,
        FORECAST_SIMULATIONS,
        SEED + list_number * 100,
    )
    lines = [
        f"🎲 <b>Прогноз: {html.escape(snapshot.name)}</b>",
        f"Код: <code>{html.escape(applicant_code)}</code>",
        f"Данные обновлены: {html.escape(snapshot.updated_at)}",
        f"Симуляций на сценарий: <b>{FORECAST_SIMULATIONS:,}</b>".replace(",", " "),
        "<i>Это сценарная модель, а не официальная вероятность.</i>",
    ]
    for result in results:
        lines.extend(
            [
                "",
                f"<b>{result['scenario']}</b>",
                f"Вероятность бюджета: {100 * float(result['probability']):.1f}%",
                f"Медианное место: {result['median']}",
                f"Вероятный диапазон: {result['p10']}–{result['p90']}",
            ]
        )
    return "\n".join(lines)


def update_marker(snapshots: list[ListSnapshot]) -> str:
    return "|".join(f"{snapshot.list_id}:{snapshot.updated_at}" for snapshot in snapshots)


def notify_all_users(snapshots: list[ListSnapshot]) -> tuple[int, int]:
    sent = 0
    failed = 0
    for chat_id, applicant_code in list_notification_users():
        try:
            send_message(
                "🔔 <b>Конкурсные списки обновились</b>\n\n"
                + status_text(snapshots, applicant_code),
                chat_id,
            )
            sent += 1
        except Exception:
            failed += 1
            log.exception("Cannot notify chat_id=%s", chat_id)
    return sent, failed


def check_for_update_and_notify() -> str:
    snapshots = load_all_snapshots()
    marker = update_marker(snapshots)
    old_marker = get_state("last_update_marker")

    if old_marker is None:
        set_state("last_update_marker", marker)
        return "initialized"

    if marker == old_marker:
        return "unchanged"

    set_state("last_update_marker", marker)
    sent, failed = notify_all_users(snapshots)
    log.info("Update marker changed: %s -> %s; sent=%s failed=%s", old_marker, marker, sent, failed)
    return f"notified:{sent}:{failed}"


async def run_forecast(chat_id: int, list_number: int, applicant_code: str) -> None:
    if forecast_lock.locked():
        send_message("Прогноз уже считается. Дождись предыдущего результата.", chat_id)
        return
    async with forecast_lock:
        direction = list(DIRECTION_NAMES.values())[list_number - 1] if 1 <= list_number <= 2 else str(list_number)
        send_message(
            f"Запускаю прогноз на {FORECAST_SIMULATIONS:,} симуляций: "
            f"{html.escape(direction)}…".replace(",", " "),
            chat_id,
        )
        try:
            message = await asyncio.to_thread(build_forecast_message, list_number, applicant_code)
            send_message(message, chat_id)
        except Exception as error:
            log.exception("Forecast failed")
            send_message(f"Ошибка прогноза: <code>{html.escape(str(error))}</code>", chat_id)


def require_code(chat_id: int) -> str | None:
    code = get_user_code(chat_id)
    if code is None:
        send_message(
            "Сначала укажи уникальный код поступающего:\n"
            "<code>/setcode 1203485</code>",
            chat_id,
        )
    return code


@app.on_event("startup")
def startup() -> None:
    init_database()
    if OWNER_CHAT_ID and DEFAULT_APPLICANT_CODE and get_user_code(OWNER_CHAT_ID) is None:
        save_user(OWNER_CHAT_ID, DEFAULT_APPLICANT_CODE)


@app.get("/")
def health() -> dict:
    return {"ok": True, "service": "etu-admission-bot"}


@app.get("/cron/check")
def cron_check(x_cron_secret: str | None = Header(default=None)) -> dict:
    if x_cron_secret != CRON_SECRET:
        raise HTTPException(status_code=403, detail="bad secret")
    try:
        result = check_for_update_and_notify()
        return {"ok": True, "result": result, "checked_at": datetime.now().isoformat()}
    except Exception as error:
        log.exception("Scheduled check failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/telegram/{secret}")
async def telegram_webhook(secret: str, request: Request, tasks: BackgroundTasks) -> dict:
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="bad secret")

    update = await request.json()
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = int(chat.get("id", 0))
    text = (message.get("text") or "").strip()
    if not chat_id or not text:
        return {"ok": True}

    command, *args = text.split()
    command = command.split("@", 1)[0].lower()

    if command in {"/start", "/help"}:
        send_message(
            "<b>Бот конкурсных списков ЛЭТИ</b>\n\n"
            "1. Сохрани свой код:\n<code>/setcode 1203485</code>\n\n"
            "/mycode — показать сохранённый код\n"
            "/status — текущие баллы и места\n"
            "/check — проверить данные сейчас\n"
            "/forecast 1 — прогноз: электроника, фотоника, приборостроение и связь\n"
            "/forecast 2 — прогноз: информатика и вычислительная техника\n"
            "/remove — удалить код и отключить уведомления",
            chat_id,
        )

    elif command == "/setcode":
        if len(args) != 1 or not args[0].isdigit():
            send_message("Использование: <code>/setcode 1203485</code>", chat_id)
            return {"ok": True}
        applicant_code = args[0]
        try:
            snapshots = await asyncio.to_thread(load_all_snapshots)
            if not code_exists(snapshots, applicant_code):
                send_message(
                    "Код не найден ни в одном из отслеживаемых направлений. "
                    "Проверь цифры и попробуй снова.",
                    chat_id,
                )
                return {"ok": True}
            save_user(chat_id, applicant_code)
            send_message(
                f"✅ Код сохранён: <code>{html.escape(applicant_code)}</code>\n"
                "Автоматические уведомления включены.",
                chat_id,
            )
        except Exception as error:
            send_message(f"Ошибка: <code>{html.escape(str(error))}</code>", chat_id)

    elif command == "/mycode":
        applicant_code = get_user_code(chat_id)
        if applicant_code:
            send_message(f"Твой код: <code>{html.escape(applicant_code)}</code>", chat_id)
        else:
            send_message("Код пока не сохранён. Используй <code>/setcode 1203485</code>", chat_id)

    elif command == "/remove":
        if delete_user(chat_id):
            send_message("Код удалён. Автоматические уведомления отключены.", chat_id)
        else:
            send_message("У тебя не было сохранённого кода.", chat_id)

    elif command in {"/status", "/check"}:
        applicant_code = require_code(chat_id)
        if applicant_code:
            try:
                snapshots = await asyncio.to_thread(load_all_snapshots)
                send_message(status_text(snapshots, applicant_code), chat_id)
            except Exception as error:
                send_message(f"Ошибка: <code>{html.escape(str(error))}</code>", chat_id)

    elif command == "/forecast":
        applicant_code = require_code(chat_id)
        if applicant_code:
            try:
                number = int(args[0]) if args else FORECAST_LIST
            except ValueError:
                send_message("Использование: <code>/forecast 1</code> или <code>/forecast 2</code>", chat_id)
            else:
                tasks.add_task(run_forecast, chat_id, number, applicant_code)

    else:
        send_message("Неизвестная команда. Используй /help", chat_id)

    return {"ok": True}
