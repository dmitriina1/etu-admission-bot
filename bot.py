from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import requests
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from etu_rank import (
    API_URL,
    LISTS,
    MODES,
    SCENARIOS,
    Applicant,
    exact_metrics,
    fetch_html,
    find_target,
    parse_applicants,
    parse_budget_places,
    sample_unknown_score,
    stay_probability,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("etu-bot")

BOT_TOKEN = os.environ["BOT_TOKEN"]
OWNER_CHAT_ID = int(os.environ["OWNER_CHAT_ID"])
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
CRON_SECRET = os.environ["CRON_SECRET"]
APPLICANT_CODE = os.getenv("APPLICANT_CODE", "1203485")
FORECAST_SIMULATIONS = int(os.getenv("FORECAST_SIMULATIONS", "300000"))
FORECAST_LIST = int(os.getenv("FORECAST_LIST", "2"))
SEED = int(os.getenv("SEED", "20260723"))

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
app = FastAPI(title="ETU admission Telegram bot")

# Пока бесплатный сервер поддерживается cron-запросами, значение хранится в памяти.
# После перезапуска первая проверка лишь запомнит актуальную дату и не пришлёт ложное обновление.
last_update_marker: str | None = None
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


def send_message(text: str, chat_id: int = OWNER_CHAT_ID) -> None:
    # Telegram ограничивает одно сообщение 4096 символами.
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
    raise RuntimeError("На странице не найдено время «Список обновлен». Возможно, сайт изменил разметку.")


def load_snapshot(list_name: str, list_id: str) -> ListSnapshot:
    with requests.Session() as session:
        full_html = fetch_html(session, list_id, [], body_only=False)
        updated_at = parse_update_time(full_html)
        budget_places = parse_budget_places(full_html)
        modes: dict[str, list[Applicant]] = {}
        for mode_name, filters in MODES:
            body = fetch_html(session, list_id, filters, body_only=True)
            modes[mode_name] = parse_applicants(body)
    return ListSnapshot(list_name, list_id, updated_at, budget_places, modes)


def load_all_snapshots() -> list[ListSnapshot]:
    return [load_snapshot(name, list_id) for name, list_id in LISTS.items()]


def format_mode(mode_name: str, applicants: list[Applicant], budget_places: int | None) -> str:
    metrics = exact_metrics(applicants, APPLICANT_CODE, budget_places)
    if metrics is None:
        return f"<b>{html.escape(mode_name)}</b>\nАбитуриент отсутствует (строк: {len(applicants)})"

    target = metrics["target"]
    assert isinstance(target, Applicant)
    lines = [
        f"<b>{html.escape(mode_name)}</b>",
        f"Место: <b>{metrics['position']}</b> из {len(applicants)}",
        f"Балл: <b>{target.real_score}</b> ({target.special_score} + {target.language_score} + {target.achievements_score})",
        f"Приоритет: {target.priority}; согласие: {'да' if target.has_consent else 'нет'}",
        f"Выше с согласием: {metrics['above_with_consent']}",
    ]
    if budget_places:
        lines.append(f"Бюджетных мест: {budget_places}; {metrics['status_text']}")
        if metrics["position"] <= budget_places:
            lines.append(f"Запас до границы: {metrics['reserve']} мест")
        else:
            lines.append(f"До бюджетной зоны: {metrics['reserve']} мест")
        lines.append(f"Текущий проходной балл: {metrics['cutoff_score']}")
        lines.append(f"Неизвестных баллов: {metrics['unknown_count']}")
    return "\n".join(lines)


def format_snapshot(snapshot: ListSnapshot) -> str:
    parts = [
        f"📋 <b>{html.escape(snapshot.name)}</b>",
        f"Обновлено: <b>{html.escape(snapshot.updated_at)}</b>",
    ]
    for mode_name, _ in MODES:
        applicants = snapshot.modes.get(mode_name)
        if applicants:
            parts.append(format_mode(mode_name, applicants, snapshot.budget_places))
    return "\n\n".join(parts)


def status_text(snapshots: list[ListSnapshot]) -> str:
    return "\n\n━━━━━━━━━━━━━━\n\n".join(format_snapshot(s) for s in snapshots)


def vectorized_forecast(
    applicants: list[Applicant],
    budget_places: int,
    simulations: int,
    seed: int,
) -> list[dict[str, object]]:
    target = find_target(applicants, APPLICANT_CODE)
    if target is None:
        raise RuntimeError("Абитуриент отсутствует в выбранном списке.")

    competitors = [a for a in applicants if a.code != APPLICANT_CODE]
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
                if not outranks:
                    continue
                p = stay_probability(applicant, scenario, unknown=False)
                above += (rng.random(simulations) <= p)
                continue

            gets_score = rng.random(simulations) <= scenario.unknown_gets_score
            p_stay = stay_probability(applicant, scenario, unknown=True)
            active = gets_score & (rng.random(simulations) <= p_stay)
            if not np.any(active):
                continue

            # Выбираем диапазон балла согласно тем же весам, что в исходной модели.
            band_prob = np.array([band[2] for band in scenario.unknown_score_bands], dtype=float)
            band_prob /= band_prob.sum()
            bands = rng.choice(len(band_prob), size=simulations, p=band_prob)
            scores = np.zeros(simulations, dtype=np.int16)
            for band_index, (low, high, _) in enumerate(scenario.unknown_score_bands):
                mask = bands == band_index
                count = int(mask.sum())
                if count:
                    scores[mask] = np.rint(rng.triangular(low, (low + high) / 2, high, count)).astype(np.int16)

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


def build_forecast_message(list_number: int) -> str:
    entries = list(LISTS.items())
    if list_number < 1 or list_number > len(entries):
        raise ValueError(f"Номер списка должен быть от 1 до {len(entries)}")
    name, list_id = entries[list_number - 1]
    snapshot = load_snapshot(name, list_id)
    applicants = snapshot.modes["Все приоритеты"]
    if not snapshot.budget_places:
        raise RuntimeError("Не удалось определить число бюджетных мест.")

    results = vectorized_forecast(
        applicants,
        snapshot.budget_places,
        FORECAST_SIMULATIONS,
        SEED + list_number * 100,
    )
    lines = [
        f"🎲 <b>Прогноз: {html.escape(name)}</b>",
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


def check_for_update_and_notify(force: bool = False) -> str:
    global last_update_marker
    snapshots = load_all_snapshots()
    marker = "|".join(f"{s.list_id}:{s.updated_at}" for s in snapshots)

    if last_update_marker is None:
        last_update_marker = marker
        if force:
            send_message(status_text(snapshots))
        return "initialized"

    if marker != last_update_marker or force:
        old = last_update_marker
        last_update_marker = marker
        send_message("🔔 <b>Конкурсные списки обновились</b>\n\n" + status_text(snapshots))
        log.info("Update marker changed: %s -> %s", old, marker)
        return "notified"

    return "unchanged"


async def run_forecast(chat_id: int, list_number: int) -> None:
    if forecast_lock.locked():
        send_message("Прогноз уже считается. Дождись предыдущего результата.", chat_id)
        return
    async with forecast_lock:
        send_message(
            f"Запускаю прогноз на {FORECAST_SIMULATIONS:,} симуляций для списка {list_number}…".replace(",", " "),
            chat_id,
        )
        try:
            message = await asyncio.to_thread(build_forecast_message, list_number)
            send_message(message, chat_id)
        except Exception as error:
            log.exception("Forecast failed")
            send_message(f"Ошибка прогноза: <code>{html.escape(str(error))}</code>", chat_id)


@app.get("/")
def health() -> dict:
    return {"ok": True, "service": "etu-admission-bot"}


@app.get("/cron/check")
def cron_check(x_cron_secret: str | None = Header(default=None)) -> dict:
    if x_cron_secret != CRON_SECRET:
        raise HTTPException(status_code=403, detail="bad secret")
    try:
        result = check_for_update_and_notify(force=False)
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

    if chat_id != OWNER_CHAT_ID:
        if chat_id:
            send_message("Этот бот закрытый.", chat_id)
        return {"ok": True}

    command, *args = text.split()
    command = command.split("@", 1)[0].lower()

    if command in {"/start", "/help"}:
        send_message(
            "<b>Бот конкурсных списков ЛЭТИ</b>\n\n"
            "/status — текущие баллы и места\n"
            "/check — проверить обновление сейчас\n"
            "/forecast — прогноз для списка по умолчанию\n"
            "/forecast 1 — прогноз для списка 1\n"
            "/forecast 2 — прогноз для списка 2",
            chat_id,
        )
    elif command == "/status":
        try:
            snapshots = await asyncio.to_thread(load_all_snapshots)
            send_message(status_text(snapshots), chat_id)
        except Exception as error:
            send_message(f"Ошибка: <code>{html.escape(str(error))}</code>", chat_id)
    elif command == "/check":
        try:
            result = await asyncio.to_thread(check_for_update_and_notify, True)
            if result == "initialized":
                send_message("Текущее состояние сохранено.", chat_id)
        except Exception as error:
            send_message(f"Ошибка: <code>{html.escape(str(error))}</code>", chat_id)
    elif command == "/forecast":
        try:
            number = int(args[0]) if args else FORECAST_LIST
        except ValueError:
            send_message("Использование: <code>/forecast 1</code> или <code>/forecast 2</code>", chat_id)
        else:
            tasks.add_task(run_forecast, chat_id, number)
    else:
        send_message("Неизвестная команда. Используй /help", chat_id)

    return {"ok": True}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "bot:app",
        host="127.0.0.1",
        port=10000,
        reload=False,
    )