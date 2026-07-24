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
FORECAST_LIST = os.getenv("FORECAST_LIST", "2.3")
SEED = int(os.getenv("SEED", "20260723"))

# Код направления -> (название, ID конкурсного списка).
# Порядок сохраняется в /directions и при обходе всех списков.
DIRECTIONS: dict[str, tuple[str, str]] = {
    "1.2": ("Компьютерные науки и информатика", "019ee53c-e6ac-7b31-af10-3a9848e2582b"),
    "1.3": ("Физические науки", "019ee53c-e6ac-7b6d-af10-3a984961511e"),
    "1.4": ("Химические науки", "019ee53c-e6ac-7a81-af10-3a9846d3f886"),
    "2.2": ("Электроника, фотоника, приборостроение и связь", "019ee53c-e6ac-7791-af10-3a9842257c11"),
    "2.3": ("Информационные технологии и телекоммуникации", "019ee53c-e6ac-7971-af10-3a9844da95ee"),
    "2.4": ("Энергетика и электротехника", "019ee53c-e6ac-7a11-af10-3a9845f54820"),
    "2.5": ("Машиностроение", "019ee53c-e6ac-7a49-af10-3a9845f7e9f6"),
    "2.6": ("Химические технологии, науки о материалах, металлургия", "019ee53c-e6ac-787d-af10-3a9842f79061"),
    "5.2": ("Экономика", "019ee53c-e6ac-7af5-af10-3a98480b19cb"),
    "5.4": ("Социология", "019ee53c-e6ac-7915-af10-3a9843ff7583"),
    "5.5": ("Политология", "019ee53c-e6ac-7abd-af10-3a9847adc04c"),
    "5.7": ("Философия", "019ee53c-e6ac-79b1-af10-3a98450d897f"),
    "5.9": ("Филология", "019ee53c-e6ac-78d5-af10-3a9843c6325c"),
}


# Режим «Только с согласием» намеренно удалён.
DISPLAY_MODES = (
    "Все приоритеты",
    "Только приоритет №1",
    "Приоритет №1 + согласие",
)

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
app = FastAPI(title="ETU admission Telegram bot")
forecast_lock = asyncio.Lock()
realplace_lock = asyncio.Lock()


@dataclass(frozen=True)
class ListSnapshot:
    direction_code: str
    name: str
    list_id: str
    updated_at: str
    budget_places: int | None
    modes: dict[str, list[Applicant]]

    @property
    def all_applicants(self) -> list[Applicant]:
        return self.modes["Все приоритеты"]


@dataclass(frozen=True)
class Choice:
    direction_code: str
    priority: int
    score: int
    numeric_code: int


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
    match = re.search(
        r"Список\s+обновл[её]н\s*:?\s*(\d{2}\.\d{2}\.\d{4}\s*,?\s*\d{2}:\d{2}:\d{2})",
        text,
        re.IGNORECASE,
    )
    if not match:
        raise RuntimeError("На странице не найдено время «Список обновлен».")
    return re.sub(r"\s*,\s*", ", ", match.group(1))


def derive_modes(applicants: list[Applicant]) -> dict[str, list[Applicant]]:
    """Строит три отображаемых режима из одной загруженной таблицы."""
    return {
        "Все приоритеты": applicants,
        "Только приоритет №1": [a for a in applicants if a.priority == 1],
        "Приоритет №1 + согласие": [
            a for a in applicants if a.priority == 1 and a.has_consent
        ],
    }


def load_snapshot(direction_code: str) -> ListSnapshot:
    try:
        name, list_id = DIRECTIONS[direction_code]
    except KeyError as error:
        raise ValueError(f"Неизвестное направление: {direction_code}") from error

    with requests.Session() as session:
        full_html = fetch_html(session, list_id, [], body_only=False)
        body_html = fetch_html(session, list_id, [], body_only=True)

    applicants = parse_applicants(body_html)
    return ListSnapshot(
        direction_code=direction_code,
        name=name,
        list_id=list_id,
        updated_at=parse_update_time(full_html),
        budget_places=parse_budget_places(full_html),
        modes=derive_modes(applicants),
    )


def load_all_snapshots() -> list[ListSnapshot]:
    snapshots: list[ListSnapshot] = []
    errors: list[str] = []
    for direction_code in DIRECTIONS:
        try:
            snapshots.append(load_snapshot(direction_code))
        except Exception as error:
            log.exception("Cannot load direction %s", direction_code)
            errors.append(f"{direction_code}: {error}")
    if not snapshots:
        raise RuntimeError("Не удалось загрузить ни одного списка: " + "; ".join(errors))
    return snapshots


def matching_snapshots(
    snapshots: list[ListSnapshot], applicant_code: str
) -> list[ListSnapshot]:
    return [
        snapshot
        for snapshot in snapshots
        if find_target(snapshot.all_applicants, applicant_code) is not None
    ]


def code_exists(snapshots: list[ListSnapshot], applicant_code: str) -> bool:
    return bool(matching_snapshots(snapshots, applicant_code))


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
        f"📋 <b>{html.escape(snapshot.direction_code)} — {html.escape(snapshot.name)}</b>",
        f"Обновлено: <b>{html.escape(snapshot.updated_at)}</b>",
    ]
    for mode_name in DISPLAY_MODES:
        applicants = snapshot.modes[mode_name]
        parts.append(format_mode(mode_name, applicants, applicant_code, snapshot.budget_places))
    return "\n\n".join(parts)


def status_text(snapshots: list[ListSnapshot], applicant_code: str) -> str:
    found = matching_snapshots(snapshots, applicant_code)
    if not found:
        return (
            f"Код <code>{html.escape(applicant_code)}</code> не найден "
            "ни в одном отслеживаемом бюджетном списке."
        )
    header = f"Код поступающего: <code>{html.escape(applicant_code)}</code>"
    body = "\n\n━━━━━━━━━━━━━━\n\n".join(
        format_snapshot(snapshot, applicant_code) for snapshot in found
    )
    return f"{header}\n\n{body}"


def resolve_direction(argument: str) -> str:
    value = argument.strip().replace(",", ".")
    if value not in DIRECTIONS:
        raise ValueError(
            "Неизвестный код направления. Используй /directions, "
            "например <code>/forecast 2.3</code>."
        )
    return value


def directions_text() -> str:
    lines = ["<b>Бюджетные направления</b>"]
    for code, (name, _) in DIRECTIONS.items():
        lines.append(f"<code>{code}</code> — {html.escape(name)}")
    lines.extend(
        [
            "",
            "Пример: <code>/forecast 2.3</code>",
        ]
    )
    return "\n".join(lines)


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
                    or (
                        applicant.real_score == target.real_score
                        and int(applicant.code) < target_code
                    )
                )
                if outranks:
                    p = stay_probability(applicant, scenario, unknown=False)
                    above += rng.random(simulations) <= p
                continue

            active = (
                (rng.random(simulations) <= scenario.unknown_gets_score)
                & (
                    rng.random(simulations)
                    <= stay_probability(applicant, scenario, unknown=True)
                )
            )
            if not np.any(active):
                continue

            band_prob = np.array(
                [band[2] for band in scenario.unknown_score_bands], dtype=float
            )
            band_prob /= band_prob.sum()
            bands = rng.choice(len(band_prob), size=simulations, p=band_prob)
            scores = np.zeros(simulations, dtype=np.int16)
            for band_index, (low, high, _) in enumerate(
                scenario.unknown_score_bands
            ):
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


def build_forecast_message(direction_code: str, applicant_code: str) -> str:
    snapshot = load_snapshot(direction_code)
    if not snapshot.budget_places:
        raise RuntimeError("Не удалось определить число бюджетных мест.")

    results = vectorized_forecast(
        snapshot.all_applicants,
        applicant_code,
        snapshot.budget_places,
        FORECAST_SIMULATIONS,
        SEED + sum(ord(ch) for ch in direction_code),
    )
    lines = [
        f"🎲 <b>Прогноз: {html.escape(direction_code)} — {html.escape(snapshot.name)}</b>",
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


def is_eligible_for_allocation(applicant: Applicant) -> bool:
    # По принятому правилу нулевой иностранный язык означает выбытие.
    # Нулевые спецдисциплина или ИД сами по себе кандидата не исключают.
    return applicant.language_score > 0 and applicant.has_consent and applicant.priority > 0


def build_choices(
    snapshots: list[ListSnapshot],
) -> dict[str, list[Choice]]:
    choices: dict[str, list[Choice]] = {}
    for snapshot in snapshots:
        if not snapshot.budget_places:
            continue
        for applicant in snapshot.all_applicants:
            if not is_eligible_for_allocation(applicant):
                continue
            choices.setdefault(applicant.code, []).append(
                Choice(
                    direction_code=snapshot.direction_code,
                    priority=applicant.priority,
                    score=applicant.real_score,
                    numeric_code=int(applicant.code),
                )
            )
    for applicant_choices in choices.values():
        applicant_choices.sort(key=lambda c: (c.priority, c.direction_code))
    return choices


def calculate_allocation(
    snapshots: list[ListSnapshot],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Распределяет кандидатов по лучшему доступному приоритету.

    Используется алгоритм последовательных предложений: кандидат сначала идёт
    на минимальный номер своего приоритета, направление временно удерживает
    лучших по баллу и коду в пределах бюджета, остальные переходят дальше.
    """
    choices = build_choices(snapshots)
    capacities = {
        snapshot.direction_code: int(snapshot.budget_places or 0)
        for snapshot in snapshots
    }
    held: dict[str, list[str]] = {code: [] for code in capacities}
    next_choice: dict[str, int] = {code: 0 for code in choices}
    pending = set(choices)

    while pending:
        proposals: dict[str, list[str]] = {code: [] for code in capacities}
        exhausted: set[str] = set()

        for applicant_code in pending:
            index = next_choice[applicant_code]
            applicant_choices = choices[applicant_code]
            if index >= len(applicant_choices):
                exhausted.add(applicant_code)
                continue
            proposals[applicant_choices[index].direction_code].append(applicant_code)

        next_pending: set[str] = set()
        for direction_code, new_codes in proposals.items():
            if not new_codes:
                continue
            pool = set(held[direction_code]) | set(new_codes)
            choice_by_code = {
                code: next(
                    c
                    for c in choices[code]
                    if c.direction_code == direction_code
                )
                for code in pool
            }
            ordered = sorted(
                pool,
                key=lambda code: (
                    -choice_by_code[code].score,
                    choice_by_code[code].numeric_code,
                ),
            )
            capacity = capacities[direction_code]
            accepted = ordered[:capacity] if capacity > 0 else []
            rejected = ordered[capacity:] if capacity > 0 else ordered
            held[direction_code] = accepted
            for code in rejected:
                next_choice[code] += 1
                if next_choice[code] < len(choices[code]):
                    next_pending.add(code)

        pending = next_pending - exhausted

    assigned = {
        applicant_code: direction_code
        for direction_code, applicant_codes in held.items()
        for applicant_code in applicant_codes
    }
    return assigned, held


def adjusted_position(
    snapshot: ListSnapshot,
    applicant_code: str,
    assigned: dict[str, str],
) -> tuple[int, int, list[tuple[str, str]]]:
    target = find_target(snapshot.all_applicants, applicant_code)
    if target is None:
        raise RuntimeError("Абитуриент отсутствует в направлении.")

    target_code = int(target.code)
    above = [
        applicant
        for applicant in snapshot.all_applicants
        if applicant.code != applicant_code
        and is_eligible_for_allocation(applicant)
        and (
            applicant.real_score > target.real_score
            or (
                applicant.real_score == target.real_score
                and int(applicant.code) < target_code
            )
        )
    ]
    removed: list[tuple[str, str]] = []
    for applicant in above:
        destination = assigned.get(applicant.code)
        if destination and destination != snapshot.direction_code:
            removed.append((applicant.code, destination))

    # «Обычное» место здесь тоже считается среди реально допущенных кандидатов
    # с согласием и ненулевым иностранным языком.
    ordinary_position = len(above) + 1
    real_position = ordinary_position - len(removed)
    return ordinary_position, real_position, removed


def build_realplace_message(applicant_code: str) -> str:
    snapshots = load_all_snapshots()
    found = matching_snapshots(snapshots, applicant_code)
    if not found:
        raise RuntimeError("Код не найден ни в одном отслеживаемом направлении.")

    assigned, _ = calculate_allocation(snapshots)
    assigned_direction = assigned.get(applicant_code)
    lines = [
        "🧭 <b>Расчёт места с учётом других приоритетов</b>",
        f"Код: <code>{html.escape(applicant_code)}</code>",
        "<i>Учитываются согласие, ненулевой иностранный язык, баллы, "
        "приоритеты и число бюджетных мест во всех 13 списках.</i>",
    ]

    if assigned_direction:
        assigned_name = DIRECTIONS[assigned_direction][0]
        lines.append(
            f"Расчётное распределение: <b>{html.escape(assigned_direction)} — "
            f"{html.escape(assigned_name)}</b>"
        )
    else:
        lines.append("Расчётное распределение: <b>бюджетное место пока не найдено</b>")

    for snapshot in found:
        ordinary, real, removed = adjusted_position(
            snapshot, applicant_code, assigned
        )
        target = find_target(snapshot.all_applicants, applicant_code)
        assert target is not None
        lines.extend(
            [
                "",
                f"<b>{html.escape(snapshot.direction_code)} — {html.escape(snapshot.name)}</b>",
                f"Приоритет: {target.priority}; бюджетных мест: {snapshot.budget_places or 'неизвестно'}",
                f"Место среди допущенных с согласием: <b>{ordinary}</b>",
                f"После распределения по другим направлениям: <b>{real}</b>",
                f"Ушли на более подходящие направления: {len(removed)}",
            ]
        )
        if removed:
            counts: dict[str, int] = {}
            for _, destination in removed:
                counts[destination] = counts.get(destination, 0) + 1
            summary = ", ".join(
                f"{code} — {count}"
                for code, count in sorted(counts.items())
            )
            lines.append(f"Куда распределились: {html.escape(summary)}")

    lines.extend(
        [
            "",
            "<i>Это расчётная модель. Она не заменяет официальный конкурсный список "
            "и может измениться при новых баллах, согласиях и приоритетах.</i>",
        ]
    )
    return "\n".join(lines)


def update_marker(snapshots: list[ListSnapshot]) -> str:
    return "|".join(
        f"{snapshot.list_id}:{snapshot.updated_at}" for snapshot in snapshots
    )


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
    old_marker = get_state("last_update_marker_v2_all_directions")

    if old_marker is None:
        set_state("last_update_marker_v2_all_directions", marker)
        return "initialized"
    if marker == old_marker:
        return "unchanged"

    set_state("last_update_marker_v2_all_directions", marker)
    sent, failed = notify_all_users(snapshots)
    log.info(
        "Update marker changed: %s -> %s; sent=%s failed=%s",
        old_marker,
        marker,
        sent,
        failed,
    )
    return f"notified:{sent}:{failed}"


async def run_forecast(
    chat_id: int, direction_code: str, applicant_code: str
) -> None:
    if forecast_lock.locked():
        send_message("Прогноз уже считается. Дождись предыдущего результата.", chat_id)
        return
    async with forecast_lock:
        direction = DIRECTIONS[direction_code][0]
        send_message(
            f"Запускаю прогноз на {FORECAST_SIMULATIONS:,} симуляций: "
            f"{html.escape(direction_code)} — {html.escape(direction)}…".replace(",", " "),
            chat_id,
        )
        try:
            message = await asyncio.to_thread(
                build_forecast_message, direction_code, applicant_code
            )
            send_message(message, chat_id)
        except Exception as error:
            log.exception("Forecast failed")
            send_message(
                f"Ошибка прогноза: <code>{html.escape(str(error))}</code>", chat_id
            )


async def run_realplace(chat_id: int, applicant_code: str) -> None:
    if realplace_lock.locked():
        send_message("Расчёт /realplace уже выполняется. Попробуй чуть позже.", chat_id)
        return
    async with realplace_lock:
        send_message(
            "Загружаю 13 списков и рассчитываю распределение по приоритетам…",
            chat_id,
        )
        try:
            message = await asyncio.to_thread(build_realplace_message, applicant_code)
            send_message(message, chat_id)
        except Exception as error:
            log.exception("Realplace failed")
            send_message(
                f"Ошибка /realplace: <code>{html.escape(str(error))}</code>",
                chat_id,
            )


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
    if (
        OWNER_CHAT_ID
        and DEFAULT_APPLICANT_CODE
        and get_user_code(OWNER_CHAT_ID) is None
    ):
        save_user(OWNER_CHAT_ID, DEFAULT_APPLICANT_CODE)


@app.get("/")
def health() -> dict:
    return {
        "ok": True,
        "service": "etu-admission-bot",
        "directions": len(DIRECTIONS),
    }


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
async def telegram_webhook(
    secret: str, request: Request, tasks: BackgroundTasks
) -> dict:
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
            "/status — места во всех списках, где найден код\n"
            "/check — проверить данные сейчас\n"
            "/directions — коды бюджетных направлений\n"
            "/forecast 2.3 — прогноз выбранного направления\n"
            "/realplace — место с учётом распределения по приоритетам\n"
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
                    "Код не найден ни в одном из 13 отслеживаемых направлений. "
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
            send_message(
                f"Твой код: <code>{html.escape(applicant_code)}</code>", chat_id
            )
        else:
            send_message(
                "Код пока не сохранён. Используй <code>/setcode 1203485</code>",
                chat_id,
            )

    elif command == "/remove":
        if delete_user(chat_id):
            send_message(
                "Код удалён. Автоматические уведомления отключены.", chat_id
            )
        else:
            send_message("У тебя не было сохранённого кода.", chat_id)

    elif command in {"/status", "/check"}:
        applicant_code = require_code(chat_id)
        if applicant_code:
            try:
                snapshots = await asyncio.to_thread(load_all_snapshots)
                send_message(status_text(snapshots, applicant_code), chat_id)
            except Exception as error:
                send_message(
                    f"Ошибка: <code>{html.escape(str(error))}</code>", chat_id
                )

    elif command == "/directions":
        send_message(directions_text(), chat_id)

    elif command == "/forecast":
        applicant_code = require_code(chat_id)
        if applicant_code:
            argument = args[0] if args else FORECAST_LIST
            try:
                direction_code = resolve_direction(argument)
            except ValueError as error:
                send_message(str(error), chat_id)
            else:
                tasks.add_task(
                    run_forecast, chat_id, direction_code, applicant_code
                )

    elif command == "/realplace":
        applicant_code = require_code(chat_id)
        if applicant_code:
            tasks.add_task(run_realplace, chat_id, applicant_code)

    else:
        send_message("Неизвестная команда. Используй /help", chat_id)

    return {"ok": True}
