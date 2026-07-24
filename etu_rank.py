from __future__ import annotations

import argparse
import math
import random
import re
import statistics
from dataclasses import dataclass
from typing import Iterable

import requests
from bs4 import BeautifulSoup


DEFAULT_APPLICANT_CODE = "1203485"
DEFAULT_SIMULATIONS = 30_000

API_URL = "https://lists.priem.etu.ru/public/list.html"

LISTS = {
    "1.2 — Компьютерные науки и информатика": "019ee53c-e6ac-7b31-af10-3a9848e2582b",
    "1.3 — Физические науки": "019ee53c-e6ac-7b6d-af10-3a984961511e",
    "1.4 — Химические науки": "019ee53c-e6ac-7a81-af10-3a9846d3f886",
    "2.2 — Электроника, фотоника, приборостроение и связь": "019ee53c-e6ac-7791-af10-3a9842257c11",
    "2.3 — Информационные технологии и телекоммуникации": "019ee53c-e6ac-7971-af10-3a9844da95ee",
    "2.4 — Энергетика и электротехника": "019ee53c-e6ac-7a11-af10-3a9845f54820",
    "2.5 — Машиностроение": "019ee53c-e6ac-7a49-af10-3a9845f7e9f6",
    "2.6 — Химические технологии, науки о материалах, металлургия": "019ee53c-e6ac-787d-af10-3a9842f79061",
    "5.2 — Экономика": "019ee53c-e6ac-7af5-af10-3a98480b19cb",
    "5.4 — Социология": "019ee53c-e6ac-7915-af10-3a9843ff7583",
    "5.5 — Политология": "019ee53c-e6ac-7abd-af10-3a9847adc04c",
    "5.7 — Философия": "019ee53c-e6ac-79b1-af10-3a98450d897f",
    "5.9 — Филология": "019ee53c-e6ac-78d5-af10-3a9843c6325c",
}

MODES = (
    ("Все приоритеты", []),
    ("Только приоритет №1", ["1st_priority"]),
    ("Приоритет №1 + согласие", ["1st_priority", "has_agreement"]),
)


@dataclass(frozen=True)
class Applicant:
    site_position: int
    code: str
    priority: int
    enrollment_terms: str
    displayed_score: int
    special_score: int
    language_score: int
    achievements_score: int
    target_achievements_score: int
    consent: str
    status: str

    @property
    def real_score(self) -> int:
        # ИД целевые намеренно не включаются.
        return (
            self.special_score
            + self.language_score
            + self.achievements_score
        )

    @property
    def has_consent(self) -> bool:
        return bool(self.consent.strip())

    @property
    def has_known_score(self) -> bool:
        # Нулевой конкурсный балл считаем неизвестным/неполным результатом.
        return self.real_score > 0


@dataclass(frozen=True)
class Scenario:
    name: str

    # Вероятность того, что кандидат с уже известным баллом
    # фактически останется в конкурсе на это направление.
    known_priority1_consent: float
    known_priority1_no_consent: float
    known_other_consent: float
    known_other_no_consent: float

    # Вероятность, что кандидат с нулевым баллом вообще получит
    # итоговый ненулевой результат.
    unknown_gets_score: float

    # Вероятность, что кандидат с нулевым баллом останется
    # в этом конкурсе после появления результата.
    unknown_priority1_consent: float
    unknown_priority1_no_consent: float
    unknown_other_consent: float
    unknown_other_no_consent: float

    # Распределение итогового балла неизвестного кандидата:
    # (нижняя граница, верхняя граница, вероятность).
    unknown_score_bands: tuple[tuple[int, int, float], ...]

    # Распределения для отдельно отсутствующих компонентов.
    special_score_bands: tuple[tuple[int, int, float], ...]
    achievements_score_bands: tuple[tuple[int, int, float], ...]


SCENARIOS = (
    Scenario(
        name="Оптимистичный",
        known_priority1_consent=0.92,
        known_priority1_no_consent=0.55,
        known_other_consent=0.42,
        known_other_no_consent=0.18,
        unknown_gets_score=0.70,
        unknown_priority1_consent=0.72,
        unknown_priority1_no_consent=0.40,
        unknown_other_consent=0.30,
        unknown_other_no_consent=0.12,
        unknown_score_bands=(
            (1, 69, 0.28),
            (70, 89, 0.32),
            (90, 100, 0.24),
            (101, 110, 0.11),
            (111, 125, 0.045),
            (126, 150, 0.005),
        ),
        special_score_bands=(
            (14, 29, 0.25),
            (30, 39, 0.35),
            (40, 45, 0.30),
            (46, 50, 0.10),
        ),
        achievements_score_bands=(
            (0, 0, 0.55),
            (1, 10, 0.20),
            (11, 20, 0.15),
            (21, 30, 0.08),
            (31, 40, 0.02),
        ),
    ),
    Scenario(
        name="Базовый",
        known_priority1_consent=0.95,
        known_priority1_no_consent=0.67,
        known_other_consent=0.55,
        known_other_no_consent=0.27,
        unknown_gets_score=0.82,
        unknown_priority1_consent=0.82,
        unknown_priority1_no_consent=0.53,
        unknown_other_consent=0.42,
        unknown_other_no_consent=0.20,
        unknown_score_bands=(
            (1, 69, 0.20),
            (70, 89, 0.28),
            (90, 100, 0.25),
            (101, 110, 0.17),
            (111, 125, 0.085),
            (126, 150, 0.015),
        ),
        special_score_bands=(
            (14, 29, 0.15),
            (30, 39, 0.30),
            (40, 45, 0.35),
            (46, 50, 0.20),
        ),
        achievements_score_bands=(
            (0, 0, 0.35),
            (1, 10, 0.20),
            (11, 20, 0.20),
            (21, 30, 0.18),
            (31, 40, 0.07),
        ),
    ),
    Scenario(
        name="Пессимистичный",
        known_priority1_consent=0.98,
        known_priority1_no_consent=0.78,
        known_other_consent=0.68,
        known_other_no_consent=0.40,
        unknown_gets_score=0.92,
        unknown_priority1_consent=0.90,
        unknown_priority1_no_consent=0.68,
        unknown_other_consent=0.58,
        unknown_other_no_consent=0.34,
        unknown_score_bands=(
            (1, 69, 0.12),
            (70, 89, 0.21),
            (90, 100, 0.24),
            (101, 110, 0.23),
            (111, 125, 0.17),
            (126, 150, 0.03),
        ),
        special_score_bands=(
            (14, 29, 0.08),
            (30, 39, 0.22),
            (40, 45, 0.38),
            (46, 50, 0.32),
        ),
        achievements_score_bands=(
            (0, 0, 0.20),
            (1, 10, 0.15),
            (11, 20, 0.20),
            (21, 30, 0.25),
            (31, 40, 0.20),
        ),
    ),
)


def clean_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split()).strip()


def parse_int(value: str, default: int = 0) -> int:
    value = clean_text(value)
    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def fetch_html(
    session: requests.Session,
    list_id: str,
    filters: Iterable[str],
    *,
    body_only: bool,
) -> str:
    params: list[tuple[str, str]] = [("id", list_id)]

    if body_only:
        params.append(("bodyOnly", "true"))

    for value in filters:
        params.append(("filters[]", value))

    response = session.get(
        API_URL,
        params=params,
        headers={
            "Accept": "text/html",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/130 Safari/537.36"
            ),
        },
        timeout=30,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def parse_budget_places(html: str) -> int | None:
    soup = BeautifulSoup(html, "html.parser")
    text = clean_text(soup.get_text(" ", strip=True))
    match = re.search(r"Бюджетных мест:\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_applicants(html: str) -> list[Applicant]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    if table is None:
        raise RuntimeError("В ответе API не найдена таблица.")

    tbody = table.find("tbody")
    if tbody is None:
        raise RuntimeError("В таблице не найден tbody.")

    applicants: list[Applicant] = []

    for row in tbody.find_all("tr", recursive=False):
        cells = [
            clean_text(cell.get_text(" ", strip=True))
            for cell in row.find_all(["td", "th"], recursive=False)
        ]

        if len(cells) < 11:
            continue

        code = cells[1]
        if not code.isdigit():
            continue

        applicants.append(
            Applicant(
                site_position=parse_int(cells[0], len(applicants) + 1),
                code=code,
                priority=parse_int(cells[2], -1),
                enrollment_terms=cells[3],
                displayed_score=parse_int(cells[4]),
                special_score=parse_int(cells[5]),
                language_score=parse_int(cells[6]),
                achievements_score=parse_int(cells[7]),
                target_achievements_score=parse_int(cells[8]),
                consent=cells[9],
                status=cells[10],
            )
        )

    if not applicants:
        raise RuntimeError("Не удалось прочитать ни одной строки таблицы.")

    return applicants


def applicant_sort_key(applicant: Applicant) -> tuple[int, int, int, int, int]:
    """Официальный порядок при равном конкурсном балле.

    Сначала сравнивается общий балл, затем специальная дисциплина,
    иностранный язык, индивидуальные достижения и только потом ID.
    Отрицательные значения нужны для сортировки по убыванию.
    """
    return (
        -applicant.real_score,
        -applicant.special_score,
        -applicant.language_score,
        -applicant.achievements_score,
        int(applicant.code),
    )


def sorted_applicants(applicants: list[Applicant]) -> list[Applicant]:
    return sorted(applicants, key=applicant_sort_key)


def applicant_outranks(left: Applicant, right: Applicant) -> bool:
    """Возвращает True, если left должен стоять выше right."""
    return applicant_sort_key(left) < applicant_sort_key(right)


def find_target(
    applicants: list[Applicant],
    applicant_code: str,
) -> Applicant | None:
    return next(
        (applicant for applicant in applicants if applicant.code == applicant_code),
        None,
    )


def exact_metrics(
    applicants: list[Applicant],
    applicant_code: str,
    budget_places: int | None,
) -> dict[str, object] | None:
    target = find_target(applicants, applicant_code)
    if target is None:
        return None

    ordered = sorted_applicants(applicants)
    position = next(
        index
        for index, applicant in enumerate(ordered, start=1)
        if applicant.code == applicant_code
    )

    same_score_ordered = [
        applicant for applicant in ordered
        if applicant.real_score == target.real_score
    ]
    same_score_position = next(
        index
        for index, applicant in enumerate(same_score_ordered, start=1)
        if applicant.code == applicant_code
    )

    above = ordered[: position - 1]

    above_with_consent = sum(applicant.has_consent for applicant in above)
    unknown = [
        applicant for applicant in applicants
        if applicant.code != applicant_code and not applicant.has_known_score
    ]

    cutoff_score = None
    reserve = None
    status_text = "неизвестно"

    if budget_places and budget_places > 0:
        if len(ordered) >= budget_places:
            cutoff_score = ordered[budget_places - 1].real_score

        if position <= budget_places:
            reserve = budget_places - position
            status_text = "в бюджетной зоне"
        else:
            reserve = position - budget_places
            status_text = "вне бюджетной зоны"

    # Сколько неизвестных должны обойти пользователя, чтобы он оказался
    # за границей бюджета при неизменности остальных известных баллов.
    unknown_needed_to_push_out = None
    if budget_places and position <= budget_places:
        unknown_needed_to_push_out = budget_places - position + 1
    elif budget_places:
        unknown_needed_to_push_out = 0

    return {
        "target": target,
        "position": position,
        "same_score_count": len(same_score),
        "same_score_position": same_score_position,
        "above_with_consent": above_with_consent,
        "budget_places": budget_places,
        "cutoff_score": cutoff_score,
        "reserve": reserve,
        "status_text": status_text,
        "unknown_count": len(unknown),
        "unknown_needed_to_push_out": unknown_needed_to_push_out,
    }


def stay_probability(applicant: Applicant, scenario: Scenario, unknown: bool) -> float:
    if unknown:
        if applicant.priority == 1 and applicant.has_consent:
            return scenario.unknown_priority1_consent
        if applicant.priority == 1 and not applicant.has_consent:
            return scenario.unknown_priority1_no_consent
        if applicant.priority != 1 and applicant.has_consent:
            return scenario.unknown_other_consent
        return scenario.unknown_other_no_consent

    if applicant.priority == 1 and applicant.has_consent:
        return scenario.known_priority1_consent
    if applicant.priority == 1 and not applicant.has_consent:
        return scenario.known_priority1_no_consent
    if applicant.priority != 1 and applicant.has_consent:
        return scenario.known_other_consent
    return scenario.known_other_no_consent


def sample_unknown_score(
    rng: random.Random,
    bands: tuple[tuple[int, int, float], ...],
) -> int:
    value = rng.random()
    cumulative = 0.0

    for low, high, probability in bands:
        cumulative += probability
        if value <= cumulative:
            # Треугольное распределение с центром ближе к середине диапазона.
            mode = (low + high) / 2
            return round(rng.triangular(low, high, mode))

    low, high, _ = bands[-1]
    return rng.randint(low, high)


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("Пустой список значений.")

    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(fraction * len(ordered)) - 1),
    )
    return ordered[index]


def simulate_scenario(
    applicants: list[Applicant],
    applicant_code: str,
    budget_places: int,
    scenario: Scenario,
    simulations: int,
    seed: int,
) -> dict[str, object] | None:
    target = find_target(applicants, applicant_code)
    if target is None:
        return None

    rng = random.Random(seed)
    final_positions: list[int] = []
    admitted_count = 0

    competitors = [
        applicant
        for applicant in applicants
        if applicant.code != applicant_code
    ]

    for _ in range(simulations):
        active: list[tuple[int, int]] = [
            (target.real_score, target.special_score, target.language_score, target.achievements_score, int(target.code))
        ]

        for applicant in competitors:
            # Ноль по иностранному языку означает недопуск: такого кандидата
            # не моделируем и не включаем в итоговое распределение.
            if applicant.language_score <= 0:
                continue

            incomplete = (
                applicant.special_score <= 0
                or applicant.achievements_score <= 0
            )
            probability = stay_probability(
                applicant,
                scenario,
                unknown=incomplete,
            )
            if rng.random() > probability:
                continue

            special = applicant.special_score
            achievements = applicant.achievements_score
            if special <= 0:
                special = sample_unknown_score(rng, scenario.special_score_bands)
            if achievements <= 0:
                achievements = sample_unknown_score(
                    rng, scenario.achievements_score_bands
                )

            score = special + applicant.language_score + achievements
            active.append(
                (
                    score,
                    special,
                    applicant.language_score,
                    achievements,
                    int(applicant.code),
                )
            )

        active.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4]))

        target_position = next(
            index
            for index, (*_, code) in enumerate(active, start=1)
            if code == int(target.code)
        )

        final_positions.append(target_position)
        if target_position <= budget_places:
            admitted_count += 1

    return {
        "scenario": scenario.name,
        "probability": admitted_count / simulations,
        "median": round(statistics.median(final_positions)),
        "p10": percentile(final_positions, 0.10),
        "p90": percentile(final_positions, 0.90),
        "worst_10_percent": percentile(final_positions, 0.90),
    }


def print_exact_metrics(
    mode_name: str,
    applicants: list[Applicant],
    applicant_code: str,
    budget_places: int | None,
) -> None:
    metrics = exact_metrics(
        applicants,
        applicant_code,
        budget_places,
    )

    if metrics is None:
        print(
            f"\n{mode_name}: абитуриент {applicant_code} отсутствует "
            f"(строк: {len(applicants)})"
        )
        return

    target = metrics["target"]
    assert isinstance(target, Applicant)

    print(f"\n{mode_name}")
    print(f"  Строк в выборке:                 {len(applicants)}")
    print(f"  Настоящее место:                 {metrics['position']}")
    print(f"  Пересчитанный балл:              {target.real_score}")
    print(
        "  Состав балла:                    "
        f"{target.special_score} + "
        f"{target.language_score} + "
        f"{target.achievements_score}"
    )
    print(
        f"  С таким же баллом:               "
        f"{metrics['same_score_count']}"
    )
    print(
        f"  Место среди людей с тем же баллом: "
        f"{metrics['same_score_position']} из "
        f"{metrics['same_score_count']}"
    )
    print(
        f"  Выше тебя с согласием:           "
        f"{metrics['above_with_consent']}"
    )

    if budget_places:
        print(f"  Бюджетных мест:                  {budget_places}")
        print(f"  Положение:                       {metrics['status_text']}")

        reserve = metrics["reserve"]
        if metrics["position"] <= budget_places:
            print(f"  Запас до границы:                {reserve} мест")
        else:
            print(f"  До бюджетной зоны:               {reserve} мест")

        cutoff = metrics["cutoff_score"]
        if cutoff is not None:
            print(f"  Текущий проходной балл:          {cutoff}")

        print(
            f"  Кандидатов с неизвестным баллом: "
            f"{metrics['unknown_count']}"
        )

        needed = metrics["unknown_needed_to_push_out"]
        if isinstance(needed, int) and needed > 0:
            unknown_count = int(metrics["unknown_count"])
            threshold = (
                100.0 * needed / unknown_count
                if unknown_count > 0
                else 0.0
            )
            print(
                f"  Чтобы вытеснить тебя из бюджета: "
                f"{needed} неизвестных кандидатов"
            )
            if unknown_count > 0:
                print(
                    f"  Это доля неизвестных:            "
                    f"{threshold:.1f}%"
                )


def main(
    applicant_code: str,
    simulations: int,
    seed: int,
) -> None:
    print(f"Абитуриент: {applicant_code}")
    print("Балл = Спец. дисциплина + Ин. яз + ИД")
    print(
        "Прогноз вероятностный: значения зависят от сценарных "
        "допущений, а не являются официальной вероятностью."
    )

    with requests.Session() as session:
        for list_number, (list_name, list_id) in enumerate(
            LISTS.items(),
            start=1,
        ):
            print("\n" + "=" * 96)
            print(list_name)
            print(f"ID списка: {list_id}")

            try:
                full_html = fetch_html(
                    session,
                    list_id,
                    [],
                    body_only=False,
                )
                budget_places = parse_budget_places(full_html)
            except Exception as error:
                budget_places = None
                print(f"Не удалось определить число бюджетных мест: {error}")

            mode_data: dict[str, list[Applicant]] = {}

            for mode_name, filters in MODES:
                try:
                    html = fetch_html(
                        session,
                        list_id,
                        filters,
                        body_only=True,
                    )
                    applicants = parse_applicants(html)
                    mode_data[mode_name] = applicants
                    print_exact_metrics(
                        mode_name,
                        applicants,
                        applicant_code,
                        budget_places,
                    )
                except requests.RequestException as error:
                    print(f"\n{mode_name}: ошибка HTTP: {error}")
                except Exception as error:
                    print(f"\n{mode_name}: ошибка обработки: {error}")

            # Монте-Карло запускаем по полному списку всех приоритетов.
            all_applicants = mode_data.get("Все приоритеты")

            if (
                all_applicants is None
                or budget_places is None
                or budget_places <= 0
            ):
                print("\nПрогноз Монте-Карло недоступен.")
                continue

            if find_target(all_applicants, applicant_code) is None:
                print(
                    "\nПрогноз Монте-Карло: абитуриент отсутствует "
                    "в полном списке."
                )
                continue

            print("\n" + "-" * 96)
            print(
                f"ПРОГНОЗ МОНТЕ-КАРЛО "
                f"({simulations:,} симуляций на сценарий)"
                .replace(",", " ")
            )

            for scenario_index, scenario in enumerate(SCENARIOS):
                result = simulate_scenario(
                    all_applicants,
                    applicant_code,
                    budget_places,
                    scenario,
                    simulations,
                    seed + list_number * 100 + scenario_index,
                )

                if result is None:
                    continue

                probability = 100.0 * float(result["probability"])

                print(f"\n  {result['scenario']} сценарий")
                print(
                    f"    Вероятность попасть в бюджет: "
                    f"{probability:.1f}%"
                )
                print(
                    f"    Медианное итоговое место:      "
                    f"{result['median']}"
                )
                print(
                    f"    Вероятный диапазон мест:       "
                    f"{result['p10']}–{result['p90']}"
                )
                print(
                    f"    Граница худших 10% сценариев:  "
                    f"{result['worst_10_percent']}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Расчёт настоящего места и сценарный прогноз "
            "поступления в конкурсных списках ЛЭТИ"
        )
    )
    parser.add_argument(
        "--code",
        default=DEFAULT_APPLICANT_CODE,
        help=(
            "Уникальный код поступающего "
            f"(по умолчанию {DEFAULT_APPLICANT_CODE})"
        ),
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help=(
            "Количество симуляций на сценарий "
            f"(по умолчанию {DEFAULT_SIMULATIONS})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260716,
        help="Seed генератора случайных чисел для повторяемости.",
    )

    args = parser.parse_args()

    if args.simulations < 1_000:
        parser.error("--simulations должно быть не меньше 1000")

    main(
        applicant_code=args.code,
        simulations=args.simulations,
        seed=args.seed,
    )
