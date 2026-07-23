from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg

DATABASE_URL = os.environ["DATABASE_URL"]


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        yield conn


def init_database() -> None:
    with connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_users (
                chat_id BIGINT PRIMARY KEY,
                applicant_code TEXT NOT NULL,
                notifications_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )


def save_user(chat_id: int, applicant_code: str) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO bot_users (chat_id, applicant_code, notifications_enabled)
            VALUES (%s, %s, TRUE)
            ON CONFLICT (chat_id) DO UPDATE SET
                applicant_code = EXCLUDED.applicant_code,
                notifications_enabled = TRUE,
                updated_at = NOW()
            """,
            (chat_id, applicant_code),
        )


def get_user_code(chat_id: int) -> str | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT applicant_code FROM bot_users WHERE chat_id = %s",
            (chat_id,),
        ).fetchone()
    return str(row[0]) if row else None


def delete_user(chat_id: int) -> bool:
    with connection() as conn:
        cursor = conn.execute("DELETE FROM bot_users WHERE chat_id = %s", (chat_id,))
        return cursor.rowcount > 0


def list_notification_users() -> list[tuple[int, str]]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT chat_id, applicant_code
            FROM bot_users
            WHERE notifications_enabled = TRUE
            ORDER BY chat_id
            """
        ).fetchall()
    return [(int(chat_id), str(code)) for chat_id, code in rows]


def get_state(key: str) -> str | None:
    with connection() as conn:
        row = conn.execute("SELECT value FROM bot_state WHERE key = %s", (key,)).fetchone()
    return str(row[0]) if row else None


def set_state(key: str, value: str) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO bot_state (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = NOW()
            """,
            (key, value),
        )
