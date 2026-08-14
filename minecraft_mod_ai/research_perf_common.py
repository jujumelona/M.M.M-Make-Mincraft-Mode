from __future__ import annotations

import os
import sqlite3


def env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw) if raw else float(default)
    except ValueError:
        value = float(default)
    return max(minimum, value)


def env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None
