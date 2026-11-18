"""Read-only SQLite execution with an authorizer and a time budget."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

# WITH RECURSIVE is a read too; SQLITE_RECURSIVE = 33
_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
            getattr(sqlite3, "SQLITE_RECURSIVE", 33)}


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


def connect_ro(db_path: str | Path) -> sqlite3.Connection:
    """Open the database read-only. Call ``lock`` once schema introspection is done."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def lock(con: sqlite3.Connection) -> sqlite3.Connection:
    """From now on SQLite refuses anything but reads (PRAGMA, ATTACH, writes...)."""
    con.set_authorizer(_authorizer)
    return con


@dataclass
class ExecResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    ms: float = 0.0


def execute(con: sqlite3.Connection, sql: str, timeout_s: float = 3.0) -> ExecResult:
    deadline = time.perf_counter() + timeout_s
    con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
    t0 = time.perf_counter()
    try:
        cur = con.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description or []]
        return ExecResult(cols, rows, None, (time.perf_counter() - t0) * 1000)
    except sqlite3.Error as exc:
        msg = str(exc)
        if "interrupted" in msg:
            msg = f"query exceeded the {timeout_s:g}s time budget"
        return ExecResult(error=msg, ms=(time.perf_counter() - t0) * 1000)
    finally:
        con.set_progress_handler(None, 0)
