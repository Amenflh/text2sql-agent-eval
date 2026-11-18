"""SQL guardrails: read-only, single statement, bounded result size.

Three independent layers, so that one bug does not open the database:
1. this static validator (before execution);
2. a SQLite authorizer that only allows read operations (see ``db.py``);
3. a read-only connection (``mode=ro``) with a time budget.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FORBIDDEN = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE",
             "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "GRANT", "REVOKE", "LOAD_EXTENSION")
DEFAULT_LIMIT = 200


@dataclass
class GuardResult:
    ok: bool
    sql: str
    reason: str = ""


def _mask(sql: str) -> str:
    """Blank out string literals and comments so keywords inside them are ignored."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate(sql: str, limit: int = DEFAULT_LIMIT) -> GuardResult:
    if not sql or not sql.strip():
        return GuardResult(False, sql, "empty query")
    cleaned = sql.strip().rstrip(";").strip()
    masked = _mask(cleaned)
    if ";" in masked:
        return GuardResult(False, sql, "multiple statements are not allowed")
    head = masked.lstrip("( \n\t").split(None, 1)[0].upper() if masked.strip() else ""
    if head not in ("SELECT", "WITH"):
        return GuardResult(False, sql, f"only SELECT queries are allowed (got {head or 'nothing'})")
    words = set(re.findall(r"[A-Za-z_]+", masked.upper()))
    bad = sorted(words.intersection(FORBIDDEN))
    if bad:
        return GuardResult(False, sql, f"forbidden keyword(s): {', '.join(bad)}")
    if not re.search(r"\bLIMIT\s+\d+\s*(OFFSET\s+\d+\s*)?$", masked, re.I):
        cleaned = f"{cleaned} LIMIT {limit}"
    return GuardResult(True, cleaned)
