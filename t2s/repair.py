"""Deterministic repair tools driven by execution feedback.

They work on the SQL text and the SQLite error (or an empty result), so they
apply to drafts from any backend. The agent tries them first and falls back to
asking the backend (LLM re-prompt with the error) when none applies.
"""
from __future__ import annotations

import difflib
import re
import sqlite3

from .schema import DIMENSION_WORDS, TABLE_WORDS, Schema, norm, tokenize

COLUMN_SYNONYMS = {
    "amount": ["principal", "balance", "amount"], "montant": ["principal", "amount", "balance"],
    "rate": ["interest_rate"], "interest": ["interest_rate"], "date": ["txn_date", "start_date", "opened_date"],
    "type": ["account_type", "loan_type", "txn_type"], "name": ["name", "last_name"],
    "category": ["merchant_category"], "duration": ["term_months"], "term": ["term_months"],
}
_STRING = r"'(?:[^']|'')*'"


def referenced_tables(sql: str, schema: Schema) -> list[str]:
    out = []
    for t in re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", sql, re.I):
        if t in schema.tables and t not in out:
            out.append(t)
    return out


def _replace_identifier(sql: str, old: str, new_fn) -> str:
    """Replace identifier ``old`` (optionally qualified) outside string literals."""
    pattern = re.compile(rf"{_STRING}|(?<![\w.])(?:(\w+)\.)?{re.escape(old)}(?!\w)")

    def repl(m: re.Match) -> str:
        if m.group(0).startswith("'"):
            return m.group(0)
        prefix = sql[max(0, m.start() - 3):m.start()].upper()
        if prefix.endswith("AS "):
            return m.group(0)
        return new_fn(m.group(1))
    return pattern.sub(repl, sql)


def fix_ambiguous_column(sql: str, error: str, schema: Schema, question: str):
    m = re.search(r"ambiguous column name: (?:\w+\.)?(\w+)", error)
    if not m:
        return None
    col = m.group(1)
    cands = [t for t in referenced_tables(sql, schema) if col in schema.tables[t]]
    if len(cands) < 2:
        return None
    choice = cands[0]
    _, toks = tokenize(question)
    col_words = {w for w, d in DIMENSION_WORDS.items() if d == col} | {col}
    col_pos = [i for i, w in enumerate(toks) if w in col_words]
    for t in cands[1:]:
        if any(w in TABLE_WORDS[t] and abs(i - j) <= 3 for i, w in enumerate(toks) for j in col_pos):
            choice = t
    new = _replace_identifier(sql, col, lambda q: f"{q}.{col}" if q else f"{choice}.{col}")
    return (new, f"qualified ambiguous column '{col}' as {choice}.{col}") if new != sql else None


def fix_unknown_column(sql: str, error: str, schema: Schema, question: str):
    m = re.search(r"no such column: (?:(\w+)\.)?(\w+)", error)
    if not m:
        return None
    qual, name = m.group(1), m.group(2)
    tables = [qual] if qual in schema.tables else referenced_tables(sql, schema)
    cols = [(t, c) for t in tables for c in schema.tables[t]]
    target = None
    for syn in COLUMN_SYNONYMS.get(name, []):
        hit = next(((t, c) for t, c in cols if c == syn), None)
        if hit:
            target = hit
            break
    if not target:
        close = difflib.get_close_matches(name, [c for _, c in cols], n=1, cutoff=0.6)
        target = next(((t, c) for t, c in cols if close and c == close[0]), None)
    if not target or target[1] == name:
        return None
    new = _replace_identifier(sql, name, lambda q: f"{q}.{target[1]}" if q else target[1])
    return (new, f"replaced unknown column '{name}' with '{target[1]}'") if new != sql else None


def fix_unknown_table(sql: str, error: str, schema: Schema, question: str):
    m = re.search(r"no such table: (\w+)", error)
    if not m:
        return None
    bad = m.group(1)
    target = next((t for t, words in TABLE_WORDS.items() if bad.lower() in words), None) or \
        next(iter(difflib.get_close_matches(bad, list(schema.tables), n=1, cutoff=0.6)), None)
    if not target:
        return None
    new = _replace_identifier(sql, bad, lambda q: target)
    return (new, f"replaced unknown table '{bad}' with '{target}'") if new != sql else None


def fix_empty_result(sql: str, con: sqlite3.Connection, schema: Schema):
    """Value linking: align literals with values that really exist in the column."""
    # strftime() returns text: comparing it with an integer is always false
    new = re.sub(r"(strftime\([^)]*\)\s*(?:=|<|>|<=|>=)\s*)(\d{4}(?:-\d{2})?)(?![\d'])", r"\1'\2'", sql)
    if new != sql:
        return new, "quoted the year compared with strftime()"
    tables = referenced_tables(sql, schema)
    notes = []
    for m in re.finditer(rf"(?:(\w+)\.)?(\w+)\s*(=|!=|<>)\s*({_STRING})", sql):
        qual, col, literal = m.group(1), m.group(2), m.group(4)[1:-1].replace("''", "'")
        owners = [qual] if qual in schema.tables else [t for t in tables if col in schema.tables[t]]
        if not owners:
            continue
        try:
            values = [str(r[0]) for r in con.execute(
                f"SELECT DISTINCT {col} FROM {owners[0]} WHERE {col} IS NOT NULL LIMIT 1000")]
        except sqlite3.Error:
            continue
        if literal in values:
            continue
        by_norm = {norm(v).replace("_", " "): v for v in values}
        key = norm(literal).replace("_", " ")
        best = by_norm.get(key)
        if best is None:
            close = difflib.get_close_matches(key, list(by_norm), n=1, cutoff=0.8)
            best = by_norm[close[0]] if close else None
        if best is not None:
            sql = sql.replace(m.group(4), "'" + best.replace("'", "''") + "'", 1)
            notes.append(f"'{literal}' -> '{best}' ({col})")
    return (sql, "value linking: " + ", ".join(notes)) if notes else None


ERROR_FIXERS = (fix_ambiguous_column, fix_unknown_column, fix_unknown_table)
