"""Text-to-SQL agent as an explicit graph of steps.

    link -> generate -> validate -> execute -> answer
                           ^           |
                           +- repair <-+  (SQLite error, or empty result on a filtered query)

Each node reads and writes a shared ``State`` and returns the name of the next
node, like a LangGraph ``StateGraph`` but with no dependency. The whole run is
recorded in ``state.trace`` for debugging and evaluation.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import guard, repair
from .answer import render_answer
from .backends import get_backend
from .data import DEFAULT_DB, ensure_db
from .db import ExecResult, connect_ro, execute, lock
from .schema import Link, Linker, Schema

END = "__end__"
FR_MARKERS = {"combien", "quel", "quelle", "quels", "quelles", "les", "des", "par", "moyenne", "clients",
              "comptes", "credits", "prets", "agence", "agences", "solde", "montant", "est", "sont", "le",
              "la", "du", "de", "ont", "sans", "liste", "donne", "nombre", "plus", "chaque", "ville"}


def detect_lang(toks: list[str]) -> str:
    fr = sum(t in FR_MARKERS for t in toks)
    return "fr" if fr >= 2 or (fr == 1 and len(toks) <= 4) else "en"


@dataclass
class State:
    question: str
    lang: str = "en"
    link: Link | None = None
    sql: str = ""
    attempts: int = 0
    result: ExecResult | None = None
    first_sql: str = ""
    first_result: ExecResult | None = None
    status: str = "pending"          # answered | empty | blocked | failed
    reason: str = ""
    answer: str = ""
    trace: list[dict] = field(default_factory=list)
    ms: float = 0.0

    def log(self, node: str, **info):
        self.trace.append({"node": node, **info})


class Agent:
    def __init__(self, db_path: str | Path = DEFAULT_DB, backend: str | None = None,
                 max_attempts: int = 3, row_limit: int = guard.DEFAULT_LIMIT):
        self.con = connect_ro(ensure_db(Path(db_path)))
        self.schema = Schema.from_db(self.con)
        self.linker = Linker(self.con, self.schema)
        lock(self.con)
        self.backend = get_backend(self.schema, backend)
        self.max_attempts = max_attempts
        self.row_limit = row_limit
        self.nodes = {"link": self.link, "generate": self.generate, "validate": self.validate,
                      "execute": self.execute, "repair": self.repair, "answer": self.answer}

    # ------------------------------------------------------------------ nodes
    def link(self, s: State) -> str:
        s.link = self.linker.link(s.question)
        s.lang = detect_lang(s.link.toks)
        s.log("link", lang=s.lang, tables=s.link.tables,
              values=[f"{v.table}.{v.column}={v.value}" for v in s.link.values])
        return "generate"

    def generate(self, s: State) -> str:
        s.attempts = 1
        try:
            s.sql = self.backend.generate(s.question, s.link, s.lang)
        except Exception as exc:  # network / provider errors for LLM backends
            s.status, s.reason = "failed", f"generation error: {exc}"
            s.log("generate", error=s.reason)
            return "answer"
        s.first_sql = s.sql
        s.log("generate", backend=self.backend.name, sql=s.sql)
        return "validate"

    def validate(self, s: State) -> str:
        res = guard.validate(s.sql, self.row_limit)
        s.log("validate", ok=res.ok, reason=res.reason)
        if not res.ok:
            s.status, s.reason = "blocked", res.reason
            return "answer"
        s.sql = res.sql
        return "execute"

    def execute(self, s: State) -> str:
        s.result = execute(self.con, s.sql)
        if s.attempts == 1:
            s.first_result = s.result
        s.log("execute", attempt=s.attempts, error=s.result.error, rows=len(s.result.rows),
              ms=round(s.result.ms, 2))
        needs_repair = s.result.error or (self._looks_empty(s.result) and re.search(r"\bWHERE\b", s.sql, re.I))
        if needs_repair and s.attempts < self.max_attempts:
            return "repair"
        if s.result.error:
            s.status, s.reason = "failed", s.result.error
        else:
            s.status = "answered" if s.result.rows else "empty"
        return "answer"

    @staticmethod
    def _looks_empty(res: ExecResult) -> bool:
        """No rows, or a single aggregate that is 0 / NULL (COUNT over an empty filter)."""
        if not res.rows:
            return True
        return len(res.rows) == 1 and len(res.columns) == 1 and res.rows[0][0] in (0, None)

    def repair(self, s: State) -> str:
        err = s.result.error
        fix = None
        if err:
            for fixer in repair.ERROR_FIXERS:
                fix = fixer(s.sql, err, self.schema, s.question)
                if fix:
                    break
            feedback = f"SQLite error: {err}"
        else:
            fix = repair.fix_empty_result(s.sql, self.con, self.schema)
            feedback = "the query returned no rows; check the filter values against the data"
        if fix:
            new_sql, how = fix
            tool = "deterministic"
        else:
            try:
                new_sql = self.backend.repair(s.question, s.sql, feedback, s.link, s.lang)
            except Exception as exc:
                new_sql = None
                feedback += f" (backend repair failed: {exc})"
            how, tool = "backend re-prompt", self.backend.name
        if not new_sql or new_sql.strip() == s.sql.strip():
            s.log("repair", applied=False, feedback=feedback)
            s.status = "failed" if err else "empty"
            s.reason = err or "no rows"
            return "answer"
        s.attempts += 1
        s.log("repair", applied=True, tool=tool, how=how, sql=new_sql)
        s.sql = new_sql
        return "validate"

    def answer(self, s: State) -> str:
        if s.status == "pending":
            s.status = "answered"
        s.answer = render_answer(s)
        return END

    # -------------------------------------------------------------------- run
    def run(self, question: str) -> State:
        s = State(question=question)
        t0 = time.perf_counter()
        node = "link"
        while node != END:
            node = self.nodes[node](s)
        s.ms = (time.perf_counter() - t0) * 1000
        return s
