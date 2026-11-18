"""Natural-language answer and table rendering.

The answer is templated from the result rows, never free-generated: numbers in
the answer are exactly the numbers SQLite returned.
"""
from __future__ import annotations

MSG = {
    "en": {"blocked": "Request refused by the SQL guardrails: {reason}.",
           "failed": "I could not build a valid query ({reason}).",
           "empty": "No matching rows.",
           "scalar": "Answer: {value}.",
           "rows": "{n} row(s) returned{more}.", "more": ", showing the first {k}"},
    "fr": {"blocked": "Demande refusée par les garde-fous SQL : {reason}.",
           "failed": "Je n'ai pas pu construire une requête valide ({reason}).",
           "empty": "Aucune ligne ne correspond.",
           "scalar": "Réponse : {value}.",
           "rows": "{n} ligne(s) renvoyée(s){more}.", "more": ", les {k} premières affichées"},
}


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}".replace(",", " ")
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")
    return "NULL" if v is None else str(v)


def markdown_table(columns, rows, max_rows: int = 10) -> str:
    if not columns:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows[:max_rows]]
    return "\n".join(lines)


def render_answer(state, max_rows: int = 10) -> str:
    m = MSG.get(state.lang, MSG["en"])
    if state.status in ("blocked", "failed"):
        return m[state.status].format(reason=state.reason)
    res = state.result
    if not res or not res.rows:
        return m["empty"]
    if len(res.rows) == 1 and len(res.columns) == 1:
        return m["scalar"].format(value=fmt(res.rows[0][0]))
    more = m["more"].format(k=max_rows) if len(res.rows) > max_rows else ""
    return m["rows"].format(n=len(res.rows), more=more) + "\n\n" + markdown_table(res.columns, res.rows, max_rows)
