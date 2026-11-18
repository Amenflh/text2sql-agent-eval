"""LLM backends (OpenAI-compatible chat API, Ollama). Optional: never needed for tests.

Configured through environment variables only, with the standard library HTTP
client (no SDK dependency):

    T2S_BACKEND=openai  T2S_OPENAI_BASE_URL=https://api.openai.com/v1
                        T2S_OPENAI_API_KEY=...  T2S_MODEL=gpt-4o-mini
    T2S_BACKEND=ollama  T2S_OLLAMA_URL=http://localhost:11434  T2S_MODEL=qwen2.5-coder:7b
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from ..schema import Link, Schema

SYSTEM = (
    "You translate questions about a banking database into ONE SQLite SELECT query. "
    "Use only the tables and columns given. Qualify columns when several tables are joined. "
    "Never modify data. Answer with the SQL only, inside a ```sql block."
)


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    sql = (m.group(1) if m else text).strip()
    return sql.rstrip(";").strip()


class _ChatBackend:
    name = "llm"

    def __init__(self, schema: Schema):
        self.schema = schema
        self.model = os.environ.get("T2S_MODEL", "")

    def _prompt(self, question: str, link: Link, lang: str) -> str:
        tables = link.tables or list(self.schema.tables)
        # always give the FK neighbours so joins stay possible
        for t in list(tables):
            p = self.schema.parent_of(t)
            if p and p[1] not in tables:
                tables.append(p[1])
        hints = "".join(f"\n- the user may mean {v.column} = '{v.value}'" for v in link.values)
        return (f"Schema (SQLite):\n{self.schema.prompt_ddl(tables)}\n"
                f"Question ({'French' if lang == 'fr' else 'English'}): {question}"
                + (f"\nValue hints:{hints}" if hints else ""))

    def generate(self, question: str, link: Link, lang: str) -> str:
        return extract_sql(self._chat([{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": self._prompt(question, link, lang)}]))

    def repair(self, question: str, sql: str, feedback: str, link: Link, lang: str) -> str | None:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": self._prompt(question, link, lang)},
                {"role": "assistant", "content": f"```sql\n{sql}\n```"},
                {"role": "user", "content": f"That query failed: {feedback}. Return a corrected query."}]
        return extract_sql(self._chat(msgs))

    def _post(self, url: str, payload: dict, headers: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=float(os.environ.get("T2S_TIMEOUT", "60"))) as resp:
            return json.loads(resp.read().decode())

    def _chat(self, messages: list[dict]) -> str:
        raise NotImplementedError


class OpenAICompatBackend(_ChatBackend):
    name = "openai"

    def _chat(self, messages):
        base = os.environ.get("T2S_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("T2S_OPENAI_API_KEY", "")
        data = self._post(f"{base}/chat/completions",
                          {"model": self.model or "gpt-4o-mini", "messages": messages, "temperature": 0},
                          {"Authorization": f"Bearer {key}"} if key else {})
        return data["choices"][0]["message"]["content"]


class OllamaBackend(_ChatBackend):
    name = "ollama"

    def _chat(self, messages):
        base = os.environ.get("T2S_OLLAMA_URL", "http://localhost:11434").rstrip("/")
        data = self._post(f"{base}/api/chat", {"model": self.model or "qwen2.5-coder:7b", "messages": messages,
                                               "stream": False, "options": {"temperature": 0}}, {})
        return data["message"]["content"]
