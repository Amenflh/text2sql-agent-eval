"""HTTP API (FastAPI).  uvicorn t2s.api:app --port 8000"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .agent import Agent

app = FastAPI(title="text2sql-agent-eval", version=__version__,
              description="Self-correcting Text-to-SQL agent over a SYNTHETIC banking database.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class AskResponse(BaseModel):
    status: str
    sql: str
    answer: str
    attempts: int
    columns: list[str]
    rows: list[list]
    trace: list[dict]


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    return Agent(backend=os.environ.get("T2S_BACKEND"))


@app.get("/health")
def health() -> dict:
    agent = get_agent()
    return {"status": "ok", "backend": agent.backend.name, "tables": sorted(agent.schema.tables)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    s = get_agent().run(req.question)
    res = s.result
    return AskResponse(status=s.status, sql=s.sql, answer=s.answer, attempts=s.attempts,
                       columns=res.columns if res else [], rows=[list(r) for r in res.rows] if res else [],
                       trace=s.trace)
