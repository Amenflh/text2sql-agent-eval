"""Interchangeable SQL generators. ``T2S_BACKEND`` selects one (default: offline)."""
from __future__ import annotations

import os

from ..schema import Schema
from .llm import OllamaBackend, OpenAICompatBackend
from .offline import OfflineBackend

BACKENDS = {"offline": OfflineBackend, "openai": OpenAICompatBackend, "ollama": OllamaBackend}


def get_backend(schema: Schema, name: str | None = None):
    name = (name or os.environ.get("T2S_BACKEND", "offline")).lower()
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose one of {sorted(BACKENDS)}")
    return BACKENDS[name](schema)
