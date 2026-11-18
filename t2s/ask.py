"""Ask a question from the command line.

    python -m t2s.ask "How many customers live in Sfax?"
    python -m t2s.ask "Quel est le solde moyen des comptes épargne ?" --trace
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Text-to-SQL agent over the synthetic bank database")
    ap.add_argument("question")
    ap.add_argument("--backend", default=None, help="offline (default) | openai | ollama")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--trace", action="store_true", help="print every step of the graph")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    state = Agent(backend=args.backend, max_attempts=args.max_attempts).run(args.question)
    if args.json:
        print(json.dumps({"status": state.status, "sql": state.sql, "answer": state.answer,
                          "attempts": state.attempts, "trace": state.trace}, ensure_ascii=False, indent=2))
        return 0
    print(f"SQL ({state.attempts} attempt(s)): {state.sql}\n")
    print(state.answer)
    if args.trace:
        print("\nTrace:")
        for step in state.trace:
            print("  " + json.dumps(step, ensure_ascii=False))
    return 0 if state.status in ("answered", "empty") else 1


if __name__ == "__main__":
    raise SystemExit(main())
