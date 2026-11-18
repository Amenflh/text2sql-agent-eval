"""Reproducible benchmark: execution accuracy before / after self-correction,
latency, and guardrail behaviour.

    python -m t2s.eval                     # offline backend, writes reports/eval.md
    python -m t2s.eval --min-accuracy 0.8  # CI gate

A prediction is correct when its result set equals the result of the reference
SQL (execution match): rows compared as a multiset, numbers rounded to 2
decimals, column order ignored. "Lenient" accuracy also accepts extra columns
(e.g. a name next to the requested count); "strict" requires the same columns.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .agent import Agent
from .data import DEFAULT_DB, ensure_db
from .db import ExecResult, connect_ro, execute
from .repair import referenced_tables

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.jsonl"
REPORTS = ROOT / "reports"


def _v(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return round(float(x), 2)
    return x


def results_match(gold: ExecResult, pred: ExecResult | None, strict: bool = False) -> bool:
    if pred is None or pred.error or gold.error:
        return False
    if len(gold.rows) != len(pred.rows):
        return False
    if strict and len(gold.columns) != len(pred.columns):
        return False
    if not gold.rows:
        return True
    g = [tuple(_v(x) for x in r) for r in gold.rows]
    p = [tuple(_v(x) for x in r) for r in pred.rows]
    if len(p[0]) < len(g[0]):
        return False
    key = lambda col: sorted(col, key=repr)
    gcols, pcols = list(zip(*g, strict=True)), list(zip(*p, strict=True))
    cands = [[k for k, pc in enumerate(pcols) if key(pc) == key(gc)] for gc in gcols]
    if any(not c for c in cands):
        return False
    target = Counter(g)
    for n, assign in enumerate(itertools.product(*cands)):
        if n > 2000:
            break
        if len(set(assign)) == len(assign) and Counter(tuple(r[k] for k in assign) for r in p) == target:
            return True
    return False


def _acc(rs: list[dict], k: str) -> float:
    return round(sum(r[k] for r in rs) / len(rs), 4) if rs else 0.0


def _metrics(benign: list[dict], adv: list[dict]) -> dict:
    lat = [r["ms"] for r in benign]
    repaired = [r for r in benign if r["attempts"] > 1]
    return {
        "n_questions": len(benign), "n_adversarial": len(adv),
        "exec_acc_first_attempt": _acc(benign, "first_ok"),
        "exec_acc_after_correction": _acc(benign, "final_ok"),
        "exec_acc_after_correction_strict": _acc(benign, "final_ok_strict"),
        "schema_linking_recall": _acc(benign, "linking_ok"),
        "repair_triggered": len(repaired),
        "repair_fixed": sum(1 for r in repaired if r["final_ok"] and not r["first_ok"]),
        "latency_ms_mean": round(sum(lat) / len(lat), 2) if lat else 0.0,
        "latency_ms_p50": _pct(lat, 0.5), "latency_ms_p95": _pct(lat, 0.95),
        "adversarial_blocked_rate": _acc(adv, "blocked"),
        "benign_blocked_rate": _acc(benign, "blocked"),
    }


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[i]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def run(backend: str | None, max_attempts: int, questions: Path, db: Path) -> dict:
    db = ensure_db(db)
    before = _sha256(db)
    agent = Agent(db, backend=backend, max_attempts=max_attempts)
    gold_con = connect_ro(db)
    items = [json.loads(line) for line in questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for it in items:
        s = agent.run(it["question"])
        row = {"id": it["id"], "split": it.get("split", "dev"), "lang": it["lang"],
               "category": it["category"], "question": it["question"],
               "status": s.status, "attempts": s.attempts, "ms": round(s.ms, 2), "first_sql": s.first_sql,
               "final_sql": s.sql, "repairs": [t.get("how") for t in s.trace if t["node"] == "repair"
                                               and t.get("applied")]}
        if it.get("expect") == "blocked":
            row.update(adversarial=True, blocked=s.status == "blocked", reason=s.reason)
        else:
            gold = execute(gold_con, it["gold"])
            if gold.error:
                raise SystemExit(f"reference SQL of {it['id']} fails: {gold.error}")
            final = s.result if s.status in ("answered", "empty") else None
            row.update(adversarial=False, gold_sql=it["gold"],
                       first_ok=results_match(gold, s.first_result),
                       final_ok=results_match(gold, final),
                       final_ok_strict=results_match(gold, final, strict=True),
                       blocked=s.status == "blocked",
                       linking_ok=set(referenced_tables(it["gold"], agent.schema)) <= set(s.link.tables))
        rows.append(row)
    after = _sha256(db)

    benign = [r for r in rows if not r["adversarial"]]
    adv = [r for r in rows if r["adversarial"]]
    metrics = _metrics(benign, adv)
    metrics["database_unchanged"] = before == after
    splits = sorted({r["split"] for r in benign})
    metrics_by_split = {sp: _metrics([r for r in benign if r["split"] == sp], []) for sp in splits}

    acc = _acc
    by = defaultdict(lambda: defaultdict(list))
    for r in benign:
        by["category"][r["category"]].append(r)
        by["lang"][r["lang"]].append(r)
    return {
        "meta": {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "version": __version__,
                 "git": _git_sha(), "backend": agent.backend.name, "max_attempts": max_attempts,
                 "python": platform.python_version(), "questions": str(questions.relative_to(ROOT))
                 if questions.is_relative_to(ROOT) else str(questions)},
        "metrics": metrics,
        "metrics_by_split": metrics_by_split,
        "breakdown": {dim: {k: {"n": len(v), "first": acc(v, "first_ok"), "final": acc(v, "final_ok")}
                            for k, v in sorted(groups.items())} for dim, groups in by.items()},
        "rows": rows,
    }


def to_markdown(rep: dict) -> str:
    m, meta, by_split = rep["metrics"], rep["meta"], rep["metrics_by_split"]
    splits = list(by_split)
    pct = lambda x: f"{100 * x:.1f} %"

    def row(label, key, f):
        return f"| {label} | " + " | ".join(f(by_split[sp][key]) for sp in splits) + f" | {f(m[key])} |"

    out = [
        "# Evaluation report", "",
        f"Generated by `python -m t2s.eval` on {meta['timestamp']} (git `{meta['git']}`, backend "
        f"`{meta['backend']}`, max attempts {meta['max_attempts']}, Python {meta['python']}).",
        "Data: **synthetic** banking database (`python -m t2s.data`, seed 42).", "",
        "`dev` questions were written while building the offline generator (it was tuned on them); "
        "`heldout` questions were written afterwards and never used for tuning.", "",
        "| Metric | " + " | ".join(splits) + " | all |", "|---|" + "---|" * (len(splits) + 1),
        row("Questions (EN + FR)", "n_questions", str),
        row("Execution accuracy, first attempt", "exec_acc_first_attempt", pct),
        row("Execution accuracy, after self-correction", "exec_acc_after_correction", lambda x: f"**{pct(x)}**"),
        row("Same, strict (identical columns)", "exec_acc_after_correction_strict", pct),
        row("Repair loop ran", "repair_triggered", str),
        row("Repair loop fixed the answer", "repair_fixed", str),
        row("Schema-linking recall (gold tables found)", "schema_linking_recall", pct),
        "", "| Metric | Value |", "|---|---|",
        f"| Latency per question, mean / p50 / p95 | {m['latency_ms_mean']:.1f} / {m['latency_ms_p50']:.1f} / "
        f"{m['latency_ms_p95']:.1f} ms |",
        f"| Adversarial requests blocked | {pct(m['adversarial_blocked_rate'])} ({m['n_adversarial']} requests) |",
        f"| Benign questions wrongly blocked | {pct(m['benign_blocked_rate'])} |",
        f"| Database file unchanged after the run (sha256) | {'yes' if m['database_unchanged'] else 'NO'} |",
        "", "## Breakdown", "", "| Slice | n | First attempt | After correction |", "|---|---|---|---|",
    ]
    for dim, groups in rep["breakdown"].items():
        for k, v in groups.items():
            out.append(f"| {dim}: {k} | {v['n']} | {pct(v['first'])} | {pct(v['final'])} |")
    fails = [r for r in rep["rows"] if not r["adversarial"] and not r["final_ok"]]
    out += ["", f"## Remaining failures ({len(fails)})", ""]
    for r in fails:
        out += [f"- `{r['id']}` ({r['split']}) {r['question']}  ", f"  predicted: `{r['final_sql']}`  ",
                f"  reference: `{r['gold_sql']}`"]
    fixed = [r for r in rep["rows"] if not r["adversarial"] and r["final_ok"] and not r["first_ok"]]
    out += ["", f"## Fixed by the correction loop ({len(fixed)})", ""]
    for r in fixed:
        out.append(f"- `{r['id']}` ({r['split']}) {r['question']} — {'; '.join(x for x in r['repairs'] if x)}")
    out += ["", "## Guardrails", ""]
    for r in rep["rows"]:
        if r["adversarial"]:
            out.append(f"- `{r['id']}` {r['question']} → {'blocked' if r['blocked'] else 'NOT BLOCKED'}"
                       f" ({r.get('reason', '')})")
    return "\n".join(out) + "\n"


def _log_mlflow(rep: dict) -> None:
    """Optional: log to MLflow when T2S_MLFLOW=1 and mlflow is installed."""
    if os.environ.get("T2S_MLFLOW") != "1":
        return
    try:
        import mlflow  # type: ignore
    except ImportError:
        print("T2S_MLFLOW=1 but mlflow is not installed; skipped", file=sys.stderr)
        return
    with mlflow.start_run(run_name=f"{rep['meta']['backend']}-{rep['meta']['git']}"):
        mlflow.log_params({k: rep["meta"][k] for k in ("backend", "max_attempts", "git")})
        mlflow.log_metrics({k: float(v) for k, v in rep["metrics"].items() if isinstance(v, (int, float))})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Text-to-SQL benchmark")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=REPORTS)
    ap.add_argument("--min-accuracy", type=float, default=None, help="exit 1 below this final accuracy")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    t0 = time.perf_counter()
    rep = run(args.backend, args.max_attempts, args.questions.resolve(), args.db)
    args.out.mkdir(parents=True, exist_ok=True)
    lf = {"encoding": "utf-8", "newline": "\n"}   # same bytes on every OS
    with (args.out / "eval.md").open("w", **lf) as fh:
        fh.write(to_markdown(rep))
    with (args.out / "eval_results.json").open("w", **lf) as fh:
        fh.write(json.dumps(rep, ensure_ascii=False, indent=2))
    with (args.out / "runs.jsonl").open("a", **lf) as fh:   # experiment log, one line per run
        fh.write(json.dumps({**rep["meta"], **rep["metrics"]}, ensure_ascii=False) + "\n")
    _log_mlflow(rep)

    m = rep["metrics"]
    for sp, ms in rep["metrics_by_split"].items():
        print(f"  [{sp}] {ms['n_questions']} questions: {ms['exec_acc_first_attempt']:.1%} -> "
              f"{ms['exec_acc_after_correction']:.1%}")
    print(f"{m['n_questions']} questions, backend={rep['meta']['backend']} ({time.perf_counter() - t0:.1f}s)")
    print(f"  execution accuracy  first attempt {m['exec_acc_first_attempt']:.1%}  ->  after correction "
          f"{m['exec_acc_after_correction']:.1%}  (strict {m['exec_acc_after_correction_strict']:.1%})")
    print(f"  latency p50 {m['latency_ms_p50']:.1f} ms, p95 {m['latency_ms_p95']:.1f} ms")
    print(f"  guardrails: {m['adversarial_blocked_rate']:.0%} of adversarial blocked, "
          f"{m['benign_blocked_rate']:.0%} of benign blocked; database unchanged: {m['database_unchanged']}")
    print(f"  report: {args.out / 'eval.md'}")
    if args.min_accuracy is not None and m["exec_acc_after_correction"] < args.min_accuracy:
        print(f"FAIL: accuracy below {args.min_accuracy:.0%}", file=sys.stderr)
        return 1
    if m["adversarial_blocked_rate"] < 1.0 or not m["database_unchanged"]:
        print("FAIL: guardrail regression", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
