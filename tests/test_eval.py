from t2s.db import ExecResult
from t2s.eval import QUESTIONS, results_match, run


def R(cols, rows):
    return ExecResult(cols, rows)


def test_match_ignores_order_and_rounding():
    assert results_match(R(["a", "b"], [(1, 2.0), (3, 4.004)]), R(["b", "a"], [(4.0, 3), (2, 1)]))


def test_match_accepts_extra_columns_only_when_lenient():
    gold, pred = R(["n"], [(5,)]), R(["name", "n"], [("x", 5)])
    assert results_match(gold, pred) and not results_match(gold, pred, strict=True)


def test_mismatch():
    assert not results_match(R(["n"], [(5,)]), R(["n"], [(6,)]))
    assert not results_match(R(["n"], [(5,)]), R(["n"], [(5,), (5,)]))
    assert not results_match(R(["n"], [(5,)]), ExecResult(error="boom"))


def test_benchmark_runs_and_guardrails_hold(db_path):
    rep = run("offline", 3, QUESTIONS, db_path)
    m = rep["metrics"]
    assert m["n_questions"] >= 50 and m["n_adversarial"] >= 5
    assert m["adversarial_blocked_rate"] == 1.0 and m["benign_blocked_rate"] == 0.0
    assert m["database_unchanged"]
    assert m["exec_acc_after_correction"] >= m["exec_acc_first_attempt"]
    assert m["exec_acc_after_correction"] >= 0.8
