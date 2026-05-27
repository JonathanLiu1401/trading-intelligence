"""Regression pin for the c65e39d worker-sleep regression.

The ``StockTwits trending symbols rank detector`` commit (c65e39d) inserted
the new ``stocktwits_trending_symbols_worker`` *between*
``stocktwits_sentiment_worker``'s body and the worker's trailing ``_sleep(300)``
call. The diff cleanly moved the new function declaration into place but the
``_sleep(300)`` line that previously closed ``stocktwits_sentiment_worker``
ended up appended *after* the new worker's own ``_sleep(300)``, leaving:

  * ``stocktwits_sentiment_worker`` — NO ``_sleep`` after the success path
    (hot CPU loop, hammering the cursor file + the source_health record
    on every iteration);
  * ``stocktwits_trending_symbols_worker`` — TWO consecutive ``_sleep(300)``
    calls, doubling the intended 5-min cadence to 10 min and silently
    halving freshness on the trending-rank signal.

Both are real bugs in HEAD. This test pins the fix structurally so a future
edit-merge that reintroduces the same pattern fails CI immediately.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


_DAEMON_PATH = Path(__file__).resolve().parent.parent / "daemon.py"


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in daemon.py")


def _count_top_level_sleep_calls(fn: ast.FunctionDef) -> int:
    """Count ``_sleep(...)`` calls that are direct top-level statements of
    the worker's ``while`` loop body (not nested inside any conditional /
    try / except).

    A worker's per-cycle pacing sleep lives at the END of the
    ``while _running:`` loop body — same place every collector worker in
    daemon.py puts it. The bug we're pinning is a missing/duplicated
    top-level _sleep; the scorer/alert workers deliberately put theirs
    behind conditionals (see ``_count_any_sleep_calls`` for those)."""
    n = 0
    for body_stmt in fn.body:
        if not isinstance(body_stmt, ast.While):
            continue
        for stmt in body_stmt.body:
            if not isinstance(stmt, ast.Expr):
                continue
            call = stmt.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id == "_sleep":
                    n += 1
    return n


def _count_any_sleep_calls(fn: ast.FunctionDef) -> int:
    """Count ``_sleep(...)`` calls anywhere inside the worker's body — at
    any nesting level. This catches a worker whose pacing sleep is nested
    behind an ``if remaining == 0:`` (like ``scorer_worker``) while still
    failing the regression case where the call was deleted entirely
    (c65e39d stripped the lone top-level _sleep from
    ``stocktwits_sentiment_worker`` leaving ZERO _sleep calls in the
    function)."""
    n = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_sleep":
                n += 1
    return n


@pytest.fixture(scope="module")
def daemon_tree() -> ast.Module:
    src = _DAEMON_PATH.read_text(encoding="utf-8")
    return ast.parse(src, filename=str(_DAEMON_PATH))


def test_stocktwits_sentiment_worker_has_pacing_sleep(daemon_tree):
    """The success path of ``stocktwits_sentiment_worker`` MUST end with a
    ``_sleep(...)`` at the top level of the while-loop body.

    If this assertion fails, the worker hot-loops on every cycle — the c65e39d
    regression. Mirrors how every other ``*_worker`` in daemon.py is
    structured."""
    fn = _find_function(daemon_tree, "stocktwits_sentiment_worker")
    n = _count_top_level_sleep_calls(fn)
    assert n >= 1, (
        "stocktwits_sentiment_worker is missing its per-cycle _sleep(...) — "
        "the worker will hot-loop, hammering the StockTwits API and the "
        "cursor file (c65e39d-class regression)."
    )


def test_stocktwits_trending_symbols_worker_has_exactly_one_pacing_sleep(
    daemon_tree,
):
    """``stocktwits_trending_symbols_worker`` MUST have exactly ONE pacing
    ``_sleep(...)`` at the top level of its while-loop body. Two would
    silently double the cadence to 10 min — exactly the c65e39d regression."""
    fn = _find_function(daemon_tree, "stocktwits_trending_symbols_worker")
    n = _count_top_level_sleep_calls(fn)
    assert n == 1, (
        f"stocktwits_trending_symbols_worker has {n} top-level _sleep() "
        f"calls; expected exactly 1. Two consecutive _sleep(300) calls "
        f"silently double the trending-symbol poll cadence to 10 min — "
        f"freshness halved on the rank signal (c65e39d regression)."
    )


def test_every_collector_worker_has_a_pacing_sleep(daemon_tree):
    """Every ``*_worker`` function declared at module level in daemon.py must
    have at least one top-level ``_sleep(...)`` in its while-loop body.

    Catches the c65e39d-class regression at large: an edit-merge that
    accidentally strips the pacing sleep from a worker silently turns it
    into a hot loop. Excludes the supervisory ``_worker`` helpers (they
    aren't workers — they're worker plumbing) and the few infrastructure
    workers that are intentionally driven entirely by event loops or
    blocking calls (web_server_worker / heartbeat_worker style)."""
    INFRASTRUCTURE_WHITELIST = {
        "web_server_worker",  # blocks in Flask.run forever
    }
    failures: list[str] = []
    for node in daemon_tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.endswith("_worker"):
            continue
        if node.name in INFRASTRUCTURE_WHITELIST:
            continue
        # Skip helper functions like _worker_health_report / _wrap_worker /
        # _worker_liveness_deadline / _worker_health_snapshot — they don't
        # follow the while-loop worker pattern.
        if node.name.startswith("_"):
            continue
        # The function MUST contain at least one While loop with a
        # ``_sleep(...)`` call SOMEWHERE in its body — top level or
        # conditional. The scorer/alert workers deliberately gate their
        # pacing sleep behind progress checks; only an entirely-missing
        # _sleep means "hot loop", which is what we're catching.
        has_while = any(isinstance(s, ast.While) for s in node.body)
        if not has_while:
            continue
        if _count_any_sleep_calls(node) < 1:
            failures.append(node.name)
    assert not failures, (
        f"workers missing per-cycle pacing _sleep: {failures}. "
        f"Each will hot-loop — same regression class as c65e39d on "
        f"stocktwits_sentiment_worker."
    )
