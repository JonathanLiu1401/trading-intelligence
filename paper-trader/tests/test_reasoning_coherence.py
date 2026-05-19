"""Tests for analytics/reasoning_coherence.py — across-time stability of
Opus's HOLD justification.

Pins:
  * pure / no I/O (degrades to NO_DATA on bad input shape, never raises)
  * extracts the standard ``{"decision": {"reasoning": "..."}}`` envelope
  * tolerates parse_failed: / retry_failed: prefixes from strategy.py
  * NO_DATA / INSUFFICIENT / OK ladder
  * regime thresholds (STABLE_THESIS / DRIFTING / RAPID_DRIFT) match the
    documented bands (≥0.60 / ≥0.30 / <0.30 by median Jaccard)
  * HOLD filter ignores NO_DECISION, BLOCKED, FILLED rows
  * pairwise Jaccard over content tokens (stopwords + len<3 dropped)
  * route exists, returns JSON, clamps ``limit`` (5..500), no @swr_cached
    (lock pinned)
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.reasoning_coherence import (
    DRIFTING_THRESHOLD,
    MIN_PAIRS_FOR_VERDICT,
    STABLE_THRESHOLD,
    build_reasoning_coherence,
)


def _hold(reasoning_text: str | None, ts="2026-05-19T00:00:00+00:00",
          ticker="NVDA"):
    """One decisions-table row in the shape ``recent_decisions`` returns,
    HOLD action with reasoning stored as the live JSON envelope shape."""
    if reasoning_text is None:
        blob = None
    else:
        blob = json.dumps({"decision": {
            "action": "HOLD", "ticker": ticker, "qty": 0,
            "confidence": 0.7, "reasoning": reasoning_text,
        }})
    return {
        "id": 1, "timestamp": ts, "action_taken": f"HOLD {ticker} → HOLD",
        "reasoning": blob, "portfolio_value": 1000.0, "cash": 500.0,
        "market_open": 0, "signal_count": 5,
    }


def _no_decision(ts="2026-05-19T00:00:00+00:00"):
    return {
        "id": 1, "timestamp": ts, "action_taken": "NO_DECISION",
        "reasoning": "skipped claude call — host saturated",
        "portfolio_value": 1000.0, "cash": 500.0,
        "market_open": 0, "signal_count": 5,
    }


def _filled(ts="2026-05-19T00:00:00+00:00"):
    return {
        "id": 1, "timestamp": ts, "action_taken": "BUY NVDA → FILLED",
        "reasoning": json.dumps({"decision": {"reasoning": "earnings beat"}}),
        "portfolio_value": 1000.0, "cash": 500.0,
        "market_open": 0, "signal_count": 5,
    }


# ── shape / degradation ladder ───────────────────────────────────────────────


def test_none_input_returns_no_data():
    out = build_reasoning_coherence(None)
    assert out["state"] == "NO_DATA"
    assert out["n_hold_decisions"] == 0
    assert out["n_pairs"] == 0
    assert out["pairs"] == []
    assert out["regime"] is None


def test_empty_list_returns_no_data():
    assert build_reasoning_coherence([])["state"] == "NO_DATA"


def test_only_non_hold_decisions_returns_no_data():
    out = build_reasoning_coherence([_no_decision(), _filled()])
    assert out["state"] == "NO_DATA"
    assert out["n_hold_decisions"] == 0


def test_single_hold_returns_insufficient_with_one_reasoned_zero_pairs():
    """A solo HOLD reasoning cannot pair with anything → INSUFFICIENT, not OK."""
    out = build_reasoning_coherence([_hold("NVDA earnings catalyst soon")])
    assert out["state"] == "INSUFFICIENT"
    assert out["n_hold_decisions"] == 1
    assert out["n_pairs"] == 0


def test_two_holds_emit_one_pair_but_still_insufficient_for_verdict():
    """Below MIN_PAIRS_FOR_VERDICT (=3) the regime is withheld."""
    decs = [
        _hold("NVDA earnings catalyst", ts="2026-05-19T00:00:00+00:00"),
        _hold("NVDA earnings catalyst", ts="2026-05-19T00:01:00+00:00"),
    ]
    out = build_reasoning_coherence(decs)
    assert out["state"] == "INSUFFICIENT"
    assert out["n_pairs"] == 1
    assert out["regime"] is None
    assert out["median_similarity"] == 1.0  # raw stat still emitted


# ── regime verdicts ──────────────────────────────────────────────────────────


def test_identical_reasonings_yield_stable_thesis():
    txt = "NVDA earnings in 0.8 days hold through print binary catalyst"
    decs = [_hold(txt, ts=f"2026-05-19T00:0{i}:00+00:00")
            for i in range(5)]
    out = build_reasoning_coherence(decs)
    assert out["state"] == "OK"
    assert out["regime"] == "STABLE_THESIS"
    assert out["median_similarity"] == 1.0
    assert out["n_pairs"] == 4


def test_completely_disjoint_reasonings_yield_rapid_drift():
    """Five HOLDs, each citing different content → near-zero Jaccard pairs."""
    decs = [
        _hold("alpha beta gamma delta epsilon zeta eta", ts="2026-05-19T00:00:00+00:00"),
        _hold("china tariffs imposed beijing washington tension", ts="2026-05-19T00:01:00+00:00"),
        _hold("portfolio paralysis cash burn rate liquidity", ts="2026-05-19T00:02:00+00:00"),
        _hold("rsi macd bollinger overbought technical squeeze", ts="2026-05-19T00:03:00+00:00"),
        _hold("dividend payout ratio yield aristocrats banking", ts="2026-05-19T00:04:00+00:00"),
    ]
    out = build_reasoning_coherence(decs)
    assert out["state"] == "OK"
    assert out["regime"] == "RAPID_DRIFT"
    assert out["median_similarity"] < DRIFTING_THRESHOLD


def test_partial_overlap_yields_drifting():
    """Pairs that share ~half their content tokens → DRIFTING band."""
    decs = [
        _hold("nvda earnings catalyst binary hold print today", ts="2026-05-19T00:00:00+00:00"),
        _hold("nvda earnings macro fed today catalyst pivot", ts="2026-05-19T00:01:00+00:00"),
        _hold("nvda macro fed pivot inflation cooling print", ts="2026-05-19T00:02:00+00:00"),
        _hold("nvda inflation cooling pivot rates hold today", ts="2026-05-19T00:03:00+00:00"),
    ]
    out = build_reasoning_coherence(decs)
    assert out["state"] == "OK"
    assert out["regime"] == "DRIFTING"
    assert DRIFTING_THRESHOLD <= out["median_similarity"] < STABLE_THRESHOLD


# ── input-shape tolerance ────────────────────────────────────────────────────


def test_unparseable_json_reasoning_is_silently_skipped():
    decs = [
        _hold("nvda earnings binary catalyst hold print"),
        {"id": 2, "timestamp": "2026-05-19T00:01:00+00:00",
         "action_taken": "HOLD NVDA → HOLD",
         "reasoning": "not json at all just free text",
         "portfolio_value": 1000.0, "cash": 500.0,
         "market_open": 0, "signal_count": 5},
        _hold("nvda earnings binary catalyst hold print"),
    ]
    # n_hold_decisions counts ALL three HOLDs; reasoned drops the middle one.
    out = build_reasoning_coherence(decs)
    assert out["n_hold_decisions"] == 3
    # Only 2 parseable HOLDs => 1 pair => INSUFFICIENT.
    assert out["n_pairs"] == 1
    assert out["state"] == "INSUFFICIENT"


def test_parse_failed_prefix_is_stripped_before_json_parse():
    """strategy.py records ``parse_failed:`` + raw response on JSON failure."""
    raw = json.dumps({"decision": {"reasoning": "earnings hold catalyst"}})
    decs = [
        _hold("earnings hold catalyst"),
        {"id": 2, "timestamp": "2026-05-19T00:01:00+00:00",
         "action_taken": "HOLD NVDA → HOLD",
         "reasoning": "parse_failed: " + raw,
         "portfolio_value": 1000.0, "cash": 500.0,
         "market_open": 0, "signal_count": 5},
        _hold("earnings hold catalyst"),
        _hold("earnings hold catalyst"),
    ]
    out = build_reasoning_coherence(decs)
    assert out["state"] == "OK"
    # All four parsed → median similarity 1.0 → STABLE
    assert out["regime"] == "STABLE_THESIS"


def test_no_decision_rows_do_not_become_holds():
    """NO_DECISION (host_saturated etc.) must not be treated as HOLD."""
    decs = [
        _hold("real hold reasoning here today"),
        _no_decision(ts="2026-05-19T00:01:00+00:00"),
        _hold("real hold reasoning here today"),
    ]
    out = build_reasoning_coherence(decs)
    assert out["n_hold_decisions"] == 2  # only the two real HOLDs


def test_filled_buy_rows_do_not_become_holds():
    decs = [
        _hold("entered position because pivotal catalyst"),
        _filled(ts="2026-05-19T00:01:00+00:00"),
        _hold("entered position because pivotal catalyst"),
    ]
    out = build_reasoning_coherence(decs)
    assert out["n_hold_decisions"] == 2  # the BUY FILLED is not a HOLD


def test_input_list_not_mutated():
    src = [_hold("alpha beta gamma delta"), _hold("alpha beta gamma delta")]
    snapshot = [dict(r) for r in src]
    build_reasoning_coherence(src)
    assert src == snapshot


def test_never_raises_on_garbage_input():
    """Defensive: any field missing / wrong-typed must degrade, not raise."""
    decs = [
        {"action_taken": None, "reasoning": None},
        {"action_taken": "HOLD", "reasoning": object()},
        {"action_taken": "HOLD", "reasoning": ""},
        {},
        None,  # well, list type-strict — skip None members in caller? Test it:
    ]
    out = build_reasoning_coherence([r for r in decs if r is not None])
    assert out["state"] == "NO_DATA"


def test_no_llm_no_subprocess_purity():
    """Builder must never call out."""
    src = inspect.getsource(build_reasoning_coherence)
    assert "subprocess" not in src
    assert "claude_call" not in src
    assert "requests" not in src
    assert "yfinance" not in src
    assert "sqlite3" not in src


# ── Flask route surface ──────────────────────────────────────────────────────


def _client():
    import importlib
    import sys as _sys
    # paper_trader.dashboard imports a module-level Flask app at import time.
    # Reuse it for test_client; create_app pattern doesn't apply here.
    _sys.modules.pop("paper_trader.dashboard", None)
    d = importlib.import_module("paper_trader.dashboard")
    d.app.testing = True
    return d, d.app.test_client()


def test_route_returns_json_envelope(monkeypatch):
    d, c = _client()

    class _FakeStore:
        def recent_decisions(self, limit=20):
            return [_hold("alpha beta gamma delta epsilon")
                    for _ in range(5)]

    monkeypatch.setattr(d, "get_store", lambda: _FakeStore())
    r = c.get("/api/reasoning-coherence")
    assert r.status_code == 200
    data = r.get_json()
    assert data["state"] == "OK"
    assert data["regime"] == "STABLE_THESIS"
    assert data["window_limit"] == 100


def test_route_clamps_limit(monkeypatch):
    d, c = _client()
    captured = {}

    class _FakeStore:
        def recent_decisions(self, limit=20):
            captured["limit"] = limit
            return []

    monkeypatch.setattr(d, "get_store", lambda: _FakeStore())

    assert c.get("/api/reasoning-coherence?limit=0").get_json()["window_limit"] == 5
    assert captured["limit"] == 5

    assert c.get("/api/reasoning-coherence?limit=99999").get_json()["window_limit"] == 500
    assert captured["limit"] == 500

    assert c.get("/api/reasoning-coherence?limit=abc").get_json()["window_limit"] == 100
    assert captured["limit"] == 100


def test_route_degrades_on_store_failure(monkeypatch):
    d, c = _client()

    class _BrokenStore:
        def recent_decisions(self, limit=20):
            raise RuntimeError("db gone")

    monkeypatch.setattr(d, "get_store", lambda: _BrokenStore())
    r = c.get("/api/reasoning-coherence")
    # Endpoint wraps the call in try/except; surfaces error JSON not 500.
    body = r.get_json()
    assert r.status_code == 500
    assert "error" in body


def test_route_not_behind_swr_cached():
    """Cheap endpoint (recent_decisions + pure builder, no yfinance/no LLM).
    No ``@swr_cached`` decorator means no obligation to update the prewarm
    coverage list (tests/test_swr_prewarm_coverage.py)."""
    d, _ = _client()
    src = inspect.getsource(d.reasoning_coherence_api)
    # ``@swr_cached`` would wrap the function; the unwrapped source contains
    # the decorator name when it IS present (decorators are above the def in
    # getsource for module-level wrappers). The route is intentionally
    # decorator-free apart from @app.route.
    # Reading the source of the function does not include outer decorators,
    # so we check the dashboard module source for the cache-list mention.
    full = inspect.getsource(d)
    assert '@swr_cached("reasoning-coherence"' not in full
