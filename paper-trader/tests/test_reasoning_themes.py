"""Tests for analytics/reasoning_themes.py — top phrases across recent
Opus decision reasonings.

Pins:
  * pure / never raises on garbage (None/missing/non-JSON/wrong shape)
  * NO_DATA when zero rows yield reasoning text; OK otherwise
  * leaderboard ranks by decisions_mentioning, NOT raw mention count
    (the breadth-not-loudness contract — a phrase 30x in one row beats
    nothing across 12 different rows)
  * bigrams compete with unigrams in one list; bigram wins on a tie
    (informativeness tie-break)
  * stopwords + len<3 dropped from BOTH unigrams and the bigram chain
  * tolerates JSON envelope shape, top-level reasoning, parse_failed:
    prefix, and bare-string NO_DECISION timeout text (themable corpus)
  * ``share_of_decisions`` is decisions_with_text-relative, not n_rows
  * route exists, clamps limit (5..500) and top_k (3..50), is not @swr_cached
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.reasoning_themes import (
    build_reasoning_themes,
)


def _hold(text: str | None, ts="2026-05-19T00:00:00+00:00", ticker="NVDA",
          confidence=0.7):
    if text is None:
        blob = None
    else:
        blob = json.dumps({"decision": {
            "action": "HOLD", "ticker": ticker, "qty": 0,
            "confidence": confidence, "reasoning": text,
        }})
    return {
        "id": 1, "timestamp": ts, "action_taken": f"HOLD {ticker} → HOLD",
        "reasoning": blob, "portfolio_value": 1000.0, "cash": 500.0,
        "market_open": 0, "signal_count": 5,
    }


def _no_decision(text="claude returned no response (timeout/empty)",
                 ts="2026-05-19T00:00:00+00:00"):
    return {
        "id": 1, "timestamp": ts, "action_taken": "NO_DECISION",
        "reasoning": text, "portfolio_value": 1000.0, "cash": 500.0,
        "market_open": 0, "signal_count": 5,
    }


# ── shape / degradation ladder ───────────────────────────────────────────────


def test_none_input_returns_no_data():
    out = build_reasoning_themes(None)
    assert out["state"] == "NO_DATA"
    assert out["n_decisions"] == 0
    assert out["n_with_reasoning"] == 0
    assert out["themes"] == []
    assert "No reasoning" in out["headline"]


def test_empty_list_returns_no_data():
    out = build_reasoning_themes([])
    assert out["state"] == "NO_DATA"
    assert out["n_decisions"] == 0


def test_all_unparseable_returns_no_data():
    rows = [
        {"reasoning": None, "action_taken": "NO_DECISION",
         "timestamp": "2026-05-19T00:00:00+00:00"},
        {"reasoning": "", "action_taken": "NO_DECISION",
         "timestamp": "2026-05-19T00:01:00+00:00"},
    ]
    out = build_reasoning_themes(rows)
    assert out["state"] == "NO_DATA"
    assert out["n_unparseable"] == 2
    assert out["themes"] == []


def test_garbage_row_keys_do_not_raise():
    """Mixed-shape rows must not crash. The non-string ``12345`` reasoning
    coerces to its string repr and may produce a low-quality numeric
    theme, which is acceptable — the contract is "never raise", not
    "always NO_DATA on malformed input"."""
    rows = [
        {"unexpected": "row", "no_reasoning_or_action": True},
        {"reasoning": None, "action_taken": None},
        {},
    ]
    out = build_reasoning_themes(rows)
    assert out["state"] in {"NO_DATA", "OK"}  # never raises
    assert out["n_decisions"] == 3


# ── reasoning extraction across the three real shapes ──────────────────────


def test_extracts_envelope_reasoning_text():
    # Disable bigrams here so the unigram leaderboard isn't crowded out.
    rows = [_hold("Earnings premium in NVDA ahead of print")]
    out = build_reasoning_themes(rows, top_k=10, include_bigrams=False)
    assert out["state"] == "OK"
    phrases = {t["phrase"] for t in out["themes"]}
    # Stopwords (in, of) gone; len>=3 content tokens present.
    assert "earnings" in phrases
    assert "premium" in phrases
    assert "nvda" in phrases
    assert "in" not in phrases
    assert "of" not in phrases


def test_extracts_top_level_reasoning_key():
    """A decision blob without the ``{"decision": {...}}`` wrap — bare
    top-level ``reasoning`` — still mines."""
    blob = json.dumps({"reasoning": "memory super cycle thesis intact"})
    rows = [{
        "reasoning": blob, "action_taken": "HOLD",
        "timestamp": "2026-05-19T00:00:00+00:00",
    }]
    out = build_reasoning_themes(rows)
    assert out["state"] == "OK"
    phrases = {t["phrase"] for t in out["themes"]}
    assert "memory" in phrases
    assert "super" in phrases


def test_tolerates_parse_failed_prefix():
    """``strategy._parse_decision`` writes ``parse_failed: <raw>`` when JSON
    extraction fails. The raw payload is still themable prose."""
    rows = [{
        "reasoning": (
            "parse_failed: NVDA concentration concentration concentration"
        ),
        "action_taken": "NO_DECISION",
        "timestamp": "2026-05-19T00:00:00+00:00",
    }]
    out = build_reasoning_themes(rows)
    assert out["state"] == "OK"
    top = out["themes"][0]
    assert top["phrase"] in {"concentration", "nvda concentration",
                             "concentration concentration"}


def test_no_decision_timeout_strings_are_themable():
    """NO_DECISION rows carry raw prose like ``"claude returned no response
    (timeout/empty)"`` — this is itself a recurring theme (host saturation
    pattern) that the operator wants to see surfaced."""
    rows = [_no_decision() for _ in range(4)]
    out = build_reasoning_themes(rows, top_k=10, include_bigrams=False)
    assert out["state"] == "OK"
    phrases = {t["phrase"] for t in out["themes"]}
    assert "claude" in phrases  # the leading content word
    assert "response" in phrases
    assert "timeout" in phrases


# ── ranking semantics: breadth (decisions_mentioning), not loudness ─────


def test_breadth_beats_loudness():
    """A phrase repeated 30x in ONE decision must NOT outrank a phrase
    that recurs across many decisions. Discriminator: 1 verbose NVDA row
    vs 5 mentions of 'earnings' across 5 separate rows.

    Construct: row A repeats "alpha" 30 times in one reasoning. Rows B-F
    each mention "earnings" exactly once. After ranking, 'earnings' must
    come above 'alpha' because 5 > 1 on decisions_mentioning even though
    total_mentions for 'alpha' is far higher.
    """
    rows = [
        _hold("alpha " * 30, ts="2026-05-19T00:00:00+00:00"),
        _hold("earnings risk", ts="2026-05-19T00:01:00+00:00"),
        _hold("earnings call ahead", ts="2026-05-19T00:02:00+00:00"),
        _hold("earnings beat expected", ts="2026-05-19T00:03:00+00:00"),
        _hold("earnings premium intact", ts="2026-05-19T00:04:00+00:00"),
        _hold("earnings revision risk", ts="2026-05-19T00:05:00+00:00"),
    ]
    out = build_reasoning_themes(rows, top_k=5, include_bigrams=False)
    leaderboard = [t["phrase"] for t in out["themes"]]
    assert leaderboard[0] == "earnings"
    earnings = next(t for t in out["themes"] if t["phrase"] == "earnings")
    alpha = next(t for t in out["themes"] if t["phrase"] == "alpha")
    assert earnings["decisions_mentioning"] == 5
    assert alpha["decisions_mentioning"] == 1
    assert alpha["total_mentions"] == 30  # loudness preserved as field
    assert earnings["total_mentions"] == 5


def test_bigram_beats_unigram_on_tie():
    """When a unigram and a bigram have the same decisions_mentioning AND
    same total_mentions, the bigram wins (informativeness tie-break)."""
    rows = [
        _hold("super cycle", ts="2026-05-19T00:00:00+00:00"),
        _hold("super cycle", ts="2026-05-19T00:01:00+00:00"),
        _hold("super cycle", ts="2026-05-19T00:02:00+00:00"),
    ]
    out = build_reasoning_themes(rows, top_k=5)
    # "super cycle" (bigram) AND "super" (unigram) AND "cycle" (unigram)
    # all have decisions_mentioning=3, total_mentions=3.
    # Bigram must come first.
    leaderboard = [t["phrase"] for t in out["themes"]]
    assert leaderboard[0] == "super cycle"
    assert out["themes"][0]["is_bigram"] is True


def test_include_bigrams_false_suppresses_bigrams():
    rows = [_hold("super cycle thesis") for _ in range(3)]
    out = build_reasoning_themes(rows, top_k=10, include_bigrams=False)
    for t in out["themes"]:
        assert " " not in t["phrase"]
        assert t["is_bigram"] is False


# ── filter semantics ───────────────────────────────────────────────────────


def test_stopwords_are_dropped():
    rows = [_hold("the and for with the and for with")]
    out = build_reasoning_themes(rows)
    # Every token here is a stopword and every bigram is stopword+stopword
    # → no content. NO_DATA, not OK.
    assert out["state"] == "NO_DATA"


def test_short_tokens_dropped():
    rows = [_hold("AI ML BI a b")]
    out = build_reasoning_themes(rows)
    # All <3 chars → no content tokens.
    assert out["state"] == "NO_DATA"


# ── share_of_decisions semantics ──────────────────────────────────────────


def test_share_is_decisions_with_text_relative():
    """share_of_decisions denominator is decisions WITH PARSEABLE TEXT,
    not raw row count. An unparseable row in the same window must NOT
    dilute the share of a phrase that appears in every text-bearing row.
    """
    rows = [
        _hold("alpha", ts="2026-05-19T00:00:00+00:00"),
        _hold("alpha", ts="2026-05-19T00:01:00+00:00"),
        {"reasoning": None, "action_taken": "NO_DECISION",
         "timestamp": "2026-05-19T00:02:00+00:00"},  # unparseable
    ]
    out = build_reasoning_themes(rows)
    assert out["n_decisions"] == 3
    assert out["n_with_reasoning"] == 2
    assert out["n_unparseable"] == 1
    alpha = next(t for t in out["themes"] if t["phrase"] == "alpha")
    assert alpha["decisions_mentioning"] == 2
    assert alpha["share_of_decisions"] == 1.0


# ── example excerpt ──────────────────────────────────────────────────────


def test_example_contains_phrase_or_falls_back():
    rows = [_hold("Memory super cycle thesis intact for NVDA")]
    out = build_reasoning_themes(rows)
    super_cycle = next(t for t in out["themes"]
                       if t["phrase"] == "super cycle")
    # Phrase found in raw text → excerpt contains it (case-insensitive).
    assert "super cycle" in super_cycle["example"].lower()


# ── top_k clamp ───────────────────────────────────────────────────────────


def test_top_k_clamps_high_and_returns_at_most_n_themes():
    rows = [_hold("alpha beta gamma delta")]
    out = build_reasoning_themes(rows, top_k=999)
    # Clamped at 50, but only N distinct themes exist in input so result
    # is bounded by that.
    assert len(out["themes"]) <= 50
    assert out["top_k"] == 50


def test_top_k_clamps_low():
    rows = [_hold("alpha beta gamma")]
    out = build_reasoning_themes(rows, top_k=0)
    assert out["top_k"] == 3
    assert len(out["themes"]) <= 3


# ── endpoint integration ─────────────────────────────────────────────────


def test_endpoint_route_exists_and_returns_json(tmp_path, monkeypatch):
    """Drive the real Flask view on a fresh temp Store. Wiring asserted,
    not the builder math (covered above)."""
    from paper_trader import dashboard as dashboard_mod
    import paper_trader.store as store_mod
    from paper_trader.store import Store

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "paper_trader.db")
    monkeypatch.setattr(store_mod, "_singleton", None)
    store = Store()
    # Two parseable HOLD rows + one NO_DECISION.
    store.record_decision(
        market_open=True, signal_count=5,
        action_taken="HOLD NVDA → HOLD",
        reasoning=json.dumps({"decision": {
            "action": "HOLD", "ticker": "NVDA", "qty": 0,
            "confidence": 0.7,
            "reasoning": "Earnings premium in NVDA ahead of print",
        }}),
        portfolio_value=1000.0, cash=500.0,
    )
    store.record_decision(
        market_open=True, signal_count=4,
        action_taken="HOLD NVDA → HOLD",
        reasoning=json.dumps({"decision": {
            "action": "HOLD", "ticker": "NVDA", "qty": 0,
            "confidence": 0.6,
            "reasoning": "Earnings risk still intact",
        }}),
        portfolio_value=1000.0, cash=500.0,
    )

    monkeypatch.setattr(dashboard_mod, "get_store", lambda: store)
    client = dashboard_mod.app.test_client()
    resp = client.get("/api/reasoning-themes?limit=10&top_k=5")
    assert resp.status_code == 200
    out = resp.get_json()
    assert out["state"] == "OK"
    assert out["window_limit"] == 10
    assert out["top_k"] == 5
    phrases = {t["phrase"] for t in out["themes"]}
    assert "earnings" in phrases


def test_endpoint_clamps_limit_and_top_k(tmp_path, monkeypatch):
    from paper_trader import dashboard as dashboard_mod
    import paper_trader.store as store_mod
    from paper_trader.store import Store

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "p.db")
    monkeypatch.setattr(store_mod, "_singleton", None)
    store = Store()
    monkeypatch.setattr(dashboard_mod, "get_store", lambda: store)
    client = dashboard_mod.app.test_client()

    # ?limit=99999 → clamped to 500; ?top_k=999 → clamped to 50
    resp = client.get("/api/reasoning-themes?limit=99999&top_k=999")
    assert resp.status_code == 200
    out = resp.get_json()
    assert out["window_limit"] == 500
    assert out["top_k"] == 50

    # ?limit=1 → clamped to 5; ?top_k=1 → clamped to 3
    resp = client.get("/api/reasoning-themes?limit=1&top_k=1")
    out = resp.get_json()
    assert out["window_limit"] == 5
    assert out["top_k"] == 3


def test_endpoint_garbage_params_use_defaults(tmp_path, monkeypatch):
    from paper_trader import dashboard as dashboard_mod
    import paper_trader.store as store_mod
    from paper_trader.store import Store

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "p.db")
    monkeypatch.setattr(store_mod, "_singleton", None)
    store = Store()
    monkeypatch.setattr(dashboard_mod, "get_store", lambda: store)
    client = dashboard_mod.app.test_client()

    resp = client.get("/api/reasoning-themes?limit=banana&top_k=oops")
    assert resp.status_code == 200
    out = resp.get_json()
    assert out["window_limit"] == 100
    assert out["top_k"] == 10


def test_endpoint_include_bigrams_off(tmp_path, monkeypatch):
    from paper_trader import dashboard as dashboard_mod
    import paper_trader.store as store_mod
    from paper_trader.store import Store

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "p.db")
    monkeypatch.setattr(store_mod, "_singleton", None)
    store = Store()
    store.record_decision(
        market_open=True, signal_count=5,
        action_taken="HOLD NVDA → HOLD",
        reasoning=json.dumps({"decision": {
            "reasoning": "super cycle thesis",
        }}),
        portfolio_value=1000.0, cash=500.0,
    )
    monkeypatch.setattr(dashboard_mod, "get_store", lambda: store)
    client = dashboard_mod.app.test_client()

    resp = client.get("/api/reasoning-themes?include_bigrams=0")
    out = resp.get_json()
    assert out["include_bigrams"] is False
    for t in out["themes"]:
        assert " " not in t["phrase"]


def test_endpoint_not_swr_cached():
    """Cheap pure builder — must not sit behind the SWR cache (which is
    reserved for endpoints with yfinance/network fan-out). Same lock as
    ``test_reasoning_coherence`` uses."""
    from paper_trader import dashboard as dashboard_mod
    src = inspect.getsource(dashboard_mod.reasoning_themes_api)
    assert "@swr_cached" not in src
    assert "swr_cached" not in src.split("def reasoning_themes_api")[0][-200:]
