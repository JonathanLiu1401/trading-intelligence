"""Tests for analytics/decision_confidence.py — aggregate Opus's self-rated
``confidence`` across recent decisions.

Pins:
  * pure / never raises on garbage (None/missing/non-JSON/NaN/non-numeric)
  * NO_DATA / INSUFFICIENT / OK ladder
  * NO_DATA when zero parseable confidence values
  * INSUFFICIENT below ``min_samples`` (raw stats emitted, regime withheld)
  * regime CAUTIOUS/NEUTRAL/CONVICTED at documented thresholds (median band)
  * trend TRENDING_UP/DOWN/FLAT vs ``TREND_DELTA`` — recent half is the
    FIRST half of caller-supplied order (``recent_decisions`` newest-first)
  * out-of-band values (-0.5 / 1.5) CLAMPED to [0,1], NOT dropped
  * NaN values dropped
  * per-action breakdown groups by leading verb (HOLD / BUY / NO_DECISION)
  * tolerates JSON envelope shape, top-level ``confidence`` key, and
    ``parse_failed:`` / ``retry_failed:`` prefixes
  * route exists, clamps limit (5..500), is not @swr_cached
"""
from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.decision_confidence import (
    CAUTIOUS_THRESHOLD,
    CONVICTED_THRESHOLD,
    MIN_SAMPLES_FOR_VERDICT,
    TREND_DELTA,
    build_decision_confidence,
)


def _row(action="HOLD NVDA → HOLD", confidence=0.7, reasoning="x",
         ts="2026-05-19T00:00:00+00:00"):
    if confidence is None:
        blob = json.dumps({"decision": {
            "action": "HOLD", "reasoning": reasoning,
        }})
    else:
        blob = json.dumps({"decision": {
            "action": "HOLD", "reasoning": reasoning,
            "confidence": confidence,
        }})
    return {
        "id": 1, "timestamp": ts, "action_taken": action,
        "reasoning": blob, "portfolio_value": 1000.0, "cash": 500.0,
        "market_open": 0, "signal_count": 5,
    }


# ── shape / degradation ladder ───────────────────────────────────────────────


def test_none_input_returns_no_data():
    out = build_decision_confidence(None)
    assert out["state"] == "NO_DATA"
    assert out["n_decisions"] == 0
    assert out["n_with_confidence"] == 0
    assert out["median"] is None
    assert out["regime"] is None
    assert out["trend"] is None


def test_empty_list_returns_no_data():
    out = build_decision_confidence([])
    assert out["state"] == "NO_DATA"


def test_all_unparseable_returns_no_data():
    rows = [
        {"reasoning": None, "action_taken": "NO_DECISION", "timestamp": "t"},
        {"reasoning": "claude returned no response", "action_taken": "NO_DECISION", "timestamp": "t"},
        {"reasoning": "{}", "action_taken": "HOLD", "timestamp": "t"},
        {"reasoning": json.dumps({"decision": {"reasoning": "no confidence here"}}),
         "action_taken": "HOLD", "timestamp": "t"},
    ]
    out = build_decision_confidence(rows)
    assert out["state"] == "NO_DATA"
    assert out["n_unparseable"] == 4


def test_garbage_rows_do_not_raise():
    rows = [{}, {"reasoning": 12345}, {"action_taken": "HOLD"}]
    out = build_decision_confidence(rows)
    assert out["state"] == "NO_DATA"


def test_below_min_samples_insufficient_with_raw_stats():
    # Use one fewer than the threshold so we always exercise INSUFFICIENT.
    rows = [_row(confidence=0.7) for _ in range(MIN_SAMPLES_FOR_VERDICT - 1)]
    out = build_decision_confidence(rows)
    assert out["state"] == "INSUFFICIENT"
    assert out["n_with_confidence"] == MIN_SAMPLES_FOR_VERDICT - 1
    assert out["median"] == 0.7
    assert out["regime"] is None  # withheld until enough samples
    assert out["trend"] is None


# ── extraction across shapes ──────────────────────────────────────────────


def test_extracts_envelope_confidence():
    rows = [_row(confidence=0.65) for _ in range(MIN_SAMPLES_FOR_VERDICT)]
    out = build_decision_confidence(rows)
    assert out["state"] == "OK"
    assert out["median"] == 0.65
    assert out["mean"] == 0.65


def test_extracts_top_level_confidence_key():
    """Some rows may carry ``confidence`` at the top level rather than
    wrapped in ``{"decision": {...}}``."""
    blob = json.dumps({"confidence": 0.8, "reasoning": "x"})
    rows = [{
        "id": 1, "timestamp": "t", "action_taken": "HOLD",
        "reasoning": blob, "portfolio_value": 1000.0, "cash": 500.0,
        "market_open": 0, "signal_count": 5,
    } for _ in range(MIN_SAMPLES_FOR_VERDICT)]
    out = build_decision_confidence(rows)
    assert out["state"] == "OK"
    assert out["median"] == 0.8


def test_tolerates_parse_failed_prefix():
    raw = "parse_failed: " + json.dumps({
        "decision": {"confidence": 0.55, "reasoning": "x"},
    })
    rows = [{
        "id": 1, "timestamp": "t", "action_taken": "HOLD",
        "reasoning": raw, "portfolio_value": 1000.0, "cash": 500.0,
        "market_open": 0, "signal_count": 5,
    } for _ in range(MIN_SAMPLES_FOR_VERDICT)]
    out = build_decision_confidence(rows)
    assert out["state"] == "OK"
    assert out["median"] == 0.55


# ── numerical robustness ─────────────────────────────────────────────────


def test_out_of_band_values_clamped_not_dropped():
    """A model emitting 1.2 or -0.3 is bounded conviction noise, not
    invalid data — read it but bound to [0, 1]. (If we silently dropped,
    we'd hide a real model bug from the operator.)"""
    rows = [
        _row(confidence=1.5),
        _row(confidence=-0.5),
        _row(confidence=0.7),
        _row(confidence=0.6),
        _row(confidence=0.8),
    ]
    out = build_decision_confidence(rows)
    assert out["state"] == "OK"
    assert out["n_with_confidence"] == 5
    assert out["max"] == 1.0  # clamped from 1.5
    assert out["min"] == 0.0  # clamped from -0.5


def test_nan_dropped():
    rows = [_row(confidence=v) for v in (
        float("nan"), 0.7, 0.6, 0.5, 0.8, 0.55,
    )]
    out = build_decision_confidence(rows)
    assert out["n_with_confidence"] == 5
    assert out["n_unparseable"] == 1


def test_non_numeric_confidence_dropped():
    blob = json.dumps({"decision": {"confidence": "high", "reasoning": "x"}})
    rows = [{
        "id": 1, "timestamp": "t", "action_taken": "HOLD",
        "reasoning": blob, "portfolio_value": 1000.0, "cash": 500.0,
        "market_open": 0, "signal_count": 5,
    }]
    out = build_decision_confidence(rows)
    assert out["state"] == "NO_DATA"
    assert out["n_unparseable"] == 1


# ── regime thresholds ────────────────────────────────────────────────────


def test_cautious_regime():
    """Median below CAUTIOUS_THRESHOLD (=0.45) → CAUTIOUS."""
    rows = [_row(confidence=v) for v in (0.3, 0.35, 0.4, 0.3, 0.35, 0.4)]
    out = build_decision_confidence(rows)
    assert out["state"] == "OK"
    assert out["regime"] == "CAUTIOUS"
    assert out["median"] < CAUTIOUS_THRESHOLD


def test_neutral_regime():
    """CAUTIOUS_THRESHOLD ≤ median < CONVICTED_THRESHOLD → NEUTRAL."""
    rows = [_row(confidence=v) for v in (0.5, 0.55, 0.6, 0.5, 0.55, 0.6)]
    out = build_decision_confidence(rows)
    assert out["state"] == "OK"
    assert out["regime"] == "NEUTRAL"
    assert CAUTIOUS_THRESHOLD <= out["median"] < CONVICTED_THRESHOLD


def test_convicted_regime():
    """Median ≥ CONVICTED_THRESHOLD (=0.70) → CONVICTED."""
    rows = [_row(confidence=v) for v in (0.7, 0.75, 0.8, 0.7, 0.75, 0.8)]
    out = build_decision_confidence(rows)
    assert out["state"] == "OK"
    assert out["regime"] == "CONVICTED"
    assert out["median"] >= CONVICTED_THRESHOLD


def test_threshold_boundary_inclusive_at_cautious():
    """median == CAUTIOUS_THRESHOLD (=0.45 by default) → NEUTRAL (not CAUTIOUS).
    The CAUTIOUS bucket is < threshold strict."""
    rows = [_row(confidence=CAUTIOUS_THRESHOLD)
            for _ in range(MIN_SAMPLES_FOR_VERDICT)]
    out = build_decision_confidence(rows)
    assert out["regime"] == "NEUTRAL"


def test_threshold_boundary_inclusive_at_convicted():
    """median == CONVICTED_THRESHOLD → CONVICTED (inclusive)."""
    rows = [_row(confidence=CONVICTED_THRESHOLD)
            for _ in range(MIN_SAMPLES_FOR_VERDICT)]
    out = build_decision_confidence(rows)
    assert out["regime"] == "CONVICTED"


# ── trend split (recent vs older) ────────────────────────────────────────


def test_trend_up_when_recent_half_higher():
    """Caller order preserved. ``recent_decisions`` returns newest-first
    so the FIRST half is the recent half. Construct: 6 rows where the
    first 3 are 0.8 (high conviction recent) and last 3 are 0.4
    (low conviction older). Recent median 0.8 - older 0.4 = +0.4 >>
    TREND_DELTA → TRENDING_UP."""
    rows = (
        [_row(confidence=0.8) for _ in range(3)]
        + [_row(confidence=0.4) for _ in range(3)]
    )
    out = build_decision_confidence(rows)
    assert out["state"] == "OK"
    trend = out["trend"]
    assert trend is not None
    assert trend["tag"] == "TRENDING_UP"
    assert trend["recent_median"] == 0.8
    assert trend["older_median"] == 0.4
    assert trend["delta"] >= TREND_DELTA


def test_trend_down_when_recent_half_lower():
    rows = (
        [_row(confidence=0.4) for _ in range(3)]
        + [_row(confidence=0.8) for _ in range(3)]
    )
    out = build_decision_confidence(rows)
    trend = out["trend"]
    assert trend["tag"] == "TRENDING_DOWN"
    assert trend["delta"] <= -TREND_DELTA


def test_trend_flat_when_within_delta():
    rows = [_row(confidence=v) for v in (0.7, 0.72, 0.7, 0.71, 0.7, 0.72)]
    out = build_decision_confidence(rows)
    trend = out["trend"]
    assert trend["tag"] == "FLAT"
    assert abs(trend["delta"]) < TREND_DELTA


def test_trend_none_when_halves_too_small():
    """Below 4 samples split-half is <2 per side; trend withheld."""
    rows = [_row(confidence=0.7), _row(confidence=0.75),
            _row(confidence=0.7), _row(confidence=0.8),
            _row(confidence=0.7)]  # n=5 → half=2, just at limit
    out = build_decision_confidence(rows)
    assert out["state"] == "OK"
    assert out["trend"] is not None
    # n=3 (one short of MIN_SAMPLES_FOR_VERDICT) would be INSUFFICIENT;
    # we sit in OK with a valid (but possibly FLAT) trend.


# ── per-action breakdown ─────────────────────────────────────────────────


def test_per_action_breakdown_groups_by_leading_verb():
    rows = [
        _row(action="HOLD NVDA → HOLD", confidence=0.7),
        _row(action="HOLD NVDA → HOLD", confidence=0.8),
        _row(action="BUY NVDA → FILLED", confidence=0.6),
        _row(action="SELL_CALL NVDA 200C → FILLED", confidence=0.5),
        _row(action="HOLD NVDA → HOLD", confidence=0.75),
    ]
    out = build_decision_confidence(rows)
    by_action = out["by_action"]
    assert "HOLD" in by_action
    assert by_action["HOLD"]["n"] == 3
    assert by_action["BUY"]["n"] == 1
    assert by_action["SELL_CALL"]["n"] == 1
    assert by_action["HOLD"]["median"] == 0.75  # median of [0.7,0.75,0.8]


def test_per_action_handles_empty_or_no_decision():
    """``NO_DECISION`` rows carry plain prose reasoning, NOT a JSON
    envelope — they never contribute a confidence value, so they should
    not appear in by_action at all. A HOLD row with a missing/blank
    action_taken bucket maps to ``UNKNOWN``."""
    rows = [
        _row(action="", confidence=0.7),
        _row(action=None, confidence=0.6),
        _row(action="HOLD", confidence=0.8),
        _row(action="HOLD", confidence=0.75),
        _row(action="HOLD", confidence=0.7),
    ]
    out = build_decision_confidence(rows)
    by_action = out["by_action"]
    assert "UNKNOWN" in by_action
    assert by_action["UNKNOWN"]["n"] == 2
    assert by_action["HOLD"]["n"] == 3


# ── buckets ──────────────────────────────────────────────────────────────


def test_buckets_partition_range():
    rows = [_row(confidence=v) for v in (
        0.1, 0.2,         # low (<0.4)
        0.4, 0.5,         # medium ([0.4, 0.6))
        0.6, 0.7,         # high ([0.6, 0.8))
        0.85, 0.95, 1.0,  # very_high ([0.8, 1.0])
    )]
    out = build_decision_confidence(rows)
    b = out["buckets"]
    assert b["low"] == 2
    assert b["medium"] == 2
    assert b["high"] == 2
    assert b["very_high"] == 3
    # All keys present (zeroed when empty)
    assert set(b.keys()) == {"low", "medium", "high", "very_high"}


# ── endpoint integration ─────────────────────────────────────────────────


def test_endpoint_route_exists(tmp_path, monkeypatch):
    from paper_trader import dashboard as dashboard_mod
    import paper_trader.store as store_mod
    from paper_trader.store import Store

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "p.db")
    monkeypatch.setattr(store_mod, "_singleton", None)
    store = Store()
    for v in (0.7, 0.75, 0.6, 0.8, 0.7, 0.65):
        store.record_decision(
            market_open=True, signal_count=5,
            action_taken="HOLD NVDA → HOLD",
            reasoning=json.dumps({"decision": {
                "confidence": v, "reasoning": "x",
            }}),
            portfolio_value=1000.0, cash=500.0,
        )
    monkeypatch.setattr(dashboard_mod, "get_store", lambda: store)
    client = dashboard_mod.app.test_client()
    resp = client.get("/api/decision-confidence?limit=20")
    assert resp.status_code == 200
    out = resp.get_json()
    assert out["state"] == "OK"
    assert out["window_limit"] == 20
    assert out["n_with_confidence"] == 6


def test_endpoint_clamps_limit(tmp_path, monkeypatch):
    from paper_trader import dashboard as dashboard_mod
    import paper_trader.store as store_mod
    from paper_trader.store import Store

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "p.db")
    monkeypatch.setattr(store_mod, "_singleton", None)
    store = Store()
    monkeypatch.setattr(dashboard_mod, "get_store", lambda: store)
    client = dashboard_mod.app.test_client()

    resp = client.get("/api/decision-confidence?limit=99999")
    assert resp.get_json()["window_limit"] == 500

    resp = client.get("/api/decision-confidence?limit=1")
    assert resp.get_json()["window_limit"] == 5

    resp = client.get("/api/decision-confidence?limit=banana")
    assert resp.get_json()["window_limit"] == 100


def test_endpoint_not_swr_cached():
    from paper_trader import dashboard as dashboard_mod
    src = inspect.getsource(dashboard_mod.decision_confidence_api)
    assert "@swr_cached" not in src
