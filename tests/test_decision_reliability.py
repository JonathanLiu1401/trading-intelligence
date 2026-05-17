"""Exact-value tests for analytics/decision_reliability.build_decision_reliability.

The feature's whole point is the *regime partition*: the headline NO_DECISION
rate is dominated by legacy pre-diagnostics rows that stop accruing once the
runner restarts onto diagnostic code. This pins the true current-regime rate,
the STALE_LEGACY_DOMINATED / INSUFFICIENT / HEALTHY/DEGRADED/CRITICAL state
machine, the unparseable-timestamp handling, the dead-cycles-per-day
arithmetic, and that the bleed / headline numbers are *passed through* from the
single-source-of-truth builders (drought / forensics) rather than re-derived.
"""
from datetime import datetime, timedelta, timezone

from paper_trader.analytics.decision_reliability import (
    build_decision_reliability,
    MIN_CURRENT,
)
from paper_trader.analytics.decision_drought import build_decision_drought
from paper_trader.analytics.decision_forensics import build_decision_forensics

NOW = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)

_GOOD_REASON = '{"decision": {"action": "HOLD"}, "confidence": 0.5}'
_LEGACY_REASON = "claude returned no parseable JSON"
_TRUNCATED_REASON = 'parse_failed: {"decision": {"action": "BUY"'   # unbalanced { → TRUNCATED
_NOJSON_REASON = "parse_failed: I cannot comply with that request"  # no { → NO_JSON


def _dec(minutes_ago, *, no_decision=False, reasoning=None, ts=None,
         market_open=1):
    """One decisions-table row dict (shape of store.recent_decisions)."""
    if ts is None:
        ts = (NOW - timedelta(minutes=minutes_ago)).isoformat()
    if no_decision:
        return {"timestamp": ts, "market_open": market_open,
                "signal_count": 0, "action_taken": "NO_DECISION",
                "reasoning": reasoning if reasoning is not None else _NOJSON_REASON,
                "portfolio_value": 1000.0, "cash": 100.0}
    return {"timestamp": ts, "market_open": market_open, "signal_count": 3,
            "action_taken": "HOLD → HOLD",
            "reasoning": reasoning if reasoning is not None else _GOOD_REASON,
            "portfolio_value": 1000.0, "cash": 100.0}


def test_empty_decisions_is_no_data():
    out = build_decision_reliability([], [], now=NOW)
    assert out["state"] == "NO_DATA"
    assert out["n_decisions"] == 0
    assert out["current_total"] == 0
    assert out["current_failures"] == 0
    assert out["legacy_failures"] == 0
    assert out["legacy_share_pct"] == 0.0
    assert out["restart_recommended"] is False
    assert out["regime_boundary"] is None
    assert out["current_mode_mix"] == []
    assert out["involuntary_alpha_bleed_pct"] == 0.0
    assert out["decisions_per_day"] is None
    assert out["dead_cycles_per_day"] is None


def test_no_legacy_path_all_rows_are_current_and_healthy():
    # 12 good + 2 parse_failed, none legacy → boundary undefined → all current.
    decs = [_dec(i) for i in range(12)]
    decs += [_dec(20, no_decision=True, reasoning=_TRUNCATED_REASON),
             _dec(25, no_decision=True, reasoning=_TRUNCATED_REASON)]
    decs.sort(key=lambda d: d["timestamp"], reverse=True)  # newest-first

    out = build_decision_reliability(decs, [], now=NOW)
    assert out["regime_boundary"] is None
    assert out["n_decisions"] == 14
    assert out["current_total"] == 14
    assert out["current_failures"] == 2
    assert out["current_failure_rate_pct"] == round(2 / 14 * 100, 1)  # 14.3
    assert out["legacy_failures"] == 0
    assert out["legacy_share_pct"] == 0.0
    assert out["state"] == "HEALTHY"          # 14 ≥ MIN_CURRENT, 14.3% < 25
    assert out["restart_recommended"] is False
    assert out["decisions_per_day"] is not None  # span > 0


def test_all_legacy_is_stale_legacy_dominated_and_recommends_restart():
    decs = [_dec(i, no_decision=True, reasoning=_LEGACY_REASON)
            for i in range(5)]  # i=0 is newest → the boundary
    out = build_decision_reliability(decs, [], now=NOW)

    assert out["regime_boundary"] == (NOW - timedelta(minutes=0)).isoformat()
    assert out["n_decisions"] == 5
    assert out["legacy_failures"] == 5
    assert out["legacy_share_pct"] == 100.0
    assert out["current_total"] == 0          # nothing strictly after newest legacy
    assert out["current_failures"] == 0
    assert out["current_failure_rate_pct"] == 0.0
    assert out["current_mode_mix"] == []
    assert out["state"] == "STALE_LEGACY_DOMINATED"
    assert out["restart_recommended"] is True
    assert "restart" in out["headline"].lower()


def test_few_current_no_legacy_is_insufficient_not_stale():
    decs = [_dec(i) for i in range(MIN_CURRENT - 1)]  # all good, no legacy
    out = build_decision_reliability(decs, [], now=NOW)
    assert out["legacy_failures"] == 0
    assert out["current_total"] == MIN_CURRENT - 1
    assert out["state"] == "INSUFFICIENT"
    assert out["restart_recommended"] is False


def test_regime_boundary_excludes_unparseable_ts_and_pre_boundary_rows():
    # 3 legacy (newest = boundary), then post-boundary: good / TRUNCATED /
    # NO_JSON / good, plus one NO_DECISION row with an unparseable timestamp
    # that must NOT enter the current partition (boundary exists).
    decs = [
        _dec(60, no_decision=True, reasoning=_LEGACY_REASON),
        _dec(55, no_decision=True, reasoning=_LEGACY_REASON),
        _dec(50, no_decision=True, reasoning=_LEGACY_REASON),  # newest legacy
        _dec(40),
        _dec(30, no_decision=True, reasoning=_TRUNCATED_REASON),
        _dec(20, no_decision=True, reasoning=_NOJSON_REASON),
        _dec(10),
        _dec(0, no_decision=True, reasoning=_TRUNCATED_REASON, ts="not-a-date"),
    ]
    decs.sort(key=lambda d: str(d["timestamp"]), reverse=True)

    out = build_decision_reliability(decs, [], now=NOW)
    assert out["regime_boundary"] == (NOW - timedelta(minutes=50)).isoformat()
    assert out["n_decisions"] == 8
    assert out["legacy_failures"] == 3
    assert out["legacy_share_pct"] == round(3 / 8 * 100, 1)  # 37.5
    # current = strictly-after-boundary AND parseable ts: the 4 rows at -40/-30/-20/-10
    assert out["current_total"] == 4
    assert out["current_failures"] == 2
    assert out["current_failure_rate_pct"] == 50.0
    # mode mix: 1 TRUNCATED + 1 NO_JSON, sorted by (-n, MODES order)
    assert out["current_mode_mix"] == [
        {"mode": "TRUNCATED", "n": 1, "pct": 50.0},
        {"mode": "NO_JSON", "n": 1, "pct": 50.0},
    ]
    assert out["state"] == "INSUFFICIENT"      # only 4 current rows
    assert out["restart_recommended"] is False


def test_critical_when_current_rate_high_and_sample_sufficient():
    # 1 legacy boundary at -13h, then MIN_CURRENT current rows all NO_DECISION
    # spanning exactly the prior 12h → full span 12h = 0.5 day.
    decs = [_dec(0, no_decision=True, reasoning=_LEGACY_REASON,
                 ts=(NOW - timedelta(hours=13)).isoformat())]
    for k in range(MIN_CURRENT):
        # newest at -1h, oldest at -MIN_CURRENT h (so full span = 12h here)
        decs.append(_dec(0, no_decision=True, reasoning=_NOJSON_REASON,
                         ts=(NOW - timedelta(hours=(MIN_CURRENT - k))).isoformat()))
    decs.sort(key=lambda d: d["timestamp"], reverse=True)

    out = build_decision_reliability(decs, [], now=NOW)
    assert out["n_decisions"] == MIN_CURRENT + 1
    assert out["current_total"] == MIN_CURRENT
    assert out["current_failures"] == MIN_CURRENT
    assert out["current_failure_rate_pct"] == 100.0
    assert out["legacy_failures"] == 1
    assert out["state"] == "CRITICAL"
    assert out["restart_recommended"] is False  # restart flag is STALE-only
    # full span: (NOW-1h) ... (NOW-13h) = 12h = 0.5 day; n = 13
    assert out["decisions_per_day"] == round((MIN_CURRENT + 1) / 0.5, 3)
    assert out["dead_cycles_per_day"] == round(
        1.0 * (MIN_CURRENT + 1) / 0.5, 2)


def test_degraded_and_dead_cycles_sub_one_day_span():
    # 24 decisions over exactly 12h (0.5 day), 6 NO_DECISION → 25% → DEGRADED.
    decs = []
    for k in range(24):
        nd = k < 6
        decs.append(_dec(0, no_decision=nd,
                         reasoning=_NOJSON_REASON if nd else None,
                         ts=(NOW - timedelta(hours=12) +
                             timedelta(minutes=k * (12 * 60 // 23))).isoformat()))
    # force exact endpoints so span is precisely 12h
    decs[0]["timestamp"] = (NOW - timedelta(hours=12)).isoformat()
    decs[-1]["timestamp"] = NOW.isoformat()
    decs.sort(key=lambda d: d["timestamp"], reverse=True)

    out = build_decision_reliability(decs, [], now=NOW)
    assert out["current_total"] == 24
    assert out["current_failures"] == 6
    assert out["current_failure_rate_pct"] == 25.0
    assert out["state"] == "DEGRADED"
    assert out["decisions_per_day"] == round(24 / 0.5, 3)        # 48.0
    assert out["dead_cycles_per_day"] == round(6 / 24 * 48.0, 2)  # 12.0


def test_zero_span_does_not_divide_by_zero():
    # all identical timestamps → no derivable cadence, never an exception.
    same = NOW.isoformat()
    decs = [_dec(0, ts=same) for _ in range(3)]
    out = build_decision_reliability(decs, [], now=NOW)
    assert out["decisions_per_day"] is None
    assert out["dead_cycles_per_day"] is None
    assert out["state"] == "INSUFFICIENT"
    assert out["restart_recommended"] is False


def test_bleed_and_headline_are_passed_through_not_rederived():
    # Single-source-of-truth contract: the bleed comes verbatim from
    # build_decision_drought and the headline rate from build_decision_forensics.
    decs = [_dec(60, no_decision=True, reasoning=_LEGACY_REASON),
            _dec(50), _dec(40, no_decision=True, reasoning=_TRUNCATED_REASON),
            _dec(30), _dec(20), _dec(10)]
    decs.sort(key=lambda d: d["timestamp"], reverse=True)
    equity = [
        {"timestamp": (NOW - timedelta(minutes=60)).isoformat(),
         "total_value": 1000.0, "cash": 100.0, "sp500_price": 5000.0},
        {"timestamp": (NOW - timedelta(minutes=5)).isoformat(),
         "total_value": 980.0, "cash": 100.0, "sp500_price": 5100.0},
    ]
    out = build_decision_reliability(decs, equity, now=NOW)
    dr = build_decision_drought(decs, equity, now=NOW)
    fz = build_decision_forensics(decs, now=NOW)
    assert out["involuntary_alpha_bleed_pct"] == dr["involuntary_alpha_bleed_pct"]
    assert out["headline_failure_rate_pct"] == fz["failure_rate_pct"]
