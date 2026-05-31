"""Exact-value tests for analytics/parse_fail_windows.build_parse_fail_windows.

Pins the per-window failure-rate math, the trend-verdict thresholds, the
MIN_WIN_N sample gate, the recent-failures forensics ordering, and the
single-source-of-truth contract (every mode label comes from
decision_forensics.classify_failure, not re-derived). Specific values are
asserted, not "no crash" — the whole point of the panel is that a 1h vs 6h
delta of ≥TREND_PP names the trend, and a regression to round-number-arithmetic
would silently break the operator's deploy-validation workflow.
"""
from datetime import datetime, timedelta, timezone

from paper_trader.analytics.parse_fail_windows import (
    MIN_WIN_N,
    TREND_PP,
    build_parse_fail_windows,
)

NOW = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)

_GOOD = '{"decision": {"action": "HOLD"}, "confidence": 0.5}'
_NOJSON = "parse_failed: I cannot comply with that request"
_TRUNCATED = 'parse_failed: {"decision": {"action": "BUY"'
_HOST_SKIP = "skipped claude call — host saturated: load too high"
_SUBPROC = "claude returned no response (nonzero_rc)"


def _dec(minutes_ago, *, no_decision=False, reasoning=None, ts=None):
    """One decisions row (shape of store.recent_decisions)."""
    if ts is None:
        ts = (NOW - timedelta(minutes=minutes_ago)).isoformat()
    if no_decision:
        return {"timestamp": ts, "action_taken": "NO_DECISION",
                "reasoning": reasoning if reasoning is not None else _NOJSON}
    return {"timestamp": ts, "action_taken": "HOLD → HOLD",
            "reasoning": reasoning if reasoning is not None else _GOOD}


def test_empty_decisions_is_no_data():
    out = build_parse_fail_windows([], now=NOW)
    assert out["state"] == "NO_DATA"
    assert out["n_decisions_total"] == 0
    assert out["windows"] == []
    assert out["trend"] == "INSUFFICIENT"
    assert out["trend_delta_pp"] is None
    assert out["recent_failures"] == []
    assert "no decisions" in out["headline"].lower()


def test_window_labels_and_order_are_pinned():
    decs = [_dec(i) for i in range(3)]  # 3 healthy rows in last 3 min
    out = build_parse_fail_windows(decs, now=NOW)
    labels = [w["label"] for w in out["windows"]]
    # Trend logic compares windows[0] (1h) vs windows[1] (6h); pinning the
    # order guarantees a future re-tune can't silently re-wire the trend.
    assert labels == ["1h", "6h", "24h", "7d"]


def test_zero_failures_clean_window_is_healthy_when_sufficient():
    # 6 healthy decisions in the last hour — clears MIN_WIN_N and the 25%
    # DEGRADED band, so the 1h state is HEALTHY.
    decs = [_dec(i) for i in range(6)]
    out = build_parse_fail_windows(decs, now=NOW)
    w1h = out["windows"][0]
    assert w1h["n_decisions"] == 6
    assert w1h["n_failures"] == 0
    assert w1h["failure_rate_pct"] == 0.0
    assert w1h["mode_mix"] == []
    assert w1h["state"] == "HEALTHY"
    assert w1h["sufficient"] is True


def test_window_failure_rate_and_mode_mix_pinned():
    # 5 healthy + 5 NO_JSON in last 30 min → 50% failure rate, 1h window.
    decs = [_dec(i) for i in range(5)]
    decs += [_dec(10 + i, no_decision=True, reasoning=_NOJSON)
             for i in range(5)]
    out = build_parse_fail_windows(decs, now=NOW)
    w1h = out["windows"][0]
    assert w1h["n_decisions"] == 10
    assert w1h["n_failures"] == 5
    assert w1h["failure_rate_pct"] == 50.0
    assert w1h["state"] == "CRITICAL"  # ≥50%
    assert w1h["mode_mix"] == [{"mode": "NO_JSON", "n": 5, "pct": 100.0}]


def test_insufficient_window_state_is_insufficient_regardless_of_rate():
    # 1/1 = 100% would be CRITICAL if not for the MIN_WIN_N=5 sample floor.
    decs = [_dec(0, no_decision=True, reasoning=_NOJSON)]
    out = build_parse_fail_windows(decs, now=NOW)
    w1h = out["windows"][0]
    assert w1h["n_decisions"] == 1
    assert w1h["n_failures"] == 1
    assert w1h["failure_rate_pct"] == 100.0  # numbers honest from n=1
    assert w1h["sufficient"] is False
    assert w1h["state"] == "INSUFFICIENT"  # label honest until N≥MIN_WIN_N


def test_unparseable_ts_excluded_from_every_window():
    # Two rows with bad timestamps — they bump n_decisions_total but no
    # window can claim them (same convention as decision_reliability /
    # decision_forensics).
    decs = [_dec(0, ts="not-a-date", no_decision=True, reasoning=_NOJSON),
            _dec(0, ts="not-a-date")]
    decs += [_dec(i) for i in range(5)]
    out = build_parse_fail_windows(decs, now=NOW)
    assert out["n_decisions_total"] == 7
    # Only the 5 dated rows land in any window; 0 failures in any.
    assert out["windows"][0]["n_decisions"] == 5
    assert out["windows"][0]["n_failures"] == 0
    # recent_failures is dated-rows-only — the bad-ts NO_DECISION must NOT leak.
    assert out["recent_failures"] == []


def test_trend_worsening_when_1h_rate_exceeds_6h_by_threshold():
    # 6h baseline: 5 healthy + 5 failures in the 90–360 min window → 50%.
    # 1h window: 2 healthy + 8 failures in last 60 min → 80%. Delta +30pp →
    # WORSENING (above TREND_PP=10).
    decs = []
    # 1h window content (minutes_ago < 60)
    decs += [_dec(2 + i) for i in range(2)]
    decs += [_dec(10 + i, no_decision=True) for i in range(8)]
    # 6h baseline outside the 1h window (90+ min ago)
    decs += [_dec(90 + i) for i in range(5)]
    decs += [_dec(120 + i, no_decision=True) for i in range(5)]

    out = build_parse_fail_windows(decs, now=NOW)
    # 6h window collects everything (1h subset + 90..125 min entries).
    # 1h rate = 8/10 = 80; 6h rate = 13/20 = 65; delta = +15pp.
    assert out["windows"][0]["failure_rate_pct"] == 80.0
    assert out["windows"][1]["failure_rate_pct"] == 65.0
    assert out["trend_delta_pp"] == 15.0
    assert out["trend"] == "WORSENING"


def test_trend_improving_when_1h_rate_below_6h_by_threshold():
    # Mirror image: 1h is much healthier than the 6h baseline.
    decs = []
    # 1h window: 8 healthy + 2 failures → 20%
    decs += [_dec(2 + i) for i in range(8)]
    decs += [_dec(15 + i, no_decision=True) for i in range(2)]
    # 6h baseline (outside 1h): 5 healthy + 10 failures
    decs += [_dec(90 + i) for i in range(5)]
    decs += [_dec(120 + i, no_decision=True) for i in range(10)]

    out = build_parse_fail_windows(decs, now=NOW)
    # 1h = 2/10 = 20.0; 6h = 12/25 = 48.0; delta = -28pp.
    assert out["windows"][0]["failure_rate_pct"] == 20.0
    assert out["windows"][1]["failure_rate_pct"] == 48.0
    assert out["trend_delta_pp"] == -28.0
    assert out["trend"] == "IMPROVING"


def test_trend_stable_when_delta_within_band():
    # Same rate in both windows → STABLE (within ±TREND_PP).
    decs = []
    # 1h: 6 healthy + 4 failures → 40%
    decs += [_dec(2 + i) for i in range(6)]
    decs += [_dec(20 + i, no_decision=True) for i in range(4)]
    # additional 6h (outside 1h): 3 healthy + 2 failures → 40% pooled inside
    # the 6h window (combined 9/15 = 40%)
    decs += [_dec(90 + i) for i in range(3)]
    decs += [_dec(120 + i, no_decision=True) for i in range(2)]

    out = build_parse_fail_windows(decs, now=NOW)
    assert out["windows"][0]["failure_rate_pct"] == 40.0
    assert out["windows"][1]["failure_rate_pct"] == 40.0
    assert out["trend"] == "STABLE"
    assert abs(out["trend_delta_pp"]) < TREND_PP


def test_trend_insufficient_when_either_window_thin():
    # 6h window has plenty; 1h has < MIN_WIN_N → trend withheld.
    decs = []
    decs += [_dec(2, no_decision=True)]  # 1h has just 1 row
    decs += [_dec(90 + i) for i in range(MIN_WIN_N + 2)]
    out = build_parse_fail_windows(decs, now=NOW)
    assert out["windows"][0]["sufficient"] is False
    assert out["windows"][1]["sufficient"] is True
    assert out["trend"] == "INSUFFICIENT"
    assert out["trend_delta_pp"] is None


def test_recent_failures_capped_and_newest_first():
    # 15 parse-fails — verify the first 10 are surfaced and they're newest first.
    decs = []
    for i in range(15):
        # newest minutes_ago=0 ⇒ first in list when sorted newest-first
        decs.append(_dec(i, no_decision=True,
                         reasoning=_NOJSON if i % 2 == 0 else _TRUNCATED))
    out = build_parse_fail_windows(decs, now=NOW)
    rf = out["recent_failures"]
    assert len(rf) == 10  # _FORENSICS_N cap
    # Newest-first contract: timestamps strictly descending.
    ts = [r["timestamp"] for r in rf]
    assert ts == sorted(ts, reverse=True)
    # First row is the newest (minutes_ago=0) and was NO_JSON (even index).
    assert rf[0]["mode"] == "NO_JSON"
    # And the next was TRUNCATED (the i=1 row).
    assert rf[1]["mode"] == "TRUNCATED"


def test_mode_mix_uses_canonical_taxonomy_and_orders_by_count():
    # 6 NO_JSON + 3 HOST_SATURATED_SKIP + 1 SUBPROCESS_ERROR, all in 1h window.
    decs = []
    decs += [_dec(2 + i, no_decision=True, reasoning=_NOJSON)
             for i in range(6)]
    decs += [_dec(15 + i, no_decision=True, reasoning=_HOST_SKIP)
             for i in range(3)]
    decs += [_dec(25, no_decision=True, reasoning=_SUBPROC)]
    out = build_parse_fail_windows(decs, now=NOW)
    mix = out["windows"][0]["mode_mix"]
    assert mix[0] == {"mode": "NO_JSON", "n": 6, "pct": 60.0}
    assert mix[1] == {"mode": "HOST_SATURATED_SKIP", "n": 3, "pct": 30.0}
    assert mix[2] == {"mode": "SUBPROCESS_ERROR", "n": 1, "pct": 10.0}


def test_headline_uses_6h_canonical_window_when_available():
    # 5 healthy in 1h (sufficient + HEALTHY) + 5 NO_JSON in the 90-150 min
    # range. The 1h window has 0 failures (HEALTHY); the 6h window has 5/10
    # = 50% (CRITICAL). Headline must surface the 6h canonical rate, not the
    # 1h one (the deploy-validation use case: a 0% last hour after a CRITICAL
    # 6h is the IMPROVING trend, not a HEALTHY system).
    decs = [_dec(2 + i) for i in range(5)]
    decs += [_dec(90 + i, no_decision=True) for i in range(5)]
    out = build_parse_fail_windows(decs, now=NOW)
    assert out["windows"][0]["failure_rate_pct"] == 0.0   # 1h all healthy
    assert out["windows"][1]["failure_rate_pct"] == 50.0  # 6h has the fails
    # State reflects the canonical 6h window, not the cosmetic 1h zero.
    assert out["state"] == "CRITICAL"
    assert "6h parse-fail 50.0%" in out["headline"]
    # Trend is IMPROVING — 1h dropped from 50% to 0% (delta -50pp).
    assert out["trend"] == "IMPROVING"


def test_falls_back_to_24h_when_6h_thin():
    # Only 24h window has ≥MIN_WIN_N. 6h must NOT drive the headline.
    decs = [_dec(60 * 7 + i) for i in range(MIN_WIN_N + 2)]  # all ~7h ago
    out = build_parse_fail_windows(decs, now=NOW)
    assert out["windows"][1]["sufficient"] is False  # 6h
    assert out["windows"][2]["sufficient"] is True   # 24h
    # State driven by 24h window (HEALTHY here — no failures).
    assert out["state"] == "HEALTHY"
    assert "24h parse-fail" in out["headline"]


def test_no_failures_at_all_still_emits_a_window_state():
    # 6 clean rows in 1h, nothing else. Trend still STABLE — both 1h and 6h
    # are at 0%, sufficient.
    decs = [_dec(i) for i in range(6)]
    out = build_parse_fail_windows(decs, now=NOW)
    assert out["windows"][0]["failure_rate_pct"] == 0.0
    assert out["windows"][1]["failure_rate_pct"] == 0.0
    assert out["windows"][0]["sufficient"] is True
    assert out["windows"][1]["sufficient"] is True
    assert out["trend"] == "STABLE"
    assert out["trend_delta_pp"] == 0.0
    assert out["state"] == "HEALTHY"
