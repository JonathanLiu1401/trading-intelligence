"""Tests for analytics/decision_drought.py — pure, deterministic.

The contract under test: between FILLED trades, segment cycles into droughts;
price each drought's drift against the S&P from the equity curve; separate
involuntary (NO_DECISION) paralysis from deliberate HOLD; sum the negative
alpha of the paralysis droughts as 'involuntary alpha bleed'.
"""
from datetime import datetime, timezone

from paper_trader.analytics.decision_drought import (
    _classify,
    build_decision_drought,
)


def _dec(ts: str, action: str) -> dict:
    return {"timestamp": ts, "market_open": 1, "signal_count": 0,
            "action_taken": action, "reasoning": "", "portfolio_value": None,
            "cash": None}


def _eq(ts: str, tv: float, spy: float) -> dict:
    return {"timestamp": ts, "total_value": tv, "cash": 0.0, "sp500_price": spy}


class TestClassify:
    def test_fill_block_hold_nodecision(self):
        assert _classify("BUY NVDA → FILLED") == "FILLED"
        assert _classify("SELL MU → BLOCKED") == "BLOCKED"
        assert _classify("HOLD NVDA → HOLD") == "HOLD"
        assert _classify("NO_DECISION") == "NO_DECISION"
        assert _classify("") == "NO_DECISION"
        assert _classify(None) == "NO_DECISION"


def _scenario():
    """Fill → 3×NO_DECISION (paralysis, -3% alpha) → Fill → 3×HOLD (ongoing).

    recent_decisions() returns newest-first, so reverse the chronological list.
    """
    chrono = [
        _dec("2026-05-01T00:00:00+00:00", "BUY NVDA → FILLED"),
        _dec("2026-05-01T01:00:00+00:00", "NO_DECISION"),
        _dec("2026-05-01T02:00:00+00:00", "NO_DECISION"),
        _dec("2026-05-01T03:00:00+00:00", "NO_DECISION"),
        _dec("2026-05-01T04:00:00+00:00", "SELL NVDA → FILLED"),
        _dec("2026-05-01T05:00:00+00:00", "HOLD NVDA → HOLD"),
        _dec("2026-05-01T06:00:00+00:00", "HOLD NVDA → HOLD"),
        _dec("2026-05-01T07:00:00+00:00", "HOLD NVDA → HOLD"),
    ]
    equity = [
        _eq("2026-05-01T00:00:00+00:00", 1000.0, 100.0),
        _eq("2026-05-01T01:00:00+00:00", 1000.0, 100.0),
        _eq("2026-05-01T02:00:00+00:00", 995.0, 101.0),
        _eq("2026-05-01T03:00:00+00:00", 990.0, 102.0),
        _eq("2026-05-01T04:00:00+00:00", 990.0, 102.0),
        _eq("2026-05-01T05:00:00+00:00", 990.0, 102.0),
        _eq("2026-05-01T06:00:00+00:00", 1000.0, 103.0),
        _eq("2026-05-01T07:00:00+00:00", 1010.0, 104.0),
    ]
    newest_first = list(reversed(chrono))
    return newest_first, equity


class TestBuildDecisionDrought:
    def test_segments_two_droughts(self):
        decs, eq = _scenario()
        r = build_decision_drought(decs, eq,
                                   now=datetime(2026, 5, 1, 8, tzinfo=timezone.utc))
        assert r["n_cycles"] == 8
        assert r["n_fills"] == 2
        assert r["n_droughts"] == 2

    def test_paralysis_drought_alpha(self):
        decs, eq = _scenario()
        r = build_decision_drought(decs, eq)
        # Droughts sorted newest-first; the HOLD run is most recent.
        para = [d for d in r["droughts"] if d["kind"] == "PARALYSIS"]
        assert len(para) == 1
        d = para[0]
        assert d["n_cycles"] == 3
        assert d["n_no_decision"] == 3
        assert d["n_hold"] == 0
        # start 01:00 (tv 1000, spy 100) → end 03:00 (tv 990, spy 102)
        assert d["portfolio_pct"] == -1.0
        assert d["spy_pct"] == 2.0
        assert d["alpha_pct"] == -3.0
        assert d["ongoing"] is False

    def test_deliberate_hold_drought_is_ongoing(self):
        decs, eq = _scenario()
        r = build_decision_drought(decs, eq)
        cur = r["current_drought"]
        assert cur is not None
        assert cur["kind"] == "DELIBERATE_HOLD"
        assert cur["ongoing"] is True
        assert cur["n_hold"] == 3
        assert cur["n_no_decision"] == 0
        # start 05:00 (tv 990, spy 102) → end 07:00 (tv 1010, spy 104)
        assert cur["portfolio_pct"] == 2.02
        assert cur["spy_pct"] == 1.961
        assert cur["alpha_pct"] == 0.059

    def test_involuntary_bleed_and_verdict(self):
        decs, eq = _scenario()
        r = build_decision_drought(decs, eq)
        # Only the PARALYSIS drought's negative alpha counts: -3.0.
        assert r["involuntary_alpha_bleed_pct"] == -3.0
        assert r["n_paralysis_droughts"] == 1
        assert r["verdict"] == "BLEEDING"
        assert r["worst_alpha_drought"]["alpha_pct"] == -3.0

    def test_min_reportable_cycles_filters_singletons(self):
        # Alternating fill/no-decision → every gap is a single cycle, none
        # reportable, but they still count toward n_fills.
        chrono = [
            _dec("2026-05-01T00:00:00+00:00", "BUY A → FILLED"),
            _dec("2026-05-01T01:00:00+00:00", "NO_DECISION"),
            _dec("2026-05-01T02:00:00+00:00", "SELL A → FILLED"),
            _dec("2026-05-01T03:00:00+00:00", "NO_DECISION"),
            _dec("2026-05-01T04:00:00+00:00", "BUY A → FILLED"),
        ]
        r = build_decision_drought(list(reversed(chrono)), [])
        assert r["n_fills"] == 3
        assert r["n_droughts"] == 0
        assert r["current_drought"] is None

    def test_never_traded_verdict(self):
        chrono = [_dec(f"2026-05-01T0{i}:00:00+00:00", "NO_DECISION")
                  for i in range(5)]
        r = build_decision_drought(list(reversed(chrono)), [])
        assert r["n_fills"] == 0
        assert r["verdict"] == "NEVER_TRADED"
        # Ongoing paralysis drought present even with no equity data.
        assert r["current_drought"]["kind"] == "PARALYSIS"
        assert r["current_drought"]["alpha_pct"] is None

    def test_empty_is_no_data(self):
        r = build_decision_drought([], [])
        assert r["verdict"] == "NO_DATA"
        assert r["n_cycles"] == 0
        assert r["droughts"] == []

    def test_alpha_none_when_spy_missing(self):
        chrono = [
            _dec("2026-05-01T00:00:00+00:00", "BUY A → FILLED"),
            _dec("2026-05-01T01:00:00+00:00", "NO_DECISION"),
            _dec("2026-05-01T02:00:00+00:00", "NO_DECISION"),
            _dec("2026-05-01T03:00:00+00:00", "BUY A → FILLED"),
        ]
        eq = [
            _eq("2026-05-01T01:00:00+00:00", 1000.0, None),
            _eq("2026-05-01T02:00:00+00:00", 990.0, None),
        ]
        r = build_decision_drought(list(reversed(chrono)), eq)
        d = r["droughts"][0]
        assert d["portfolio_pct"] == -1.0
        assert d["spy_pct"] is None
        assert d["alpha_pct"] is None
