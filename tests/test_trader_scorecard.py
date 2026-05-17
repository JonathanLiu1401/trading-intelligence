"""Tests for analytics/trader_scorecard.py — the behavioural-verdict
alignment router.

These assert *correct behaviour*, not "no crash". The scorecard mints no new
opinion: it routes the five pure behavioural builders' own verdicts into a
descriptive concordance view. A wrong FLAG/OK/IMMATURE classification, a
verdict invented out of an unknown label, a builder fault that sinks the whole
scorecard, a drifted (non-verbatim) headline, or a mis-ordered ``focus`` all
fail an assertion here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import trader_scorecard as tsc
from paper_trader.analytics.trader_scorecard import (
    build_trader_scorecard,
    classify_check,
)
from paper_trader.analytics.trade_asymmetry import build_trade_asymmetry

NOW = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)


# ───────────────────────── fixtures (shared shape) ──────────────────────────

def _trade(tid, ticker, action, ts, qty, px):
    return {"id": tid, "timestamp": ts, "ticker": ticker, "action": action,
            "qty": qty, "price": px, "value": qty * px,
            "strike": None, "expiry": None, "option_type": None}


def _loss_ledger(n):
    """`n` independent losing LITE round-trips, store-native newest-first.
    Each buy 100 → sell 90 in a strictly increasing window so
    build_round_trips closes each as its own round-trip; consecutive
    re-buys are 1 day apart (< REENTRY_WINDOW_DAYS) so churn sees them as
    fast same-name re-entries."""
    rows = []
    tid = 1
    for k in range(n):
        buy_ts = (NOW - timedelta(days=2 * (n - k) + 1)).isoformat()
        sell_ts = (NOW - timedelta(days=2 * (n - k))).isoformat()
        rows.append(_trade(tid, "LITE", "BUY", buy_ts, 1.0, 100.0))
        rows.append(_trade(tid + 1, "LITE", "SELL", sell_ts, 1.0, 90.0))
        tid += 2
    rows.reverse()
    return rows


def _pos(ticker, qty, avg, cur, opened_days_ago):
    return {"ticker": ticker, "type": "stock", "qty": qty, "avg_cost": avg,
            "current_price": cur,
            "opened_at": (NOW - timedelta(days=opened_days_ago)).isoformat()}


def _eq(total, sp500, mins_ago):
    return {"timestamp": (NOW - timedelta(minutes=mins_ago)).isoformat(),
            "total_value": total, "cash": 6.0, "sp500_price": sp500}


# ───────────────────────────── NO_DATA path ─────────────────────────────

class TestNoData:
    def test_empty_inputs_are_no_data_and_never_raise(self):
        r = build_trader_scorecard({}, [], [], [], [], now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["n_flags"] == 0
        assert r["headline"] == "No mature behavioural history yet."
        assert r["focus"] is None
        assert r["concordance"] == []
        # every constituent is still composed (not omitted) and immature
        names = {c["name"] for c in r["checks"]}
        assert names == {"trade_asymmetry", "churn", "capital_paralysis",
                         "decision_reliability", "open_attribution"}
        assert all(c["klass"] == "IMMATURE" for c in r["checks"])
        assert r["as_of"] == NOW.isoformat(timespec="seconds")


# ───────────────────── FLAGS + concordance + focus ──────────────────────

class TestFlagsPresentConcordance:
    """21 all-loss LITE round-trips → STABLE PAYOFF_TRAP (trade_asymmetry)
    AND CHURNING (churn). $6 cash of $1000 across two underwater names →
    PINNED (capital_paralysis). PAYOFF_TRAP+CHURNING share the
    EXIT_DISCIPLINE theme → a concordance note; CAPITAL_TRAP outranks
    EXIT_DISCIPLINE so focus is the capital headline."""

    def setup_method(self):
        trades = _loss_ledger(21)
        portfolio = {"cash": 6.0, "total_value": 1000.0}
        positions = [_pos("LITE", 1.0, 800.0, 790.0, 5),
                     _pos("NVDA", 1.0, 200.0, 180.0, 5)]
        equity = [_eq(1010.0, 5000.0, 60), _eq(972.0, 5050.0, 1)]
        self.trades = trades
        self.r = build_trader_scorecard(portfolio, positions, trades, [],
                                        equity, now=NOW)

    def _check(self, name):
        return next(c for c in self.r["checks"] if c["name"] == name)

    def test_asymmetry_flagged_payoff_trap(self):
        c = self._check("trade_asymmetry")
        assert c["klass"] == "FLAG"
        assert c["label"] == "PAYOFF_TRAP"

    def test_churn_flagged_churning(self):
        c = self._check("churn")
        assert c["klass"] == "FLAG"
        assert c["label"] == "CHURNING"

    def test_capital_flagged_pinned(self):
        c = self._check("capital_paralysis")
        assert c["klass"] == "FLAG"
        assert c["label"] == "PINNED"

    def test_open_attribution_flagged_selection_drag(self):
        # losing on both names while SPY rose 5000→5050 is a real pathology
        c = self._check("open_attribution")
        assert c["klass"] == "FLAG"
        assert c["label"] == "SELECTION_DRAG"
        # decisions=[] → reliability is immature, never an invented flag
        assert self._check("decision_reliability")["klass"] == "IMMATURE"

    def test_state_and_counts(self):
        assert self.r["state"] == "FLAGS_PRESENT"
        assert self.r["n_flags"] == 4
        # headline names the verbatim labels and the count
        assert self.r["headline"].startswith("4 of 5 behavioural checks flagging")
        for lbl in ("PAYOFF_TRAP", "CHURNING", "PINNED", "SELECTION_DRAG"):
            assert lbl in self.r["headline"]

    def test_exit_discipline_concordance_fires(self):
        notes = self.r["concordance"]
        ed = [n for n in notes if n["theme"] == "EXIT_DISCIPLINE"]
        assert len(ed) == 1
        assert ed[0]["count"] == 2
        assert set(ed[0]["labels"]) == {"PAYOFF_TRAP", "CHURNING"}
        # CAPITAL_TRAP has only one flagging builder → no concordance note
        assert not [n for n in notes if n["theme"] == "CAPITAL_TRAP"]

    def test_focus_is_capital_by_documented_precedence(self):
        f = self.r["focus"]
        assert f is not None
        assert f["name"] == "capital_paralysis"
        # focus headline is the builder's own, forwarded verbatim
        assert f["headline"] == self._check("capital_paralysis")["headline"]

    def test_headline_is_builder_verbatim_no_drift(self):
        # single source of truth: each check carries the builder's own headline
        asym = build_trade_asymmetry(list(reversed(self.trades)), now=NOW)
        assert self._check("trade_asymmetry")["headline"] == asym["headline"]


# ───────────────────────── ALIGNED_HEALTHY path ─────────────────────────

class TestAlignedHealthy:
    """Cash-rich book, no trades/decisions: asymmetry/churn/reliability are
    IMMATURE, but capital_paralysis is a mature FREE (OK) with zero flags →
    ALIGNED_HEALTHY."""

    def setup_method(self):
        portfolio = {"cash": 900.0, "total_value": 1000.0}
        positions = [_pos("NVDA", 1.0, 100.0, 100.0, 2)]
        equity = [_eq(1000.0, 5000.0, 30), _eq(1000.0, 5000.0, 1)]
        self.r = build_trader_scorecard(portfolio, positions, [], [], equity,
                                        now=NOW)

    def test_capital_is_ok(self):
        c = next(c for c in self.r["checks"]
                 if c["name"] == "capital_paralysis")
        assert c["klass"] == "OK"
        assert c["label"] == "FREE"

    def test_state_aligned_healthy_zero_flags(self):
        assert self.r["n_flags"] == 0
        assert self.r["n_ok"] >= 1
        assert self.r["state"] == "ALIGNED_HEALTHY"
        assert self.r["focus"] is None
        assert "healthy" in self.r["headline"].lower()


# ─────────────────── classification table (correctness) ──────────────────

# (builder name, verdict/state/status dict, expected klass)
_CLASSIFY_CASES = [
    ("trade_asymmetry", {"state": "STABLE", "verdict": "PAYOFF_TRAP"}, "FLAG"),
    ("trade_asymmetry", {"state": "STABLE", "verdict": "DISPOSITION_BLEED"}, "FLAG"),
    ("trade_asymmetry", {"state": "STABLE", "verdict": "EDGE_POSITIVE"}, "OK"),
    ("trade_asymmetry", {"state": "STABLE", "verdict": "FLAT"}, "OK"),
    ("trade_asymmetry", {"state": "EMERGING", "verdict": None}, "IMMATURE"),
    ("trade_asymmetry", {"state": "NO_DATA", "verdict": None}, "IMMATURE"),
    ("churn", {"state": "STABLE", "verdict": "CHURNING"}, "FLAG"),
    ("churn", {"state": "STABLE", "verdict": "BUY_AND_HOLD"}, "OK"),
    ("churn", {"state": "STABLE", "verdict": "ACTIVE_TURNOVER"}, "OK"),
    ("churn", {"state": "EMERGING", "verdict": None}, "IMMATURE"),
    ("capital_paralysis", {"state": "PINNED"}, "FLAG"),
    ("capital_paralysis", {"state": "EMPTY"}, "FLAG"),
    ("capital_paralysis", {"state": "FREE"}, "OK"),
    ("capital_paralysis", {"state": "NO_DATA"}, "IMMATURE"),
    ("decision_reliability", {"state": "CRITICAL"}, "FLAG"),
    ("decision_reliability", {"state": "DEGRADED"}, "FLAG"),
    ("decision_reliability", {"state": "STALE_LEGACY_DOMINATED"}, "FLAG"),
    ("decision_reliability", {"state": "HEALTHY"}, "OK"),
    ("decision_reliability", {"state": "INSUFFICIENT"}, "IMMATURE"),
    ("decision_reliability", {"state": "NO_DATA"}, "IMMATURE"),
    ("open_attribution", {"status": "SELECTION_DRAG"}, "FLAG"),
    ("open_attribution", {"status": "SELECTION_ADDING"}, "OK"),
    ("open_attribution", {"status": "FLAT_VS_SPY"}, "OK"),
    ("open_attribution", {"status": "NO_BENCHMARK"}, "IMMATURE"),
    ("open_attribution", {"status": "NO_DATA"}, "IMMATURE"),
    # unknown label must fail safe to IMMATURE, never invent a FLAG
    ("trade_asymmetry", {"state": "STABLE", "verdict": "WAT"}, "IMMATURE"),
    ("capital_paralysis", {"state": "SOMETHING_NEW"}, "IMMATURE"),
    # an ERROR marker from _safe is its own class
    ("churn", {"state": "ERROR", "error": "boom"}, "ERROR"),
]


@pytest.mark.parametrize("name,result,expected", _CLASSIFY_CASES)
def test_classify_check_table(name, result, expected):
    assert classify_check(name, result) == expected


# ──────────────────── a faulting builder is contained ───────────────────

class TestBuilderFaultContained:
    def test_one_builder_raising_does_not_sink_scorecard(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("synthetic builder failure")

        monkeypatch.setattr(tsc, "build_trade_asymmetry", _boom)
        # must not raise
        r = build_trader_scorecard({"cash": 900.0, "total_value": 1000.0},
                                   [_pos("NVDA", 1.0, 100.0, 100.0, 2)],
                                   [], [], [_eq(1000.0, 5000.0, 1)], now=NOW)
        c = next(c for c in r["checks"] if c["name"] == "trade_asymmetry")
        assert c["klass"] == "ERROR"
        # ERROR is not counted as a flag (no invented pathology)
        assert all(c["name"] != "trade_asymmetry"
                   for c in r["checks"] if c["klass"] == "FLAG")
        assert r["n_error"] >= 1


# ──────────────── endpoint end-to-end (Flask test client) ────────────────
# Per project memory: verify endpoints via the Flask test client, NOT a
# module __main__ smoke (that hits a different/empty DB).

@pytest.fixture
def seeded_client(tmp_path, monkeypatch):
    from paper_trader import store as store_mod
    from paper_trader.store import Store

    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    # 21 losing LITE round-trips → STABLE PAYOFF_TRAP + CHURNING; pinned book.
    for _ in range(21):
        s.record_trade("LITE", "BUY", 1.0, 100.0)
        s.record_trade("LITE", "SELL", 1.0, 90.0)
    s.update_portfolio(cash=6.0, total_value=1000.0, positions=[])

    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    try:
        with dashboard.app.test_client() as client:
            yield client
    finally:
        s.close()


class TestScorecardEndpoint:
    def test_endpoint_returns_scorecard_shape(self, seeded_client):
        resp = seeded_client.get("/api/scorecard")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "error" not in data, data
        assert data["state"] in ("NO_DATA", "ALIGNED_HEALTHY", "FLAGS_PRESENT")
        assert {"checks", "flags", "concordance", "focus", "headline",
                "n_flags", "n_total"} <= set(data)
        # the seeded pathological ledger must surface the real verdicts
        assert data["state"] == "FLAGS_PRESENT"
        labels = {c["name"]: c["label"] for c in data["checks"]}
        assert labels["trade_asymmetry"] == "PAYOFF_TRAP"
        assert labels["churn"] == "CHURNING"
