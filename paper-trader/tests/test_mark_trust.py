"""End-to-end Flask-client tests for the `mark_trust` honesty key folded
into the equity-derived risk endpoints (/api/tail-risk, /api/drawdown).

Why this exists: `build_mark_integrity` quantifies how much of the current
book is marked at cost (stale price feed) — and its own docstring names the
victims it does NOT fix: "/api/analytics Sharpe, /api/drawdown, the equity
curve ... all quietly partially false, with nothing saying so". A grep shows
`stale_mark` reaches only mark_integrity/strategy/dashboard/reporter — the
tail-risk & drawdown surfaces compute VaR/vol/skew/max-DD over an equity
series whose stale-cycle points are cost-frozen flats, with zero caveat
(flat artefacts deflate vol & drawdown, inflate Sharpe, truncate the tail).

This wires the EXISTING single source of truth (`build_mark_integrity`,
computed from the SAME write-free `portfolio_snapshot_readonly` snapshot the
/api/mark-integrity endpoint uses) into those endpoints as an additive
`mark_trust` key. These tests lock: the key surfaces a stale book, reads
CLEAN otherwise, is purely additive (no risk number drifts), is `_safe`
(a snapshot fault → no key, never a 500), and never re-derives staleness
(single source of truth, AGENTS.md #10).

Convention mirrors test_decision_context_endpoint.py (real Flask app, real
temp Store, get_prices→{} = the realistic yfinance-starvation stale shape).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.dashboard as d
import paper_trader.market as market_mod
import paper_trader.store as store_mod
from paper_trader import strategy
from paper_trader.analytics.mark_integrity import build_mark_integrity
from paper_trader.analytics.tail_risk import build_tail_risk
from paper_trader.store import Store


@pytest.fixture
def client_store(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    s.record_trade("NVDA", "BUY", 2, 100.0, reason="momentum")
    s.upsert_position("NVDA", "stock", 2, 100.0)
    s.record_equity_point(200.0, 0.0, 5800.0)
    s.record_equity_point(180.0, 0.0, 5790.0)
    d.app.config["TESTING"] = True
    try:
        with d.app.test_client() as client:
            yield client, s
    finally:
        s.close()


def _stale(monkeypatch):
    """Realistic yfinance-starvation: no price for the held name → the
    readonly snapshot marks it at avg_cost and flags stale_mark=True."""
    monkeypatch.setattr(market_mod, "get_prices", lambda tks: {})


def _live(monkeypatch):
    monkeypatch.setattr(market_mod, "get_prices", lambda tks: {"NVDA": 120.0})


class TestTailRiskMarkTrust:
    def test_stale_book_surfaces_mark_trust(self, client_store, monkeypatch):
        client, _s = client_store
        _stale(monkeypatch)
        j = client.get("/api/tail-risk").get_json()
        assert "mark_trust" in j, "stale book must surface the honesty key"
        mt = j["mark_trust"]
        # single NVDA position, no price → 100% of gross marked at cost
        assert mt["verdict"] == "UNTRUSTWORTHY"
        assert mt["n_positions"] == 1
        assert mt["n_stale"] == 1
        assert mt["stale_value_pct"] == 100.0
        assert mt["stale_tickers"] == ["NVDA"]
        assert isinstance(mt["note"], str) and "cost" in mt["note"].lower()

    def test_clean_book_mark_trust_is_clean(self, client_store, monkeypatch):
        client, _s = client_store
        _live(monkeypatch)
        mt = client.get("/api/tail-risk").get_json()["mark_trust"]
        assert mt["verdict"] == "CLEAN"
        assert mt["n_stale"] == 0
        # a trustworthy book carries no scary caveat
        assert mt["note"] is None

    def test_fold_is_additive_no_risk_drift(self, client_store, monkeypatch):
        """The fold must add EXACTLY one key and mutate no risk number —
        a regression that recomputed the curve or dropped a field fails
        here (build_tail_risk is deterministic given the same curve;
        `as_of` is the only wall-clock field, excluded)."""
        client, s = client_store
        _stale(monkeypatch)
        j = client.get("/api/tail-risk").get_json()
        direct = build_tail_risk(s.equity_curve(5000))
        assert set(j) - set(direct) == {"mark_trust"}
        for k, v in direct.items():
            if k == "as_of":
                continue
            assert j[k] == v, f"risk field {k!r} drifted: {j[k]!r} != {v!r}"

    def test_safe_contract_snapshot_fault_keeps_200_no_key(
        self, client_store, monkeypatch
    ):
        """A diagnostics fault must not 500 the risk endpoint or leak a
        half-built key — the behavioural-builder _safe contract. A naive
        try-less fold 500s here."""
        client, _s = client_store

        def _boom(*a, **k):
            raise RuntimeError("snapshot exploded")

        monkeypatch.setattr(strategy, "portfolio_snapshot_readonly", _boom)
        r = client.get("/api/tail-risk")
        assert r.status_code == 200
        j = r.get_json()
        assert "mark_trust" not in j
        assert "error" not in j
        # the real risk payload is untouched (still the tail-risk shape)
        assert "n_returns" in j and "var_95_pct" in j


class TestDrawdownMarkTrust:
    def test_stale_book_surfaces_mark_trust(self, client_store, monkeypatch):
        client, _s = client_store
        _stale(monkeypatch)
        j = client.get("/api/drawdown").get_json()
        assert "mark_trust" in j
        mt = j["mark_trust"]
        assert mt["verdict"] == "UNTRUSTWORTHY"
        assert mt["n_stale"] == 1
        assert mt["stale_tickers"] == ["NVDA"]

    def test_clean_book_mark_trust_is_clean(self, client_store, monkeypatch):
        client, _s = client_store
        _live(monkeypatch)
        assert client.get("/api/drawdown").get_json()["mark_trust"][
            "verdict"
        ] == "CLEAN"

    def test_safe_contract_snapshot_fault_keeps_200_no_key(
        self, client_store, monkeypatch
    ):
        client, _s = client_store

        def _boom(*a, **k):
            raise RuntimeError("snapshot exploded")

        monkeypatch.setattr(strategy, "portfolio_snapshot_readonly", _boom)
        r = client.get("/api/drawdown")
        assert r.status_code == 200
        j = r.get_json()
        assert "mark_trust" not in j and "error" not in j
        # drawdown payload intact
        assert "drawdown_pct" in j and "at_high_water" in j


class TestAnalyticsMarkTrust:
    """/api/analytics is the #1 victim mark_integrity's docstring names
    ("/api/analytics Sharpe ... quietly partially false"). It already folds
    build_tail_risk as an additive key with keyed assertions — the same
    contract carries mark_trust with no whole-dict-equality risk."""

    def test_stale_book_surfaces_mark_trust(self, client_store, monkeypatch):
        client, _s = client_store
        _stale(monkeypatch)
        j = client.get("/api/analytics").get_json()
        assert "mark_trust" in j
        mt = j["mark_trust"]
        assert mt["verdict"] == "UNTRUSTWORTHY"
        assert mt["n_stale"] == 1
        assert mt["stale_tickers"] == ["NVDA"]
        # the headline Sharpe is still present — additive, not replaced
        assert "sharpe_annualized" in j and "tail_risk" in j

    def test_clean_book_mark_trust_is_clean(self, client_store, monkeypatch):
        client, _s = client_store
        _live(monkeypatch)
        assert client.get("/api/analytics").get_json()["mark_trust"][
            "verdict"
        ] == "CLEAN"

    def test_safe_contract_snapshot_fault_keeps_200_no_key(
        self, client_store, monkeypatch
    ):
        client, _s = client_store

        def _boom(*a, **k):
            raise RuntimeError("snapshot exploded")

        monkeypatch.setattr(strategy, "portfolio_snapshot_readonly", _boom)
        r = client.get("/api/analytics")
        assert r.status_code == 200
        j = r.get_json()
        assert "mark_trust" not in j and "error" not in j
        # full analytics payload intact (Sharpe + the existing tail_risk fold)
        assert "sharpe_annualized" in j and "tail_risk" in j


class TestSingleSourceOfTruth:
    def test_mark_trust_equals_build_mark_integrity_verbatim(
        self, client_store, monkeypatch
    ):
        """The fold must COMPOSE build_mark_integrity (AGENTS.md #10), not
        re-derive staleness. Independently run the canonical builder over
        the same readonly snapshot and assert no drift."""
        client, s = client_store
        _stale(monkeypatch)
        mt = client.get("/api/tail-risk").get_json()["mark_trust"]
        snap = strategy.portfolio_snapshot_readonly(s)
        canonical = build_mark_integrity(snap.get("positions") or [])
        assert mt["verdict"] == canonical["verdict"]
        assert mt["n_stale"] == canonical["n_stale"]
        assert mt["n_positions"] == canonical["n_positions"]
        assert mt["stale_value_pct"] == canonical["stale_value_pct"]
        assert mt["stale_tickers"] == canonical["stale_tickers"]
        assert mt["headline"] == canonical["headline"]
