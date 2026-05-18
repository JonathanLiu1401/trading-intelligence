"""Tests for analytics/stress_scenarios.py — the forward beta/concentration
shock builder (the day-one complement to the history-gated tail_risk).

The discriminating locks:

* **SSOT no-drift**: the −3 % market scenario is asserted equal to an
  independent recompute of ``/api/risk``'s ``Σ −0.03·β·val`` shock — a
  drift in either the builder or the risk endpoint fails loudly.
* **Exact hand-computed P&L** for every scenario family on a pinned book
  (an off-by-sign or a dropped beta is caught, not "no crash").
* **Monotonicity**: |loss| strictly grows −1 → −3 → −5 → −10 %.
* **Option beta path**: ×3 capped at 4, **negated for puts** (a put book
  *gains* on an SPY sell-off).
* **No sample-size gate** — the whole point vs tail_risk — verified by an
  OK verdict on a one-position book.
* **Never raises / `_safe`**: garbage rows, None, zero book.
* Prompt render order + reporter line + endpoint↔builder parity.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.stress_scenarios import (
    _LEVERAGE_BETA as _SS_BETA,
    build_stress_scenarios,
)


class TestBetaMapIsPinnedToDashboard:
    """The hot-path copy must never drift from the /api/risk SSOT (the
    sector_exposure SECTOR_MAP == dashboard.SECTOR_MAP precedent)."""

    def test_leverage_beta_equals_dashboard(self):
        from paper_trader import dashboard
        assert _SS_BETA == dashboard._LEVERAGE_BETA

    def test_sector_exposure_classify_matches_dashboard(self):
        # The strategy hot path passes sector_exposure.classify; it must
        # agree with dashboard._classify so the prompt block and
        # /api/stress-scenarios cannot diverge.
        from paper_trader import dashboard
        from paper_trader.analytics.sector_exposure import classify as sx_cls
        for t in ("MU", "NVDA", "LITE", "TQQQ", "SOXL", "UNKNOWNX"):
            assert sx_cls(t) == dashboard._classify(t)

# Fixture book mirroring the live pathology shape (one optical, one semis),
# clean numbers so every figure is hand-checkable.
_BETA = {"optical": 1.4, "semis": 1.5}
_SEC = {"LITE": "optical", "MU": "semis", "NVDA": "semis"}


def _classify(t):
    return _SEC.get((t or "").upper(), "other")


_BOOK = [
    {"ticker": "LITE", "type": "stock", "qty": 10, "current_price": 100.0,
     "avg_cost": 90.0},
    {"ticker": "MU", "type": "stock", "qty": 5, "current_price": 50.0,
     "avg_cost": 50.0},
]
# LITE val=1000 (β1.4), MU val=250 (β1.5). gross=1250. β-weighted base=1775.
_TV = 1250.0
_NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _build(positions=_BOOK, tv=_TV):
    return build_stress_scenarios(positions, tv, _classify, _BETA, now=_NOW)


def _sc(res, label):
    return next(s for s in res["scenarios"] if s["label"] == label)


class TestExactScenarioMath:
    def test_market_scenarios_exact(self):
        r = _build()
        assert r["state"] == "OK"
        assert r["gross_value_usd"] == 1250.0
        # frac * (1.4*1000 + 1.5*250) = frac * 1775
        assert _sc(r, "SPY -1%")["pnl_usd"] == -17.75
        assert _sc(r, "SPY -3%")["pnl_usd"] == -53.25
        assert _sc(r, "SPY -5%")["pnl_usd"] == -88.75
        assert _sc(r, "SPY -10%")["pnl_usd"] == -177.5
        assert _sc(r, "SPY +3%")["pnl_usd"] == 53.25
        # pnl_pct is against the passed total_value (1250), not gross.
        assert _sc(r, "SPY -10%")["pnl_pct"] == -14.2
        assert _sc(r, "SPY -3%")["pnl_pct"] == -4.26

    def test_minus3_market_equals_api_risk_shock_formula(self):
        """SSOT (AGENTS.md #10): the −3 % line MUST equal an independent
        recompute of /api/risk's `shock_usd = Σ -0.03·β·val`."""
        r = _build()
        independent = 0.0
        for p in _BOOK:
            val = p["current_price"] * p["qty"]
            independent += -0.03 * _BETA[_classify(p["ticker"])] * val
        assert _sc(r, "SPY -3%")["pnl_usd"] == round(independent, 2) == -53.25

    def test_loss_is_monotone_in_shock_size(self):
        r = _build()
        losses = [abs(_sc(r, f"SPY -{m}%")["pnl_usd"]) for m in (1, 3, 5, 10)]
        assert losses == sorted(losses)
        assert len(set(losses)) == 4  # strictly increasing, no ties

    def test_single_name_gap_is_top_position_no_beta(self):
        r = _build()
        sn = r["single_name_gap"]
        assert sn["ticker"] == "LITE"            # 1000 > 250
        assert sn["gap_pct"] == -10.0
        assert sn["pnl_usd"] == -100.0           # -0.10 * 1000, NO beta
        assert sn["weight_pct"] == 80.0          # 1000 / 1250
        assert sn["pnl_pct"] == -8.0

    def test_sector_shock_isolates_heaviest_sector_no_beta(self):
        r = _build()
        ss = r["sector_shock"]
        assert ss["sector"] == "optical"         # 1000 > 250 (semis)
        assert ss["n_names"] == 1
        assert ss["pnl_usd"] == -100.0           # -0.10 * 1000, NO beta
        assert ss["weight_pct"] == 80.0

    def test_headline_names_the_worst_scenario(self):
        r = _build()
        # Worst over all 7 candidates is SPY -10% (-177.50) < single/sector -100.
        assert "SPY -10%" in r["headline"]
        assert "-177.50" in r["headline"]
        assert "beta-approx" in r["headline"]    # honesty disclosure


class TestOptionBetaPath:
    def test_put_book_gains_on_selloff_beta_capped_and_negated(self):
        # NVDA put: val = 3 * 2 * 100 = 600; β = min(1.5*3,4)=4.0 → -4.0.
        book = [{"ticker": "NVDA", "type": "put", "qty": 2,
                 "current_price": 3.0, "avg_cost": 3.0,
                 "strike": 100.0, "expiry": "2026-06-19"}]
        r = build_stress_scenarios(book, 600.0, _classify, _BETA, now=_NOW)
        # SPY -10%: -0.10 * (-4.0) * 600 = +240 (a put profits on a drop).
        assert _sc(r, "SPY -10%")["pnl_usd"] == 240.0
        assert _sc(r, "SPY +3%")["pnl_usd"] == -72.0


class TestStateLadderAndSafety:
    def test_one_position_book_is_OK_no_sample_gate(self):
        """The feature's reason to exist: a real number on day one, where
        tail_risk would read INSUFFICIENT."""
        r = build_stress_scenarios(
            [{"ticker": "MU", "type": "stock", "qty": 1,
              "current_price": 100.0, "avg_cost": 100.0}],
            100.0, _classify, _BETA, now=_NOW)
        assert r["state"] == "OK"
        assert _sc(r, "SPY -10%")["pnl_usd"] == -15.0   # -0.10*1.5*100

    @pytest.mark.parametrize("positions,tv", [
        ([], 1000.0),                                   # no positions
        (None, 1000.0),                                 # None
        (_BOOK, 0.0),                                    # zero total value
        ([{"ticker": "X", "type": "stock", "qty": 0,
           "current_price": 0.0, "avg_cost": 0.0}], 100.0),  # unpriceable
    ])
    def test_no_data_states(self, positions, tv):
        r = build_stress_scenarios(positions, tv, _classify, _BETA, now=_NOW)
        assert r["state"] == "NO_DATA"
        assert r["scenarios"] == []
        assert r["prompt_block"] is None
        assert "no priced book" in r["headline"]

    def test_garbage_row_skipped_never_raises(self):
        book = [
            {"ticker": "MU", "type": "stock", "qty": "abc",
             "current_price": 50.0, "avg_cost": 50.0},          # bad qty
            {"ticker": "LITE", "type": "stock", "qty": 10,
             "current_price": 100.0, "avg_cost": 90.0},         # good
        ]
        r = build_stress_scenarios(book, 1000.0, _classify, _BETA, now=_NOW)
        assert r["state"] == "OK"
        assert r["n_positions"] == 1                            # bad row dropped
        assert r["gross_value_usd"] == 1000.0

    def test_classify_raising_does_not_sink_builder(self):
        def boom(_t):
            raise RuntimeError("classify exploded")
        r = build_stress_scenarios(_BOOK, _TV, boom, _BETA, now=_NOW)
        # Every row drops (classify raises inside the per-row guard) → NO_DATA,
        # never an exception (the _safe contract).
        assert r["state"] == "NO_DATA"


# ───────────────────────── prompt render order ─────────────────────────

def test_build_payload_renders_stress_after_sector_before_event():
    from paper_trader import strategy
    snap = {"positions": [], "cash": 1000.0,
            "open_value": 0.0, "total_value": 1000.0}
    payload = strategy._build_payload(
        snap, [], [], {}, {}, None, True,
        quant_signals={},
        sector_exposure_block="SECTOR-MARKER",
        stress_block="STRESS-BLOCK-MARKER forward shock",
        event_calendar_block="EVENT-MARKER",
    )
    assert "STRESS-BLOCK-MARKER" in payload
    assert payload.index("SECTOR-MARKER") < payload.index("STRESS-BLOCK-MARKER")
    assert payload.index("STRESS-BLOCK-MARKER") < payload.index("EVENT-MARKER")
    assert payload.index("STRESS-BLOCK-MARKER") < payload.index("WATCHLIST PRICES")


def test_build_payload_none_stress_renders_no_stray_text():
    from paper_trader import strategy
    snap = {"positions": [], "cash": 1000.0,
            "open_value": 0.0, "total_value": 1000.0}
    payload = strategy._build_payload(
        snap, [], [], {}, {}, None, True,
        quant_signals={}, stress_block=None)
    assert "FORWARD STRESS" not in payload
    assert "None" not in payload.split("PORTFOLIO")[1].split("WATCHLIST")[0]


# ───────────────────────── reporter line ─────────────────────────

class TestReporterStressLine:
    def test_no_data_suppressed(self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        from paper_trader import reporter
        db = tmp_path / "pt.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        try:
            assert reporter._stress_line(s) == ""   # empty book → silent
        finally:
            s.close()

    def test_headline_verbatim_when_book_priced(self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        from paper_trader import reporter
        db = tmp_path / "pt.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        try:
            s.upsert_position("MU", "stock", 5.0, 50.0)
            s.update_portfolio(cash=0.0, total_value=250.0, positions=[])
            line = reporter._stress_line(s)
            assert line != ""
            assert "FORWARD STRESS" in line
            # Composed verbatim from the builder headline (SSOT, no re-derive).
            from paper_trader.analytics.stress_scenarios import (
                build_stress_scenarios)
            from paper_trader.dashboard import _classify as dcl, _LEVERAGE_BETA
            ref = build_stress_scenarios(
                s.open_positions(), 250.0, dcl, _LEVERAGE_BETA)
            assert ref["headline"] in line
        finally:
            s.close()

    def test_builder_fault_degrades_to_empty_never_raises(
            self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        from paper_trader import reporter
        db = tmp_path / "pt.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        monkeypatch.setattr(
            reporter, "build_stress_scenarios",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        try:
            assert reporter._stress_line(s) == ""   # fault → no line, no raise
        finally:
            s.close()


# ───────────────────────── /api/stress-scenarios parity ─────────────────────────

class TestStressScenariosEndpoint:
    """Endpoint↔builder no-drift (the tail_risk discipline): the route and
    the additive /api/analytics key must both equal build_stress_scenarios
    on the same live store, recomputed with the dashboard's own
    _classify / _LEVERAGE_BETA SSOT (no hardcoded sector literals → robust
    to a SECTOR_MAP change)."""

    def _seed(self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        s.upsert_position("MU", "stock", 5.0, 50.0)
        s.upsert_position("LITE", "stock", 10.0, 100.0)
        s.update_portfolio(cash=10.0, total_value=760.0, positions=[])
        return s

    def test_endpoint_and_analytics_key_match_builder(
            self, tmp_path, monkeypatch):
        s = self._seed(tmp_path, monkeypatch)
        from paper_trader import dashboard
        from paper_trader.analytics.stress_scenarios import (
            build_stress_scenarios)
        dashboard.app.config["TESTING"] = True
        try:
            with dashboard.app.test_client() as client:
                ep = client.get("/api/stress-scenarios").get_json()
                an = client.get("/api/analytics").get_json()
            pf = s.get_portfolio()
            ref = build_stress_scenarios(
                s.open_positions(), pf.get("total_value") or 0.0,
                dashboard._classify, dashboard._LEVERAGE_BETA)
        finally:
            s.close()
        assert "error" not in ep, ep
        assert "error" not in an, an
        assert "stress_scenarios" in an            # additive key present
        for got in (ep, an["stress_scenarios"]):
            assert got["state"] == "OK" == ref["state"]
            assert got["scenarios"] == ref["scenarios"]
            assert got["single_name_gap"] == ref["single_name_gap"]
            assert got["sector_shock"] == ref["sector_shock"]
            assert got["headline"] == ref["headline"]

    def test_empty_book_endpoint_is_no_data_not_500(
            self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        db = tmp_path / "empty.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        from paper_trader import dashboard
        try:
            with dashboard.app.test_client() as client:
                resp = client.get("/api/stress-scenarios")
                an = client.get("/api/analytics").get_json()
        finally:
            s.close()
        assert resp.status_code == 200
        assert resp.get_json()["state"] == "NO_DATA"
        assert an["stress_scenarios"]["state"] == "NO_DATA"
