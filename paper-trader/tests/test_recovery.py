"""Tests for analytics/recovery.py — the forward "path back to even"
builder (the forward complement to the backward /api/drawdown).

Discriminating locks (each fails on a specific regression, not "no crash"):

* **Exact hand-computed** per-position breakeven %/$ and book
  to_initial/to_peak %/$ on a pinned underwater book (a sign flip or a
  dropped figure is caught).
* **Option lot ratio is multiplier-invariant** — breakeven % uses the
  raw price ratio, NOT ×100; a reviewer "fixing" it by multiplying
  through turns the test red (advisor discriminator).
* **σ no-drift**: the σ-day figure recomputed from an independent
  ``annualized_vol_pct / √252`` must equal the builder's.
* **Dispersion honesty gate**: ``tail_risk.state != OK`` → %/$ still
  emitted, σ withheld with an honest sentence; ``OK`` → σ emitted. A
  regression that reads vol regardless of the verdict fails.
* **State ladder**: NO_DATA / ABOVE_WATER (line suppressed) / UNDERWATER.
* **Endpoint↔fold↔builder no-drift** via the real Flask test client.
* **Reporter line**: suppressed on ABOVE_WATER/NO_DATA, builder headline
  verbatim on UNDERWATER, builder fault → "" and the summary still sends.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.recovery import build_recovery

_SQRT_TD = math.sqrt(252)


def _dd(current, peak, contributors=None):
    """Minimal compute_drawdown-shaped result the builder consumes."""
    return {
        "current_value": current,
        "peak_value": peak,
        "contributors": contributors or [],
    }


def _contrib(ticker, avg_cost, current_price, unrealized_pl, type_="stock"):
    return {
        "ticker": ticker,
        "type": type_,
        "avg_cost": avg_cost,
        "current_price": current_price,
        "unrealized_pl": unrealized_pl,
    }


class TestExactRecoveryMath:
    """Pinned 2-name underwater book: every figure hand-checkable."""

    def _build(self):
        dd = _dd(800.0, 1100.0, [
            _contrib("MU", 100.0, 80.0, -40.0),
            _contrib("LITE", 200.0, 150.0, -100.0),
        ])
        tr = {"state": "OK", "annualized_vol_pct": 40.0}
        return build_recovery(dd, tr, 1000.0)

    def test_state_and_book_targets_exact(self):
        r = self._build()
        assert r["state"] == "UNDERWATER"
        # to even: (1000/800 - 1)*100 = 25.0 %, $200
        assert r["to_initial_pct"] == 25.0
        assert r["to_initial_usd"] == 200.0
        # to peak: (1100/800 - 1)*100 = 37.5 %, $300
        assert r["to_peak_pct"] == 37.5
        assert r["to_peak_usd"] == 300.0
        assert r["current_value"] == 800.0
        assert r["peak_value"] == 1100.0

    def test_per_position_breakeven_exact_and_sorted_heaviest_first(self):
        r = self._build()
        # Heaviest $-to-recover leads (LITE $100 before MU $40).
        assert [p["ticker"] for p in r["positions"]] == ["LITE", "MU"]
        lite, mu = r["positions"]
        # (200/150 - 1)*100 = 33.33 %, recover $100
        assert lite["breakeven_pct"] == 33.33
        assert lite["dollars_to_recover"] == 100.0
        # (100/80 - 1)*100 = 25.0 %, recover $40
        assert mu["breakeven_pct"] == 25.0
        assert mu["dollars_to_recover"] == 40.0

    def test_sigma_days_independent_recompute_no_drift(self):
        r = self._build()
        daily_sigma = 40.0 / _SQRT_TD
        assert r["daily_vol_pct"] == round(daily_sigma, 2)
        assert r["dispersion_state"] == "OK"
        assert r["to_initial_sigma_days"] == round(25.0 / daily_sigma, 1)
        assert r["to_peak_sigma_days"] == round(37.5 / daily_sigma, 1)

    def test_headline_carries_the_decision_relevant_figures(self):
        r = self._build()
        h = r["headline"]
        assert "underwater $-200.00 (-25.00%)" in h
        assert "+25.00% ($200.00) to even" in h
        assert "+37.50% to the $1,100.00 peak" in h
        assert "dispersion scale" in h.lower()
        assert "LITE carries the most ($+100.00" in h
        # prompt block restates the no-forecast honesty.
        assert "not an estimated time-to-recover" in r["prompt_block"]


class TestWinnerAndOptionLot:
    def test_winner_needs_zero_never_negative_noise(self):
        dd = _dd(950.0, 1000.0, [
            _contrib("NVDA", 50.0, 80.0, 60.0),   # in profit
            _contrib("MU", 100.0, 90.0, -10.0),
        ])
        r = build_recovery(dd, {"state": "INSUFFICIENT"}, 1000.0)
        by = {p["ticker"]: p for p in r["positions"]}
        assert by["NVDA"]["breakeven_pct"] == 0.0
        assert by["NVDA"]["dollars_to_recover"] == 0.0
        assert by["MU"]["breakeven_pct"] == round((100.0 / 90.0 - 1) * 100, 2)

    def test_option_breakeven_is_price_ratio_not_times_100(self):
        # avg 2.00 → now 1.00, qty 1 (×100 baked into unrealized_pl=-100).
        # Breakeven % is the RAW price ratio (multiplier-invariant): a
        # reviewer multiplying by 100 would make this 10000.0 — RED.
        dd = _dd(400.0, 600.0, [
            _contrib("MU260116C00100", 2.00, 1.00, -100.0, type_="call"),
        ])
        r = build_recovery(dd, {"state": "OK", "annualized_vol_pct": 50.0},
                           1000.0)
        opt = r["positions"][0]
        assert opt["breakeven_pct"] == 100.0          # (2/1 - 1)*100
        assert opt["dollars_to_recover"] == 100.0     # = -unrealized_pl


class TestDispersionHonestyGate:
    """The discriminating lock: tail_risk numerics exist even when the
    verdict is withheld; the σ figure must still be suppressed unless
    state == OK (the young-book honesty precedent)."""

    def test_insufficient_emits_targets_but_withholds_sigma(self):
        dd = _dd(800.0, 900.0, [_contrib("MU", 100.0, 80.0, -40.0)])
        # tail_risk emits annualized_vol_pct even when INSUFFICIENT.
        tr = {"state": "INSUFFICIENT", "annualized_vol_pct": 33.0}
        r = build_recovery(dd, tr, 1000.0)
        assert r["to_initial_pct"] == 25.0            # %/$ still emitted
        assert r["to_initial_usd"] == 200.0
        assert r["to_initial_sigma_days"] is None     # σ withheld
        assert r["daily_vol_pct"] is None
        assert r["dispersion_state"] == "WITHHELD"
        assert "withheld" in r["headline"].lower()

    def test_none_tail_risk_withholds_sigma(self):
        dd = _dd(800.0, 900.0, [_contrib("MU", 100.0, 80.0, -40.0)])
        r = build_recovery(dd, None, 1000.0)
        assert r["state"] == "UNDERWATER"
        assert r["to_initial_pct"] == 25.0
        assert r["dispersion_state"] == "WITHHELD"

    def test_ok_emits_sigma(self):
        dd = _dd(800.0, 900.0, [_contrib("MU", 100.0, 80.0, -40.0)])
        r = build_recovery(dd, {"state": "OK", "annualized_vol_pct": 30.0},
                           1000.0)
        assert r["dispersion_state"] == "OK"
        assert r["to_initial_sigma_days"] == round(
            25.0 / (30.0 / _SQRT_TD), 1)


class TestStateLadder:
    def test_not_a_dict_is_no_data(self):
        r = build_recovery(None, {"state": "OK"}, 1000.0)
        assert r["state"] == "NO_DATA"
        assert r["positions"] == []

    @pytest.mark.parametrize("cur", [None, 0.0, -5.0, "junk"])
    def test_non_positive_or_unparseable_current_is_no_data(self, cur):
        r = build_recovery(_dd(cur, 1000.0), {"state": "OK"}, 1000.0)
        assert r["state"] == "NO_DATA"

    def test_above_water_zeros_to_initial_and_sets_state(self):
        # Book above the $1000 start but below an intra-history peak.
        dd = _dd(1100.0, 1300.0, [_contrib("NVDA", 50.0, 80.0, 60.0)])
        r = build_recovery(dd, {"state": "OK", "annualized_vol_pct": 20.0},
                           1000.0)
        assert r["state"] == "ABOVE_WATER"
        assert r["to_initial_pct"] == 0.0
        assert r["to_initial_usd"] == 0.0
        # peak gap still reported for completeness
        assert r["to_peak_pct"] == round((1300.0 / 1100.0 - 1) * 100, 2)

    def test_underwater_with_peak_below_current_zeros_peak_gap(self):
        # Fresh low: peak == current (drawdown's running-peak == latest).
        dd = _dd(800.0, 800.0, [_contrib("MU", 100.0, 80.0, -40.0)])
        r = build_recovery(dd, {"state": "OK", "annualized_vol_pct": 25.0},
                           1000.0)
        assert r["state"] == "UNDERWATER"
        assert r["to_peak_pct"] == 0.0
        assert r["to_peak_usd"] == 0.0


class TestSafetyNeverRaises:
    def test_garbage_contributors_never_raise(self):
        dd = {
            "current_value": 800.0,
            "peak_value": 1000.0,
            "contributors": [
                None, 42, "nope",
                {"ticker": "X", "avg_cost": None,
                 "current_price": None, "unrealized_pl": None},
            ],
        }
        r = build_recovery(dd, {"state": "OK", "annualized_vol_pct": 30.0},
                           1000.0)
        assert r["state"] == "UNDERWATER"
        # The all-None row coerces to a flat (0.0) breakeven, never raises.
        assert r["positions"][0]["breakeven_pct"] == 0.0

    def test_missing_peak_defaults_to_current(self):
        r = build_recovery(
            {"current_value": 800.0, "contributors": []},
            {"state": "OK", "annualized_vol_pct": 30.0}, 1000.0)
        assert r["peak_value"] == 800.0
        assert r["to_peak_pct"] == 0.0


# ─────────────────────────── /api/recovery parity ───────────────────────────

class TestRecoveryEndpoint:
    """Endpoint↔/api/analytics-fold↔builder no-drift (the tail_risk /
    stress_scenarios discipline) on a real seeded Store."""

    def _seed_underwater(self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        s.upsert_position("MU", "stock", 5.0, 100.0)
        s.upsert_position("LITE", "stock", 3.0, 200.0)
        # Underwater equity points (all today → tail_risk reads young).
        s.record_equity_point(1000.0, 100.0, None)
        s.record_equity_point(820.0, 20.0, None)
        s.update_portfolio(cash=20.0, total_value=820.0, positions=[])
        return s

    def test_endpoint_and_fold_match_builder(self, tmp_path, monkeypatch):
        s = self._seed_underwater(tmp_path, monkeypatch)
        from paper_trader import dashboard
        from paper_trader.analytics.drawdown import compute_drawdown
        from paper_trader.analytics.recovery import build_recovery as br
        from paper_trader.analytics.tail_risk import build_tail_risk
        from paper_trader.store import INITIAL_CASH
        dashboard.app.config["TESTING"] = True
        try:
            with dashboard.app.test_client() as client:
                ep = client.get("/api/recovery").get_json()
                an = client.get("/api/analytics").get_json()
            eq = s.equity_curve(2000)
            pos = s.open_positions()
            ref = br(compute_drawdown(eq, pos, starting_equity=INITIAL_CASH),
                     build_tail_risk(eq), INITIAL_CASH)
        finally:
            s.close()
        assert "error" not in ep, ep
        assert "recovery" in an, list(an)            # additive key present
        for got in (ep, an["recovery"]):
            assert got["state"] == "UNDERWATER" == ref["state"]
            assert got["to_initial_pct"] == ref["to_initial_pct"]
            assert got["to_initial_usd"] == ref["to_initial_usd"]
            assert got["positions"] == ref["positions"]
            assert got["headline"] == ref["headline"]

    def test_empty_book_is_above_water_not_500(self, tmp_path, monkeypatch):
        # compute_drawdown's empty-book fallback returns
        # current_value == starting_equity (the green at-high-water badge
        # contract), so a fresh $1000 book is correctly ABOVE_WATER
        # (at even, nothing to recover) — never a 500, never a false
        # UNDERWATER alarm, and the Discord line self-suppresses.
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        db = tmp_path / "empty.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        from paper_trader import dashboard
        try:
            with dashboard.app.test_client() as client:
                resp = client.get("/api/recovery")
        finally:
            s.close()
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["state"] == "ABOVE_WATER"
        assert body["to_initial_pct"] == 0.0


# ───────────────────────── reporter _recovery_line ──────────────────────────

class TestReporterRecoveryLine:
    def _store(self, tmp_path, monkeypatch, name):
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        db = tmp_path / name
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        return Store()

    def test_no_data_book_suppressed(self, tmp_path, monkeypatch):
        from paper_trader import reporter
        s = self._store(tmp_path, monkeypatch, "a.db")
        try:
            assert reporter._recovery_line(s) == ""    # empty book → silent
        finally:
            s.close()

    def test_above_water_suppressed(self, tmp_path, monkeypatch):
        from paper_trader import reporter
        s = self._store(tmp_path, monkeypatch, "b.db")
        try:
            # total_value above the $1000 start → nothing to recover.
            s.record_equity_point(1200.0, 1200.0, None)
            s.update_portfolio(cash=1200.0, total_value=1200.0, positions=[])
            assert reporter._recovery_line(s) == ""
        finally:
            s.close()

    def test_underwater_emits_builder_headline_verbatim(
            self, tmp_path, monkeypatch):
        from paper_trader import reporter
        from paper_trader.analytics.drawdown import compute_drawdown
        from paper_trader.analytics.recovery import build_recovery as br
        from paper_trader.analytics.tail_risk import build_tail_risk
        s = self._store(tmp_path, monkeypatch, "c.db")
        try:
            s.upsert_position("MU", "stock", 5.0, 100.0)
            s.record_equity_point(820.0, 20.0, None)
            s.update_portfolio(cash=20.0, total_value=820.0, positions=[])
            line = reporter._recovery_line(s)
            assert line != "" and "RECOVERY" in line
            eq = s.equity_curve(2000)
            ref = br(compute_drawdown(eq, s.open_positions(),
                                      starting_equity=reporter._INITIAL_EQUITY),
                     build_tail_risk(eq), reporter._INITIAL_EQUITY)
            assert ref["headline"] in line          # verbatim, no re-derive
        finally:
            s.close()

    def test_builder_fault_degrades_to_empty_never_raises(
            self, tmp_path, monkeypatch):
        from paper_trader import reporter
        s = self._store(tmp_path, monkeypatch, "d.db")
        s.upsert_position("MU", "stock", 5.0, 100.0)
        s.record_equity_point(820.0, 20.0, None)
        s.update_portfolio(cash=20.0, total_value=820.0, positions=[])
        monkeypatch.setattr(
            "paper_trader.analytics.recovery.build_recovery",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        try:
            assert reporter._recovery_line(s) == ""   # fault → "", no raise
        finally:
            s.close()

    def test_summary_still_sends_when_recovery_builder_faults(
            self, tmp_path, monkeypatch):
        """The "no block, never no summary" failure contract."""
        from paper_trader import reporter
        s = self._store(tmp_path, monkeypatch, "e.db")
        s.upsert_position("MU", "stock", 5.0, 100.0)
        s.record_equity_point(820.0, 20.0, None)
        s.update_portfolio(cash=20.0, total_value=820.0, positions=[])
        monkeypatch.setattr(reporter, "get_store", lambda: s)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: 5000.0)
        monkeypatch.setattr(
            "paper_trader.analytics.recovery.build_recovery",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        sent = {}

        def _fake_send(body):
            sent["body"] = body
            return True

        monkeypatch.setattr(reporter, "_send", _fake_send)
        try:
            assert reporter.send_hourly_summary() is True
            assert "HOURLY" in sent["body"]           # summary still sent
            assert "RECOVERY" not in sent["body"]     # just no recovery line
        finally:
            s.close()
