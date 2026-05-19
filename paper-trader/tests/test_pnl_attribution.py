"""Tests for analytics/pnl_attribution.py — β-adjusted unrealized P/L
decomposition.

The discriminator vs ``open_attribution`` (β=1 implicit) is that this
builder actually uses the sector β-map. On a leveraged-ETF book — TQQQ
β=3, NVDA β=1.5 — the two surfaces should *disagree* by design: that
disagreement IS the value-add. A regression that re-uses
``open_attribution``'s formula here would zero out the disagreement and
silently revert this builder to the existing surface.

Discriminating regressions locked here:
* the exact β·SPY decomposition arithmetic (hand-computed on a pinned
  TQQQ vs NVDA scenario — a sign flip / dropped β fails RED);
* β source is the sector→β map, not 1.0 by default (TQQQ in "broad_lev"
  → β=3, not 1.0);
* options skipped (β-attribution on a Greeks instrument is its own
  surface);
* SPY anchor is the equity-curve sp500_price at-or-after opened_at —
  same SSOT as ``open_attribution``;
* state ladder NO_DATA / NO_BENCHMARK / INSUFFICIENT / OK
  (no-anchor degrades honestly, never 0);
* totals math (book unrealized = beta_explained + idiosyncratic, exact);
* never raises on garbage rows (the ``_safe`` discipline);
* endpoint↔builder no-drift via the real Flask test_client.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.pnl_attribution import build_pnl_attribution


# ─────────────────────────── fixtures ───────────────────────────

def _classify(t: str) -> str:
    """Minimal local classifier mirroring dashboard._classify shape; tests
    don't depend on the real sector map."""
    return {
        "NVDA": "semis", "AMD": "semis", "MU": "semis",
        "TQQQ": "broad_lev", "SOXL": "semis_lev",
        "SPY": "broad", "QQQ": "broad",
    }.get((t or "").upper(), "other")


_BETA_MAP = {
    "semis": 1.5,
    "broad_lev": 3.0,
    "semis_lev": 3.0,
    "broad": 1.0,
    "other": 1.0,
}


def _curve(entries: list[tuple[str, float]]) -> list[dict]:
    """[(ts_iso, sp500_price), ...] → equity_curve rows ordered as the store
    returns them. ts is treated as the wall-clock anchor."""
    return [{"timestamp": ts, "sp500_price": px} for ts, px in entries]


# ─────────────────────────── state ladder ───────────────────────────

class TestStateLadder:
    def test_no_data_empty_positions(self):
        r = build_pnl_attribution([], _curve([("2026-05-19T00:00:00+00:00", 5000.0)]),
                                  _classify, _BETA_MAP)
        assert r["state"] == "NO_DATA"
        assert r["n_positions"] == 0

    def test_no_data_only_options(self):
        r = build_pnl_attribution(
            [{"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 1.0,
              "current_price": 2.0, "opened_at": "2026-05-19T00:00:00+00:00",
              "strike": 100, "expiry": "2026-06-19"}],
            _curve([("2026-05-19T00:00:00+00:00", 5000.0)]),
            _classify, _BETA_MAP,
        )
        # Options → skipped → no rows → NO_DATA
        assert r["state"] == "NO_DATA"
        assert any("option" in s["reason"] for s in r["skipped"])

    def test_no_benchmark_when_curve_has_no_spy(self):
        # Curve rows lack sp500_price → series empty → NO_BENCHMARK
        r = build_pnl_attribution(
            [{"ticker": "NVDA", "type": "stock", "qty": 1, "avg_cost": 100,
              "current_price": 110, "opened_at": "2026-05-19T00:00:00+00:00"}],
            [{"timestamp": "2026-05-19T00:00:00+00:00", "sp500_price": None}],
            _classify, _BETA_MAP,
        )
        assert r["state"] == "NO_BENCHMARK"

    def test_insufficient_when_opened_at_postdates_curve(self):
        # Position opened AFTER the last equity tick → no anchor (the SPY
        # series ends before opened_at; the "at-or-after" lookup fails).
        # The row is still emitted with anchored=False (honest withholding).
        r = build_pnl_attribution(
            [{"ticker": "NVDA", "type": "stock", "qty": 1, "avg_cost": 100,
              "current_price": 110, "opened_at": "2027-01-01T00:00:00+00:00"}],
            _curve([("2026-05-19T00:00:00+00:00", 5000.0)]),
            _classify, _BETA_MAP,
        )
        assert r["state"] == "INSUFFICIENT"
        assert r["rows"][0]["anchored"] is False
        assert r["rows"][0]["beta_explained_pct"] is None


# ─────────────────────────── exact arithmetic ───────────────────────────

class TestExactArithmetic:
    def test_nvda_beta_decomposition(self):
        # NVDA opened at $100, now $110 → position +10%.
        # SPY at-entry 5000, now 5050 → SPY +1%.
        # β=1.5 → beta_explained = 1.5 * 1% = 1.5%.
        # idiosyncratic = 10% - 1.5% = 8.5%.
        # Cost = 1 share * $100 = $100. Unrealized $ = $10.
        # beta_$ = $100 * 1.5% = $1.50; idio_$ = $10 - $1.50 = $8.50.
        r = build_pnl_attribution(
            [{"ticker": "NVDA", "type": "stock", "qty": 1, "avg_cost": 100.0,
              "current_price": 110.0, "opened_at": "2026-05-19T00:00:00+00:00"}],
            _curve([("2026-05-19T00:00:00+00:00", 5000.0),
                    ("2026-05-19T12:00:00+00:00", 5050.0)]),
            _classify, _BETA_MAP,
        )
        assert r["state"] == "OK"
        row = r["rows"][0]
        assert row["beta"] == 1.5
        assert row["position_return_pct"] == 10.0
        assert row["spy_return_pct"] == 1.0
        assert row["beta_explained_pct"] == 1.5
        assert row["idiosyncratic_pct"] == 8.5
        assert row["unrealized_usd"] == 10.0
        assert row["beta_explained_usd"] == 1.5
        assert row["idiosyncratic_usd"] == 8.5
        assert row["sector"] == "semis"
        assert row["anchored"] is True

    def test_tqqq_leveraged_disagrees_with_naive_alpha(self):
        # TQQQ β=3 in "broad_lev". TQQQ +6%, SPY +2%.
        # naive alpha = 6 - 2 = 4 (the open_attribution number).
        # β-adjusted: beta_explained = 3*2 = 6%; idiosyncratic = 6 - 6 = 0%.
        # THE DISCRIMINATOR: this builder must report 0 idio while the
        # naive surface would report +4 — that disagreement is the value.
        r = build_pnl_attribution(
            [{"ticker": "TQQQ", "type": "stock", "qty": 10, "avg_cost": 50.0,
              "current_price": 53.0, "opened_at": "2026-05-19T00:00:00+00:00"}],
            _curve([("2026-05-19T00:00:00+00:00", 5000.0),
                    ("2026-05-19T12:00:00+00:00", 5100.0)]),
            _classify, _BETA_MAP,
        )
        row = r["rows"][0]
        assert row["beta"] == 3.0
        assert row["position_return_pct"] == 6.0
        assert row["spy_return_pct"] == 2.0
        assert row["beta_explained_pct"] == 6.0
        assert row["idiosyncratic_pct"] == 0.0
        # Unrealized: $30, all of it explained by β·SPY, $0 idiosyncratic.
        assert row["unrealized_usd"] == 30.0
        assert row["beta_explained_usd"] == 30.0
        assert row["idiosyncratic_usd"] == 0.0

    def test_negative_idiosyncratic_on_losing_leveraged_etf(self):
        # TQQQ flat, SPY up 1% → β·SPY = +3% explains nothing the position
        # didn't capture → idiosyncratic = 0 - 3 = -3% (the desk's read:
        # "I should have gained 3% but didn't — selection cost 3%").
        r = build_pnl_attribution(
            [{"ticker": "TQQQ", "type": "stock", "qty": 1, "avg_cost": 50.0,
              "current_price": 50.0, "opened_at": "2026-05-19T00:00:00+00:00"}],
            _curve([("2026-05-19T00:00:00+00:00", 5000.0),
                    ("2026-05-19T12:00:00+00:00", 5050.0)]),
            _classify, _BETA_MAP,
        )
        row = r["rows"][0]
        assert row["position_return_pct"] == 0.0
        assert row["spy_return_pct"] == 1.0
        assert row["beta_explained_pct"] == 3.0
        assert row["idiosyncratic_pct"] == -3.0
        assert row["unrealized_usd"] == 0.0
        assert row["beta_explained_usd"] == 1.5
        assert row["idiosyncratic_usd"] == -1.5

    def test_totals_sum_correctly(self):
        # NVDA β=1.5 +10%, TQQQ β=3 +6%. Combine on a $5000→$5050 SPY tape.
        # NVDA: cost $100, beta_$ = $1.50, idio_$ = $8.50, unreal $10
        # TQQQ: cost $50, beta_$ = $0.45 (50 * 0.9%), idio_$ = TQQQ +6% - 1.5×SPY%
        # Use distinct entry SPY by giving them different opened_at to test
        # only the sum (mathematical equality should still hold).
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 1, "avg_cost": 100.0,
             "current_price": 110.0, "opened_at": "2026-05-19T00:00:00+00:00"},
            {"ticker": "TQQQ", "type": "stock", "qty": 1, "avg_cost": 50.0,
             "current_price": 53.0, "opened_at": "2026-05-19T00:00:00+00:00"},
        ]
        r = build_pnl_attribution(
            positions,
            _curve([("2026-05-19T00:00:00+00:00", 5000.0),
                    ("2026-05-19T12:00:00+00:00", 5050.0)]),
            _classify, _BETA_MAP,
        )
        tot = r["totals"]
        # Invariant: book unrealized $ = book beta_explained $ + book idio $.
        assert tot["unrealized_usd"] == pytest.approx(
            tot["beta_explained_usd"] + tot["idiosyncratic_usd"], abs=0.02
        )
        # Cost basis = $150.
        assert tot["cost_basis_usd"] == 150.0
        # Unrealized $13 (NVDA $10 + TQQQ $3).
        assert tot["unrealized_usd"] == 13.0
        # NVDA β·SPY = $1.50; TQQQ β·SPY = $50 * 3.0% = $1.50. Total $3.
        assert tot["beta_explained_usd"] == 3.0
        # Idiosyncratic = $13 - $3 = $10.
        assert tot["idiosyncratic_usd"] == 10.0


# ─────────────────────────── degrade-honestly paths ───────────────────────────

class TestDegradePaths:
    def test_unknown_ticker_defaults_to_beta_1(self):
        # Unmapped ticker → classify→"other" → β=1.0 (market-beta default).
        r = build_pnl_attribution(
            [{"ticker": "UNMAPPED", "type": "stock", "qty": 1, "avg_cost": 100.0,
              "current_price": 105.0, "opened_at": "2026-05-19T00:00:00+00:00"}],
            _curve([("2026-05-19T00:00:00+00:00", 5000.0),
                    ("2026-05-19T12:00:00+00:00", 5100.0)]),
            _classify, _BETA_MAP,
        )
        row = r["rows"][0]
        assert row["beta"] == 1.0
        assert row["sector"] == "other"
        assert row["beta_explained_pct"] == 2.0  # 1 * 2%
        assert row["idiosyncratic_pct"] == 3.0    # 5 - 2

    def test_option_position_skipped(self):
        r = build_pnl_attribution(
            [{"ticker": "NVDA", "type": "call", "qty": 1, "avg_cost": 5.0,
              "current_price": 10.0, "opened_at": "2026-05-19T00:00:00+00:00",
              "strike": 100, "expiry": "2026-06-19"},
             {"ticker": "NVDA", "type": "stock", "qty": 1, "avg_cost": 100.0,
              "current_price": 110.0, "opened_at": "2026-05-19T00:00:00+00:00"}],
            _curve([("2026-05-19T00:00:00+00:00", 5000.0),
                    ("2026-05-19T12:00:00+00:00", 5050.0)]),
            _classify, _BETA_MAP,
        )
        assert r["n_positions"] == 1  # only the stock
        assert any(s["type"] == "call" for s in r["skipped"])

    def test_classify_raises_falls_back_to_other(self):
        # The ``_safe`` contract — a broken classifier never raises.
        def boom(t):
            raise RuntimeError("classifier hiccup")

        r = build_pnl_attribution(
            [{"ticker": "NVDA", "type": "stock", "qty": 1, "avg_cost": 100.0,
              "current_price": 110.0, "opened_at": "2026-05-19T00:00:00+00:00"}],
            _curve([("2026-05-19T00:00:00+00:00", 5000.0),
                    ("2026-05-19T12:00:00+00:00", 5050.0)]),
            boom, _BETA_MAP,
        )
        # The discriminator: the row is still emitted with β=1.0 fallback.
        assert r["state"] == "OK"
        assert r["rows"][0]["beta"] == 1.0
        assert r["rows"][0]["sector"] == "other"

    def test_garbage_position_never_raises(self):
        r = build_pnl_attribution(
            [None, "not-a-dict",
             {"ticker": "NVDA", "qty": "garbage", "type": "stock"},
             {"ticker": "NVDA", "qty": 0, "avg_cost": 0, "type": "stock"},
             {"ticker": "NVDA", "qty": 1, "avg_cost": 100.0, "current_price": None,
              "type": "stock"},
             {"ticker": "NVDA", "type": "stock", "qty": 1, "avg_cost": 100.0,
              "current_price": 110.0, "opened_at": "2026-05-19T00:00:00+00:00"}],
            _curve([("2026-05-19T00:00:00+00:00", 5000.0),
                    ("2026-05-19T12:00:00+00:00", 5050.0)]),
            _classify, _BETA_MAP,
        )
        # Exactly one good row makes it through.
        assert r["n_positions"] == 1
        assert r["state"] == "OK"

    def test_corrupt_equity_curve_row_skipped(self):
        # A row with garbage sp500_price should be skipped, not raise.
        r = build_pnl_attribution(
            [{"ticker": "NVDA", "type": "stock", "qty": 1, "avg_cost": 100.0,
              "current_price": 110.0, "opened_at": "2026-05-19T00:00:00+00:00"}],
            [{"timestamp": "garbage", "sp500_price": "bad"},
             {"timestamp": "2026-05-19T00:00:00+00:00", "sp500_price": 5000.0},
             {"timestamp": "2026-05-19T12:00:00+00:00", "sp500_price": -1.0},  # rejected
             {"timestamp": "2026-05-19T13:00:00+00:00", "sp500_price": 5050.0}],
            _classify, _BETA_MAP,
        )
        # Anchor lands on 5000, latest is 5050 → +1%. OK with full numerics.
        assert r["state"] == "OK"
        assert r["rows"][0]["spy_return_pct"] == 1.0

    def test_unmapped_beta_falls_back_to_one(self):
        # classify returns a sector not in beta_map → β=1.0
        def classify_only_other(t):
            return "totally_unknown_sector"

        r = build_pnl_attribution(
            [{"ticker": "NVDA", "type": "stock", "qty": 1, "avg_cost": 100.0,
              "current_price": 110.0, "opened_at": "2026-05-19T00:00:00+00:00"}],
            _curve([("2026-05-19T00:00:00+00:00", 5000.0),
                    ("2026-05-19T12:00:00+00:00", 5050.0)]),
            classify_only_other, _BETA_MAP,
        )
        assert r["rows"][0]["beta"] == 1.0


# ─────────────────────────── sort + biggest-drag-first ───────────────────────────

class TestSortOrder:
    def test_biggest_abs_idiosyncratic_first(self):
        # NVDA: cost $100, +11.5% → unreal $11.5; β=1.5×SPY+1% = $1.50 β-explained;
        #       idio $10.
        # TQQQ: cost $500, -2% → unreal -$10; β=3×1% = $15 β-explained;
        #       idio = -$10 - $15 = -$25 (the leveraged-ETF read: you LOST
        #       $10 but the market would have given you +$15, so selection
        #       cost you $25).
        # |TQQQ idio|=$25 > |NVDA idio|=$10 → TQQQ sorts first.
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 1, "avg_cost": 100.0,
             "current_price": 111.5, "opened_at": "2026-05-19T00:00:00+00:00"},
            {"ticker": "TQQQ", "type": "stock", "qty": 10, "avg_cost": 50.0,
             "current_price": 49.0, "opened_at": "2026-05-19T00:00:00+00:00"},
        ]
        r = build_pnl_attribution(
            positions,
            _curve([("2026-05-19T00:00:00+00:00", 5000.0),
                    ("2026-05-19T12:00:00+00:00", 5050.0)]),
            _classify, _BETA_MAP,
        )
        # Sort order asserted by ticker.
        assert r["rows"][0]["ticker"] == "TQQQ"
        assert r["rows"][1]["ticker"] == "NVDA"
        assert r["rows"][0]["idiosyncratic_usd"] == -25.0
        assert r["rows"][1]["idiosyncratic_usd"] == 10.0

    def test_unanchored_sorts_last(self):
        # NVDA opened AFTER the curve ends → unanchorable (the "at-or-after"
        # lookup returns None). Even with a large would-be return, the row
        # sorts after anchored rows because its idiosyncratic_usd is None.
        positions = [
            {"ticker": "NVDA", "type": "stock", "qty": 1, "avg_cost": 100.0,
             "current_price": 200.0, "opened_at": "2027-01-01T00:00:00+00:00"},
            {"ticker": "AMD", "type": "stock", "qty": 1, "avg_cost": 100.0,
             "current_price": 105.0, "opened_at": "2026-05-19T00:00:00+00:00"},
        ]
        r = build_pnl_attribution(
            positions,
            _curve([("2026-05-19T00:00:00+00:00", 5000.0),
                    ("2026-05-19T12:00:00+00:00", 5050.0)]),
            _classify, _BETA_MAP,
        )
        # AMD anchored → first; NVDA unanchored → last.
        assert r["rows"][0]["ticker"] == "AMD"
        assert r["rows"][1]["ticker"] == "NVDA"
        assert r["rows"][1]["anchored"] is False


# ─────────────────────────── /api/pnl-attribution parity ───────────────────────────

class TestPnlAttributionEndpoint:
    """Endpoint↔builder no-drift via the real Flask test_client. The
    discriminator: ``/api/pnl-attribution`` is THE true SSOT — the
    strategy-side pinned copies are CI-pinned to its ``_classify`` /
    ``_LEVERAGE_BETA``. A regression that swaps the route's classifier
    for an inline literal would silently desync.
    """

    def test_endpoint_serves_builder_output(self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store

        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        # Seed an open NVDA position at $100, current $110 (set via positions
        # update so current_price comes through the snapshot).
        s.upsert_position("NVDA", "stock", 1.0, 100.0)
        # Mark the position to $110 — upsert_position resets current_price=0,
        # so we need update_position_marks to seed it (the live runner does
        # this via strategy._portfolio_snapshot every cycle).
        pos = s.open_positions()
        s.update_position_marks({pos[0]["id"]: (110.0, 10.0)})
        # Record an equity point so the curve has SPY history at-entry.
        s.record_equity_point(total_value=110.0, cash=0.0, sp500=5000.0)
        # Also seed a "now" tick so SPY has moved.
        s.record_equity_point(total_value=110.0, cash=0.0, sp500=5050.0)

        from paper_trader import dashboard
        dashboard.app.config["TESTING"] = True

        try:
            with dashboard.app.test_client() as client:
                resp = client.get("/api/pnl-attribution")
        finally:
            s.close()

        assert resp.status_code == 200, resp.data
        data = resp.get_json()
        assert "error" not in data, data
        assert data["state"] == "OK"
        nvda = [r for r in data["rows"] if r["ticker"] == "NVDA"][0]
        assert nvda["sector"] == "semis"
        # The dashboard's SSOT β for "semis" is 1.5.
        assert nvda["beta"] == 1.5
        # Position +10%, SPY +1%, β=1.5 → beta_explained=1.5, idio=8.5.
        assert nvda["position_return_pct"] == 10.0
        assert nvda["spy_return_pct"] == 1.0
        assert nvda["beta_explained_pct"] == 1.5
        assert nvda["idiosyncratic_pct"] == 8.5

    def test_endpoint_empty_book_no_data_not_500(
            self, tmp_path, monkeypatch):
        # Empty book → NO_DATA, not a 500 — the chat /api/analytics
        # cross-fetch must keep working on a fresh book.
        from paper_trader import store as store_mod
        from paper_trader.store import Store

        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()

        from paper_trader import dashboard
        dashboard.app.config["TESTING"] = True
        try:
            with dashboard.app.test_client() as client:
                resp = client.get("/api/pnl-attribution")
        finally:
            s.close()
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["state"] == "NO_DATA"


class TestAnalyticsFold:
    """The /api/analytics fold inherits pnl_attribution for the
    digital-intern analyst chat. SSOT no-drift: the fold and the dedicated
    endpoint must serve byte-equal numerics for the same inputs."""

    def test_analytics_carries_pnl_attribution_key(self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store

        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        s.upsert_position("NVDA", "stock", 1.0, 100.0)
        pos = s.open_positions()
        s.update_position_marks({pos[0]["id"]: (110.0, 10.0)})
        s.record_equity_point(total_value=110.0, cash=0.0, sp500=5000.0)
        s.record_equity_point(total_value=110.0, cash=0.0, sp500=5050.0)

        from paper_trader import dashboard
        dashboard.app.config["TESTING"] = True
        try:
            with dashboard.app.test_client() as client:
                ana = client.get("/api/analytics").get_json()
                pa = client.get("/api/pnl-attribution").get_json()
        finally:
            s.close()
        assert "pnl_attribution" in ana, "fold missing"
        # No-drift on the load-bearing fields.
        assert ana["pnl_attribution"]["state"] == pa["state"]
        nvda_a = [r for r in ana["pnl_attribution"]["rows"]
                  if r["ticker"] == "NVDA"][0]
        nvda_p = [r for r in pa["rows"] if r["ticker"] == "NVDA"][0]
        for k in ("beta", "beta_explained_pct", "idiosyncratic_pct",
                  "beta_explained_usd", "idiosyncratic_usd"):
            assert nvda_a[k] == nvda_p[k], (k, nvda_a[k], nvda_p[k])
