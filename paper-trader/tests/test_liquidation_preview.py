"""Tests for analytics/liquidation_preview.py + /api/liquidation-preview.

Contract:
* Pure ``build_liquidation_preview`` is offline / never raises on garbage.
* Empty book → NO_POSITIONS with current_cash == liquidation_cash.
* Single stock position → liquidation_cash adds market_value;
  realized_pl_if_closed equals position.unrealized_pl.
* Stock + option mix → option ×100 multiplier honored.
* Stale-mark positions are flagged and counted; lock-in noted as unreliable
  in the headline.
* Per-position rows are sorted by |realized_pl| DESC (biggest contributor
  first — partial-liquidation reading order).
* Garbage cells (None / 'x' / NaN) don't raise; they degrade to 0.0.
* SSOT with _mark_to_market: when ``unrealized_pl`` is present on the
  position, it is used verbatim; only absent → derive from cur−avg.
* Endpoint returns 200 with shape ``{state, current_cash, liquidation_cash,
  realized_pl_if_closed, positions: [...]}`` over a real read-only
  snapshot; returns 500 on builder fault.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from paper_trader.analytics.liquidation_preview import (
    build_liquidation_preview,
)


# ── Pure builder ───────────────────────────────────────────────────────────


class TestEmpty:
    def test_empty_snapshot_returns_no_positions(self):
        r = build_liquidation_preview({})
        assert r["state"] == "NO_POSITIONS"
        assert r["n_positions"] == 0
        assert r["current_cash"] == 0.0
        assert r["liquidation_cash"] == 0.0
        assert r["cash_freed"] == 0.0
        assert r["realized_pl_if_closed"] == 0.0
        assert r["realized_pl_pct"] is None
        assert r["positions"] == []
        assert "nothing to liquidate" in r["headline"].lower()

    def test_no_positions_with_cash_preserves_cash(self):
        r = build_liquidation_preview(
            {"cash": 1000.0, "total_value": 1000.0, "positions": []},
        )
        assert r["state"] == "NO_POSITIONS"
        assert r["current_cash"] == 1000.0
        assert r["liquidation_cash"] == 1000.0
        assert r["cash_freed"] == 0.0

    def test_non_dict_input_does_not_raise(self):
        r = build_liquidation_preview(None)  # type: ignore[arg-type]
        assert r["state"] == "NO_POSITIONS"
        r2 = build_liquidation_preview("garbage")  # type: ignore[arg-type]
        assert r2["state"] == "NO_POSITIONS"


# ── Single position ────────────────────────────────────────────────────────


class TestSingleStock:
    def test_single_winning_stock(self):
        # 10 shares bought at $100, now $120 → market_value $1200,
        # realized +$200 on close, cash from $500 → $1700.
        snap = {
            "cash": 500.0,
            "total_value": 1700.0,
            "positions": [{
                "ticker": "NVDA",
                "type": "stock",
                "qty": 10.0,
                "avg_cost": 100.0,
                "current_price": 120.0,
                "market_value": 1200.0,
                "unrealized_pl": 200.0,
            }],
        }
        r = build_liquidation_preview(snap)
        assert r["state"] == "OK"
        assert r["current_cash"] == 500.0
        assert r["liquidation_cash"] == 1700.0
        assert r["cash_freed"] == 1200.0
        assert r["realized_pl_if_closed"] == 200.0
        assert r["realized_pl_pct"] == round(200.0 / 1700.0 * 100, 2)
        assert r["n_positions"] == 1
        assert r["n_stale_marks"] == 0
        p = r["positions"][0]
        assert p["ticker"] == "NVDA"
        assert p["label"] == "NVDA"
        assert p["market_value"] == 1200.0
        assert p["realized_pl"] == 200.0
        assert p["pl_pct"] == 20.0
        assert p["stale_mark"] is False

    def test_single_losing_stock_signs_correctly(self):
        snap = {
            "cash": 500.0,
            "total_value": 1300.0,
            "positions": [{
                "ticker": "MU",
                "type": "stock",
                "qty": 10.0,
                "avg_cost": 100.0,
                "current_price": 80.0,
                "market_value": 800.0,
                "unrealized_pl": -200.0,
            }],
        }
        r = build_liquidation_preview(snap)
        assert r["realized_pl_if_closed"] == -200.0
        assert r["positions"][0]["pl_pct"] == -20.0
        # No '+' in front of a negative number.
        assert "-$200.00" in r["headline"]


# ── Option positions ───────────────────────────────────────────────────────


class TestOption:
    def test_option_multiplier_honored_via_market_value(self):
        # Bought 2 calls at $5, now $7. market_value field already bakes the
        # ×100 multiplier (this is the post-mark-to-market shape).
        # realized = (7-5)*2*100 = $400.
        snap = {
            "cash": 100.0,
            "total_value": 1500.0,
            "positions": [{
                "ticker": "NVDA",
                "type": "call",
                "qty": 2.0,
                "strike": 600.0,
                "expiry": "2026-06-19",
                "avg_cost": 5.0,
                "current_price": 7.0,
                "market_value": 1400.0,  # 7*2*100
                "unrealized_pl": 400.0,
            }],
        }
        r = build_liquidation_preview(snap)
        assert r["liquidation_cash"] == 1500.0
        assert r["realized_pl_if_closed"] == 400.0
        assert r["positions"][0]["label"] == "NVDA 600C 2026-06-19"
        assert r["positions"][0]["market_value"] == 1400.0

    def test_option_multiplier_derived_when_market_value_absent(self):
        # Stripped row (no market_value / unrealized_pl) — exercises the
        # fallback path that derives both via the ×100 multiplier.
        snap = {
            "cash": 100.0,
            "total_value": 1500.0,
            "positions": [{
                "ticker": "NVDA",
                "type": "put",
                "qty": 1.0,
                "strike": 500.0,
                "expiry": "2026-06-19",
                "avg_cost": 3.0,
                "current_price": 5.0,
            }],
        }
        r = build_liquidation_preview(snap)
        # market_value = 5 * 1 * 100 = $500; realized = (5-3)*1*100 = $200.
        assert r["positions"][0]["market_value"] == 500.0
        assert r["positions"][0]["realized_pl"] == 200.0
        assert r["positions"][0]["label"] == "NVDA 500P 2026-06-19"

    def test_option_label_renders_whole_strike_without_decimal(self):
        snap = {
            "cash": 0.0,
            "total_value": 100.0,
            "positions": [{
                "ticker": "AAPL",
                "type": "call",
                "qty": 1.0,
                "strike": 250.0,
                "expiry": "2026-06-19",
                "avg_cost": 1.0,
                "current_price": 1.0,
                "market_value": 100.0,
                "unrealized_pl": 0.0,
            }],
        }
        r = build_liquidation_preview(snap)
        # No ".0" trailing the strike.
        assert r["positions"][0]["label"] == "AAPL 250C 2026-06-19"

    def test_option_label_keeps_fractional_strike(self):
        snap = {
            "cash": 0.0,
            "total_value": 100.0,
            "positions": [{
                "ticker": "SPY",
                "type": "put",
                "qty": 1.0,
                "strike": 552.5,
                "expiry": "2026-06-19",
                "avg_cost": 1.0,
                "current_price": 1.0,
                "market_value": 100.0,
                "unrealized_pl": 0.0,
            }],
        }
        r = build_liquidation_preview(snap)
        assert r["positions"][0]["label"] == "SPY 552.5P 2026-06-19"


# ── Mixed book + sorting ───────────────────────────────────────────────────


class TestMixed:
    def test_sort_by_absolute_realized_desc(self):
        snap = {
            "cash": 0.0,
            "total_value": 5000.0,
            "positions": [
                {"ticker": "A", "type": "stock", "qty": 1, "avg_cost": 100,
                 "current_price": 110, "market_value": 110,
                 "unrealized_pl": 10.0},
                {"ticker": "B", "type": "stock", "qty": 1, "avg_cost": 100,
                 "current_price": 50, "market_value": 50,
                 "unrealized_pl": -50.0},
                {"ticker": "C", "type": "stock", "qty": 1, "avg_cost": 100,
                 "current_price": 130, "market_value": 130,
                 "unrealized_pl": 30.0},
            ],
        }
        r = build_liquidation_preview(snap)
        # Absolute realized: 50 (B), 30 (C), 10 (A). Biggest first.
        assert [p["ticker"] for p in r["positions"]] == ["B", "C", "A"]


# ── Stale marks ────────────────────────────────────────────────────────────


class TestStale:
    def test_stale_mark_counted_and_flagged(self):
        snap = {
            "cash": 0.0,
            "total_value": 200.0,
            "positions": [{
                "ticker": "MU",
                "type": "stock",
                "qty": 1.0,
                "avg_cost": 200.0,
                "current_price": 200.0,
                "market_value": 200.0,
                "unrealized_pl": 0.0,
                "stale_mark": True,
            }],
        }
        r = build_liquidation_preview(snap)
        assert r["n_stale_marks"] == 1
        assert r["positions"][0]["stale_mark"] is True
        assert "stale" in r["headline"].lower()
        # The mark-at-cost row still contributes market_value to liquidation
        # (the position genuinely IS owned, just priced at the floor).
        assert r["liquidation_cash"] == 200.0


# ── SSOT with _mark_to_market ──────────────────────────────────────────────


class TestSSOT:
    def test_uses_unrealized_pl_when_present(self):
        # If a row has unrealized_pl already (post-mark-to-market shape),
        # the builder MUST trust it verbatim — not re-derive. This is the
        # invariant #10 single-source-of-truth contract.
        snap = {
            "cash": 0.0,
            "total_value": 100.0,
            "positions": [{
                "ticker": "X",
                "type": "stock",
                "qty": 10.0,
                "avg_cost": 100.0,
                "current_price": 100.0,
                # mtm says +$777 even though prices imply 0. Trust mtm.
                "market_value": 1000.0,
                "unrealized_pl": 777.0,
            }],
        }
        r = build_liquidation_preview(snap)
        assert r["realized_pl_if_closed"] == 777.0
        assert r["positions"][0]["realized_pl"] == 777.0

    def test_falls_back_to_derive_when_unrealized_pl_absent(self):
        snap = {
            "cash": 0.0,
            "total_value": 100.0,
            "positions": [{
                "ticker": "X",
                "type": "stock",
                "qty": 10.0,
                "avg_cost": 100.0,
                "current_price": 120.0,
                "market_value": 1200.0,
                # unrealized_pl deliberately absent.
            }],
        }
        r = build_liquidation_preview(snap)
        assert r["realized_pl_if_closed"] == 200.0
        assert r["positions"][0]["realized_pl"] == 200.0


# ── Garbage / degrade-safe ─────────────────────────────────────────────────


class TestDegrade:
    def test_none_qty_does_not_raise(self):
        snap = {
            "cash": 100.0,
            "total_value": 100.0,
            "positions": [
                {"ticker": "X", "type": "stock", "qty": None,
                 "avg_cost": None, "current_price": None},
            ],
        }
        r = build_liquidation_preview(snap)
        # Garbage row contributes nothing; we still report it as a position
        # row (count=1) but realized_pl=0 and market_value=0.
        assert r["state"] == "OK"
        assert r["n_positions"] == 1
        assert r["realized_pl_if_closed"] == 0.0
        assert r["positions"][0]["market_value"] == 0.0

    def test_zero_total_value_yields_none_pct(self):
        snap = {
            "cash": 0.0,
            "total_value": 0.0,
            "positions": [
                {"ticker": "X", "type": "stock", "qty": 1,
                 "avg_cost": 100, "current_price": 100,
                 "market_value": 100, "unrealized_pl": 0.0},
            ],
        }
        r = build_liquidation_preview(snap)
        assert r["realized_pl_pct"] is None

    def test_non_dict_position_skipped(self):
        snap = {
            "cash": 50.0,
            "total_value": 150.0,
            "positions": [
                "garbage_string",
                None,
                {"ticker": "REAL", "type": "stock", "qty": 1.0,
                 "avg_cost": 50.0, "current_price": 100.0,
                 "market_value": 100.0, "unrealized_pl": 50.0},
            ],
        }
        r = build_liquidation_preview(snap)
        assert r["n_positions"] == 1
        assert r["positions"][0]["ticker"] == "REAL"

    def test_avg_cost_zero_gives_none_pl_pct(self):
        # A division-by-zero on cost would otherwise raise.
        snap = {
            "cash": 0.0,
            "total_value": 100.0,
            "positions": [
                {"ticker": "FREE", "type": "stock", "qty": 1.0,
                 "avg_cost": 0.0, "current_price": 100.0,
                 "market_value": 100.0, "unrealized_pl": 100.0},
            ],
        }
        r = build_liquidation_preview(snap)
        assert r["positions"][0]["pl_pct"] is None


# ── Headline ───────────────────────────────────────────────────────────────


class TestHeadline:
    def test_headline_singularizes_one_position(self):
        snap = {
            "cash": 100.0,
            "total_value": 200.0,
            "positions": [{"ticker": "X", "type": "stock", "qty": 1,
                           "avg_cost": 100, "current_price": 100,
                           "market_value": 100, "unrealized_pl": 0.0}],
        }
        r = build_liquidation_preview(snap)
        assert "1 position" in r["headline"]
        # No "1 positions" — explicit singular.
        assert "1 positions" not in r["headline"]

    def test_headline_pluralizes_multiple(self):
        snap = {
            "cash": 0.0,
            "total_value": 200.0,
            "positions": [
                {"ticker": "A", "type": "stock", "qty": 1, "avg_cost": 100,
                 "current_price": 100, "market_value": 100,
                 "unrealized_pl": 0.0},
                {"ticker": "B", "type": "stock", "qty": 1, "avg_cost": 100,
                 "current_price": 100, "market_value": 100,
                 "unrealized_pl": 0.0},
            ],
        }
        r = build_liquidation_preview(snap)
        assert "2 positions" in r["headline"]


# ── Endpoint integration ───────────────────────────────────────────────────


class TestEndpoint:
    def _bootstrap(self, tmp_path, monkeypatch):
        from paper_trader import dashboard as dash
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        monkeypatch.setattr(store_mod, "DB_PATH",
                            tmp_path / "paper_trader.db")
        monkeypatch.setattr(store_mod, "_singleton", None)
        store = Store()
        monkeypatch.setattr(dash, "get_store", lambda: store)
        dash.app.config["TESTING"] = True
        return dash, store

    def test_endpoint_empty_book(self, tmp_path, monkeypatch):
        dash, store = self._bootstrap(tmp_path, monkeypatch)
        client = dash.app.test_client()
        resp = client.get("/api/liquidation-preview")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["state"] == "NO_POSITIONS"
        assert body["n_positions"] == 0
        # Fresh store: starting cash $1000.
        assert body["current_cash"] == 1000.0
        assert body["liquidation_cash"] == 1000.0

    def test_endpoint_with_real_position(self, tmp_path, monkeypatch):
        dash, store = self._bootstrap(tmp_path, monkeypatch)
        # Buy 1 NVDA @ $100 (cash 1000 → 900, position value 100).
        store.record_trade("NVDA", "BUY", 1.0, 100.0, "seed")
        store.upsert_position("NVDA", "stock", 1.0, 100.0)
        store.update_portfolio(900.0, 1000.0, [])
        # Stub market.get_prices used by portfolio_snapshot_readonly so the
        # mark-to-market is deterministic.
        from paper_trader import market as _market
        monkeypatch.setattr(_market, "get_prices",
                            lambda tickers: {"NVDA": 150.0})
        client = dash.app.test_client()
        resp = client.get("/api/liquidation-preview")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["state"] == "OK"
        assert body["n_positions"] == 1
        # Cash 900 + market 150 = 1050; realized = (150-100)*1 = 50.
        assert body["current_cash"] == 900.0
        assert body["liquidation_cash"] == 1050.0
        assert body["realized_pl_if_closed"] == 50.0
        p = body["positions"][0]
        assert p["ticker"] == "NVDA"
        assert p["market_value"] == 150.0
        assert p["realized_pl"] == 50.0
        assert p["pl_pct"] == 50.0

    def test_endpoint_returns_500_on_builder_fault(self, tmp_path,
                                                    monkeypatch):
        dash, store = self._bootstrap(tmp_path, monkeypatch)

        def boom(snap):
            raise RuntimeError("synthetic fault")

        # Patch the builder symbol the endpoint imports lazily.
        import paper_trader.analytics.liquidation_preview as lp
        monkeypatch.setattr(lp, "build_liquidation_preview", boom)
        client = dash.app.test_client()
        resp = client.get("/api/liquidation-preview")
        assert resp.status_code == 500
        body = resp.get_json()
        assert "synthetic fault" in body["error"]
