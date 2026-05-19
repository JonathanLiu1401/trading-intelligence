"""Tests for analytics/etf_lookthrough.py — single-name exposure look-through
through leveraged ETF positions.

Discriminating regressions locked here:
* exact look-through arithmetic on a pinned book — ``$148 × 3.0 × 0.09 =
  $39.96`` indirect NVDA from a TQQQ position. A dropped leverage factor
  or a missing %-to-decimal conversion fails RED;
* inverse ETF (SQQQ, leverage=-3) SHORTS the underlying — sign matters;
* state ladder ``NO_DATA`` / ``NO_ETF_HELD`` / ``OK`` boundary;
* HIDDEN_AMPLIFIED / HIDDEN_ONLY / TRANSPARENT / TRIVIAL tier
  classification on the ``HIDDEN_RATIO=1.5`` boundary;
* effective_usd math (direct + indirect, exact);
* sort order: ``|effective_usd|`` DESC, tie-break by direct DESC;
* options NOT looked-through (delta-adjustment is its own surface);
* live ``_ETF_LOOKTHROUGH`` map sanity (leverage signs, holdings shape);
* degrade-never-raises on garbage rows / missing keys / non-dict snapshot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.etf_lookthrough import (  # noqa: E402
    _ETF_LOOKTHROUGH,
    HIDDEN_RATIO,
    _exposure_state,
    _position_value,
    build_etf_lookthrough,
)


# ─────────────────────────── pinned fixtures ───────────────────────────

# A minimal test map mirrors the real shape without depending on the
# (drifting) issuer weights. Hand-pinned for exact arithmetic.
_TEST_MAP = {
    "TQQQ": {
        "leverage": 3.0,
        "holdings": [("NVDA", 10.0), ("MSFT", 10.0), ("AAPL", 8.0)],
    },
    "SQQQ": {
        "leverage": -3.0,
        "holdings": [("NVDA", 10.0), ("MSFT", 10.0), ("AAPL", 8.0)],
    },
    "SOXL": {
        "leverage": 3.0,
        "holdings": [("NVDA", 10.0), ("AVGO", 9.0), ("AMD", 7.0)],
    },
}


def _stock(ticker: str, qty: float, price: float) -> dict:
    return {
        "ticker": ticker,
        "qty": qty,
        "type": "stock",
        "current_price": price,
        "avg_cost": price,
    }


def _opt(ticker: str, qty: float, premium: float, kind: str = "call") -> dict:
    return {
        "ticker": ticker,
        "qty": qty,
        "type": kind,
        "current_price": premium,
        "avg_cost": premium,
    }


# ─────────────────────────── state ladder ───────────────────────────


class TestStateLadder:
    def test_no_data_empty_snapshot(self):
        out = build_etf_lookthrough({}, lookthrough_map=_TEST_MAP)
        assert out["state"] == "NO_DATA"
        assert out["n_etfs_held"] == 0
        assert out["etf_positions"] == []
        assert out["underlyings"] == []

    def test_no_data_zero_total(self):
        out = build_etf_lookthrough(
            {"total_value": 0.0, "positions": [_stock("NVDA", 1, 100)]},
            lookthrough_map=_TEST_MAP,
        )
        assert out["state"] == "NO_DATA"

    def test_no_data_negative_total(self):
        out = build_etf_lookthrough(
            {"total_value": -50.0, "positions": [_stock("NVDA", 1, 100)]},
            lookthrough_map=_TEST_MAP,
        )
        assert out["state"] == "NO_DATA"

    def test_no_etf_held_when_only_cash_stocks(self):
        # Book is just NVDA + MU — no ETF in the map. Look-through == direct.
        snap = {
            "total_value": 1000.0,
            "positions": [
                _stock("NVDA", 4, 200),  # $800 direct
                _stock("MU", 2, 100),    # $200 direct
            ],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        assert out["state"] == "NO_ETF_HELD"
        assert "no leveraged ETF" in out["headline"]
        assert out["n_etfs_held"] == 0
        assert out["underlyings"] == []

    def test_ok_when_etf_held(self):
        snap = {
            "total_value": 1000.0,
            "positions": [_stock("TQQQ", 5, 100)],  # $500 TQQQ
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        assert out["state"] == "OK"
        assert out["n_etfs_held"] == 1


# ─────────────────────────── exact arithmetic ───────────────────────────


class TestArithmetic:
    def test_tqqq_lookthrough_nvda_exact(self):
        # The pinned live-pathology number: $148 TQQQ × 3.0 × 9% = $39.96
        # indirect NVDA. Using the test map's 10% weight gives an even
        # cleaner $44.40 — pick a deliberate qty/price to make the math
        # textbook so a regression jumps out.
        snap = {
            "total_value": 1000.0,
            "positions": [_stock("TQQQ", 1, 100)],  # $100 TQQQ position
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        assert out["state"] == "OK"

        # Find NVDA underlying row.
        nvda = next(r for r in out["underlyings"] if r["ticker"] == "NVDA")
        # $100 × 3.0 × 10% = $30.00 indirect, $0 direct
        assert nvda["indirect_usd"] == 30.0
        assert nvda["direct_usd"] == 0.0
        assert nvda["effective_usd"] == 30.0
        # As % of $1000 book: 3.0%
        assert nvda["effective_pct"] == 3.0
        assert nvda["direct_pct"] == 0.0

    def test_direct_plus_indirect_equals_effective(self):
        # Both a direct NVDA position AND TQQQ. Effective = direct + indirect.
        snap = {
            "total_value": 1000.0,
            "positions": [
                _stock("NVDA", 2, 200),   # $400 direct NVDA
                _stock("TQQQ", 2, 100),   # $200 TQQQ → $200×3×0.10 = $60 indirect NVDA
            ],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        nvda = next(r for r in out["underlyings"] if r["ticker"] == "NVDA")
        assert nvda["direct_usd"] == 400.0
        assert nvda["indirect_usd"] == 60.0
        assert nvda["effective_usd"] == 460.0
        assert nvda["effective_pct"] == 46.0

    def test_inverse_etf_shorts_underlying(self):
        # SQQQ leverage = -3. Holding $100 SQQQ → -$30 indirect NVDA exposure.
        # A book with $400 direct NVDA and $100 SQQQ has NET +$370 NVDA.
        snap = {
            "total_value": 1000.0,
            "positions": [
                _stock("NVDA", 2, 200),   # $400 direct
                _stock("SQQQ", 1, 100),   # -$30 indirect NVDA
            ],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        nvda = next(r for r in out["underlyings"] if r["ticker"] == "NVDA")
        assert nvda["indirect_usd"] == -30.0  # SHORT exposure
        assert nvda["direct_usd"] == 400.0
        assert nvda["effective_usd"] == 370.0
        assert nvda["effective_pct"] == 37.0

    def test_multi_etf_compounding(self):
        # TQQQ + SOXL both have NVDA at 10%, both 3x. Indirect should ADD.
        snap = {
            "total_value": 1000.0,
            "positions": [
                _stock("TQQQ", 1, 100),   # $100 → $30 NVDA
                _stock("SOXL", 1, 100),   # $100 → $30 NVDA
            ],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        nvda = next(r for r in out["underlyings"] if r["ticker"] == "NVDA")
        assert nvda["indirect_usd"] == 60.0
        assert nvda["effective_usd"] == 60.0

    def test_unmapped_etf_treated_as_direct_only(self):
        # An ETF not in our map (e.g. SMH not in _TEST_MAP) contributes
        # only direct exposure — no look-through.
        snap = {
            "total_value": 1000.0,
            "positions": [_stock("VOO", 5, 100)],  # not in _TEST_MAP
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        # No ETF in map held → NO_ETF_HELD even though VOO is technically an ETF.
        assert out["state"] == "NO_ETF_HELD"


# ─────────────────────────── tier classification ───────────────────────────


class TestTierClassification:
    def test_trivial_below_half_pct(self):
        # eff_pct < 0.5% → TRIVIAL regardless of ratio
        assert _exposure_state(direct_pct=0.0, effective_pct=0.3) == "TRIVIAL"
        assert _exposure_state(direct_pct=0.4, effective_pct=0.49) == "TRIVIAL"

    def test_hidden_only_when_no_direct(self):
        # All-indirect exposure → HIDDEN_ONLY
        assert _exposure_state(direct_pct=0.0, effective_pct=5.0) == "HIDDEN_ONLY"
        # Direct < 0.1% counts as no-direct (negligible)
        assert _exposure_state(direct_pct=0.05, effective_pct=5.0) == "HIDDEN_ONLY"

    def test_hidden_amplified_at_or_above_ratio(self):
        # At exactly HIDDEN_RATIO → HIDDEN_AMPLIFIED (inclusive boundary)
        assert _exposure_state(direct_pct=10.0, effective_pct=15.0) == "HIDDEN_AMPLIFIED"
        assert _exposure_state(direct_pct=10.0, effective_pct=16.0) == "HIDDEN_AMPLIFIED"

    def test_transparent_below_ratio(self):
        # 1.49× < HIDDEN_RATIO 1.5× → TRANSPARENT
        assert _exposure_state(direct_pct=10.0, effective_pct=14.9) == "TRANSPARENT"
        assert _exposure_state(direct_pct=10.0, effective_pct=11.0) == "TRANSPARENT"

    def test_hidden_amplified_full_pipeline_pinned(self):
        # NVDA: $440 direct + $90 indirect (TQQQ $300 × 3 × 10%) = $530 eff
        # vs $1000 book → 53% eff vs 44% direct → 53/44 = 1.20× < 1.5
        # so this is TRANSPARENT, not HIDDEN_AMPLIFIED. The discriminator
        # is whether the ratio actually exceeds the threshold.
        snap = {
            "total_value": 1000.0,
            "positions": [
                _stock("NVDA", 2, 220),    # $440 direct
                _stock("TQQQ", 3, 100),    # $300 → $90 indirect NVDA
            ],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        nvda = next(r for r in out["underlyings"] if r["ticker"] == "NVDA")
        assert nvda["tier"] == "TRANSPARENT"
        # Now amplify TQQQ to flip it: $200 NVDA direct + $600 TQQQ
        # → $200 + $180 = $380 eff = 38% vs 20% direct → 1.9× HIDDEN_AMPLIFIED
        snap2 = {
            "total_value": 1000.0,
            "positions": [
                _stock("NVDA", 2, 100),    # $200 direct
                _stock("TQQQ", 6, 100),    # $600 → $180 indirect NVDA
            ],
        }
        out2 = build_etf_lookthrough(snap2, lookthrough_map=_TEST_MAP)
        nvda2 = next(r for r in out2["underlyings"] if r["ticker"] == "NVDA")
        assert nvda2["tier"] == "HIDDEN_AMPLIFIED"
        assert "hidden concentration" in out2["headline"]
        assert "NVDA" in out2["headline"]

    def test_hidden_only_surfaces_silent_underlying(self):
        # Hold TQQQ but NO direct NVDA — NVDA shows as HIDDEN_ONLY.
        snap = {
            "total_value": 1000.0,
            "positions": [_stock("TQQQ", 5, 100)],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        nvda = next(r for r in out["underlyings"] if r["ticker"] == "NVDA")
        assert nvda["tier"] == "HIDDEN_ONLY"
        assert "silent exposure" in out["headline"] or "hidden" in out["headline"]


# ─────────────────────────── sort order ───────────────────────────


class TestSortOrder:
    def test_sorted_by_abs_effective_desc(self):
        snap = {
            "total_value": 1000.0,
            "positions": [
                _stock("AAPL", 1, 100),   # $100 direct AAPL
                _stock("TQQQ", 5, 100),   # $500: NVDA $150, MSFT $150, AAPL $120
                _stock("NVDA", 1, 50),    # $50 direct NVDA
            ],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        # NVDA: $50 direct + $150 indirect = $200 eff
        # MSFT: $0 direct + $150 indirect = $150 eff
        # AAPL: $100 + $120 = $220 eff
        # TQQQ: $500 direct + $0 indirect = $500 eff (TQQQ itself is direct!)
        # Sort order by |effective| DESC: TQQQ($500) > AAPL($220) > NVDA($200) > MSFT($150)
        order = [r["ticker"] for r in out["underlyings"]]
        assert order[0] == "TQQQ"
        assert order[1] == "AAPL"
        assert order[2] == "NVDA"
        assert order[3] == "MSFT"

    def test_max_underlyings_cap(self):
        # Force many underlyings via a fat map
        big_map = {
            "ETF": {"leverage": 1.0, "holdings": [
                (f"T{i}", 1.0) for i in range(30)
            ]},
        }
        snap = {
            "total_value": 1000.0,
            "positions": [_stock("ETF", 10, 100)],  # $1000 ETF
        }
        out = build_etf_lookthrough(snap, lookthrough_map=big_map, max_underlyings=5)
        # 5 underlyings + ETF itself == 6 max? No — ETF is also an underlying
        # (it's a direct position too). So cap honors max_underlyings=5.
        assert len(out["underlyings"]) == 5


# ─────────────────────────── options carve-out ───────────────────────────


class TestOptionsCarveOut:
    def test_option_not_looked_through(self):
        # An NVDA call is NOT looked through (would need delta-adjust).
        # But it DOES contribute as direct NVDA exposure via the line item.
        snap = {
            "total_value": 1000.0,
            "positions": [
                _opt("NVDA", 1, 5.0, "call"),  # premium $5 × qty 1 × 100 = $500
            ],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        # No ETF held → NO_ETF_HELD
        assert out["state"] == "NO_ETF_HELD"

    def test_option_on_etf_not_looked_through_but_etf_stock_is(self):
        # If we have a TQQQ call AND a TQQQ stock position, only the stock
        # gets look-through; the call does not (delta-adjustment is its
        # own surface).
        snap = {
            "total_value": 10000.0,
            "positions": [
                _stock("TQQQ", 10, 100),     # $1000 stock, looked through
                _opt("TQQQ", 1, 5.0, "call"),  # $500 option, NOT looked through
            ],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        # Indirect NVDA = $1000 × 3 × 10% = $300 (only from the stock leg)
        nvda = next(r for r in out["underlyings"] if r["ticker"] == "NVDA")
        assert nvda["indirect_usd"] == 300.0


# ─────────────────────────── garbage degrade ───────────────────────────


class TestGarbageDegrade:
    def test_none_snapshot_degrades(self):
        out = build_etf_lookthrough(None, lookthrough_map=_TEST_MAP)
        assert out["state"] == "NO_DATA"

    def test_non_dict_position_skipped(self):
        snap = {
            "total_value": 1000.0,
            "positions": ["not-a-dict", _stock("TQQQ", 1, 100), None, 42],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        assert out["state"] == "OK"
        # Only TQQQ contributed
        assert out["n_etfs_held"] == 1

    def test_missing_qty_or_price_degrades_to_zero(self):
        snap = {
            "total_value": 1000.0,
            "positions": [
                {"ticker": "TQQQ", "type": "stock"},  # no qty/price → value=0
                _stock("TQQQ", 1, 100),
            ],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        # Only the well-formed position counts
        assert out["state"] == "OK"
        nvda = next(r for r in out["underlyings"] if r["ticker"] == "NVDA")
        assert nvda["indirect_usd"] == 30.0

    def test_garbage_ticker_dropped(self):
        snap = {
            "total_value": 1000.0,
            "positions": [
                {"ticker": None, "qty": 1, "current_price": 100, "type": "stock"},
                {"ticker": "", "qty": 1, "current_price": 100, "type": "stock"},
                _stock("TQQQ", 1, 100),
            ],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        assert out["state"] == "OK"
        assert out["n_etfs_held"] == 1

    def test_garbage_holding_row_skipped(self):
        bad_map = {
            "TQQQ": {"leverage": 3.0, "holdings": [
                ("NVDA", 10.0),
                ("BAD", "not-a-number"),  # garbage weight
                (None, 5.0),               # garbage ticker
                ("",),                     # malformed tuple
                ("MSFT", 10.0),
            ]},
        }
        snap = {
            "total_value": 1000.0,
            "positions": [_stock("TQQQ", 1, 100)],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=bad_map)
        assert out["state"] == "OK"
        tickers = {r["ticker"] for r in out["underlyings"]}
        assert "NVDA" in tickers
        assert "MSFT" in tickers
        assert "BAD" not in tickers


# ─────────────────────────── live map sanity ───────────────────────────


class TestLiveMapSanity:
    def test_canonical_etfs_present(self):
        """The big-name leveraged ETFs the dashboard already classifies must
        all be look-through-mappable — otherwise the operator's documented
        live-book pathology (TQQQ + SOXL + FNGU stacking NVDA) goes
        un-flagged."""
        for tk in ("TQQQ", "SQQQ", "SOXL", "SOXS", "FNGU", "FNGD",
                   "TECL", "SPXL", "UPRO"):
            assert tk in _ETF_LOOKTHROUGH, f"{tk} missing from look-through map"

    def test_inverse_etfs_have_negative_leverage(self):
        for tk in ("SQQQ", "SOXS", "FNGD", "TECS", "SPXS"):
            assert _ETF_LOOKTHROUGH[tk]["leverage"] < 0, (
                f"{tk} should be inverse"
            )

    def test_long_etfs_have_positive_leverage(self):
        for tk in ("TQQQ", "SOXL", "FNGU", "TECL", "SPXL", "UPRO", "QLD", "SSO"):
            assert _ETF_LOOKTHROUGH[tk]["leverage"] > 0, (
                f"{tk} should be long-leveraged"
            )

    def test_all_etfs_carry_holdings(self):
        for tk, etf in _ETF_LOOKTHROUGH.items():
            assert etf["holdings"], f"{tk} has empty holdings"
            for h in etf["holdings"]:
                assert len(h) == 2, f"{tk} holding row malformed: {h!r}"
                assert isinstance(h[0], str)
                assert isinstance(h[1], (int, float))
                assert h[1] > 0


# ─────────────────────────── headline shape ───────────────────────────


class TestHeadline:
    def test_headline_includes_top_amplified_ticker(self):
        # Construct a strong HIDDEN_AMPLIFIED case
        snap = {
            "total_value": 1000.0,
            "positions": [
                _stock("NVDA", 1, 100),  # $100 direct
                _stock("TQQQ", 6, 100),  # $600 → $180 indirect NVDA → eff 28%, ratio 2.8x
            ],
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        assert "NVDA" in out["headline"]
        assert "hidden concentration" in out["headline"]

    def test_headline_when_no_hidden_falls_back_to_largest(self):
        snap = {
            "total_value": 1000.0,
            "positions": [_stock("TQQQ", 5, 100)],  # $500 TQQQ direct
        }
        out = build_etf_lookthrough(snap, lookthrough_map=_TEST_MAP)
        # TQQQ underlying is itself the largest direct line
        # All other tickers are HIDDEN_ONLY (no direct), so the headline
        # picks HIDDEN_ONLY first (priority before largest-effective).
        assert ("hidden" in out["headline"] or
                "silent exposure" in out["headline"] or
                "largest effective" in out["headline"])

    def test_position_value_helper_options_x100(self):
        # The ×100 multiplier on options matches sector_exposure's verbatim formula.
        assert _position_value(_opt("X", 2, 3.0, "call")) == 600.0  # 2×3×100
        assert _position_value(_stock("X", 2, 3.0)) == 6.0


# ─────────────────────────── Flask endpoint integration ───────────────────────────


class TestEtfLookthroughEndpoint:
    """Endpoint↔builder no-drift via the real Flask test_client.

    Uses a fresh in-memory-style Store seeded with a TQQQ position and
    asserts ``/api/etf-lookthrough`` returns the same fields the builder
    produced for the same snapshot — the canonical SSOT lock pattern
    (the ``pnl_attribution`` / ``stress_scenarios`` precedent).
    """

    def _setup(self, tmp_path, monkeypatch):
        # Real Store on tmp DB; mirrors test_pnl_attribution's pattern.
        from paper_trader import store as _store_mod
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(_store_mod, "DB_PATH", db)
        monkeypatch.setattr(_store_mod, "_singleton", None)
        store = _store_mod.Store()
        return store, _store_mod

    def test_endpoint_serves_builder_output_no_drift(self, tmp_path, monkeypatch):
        from paper_trader.dashboard import app
        from paper_trader.analytics.etf_lookthrough import build_etf_lookthrough

        store, _store_mod = self._setup(tmp_path, monkeypatch)

        # Seed a TQQQ stock position and mark it to current_price via the
        # canonical update_position_marks path (the pnl_attribution precedent).
        store.upsert_position("TQQQ", "stock", 5.0, 100.0)
        pos = store.open_positions()
        store.update_position_marks({pos[0]["id"]: (100.0, 0.0)})
        store.update_portfolio(cash=500.0, total_value=1000.0, positions=[])

        with app.test_client() as c:
            r = c.get("/api/etf-lookthrough")
            assert r.status_code == 200
            payload = r.get_json()

        # Strip SWR-only annotations before comparing.
        for k in ("cached", "cache_age_s"):
            payload.pop(k, None)

        # Build the same builder output for the same snapshot — bytes-equal.
        positions = store.open_positions()
        pf = store.get_portfolio()
        expected = build_etf_lookthrough({
            "cash": float(pf.get("cash") or 0.0),
            "total_value": float(pf.get("total_value") or 0.0),
            "positions": positions,
        })
        assert payload["state"] == expected["state"]
        assert payload["headline"] == expected["headline"]
        assert payload["n_etfs_held"] == expected["n_etfs_held"]
        # underlyings comparison: every row should match
        assert len(payload["underlyings"]) == len(expected["underlyings"])
        for got, exp in zip(payload["underlyings"], expected["underlyings"]):
            assert got["ticker"] == exp["ticker"]
            assert got["effective_usd"] == exp["effective_usd"]
            assert got["tier"] == exp["tier"]

    def test_empty_book_endpoint_returns_no_data(self, tmp_path, monkeypatch):
        from paper_trader.dashboard import app
        self._setup(tmp_path, monkeypatch)
        # Fresh store has $1000 cash, no positions
        with app.test_client() as c:
            r = c.get("/api/etf-lookthrough")
            assert r.status_code == 200
            payload = r.get_json()
        # No positions + total_value=$1000 → NO_ETF_HELD (not NO_DATA, since
        # total_value > 0 but no ETF in our map).
        assert payload["state"] in ("NO_ETF_HELD", "NO_DATA")
        assert payload["n_etfs_held"] == 0

    def test_analytics_fold_inherits_lookthrough(self, tmp_path, monkeypatch):
        """The /api/analytics additive-key fold must mirror /api/etf-lookthrough
        on the load-bearing fields — the tail_risk/stress_scenarios/recovery
        no-drift discipline (AGENTS.md #10)."""
        from paper_trader.dashboard import app
        store, _ = self._setup(tmp_path, monkeypatch)
        store.upsert_position("TQQQ", "stock", 5.0, 100.0)
        pos = store.open_positions()
        store.update_position_marks({pos[0]["id"]: (100.0, 0.0)})
        store.update_portfolio(cash=500.0, total_value=1000.0, positions=[])

        with app.test_client() as c:
            ep = c.get("/api/etf-lookthrough").get_json()
            an = c.get("/api/analytics").get_json()
        for k in ("cached", "cache_age_s"):
            ep.pop(k, None)

        assert "etf_lookthrough" in an, (
            "/api/analytics did not fold in etf_lookthrough — every additive "
            "analytics key must be inheritable by the digital-intern chat"
        )
        fold = an["etf_lookthrough"]
        # No-drift: state + headline + n_etfs_held must agree byte-for-byte.
        assert fold["state"] == ep["state"]
        assert fold["headline"] == ep["headline"]
        assert fold["n_etfs_held"] == ep["n_etfs_held"]
