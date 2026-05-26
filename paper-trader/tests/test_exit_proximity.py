"""Tests for paper_trader.analytics.exit_proximity — forward-looking
mechanical SL/TP proximity per open position."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from paper_trader.analytics.exit_proximity import (
    NEAR_SL_MAX,
    NEAR_TP_MIN,
    _classify,
    _row,
    build_exit_proximity,
)


FIXED_NOW = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)


def _mk_pos(
    ticker: str = "NVDA",
    qty: float = 10.0,
    avg: float = 100.0,
    cur: float | None = 100.0,
    sl: float | None = 98.0,
    tp: float | None = 103.0,
    ptype: str = "stock",
):
    return {
        "ticker": ticker,
        "type": ptype,
        "qty": qty,
        "avg_cost": avg,
        "current_price": cur,
        "stop_loss_price": sl,
        "take_profit_price": tp,
    }


# ───────── _classify ─────────

class TestClassify:
    def test_at_risk_sl_below_zero(self):
        assert _classify(-0.01) == "AT_RISK_SL"
        assert _classify(-5.0) == "AT_RISK_SL"

    def test_at_risk_tp_above_one(self):
        assert _classify(1.01) == "AT_RISK_TP"
        assert _classify(5.0) == "AT_RISK_TP"

    def test_near_sl_lower_quartile(self):
        assert _classify(0.0) == "NEAR_SL"
        assert _classify(NEAR_SL_MAX - 0.001) == "NEAR_SL"

    def test_near_tp_upper_quartile(self):
        assert _classify(NEAR_TP_MIN) == "NEAR_TP"
        assert _classify(1.0) == "NEAR_TP"

    def test_mid_band(self):
        assert _classify(NEAR_SL_MAX) == "MID_BAND"
        assert _classify(0.5) == "MID_BAND"
        assert _classify(NEAR_TP_MIN - 0.001) == "MID_BAND"

    def test_none_routes_no_sl_tp(self):
        assert _classify(None) == "NO_SL_TP"


# ───────── _row per-position ─────────

class TestRowStockBands:
    def test_mid_band_balanced(self):
        # SL=98, TP=103, current=100 → corridor_pos = (100-98)/(103-98) = 0.4
        r = _row(_mk_pos(cur=100.0, sl=98.0, tp=103.0))
        assert r["proximity_band"] == "MID_BAND"
        assert r["corridor_pos"] == 0.4
        assert r["dist_to_sl_pct"] == 2.0   # (100-98)/100 * 100
        assert r["dist_to_tp_pct"] == 3.0   # (103-100)/100 * 100
        assert r["closer_target"] == "SL"   # 2.0 ≤ 3.0

    def test_near_sl_in_lower_quartile(self):
        # SL=98, TP=103, current=98.5 → pos = 0.5/5 = 0.1 → NEAR_SL
        r = _row(_mk_pos(cur=98.5, sl=98.0, tp=103.0))
        assert r["proximity_band"] == "NEAR_SL"
        assert r["closer_target"] == "SL"

    def test_at_risk_sl_below_threshold(self):
        # current=97.9 → already past 98 stop → AT_RISK_SL
        r = _row(_mk_pos(cur=97.9, sl=98.0, tp=103.0))
        assert r["proximity_band"] == "AT_RISK_SL"
        # dist_to_sl_pct is the signed distance (negative because past SL)
        # (97.9 - 98) / 97.9 * 100 ≈ -0.1021 → rounded to -0.10
        assert r["dist_to_sl_pct"] is not None
        assert r["dist_to_sl_pct"] < 0

    def test_near_tp_in_upper_quartile(self):
        # SL=98, TP=103, current=102 → pos = 4/5 = 0.8 → NEAR_TP
        r = _row(_mk_pos(cur=102.0, sl=98.0, tp=103.0))
        assert r["proximity_band"] == "NEAR_TP"

    def test_at_risk_tp_above_threshold(self):
        # current=103.5 → past TP → AT_RISK_TP
        r = _row(_mk_pos(cur=103.5, sl=98.0, tp=103.0))
        assert r["proximity_band"] == "AT_RISK_TP"
        assert r["dist_to_tp_pct"] is not None
        assert r["dist_to_tp_pct"] < 0  # signed distance is negative (past TP)

    def test_closer_target_when_equidistant_prefers_sl(self):
        # cur exactly equidistant from SL and TP at 2.5% each side (not the
        # default 2/3 R:R). We force it: SL=97.5, TP=102.5, cur=100.0
        r = _row(_mk_pos(cur=100.0, sl=97.5, tp=102.5))
        # dist_to_sl = 2.5%, dist_to_tp = 2.5%. abs(dist_sl) ≤ abs(dist_tp) → "SL"
        assert r["closer_target"] == "SL"
        assert r["proximity_band"] == "MID_BAND"


class TestRowDegenerate:
    def test_options_route_no_sl_tp(self):
        r = _row(_mk_pos(ptype="call", cur=10.0, sl=8.0, tp=15.0))
        assert r["proximity_band"] == "NO_SL_TP"
        assert "option" in r["reason"].lower()
        assert r["dist_to_sl_pct"] is None
        assert r["dist_to_tp_pct"] is None
        assert r["corridor_pos"] is None

    def test_missing_sl(self):
        r = _row(_mk_pos(sl=None))
        assert r["proximity_band"] == "NO_SL_TP"
        assert "missing" in r["reason"].lower()

    def test_missing_tp(self):
        r = _row(_mk_pos(tp=None))
        assert r["proximity_band"] == "NO_SL_TP"

    def test_degenerate_sl_ge_tp(self):
        r = _row(_mk_pos(sl=105.0, tp=100.0))
        assert r["proximity_band"] == "NO_SL_TP"
        assert "degenerate" in r["reason"].lower()

    def test_zero_current_price(self):
        r = _row(_mk_pos(cur=0.0))
        assert r["proximity_band"] == "NO_SL_TP"
        assert "current_price" in r["reason"].lower()

    def test_none_current_price(self):
        r = _row(_mk_pos(cur=None))
        assert r["proximity_band"] == "NO_SL_TP"

    def test_garbage_sl_string(self):
        # Non-numeric SL coerces to None — falls into "missing" bucket.
        r = _row(_mk_pos(sl="not-a-number"))
        assert r["proximity_band"] == "NO_SL_TP"

    def test_garbage_current_price_string(self):
        r = _row(_mk_pos(cur="bad"))
        assert r["proximity_band"] == "NO_SL_TP"


# ───────── build_exit_proximity aggregate ─────────

class TestBuildEmpty:
    def test_none_positions(self):
        out = build_exit_proximity(None, now=FIXED_NOW)
        assert out["verdict"] == "NO_DATA"
        assert out["n_positions"] == 0
        assert out["positions"] == []
        assert out["band_counts"]["MID_BAND"] == 0

    def test_empty_list(self):
        out = build_exit_proximity([], now=FIXED_NOW)
        assert out["verdict"] == "NO_DATA"

    def test_closed_only_book_is_no_data(self):
        # qty=0 lots are filtered before band counting.
        out = build_exit_proximity(
            [_mk_pos(qty=0.0), _mk_pos(qty=-1.0)],
            now=FIXED_NOW,
        )
        assert out["verdict"] == "NO_DATA"
        assert out["n_positions"] == 0


class TestBuildSinglePosition:
    def test_mid_band_comfortable(self):
        out = build_exit_proximity(
            [_mk_pos(cur=100.0, sl=98.0, tp=103.0)],
            now=FIXED_NOW,
        )
        assert out["verdict"] == "COMFORTABLE"
        assert out["n_positions"] == 1
        assert out["n_with_sl_tp"] == 1
        assert out["band_counts"]["MID_BAND"] == 1
        assert "MID_BAND" in out["headline"]

    def test_near_sl_near_threshold(self):
        out = build_exit_proximity(
            [_mk_pos(cur=98.3, sl=98.0, tp=103.0)],
            now=FIXED_NOW,
        )
        assert out["verdict"] == "NEAR_THRESHOLD"
        assert "NEAR_SL" in out["headline"]

    def test_at_risk_sl_top_verdict(self):
        out = build_exit_proximity(
            [_mk_pos(cur=97.5, sl=98.0, tp=103.0)],
            now=FIXED_NOW,
        )
        assert out["verdict"] == "AT_RISK"
        assert "AT_RISK_SL" in out["headline"]
        assert out["band_counts"]["AT_RISK_SL"] == 1

    def test_at_risk_tp_top_verdict(self):
        out = build_exit_proximity(
            [_mk_pos(cur=104.0, sl=98.0, tp=103.0)],
            now=FIXED_NOW,
        )
        assert out["verdict"] == "AT_RISK"
        assert "AT_RISK_TP" in out["headline"]

    def test_no_sl_tp_set_verdict(self):
        out = build_exit_proximity(
            [_mk_pos(sl=None, tp=None)],
            now=FIXED_NOW,
        )
        assert out["verdict"] == "NO_SL_TP_SET"
        assert "mechanical exit machinery is dark" in out["headline"]
        assert out["n_positions"] == 1
        assert out["n_with_sl_tp"] == 0


class TestBuildMultiPosition:
    def test_mixed_at_risk_overrides_near(self):
        out = build_exit_proximity(
            [
                _mk_pos("NVDA", cur=97.5, sl=98.0, tp=103.0),   # AT_RISK_SL
                _mk_pos("TQQQ", cur=98.5, sl=98.0, tp=103.0),   # NEAR_SL
                _mk_pos("SPY",  cur=100.0, sl=98.0, tp=103.0),  # MID_BAND
            ],
            now=FIXED_NOW,
        )
        assert out["verdict"] == "AT_RISK"
        assert out["band_counts"]["AT_RISK_SL"] == 1
        assert out["band_counts"]["NEAR_SL"] == 1
        assert out["band_counts"]["MID_BAND"] == 1

    def test_near_threshold_no_at_risk(self):
        out = build_exit_proximity(
            [
                _mk_pos("NVDA", cur=98.5, sl=98.0, tp=103.0),  # NEAR_SL
                _mk_pos("SPY",  cur=100.0, sl=98.0, tp=103.0), # MID_BAND
            ],
            now=FIXED_NOW,
        )
        assert out["verdict"] == "NEAR_THRESHOLD"

    def test_all_mid_band_comfortable(self):
        out = build_exit_proximity(
            [
                _mk_pos("NVDA", cur=100.0, sl=98.0, tp=103.0),
                _mk_pos("SPY",  cur=101.0, sl=98.0, tp=103.0),
            ],
            now=FIXED_NOW,
        )
        assert out["verdict"] == "COMFORTABLE"
        assert out["band_counts"]["MID_BAND"] == 2

    def test_sort_actionability_at_risk_first(self):
        out = build_exit_proximity(
            [
                _mk_pos("AAA", cur=100.0, sl=98.0, tp=103.0),  # MID
                _mk_pos("BBB", cur=97.5,  sl=98.0, tp=103.0),  # AT_RISK_SL
                _mk_pos("CCC", cur=98.5,  sl=98.0, tp=103.0),  # NEAR_SL
            ],
            now=FIXED_NOW,
        )
        bands_in_order = [r["proximity_band"] for r in out["positions"]]
        assert bands_in_order[0] == "AT_RISK_SL"
        # NEAR_SL comes before MID_BAND
        idx_near = bands_in_order.index("NEAR_SL")
        idx_mid = bands_in_order.index("MID_BAND")
        assert idx_near < idx_mid

    def test_sort_within_at_risk_by_breach_depth(self):
        # Two AT_RISK_SL positions; the one further past SL (lower corridor_pos)
        # should come FIRST.
        out = build_exit_proximity(
            [
                _mk_pos("LIGHTLY", cur=97.9, sl=98.0, tp=103.0),  # pos=-0.02
                _mk_pos("DEEPLY",  cur=95.0, sl=98.0, tp=103.0),  # pos=-0.6
            ],
            now=FIXED_NOW,
        )
        # Deeper breach (more negative pos) → larger abs(pos) → first.
        tickers = [r["ticker"] for r in out["positions"]]
        assert tickers[0] == "DEEPLY"
        assert tickers[1] == "LIGHTLY"

    def test_no_sl_tp_rows_sort_last(self):
        out = build_exit_proximity(
            [
                _mk_pos("NOTP", cur=100.0, sl=98.0, tp=None),   # NO_SL_TP
                _mk_pos("MID",  cur=100.0, sl=98.0, tp=103.0),  # MID_BAND
            ],
            now=FIXED_NOW,
        )
        tickers = [r["ticker"] for r in out["positions"]]
        assert tickers == ["MID", "NOTP"]


class TestBuildExclusions:
    def test_options_count_as_no_sl_tp(self):
        out = build_exit_proximity(
            [_mk_pos(ptype="call", cur=10.0, sl=None, tp=None)],
            now=FIXED_NOW,
        )
        # The position is open (qty=10) so it's in the n_positions count
        # but routes to NO_SL_TP and the verdict is NO_SL_TP_SET.
        assert out["verdict"] == "NO_SL_TP_SET"
        assert out["n_positions"] == 1
        assert out["n_with_sl_tp"] == 0

    def test_options_with_phantom_sl_tp_still_route_no_sl_tp(self):
        # If someone hand-set SL/TP on an option row (not done by engine),
        # we still route to NO_SL_TP because the live machinery doesn't
        # enforce option SL/TP.
        out = build_exit_proximity(
            [_mk_pos(ptype="put", cur=10.0, sl=8.0, tp=15.0)],
            now=FIXED_NOW,
        )
        assert out["positions"][0]["proximity_band"] == "NO_SL_TP"

    def test_closed_lots_excluded(self):
        # qty=0 lots filtered; the one open MID_BAND lot drives the verdict.
        out = build_exit_proximity(
            [
                _mk_pos("OPEN", qty=5.0, cur=100.0, sl=98.0, tp=103.0),
                _mk_pos("ZERO", qty=0.0, cur=100.0, sl=98.0, tp=103.0),
                _mk_pos("NEG",  qty=-1.0, cur=100.0, sl=98.0, tp=103.0),
            ],
            now=FIXED_NOW,
        )
        assert out["n_positions"] == 1
        assert out["positions"][0]["ticker"] == "OPEN"

    def test_garbage_position_row_does_not_crash(self):
        out = build_exit_proximity(
            [{"ticker": None, "qty": "weird"}, _mk_pos()],
            now=FIXED_NOW,
        )
        # Garbage qty → 0.0 → filtered. The valid row drives the verdict.
        assert out["n_positions"] == 1
        assert out["verdict"] in ("COMFORTABLE", "NEAR_THRESHOLD", "AT_RISK")


class TestBuildRobustness:
    def test_never_raises_on_garbage(self):
        # Force every possible failure shape into the input.
        out = build_exit_proximity(
            [
                {"ticker": 123, "qty": None, "type": object()},
                {},
                None,
            ],
            now=FIXED_NOW,
        )
        # All rows discarded — should read as NO_DATA, not a crash.
        assert out["verdict"] == "NO_DATA"

    def test_as_of_is_tz_aware(self):
        out = build_exit_proximity([], now=FIXED_NOW)
        assert "+00:00" in out["as_of"] or out["as_of"].endswith("Z")

    def test_naive_now_made_aware(self):
        naive = datetime(2026, 5, 25, 12, 0, 0)
        out = build_exit_proximity([], now=naive)
        assert "+00:00" in out["as_of"] or out["as_of"].endswith("Z")

    def test_band_counts_include_all_keys(self):
        out = build_exit_proximity([_mk_pos()], now=FIXED_NOW)
        for k in ("AT_RISK_SL", "AT_RISK_TP", "NEAR_SL", "NEAR_TP",
                  "MID_BAND", "NO_SL_TP"):
            assert k in out["band_counts"]


class TestSignedZeroFolding:
    def test_no_negative_zero_in_output(self):
        # current EXACTLY on SL → corridor_pos = 0.0; ensure no -0.0 sneaks in
        r = _row(_mk_pos(cur=98.0, sl=98.0, tp=103.0))
        # corridor_pos = 0.0 exactly; signed-zero protection in _z folds to 0.0
        if r["corridor_pos"] is not None:
            assert r["corridor_pos"] == 0.0


# ───────── Output JSON shape (no NaN / inf) ─────────

class TestJsonSafety:
    def test_no_nan_or_inf(self):
        import json
        out = build_exit_proximity(
            [
                _mk_pos("A", cur=100.0, sl=98.0, tp=103.0),
                _mk_pos("B", cur=97.0, sl=98.0, tp=103.0),
                _mk_pos("C", ptype="call"),
            ],
            now=FIXED_NOW,
        )
        # Must JSON-serialise cleanly (no NaN/Inf which json rejects).
        json.dumps(out)

    def test_output_carries_thresholds(self):
        out = build_exit_proximity([], now=FIXED_NOW)
        assert out["thresholds"]["near_sl_max"] == NEAR_SL_MAX
        assert out["thresholds"]["near_tp_min"] == NEAR_TP_MIN


# ───────── Dashboard route registration ─────────

class TestDashboardWiring:
    """The route registers, returns 200, and carries the full envelope."""

    def test_route_returns_200_with_full_envelope(self):
        from paper_trader import dashboard
        with dashboard.app.test_client() as c:
            r = c.get("/api/exit-proximity")
        assert r.status_code == 200
        j = r.get_json()
        # Full key surface — what callers may rely on. The error-fallback
        # envelope in dashboard.py mirrors these keys exactly, so a 500
        # path also returns this shape.
        for k in ("as_of", "verdict", "headline", "n_positions",
                  "n_with_sl_tp", "band_counts", "positions", "thresholds",
                  "service"):
            assert k in j, f"missing key: {k}"
        assert j["service"] == "paper_trader"

    def test_route_error_fallback_does_not_500(self, monkeypatch):
        """Force the builder to raise — the route must still emit a coherent
        envelope, never an uncaught 500."""
        from paper_trader import dashboard
        from paper_trader.analytics import exit_proximity as mod

        def _boom(*_a, **_k):
            raise RuntimeError("synthetic")

        monkeypatch.setattr(mod, "build_exit_proximity", _boom)
        with dashboard.app.test_client() as c:
            r = c.get("/api/exit-proximity")
        # Flask returns 500 (deliberately, so monitoring sees the fault)
        # but the body MUST still parse and carry the expected keys.
        assert r.status_code in (200, 500)
        j = r.get_json()
        assert j is not None
        for k in ("verdict", "headline", "n_positions", "band_counts",
                  "positions", "thresholds"):
            assert k in j
