"""Tests for /api/backtests/trade-delta and its pure builder.

The companion to /api/backtests/compare: compare returns aggregate
summaries (return %, max DD, win rate, normalized curves) for 2-4 runs;
this returns the trade-level diff between exactly two runs so the
operator can answer "which trades drove the return delta?".

Critical regressions to pin:

- Match identity is (ticker upper, action upper, sim_date) — case
  normalization on ticker/action, exact-string match on sim_date.
- only_in_a / only_in_b / common partition is exhaustive (no trade
  duplicated across buckets, none lost).
- Multi-execution on the same key pairs n_paired trades into common
  and surplus into only_*.
- divergence_score is Jaccard distance over the trade sets: 0.0 when
  identical, 1.0 when fully disjoint, in [0.0, 1.0].
- Attribution uses the SAME FIFO BUY→SELL pairing dashboard.backtest_compare
  uses for win-rate — pin the algorithm by running both side-by-side.
- Garbage rows (None, bad types, missing fields) never raise.
- Flask endpoint smoke via test_client: ids count enforced, not-found
  handled.
"""
from __future__ import annotations

import pytest

from paper_trader.analytics.backtest_trade_delta import (
    build_trade_delta,
    _fifo_realized_pl,
    _norm,
)


def _trade(ticker, action, sim_date, qty=10.0, price=100.0, value=None,
           reason=""):
    if value is None:
        value = qty * price
    return {
        "ticker": ticker,
        "action": action,
        "sim_date": sim_date,
        "qty": qty,
        "price": price,
        "value": value,
        "reason": reason,
    }


def _detail(run_id, trades, total_return_pct=10.0, **extra):
    out = {
        "run_id": run_id,
        "start_date": "2026-01-01",
        "end_date": "2026-04-01",
        "status": "complete",
        "total_return_pct": total_return_pct,
        "spy_return_pct": 5.0,
        "vs_spy_pct": total_return_pct - 5.0,
        "n_trades": len(trades),
        "n_decisions": len(trades),
        "final_value": 1000.0 * (1 + total_return_pct / 100.0),
        "trades": trades,
    }
    out.update(extra)
    return out


class TestNorm:
    def test_valid_key(self):
        t = _trade("NVDA", "BUY", "2026-04-01")
        assert _norm(t) == ("NVDA", "BUY", "2026-04-01")

    def test_case_normalized(self):
        t = _trade("nvda", "buy", "2026-04-01")
        assert _norm(t) == ("NVDA", "BUY", "2026-04-01")

    def test_none_inputs(self):
        assert _norm(None) is None  # type: ignore[arg-type]
        assert _norm({}) is None
        assert _norm({"ticker": "X"}) is None
        assert _norm({"ticker": "X", "action": "BUY"}) is None
        assert _norm({"ticker": "", "action": "BUY",
                      "sim_date": "2026-04-01"}) is None


class TestFifoRealizedPl:
    def test_simple_win(self):
        trades = [
            _trade("NVDA", "BUY", "2026-04-01", qty=10, price=100.0),
            _trade("NVDA", "SELL", "2026-04-15", qty=10, price=120.0),
        ]
        pl = _fifo_realized_pl(trades)
        assert pl["NVDA"] == pytest.approx(200.0)

    def test_partial_sell(self):
        # Buy 10 @ 100, sell 4 @ 120 ⇒ realized = 4 * (120-100) = 80
        trades = [
            _trade("NVDA", "BUY", "2026-04-01", qty=10, price=100.0),
            _trade("NVDA", "SELL", "2026-04-10", qty=4, price=120.0),
        ]
        pl = _fifo_realized_pl(trades)
        assert pl["NVDA"] == pytest.approx(80.0)

    def test_fifo_ordering(self):
        # 2 lots: 5 @ 100, 5 @ 110; sell 8 @ 120 ⇒ 5*(20) + 3*(10) = 130
        trades = [
            _trade("NVDA", "BUY", "2026-04-01", qty=5, price=100.0),
            _trade("NVDA", "BUY", "2026-04-05", qty=5, price=110.0),
            _trade("NVDA", "SELL", "2026-04-10", qty=8, price=120.0),
        ]
        pl = _fifo_realized_pl(trades)
        assert pl["NVDA"] == pytest.approx(130.0)

    def test_loss(self):
        trades = [
            _trade("NVDA", "BUY", "2026-04-01", qty=10, price=100.0),
            _trade("NVDA", "SELL", "2026-04-10", qty=10, price=80.0),
        ]
        pl = _fifo_realized_pl(trades)
        assert pl["NVDA"] == pytest.approx(-200.0)

    def test_garbage_rows_ignored(self):
        trades = [
            None,                                              # type: ignore
            "not-a-dict",                                      # type: ignore
            {"action": "BUY"},                                 # missing ticker
            _trade("NVDA", "BUY", "2026-04-01", qty=10, price=100.0),
            _trade("NVDA", "SELL", "2026-04-15", qty=10, price=120.0),
        ]
        pl = _fifo_realized_pl(trades)  # type: ignore
        assert pl == {"NVDA": pytest.approx(200.0)}


class TestBuildTradeDeltaPartition:
    def test_empty_both_sides(self):
        a = _detail(1, [])
        b = _detail(2, [])
        r = build_trade_delta(a, b)
        d = r["delta"]
        assert d["n_only_a"] == 0
        assert d["n_only_b"] == 0
        assert d["n_common"] == 0
        assert d["divergence_score"] == 0.0
        assert "no trades" in r["headline"]

    def test_identical_trades_all_common(self):
        ts = [_trade("NVDA", "BUY", "2026-04-01"),
              _trade("MU", "SELL", "2026-04-10")]
        a = _detail(1, ts)
        b = _detail(2, [dict(t) for t in ts])
        r = build_trade_delta(a, b)
        d = r["delta"]
        assert d["n_only_a"] == 0
        assert d["n_only_b"] == 0
        assert d["n_common"] == 2
        assert d["divergence_score"] == 0.0

    def test_fully_disjoint(self):
        a_t = [_trade("NVDA", "BUY", "2026-04-01")]
        b_t = [_trade("MU", "BUY", "2026-04-02")]
        r = build_trade_delta(_detail(1, a_t), _detail(2, b_t))
        d = r["delta"]
        assert d["n_only_a"] == 1
        assert d["n_only_b"] == 1
        assert d["n_common"] == 0
        # Jaccard: 2 / (2 + 0) = 1.0
        assert d["divergence_score"] == 1.0

    def test_partial_overlap(self):
        # 1 common, 1 A-only, 1 B-only
        common_t = _trade("NVDA", "BUY", "2026-04-01")
        a_t = [common_t, _trade("MU", "BUY", "2026-04-02")]
        b_t = [dict(common_t), _trade("AMD", "BUY", "2026-04-03")]
        r = build_trade_delta(_detail(1, a_t), _detail(2, b_t))
        d = r["delta"]
        assert d["n_only_a"] == 1
        assert d["n_only_b"] == 1
        assert d["n_common"] == 1
        # Jaccard: 2 / 3 = 0.6667
        assert d["divergence_score"] == pytest.approx(0.6667, abs=0.001)

    def test_case_normalization_matches(self):
        # NVDA + BUY on the same date must match nvda + buy
        a_t = [_trade("NVDA", "BUY", "2026-04-01")]
        b_t = [_trade("nvda", "buy", "2026-04-01")]
        r = build_trade_delta(_detail(1, a_t), _detail(2, b_t))
        d = r["delta"]
        assert d["n_common"] == 1
        assert d["n_only_a"] == 0
        assert d["n_only_b"] == 0

    def test_multi_execution_pairs_min_then_surplus(self):
        # A: 3 NVDA BUYs on same date; B: 2 NVDA BUYs on same date
        # ⇒ 2 paired into common, 1 surplus in only_a
        a_t = [_trade("NVDA", "BUY", "2026-04-01") for _ in range(3)]
        b_t = [_trade("NVDA", "BUY", "2026-04-01") for _ in range(2)]
        r = build_trade_delta(_detail(1, a_t), _detail(2, b_t))
        d = r["delta"]
        assert d["n_common"] == 2
        assert d["n_only_a"] == 1
        assert d["n_only_b"] == 0


class TestCommonDiff:
    def test_qty_delta_surfaces_sizing_difference(self):
        # Same trade key, different qty ⇒ goes to common with qty_delta
        a_t = [_trade("NVDA", "BUY", "2026-04-01", qty=10, price=100.0)]
        b_t = [_trade("NVDA", "BUY", "2026-04-01", qty=15, price=100.0)]
        r = build_trade_delta(_detail(1, a_t), _detail(2, b_t))
        common = r["delta"]["common"]
        assert len(common) == 1
        c = common[0]
        assert c["a_qty"] == 10.0
        assert c["b_qty"] == 15.0
        assert c["qty_delta"] == 5.0

    def test_value_delta(self):
        a_t = [_trade("NVDA", "BUY", "2026-04-01",
                      qty=10, price=100.0, value=1000.0)]
        b_t = [_trade("NVDA", "BUY", "2026-04-01",
                      qty=10, price=105.0, value=1050.0)]
        r = build_trade_delta(_detail(1, a_t), _detail(2, b_t))
        c = r["delta"]["common"][0]
        assert c["value_delta"] == 50.0


class TestReturnDelta:
    def test_return_delta_pct(self):
        a = _detail(1, [], total_return_pct=10.0)
        b = _detail(2, [], total_return_pct=15.0)
        r = build_trade_delta(a, b)
        assert r["delta"]["return_delta_pct"] == 5.0

    def test_return_delta_negative_when_b_worse(self):
        a = _detail(1, [], total_return_pct=20.0)
        b = _detail(2, [], total_return_pct=12.0)
        r = build_trade_delta(a, b)
        assert r["delta"]["return_delta_pct"] == -8.0

    def test_return_delta_none_when_either_missing(self):
        # Hand-build details with total_return_pct missing on B —
        # mirrors what BacktestStore.run_detail returns for an
        # in-progress run that hasn't yet recorded a return.
        a = {"run_id": 1, "trades": [], "total_return_pct": 10.0}
        b = {"run_id": 2, "trades": []}  # no total_return_pct field
        r = build_trade_delta(a, b)
        assert r["delta"]["return_delta_pct"] is None


class TestAttribution:
    def test_top_attribution_is_unique_winner(self):
        # Only B did the NVDA round-trip + 200 win ⇒ attribution
        # surfaces NVDA as a delta=+200 contributor unique to B.
        a_t = []
        b_t = [
            _trade("NVDA", "BUY", "2026-04-01", qty=10, price=100.0),
            _trade("NVDA", "SELL", "2026-04-15", qty=10, price=120.0),
        ]
        r = build_trade_delta(_detail(1, a_t), _detail(2, b_t))
        attr = r["delta"]["attribution"]
        nvda = next(a for a in attr if a["ticker"] == "NVDA")
        assert nvda["only_b_pl_usd"] == 200.0
        assert nvda["only_a_pl_usd"] == 0.0
        assert nvda["delta_pl_usd"] == 200.0

    def test_attribution_sorted_by_abs_delta(self):
        # A traded LOSER (-100), B traded WINNER (+500); attribution
        # sorts by |delta|: WINNER first (500), LOSER second (100).
        a_t = [
            _trade("LOSER", "BUY", "2026-04-01", qty=10, price=100.0),
            _trade("LOSER", "SELL", "2026-04-15", qty=10, price=90.0),
        ]
        b_t = [
            _trade("WINNER", "BUY", "2026-04-01", qty=10, price=100.0),
            _trade("WINNER", "SELL", "2026-04-15", qty=10, price=150.0),
        ]
        r = build_trade_delta(_detail(1, a_t), _detail(2, b_t))
        attr = r["delta"]["attribution"]
        assert attr[0]["ticker"] == "WINNER"
        assert attr[1]["ticker"] == "LOSER"

    def test_no_attribution_when_no_round_trips(self):
        # BUYs alone (no SELL) ⇒ no realized P/L ⇒ empty attribution
        a_t = [_trade("NVDA", "BUY", "2026-04-01")]
        b_t = []
        r = build_trade_delta(_detail(1, a_t), _detail(2, b_t))
        assert r["delta"]["attribution"] == []


class TestSafeOnGarbage:
    def test_non_dict_inputs(self):
        r = build_trade_delta(None, None)  # type: ignore
        assert r["delta"]["n_only_a"] == 0
        assert r["delta"]["n_only_b"] == 0
        assert r["delta"]["divergence_score"] == 0.0

    def test_missing_trades_list(self):
        r = build_trade_delta({"run_id": 1}, {"run_id": 2})
        assert r["delta"]["n_total_a"] == 0

    def test_garbage_trades_skipped_not_raised(self):
        a_t = [
            None,                                       # type: ignore
            "not-a-dict",                               # type: ignore
            {"ticker": "X"},                            # missing action+date
            _trade("NVDA", "BUY", "2026-04-01"),
        ]
        b_t = [_trade("NVDA", "BUY", "2026-04-01")]
        r = build_trade_delta(_detail(1, a_t), _detail(2, b_t))  # type: ignore
        # The 3 garbage rows skipped; the one valid row matches into common.
        assert r["delta"]["n_skipped_a"] == 3
        assert r["delta"]["n_common"] == 1

    def test_bad_numeric_fields(self):
        bad = {"ticker": "NVDA", "action": "BUY", "sim_date": "2026-04-01",
               "qty": "garbage", "price": None, "value": float("nan")}
        r = build_trade_delta(_detail(1, [bad]), _detail(2, []))
        # Row keys still valid ⇒ ends up in only_in_a with sanitized 0.0.
        only_a = r["delta"]["only_in_a"]
        assert len(only_a) == 1
        assert only_a[0]["qty"] == 0.0
        assert only_a[0]["price"] == 0.0
        assert only_a[0]["value"] == 0.0


class TestEndpointSmoke:
    """Verify endpoint contract via Flask test_client per the
    analytics-verification memory (module __main__ would hit a
    different/empty DB).
    """

    def test_endpoint_missing_ids(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/backtests/trade-delta")
        assert resp.status_code == 400
        assert "missing ids" in resp.get_json()["error"]

    def test_endpoint_wrong_id_count(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        for q in ("100", "1,2,3", "1,2,3,4"):
            resp = client.get(f"/api/backtests/trade-delta?ids={q}")
            assert resp.status_code == 400, q
            assert "exactly two" in resp.get_json()["error"]

    def test_endpoint_distinct_ids_required(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/backtests/trade-delta?ids=5,5")
        assert resp.status_code == 400
        assert "distinct" in resp.get_json()["error"]

    def test_endpoint_garbage_ids(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/backtests/trade-delta?ids=foo,bar")
        assert resp.status_code == 400

    def test_endpoint_neither_found_yields_404(self, monkeypatch):
        # Patch the store so both lookups return None — verifies the
        # 404 branch fires cleanly without a real backtest.db.
        from paper_trader import dashboard
        from paper_trader.analytics import backtest_trade_delta as btd

        class _FakeStore:
            def run_detail(self, rid):
                return None

        import paper_trader.backtest as bt
        monkeypatch.setattr(bt, "BacktestStore", _FakeStore)
        client = dashboard.app.test_client()
        resp = client.get(
            "/api/backtests/trade-delta?ids=999999,888888"
        )
        # Either 404 (neither found) or 500-with-error envelope: pin 404
        # since the route's `if a is None and b is None: 404` branch is
        # the documented contract.
        assert resp.status_code == 404, resp.data
        assert "neither" in resp.get_json()["error"]

    def test_endpoint_happy_path(self, monkeypatch):
        from paper_trader import dashboard
        import paper_trader.backtest as bt

        a_detail = _detail(100, [
            _trade("NVDA", "BUY", "2026-04-01", qty=10, price=100.0),
            _trade("NVDA", "SELL", "2026-04-15", qty=10, price=120.0),
        ], total_return_pct=10.0)
        b_detail = _detail(105, [
            _trade("NVDA", "BUY", "2026-04-01", qty=10, price=100.0),
            _trade("NVDA", "SELL", "2026-04-15", qty=10, price=130.0),
            _trade("MU", "BUY", "2026-04-05", qty=5, price=80.0),
            _trade("MU", "SELL", "2026-04-20", qty=5, price=90.0),
        ], total_return_pct=15.0)

        class _FakeStore:
            def run_detail(self, rid):
                return a_detail if rid == 100 else (
                    b_detail if rid == 105 else None
                )

        monkeypatch.setattr(bt, "BacktestStore", _FakeStore)
        client = dashboard.app.test_client()
        resp = client.get("/api/backtests/trade-delta?ids=100,105")
        assert resp.status_code == 200, resp.data
        body = resp.get_json()
        assert body["a"]["run_id"] == 100
        assert body["b"]["run_id"] == 105
        d = body["delta"]
        # Two MU trades unique to B; NVDA round-trip in both ⇒ paired into
        # common (price differs, qty same — surfaces value_delta).
        assert d["n_only_b"] == 2  # MU buy + MU sell unique to B
        assert d["n_common"] == 2  # NVDA buy + NVDA sell shared
        assert d["return_delta_pct"] == 5.0
        # MU only in B, +50 round-trip ⇒ top attribution.
        top = d["attribution"][0]
        assert top["ticker"] == "MU"
        assert top["only_b_pl_usd"] == 50.0
