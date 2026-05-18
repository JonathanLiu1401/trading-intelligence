"""Equity-curve integrity builder + /api/equity-integrity endpoint.

Asserts exact verdicts/values for the three failure modes the builder
exists to catch (negative cash, non-positive equity, no-trade jump) and the
explain-away contract (a jump WITH a trade in its window is NOT suspect),
plus the never-raises degrade contract. Pure — no DB, no network.
"""
import paper_trader.analytics.equity_integrity as ei


def _pt(ts, tv, cash):
    return {"timestamp": ts, "total_value": tv, "cash": cash,
            "sp500_price": None}


def test_no_data_below_min_points():
    out = ei.build_equity_integrity([_pt("2026-05-18T00:00:00+00:00", 1000, 1000)],
                                    [], min_points=3)
    assert out["verdict"] == "NO_DATA"
    assert out["n_points"] == 1
    assert out["n_suspect_jumps"] == 0


def test_clean_flat_curve():
    pts = [_pt(f"2026-05-18T0{i}:00:00+00:00", 1000.0 + i, 500.0)
           for i in range(5)]
    out = ei.build_equity_integrity(pts, [])
    assert out["verdict"] == "CLEAN"
    assert out["n_negative_cash"] == 0
    assert out["n_suspect_jumps"] == 0
    assert out["min_cash_usd"] == 500.0
    assert out["window_start"] == "2026-05-18T00:00:00+00:00"
    assert out["window_end"] == "2026-05-18T04:00:00+00:00"


def test_negative_cash_is_corrupt_and_dominates():
    # A 50% jump (would otherwise be SUSPECT) AND a negative-cash point:
    # CORRUPT must dominate SUSPECT.
    pts = [
        _pt("2026-05-18T00:00:00+00:00", 1000.0, 200.0),
        _pt("2026-05-18T01:00:00+00:00", 1500.0, -25.50),  # +50% & cash<0
        _pt("2026-05-18T02:00:00+00:00", 1500.0, -25.50),
    ]
    out = ei.build_equity_integrity(pts, [])  # no trades → jump is no-trade
    assert out["verdict"] == "CORRUPT"
    assert out["n_negative_cash"] == 2
    assert out["min_cash_usd"] == -25.5
    assert out["negative_cash_points"][0]["cash"] == -25.5
    assert "over-drawn" in out["headline"]


def test_nonpositive_equity_is_corrupt():
    pts = [
        _pt("2026-05-18T00:00:00+00:00", 1000.0, 100.0),
        _pt("2026-05-18T01:00:00+00:00", 0.0, 100.0),     # ruin / corrupt
        _pt("2026-05-18T02:00:00+00:00", 5.0, 100.0),
    ]
    out = ei.build_equity_integrity(pts, [])
    assert out["verdict"] == "CORRUPT"
    assert out["n_nonpositive_equity"] == 1
    assert out["nonpositive_equity_points"][0]["total_value"] == 0.0


def test_no_trade_jump_is_suspect():
    pts = [
        _pt("2026-05-18T00:00:00+00:00", 1000.0, 500.0),
        _pt("2026-05-18T01:00:00+00:00", 1000.0, 500.0),
        _pt("2026-05-18T02:00:00+00:00", 1200.0, 500.0),  # +20% no trade
        _pt("2026-05-18T03:00:00+00:00", 1205.0, 500.0),
    ]
    out = ei.build_equity_integrity(pts, [])
    assert out["verdict"] == "SUSPECT"
    assert out["n_suspect_jumps"] == 1
    wj = out["worst_jump"]
    assert wj["delta_pct"] == 20.0
    assert wj["delta_usd"] == 200.0
    assert wj["from_ts"] == "2026-05-18T01:00:00+00:00"
    assert wj["to_ts"] == "2026-05-18T02:00:00+00:00"


def test_jump_with_trade_in_window_is_explained_away():
    pts = [
        _pt("2026-05-18T00:00:00+00:00", 1000.0, 500.0),
        _pt("2026-05-18T01:00:00+00:00", 1000.0, 500.0),
        _pt("2026-05-18T02:00:00+00:00", 1200.0, 500.0),  # +20% ...
        _pt("2026-05-18T03:00:00+00:00", 1205.0, 500.0),
    ]
    # ...but a trade fired inside (01:00, 02:00] → expected, not suspect.
    trades = [{"timestamp": "2026-05-18T01:30:00+00:00", "action": "BUY",
               "ticker": "NVDA", "value": 200.0}]
    out = ei.build_equity_integrity(pts, trades)
    assert out["verdict"] == "CLEAN"
    assert out["n_suspect_jumps"] == 0


def test_window_boundary_is_half_open_lo_exclusive_hi_inclusive():
    # Trade exactly AT the lower bound (prev ts) must NOT explain the window
    # (it belongs to the prior window); a trade exactly at the upper bound
    # (cur ts) MUST explain it.
    pts = [
        _pt("2026-05-18T00:00:00+00:00", 1000.0, 500.0),
        _pt("2026-05-18T01:00:00+00:00", 1300.0, 500.0),  # +30%
    ]
    # min_points=2 so the single transition is audited.
    at_lo = [{"timestamp": "2026-05-18T00:00:00+00:00", "action": "BUY",
              "ticker": "X", "value": 1.0}]
    out_lo = ei.build_equity_integrity(pts, at_lo, min_points=2)
    assert out_lo["verdict"] == "SUSPECT", "trade at lo bound must not explain"

    at_hi = [{"timestamp": "2026-05-18T01:00:00+00:00", "action": "BUY",
              "ticker": "X", "value": 1.0}]
    out_hi = ei.build_equity_integrity(pts, at_hi, min_points=2)
    assert out_hi["verdict"] == "CLEAN", "trade at hi bound must explain"


def test_unsorted_input_is_sorted_defensively():
    pts = [
        _pt("2026-05-18T02:00:00+00:00", 1200.0, 500.0),
        _pt("2026-05-18T00:00:00+00:00", 1000.0, 500.0),
        _pt("2026-05-18T01:00:00+00:00", 1000.0, 500.0),
    ]
    out = ei.build_equity_integrity(pts, [])
    assert out["window_start"] == "2026-05-18T00:00:00+00:00"
    assert out["window_end"] == "2026-05-18T02:00:00+00:00"
    # 00:00→01:00 flat, 01:00→02:00 +20% no-trade → SUSPECT (proves the
    # transitions were evaluated in chronological, not input, order).
    assert out["verdict"] == "SUSPECT"


def test_never_raises_on_garbage():
    garbage = [
        None,
        {"timestamp": None, "total_value": 1.0},
        {"timestamp": "2026-05-18T00:00:00+00:00", "total_value": "NaNish"},
        {"timestamp": "2026-05-18T01:00:00+00:00", "total_value": 1000.0,
         "cash": "oops"},
        {"timestamp": "2026-05-18T02:00:00+00:00", "total_value": 1000.0,
         "cash": None},
        {"timestamp": "2026-05-18T03:00:00+00:00", "total_value": 1000.0,
         "cash": 50.0},
    ]
    out = ei.build_equity_integrity(garbage, [{"timestamp": None}, "junk"])
    # 3 parseable points (rows with a usable total_value + ts) → audited,
    # no crash; unparseable cash is simply not counted as negative.
    assert out["verdict"] in ("CLEAN", "SUSPECT", "CORRUPT", "NO_DATA")
    assert out["n_negative_cash"] == 0


def test_endpoint_smoke(monkeypatch):
    """/api/equity-integrity returns 200 + the builder's keys, sourced from
    the store (no network)."""
    import paper_trader.dashboard as d

    class _FakeStore:
        def equity_curve(self, limit=5000):
            return [_pt(f"2026-05-18T0{i}:00:00+00:00", 1000.0 + i, 500.0)
                    for i in range(4)]

        def recent_trades(self, limit=5000):
            return []

    monkeypatch.setattr(d, "get_store", lambda: _FakeStore())
    client = d.app.test_client()
    r = client.get("/api/equity-integrity")
    assert r.status_code == 200
    body = r.get_json()
    assert body["verdict"] == "CLEAN"
    assert body["n_points"] == 4
    assert "as_of" in body and body["headline"]
