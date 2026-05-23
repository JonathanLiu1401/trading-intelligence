"""Tests for the per-name concentration-cap recommender.

* **Builder** — ``paper_trader.analytics.concentration_cap.build_concentration_cap``
  pins the cap-clamp band, the trim arithmetic, the AT_CAP vs OVER_CAP
  verdict, the worst-first sort, and the never-raises / NO_DATA degradation
  contract.

* **Endpoint** — ``/api/concentration-cap`` is a pure passthrough; tests cover
  the ``?cap_pct=`` query parse, builder-exception → 500, prewarm
  registration.

No live process, no network.
"""
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paper_trader.dashboard as dash
import paper_trader.analytics.concentration_cap as cc
from paper_trader.analytics.concentration_cap import (
    DEFAULT_CAP_PCT,
    MIN_CAP_PCT,
    MAX_CAP_PCT,
    build_concentration_cap,
)


# ─────────────────────────────── builder: degradation ──────────────────────
def test_no_positions_is_no_data():
    res = build_concentration_cap([], 1000.0)
    assert res["state"] == "NO_DATA"
    assert res["over_cap_positions"] == []
    assert "no priced book" in res["headline"]


def test_none_positions_is_no_data():
    assert build_concentration_cap(None, 1000.0)["state"] == "NO_DATA"


def test_zero_total_value_is_no_data():
    rows = [{"ticker": "NVDA", "current_price": 100.0, "qty": 1}]
    assert build_concentration_cap(rows, 0.0)["state"] == "NO_DATA"


def test_all_rows_zero_value_is_no_data():
    rows = [{"ticker": "DEAD", "current_price": 0.0, "qty": 0}]
    assert build_concentration_cap(rows, 1000.0)["state"] == "NO_DATA"


def test_builder_never_raises_on_malformed_rows():
    rows = [{"ticker": "JUNK", "current_price": "abc", "qty": "xyz"},
            {"ticker": "NVDA", "current_price": 200.0, "qty": 3}]
    res = build_concentration_cap(rows, 1000.0)
    assert res["n_positions"] == 1


# ───────────────────────────── builder: arithmetic ─────────────────────────
def test_over_cap_trim_qty_is_exact():
    """NVDA $800 in $1000 book = 80%; cap 25% → target $250; trim $550;
    shares to trim = 4 × (550/800) = 2.75."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]   # $800
    res = build_concentration_cap(rows, 1000.0, cap_pct=25.0)
    assert res["state"] == "OVER_CAP"
    assert res["n_over_cap"] == 1
    o = res["over_cap_positions"][0]
    assert o["ticker"] == "NVDA"
    assert o["current_weight_pct"] == 80.0
    assert o["target_weight_pct"] == 25.0
    assert o["target_market_value_usd"] == 250.0
    assert o["cash_freed_usd"] == 550.0
    assert o["shares_to_trim"] == 2.75


def test_total_cash_freed_sums_across_over_cap_rows():
    """Two over-cap names → ``total_cash_freed_usd`` is the sum."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 3},   # $600 = 60%
            {"ticker": "AMD", "current_price": 100.0, "qty": 3}]    # $300 = 30%
    res = build_concentration_cap(rows, 1000.0, cap_pct=25.0)
    # NVDA: trim $600-$250 = $350. AMD: trim $300-$250 = $50.
    assert res["n_over_cap"] == 2
    assert res["total_cash_freed_usd"] == 400.0


def test_at_cap_verdict_when_no_position_exceeds():
    """Heaviest 25% in a $1000 book with cap 25 → AT_CAP, no trims."""
    rows = [{"ticker": "NVDA", "current_price": 250.0, "qty": 1},    # 25%
            {"ticker": "AMD", "current_price": 100.0, "qty": 1}]     # 10%
    res = build_concentration_cap(rows, 1000.0, cap_pct=25.0)
    assert res["state"] == "AT_CAP"
    assert res["n_over_cap"] == 0
    assert res["over_cap_positions"] == []


def test_baseline_top1_top3_computed():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 3},   # $600 = 60%
            {"ticker": "AMD", "current_price": 100.0, "qty": 2},    # $200 = 20%
            {"ticker": "MU", "current_price": 50.0, "qty": 2}]      # $100 = 10%
    res = build_concentration_cap(rows, 1000.0, cap_pct=25.0)
    assert res["baseline"]["top1_ticker"] == "NVDA"
    assert res["baseline"]["top1_pct"] == 60.0
    # top3 = 60 + 20 + 10 = 90 (all positions).
    assert res["baseline"]["top3_pct"] == 90.0


def test_projected_top1_drops_after_trim():
    """After capping NVDA at 25%, the projected top1 should be 25% (NVDA
    still heaviest at the cap) — the operator sees the reduced exposure."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 3},   # 60%
            {"ticker": "AMD", "current_price": 100.0, "qty": 2}]    # 20%
    res = build_concentration_cap(rows, 1000.0, cap_pct=25.0)
    # NVDA capped at 25%; AMD at 20%. Top1 = NVDA 25%.
    assert res["projected"]["top1_ticker"] == "NVDA"
    assert res["projected"]["top1_pct"] == 25.0


def test_projected_top1_equals_cap_when_only_top1_was_over():
    """Trimming only the over-cap name reweights it to exactly the cap; with
    no other name above the cap, the projected top1 is cap%."""
    rows = [{"ticker": "NVDA", "current_price": 600.0, "qty": 1},   # $600 = 60%
            {"ticker": "AMD", "current_price": 150.0, "qty": 1}]    # $150 = 15%
    # cap 25%: NVDA capped at $250 (25%); AMD unchanged at $150 (15%).
    res = build_concentration_cap(rows, 1000.0, cap_pct=25.0)
    assert res["projected"]["top1_ticker"] == "NVDA"
    assert res["projected"]["top1_pct"] == 25.0
    # And top3 is the sum of all (NVDA 25 + AMD 15) = 40%.
    assert res["projected"]["top3_pct"] == 40.0


def test_options_multiplier_in_value():
    rows = [{"ticker": "NVDA", "type": "call",
             "current_price": 5.0, "qty": 2}]   # $1000 notional
    res = build_concentration_cap(rows, 2000.0, cap_pct=25.0)
    assert res["over_cap_positions"][0]["current_market_value_usd"] == 1000.0


# ───────────────────────────── cap clamp ───────────────────────────────────
def test_cap_clamped_low():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 1}]
    res = build_concentration_cap(rows, 1000.0, cap_pct=0.0)
    assert res["cap_pct"] == MIN_CAP_PCT


def test_cap_clamped_high():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 1}]
    res = build_concentration_cap(rows, 1000.0, cap_pct=999.0)
    assert res["cap_pct"] == MAX_CAP_PCT


def test_cap_garbage_falls_back_to_default():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 1}]
    res = build_concentration_cap(rows, 1000.0, cap_pct="abc")
    assert res["cap_pct"] == DEFAULT_CAP_PCT


def test_cap_threshold_is_inclusive_at_exactly_cap():
    """Position at exactly cap%% is NOT over-cap (inclusive boundary)."""
    rows = [{"ticker": "NVDA", "current_price": 250.0, "qty": 1}]   # exactly 25%
    res = build_concentration_cap(rows, 1000.0, cap_pct=25.0)
    assert res["n_over_cap"] == 0
    assert res["state"] == "AT_CAP"


# ───────────────────────────── ordering / headline ────────────────────────
def test_over_cap_rows_worst_first():
    """Largest weight_pct_reduction first — operator sees the biggest cut
    requirement at the top."""
    rows = [{"ticker": "MID", "current_price": 300.0, "qty": 1},    # 30%
            {"ticker": "BIG", "current_price": 600.0, "qty": 1},    # 60%
            {"ticker": "SMALL", "current_price": 100.0, "qty": 1}]  # 10%
    res = build_concentration_cap(rows, 1000.0, cap_pct=25.0)
    order = [r["ticker"] for r in res["over_cap_positions"]]
    assert order == ["BIG", "MID"]   # SMALL is under cap


def test_headline_over_cap_mentions_worst_and_target():
    rows = [{"ticker": "NVDA", "current_price": 300.0, "qty": 2}]   # 60%
    res = build_concentration_cap(rows, 1000.0, cap_pct=25.0)
    h = res["headline"]
    assert "NVDA" in h
    assert "25.0%" in h or "25%" in h
    assert "→" in h    # baseline → projected arrow


def test_headline_at_cap_mentions_largest():
    rows = [{"ticker": "NVDA", "current_price": 150.0, "qty": 1}]   # 15%
    res = build_concentration_cap(rows, 1000.0, cap_pct=25.0)
    assert "NVDA" in res["headline"]
    assert "within cap" in res["headline"]


# ─────────────────────────── endpoint / route ──────────────────────────────
def _client():
    return dash.app.test_client()


def test_endpoint_uses_default_cap_when_param_absent(monkeypatch):
    seen = {}

    def _spy(positions, total_value, cap, *a, **k):
        seen["cap"] = cap
        return {"as_of": "now", "state": "AT_CAP", "cap_pct": cap,
                "total_value_usd": 0.0, "n_positions": 0, "n_over_cap": 0,
                "total_cash_freed_usd": 0.0, "over_cap_positions": [],
                "baseline": {}, "projected": {}, "headline": "ok"}

    monkeypatch.setattr(cc, "build_concentration_cap", _spy)
    resp = _client().get("/api/concentration-cap")
    assert resp.status_code == 200
    assert seen["cap"] == DEFAULT_CAP_PCT


def test_endpoint_reads_cap_pct_query_param(monkeypatch):
    seen = {}

    def _spy(positions, total_value, cap, *a, **k):
        seen["cap"] = cap
        return {"as_of": "now", "state": "AT_CAP", "cap_pct": cap,
                "total_value_usd": 0.0, "n_positions": 0, "n_over_cap": 0,
                "total_cash_freed_usd": 0.0, "over_cap_positions": [],
                "baseline": {}, "projected": {}, "headline": "ok"}

    monkeypatch.setattr(cc, "build_concentration_cap", _spy)
    resp = _client().get("/api/concentration-cap?cap_pct=40")
    assert resp.status_code == 200
    assert seen["cap"] == 40.0


def test_endpoint_garbage_cap_param_falls_back_to_default(monkeypatch):
    """A non-numeric ``?cap_pct=`` doesn't 500 — it falls back to the default
    (route-level) which the builder then clamps."""
    seen = {}

    def _spy(positions, total_value, cap, *a, **k):
        seen["cap"] = cap
        return {"as_of": "now", "state": "AT_CAP", "cap_pct": cap,
                "total_value_usd": 0.0, "n_positions": 0, "n_over_cap": 0,
                "total_cash_freed_usd": 0.0, "over_cap_positions": [],
                "baseline": {}, "projected": {}, "headline": "ok"}

    monkeypatch.setattr(cc, "build_concentration_cap", _spy)
    resp = _client().get("/api/concentration-cap?cap_pct=abc")
    assert resp.status_code == 200
    assert seen["cap"] == DEFAULT_CAP_PCT


def test_endpoint_exception_yields_500(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(cc, "build_concentration_cap", _boom)
    resp = _client().get("/api/concentration-cap")
    assert resp.status_code == 500


def test_endpoint_ssot_parity_with_builder():
    """Route fed by the live store == direct ``build_concentration_cap``
    call, minus ``as_of``. Bypasses @swr_cached via ``__wrapped__`` so a
    prior test's cache state can never make the parity flake."""
    from paper_trader.store import get_store
    store = get_store()
    pf = store.get_portfolio()
    direct = build_concentration_cap(
        store.open_positions(),
        float(pf.get("total_value") or 0.0),
        DEFAULT_CAP_PCT,
    )
    raw = dash.concentration_cap_api.__wrapped__
    with dash.app.test_request_context("/api/concentration-cap"):
        resp = raw()
    routed = resp.get_json() if hasattr(resp, "get_json") else json.loads(resp.data)
    for k in ("as_of", "cached", "cache_age_s"):
        routed.pop(k, None)
        direct.pop(k, None)
    assert routed == direct


# ─────────────────────── prewarm coverage (regression lock) ────────────────
def test_endpoint_listed_in_swr_prewarm_targets():
    src = inspect.getsource(dash._swr_prewarm)
    assert '("concentration_cap", concentration_cap_api)' in src
