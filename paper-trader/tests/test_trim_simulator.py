"""Tests for the per-position trim simulator.

Two layers:

* **Builder** — ``paper_trader.analytics.trim_simulator.build_trim_simulator``
  pins the rung arithmetic, the scorer-EV math, the per-position verdict
  ladder (RECOMMEND_EXIT / RECOMMEND_TRIM / NEUTRAL / HOLD), the worst-
  first sort, the options ×100 multiplier, and the never-raises / NO_DATA
  degradation contract — the ``test_position_blowup.py`` pattern.

* **Endpoint** — ``/api/trim-simulator`` is a thin SWR wrapper that fans the
  scorer_predictions handler out so the ladder can carry EV math; the route
  tests cover passthrough, NO_DATA→200, scorer-failure degrade, builder-
  exception→500, and the prewarm registration invariant.

No live process, no network.
"""
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paper_trader.dashboard as dash
import paper_trader.analytics.trim_simulator as ts
from paper_trader.analytics.trim_simulator import (
    DEFAULT_RUNGS,
    build_trim_simulator,
)


# ─────────────────────────────── builder: degradation ──────────────────────
def test_no_positions_is_no_data():
    res = build_trim_simulator([], 1000.0)
    assert res["state"] == "NO_DATA"
    assert res["positions"] == []
    assert res["n_positions"] == 0
    assert "no priced book" in res["headline"]


def test_none_positions_is_no_data():
    res = build_trim_simulator(None, 1000.0)
    assert res["state"] == "NO_DATA"


def test_zero_total_value_is_no_data():
    """Positions present but tv=0 → can't compute weights → NO_DATA, never a
    ZeroDivisionError."""
    rows = [{"ticker": "NVDA", "current_price": 100.0, "qty": 1}]
    res = build_trim_simulator(rows, 0.0)
    assert res["state"] == "NO_DATA"


def test_all_rows_zero_value_is_no_data():
    """Every row prices to 0 → NO_DATA, not a phantom verdict."""
    rows = [{"ticker": "DEAD", "current_price": 0.0, "qty": 0},
            {"ticker": "NULL", "current_price": None, "qty": None}]
    res = build_trim_simulator(rows, 1000.0)
    assert res["state"] == "NO_DATA"
    assert res["positions"] == []


def test_builder_never_raises_on_malformed_rows():
    """A row with non-numeric fields contributes 0 and is skipped — it never
    sinks the whole result."""
    rows = [{"ticker": "JUNK", "current_price": "abc", "qty": "xyz"},
            {"ticker": "NVDA", "current_price": 200.0, "qty": 3}]
    res = build_trim_simulator(rows, 1000.0)
    assert res["n_positions"] == 1
    assert res["positions"][0]["ticker"] == "NVDA"


def test_garbage_total_value_is_no_data():
    rows = [{"ticker": "NVDA", "current_price": 100.0, "qty": 1}]
    assert build_trim_simulator(rows, "not-a-number")["state"] == "NO_DATA"


# ───────────────────────────── builder: arithmetic ─────────────────────────
def test_default_rungs_emit_quarter_half_threequarter():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]   # $800
    res = build_trim_simulator(rows, 1000.0)
    assert res["rungs_pct"] == [25.0, 50.0, 75.0]
    rungs = res["positions"][0]["rungs"]
    assert [r["trim_pct"] for r in rungs] == [25.0, 50.0, 75.0]


def test_shares_to_trim_is_exact_fraction_of_qty():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]   # 4 shares
    res = build_trim_simulator(rows, 1000.0)
    rungs = {r["trim_pct"]: r for r in res["positions"][0]["rungs"]}
    assert rungs[25.0]["shares_to_trim"] == 1.0
    assert rungs[50.0]["shares_to_trim"] == 2.0
    assert rungs[75.0]["shares_to_trim"] == 3.0


def test_cash_freed_matches_fraction_of_market_value():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]   # $800 MV
    res = build_trim_simulator(rows, 1000.0)
    rungs = {r["trim_pct"]: r for r in res["positions"][0]["rungs"]}
    assert rungs[25.0]["cash_freed_usd"] == 200.0
    assert rungs[50.0]["cash_freed_usd"] == 400.0
    assert rungs[75.0]["cash_freed_usd"] == 600.0


def test_new_weight_pct_drops_correctly():
    """$800 NVDA in $1000 book = 80%. Trim 50% → $400 left = 40% of book."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]
    res = build_trim_simulator(rows, 1000.0)
    rungs = {r["trim_pct"]: r for r in res["positions"][0]["rungs"]}
    # Pre-trim weight in the row.
    assert res["positions"][0]["current_weight_pct"] == 80.0
    assert rungs[25.0]["new_weight_pct"] == 60.0
    assert rungs[50.0]["new_weight_pct"] == 40.0
    assert rungs[75.0]["new_weight_pct"] == 20.0


def test_remaining_market_value_is_complement_of_cash_freed():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]   # $800
    res = build_trim_simulator(rows, 1000.0)
    for r in res["positions"][0]["rungs"]:
        assert r["remaining_market_value_usd"] + r["cash_freed_usd"] == 800.0


def test_options_multiplier_applied():
    """An option position contributes its true ×100 notional, mirroring
    the ``position_blowup`` precedent — so the trim ladder reflects the
    real risk, not the per-share premium."""
    rows = [{"ticker": "NVDA", "type": "call",
             "current_price": 5.0, "qty": 2}]   # 5 × 2 × 100 = $1000
    res = build_trim_simulator(rows, 1000.0)
    pos = res["positions"][0]
    assert pos["current_market_value_usd"] == 1000.0
    half = next(r for r in pos["rungs"] if r["trim_pct"] == 50.0)
    assert half["cash_freed_usd"] == 500.0


def test_market_value_field_preferred_over_price_times_qty():
    """When ``_mark_to_market`` has enriched the row, trust ``market_value``
    over recomputing — same SSOT precedent as ``position_blowup``."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4,
             "market_value": 777.0}]
    res = build_trim_simulator(rows, 1000.0)
    assert res["positions"][0]["current_market_value_usd"] == 777.0


# ───────────────────────────── scorer-EV math ──────────────────────────────
def test_negative_pred_yields_avoided_loss_only():
    """A bearish pred (-10% 5d) on a $800 position trimmed 50% means the
    freed $400 was expected to lose 10% = $40 over 5d. ``ev_avoided_loss_usd``
    is the *positive* magnitude of that avoided loss; ``ev_forgone_upside_usd``
    is None (we only forgo upside on positive preds)."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]
    preds = [{"ticker": "NVDA", "pred_5d_return_pct": -10.0, "verdict": "EXIT"}]
    res = build_trim_simulator(rows, 1000.0, preds)
    half = next(r for r in res["positions"][0]["rungs"] if r["trim_pct"] == 50.0)
    assert half["ev_avoided_loss_usd"] == 40.0
    assert half["ev_forgone_upside_usd"] is None
    # ev_freed_5d is the signed scorer-EV on the freed slice ($400 × -10% = -$40).
    assert half["ev_freed_5d_usd"] == -40.0
    # ev_kept_5d is the signed scorer-EV on the remaining slice ($400 × -10% = -$40).
    assert half["ev_kept_5d_usd"] == -40.0


def test_positive_pred_yields_forgone_upside_only():
    """A bullish pred (+10% 5d) on the same trim means the freed $400 was
    expected to gain $40 over 5d — ``ev_forgone_upside_usd`` = $40,
    ``ev_avoided_loss_usd`` = None."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]
    preds = [{"ticker": "NVDA", "pred_5d_return_pct": 10.0, "verdict": "STRONG_HOLD"}]
    res = build_trim_simulator(rows, 1000.0, preds)
    half = next(r for r in res["positions"][0]["rungs"] if r["trim_pct"] == 50.0)
    assert half["ev_forgone_upside_usd"] == 40.0
    assert half["ev_avoided_loss_usd"] is None


def test_missing_scorer_pred_yields_none_ev_fields():
    """No scorer for this ticker → every EV field is None; ladder math still
    populates the mechanical fields (shares/cash/weight)."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]
    res = build_trim_simulator(rows, 1000.0, scorer_predictions=[])
    pos = res["positions"][0]
    assert pos["scorer_pred_5d_pct"] is None
    half = next(r for r in pos["rungs"] if r["trim_pct"] == 50.0)
    assert half["ev_avoided_loss_usd"] is None
    assert half["ev_forgone_upside_usd"] is None
    assert half["ev_kept_5d_usd"] is None
    assert half["ev_freed_5d_usd"] is None
    # Mechanical fields populated as usual.
    assert half["cash_freed_usd"] == 400.0


def test_scorer_pred_indexed_case_insensitively():
    """Scorer rows might come back lowercase; the lookup uppercases both
    sides so a ``"nvda"`` pred still matches a ``"NVDA"`` position."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]
    preds = [{"ticker": "nvda", "pred_5d_return_pct": -10.0}]
    res = build_trim_simulator(rows, 1000.0, preds)
    assert res["positions"][0]["scorer_pred_5d_pct"] == -10.0


def test_scorer_pred_malformed_is_treated_as_missing():
    """A pred row with garbage ``pred_5d_return_pct`` degrades to no EV — the
    row never sinks the result."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]
    preds = [{"ticker": "NVDA", "pred_5d_return_pct": "abc"}]
    res = build_trim_simulator(rows, 1000.0, preds)
    half = next(r for r in res["positions"][0]["rungs"] if r["trim_pct"] == 50.0)
    assert half["ev_avoided_loss_usd"] is None
    assert half["ev_forgone_upside_usd"] is None


# ───────────────────────────── verdicts ────────────────────────────────────
def test_exit_verdict_when_bearish_pred_and_heavy_weight():
    """The live scenario: bearish pred (-22.6%) on a heavy position (NVDA
    65.7% of book) → RECOMMEND_EXIT, deepest rung recommended."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]   # 80% of $1000
    preds = [{"ticker": "NVDA", "pred_5d_return_pct": -22.6, "verdict": "EXIT"}]
    res = build_trim_simulator(rows, 1000.0, preds)
    pos = res["positions"][0]
    assert pos["verdict"] == "RECOMMEND_EXIT"
    # Recommended rung is the deepest (0.75 in the default ladder).
    assert pos["recommended_rung"]["trim_pct"] == 75.0
    assert res["state"] == "EXIT_RECOMMENDED"
    assert res["n_exit_recommended"] == 1


def test_trim_verdict_when_bearish_pred_alone():
    """Bearish pred (-12%) on a small position (5% of book) → still a TRIM —
    the scorer is shouting, weight is incidental."""
    rows = [{"ticker": "SOXL", "current_price": 50.0, "qty": 1}]    # 5% of $1000
    preds = [{"ticker": "SOXL", "pred_5d_return_pct": -12.0, "verdict": "EXIT"}]
    res = build_trim_simulator(rows, 1000.0, preds)
    pos = res["positions"][0]
    assert pos["verdict"] == "RECOMMEND_TRIM"
    # Recommended rung is the middle one (0.50 in the default 3-rung ladder).
    assert pos["recommended_rung"]["trim_pct"] == 50.0
    assert res["state"] == "TRIM_RECOMMENDED"


def test_trim_verdict_when_concentrated_no_scorer():
    """Heavy concentration (50% of book) with NO scorer signal → still flagged
    RECOMMEND_TRIM — concentration alone is decision-relevant. Weight-only
    triage path."""
    rows = [{"ticker": "NVDA", "current_price": 500.0, "qty": 1}]   # 50% of $1000
    res = build_trim_simulator(rows, 1000.0)
    pos = res["positions"][0]
    assert pos["verdict"] == "RECOMMEND_TRIM"
    assert res["state"] == "TRIM_RECOMMENDED"


def test_hold_verdict_when_bullish_pred():
    """Bullish pred (+5% 5d) on a non-concentrated position → HOLD."""
    rows = [{"ticker": "AMD", "current_price": 100.0, "qty": 1}]    # 10% of $1000
    preds = [{"ticker": "AMD", "pred_5d_return_pct": 5.0, "verdict": "STRONG_HOLD"}]
    res = build_trim_simulator(rows, 1000.0, preds)
    pos = res["positions"][0]
    assert pos["verdict"] == "HOLD"
    assert pos["recommended_rung"] is None
    assert res["state"] == "ALL_HOLD"


def test_neutral_verdict_when_flat_pred_small_weight():
    """Flat scorer (~0%) on a small position → NEUTRAL, no rec rung."""
    rows = [{"ticker": "AMD", "current_price": 100.0, "qty": 1}]    # 10%
    preds = [{"ticker": "AMD", "pred_5d_return_pct": 0.5, "verdict": "NEUTRAL"}]
    res = build_trim_simulator(rows, 1000.0, preds)
    pos = res["positions"][0]
    assert pos["verdict"] == "NEUTRAL"
    assert pos["recommended_rung"] is None
    assert res["state"] == "ALL_HOLD"


def test_exit_threshold_is_inclusive_at_minus_10():
    """A scorer pred of exactly -10% on a heavy position is still EXIT —
    the threshold is inclusive (``≤ -10.0``)."""
    rows = [{"ticker": "NVDA", "current_price": 400.0, "qty": 1}]   # 40% — heavy
    preds = [{"ticker": "NVDA", "pred_5d_return_pct": -10.0}]
    res = build_trim_simulator(rows, 1000.0, preds)
    assert res["positions"][0]["verdict"] == "RECOMMEND_EXIT"


# ───────────────────────────── ordering / headline ────────────────────────
def test_positions_sorted_by_verdict_urgency_then_weight():
    """EXIT first, then TRIM, then by descending weight — operator sees the
    biggest pending action first."""
    rows = [
        {"ticker": "SMALL_HOLD", "current_price": 50.0, "qty": 1},     # 5%
        {"ticker": "BIG_EXIT", "current_price": 200.0, "qty": 3},      # 60%
        {"ticker": "MID_TRIM", "current_price": 100.0, "qty": 2},      # 20%
    ]
    preds = [
        {"ticker": "SMALL_HOLD", "pred_5d_return_pct": 5.0},
        {"ticker": "BIG_EXIT", "pred_5d_return_pct": -15.0},
        {"ticker": "MID_TRIM", "pred_5d_return_pct": -7.0},
    ]
    res = build_trim_simulator(rows, 1000.0, preds)
    order = [p["ticker"] for p in res["positions"]]
    assert order == ["BIG_EXIT", "MID_TRIM", "SMALL_HOLD"]


def test_headline_includes_top_ticker_and_rung_when_trim_recommended():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]   # 80%
    preds = [{"ticker": "NVDA", "pred_5d_return_pct": -15.0}]
    res = build_trim_simulator(rows, 1000.0, preds)
    h = res["headline"]
    assert "NVDA" in h
    assert "trim" in h.lower()
    assert "%" in h


def test_headline_when_all_hold_mentions_no_pressure():
    rows = [{"ticker": "AMD", "current_price": 100.0, "qty": 1}]    # 10%
    preds = [{"ticker": "AMD", "pred_5d_return_pct": 5.0}]
    res = build_trim_simulator(rows, 1000.0, preds)
    assert "no scorer-driven trim pressure" in res["headline"]


# ───────────────────────────── rung cleaning ──────────────────────────────
def test_rungs_outside_unit_interval_are_dropped():
    """Rungs <=0 or >1 are silently dropped; an all-invalid set falls back
    to the default 0.25/0.50/0.75 ladder."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]
    res = build_trim_simulator(rows, 1000.0, rungs=(-0.5, 0.0, 1.5, 2.0))
    assert res["rungs_pct"] == [25.0, 50.0, 75.0]


def test_rungs_deduped_and_sorted():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]
    res = build_trim_simulator(rows, 1000.0, rungs=(0.5, 0.25, 0.5, 0.75))
    assert res["rungs_pct"] == [25.0, 50.0, 75.0]


def test_custom_rungs_respected():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 4}]
    res = build_trim_simulator(rows, 1000.0, rungs=(0.10, 0.40))
    assert res["rungs_pct"] == [10.0, 40.0]
    assert [r["trim_pct"] for r in res["positions"][0]["rungs"]] == [10.0, 40.0]


# ─────────────────────────── endpoint / route ──────────────────────────────
def _client():
    return dash.app.test_client()


_OK_PAYLOAD = {
    "as_of": "2026-05-22T06:00:00+00:00",
    "state": "EXIT_RECOMMENDED",
    "n_positions": 1,
    "total_value_usd": 1000.0,
    "rungs_pct": [25.0, 50.0, 75.0],
    "n_trim_recommended": 0,
    "n_exit_recommended": 1,
    "positions": [{
        "ticker": "NVDA", "type": "stock", "qty": 4.0,
        "current_market_value_usd": 800.0, "current_weight_pct": 80.0,
        "scorer_pred_5d_pct": -22.6, "scorer_verdict": "EXIT",
        "off_distribution": False,
        "rungs": [],
        "verdict": "RECOMMEND_EXIT",
        "recommended_rung": None,
    }],
    "headline": "Trim simulator (EXIT_RECOMMENDED): NVDA …",
}


def test_endpoint_passes_builder_payload_through(monkeypatch):
    """The route forks no logic — its body is the builder payload (a pure
    passthrough; it stamps no extra key)."""
    monkeypatch.setattr(ts, "build_trim_simulator",
                        lambda *a, **k: dict(_OK_PAYLOAD))
    resp = _client().get("/api/trim-simulator")
    assert resp.status_code == 200
    # Ignore SWR cache metadata added by @swr_cached.
    body = json.loads(resp.data)
    body.pop("cached", None)
    body.pop("cache_age_s", None)
    assert body == _OK_PAYLOAD


def test_endpoint_no_data_is_200_not_500(monkeypatch):
    """NO_DATA (no positions / zero tv) is a 200 with state=NO_DATA, never
    a 500. Matches the position-blowup precedent."""
    no_data = {"as_of": "2026-05-22T06:00:00+00:00", "state": "NO_DATA",
               "n_positions": 0, "total_value_usd": 0.0,
               "rungs_pct": [25.0, 50.0, 75.0], "positions": [],
               "n_trim_recommended": 0, "n_exit_recommended": 0,
               "headline": "Trim simulator: no priced book to simulate yet."}
    monkeypatch.setattr(ts, "build_trim_simulator",
                        lambda *a, **k: dict(no_data))
    resp = _client().get("/api/trim-simulator")
    assert resp.status_code == 200
    assert json.loads(resp.data)["state"] == "NO_DATA"


def test_endpoint_unexpected_exception_yields_500(monkeypatch):
    """A builder raise (contract violation) returns a shaped 500 not a 200
    with garbage — the position-blowup precedent."""
    def _boom(*a, **k):
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(ts, "build_trim_simulator", _boom)
    resp = _client().get("/api/trim-simulator")
    assert resp.status_code == 500
    assert "error" in json.loads(resp.data)


def test_endpoint_degrades_when_scorer_predictions_fails(monkeypatch):
    """If the scorer_predictions handler returns ``{"error": ...}`` the
    route still serves the trim simulator with no scorer EV (degrades, never
    surfaces an error). Pin the contract: an empty preds list reaches the
    builder, and the route returns 200."""
    captured = {}

    def _fake_builder(positions, total_value, preds, *a, **k):
        captured["preds"] = preds
        return {"as_of": "now", "state": "ALL_HOLD", "n_positions": 0,
                "total_value_usd": 0.0, "rungs_pct": [25.0, 50.0, 75.0],
                "positions": [], "n_trim_recommended": 0,
                "n_exit_recommended": 0, "headline": "ok"}

    class _FakeResp:
        def get_json(self):
            return {"error": "scorer offline"}

    monkeypatch.setattr(ts, "build_trim_simulator", _fake_builder)
    monkeypatch.setattr(dash, "scorer_predictions_api", lambda: _FakeResp())
    resp = _client().get("/api/trim-simulator")
    assert resp.status_code == 200
    assert captured["preds"] == []


def test_endpoint_ssot_parity_with_builder(monkeypatch):
    """The route reuses ``build_trim_simulator`` verbatim — no forked logic
    (AGENTS.md #10). Pinning the same store snapshot and same scorer-preds
    list, the unwrapped route handler produces the same body as the direct
    builder call (minus the per-call wall-clock ``as_of``).

    Bypasses the route's @swr_cached wrapper (via ``__wrapped__``) so the
    parity is deterministic; otherwise the SWR cache state for both
    ``trim_simulator`` and the in-handler ``scorer_predictions`` would race
    a prior test's warmed cache.
    """
    from paper_trader.store import get_store
    store = get_store()
    pf = store.get_portfolio()

    fixed_preds = [{
        "ticker": "NVDA", "pred_5d_return_pct": -12.0,
        "verdict": "EXIT", "off_distribution": False,
    }]

    class _FakeResp:
        def get_json(self):
            return {"predictions": list(fixed_preds)}

    monkeypatch.setattr(dash, "scorer_predictions_api", lambda: _FakeResp())

    direct = build_trim_simulator(
        store.open_positions(),
        float(pf.get("total_value") or 0.0),
        list(fixed_preds),
    )
    # Bypass @swr_cached by calling the raw handler (set on .__wrapped__ by
    # functools.wraps) inside a request context so flask.request works.
    raw = dash.trim_simulator_api.__wrapped__
    with dash.app.test_request_context("/api/trim-simulator"):
        resp = raw()
    routed = resp.get_json() if hasattr(resp, "get_json") else json.loads(resp.data)
    for k in ("as_of", "cached", "cache_age_s"):
        routed.pop(k, None)
        direct.pop(k, None)
    assert routed == direct


# ─────────────────────── prewarm coverage (regression lock) ────────────────
def test_endpoint_listed_in_swr_prewarm_targets():
    """``test_swr_prewarm_coverage`` is the canonical sweep, but pinning the
    exact ``trim_simulator`` name here means a typo in the prewarm tuple is
    caught locally too — same insurance the position_blowup tests carry."""
    src = inspect.getsource(dash._swr_prewarm)
    assert '("trim_simulator", trim_simulator_api)' in src
