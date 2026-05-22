"""Tests for the per-position single-name blow-up ladder.

Two layers:

* **Builder** — ``paper_trader.analytics.position_blowup.build_position_blowup``
  shipped with a CLI but **no test coverage at all**. These tests pin the
  exact shock arithmetic, the concentration-aware verdict ladder, the
  worst-first sort, the options ×100 multiplier, and the never-raises /
  NO_DATA degradation contract.

* **Endpoint** — ``/api/position-blowup`` was added to surface the builder
  on the dashboard (the recurring "no operator can see it" gap). The route
  is a *pure passthrough* — unlike ``/api/per-ticker-skill`` it stamps no
  extra key (the builder already sets ``as_of``) — so the parity assertion
  is byte-identity, not byte-identity-minus-``as_of``.

No live process, no network.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paper_trader.dashboard as dash
import paper_trader.analytics.position_blowup as pb
from paper_trader.analytics.position_blowup import build_position_blowup


# ───────────────────────────── builder: degradation ────────────────────────
def test_no_positions_is_no_data():
    res = build_position_blowup([], 1000.0)
    assert res["state"] == "NO_DATA"
    assert res["positions"] == []
    assert res["n_positions"] == 0
    assert "no priced book" in res["headline"]


def test_none_positions_is_no_data():
    res = build_position_blowup(None, 1000.0)
    assert res["state"] == "NO_DATA"


def test_zero_total_value_is_no_data():
    """A book with positions but zero total value can't be weighted —
    NO_DATA, never a ZeroDivisionError."""
    res = build_position_blowup(
        [{"ticker": "NVDA", "current_price": 100.0, "qty": 1}], 0.0)
    assert res["state"] == "NO_DATA"


def test_garbage_total_value_is_no_data():
    res = build_position_blowup(
        [{"ticker": "NVDA", "current_price": 100.0, "qty": 1}], "not-a-number")
    assert res["state"] == "NO_DATA"


def test_all_rows_zero_value_is_no_data():
    """Every row prices to 0 (closed mid-cycle / garbage qty) — the result
    is NO_DATA, not a CONCENTRATED verdict over phantom rows."""
    rows = [{"ticker": "NVDA", "current_price": 0.0, "qty": 0},
            {"ticker": "AMD", "current_price": None, "qty": None}]
    res = build_position_blowup(rows, 1000.0)
    assert res["state"] == "NO_DATA"
    assert res["positions"] == []


# ───────────────────────────── builder: arithmetic ─────────────────────────
def test_shock_arithmetic_is_exact():
    """A $600 single name in a $1000 book: −25 % loses $150 (15 % of book),
    −50 % loses $300, to-zero loses $600."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 3}]  # $600
    res = build_position_blowup(rows, 1000.0)
    pos = res["positions"][0]
    assert pos["market_value_usd"] == 600.0
    assert pos["weight_pct"] == 60.0
    by_mag = {s["shock_pct"]: s for s in pos["shocks"]}
    assert by_mag[-10.0]["pnl_usd"] == -60.0
    assert by_mag[-10.0]["pnl_pct_of_book"] == -6.0
    assert by_mag[-25.0]["pnl_usd"] == -150.0
    assert by_mag[-25.0]["pnl_pct_of_book"] == -15.0
    assert by_mag[-50.0]["pnl_usd"] == -300.0
    assert by_mag[-100.0]["pnl_usd"] == -600.0


def test_max_loss_equals_negative_market_value():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 3}]
    pos = build_position_blowup(rows, 1000.0)["positions"][0]
    assert pos["max_loss_usd"] == -600.0
    assert pos["max_loss_pct_of_book"] == -60.0


def test_shock_magnitudes_are_the_four_rungs():
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 3}]
    res = build_position_blowup(rows, 1000.0)
    assert res["shock_magnitudes_pct"] == [-10.0, -25.0, -50.0, -100.0]
    assert [s["shock_pct"] for s in res["positions"][0]["shocks"]] == \
        [-10.0, -25.0, -50.0, -100.0]


def test_options_multiplied_by_100():
    """An option contributes its true ×100 notional, not the per-share
    premium (the stress_scenarios._position_betas precedent)."""
    rows = [{"ticker": "NVDA", "type": "call", "current_price": 5.0, "qty": 2}]
    # 5.0 × 2 × 100 = $1000 notional
    pos = build_position_blowup(rows, 1000.0)["positions"][0]
    assert pos["market_value_usd"] == 1000.0


def test_market_value_field_preferred_over_price_times_qty():
    """When ``_mark_to_market`` has enriched the row with ``market_value``
    the builder trusts it over recomputing current_price×qty."""
    rows = [{"ticker": "NVDA", "current_price": 200.0, "qty": 3,
             "market_value": 777.0}]
    pos = build_position_blowup(rows, 1000.0)["positions"][0]
    assert pos["market_value_usd"] == 777.0


def test_zero_value_row_skipped_others_kept():
    """A garbage zero-value row is dropped silently; the real lot survives
    and the headline counts only real positions."""
    rows = [{"ticker": "DEAD", "current_price": 0.0, "qty": 0},
            {"ticker": "NVDA", "current_price": 200.0, "qty": 3}]
    res = build_position_blowup(rows, 1000.0)
    assert res["n_positions"] == 1
    assert res["positions"][0]["ticker"] == "NVDA"


# ───────────────────────────── builder: verdicts ───────────────────────────
def test_concentrated_verdict_when_one_name_over_40pct():
    """A single name worth >40 % of the book → CONCENTRATED (a single-name
    surprise is the dominant tail)."""
    rows = [{"ticker": "NVDA", "current_price": 500.0, "qty": 1}]  # $500 = 50%
    res = build_position_blowup(rows, 1000.0)
    assert res["state"] == "CONCENTRATED"
    assert "NVDA" in res["headline"]


def test_moderate_verdict_between_20_and_40pct():
    rows = [{"ticker": "NVDA", "current_price": 300.0, "qty": 1},   # 30%
            {"ticker": "AMD", "current_price": 150.0, "qty": 1}]    # 15%
    res = build_position_blowup(rows, 1000.0)
    assert res["state"] == "MODERATE"


def test_diffuse_verdict_when_no_name_over_20pct():
    """Ten equal names at 8 % each — no single-name surprise loses >20 %
    of the book → DIFFUSE."""
    rows = [{"ticker": f"T{i}", "current_price": 80.0, "qty": 1}
            for i in range(10)]
    res = build_position_blowup(rows, 1000.0)
    assert res["state"] == "DIFFUSE"


def test_verdict_threshold_is_inclusive_at_40():
    """Exactly 40 % to-zero loss → CONCENTRATED (the >= boundary)."""
    rows = [{"ticker": "NVDA", "current_price": 400.0, "qty": 1}]   # 40%
    res = build_position_blowup(rows, 1000.0)
    assert res["state"] == "CONCENTRATED"


# ───────────────────────────── builder: ordering ───────────────────────────
def test_rows_sorted_worst_first():
    """Positions are ordered most-damaging-first by max_loss_usd so the
    operator's first read is the biggest single-name risk."""
    rows = [{"ticker": "SMALL", "current_price": 100.0, "qty": 1},
            {"ticker": "BIG", "current_price": 600.0, "qty": 1},
            {"ticker": "MID", "current_price": 300.0, "qty": 1}]
    res = build_position_blowup(rows, 1000.0)
    assert [p["ticker"] for p in res["positions"]] == ["BIG", "MID", "SMALL"]
    assert res["headline"].startswith("Position blow-up (CONCENTRATED): BIG")


def test_builder_never_raises_on_malformed_rows():
    """A row with non-numeric fields contributes 0 and is skipped — it
    never sinks the whole result."""
    rows = [{"ticker": "JUNK", "current_price": "abc", "qty": "xyz"},
            {"ticker": "NVDA", "current_price": 200.0, "qty": 3}]
    res = build_position_blowup(rows, 1000.0)
    assert res["n_positions"] == 1
    assert res["positions"][0]["ticker"] == "NVDA"


# ───────────────────────────── endpoint: HTTP ──────────────────────────────
_OK_PAYLOAD = {
    "as_of": "2026-05-22T06:00:00+00:00",
    "state": "CONCENTRATED",
    "n_positions": 1,
    "total_value_usd": 1000.0,
    "shock_magnitudes_pct": [-10.0, -25.0, -50.0, -100.0],
    "positions": [
        {"ticker": "NVDA", "type": "stock", "market_value_usd": 658.5,
         "weight_pct": 65.8, "max_loss_usd": -658.5,
         "max_loss_pct_of_book": -65.8,
         "shocks": [{"shock_pct": -10.0, "pnl_usd": -65.85,
                     "pnl_pct_of_book": -6.58}]},
    ],
    "headline": "Position blow-up (CONCENTRATED): NVDA …",
}


def _client():
    return dash.app.test_client()


def test_endpoint_passes_builder_payload_through(monkeypatch):
    """The route forks no logic — its body is byte-identical to the builder
    payload (a pure passthrough; it stamps no extra key)."""
    monkeypatch.setattr(pb, "build_position_blowup",
                        lambda *a, **k: dict(_OK_PAYLOAD))
    resp = _client().get("/api/position-blowup")
    assert resp.status_code == 200
    assert json.loads(resp.data) == _OK_PAYLOAD


def test_endpoint_no_data_is_200_not_500(monkeypatch):
    """The builder degrades to NO_DATA (never raises); the route must
    surface that at HTTP 200."""
    no_data = {"as_of": "2026-05-22T06:00:00+00:00", "state": "NO_DATA",
               "n_positions": 0, "total_value_usd": 0.0,
               "shock_magnitudes_pct": [-10.0, -25.0, -50.0, -100.0],
               "positions": [], "headline": "no priced book to shock yet."}
    monkeypatch.setattr(pb, "build_position_blowup",
                        lambda *a, **k: dict(no_data))
    resp = _client().get("/api/position-blowup")
    assert resp.status_code == 200
    assert json.loads(resp.data)["state"] == "NO_DATA"


def test_endpoint_unexpected_exception_yields_500(monkeypatch):
    """If the builder raises (a contract violation) the route's defensive
    guard returns a shaped 500 rather than crashing the worker."""
    def _boom(*a, **k):
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(pb, "build_position_blowup", _boom)
    resp = _client().get("/api/position-blowup")
    assert resp.status_code == 500
    assert "error" in json.loads(resp.data)


def test_endpoint_ssot_parity_with_builder():
    """The route fed by the live store produces exactly what calling
    ``build_position_blowup`` on the same store snapshot produces — no
    forked logic (AGENTS.md #10)."""
    from paper_trader.store import get_store
    store = get_store()
    pf = store.get_portfolio()
    direct = build_position_blowup(
        store.open_positions(), float(pf.get("total_value") or 0.0))
    resp = _client().get("/api/position-blowup")
    assert resp.status_code == 200
    routed = json.loads(resp.data)
    # ``as_of`` is a wall-clock stamp set independently in each call — drop
    # it; every other key must match byte-for-byte.
    routed.pop("as_of", None)
    direct.pop("as_of", None)
    assert routed == direct
