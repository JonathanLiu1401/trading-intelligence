"""Behaviour lock for paper_trader.analytics.deployment_plan and the
/api/deployment-plan Flask endpoint.

These tests assert the exact ranking, sizing, and constraint behaviour
of the multi-trade allocator. They would fail if:

  * an already-held name leaked into the plan,
  * a NEUTRAL / TRIM / EXIT verdict were ever sized,
  * a sub-floor ``pred_5d_return_pct`` were ever sized,
  * the per-name cap were ignored when Kelly target exceeds it,
  * the per-leverage cap let the plan stack 100% into +3x ETFs,
  * the implied-book preview drifted from the actual plan sum,
  * insufficient-cash / empty-slate / all-gated states slipped through
    as READY,
  * the response constraints echo failed to clamp absurd query params.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.deployment_plan import (
    build_deployment_plan,
    DEFAULT_MIN_ALLOC_USD,
)


def _opp(ticker, pred, verdict="STRONG_HOLD"):
    return {"ticker": ticker, "pred_5d_return_pct": pred, "verdict": verdict}


def _pos(ticker, qty, price):
    return {"ticker": ticker, "type": "stock", "qty": qty,
            "current_price": price, "market_value": qty * price}


# ───────────── Verdict ladder ───────────────────────────────────────────

def test_empty_slate_yields_no_opportunities():
    out = build_deployment_plan(
        opportunities=[], positions=[], cash_usd=1000.0,
        total_value=1000.0, kelly_pct=20.0,
    )
    assert out["verdict"] == "NO_OPPORTUNITIES"
    assert out["plan"] == []
    assert out["n_plan"] == 0


def test_insufficient_cash_yields_insufficient_cash():
    out = build_deployment_plan(
        opportunities=[_opp("AAPL", 5.0)], positions=[],
        cash_usd=5.0, total_value=5.0, kelly_pct=20.0,
        min_alloc_usd=50.0,
    )
    assert out["verdict"] == "INSUFFICIENT_CASH"
    assert out["plan"] == []


def test_ready_when_anything_sizes():
    out = build_deployment_plan(
        opportunities=[_opp("AAPL", 8.0)], positions=[],
        cash_usd=1000.0, total_value=1000.0, kelly_pct=20.0,
    )
    assert out["verdict"] == "READY"
    assert out["n_plan"] == 1
    assert out["plan"][0]["ticker"] == "AAPL"
    assert out["plan"][0]["alloc_usd"] >= DEFAULT_MIN_ALLOC_USD


# ───────────── Filter behaviour ─────────────────────────────────────────

def test_skips_already_held():
    out = build_deployment_plan(
        opportunities=[_opp("AAPL", 8.0), _opp("MSFT", 5.0)],
        positions=[_pos("AAPL", 1, 100)],
        cash_usd=1000.0, total_value=1100.0, kelly_pct=20.0,
    )
    plan_tickers = [t["ticker"] for t in out["plan"]]
    assert "AAPL" not in plan_tickers
    assert "MSFT" in plan_tickers
    skipped = [s["ticker"] for s in out["skipped"]]
    assert "AAPL" in skipped


def test_skips_non_buy_verdicts():
    out = build_deployment_plan(
        opportunities=[_opp("AAPL", 8.0, "STRONG_HOLD"),
                       _opp("MSFT", 6.0, "NEUTRAL"),
                       _opp("GOOG", 5.0, "TRIM"),
                       _opp("AMZN", 4.0, "HOLD")],
        positions=[], cash_usd=2000.0, total_value=2000.0,
        kelly_pct=15.0,
    )
    plan_tickers = [t["ticker"] for t in out["plan"]]
    assert "AAPL" in plan_tickers
    assert "AMZN" in plan_tickers
    assert "MSFT" not in plan_tickers
    assert "GOOG" not in plan_tickers


def test_skips_below_pred_floor():
    out = build_deployment_plan(
        opportunities=[_opp("AAPL", 8.0), _opp("LOW1", 0.5), _opp("LOW2", -1.0)],
        positions=[], cash_usd=1000.0, total_value=1000.0,
        kelly_pct=20.0, min_pred_pct=1.0,
    )
    plan_tickers = [t["ticker"] for t in out["plan"]]
    assert "AAPL" in plan_tickers
    assert "LOW1" not in plan_tickers
    assert "LOW2" not in plan_tickers


# ───────────── Ranking & sizing ─────────────────────────────────────────

def test_plan_ranked_by_pred_descending():
    # All three test tickers map to sector "other"; relax the sector
    # cap so it doesn't drop the third candidate before the rank
    # check (the cap is asserted separately).
    out = build_deployment_plan(
        opportunities=[_opp("LOW", 2.0), _opp("HIGH", 12.0), _opp("MID", 6.0)],
        positions=[], cash_usd=1000.0, total_value=1000.0,
        kelly_pct=15.0, per_sector_cap_pct=100.0,
    )
    plan_tickers = [t["ticker"] for t in out["plan"]]
    assert plan_tickers[0] == "HIGH"
    assert plan_tickers.index("MID") < plan_tickers.index("LOW")


def test_per_name_cap_enforced():
    # Kelly target 50% of book * edge-mult > 25% per-name cap.
    out = build_deployment_plan(
        opportunities=[_opp("AAPL", 20.0)], positions=[],
        cash_usd=1000.0, total_value=1000.0,
        kelly_pct=50.0, per_name_cap_pct=25.0,
    )
    assert out["plan"][0]["alloc_usd"] == pytest.approx(250.0, abs=1.0)


def test_reserve_cash_pct_held_back():
    # 30% reserve out of $1000 = $700 deployable; with 1 candidate at
    # kelly=50% per-name=100% the planner takes the full deployable.
    out = build_deployment_plan(
        opportunities=[_opp("AAPL", 20.0)], positions=[],
        cash_usd=1000.0, total_value=1000.0,
        kelly_pct=50.0, per_name_cap_pct=100.0,
        reserve_cash_pct=30.0,
    )
    assert out["deployable_usd"] == 700.0
    assert sum(t["alloc_usd"] for t in out["plan"]) <= 700.0


def test_leverage_cap_blocks_3x_stacking():
    # All candidates are leveraged ETFs. With a 30% leverage cap and
    # kelly=20%, the planner should not let leveraged dollars exceed
    # $300 of a $1000 book.
    out = build_deployment_plan(
        opportunities=[_opp("SOXL", 20.0), _opp("TQQQ", 15.0),
                       _opp("NAIL", 12.0), _opp("DPST", 10.0)],
        positions=[], cash_usd=1000.0, total_value=1000.0,
        kelly_pct=20.0, leveraged_cap_pct=30.0,
        per_name_cap_pct=25.0,
    )
    lev_total = sum(t["alloc_usd"] for t in out["plan"] if t["is_leveraged"])
    assert lev_total <= 300.0 + 0.01
    # At least one of the four should have been rejected by the cap.
    rej_reasons = " ".join(r["reason"] for r in out["rejected_by_constraint"])
    assert "leveraged cap" in rej_reasons


def test_sector_cap_blocks_over_concentration():
    # Two semis names with a tight 10% sector cap on a $1000 book =
    # at most $100 of semis exposure total.
    out = build_deployment_plan(
        opportunities=[_opp("NVDA", 15.0), _opp("AMD", 10.0),
                       _opp("AAPL", 8.0)],  # AAPL is tech, not semis
        positions=[], cash_usd=1000.0, total_value=1000.0,
        kelly_pct=20.0, per_sector_cap_pct=10.0,
        per_name_cap_pct=25.0,
    )
    semis_total = sum(t["alloc_usd"] for t in out["plan"]
                      if t["sector"] == "semis")
    assert semis_total <= 100.0 + 0.01


# ───────────── Implied book preview ─────────────────────────────────────

def test_implied_book_math_matches_plan():
    out = build_deployment_plan(
        opportunities=[_opp("AAPL", 8.0), _opp("MSFT", 6.0)],
        positions=[], cash_usd=1000.0, total_value=1000.0,
        kelly_pct=20.0, per_name_cap_pct=25.0,
    )
    deployed = sum(t["alloc_usd"] for t in out["plan"])
    assert out["deployed_usd"] == pytest.approx(deployed, abs=0.01)
    assert out["implied_book"]["post_cash_usd"] == pytest.approx(
        1000.0 - deployed, abs=0.01
    )
    # Blended pred matches alloc-weighted average.
    if out["plan"]:
        expected_blend = sum(
            t["pred_5d_return_pct"] * t["alloc_usd"] for t in out["plan"]
        ) / max(deployed, 1e-9)
        assert out["implied_book"]["blended_pred_5d_return_pct"] == pytest.approx(
            expected_blend, abs=0.01
        )


def test_max_trades_caps_plan_length():
    opps = [_opp(f"T{i}", 5.0) for i in range(12)]
    out = build_deployment_plan(
        opportunities=opps, positions=[],
        cash_usd=10000.0, total_value=10000.0,
        kelly_pct=10.0, per_name_cap_pct=10.0,
        max_trades=3,
    )
    assert len(out["plan"]) == 3


# ───────────── Constraint clamping ──────────────────────────────────────

def test_constraints_echoed_under_caps():
    # Absurd inputs must clamp into the documented bands.
    out = build_deployment_plan(
        opportunities=[_opp("AAPL", 5.0)], positions=[],
        cash_usd=1000.0, total_value=1000.0, kelly_pct=10.0,
        reserve_cash_pct=999.0,   # → clamped to 100
        per_name_cap_pct=-50.0,   # → clamped to 1
        min_pred_pct=999.0,       # → clamped to MAX_PRED_PCT_FLOOR
    )
    eff = out["constraints"]
    assert eff["reserve_cash_pct"] == 100.0
    assert eff["per_name_cap_pct"] == 1.0
    assert eff["min_pred_pct"] <= 50.0


# ───────────── Robustness ───────────────────────────────────────────────

def test_garbage_opportunity_rows_dont_raise():
    out = build_deployment_plan(
        opportunities=[
            {"ticker": None, "pred_5d_return_pct": 5.0, "verdict": "STRONG_HOLD"},
            {"ticker": "AAPL", "pred_5d_return_pct": "abc", "verdict": "STRONG_HOLD"},
            _opp("MSFT", 5.0),
        ],
        positions=[], cash_usd=1000.0, total_value=1000.0,
        kelly_pct=10.0,
    )
    # The MSFT row must still survive and size.
    plan_tickers = [t["ticker"] for t in out["plan"]]
    assert "MSFT" in plan_tickers


# ───────────── Flask endpoint wiring ────────────────────────────────────

@pytest.fixture
def stub_client(tmp_path, monkeypatch):
    from paper_trader import store as store_mod
    from paper_trader.store import Store

    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()  # all-cash $1000 default

    from paper_trader import dashboard
    monkeypatch.setattr(dashboard, "get_store", lambda: s)

    monkeypatch.setattr(dashboard, "_scorer_opportunities_inline", lambda: [
        {"ticker": "NVDA", "pred_5d_return_pct": 8.0, "verdict": "STRONG_HOLD"},
        {"ticker": "AAPL", "pred_5d_return_pct": 5.0, "verdict": "STRONG_HOLD"},
        {"ticker": "GOOG", "pred_5d_return_pct": 0.2, "verdict": "NEUTRAL"},
    ])
    monkeypatch.setattr(dashboard, "_spy_regime_label", lambda: ("bull", 5.0))

    dashboard.app.config.update(TESTING=True)
    return dashboard.app.test_client()


def test_endpoint_returns_plan_with_expected_shape(stub_client):
    r = stub_client.get("/api/deployment-plan")
    assert r.status_code == 200
    body = r.get_json()
    for key in ("as_of", "verdict", "headline", "cash_available_usd",
                "deployable_usd", "deployed_usd", "n_plan",
                "plan", "skipped", "rejected_by_constraint",
                "constraints", "implied_book", "regime",
                "n_opportunities_input"):
        assert key in body, f"missing {key}"
    # NEUTRAL verdict must be skipped.
    plan_tickers = [t["ticker"] for t in body["plan"]]
    assert "GOOG" not in plan_tickers
    assert body["regime"] == "bull"
    assert body["n_opportunities_input"] == 3


def test_endpoint_respects_query_params(stub_client):
    # max_trades=1 with two surviving candidates → only the top one.
    r = stub_client.get("/api/deployment-plan?max_trades=1")
    body = r.get_json()
    assert body["n_plan"] == 1
    # Top survivor by pred = NVDA (8.0 > 5.0).
    assert body["plan"][0]["ticker"] == "NVDA"
