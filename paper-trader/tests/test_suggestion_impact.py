"""/api/suggestion-impact — per-trade portfolio projection layered on top of /api/suggestions.

The pin set is small but load-bearing:

  * pure builder: no DB, no network — testable in isolation from Flask
  * BUY adds default 5% sizing, projects cash burn + post-trade concentration
  * BUY is cash-capped (cash_constrained flag fires when default sizing
    exceeds available cash; sized_usd shrinks to cash)
  * EXIT projects 100% liquidation, cash+proceeds, realized P/L at current
    price (current - avg_cost) × qty, AND the post-trade concentration drop
  * TRIM defaults to 50% (matches _SUGGESTION_TRIM_FRACTION constant)
  * SSOT: the projected severity comes from the same _concentration_severity
    helper /api/risk uses — anything that changes the LOW/MEDIUM/HIGH bucket
    thresholds in one place must update both reads consistently
  * would_overconcentrate fires when a BUY pushes severity LOW → ≥MEDIUM
  * frees_concentration fires when an EXIT pushes severity ≥MEDIUM → LOW
  * HOLD / WATCH pass through with would_act=false (no projection fields)
  * total/total_value=0 (or empty book) collapses gracefully — never raises
  * the Flask route exists and clamps ``size_pct`` to 0..100
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_trader.dashboard import (
    app,
    build_suggestion_impact,
    _concentration_severity,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _row(ticker, market_value, *, avg_cost=100.0, current_price=110.0, qty=1.0):
    return {
        "ticker": ticker,
        "type": "stock",
        "sector": "semis",
        "market_value": market_value,
        "avg_cost": avg_cost,
        "current_price": current_price,
        "qty": qty,
        "multiplier": 1,
    }


def _sug(ticker, action, *, price=100.0, held_qty=0.0):
    return {
        "ticker": ticker,
        "action": action,
        "price": price,
        "held_qty": held_qty,
        "conviction": 0.7,
        "news_max_score": 8.0,
        "news_urgent": True,
        "reasons": ["test"],
    }


# ── pure-builder shape ───────────────────────────────────────────────────────


def test_empty_inputs_collapse_to_well_formed_envelope():
    out = build_suggestion_impact([], [], cash=0.0, total_value=0.0)
    assert out["cards"] == []
    assert out["n_cards"] == 0
    assert out["baseline_top1_pct"] == 0.0
    assert out["baseline_severity"] == "LOW"
    assert out["default_size_pct"] == 5.0
    assert "as_of" in out


def test_non_list_suggestions_does_not_raise():
    for bad in (None, "string", {"x": 1}, 42):
        out = build_suggestion_impact(bad, [], cash=100, total_value=100)  # type: ignore[arg-type]
        assert out["cards"] == []


def test_size_pct_param_overrides_default_dollar_sizing():
    out = build_suggestion_impact(
        [_sug("MU", "BUY", price=50.0)],
        [],
        cash=1000.0,
        total_value=1000.0,
        size_pct=10.0,
    )
    card = out["cards"][0]
    # 10% of $1000 = $100 → qty = 2.0
    assert card["projected_size_usd"] == 100.0
    assert card["projected_qty"] == 2.0


# ── BUY: sizing, cash burn, post-trade concentration ────────────────────────


def test_buy_uses_default_5pct_sizing_when_cash_sufficient():
    out = build_suggestion_impact(
        [_sug("MU", "BUY", price=50.0)],
        [_row("NVDA", 500.0)],
        cash=500.0,
        total_value=1000.0,  # $500 NVDA + $500 cash
    )
    card = out["cards"][0]
    assert card["action"] == "BUY"
    assert card["would_act"] is True
    # 5% of $1000 = $50; ample cash; not constrained.
    assert card["projected_size_usd"] == 50.0
    assert card["cash_constrained"] is False
    # qty = 50 / 50 = 1.0
    assert card["projected_qty"] == 1.0
    # cash 500 → 450
    assert card["projected_cash_after"] == 450.0


def test_buy_caps_at_available_cash_and_flags_constraint():
    """5% of book = $100 but only $20 cash on hand → sized_usd = $20."""
    out = build_suggestion_impact(
        [_sug("MU", "BUY", price=50.0)],
        [_row("NVDA", 1980.0)],
        cash=20.0,
        total_value=2000.0,
    )
    card = out["cards"][0]
    # min($100, $20) = $20
    assert card["projected_size_usd"] == 20.0
    assert card["cash_constrained"] is True
    assert card["projected_cash_after"] == 0.0


def test_buy_post_trade_concentration_uses_same_severity_taxonomy_as_risk():
    """A BUY that pushes top1_pct past the 40% MEDIUM threshold must flip
    severity LOW → MEDIUM via the SAME _concentration_severity helper /api/risk
    uses — single taxonomy, no drift between the two reads."""
    out = build_suggestion_impact(
        [_sug("MU", "BUY", price=100.0)],
        [_row("NVDA", 350.0), _row("MU", 250.0, qty=2.5)],
        cash=1000.0,
        total_value=1600.0,  # +$1000 cash → top1 baseline = 350/1600 = 21.9%
        size_pct=30.0,
    )
    card = out["cards"][0]
    # 30% sizing = $480 → projected MU = 250 + 480 = $730 = 45.6% > 40% MEDIUM
    assert card["baseline_severity"] == "LOW"
    assert card["projected_severity_after"] == "MEDIUM"
    assert card["would_overconcentrate"] is True


def test_buy_into_new_ticker_creates_new_row_in_projection():
    out = build_suggestion_impact(
        [_sug("AMD", "BUY", price=200.0)],
        [_row("NVDA", 100.0)],
        cash=500.0,
        total_value=600.0,
        size_pct=50.0,
    )
    card = out["cards"][0]
    # 50% sizing = $300 → AMD = $300 = 50% of $600 → severity MEDIUM
    assert card["projected_severity_after"] == "MEDIUM"
    assert card["projected_top1_ticker_after"] == "AMD"


# ── EXIT / TRIM: proceeds, P/L, frees_concentration ─────────────────────────


def test_exit_projects_full_liquidation_and_realized_pnl():
    out = build_suggestion_impact(
        [_sug("NVDA", "EXIT", price=110.0, held_qty=10.0)],
        [_row("NVDA", 1100.0, avg_cost=100.0, current_price=110.0, qty=10.0)],
        cash=100.0,
        total_value=1200.0,
    )
    card = out["cards"][0]
    # Full sell of $1100 NVDA at avg 100 / current 110 → realized = 10 × 10 = $100
    assert card["projected_proceeds_usd"] == 1100.0
    assert card["projected_realized_pnl_usd"] == 100.0
    assert card["projected_cash_after"] == 1200.0  # 100 + 1100
    assert card["projected_position_pct_after"] == 0.0


def test_trim_uses_default_50_percent_fraction():
    out = build_suggestion_impact(
        [_sug("NVDA", "TRIM", price=110.0, held_qty=10.0)],
        [_row("NVDA", 1100.0, avg_cost=100.0, current_price=110.0, qty=10.0)],
        cash=100.0,
        total_value=1200.0,
    )
    card = out["cards"][0]
    # 50% trim of $1100 → $550 proceeds, $50 realized P/L (5 shares × $10 gain)
    assert card["projected_proceeds_usd"] == 550.0
    assert card["projected_realized_pnl_usd"] == 50.0
    assert card["projected_cash_after"] == 650.0


def test_exit_frees_concentration_flag_fires_when_high_drops_to_low():
    """EXIT on a 70%-of-book concentrated position must trip frees_concentration."""
    out = build_suggestion_impact(
        [_sug("NVDA", "EXIT", price=100.0, held_qty=7.0)],
        [_row("NVDA", 700.0, avg_cost=100.0, current_price=100.0, qty=7.0),
         _row("MU", 100.0),
         _row("AMD", 100.0)],
        cash=100.0,
        total_value=1000.0,
    )
    card = out["cards"][0]
    # Baseline top1 NVDA = 70% → HIGH; after EXIT, top1 falls to MU 10% → LOW
    assert card["baseline_severity"] == "HIGH"
    assert card["projected_severity_after"] == "LOW"
    assert card["frees_concentration"] is True


def test_realized_loss_is_negative():
    out = build_suggestion_impact(
        [_sug("NVDA", "EXIT", price=80.0, held_qty=10.0)],
        [_row("NVDA", 800.0, avg_cost=100.0, current_price=80.0, qty=10.0)],
        cash=100.0,
        total_value=900.0,
    )
    card = out["cards"][0]
    # 10 shares × ($80 - $100) = -$200
    assert card["projected_realized_pnl_usd"] == -200.0


# ── HOLD / WATCH: passthrough, no projection fields ─────────────────────────


def test_hold_passes_through_with_would_act_false():
    out = build_suggestion_impact(
        [_sug("NVDA", "HOLD", price=100.0, held_qty=1.0)],
        [_row("NVDA", 100.0)],
        cash=100.0,
        total_value=200.0,
    )
    card = out["cards"][0]
    assert card["would_act"] is False
    assert "projected_size_usd" not in card
    assert "projected_proceeds_usd" not in card
    # baseline still surfaces (the trader still wants to see what they hold)
    assert card["baseline_position_pct"] == 50.0


def test_watch_passes_through_with_would_act_false():
    out = build_suggestion_impact(
        [_sug("AMD", "WATCH", price=100.0)],
        [],
        cash=100.0,
        total_value=100.0,
    )
    card = out["cards"][0]
    assert card["would_act"] is False
    assert "projected_size_usd" not in card


# ── SSOT: severity taxonomy must match /api/risk's _concentration_severity ──


def test_severity_taxonomy_is_shared_with_risk_endpoint():
    """Source proof: the BUY case must use _concentration_severity for its
    projected_severity_after — if a copy-pasted threshold table appeared
    here, that severity would drift from /api/risk's display. Pin via an
    end-to-end check: the same inputs through both paths agree."""
    rows = [_row("NVDA", 450.0), _row("MU", 150.0)]
    total = 1000.0
    # Build a BUY of $0 (no change) — projected severity must equal baseline.
    out = build_suggestion_impact(
        [_sug("MU", "BUY", price=10.0)],
        rows,
        cash=1000.0,
        total_value=total,
        size_pct=0.0,
    )
    card = out["cards"][0]
    # Sanity: with 0% sizing, projected concentration == baseline.
    assert card["projected_top1_pct_after"] == card["baseline_top1_pct"]
    assert card["projected_severity_after"] == card["baseline_severity"]
    # And the severity matches the canonical helper directly.
    expected, _ = _concentration_severity(
        card["baseline_top1_pct"], card["baseline_top3_pct"]
    )
    assert card["baseline_severity"] == expected


# ── Flask route surface ─────────────────────────────────────────────────────


def test_route_exists_and_returns_json():
    client = app.test_client()
    r = client.get("/api/suggestion-impact")
    assert r.status_code == 200
    data = r.get_json()
    # Whether or not /api/suggestions returns rows in this sandbox, the
    # envelope must always be well-formed (cards key present, never error
    # by default).
    assert "cards" in data
    assert isinstance(data["cards"], list)


def test_route_clamps_size_pct():
    client = app.test_client()
    r = client.get("/api/suggestion-impact?size_pct=999")
    assert r.status_code == 200
    data = r.get_json()
    if "default_size_pct" in data:
        assert data["default_size_pct"] == 100.0
    r = client.get("/api/suggestion-impact?size_pct=-50")
    assert r.status_code == 200
    data = r.get_json()
    if "default_size_pct" in data:
        assert data["default_size_pct"] == 0.0


def test_route_tolerates_garbage_size_pct():
    client = app.test_client()
    r = client.get("/api/suggestion-impact?size_pct=abc")
    assert r.status_code == 200
    data = r.get_json()
    if "default_size_pct" in data:
        # Garbage → falls back to 5.0 default.
        assert data["default_size_pct"] == 5.0
