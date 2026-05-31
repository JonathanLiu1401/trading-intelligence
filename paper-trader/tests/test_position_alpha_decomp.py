"""Exact-value tests for analytics/position_alpha_decomp.

Pins the per-position alpha = pos_return - beta * spy_return math, the options
×3-capped-at-4 / put-negation beta convention (shared with stress_scenarios),
the MAX_OPEN_LAG_S sample gate for the spy_at_open match, the market-value-
weighted aggregate verdict, and the INSUFFICIENT_SPY_DATA degrade path. Every
assertion targets a specific value, not "no crash" — the whole point of the
panel is the residual after stripping beta, and a regression to a simpler
"absolute return" would silently mis-label a beta-1.5 winner as positive alpha.
"""
from datetime import datetime, timedelta, timezone

from paper_trader.analytics.position_alpha_decomp import (
    ALPHA_BAND_PP,
    MAX_OPEN_LAG_S,
    build_position_alpha_decomp,
)

NOW = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)

# Toy sector/beta tables used by these tests — keeps the test independent
# of the live SECTOR_MAP / _LEVERAGE_BETA (which evolve). The shape is the
# only contract: classify(ticker) -> sector, beta_map[sector] -> beta.
_SECTORS = {"NVDA": "semis", "MU": "semis", "SPY": "broad", "SOXL": "semis_lev"}
_BETAS = {"semis": 1.5, "broad": 1.0, "semis_lev": 3.0, "other": 1.0}


def _classify(t: str) -> str:
    return _SECTORS.get(t, "other")


def _pos(ticker, *, qty=1.0, avg=100.0, cur=110.0, opened_min_ago=60,
         ptype="stock", expiry=None, strike=None):
    """One open-position dict (shape of store.open_positions)."""
    return {
        "ticker": ticker,
        "type": ptype,
        "qty": qty,
        "avg_cost": avg,
        "current_price": cur,
        "expiry": expiry,
        "strike": strike,
        "opened_at": (NOW - timedelta(minutes=opened_min_ago)).isoformat(),
    }


def _eq(minutes_ago, sp500, total=1000.0, cash=500.0):
    return {"timestamp": (NOW - timedelta(minutes=minutes_ago)).isoformat(),
            "total_value": total, "cash": cash, "sp500_price": sp500}


def test_no_positions_is_no_data():
    out = build_position_alpha_decomp([], [], _classify, _BETAS, now=NOW)
    assert out["state"] == "NO_DATA"
    assert out["verdict"] == "NO_DATA"
    assert out["n_positions"] == 0
    assert out["positions"] == []
    assert out["weighted_alpha_pp"] is None
    assert "no open positions" in out["headline"].lower()


def test_positive_alpha_when_beating_beta_implied_spy():
    # NVDA semis (beta 1.5), bought at 100, now 130 ⇒ +30%.
    # SPY moved from 5000 → 5100 over the hold ⇒ +2% SPY return.
    # Beta-implied: 1.5 × 2% = +3%. Alpha = 30 - 3 = +27pp ⇒ ALPHA_POS.
    pos = [_pos("NVDA", avg=100, cur=130, opened_min_ago=60)]
    eq = [_eq(60, 5000), _eq(0, 5100)]
    out = build_position_alpha_decomp(pos, eq, _classify, _BETAS, now=NOW)
    assert out["n_positions"] == 1
    assert out["n_judged"] == 1
    row = out["positions"][0]
    assert row["ticker"] == "NVDA"
    assert row["sector"] == "semis"
    assert row["beta_est"] == 1.5
    assert row["pos_return_pct"] == 30.0
    assert row["spy_return_pct"] == 2.0
    assert row["pure_beta_pct"] == 3.0
    assert row["alpha_pp"] == 27.0
    assert row["verdict"] == "ALPHA_POS"
    assert out["verdict"] == "ALPHA_ADDING"
    assert "+27.00pp" in out["headline"]


def test_negative_alpha_when_position_lags_beta_implied():
    # NVDA semis (beta 1.5), bought at 100, now 105 ⇒ +5%.
    # SPY +4% over hold ⇒ pure_beta = 6.0%. Alpha = 5 - 6 = -1pp ⇒ ALPHA_NEG.
    # This is the classic "celebrated as a winner but actually beta-lagging" case.
    pos = [_pos("NVDA", avg=100, cur=105, opened_min_ago=120)]
    eq = [_eq(120, 5000), _eq(0, 5200)]
    out = build_position_alpha_decomp(pos, eq, _classify, _BETAS, now=NOW)
    row = out["positions"][0]
    assert row["pos_return_pct"] == 5.0
    assert row["spy_return_pct"] == 4.0
    assert row["pure_beta_pct"] == 6.0
    assert row["alpha_pp"] == -1.0
    assert row["verdict"] == "ALPHA_NEG"
    assert out["verdict"] == "ALPHA_BLEEDING"


def test_pure_beta_verdict_within_band():
    # Position return exactly matches beta-implied → alpha 0pp → PURE_BETA.
    # SPY return 2%, beta 1.5, beta-implied 3%; position +3% ⇒ alpha 0.
    pos = [_pos("NVDA", avg=100, cur=103, opened_min_ago=60)]
    eq = [_eq(60, 5000), _eq(0, 5100)]
    out = build_position_alpha_decomp(pos, eq, _classify, _BETAS, now=NOW)
    row = out["positions"][0]
    assert row["alpha_pp"] == 0.0
    assert row["verdict"] == "PURE_BETA"
    assert out["verdict"] == "BETA_RIDING"


def test_alpha_band_boundary_is_pure_beta_not_alpha_pos():
    # Exactly +ALPHA_BAND_PP alpha — equality stays PURE_BETA (conservative
    # band logic, pinned so a >= → > slip can't flip a noise-floor reading).
    # Need pos_return - pure_beta = ALPHA_BAND_PP exactly.
    # beta 1.5, SPY +2% → pure_beta = 3.0. pos_return = 3.0 + 0.5 = 3.5.
    pos = [_pos("NVDA", avg=100, cur=103.5, opened_min_ago=60)]
    eq = [_eq(60, 5000), _eq(0, 5100)]
    out = build_position_alpha_decomp(pos, eq, _classify, _BETAS, now=NOW)
    row = out["positions"][0]
    assert row["alpha_pp"] == ALPHA_BAND_PP
    assert row["verdict"] == "PURE_BETA"


def test_put_option_beta_is_negated():
    # Puts negate beta (a -1.5 beta on a semis put) — verbatim contract from
    # stress_scenarios._position_betas. Beta 1.5 → option ×3 = 4.5, capped at
    # 4 → negated to -4 for a put.
    pos = [_pos("NVDA", avg=10.0, cur=15.0, opened_min_ago=60,
                ptype="put", expiry="2026-06-20", strike=120.0)]
    eq = [_eq(60, 5000), _eq(0, 5100)]  # SPY +2%
    out = build_position_alpha_decomp(pos, eq, _classify, _BETAS, now=NOW)
    row = out["positions"][0]
    assert row["beta_est"] == -4.0
    assert row["spy_return_pct"] == 2.0
    assert row["pure_beta_pct"] == -8.0     # -4 × 2%
    # Premium +50%; pure_beta -8 → alpha = 50 - (-8) = +58pp.
    assert row["pos_return_pct"] == 50.0
    assert row["alpha_pp"] == 58.0
    assert row["verdict"] == "ALPHA_POS"


def test_call_option_beta_caps_at_four():
    # Call: beta_map[semis]=1.5, ×3 = 4.5, cap = 4.0 (NOT 4.5).
    pos = [_pos("NVDA", avg=10.0, cur=12.0, opened_min_ago=60,
                ptype="call", expiry="2026-06-20", strike=130.0)]
    eq = [_eq(60, 5000), _eq(0, 5100)]
    out = build_position_alpha_decomp(pos, eq, _classify, _BETAS, now=NOW)
    row = out["positions"][0]
    assert row["beta_est"] == 4.0   # capped at 4, not 4.5
    assert row["pure_beta_pct"] == 8.0   # 4 × 2%
    # Premium +20%, pure_beta 8 → alpha = +12pp.
    assert row["alpha_pp"] == 12.0


def test_insufficient_when_no_equity_point_covers_opened_at():
    # Equity curve only starts AFTER the position was opened by >MAX_OPEN_LAG_S
    # — no usable baseline → INSUFFICIENT_SPY_DATA.
    # opened_min_ago=240 (4h); MAX_OPEN_LAG_S/60 = 120 min. So a curve that
    # starts only 60 min ago has a 180-min gap → exceeds the lag tolerance.
    pos = [_pos("NVDA", avg=100, cur=110, opened_min_ago=240)]
    eq = [_eq(60, 5000), _eq(0, 5100)]
    out = build_position_alpha_decomp(pos, eq, _classify, _BETAS, now=NOW)
    row = out["positions"][0]
    assert row["verdict"] == "INSUFFICIENT_SPY_DATA"
    assert row["spy_at_open"] is None
    assert row["alpha_pp"] is None
    assert out["state"] == "NO_DATA"
    assert out["verdict"] == "NO_DATA"
    assert out["n_judged"] == 0
    assert "insufficient" in out["headline"].lower()


def test_max_open_lag_exactly_at_boundary_is_accepted():
    # opened exactly MAX_OPEN_LAG_S before the first eq point — accepted
    # (the bound is inclusive; pinned so a < vs <= slip can't change behavior).
    lag_min = MAX_OPEN_LAG_S / 60.0  # 120 minutes
    # Position opened at NOW - lag_min - 1 min; first eq point at NOW - 1 min.
    pos = [_pos("NVDA", avg=100, cur=110,
                opened_min_ago=lag_min + 1)]
    eq = [_eq(1, 5000), _eq(0, 5100)]
    out = build_position_alpha_decomp(pos, eq, _classify, _BETAS, now=NOW)
    row = out["positions"][0]
    # Bound is inclusive (lag == MAX_OPEN_LAG_S exactly).
    assert row["spy_at_open"] == 5000.0
    assert row["spy_open_lag_s"] == MAX_OPEN_LAG_S
    assert row["verdict"] in ("ALPHA_POS", "ALPHA_NEG", "PURE_BETA")


def test_first_eq_point_at_or_after_opened_at_is_chosen():
    # Two equity points after opened_at — the FIRST (closest to open) is the
    # spy_at_open anchor, NOT the latest one (which would zero out the spread
    # and dishonestly hide the move).
    pos = [_pos("NVDA", avg=100, cur=120, opened_min_ago=60)]
    eq = [
        _eq(80, 4900),   # BEFORE opened_at — must be skipped
        _eq(55, 5000),   # first at-or-after opened_at — this one
        _eq(30, 5050),
        _eq(0, 5100),
    ]
    out = build_position_alpha_decomp(pos, eq, _classify, _BETAS, now=NOW)
    row = out["positions"][0]
    assert row["spy_at_open"] == 5000.0   # NOT 4900 (before) or 5050/5100
    assert row["spy_return_pct"] == 2.0
    # pos_return 20%, beta 1.5, pure_beta 3% → alpha 17pp
    assert row["alpha_pp"] == 17.0


def test_weighted_aggregate_alpha_uses_market_value_weights():
    # Two positions with different notionals — aggregate alpha weights by
    # market value, NOT a naive mean. Big position dominates.
    # P1: NVDA qty=10 cur=110 → MV=1100 (large), pos_return +10%, beta 1.5,
    #     SPY +2% → pure 3% → alpha +7pp
    # P2: MU qty=1 cur=110 → MV=110 (small), same +10% return, alpha +7pp
    # Equal alpha so weighted average is +7pp regardless — flip the small
    # one's return to test the weighting actually engaged.
    p1 = _pos("NVDA", qty=10, avg=100, cur=110, opened_min_ago=60)
    p2 = _pos("MU", qty=1, avg=100, cur=90, opened_min_ago=60)
    eq = [_eq(60, 5000), _eq(0, 5100)]  # SPY +2%, pure beta 3%
    out = build_position_alpha_decomp([p1, p2], eq, _classify, _BETAS,
                                      now=NOW)
    # P1 alpha = 10 - 3 = +7; P2 alpha = -10 - 3 = -13.
    # MV uses current_price: P1 = 110×10 = 1100; P2 = 90×1 = 90 (NOT avg_cost).
    # Weighted alpha = (7×1100 + -13×90) / (1100+90) = 6530 / 1190 ≈ 5.4874
    assert out["positions"][0]["ticker"] == "NVDA"  # largest MV first
    assert out["positions"][1]["ticker"] == "MU"
    expected = round((7.0 * 1100 + -13.0 * 90) / (1100 + 90), 4)
    assert out["weighted_alpha_pp"] == expected
    assert out["verdict"] == "ALPHA_ADDING"


def test_garbage_position_does_not_raise():
    # Garbage rows shouldn't crash; they degrade to INSUFFICIENT_SPY_DATA.
    bad = [{"ticker": "X", "type": "stock", "qty": "not-a-number",
            "avg_cost": None, "current_price": None,
            "opened_at": "not-a-date"}]
    out = build_position_alpha_decomp(bad, [_eq(60, 5000), _eq(0, 5100)],
                                      _classify, _BETAS, now=NOW)
    assert out["n_positions"] == 1
    assert out["n_judged"] == 0
    assert out["positions"][0]["verdict"] == "INSUFFICIENT_SPY_DATA"
    # Aggregate degrades to NO_DATA, not an exception.
    assert out["state"] == "NO_DATA"


def test_zero_avg_cost_does_not_divide_by_zero():
    # avg_cost = 0 → pos_return undefined → INSUFFICIENT, never ZeroDivision.
    pos = [_pos("NVDA", avg=0, cur=110, opened_min_ago=60)]
    eq = [_eq(60, 5000), _eq(0, 5100)]
    out = build_position_alpha_decomp(pos, eq, _classify, _BETAS, now=NOW)
    assert out["positions"][0]["verdict"] == "INSUFFICIENT_SPY_DATA"
    assert out["positions"][0]["alpha_pp"] is None


def test_unknown_sector_defaults_to_beta_one():
    # Unknown ticker → classify returns "other"; beta_map["other"]=1.0.
    pos = [_pos("ZZZUNKNOWN", avg=100, cur=110, opened_min_ago=60)]
    eq = [_eq(60, 5000), _eq(0, 5100)]
    out = build_position_alpha_decomp(pos, eq, _classify, _BETAS, now=NOW)
    row = out["positions"][0]
    assert row["sector"] == "other"
    assert row["beta_est"] == 1.0
    # pos +10%, SPY +2%, pure beta 2% → alpha +8pp.
    assert row["alpha_pp"] == 8.0


def test_judged_rows_ordered_before_insufficient_by_market_value():
    # One judged big-MV position, one insufficient small-MV position. Judged
    # must appear first regardless of MV-vs-insufficient comparison.
    p_judged = _pos("NVDA", qty=10, avg=100, cur=110, opened_min_ago=60)
    p_insuf = _pos("MU", qty=1, avg=100, cur=110, opened_min_ago=240)
    eq = [_eq(60, 5000), _eq(0, 5100)]
    out = build_position_alpha_decomp([p_insuf, p_judged], eq, _classify,
                                      _BETAS, now=NOW)
    assert [r["ticker"] for r in out["positions"]] == ["NVDA", "MU"]
    assert out["positions"][0]["verdict"] == "ALPHA_POS"
    assert out["positions"][1]["verdict"] == "INSUFFICIENT_SPY_DATA"
