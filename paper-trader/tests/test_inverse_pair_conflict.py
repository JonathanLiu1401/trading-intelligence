"""Unit tests for paper_trader.analytics.inverse_pair_conflict.

Locks the verdict ladder (NO_BOOK / CLEAN / OPPOSING_UNLEVERED /
CARRY_WASTE), the per-family delta math, severity bucketing, daily-drag
ballpark, and the never-raises / advisory contract. Pure builder ⇒ fast
exact-value asserts.
"""
from __future__ import annotations

from paper_trader.analytics.inverse_pair_conflict import (
    _PAIR_FAMILIES,
    _TICKER_INDEX,
    SEVERITY_HIGH_PCT,
    SEVERITY_MEDIUM_PCT,
    build_inverse_pair_conflict,
)


def _pos(ticker, qty, price, *, ptype="stock"):
    return {"ticker": ticker, "qty": qty, "current_price": price, "type": ptype}


# ─── verdict ladder ──────────────────────────────────────────────────────


def test_no_book_when_positions_empty():
    out = build_inverse_pair_conflict([])
    assert out["verdict"] == "NO_BOOK"
    assert out["state"] == "NO_BOOK"
    assert out["n_book_positions"] == 0
    assert out["conflicts"] == []
    assert out["n_conflicts"] == 0


def test_no_book_when_positions_none():
    out = build_inverse_pair_conflict(None)
    assert out["verdict"] == "NO_BOOK"


def test_clean_when_no_opposing_pairs():
    # Long-only leveraged book — no inverse, no conflict.
    positions = [
        _pos("TQQQ", 10, 100.0),
        _pos("SOXL", 5, 50.0),
        _pos("NVDA", 1, 200.0),  # non-family ticker
    ]
    out = build_inverse_pair_conflict(positions)
    assert out["verdict"] == "CLEAN"
    assert out["state"] == "READY"
    assert out["n_book_positions"] == 3
    assert out["n_conflicts"] == 0


def test_carry_waste_balanced_pair():
    # Equal-dollar TQQQ + SQQQ — fully cancelling deltas, both paying
    # the 3x leverage tab. Cancelled fraction = 100% of bigger side ⇒
    # HIGH severity.
    positions = [
        _pos("TQQQ", 10, 100.0),  # $1000
        _pos("SQQQ", 50, 20.0),    # $1000
    ]
    out = build_inverse_pair_conflict(positions)
    assert out["verdict"] == "CARRY_WASTE"
    assert out["n_conflicts"] == 1
    c = out["conflicts"][0]
    assert c["family"] == "QQQ"
    assert c["classification"] == "CARRY_WASTE"
    assert c["long_notional_usd"] == 1000.0
    assert c["inverse_notional_usd"] == 1000.0
    # Deltas: long = 1000 * 3 = 3000; inverse = 1000 * -3 = -3000.
    assert c["long_delta_usd"] == 3000.0
    assert c["inverse_delta_usd"] == -3000.0
    assert c["cancelled_delta_usd"] == 3000.0
    assert c["net_delta_usd"] == 0.0
    assert c["severity"] == "HIGH"
    # Daily drag ≈ 1000 * 6bps + 1000 * 6bps = $1.20/day.
    assert c["daily_drag_estimate_usd"] == 1.2


def test_carry_waste_lopsided_pair_medium_severity():
    # 4:1 imbalance ⇒ cancelled / max = 25% — should NOT be HIGH but
    # also NOT be MEDIUM by the >40% rule. Verify the LOW bucket fires.
    positions = [
        _pos("TQQQ", 40, 100.0),   # $4000  → delta +12000
        _pos("SQQQ", 10, 100.0),    # $1000  → delta -3000
    ]
    out = build_inverse_pair_conflict(positions)
    c = out["conflicts"][0]
    assert c["classification"] == "CARRY_WASTE"
    assert c["cancelled_delta_usd"] == 3000.0
    assert c["net_delta_usd"] == 9000.0
    # cancelled / max(long, inv) = 3000 / 12000 = 25% — below MEDIUM floor.
    assert c["severity"] == "LOW"


def test_carry_waste_two_to_one_imbalance_is_medium():
    # 2:1 ⇒ cancelled / max = 50% ⇒ MEDIUM.
    positions = [
        _pos("TQQQ", 20, 100.0),   # $2000 long_delta 6000
        _pos("SQQQ", 10, 100.0),    # $1000 inverse_delta -3000
    ]
    out = build_inverse_pair_conflict(positions)
    c = out["conflicts"][0]
    assert c["severity"] == "MEDIUM"


def test_opposing_unlevered_when_only_core_vs_inverse():
    # SPY + SPXS — 1x core vs -3x inverse. Directional offset, no
    # leveraged-long tab on the SPY sleeve ⇒ OPPOSING_UNLEVERED, NOT
    # CARRY_WASTE.
    positions = [
        _pos("SPY", 2, 500.0),   # $1000 core, delta +1000
        _pos("SPXS", 100, 10.0),  # $1000 inverse, delta -3000
    ]
    out = build_inverse_pair_conflict(positions)
    assert out["verdict"] == "OPPOSING_UNLEVERED"
    assert out["n_conflicts"] == 1
    c = out["conflicts"][0]
    assert c["classification"] == "OPPOSING_UNLEVERED"
    assert c["core_notional_usd"] == 1000.0
    assert c["inverse_notional_usd"] == 1000.0
    # Daily drag — only the inverse leveraged sleeve pays the tab.
    # 1000 * 6bps = $0.60/day.
    assert c["daily_drag_estimate_usd"] == 0.6


def test_carry_waste_takes_precedence_over_opposing_unlevered():
    # Both flavours present in different families — overall verdict
    # escalates to CARRY_WASTE.
    positions = [
        _pos("SPY", 2, 500.0),
        _pos("SPXS", 100, 10.0),
        _pos("TQQQ", 10, 100.0),
        _pos("SQQQ", 50, 20.0),
    ]
    out = build_inverse_pair_conflict(positions)
    assert out["verdict"] == "CARRY_WASTE"
    # Both conflicts surfaced.
    assert out["n_conflicts"] == 2
    families = {c["family"] for c in out["conflicts"]}
    assert families == {"QQQ", "SP500"}
    # The CARRY_WASTE conflict sorts before OPPOSING_UNLEVERED.
    assert out["conflicts"][0]["classification"] == "CARRY_WASTE"
    assert out["conflicts"][1]["classification"] == "OPPOSING_UNLEVERED"


def test_multi_family_carry_waste_aggregates():
    # Two leveraged inverse-pair conflicts (semis + S&P).
    positions = [
        _pos("SOXL", 1, 100.0),   # $100  delta +300
        _pos("SOXS", 1, 50.0),     # $50   delta -150
        _pos("SPXL", 1, 100.0),   # $100  delta +300
        _pos("SPXS", 1, 50.0),     # $50   delta -150
    ]
    out = build_inverse_pair_conflict(positions)
    assert out["verdict"] == "CARRY_WASTE"
    assert out["n_conflicts"] == 2
    # 150 cancelled per family × 2.
    assert out["total_cancelled_delta_usd"] == 300.0
    # Drag: per family ($100 + $50) × 6bps = $0.09 → 2 families = $0.18.
    assert out["total_daily_drag_usd"] == 0.18


def test_severity_thresholds_exposed_as_constants():
    # Locked module constants — used by chat enrichment and tests.
    assert SEVERITY_HIGH_PCT == 80.0
    assert SEVERITY_MEDIUM_PCT == 40.0


def test_taxonomy_consistency_no_ticker_in_two_roles():
    # A ticker must belong to exactly one (family, role) slot; the
    # _TICKER_INDEX is built from the families map so any drift here
    # would silently break delta accounting.
    seen: dict[str, tuple[str, str]] = {}
    for fam_key, fam in _PAIR_FAMILIES.items():
        for role in ("long", "inverse", "core"):
            for tkr, lev in fam[role].items():
                assert (fam[role][tkr] == lev), tkr  # leverage stable
                key = tkr
                assert key not in seen, f"{tkr} double-listed"
                seen[key] = (fam_key, role)
                # Sign discipline: long > 0, inverse < 0, core == 1.0.
                if role == "long":
                    assert lev > 1.0
                elif role == "inverse":
                    assert lev < 0
                else:
                    assert lev == 1.0
    # Index agrees with the families map.
    for tkr, (fam_key, role, lev) in _TICKER_INDEX.items():
        assert tkr in _PAIR_FAMILIES[fam_key][role]


# ─── _safe / never-raises discipline ─────────────────────────────────────


def test_garbage_rows_do_not_raise():
    positions = [
        None,
        {"ticker": None},
        {"ticker": "TQQQ", "qty": "x", "current_price": None},
        {"ticker": "", "qty": 1, "current_price": 1},
        _pos("SQQQ", 5, 100.0),
    ]
    out = build_inverse_pair_conflict(positions)
    # Only SQQQ has any value ⇒ no conflict (no opposing long-leveraged).
    assert out["verdict"] == "CLEAN"


def test_options_dropped_from_lookup():
    # An option on TQQQ doesn't count as a TQQQ stock position. So
    # holding TQQQ_CALL + SQQQ should read CLEAN (no stock-leveraged
    # opposing pair).
    positions = [
        _pos("TQQQ", 1, 5.0, ptype="call"),
        _pos("SQQQ", 5, 100.0),
    ]
    out = build_inverse_pair_conflict(positions)
    # Options skipped from the inverse-pair lookup; SQQQ alone has no
    # long sleeve in the QQQ family ⇒ CLEAN.
    assert out["verdict"] == "CLEAN"


def test_non_family_tickers_do_not_count():
    # Holding NVDA + SQQQ — NVDA isn't in the QQQ family taxonomy so
    # there's no conflict even though SQQQ shorts NVDA via lookthrough.
    # (Single-name correlation overlap is etf_lookthrough's job.)
    positions = [
        _pos("NVDA", 10, 200.0),
        _pos("SQQQ", 50, 20.0),
    ]
    out = build_inverse_pair_conflict(positions)
    assert out["verdict"] == "CLEAN"


def test_headline_contains_family_label_on_conflict():
    positions = [
        _pos("SOXL", 1, 100.0),
        _pos("SOXS", 1, 50.0),
    ]
    out = build_inverse_pair_conflict(positions)
    assert "Semis" in out["headline"]
    assert "CARRY_WASTE" in out["headline"]


def test_returns_safe_zeros_when_total_book_zero():
    # All-zero-priced positions degrade to no positions effective.
    positions = [
        _pos("TQQQ", 10, 0.0),
        _pos("SQQQ", 5, 0.0),
    ]
    out = build_inverse_pair_conflict(positions)
    assert out["verdict"] == "NO_BOOK"
