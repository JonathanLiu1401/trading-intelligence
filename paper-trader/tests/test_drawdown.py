"""Real-logic tests for analytics/drawdown.py + the /api/drawdown invariant-#12 lock.

`drawdown.py` had no dedicated test file. These exercise the actual peak/
trough/recovery math with hand-computed expected values (not "does it run"),
the empty-curve fallback, the contributor decomposition, and — the regression
lock for this review's fix — that `/api/drawdown` threads the module
`INITIAL_CASH` rather than the builder's hardcoded 1000.0 default
(the `benchmark_api`/`analytics_api` single-source-of-truth pattern,
invariant #12). The endpoint test FAILS against the pre-fix code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.drawdown import compute_drawdown


def _eq(points):
    """points: list of (ts, total_value) → equity_curve row dicts."""
    return [{"timestamp": ts, "total_value": tv, "cash": 0.0,
             "sp500_price": None} for ts, tv in points]


# ── empty-curve fallback honours the PASSED starting_equity (the bug) ──────
def test_empty_curve_uses_passed_starting_equity_not_1000():
    r = compute_drawdown([], [], starting_equity=2000.0)
    # Pre-fix the endpoint never passed this and the builder's 1000.0 default
    # leaked; the builder itself must echo whatever it is given.
    assert r["starting_equity"] == 2000.0
    assert r["current_value"] == 2000.0
    assert r["peak_value"] == 2000.0
    assert r["trough_value"] == 2000.0
    assert r["drawdown_pct"] == 0.0
    assert r["at_high_water"] is True
    assert r["history"] == []


def test_empty_curve_default_is_still_1000():
    # The default exists for back-compat; the fix is that the *endpoint* must
    # not rely on it. The builder default itself stays 1000.0.
    assert compute_drawdown([], [])["starting_equity"] == 1000.0


# ── peak / trough / recovery math (hand-computed) ─────────────────────────
def test_peak_then_drop_then_partial_recovery():
    eq = _eq([
        ("2026-05-17T00:00:00+00:00", 1000.0),
        ("2026-05-17T01:00:00+00:00", 1200.0),  # all-time peak
        ("2026-05-17T02:00:00+00:00", 900.0),   # trough since peak
        ("2026-05-17T03:00:00+00:00", 1050.0),  # current (partial recovery)
    ])
    r = compute_drawdown(eq, [], starting_equity=1000.0)
    assert r["current_value"] == 1050.0
    assert r["peak_value"] == 1200.0
    assert r["peak_ts"] == "2026-05-17T01:00:00+00:00"
    assert r["drawdown_abs"] == -150.0          # 1050 - 1200
    assert r["drawdown_pct"] == -12.5           # -150 / 1200 * 100
    assert r["trough_value"] == 900.0
    assert r["trough_pct"] == -25.0             # -300 / 1200 * 100
    # recovered (1050-900)/(1200-900) = 150/300 = 50%
    assert r["recovery_pct"] == 50.0
    assert r["at_high_water"] is False


def test_monotonic_rising_is_at_high_water_zero_dd():
    eq = _eq([
        ("2026-05-17T00:00:00+00:00", 1000.0),
        ("2026-05-17T01:00:00+00:00", 1100.0),
        ("2026-05-17T02:00:00+00:00", 1200.0),
    ])
    r = compute_drawdown(eq, [], starting_equity=1000.0)
    assert r["at_high_water"] is True
    assert r["drawdown_pct"] == 0.0
    assert r["drawdown_abs"] == 0.0
    assert r["peak_value"] == 1200.0
    # trough == peak ⇒ recovery stays 0.0 (no drawdown to recover)
    assert r["recovery_pct"] == 0.0


def test_trough_is_lowest_AFTER_the_most_recent_peak_not_global_min():
    # A deep early dip BEFORE a later all-time peak must NOT count as the
    # current-drawdown trough — the trough resets on every new peak.
    eq = _eq([
        ("2026-05-17T00:00:00+00:00", 1000.0),
        ("2026-05-17T01:00:00+00:00", 500.0),   # deep early dip
        ("2026-05-17T02:00:00+00:00", 2000.0),  # new all-time peak (resets trough)
        ("2026-05-17T03:00:00+00:00", 1800.0),  # mild pullback
    ])
    r = compute_drawdown(eq, [], starting_equity=1000.0)
    assert r["peak_value"] == 2000.0
    assert r["trough_value"] == 1800.0          # NOT 500
    assert r["drawdown_pct"] == -10.0           # (1800-2000)/2000*100


def test_at_high_water_boundary_within_1bp():
    # -0.01% is the documented at-high-water tolerance (rounding noise).
    eq = _eq([
        ("2026-05-17T00:00:00+00:00", 100000.0),
        ("2026-05-17T01:00:00+00:00", 99999.99),  # -1e-3 % ≈ within 1bp
    ])
    r = compute_drawdown(eq, [], starting_equity=1000.0)
    assert r["at_high_water"] is True


# ── per-position contribution decomposition ───────────────────────────────
def test_contributors_sorted_most_negative_first_and_skip_closed():
    positions = [
        {"ticker": "A", "type": "stock", "qty": 1, "avg_cost": 100.0,
         "current_price": 90.0, "unrealized_pl": -10.0},
        {"ticker": "B", "type": "stock", "qty": 2, "avg_cost": 50.0,
         "current_price": 60.0, "unrealized_pl": 20.0},
        {"ticker": "C", "type": "stock", "qty": 0, "avg_cost": 10.0,
         "current_price": 9.0, "unrealized_pl": -1.0},   # qty<=0 → skipped
        {"ticker": "D", "type": "stock", "qty": 1, "avg_cost": 100.0,
         "current_price": 70.0, "unrealized_pl": -30.0},
    ]
    r = compute_drawdown(_eq([("2026-05-17T00:00:00+00:00", 1000.0)]),
                         positions, starting_equity=1000.0)
    c = r["contributors"]
    assert [x["ticker"] for x in c] == ["D", "A", "B"]   # most-negative first
    assert c[0]["pl_pct"] == -30.0    # -30 / (100*1) * 100
    assert c[0]["drag"] is True
    assert c[0]["cost_basis"] == 100.0
    assert c[2]["ticker"] == "B" and c[2]["drag"] is False
    assert c[2]["pl_pct"] == 20.0     # 20 / (50*2) * 100


def test_contributor_zero_cost_basis_no_zero_division():
    positions = [{"ticker": "X", "type": "stock", "qty": 1, "avg_cost": 0.0,
                  "current_price": 5.0, "unrealized_pl": 5.0}]
    r = compute_drawdown(_eq([("2026-05-17T00:00:00+00:00", 1000.0)]),
                         positions, starting_equity=1000.0)
    assert r["contributors"][0]["pl_pct"] == 0.0   # guarded, not ZeroDivision


# ── history down-sample: tail is always pinned (no data loss at the end) ───
def test_history_downsampled_and_last_point_pinned():
    pts = [(f"2026-05-17T{h:02d}:{m:02d}:00+00:00", 1000.0 + i)
           for i, (h, m) in enumerate(
               (i // 60, i % 60) for i in range(500))]
    r = compute_drawdown(_eq(pts), [], starting_equity=1000.0)
    # Down-sampled (fewer than the 500 raw points) ...
    assert len(r["history"]) < 500
    # ... but the most recent point is never dropped (a chart that loses its
    # tail mis-states the *current* value).
    assert r["history"][-1]["v"] == round(pts[-1][1], 2)


# ── /api/drawdown threads INITIAL_CASH (the regression lock) ──────────────
def test_endpoint_threads_initial_cash_not_literal_1000(tmp_path, monkeypatch):
    import paper_trader.store as store_mod
    import paper_trader.dashboard as d
    from paper_trader.store import Store

    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    # dashboard did `from .store import INITIAL_CASH` → it has its own binding.
    monkeypatch.setattr(d, "INITIAL_CASH", 2500.0)

    s = Store()                       # fresh book: equity_curve is empty
    try:
        d.app.config["TESTING"] = True
        with d.app.test_client() as client:
            resp = client.get("/api/drawdown")
            assert resp.status_code == 200
            body = resp.get_json()
        # Pre-fix: builder's 1000.0 default leaked → these would be 1000.0.
        assert body["starting_equity"] == 2500.0
        assert body["peak_value"] == 2500.0
        assert body["current_value"] == 2500.0
        assert body["at_high_water"] is True
    finally:
        s.close()
        store_mod._singleton = None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
