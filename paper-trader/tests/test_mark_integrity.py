"""Tests for analytics/mark_integrity.py — mark-trust meta-metric.

`stale_mark` is surfaced per-position to Opus and Discord, but nothing
answers "what fraction of the displayed book value is fictional right
now?". A stale book makes every P/L panel partially false. These lock the
exact arithmetic against the live shape observed 2026-05-17 (MU stale @
cost while LITE marks live), the verdict thresholds, and the no-divide-by-
zero / never-raises contracts.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.mark_integrity import (
    UNTRUSTWORTHY_PCT,
    build_mark_integrity,
)


def _pos(ticker, qty, avg, cur, stale, ptype="stock"):
    mult = 100 if ptype in ("call", "put") else 1
    return {
        "ticker": ticker, "type": ptype, "qty": qty, "avg_cost": avg,
        "current_price": cur, "unrealized_pl": (cur - avg) * qty * mult,
        "market_value": cur * qty * mult, "stale_mark": stale,
    }


class TestVerdict:
    def test_no_positions_is_no_data(self):
        r = build_mark_integrity([])
        assert r["verdict"] == "NO_DATA"
        assert r["n_positions"] == 0
        assert r["stale_value_pct"] is None

    def test_all_live_is_clean(self):
        r = build_mark_integrity([
            _pos("NVDA", 2, 100.0, 110.0, False),
            _pos("AMD", 3, 50.0, 48.0, False),
        ])
        assert r["verdict"] == "CLEAN"
        assert r["n_stale"] == 0
        assert r["stale_value_usd"] == 0.0
        assert r["stale_value_pct"] == 0.0
        assert r["stale_tickers"] == []

    def test_live_mu_stale_shape_is_degraded_exact(self):
        """The exact 2026-05-17 live book: MU marked at cost (stale), LITE
        marks live. 37.95% of gross is fictional → DEGRADED, not CLEAN."""
        r = build_mark_integrity([
            # MU 0.5 @ 724.12, no live price → marked at cost, P/L $0.00
            _pos("MU", 0.5, 724.12, 724.12, True),
            # LITE 0.61 @ 980.90, marks 970.71 live
            _pos("LITE", 0.61, 980.90, 970.71, False),
        ])
        assert r["n_positions"] == 2
        assert r["n_stale"] == 1
        assert r["stale_tickers"] == ["MU"]
        assert r["stale_value_usd"] == 362.06          # 724.12 * 0.5
        assert r["gross_value_usd"] == 954.19           # 362.06 + 592.1331
        # 362.06 / 954.1931 * 100 = 37.9446 → 37.94 (pct off the raw gross)
        assert r["stale_value_pct"] == 37.94
        assert r["verdict"] == "DEGRADED"

    def test_majority_stale_is_untrustworthy(self):
        """>= UNTRUSTWORTHY_PCT of book value at cost ⇒ every panel's P/L is
        substantially fictional — the actionable 'do not trust the numbers'
        state."""
        r = build_mark_integrity([
            _pos("MU", 1.0, 700.0, 700.0, True),     # 700 stale
            _pos("LITE", 1.0, 300.0, 300.0, False),  # 300 live
        ])
        assert r["stale_value_pct"] == 70.0
        assert r["stale_value_pct"] >= UNTRUSTWORTHY_PCT
        assert r["verdict"] == "UNTRUSTWORTHY"

    def test_boundary_exactly_threshold_is_untrustworthy(self):
        r = build_mark_integrity([
            _pos("A", 1.0, 50.0, 50.0, True),
            _pos("B", 1.0, 50.0, 50.0, False),
        ])
        assert r["stale_value_pct"] == 50.0 == UNTRUSTWORTHY_PCT
        assert r["verdict"] == "UNTRUSTWORTHY"


class TestEdges:
    def test_zero_gross_value_no_divide_by_zero(self):
        """A worthless-marked stale book (gross 0) must not raise — pct is
        honestly None and the verdict still flags the stale marks."""
        r = build_mark_integrity([_pos("DEAD", 1.0, 0.0, 0.0, True)])
        assert r["gross_value_usd"] == 0.0
        assert r["stale_value_pct"] is None
        assert r["n_stale"] == 1
        assert r["verdict"] == "DEGRADED"

    def test_option_multiplier_in_value(self):
        r = build_mark_integrity([
            _pos("NVDA", 1.0, 2.0, 2.0, True, ptype="call"),  # 2*1*100=200
        ])
        assert r["gross_value_usd"] == 200.0
        assert r["stale_value_usd"] == 200.0

    def test_never_raises_on_garbage_rows(self):
        r = build_mark_integrity([
            {"ticker": "X"},                       # missing everything
            {"market_value": None, "stale_mark": True},
            {"stale_mark": "yes"},                  # truthy non-bool
        ])
        assert "verdict" in r
        assert isinstance(r["n_positions"], int)

    def test_headline_is_a_nonempty_string(self):
        for rows in ([], [_pos("A", 1, 1.0, 1.0, False)],
                     [_pos("A", 1, 1.0, 1.0, True)]):
            assert isinstance(build_mark_integrity(rows)["headline"], str)
            assert build_mark_integrity(rows)["headline"]
