"""Unit tests for analytics.spy_valuation.build_spy_valuation.

Pins the title parser against the digital-intern collector's exact format,
the CAPE→regime band classification, the staleness threshold, and the
NO_DATA / PARSE_FAILED degrade-soft contracts.

A wrong CAPE parse would silently mis-label the macro backdrop; these
assertions catch that deterministically.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.spy_valuation import build_spy_valuation

# A real title shape captured live from articles.db on 2026-05-25 — pinning
# this directly catches collector-format drift on the next run.
_LIVE_TITLE = ("S&P 500 valuation: Extreme Overvaluation — "
               "CAPE 42.04 (2.4x historical mean (95% of 44.19 all-time peak)), "
               "P/E 32.19")

_FIXED_NOW = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)


def _art(title=_LIVE_TITLE, first_seen=None, url="internal://market_valuation/2026-05-25/extreme_overvalued"):
    return {
        "title": title,
        "url": url,
        "first_seen": first_seen or "2026-05-25 00:25:22",
        "published": None,
    }


class TestParserAgainstLiveFormat:
    def test_live_title_parses_cape_and_pe(self):
        out = build_spy_valuation(_art(), now=_FIXED_NOW)
        assert out["state"] == "REGIME_READ"
        assert out["cape"] == 42.04
        assert out["pe"] == 32.19
        assert out["regime"] == "EXTREME_OVERVALUED"
        assert out["regime_label"] == "Extreme Overvaluation"
        assert "Extreme Overvaluation" in out["headline"]
        assert "42.04" in out["headline"]
        # CAPE 42.04 / mean 17.38 ≈ 2.42×
        assert out["cape_vs_mean"] == pytest.approx(42.04 / 17.38, rel=1e-3)
        # 42.04 / 44.19 ≈ 95.1% of peak
        assert out["cape_pct_of_peak"] == pytest.approx(42.04 / 44.19 * 100, rel=1e-3)

    def test_title_without_pe_still_parses_cape(self):
        # CAPE 22 lands cleanly in FAIR_VALUE (band: 20 ≤ x < 30).
        title = "S&P 500 valuation: Fair Value — CAPE 22.0 (1.3x mean)"
        out = build_spy_valuation(_art(title=title), now=_FIXED_NOW)
        assert out["state"] == "REGIME_READ"
        assert out["cape"] == 22.0
        assert out["pe"] is None
        assert out["regime"] == "FAIR_VALUE"


class TestRegimeBands:
    @pytest.mark.parametrize("cape,expected_regime", [
        (45.0, "EXTREME_OVERVALUED"),
        (40.0, "EXTREME_OVERVALUED"),    # boundary inclusive
        (39.99, "EXPENSIVE"),
        (30.0, "EXPENSIVE"),
        (29.99, "FAIR_VALUE"),
        (20.0, "FAIR_VALUE"),
        (19.99, "UNDERVALUED"),
        (15.0, "UNDERVALUED"),
        (14.99, "DEEPLY_UNDERVALUED"),
        (8.0,  "DEEPLY_UNDERVALUED"),
    ])
    def test_band_classification(self, cape, expected_regime):
        title = f"S&P 500 valuation: X — CAPE {cape:.2f} (foo), P/E 20.0"
        out = build_spy_valuation(_art(title=title), now=_FIXED_NOW)
        assert out["regime"] == expected_regime


class TestStaleness:
    def test_fresh_article_not_stale(self):
        fresh = (_FIXED_NOW - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")
        out = build_spy_valuation(_art(first_seen=fresh), now=_FIXED_NOW)
        assert out["stale"] is False
        assert out["article_age_hours"] == pytest.approx(4.0, rel=1e-3)

    def test_old_article_marked_stale(self):
        old = (_FIXED_NOW - timedelta(hours=72)).strftime("%Y-%m-%d %H:%M:%S")
        out = build_spy_valuation(_art(first_seen=old), now=_FIXED_NOW)
        assert out["stale"] is True
        assert out["article_age_hours"] == pytest.approx(72.0, rel=1e-3)

    def test_boundary_36h_inclusive(self):
        at_36 = (_FIXED_NOW - timedelta(hours=36)).strftime("%Y-%m-%d %H:%M:%S")
        out = build_spy_valuation(_art(first_seen=at_36), now=_FIXED_NOW)
        # > 36 ⇒ stale; exactly 36 ⇒ not stale
        assert out["stale"] is False


class TestDegradeSoft:
    def test_none_article_returns_no_data(self):
        out = build_spy_valuation(None, now=_FIXED_NOW)
        assert out["state"] == "NO_DATA"
        assert out["regime"] is None
        assert out["cape"] is None
        assert out["stale"] is True

    def test_empty_dict_returns_no_data(self):
        out = build_spy_valuation({}, now=_FIXED_NOW)
        assert out["state"] == "NO_DATA"

    def test_unparseable_title_returns_parse_failed(self):
        out = build_spy_valuation(_art(title="market closed today"),
                                    now=_FIXED_NOW)
        assert out["state"] == "PARSE_FAILED"
        assert out["cape"] is None
        assert out["regime"] is None
        # Should still pass through the article metadata for the operator.
        assert out["source_url"] is not None

    def test_iso_with_T_separator_also_parses(self):
        ts = (_FIXED_NOW - timedelta(hours=2)).isoformat()
        out = build_spy_valuation(_art(first_seen=ts), now=_FIXED_NOW)
        assert out["state"] == "REGIME_READ"
        assert out["article_age_hours"] == pytest.approx(2.0, rel=1e-3)

    def test_bad_iso_degrades_to_stale_unknown_age(self):
        out = build_spy_valuation(_art(first_seen="not-a-date"),
                                    now=_FIXED_NOW)
        # Parse still succeeds for CAPE; only the age is unknown.
        assert out["state"] == "REGIME_READ"
        assert out["article_age_hours"] is None
        assert out["stale"] is True
