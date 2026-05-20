"""Tests for paper_trader.analytics.held_theme_decay.

Per-held-ticker fresh-vs-prior decayed-news-score velocity. Asserts the
exact arithmetic of the decayed weight sum, the state ladder cutoffs
(FADING / BUILDING / STABLE / DARK), held-only article weighting,
multi-ticker article split, defense-in-depth backtest filter, and the
collapse-to-silence NO_HELD path.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.held_theme_decay import (
    BUILD_RATIO,
    DECAY_HALF_LIFE_HOURS,
    FADE_RATIO,
    FRESH_WINDOW_HOURS,
    MIN_FRESH_SCORE,
    build_held_theme_decay,
)
from paper_trader.analytics import news_themes as nt_mod

NOW = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)


def _art(*, ts_h_ago, tickers, ai_score=8.0, title="hl",
         url="https://example.com/a", source="rss"):
    ts = (NOW - timedelta(hours=ts_h_ago)).isoformat()
    return {
        "title": title,
        "url": url,
        "source": source,
        "ai_score": ai_score,
        "urgency": 0,
        "first_seen": ts,
        "tickers": list(tickers),
    }


class TestSSOTDecayHalfLife:
    """The half-life constant MUST come from news_themes — any drift in
    the decay shape between this endpoint and /api/news-themes is a
    silent operator-confusion bug (the fresh_score here is supposed to
    line up with the news-themes top-themes contribution magnitude)."""

    def test_decay_half_life_is_news_themes_ssot(self):
        assert DECAY_HALF_LIFE_HOURS == nt_mod.DECAY_HALF_LIFE_HOURS

    def test_default_fresh_window_matches_half_life(self):
        # Picking fresh window = half life is what makes the fresh-score
        # magnitudes from this endpoint directly comparable to news-themes.
        assert FRESH_WINDOW_HOURS == DECAY_HALF_LIFE_HOURS


class TestStateLadderEdges:
    """Verdict cutoff arithmetic — exact ratios, not 'about right'."""

    def test_no_held_collapses_to_silence(self):
        out = build_held_theme_decay([], held_tickers=[], now=NOW)
        assert out["state"] == "NO_HELD"
        assert out["holds"] == []
        assert out["n_held"] == 0
        assert out["worst_verdict"] is None

    def test_held_with_zero_articles_marks_dark(self):
        out = build_held_theme_decay([], held_tickers=["NVDA"], now=NOW)
        assert out["state"] == "OK"
        assert out["n_held"] == 1
        assert out["holds"][0]["ticker"] == "NVDA"
        assert out["holds"][0]["verdict"] == "DARK"
        assert out["holds"][0]["fresh_score"] == 0.0
        assert out["holds"][0]["prior_score"] == 0.0
        assert out["holds"][0]["ratio"] is None
        assert out["dark_tickers"] == ["NVDA"]
        assert out["worst_verdict"] == "DARK"

    def test_fresh_alone_above_floor_marks_building(self):
        # No prior coverage, fresh meaningful → BUILDING (not DARK).
        # Score 8.0 at age 0 → fresh_score = 8.0 (well above MIN_FRESH).
        arts = [_art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=8.0)]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        h = out["holds"][0]
        assert h["verdict"] == "BUILDING"
        assert h["prior_score"] == 0.0
        assert h["ratio"] is None  # prior 0 must not fabricate +inf
        assert h["fresh_n"] == 1
        assert h["prior_n"] == 0

    def test_fresh_alone_below_floor_stays_dark(self):
        # A tiny fresh score (<MIN_FRESH_SCORE) with no prior must NOT
        # claim BUILDING — absolute prominence floor honesty.
        arts = [_art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=0.5)]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        assert out["holds"][0]["verdict"] == "DARK"
        assert out["holds"][0]["fresh_score"] == 0.5
        assert MIN_FRESH_SCORE > 0.5

    def test_fading_when_fresh_drops_below_fade_ratio(self):
        # Prior window: one 8.0 article at 9h ago (within prior 6-12h band).
        # Decay weight at 9h: 8.0 * exp(-9/6 * ln2) = 8.0 * 2^-1.5 ≈ 2.828
        # Fresh window: one 8.0 article at 5h ago (within fresh 0-6h).
        # Decay weight at 5h: 8.0 * exp(-5/6 * ln2) = 8.0 * 2^-5/6 ≈ 4.49
        # Ratio fresh/prior ≈ 4.49 / 2.828 ≈ 1.59 → BUILDING, not FADING.
        # For FADING we need fresh < prior * 0.7. Put PRIOR high & fresh low:
        # Prior: 8.0 article at 6.1h ago (just inside prior band).
        # Fresh: 8.0 article at 5.9h ago (just inside fresh band).
        # Both decay roughly equally (~3.99 vs ~4.04) → STABLE.
        # Easier: prior loud (8.0 @ 6.1h), fresh quiet (2.0 @ 0h).
        arts = [
            _art(ts_h_ago=6.1, tickers=["NVDA"], ai_score=8.0,
                 title="loud-prior"),
            _art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=2.0,
                 title="quiet-fresh"),
        ]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        h = out["holds"][0]
        # Prior weight: 8.0 * 2^(-6.1/6) ≈ 3.94. Fresh weight: 2.0.
        # Ratio ≈ 2.0 / 3.94 ≈ 0.508 < FADE_RATIO=0.7 → FADING.
        prior_expect = 8.0 * math.exp(-6.1 / 6.0 * math.log(2))
        fresh_expect = 2.0
        assert abs(h["fresh_score"] - fresh_expect) < 1e-3
        assert abs(h["prior_score"] - prior_expect) < 1e-3
        assert h["verdict"] == "FADING"
        assert h["ratio"] is not None and h["ratio"] < FADE_RATIO

    def test_building_when_fresh_exceeds_build_ratio(self):
        # Prior quiet (1.5 @ 6.1h ≈ 0.74), fresh loud (8.0 @ 0h = 8.0).
        # Ratio 8.0 / 0.74 ≈ 10.8 > BUILD_RATIO=1.43 → BUILDING.
        arts = [
            _art(ts_h_ago=6.1, tickers=["NVDA"], ai_score=1.5,
                 title="quiet-prior"),
            _art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=8.0,
                 title="loud-fresh"),
        ]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        h = out["holds"][0]
        assert h["verdict"] == "BUILDING"
        assert h["ratio"] is not None and h["ratio"] > BUILD_RATIO
        # Top fresh title surfaced.
        assert h["top_fresh_title"] == "loud-fresh"

    def test_stable_when_ratio_inside_band(self):
        # Symmetric prior/fresh: both 5.0 @ 3h and 9h (just inside windows).
        # Prior: 5.0 * 2^(-9/6)  ≈ 1.768
        # Fresh: 5.0 * 2^(-3/6)  ≈ 3.536
        # Ratio ≈ 2.0 — above BUILD_RATIO (1.43) → BUILDING.
        # For STABLE use ages that produce ratio ∈ [0.7, 1.43].
        # Prior 5.0 @ 7h → 5.0 * 2^(-7/6) ≈ 2.227
        # Fresh 5.0 @ 5h → 5.0 * 2^(-5/6) ≈ 2.806. Ratio ≈ 1.26 → STABLE.
        arts = [
            _art(ts_h_ago=7.0, tickers=["NVDA"], ai_score=5.0),
            _art(ts_h_ago=5.0, tickers=["NVDA"], ai_score=5.0),
        ]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        h = out["holds"][0]
        assert h["verdict"] == "STABLE"


class TestArithmetic:
    """Pin the exact decayed-weight formula so a future tune is loud."""

    def test_fresh_score_is_decayed_ai_score(self):
        # Single article: ai_score 9.0 @ 3h → 9.0 * exp(-3/6 * ln2) = 9/√2.
        arts = [_art(ts_h_ago=3.0, tickers=["NVDA"], ai_score=9.0)]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        expected = 9.0 * math.exp(-3.0 / 6.0 * math.log(2))
        assert abs(out["holds"][0]["fresh_score"] - expected) < 1e-3

    def test_multi_ticker_article_splits_weight(self):
        # 4-ticker article (3 held, 1 unheld) at age 0: per-ticker fresh
        # contribution must be (ai/4) not ai. Held-mention loop only
        # writes to held tickers — but the SPLIT denominator counts ALL
        # mentioned tickers (anti-inflation).
        arts = [
            _art(
                ts_h_ago=0.0,
                tickers=["NVDA", "TQQQ", "AMD", "MU"],
                ai_score=8.0,
                title="quad",
            )
        ]
        out = build_held_theme_decay(
            arts, held_tickers=["NVDA", "TQQQ", "AMD"], now=NOW,
        )
        by = {h["ticker"]: h for h in out["holds"]}
        # Each held ticker should get 8.0 / 4 = 2.0 fresh score.
        for tk in ("NVDA", "TQQQ", "AMD"):
            assert abs(by[tk]["fresh_score"] - 2.0) < 1e-3, tk

    def test_articles_with_zero_score_contribute_nothing(self):
        arts = [
            _art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=0.0),
            _art(ts_h_ago=3.0, tickers=["NVDA"], ai_score=0.0),
        ]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        h = out["holds"][0]
        assert h["fresh_score"] == 0.0
        assert h["prior_score"] == 0.0
        assert h["verdict"] == "DARK"

    def test_articles_outside_both_windows_ignored(self):
        # Beyond 2*window → must NOT count toward prior.
        arts = [_art(ts_h_ago=24.0, tickers=["NVDA"], ai_score=10.0)]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        h = out["holds"][0]
        assert h["fresh_score"] == 0.0
        assert h["prior_score"] == 0.0
        assert h["fresh_n"] == 0
        assert h["prior_n"] == 0

    def test_unheld_ticker_articles_are_ignored(self):
        # An article that only mentions an unheld ticker must not show up.
        arts = [_art(ts_h_ago=0.0, tickers=["GOOG"], ai_score=9.0)]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        h = out["holds"][0]
        assert h["fresh_score"] == 0.0
        assert h["verdict"] == "DARK"


class TestBacktestDefenseInDepth:
    """A leaked synthetic row must not corrupt the held-position view."""

    def test_backtest_url_dropped(self):
        arts = [_art(
            ts_h_ago=0.0, tickers=["NVDA"], ai_score=9.0,
            url="backtest://run_42/2025-09-15/BUY/NVDA",
        )]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        assert out["holds"][0]["fresh_score"] == 0.0

    def test_backtest_source_dropped(self):
        arts = [_art(
            ts_h_ago=0.0, tickers=["NVDA"], ai_score=9.0,
            source="backtest_run_42_winner",
        )]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        assert out["holds"][0]["fresh_score"] == 0.0

    def test_opus_annotation_dropped(self):
        arts = [_art(
            ts_h_ago=0.0, tickers=["NVDA"], ai_score=9.0,
            source="opus_annotation_cycle_5",
        )]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        assert out["holds"][0]["fresh_score"] == 0.0


class TestAggregateAndHeadline:
    """Aggregate counts and the operator-facing headline pick the right
    severity bucket."""

    def test_fading_leads_aggregate_and_headline(self):
        # Two holds: NVDA FADING, TQQQ BUILDING. Worst must be FADING.
        arts = [
            _art(ts_h_ago=6.1, tickers=["NVDA"], ai_score=8.0),
            _art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=2.0),
            _art(ts_h_ago=6.1, tickers=["TQQQ"], ai_score=1.5),
            _art(ts_h_ago=0.0, tickers=["TQQQ"], ai_score=8.0),
        ]
        out = build_held_theme_decay(
            arts, held_tickers=["NVDA", "TQQQ"], now=NOW,
        )
        assert out["n_fading"] == 1
        assert out["n_building"] == 1
        assert out["fading_tickers"] == ["NVDA"]
        assert out["building_tickers"] == ["TQQQ"]
        assert out["worst_verdict"] == "FADING"
        assert "FADING" in out["headline"]
        assert "NVDA" in out["headline"]

    def test_dark_only_path(self):
        out = build_held_theme_decay(
            [], held_tickers=["NVDA", "TQQQ"], now=NOW,
        )
        assert out["n_dark"] == 2
        assert out["worst_verdict"] == "DARK"
        assert "DARK" in out["headline"]

    def test_holds_sorted_by_severity(self):
        # NVDA BUILDING, TQQQ FADING, MU DARK.
        arts = [
            # NVDA: prior quiet, fresh loud → BUILDING
            _art(ts_h_ago=6.1, tickers=["NVDA"], ai_score=1.5),
            _art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=8.0),
            # TQQQ: prior loud, fresh quiet → FADING
            _art(ts_h_ago=6.1, tickers=["TQQQ"], ai_score=8.0),
            _art(ts_h_ago=0.0, tickers=["TQQQ"], ai_score=2.0),
            # MU: no articles → DARK
        ]
        out = build_held_theme_decay(
            arts, held_tickers=["NVDA", "TQQQ", "MU"], now=NOW,
        )
        verdicts = [h["verdict"] for h in out["holds"]]
        # FADING, DARK, STABLE-or-BUILDING (BUILDING in this case).
        assert verdicts[0] == "FADING"
        assert verdicts[1] == "DARK"
        assert verdicts[-1] == "BUILDING"

    def test_held_tickers_case_insensitive_and_dedupe(self):
        arts = [_art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=8.0)]
        out = build_held_theme_decay(
            arts, held_tickers=["nvda", "NVDA", "Nvda"], now=NOW,
        )
        assert out["n_held"] == 1
        assert out["holds"][0]["ticker"] == "NVDA"
        assert out["holds"][0]["verdict"] == "BUILDING"


class TestGarbageInput:
    """Never-raise contract under degraded/odd inputs."""

    def test_garbage_rows_skipped(self):
        arts = [
            None,
            "not-a-dict",
            {},
            {"first_seen": "garbage"},
            {"first_seen": None, "tickers": ["NVDA"]},
            _art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=8.0),
        ]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        # Only the one well-formed article counts.
        assert out["holds"][0]["fresh_n"] == 1
        assert out["holds"][0]["verdict"] == "BUILDING"

    def test_non_list_tickers_skipped(self):
        arts = [{
            "title": "t", "url": "u", "source": "s",
            "ai_score": 8.0, "urgency": 0,
            "first_seen": NOW.isoformat(),
            "tickers": "NVDA",  # string, not list
        }]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        assert out["holds"][0]["fresh_score"] == 0.0

    def test_string_ai_score_tolerated(self):
        arts = [{
            "title": "t", "url": "u", "source": "s",
            "ai_score": "not-a-number", "urgency": 0,
            "first_seen": NOW.isoformat(),
            "tickers": ["NVDA"],
        }]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        assert out["holds"][0]["fresh_score"] == 0.0


class TestShape:
    """Stable output keys regardless of input."""

    REQUIRED_TOP = {
        "as_of", "fresh_window_hours", "prior_window_hours",
        "decay_half_life_hours", "state", "holds", "n_held",
        "n_fading", "n_building", "n_dark", "n_stable",
        "fading_tickers", "building_tickers", "dark_tickers",
        "worst_verdict", "headline",
    }
    REQUIRED_HOLD = {
        "ticker", "fresh_score", "prior_score", "fresh_n", "prior_n",
        "ratio", "verdict", "top_fresh_title", "top_fresh_url",
    }

    def test_empty_input_has_full_shape(self):
        out = build_held_theme_decay([], held_tickers=[], now=NOW)
        assert self.REQUIRED_TOP <= set(out.keys())

    def test_ok_path_has_full_shape(self):
        arts = [_art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=8.0)]
        out = build_held_theme_decay(arts, held_tickers=["NVDA"], now=NOW)
        assert self.REQUIRED_TOP <= set(out.keys())
        assert self.REQUIRED_HOLD <= set(out["holds"][0].keys())
