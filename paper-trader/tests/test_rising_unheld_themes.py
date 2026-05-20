"""Tests for paper_trader.analytics.rising_unheld_themes.

The unheld complement to held_theme_decay. Asserts:
  * SSOT constants come from held_theme_decay (rotation-pair invariant)
  * State ladder: DARK / FADING / STABLE / BUILDING / BREAKING
  * Held tickers are EXCLUDED from the output (no overlap with
    held-theme-decay)
  * Multi-ticker article split denominator counts ALL tickers
    (held+unheld) so per-ticker weights match held-theme-decay's weights
  * Defense-in-depth backtest filter
  * Sort order BREAKING > BUILDING > STABLE > FADING > DARK, within
    each bucket by descending fresh_score
  * max_themes cap on rows; aggregate counts span the full universe
  * Ratio is None when prior==0 (don't fabricate +inf)
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.rising_unheld_themes import (
    BREAKING_FRESH_SCORE,
    DEFAULT_MAX_THEMES,
    build_rising_unheld_themes,
)
from paper_trader.analytics import held_theme_decay as htd_mod
from paper_trader.analytics import rising_unheld_themes as run_mod

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


class TestSSOTSharedConstants:
    """The rotation-pair invariant: re-tune held_theme_decay's window /
    decay shape and rising_unheld_themes follows automatically. Drift
    between the two velocity surfaces silently confuses the operator
    (held FADING + unheld BUILDING is the actionable rotation signal)."""

    def test_fresh_window_is_held_theme_decay_ssot(self):
        assert run_mod.FRESH_WINDOW_HOURS == htd_mod.FRESH_WINDOW_HOURS

    def test_min_fresh_score_is_held_theme_decay_ssot(self):
        assert run_mod.MIN_FRESH_SCORE == htd_mod.MIN_FRESH_SCORE

    def test_fade_ratio_is_held_theme_decay_ssot(self):
        assert run_mod.FADE_RATIO == htd_mod.FADE_RATIO

    def test_build_ratio_is_held_theme_decay_ssot(self):
        assert run_mod.BUILD_RATIO == htd_mod.BUILD_RATIO

    def test_breaking_floor_is_strictly_above_min_fresh(self):
        # BREAKING (brand-new catalyst) must require more absolute
        # weight than BUILDING (accelerating existing story) to be
        # honest — a 1.0-score single article with no prior shouldn't
        # claim "brand new theme."
        assert BREAKING_FRESH_SCORE > run_mod.MIN_FRESH_SCORE


class TestHeldExclusion:
    """Held tickers must NEVER appear in the unheld output, regardless
    of how loud their coverage is. Overlap with held-theme-decay would
    violate invariant #10 (single source of truth per surface)."""

    def test_held_ticker_excluded_even_when_loud(self):
        arts = [_art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=9.5)]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        # No unheld ticker scored — wire was held-only.
        assert all(r["ticker"] != "NVDA" for r in out["themes"])

    def test_held_set_is_case_insensitive(self):
        # Held set normalized to upper-strip; an article with the
        # same ticker in any case must still be excluded.
        arts = [_art(ts_h_ago=0.0, tickers=["nvda"], ai_score=9.0)]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NvDa"], now=NOW,
        )
        assert all(r["ticker"] != "NVDA" for r in out["themes"])

    def test_unheld_tickers_in_same_article_still_counted(self):
        # 2-ticker article — one held, one unheld. The unheld one
        # MUST appear; the held one MUST NOT. Split denominator counts
        # both (anti-inflation rule), so unheld gets 0.5× weight.
        arts = [_art(ts_h_ago=0.0, tickers=["NVDA", "MU"],
                     ai_score=8.0, title="pair")]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        by = {r["ticker"]: r for r in out["themes"]}
        assert "NVDA" not in by
        assert "MU" in by
        # 8.0 / 2 tickers (NVDA+MU, both counted in denominator) = 4.0.
        assert abs(by["MU"]["fresh_score"] - 4.0) < 1e-3


class TestSplitDenominatorMatchesHeldThemeDecay:
    """The split denominator (all-mentioned-tickers count) is the
    anti-inflation rule both surfaces share. The unheld weights here
    MUST equal the weights held_theme_decay would assign to those
    same tickers if they were held — that's what makes the two
    surfaces directly comparable."""

    def test_split_uses_all_tickers_held_plus_unheld(self):
        from paper_trader.analytics.held_theme_decay import build_held_theme_decay

        arts = [_art(
            ts_h_ago=0.0,
            tickers=["NVDA", "MU", "AMD", "INTC"],
            ai_score=8.0,
            title="quad",
        )]
        # Case A: NVDA held → MU/AMD/INTC are unheld, each gets 8/4=2.0.
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        by = {r["ticker"]: r for r in out["themes"]}
        for tk in ("MU", "AMD", "INTC"):
            assert abs(by[tk]["fresh_score"] - 2.0) < 1e-3, tk

        # Case B: MU held → held_theme_decay should give MU the SAME
        # 2.0 weight (4-way split). Pinning the cross-surface invariant.
        out_held = build_held_theme_decay(
            arts, held_tickers=["MU"], now=NOW,
        )
        assert abs(out_held["holds"][0]["fresh_score"] - 2.0) < 1e-3


class TestStateLadder:
    """Verdict cutoffs — exact ratios, not 'about right'."""

    def test_no_articles_no_data(self):
        out = build_rising_unheld_themes([], held_tickers=["NVDA"], now=NOW)
        assert out["state"] == "NO_DATA"
        assert out["themes"] == []
        assert out["n_unheld_seen"] == 0
        assert out["top_rising"] is None

    def test_held_only_wire_no_data_with_explanation(self):
        # Articles exist but every one is held-only — honest NO_DATA
        # with a held-only flavor in the headline.
        arts = [_art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=9.0)]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["state"] == "OK"
        assert out["themes"] == []
        assert "held-only" in out["headline"]

    def test_breaking_when_no_prior_and_loud_fresh(self):
        # Fresh 8.0 @ 0.5h → ~8.0 decayed; prior 0 → BREAKING.
        arts = [_art(ts_h_ago=0.5, tickers=["MU"], ai_score=8.0,
                     title="new-mu")]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        row = out["themes"][0]
        assert row["ticker"] == "MU"
        assert row["verdict"] == "BREAKING"
        assert row["prior_score"] == 0.0
        assert row["ratio"] is None  # don't fabricate +inf
        assert row["fresh_n"] == 1
        assert row["top_fresh_title"] == "new-mu"

    def test_no_prior_below_breaking_floor_is_building(self):
        # Fresh score in [MIN_FRESH_SCORE, BREAKING_FRESH_SCORE) with
        # prior==0 — qualifies as BUILDING (accelerating from no
        # coverage), NOT BREAKING.
        # 2.0 @ 0h → fresh_score = 2.0 (between 1.0 and 3.0).
        arts = [_art(ts_h_ago=0.0, tickers=["MU"], ai_score=2.0)]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["themes"][0]["verdict"] == "BUILDING"

    def test_no_prior_below_min_fresh_is_dark(self):
        # 0.5 @ 0h → fresh_score = 0.5 (below MIN_FRESH_SCORE=1.0).
        # Honest DARK even though fresh > prior numerically.
        arts = [_art(ts_h_ago=0.0, tickers=["MU"], ai_score=0.5)]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        # DARK rows should still appear in the per-ticker list (sorted
        # last), but n_breaking/n_building == 0.
        assert out["n_breaking"] == 0
        assert out["n_building"] == 0
        if out["themes"]:
            assert out["themes"][0]["verdict"] == "DARK"

    def test_building_when_ratio_above_threshold_with_existing_prior(self):
        # Prior quiet (1.5 @ 6.1h ≈ 0.74), fresh loud (8.0 @ 0h = 8.0).
        # Ratio ≈ 10.8 > BUILD_RATIO=1.43, fresh >= MIN_FRESH → BUILDING.
        arts = [
            _art(ts_h_ago=6.1, tickers=["MU"], ai_score=1.5, title="q"),
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=8.0, title="loud"),
        ]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        row = out["themes"][0]
        assert row["verdict"] == "BUILDING"
        assert row["ratio"] is not None and row["ratio"] > run_mod.BUILD_RATIO
        assert row["top_fresh_title"] == "loud"

    def test_fading_when_fresh_drops_below_fade_ratio(self):
        # Prior loud (8.0 @ 6.1h ≈ 3.94), fresh quiet (2.0 @ 0h = 2.0).
        # Ratio ≈ 0.508 < FADE_RATIO=0.7 → FADING. Surfaced for context
        # but NOT actionable (the operator wouldn't enter a fading
        # unheld name — informational only).
        arts = [
            _art(ts_h_ago=6.1, tickers=["MU"], ai_score=8.0),
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=2.0),
        ]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        # FADING goes in the per-ticker list (per spec) but n_fading
        # counts it and it does NOT count toward actionable rotation.
        assert out["n_fading"] == 1
        # FADING outranks DARK but sits below BREAKING/BUILDING/STABLE
        # in the sort.
        assert any(r["ticker"] == "MU" and r["verdict"] == "FADING"
                   for r in out["themes"])

    def test_stable_when_ratio_inside_band(self):
        # Prior 5.0 @ 7h → 5.0 * 2^(-7/6) ≈ 2.227
        # Fresh 5.0 @ 5h → 5.0 * 2^(-5/6) ≈ 2.806. Ratio ≈ 1.26.
        arts = [
            _art(ts_h_ago=7.0, tickers=["MU"], ai_score=5.0),
            _art(ts_h_ago=5.0, tickers=["MU"], ai_score=5.0),
        ]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        row = next(r for r in out["themes"] if r["ticker"] == "MU")
        assert row["verdict"] == "STABLE"


class TestArithmetic:
    """Pin the exact decayed-weight formula and split rule."""

    def test_fresh_score_matches_decay_formula(self):
        # Single article: ai 9.0 @ 3h → 9.0 * exp(-3/6 * ln2) = 9/√2.
        arts = [_art(ts_h_ago=3.0, tickers=["MU"], ai_score=9.0)]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        expected = 9.0 * math.exp(-3.0 / 6.0 * math.log(2))
        row = next(r for r in out["themes"] if r["ticker"] == "MU")
        assert abs(row["fresh_score"] - expected) < 1e-3

    def test_zero_score_articles_contribute_nothing(self):
        arts = [
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=0.0),
            _art(ts_h_ago=3.0, tickers=["MU"], ai_score=0.0),
        ]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        # MU should either be absent (no weight accumulated) or DARK.
        for r in out["themes"]:
            if r["ticker"] == "MU":
                assert r["fresh_score"] == 0.0
                assert r["prior_score"] == 0.0

    def test_articles_outside_both_windows_ignored(self):
        # Beyond 2*window → must NOT count toward prior.
        arts = [_art(ts_h_ago=24.0, tickers=["MU"], ai_score=10.0)]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        # Article was entirely dropped → MU should not be in themes.
        assert all(r["ticker"] != "MU" for r in out["themes"])


class TestBacktestDefenseInDepth:
    """A leaked synthetic row must not corrupt the rotation surface."""

    def test_backtest_url_dropped(self):
        arts = [_art(
            ts_h_ago=0.0, tickers=["MU"], ai_score=9.0,
            url="backtest://run_42/2025-09-15/BUY/MU",
        )]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["themes"] == []
        assert out["n_breaking"] == 0

    def test_backtest_source_dropped(self):
        arts = [_art(
            ts_h_ago=0.0, tickers=["MU"], ai_score=9.0,
            source="backtest_run_42_winner",
        )]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["themes"] == []

    def test_opus_annotation_dropped(self):
        arts = [_art(
            ts_h_ago=0.0, tickers=["MU"], ai_score=9.0,
            source="opus_annotation_cycle_5",
        )]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["themes"] == []


class TestSortOrderAndCap:
    """BREAKING > BUILDING > STABLE > FADING > DARK; within bucket by
    fresh_score descending."""

    def test_breaking_outranks_building(self):
        # MU: prior 0, fresh 8.0 @ 0h → BREAKING (≥3.0 floor).
        # INTC: prior 1.5 @ 6.1h (≈0.74), fresh 4.0 @ 0h →
        #   ratio ~5.4 > 1.43, fresh > MIN_FRESH → BUILDING.
        arts = [
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=8.0),
            _art(ts_h_ago=6.1, tickers=["INTC"], ai_score=1.5),
            _art(ts_h_ago=0.0, tickers=["INTC"], ai_score=4.0),
        ]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["themes"][0]["ticker"] == "MU"
        assert out["themes"][0]["verdict"] == "BREAKING"
        assert out["themes"][1]["ticker"] == "INTC"
        assert out["themes"][1]["verdict"] == "BUILDING"

    def test_within_breaking_loudest_first(self):
        # Two BREAKING themes — louder fresh_score goes first.
        arts = [
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=4.0),
            _art(ts_h_ago=0.0, tickers=["AMD"], ai_score=9.0),
        ]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["themes"][0]["ticker"] == "AMD"
        assert out["themes"][1]["ticker"] == "MU"

    def test_max_themes_caps_returned_rows_not_counters(self):
        # Generate 5 unheld BREAKING themes; cap to 2. Counters must
        # see all 5; rows must trim to 2.
        arts = [
            _art(ts_h_ago=0.0, tickers=[tk], ai_score=8.0)
            for tk in ("MU", "AMD", "INTC", "AVGO", "TSM")
        ]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW, max_themes=2,
        )
        assert len(out["themes"]) == 2
        assert out["n_breaking"] == 5
        assert out["n_unheld_seen"] == 5

    def test_top_rising_picks_breaking_over_building(self):
        arts = [
            _art(ts_h_ago=6.1, tickers=["INTC"], ai_score=1.5),
            _art(ts_h_ago=0.0, tickers=["INTC"], ai_score=4.0),
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=8.0),
        ]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["top_rising"] is not None
        assert out["top_rising"]["ticker"] == "MU"
        assert out["top_rising"]["verdict"] == "BREAKING"

    def test_top_rising_none_when_no_actionable(self):
        # Only DARK/FADING/STABLE in the universe.
        arts = [_art(ts_h_ago=0.0, tickers=["MU"], ai_score=0.5)]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["top_rising"] is None


class TestAggregateAndHeadline:
    """Counters and the operator-facing headline pick the right
    severity / phrasing."""

    def test_breaking_leads_headline(self):
        arts = [_art(ts_h_ago=0.0, tickers=["MU"], ai_score=8.0)]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["n_breaking"] == 1
        assert "BREAKING" in out["headline"]
        assert "MU" in out["headline"]

    def test_building_only_headline(self):
        # One BUILDING, no BREAKING.
        arts = [
            _art(ts_h_ago=6.1, tickers=["INTC"], ai_score=1.5),
            _art(ts_h_ago=0.0, tickers=["INTC"], ai_score=4.0),
        ]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["n_breaking"] == 0
        assert out["n_building"] == 1
        assert "BUILDING" in out["headline"]
        assert "INTC" in out["headline"]

    def test_no_rotation_candidates_headline(self):
        # FADING + DARK only — no actionable rotation. Headline should
        # say so honestly.
        arts = [
            _art(ts_h_ago=6.1, tickers=["MU"], ai_score=8.0),
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=2.0),
        ]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        assert out["n_breaking"] == 0
        assert out["n_building"] == 0
        assert "No rotation candidates" in out["headline"]


class TestParamClamps:
    """Builder must clamp out-of-range knobs (the dashboard layer also
    clamps but the builder is the SSOT for safe defaults)."""

    def test_max_themes_clamps_low_and_high(self):
        arts = [_art(ts_h_ago=0.0, tickers=["MU"], ai_score=8.0)]
        # max_themes=0 must clamp to >=1, returning at least 1 row.
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW, max_themes=0,
        )
        assert out["max_themes"] >= 1
        # max_themes=10000 must clamp <=100 (defensive).
        out_hi = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW, max_themes=10000,
        )
        assert out_hi["max_themes"] <= 100

    def test_fresh_window_clamp(self):
        # Builder clamps to >=0.5 hours; pass 0 → 0.5.
        out = build_rising_unheld_themes(
            [], held_tickers=[], now=NOW, fresh_window_hours=0.0,
        )
        assert out["fresh_window_hours"] >= 0.5

    def test_garbage_max_themes_falls_back_to_default(self):
        out = build_rising_unheld_themes(
            [], held_tickers=[], now=NOW, max_themes="garbage",  # type: ignore[arg-type]
        )
        assert out["max_themes"] == DEFAULT_MAX_THEMES


class TestShapeStability:
    """The response shape is stable regardless of input — downstream
    callers (UI, panel-health, command-center) must not need a try-cast."""

    def test_no_data_path_has_all_keys(self):
        out = build_rising_unheld_themes([], held_tickers=[], now=NOW)
        expected = {
            "as_of", "fresh_window_hours", "prior_window_hours",
            "decay_half_life_hours", "max_themes", "state", "themes",
            "n_unheld_seen", "n_building", "n_breaking", "n_fading",
            "n_dark", "n_stable", "building_tickers",
            "breaking_tickers", "top_rising", "headline",
        }
        assert expected <= set(out.keys())

    def test_ok_path_has_all_keys(self):
        arts = [_art(ts_h_ago=0.0, tickers=["MU"], ai_score=8.0)]
        out = build_rising_unheld_themes(
            arts, held_tickers=["NVDA"], now=NOW,
        )
        expected = {
            "as_of", "fresh_window_hours", "prior_window_hours",
            "decay_half_life_hours", "max_themes", "state", "themes",
            "n_unheld_seen", "n_building", "n_breaking", "n_fading",
            "n_dark", "n_stable", "building_tickers",
            "breaking_tickers", "top_rising", "headline",
        }
        assert expected <= set(out.keys())

    def test_garbage_articles_dont_raise(self):
        garbage = [None, "string", {"no_first_seen": True},
                   {"first_seen": "not-iso", "ai_score": 1.0,
                    "tickers": ["MU"]},
                   {"first_seen": NOW.isoformat(), "ai_score": "bad",
                    "tickers": "not-a-list"}]
        out = build_rising_unheld_themes(
            garbage, held_tickers=["NVDA"], now=NOW,
        )
        assert out["state"] in ("NO_DATA", "OK")
