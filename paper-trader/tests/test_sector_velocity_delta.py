"""Tests for paper_trader.analytics.sector_velocity_delta.

Per-sector-bucket news-velocity delta. Asserts:
  * SSOT constants come from held_theme_decay / news_themes
    (rotation-pair invariant, extended to bucket level)
  * Bucket SSOT is sector_heatmap.HEATMAP_BUCKETS
  * Verdict ladder: DARK / FADING / DECELERATING / STABLE /
    BUILDING / ACCELERATING — with bucket-prominence floor scaling
    by bucket size
  * Multi-ticker split denominator counts ALL tickers (anti-inflation,
    matches news_themes / held_theme_decay)
  * Multi-bucket articles attribute weight to every bucket the
    ticker(s) belong to (no double-deduction across buckets)
  * Defense-in-depth backtest filter
  * Sort order ACCELERATING > BUILDING > STABLE > FADING >
    DECELERATING > DARK
  * Ratio is None when prior == 0 (don't fabricate +inf)
  * top_accelerating / top_decelerating / rotating_in / rotating_out
    surfacing
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.sector_velocity_delta import (
    ACCEL_RATIO,
    DECEL_RATIO,
    build_sector_velocity_delta,
)
from paper_trader.analytics import held_theme_decay as htd_mod
from paper_trader.analytics import news_themes as nt_mod
from paper_trader.analytics import sector_velocity_delta as svd_mod
from paper_trader.analytics import sector_heatmap as sh_mod

NOW = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)

# Small test bucket map — lets us pin the exact arithmetic without
# depending on the production HEATMAP_BUCKETS list changing under us.
TEST_BUCKETS = {
    "memory_core":      ["MU", "WDC", "STX"],          # 3 tickers
    "semis_equipment":  ["LRCX", "AMAT", "KLAC", "ASML"],  # 4 tickers
    "foundry":          ["TSM"],                       # 1 ticker
    "design":           ["NVDA", "AMD"],               # 2 tickers
}


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


class TestSSOTConstantsShared:
    """Rotation-pair invariant, extended to the bucket level. Drift
    between the held-ticker / unheld-ticker / sector velocity surfaces
    silently confuses the operator (FADING + BUILDING + STABLE at the
    same ticker should always cross-correlate with the bucket-level
    verdict)."""

    def test_decay_half_life_from_news_themes(self):
        from paper_trader.analytics.sector_velocity_delta import (
            DECAY_HALF_LIFE_HOURS,
        )
        assert DECAY_HALF_LIFE_HOURS == nt_mod.DECAY_HALF_LIFE_HOURS

    def test_window_constants_from_held_theme_decay(self):
        assert svd_mod.FRESH_WINDOW_HOURS == htd_mod.FRESH_WINDOW_HOURS
        assert svd_mod.MIN_FRESH_SCORE == htd_mod.MIN_FRESH_SCORE
        assert svd_mod.BUILD_RATIO == htd_mod.BUILD_RATIO
        assert svd_mod.FADE_RATIO == htd_mod.FADE_RATIO

    def test_bucket_ssot_from_sector_heatmap(self):
        # The production builder uses sector_heatmap.HEATMAP_BUCKETS
        # by default (the bucket override is a test seam only).
        from paper_trader.analytics.sector_velocity_delta import HEATMAP_BUCKETS
        assert HEATMAP_BUCKETS is sh_mod.HEATMAP_BUCKETS

    def test_accel_ratio_strictly_above_build_ratio(self):
        # ACCELERATING (sector-level) must require MORE absolute
        # acceleration than BUILDING (per-ticker-level) — otherwise
        # the two verdicts collapse to the same threshold.
        assert ACCEL_RATIO > svd_mod.BUILD_RATIO

    def test_decel_ratio_equals_fade_ratio(self):
        # Symmetric: a bucket DECELERATING uses the same ratio
        # threshold as a single ticker FADING; the distinction is the
        # bucket-prominence floor (absolute weight, not ratio).
        assert DECEL_RATIO == svd_mod.FADE_RATIO


class TestStateLadder:
    """Verdict cutoffs at the bucket level — exact arithmetic."""

    def test_no_articles_no_data(self):
        out = build_sector_velocity_delta([], now=NOW, buckets=TEST_BUCKETS)
        assert out["state"] == "NO_DATA"
        assert all(b["verdict"] == "DARK" for b in out["buckets"])
        # Aggregate counts on the no-data path collapse to 0 — operator
        # reads NO_DATA, not "every bucket is DARK".
        assert out["n_accelerating"] == 0
        assert out["n_decelerating"] == 0
        assert out["rotating_in"] == []
        assert out["top_accelerating"] is None

    def test_dark_when_neither_window_crosses_single_floor(self):
        # A single ticker in foundry (TSM, n=1) with fresh 0.5 and
        # no prior — below MIN_FRESH (1.0) → DARK even though prior==0.
        arts = [_art(ts_h_ago=0.0, tickers=["TSM"], ai_score=0.5)]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        foundry = next(b for b in out["buckets"] if b["name"] == "foundry")
        assert foundry["verdict"] == "DARK"
        assert foundry["fresh_score"] == 0.5

    def test_building_when_no_prior_and_per_ticker_floor_met(self):
        # 1-ticker bucket (foundry, TSM), fresh ai 2.0 — above
        # MIN_FRESH (1.0) but below bucket floor (1 × 1.0 = 1.0 →
        # actually equal). The bucket floor for n=1 is MIN_FRESH, so
        # 2.0 fresh > floor → ACCELERATING. Confirms the floor scales
        # correctly for 1-ticker buckets.
        arts = [_art(ts_h_ago=0.0, tickers=["TSM"], ai_score=2.0)]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        foundry = next(b for b in out["buckets"] if b["name"] == "foundry")
        assert foundry["verdict"] == "ACCELERATING"

    def test_no_prior_low_absolute_is_building_not_accelerating(self):
        # semis_equipment (4 tickers, floor = 4.0). Single fresh
        # article at 2.0 → above per-ticker MIN_FRESH but below
        # bucket floor → BUILDING (not ACCELERATING).
        arts = [_art(ts_h_ago=0.0, tickers=["LRCX"], ai_score=2.0)]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        eq = next(b for b in out["buckets"] if b["name"] == "semis_equipment")
        assert eq["verdict"] == "BUILDING"

    def test_accelerating_requires_bucket_floor(self):
        # semis_equipment fresh 5.0 (above 4.0 floor) with no prior →
        # ACCELERATING.
        arts = [
            _art(ts_h_ago=0.0, tickers=["LRCX"], ai_score=5.0),
        ]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        eq = next(b for b in out["buckets"] if b["name"] == "semis_equipment")
        assert eq["verdict"] == "ACCELERATING"

    def test_accelerating_with_existing_prior(self):
        # memory_core: prior 3.0 @ 6.1h (weight ~1.48) and fresh
        # 9.0 @ 0h (weight 9.0). Ratio 9/1.48 ≈ 6.1 > ACCEL_RATIO
        # (1.6); fresh 9.0 >= bucket floor (3.0 for 3-ticker bucket)
        # → ACCELERATING.
        arts = [
            _art(ts_h_ago=6.1, tickers=["MU"], ai_score=3.0),
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=9.0),
        ]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        mc = next(b for b in out["buckets"] if b["name"] == "memory_core")
        assert mc["verdict"] == "ACCELERATING"
        assert mc["ratio"] is not None and mc["ratio"] > ACCEL_RATIO

    def test_decelerating_when_prior_loud_fresh_quiet(self):
        # memory_core prior loud (8.0 @ 6.1h ≈ 3.94, > 3.0 bucket floor),
        # fresh quiet (1.5 @ 0h = 1.5). Ratio ~0.38 < FADE_RATIO
        # → DECELERATING (rotation OUT) because prior >= bucket floor.
        arts = [
            _art(ts_h_ago=6.1, tickers=["MU"], ai_score=8.0),
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=1.5),
        ]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        mc = next(b for b in out["buckets"] if b["name"] == "memory_core")
        assert mc["verdict"] == "DECELERATING"

    def test_fading_when_marginal_bucket_cools(self):
        # design bucket (2 tickers, floor = 2.0). Prior 2.5 @ 6.1h
        # (weight ~1.23, BELOW bucket floor 2.0), fresh 0.5 @ 0h
        # (below MIN_FRESH). Both windows have SOMETHING but neither
        # crosses MIN_FRESH (1.0) for fresh_score, neither crosses
        # bucket floor for prior_score → DARK. Use a different shape:
        # prior 5.0 @ 7h (weight ~2.227, above bucket floor 2.0),
        # fresh 1.2 @ 0h (above MIN_FRESH, below bucket floor) — drop
        # ratio 1.2/2.227 ≈ 0.54 < FADE_RATIO → DECELERATING (prior
        # crossed floor). Need a TRULY marginal prior to get FADING.
        # Prior 1.5 @ 7h (weight ~0.67, below bucket floor AND below
        # MIN_FRESH), fresh 1.2 @ 0h (above MIN_FRESH). prior < MIN
        # AND fresh > MIN — not the "neither crosses MIN" DARK case.
        # ratio = 1.2/0.67 ≈ 1.8 → above FADE_RATIO → not FADING.
        # This shows FADING is a narrow case: prior > MIN, prior <
        # bucket floor, fresh < prior × FADE. Try:
        # Prior 2.0 @ 7h (weight ~0.89, below MIN — drops to DARK).
        # Prior 3.0 @ 7h (weight ~1.33, above MIN, below bucket floor
        # for n=2 bucket which is 2.0... wait, floor for n=2 is
        # MIN_FRESH * 2 = 2.0; 1.33 < 2.0 — below floor.)
        # Fresh 0.5 @ 0h (below MIN). ratio = 0.5/1.33 = 0.376 <
        # FADE_RATIO. prior > MIN, fresh < MIN, prior < bucket floor.
        # Neither-MIN check: max(fresh, prior) = 1.33 > MIN — passes.
        # prior < bucket floor → not DECELERATING. ratio < FADE →
        # FADING. ✓
        arts = [
            _art(ts_h_ago=7.0, tickers=["NVDA"], ai_score=3.0),
            _art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=0.5),
        ]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        design = next(b for b in out["buckets"] if b["name"] == "design")
        assert design["verdict"] == "FADING"

    def test_stable_when_ratio_inside_band(self):
        # memory_core prior 5.0 @ 7h (weight ~2.23) + fresh 5.0 @ 5h
        # (weight ~2.81). Ratio ~1.26 — inside [FADE, BUILD] → STABLE.
        # Both windows above per-ticker MIN_FRESH but below bucket
        # floor 3.0 → STABLE, not DARK.
        arts = [
            _art(ts_h_ago=7.0, tickers=["MU"], ai_score=5.0),
            _art(ts_h_ago=5.0, tickers=["MU"], ai_score=5.0),
        ]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        mc = next(b for b in out["buckets"] if b["name"] == "memory_core")
        assert mc["verdict"] == "STABLE"


class TestMultiTickerSplit:
    """Anti-inflation rule + multi-bucket attribution."""

    def test_split_uses_all_tickers_denominator(self):
        # 4-ticker article (all memory_core), ai 8.0. Each ticker
        # contributes 8/4 = 2.0 to fresh_score. memory_core has 3
        # tickers in TEST_BUCKETS (MU/WDC/STX); the 4th (MUU) is in
        # memory_leveraged which is NOT in TEST_BUCKETS — so only
        # 3 of 4 mentions land in tracked buckets. memory_core
        # fresh_score = 2.0 × 3 = 6.0.
        arts = [_art(
            ts_h_ago=0.0,
            tickers=["MU", "WDC", "STX", "MUU"],
            ai_score=8.0,
        )]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        mc = next(b for b in out["buckets"] if b["name"] == "memory_core")
        # Each of MU/WDC/STX contributed 2.0 → bucket sum 6.0.
        assert abs(mc["fresh_score"] - 6.0) < 1e-3

    def test_cross_bucket_article_attributes_to_both(self):
        # 2-ticker article: MU (memory_core) + NVDA (design).
        # ai 8.0, split 4.0 per ticker. Each bucket gets 4.0.
        arts = [_art(
            ts_h_ago=0.0,
            tickers=["MU", "NVDA"],
            ai_score=8.0,
        )]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        mc = next(b for b in out["buckets"] if b["name"] == "memory_core")
        design = next(b for b in out["buckets"] if b["name"] == "design")
        assert abs(mc["fresh_score"] - 4.0) < 1e-3
        assert abs(design["fresh_score"] - 4.0) < 1e-3
        # Each bucket counts the article ONCE (not twice for
        # multi-ticker).
        assert mc["fresh_n"] == 1
        assert design["fresh_n"] == 1

    def test_per_ticker_weights_match_held_theme_decay(self):
        # Cross-surface invariant: the bucket's fresh_score should
        # equal the sum of the per-held-ticker fresh_scores from
        # held_theme_decay over the same input + held set.
        from paper_trader.analytics.held_theme_decay import build_held_theme_decay
        arts = [_art(
            ts_h_ago=0.0, tickers=["MU", "WDC"], ai_score=10.0,
        )]
        svd = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        htd = build_held_theme_decay(
            arts, held_tickers=["MU", "WDC"], now=NOW,
        )
        mc = next(b for b in svd["buckets"] if b["name"] == "memory_core")
        sum_held = sum(h["fresh_score"] for h in htd["holds"])
        assert abs(mc["fresh_score"] - sum_held) < 1e-3


class TestTopBucketSurfacing:
    """top_accelerating, top_decelerating, rotating_in/out lists."""

    def test_rotation_in_and_out_simultaneously(self):
        # memory_core ACCELERATING (prior small, fresh large),
        # semis_equipment DECELERATING (prior large above floor,
        # fresh small).
        arts = [
            # memory_core: prior tiny + fresh loud.
            _art(ts_h_ago=6.1, tickers=["MU"], ai_score=2.0),
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=9.0),
            # semis_equipment: prior loud (well above 4.0 floor),
            # fresh quiet. Use 4 different tickers @ 6.1h with high
            # ai_score so the bucket prior crosses the 4.0 floor.
            _art(ts_h_ago=6.1, tickers=["LRCX"], ai_score=10.0),
            _art(ts_h_ago=6.1, tickers=["AMAT"], ai_score=10.0),
            _art(ts_h_ago=6.1, tickers=["KLAC"], ai_score=10.0),
            _art(ts_h_ago=0.0, tickers=["LRCX"], ai_score=0.5),
        ]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        assert "memory_core" in out["rotating_in"]
        assert "semis_equipment" in out["rotating_out"]
        assert out["top_accelerating"]["name"] == "memory_core"
        assert out["top_decelerating"]["name"] == "semis_equipment"
        # Headline surfaces both rotations.
        assert "ACCELERATING" in out["headline"]
        assert "DECELERATING" in out["headline"]

    def test_top_accelerating_picks_loudest_fresh(self):
        # Two ACCELERATING buckets — the one with larger fresh_score wins.
        arts = [
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=5.0),    # mc ≈ 5
            _art(ts_h_ago=0.0, tickers=["NVDA"], ai_score=9.0),  # design ≈ 4.5
        ]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        # memory_core fresh 5.0 wins over design fresh 9.0 / 1 = 4.5
        # (NVDA is single-ticker article so no split; per-bucket
        # weight = 9.0 / 1 = 9.0 — design wins). Confirm reality.
        # Design: NVDA fresh = 9.0 (single ticker, full weight).
        # memory_core: MU fresh = 5.0.
        # → design fresh 9.0 > memory_core fresh 5.0; design wins.
        assert out["top_accelerating"]["name"] == "design"

    def test_no_acceleration_returns_none_top(self):
        # Only STABLE / DARK / FADING.
        arts = [
            _art(ts_h_ago=7.0, tickers=["MU"], ai_score=5.0),
            _art(ts_h_ago=5.0, tickers=["MU"], ai_score=5.0),
        ]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        assert out["top_accelerating"] is None
        assert out["top_decelerating"] is None
        assert out["rotating_in"] == []
        assert out["rotating_out"] == []


class TestSortOrder:
    """ACCELERATING > BUILDING > STABLE > FADING > DECELERATING > DARK
    in the buckets list (within each, descending fresh_score)."""

    def test_accelerating_at_top(self):
        arts = [
            # memory_core ACCELERATING (fresh 9 > 3.0 floor, no prior).
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=9.0),
            # semis_equipment STABLE (prior + fresh both ~2.2/2.8).
            _art(ts_h_ago=7.0, tickers=["LRCX"], ai_score=5.0),
            _art(ts_h_ago=5.0, tickers=["LRCX"], ai_score=5.0),
        ]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        # First bucket in the list must be the ACCELERATING one.
        assert out["buckets"][0]["name"] == "memory_core"
        assert out["buckets"][0]["verdict"] == "ACCELERATING"


class TestBacktestDefenseInDepth:
    """A leaked synthetic row must not corrupt the bucket view."""

    def test_backtest_url_dropped(self):
        arts = [_art(
            ts_h_ago=0.0, tickers=["MU"], ai_score=9.0,
            url="backtest://run_42/2025-09-15/BUY/MU",
        )]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        # No qualifying articles → NO_DATA.
        assert out["state"] == "NO_DATA"

    def test_backtest_source_dropped(self):
        arts = [_art(
            ts_h_ago=0.0, tickers=["MU"], ai_score=9.0,
            source="backtest_run_42_winner",
        )]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        assert out["state"] == "NO_DATA"

    def test_opus_annotation_dropped(self):
        arts = [_art(
            ts_h_ago=0.0, tickers=["MU"], ai_score=9.0,
            source="opus_annotation_cycle_5",
        )]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        assert out["state"] == "NO_DATA"


class TestShapeStability:
    """Response shape stable across input — UI / panel-health rely on
    keys being present."""

    EXPECTED_KEYS = {
        "as_of", "fresh_window_hours", "prior_window_hours",
        "decay_half_life_hours", "accel_ratio", "build_ratio",
        "fade_ratio", "decel_ratio", "min_fresh_score", "state",
        "buckets", "n_buckets", "n_accelerating", "n_building",
        "n_decelerating", "n_fading", "n_dark", "n_stable",
        "rotating_in", "rotating_out", "top_accelerating",
        "top_decelerating", "headline",
    }

    def test_no_data_path_has_all_keys(self):
        out = build_sector_velocity_delta([], now=NOW, buckets=TEST_BUCKETS)
        assert self.EXPECTED_KEYS <= set(out.keys())

    def test_ok_path_has_all_keys(self):
        arts = [_art(ts_h_ago=0.0, tickers=["MU"], ai_score=9.0)]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        assert self.EXPECTED_KEYS <= set(out.keys())

    def test_garbage_articles_dont_raise(self):
        garbage = [None, "string", {"no_first_seen": True},
                   {"first_seen": "not-iso", "ai_score": 1.0,
                    "tickers": ["MU"]},
                   {"first_seen": NOW.isoformat(), "ai_score": "bad",
                    "tickers": "not-a-list"}]
        out = build_sector_velocity_delta(
            garbage, now=NOW, buckets=TEST_BUCKETS,
        )
        assert out["state"] in ("NO_DATA", "OK")

    def test_articles_outside_both_windows_ignored(self):
        arts = [_art(ts_h_ago=24.0, tickers=["MU"], ai_score=10.0)]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        assert out["state"] == "NO_DATA"

    def test_ratio_none_when_prior_zero(self):
        arts = [_art(ts_h_ago=0.0, tickers=["MU"], ai_score=9.0)]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        mc = next(b for b in out["buckets"] if b["name"] == "memory_core")
        assert mc["ratio"] is None  # never +inf

    def test_top_fresh_ticker_is_loudest_contributor(self):
        # 2 articles in memory_core: MU 9.0 + WDC 5.0. MU is the
        # bucket's top contributor.
        arts = [
            _art(ts_h_ago=0.0, tickers=["MU"], ai_score=9.0),
            _art(ts_h_ago=0.0, tickers=["WDC"], ai_score=5.0),
        ]
        out = build_sector_velocity_delta(
            arts, now=NOW, buckets=TEST_BUCKETS,
        )
        mc = next(b for b in out["buckets"] if b["name"] == "memory_core")
        assert mc["top_fresh_ticker"] == "MU"


class TestParamClamps:
    """Builder clamps out-of-range knobs (dashboard layer also clamps;
    builder is the SSOT for safe defaults)."""

    def test_fresh_window_clamp(self):
        out = build_sector_velocity_delta(
            [], now=NOW, fresh_window_hours=0.0, buckets=TEST_BUCKETS,
        )
        assert out["fresh_window_hours"] >= 0.5
