"""Pure-helper tests for the /api/chat held-ticker conviction-decay enrichment.

`build_position_conviction_decay` buckets the last 24h of articles into 4 × 6h
slices per held ticker and reports the avg ai_score per bucket + a coarse
RISING / STABLE / FADING trend. `_position_conviction_decay_chat_lines` renders
the report as chat-context lines.

The surrounding chat handler is one large inline closure, so per the
established design (cf. `_baseline_compare_chat_lines` /
`_macro_calendar_chat_lines`) the logic is a total/pure pair unit-tested here
— no Flask, no :8090, no articles.db. The handler's only contribution is the
guarded sub-fetch + SQL query.

Discriminating locks:
- **bucket boundaries**: 0-6h / 6-12h / 12-18h / 18-24h, oldest → newest in
  the buckets array; articles outside the 24h window are silently dropped.
- **ticker matching**: word-boundary case-insensitive on title; "MUST" does
  NOT match "MU".
- **STABLE / INSUFFICIENT_DATA → silence**: chat budget is finite; only
  RISING / FADING surface as lines. An all-silent block produces [].
- **pure / total**: non-dict / missing fields / unknown trend never raises
  and degrades to silence or the safe subset.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import (  # noqa: E402
    build_position_conviction_decay,
    _position_conviction_decay_chat_lines,
)


_NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _art(title: str, score: float, hours_ago: float) -> dict:
    ts = _NOW - timedelta(hours=hours_ago)
    return {"title": title, "ai_score": score, "first_seen": ts.isoformat()}


# ── builder: bucketization correctness ────────────────────────────────
def test_builder_buckets_by_age_oldest_to_newest():
    """A single ticker with one article per 6h slot — every bucket should
    have n=1 and the avg should match the article's score."""
    arts = [
        _art("MU reports earnings", 9.0, hours_ago=1.0),    # 0-6h → idx 3
        _art("MU memory cycle peak", 7.0, hours_ago=8.0),   # 6-12h → idx 2
        _art("MU DRAM analyst note", 5.0, hours_ago=15.0),  # 12-18h → idx 1
        _art("MU production cut", 3.0, hours_ago=22.0),     # 18-24h → idx 0
    ]
    rep = build_position_conviction_decay(["MU"], arts, now=_NOW)
    assert rep["window_hours"] == 24
    assert rep["bucket_hours"] == 6
    assert len(rep["tickers"]) == 1
    row = rep["tickers"][0]
    assert row["ticker"] == "MU"
    assert row["n_articles"] == 4
    # buckets[0] = 18-24h ... buckets[3] = 0-6h
    assert row["buckets"][0] == {"n": 1, "avg": 3.0}
    assert row["buckets"][1] == {"n": 1, "avg": 5.0}
    assert row["buckets"][2] == {"n": 1, "avg": 7.0}
    assert row["buckets"][3] == {"n": 1, "avg": 9.0}


def test_builder_drops_articles_outside_24h_window():
    """Articles older than 24h must NEVER land in a bucket — otherwise the
    'last 24h' contract is violated and stale stories pollute the trend."""
    arts = [
        _art("MU now-ish", 8.0, hours_ago=2.0),
        _art("MU 25h old — out of window", 9.5, hours_ago=25.0),
        _art("MU 48h old — out of window", 9.9, hours_ago=48.0),
        _art("MU future timestamp — out of window", 9.0, hours_ago=-0.5),
    ]
    rep = build_position_conviction_decay(["MU"], arts, now=_NOW)
    row = rep["tickers"][0]
    assert row["n_articles"] == 1
    # Only the 2h-ago article survives — score 8.0 in bucket idx 3 (0-6h).
    assert row["buckets"][3] == {"n": 1, "avg": 8.0}
    assert row["buckets"][0]["n"] == 0
    assert row["buckets"][0]["avg"] is None


def test_builder_word_boundary_match_avoids_substring_collision():
    """The ticker matcher MUST use word boundaries — 'MU' must not match
    'MUST', 'MUSK', 'MUSEUM', etc. Otherwise held-ticker conviction is
    poisoned by every news story containing the ticker as a substring."""
    arts = [
        _art("MU memory analyst raises target", 8.0, hours_ago=1.0),
        _art("Elon MUSK posts cryptic tweet", 9.5, hours_ago=2.0),
        _art("Investors MUST read this MUSEUM piece", 9.0, hours_ago=3.0),
    ]
    rep = build_position_conviction_decay(["MU"], arts, now=_NOW)
    row = rep["tickers"][0]
    assert row["n_articles"] == 1, f"unexpected matches: {row}"
    assert row["buckets"][3]["avg"] == 8.0


def test_builder_case_insensitive_ticker_match():
    """Titles often lowercase the ticker. Match must survive that."""
    arts = [
        _art("mu hits new high", 7.0, hours_ago=1.0),
        _art("Mu analyst day recap", 8.0, hours_ago=5.0),
    ]
    rep = build_position_conviction_decay(["MU"], arts, now=_NOW)
    assert rep["tickers"][0]["n_articles"] == 2


def test_builder_handles_dict_form_held_tickers():
    """The chat handler extracts held tickers from /api/state.positions —
    a list of dicts with a 'ticker' key. The builder must accept either
    shape so callers don't need to pre-flatten."""
    held = [{"ticker": "NVDA", "qty": 10, "type": "stock"},
            {"ticker": "AMD", "qty": 5, "type": "stock"}]
    arts = [_art("NVDA earnings beat", 9.0, hours_ago=2.0)]
    rep = build_position_conviction_decay(held, arts, now=_NOW)
    tickers = {r["ticker"] for r in rep["tickers"]}
    assert tickers == {"NVDA", "AMD"}


# ── builder: trend classification ─────────────────────────────────────
def test_builder_trend_rising_when_recent_half_dominates():
    """Articles concentrated in the recent 12h with higher avg score →
    RISING. The threshold is +0.5 on (recent_avg − earlier_avg)."""
    arts = [
        _art("MU 1", 9.5, hours_ago=2.0),    # recent half
        _art("MU 2", 9.0, hours_ago=4.0),    # recent half
        _art("MU 3", 5.0, hours_ago=15.0),   # earlier half
        _art("MU 4", 5.5, hours_ago=20.0),   # earlier half
    ]
    rep = build_position_conviction_decay(["MU"], arts, now=_NOW)
    row = rep["tickers"][0]
    assert row["trend"] == "RISING"
    assert row["recent_minus_earlier"] > 0.5


def test_builder_trend_fading_when_earlier_half_dominates():
    arts = [
        _art("MU 1", 4.0, hours_ago=2.0),
        _art("MU 2", 5.0, hours_ago=4.0),
        _art("MU 3", 9.0, hours_ago=15.0),
        _art("MU 4", 9.5, hours_ago=20.0),
    ]
    rep = build_position_conviction_decay(["MU"], arts, now=_NOW)
    row = rep["tickers"][0]
    assert row["trend"] == "FADING"
    assert row["recent_minus_earlier"] < -0.5


def test_builder_trend_stable_within_threshold():
    """Small drift below the threshold → STABLE, not noise-amplified to a
    direction. Discriminating: a +0.3 delta must NOT trip RISING."""
    arts = [
        _art("MU 1", 7.3, hours_ago=2.0),
        _art("MU 2", 7.2, hours_ago=4.0),
        _art("MU 3", 7.0, hours_ago=15.0),
        _art("MU 4", 6.9, hours_ago=20.0),
    ]
    rep = build_position_conviction_decay(["MU"], arts, now=_NOW)
    row = rep["tickers"][0]
    assert row["trend"] == "STABLE"


def test_builder_trend_insufficient_data_when_one_half_empty():
    """A ticker with articles in only the recent half cannot have a trend
    (the comparison needs both halves). Verdict: INSUFFICIENT_DATA, raw
    bucket counts still surface honestly."""
    arts = [
        _art("MU 1", 9.0, hours_ago=2.0),
        _art("MU 2", 9.0, hours_ago=4.0),
    ]
    rep = build_position_conviction_decay(["MU"], arts, now=_NOW)
    row = rep["tickers"][0]
    assert row["trend"] == "INSUFFICIENT_DATA"
    assert "recent_minus_earlier" not in row
    assert row["n_articles"] == 2


def test_builder_trend_insufficient_data_below_min_count():
    """Below 2 total articles → INSUFFICIENT_DATA regardless of distribution."""
    arts = [_art("MU 1", 9.0, hours_ago=2.0)]
    rep = build_position_conviction_decay(["MU"], arts, now=_NOW)
    row = rep["tickers"][0]
    assert row["trend"] == "INSUFFICIENT_DATA"


# ── builder: pure / total contract ────────────────────────────────────
def test_builder_no_held_tickers_returns_empty_tickers():
    rep = build_position_conviction_decay([], [_art("x", 1.0, 1.0)], now=_NOW)
    assert rep["tickers"] == []
    assert rep["window_hours"] == 24


def test_builder_no_articles_returns_zero_bucket_rows():
    """A held ticker with no matching news still appears with all-empty
    buckets and INSUFFICIENT_DATA trend (silence in the chat, but the
    builder's contract is observational completeness)."""
    rep = build_position_conviction_decay(["MU", "NVDA"], [], now=_NOW)
    assert len(rep["tickers"]) == 2
    for row in rep["tickers"]:
        assert row["n_articles"] == 0
        assert row["trend"] == "INSUFFICIENT_DATA"


@pytest.mark.parametrize("bad", [None, "x", 42, object()])
def test_builder_non_list_articles_is_silence(bad):
    rep = build_position_conviction_decay(["MU"], bad, now=_NOW)
    assert rep["tickers"] == []


def test_builder_skips_garbage_article_rows():
    """A malformed row (missing first_seen, non-numeric ai_score) must NOT
    raise — it silently drops out of the aggregation."""
    arts = [
        {"title": "MU good", "ai_score": 8.0,
         "first_seen": (_NOW - timedelta(hours=2)).isoformat()},
        {"title": "MU broken score", "ai_score": "nope",
         "first_seen": (_NOW - timedelta(hours=2)).isoformat()},
        {"title": "MU no timestamp", "ai_score": 5.0},
        "not a dict",
        None,
    ]
    rep = build_position_conviction_decay(["MU"], arts, now=_NOW)
    row = rep["tickers"][0]
    # Two rows have timestamps; the 'broken score' one coerces to 0.0
    # which is a valid count, just a 0 contribution to the avg.
    assert row["n_articles"] == 2


# ── render: contract ──────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], ["a"], (), object()])
def test_render_non_dict_is_silence(bad):
    assert _position_conviction_decay_chat_lines(bad) == []


def test_render_missing_tickers_is_silence():
    assert _position_conviction_decay_chat_lines({}) == []
    assert _position_conviction_decay_chat_lines({"tickers": []}) == []
    assert _position_conviction_decay_chat_lines({"tickers": "wat"}) == []


def test_render_stable_and_insufficient_collapse_to_silence():
    """The chat budget is finite — STABLE and INSUFFICIENT_DATA must NOT
    produce lines (otherwise a quiet held book would push every other
    sub-block off the screen). Only RISING / FADING surface."""
    rep = {
        "tickers": [
            {"ticker": "MU", "trend": "STABLE", "n_articles": 4,
             "recent_minus_earlier": 0.1,
             "buckets": [{"n": 1, "avg": 7.0}] * 4},
            {"ticker": "NVDA", "trend": "INSUFFICIENT_DATA", "n_articles": 1,
             "buckets": [{"n": 0, "avg": None}] * 4},
        ],
    }
    assert _position_conviction_decay_chat_lines(rep) == []


def test_render_rising_includes_ticker_trend_and_buckets():
    rep = {
        "tickers": [
            {
                "ticker": "MU", "trend": "RISING", "n_articles": 4,
                "recent_minus_earlier": 3.5,
                "buckets": [
                    {"n": 1, "avg": 5.5}, {"n": 1, "avg": 5.0},
                    {"n": 1, "avg": 9.0}, {"n": 1, "avg": 9.5},
                ],
            }
        ],
    }
    out = _position_conviction_decay_chat_lines(rep)
    assert len(out) == 1
    line = out[0]
    assert "MU" in line
    assert "RISING" in line
    assert "n=4" in line
    # Bucket render is oldest → newest so the chat reads time-forward.
    assert "18-24h" in line and "0-6h" in line
    # 0-6h bucket's avg must appear so the analyst can audit the verdict.
    assert "9.5" in line


def test_render_does_not_fabricate_buckets_when_shape_is_wrong():
    """If the builder's bucket array is malformed (wrong length, non-dicts),
    the renderer drops the bucket render rather than crashing or inventing
    numbers. Verdict line must still appear."""
    rep = {
        "tickers": [
            {"ticker": "MU", "trend": "FADING", "n_articles": 3,
             "recent_minus_earlier": -1.2,
             "buckets": [{"n": 1, "avg": 5.0}]},  # only 1 bucket
        ],
    }
    out = _position_conviction_decay_chat_lines(rep)
    assert len(out) == 1
    assert "FADING" in out[0]
    # No fabricated "0-6h" string from a missing bucket.
    assert "0-6h" not in out[0]


def test_render_multiple_tickers_each_get_their_own_line():
    rep = {
        "tickers": [
            {"ticker": "MU", "trend": "RISING", "n_articles": 3,
             "recent_minus_earlier": 2.0,
             "buckets": [{"n": 1, "avg": 6.0}] * 4},
            {"ticker": "NVDA", "trend": "FADING", "n_articles": 4,
             "recent_minus_earlier": -2.0,
             "buckets": [{"n": 1, "avg": 6.0}] * 4},
            {"ticker": "AMD", "trend": "STABLE", "n_articles": 2,
             "recent_minus_earlier": 0.0,
             "buckets": [{"n": 1, "avg": 6.0}] * 4},
        ],
    }
    out = _position_conviction_decay_chat_lines(rep)
    assert len(out) == 2  # STABLE filtered
    blob = "\n".join(out)
    assert "MU" in blob and "RISING" in blob
    assert "NVDA" in blob and "FADING" in blob
    assert "AMD" not in blob


# ── end-to-end pure pipeline ──────────────────────────────────────────
def test_end_to_end_pipeline_with_realistic_held_book():
    """Realistic: a 2-name held book, one trending up, one fading. The
    render output should contain exactly 2 lines — one per non-stable
    trend — and the analyst should see which way each is moving."""
    held = ["MU", "NVDA"]
    arts = [
        # MU: ramping up
        _art("MU memory ASP guidance raise", 9.5, hours_ago=1.0),
        _art("MU analyst day standout", 9.0, hours_ago=3.0),
        _art("MU early note from boutique", 4.0, hours_ago=18.0),
        # NVDA: fading
        _art("NVDA Beijing trip headline", 8.5, hours_ago=22.0),
        _art("NVDA 18h-old recap", 8.0, hours_ago=18.0),
        _art("NVDA muted post-print follow-up", 4.0, hours_ago=2.0),
    ]
    rep = build_position_conviction_decay(held, arts, now=_NOW)
    out = _position_conviction_decay_chat_lines(rep)
    blob = "\n".join(out)
    assert "MU" in blob and "RISING" in blob
    assert "NVDA" in blob and "FADING" in blob
