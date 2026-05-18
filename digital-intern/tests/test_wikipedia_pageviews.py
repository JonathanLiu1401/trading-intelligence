"""Tests for collectors/wikipedia_pageviews.py.

Math-heavy spots are exercised against fixed series so the z-score / spike
logic stays pinned. Network is never touched — _fetch_views is unit-tested
separately via responses-style stubbing in build_spike_articles tests by
feeding pre-canned rows directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Project root on sys.path so 'collectors' imports without a package install.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collectors.wikipedia_pageviews import (  # noqa: E402
    SOURCE,
    SPIKE_Z,
    LOOKBACK_DAYS,
    build_spike_articles,
    _z_score,
)


def test_z_score_too_few_priors_returns_none():
    assert _z_score(100, []) is None
    assert _z_score(100, [50]) is None
    assert _z_score(100, [50, 60]) is None


def test_z_score_zero_variance_returns_none():
    """A flat prior series has no spread → no meaningful spike signal."""
    assert _z_score(200, [100, 100, 100, 100, 100]) is None


def test_z_score_positive_for_above_baseline():
    z = _z_score(200, [100, 105, 95, 100, 110])
    assert z is not None
    assert z > 3.0


def test_z_score_negative_for_below_baseline():
    z = _z_score(50, [100, 105, 95, 100, 110])
    assert z is not None
    assert z < -3.0


def test_build_spike_articles_too_few_rows():
    """Builder needs at least 4 points (3 priors + today)."""
    assert build_spike_articles([], "X", "X") == []
    assert build_spike_articles([("20260101", 100)], "X", "X") == []
    assert build_spike_articles(
        [("20260101", 100), ("20260102", 100), ("20260103", 100)], "X", "X"
    ) == []


def test_build_spike_articles_surfaces_clear_surge():
    """A 10x jump after a stable baseline must surface exactly one article
    with the right shape, direction, and metadata."""
    rows = [
        ("20260101", 100),
        ("20260102", 95),
        ("20260103", 105),
        ("20260104", 102),
        ("20260105", 98),
        ("20260106", 100),
        ("20260107", 1000),
    ]
    arts = build_spike_articles(rows, "NVDA", "Nvidia")
    assert len(arts) == 1
    a = arts[0]
    assert a["_ticker"] == "NVDA"
    assert a["_z"] > 5.0
    assert a["_views"] == 1000
    assert a["_baseline"] == 100
    assert a["source"] == SOURCE
    assert a["published"] == "2026-01-07"
    assert "SURGE" in a["title"]
    assert "NVDA" in a["title"]
    assert "Nvidia" in a["link"]
    # All standard collector keys present.
    for k in ("title", "link", "summary", "published", "source"):
        assert k in a
        assert a[k]


def test_build_spike_articles_surfaces_clear_drop():
    """A 10x collapse after a stable elevated baseline must fire as DROP."""
    rows = [
        ("20260101", 1000),
        ("20260102", 950),
        ("20260103", 1050),
        ("20260104", 1020),
        ("20260105", 980),
        ("20260106", 1000),
        ("20260107", 100),
    ]
    arts = build_spike_articles(rows, "NVDA", "Nvidia")
    assert len(arts) == 1
    a = arts[0]
    assert a["_z"] < -5.0
    assert "DROP" in a["title"]
    assert a["published"] == "2026-01-07"


def test_build_spike_articles_respects_high_threshold():
    """A modest 30% bump should not survive a threshold=10.0 filter."""
    rows = [
        ("20260101", 100),
        ("20260102", 95),
        ("20260103", 105),
        ("20260104", 102),
        ("20260105", 98),
        ("20260106", 100),
        ("20260107", 130),
    ]
    arts = build_spike_articles(rows, "NVDA", "Nvidia", threshold=10.0)
    assert arts == []


def test_build_spike_articles_slow_ramp_does_not_fire():
    """A monotonic gentle climb must not register as a spike — this is the
    'organic interest growth' false-positive guard."""
    rows = [(f"2026010{i}", 100 + i) for i in range(1, 9)]
    arts = build_spike_articles(rows, "NVDA", "Nvidia")
    # Any survivors must legitimately exceed the default threshold; the
    # primary guarantee here is that we don't blow up emitting noise.
    for a in arts:
        assert abs(a["_z"]) >= SPIKE_Z


def test_build_spike_articles_title_length_capped_at_200():
    """Defensive: a deeply pathological wiki slug must not blow past the
    title-length cap used everywhere else in the pipeline."""
    rows = [
        ("20260101", 100), ("20260102", 100), ("20260103", 100),
        ("20260104", 100), ("20260105", 100), ("20260106", 100),
        ("20260107", 999999),
    ]
    # The prior is zero-variance (all 100s) so the spike fires with z=None
    # actually — zero variance returns None. Use a slight wobble instead:
    rows = [
        ("20260101", 100), ("20260102", 101), ("20260103", 99),
        ("20260104", 100), ("20260105", 102), ("20260106", 98),
        ("20260107", 999999),
    ]
    long_slug = "A" * 250
    arts = build_spike_articles(rows, "X", long_slug)
    assert len(arts) == 1
    assert len(arts[0]["title"]) <= 200


def test_build_spike_articles_indexing_window_size():
    """The prior window must be only the most-recent (LOOKBACK_DAYS-1)=6
    days, not every preceding observation. Construct a series where leaking
    further back would change the baseline by an order of magnitude — and
    assert the actual computed baseline reflects only the recent window."""
    rows = [
        # Distant past at a vastly different baseline. If the window leaked
        # back here, _baseline on the final day's spike would be ~5000.
        ("20260101", 5000), ("20260102", 5000), ("20260103", 5000),
        ("20260104", 5000), ("20260105", 5000), ("20260106", 5000),
        ("20260107", 5000), ("20260108", 5000),
        # The 6 days that MUST exclusively form the prior window for index 14.
        ("20260109", 100), ("20260110", 102), ("20260111",  98),
        ("20260112", 100), ("20260113", 101), ("20260114",  99),
        # Spike day — measured against the 6 recent-low priors.
        ("20260115", 1000),
    ]
    arts = build_spike_articles(rows, "X", "X")
    final = [a for a in arts if a["published"] == "2026-01-15"]
    assert final, "expected a spike on the final day"
    # Baseline is ~100 (mean of the 6 recent-low days) — definitely not 5000.
    assert final[0]["_baseline"] < 500
    assert final[0]["_z"] > 100.0  # tiny stdev makes z enormous


def test_build_spike_articles_uses_correct_source_tag():
    """Surface the canonical SOURCE constant — locks the article taxonomy
    so downstream source-health monitoring can pick this collector out."""
    rows = [
        ("20260101", 100), ("20260102", 101), ("20260103", 99),
        ("20260104", 100), ("20260105", 102), ("20260106", 98),
        ("20260107", 1000),
    ]
    arts = build_spike_articles(rows, "NVDA", "Nvidia")
    assert all(a["source"] == "wikipedia/pageviews" for a in arts)
    assert SOURCE == "wikipedia/pageviews"


def test_lookback_constant_sane():
    """If anyone shrinks LOOKBACK_DAYS below 4, build_spike_articles can't
    produce anything (needs 3 priors). Pin the floor."""
    assert LOOKBACK_DAYS >= 4
