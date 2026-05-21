"""Tests for analytics/news_arrival_rhythm.py — per-source hour-of-day distribution.

A bucketing regression where urgency floor leaks, where the window cutoff
shifts by an hour, where the circular quiet-window misses the wrap-around,
where the per-source rollup totals don't reconcile to the aggregate, or
where the empty / sparse states are misclassified would all fail an
assertion here. Mirrors the test_event_threads.py / test_portfolio_signals.py
structure — pure builder, injected ``now``, hand-computed arithmetic.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.news_arrival_rhythm import (
    DEFAULT_MIN_URGENCY,
    DEFAULT_TOP_SOURCES,
    DEFAULT_WINDOW_HOURS,
    _longest_zero_run_circular,
    build_news_arrival_rhythm,
)

# An anchor `now` deep inside the hour so window arithmetic is unambiguous.
NOW = datetime(2026, 5, 21, 12, 30, 0, tzinfo=timezone.utc)


def _art(source: str, urgency: int, age_hours: float = 0.0,
         first_seen: datetime | None = None) -> dict:
    if first_seen is None:
        first_seen = NOW - timedelta(hours=age_hours)
    return {
        "source": source,
        "urgency": urgency,
        "first_seen": first_seen.isoformat(),
    }


# ───────────────────── empty / degenerate inputs ────────────────────────

class TestEmptyAndDefensive:
    def test_empty_list_returns_no_data_envelope(self):
        rep = build_news_arrival_rhythm([], now=NOW)
        assert rep["state"] == "NO_DATA"
        assert rep["n_articles_scanned"] == 0
        assert rep["n_articles_kept"] == 0
        assert rep["n_sources"] == 0
        assert rep["hour_of_day_totals"] == [0] * 24
        assert rep["peak_hour"] is None
        assert rep["sources"] == []
        assert "nothing to plot" in rep["headline"].lower()

    def test_non_list_input_does_not_raise(self):
        for bad in (None, 0, "string", 12.3, {"not": "a list"}):
            rep = build_news_arrival_rhythm(bad, now=NOW)  # type: ignore[arg-type]
            assert rep["state"] == "NO_DATA"
            assert rep["hour_of_day_totals"] == [0] * 24

    def test_non_dict_article_rows_skipped(self):
        rep = build_news_arrival_rhythm(
            [None, "string", 42, _art("rss", 2, 1.0)], now=NOW,
        )
        # n_scanned counts dict rows only (the non-dict are filtered at
        # the top of the loop). The kept article is the rss one.
        assert rep["n_articles_kept"] == 1
        # Total scanned counts every dict; the non-dicts are skipped silently.
        assert rep["n_articles_scanned"] >= 1

    def test_invalid_urgency_skipped(self):
        # An urgency of "high" / None should not blow up bucketing.
        arts = [
            _art("rss", "high"),  # type: ignore[arg-type]
            _art("rss", None),  # type: ignore[arg-type]
            _art("rss", 2, age_hours=1.0),
        ]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        assert rep["n_articles_kept"] == 1

    def test_invalid_first_seen_skipped(self):
        arts = [
            {"source": "rss", "urgency": 2, "first_seen": None},
            {"source": "rss", "urgency": 2, "first_seen": "not-a-date"},
            _art("rss", 2, age_hours=1.0),
        ]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        assert rep["n_articles_kept"] == 1

    def test_zero_hours_window_returns_no_data(self):
        # A 0h window is a programmer error; the builder collapses to
        # empty rather than dividing by zero or scanning the full ledger.
        rep = build_news_arrival_rhythm(
            [_art("rss", 2, age_hours=0.5)], hours=0, now=NOW,
        )
        assert rep["state"] == "NO_DATA"
        assert rep["n_articles_kept"] == 0

    def test_constants_have_expected_defaults(self):
        # Pin the documented defaults — a flip should be deliberate.
        assert DEFAULT_WINDOW_HOURS == 24
        assert DEFAULT_MIN_URGENCY == 1
        assert DEFAULT_TOP_SOURCES == 10


# ───────────────────── urgency / window filtering ───────────────────────

class TestFilters:
    def test_min_urgency_floor_excludes_below(self):
        arts = [
            _art("rss", 0, age_hours=1.0),
            _art("rss", 1, age_hours=2.0),
            _art("rss", 2, age_hours=3.0),
        ]
        # Default floor is 1 → keeps 2 of 3.
        rep = build_news_arrival_rhythm(arts, now=NOW)
        assert rep["n_articles_kept"] == 2
        # Floor 2 keeps only the urgency=2 row.
        rep2 = build_news_arrival_rhythm(arts, min_urgency=2, now=NOW)
        assert rep2["n_articles_kept"] == 1

    def test_min_urgency_zero_keeps_everything(self):
        arts = [
            _art("rss", 0, age_hours=1.0),
            _art("rss", 1, age_hours=2.0),
            _art("rss", 2, age_hours=3.0),
        ]
        rep = build_news_arrival_rhythm(arts, min_urgency=0, now=NOW)
        assert rep["n_articles_kept"] == 3

    def test_window_cutoff_excludes_old_articles(self):
        # 24h window: article 25h ago must be dropped; article 23h ago kept.
        arts = [
            _art("rss", 2, age_hours=23.0),
            _art("rss", 2, age_hours=25.0),  # outside window
        ]
        rep = build_news_arrival_rhythm(arts, hours=24, now=NOW)
        assert rep["n_articles_kept"] == 1

    def test_future_articles_excluded(self):
        # An article timestamped after `now` shouldn't be counted (clock
        # skew defense).
        arts = [
            _art("rss", 2, age_hours=1.0),
            _art("rss", 2, first_seen=NOW + timedelta(hours=1)),
        ]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        assert rep["n_articles_kept"] == 1


# ────────────────────── hour-of-day bucketing ───────────────────────────

class TestHourBucketing:
    def test_hour_of_day_uses_utc_hour(self):
        # NOW is 12:30 UTC. Insert articles at distinct hours and verify
        # each lands in its OWN bucket.
        arts = []
        for h in (1, 5, 11):
            # h hours ago → hour-of-day = (12 - h) mod 24 since NOW=12:30
            arts.append(_art("rss", 2, age_hours=h))
        rep = build_news_arrival_rhythm(arts, now=NOW)
        # Expected hours: 12-1=11, 12-5=7, 12-11=1.
        for h in (11, 7, 1):
            assert rep["hour_of_day_totals"][h] == 1
        # All other hours zero.
        for h in range(24):
            if h not in (11, 7, 1):
                assert rep["hour_of_day_totals"][h] == 0

    def test_per_source_sums_match_aggregate(self):
        # Reconciliation check: sum of per-source totals = n_articles_kept
        # = sum of aggregate hour_of_day_totals.
        arts = [
            _art("rss",      2, age_hours=1.5),
            _art("rss",      2, age_hours=2.5),
            _art("gdelt",    1, age_hours=3.5),
            _art("reuters",  2, age_hours=4.5),
        ]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        src_total = sum(s["total"] for s in rep["sources"])
        assert src_total == rep["n_articles_kept"]
        assert sum(rep["hour_of_day_totals"]) == rep["n_articles_kept"]

    def test_peak_hour_is_highest_count(self):
        # 3 articles in hour 11, 1 in hour 7 → peak should be 11.
        arts = [
            _art("rss", 2, age_hours=1.0),
            _art("rss", 2, age_hours=1.5),
            _art("rss", 2, age_hours=1.25),
            _art("rss", 2, age_hours=5.0),
        ]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        assert rep["peak_hour"] == 11
        assert rep["hour_of_day_totals"][11] == 3
        assert rep["hour_of_day_totals"][7] == 1


# ───────────────────────── source ranking ──────────────────────────────

class TestSourceRanking:
    def test_sources_sorted_by_total_descending(self):
        arts = (
            [_art("rss", 2, age_hours=0.5 + i) for i in range(5)] +
            [_art("gdelt", 2, age_hours=0.5 + i) for i in range(3)] +
            [_art("reuters", 2, age_hours=1.0)]
        )
        rep = build_news_arrival_rhythm(arts, now=NOW)
        totals = [s["total"] for s in rep["sources"]]
        assert totals == sorted(totals, reverse=True)
        assert rep["sources"][0]["source"] == "rss"
        assert rep["sources"][0]["total"] == 5

    def test_alphabetical_tie_break(self):
        arts = [
            _art("zebra",  2, age_hours=1.0),
            _art("alpha",  2, age_hours=2.0),
            _art("middle", 2, age_hours=3.0),
        ]
        # All sources tied at total=1; expect alphabetical order.
        rep = build_news_arrival_rhythm(arts, now=NOW)
        ordered = [s["source"] for s in rep["sources"]]
        assert ordered == ["alpha", "middle", "zebra"]

    def test_top_sources_cap_truncates_display_not_aggregate(self):
        # 5 distinct sources; cap to 2 → only 2 cards, but aggregate
        # counts the full 5.
        arts = [_art(f"src{i}", 2, age_hours=0.5 + i) for i in range(5)]
        rep = build_news_arrival_rhythm(arts, top_sources=2, now=NOW)
        assert len(rep["sources"]) == 2
        assert rep["n_articles_kept"] == 5
        # aggregate sums still match the full count.
        assert sum(rep["hour_of_day_totals"]) == 5
        # n_sources reflects the pre-cap distinct count.
        assert rep["n_sources"] == 5

    def test_missing_or_non_string_source_collapses_to_unknown(self):
        arts = [
            {"source": None, "urgency": 2,
             "first_seen": (NOW - timedelta(hours=1)).isoformat()},
            {"source": 42, "urgency": 2,
             "first_seen": (NOW - timedelta(hours=2)).isoformat()},
        ]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        # Both fall into a single '(unknown)' bucket.
        assert rep["n_articles_kept"] == 2
        names = [s["source"] for s in rep["sources"]]
        assert "(unknown)" in names


# ───────────────────── quiet window detection ──────────────────────────

class TestQuietWindow:
    def test_longest_zero_run_simple(self):
        counts = [1, 0, 0, 0, 1] + [1] * 19  # 24 elements
        length, start, end = _longest_zero_run_circular(counts)
        assert length == 3
        assert start == 1
        assert end == 3

    def test_longest_zero_run_wraps_around(self):
        # Zeros at hours 22, 23, 0, 1 → length 4, wrap from 22→1.
        counts = [0, 0] + [1] * 20 + [0, 0]  # 24 elements: 0=zero, 1=zero, 2..21=nonzero, 22=zero, 23=zero
        length, start, end = _longest_zero_run_circular(counts)
        assert length == 4
        # Start must be the FIRST hour of the wrap-stretch (22),
        # end must be the LAST (1).
        assert start == 22
        assert end == 1

    def test_longest_zero_run_all_zero(self):
        assert _longest_zero_run_circular([0] * 24) == (24, 0, 23)

    def test_longest_zero_run_all_nonzero(self):
        assert _longest_zero_run_circular([1] * 24) == (0, -1, -1)

    def test_longest_zero_run_empty(self):
        assert _longest_zero_run_circular([]) == (0, -1, -1)

    def test_envelope_quiet_window_reflects_aggregate(self):
        # Insert articles only at hour 11 (NOW=12:30 UTC, age 1.5h).
        # The other 23 hours are zero → quiet length = 23, the longest run.
        arts = [_art("rss", 2, age_hours=1.5) for _ in range(5)]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        assert rep["quiet_window"]["length_hours"] == 23
        # Hour 11 is the only nonzero, so the gap starts at 12 and ends at 10
        # (wrap-around) for a 23-hour stretch.
        assert rep["quiet_window"]["start_hour"] == 12
        assert rep["quiet_window"]["end_hour"] == 10


# ────────────────────── state + headline ─────────────────────────────

class TestStateAndHeadline:
    def test_sparse_state_below_five_kept(self):
        arts = [_art("rss", 2, age_hours=0.5 + i) for i in range(4)]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        assert rep["state"] == "SPARSE"
        assert "sparse" in rep["headline"].lower()

    def test_stable_state_at_five_kept(self):
        arts = [_art("rss", 2, age_hours=0.5 + i) for i in range(5)]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        assert rep["state"] == "STABLE"
        assert "peak urgent-news hour" in rep["headline"].lower()

    def test_stable_headline_includes_loudest_source(self):
        # rss=5, gdelt=3, reuters=1 → rss is the loudest.
        arts = (
            [_art("rss", 2, age_hours=0.5 + i) for i in range(5)] +
            [_art("gdelt", 2, age_hours=0.5 + i) for i in range(3)] +
            [_art("reuters", 2, age_hours=1.0)]
        )
        rep = build_news_arrival_rhythm(arts, now=NOW)
        assert "rss" in rep["headline"]


# ───────────────────── envelope shape ──────────────────────────────

class TestResponseShape:
    def test_envelope_keys_present_on_empty(self):
        rep = build_news_arrival_rhythm([], now=NOW)
        for key in (
            "as_of", "state", "headline", "window_hours", "min_urgency",
            "n_articles_scanned", "n_articles_kept", "n_sources",
            "hour_of_day_totals", "peak_hour", "trough_hour",
            "quiet_window", "sources", "top_sources_cap",
        ):
            assert key in rep, f"envelope key '{key}' missing"
        for sub in ("length_hours", "start_hour", "end_hour"):
            assert sub in rep["quiet_window"]

    def test_envelope_keys_present_on_populated(self):
        arts = [_art("rss", 2, age_hours=0.5 + i) for i in range(5)]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        # Same keys as the empty envelope — the binding contract must
        # hold regardless of state.
        for key in (
            "as_of", "state", "headline", "window_hours", "min_urgency",
            "n_articles_scanned", "n_articles_kept", "n_sources",
            "hour_of_day_totals", "peak_hour", "trough_hour",
            "quiet_window", "sources", "top_sources_cap",
        ):
            assert key in rep, f"envelope key '{key}' missing on populated"
        # Source rows expose the documented field set.
        for src in rep["sources"]:
            for k in ("source", "total", "hourly_counts",
                      "peak_hour", "n_quiet_hours"):
                assert k in src, f"source-row key '{k}' missing"
            assert len(src["hourly_counts"]) == 24

    def test_as_of_uses_injected_now(self):
        rep = build_news_arrival_rhythm([], now=NOW)
        assert rep["as_of"].startswith("2026-05-21T12:30:00")

    def test_hour_of_day_totals_length_is_24(self):
        rep = build_news_arrival_rhythm([], now=NOW)
        assert len(rep["hour_of_day_totals"]) == 24
        arts = [_art("rss", 2, age_hours=1.0)]
        rep2 = build_news_arrival_rhythm(arts, now=NOW)
        assert len(rep2["hour_of_day_totals"]) == 24


# ────────────────────── timezone tolerance ─────────────────────────

class TestTimestampParsing:
    def test_z_suffixed_iso_string_accepted(self):
        # digital-intern occasionally writes "...Z" in legacy paths;
        # builder must tolerate both "+00:00" and "Z".
        arts = [
            {"source": "rss", "urgency": 2,
             "first_seen": "2026-05-21T12:00:00Z"},
        ]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        assert rep["n_articles_kept"] == 1

    def test_naive_iso_string_treated_as_utc(self):
        # A naive timestamp (no offset) is treated as UTC — same
        # convention as portfolio_signals._parse_ts.
        arts = [
            {"source": "rss", "urgency": 2,
             "first_seen": "2026-05-21T12:00:00"},
        ]
        rep = build_news_arrival_rhythm(arts, now=NOW)
        assert rep["n_articles_kept"] == 1
