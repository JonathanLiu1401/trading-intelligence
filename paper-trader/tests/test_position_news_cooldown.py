"""Tests for analytics/position_news_cooldown.py — per-position news
cooldown diagnostic.

Verdict-threshold branching and rollup logic are locked to exact-value
checks following the test_position_attention.py / test_correlation.py
patterns. The ``now`` argument is injected so all time arithmetic is
deterministic — no wall-clock dependence.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.position_news_cooldown import (  # noqa: E402
    COOL_H,
    FRESH_H,
    MIN_SCORE_THRESHOLD,
    WARM_H,
    _classify,
    build_position_news_cooldown,
)


NOW = datetime(2026, 5, 19, 18, 0, 0, tzinfo=timezone.utc)


def _ts(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


# ───────────────────────────── _classify ─────────────────────────────────


class TestClassify:
    def test_fresh_at_boundary(self):
        # FRESH_H = 6.0 — ≤ FRESH_H is FRESH.
        assert _classify(0.0) == "FRESH"
        assert _classify(FRESH_H) == "FRESH"

    def test_warm_at_boundary(self):
        assert _classify(FRESH_H + 0.01) == "WARM"
        assert _classify(WARM_H) == "WARM"

    def test_cool_at_boundary(self):
        assert _classify(WARM_H + 0.01) == "COOL"
        assert _classify(COOL_H) == "COOL"

    def test_dark_past_cool(self):
        assert _classify(COOL_H + 0.01) == "DARK"
        assert _classify(500.0) == "DARK"

    def test_none_is_dark(self):
        # No news at all ⇒ DARK (worst bucket). This is the load-bearing
        # contract — a ticker missing from the news map must not silently
        # downgrade to "FRESH".
        assert _classify(None) == "DARK"


# ───────────────────── empty / insufficient input ────────────────────────


class TestInsufficient:
    def test_no_positions(self):
        r = build_position_news_cooldown([], {}, now=NOW)
        assert r["verdict"] == "INSUFFICIENT_DATA"
        assert r["n_positions"] == 0
        assert r["positions"] == []
        assert r["summary"] == {"fresh": 0, "warm": 0, "cool": 0, "dark": 0}

    def test_skips_position_with_no_ticker(self):
        # A row with no ticker is skipped silently — same behaviour as
        # position_attention.
        r = build_position_news_cooldown(
            [{"ticker": "", "qty": 1}, {"ticker": None, "qty": 1}],
            {}, now=NOW,
        )
        assert r["n_positions"] == 0
        assert r["verdict"] == "INSUFFICIENT_DATA"


# ─────────────────────── per-position classification ─────────────────────


class TestPerPositionClassification:
    def test_fresh_position(self):
        # 3h since last news (≤ FRESH_H = 6.0) ⇒ FRESH.
        positions = [{"ticker": "NVDA", "type": "stock", "qty": 10}]
        news = {"NVDA": {
            "last_first_seen": _ts(3.0),
            "top_score": 7.5,
            "top_title": "NVDA earnings beat",
            "n_24h": 4,
            "n_72h": 9,
        }}
        r = build_position_news_cooldown(positions, news, now=NOW)
        assert r["n_positions"] == 1
        p = r["positions"][0]
        assert p["ticker"] == "NVDA"
        assert p["verdict"] == "FRESH"
        assert p["hours_since_last_news"] == 3.0
        assert p["top_score_72h"] == 7.5
        assert p["top_title_72h"] == "NVDA earnings beat"
        assert p["n_articles_24h"] == 4
        assert p["n_articles_72h"] == 9
        assert r["summary"] == {"fresh": 1, "warm": 0, "cool": 0, "dark": 0}
        assert r["verdict"] == "OK"

    def test_warm_position(self):
        positions = [{"ticker": "AMD", "type": "stock", "qty": 5}]
        news = {"AMD": {"last_first_seen": _ts(12.0), "top_score": 5.0,
                        "top_title": "x", "n_24h": 1, "n_72h": 3}}
        r = build_position_news_cooldown(positions, news, now=NOW)
        assert r["positions"][0]["verdict"] == "WARM"
        assert r["verdict"] == "OK"  # WARM doesn't escalate the book

    def test_cool_position(self):
        positions = [{"ticker": "MSFT", "type": "stock", "qty": 1}]
        news = {"MSFT": {"last_first_seen": _ts(48.0), "top_score": 4.5,
                         "top_title": "x", "n_24h": 0, "n_72h": 1}}
        r = build_position_news_cooldown(positions, news, now=NOW)
        assert r["positions"][0]["verdict"] == "COOL"
        assert r["verdict"] == "COOLING_BOOK"

    def test_dark_position_via_age(self):
        # >COOL_H ⇒ DARK.
        positions = [{"ticker": "T", "type": "stock", "qty": 100}]
        news = {"T": {"last_first_seen": _ts(COOL_H + 5.0), "top_score": 6.0,
                      "top_title": "x", "n_24h": 0, "n_72h": 0}}
        r = build_position_news_cooldown(positions, news, now=NOW)
        assert r["positions"][0]["verdict"] == "DARK"
        assert r["verdict"] == "DARK_BOOK"

    def test_dark_position_via_missing_news_entry(self):
        # No entry in the news map at all ⇒ DARK (the catch-all branch in
        # _classify(None) must propagate up).
        positions = [{"ticker": "F", "type": "stock", "qty": 50}]
        r = build_position_news_cooldown(positions, {}, now=NOW)
        p = r["positions"][0]
        assert p["verdict"] == "DARK"
        assert p["hours_since_last_news"] is None
        assert p["last_news_ts"] is None
        assert p["n_articles_24h"] == 0
        assert p["n_articles_72h"] == 0
        assert r["verdict"] == "DARK_BOOK"


# ─────────────────────────── rollup verdict ──────────────────────────────


class TestRollup:
    def test_dark_dominates_cool_in_rollup(self):
        positions = [
            {"ticker": "A", "type": "stock", "qty": 1},
            {"ticker": "B", "type": "stock", "qty": 1},
        ]
        news = {
            "A": {"last_first_seen": _ts(40.0), "top_score": 5,
                  "top_title": None, "n_24h": 0, "n_72h": 1},  # COOL
            "B": {"last_first_seen": _ts(120.0), "top_score": 5,
                  "top_title": None, "n_24h": 0, "n_72h": 0},  # DARK
        }
        r = build_position_news_cooldown(positions, news, now=NOW)
        assert r["summary"] == {"fresh": 0, "warm": 0, "cool": 1, "dark": 1}
        assert r["verdict"] == "DARK_BOOK"

    def test_all_fresh_or_warm_is_ok(self):
        positions = [{"ticker": "A", "type": "stock", "qty": 1},
                     {"ticker": "B", "type": "stock", "qty": 1}]
        news = {
            "A": {"last_first_seen": _ts(2.0), "top_score": 8,
                  "top_title": "x", "n_24h": 2, "n_72h": 3},
            "B": {"last_first_seen": _ts(18.0), "top_score": 6,
                  "top_title": "y", "n_24h": 1, "n_72h": 2},
        }
        r = build_position_news_cooldown(positions, news, now=NOW)
        assert r["verdict"] == "OK"
        assert r["summary"]["fresh"] == 1
        assert r["summary"]["warm"] == 1


# ──────────────────────────── sort order ─────────────────────────────────


class TestSortOrder:
    def test_worst_first(self):
        # DARK (with None hours, i.e. never-seen) should sort above DARK
        # (with a finite age) which should sort above COOL, WARM, FRESH.
        positions = [
            {"ticker": "FRESH", "type": "stock", "qty": 1},
            {"ticker": "COOL", "type": "stock", "qty": 1},
            {"ticker": "DARK_OLD", "type": "stock", "qty": 1},
            {"ticker": "DARK_NONE", "type": "stock", "qty": 1},
            {"ticker": "WARM", "type": "stock", "qty": 1},
        ]
        news = {
            "FRESH": {"last_first_seen": _ts(1.0), "top_score": 5,
                      "top_title": None, "n_24h": 0, "n_72h": 1},
            "WARM": {"last_first_seen": _ts(20.0), "top_score": 5,
                     "top_title": None, "n_24h": 0, "n_72h": 1},
            "COOL": {"last_first_seen": _ts(50.0), "top_score": 5,
                     "top_title": None, "n_24h": 0, "n_72h": 1},
            "DARK_OLD": {"last_first_seen": _ts(120.0), "top_score": 5,
                         "top_title": None, "n_24h": 0, "n_72h": 0},
            # DARK_NONE has no entry — silent / never seen.
        }
        r = build_position_news_cooldown(positions, news, now=NOW)
        ordered = [p["ticker"] for p in r["positions"]]
        assert ordered == ["DARK_NONE", "DARK_OLD", "COOL", "WARM", "FRESH"]


# ────────────────────────── metadata surface ─────────────────────────────


class TestMetadata:
    def test_threshold_metadata_exposed(self):
        # The thresholds the verdicts are read off must be in the
        # response so a UI can render the ladder without hard-coding.
        positions = [{"ticker": "X", "type": "stock", "qty": 1}]
        r = build_position_news_cooldown(positions, {}, now=NOW)
        assert r["thresholds_hours"]["fresh_le"] == FRESH_H
        assert r["thresholds_hours"]["warm_le"] == WARM_H
        assert r["thresholds_hours"]["cool_le"] == COOL_H
        assert r["min_score_threshold"] == MIN_SCORE_THRESHOLD

    def test_as_of_present_and_iso(self):
        r = build_position_news_cooldown([], {}, now=NOW)
        # Must round-trip through fromisoformat.
        datetime.fromisoformat(r["as_of"])

    def test_min_score_threshold_overridable(self):
        # Callers can pass a tighter / looser threshold; it's surfaced for
        # the UI so the note text can interpolate the actual value.
        r = build_position_news_cooldown(
            [{"ticker": "A", "type": "stock", "qty": 1}],
            {"A": {"last_first_seen": _ts(2.0), "top_score": 9,
                   "top_title": None, "n_24h": 1, "n_72h": 2}},
            now=NOW,
            min_score_threshold=6.5,
        )
        assert r["min_score_threshold"] == 6.5


# ──────────────────────── ticker uppercase contract ──────────────────────


class TestTickerNormalisation:
    def test_uppercase_lookup(self):
        # Positions can arrive with lowercase tickers; the builder must
        # uppercase them before matching the news map (the upstream caller
        # already passes uppercased keys).
        positions = [{"ticker": "nvda", "type": "stock", "qty": 1}]
        news = {"NVDA": {"last_first_seen": _ts(1.0), "top_score": 9,
                         "top_title": None, "n_24h": 1, "n_72h": 1}}
        r = build_position_news_cooldown(positions, news, now=NOW)
        assert r["positions"][0]["ticker"] == "NVDA"
        assert r["positions"][0]["verdict"] == "FRESH"
