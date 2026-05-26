"""Tests for analytics/sentiment_reversal.py — per-ticker avg ml_score
sign-flip detector across two consecutive 2h windows.

Critical regressions to pin:
  * sign-flip detection (a true neg→pos and pos→neg flip must fire);
  * MIN_DELTA gate (a tiny flip from -0.01 to +0.01 must not fire);
  * MIN_ARTICLES gate (a one-article window must not produce a verdict);
  * windowing (an article older than 2× WINDOW_HOURS must be dropped);
  * ticker extraction (STOPwords / lowercase / short tokens excluded);
  * non-actionable rows → chat helper returns ``[]`` (silence precedent);
  * chat helper emits the verbatim direction/avg/article fields, not a
    re-derived verdict.

Pure-helper tests — no Flask (project_digital_intern_chat_enrichment_pattern).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analytics.sentiment_reversal import (  # noqa: E402
    MIN_ARTICLES,
    MIN_DELTA,
    WINDOW_HOURS,
    _extract_tickers,
    build_sentiment_reversal,
)
from dashboard.web_server import _sentiment_reversal_chat_lines  # noqa: E402


NOW = datetime(2026, 5, 25, 18, 0, tzinfo=timezone.utc)


def _at(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _a(title: str, ml_score: float, hours_ago: float) -> dict:
    return {"title": title, "ml_score": ml_score, "first_seen": _at(hours_ago)}


class TestExtractTickers:
    def test_basic_extraction(self):
        assert "NVDA" in _extract_tickers("NVDA beats earnings")
        assert "MU" in _extract_tickers("$MU surge after HBM news")

    def test_stopwords_excluded(self):
        # Stopwords from the same STOP set as trend_velocity must be skipped.
        assert "CEO" not in _extract_tickers("CEO of NVDA speaks")
        assert "IPO" not in _extract_tickers("NVDA IPO rumour false")
        assert "FED" not in _extract_tickers("FED meeting Tuesday")

    def test_too_short_excluded(self):
        # Single-letter or zero-letter tokens never match.
        # The regex itself caps at 2..5 char tokens.
        assert _extract_tickers("A") == []
        assert _extract_tickers("") == []


class TestBuilderEmpty:
    def test_empty_articles_returns_zero_reversals(self):
        r = build_sentiment_reversal([], now=NOW)
        assert r["reversals_found"] == 0
        assert r["reversals"] == []
        assert r["rows_scanned"] == 0
        assert r["window_hours"] == WINDOW_HOURS
        # Always carry the gate values so the chat helper can describe them.
        assert r["min_articles_per_window"] == MIN_ARTICLES
        assert r["min_delta"] == MIN_DELTA


class TestSignFlip:
    def _prev_neg_curr_pos(self) -> list[dict]:
        # PREV window (3h ago, ~middle of [2h, 4h)) → strongly negative.
        # CURR window (0.5h ago, in [0, 2h))         → strongly positive.
        rows = []
        for _ in range(MIN_ARTICLES):
            rows.append(_a("NVDA selloff broadens", -3.0, 3.0))
        for _ in range(MIN_ARTICLES):
            rows.append(_a("NVDA rallies on HBM beat", +3.0, 0.5))
        return rows

    def test_neg_to_pos_flip_fires(self):
        r = build_sentiment_reversal(self._prev_neg_curr_pos(), now=NOW)
        assert r["reversals_found"] == 1
        rev = r["reversals"][0]
        assert rev["ticker"] == "NVDA"
        assert rev["direction"] == "neg→pos"
        assert rev["avg_prev"] < 0 < rev["avg_curr"]
        assert rev["articles_prev"] == MIN_ARTICLES
        assert rev["articles_curr"] == MIN_ARTICLES
        # delta = curr - prev should be a positive, large number.
        assert rev["delta"] >= MIN_DELTA
        assert rev["delta"] == round(rev["avg_curr"] - rev["avg_prev"], 4)

    def test_pos_to_neg_flip_fires(self):
        rows = []
        for _ in range(MIN_ARTICLES):
            rows.append(_a("MU upgraded sharply", +2.5, 3.0))
        for _ in range(MIN_ARTICLES):
            rows.append(_a("MU plunges on guide cut", -2.5, 0.5))
        r = build_sentiment_reversal(rows, now=NOW)
        assert r["reversals_found"] == 1
        rev = r["reversals"][0]
        assert rev["ticker"] == "MU"
        assert rev["direction"] == "pos→neg"
        assert rev["avg_prev"] > 0 > rev["avg_curr"]


class TestNoFlip:
    def test_same_sign_does_not_fire(self):
        # Both windows positive — sign agrees, no flip.
        rows = []
        for _ in range(MIN_ARTICLES):
            rows.append(_a("AMD rising on AI demand", +2.0, 3.0))
        for _ in range(MIN_ARTICLES):
            rows.append(_a("AMD ATH on Nvidia partnership", +1.0, 0.5))
        r = build_sentiment_reversal(rows, now=NOW)
        assert r["reversals_found"] == 0

    def test_tiny_flip_below_min_delta_does_not_fire(self):
        # Sign DOES flip but the magnitudes are tiny — must respect MIN_DELTA.
        rows = []
        for _ in range(MIN_ARTICLES):
            rows.append(_a("INTC neutral note", -0.02, 3.0))
        for _ in range(MIN_ARTICLES):
            rows.append(_a("INTC neutral note 2", +0.02, 0.5))
        r = build_sentiment_reversal(rows, now=NOW)
        # |delta| ~= 0.04 < MIN_DELTA (0.15) → no fire.
        assert r["reversals_found"] == 0

    def test_one_article_per_window_does_not_fire(self):
        # Sign flips but each window has only 1 article — below MIN_ARTICLES.
        rows = [
            _a("TSLA gloom", -2.0, 3.0),
            _a("TSLA pop", +2.0, 0.5),
        ]
        r = build_sentiment_reversal(rows, now=NOW)
        assert r["reversals_found"] == 0

    def test_out_of_window_articles_ignored(self):
        # An article older than 2 × WINDOW_HOURS must not seed the PREV
        # window — otherwise an 8h-old item could create a phantom reversal.
        rows = []
        for _ in range(MIN_ARTICLES):
            rows.append(_a("NVDA crash years ago", -5.0, 8.0))  # too old
        for _ in range(MIN_ARTICLES):
            rows.append(_a("NVDA new bull run", +3.0, 0.5))
        r = build_sentiment_reversal(rows, now=NOW)
        # PREV window is empty after the 4h cutoff → can't compare → no fire.
        assert r["reversals_found"] == 0


class TestNullSafety:
    def test_missing_ml_score_skipped(self):
        rows = []
        rows.append({"title": "NVDA", "ml_score": None,
                     "first_seen": _at(0.5)})
        rows.append({"title": "NVDA", "ml_score": None,
                     "first_seen": _at(3.0)})
        r = build_sentiment_reversal(rows, now=NOW)
        assert r["reversals_found"] == 0
        assert r["skipped"] == 2

    def test_malformed_timestamp_skipped(self):
        rows = [{"title": "NVDA", "ml_score": 1.0,
                 "first_seen": "not-a-date"}]
        r = build_sentiment_reversal(rows, now=NOW)
        assert r["reversals_found"] == 0
        assert r["skipped"] == 1


class TestRankingAndCap:
    def test_reversals_sorted_by_abs_delta_desc(self):
        # Two reversals with different magnitudes; the larger must be first.
        rows = []
        # Smaller flip: ABC at ±0.5
        for _ in range(MIN_ARTICLES):
            rows.append(_a("ABCD small dip", -0.5, 3.0))
        for _ in range(MIN_ARTICLES):
            rows.append(_a("ABCD small lift", +0.5, 0.5))
        # Bigger flip: XYZ at ±4.0
        for _ in range(MIN_ARTICLES):
            rows.append(_a("XYZA huge plunge", -4.0, 3.0))
        for _ in range(MIN_ARTICLES):
            rows.append(_a("XYZA huge surge", +4.0, 0.5))
        r = build_sentiment_reversal(rows, now=NOW)
        assert r["reversals_found"] == 2
        assert r["reversals"][0]["ticker"] == "XYZA"
        assert abs(r["reversals"][0]["delta"]) > abs(r["reversals"][1]["delta"])


class TestChatHelperContract:
    def test_non_dict_returns_empty(self):
        assert _sentiment_reversal_chat_lines(None) == []
        assert _sentiment_reversal_chat_lines("oops") == []
        assert _sentiment_reversal_chat_lines([]) == []

    def test_zero_reversals_silence(self):
        # silence precedent — when nothing actionable, the block is omitted.
        r = build_sentiment_reversal([], now=NOW)
        assert _sentiment_reversal_chat_lines(r) == []

    def test_reversals_emit_headline_and_per_ticker_line(self):
        rows = []
        for _ in range(MIN_ARTICLES):
            rows.append(_a("NVDA selloff", -3.0, 3.0))
        for _ in range(MIN_ARTICLES):
            rows.append(_a("NVDA rally", +3.0, 0.5))
        r = build_sentiment_reversal(rows, now=NOW)
        lines = _sentiment_reversal_chat_lines(r)
        # headline + 1 ticker line
        assert len(lines) == 2
        assert "1 ticker(s) flipped sentiment" in lines[0]
        # The per-ticker line carries the BUILDER's own direction + numbers.
        assert "NVDA" in lines[1]
        assert "neg→pos" in lines[1]
        assert "prev" in lines[1] and "curr" in lines[1]

    def test_helper_caps_per_ticker_lines(self):
        # Build 9 reversals; the helper should cap detail lines.
        rows = []
        tickers = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE",
                   "FFFF", "GGGG", "HHHH", "IIII"]
        for t in tickers:
            for _ in range(MIN_ARTICLES):
                rows.append(_a(f"{t} crash", -3.0, 3.0))
            for _ in range(MIN_ARTICLES):
                rows.append(_a(f"{t} pop", +3.0, 0.5))
        r = build_sentiment_reversal(rows, now=NOW)
        assert r["reversals_found"] >= 7
        lines = _sentiment_reversal_chat_lines(r)
        # headline + at most _SENTIMENT_REVERSAL_TOP_SHOWN per-ticker rows
        from dashboard.web_server import _SENTIMENT_REVERSAL_TOP_SHOWN
        assert len(lines) <= 1 + _SENTIMENT_REVERSAL_TOP_SHOWN
