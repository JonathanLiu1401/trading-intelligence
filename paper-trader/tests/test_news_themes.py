"""Tests for paper_trader.analytics.news_themes.

Per-ticker theme aggregation over the live news feed: which tickers are
the wire spending its breath on right now? Asserts exact arithmetic of
the recency-decayed score sum, held-vs-unheld classification, NO_DATA
honesty, and degrade-never-raise on garbage rows.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.news_themes import (
    DECAY_HALF_LIFE_HOURS,
    build_news_themes,
)

NOW = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)


def _art(*, ts_h_ago, tickers, ai_score=5.0, urgency=0, title="hl",
         url="u", source="src"):
    ts = (NOW - timedelta(hours=ts_h_ago)).isoformat()
    return {
        "title": title,
        "url": url,
        "source": source,
        "ai_score": ai_score,
        "urgency": urgency,
        "first_seen": ts,
        "tickers": list(tickers),
    }


class TestStateLadder:
    def test_no_data_on_empty_articles(self):
        out = build_news_themes([], held_tickers=[], now=NOW)
        assert out["state"] == "NO_DATA"
        assert out["themes"] == []
        assert out["n_articles"] == 0

    def test_no_data_on_zero_recency_articles(self):
        # Articles outside the window (older than window_hours).
        arts = [_art(ts_h_ago=72.0, tickers=["NVDA"], ai_score=8.0)]
        out = build_news_themes(arts, held_tickers=[], now=NOW,
                                window_hours=4.0)
        assert out["state"] == "NO_DATA"

    def test_ok_when_at_least_one_theme(self):
        arts = [_art(ts_h_ago=1.0, tickers=["NVDA"], ai_score=8.0)]
        out = build_news_themes(arts, held_tickers=[], now=NOW)
        assert out["state"] == "OK"
        assert len(out["themes"]) >= 1


class TestDecay:
    def test_fresh_article_carries_more_weight_than_old(self):
        fresh = _art(ts_h_ago=0.0, tickers=["AMD"], ai_score=8.0,
                     title="fresh")
        old = _art(ts_h_ago=DECAY_HALF_LIFE_HOURS, tickers=["NVDA"],
                   ai_score=8.0, title="old")
        out = build_news_themes([fresh, old], held_tickers=[], now=NOW)
        # AMD fresh weight = 8.0 * exp(0) = 8.0
        # NVDA old weight  = 8.0 * exp(-ln(2)) = 4.0
        by_ticker = {t["ticker"]: t for t in out["themes"]}
        assert abs(by_ticker["AMD"]["decayed_score"] - 8.0) < 1e-3
        assert abs(by_ticker["NVDA"]["decayed_score"] - 4.0) < 1e-3

    def test_themes_sorted_by_decayed_score(self):
        arts = [
            _art(ts_h_ago=0.5, tickers=["MU"], ai_score=3.0),
            _art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=9.0),
            _art(ts_h_ago=0.5, tickers=["AMD"], ai_score=6.0),
        ]
        out = build_news_themes(arts, held_tickers=[], now=NOW)
        tickers = [t["ticker"] for t in out["themes"]]
        assert tickers[0] == "NVDA"
        assert tickers[-1] == "MU"


class TestAggregation:
    def test_per_ticker_count_aggregation(self):
        arts = [
            _art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=7.0),
            _art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=6.0),
            _art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=8.0),
            _art(ts_h_ago=0.5, tickers=["AMD"], ai_score=5.0),
        ]
        out = build_news_themes(arts, held_tickers=[], now=NOW)
        by = {t["ticker"]: t for t in out["themes"]}
        assert by["NVDA"]["n_articles"] == 3
        assert by["AMD"]["n_articles"] == 1

    def test_max_urgency_aggregation(self):
        arts = [
            _art(ts_h_ago=0.5, tickers=["TSM"], ai_score=5.0, urgency=0),
            _art(ts_h_ago=0.5, tickers=["TSM"], ai_score=5.0, urgency=2),
            _art(ts_h_ago=0.5, tickers=["TSM"], ai_score=5.0, urgency=1),
        ]
        out = build_news_themes(arts, held_tickers=[], now=NOW)
        assert out["themes"][0]["max_urgency"] == 2

    def test_top_title_is_highest_decayed_score_article(self):
        arts = [
            _art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=5.0,
                 title="meh", url="u1"),
            _art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=9.0,
                 title="big", url="u2"),
            _art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=4.0,
                 title="small", url="u3"),
        ]
        out = build_news_themes(arts, held_tickers=[], now=NOW)
        theme = out["themes"][0]
        assert theme["top_title"] == "big"
        assert theme["top_url"] == "u2"

    def test_multi_ticker_article_split_evenly(self):
        # One article mentioning 4 tickers should contribute its score
        # SPLIT across the four — not full weight to each (avoids a
        # 4-ticker article inflating four themes simultaneously).
        arts = [_art(ts_h_ago=0.0, tickers=["A", "B", "C", "D"],
                     ai_score=8.0)]
        out = build_news_themes(arts, held_tickers=[], now=NOW)
        for theme in out["themes"]:
            # 8.0 / 4 = 2.0 contributed to each ticker
            assert abs(theme["decayed_score"] - 2.0) < 1e-3


class TestHeldFlag:
    def test_held_flag_set_for_held_tickers(self):
        arts = [
            _art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=7.0),
            _art(ts_h_ago=0.5, tickers=["AAPL"], ai_score=6.0),
        ]
        out = build_news_themes(arts, held_tickers=["NVDA"], now=NOW)
        by = {t["ticker"]: t for t in out["themes"]}
        assert by["NVDA"]["held"] is True
        assert by["AAPL"]["held"] is False

    def test_held_lookup_case_insensitive(self):
        arts = [_art(ts_h_ago=0.5, tickers=["MU"], ai_score=7.0)]
        out = build_news_themes(arts, held_tickers=["mu"], now=NOW)
        assert out["themes"][0]["held"] is True

    def test_unheld_themes_count_separately(self):
        arts = [
            _art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=7.0),
            _art(ts_h_ago=0.5, tickers=["AAPL"], ai_score=8.0),
            _art(ts_h_ago=0.5, tickers=["MSFT"], ai_score=4.0),
        ]
        out = build_news_themes(arts, held_tickers=["NVDA"], now=NOW)
        assert out["n_held_themes"] == 1
        assert out["n_unheld_themes"] == 2
        # The top unheld theme is the highest-scored ticker we don't hold.
        assert out["top_unheld_ticker"] == "AAPL"


class TestSummary:
    def test_summary_block_present(self):
        arts = [_art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=7.0)]
        out = build_news_themes(arts, held_tickers=["NVDA"], now=NOW)
        assert "total_decayed_score" in out
        assert "n_articles" in out
        assert out["n_articles"] == 1
        # Total ≈ 7.0 * decay(0.5h) ≈ 7.0 * exp(-0.5/HALFLIFE * ln 2)
        import math
        expected = 7.0 * math.exp(-0.5 / DECAY_HALF_LIFE_HOURS * math.log(2))
        assert abs(out["total_decayed_score"] - round(expected, 4)) < 0.01

    def test_max_themes_clipping(self):
        # 15 tickers, max_themes=5 → only top 5 surface.
        arts = [_art(ts_h_ago=0.5, tickers=[f"T{i}"], ai_score=float(i + 1))
                for i in range(15)]
        out = build_news_themes(arts, held_tickers=[], now=NOW,
                                max_themes=5)
        assert len(out["themes"]) == 5
        # Highest-scored first: ai_score 15 (ticker T14).
        assert out["themes"][0]["ticker"] == "T14"


class TestBacktestFilter:
    def test_synthetic_backtest_rows_dropped(self):
        # If a backtest URL leaks in, the builder MUST drop it (defense
        # in depth — the canonical filter is in the SQL, this is the
        # last gate before user-facing JSON).
        arts = [
            _art(ts_h_ago=0.5, tickers=["NVDA"], ai_score=8.0),
            {
                "title": "synthetic", "url": "backtest://run_99/x/y/z",
                "source": "backtest_run_99_winner", "ai_score": 9.0,
                "urgency": 1, "first_seen": (NOW - timedelta(hours=0.5)).isoformat(),
                "tickers": ["FAKE"],
            },
            {
                "title": "opus", "url": "real://x",
                "source": "opus_annotation_cycle_3",
                "ai_score": 10.0, "urgency": 2,
                "first_seen": (NOW - timedelta(hours=0.5)).isoformat(),
                "tickers": ["FAKE2"],
            },
        ]
        out = build_news_themes(arts, held_tickers=[], now=NOW)
        tickers = [t["ticker"] for t in out["themes"]]
        assert "FAKE" not in tickers
        assert "FAKE2" not in tickers
        assert "NVDA" in tickers


class TestDegradeNeverRaises:
    def test_none_input_returns_no_data(self):
        out = build_news_themes(None, held_tickers=None, now=NOW)
        assert out["state"] == "NO_DATA"

    def test_garbage_row_skipped(self):
        arts = [
            {"title": "good", "tickers": ["A"], "ai_score": 5.0,
             "first_seen": (NOW - timedelta(hours=0.5)).isoformat()},
            "not a dict",
            {"title": "no time"},
            {"first_seen": "junk-not-a-timestamp", "tickers": ["B"]},
        ]
        out = build_news_themes(arts, held_tickers=[], now=NOW)
        assert out["state"] == "OK"
        assert {t["ticker"] for t in out["themes"]} == {"A"}

    def test_no_tickers_field_counts_separately(self):
        # Article with empty/missing tickers contributes to a counter,
        # not to any ticker's score.
        arts = [
            _art(ts_h_ago=0.5, tickers=[], ai_score=8.0),
            _art(ts_h_ago=0.5, tickers=["A"], ai_score=4.0),
        ]
        out = build_news_themes(arts, held_tickers=[], now=NOW)
        assert out["n_articles_with_no_tickers"] == 1
        # Total only counts the ticker'd article's contribution.
        assert {t["ticker"] for t in out["themes"]} == {"A"}
