"""Tests for analytics/news_action_funnel.py + /api/news-action-funnel.

Contract:
* Pure ``build_news_action_funnel`` is offline / never raises on garbage.
* State ladder: NO_DATA → OK.
* Verdict ladder per ticker:
  IGNORED (≥MIN articles, 0 decisions)
  DECIDED_NO_FILL (≥MIN articles, ≥1 decision, 0 fills)
  RESPONSIVE (≥MIN articles, ≥1 fill)
  ACTED_WITHOUT_NEWS (<MIN articles, ≥1 fill)
  QUIET (everything else).
* Sort priority IGNORED first then DECIDED_NO_FILL > ACTED_WITHOUT_NEWS >
  RESPONSIVE > QUIET, then within each tier by article count DESC then
  decision count DESC then ticker ASC.
* Word-boundary regex (MU ≠ MUTUAL, AMD ≠ AMDOCS, $NVDA cashtag) matches
  news_velocity / idle_opportunity / trade_attribution.
* Window cutoff strict inclusive (first_seen ≥ now - window_h).
* NaN/Inf ai_score rejected.
* Decision ticker parsed via the dashboard SSOT (NO_DECISION/BLOCKED/CASH
  sentinels never bucket).
* Held flag set when ticker is in held_tickers.
* unrealized_pl carried only for held positions.
* Endpoint Flask test_client: real seeded articles.db, real Store with
  decisions+trades, parameter clamps, live-only SQL filter.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from paper_trader.analytics.news_action_funnel import (
    DEFAULT_MAX_TICKERS,
    DEFAULT_MIN_AI_SCORE,
    DEFAULT_WINDOW_HOURS,
    MIN_ARTICLES_FOR_VERDICT,
    _parse_action_ticker,
    _safe_float,
    _ticker_regex,
    _verdict,
    build_news_action_funnel,
)

NOW = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)


def _article(ticker_in_title: str, ai_score: float, hours_ago: float,
             title_extra: str = "", source: str = "rss",
             url: str | None = None) -> dict:
    fs = NOW - timedelta(hours=hours_ago)
    title = f"{ticker_in_title} {title_extra}".strip()
    return {
        "title": title,
        "ai_score": ai_score,
        "first_seen": fs.isoformat(timespec="seconds"),
        "source": source,
        "url": url or f"https://example.com/{abs(hash(title)) % 99999}",
    }


def _decision(action: str, hours_ago: float) -> dict:
    return {
        "timestamp": (NOW - timedelta(hours=hours_ago))
                     .isoformat(timespec="seconds"),
        "action_taken": action,
    }


def _trade(ticker: str, hours_ago: float, status: str = "FILLED") -> dict:
    return {
        "timestamp": (NOW - timedelta(hours=hours_ago))
                     .isoformat(timespec="seconds"),
        "ticker": ticker,
        "status": status,
    }


def _position(ticker: str, unrealized_pl: float) -> dict:
    return {
        "ticker": ticker,
        "type": "stock",
        "qty": 1.0,
        "avg_cost": 100.0,
        "current_price": 100.0 + unrealized_pl,
        "unrealized_pl": unrealized_pl,
    }


# ── Helper-fn lockdowns ────────────────────────────────────────────


class TestHelpers:
    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_nan_rejected(self):
        assert _safe_float(float("nan")) is None

    def test_safe_float_inf_rejected(self):
        assert _safe_float(float("inf")) is None
        assert _safe_float(float("-inf")) is None

    def test_safe_float_garbage_rejected(self):
        assert _safe_float("not-a-number") is None

    def test_safe_float_passes_zero(self):
        assert _safe_float(0.0) == 0.0

    def test_ticker_regex_word_boundary_MU_not_MUTUAL(self):
        pat = _ticker_regex("MU")
        assert pat.search("MU EARNINGS BEAT")
        assert pat.search("$MU CASHTAG HITS")
        # MUTUAL must NOT alias MU.
        assert not pat.search("MUTUAL FUND OUTFLOWS")

    def test_ticker_regex_word_boundary_AMD_not_AMDOCS(self):
        pat = _ticker_regex("AMD")
        assert pat.search("AMD STRONG QUARTER")
        assert not pat.search("AMDOCS REPORTS Q1")

    def test_ticker_regex_nvda_cashtag_hits(self):
        pat = _ticker_regex("NVDA")
        assert pat.search("$NVDA TO THE MOON")

    def test_parse_action_ticker_buy_filled(self):
        assert _parse_action_ticker("BUY NVDA → FILLED") == ("BUY", "NVDA")

    def test_parse_action_ticker_no_decision_sentinel(self):
        assert _parse_action_ticker("NO_DECISION") == ("NO_DECISION", None)

    def test_parse_action_ticker_blocked_sentinel(self):
        assert _parse_action_ticker("BLOCKED") == ("BLOCKED", None)

    def test_parse_action_ticker_cash_pseudo(self):
        verb, tk = _parse_action_ticker("HOLD CASH → HOLD")
        assert verb == "HOLD"
        assert tk is None  # CASH carve-out

    def test_parse_action_ticker_empty(self):
        assert _parse_action_ticker("") == ("", None)

    def test_verdict_ignored_loud_no_decision(self):
        assert _verdict(MIN_ARTICLES_FOR_VERDICT, 0, 0) == "IGNORED"

    def test_verdict_decided_no_fill(self):
        assert _verdict(MIN_ARTICLES_FOR_VERDICT, 1, 0) == "DECIDED_NO_FILL"

    def test_verdict_responsive(self):
        assert _verdict(MIN_ARTICLES_FOR_VERDICT, 2, 1) == "RESPONSIVE"

    def test_verdict_acted_without_news(self):
        # 1 article < MIN, but 1 fill
        assert _verdict(1, 0, 1) == "ACTED_WITHOUT_NEWS"

    def test_verdict_quiet(self):
        assert _verdict(0, 0, 0) == "QUIET"
        assert _verdict(MIN_ARTICLES_FOR_VERDICT - 1, 0, 0) == "QUIET"

    def test_verdict_quiet_with_decision_but_low_articles(self):
        # Below MIN_ARTICLES_FOR_VERDICT, no fills → QUIET. A single article
        # is not enough to call a decision DECIDED_NO_FILL (sample size honesty).
        assert _verdict(1, 1, 0) == "QUIET"


# ── State ladder ──────────────────────────────────────────────────


class TestStateLadder:
    def test_no_tickers_returns_no_data(self):
        r = build_news_action_funnel([], [], [], [], [], now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["n_tickers"] == 0
        assert r["tickers"] == []
        assert "no tickers" in r["headline"]

    def test_none_inputs_degrade_to_no_data(self):
        r = build_news_action_funnel(None, None, None, None, None, now=NOW)
        assert r["state"] == "NO_DATA"

    def test_tickers_present_returns_ok_even_when_all_quiet(self):
        # MU and NVDA in universe, nothing else — should be OK with two QUIET
        # rows so the panel renders.
        r = build_news_action_funnel(
            [], [], [], [], ["MU", "NVDA"], now=NOW,
        )
        assert r["state"] == "OK"
        assert r["n_tickers"] == 2
        assert {row["ticker"] for row in r["tickers"]} == {"MU", "NVDA"}
        for row in r["tickers"]:
            assert row["verdict"] == "QUIET"

    def test_universe_deduplicated_preserves_first_position(self):
        # Held NVDA listed twice + watchlist NVDA → one row.
        r = build_news_action_funnel(
            [], [], [], [], ["NVDA", "MU", "NVDA"], now=NOW,
        )
        tickers = [row["ticker"] for row in r["tickers"]]
        assert tickers.count("NVDA") == 1
        assert "MU" in tickers


# ── Verdict ladder (against real article windows) ─────────────────


class TestVerdictLadder:
    def test_loud_news_no_decision_is_ignored(self):
        articles = [
            _article("MU", 9.0, 1.0),
            _article("MU", 8.0, 2.0),
            _article("MU", 7.0, 3.0),
        ]
        r = build_news_action_funnel(
            articles, [], [], [], ["MU"], now=NOW,
        )
        mu = r["tickers"][0]
        assert mu["ticker"] == "MU"
        assert mu["n_articles"] == 3
        assert mu["n_decisions"] == 0
        assert mu["n_fills"] == 0
        assert mu["verdict"] == "IGNORED"
        assert r["n_ignored"] == 1
        # Top score carries the loudest article and its title.
        assert mu["top_score"] == 9.0
        assert mu["top_title"].startswith("MU")

    def test_loud_news_with_decision_no_fill_is_decided_no_fill(self):
        articles = [_article("NVDA", 8.0, h) for h in (1.0, 2.0, 3.0)]
        decisions = [_decision("HOLD NVDA → HOLD", 1.5)]
        r = build_news_action_funnel(
            articles, decisions, [], [], ["NVDA"], now=NOW,
        )
        row = r["tickers"][0]
        assert row["verdict"] == "DECIDED_NO_FILL"
        assert row["n_articles"] == 3
        assert row["n_decisions"] == 1
        assert row["n_fills"] == 0

    def test_loud_news_with_fill_is_responsive(self):
        articles = [_article("NVDA", 8.0, h) for h in (1.0, 2.0, 3.0)]
        decisions = [_decision("BUY NVDA → FILLED", 1.0)]
        trades = [_trade("NVDA", 1.0)]
        r = build_news_action_funnel(
            articles, decisions, trades, [], ["NVDA"], now=NOW,
        )
        row = r["tickers"][0]
        assert row["verdict"] == "RESPONSIVE"
        assert row["n_articles"] == 3
        assert row["n_fills"] == 1

    def test_fill_without_loud_news_is_acted_without_news(self):
        # One article (< MIN_ARTICLES_FOR_VERDICT) + one fill → ACTED.
        articles = [_article("NVDA", 8.0, 1.0)]
        trades = [_trade("NVDA", 0.5)]
        r = build_news_action_funnel(
            articles, [], trades, [], ["NVDA"], now=NOW,
        )
        row = r["tickers"][0]
        assert row["verdict"] == "ACTED_WITHOUT_NEWS"
        assert row["n_articles"] == 1
        assert row["n_fills"] == 1

    def test_below_floor_articles_do_not_count_as_loud(self):
        # Three articles but ai_score < min_ai_score floor → none count.
        articles = [_article("MU", 4.0, h) for h in (1.0, 2.0, 3.0)]
        r = build_news_action_funnel(
            articles, [], [], [], ["MU"], now=NOW, min_ai_score=6.0,
        )
        row = r["tickers"][0]
        assert row["n_articles"] == 0
        assert row["verdict"] == "QUIET"

    def test_at_min_articles_boundary_is_ignored_not_quiet(self):
        # Exactly MIN_ARTICLES_FOR_VERDICT (default 3) → IGNORED.
        articles = [_article("MU", 7.0, h)
                    for h in range(1, MIN_ARTICLES_FOR_VERDICT + 1)]
        r = build_news_action_funnel(
            articles, [], [], [], ["MU"], now=NOW,
        )
        row = r["tickers"][0]
        assert row["n_articles"] == MIN_ARTICLES_FOR_VERDICT
        assert row["verdict"] == "IGNORED"

    def test_one_below_min_articles_is_quiet(self):
        # MIN_ARTICLES_FOR_VERDICT - 1 → QUIET (sample-size honesty).
        articles = [_article("MU", 7.0, h)
                    for h in range(1, MIN_ARTICLES_FOR_VERDICT)]
        r = build_news_action_funnel(
            articles, [], [], [], ["MU"], now=NOW,
        )
        assert r["tickers"][0]["verdict"] == "QUIET"


# ── Word-boundary discrimination ──────────────────────────────────


class TestWordBoundary:
    def test_MU_does_not_alias_MUTUAL(self):
        articles = [
            _article("MUTUAL", 9.0, 1.0, title_extra="FUND OUTFLOWS Q1"),
            _article("MUTUAL", 9.0, 2.0, title_extra="REPORT"),
            _article("MUTUAL", 9.0, 3.0, title_extra="LOSSES"),
        ]
        r = build_news_action_funnel(
            articles, [], [], [], ["MU"], now=NOW,
        )
        # MU has 0 articles — MUTUAL must not be bucketed under MU.
        assert r["tickers"][0]["n_articles"] == 0

    def test_AMD_does_not_alias_AMDOCS(self):
        articles = [
            _article("AMDOCS", 9.0, 1.0),
            _article("AMDOCS", 9.0, 2.0),
            _article("AMDOCS", 9.0, 3.0),
        ]
        r = build_news_action_funnel(
            articles, [], [], [], ["AMD"], now=NOW,
        )
        assert r["tickers"][0]["n_articles"] == 0

    def test_nvda_cashtag_hits(self):
        articles = [_article("", 9.0, h, title_extra="$NVDA TO 1000")
                    for h in (1.0, 2.0, 3.0)]
        r = build_news_action_funnel(
            articles, [], [], [], ["NVDA"], now=NOW,
        )
        assert r["tickers"][0]["n_articles"] == 3


# ── Window cutoff ──────────────────────────────────────────────────


class TestWindowCutoff:
    def test_strict_inclusive_at_window_boundary(self):
        # An article exactly at the cutoff (now - window_h) is included.
        articles = [
            _article("NVDA", 9.0, hours_ago=24.0),  # at cutoff
            _article("NVDA", 9.0, hours_ago=1.0),
            _article("NVDA", 9.0, hours_ago=2.0),
        ]
        r = build_news_action_funnel(
            articles, [], [], [], ["NVDA"], now=NOW, window_hours=24.0,
        )
        # All three within (or at) cutoff.
        assert r["tickers"][0]["n_articles"] == 3

    def test_outside_window_excluded(self):
        # 25h ago > 24h window → excluded.
        articles = [
            _article("NVDA", 9.0, hours_ago=25.0),
            _article("NVDA", 9.0, hours_ago=1.0),
            _article("NVDA", 9.0, hours_ago=2.0),
        ]
        r = build_news_action_funnel(
            articles, [], [], [], ["NVDA"], now=NOW, window_hours=24.0,
        )
        assert r["tickers"][0]["n_articles"] == 2

    def test_decisions_outside_window_excluded(self):
        articles = [_article("NVDA", 9.0, h) for h in (1.0, 2.0, 3.0)]
        decisions = [_decision("BUY NVDA → FILLED", hours_ago=30.0)]
        r = build_news_action_funnel(
            articles, decisions, [], [], ["NVDA"],
            now=NOW, window_hours=24.0,
        )
        # 30h-old decision out of window → 0 decisions counted.
        assert r["tickers"][0]["n_decisions"] == 0
        # And so the row is IGNORED (loud articles, no in-window decisions).
        assert r["tickers"][0]["verdict"] == "IGNORED"

    def test_trades_outside_window_excluded(self):
        articles = [_article("NVDA", 9.0, h) for h in (1.0, 2.0, 3.0)]
        trades = [_trade("NVDA", hours_ago=30.0)]
        r = build_news_action_funnel(
            articles, [], trades, [], ["NVDA"], now=NOW, window_hours=24.0,
        )
        assert r["tickers"][0]["n_fills"] == 0


# ── Sort priority ─────────────────────────────────────────────────


class TestSortPriority:
    def test_ignored_first_then_decided_no_fill_then_quiet(self):
        # MU: IGNORED (3 art, 0 dec)
        # NVDA: DECIDED_NO_FILL (3 art, 1 dec, 0 fill)
        # AMD: RESPONSIVE (3 art, 1 dec, 1 fill)
        # DRAM: QUIET (0 art, 0 dec)
        articles = (
            [_article("MU", 8.0, h) for h in (1.0, 2.0, 3.0)]
            + [_article("NVDA", 8.0, h) for h in (1.0, 2.0, 3.0)]
            + [_article("AMD", 8.0, h) for h in (1.0, 2.0, 3.0)]
        )
        decisions = [
            _decision("HOLD NVDA → HOLD", 1.5),
            _decision("BUY AMD → FILLED", 1.0),
        ]
        trades = [_trade("AMD", 1.0)]
        r = build_news_action_funnel(
            articles, decisions, trades, [],
            ["MU", "NVDA", "AMD", "DRAM"], now=NOW,
        )
        verdicts = [row["verdict"] for row in r["tickers"]]
        # Per the priority map: IGNORED < DECIDED_NO_FILL < ACTED < RESPONSIVE < QUIET
        assert verdicts == ["IGNORED", "DECIDED_NO_FILL", "RESPONSIVE", "QUIET"]
        assert r["n_ignored"] == 1
        assert r["n_decided_no_fill"] == 1
        assert r["n_responsive"] == 1
        assert r["n_quiet"] == 1

    def test_within_tier_sorted_by_article_count_desc(self):
        # Two IGNORED tickers, MU has more articles → MU first.
        articles = (
            [_article("MU", 8.0, h) for h in (1.0, 2.0, 3.0, 4.0, 5.0)]
            + [_article("AMD", 8.0, h) for h in (1.0, 2.0, 3.0)]
        )
        r = build_news_action_funnel(
            articles, [], [], [], ["AMD", "MU"], now=NOW,
        )
        assert r["tickers"][0]["ticker"] == "MU"
        assert r["tickers"][0]["n_articles"] == 5
        assert r["tickers"][1]["ticker"] == "AMD"
        assert r["tickers"][1]["n_articles"] == 3

    def test_max_tickers_caps_panel(self):
        # 5 tickers in universe, all QUIET; max_tickers=2 → 2 rows.
        r = build_news_action_funnel(
            [], [], [], [], ["A", "B", "C", "D", "E"],
            now=NOW, max_tickers=2,
        )
        assert r["n_tickers"] == 2


# ── Held flag + P&L attach ────────────────────────────────────────


class TestHeldAndPL:
    def test_held_flag_set_when_in_held_tickers(self):
        r = build_news_action_funnel(
            [], [], [], [], ["NVDA", "MU"],
            held_tickers=["NVDA"], now=NOW,
        )
        rows = {row["ticker"]: row for row in r["tickers"]}
        assert rows["NVDA"]["held"] is True
        assert rows["MU"]["held"] is False

    def test_unrealized_pl_attached_for_held(self):
        positions = [_position("NVDA", unrealized_pl=-3.44)]
        r = build_news_action_funnel(
            [], [], [], positions, ["NVDA"],
            held_tickers=["NVDA"], now=NOW,
        )
        assert r["tickers"][0]["unrealized_pl"] == pytest.approx(-3.44)

    def test_unrealized_pl_none_for_unheld(self):
        r = build_news_action_funnel(
            [], [], [], [], ["MU"], held_tickers=[], now=NOW,
        )
        assert r["tickers"][0]["unrealized_pl"] is None

    def test_held_tag_in_headline_when_ignored_is_held(self):
        # 3 loud MU articles, MU held → headline includes "(HELD)".
        articles = [_article("MU", 9.0, h) for h in (1.0, 2.0, 3.0)]
        r = build_news_action_funnel(
            articles, [], [], [_position("MU", -1.0)],
            ["MU"], held_tickers=["MU"], now=NOW,
        )
        assert "(HELD)" in r["headline"]
        assert "MU" in r["headline"]


# ── NaN/Inf rejection ─────────────────────────────────────────────


class TestScoreRejection:
    def test_nan_ai_score_rejected(self):
        articles = [
            {"title": "MU news",
             "ai_score": float("nan"),
             "first_seen": (NOW - timedelta(hours=1)).isoformat()},
        ]
        r = build_news_action_funnel(
            articles, [], [], [], ["MU"], now=NOW,
        )
        assert r["tickers"][0]["n_articles"] == 0

    def test_inf_ai_score_rejected(self):
        articles = [
            {"title": "MU news",
             "ai_score": float("inf"),
             "first_seen": (NOW - timedelta(hours=1)).isoformat()},
        ]
        r = build_news_action_funnel(
            articles, [], [], [], ["MU"], now=NOW,
        )
        assert r["tickers"][0]["n_articles"] == 0


# ── Decision sentinel + parsed ticker correctness ─────────────────


class TestDecisionParsing:
    def test_no_decision_does_not_bucket(self):
        articles = [_article("NVDA", 9.0, h) for h in (1.0, 2.0, 3.0)]
        decisions = [_decision("NO_DECISION", 1.0)]
        r = build_news_action_funnel(
            articles, decisions, [], [], ["NVDA"], now=NOW,
        )
        # NO_DECISION sentinel → no ticker → no bucket.
        assert r["tickers"][0]["n_decisions"] == 0
        assert r["tickers"][0]["verdict"] == "IGNORED"

    def test_blocked_does_not_bucket(self):
        articles = [_article("NVDA", 9.0, h) for h in (1.0, 2.0, 3.0)]
        decisions = [_decision("BLOCKED", 1.0)]
        r = build_news_action_funnel(
            articles, decisions, [], [], ["NVDA"], now=NOW,
        )
        assert r["tickers"][0]["n_decisions"] == 0

    def test_hold_cash_does_not_bucket(self):
        articles = [_article("NVDA", 9.0, h) for h in (1.0, 2.0, 3.0)]
        decisions = [_decision("HOLD CASH → HOLD", 1.0)]
        r = build_news_action_funnel(
            articles, decisions, [], [], ["NVDA"], now=NOW,
        )
        assert r["tickers"][0]["n_decisions"] == 0  # CASH carve-out

    def test_decision_for_different_ticker_does_not_count(self):
        articles = [_article("NVDA", 9.0, h) for h in (1.0, 2.0, 3.0)]
        decisions = [_decision("BUY MU → FILLED", 1.0)]
        r = build_news_action_funnel(
            articles, decisions, [], [], ["NVDA", "MU"], now=NOW,
        )
        nvda = [row for row in r["tickers"] if row["ticker"] == "NVDA"][0]
        mu = [row for row in r["tickers"] if row["ticker"] == "MU"][0]
        assert nvda["n_decisions"] == 0
        assert mu["n_decisions"] == 1

    def test_unfilled_trades_excluded(self):
        # Status != FILLED → not counted as a fill.
        articles = [_article("NVDA", 9.0, h) for h in (1.0, 2.0, 3.0)]
        trades = [_trade("NVDA", 1.0, status="REJECTED")]
        r = build_news_action_funnel(
            articles, [], trades, [], ["NVDA"], now=NOW,
        )
        assert r["tickers"][0]["n_fills"] == 0

    def test_live_trades_no_status_counted(self):
        # The live paper_trader.db trades table has NO status column — every
        # row in the table IS a fill. The builder must count those even when
        # status is missing from the dict.
        articles = [_article("NVDA", 9.0, h) for h in (1.0, 2.0, 3.0)]
        trades = [{
            "timestamp": (NOW - timedelta(hours=1)).isoformat(),
            "ticker": "NVDA",
            "action": "BUY",
            # NO status field — the live shape.
        }]
        r = build_news_action_funnel(
            articles, [], trades, [], ["NVDA"], now=NOW,
        )
        assert r["tickers"][0]["n_fills"] == 1


# ── Degrade-never-raise (the _safe contract) ──────────────────────


class TestDegrade:
    def test_none_article_in_list(self):
        articles = [None, _article("NVDA", 9.0, 1.0)]
        r = build_news_action_funnel(
            articles, [], [], [], ["NVDA"], now=NOW,
        )
        assert r["state"] == "OK"
        # The good article still counted.
        assert r["tickers"][0]["n_articles"] == 1

    def test_non_dict_article(self):
        articles = ["garbage", 42, _article("NVDA", 9.0, 1.0)]
        r = build_news_action_funnel(
            articles, [], [], [], ["NVDA"], now=NOW,
        )
        assert r["state"] == "OK"

    def test_garbage_timestamps_dropped(self):
        articles = [
            {"title": "NVDA news", "ai_score": 9.0, "first_seen": "garbage"},
            _article("NVDA", 9.0, 1.0),
        ]
        r = build_news_action_funnel(
            articles, [], [], [], ["NVDA"], now=NOW,
        )
        assert r["tickers"][0]["n_articles"] == 1

    def test_missing_fields_dont_crash(self):
        articles = [{"title": None, "ai_score": None,
                     "first_seen": None}]
        r = build_news_action_funnel(
            articles, [], [], [], ["NVDA"], now=NOW,
        )
        assert r["state"] == "OK"
        assert r["tickers"][0]["n_articles"] == 0

    def test_garbage_decisions_skipped(self):
        articles = [_article("NVDA", 9.0, h) for h in (1.0, 2.0, 3.0)]
        decisions = [None, "garbage", {"timestamp": "bad",
                                       "action_taken": "BUY NVDA → FILLED"}]
        r = build_news_action_funnel(
            articles, decisions, [], [], ["NVDA"], now=NOW,
        )
        # Bad-timestamp row dropped → 0 decisions.
        assert r["tickers"][0]["n_decisions"] == 0

    def test_garbage_trades_skipped(self):
        articles = [_article("NVDA", 9.0, h) for h in (1.0, 2.0, 3.0)]
        trades = [None, 42, {"ticker": None, "status": "FILLED",
                             "timestamp": (NOW).isoformat()}]
        r = build_news_action_funnel(
            articles, [], trades, [], ["NVDA"], now=NOW,
        )
        assert r["tickers"][0]["n_fills"] == 0

    def test_garbage_position_does_not_crash_pl_attach(self):
        positions = [None, {"ticker": None, "unrealized_pl": 5.0},
                     {"ticker": "NVDA", "unrealized_pl": "not-a-number"},
                     _position("NVDA", -3.44)]
        r = build_news_action_funnel(
            [], [], [], positions, ["NVDA"],
            held_tickers=["NVDA"], now=NOW,
        )
        # The one good row wins.
        assert r["tickers"][0]["unrealized_pl"] == pytest.approx(-3.44)

    def test_empty_ticker_string_dropped(self):
        # Empty / non-string entries in universe dropped before bucketing.
        r = build_news_action_funnel(
            [], [], [], [], ["NVDA", "", None, 42], now=NOW,
        )
        tickers = [row["ticker"] for row in r["tickers"]]
        assert tickers == ["NVDA"]


# ── Tie-break / top-article correctness ───────────────────────────


class TestTopArticle:
    def test_top_article_score_desc_then_newer_first_seen(self):
        # Two equal-score articles for MU; the NEWER one wins the tie.
        articles = [
            _article("MU", 8.0, hours_ago=5.0, title_extra="OLDER"),
            _article("MU", 8.0, hours_ago=1.0, title_extra="NEWER"),
            _article("MU", 7.0, hours_ago=0.5, title_extra="LOWER"),
        ]
        r = build_news_action_funnel(
            articles, [], [], [], ["MU"], now=NOW,
        )
        # Top score is 8.0, tie-break to newer → "NEWER" article.
        assert r["tickers"][0]["top_score"] == 8.0
        assert "NEWER" in r["tickers"][0]["top_title"]


# ── Endpoint Flask test_client ────────────────────────────────────


@pytest.fixture
def seeded_articles_db(tmp_path) -> Path:
    """Hand-craft a tiny articles.db with one IGNORED row and one synthetic
    backtest row that MUST be filtered by the SQL-side live-only clause."""
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE articles (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            published TEXT,
            kw_score REAL DEFAULT 0,
            ai_score REAL DEFAULT 0,
            urgency INTEGER DEFAULT 0,
            full_text BLOB,
            first_seen TEXT NOT NULL,
            cycle INTEGER DEFAULT 0,
            time_sensitivity REAL
        );
        """
    )
    rows = [
        ("a1", "https://x/1", "MU on its way to 800 — BofA target raise",
         "rss", 9.0, 0,
         (NOW - timedelta(hours=1)).isoformat(timespec="seconds")),
        ("a2", "https://x/2", "MU strong demand sustained",
         "rss", 8.0, 0,
         (NOW - timedelta(hours=3)).isoformat(timespec="seconds")),
        ("a3", "https://x/3", "MU upgrade analyst",
         "rss", 7.0, 0,
         (NOW - timedelta(hours=5)).isoformat(timespec="seconds")),
        # Synthetic backtest row — SQL live-only filter MUST drop it.
        ("a4", "backtest://run_1/MU", "MU backtest winner annotation",
         "backtest_run_1_winner", 10.0, 0,
         (NOW - timedelta(hours=1)).isoformat(timespec="seconds")),
        # Old article outside the window — dropped by cutoff.
        ("a5", "https://x/5", "MU ancient news",
         "rss", 9.0, 0,
         (NOW - timedelta(hours=48)).isoformat(timespec="seconds")),
    ]
    conn.executemany(
        "INSERT INTO articles (id,url,title,source,ai_score,urgency,first_seen) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


def _bootstrap_dashboard_app(tmp_path, monkeypatch,
                             seeded_db: Path | None = None):
    """Spin up Flask app on a temp paper_trader.db + optional seeded
    articles.db. Same pattern as test_idle_opportunity."""
    from paper_trader import dashboard as dash
    from paper_trader import store as store_mod
    from paper_trader.store import Store
    monkeypatch.setattr(store_mod, "DB_PATH",
                        tmp_path / "paper_trader.db")
    monkeypatch.setattr(store_mod, "_singleton", None)
    fresh_store = Store()
    monkeypatch.setattr(dash, "get_store", lambda: fresh_store)
    if seeded_db is not None:
        monkeypatch.setattr(dash, "_articles_db_path", lambda: seeded_db)
    else:
        # Sentinel: any attempt to open the live DB returns None
        # (build still degrades cleanly).
        monkeypatch.setattr(dash, "_articles_db_path", lambda: None)
    dash.app.config["TESTING"] = True
    return dash, fresh_store


class TestNewsActionFunnelEndpoint:
    def test_endpoint_serves_ok_when_db_absent(self, tmp_path, monkeypatch):
        # No articles.db → empty article list → tickers all QUIET; state OK.
        dash, store = _bootstrap_dashboard_app(tmp_path, monkeypatch)
        client = dash.app.test_client()
        resp = client.get("/api/news-action-funnel?tickers=MU,NVDA")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["state"] == "OK"
        assert body["n_ignored"] == 0
        verdicts = {row["verdict"] for row in body["tickers"]}
        assert verdicts == {"QUIET"}

    def test_endpoint_against_seeded_db_returns_ignored(
            self, tmp_path, monkeypatch, seeded_articles_db):
        dash, store = _bootstrap_dashboard_app(
            tmp_path, monkeypatch, seeded_db=seeded_articles_db,
        )
        client = dash.app.test_client()
        resp = client.get(
            "/api/news-action-funnel?tickers=MU,NVDA&window_hours=24")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["state"] == "OK"
        mu = [row for row in body["tickers"] if row["ticker"] == "MU"][0]
        # Three live articles in window (a1, a2, a3). The synthetic
        # a4 (backtest_run_1_winner) MUST be filtered by SQL. a5 is out of
        # the 24h window.
        assert mu["n_articles"] == 3
        assert mu["verdict"] == "IGNORED"
        # Top score is 9.0 — the legitimate top row, NOT the 10.0 synthetic.
        assert mu["top_score"] == 9.0
        assert mu["top_score"] != 10.0

    def test_endpoint_clamps_window_hours(
            self, tmp_path, monkeypatch, seeded_articles_db):
        dash, _ = _bootstrap_dashboard_app(
            tmp_path, monkeypatch, seeded_db=seeded_articles_db,
        )
        client = dash.app.test_client()
        # Garbage → default.
        resp = client.get(
            "/api/news-action-funnel?tickers=MU&window_hours=nope")
        body = resp.get_json()
        assert body["window_hours"] == DEFAULT_WINDOW_HOURS
        # Over cap → clamped to 72.
        resp = client.get(
            "/api/news-action-funnel?tickers=MU&window_hours=9999")
        body = resp.get_json()
        assert body["window_hours"] == 72.0
        # Below floor → clamped to 1.
        resp = client.get(
            "/api/news-action-funnel?tickers=MU&window_hours=0")
        body = resp.get_json()
        assert body["window_hours"] == 1.0

    def test_endpoint_clamps_min_ai_score(
            self, tmp_path, monkeypatch, seeded_articles_db):
        dash, _ = _bootstrap_dashboard_app(
            tmp_path, monkeypatch, seeded_db=seeded_articles_db,
        )
        client = dash.app.test_client()
        resp = client.get(
            "/api/news-action-funnel?tickers=MU&min_ai_score=nope")
        body = resp.get_json()
        assert body["min_ai_score"] == DEFAULT_MIN_AI_SCORE
        resp = client.get(
            "/api/news-action-funnel?tickers=MU&min_ai_score=99")
        body = resp.get_json()
        assert body["min_ai_score"] == 10.0

    def test_endpoint_clamps_max_tickers(
            self, tmp_path, monkeypatch, seeded_articles_db):
        dash, _ = _bootstrap_dashboard_app(
            tmp_path, monkeypatch, seeded_db=seeded_articles_db,
        )
        client = dash.app.test_client()
        resp = client.get(
            "/api/news-action-funnel?tickers=A,B,C,D,E&max_tickers=2")
        body = resp.get_json()
        assert body["n_tickers"] == 2

    def test_endpoint_with_real_decision_in_store(
            self, tmp_path, monkeypatch, seeded_articles_db):
        # Seed a real DECIDED_NO_FILL state: 3 MU articles + 1 HOLD MU
        # decision but 0 fills.
        dash, store = _bootstrap_dashboard_app(
            tmp_path, monkeypatch, seeded_db=seeded_articles_db,
        )
        ts = (NOW - timedelta(hours=1)).isoformat(timespec="seconds")
        store.conn.execute(
            "INSERT INTO decisions (timestamp, market_open, signal_count, "
            "action_taken, reasoning, portfolio_value, cash) "
            "VALUES (?, 1, 1, ?, 'seed', 1000.0, 500.0)",
            (ts, "HOLD MU → HOLD"),
        )
        store.conn.commit()
        client = dash.app.test_client()
        resp = client.get(
            "/api/news-action-funnel?tickers=MU&window_hours=24")
        body = resp.get_json()
        mu = body["tickers"][0]
        assert mu["n_articles"] == 3
        assert mu["n_decisions"] == 1
        assert mu["n_fills"] == 0
        assert mu["verdict"] == "DECIDED_NO_FILL"

    def test_endpoint_with_real_fill_in_store(
            self, tmp_path, monkeypatch, seeded_articles_db):
        # Three MU articles + 1 FILLED MU trade → RESPONSIVE.
        dash, store = _bootstrap_dashboard_app(
            tmp_path, monkeypatch, seeded_db=seeded_articles_db,
        )
        ts = (NOW - timedelta(hours=1)).isoformat(timespec="seconds")
        store.conn.execute(
            "INSERT INTO decisions (timestamp, market_open, signal_count, "
            "action_taken, reasoning, portfolio_value, cash) "
            "VALUES (?, 1, 1, ?, 'seed', 1000.0, 500.0)",
            (ts, "BUY MU → FILLED"),
        )
        store.conn.execute(
            "INSERT INTO trades (timestamp, ticker, action, qty, "
            "price, value, reason) "
            "VALUES (?, 'MU', 'BUY', 1.0, 100.0, 100.0, 'seed')",
            (ts,),
        )
        store.conn.commit()
        client = dash.app.test_client()
        resp = client.get(
            "/api/news-action-funnel?tickers=MU&window_hours=24")
        body = resp.get_json()
        mu = body["tickers"][0]
        assert mu["n_articles"] == 3
        assert mu["n_fills"] == 1
        assert mu["verdict"] == "RESPONSIVE"
