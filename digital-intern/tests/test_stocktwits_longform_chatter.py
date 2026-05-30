"""Tests for ``analytics.stocktwits_longform_chatter``.

Verifies the predicate's high-precision behaviour (catches the live chatter
patterns the 50-char cap misses, survives every must-survive sample from the
7d ml>=9 stocktwits audit), the builder envelope contract (verdict ladder,
exact mean_ml_score arithmetic, sample caps, ranking ties), and SSOT lockstep
with the alert-side news-keyword exit list.

Discipline mirrors ``tests/test_emerging_press_mill.py`` — the canonical
analytics-test shape from the prior agent rotation. Assertions are specific
values, not "no crash". Backtest isolation is N/A at the builder layer (caller
filters upstream via ``_LIVE_ONLY_CLAUSE``); the predicate cannot match a
synthetic row by construction (source check requires ``stocktwits``), but the
builder still never raises on a malformed row.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from analytics.stocktwits_longform_chatter import (
    LONGFORM_MAX_TITLE_LEN,
    LONGFORM_MIN_TITLE_LEN,
    LONGFORM_PREDICATES,
    build_longform_chatter_report,
    is_longform_stocktwits_chatter,
)


# ── Predicate: positive (must-catch) ──────────────────────────────────────────
class TestPredicatePositive:
    """The live-DB urgency=2 chatter rows the 50-char short-form gate misses
    must all trigger the predicate. Titles are verbatim from the 2026-05-30
    audit, truncated to the live width."""

    def test_multi_ticker_lead_with_slang(self):
        # 101 chars, stocktwits, urgency=2, ml_score=9.9 in live DB.
        art = {
            "source": "stocktwits",
            "title": (
                "$IONQ $QBTS $QUBT $RGTI $XRP.X nobody wants to buy your "
                "dumb shtcoin. Go knock on someone"
            ),
        }
        assert is_longform_stocktwits_chatter(art)

    def test_held_ticker_chatter_with_emoji(self):
        art = {
            "source": "stocktwits",
            "title": "$MU joining the AI momentum is exactly why I want better alerts. Memory demand d",
        }
        assert is_longform_stocktwits_chatter(art)

    def test_held_ticker_chatter_simple(self):
        art = {
            "source": "stocktwits",
            "title": "$MU Sold-Out Inventory: The explosive demand for Artificial Intelligence require",
        }
        assert is_longform_stocktwits_chatter(art)

    def test_political_chatter_with_tickers(self):
        # 76 chars: between the existing 50-char cap and a long real article.
        art = {
            "source": "stocktwits",
            "title": "$SPY $QQQ $USO $NVDA if Israel is gonna keep shooting up places or bombing now",
        }
        assert is_longform_stocktwits_chatter(art)

    def test_stocktwits_trending_collector_source_caught(self):
        # Predicate is source-prefix-scoped — sibling collectors named
        # "stocktwits_trending", "StockTwits Trending" must also match.
        art = {
            "source": "stocktwits_trending",
            "title": "$MU another red day, but I'm holding strong. AI demand is structural. nothing to",
        }
        assert is_longform_stocktwits_chatter(art)


# ── Predicate: negative (must-survive) ────────────────────────────────────────
class TestPredicateNegative:
    """Every ``has-news`` survivor from the 7d audit, plus every adjacent
    edge case, must NOT trigger. These are the precision-anchors."""

    def test_short_title_below_floor_not_caught(self):
        # The existing short-form gate's responsibility, not ours.
        art = {"source": "stocktwits", "title": "$MU lol"}
        assert is_longform_stocktwits_chatter(art) is False

    def test_exactly_at_floor_caught(self):
        # The boundary: len == LONGFORM_MIN_TITLE_LEN (50) is the FIRST length
        # this gate engages on (the existing gate stops at < 50). Build a
        # 50-char chatter title.
        title = "$MU $NVDA $SPY chatter with NO news in this title."
        assert len(title) == 50
        art = {"source": "stocktwits", "title": title}
        assert is_longform_stocktwits_chatter(art) is True

    def test_one_char_below_floor_not_caught(self):
        title = "$MU $NVDA $SPY chatter with NO news in this title"
        assert len(title) == 49
        art = {"source": "stocktwits", "title": title}
        assert is_longform_stocktwits_chatter(art) is False

    def test_above_ceiling_not_caught(self):
        # Very long titles are likely real syndicated content; survive.
        title = "x" * (LONGFORM_MAX_TITLE_LEN + 5)
        art = {"source": "stocktwits", "title": title}
        assert is_longform_stocktwits_chatter(art) is False

    def test_news_keyword_earnings_survives(self):
        # 7d audit survivor: "Next Week's Trading Outlook ... earnings".
        art = {
            "source": "stocktwits",
            "title": "Next Week's Trading Outlook — Wishing You Successful Trades! Here are the five key factors influencing earnings",
        }
        assert is_longform_stocktwits_chatter(art) is False

    def test_news_keyword_price_target_survives(self):
        # 7d audit survivor: "$MU Price Target Alert: $1,175.00. Issued by Barclays".
        art = {
            "source": "stocktwits",
            "title": "$MU MU Price Target Alert: $1,175.00. Issued by Barclays — full coverage in the wire",
        }
        assert is_longform_stocktwits_chatter(art) is False

    def test_news_keyword_analyst_survives(self):
        # 7d audit survivor: "$MU UBS Analyst Timothy Arcuri expects ...".
        art = {
            "source": "stocktwits",
            "title": "$MU UBS Analyst Timothy Arcuri expects to keep Micron's earnings per share above $100",
        }
        assert is_longform_stocktwits_chatter(art) is False

    def test_news_keyword_merger_survives(self):
        # 7d audit survivor: "$ASRT ... merger can still go through".
        art = {
            "source": "stocktwits",
            "title": (
                "$ASRT The blunt answer: you do not get a 'No' vote button. "
                "To show disapproval mechanically, you do not tender. But if "
                "more than 50% tender, the merger can still go through and "
                "your remaining shares w"
            ),
        }
        assert is_longform_stocktwits_chatter(art) is False

    def test_shared_bloomberg_url_survives(self):
        # 7d audit survivor: "$NVDA $TSM $SMCI $MU $DRAM ... bloomberg.com/...".
        art = {
            "source": "stocktwits",
            "title": "$NVDA $TSM $SMCI $MU $DRAM #SKYHNIX https://www.bloomberg.com/news/articles/2026-05-27/sk-hynix-joins-1",
        }
        assert is_longform_stocktwits_chatter(art) is False

    def test_shared_marketwirenews_url_survives(self):
        # User-shared real news on marketwirenews URL — even without an
        # obvious news-keyword in the prose, the URL is the discriminator.
        art = {
            "source": "stocktwits",
            "title": "$RDW https://marketbeat.com/a/8675828/  The SpaceX IPO Frenzy Is Creating 2 Very Real Beneficiaries",
        }
        assert is_longform_stocktwits_chatter(art) is False

    def test_stocktwits_sentiment_digest_source_excluded(self):
        # The structured sentiment digest carries real signal.
        art = {
            "source": "stocktwits/sentiment",
            "title": "[StockTwits Sentiment] NVDA Bullish: 53% Bullish / 3% Bearish (16 1 of 30)",
        }
        assert is_longform_stocktwits_chatter(art) is False

    def test_non_stocktwits_source_not_caught(self):
        # Same chatter-shape title but from a non-stocktwits source: the
        # alert-side gate is source-scoped, so this predicate must be too.
        art = {
            "source": "reddit",
            "title": "$IONQ $QBTS $QUBT $RGTI $XRP.X nobody wants to buy your dumb shtcoin",
        }
        assert is_longform_stocktwits_chatter(art) is False

    def test_empty_source_not_caught(self):
        art = {"source": "", "title": "$MU $NVDA random long chatter from an unknown collector"}
        assert is_longform_stocktwits_chatter(art) is False

    def test_empty_title_not_caught(self):
        art = {"source": "stocktwits", "title": ""}
        assert is_longform_stocktwits_chatter(art) is False


# ── Predicate: defensive ────────────────────────────────────────────────────
class TestPredicateDefensive:
    def test_non_dict_input_returns_false(self):
        assert is_longform_stocktwits_chatter(None) is False
        assert is_longform_stocktwits_chatter("not a dict") is False
        assert is_longform_stocktwits_chatter(42) is False
        assert is_longform_stocktwits_chatter([1, 2, 3]) is False

    def test_missing_keys_returns_false(self):
        assert is_longform_stocktwits_chatter({}) is False
        assert is_longform_stocktwits_chatter({"source": "stocktwits"}) is False
        assert is_longform_stocktwits_chatter(
            {"title": "$MU some long chatter that has no news keyword whatsoever"}
        ) is False

    def test_none_values_return_false(self):
        assert is_longform_stocktwits_chatter(
            {"source": None, "title": "$MU long enough title with no news keyword anywhere"}
        ) is False
        assert is_longform_stocktwits_chatter(
            {"source": "stocktwits", "title": None}
        ) is False


# ── Builder: verdict ladder ─────────────────────────────────────────────────
class TestBuilderVerdictLadder:
    """The verdict transitions across the NO_DATA / NO_CHATTER /
    CHATTER_LEAKING_PAST_GATE ladder. Mirrors the
    NO_DATA / ALL_GATED / EMERGING_NOISE precedent from emerging_press_mill."""

    def test_empty_input_returns_no_data(self):
        out = build_longform_chatter_report([])
        assert out["verdict"] == "NO_DATA"
        assert out["n_audited"] == 0
        assert out["n_chatter_caught"] == 0
        assert out["n_uncaught"] == 0

    def test_only_real_news_returns_no_chatter(self):
        rows = [
            {"source": "rss", "title": "Nvidia Q1 revenue rises 22%", "ml_score": 9.5},
            {"source": "reuters", "title": "Fed cuts rates 50bp", "ml_score": 9.0},
        ]
        out = build_longform_chatter_report(rows)
        assert out["verdict"] == "NO_CHATTER"
        assert out["n_audited"] == 2
        assert out["n_chatter_caught"] == 0
        assert out["n_uncaught"] == 2

    def test_one_chatter_returns_leaking(self):
        rows = [
            {
                "source": "stocktwits",
                "title": "$MU joining the AI momentum is exactly why I want better alerts here",
                "ml_score": 9.9,
            },
        ]
        out = build_longform_chatter_report(rows)
        assert out["verdict"] == "CHATTER_LEAKING_PAST_GATE"
        assert out["n_audited"] == 1
        assert out["n_chatter_caught"] == 1
        assert out["n_uncaught"] == 0


# ── Builder: bucket counts + arithmetic ──────────────────────────────────────
class TestBuilderArithmetic:
    def test_mean_ml_score_arithmetic_exact(self):
        # 3 chatter rows with ml_scores [9.97, 9.9, 9.5] → mean = 9.79.
        rows = [
            {
                "source": "stocktwits",
                "title": "$MU random chatter that has no news keywords in it at all here",
                "ml_score": 9.97,
            },
            {
                "source": "stocktwits",
                "title": "$NVDA more chatter without any news keyword whatsoever folks",
                "ml_score": 9.9,
            },
            {
                "source": "stocktwits",
                "title": "$MSFT yet more chatter with no news keyword to be found here today",
                "ml_score": 9.5,
            },
        ]
        out = build_longform_chatter_report(rows)
        assert out["n_chatter_caught"] == 3
        pred = out["by_predicate"]["longform_stocktwits_chatter"]
        assert pred["count"] == 3
        # Exact arithmetic to 1e-3.
        assert pred["mean_ml_score"] == pytest.approx(
            (9.97 + 9.9 + 9.5) / 3.0, abs=1e-3
        )

    def test_chatter_count_is_predicate_count(self):
        rows = [
            {"source": "stocktwits", "title": "$MU long chatter without any newsy keyword in this entire title at all here", "ml_score": 9.0},
            {"source": "rss", "title": "Real news here", "ml_score": 8.0},
            {"source": "stocktwits", "title": "$NVDA $MU more long chatter without any newsy content of any kind whatsoever", "ml_score": 9.5},
        ]
        out = build_longform_chatter_report(rows)
        assert out["n_chatter_caught"] == 2
        assert out["n_uncaught"] == 1
        assert out["n_audited"] == 3

    def test_uncaught_source_breakdown_ranking(self):
        # Two real-news sources, three rss rows + two reuters rows. rss leads
        # with 3 count, then reuters with 2.
        rows = [
            {"source": "rss", "title": "real news A", "ml_score": 8.0},
            {"source": "rss", "title": "real news B", "ml_score": 9.0},
            {"source": "rss", "title": "real news C", "ml_score": 7.0},
            {"source": "reuters", "title": "real news D", "ml_score": 9.5},
            {"source": "reuters", "title": "real news E", "ml_score": 8.5},
        ]
        out = build_longform_chatter_report(rows)
        assert out["by_uncaught_source"][0]["source"] == "rss"
        assert out["by_uncaught_source"][0]["count"] == 3
        assert out["by_uncaught_source"][0]["mean_ml_score"] == pytest.approx(
            (8.0 + 9.0 + 7.0) / 3.0, abs=1e-3
        )
        assert out["by_uncaught_source"][1]["source"] == "reuters"
        assert out["by_uncaught_source"][1]["count"] == 2
        assert out["by_uncaught_source"][1]["mean_ml_score"] == pytest.approx(
            (9.5 + 8.5) / 2.0, abs=1e-3
        )

    def test_uncaught_source_tiebreak_alphabetical(self):
        # Two sources with same count → alphabetical tiebreak.
        rows = [
            {"source": "zzz_late", "title": "real news A", "ml_score": 8.0},
            {"source": "aaa_first", "title": "real news B", "ml_score": 9.0},
        ]
        out = build_longform_chatter_report(rows)
        # Both have count 1 → alphabetical.
        assert out["by_uncaught_source"][0]["source"] == "aaa_first"
        assert out["by_uncaught_source"][1]["source"] == "zzz_late"


# ── Builder: caps ───────────────────────────────────────────────────────────
class TestBuilderCaps:
    def test_max_samples_per_pattern_respected(self):
        # 10 distinct chatter rows; cap to 3 samples per pattern.
        rows = [
            {
                "source": "stocktwits",
                "title": f"$MU chatter sample number {i} with no news in this title at all",
                "ml_score": 9.0,
            }
            for i in range(10)
        ]
        out = build_longform_chatter_report(rows, max_samples_per_pattern=3)
        samples = out["by_predicate"]["longform_stocktwits_chatter"]["sample_titles"]
        assert len(samples) == 3
        # Bucket count is still 10 (only the sample list is capped).
        assert out["by_predicate"]["longform_stocktwits_chatter"]["count"] == 10

    def test_max_uncaught_sources_respected(self):
        # 10 distinct non-stocktwits sources, cap to 3.
        rows = [
            {"source": f"src_{i:02d}", "title": "real news A", "ml_score": 8.0}
            for i in range(10)
        ]
        out = build_longform_chatter_report(rows, max_uncaught_sources=3)
        assert len(out["by_uncaught_source"]) == 3

    def test_samples_dedupe_within_bucket(self):
        # Three rows with the IDENTICAL title — sample list should hold one.
        rows = [
            {"source": "stocktwits", "title": "$MU same chatter sample with no newsy keyword in the title anywhere here", "ml_score": 9.0},
            {"source": "stocktwits", "title": "$MU same chatter sample with no newsy keyword in the title anywhere here", "ml_score": 9.0},
            {"source": "stocktwits", "title": "$MU same chatter sample with no newsy keyword in the title anywhere here", "ml_score": 9.0},
        ]
        out = build_longform_chatter_report(rows, max_samples_per_pattern=5)
        samples = out["by_predicate"]["longform_stocktwits_chatter"]["sample_titles"]
        assert len(samples) == 1
        # Count still reflects every row.
        assert out["by_predicate"]["longform_stocktwits_chatter"]["count"] == 3


# ── Builder: defensive ──────────────────────────────────────────────────────
class TestBuilderDefensive:
    def test_skips_non_dict_rows_without_raising(self):
        # Mixed list with garbage entries.
        rows = [
            None,
            "not a dict",
            42,
            {"source": "stocktwits", "title": "$MU long chatter sample with no news keyword present at all", "ml_score": 9.0},
            [1, 2, 3],
            {"source": "rss", "title": "real news", "ml_score": 8.0},
        ]
        out = build_longform_chatter_report(rows)
        # Only the 2 dict rows are audited; garbage skipped silently.
        assert out["n_audited"] == 2
        assert out["n_chatter_caught"] == 1
        assert out["n_uncaught"] == 1

    def test_malformed_ml_score_falls_to_zero(self):
        rows = [
            {
                "source": "stocktwits",
                "title": "$MU chatter row with garbage ml_score field on it here",
                "ml_score": "not a number",
            },
        ]
        out = build_longform_chatter_report(rows)
        # Mean for one row with parse-failed score is 0.0 (the documented
        # fallback in the builder).
        assert out["by_predicate"]["longform_stocktwits_chatter"]["mean_ml_score"] == 0.0

    def test_as_of_uses_now_when_unset(self):
        out = build_longform_chatter_report([])
        # Parses without raising.
        dt = datetime.fromisoformat(out["as_of"])
        assert dt.tzinfo is not None

    def test_as_of_honors_override(self):
        fixed = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
        out = build_longform_chatter_report([], now=fixed)
        assert out["as_of"] == fixed.isoformat()


# ── Drift locks (SSOT) ──────────────────────────────────────────────────────
class TestDriftLocks:
    """Lock invariants that a future regression would silently violate."""

    def test_predicates_tuple_length_pinned(self):
        # If a future agent drops a predicate without intent, this fails.
        assert len(LONGFORM_PREDICATES) == 1

    def test_predicates_first_match_is_stocktwits_chatter(self):
        label, pred = LONGFORM_PREDICATES[0]
        assert label == "longform_stocktwits_chatter"
        assert pred is is_longform_stocktwits_chatter

    def test_news_exit_imported_from_alert_agent(self):
        """SSOT: the news-keyword exit list is imported verbatim from
        ``watchers.alert_agent``, not redeclared. A future widening on the
        alert side must automatically extend the must-survive corpus for
        this predicate — there is no other path to that consistency."""
        from analytics import stocktwits_longform_chatter as mod
        from watchers import alert_agent
        assert mod._STOCKTWITS_NEWS_EXIT is alert_agent._STOCKTWITS_NEWS_EXIT

    def test_min_floor_matches_short_form_gate_ceiling(self):
        """The long-form gate must start EXACTLY where the short-form gate
        stops — same boundary on both sides so titles are covered exactly
        once across the two predicates."""
        from watchers.alert_agent import _STOCKTWITS_CHATTER_TITLE_MAX
        assert LONGFORM_MIN_TITLE_LEN == _STOCKTWITS_CHATTER_TITLE_MAX

    def test_envelope_keys_pinned(self):
        out = build_longform_chatter_report([])
        # The envelope shape downstream readers (dashboards, briefings) will
        # consume. Pin every top-level key.
        assert set(out.keys()) == {
            "as_of",
            "n_audited",
            "n_chatter_caught",
            "n_uncaught",
            "by_predicate",
            "by_uncaught_source",
            "verdict",
        }
