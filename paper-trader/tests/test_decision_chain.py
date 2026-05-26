"""Tests for analytics/decision_chain.py + /api/decision-chain.

Contract:
* Pure ``build_decision_chain`` takes pre-fetched trades + articles +
  current_prices + (optional) decisions. Never raises on garbage input.
* Verdict ladder per trade: GOOD / NEUTRAL / BAD / OPEN / NO_PRICE.
  - BUY: intended_pct = +abs_pct → GOOD when intended ≥ good_pct
  - SELL: intended_pct = -abs_pct → GOOD when intended ≥ good_pct
  - OPEN: trades younger than open_min_age_hours (default 24h)
  - NO_PRICE: option rows OR missing current_price OR missing decision_price
  - NEUTRAL: between thresholds
* Pre-decision news: top-k articles whose title regex-matches the
  ticker AND first_seen ∈ [decision_ts - lookback_h, decision_ts].
  Word-boundary regex (MU ≠ MUTUAL, AMD ≠ AMDOCS, $NVDA cashtag hits).
* Articles AFTER decision_ts MUST be excluded (the test catches the
  off-by-one a naive implementation introduces).
* Reasoning preference: longer ``decisions.reasoning`` joined by
  timestamp+ticker proximity if present; falls back to ``trades.reason``.
* Empty / garbage trades → state=NO_DATA, n_chains=0.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.decision_chain import (
    DEFAULT_GOOD_PCT,
    DEFAULT_BAD_PCT,
    DEFAULT_LOOKBACK_HOURS,
    _ticker_regex,
    _safe_float,
    _verdict,
    _pick_top_articles,
    build_decision_chain,
)


NOW = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)


def _trade(ticker: str, action: str, price: float, hours_ago: float,
           id_: int = 1, qty: float = 1.0, reason: str = "",
           option_type: str | None = None,
           expiry: str | None = None, strike: float | None = None) -> dict:
    ts = (NOW - timedelta(hours=hours_ago)).isoformat()
    return {
        "id": id_, "timestamp": ts, "ticker": ticker, "action": action,
        "qty": qty, "price": price, "value": qty * price,
        "reason": reason, "expiry": expiry, "strike": strike,
        "option_type": option_type,
    }


def _article(title: str, ai_score: float, hours_ago: float,
             urgency: int = 0, source: str = "rss") -> dict:
    return {
        "title": title, "ai_score": ai_score, "urgency": urgency,
        "first_seen": (NOW - timedelta(hours=hours_ago)).isoformat(),
        "source": source,
        "url": f"https://example.com/{abs(hash(title)) % 99999}",
    }


# ─── State ladder ──────────────────────────────────────────────────


class TestStateLadder:
    def test_empty_trades_is_no_data(self):
        r = build_decision_chain([], [], {}, now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["n_chains"] == 0
        assert r["chains"] == []
        assert "no FILLED trades" in r["headline"]

    def test_none_trades_is_no_data(self):
        r = build_decision_chain(None, None, None, now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["chains"] == []

    def test_trades_with_garbage_rows_filtered_silently(self):
        bad = [
            None,
            {"timestamp": "not-a-date", "ticker": "NVDA"},
            {"timestamp": NOW.isoformat(), "ticker": ""},   # empty ticker
            {"timestamp": NOW.isoformat(), "ticker": None},  # None ticker
        ]
        r = build_decision_chain(bad, [], {}, now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["chains"] == []


# ─── Verdict ladder ────────────────────────────────────────────────


class TestVerdictLadder:
    def test_buy_winning_above_good_pct_is_good(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=48)
        r = build_decision_chain([t], [], {"NVDA": 110.0}, now=NOW)
        assert r["chains"][0]["outcome"]["verdict"] == "GOOD"
        assert r["chains"][0]["outcome"]["intended_move_pct"] == 10.0

    def test_buy_losing_above_bad_pct_is_bad(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=72)
        r = build_decision_chain([t], [], {"NVDA": 95.0}, now=NOW)
        # -5% move on a BUY → intended -5%, ≤ -bad_pct (3) → BAD.
        assert r["chains"][0]["outcome"]["verdict"] == "BAD"
        assert r["chains"][0]["outcome"]["intended_move_pct"] == -5.0

    def test_sell_winning_when_price_fell_is_good(self):
        t = _trade("TQQQ", "SELL", 60.0, hours_ago=48)
        r = build_decision_chain([t], [], {"TQQQ": 55.0}, now=NOW)
        # SELL @ 60, now 55 → -8.33% abs but +8.33% intended → GOOD.
        assert r["chains"][0]["outcome"]["verdict"] == "GOOD"
        assert r["chains"][0]["outcome"]["intended_move_pct"] > 0

    def test_sell_losing_when_price_rose_is_bad(self):
        t = _trade("TQQQ", "SELL", 60.0, hours_ago=48)
        r = build_decision_chain([t], [], {"TQQQ": 70.0}, now=NOW)
        # SELL @ 60, now 70 → +16.67% abs but -16.67% intended → BAD.
        assert r["chains"][0]["outcome"]["verdict"] == "BAD"
        assert r["chains"][0]["outcome"]["intended_move_pct"] < 0

    def test_neutral_in_between(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=48)
        r = build_decision_chain([t], [], {"NVDA": 100.4}, now=NOW,
                                 good_pct=1.0, bad_pct=3.0)
        # +0.4% < 1% good_pct, > -3% bad_pct → NEUTRAL.
        assert r["chains"][0]["outcome"]["verdict"] == "NEUTRAL"

    def test_open_when_younger_than_open_min_age(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=2)
        r = build_decision_chain([t], [], {"NVDA": 120.0}, now=NOW,
                                 open_min_age_h=24)
        # Even a +20% move on a 2h-old trade stays OPEN — too early to judge.
        assert r["chains"][0]["outcome"]["verdict"] == "OPEN"
        assert r["chains"][0]["outcome"]["intended_move_pct"] == 20.0

    def test_no_price_when_current_price_missing(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=48)
        r = build_decision_chain([t], [], {"NVDA": None}, now=NOW)
        assert r["chains"][0]["outcome"]["verdict"] == "NO_PRICE"
        assert r["chains"][0]["outcome"]["abs_move_pct"] is None

    def test_option_row_always_no_price(self):
        # BUY_CALL with both prices available → still NO_PRICE.
        t = _trade("NVDA", "BUY_CALL", 2.0, hours_ago=48,
                   option_type="call", strike=120.0, expiry="2026-06-19")
        r = build_decision_chain([t], [], {"NVDA": 110.0}, now=NOW)
        assert r["chains"][0]["outcome"]["verdict"] == "NO_PRICE"
        assert r["chains"][0]["is_option"] is True

    def test_zero_decision_price_is_no_price(self):
        t = _trade("NVDA", "BUY", 0.0, hours_ago=48)
        r = build_decision_chain([t], [], {"NVDA": 110.0}, now=NOW)
        assert r["chains"][0]["outcome"]["verdict"] == "NO_PRICE"


# ─── Pre-decision news bucketing ───────────────────────────────────


class TestPreDecisionNews:
    def test_only_articles_within_lookback_before_decision_returned(self):
        # Decision at t-48h. Lookback 6h. Three articles:
        #   - 49h ago (1h BEFORE decision) → IN
        #   - 55h ago (7h before decision) → OUT (too old)
        #   - 47h ago (1h AFTER decision)  → OUT (post-decision — would
        #     contaminate the audit; this is the off-by-one a naive
        #     implementation will miss).
        t = _trade("NVDA", "BUY", 100.0, hours_ago=48)
        arts = [
            _article("NVDA: pre-decision tailwind", 9.0, hours_ago=49),
            _article("NVDA: pre-lookback noise", 5.0, hours_ago=55),
            _article("NVDA: AFTER decision (must exclude)", 9.5, hours_ago=47),
        ]
        r = build_decision_chain([t], arts, {"NVDA": 100.0}, now=NOW,
                                 lookback_h=6.0)
        news = r["chains"][0]["pre_decision_news"]
        titles = [a["title"] for a in news["top_articles"]]
        assert any("pre-decision tailwind" in t for t in titles)
        assert not any("AFTER decision" in t for t in titles)
        assert not any("pre-lookback noise" in t for t in titles)

    def test_topk_capped(self):
        # Trade 10h ago; articles in 11..22h ago (BEFORE decision, in window).
        t = _trade("NVDA", "BUY", 100.0, hours_ago=10)
        arts = [_article(f"NVDA news #{i}", 5.0 + i * 0.1,
                         hours_ago=11 + i * 0.1)
                for i in range(12)]
        r = build_decision_chain([t], arts, {"NVDA": 100.0}, now=NOW,
                                 article_top_k=3, lookback_h=24.0)
        assert len(r["chains"][0]["pre_decision_news"]["top_articles"]) == 3

    def test_topk_zero_returns_empty_list(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=10)
        # Article BEFORE the trade (hours_ago > trade hours_ago).
        arts = [_article("NVDA news", 9.0, hours_ago=12)]
        r = build_decision_chain([t], arts, {"NVDA": 100.0}, now=NOW,
                                 article_top_k=0, lookback_h=12.0)
        assert r["chains"][0]["pre_decision_news"]["top_articles"] == []

    def test_ticker_word_boundary_no_mu_in_mutual(self):
        # MUTUAL must NOT match the MU ticker.
        t = _trade("MU", "BUY", 100.0, hours_ago=10)
        # Both articles BEFORE the trade.
        arts = [
            _article("MUTUAL FUND inflows hit record", 9.0, hours_ago=12),
            _article("MU memory pricing rebound", 9.0, hours_ago=13),
        ]
        r = build_decision_chain([t], arts, {"MU": 100.0}, now=NOW,
                                 lookback_h=24.0)
        titles = [a["title"] for a in
                  r["chains"][0]["pre_decision_news"]["top_articles"]]
        assert any("MU memory" in t for t in titles)
        assert not any("MUTUAL FUND" in t for t in titles)

    def test_cashtag_match(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=10)
        # Article BEFORE the trade.
        arts = [_article("Strong $NVDA setup", 9.0, hours_ago=12)]
        r = build_decision_chain([t], arts, {"NVDA": 100.0}, now=NOW,
                                 lookback_h=24.0)
        assert r["chains"][0]["pre_decision_news"]["n_articles_returned"] == 1

    def test_articles_per_ticker_independent(self):
        # Two trades on different tickers. NVDA's article must NOT show
        # under AMD's chain, and vice versa. Articles BEFORE the matching
        # trade so they fall within the lookback window.
        ta = _trade("NVDA", "BUY", 100.0, hours_ago=20, id_=1)
        tb = _trade("AMD", "BUY", 80.0, hours_ago=10, id_=2)
        arts = [
            _article("NVDA earnings tailwind", 9.0, hours_ago=22),
            _article("AMD product launch", 8.0, hours_ago=12),
        ]
        r = build_decision_chain([ta, tb], arts,
                                 {"NVDA": 110.0, "AMD": 88.0}, now=NOW,
                                 lookback_h=12.0)
        by_tk = {c["ticker"]: c for c in r["chains"]}
        nvda_titles = [a["title"] for a in
                       by_tk["NVDA"]["pre_decision_news"]["top_articles"]]
        amd_titles = [a["title"] for a in
                      by_tk["AMD"]["pre_decision_news"]["top_articles"]]
        assert "NVDA earnings tailwind" in nvda_titles
        assert "NVDA earnings tailwind" not in amd_titles
        assert "AMD product launch" in amd_titles
        assert "AMD product launch" not in nvda_titles


# ─── Reasoning attachment ──────────────────────────────────────────


class TestReasoningAttachment:
    def test_falls_back_to_trade_reason_when_no_decisions(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=48,
                   reason="quick trade reason")
        r = build_decision_chain([t], [], {"NVDA": 100.0}, now=NOW)
        assert "quick trade reason" in r["chains"][0]["reason_excerpt"]

    def test_prefers_decisions_reasoning_when_matched_by_ts_ticker(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=48,
                   reason="short trade reason")
        # Decision row written ~5s after the trade — same ticker.
        d_ts = NOW - timedelta(hours=48, seconds=-2)
        decisions = [{
            "id": 1,
            "timestamp": d_ts.isoformat(),
            "action_taken": "BUY NVDA → FILLED",
            "reasoning": "FULL OPUS REASONING " * 5,
        }]
        r = build_decision_chain([t], [], {"NVDA": 100.0},
                                 decisions=decisions, now=NOW)
        excerpt = r["chains"][0]["reason_excerpt"]
        assert "FULL OPUS" in excerpt
        # Verify it's the long reasoning, NOT the short trade.reason.
        assert "short trade reason" not in excerpt

    def test_does_not_match_different_ticker(self):
        # Decision row near the trade ts but for a DIFFERENT ticker — the
        # builder must not mis-attach.
        t = _trade("NVDA", "BUY", 100.0, hours_ago=48,
                   reason="nvda short reason")
        d_ts = NOW - timedelta(hours=48, seconds=-1)
        decisions = [{
            "id": 1,
            "timestamp": d_ts.isoformat(),
            "action_taken": "SELL AMD → FILLED",   # different ticker
            "reasoning": "WRONG-TICKER reasoning",
        }]
        r = build_decision_chain([t], [], {"NVDA": 100.0},
                                 decisions=decisions, now=NOW)
        excerpt = r["chains"][0]["reason_excerpt"]
        assert "WRONG-TICKER" not in excerpt
        assert "nvda short reason" in excerpt


# ─── Output shape & sort ───────────────────────────────────────────


class TestOutputShape:
    def test_newest_first_in_chains(self):
        t_old = _trade("NVDA", "BUY", 100.0, hours_ago=72, id_=1)
        t_new = _trade("AMD", "SELL", 80.0, hours_ago=2, id_=2)
        r = build_decision_chain([t_old, t_new], [],
                                 {"NVDA": 105.0, "AMD": 80.0}, now=NOW)
        assert [c["ticker"] for c in r["chains"]] == ["AMD", "NVDA"]

    def test_n_cap_respected(self):
        trades = [_trade("NVDA", "BUY", 100.0, hours_ago=h, id_=h)
                  for h in range(1, 21)]
        r = build_decision_chain(trades, [], {"NVDA": 100.0}, n=5, now=NOW)
        assert len(r["chains"]) == 5

    def test_verdict_counts_sum_to_n_chains(self):
        ts = [
            _trade("NVDA", "BUY", 100.0, hours_ago=48),  # +10% → GOOD
            _trade("AMD",  "BUY", 80.0,  hours_ago=2),   # OPEN (too recent)
            _trade("XYZ",  "BUY", 50.0,  hours_ago=48),  # no price → NO_PRICE
            _trade("FOO",  "BUY_CALL", 2.0, hours_ago=48, option_type="call"),
        ]
        prices = {"NVDA": 110.0, "AMD": 80.0, "XYZ": None, "FOO": 5.0}
        r = build_decision_chain(ts, [], prices, now=NOW)
        counts = r["verdict_counts"]
        assert sum(counts.values()) == r["n_chains"] == 4
        assert counts["GOOD"] == 1
        assert counts["OPEN"] == 1
        assert counts["NO_PRICE"] == 2  # XYZ missing + FOO option

    def test_headline_names_counts(self):
        ts = [_trade("NVDA", "BUY", 100.0, hours_ago=48)]
        r = build_decision_chain(ts, [], {"NVDA": 110.0}, now=NOW)
        assert "GOOD" in r["headline"]


# ─── Robustness ────────────────────────────────────────────────────


class TestRobustness:
    def test_nan_ai_score_rejected_silently(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=48)
        arts = [{
            "title": "NVDA news", "ai_score": float("nan"),
            "first_seen": (NOW - timedelta(hours=48.5)).isoformat(),
            "source": "rss",
        }]
        r = build_decision_chain([t], arts, {"NVDA": 100.0}, now=NOW)
        # NaN ai_score article must be filtered, not raise.
        assert r["chains"][0]["pre_decision_news"]["n_articles_returned"] == 0

    def test_unparseable_article_first_seen_dropped(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=48)
        arts = [{
            "title": "NVDA stale", "ai_score": 9.0,
            "first_seen": "garbage-date", "source": "rss",
        }]
        r = build_decision_chain([t], arts, {"NVDA": 100.0}, now=NOW)
        assert r["chains"][0]["pre_decision_news"]["n_articles_returned"] == 0

    def test_article_missing_title_skipped(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=48)
        arts = [{
            "title": None, "ai_score": 9.0,
            "first_seen": (NOW - timedelta(hours=48.5)).isoformat(),
        }, {
            "title": "", "ai_score": 9.0,
            "first_seen": (NOW - timedelta(hours=48.5)).isoformat(),
        }]
        r = build_decision_chain([t], arts, {"NVDA": 100.0}, now=NOW)
        assert r["chains"][0]["pre_decision_news"]["n_articles_returned"] == 0

    def test_huge_reasoning_truncated_with_flag(self):
        t = _trade("NVDA", "BUY", 100.0, hours_ago=48,
                   reason="x" * 5000)
        r = build_decision_chain([t], [], {"NVDA": 100.0}, now=NOW)
        chain = r["chains"][0]
        assert len(chain["reason_excerpt"]) <= 400
        assert chain["reason_truncated"] is True


# ─── Pure helpers ──────────────────────────────────────────────────


class TestHelpers:
    def test_safe_float_rejects_nan_inf(self):
        assert _safe_float(float("nan")) is None
        assert _safe_float(float("inf")) is None
        assert _safe_float(float("-inf")) is None
        assert _safe_float("not-a-number") is None
        assert _safe_float(None) is None
        assert _safe_float("3.14") == 3.14

    def test_ticker_regex_boundaries(self):
        p = _ticker_regex("MU")
        assert p.search("MU MEMORY") is not None
        assert p.search("$MU CASHTAG") is not None
        assert p.search("MUTUAL FUND") is None
        assert p.search("MMU SOMETHING") is None
        assert p.search("FOOMU") is None

    def test_verdict_helper_directly(self):
        # +5% on BUY at 48h old, good=1, bad=3, open_min_age=24 → GOOD
        v, abs_p, int_p = _verdict("BUY", 100.0, 105.0, False, 48.0,
                                   1.0, 3.0, 24.0)
        assert v == "GOOD"
        assert abs_p == pytest.approx(5.0)
        assert int_p == pytest.approx(5.0)
