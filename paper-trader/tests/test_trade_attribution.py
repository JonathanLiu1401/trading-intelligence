"""Pure-function tests for ``build_trade_attribution``.

The endpoint does the DB I/O (paper_trader.db trades + articles.db
articles, applying the canonical live-only SQL clause); the *builder* is
the deterministic core — given (trades, articles), what does the
attribution payload look like? Mirrors ``test_correlation.py`` /
``test_news_edge.py`` (pure, no network, no DB).

Discriminating locks:
- **Ticker word-boundary**: bare-substring would alias ``MU`` ⊂ ``MUTUAL``
  / ``MA`` ⊂ ``MARKET``; ``\\bMU\\b`` does not.
- **Pre-trade window**: an article first_seen AFTER the fill cannot have
  driven it and must be excluded.
- **Pseudo-tickers**: ``CASH`` / ``NO_DECISION`` / ``BLOCKED`` are not
  fills — they must be dropped (invariant #11 / ``_parse_action_ticker``).
- **No fabrication**: a trade with zero matches surfaces
  ``n_attributed: 0`` and an honest "no article in window" headline,
  never a filler top article.
- **Determinism**: ai_score DESC; tie-break by recency-closest-to-fill
  (more plausibly causal). Same input ⇒ same output.
- **min_ai_score honesty**: the builder uses the live trader's own
  signal cutoff so the panel reflects what would have been in the
  prompt, not stream dregs.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.trade_attribution import (
    DEFAULT_MAX_PER_TRADE,
    DEFAULT_MIN_AI_SCORE,
    DEFAULT_WINDOW_HOURS,
    build_trade_attribution,
)


# ── helpers ────────────────────────────────────────────────────────────
NOW = datetime(2026, 5, 18, 14, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _trade(ticker: str, *, minutes_ago: float, action: str = "BUY",
           tid: int = 1, qty: float = 10.0, price: float = 100.0) -> dict:
    return {
        "id": tid,
        "timestamp": _iso(NOW - timedelta(minutes=minutes_ago)),
        "ticker": ticker,
        "action": action,
        "qty": qty,
        "price": price,
        "value": qty * price,
    }


def _article(title: str, *, minutes_ago: float, ai_score: float = 5.0,
             urgency: int = 0, url: str | None = None,
             source: str = "rss") -> dict:
    return {
        "title": title,
        "url": url or f"https://example.com/{abs(hash(title)) % 99999}",
        "source": source,
        "ai_score": ai_score,
        "urgency": urgency,
        "first_seen": _iso(NOW - timedelta(minutes=minutes_ago)),
    }


# ── pure/total contract ────────────────────────────────────────────────
def test_empty_inputs_state_no_data():
    out = build_trade_attribution([], [], now=NOW)
    assert out["state"] == "NO_DATA"
    assert out["n_trades"] == 0
    assert out["n_articles_examined"] == 0
    assert out["trades"] == []


def test_empty_articles_trade_still_appears_with_zero_attribution():
    """A trade with zero matching articles must appear with n_attributed=0;
    the operator wants to see the negative space (a fill that came from
    quant alone, no news catalyst) — not a missing row."""
    trades = [_trade("NVDA", minutes_ago=30)]
    out = build_trade_attribution(trades, [], now=NOW)
    assert out["state"] == "OK"
    assert out["n_trades"] == 1
    row = out["trades"][0]
    assert row["ticker"] == "NVDA"
    assert row["n_attributed"] == 0
    assert row["attributed_articles"] == []
    assert "no live-only article" in row["headline"]


def test_non_dict_inputs_never_raise():
    out = build_trade_attribution(
        [None, "x", 42, {}, _trade("NVDA", minutes_ago=30)],
        [None, "y", 7, {}, _article("NVDA jumps", minutes_ago=10)],
        now=NOW,
    )
    assert out["state"] == "OK"
    # only the well-formed trade survives
    assert out["n_trades"] == 1


# ── pre-trade window correctness ──────────────────────────────────────
def test_article_after_trade_is_excluded():
    """Article first_seen AFTER the fill cannot have caused it."""
    trades = [_trade("NVDA", minutes_ago=60)]  # 13:00
    arts = [_article("NVDA breakout", minutes_ago=30)]  # 13:30 — after fill
    out = build_trade_attribution(trades, arts, now=NOW)
    assert out["trades"][0]["n_attributed"] == 0


def test_article_at_exact_trade_time_is_included():
    trades = [_trade("NVDA", minutes_ago=60)]
    arts = [_article("NVDA breakout", minutes_ago=60)]  # tie → include
    out = build_trade_attribution(trades, arts, now=NOW)
    assert out["trades"][0]["n_attributed"] == 1


def test_article_older_than_window_is_excluded():
    """Default window 4h; an article 5h before the fill is out."""
    trades = [_trade("NVDA", minutes_ago=30)]
    arts = [
        _article("NVDA in window", minutes_ago=60),     # in
        _article("NVDA way old", minutes_ago=60 + 240 + 30),  # out (>4h before)
    ]
    out = build_trade_attribution(trades, arts, now=NOW)
    row = out["trades"][0]
    titles = [a["title"] for a in row["attributed_articles"]]
    assert "NVDA in window" in titles
    assert "NVDA way old" not in titles


def test_window_hours_param_widens_the_match():
    trades = [_trade("NVDA", minutes_ago=30)]
    arts = [_article("NVDA 6h before", minutes_ago=30 + 6 * 60)]
    # default 4h → 0 matches
    assert build_trade_attribution(
        trades, arts, now=NOW)["trades"][0]["n_attributed"] == 0
    # 8h window → 1 match
    assert build_trade_attribution(
        trades, arts, window_hours=8.0,
        now=NOW)["trades"][0]["n_attributed"] == 1


# ── ticker matching (word boundary, case-insensitive) ─────────────────
def test_ticker_word_boundary_blocks_substring_alias():
    """``MU`` (Micron) must not match ``MUTUAL`` or ``AMUSEMENT``."""
    trades = [_trade("MU", minutes_ago=10)]
    arts = [
        _article("MUTUAL fund flows hit record", minutes_ago=30),  # NO
        _article("MU surges on HBM design win", minutes_ago=30),    # YES
        _article("Amusement park stocks rally", minutes_ago=30),    # NO
    ]
    out = build_trade_attribution(trades, arts, now=NOW)
    row = out["trades"][0]
    assert row["n_attributed"] == 1
    assert row["attributed_articles"][0]["title"].startswith("MU surges")


def test_cashtag_dollar_sign_still_matches():
    """``$NVDA`` is a common cashtag; the ``$`` is non-word so the leading
    boundary fires."""
    trades = [_trade("NVDA", minutes_ago=15)]
    arts = [_article("$NVDA analyst upgrade lifts the floor", minutes_ago=30)]
    out = build_trade_attribution(trades, arts, now=NOW)
    assert out["trades"][0]["n_attributed"] == 1


def test_ticker_matching_is_case_insensitive():
    trades = [_trade("NVDA", minutes_ago=15)]
    arts = [_article("nvda jumps on guidance", minutes_ago=30)]
    out = build_trade_attribution(trades, arts, now=NOW)
    assert out["trades"][0]["n_attributed"] == 1


# ── pseudo-ticker carve-out (invariant #11) ───────────────────────────
@pytest.mark.parametrize("pseudo", ["CASH", "NONE", "NO_DECISION", "BLOCKED", ""])
def test_pseudo_tickers_dropped(pseudo):
    trades = [
        _trade(pseudo, minutes_ago=30, tid=1),
        _trade("NVDA", minutes_ago=30, tid=2),
    ]
    arts = [_article("NVDA up", minutes_ago=10)]
    out = build_trade_attribution(trades, arts, now=NOW)
    assert out["n_trades"] == 1
    assert out["n_skipped_pseudo_ticker"] == 1
    assert out["trades"][0]["ticker"] == "NVDA"


# ── ranking + dedup-by-max ────────────────────────────────────────────
def test_top_per_trade_capped_and_sorted_by_score_desc():
    """``max_per_trade`` caps the per-trade list; the top is highest
    ai_score; ties break by recency-closest-to-fill (more plausibly
    causal). Deterministic — same input ⇒ same output.

    Trade at minutes_ago=5; articles must precede it (minutes_ago > 5)."""
    trades = [_trade("NVDA", minutes_ago=5)]
    arts = [
        _article("NVDA mid 1", minutes_ago=120, ai_score=6.0),   # 115min before fill
        _article("NVDA top",   minutes_ago=60,  ai_score=9.0),
        _article("NVDA low",   minutes_ago=30,  ai_score=3.0),
        _article("NVDA mid 2", minutes_ago=15,  ai_score=6.0),   # tie w/ mid 1, closer to fill
    ]
    out = build_trade_attribution(
        trades, arts, max_per_trade=3, now=NOW)
    row = out["trades"][0]
    assert row["n_candidates"] == 4
    assert row["n_attributed"] == 3
    titles = [a["title"] for a in row["attributed_articles"]]
    # 9.0 first
    assert titles[0] == "NVDA top"
    # tie at 6.0: the one CLOSER to fill (mid 2, 10min before) wins over
    # mid 1 (115min before)
    assert titles[1] == "NVDA mid 2"
    assert titles[2] == "NVDA mid 1"
    # 3.0 dropped by cap
    assert "NVDA low" not in titles


# ── min_ai_score cutoff honesty ───────────────────────────────────────
def test_min_ai_score_cuts_below_threshold():
    """Below ``min_ai_score`` is excluded — the panel reflects what WOULD
    HAVE BEEN IN THE PROMPT (live trader's own signals.py min_score=4
    cutoff), not the long tail of the article stream."""
    trades = [_trade("NVDA", minutes_ago=5)]
    arts = [
        _article("NVDA noise", minutes_ago=30, ai_score=1.0),  # below 2.0 default
        _article("NVDA real",  minutes_ago=30, ai_score=4.0),  # above
    ]
    out = build_trade_attribution(trades, arts, now=NOW)
    assert out["trades"][0]["n_attributed"] == 1
    assert out["trades"][0]["attributed_articles"][0]["title"] == "NVDA real"


def test_min_ai_score_zero_keeps_everything_above_zero():
    trades = [_trade("NVDA", minutes_ago=5)]
    arts = [_article("NVDA bare", minutes_ago=30, ai_score=0.1)]
    out = build_trade_attribution(
        trades, arts, min_ai_score=0.0, now=NOW)
    assert out["trades"][0]["n_attributed"] == 1


# ── newest-first trade ordering ───────────────────────────────────────
def test_trades_returned_newest_first():
    trades = [
        _trade("AAPL", minutes_ago=180, tid=1),
        _trade("NVDA", minutes_ago=30,  tid=2),
        _trade("MSFT", minutes_ago=90,  tid=3),
    ]
    out = build_trade_attribution(trades, [], now=NOW)
    order = [r["ticker"] for r in out["trades"]]
    assert order == ["NVDA", "MSFT", "AAPL"]


# ── shape contract ─────────────────────────────────────────────────────
def test_output_contains_documented_params():
    out = build_trade_attribution([], [], now=NOW)
    assert out["window_hours"] == float(DEFAULT_WINDOW_HOURS)
    assert out["max_per_trade"] == DEFAULT_MAX_PER_TRADE
    assert out["min_ai_score"] == float(DEFAULT_MIN_AI_SCORE)
    assert "as_of" in out and out["as_of"].startswith("2026-05-18")


def test_attributed_article_row_shape():
    trades = [_trade("NVDA", minutes_ago=5, tid=42, action="BUY")]
    arts = [_article("NVDA earnings beat", minutes_ago=15,
                     ai_score=8.5, urgency=1, source="reuters")]
    row = build_trade_attribution(trades, arts, now=NOW)["trades"][0]
    art = row["attributed_articles"][0]
    # Every documented field is present and carries the right type
    assert art["title"] == "NVDA earnings beat"
    assert art["source"] == "reuters"
    assert isinstance(art["url"], str)
    assert art["ai_score"] == 8.5
    assert art["urgency"] == 1
    assert isinstance(art["first_seen"], str)
    # minutes_before_trade ≈ 15 - 5 = 10 minutes
    assert abs(art["minutes_before_trade"] - 10.0) < 0.1
    # per-trade headline carries the action + ticker + top title prefix
    assert row["headline"].startswith("BUY NVDA")
    assert "NVDA earnings beat" in row["headline"]


# ── malformed inputs degrade silently ─────────────────────────────────
def test_unparseable_article_timestamp_is_dropped_not_raised():
    trades = [_trade("NVDA", minutes_ago=5)]
    arts = [
        {"title": "NVDA up", "ai_score": 9, "first_seen": "not-an-iso"},
        _article("NVDA real", minutes_ago=30, ai_score=5.0),
    ]
    out = build_trade_attribution(trades, arts, now=NOW)
    assert out["trades"][0]["n_attributed"] == 1


def test_unparseable_trade_timestamp_drops_trade():
    trades = [
        {"id": 1, "ticker": "NVDA", "action": "BUY", "qty": 1, "price": 1,
         "timestamp": "garbage"},
        _trade("MSFT", minutes_ago=30, tid=2),
    ]
    out = build_trade_attribution(trades, [], now=NOW)
    assert out["n_trades"] == 1
    assert out["trades"][0]["ticker"] == "MSFT"
