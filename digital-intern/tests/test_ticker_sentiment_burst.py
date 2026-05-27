"""Pin ``ArticleStore.ticker_sentiment_burst`` — the title-keyword directional
sentiment primitive (BULLISH / BEARISH / MIXED / QUIET / DARK per held ticker).

Sibling to ``ticker_news_burst`` (volume) and ``ticker_mention_velocity``
(rate-of-change). This primitive carries the orthogonal DIRECTION axis the
score-based sentiment analytics (ml_score deltas) can't separate from
relevance/urgency.

Load-bearing invariants verified:
  * backtest:// URLs and backtest_* / opus_annotation* sources are excluded
    from the count denominator and the verdict (otherwise a synthetic
    backtest row could flip a held ticker's live verdict).
  * Read-only: no ai_score / ml_score / score_source / urgency mutation
    (pinned by an explicit pre/post DB row read).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _stale_iso(hours_ago: int = 100) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _insert_raw(store, *, id, url, title, source="rss", first_seen=None,
                ai_score=0.0, ml_score=None, score_source=None,
                urgency=0, kw_score=1.0):
    if first_seen is None:
        first_seen = _recent_iso()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, urgency,
             first_seen, 0, ml_score, score_source),
        )
        store.conn.commit()


class TestBullishVerdict:
    def test_unanimous_bull_yields_bullish(self, store):
        # Use ALL-CAPS NVDA throughout — same word-boundary discipline as
        # ticker_news_burst / held_ticker_latest_article. "Nvidia" prose
        # doesn't match NVDA, which is the documented behaviour.
        for i, title in enumerate([
            "NVDA surges 10% on earnings beat",
            "NVDA jumps after upgrade by Morgan Stanley",
            "NVDA rallies to record high — analysts boost target",
        ]):
            _insert_raw(store, id=f"a{i}", url=f"https://x.com/{i}",
                        title=title)
        out = store.ticker_sentiment_burst(tickers=["NVDA"], window_h=24.0)
        row = out["by_ticker"][0]
        assert row["ticker"] == "NVDA"
        assert row["count_window"] == 3
        assert row["bull"] == 3
        assert row["bear"] == 0
        assert row["intensity"] == 1.0
        assert row["verdict"] == "BULLISH"
        assert out["most_bullish"] == "NVDA"
        assert out["most_bearish"] is None

    def test_single_bull_is_quiet(self, store):
        """One bullish mention is not enough — magnitude bar = 2.

        Otherwise a single benign headline ("NVDA rises slightly")
        would flip the verdict and produce noise."""
        _insert_raw(store, id="a", url="https://x.com/1",
                    title="NVDA rises modestly to $200")
        out = store.ticker_sentiment_burst(tickers=["NVDA"], window_h=24.0)
        row = out["by_ticker"][0]
        assert row["count_window"] == 1
        assert row["bull"] == 1
        assert row["verdict"] == "QUIET"


class TestBearishVerdict:
    def test_unanimous_bear_yields_bearish(self, store):
        # Use ALL-CAPS MU throughout — \b\$?TICKER\b discipline.
        for i, title in enumerate([
            "MU plunges 15% after guidance miss",
            "MU drops on downgrade by Goldman",
            "MU tumbles to 6-month low — lawsuit fears mount",
        ]):
            _insert_raw(store, id=f"a{i}", url=f"https://x.com/{i}",
                        title=title)
        out = store.ticker_sentiment_burst(tickers=["MU"], window_h=24.0)
        row = out["by_ticker"][0]
        assert row["bull"] == 0
        assert row["bear"] == 3
        assert row["intensity"] == -1.0
        assert row["verdict"] == "BEARISH"
        assert out["most_bearish"] == "MU"

    def test_two_bears_against_one_bull_yields_bearish(self, store):
        """Verdict requires 2× dominance. 2 bear vs 1 bull = 2× exactly,
        verdict should be BEARISH."""
        _insert_raw(store, id="b1", url="https://x.com/1",
                    title="MU plunges 8% after CPI miss")
        _insert_raw(store, id="b2", url="https://x.com/2",
                    title="MU tumbles to new monthly low")
        _insert_raw(store, id="b3", url="https://x.com/3",
                    title="MU rallies briefly in afterhours")
        out = store.ticker_sentiment_burst(tickers=["MU"], window_h=24.0)
        row = out["by_ticker"][0]
        assert row["bull"] == 1
        assert row["bear"] == 2
        assert row["verdict"] == "BEARISH"


class TestMixedVerdict:
    def test_balanced_bull_bear_is_mixed(self, store):
        for i, title in enumerate([
            "AMD jumps 5% on Q1 beat",
            "AMD surges to new high",
            "AMD plunges on broader chip selloff",
            "AMD tumbles 7% in afterhours",
        ]):
            _insert_raw(store, id=f"a{i}", url=f"https://x.com/{i}",
                        title=title)
        out = store.ticker_sentiment_burst(tickers=["AMD"], window_h=24.0)
        row = out["by_ticker"][0]
        assert row["bull"] == 2
        assert row["bear"] == 2
        # intensity = 0 because (2-2)/4 = 0
        assert row["intensity"] == 0.0
        # 4 directional rows, neither dominates 2× → MIXED
        assert row["verdict"] == "MIXED"
        assert out["most_bullish"] is None
        assert out["most_bearish"] is None


class TestDarkVerdict:
    def test_no_mentions_yields_dark(self, store):
        """A held ticker the wire didn't touch in the window → DARK,
        zero counts. The analyst needs to know this is silence, not signal."""
        # Insert articles that DON'T mention NVDA — different ticker entirely.
        _insert_raw(store, id="a", url="https://x.com/1",
                    title="MSFT rises on AI announcement")
        out = store.ticker_sentiment_burst(tickers=["NVDA"], window_h=24.0)
        row = out["by_ticker"][0]
        assert row["count_window"] == 0
        assert row["bull"] == 0
        assert row["bear"] == 0
        assert row["verdict"] == "DARK"


class TestQuietVerdict:
    def test_one_neutral_mention_is_quiet(self, store):
        """An article that mentions the ticker but has no directional verb
        (CEO speaks, conference, profile piece) is neutral coverage — QUIET."""
        _insert_raw(store, id="a", url="https://x.com/1",
                    title="NVDA CEO speaks at GTC conference today")
        out = store.ticker_sentiment_burst(tickers=["NVDA"], window_h=24.0)
        row = out["by_ticker"][0]
        assert row["count_window"] == 1
        assert row["neutral"] == 1
        assert row["bull"] == 0
        assert row["bear"] == 0
        assert row["verdict"] == "QUIET"


class TestBacktestIsolation:
    def test_backtest_url_does_not_count(self, store):
        """Critical invariant #1: a backtest:// row mentioning a held ticker
        with bullish keywords MUST NOT register in the live sentiment count.
        Otherwise a synthetic injection could flip a real held name's verdict."""
        # 3 backtest rows that would have been BULLISH
        for i, title in enumerate([
            "NVDA surges in backtest replay",
            "NVDA jumps in synthetic run",
            "NVDA rallies in cycle 42",
        ]):
            _insert_raw(store, id=f"bt{i}",
                        url=f"backtest://run_1/2026-01-01/BUY/NVDA/{i}",
                        title=title, source="backtest_run_1")
        # 2 LIVE rows that would NOT trigger BULLISH alone
        _insert_raw(store, id="live1", url="https://reuters.com/x",
                    title="NVDA rises modestly")  # 1 bull, neutral verdict
        out = store.ticker_sentiment_burst(tickers=["NVDA"], window_h=24.0)
        row = out["by_ticker"][0]
        # backtest rows excluded — only the 1 live row counts → QUIET
        assert row["count_window"] == 1
        assert row["bull"] == 1
        assert row["verdict"] == "QUIET"

    def test_opus_annotation_source_does_not_count(self, store):
        _insert_raw(store, id="op1", url="https://x.com/1",
                    title="NVDA surges in Opus annotation pass",
                    source="opus_annotation_cycle_3")
        _insert_raw(store, id="op2", url="https://x.com/2",
                    title="NVDA jumps after opus annotation pass",
                    source="opus_annotation_cycle_4")
        out = store.ticker_sentiment_burst(tickers=["NVDA"], window_h=24.0)
        row = out["by_ticker"][0]
        assert row["count_window"] == 0
        assert row["verdict"] == "DARK"


class TestWindowSlice:
    def test_stale_rows_outside_window_excluded(self, store):
        """A 100h-old row must not count in a 24h window — otherwise a
        week-old earnings beat would still register as 'NVDA bullish today'."""
        _insert_raw(store, id="old1", url="https://x.com/1",
                    title="NVDA surged last week",
                    first_seen=_stale_iso(hours_ago=100))
        _insert_raw(store, id="old2", url="https://x.com/2",
                    title="NVDA jumped last month",
                    first_seen=_stale_iso(hours_ago=200))
        out = store.ticker_sentiment_burst(tickers=["NVDA"], window_h=24.0)
        row = out["by_ticker"][0]
        assert row["count_window"] == 0
        assert row["verdict"] == "DARK"


class TestWordBoundary:
    def test_AMDOCS_does_not_match_AMD(self, store):
        r"""\b\$?TICKER\b discipline — AMDOCS must not leak as AMD."""
        _insert_raw(store, id="a", url="https://x.com/1",
                    title="AMDOCS surges 8% on contract win")
        out = store.ticker_sentiment_burst(tickers=["AMD"], window_h=24.0)
        row = out["by_ticker"][0]
        assert row["count_window"] == 0, "AMDOCS title leaked as AMD mention"

    def test_dollar_prefix_matches(self, store):
        r"""$NVDA must match the NVDA ticker — same discipline as
        ticker_news_burst / held_ticker_latest_article."""
        for i in range(2):
            _insert_raw(store, id=f"a{i}", url=f"https://x.com/{i}",
                        title=f"$NVDA surges {5+i}% in pre-market")
        out = store.ticker_sentiment_burst(tickers=["NVDA"], window_h=24.0)
        row = out["by_ticker"][0]
        assert row["count_window"] == 2
        assert row["bull"] == 2
        assert row["verdict"] == "BULLISH"


class TestReadOnly:
    def test_no_db_mutation(self, store):
        """All four load-bearing invariants intact: pre/post DB row state must be
        byte-identical after a ticker_sentiment_burst call (no ai_score /
        ml_score / score_source / urgency mutation)."""
        _insert_raw(store, id="a", url="https://x.com/1",
                    title="NVDA surges 10%",
                    ai_score=5.0, ml_score=6.5, score_source="llm", urgency=1)
        before = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id='a'"
        ).fetchone()
        store.ticker_sentiment_burst(tickers=["NVDA"], window_h=24.0)
        after = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id='a'"
        ).fetchone()
        assert before == after, "ticker_sentiment_burst mutated the DB row"


class TestInputHygiene:
    def test_empty_ticker_list_returns_empty_struct(self, store):
        out = store.ticker_sentiment_burst(tickers=[], window_h=24.0)
        assert out["by_ticker"] == []
        assert out["n_window"] == 0
        assert out["most_bullish"] is None
        assert out["most_bearish"] is None

    def test_short_ticker_filtered(self, store):
        """Sub-2-char "tickers" (overmatching, e.g. "A") are filtered.
        Same hygiene as ticker_news_burst / held_ticker_latest_article."""
        out = store.ticker_sentiment_burst(tickers=["A"], window_h=24.0)
        assert out["by_ticker"] == []

    def test_long_ticker_filtered(self, store):
        """>8 char tickers (foreign / suffix-bearing) are filtered.
        Same hygiene as ticker_news_burst."""
        out = store.ticker_sentiment_burst(tickers=["005930.KS"], window_h=24.0)
        assert out["by_ticker"] == []

    def test_dedupes_case_variants(self, store):
        out = store.ticker_sentiment_burst(tickers=["NVDA", "nvda", "Nvda"],
                                            window_h=24.0)
        # All three should canonicalize to one "NVDA" entry.
        tickers_seen = [r["ticker"] for r in out["by_ticker"]]
        assert tickers_seen == ["NVDA"]


class TestSortOrder:
    def test_strongest_direction_first(self, store):
        """Sort: |intensity| desc, count_window desc, ticker asc."""
        # NVDA: 3 bull 0 bear → +1.0
        for i in range(3):
            _insert_raw(store, id=f"n{i}", url=f"https://x.com/n{i}",
                        title=f"NVDA surges {5+i}% today")
        # MU: 1 bull 1 bear → 0.0
        _insert_raw(store, id="m1", url="https://x.com/m1",
                    title="MU rises modestly")
        _insert_raw(store, id="m2", url="https://x.com/m2",
                    title="MU drops in afterhours")
        # MSFT: 2 bear 0 bull → -1.0
        _insert_raw(store, id="s1", url="https://x.com/s1",
                    title="MSFT plunges on lawsuit")
        _insert_raw(store, id="s2", url="https://x.com/s2",
                    title="MSFT downgraded by analyst")
        out = store.ticker_sentiment_burst(tickers=["NVDA", "MU", "MSFT"],
                                            window_h=24.0)
        # NVDA and MSFT both at |intensity|=1.0; NVDA wins on count_window=3 vs 2
        order = [r["ticker"] for r in out["by_ticker"]]
        assert order[0] == "NVDA"
        assert order[1] == "MSFT"
        assert order[2] == "MU"
        assert out["most_bullish"] == "NVDA"
        assert out["most_bearish"] == "MSFT"


class TestSchemaShape:
    def test_top_level_keys(self, store):
        """Pin the schema: the chat/dashboard consumers depend on this shape
        being stable. Same discipline as ticker_news_burst."""
        out = store.ticker_sentiment_burst(tickers=["NVDA"], window_h=1.0)
        assert set(out.keys()) >= {
            "window_h", "n_window", "by_ticker",
            "n_bullish", "n_bearish",
            "most_bullish", "most_bearish",
        }
        assert isinstance(out["by_ticker"], list)

    def test_per_ticker_keys(self, store):
        _insert_raw(store, id="a", url="https://x.com/1",
                    title="NVDA surges 10%")
        out = store.ticker_sentiment_burst(tickers=["NVDA"], window_h=1.0)
        row = out["by_ticker"][0]
        assert set(row.keys()) >= {
            "ticker", "count_window", "bull", "bear", "neutral",
            "intensity", "verdict",
        }


class TestLazyTickerDefault:
    def test_default_uses_live_portfolio_tickers(self, store):
        """When ``tickers=None`` the primitive must default to
        ``ml.features.LIVE_PORTFOLIO_TICKERS`` — the SSOT for the held +
        watched universe. Mirrors ticker_news_burst / held_ticker_latest_article."""
        from ml.features import LIVE_PORTFOLIO_TICKERS
        out = store.ticker_sentiment_burst(tickers=None, window_h=1.0)
        returned = {r["ticker"] for r in out["by_ticker"]}
        # Every cleaned (2..8 char) live portfolio ticker should be in the output.
        for t in LIVE_PORTFOLIO_TICKERS:
            t_up = t.strip().upper()
            if 2 <= len(t_up) <= 8:
                assert t_up in returned, (
                    f"LIVE_PORTFOLIO_TICKERS member {t_up!r} missing from default output"
                )
