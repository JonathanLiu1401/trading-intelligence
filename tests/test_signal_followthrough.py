"""Tests for analytics/signal_followthrough.py — pure, deterministic.

Contract under test: for every high-``ai_score`` live news signal that named a
watchlist ticker and was *visible to the trader at decision time* (its
``first_seen`` fell in the ``lookback_hours`` window ending at a decision's
timestamp), did the trader actually transact that ticker that cycle, and did
the signals it ACTED on outperform (forward, SPY-abnormal) the ones it
IGNORED?  This grades the *interaction* between the news pipeline and the
trader's decisions — news_edge grades the signal alone (ignoring the bot);
decision-drought grades inaction vs SPY (not vs the specific signals present).

Tests assert *exact* metric values for fixed fixtures, not "no crash".
"""
import sqlite3

import pytest

from paper_trader.analytics.signal_followthrough import (
    _fetch_live_articles,
    build_signal_followthrough,
)

TICKERS = ["NVDA", "AMD"]


def _series(start_close: float, step: float, n: int = 14,
            start_day: int = 1) -> list[tuple[str, float]]:
    """n daily bars 2026-06-DD with linear closes."""
    return [(f"2026-06-{start_day + i:02d}", start_close + step * i)
            for i in range(n)]


def _dec(ts_hour: int, action: str) -> dict:
    """A store-native decision row at 2026-06-01THH:00 (newest-first caller)."""
    return {"timestamp": f"2026-06-01T{ts_hour:02d}:00:00+00:00",
            "action_taken": action, "signal_count": 5}


def _art(ticker_text: str, score: float, hour: int, urgency: int = 0) -> dict:
    """An article first_seen at 2026-06-01THH:00 (before the decision hour)."""
    return {"text": f"{ticker_text} headline body",
            "ai_score": score, "urgency": urgency,
            "first_seen": f"2026-06-01T{hour:02d}:00:00+00:00"}


class TestExploitingSignals:
    """12 cycles: each sees an NVDA(+) and an AMD(flat) score-9 signal; the
    trader BUYs NVDA every cycle. Acted (NVDA) crushes ignored (AMD)."""

    def _data(self):
        decs, arts = [], []
        for h in range(12):
            decs.append(_dec(h + 1, "BUY NVDA → FILLED"))
            arts.append(_art("NVDA", 9.0, h + 1, urgency=1))
            arts.append(_art("AMD", 9.0, h + 1))
        ph = {"NVDA": _series(100.0, 2.0), "AMD": _series(50.0, 0.0)}
        spy = _series(400.0, 0.0)  # flat → abnormal == raw
        return decs, arts, ph, spy

    def test_counts_and_follow_through(self):
        decs, arts, ph, spy = self._data()
        r = build_signal_followthrough(decs, arts, ph, spy, TICKERS)
        assert r["n_decisions"] == 12
        assert r["n_signals"] == 24          # 12 NVDA + 12 AMD
        assert r["n_acted"] == 12            # NVDA transacted each cycle
        assert r["n_ignored"] == 12          # AMD never transacted
        assert r["follow_through_rate_pct"] == 50.0

    def test_acted_beats_ignored_exact(self):
        decs, arts, ph, spy = self._data()
        r = build_signal_followthrough(decs, arts, ph, spy, TICKERS)
        # All horizons well-sampled (12 ≥ _MIN_ACTED) → adaptive ref = 5d.
        assert r["reference_horizon"] == 5
        acted = r["acted"]
        assert acted["1"]["mean_raw_pct"] == 2.0    # 102/100-1
        assert acted["3"]["mean_raw_pct"] == 6.0    # 106/100-1
        assert acted["5"]["mean_raw_pct"] == 10.0   # 110/100-1
        assert acted["5"]["mean_abnormal_pct"] == 10.0  # SPY flat
        assert r["ignored"]["5"]["mean_abnormal_pct"] == 0.0
        assert r["selection_edge_pct"] == 10.0
        assert r["verdict"] == "EXPLOITING_SIGNALS"
        assert r["spy_adjusted"] is True


class TestMisusingSignals:
    """Mirror image: the trader BUYs the flat AMD and ignores the rising
    NVDA every cycle — negative selection edge."""

    def test_verdict_misusing(self):
        decs, arts = [], []
        for h in range(12):
            decs.append(_dec(h + 1, "BUY AMD → FILLED"))
            arts.append(_art("NVDA", 9.0, h + 1))
            arts.append(_art("AMD", 9.0, h + 1))
        ph = {"NVDA": _series(100.0, 2.0), "AMD": _series(50.0, 0.0)}
        spy = _series(400.0, 0.0)
        r = build_signal_followthrough(decs, arts, ph, spy, TICKERS)
        assert r["n_acted"] == 12 and r["n_ignored"] == 12
        assert r["acted"]["5"]["mean_abnormal_pct"] == 0.0   # AMD flat
        assert r["ignored"]["5"]["mean_abnormal_pct"] == 10.0  # NVDA +10
        assert r["selection_edge_pct"] == -10.0
        assert r["verdict"] == "MISUSING_SIGNALS"


class TestIgnoringFeed:
    """The real paper book: a steady high-score NVDA feed, trader always
    HOLDs. Follow-through is 0% → IGNORING_FEED, numerics still emitted."""

    def test_verdict_and_numerics(self):
        decs = [_dec(h + 1, "HOLD NVDA → HOLD") for h in range(20)]
        arts = [_art("NVDA", 9.0, h + 1) for h in range(20)]
        ph = {"NVDA": _series(100.0, 2.0)}
        spy = _series(400.0, 0.0)
        r = build_signal_followthrough(decs, arts, ph, spy, TICKERS)
        assert r["n_acted"] == 0
        assert r["n_ignored"] == 20
        assert r["follow_through_rate_pct"] == 0.0
        assert r["verdict"] == "IGNORING_FEED"
        # The ignored-bucket forward numbers must still be there (so the panel
        # can show "the signals you ignored moved +X%").
        assert r["ignored"]["5"]["mean_raw_pct"] == 10.0


class TestSpyAbnormalApplied:
    """abnormal = raw − SPY return over the identical span. With SPY +1/day,
    a raw +10% at 5d must report +8.75% abnormal (405/400-1 = 1.25%)."""

    def test_abnormal_subtraction(self):
        decs, arts = [], []
        for h in range(12):
            decs.append(_dec(h + 1, "BUY NVDA → FILLED"))
            arts.append(_art("NVDA", 9.0, h + 1))
        ph = {"NVDA": _series(100.0, 2.0)}
        spy = _series(400.0, 1.0)  # +1/day → +1.25% over 5d
        r = build_signal_followthrough(decs, arts, ph, spy, TICKERS)
        assert r["acted"]["5"]["mean_raw_pct"] == 10.0
        assert r["acted"]["5"]["mean_abnormal_pct"] == 8.75


class TestSampleSizeHonesty:
    def test_insufficient_withholds_verdict_keeps_numerics(self):
        # 3 cycles only — below _MIN_RESOLVED. Numerics present, verdict gated.
        decs, arts = [], []
        for h in range(3):
            decs.append(_dec(h + 1, "BUY NVDA → FILLED"))
            arts.append(_art("NVDA", 9.0, h + 1))
        ph = {"NVDA": _series(100.0, 2.0)}
        spy = _series(400.0, 0.0)
        r = build_signal_followthrough(decs, arts, ph, spy, TICKERS)
        assert r["verdict"] == "INSUFFICIENT"
        assert r["n_signals"] == 3
        assert r["acted"]["1"]["mean_raw_pct"] == 2.0  # still computed

    def test_no_data_on_empty(self):
        ph = {"NVDA": _series(100.0, 2.0)}
        r = build_signal_followthrough([], [], ph, [], TICKERS)
        assert r["verdict"] == "NO_DATA"
        r2 = build_signal_followthrough(
            [_dec(1, "HOLD NVDA → HOLD")], [], ph, [], TICKERS)
        assert r2["verdict"] == "NO_DATA"
        assert r2["n_signals"] == 0


class TestWatchlistWordBoundary:
    """AMDOCS must not resolve to AMD (news_edge regex precedent)."""

    def test_amdocs_does_not_match_amd(self):
        decs = [_dec(h + 1, "HOLD CASH → HOLD") for h in range(12)]
        arts = [_art("AMDOCS", 9.0, h + 1) for h in range(12)]
        ph = {"AMD": _series(50.0, 1.0)}
        r = build_signal_followthrough(decs, arts, ph, [], ["AMD"])
        assert r["n_signals"] == 0
        assert r["verdict"] == "NO_DATA"


class TestPerCycleTickerDedup:
    """Three NVDA articles in one window count as ONE signal that cycle —
    the unit is 'did it act on the NVDA news', not article spam."""

    def test_dedup_to_one_signal(self):
        decs = [_dec(5, "HOLD NVDA → HOLD")]
        arts = [_art("NVDA", 5.0, 3), _art("NVDA", 9.0, 4),
                _art("NVDA", 7.0, 4)]
        ph = {"NVDA": _series(100.0, 2.0)}
        r = build_signal_followthrough(decs, arts, ph, [], ["NVDA"])
        assert r["n_signals"] == 1


class TestWindowBoundary:
    """Only articles whose first_seen is in (decision − lookback, decision]
    are visible. Future news and stale news must not count."""

    def test_only_in_window_articles_count(self):
        # decision at 10:00, lookback 2h → window (08:00, 10:00].
        dec = {"timestamp": "2026-06-01T10:00:00+00:00",
               "action_taken": "HOLD NVDA → HOLD", "signal_count": 1}
        in_win = _art("NVDA", 9.0, 9)      # 09:00 — visible
        too_old = _art("AMD", 9.0, 7)      # 07:00 — before window
        future = _art("NVDA", 9.0, 11)     # 11:00 — after decision
        ph = {"NVDA": _series(100.0, 2.0), "AMD": _series(50.0, 1.0)}
        r = build_signal_followthrough(
            [dec], [in_win, too_old, future], ph, [], TICKERS,
            lookback_hours=2)
        assert r["n_signals"] == 1  # only the 09:00 NVDA


class TestFetchLiveArticlesFilter:
    """The reusable SQL fetch must apply the canonical live-only filter so a
    backtest:// / backtest_* / opus_annotation* row can never be scored."""

    def _db(self, tmp_path):
        p = tmp_path / "articles.db"
        conn = sqlite3.connect(p)
        conn.execute(
            "CREATE TABLE articles (id TEXT PRIMARY KEY, url TEXT, title TEXT, "
            "source TEXT, ai_score REAL, urgency INTEGER, first_seen TEXT, "
            "full_text BLOB)")
        rows = [
            ("1", "https://reuters.com/a", "NVDA live news", "rss",
             9.0, 1, "2026-06-01T09:00:00+00:00"),
            ("2", "backtest://run_1/2026-06-01/BUY/NVDA", "NVDA bt", "rss",
             9.0, 0, "2026-06-01T09:00:00+00:00"),
            ("3", "https://x.com/b", "NVDA winner", "backtest_run_42_winner",
             9.0, 0, "2026-06-01T09:00:00+00:00"),
            ("4", "https://x.com/c", "NVDA lesson", "opus_annotation_cycle_3",
             9.0, 0, "2026-06-01T09:00:00+00:00"),
            ("5", "https://bloomberg.com/d", "NVDA low score", "rss",
             1.0, 0, "2026-06-01T09:00:00+00:00"),
        ]
        for r in rows:
            conn.execute(
                "INSERT INTO articles (id,url,title,source,ai_score,urgency,"
                "first_seen,full_text) VALUES (?,?,?,?,?,?,?,NULL)", r)
        conn.commit()
        conn.close()
        return str(p)

    def test_only_live_high_score_rows(self, tmp_path):
        path = self._db(tmp_path)
        arts = _fetch_live_articles(path, "2026-06-01T00:00:00+00:00",
                                    min_score=4.0)
        titles = sorted(a["text"].split(" headline")[0]
                        if " headline" in a["text"] else a["text"]
                        for a in arts)
        # Only the single live, ai_score≥4 row survives both filters.
        assert len(arts) == 1
        assert "NVDA live news" in arts[0]["text"]
        assert arts[0]["ai_score"] == 9.0
