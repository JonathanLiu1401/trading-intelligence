"""Tests for analytics/news_edge.py — pure, deterministic.

Contract under test: resolve a watchlist ticker from article text, find its
close on/after the article day, compute 1/3/5-trading-day forward return both
raw and SPY-abnormal, band by ai_score, and judge whether the top score band
shows a real (positive, monotonic, well-sampled) abnormal edge.
"""
from paper_trader.analytics.news_edge import (
    _index_at_or_after,
    build_news_edge,
)

TICKERS = ["NVDA", "AMD"]


def _series(start_close: float, step: float, n: int = 12,
            start_day: int = 1) -> list[tuple[str, float]]:
    """n daily bars 2026-06-DD with linear closes."""
    return [(f"2026-06-{start_day + i:02d}", start_close + step * i)
            for i in range(n)]


class TestIndexAtOrAfter:
    def test_exact_and_gap_and_overflow(self):
        dates = ["2026-06-03", "2026-06-04", "2026-06-05"]
        assert _index_at_or_after(dates, "2026-06-03") == 0
        # Article before the first bar resolves to the first bar.
        assert _index_at_or_after(dates, "2026-06-01") == 0
        assert _index_at_or_after(dates, "2026-06-04") == 1
        # Article after the last bar → no forward window.
        assert _index_at_or_after(dates, "2026-06-09") is None


class TestEdgeConfirmed:
    """10 score-9 NVDA headlines (price +2/day) vs 10 score-3 AMD (flat)."""

    def _data(self):
        arts = []
        for _ in range(10):
            arts.append({"text": "NVDA AI chip demand surges on hyperscaler capex",
                         "ai_score": 9.0, "urgency": 1,
                         "published": "2026-06-01T12:00:00+00:00"})
        for _ in range(10):
            arts.append({"text": "AMD product gets a lukewarm reception",
                         "ai_score": 3.0, "urgency": 0,
                         "published": "2026-06-01T12:00:00+00:00"})
        ph = {
            "NVDA": _series(100.0, 2.0),   # +2/day
            "AMD": _series(50.0, 0.0),     # flat
        }
        spy = _series(400.0, 0.0)          # flat → abnormal == raw
        return arts, ph, spy

    def test_top_band_forward_returns(self):
        arts, ph, spy = self._data()
        r = build_news_edge(arts, ph, spy, TICKERS)
        top = next(b for b in r["bands"] if b["band"] == "8.0+")["horizons"]
        # entry close 100 (2026-06-01). +2/day linear.
        assert top["1"]["n"] == 10
        assert top["1"]["mean_raw_pct"] == 2.0      # 102/100-1
        assert top["3"]["mean_raw_pct"] == 6.0      # 106/100-1
        assert top["5"]["mean_raw_pct"] == 10.0     # 110/100-1
        # SPY flat → abnormal equals raw exactly.
        assert top["3"]["mean_abnormal_pct"] == 6.0
        assert top["3"]["abnormal_hit_rate"] == 100.0
        assert top["5"]["raw_up_rate"] == 100.0

    def test_bottom_band_flat(self):
        arts, ph, spy = self._data()
        r = build_news_edge(arts, ph, spy, TICKERS)
        bot = next(b for b in r["bands"] if b["band"] == "2.0-4.0")["horizons"]
        assert bot["3"]["n"] == 10
        assert bot["3"]["mean_raw_pct"] == 0.0
        assert bot["3"]["abnormal_hit_rate"] == 0.0   # 0 is not > 0

    def test_verdict_edge_confirmed(self):
        arts, ph, spy = self._data()
        r = build_news_edge(arts, ph, spy, TICKERS)
        assert r["verdict"] == "EDGE_CONFIRMED"
        # All horizons are well-sampled here, so the adaptive reference picks
        # the *longest* one (a 5d edge is the strongest claim).
        assert r["reference_horizon"] == 5
        assert r["n_resolved"] == 20
        assert r["spy_adjusted"] is True


class TestAdaptiveReferenceHorizon:
    """The real-world early-data case: only a 1-day forward window exists.

    digital-intern's articles.db starts with only ~days of history, so 3d/5d
    forward bars don't exist yet. The panel must report the 1d edge instead of
    going all-dashes / INSUFFICIENT_DATA on a horizon that *can't* have data.
    """

    def test_degrades_to_one_day_when_only_1d_has_a_window(self):
        arts = [{"text": "NVDA breakout on AI demand", "ai_score": 9.0,
                 "urgency": 1, "published": "2026-06-01T00:00:00+00:00"}
                for _ in range(8)]
        # Only two bars → i0=0 has a +1d bar, but +3d / +5d fall off the end.
        ph = {"NVDA": _series(100.0, 2.0, n=2)}
        spy = _series(400.0, 0.0, n=2)
        r = build_news_edge(arts, ph, spy, TICKERS)
        assert r["reference_horizon"] == 1
        assert r["verdict"] == "EDGE_CONFIRMED"
        top = next(b for b in r["bands"] if b["band"] == "8.0+")["horizons"]
        assert top["1"]["n"] == 8
        assert top["1"]["mean_abnormal_pct"] == 2.0
        assert top["3"]["n"] == 0       # no 3d window yet — honest zero
        assert top["5"]["mean_abnormal_pct"] is None


class TestSpyAdjustmentIsApplied:
    """The core honesty control: abnormal = raw − spy over the same span."""

    def test_abnormal_subtracts_spy(self):
        arts = [{"text": "NVDA rallies", "ai_score": 8.5, "urgency": 0,
                 "published": "2026-06-01T00:00:00+00:00"} for _ in range(8)]
        ph = {"NVDA": _series(100.0, 2.0)}   # +2/day → +2% at 1d
        spy = _series(400.0, 4.0)            # +4/day → +1% at 1d
        r = build_news_edge(arts, ph, spy, TICKERS)
        h1 = next(b for b in r["bands"] if b["band"] == "8.0+")["horizons"]["1"]
        assert h1["mean_raw_pct"] == 2.0
        # spy 400→404 over the same day = +1.0%; abnormal = 2.0 − 1.0
        assert h1["mean_abnormal_pct"] == 1.0
        assert h1["abnormal_hit_rate"] == 100.0


class TestNoEdge:
    def test_high_score_ticker_falls(self):
        arts = [{"text": "NVDA news", "ai_score": 9.0, "urgency": 0,
                 "published": "2026-06-01T00:00:00+00:00"} for _ in range(10)]
        ph = {"NVDA": _series(100.0, -2.0)}  # falls 2/day
        spy = _series(400.0, 0.0)
        r = build_news_edge(arts, ph, spy, TICKERS)
        assert r["verdict"] == "NO_EDGE"
        top = next(b for b in r["bands"] if b["band"] == "8.0+")["horizons"]
        assert top["3"]["mean_abnormal_pct"] == -6.0


class TestInsufficientData:
    def test_too_few_top_band_samples(self):
        arts = [{"text": "NVDA news", "ai_score": 9.0, "urgency": 0,
                 "published": "2026-06-01T00:00:00+00:00"}]
        ph = {"NVDA": _series(100.0, 2.0)}
        spy = _series(400.0, 0.0)
        r = build_news_edge(arts, ph, spy, TICKERS)
        assert r["verdict"] == "INSUFFICIENT_DATA"


class TestTickerResolution:
    def test_resolution_order_and_unmatched(self):
        arts = [
            {"text": "no tickers here", "ai_score": 9.0, "urgency": 0,
             "published": "2026-06-01T00:00:00+00:00"},
            {"text": "$AMD breaks out", "ai_score": 9.0, "urgency": 0,
             "published": "2026-06-01T00:00:00+00:00"},
        ]
        ph = {"AMD": _series(50.0, 1.0)}     # only AMD priced
        spy = _series(400.0, 0.0)
        r = build_news_edge(arts, ph, spy, TICKERS)
        # First article matches nothing; second resolves AMD via $-prefix.
        assert r["n_articles"] == 2
        assert r["n_resolved"] == 1

    def test_substring_does_not_falsely_match(self):
        # "AMDOCS" must not match the AMD word-boundary pattern.
        arts = [{"text": "AMDOCS earnings", "ai_score": 9.0, "urgency": 0,
                 "published": "2026-06-01T00:00:00+00:00"}]
        ph = {"AMD": _series(50.0, 1.0)}
        spy = _series(400.0, 0.0)
        r = build_news_edge(arts, ph, spy, TICKERS)
        assert r["n_resolved"] == 0


class TestEmpty:
    def test_no_articles(self):
        r = build_news_edge([], {}, [], TICKERS)
        assert r["verdict"] == "INSUFFICIENT_DATA"
        assert r["n_articles"] == 0
        assert r["n_resolved"] == 0
