"""Tests for analytics/source_edge.py — pure, deterministic.

Contract under test: normalise digital-intern's high-cardinality ``source``
column into a *collector family*, resolve a watchlist ticker from the article
text (same word-boundary rule as news_edge), compute 1/3/5-trading-day forward
return both raw and SPY-abnormal **pooled across score bands** per family, and
judge which collector's scored news actually precedes abnormal moves — the
"which of my ~17 collectors is worth trusting" question nothing else answers.

Exact-value fixtures mirror test_news_edge.py: linear price series so every
bucket mean is hand-checkable, SPY flat ⇒ abnormal == raw.
"""
import sqlite3

from paper_trader.analytics.source_edge import (
    _MIN_SOURCE_N,
    _fetch_source_articles,
    _source_family,
    build_source_edge,
)

TICKERS = ["NVDA", "AMD"]


def _series(start_close: float, step: float, n: int = 12,
            start_day: int = 1) -> list[tuple[str, float]]:
    """n daily bars 2026-06-DD with linear closes."""
    return [(f"2026-06-{start_day + i:02d}", start_close + step * i)
            for i in range(n)]


class TestSourceFamily:
    """The normalisation rule is load-bearing: without it the leaderboard
    fragments into dozens of low-n buckets and every verdict reads NOISE."""

    def test_slash_prefix_collapses(self):
        # Two different scraped domains pool into one collector family.
        assert _source_family("scraped/finance.yahoo.com") == "scraped"
        assert _source_family("scraped/www.cnbc.com") == "scraped"
        assert _source_family("reddit/r/wallstreetbets") == "reddit"
        assert _source_family("GoogleNews/Yahoo Finance") == "googlenews"
        assert _source_family("YahooFinance/NVDA") == "yahoofinance"

    def test_date_suffix_stripped_and_case_folded(self):
        # The schema-doc'd rolling form and the live slash form are the SAME
        # collector and must pool together.
        assert _source_family("gdelt_2025-09") == "gdelt"
        assert _source_family("GDELT/finance.yahoo.com") == "gdelt"
        assert _source_family("gdelt_2025-09") == _source_family(
            "GDELT/finance.yahoo.com")

    def test_bare_source_preserved(self):
        # An RSS feed name with no prefix stays its own family (lower-cased).
        assert _source_family("Investing.com") == "investing.com"
        assert _source_family("Seeking Alpha") == "seeking alpha"

    def test_empty_is_unknown(self):
        assert _source_family("") == "unknown"
        assert _source_family(None) == "unknown"
        assert _source_family("   ") == "unknown"


class TestEdgeFound:
    """Family `scraped`: 10 NVDA headlines, price +2/day.
    Family `reddit`: 10 AMD headlines, price flat. SPY flat ⇒ abnormal==raw."""

    def _data(self):
        arts = []
        for _ in range(10):
            arts.append({"text": "NVDA AI chip demand surges on capex",
                         "source": "scraped/finance.yahoo.com",
                         "ai_score": 7.0, "urgency": 1,
                         "published": "2026-06-01T12:00:00+00:00"})
        for _ in range(10):
            arts.append({"text": "AMD product gets a lukewarm reception",
                         "source": "reddit/r/AMD_Stock",
                         "ai_score": 5.0, "urgency": 0,
                         "published": "2026-06-01T12:00:00+00:00"})
        ph = {"NVDA": _series(100.0, 2.0), "AMD": _series(50.0, 0.0)}
        spy = _series(400.0, 0.0)
        return arts, ph, spy

    def test_per_source_forward_returns_exact(self):
        arts, ph, spy = self._data()
        r = build_source_edge(arts, ph, spy, TICKERS)
        by = {s["source"]: s for s in r["sources"]}
        sc = by["scraped"]["horizons"]
        assert sc["1"]["n"] == 10
        assert sc["1"]["mean_raw_pct"] == 2.0     # 102/100-1
        assert sc["3"]["mean_raw_pct"] == 6.0
        assert sc["5"]["mean_raw_pct"] == 10.0
        assert sc["5"]["mean_abnormal_pct"] == 10.0   # SPY flat
        assert sc["5"]["abnormal_hit_rate"] == 100.0
        rd = by["reddit"]["horizons"]
        assert rd["3"]["mean_raw_pct"] == 0.0
        assert rd["3"]["abnormal_hit_rate"] == 0.0    # 0 is not > 0

    def test_verdict_and_ranking(self):
        arts, ph, spy = self._data()
        r = build_source_edge(arts, ph, spy, TICKERS)
        assert r["verdict"] == "EDGE_FOUND"
        assert r["best_source"] == "scraped"
        assert r["worst_source"] == "reddit"
        # All horizons well-sampled (n=10 ≥ 8) ⇒ longest reference horizon.
        assert r["reference_horizon"] == 5
        assert r["n_resolved"] == 20
        assert r["spy_adjusted"] is True
        # Ranked best-first: graded scraped (+10pp) before graded reddit (0pp).
        assert [s["source"] for s in r["sources"]][:2] == ["scraped", "reddit"]
        assert by_v(r, "scraped") == "EXPLOITABLE"
        assert by_v(r, "reddit") in ("WEAK", "NEGATIVE")
        # Headline is the single source of truth for chat/UI — must name the
        # verdict and the winning family so the two surfaces can't drift.
        assert "EDGE_FOUND" in r["headline"]
        assert "scraped" in r["headline"]


def by_v(r, src):
    return next(s["verdict"] for s in r["sources"] if s["source"] == src)


class TestSpyAbnormalSubtraction:
    """SPY +1/day must be subtracted from the raw ticker return."""

    def test_abnormal_is_raw_minus_spy(self):
        arts = [{"text": "NVDA momentum", "source": "scraped/x",
                 "ai_score": 6.0, "urgency": 0,
                 "published": "2026-06-01T00:00:00+00:00"} for _ in range(9)]
        ph = {"NVDA": _series(100.0, 2.0)}     # +2/day raw
        spy = _series(400.0, 4.0)              # +1%/day (4/400)
        r = build_source_edge(arts, ph, spy, TICKERS)
        h = next(s for s in r["sources"] if s["source"] == "scraped")["horizons"]
        # 5d raw = 110/100-1 = 10.0 ; SPY 5d = 420/400-1 = 5.0 ; abn = 5.0
        assert h["5"]["mean_raw_pct"] == 10.0
        assert h["5"]["mean_abnormal_pct"] == 5.0


class TestMinScoreFilter:
    """Articles below min_score never enter any bucket (the 'worth scoring'
    gate — pooling is across bands *above* this floor only)."""

    def test_low_score_excluded(self):
        arts = [{"text": "NVDA noise", "source": "reddit/r/x",
                 "ai_score": 1.0, "urgency": 0,
                 "published": "2026-06-01T00:00:00+00:00"} for _ in range(20)]
        r = build_source_edge(arts, ph_flat(), spy_flat(), TICKERS,
                              min_score=2.0)
        assert r["n_articles"] == 20
        assert r["n_scored"] == 0
        assert r["n_resolved"] == 0
        assert r["verdict"] == "NO_DATA"


def ph_flat():
    return {"NVDA": _series(100.0, 0.0), "AMD": _series(50.0, 0.0)}


def spy_flat():
    return _series(400.0, 0.0)


class TestSampleSizeHonesty:
    """NO_DATA → INSUFFICIENT_DATA → graded, matching news_edge's idiom.
    digital-intern only retains ~days of live news, so the live early state
    is honestly INSUFFICIENT_DATA, never a fabricated edge."""

    def test_no_resolved_is_no_data(self):
        r = build_source_edge([], {}, [], TICKERS)
        assert r["verdict"] == "NO_DATA"
        assert r["n_resolved"] == 0

    def test_under_gate_is_insufficient_but_numerics_emitted(self):
        # 5 resolved (< _MIN_SOURCE_N=8): verdict withheld, numbers still there.
        arts = [{"text": "NVDA up", "source": "scraped/x", "ai_score": 6.0,
                 "urgency": 0, "published": "2026-06-01T00:00:00+00:00"}
                for _ in range(5)]
        r = build_source_edge(arts, ph_flat(), spy_flat(), TICKERS)
        assert r["n_resolved"] == 5
        assert r["verdict"] == "INSUFFICIENT_DATA"
        sc = next(s for s in r["sources"] if s["source"] == "scraped")
        assert sc["verdict"] == "INSUFFICIENT"
        assert sc["horizons"]["1"]["n"] == 5      # numerics still emitted
        assert _MIN_SOURCE_N == 8


class TestWordBoundary:
    """AMDOCS must not resolve to AMD (same regex as news_edge/position_thesis;
    a substring false-match would mis-attribute a whole source's edge)."""

    def test_amdocs_does_not_match_amd(self):
        arts = [{"text": "AMDOCS posts quarterly results", "source": "scraped/x",
                 "ai_score": 9.0, "urgency": 0,
                 "published": "2026-06-01T00:00:00+00:00"} for _ in range(10)]
        r = build_source_edge(arts, ph_flat(), spy_flat(), ["AMD"])
        assert r["n_resolved"] == 0
        assert r["verdict"] == "NO_DATA"


class TestFetchSourceArticlesBacktestIsolation:
    """The fetch helper inlines the canonical live-only clause verbatim
    (invariant #1 / the signals.py mirror). A planted backtest://, backtest_*
    or opus_annotation* row must never be graded as a real signal — and the
    row carries the `source` column this module is built around."""

    def _db(self, tmp_path):
        p = tmp_path / "articles.db"
        c = sqlite3.connect(str(p))
        c.execute(
            "CREATE TABLE articles (id TEXT PRIMARY KEY, url TEXT, title TEXT, "
            "source TEXT, ai_score REAL, urgency INTEGER, full_text BLOB, "
            "first_seen TEXT)")
        rows = [
            ("1", "https://news.example/a", "NVDA live", "scraped/x",
             6.0, 0, None, "2026-06-01T00:00:00+00:00"),
            ("2", "backtest://run_1/2026-06-01/BUY/NVDA", "NVDA bt", "rss",
             6.0, 0, None, "2026-06-01T00:00:00+00:00"),
            ("3", "https://news.example/c", "NVDA bt2", "backtest_run_7_winner",
             6.0, 0, None, "2026-06-01T00:00:00+00:00"),
            ("4", "https://news.example/d", "NVDA opus", "opus_annotation_3",
             6.0, 0, None, "2026-06-01T00:00:00+00:00"),
        ]
        c.executemany(
            "INSERT INTO articles VALUES (?,?,?,?,?,?,?,?)", rows)
        c.commit()
        c.close()
        return str(p)

    def test_only_live_row_returned(self, tmp_path):
        out = _fetch_source_articles(
            self._db(tmp_path), "2026-05-01T00:00:00+00:00", min_score=2.0)
        assert len(out) == 1
        assert out[0]["source"] == "scraped/x"
        assert out[0]["text"].startswith("NVDA live")


class TestSourceEdgeEndpoint:
    """End-to-end via the Flask test client (memory:
    project_paper_trader_analytics_verification — verify endpoints via the
    test client, not module __main__). Asserts the route composes the builder
    correctly AND that a planted backtest row never reaches the leaderboard."""

    def _seed_db(self, tmp_path):
        import sqlite3
        p = tmp_path / "articles.db"
        c = sqlite3.connect(str(p))
        c.execute(
            "CREATE TABLE articles (id TEXT PRIMARY KEY, url TEXT, title TEXT, "
            "source TEXT, ai_score REAL, urgency INTEGER, full_text BLOB, "
            "first_seen TEXT)")
        live = [(str(i), f"https://news/{i}", "NVDA chip demand surges",
                 "scraped/finance.yahoo.com", 7.0, 0, None,
                 "2026-06-01T12:00:00+00:00") for i in range(10)]
        # A synthetic row that MUST NOT be graded as a real collector.
        live.append(("bt", "backtest://run_1/2026-06-01/BUY/NVDA",
                     "NVDA backtest", "backtest_run_1_winner", 9.0, 0, None,
                     "2026-06-01T12:00:00+00:00"))
        c.executemany("INSERT INTO articles VALUES (?,?,?,?,?,?,?,?)", live)
        c.commit()
        c.close()
        return p

    def test_route_grades_only_live_sources(self, tmp_path, monkeypatch):
        from pathlib import Path

        from paper_trader import dashboard

        db = self._seed_db(tmp_path)
        monkeypatch.setattr(dashboard, "_articles_db_path",
                            lambda: Path(db))

        def _fake_hist(ticker, period="3mo"):
            if ticker == "SPY":
                return _series(400.0, 0.0)
            if ticker == "NVDA":
                return _series(100.0, 2.0)   # +2/day
            return []

        monkeypatch.setattr(dashboard, "_daily_history_cached", _fake_hist)
        dashboard.app.config["TESTING"] = True
        with dashboard.app.test_client() as client:
            r = client.get("/api/source-edge?days=60&min_score=2.0")
        assert r.status_code == 200
        data = r.get_json()
        fams = {s["source"] for s in data["sources"]}
        assert "scraped" in fams
        # The backtest row's source ("backtest_run_1_winner") is filtered by
        # the live-only SQL before normalisation — it can never appear.
        assert not any("backtest" in f for f in fams)
        assert data["n_resolved"] == 10
        assert data["best_source"] == "scraped"
        assert data["verdict"] == "EDGE_FOUND"
