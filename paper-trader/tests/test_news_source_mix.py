"""Tests for analytics/news_source_mix.py — per-held-ticker source-diversity.

Discriminating locks:

* **ECHO** = ≥3 articles AND single source ≥70% share — captures the
  syndication-artifact false-positive that ``news_velocity`` cannot see.
* **STRONG** = ≥3 articles AND ≥4 distinct sources AND no source dominates.
* **QUIET** = <2 articles (sample-size honesty — one mention is not a
  breadth signal).
* **MODERATE** = everything in between.
* **Window cutoff** is strict-inclusive on the lower bound (the standard
  ``signals.py`` precedent — ``first_seen >= now - window_hours``).
* **Ticker regex** uses the same word-boundary SSOT as
  ``news_velocity`` / ``trade_attribution``: ``MU`` must NOT alias
  ``MUSE`` / ``MUTUAL``; ``$MU`` cashtag still hits; ``AMDOCS`` ≠ ``AMD``.
* **``_safe`` contract** — garbage rows (None, non-dict, missing /
  unparseable ``first_seen``, missing source) must NOT raise; they
  degrade to a skip for that row only.
* **Headline priority** — ECHO warning beats STRONG confirmation beats
  QUIET vacuum so the most-actionable false-signal surfaces first.
* **Sort priority** — per_ticker is ECHO-first then STRONG/MODERATE/
  QUIET, with n_articles DESC tie-break inside each band.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.news_source_mix import (  # noqa: E402
    ECHO_MIN_ARTICLES,
    ECHO_THRESHOLD_PCT,
    STRONG_MIN_ARTICLES,
    STRONG_MIN_SOURCES,
    build_news_source_mix,
)

_NOW = datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)


def _art(ticker_in_title: str, source: str, hours_ago: float = 1.0,
         title_override: str | None = None, body: str = "") -> dict:
    ts = _NOW - timedelta(hours=hours_ago)
    title = title_override if title_override is not None else (
        f"{ticker_in_title} update from the wire")
    return {
        "title": title,
        "body": body,
        "first_seen": ts.isoformat(),
        "source": source,
    }


def _build(articles, held, **kw):
    kw.setdefault("now", _NOW)
    return build_news_source_mix(articles, held, **kw)


# ───────────────────────── State ladder ─────────────────────────


class TestStateLadder:

    def test_no_held_is_no_data(self):
        r = _build([_art("NVDA", "rss")], held=[])
        assert r["state"] == "NO_DATA"
        assert r["per_ticker"] == []
        assert r["n_held"] == 0
        assert "no held" in r["headline"].lower()
        assert r["any_echo"] is False

    def test_zero_window_is_no_data(self):
        r = _build([_art("NVDA", "rss")], held=["NVDA"], window_hours=0.0)
        assert r["state"] == "NO_DATA"
        assert r["per_ticker"] == []
        assert r["any_echo"] is False

    def test_held_but_no_articles_state_no_data_per_row_quiet(self):
        r = _build([], held=["NVDA"])
        assert r["state"] == "NO_DATA"
        assert len(r["per_ticker"]) == 1
        # Even with zero matches we still surface the held name; its state is
        # QUIET (the explicit "no articles" verdict, not a missing row).
        assert r["per_ticker"][0]["ticker"] == "NVDA"
        assert r["per_ticker"][0]["state"] == "QUIET"
        assert r["per_ticker"][0]["n_articles"] == 0
        assert r["per_ticker"][0]["n_unique_sources"] == 0
        assert r["per_ticker"][0]["top_source"] is None
        assert r["any_echo"] is False

    def test_quiet_is_below_min_for_verdict(self):
        # 1 article = QUIET (sample-size honesty)
        r = _build([_art("NVDA", "rss")], held=["NVDA"])
        row = r["per_ticker"][0]
        assert row["state"] == "QUIET"
        assert row["n_articles"] == 1

    def test_echo_single_source_dominates(self):
        # 5 articles, all yahoo → ECHO (top_share=100%)
        arts = [_art("NVDA", "yahoo", h) for h in [1, 2, 3, 4, 5]]
        r = _build(arts, held=["NVDA"])
        row = r["per_ticker"][0]
        assert row["state"] == "ECHO"
        assert row["n_articles"] == 5
        assert row["n_unique_sources"] == 1
        assert row["top_source"] == "yahoo"
        assert row["top_source_share_pct"] == 100.0
        assert r["any_echo"] is True
        assert "ECHO" in r["headline"]
        assert "yahoo" in r["headline"]

    def test_echo_threshold_exact_70pct(self):
        # 7 yahoo + 3 google_news = 70% yahoo share → ECHO
        arts = [_art("NVDA", "yahoo", h) for h in range(1, 8)] + \
               [_art("NVDA", "google_news", h) for h in [1, 2, 3]]
        r = _build(arts, held=["NVDA"])
        row = r["per_ticker"][0]
        assert row["n_articles"] == 10
        assert row["top_source_share_pct"] == 70.0
        assert row["state"] == "ECHO"

    def test_echo_below_threshold_is_moderate(self):
        # 6 yahoo + 4 google_news + 1 finnhub = 54.5% top share, 3 sources
        # → STRONG fails (only 3 sources, need 4), ECHO fails (<70%) → MODERATE
        arts = [_art("NVDA", "yahoo", h) for h in range(1, 7)] + \
               [_art("NVDA", "google_news", h) for h in range(1, 5)] + \
               [_art("NVDA", "finnhub", 1.0)]
        r = _build(arts, held=["NVDA"])
        row = r["per_ticker"][0]
        assert row["state"] == "MODERATE"
        assert row["n_unique_sources"] == 3

    def test_strong_requires_four_sources_and_no_dominance(self):
        # 1 per source × 5 sources = 5 articles, 5 sources, top share 20%
        # → STRONG (≥3 articles, ≥4 sources, top<70%)
        arts = [
            _art("NVDA", "yahoo", 1.0),
            _art("NVDA", "google_news", 2.0),
            _art("NVDA", "finnhub", 3.0),
            _art("NVDA", "polygon", 4.0),
            _art("NVDA", "rss", 5.0),
        ]
        r = _build(arts, held=["NVDA"])
        row = r["per_ticker"][0]
        assert row["state"] == "STRONG"
        assert row["n_unique_sources"] == 5
        assert row["top_source_share_pct"] == 20.0
        assert r["any_echo"] is False

    def test_strong_threshold_exactly_four_sources(self):
        # Exactly STRONG_MIN_SOURCES: 4 distinct sources, 4 articles → STRONG
        arts = [
            _art("NVDA", f"src{i}", float(i)) for i in range(1, 5)
        ]
        r = _build(arts, held=["NVDA"])
        row = r["per_ticker"][0]
        assert row["n_unique_sources"] == STRONG_MIN_SOURCES == 4
        assert row["state"] == "STRONG"

    def test_three_sources_three_articles_is_moderate_not_strong(self):
        # Just below STRONG_MIN_SOURCES: 3 sources × 1 each = MODERATE
        arts = [_art("NVDA", f"src{i}", float(i)) for i in range(1, 4)]
        r = _build(arts, held=["NVDA"])
        row = r["per_ticker"][0]
        assert row["state"] == "MODERATE"

    def test_two_articles_is_moderate_not_quiet(self):
        # MIN_FOR_VERDICT == 2; exactly 2 articles → MODERATE, not QUIET.
        arts = [_art("NVDA", "yahoo", 1.0), _art("NVDA", "google_news", 2.0)]
        r = _build(arts, held=["NVDA"])
        assert r["per_ticker"][0]["state"] == "MODERATE"


# ───────────────────────── Window boundary ─────────────────────────


class TestWindowBoundary:

    def test_window_cutoff_strict_inclusive(self):
        # An article at exactly the cutoff timestamp IS in the window.
        arts = [_art("NVDA", "yahoo", hours_ago=24.0)]
        r = _build(arts, held=["NVDA"], window_hours=24.0)
        row = r["per_ticker"][0]
        # That single article puts us in QUIET (need ≥2 for verdict).
        assert row["n_articles"] == 1
        assert row["state"] == "QUIET"

    def test_article_just_outside_window_is_excluded(self):
        # An article past the cutoff is dropped.
        arts = [_art("NVDA", "yahoo", hours_ago=24.01)]
        r = _build(arts, held=["NVDA"], window_hours=24.0)
        row = r["per_ticker"][0]
        assert row["n_articles"] == 0
        assert row["state"] == "QUIET"


# ───────────────────────── Ticker regex ─────────────────────────


class TestTickerRegex:

    def test_amdocs_does_not_alias_amd(self):
        # The substring-leak regression the news_velocity locks too.
        arts = [
            _art("X", "yahoo", h, title_override="AMDOCS earnings beat")
            for h in [1, 2, 3, 4, 5]
        ]
        r = _build(arts, held=["AMD"])
        assert r["per_ticker"][0]["n_articles"] == 0
        assert r["per_ticker"][0]["state"] == "QUIET"

    def test_mu_does_not_alias_mutual(self):
        arts = [
            _art("X", "rss", h, title_override="MUTUAL fund flows update")
            for h in [1, 2, 3]
        ]
        r = _build(arts, held=["MU"])
        assert r["per_ticker"][0]["n_articles"] == 0

    def test_cashtag_dollar_prefix_hits(self):
        arts = [
            _art("X", "rss", h, title_override=f"Article {i}: $NVDA up")
            for i, h in enumerate([1, 2, 3, 4])
        ]
        r = _build(arts, held=["NVDA"])
        assert r["per_ticker"][0]["n_articles"] == 4

    def test_body_field_scanned_too(self):
        # Title doesn't mention NVDA but body does.
        arts = [
            _art("X", "rss", h,
                 title_override="Tech sector update",
                 body=f"NVDA quote {h}")
            for h in [1, 2, 3, 4]
        ]
        r = _build(arts, held=["NVDA"])
        assert r["per_ticker"][0]["n_articles"] == 4

    def test_held_ticker_case_insensitive_dedup(self):
        # NVDA passed twice in mixed case → deduped to one ticker.
        arts = [_art("NVDA", "yahoo", 1.0)]
        r = _build(arts, held=["nvda", "NVDA", "Nvda"])
        assert r["n_held"] == 1
        assert len(r["per_ticker"]) == 1
        assert r["per_ticker"][0]["ticker"] == "NVDA"


# ───────────────────────── Sort + headline priority ─────────────────────────


class TestSortAndHeadline:

    def test_echo_first_then_strong_then_quiet(self):
        # NVDA: 4 yahoo → ECHO (100% share, 4 articles)
        # MSFT: 4 sources × 1 = 4 articles → STRONG
        # AAPL: 0 articles → QUIET
        arts = [
            _art("NVDA", "yahoo", h) for h in [1, 2, 3, 4]
        ] + [
            _art("MSFT", "yahoo", 1.0),
            _art("MSFT", "google_news", 2.0),
            _art("MSFT", "finnhub", 3.0),
            _art("MSFT", "polygon", 4.0),
        ]
        r = _build(arts, held=["AAPL", "MSFT", "NVDA"])
        # Sort: ECHO (NVDA) → STRONG (MSFT) → QUIET (AAPL)
        order = [row["ticker"] for row in r["per_ticker"]]
        assert order == ["NVDA", "MSFT", "AAPL"]
        # Headline calls out the ECHO warning specifically.
        assert "ECHO" in r["headline"]
        assert "NVDA" in r["headline"]
        assert r["any_echo"] is True

    def test_headline_strong_when_no_echo(self):
        arts = [_art("NVDA", f"s{i}", float(i)) for i in range(1, 5)]
        r = _build(arts, held=["NVDA"])
        assert "STRONG" in r["headline"]
        assert "NVDA" in r["headline"]

    def test_headline_quiet_when_all_zero(self):
        # No matching articles at all; held names present.
        arts = [_art("XYZ", "rss", 1.0) for _ in range(5)]
        r = _build(arts, held=["NVDA", "MSFT"])
        # n_with_data == 0 → NO_DATA top-level
        assert r["state"] == "NO_DATA"
        assert "0 articles" in r["headline"]
        # Per-ticker rows still QUIET, n=0.
        for row in r["per_ticker"]:
            assert row["state"] == "QUIET"
            assert row["n_articles"] == 0

    def test_sort_within_band_by_n_articles_desc(self):
        # Two ECHO tickers: NVDA 5 articles, AAPL 3 articles (both yahoo only)
        arts = [
            _art("NVDA", "yahoo", h) for h in [1, 2, 3, 4, 5]
        ] + [
            _art("AAPL", "yahoo", h) for h in [1, 2, 3]
        ]
        r = _build(arts, held=["AAPL", "NVDA"])
        order = [row["ticker"] for row in r["per_ticker"]]
        # Both ECHO; larger n_articles wins.
        assert order == ["NVDA", "AAPL"]


# ───────────────────────── `_safe` discipline ─────────────────────────


class TestSafeDegrade:

    def test_none_articles_does_not_raise(self):
        r = _build(None, held=["NVDA"])
        assert r["state"] == "NO_DATA"
        assert r["per_ticker"][0]["n_articles"] == 0

    def test_non_dict_rows_skipped(self):
        arts = [_art("NVDA", "yahoo", 1.0), "garbage", 42, None,
                _art("NVDA", "google_news", 2.0)]
        r = _build(arts, held=["NVDA"])
        # The two valid articles survive.
        assert r["per_ticker"][0]["n_articles"] == 2

    def test_missing_first_seen_skipped(self):
        arts = [
            {"title": "NVDA news", "source": "yahoo"},  # no first_seen
            _art("NVDA", "google_news", 1.0),
        ]
        r = _build(arts, held=["NVDA"])
        assert r["per_ticker"][0]["n_articles"] == 1

    def test_unparseable_first_seen_skipped(self):
        arts = [
            {"title": "NVDA news", "source": "yahoo", "first_seen": "not-a-date"},
            _art("NVDA", "yahoo", 1.0),
        ]
        r = _build(arts, held=["NVDA"])
        assert r["per_ticker"][0]["n_articles"] == 1

    def test_missing_source_falls_to_unknown(self):
        arts = [
            {"title": "NVDA news", "first_seen": _NOW.isoformat()},
            {"title": "NVDA news2", "first_seen": (_NOW - timedelta(hours=1)).isoformat()},
            {"title": "NVDA news3", "first_seen": (_NOW - timedelta(hours=2)).isoformat()},
        ]
        r = _build(arts, held=["NVDA"])
        row = r["per_ticker"][0]
        # All three rolled into the "(unknown)" bucket → ECHO.
        assert row["n_articles"] == 3
        assert row["top_source"] == "(unknown)"
        assert row["state"] == "ECHO"

    def test_blank_and_none_held_tickers_dropped(self):
        arts = [_art("NVDA", "yahoo", 1.0)]
        r = _build(arts, held=[None, "", "  ", "NVDA"])
        assert r["n_held"] == 1
        assert r["per_ticker"][0]["ticker"] == "NVDA"


# ───────────────────────── Sources breakdown ─────────────────────────


class TestSourcesBreakdown:

    def test_breakdown_capped_to_top_n_and_sorted(self):
        # 6 distinct sources, varying counts. Top 5 retained, sorted count DESC.
        arts = [
            _art("NVDA", "yahoo", 1.0),
            _art("NVDA", "yahoo", 2.0),
            _art("NVDA", "yahoo", 3.0),
            _art("NVDA", "google_news", 4.0),
            _art("NVDA", "google_news", 5.0),
            _art("NVDA", "finnhub", 6.0),
            _art("NVDA", "polygon", 7.0),
            _art("NVDA", "rss", 8.0),
            _art("NVDA", "reddit", 9.0),
        ]
        r = _build(arts, held=["NVDA"])
        row = r["per_ticker"][0]
        bd = row["sources_breakdown"]
        assert len(bd) == 5
        assert bd[0] == {"source": "yahoo", "n": 3}
        assert bd[1] == {"source": "google_news", "n": 2}
        # All others tied at 1 — alphabetical tie-break.
        names = [e["source"] for e in bd[2:]]
        assert names == sorted(names)

    def test_constants_pinned(self):
        # Live behaviour depends on these — any change without test update
        # implies an unannounced verdict shift.
        assert STRONG_MIN_SOURCES == 4
        assert STRONG_MIN_ARTICLES == 3
        assert ECHO_MIN_ARTICLES == 3
        assert ECHO_THRESHOLD_PCT == 70.0


# ───────────────────────── Endpoint integration ─────────────────────────


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """Fresh paper-trader Store under tmp_path."""
    from paper_trader import store as store_mod
    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "paper_trader.db")
    if hasattr(store_mod, "_STORE"):
        monkeypatch.setattr(store_mod, "_STORE", None, raising=False)
    s = store_mod.Store()
    return s


class TestNewsSourceMixEndpoint:
    """Drive the real Flask view on a fresh Store with no articles.db so the
    documented no-DB degrade branch (NO_DATA, never 500) is exercised."""

    def test_no_db_no_held_returns_no_data_not_500(self, fresh_store, monkeypatch):
        from paper_trader import dashboard
        monkeypatch.setattr(dashboard, "get_store", lambda: fresh_store)
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        client = dashboard.app.test_client()
        r = client.get("/api/news-source-mix")
        assert r.status_code == 200
        body = r.get_json()
        assert body["state"] == "NO_DATA"
        assert body["per_ticker"] == []
        assert body["any_echo"] is False

    def test_ticker_override_param_resolves(self, fresh_store, monkeypatch):
        from paper_trader import dashboard
        monkeypatch.setattr(dashboard, "get_store", lambda: fresh_store)
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        client = dashboard.app.test_client()
        r = client.get("/api/news-source-mix?tickers=NVDA,MU")
        assert r.status_code == 200
        body = r.get_json()
        # No DB path → degrade body still records the held count from override.
        assert body["n_held"] == 2

    def test_param_clamping_no_500_on_garbage(self, fresh_store, monkeypatch):
        from paper_trader import dashboard
        monkeypatch.setattr(dashboard, "get_store", lambda: fresh_store)
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        client = dashboard.app.test_client()
        r = client.get(
            "/api/news-source-mix?window_hours=notanumber&tickers=NVDA")
        assert r.status_code == 200
        body = r.get_json()
        assert body["state"] in {"NO_DATA", "OK"}
        # Default 24h after garbage-clamp.
        assert body["window_hours"] == 24.0

    def test_endpoint_with_real_articles_db_echo(self, fresh_store, tmp_path,
                                                  monkeypatch):
        """Seed a tiny articles.db; assert the SQL-side live-only clause
        filters synthetic rows and the ECHO state surfaces from a real
        all-yahoo bucket."""
        import sqlite3
        from paper_trader import dashboard

        db_path = tmp_path / "articles.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE articles ("
            "id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT, "
            "first_seen TEXT NOT NULL, ai_score REAL, urgency INTEGER, "
            "full_text BLOB)"
        )
        ts = datetime.now(timezone.utc)
        for i in range(5):
            conn.execute(
                "INSERT INTO articles (id, url, title, source, first_seen, "
                "ai_score, urgency) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"a{i}", f"https://example.com/{i}",
                 f"NVDA up on chip demand {i}", "yahoo",
                 (ts - timedelta(hours=i)).isoformat(), 5.0, 0),
            )
        # Backtest-poison row that MUST be filtered by the SQL live-only clause.
        conn.execute(
            "INSERT INTO articles (id, url, title, source, first_seen, "
            "ai_score, urgency) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("synthetic", "backtest://run/1/nvda", "NVDA backtest winner",
             "backtest_run_1_winner", ts.isoformat(), 5.0, 0),
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(dashboard, "get_store", lambda: fresh_store)
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: db_path)
        client = dashboard.app.test_client()
        r = client.get("/api/news-source-mix?tickers=NVDA&window_hours=24")
        assert r.status_code == 200
        body = r.get_json()
        assert body["state"] == "OK"
        assert body["any_echo"] is True
        row = body["per_ticker"][0]
        assert row["ticker"] == "NVDA"
        assert row["state"] == "ECHO"
        # 5 live rows (the backtest row filtered by SQL); single source.
        assert row["n_articles"] == 5
        assert row["n_unique_sources"] == 1
        assert row["top_source"] == "yahoo"
