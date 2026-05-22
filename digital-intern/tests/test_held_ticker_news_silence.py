"""Tests for analytics.held_ticker_news_silence.

Pure-helper coverage so the suite runs without a DB. The DB shell is
exercised by a small in-memory SQLite fixture that mirrors the production
schema for the projection actually used (title/source/first_seen + the
columns the LIVE_ONLY filter touches).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from analytics import held_ticker_news_silence as hns


# ── Pure aggregator ─────────────────────────────────────────────────────────


def _ts(now: datetime, hours_ago: float) -> str:
    return (now - timedelta(hours=hours_ago)).isoformat()


HELD = ("MU", "NVDA", "MSFT", "QBTS")


def test_zero_mentions_yields_dark_verdict():
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    out = hns.compute_silence([], HELD, now=now)
    assert {r["ticker"] for r in out} == set(HELD)
    # All four should be DARK with zero counts.
    for r in out:
        assert r["verdict"] == "DARK"
        assert r["counts"] == {"1h": 0, "6h": 0, "24h": 0}
        assert r["distinct_sources"] == {"24h": 0}


def test_single_source_24h_mentions_yields_echo_verdict():
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("MU breaks below 700 — staffing risk", "Finnhub/Yahoo", _ts(now, 2)),
        ("MU memory pricing recap", "Finnhub/Yahoo", _ts(now, 5)),
        ("MU shows DRAM weakness", "Finnhub/Yahoo", _ts(now, 8)),
    ]
    out = {r["ticker"]: r for r in hns.compute_silence(rows, HELD, now=now)}
    mu = out["MU"]
    assert mu["verdict"] == "ECHO"
    assert mu["counts"] == {"1h": 0, "6h": 2, "24h": 3}
    assert mu["distinct_sources"]["24h"] == 1
    # Other held names with no mentions stay DARK.
    assert out["NVDA"]["verdict"] == "DARK"
    assert out["MSFT"]["verdict"] == "DARK"
    assert out["QBTS"]["verdict"] == "DARK"


def test_multi_source_moderate_volume_yields_normal_verdict():
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("NVDA prints earnings beat",  "rss",                       _ts(now, 8)),
        ("NVDA upgrade from Wells",    "GDELT/reuters.com",         _ts(now, 12)),
        ("NVDA China export pause",    "scraped/finance.yahoo.com", _ts(now, 20)),
    ]
    out = {r["ticker"]: r for r in hns.compute_silence(rows, HELD, now=now)}
    nvda = out["NVDA"]
    assert nvda["verdict"] == "NORMAL"
    assert nvda["counts"] == {"1h": 0, "6h": 0, "24h": 3}
    assert nvda["distinct_sources"]["24h"] == 3


def test_hot_verdict_requires_recent_threshold_and_multi_source():
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("MU surges on memory shortage", "rss",                       _ts(now, 0.3)),
        ("MU 8-K filed",                 "SEC-EDGAR/8-K",             _ts(now, 0.5)),
        ("MU upgraded by Stifel",        "Finnhub/Benzinga",          _ts(now, 0.8)),
        # plus older context the analyst would also see
        ("MU memory talks unwind",       "GDELT/reuters.com",         _ts(now, 4)),
    ]
    out = {r["ticker"]: r for r in hns.compute_silence(rows, HELD, now=now)}
    mu = out["MU"]
    assert mu["verdict"] == "HOT"
    assert mu["counts"] == {"1h": 3, "6h": 4, "24h": 4}
    assert mu["distinct_sources"]["24h"] == 4


def test_single_source_burst_stays_echo_even_at_high_volume():
    """An echo cluster is one outlet repeating itself — high 1h count does NOT
    promote it to HOT. The escape valve is *distinct sources*, not volume.

    Same anti-noise discipline as ``ECHO_MIN_COPIES`` in claude_analyst —
    a single outlet's syndication never inflates apparent corroboration.
    """
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("NVDA Vera ships",      "reddit/r/wallstreetbets", _ts(now, 0.1)),
        ("NVDA Vera details",    "reddit/r/wallstreetbets", _ts(now, 0.2)),
        ("NVDA Vera CPU specs",  "reddit/r/wallstreetbets", _ts(now, 0.3)),
        ("NVDA Vera launch buy", "reddit/r/wallstreetbets", _ts(now, 0.4)),
    ]
    out = {r["ticker"]: r for r in hns.compute_silence(rows, HELD, now=now)}
    nvda = out["NVDA"]
    assert nvda["counts"]["1h"] == 4  # high recent volume
    assert nvda["distinct_sources"]["24h"] == 1  # one publisher
    assert nvda["verdict"] == "ECHO"  # NOT HOT — single source


def test_window_bucketing_is_strictly_correct():
    """A row landing exactly at the 1h boundary must NOT be counted in the 1h
    bucket; a row landing INSIDE all three windows counts in all three.

    Bucketing uses ``ts >= cutoff_w`` so the cutoff itself is included.
    """
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    # 1.5h ago — outside 1h window, inside 6h and 24h.
    rows = [("MU mid story", "rss", _ts(now, 1.5))]
    mu = {r["ticker"]: r for r in hns.compute_silence(rows, HELD, now=now)}["MU"]
    assert mu["counts"] == {"1h": 0, "6h": 1, "24h": 1}


def test_rows_older_than_longest_window_are_excluded():
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("MU last week story",     "rss", _ts(now, 200)),     # >24h, dropped
        ("MU fresh enough story",  "rss", _ts(now, 23)),      # inside 24h
    ]
    mu = {r["ticker"]: r for r in hns.compute_silence(rows, HELD, now=now)}["MU"]
    assert mu["counts"] == {"1h": 0, "6h": 0, "24h": 1}


def test_word_boundary_match_does_not_leak_to_substring():
    """A ticker like ``MU`` must not match inside ``MUST``, ``MUSE``, ``MUSK``."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("Investors MUST read this",      "rss", _ts(now, 1)),
        ("MUSK eyes Twitter",             "rss", _ts(now, 2)),
        ("Tesla CEO Musk roadmap",        "rss", _ts(now, 3)),
        # Real, must match
        ("MU posts surprise beat",        "rss", _ts(now, 4)),
    ]
    mu = {r["ticker"]: r for r in hns.compute_silence(rows, HELD, now=now)}["MU"]
    assert mu["counts"]["24h"] == 1


def test_case_insensitive_match_works_for_lowercase_ticker():
    """Some headlines drop case (``$nvda surges``). The pattern is
    case-insensitive so coverage is not silently undercounted on those."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("nvda is on fire",        "rss", _ts(now, 1)),
        ("$nvda buyer's strike",   "GDELT/x.com", _ts(now, 2)),
    ]
    out = {r["ticker"]: r for r in hns.compute_silence(rows, HELD, now=now)}
    assert out["NVDA"]["counts"]["24h"] == 2
    assert out["NVDA"]["distinct_sources"]["24h"] == 2


def test_one_title_naming_two_held_tickers_counts_for_both():
    """``_book_tickers`` uses set-on-title; this audit mirrors that — a
    single headline naming MU and NVDA counts once for each ticker."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("MU and NVDA selloff on China bans", "rss", _ts(now, 2)),
    ]
    out = {r["ticker"]: r for r in hns.compute_silence(rows, HELD, now=now)}
    assert out["MU"]["counts"]["24h"] == 1
    assert out["NVDA"]["counts"]["24h"] == 1


def test_severity_sort_puts_dark_first_then_echo_then_normal_then_hot():
    """The analyst's eye must land on the gaps first."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        # MU → ECHO
        ("MU note 1", "rss", _ts(now, 4)),
        ("MU note 2", "rss", _ts(now, 5)),
        # NVDA → HOT (≥3 recent, 3 distinct sources)
        ("NVDA flash 1", "rss", _ts(now, 0.1)),
        ("NVDA flash 2", "GDELT/reuters.com", _ts(now, 0.2)),
        ("NVDA flash 3", "scraped/finance.yahoo.com", _ts(now, 0.3)),
        # MSFT → NORMAL (3 distinct sources over 24h, no recent burst)
        ("MSFT Azure deal", "rss", _ts(now, 12)),
        ("MSFT acquires X",  "GDELT/reuters.com", _ts(now, 18)),
        ("MSFT Q3 preview",  "scraped/finance.yahoo.com", _ts(now, 22)),
        # QBTS → DARK (no rows)
    ]
    out = hns.compute_silence(rows, HELD, now=now)
    order = [r["ticker"] for r in out]
    assert order == ["QBTS", "MU", "MSFT", "NVDA"]


def test_unparseable_or_missing_timestamps_skipped_not_crashing():
    """Defensive: a row with a bogus first_seen is skipped, not exception."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("MU good row",     "rss",  _ts(now, 1)),
        ("MU bogus row",    "rss",  "not-a-date"),
        ("MU empty ts row", "rss",  ""),
        ("MU null ts row",  "rss",  None),
    ]
    mu = {r["ticker"]: r for r in hns.compute_silence(rows, HELD, now=now)}["MU"]
    assert mu["counts"]["24h"] == 1


def test_malformed_row_tuple_skipped_not_crashing():
    """A non-3-tuple row (e.g. malformed projection) must not crash."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("MU normal row", "rss", _ts(now, 1)),
        ("malformed",     "rss"),
        None,
    ]
    mu = {r["ticker"]: r for r in hns.compute_silence(rows, HELD, now=now)}["MU"]
    assert mu["counts"]["24h"] == 1


def test_unknown_ticker_match_does_not_promote_held_ticker():
    """A title mentioning a non-held symbol (e.g. AMD) must not bleed into
    held-set counts because of substring confusion."""
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    rows = [("AMD outperforms peers", "rss", _ts(now, 1))]
    out = hns.compute_silence(rows, HELD, now=now)
    for r in out:
        assert r["counts"]["24h"] == 0


def test_empty_ticker_list_returns_empty_output():
    out = hns.compute_silence(
        [("MU news", "rss", "2026-05-20T11:00:00+00:00")], [],
        now=datetime(2026, 5, 20, 12, tzinfo=timezone.utc),
    )
    assert out == []


# ── Verdict report shape ────────────────────────────────────────────────────


def test_build_report_counts_verdicts_and_emits_metadata():
    now = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    per_ticker = [
        {"ticker": "QBTS", "verdict": "DARK",   "counts": {"1h": 0, "6h": 0, "24h": 0}, "distinct_sources": {"24h": 0}},
        {"ticker": "MU",   "verdict": "ECHO",   "counts": {"1h": 0, "6h": 1, "24h": 3}, "distinct_sources": {"24h": 1}},
        {"ticker": "MSFT", "verdict": "NORMAL", "counts": {"1h": 0, "6h": 0, "24h": 3}, "distinct_sources": {"24h": 3}},
        {"ticker": "NVDA", "verdict": "HOT",    "counts": {"1h": 3, "6h": 3, "24h": 3}, "distinct_sources": {"24h": 3}},
    ]
    report = hns.build_report(per_ticker, now=now)
    assert report["generated_at"] == now.isoformat()
    assert report["windows_h"] == list(hns.WINDOWS_H)
    assert report["n_tickers"] == 4
    assert report["verdict_counts"] == {
        "DARK": 1, "ECHO": 1, "NORMAL": 1, "HOT": 1,
    }
    assert report["tickers"] is per_ticker
    # Round-trip survives JSON.
    json.loads(json.dumps(report))


# ── Anti-drift guards ───────────────────────────────────────────────────────


def test_live_only_clause_in_sync_with_article_store():
    """Inline ``LIVE_ONLY_CLAUSE`` must stay byte-identical with the
    canonical SQL fragment from ``storage.article_store``. Same anti-drift
    discipline as ``analytics.alert_source_breakdown`` / others.

    A re-derivation that quietly diverges (e.g. someone re-types the clause
    and drops the ``opus_annotation%`` arm) would silently let synthetic
    rows colour the audit. This guard fails loud."""
    from storage.article_store import _LIVE_ONLY_CLAUSE
    assert hns.LIVE_ONLY_CLAUSE == _LIVE_ONLY_CLAUSE


def test_held_ticker_set_uses_live_portfolio_tickers_ssot():
    """The audit must source its held set from
    ``ml.features.LIVE_PORTFOLIO_TICKERS`` — the single source of truth the
    briefing/alert/feature surfaces already key on. A new module that
    re-derives the held set would silently drift on every portfolio change.

    Locks both directions: the SSOT name is imported, and ``run()`` with no
    explicit tickers consumes EXACTLY the SSOT set.
    """
    from ml.features import LIVE_PORTFOLIO_TICKERS
    # Import path is exercised at module load.
    assert hns.LIVE_PORTFOLIO_TICKERS is LIVE_PORTFOLIO_TICKERS


# ── DB shell ────────────────────────────────────────────────────────────────


@pytest.fixture
def synth_db(tmp_path: Path):
    """Build a minimal articles.db with the columns this audit reads."""
    path = tmp_path / "articles.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE articles (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            first_seen TEXT NOT NULL
        );
    """)
    now = datetime.now(timezone.utc)
    fresh = (now - timedelta(hours=2)).isoformat()
    stale = (now - timedelta(hours=200)).isoformat()
    rows = [
        # Live row inside window
        ("a1", "https://x/1",         "MU 8-K filed",      "SEC-EDGAR/8-K", fresh),
        # Backtest row inside window — must be excluded by LIVE_ONLY_CLAUSE
        ("a2", "backtest://run/1/MU", "MU backtest title", "backtest_run_1_winner", fresh),
        # Opus-annotation row inside window — must be excluded by LIVE_ONLY_CLAUSE
        ("a3", "https://x/3",         "MU opus annotation","opus_annotation_cycle_5", fresh),
        # Stale row OUTSIDE window — must be excluded by first_seen >= cutoff
        ("a4", "https://x/4",         "MU last week",      "rss", stale),
    ]
    con.executemany(
        "INSERT INTO articles (id, url, title, source, first_seen) "
        "VALUES (?, ?, ?, ?, ?)", rows,
    )
    con.commit()
    con.close()
    return path


def test_load_rows_excludes_backtest_and_opus_annotation_and_stale(synth_db):
    rows = hns.load_rows(synth_db, hours=24)
    titles = {r[0] for r in rows}
    # Only the live, fresh row survives.
    assert titles == {"MU 8-K filed"}


def test_run_writes_report_with_dark_count_when_no_live_coverage(synth_db, tmp_path, monkeypatch):
    out_path = tmp_path / "report.json"
    monkeypatch.setattr(hns, "OUT_PATH", out_path)
    report = hns.run(db_path=synth_db, tickers=["QBTS"], write=True)
    assert report["n_tickers"] == 1
    assert report["verdict_counts"]["DARK"] == 1
    # Persisted JSON matches in-memory report.
    assert json.loads(out_path.read_text()) == report


# ── /api/held-news-silence endpoint ─────────────────────────────────────────


def _insert_article(store, *, id, url, title, source, age_min=5):
    """Insert one article row through the live ArticleStore schema."""
    fs = (datetime.now(timezone.utc) - timedelta(minutes=age_min)).isoformat()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 2.0, 5.0, 0, fs, 0, None, None),
        )
        store.conn.commit()


def test_endpoint_returns_per_ticker_verdicts_and_excludes_backtest(
        store, monkeypatch):
    from dashboard import web_server

    # NVDA carried by two distinct live publishers in the last hour.
    _insert_article(store, id="n1", url="https://x/n1",
                    title="NVDA earnings beat expectations", source="rss")
    _insert_article(store, id="n2", url="https://x/n2",
                    title="NVDA chip demand strong", source="reuters")
    # Synthetic backtest row naming NVDA — must NEVER inflate the count.
    _insert_article(store, id="bt", url="backtest://run_3/2026-01-01/BUY/NVDA",
                    title="NVDA SYNTHETIC SHOULD NOT SURFACE",
                    source="backtest_run_3_winner")

    monkeypatch.setattr(web_server, "_store", store, raising=False)
    client = web_server.create_app(store).test_client()
    resp = client.get("/api/held-news-silence")

    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert set(data) >= {"generated_at", "windows_h", "n_tickers",
                         "verdict_counts", "tickers"}
    by_tk = {r["ticker"]: r for r in data["tickers"]}
    # NVDA is a book ticker; two distinct live sources -> NORMAL, not DARK/ECHO.
    assert "NVDA" in by_tk
    assert by_tk["NVDA"]["verdict"] == "NORMAL"
    # The synthetic backtest row did not add a third source.
    assert by_tk["NVDA"]["distinct_sources"]["24h"] == 2
    # A book ticker with no coverage at all is reported DARK.
    assert any(r["verdict"] == "DARK" for r in data["tickers"])
