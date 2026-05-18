"""Cross-source corroboration detector — clustering correctness + the
backtest-isolation invariant (synthetic rows must never inflate a story's
corroboration count or surface it as breaking)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from storage.story_corroboration import (
    _jaccard,
    _normalize,
    corroborated_breaking,
)

NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _iso(minutes_ago: float) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute(
        "CREATE TABLE articles ("
        "id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT, "
        "published TEXT, kw_score REAL, ai_score REAL, urgency INTEGER, "
        "full_text BLOB, first_seen TEXT, cycle INTEGER, "
        "time_sensitivity REAL, ml_score REAL, score_source TEXT)"
    )
    yield c
    c.close()


def _add(conn, *, title, source, url, first_seen, ai_score=0.0, urgency=0):
    conn.execute(
        "INSERT INTO articles (id, url, title, source, first_seen, ai_score, "
        "urgency, kw_score, cycle) VALUES (?,?,?,?,?,?,?,?,?)",
        (f"{source}:{title}:{url}", url, title, source, first_seen,
         ai_score, urgency, 1.0, 0),
    )


# ── unit: normalization / similarity ────────────────────────────────────────

def test_normalize_drops_stopwords_and_punctuation():
    a = _normalize("The Fed has CUT rates by 50bps!!!")
    b = _normalize("Fed cuts rates 50bps")
    assert "fed" in a and "rates" in a and "50bps" in a
    assert "the" not in a and "has" not in a
    assert _jaccard(a, b) >= 0.6  # same story after normalization


def test_unrelated_headlines_have_low_similarity():
    a = _normalize("Apple unveils new iPhone at fall event")
    b = _normalize("Fed holds interest rates steady amid inflation")
    assert _jaccard(a, b) < 0.6


# ── corroboration counting ──────────────────────────────────────────────────

def test_distinct_sources_are_counted(conn):
    base = "Nvidia beats earnings, raises guidance on AI demand"
    for i, src in enumerate(["rss", "gdelt", "finnhub", "reddit"]):
        _add(conn, title=f"{base}" if i % 2 else base + "!",
             source=src, url=f"https://site{i}.com/a", first_seen=_iso(10 + i))
    out = corroborated_breaking(conn, hours=3, min_sources=3, now=NOW)
    assert len(out) == 1
    story = out[0]
    assert story["source_count"] == 4
    assert sorted(story["sources"]) == ["finnhub", "gdelt", "reddit", "rss"]
    assert story["domain_count"] == 4
    assert story["article_count"] == 4


def test_single_source_not_corroborated(conn):
    _add(conn, title="Tesla recalls 12000 vehicles over brake fault",
         source="rss", url="https://x.com/1", first_seen=_iso(5))
    assert corroborated_breaking(conn, hours=3, min_sources=3, now=NOW) == []


def test_out_of_window_excluded(conn):
    base = "ECB signals pause in rate hiking cycle"
    for src in ["rss", "gdelt", "finnhub"]:
        _add(conn, title=base, source=src, url=f"https://{src}.com/x",
             first_seen=_iso(400))  # ~6.6h ago, outside a 3h window
    assert corroborated_breaking(conn, hours=3, min_sources=3, now=NOW) == []


def test_distinct_stories_not_merged(conn):
    for src in ["rss", "gdelt", "finnhub"]:
        _add(conn, title="Apple unveils new iPhone at fall hardware event",
             source=src, url=f"https://{src}.com/apple", first_seen=_iso(8))
    for src in ["reddit", "polygon", "yahoo"]:
        _add(conn, title="Fed holds interest rates steady amid sticky inflation",
             source=src, url=f"https://{src}.com/fed", first_seen=_iso(9))
    out = corroborated_breaking(conn, hours=3, min_sources=3, now=NOW)
    assert len(out) == 2
    titles = {s["title"] for s in out}
    assert any("iPhone" in t for t in titles)
    assert any("Fed holds" in t for t in titles)


# ── CRITICAL: backtest isolation ────────────────────────────────────────────

def test_backtest_rows_never_inflate_corroboration(conn):
    """A story with only 2 live sources must NOT clear a 3-source threshold
    just because backtest:// / backtest_* / opus_annotation rows share the
    headline. Synthetic training rows must stay out of the live signal."""
    headline = "Microsoft to acquire gaming studio in $4B all-cash deal"
    _add(conn, title=headline, source="rss",
         url="https://reuters.com/m", first_seen=_iso(6))
    _add(conn, title=headline, source="gdelt",
         url="https://bloomberg.com/m", first_seen=_iso(7))
    # Synthetic rows that share the exact headline:
    _add(conn, title=headline, source="backtest_run_42_winner",
         url="backtest://run_42/2026-05-18/BUY/MSFT", first_seen=_iso(5))
    _add(conn, title=headline, source="opus_annotation_cycle_9",
         url="https://opus.local/note", first_seen=_iso(5))
    _add(conn, title=headline, source="rss",
         url="backtest://run_42/2026-05-18/SELL/MSFT", first_seen=_iso(5))

    # Only 2 genuine sources → must not reach a 3-source threshold.
    assert corroborated_breaking(conn, hours=3, min_sources=3, now=NOW) == []

    # At a 2-source threshold it surfaces, but with ZERO synthetic
    # contamination in the source list or the counts.
    out = corroborated_breaking(conn, hours=3, min_sources=2, now=NOW)
    assert len(out) == 1
    s = out[0]
    assert s["source_count"] == 2
    assert sorted(s["sources"]) == ["gdelt", "rss"]
    assert all("backtest" not in src and "opus_annotation" not in src
               for src in s["sources"])
    assert s["article_count"] == 2  # synthetic rows not clustered at all


def test_results_sorted_by_corroboration_then_burst(conn):
    # Story A: 5 sources over a wide span.
    for i, src in enumerate(["rss", "gdelt", "finnhub", "reddit", "polygon"]):
        _add(conn, title="Oil prices surge on supply disruption fears",
             source=src, url=f"https://{src}.com/oil", first_seen=_iso(120 - i * 20))
    # Story B: 3 sources in a tight 2-minute burst.
    for i, src in enumerate(["yahoo", "newsapi", "substack"]):
        _add(conn, title="Bank of Japan unexpectedly hikes policy rate",
             source=src, url=f"https://{src}.com/boj", first_seen=_iso(10 + i))
    out = corroborated_breaking(conn, hours=3, min_sources=3, now=NOW)
    assert [s["source_count"] for s in out] == [5, 3]
    assert "Oil prices" in out[0]["title"]
