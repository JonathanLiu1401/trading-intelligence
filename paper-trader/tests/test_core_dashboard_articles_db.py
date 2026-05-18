"""Regression lock: dashboard._articles_db_path() must resolve the digital-intern
articles.db through the SAME freshness-aware single source of truth the live
trader uses (signals._db_path(), AGENTS.md invariant #15/#16) — not its own
legacy USB-first existence probe.

Before this fix, `_articles_db_path()` was:

    if usb.exists(): return usb
    elif local.exists(): return local

i.e. it returned the USB mirror whenever it merely *existed*. The digital-intern
daemon falls back to writing the LOCAL copy when the USB mount is unavailable,
leaving a USB mirror that keeps `exists()`-ing while going day-stale. So
`/api/news-edge`, `/api/source-edge`, `/api/signal-followthrough`,
`/api/sector-pulse` and `/api/thesis-drift` read the STALE feed while the live
trader (`signals._db_path()`, freshness-aware since invariant #15) read the
FRESH one — the documented split-brain, root-fixed everywhere in signals.py but
left un-fixed in this one dashboard helper whose docstring still *claimed* to
"Match how paper_trader.signals discovers the digital-intern articles.db".

The discriminating assertion (fails on the old code, passes on the fixed code):
with LOCAL fresh and USB stale, the old code returns USB; the fixed code returns
LOCAL because it delegates to the freshness-aware resolver.
"""
from __future__ import annotations

import sqlite3
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard, signals


def _iso_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _build_articles_db(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY,
            url TEXT,
            title TEXT,
            source TEXT,
            ai_score REAL,
            urgency REAL,
            first_seen TEXT,
            full_text BLOB
        )
        """
    )
    for r in rows:
        conn.execute(
            "INSERT INTO articles (id, url, title, source, ai_score, urgency, "
            "first_seen, full_text) VALUES (?,?,?,?,?,?,?,?)",
            (
                r.get("id"),
                r.get("url"),
                r.get("title"),
                r.get("source"),
                r.get("ai_score"),
                r.get("urgency"),
                r.get("first_seen"),
                zlib.compress(r.get("body", "").encode("utf-8")) if r.get("body") else None,
            ),
        )
    conn.commit()
    conn.close()


def _make_db(path: Path, live_ago_h: float | None, backtest_ago_h: float | None = None):
    rows = []
    if live_ago_h is not None:
        rows.append({"id": 1, "url": "https://x/a", "title": "live", "source": "rss",
                     "ai_score": 8.0, "urgency": 0, "first_seen": _iso_ago(live_ago_h)})
    if backtest_ago_h is not None:
        rows.append({"id": 2, "url": "backtest://run_1/2026-05-16/BUY/NVDA",
                     "title": "synthetic", "source": "backtest_run_1_winner",
                     "ai_score": 5.0, "urgency": 0,
                     "first_seen": _iso_ago(backtest_ago_h)})
    _build_articles_db(path, rows)


@pytest.fixture
def two_dbs(tmp_path, monkeypatch):
    """USB + LOCAL temp paths wired into the freshness-aware resolver."""
    usb = tmp_path / "usb" / "articles.db"
    local = tmp_path / "local" / "articles.db"
    usb.parent.mkdir()
    local.parent.mkdir()
    monkeypatch.setattr(signals, "USB_DB", usb)
    monkeypatch.setattr(signals, "LOCAL_DB", local)
    signals._reset_resolver_cache()
    yield usb, local
    signals._reset_resolver_cache()


class TestArticlesDbPathIsFreshnessAware:
    def test_stale_usb_loses_to_fresh_local(self, two_dbs):
        """THE discriminating test. Old USB-first code returns USB (stale);
        the fixed delegate-to-signals code returns LOCAL (fresh)."""
        usb, local = two_dbs
        _make_db(usb, live_ago_h=30)
        _make_db(local, live_ago_h=1)
        assert dashboard._articles_db_path() == local

    def test_fresher_usb_still_wins_when_it_is_actually_fresher(self, two_dbs):
        """The fix is freshness-aware, not blindly LOCAL-first: a genuinely
        fresher USB must still be chosen."""
        usb, local = two_dbs
        _make_db(usb, live_ago_h=1)
        _make_db(local, live_ago_h=30)
        assert dashboard._articles_db_path() == usb

    def test_newer_backtest_row_on_stale_usb_does_not_win(self, two_dbs):
        """A fresh batch of injected backtest:// rows on a stale USB mirror
        must NOT make it look current (live-only filter, invariant #1/#15)."""
        usb, local = two_dbs
        _make_db(usb, live_ago_h=30, backtest_ago_h=0.1)
        _make_db(local, live_ago_h=1)
        assert dashboard._articles_db_path() == local

    def test_returns_none_when_no_db_exists(self, two_dbs):
        """Caller contract preserved: callers do `if path is None: <graceful>`,
        so a non-existent resolved DB must still surface as None, not a Path
        to a missing file."""
        # Neither temp DB created.
        assert dashboard._articles_db_path() is None

    def test_local_only_when_usb_missing(self, two_dbs):
        usb, local = two_dbs
        _make_db(local, live_ago_h=2)
        assert dashboard._articles_db_path() == local

    def test_agrees_with_signals_db_path(self, two_dbs):
        """The whole point: the dashboard helper and the live trader's
        resolver must never disagree on which feed is canonical."""
        usb, local = two_dbs
        _make_db(usb, live_ago_h=10)
        _make_db(local, live_ago_h=2)
        assert dashboard._articles_db_path() == signals._db_path()
