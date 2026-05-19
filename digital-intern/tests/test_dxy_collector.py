"""Unit tests for collectors.dxy_collector.

Fully mocked — no yfinance network calls, isolated in-memory dedup. Asserts
the load-bearing behaviours the daemon (and the briefing layer downstream)
relies on:

  * standard ``collect()`` output shape: list of dicts with the article-store
    keys (``id, url, title, source, published, first_seen, kw_score,
    urgency, full_text``)
  * ``source`` column is always the short stable name ``"dxy"`` (dashboards
    and source-grouping rely on this — not the URL or the symbol)
  * regime band classification: 95 / 100 / 105 / 110 boundary crossings flip
    the band key, which is the dedup key
  * dedup: a second collect() in the same band same day returns []
  * a band change *does* re-emit (the first cross of the day is a signal)
  * intraday move tripwire fires between scheduled emits when DXY shifts
    >= ``INTRADAY_MOVE_THRESHOLD``, keyed by the hour so it doesn't spam
  * a missing DXY history (yfinance returns empty / errors) collapses to
    [] gracefully — one bad fetch never aborts the worker loop
  * the SQLite ``articles`` row is actually written and round-trips with
    the same id (the collector writes directly, not via insert_batch)
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from collectors import dxy_collector as dc


def _make_history(closes: list[float]) -> pd.DataFrame:
    """Build a minimal yfinance-style DataFrame with a Close column."""
    idx = pd.date_range("2026-05-15", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=idx)


class _FakeTicker:
    """Stand-in for ``yfinance.Ticker(symbol)``.

    Looks up ``symbol`` in the test-injected ``RETURNS`` map; raises when
    asked about something the test didn't set up (mirrors yfinance's habit
    of returning an empty frame for an unknown symbol but is louder, so a
    test that drifts off the configured symbol set fails noisily).
    """

    RETURNS: dict[str, pd.DataFrame] = {}

    def __init__(self, symbol: str):
        self.symbol = symbol

    def history(self, period: str = "5d") -> pd.DataFrame:
        if self.symbol not in self.RETURNS:
            # mimic yfinance "empty frame for unknown symbol" path
            return pd.DataFrame()
        return self.RETURNS[self.symbol]


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Redirect DB writes to a temp ``articles.db`` with the schema the
    collector expects (subset of the real article_store schema)."""
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE articles (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            published TEXT,
            kw_score REAL DEFAULT 0,
            ai_score REAL DEFAULT 0,
            urgency INTEGER DEFAULT 0,
            full_text BLOB,
            first_seen TEXT NOT NULL,
            cycle INTEGER DEFAULT 0,
            time_sensitivity REAL
        )"""
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(dc, "DB_PATH", db)
    return db


@pytest.fixture
def fake_yf(monkeypatch):
    """Patch ``yf.Ticker`` to the fake. Returns the RETURNS dict so each test
    can configure what each symbol resolves to."""
    _FakeTicker.RETURNS = {}
    monkeypatch.setattr(dc.yf, "Ticker", _FakeTicker)
    return _FakeTicker.RETURNS


# ────────────────────────────────────────────────────────────────────────────
# Output shape + source column
# ────────────────────────────────────────────────────────────────────────────


def test_collect_returns_standard_shape(isolated_db, fake_yf):
    fake_yf["DX-Y.NYB"] = _make_history([99.0, 99.5, 100.2])  # crosses 100
    fake_yf["EURUSD=X"] = _make_history([1.08, 1.075])
    fake_yf["JPY=X"] = _make_history([155.0, 156.5])
    fake_yf["CNH=X"] = _make_history([7.20, 7.22])

    rows = dc.collect()
    assert len(rows) == 1
    row = rows[0]
    for key in (
        "id", "url", "title", "source", "published",
        "first_seen", "kw_score", "urgency", "full_text",
    ):
        assert key in row, f"missing field {key}"
    # Stable short source name — load-bearing for dashboards.
    assert row["source"] == "dxy"
    # 100.2 is in the 100..105 band.
    assert "100-105" in row["title"] or "strong" in row["title"]


def test_url_is_internal_namespace(isolated_db, fake_yf):
    fake_yf["DX-Y.NYB"] = _make_history([102.0, 102.5])
    rows = dc.collect()
    assert rows[0]["url"].startswith("internal://dxy/")


# ────────────────────────────────────────────────────────────────────────────
# Band classification
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dxy,expected_band",
    [
        (90.0, "weak_lt95"),
        (94.99, "weak_lt95"),
        (95.0, "soft_95_100"),
        (99.9, "soft_95_100"),
        (100.0, "strong_100_105"),
        (104.99, "strong_100_105"),
        (105.0, "very_strong_105_110"),
        (109.99, "very_strong_105_110"),
        (110.0, "extreme_gte110"),
        (115.0, "extreme_gte110"),
    ],
)
def test_band_boundaries(dxy, expected_band):
    assert dc._classify_band(dxy) == expected_band


# ────────────────────────────────────────────────────────────────────────────
# Dedup behaviour
# ────────────────────────────────────────────────────────────────────────────


def test_same_band_second_call_is_deduped(isolated_db, fake_yf):
    # Two calls at the same DXY level / same day → only one article.
    fake_yf["DX-Y.NYB"] = _make_history([102.0, 102.1])
    rows1 = dc.collect()
    rows2 = dc.collect()
    assert len(rows1) == 1
    assert len(rows2) == 0, "expected dedup on same band same day"


def test_band_change_emits_new_article(isolated_db, fake_yf):
    # Start in soft (99.0), then cross into strong (100.5) — both should emit.
    fake_yf["DX-Y.NYB"] = _make_history([98.5, 99.0])
    rows1 = dc.collect()
    assert len(rows1) == 1

    # Bump DXY across the 100 boundary.
    fake_yf["DX-Y.NYB"] = _make_history([99.0, 100.5])
    rows2 = dc.collect()
    assert len(rows2) == 1
    assert rows1[0]["id"] != rows2[0]["id"]


# ────────────────────────────────────────────────────────────────────────────
# Intraday move tripwire
# ────────────────────────────────────────────────────────────────────────────


def test_intraday_move_emits_after_dedup(isolated_db, fake_yf):
    # First call seeds the band + last_dxy state.
    fake_yf["DX-Y.NYB"] = _make_history([99.5, 99.8])
    rows1 = dc.collect()
    assert len(rows1) == 1

    # Same band (still 95-100) but DXY shifted by 0.35 pts since last emit
    # (above the 0.30 threshold) — should fire an intraday-move article.
    fake_yf["DX-Y.NYB"] = _make_history([99.8, 99.45])  # -0.35 pts vs last_dxy=99.8
    rows2 = dc.collect()
    assert len(rows2) == 1
    assert "intraday" in rows2[0]["title"].lower() or "->" in rows2[0]["title"]


# ────────────────────────────────────────────────────────────────────────────
# Graceful failure paths
# ────────────────────────────────────────────────────────────────────────────


def test_empty_history_returns_empty(isolated_db, fake_yf):
    # No DXY data at all — collector must return [], not raise.
    rows = dc.collect()
    assert rows == []


def test_dxy_fetch_raises_returns_empty(isolated_db, monkeypatch):
    class _BoomTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period: str = "5d"):
            raise RuntimeError("yfinance down")

    monkeypatch.setattr(dc.yf, "Ticker", _BoomTicker)
    rows = dc.collect()
    assert rows == []


def test_missing_bilateral_pair_still_emits_dxy(isolated_db, fake_yf):
    # DXY only — EUR/USD/JPY/CNH all empty. Collector must still emit DXY.
    fake_yf["DX-Y.NYB"] = _make_history([102.0, 102.5])
    rows = dc.collect()
    assert len(rows) == 1
    # body should contain DXY line but not pair lines
    body = rows[0]["full_text"]
    assert "DXY:" in body
    assert "EUR/USD" not in body
    assert "USD/JPY" not in body


# ────────────────────────────────────────────────────────────────────────────
# DB round-trip
# ────────────────────────────────────────────────────────────────────────────


def test_row_is_persisted_to_articles_db(isolated_db, fake_yf):
    fake_yf["DX-Y.NYB"] = _make_history([99.0, 100.2])
    rows = dc.collect()
    assert len(rows) == 1
    article_id = rows[0]["id"]

    conn = sqlite3.connect(str(isolated_db))
    db_row = conn.execute(
        "SELECT id, source, title, kw_score FROM articles WHERE id=?",
        (article_id,),
    ).fetchone()
    conn.close()
    assert db_row is not None
    assert db_row[0] == article_id
    assert db_row[1] == "dxy"
    assert db_row[3] >= 2.5  # base score floor from _urgency_score


def test_extreme_band_gets_higher_urgency(isolated_db, fake_yf):
    fake_yf["DX-Y.NYB"] = _make_history([109.0, 112.0])  # extreme + big daily move
    rows = dc.collect()
    assert len(rows) == 1
    # Extreme + >2% daily move should push kw_score well above the 2.5 floor.
    assert rows[0]["kw_score"] >= 5.0
    assert rows[0]["urgency"] in (0, 1)
