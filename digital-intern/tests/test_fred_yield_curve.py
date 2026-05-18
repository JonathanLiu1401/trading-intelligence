"""Tests for collectors/fred_collector.py — 10Y-2Y yield-curve spread.

The yield-curve spread is the recession leading indicator; the synthetic
article it emits must:
  * compute spread = DGS10 - DGS2 for dates BOTH series cover (FRED publishes
    Treasury constant-maturity series on the same daily calendar — but a
    revision lag can leave one leg trailing the other for a day, so we
    intersect-by-date rather than naively zipping by index)
  * tag the regime correctly: "INVERTED" iff spread < 0, "positive" otherwise
    (the briefing's keyword scorer keys on this token)
  * dedup by date so a same-day re-run does NOT emit a duplicate (would
    pollute the briefing's top-N with copies of one synthetic row)
  * survive a missing leg gracefully — return ``[]`` rather than raise — so a
    transient FRED hiccup on one series never breaks the whole cycle.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from collectors import fred_collector


# ── helpers ────────────────────────────────────────────────────────────────
def _mem_seen_conn():
    """In-memory replica of the seen_articles.db schema collect_fred opens."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE seen_articles (
            id TEXT PRIMARY KEY, link TEXT, title TEXT,
            source TEXT, first_seen TEXT
        )"""
    )
    return conn


# ── _yield_curve_articles unit tests ───────────────────────────────────────
def test_yield_curve_spread_inverted_regime():
    """Spread of -0.40 (10Y=4.10, 2Y=4.50) must be tagged INVERTED."""
    conn = _mem_seen_conn()
    dgs10 = [("2026-05-15", 4.20), ("2026-05-16", 4.10)]
    dgs2 = [("2026-05-15", 4.30), ("2026-05-16", 4.50)]
    arts = fred_collector._yield_curve_articles(conn, dgs10, dgs2)
    assert len(arts) == 2
    latest = arts[-1]
    assert latest["published"] == "2026-05-16"
    assert latest["source"] == "fred/10y2y_spread"
    assert "INVERTED" in latest["title"]
    # spread = 4.10 - 4.50 = -0.40 — token must be in body for the briefing
    # keyword scorer to surface "inverted" / "recession" correctly.
    assert "-0.4" in latest["title"] or "-0.40" in latest["summary"]
    assert "inverted yield" in latest["summary"].lower()


def test_yield_curve_spread_positive_regime():
    """Positive spread (10Y > 2Y) must be tagged 'positive', not INVERTED."""
    conn = _mem_seen_conn()
    dgs10 = [("2026-05-16", 4.50)]
    dgs2 = [("2026-05-16", 4.10)]
    arts = fred_collector._yield_curve_articles(conn, dgs10, dgs2)
    assert len(arts) == 1
    a = arts[0]
    assert "positive" in a["title"] and "INVERTED" not in a["title"]
    # spread = +0.40
    assert "0.4" in a["title"]


def test_yield_curve_spread_intersects_dates_not_zips():
    """If one series trails the other by a day, we must use the LATEST
    common date — never zip by index, which would pair 10Y on day N with 2Y
    on day N-1 and silently corrupt the spread."""
    conn = _mem_seen_conn()
    dgs10 = [("2026-05-14", 4.20), ("2026-05-15", 4.30), ("2026-05-16", 4.40)]
    # DGS2 missing 2026-05-16 (FRED publication lag on one leg).
    dgs2 = [("2026-05-14", 4.00), ("2026-05-15", 4.10)]
    arts = fred_collector._yield_curve_articles(conn, dgs10, dgs2)
    assert len(arts) == 2
    pubs = [a["published"] for a in arts]
    assert pubs == ["2026-05-14", "2026-05-15"]
    # Validate the 05-15 spread used the right pair: 4.30 - 4.10 = 0.20
    art_15 = next(a for a in arts if a["published"] == "2026-05-15")
    assert "0.2" in art_15["title"]


def test_yield_curve_spread_dedup_by_date():
    """A second call on the same conn for the same date must emit no
    article — would otherwise duplicate the synthetic row every cycle."""
    conn = _mem_seen_conn()
    dgs10 = [("2026-05-16", 4.50)]
    dgs2 = [("2026-05-16", 4.10)]
    first = fred_collector._yield_curve_articles(conn, dgs10, dgs2)
    conn.commit()
    assert len(first) == 1
    second = fred_collector._yield_curve_articles(conn, dgs10, dgs2)
    assert second == []


def test_yield_curve_spread_missing_leg_safe():
    """If either DGS10 or DGS2 is empty (transient FRED fetch failure on one
    side), return [] rather than raise — never break the whole collect cycle."""
    conn = _mem_seen_conn()
    assert fred_collector._yield_curve_articles(conn, [], [("2026-05-16", 4.1)]) == []
    assert fred_collector._yield_curve_articles(conn, [("2026-05-16", 4.5)], []) == []
    assert fred_collector._yield_curve_articles(conn, [], []) == []


def test_yield_curve_spread_no_overlap_safe():
    """Series with no common dates must yield no articles (rather than
    crash) — this can happen briefly during a FRED revision window."""
    conn = _mem_seen_conn()
    dgs10 = [("2026-05-15", 4.30)]
    dgs2 = [("2026-05-16", 4.10)]
    assert fred_collector._yield_curve_articles(conn, dgs10, dgs2) == []


def test_yield_curve_change_direction_flatten_vs_steepen():
    """Direction tag: spread shrinking (or going more negative) = flattening;
    spread widening = steepening. Drives the briefing's curve-move narrative."""
    conn = _mem_seen_conn()
    # 05-15 spread=+0.30; 05-16 spread=+0.10 → flattening.
    dgs10 = [("2026-05-15", 4.40), ("2026-05-16", 4.30)]
    dgs2 = [("2026-05-15", 4.10), ("2026-05-16", 4.20)]
    arts = fred_collector._yield_curve_articles(conn, dgs10, dgs2)
    latest = arts[-1]
    assert "flattening" in latest["title"]
    assert "steepening" not in latest["title"]


# ── DGS2 is wired into FRED_SERIES ─────────────────────────────────────────
def test_dgs2_registered_in_fred_series():
    """The 2-year Treasury must be in FRED_SERIES — without it the spread
    builder receives an empty DGS2 leg in production."""
    assert "DGS2" in fred_collector.FRED_SERIES
    assert "2-year" in fred_collector.FRED_SERIES["DGS2"].lower()


# ── End-to-end collect_fred integration (mocked HTTP) ──────────────────────
def test_collect_fred_emits_yield_curve_spread(tmp_path, monkeypatch):
    """collect_fred() must include the synthetic spread article when both
    DGS10 and DGS2 fetches succeed. End-to-end through the same dedup DB
    path the real collector uses, so a refactor of the dedup wiring is
    caught.
    """
    # Redirect the dedup DB into tmp_path so the test never touches
    # data/seen_articles.db.
    monkeypatch.setattr(fred_collector, "DB_PATH", tmp_path / "seen_articles.db")

    def fake_fetch(series: str):
        # Deterministic fixtures sized so RECENT_N (=3) surfaces all rows but
        # the spread builder only needs 2 dates of overlap to compute change.
        if series == "DGS10":
            return [("2026-05-14", 4.20), ("2026-05-15", 4.25), ("2026-05-16", 4.10)]
        if series == "DGS2":
            return [("2026-05-14", 4.00), ("2026-05-15", 4.05), ("2026-05-16", 4.50)]
        # Other series: return a minimal 2-row series so the per-series block
        # doesn't fail-out on the empty-rows branch (which would still be fine
        # but obscures the intent of this test).
        return [("2026-05-15", 1.0), ("2026-05-16", 1.1)]

    monkeypatch.setattr(fred_collector, "_fetch_series", fake_fetch)
    items = fred_collector.collect_fred()
    spread_items = [a for a in items if a["source"] == "fred/10y2y_spread"]
    assert spread_items, "expected at least one fred/10y2y_spread article"
    # The 2026-05-16 row is INVERTED (4.10 - 4.50 = -0.40)
    inv = next((a for a in spread_items if a["published"] == "2026-05-16"), None)
    assert inv is not None
    assert "INVERTED" in inv["title"]
    # A second collect on the same dedup DB must NOT re-emit the spread.
    items2 = fred_collector.collect_fred()
    spread_items2 = [a for a in items2 if a["source"] == "fred/10y2y_spread"]
    assert spread_items2 == []


def test_collect_fred_missing_dgs2_does_not_raise(tmp_path, monkeypatch):
    """If DGS2 fetch raises but DGS10 succeeds, the collector must still
    emit DGS10's articles and quietly skip the spread block. A transient
    FRED outage on one leg should never blow up the whole cycle."""
    monkeypatch.setattr(fred_collector, "DB_PATH", tmp_path / "seen_articles.db")

    def fake_fetch(series: str):
        if series == "DGS2":
            raise RuntimeError("simulated FRED 503 on DGS2")
        return [("2026-05-16", 4.10)]

    monkeypatch.setattr(fred_collector, "_fetch_series", fake_fetch)
    items = fred_collector.collect_fred()
    # No spread (DGS2 failed)
    assert all(a["source"] != "fred/10y2y_spread" for a in items)
    # DGS10 should still be present
    assert any(a["source"] == "fred/DGS10" for a in items)
