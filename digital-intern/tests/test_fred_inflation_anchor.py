"""Tests for collectors/fred_collector.py — 5y5y forward inflation anchor.

The T5YIFR series is the Fed's preferred long-term inflation expectations
gauge. The synthetic article it emits must:
  * tag the regime correctly per the band thresholds — "DEANCHORED" iff
    value >= 2.50, "BELOW_TARGET" iff < 1.75, "elevated" in [2.25, 2.50),
    "anchored" in [1.75, 2.25) (the briefing's keyword scorer keys on the
    DEANCHORED token to surface a credibility-stress headline)
  * dedup by date so a same-day re-run does NOT emit a duplicate
  * survive an empty input safely — return ``[]`` rather than raise — so a
    transient FRED hiccup on T5YIFR never breaks the whole cycle.
"""
from __future__ import annotations

import sqlite3

from collectors import fred_collector


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


def test_inflation_anchor_deanchored_regime():
    """>=2.50% must be tagged DEANCHORED — the briefing scorer keys on it."""
    conn = _mem_seen_conn()
    series = [("2026-05-15", 2.40), ("2026-05-16", 2.62)]
    arts = fred_collector._inflation_anchor_articles(conn, series)
    assert len(arts) == 2
    latest = arts[-1]
    assert latest["published"] == "2026-05-16"
    assert latest["source"] == "fred/inflation_anchor_5y5y"
    assert "DEANCHORED" in latest["title"]
    assert "2.62" in latest["title"]
    # Body must carry the regime narrative so the keyword scorer surfaces
    # "credibility risk" too.
    assert "DEANCHORED" in latest["summary"]
    assert "credibility" in latest["summary"].lower()


def test_inflation_anchor_anchored_regime():
    """In [1.75, 2.25) must read 'anchored' (lowercase, not the alert token)."""
    conn = _mem_seen_conn()
    arts = fred_collector._inflation_anchor_articles(conn, [("2026-05-16", 2.05)])
    assert len(arts) == 1
    a = arts[0]
    assert "anchored" in a["title"] and "DEANCHORED" not in a["title"]


def test_inflation_anchor_elevated_regime():
    """In [2.25, 2.50) must read 'elevated' — between target and de-anchor."""
    conn = _mem_seen_conn()
    arts = fred_collector._inflation_anchor_articles(conn, [("2026-05-16", 2.30)])
    assert len(arts) == 1
    a = arts[0]
    assert "elevated" in a["title"]
    assert "DEANCHORED" not in a["title"]


def test_inflation_anchor_below_target_regime():
    """<1.75% must be tagged BELOW_TARGET — deflation/demand-trap signal."""
    conn = _mem_seen_conn()
    arts = fred_collector._inflation_anchor_articles(conn, [("2026-05-16", 1.40)])
    assert len(arts) == 1
    a = arts[0]
    assert "BELOW_TARGET" in a["title"]


def test_inflation_anchor_band_boundary_2_25_is_elevated_not_anchored():
    """Exactly at the 2.25 boundary — the lower edge of 'elevated' must
    capture the boundary value (the >= comparison is intentional)."""
    conn = _mem_seen_conn()
    arts = fred_collector._inflation_anchor_articles(conn, [("2026-05-16", 2.25)])
    assert len(arts) == 1
    assert "elevated" in arts[0]["title"]


def test_inflation_anchor_band_boundary_2_50_is_deanchored():
    """Exactly at the 2.50 boundary — DEANCHORED must trigger at the edge,
    not one cent above. Crossing 2.50 IS the credibility-stress trigger."""
    conn = _mem_seen_conn()
    arts = fred_collector._inflation_anchor_articles(conn, [("2026-05-16", 2.50)])
    assert len(arts) == 1
    assert "DEANCHORED" in arts[0]["title"]


def test_inflation_anchor_dedup_by_date():
    """A second call on the same conn for the same date must emit no
    article — would otherwise duplicate the synthetic row every cycle."""
    conn = _mem_seen_conn()
    series = [("2026-05-16", 2.30)]
    first = fred_collector._inflation_anchor_articles(conn, series)
    conn.commit()
    assert len(first) == 1
    second = fred_collector._inflation_anchor_articles(conn, series)
    assert second == []


def test_inflation_anchor_empty_input_safe():
    """Empty input must return [] rather than raise — never break the cycle."""
    conn = _mem_seen_conn()
    assert fred_collector._inflation_anchor_articles(conn, []) == []


def test_inflation_anchor_change_direction_rising_vs_falling():
    """Direction tag: value going up = rising; going down = falling."""
    conn = _mem_seen_conn()
    series = [("2026-05-15", 2.20), ("2026-05-16", 2.32)]
    arts = fred_collector._inflation_anchor_articles(conn, series)
    latest = arts[-1]
    assert "rising" in latest["title"]
    assert "falling" not in latest["title"]


def test_t5yifr_registered_in_fred_series():
    """The 5y5y forward must be in FRED_SERIES — without it the anchor
    builder receives empty rows in production."""
    assert "T5YIFR" in fred_collector.FRED_SERIES
    assert "forward" in fred_collector.FRED_SERIES["T5YIFR"].lower()


def test_breakevens_registered_in_fred_series():
    """5Y and 10Y breakevens must be in FRED_SERIES — the operator wants
    both the spot expectations curve and the 5y5y forward."""
    assert "T5YIE" in fred_collector.FRED_SERIES
    assert "T10YIE" in fred_collector.FRED_SERIES


def test_collect_fred_emits_inflation_anchor(tmp_path, monkeypatch):
    """collect_fred() must include the synthetic anchor article when T5YIFR
    fetch succeeds. End-to-end through the dedup DB path so a refactor of
    the wiring is caught."""
    monkeypatch.setattr(fred_collector, "DB_PATH", tmp_path / "seen_articles.db")

    def fake_fetch(series: str):
        if series == "T5YIFR":
            return [("2026-05-14", 2.20), ("2026-05-15", 2.40), ("2026-05-16", 2.55)]
        return [("2026-05-15", 1.0), ("2026-05-16", 1.1)]

    monkeypatch.setattr(fred_collector, "_fetch_series", fake_fetch)
    items = fred_collector.collect_fred()
    anchor_items = [a for a in items if a["source"] == "fred/inflation_anchor_5y5y"]
    assert anchor_items, "expected at least one inflation-anchor article"
    deanchored = next(
        (a for a in anchor_items if a["published"] == "2026-05-16"), None
    )
    assert deanchored is not None
    assert "DEANCHORED" in deanchored["title"]
    items2 = fred_collector.collect_fred()
    anchor_items2 = [a for a in items2 if a["source"] == "fred/inflation_anchor_5y5y"]
    assert anchor_items2 == []


def test_collect_fred_missing_t5yifr_does_not_raise(tmp_path, monkeypatch):
    """If T5YIFR fetch raises, the collector still emits other series and
    quietly skips the anchor block. A transient FRED outage on one leg
    should never blow up the whole cycle."""
    monkeypatch.setattr(fred_collector, "DB_PATH", tmp_path / "seen_articles.db")

    def fake_fetch(series: str):
        if series == "T5YIFR":
            raise RuntimeError("simulated FRED 503 on T5YIFR")
        return [("2026-05-16", 4.10)]

    monkeypatch.setattr(fred_collector, "_fetch_series", fake_fetch)
    items = fred_collector.collect_fred()
    assert all(a["source"] != "fred/inflation_anchor_5y5y" for a in items)
    # Other series still ingested
    assert any(a["source"] == "fred/DGS10" for a in items)
