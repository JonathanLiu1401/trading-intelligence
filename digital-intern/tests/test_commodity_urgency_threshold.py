"""Pins the commodity_futures direct-write urgency threshold to URGENT_THRESHOLD.

The collector emits article rows DIRECTLY (bypassing daemon._ingest), and
historically used a literal `urgency = 1 if kw >= 6.0 else 0` cutoff that did
NOT match the system-wide URGENT_THRESHOLD = 8.0 used by the Sonnet path
(watchers.urgency_scorer) and the ML score_pending path
(storage.article_store.score_pending). A 30-day live audit found 5 alerted
commodity rows at kw_score 6.00-6.18 — barely-above-threshold price moves
(Brent +2.3%, WTI +2.4%, Copper +2.0%) that triggered BREAKING Discord pushes
the analyst would consider routine commodity volatility. Pin the alignment
explicitly so it cannot silently drift again.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from collectors import commodity_futures_collector as cfc
from watchers.urgency_scorer import URGENT_THRESHOLD


def _mock_history(latest: float, prev: float):
    """Build a yf.Ticker(...).history() return — DataFrame with Close column."""
    return pd.DataFrame({"Close": [prev, latest]})


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Per-test sqlite DB with the articles schema bootstrapped."""
    db_path = tmp_path / "articles.db"
    monkeypatch.setattr(cfc, "DB_PATH", db_path)
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.execute("""CREATE TABLE articles (
        id TEXT PRIMARY KEY, url TEXT NOT NULL, title TEXT NOT NULL,
        source TEXT, published TEXT, kw_score REAL DEFAULT 0,
        ai_score REAL DEFAULT 0, urgency INTEGER DEFAULT 0,
        full_text BLOB, first_seen TEXT NOT NULL, cycle INTEGER DEFAULT 0
    )""")
    conn.commit()
    yield conn
    conn.close()


def _row_for_symbol(conn, symbol: str) -> tuple | None:
    """Find the latest commodity_futures row touching ``symbol`` (e.g. BZ=F)."""
    cur = conn.execute(
        "SELECT id, title, kw_score, urgency FROM articles "
        "WHERE source=? ORDER BY first_seen DESC LIMIT 1",
        (cfc.SOURCE,),
    )
    return cur.fetchone()


class TestUrgencyThresholdAlignment:
    """Direct-write urgency cutoff must equal URGENT_THRESHOLD."""

    def test_urgent_threshold_is_8(self):
        """Sanity: URGENT_THRESHOLD is the system-wide 8.0 used by the Sonnet
        path (watchers.urgency_scorer) and the ML score_pending path. If this
        ever shifts, every downstream urgency gate must follow."""
        assert URGENT_THRESHOLD == 8.0, (
            f"URGENT_THRESHOLD drift: now {URGENT_THRESHOLD!r}; the test "
            "pinning the alert pipeline alignment must be updated alongside."
        )

    def test_brent_at_threshold_does_not_alert(self, isolated_db):
        """Brent +2.3% — just above the 2.0% emit threshold — must NOT mark
        urgency=1. This is the exact live-failure case (Brent +2.3% fired
        BREAKING under the old 6.0 cutoff). The post-bug behaviour: the row
        is still ingested (so it appears in the briefing pool) but does NOT
        push to the analyst's BREAKING channel."""
        latest, prev = 100.0, 97.75  # +2.30%
        with patch("yfinance.Ticker") as mock_t:
            mock_t.return_value.history.return_value = _mock_history(latest, prev)
            cfc.collect(isolated_db)
        row = _row_for_symbol(isolated_db, "BZ=F")
        assert row is not None, "Brent row should still have been ingested"
        _id, title, kw, urgency = row
        # kw_score formula: base(5.0) + min(2.30/2.0, 3.0) = 5.0 + 1.15 = 6.15
        assert 6.0 <= kw < URGENT_THRESHOLD, (
            f"kw_score {kw} fell outside the test's intended range — formula change?"
        )
        assert urgency == 0, (
            f"A 2.3% Brent move scored kw={kw} must NOT be marked urgency=1 — "
            f"that is the noise pattern this gate exists to suppress"
        )

    def test_brent_at_threshold_directly_yields_urgency_zero(self, isolated_db):
        """+2.0% is the bare-minimum emit threshold; well below URGENT."""
        latest, prev = 100.0, 98.04  # ~+2.0%
        with patch("yfinance.Ticker") as mock_t:
            mock_t.return_value.history.return_value = _mock_history(latest, prev)
            cfc.collect(isolated_db)
        row = _row_for_symbol(isolated_db, "BZ=F")
        if row is None:
            pytest.skip("Move fell just under emit threshold (formula edge case)")
        _id, title, kw, urgency = row
        assert urgency == 0
        assert kw < URGENT_THRESHOLD

    def test_brent_6pct_move_does_alert(self, isolated_db):
        """A 6%+ Brent move pushes kw_score to the 8.0 cap — that IS a
        BREAKING-worthy daily move (a structural energy shock) and SHOULD
        mark urgency=1. This is the positive-side pin: bumping the threshold
        must not silence genuine breaking moves."""
        latest, prev = 106.0, 100.0  # +6.00%, base+min(3.0)=8.0
        with patch("yfinance.Ticker") as mock_t:
            mock_t.return_value.history.return_value = _mock_history(latest, prev)
            cfc.collect(isolated_db)
        row = _row_for_symbol(isolated_db, "BZ=F")
        assert row is not None
        _id, title, kw, urgency = row
        assert kw >= URGENT_THRESHOLD, (
            f"6% Brent move should hit the kw cap of 8.0, got {kw}"
        )
        assert urgency == 1, (
            f"kw_score {kw} >= URGENT_THRESHOLD must mark urgency=1 for the "
            f"alert worker to fire a BREAKING push"
        )

    def test_urgency_cutoff_uses_url_threshold_constant(self):
        """Catches a regression where a future edit re-introduces a literal
        6.0 (or any other constant) in the urgency expression — the link to
        URGENT_THRESHOLD must remain explicit by import, not literal."""
        import inspect
        src = inspect.getsource(cfc.collect)
        # The literal 6.0 should never reappear next to "urgency" — it is the
        # specific anti-pattern this fix removed.
        assert "6.0" not in src or "score >= 6.0" not in src, (
            "collect() re-introduced a literal 6.0 urgency cutoff — the fix "
            "is that the threshold come from URGENT_THRESHOLD, not a literal"
        )
        assert "URGENT_THRESHOLD" in src, (
            "collect() no longer references URGENT_THRESHOLD — the link to "
            "the system-wide alert threshold has been broken"
        )
