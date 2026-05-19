"""Tests for analytics.ticker_concentration.

The audit is read-only and operates against a tmp SQLite file with the minimum
``articles`` columns it actually selects on. We do not stand up a full
``ArticleStore`` — its migrations are irrelevant here and would only slow the
test (same pattern as tests/test_publish_lag_audit.py).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from analytics import ticker_concentration


_MIN_COLS = """
CREATE TABLE articles (
    id          TEXT PRIMARY KEY,
    url         TEXT NOT NULL,
    title       TEXT,
    source      TEXT,
    ai_score    REAL,
    urgency     INTEGER
)
"""


def _build_db(path: Path, rows: list[tuple]) -> None:
    """rows: (id, url, title, source, ai_score, urgency)."""
    conn = sqlite3.connect(str(path))
    conn.execute(_MIN_COLS)
    conn.executemany(
        "INSERT INTO articles (id, url, title, source, ai_score, urgency) "
        "VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


@pytest.fixture
def patched_db(tmp_path, monkeypatch):
    db_path = tmp_path / "tc.db"

    def build(rows):
        _build_db(db_path, rows)
        monkeypatch.setattr(ticker_concentration, "_get_db_path", lambda: db_path)
        monkeypatch.setattr(
            ticker_concentration, "SNAPSHOT_PATH", tmp_path / "snap.json"
        )

    return db_path, build


# ── tests ───────────────────────────────────────────────────────────────


def test_empty_db_yields_zero_mentions(patched_db):
    db, build = patched_db
    build([])
    report = ticker_concentration.compute()
    assert report["scanned"] == 0
    assert report["articles_with_book_ticker"] == 0
    assert report["tickers"] == {}
    assert report["hhi"] == 0
    assert report["over_saturated"] == []
    # Every held ticker is under-covered when the corpus is empty.
    assert set(report["under_covered"]) == set(ticker_concentration._BOOK_TICKERS)


def test_book_ticker_parity():
    """ticker_concentration._BOOK_TICKERS must mirror the briefing's source so
    the audit counts the same held universe that BOOK HEAT / BOOK COVERAGE
    rank against. Drifting silently here would make the under_covered list lie.
    """
    from analysis import claude_analyst

    assert ticker_concentration._BOOK_TICKERS == claude_analyst._BOOK_TICKERS


def test_basic_mention_counting(patched_db):
    db, build = patched_db
    rows = [
        ("a1", "https://x/1", "NVDA earnings beat", "rss", 7.0, 0),
        ("a2", "https://x/2", "MSFT cloud growth", "rss", 6.0, 0),
        ("a3", "https://x/3", "NVDA partners with TSEM", "rss", 8.0, 2),
        ("a4", "https://x/4", "weather report", "rss", 1.0, 0),
    ]
    build(rows)
    report = ticker_concentration.compute()
    assert report["scanned"] == 4
    # 3 of 4 mention a held ticker (a4 has none).
    assert report["articles_with_book_ticker"] == 3
    tickers = report["tickers"]
    assert tickers["NVDA"]["n_mentions"] == 2
    assert tickers["MSFT"]["n_mentions"] == 1
    assert tickers["TSEM"]["n_mentions"] == 1
    # 2/3 of mentioning articles touched NVDA.
    assert tickers["NVDA"]["pct_share"] == pytest.approx(2 / 3 * 100, rel=1e-3)
    # Only a3 has urgency >= 2; it mentions NVDA + TSEM.
    assert tickers["NVDA"]["n_urgent"] == 1
    assert tickers["TSEM"]["n_urgent"] == 1
    assert tickers["MSFT"]["n_urgent"] == 0


def test_word_boundary_prevents_substring_false_positives(patched_db):
    """``MU`` must not match inside ``Micron`` or ``MUSEUM``, ``MUU`` must
    win over ``MU`` for ``MUU``-token text (longest-first alternation)."""
    db, build = patched_db
    rows = [
        # "Micron" contains the letters MU — must NOT match _BOOK_RE.
        ("a1", "https://x/1", "Micron raises guidance", "rss", 5.0, 0),
        # Explicit MU mention as a standalone word — must match.
        ("a2", "https://x/2", "MU shares rally on demand", "rss", 5.0, 0),
        # MUU is its own ticker — must register as MUU, not MU.
        ("a3", "https://x/3", "MUU ETF rebalances", "rss", 5.0, 0),
    ]
    build(rows)
    report = ticker_concentration.compute()
    tickers = report["tickers"]
    # Only a2 should hit MU.
    assert tickers.get("MU", {}).get("n_mentions") == 1
    assert tickers.get("MUU", {}).get("n_mentions") == 1
    # a1 ("Micron") had no held-ticker hit at all → only 2 mentioning articles.
    assert report["articles_with_book_ticker"] == 2


def test_synthetic_rows_excluded(patched_db):
    """``backtest://`` URLs and ``backtest_*``/``opus_annotation*`` sources are
    training-only and must never enter the audit — they would silently inflate
    NVDA/MSFT mention counts because backtest titles repeat the ticker."""
    db, build = patched_db
    rows = [
        ("live1", "https://x/1", "NVDA real headline", "rss", 7.0, 0),
        ("live2", "https://x/2", "MSFT real headline", "rss", 6.0, 0),
        (
            "bt1", "backtest://run_1/foo", "NVDA NVDA NVDA",
            "backtest_run_1_winner", 5.0, 0,
        ),
        ("bt2", "https://x/y", "NVDA bulk reweight", "backtest_run_2_rank1", 5.0, 0),
        ("op1", "https://x/z", "NVDA opus lesson", "opus_annotation_cycle_42", 5.0, 0),
    ]
    build(rows)
    report = ticker_concentration.compute()
    # Only the 2 live rows are scanned at all.
    assert report["scanned"] == 2
    assert report["tickers"]["NVDA"]["n_mentions"] == 1
    assert report["tickers"]["MSFT"]["n_mentions"] == 1


def test_over_saturation_flagged_at_threshold(patched_db):
    """A single ticker dominating ≥ SATURATION_PCT of book-mentioning
    articles must appear in over_saturated. The threshold is the operator's
    early-warning that the model's training signal is skewing one-sided."""
    db, build = patched_db
    rows = []
    # 8 NVDA-only articles, 2 MSFT-only — NVDA share is 80 %.
    for i in range(8):
        rows.append((f"n{i}", f"https://x/n{i}", f"NVDA news {i}", "rss", 7.0, 0))
    for i in range(2):
        rows.append((f"m{i}", f"https://x/m{i}", f"MSFT news {i}", "rss", 5.0, 0))
    build(rows)
    report = ticker_concentration.compute()
    assert "NVDA" in report["over_saturated"]
    assert "MSFT" not in report["over_saturated"]
    # HHI: 80^2 + 20^2 = 6400 + 400 = 6800.
    assert report["hhi"] == pytest.approx(6800.0)


def test_under_covered_lists_zero_mention_held_tickers(patched_db):
    """A held ticker with zero mentions in the window should appear in
    under_covered — the BOOK HEAT / coverage lines for that name go silent
    and the operator deserves to know."""
    db, build = patched_db
    rows = [
        ("a1", "https://x/1", "NVDA up 3 %", "rss", 7.0, 0),
    ]
    build(rows)
    report = ticker_concentration.compute()
    # NVDA was mentioned; every other held ticker should be under-covered.
    expected_uncovered = [
        t for t in ticker_concentration._BOOK_TICKERS if t != "NVDA"
    ]
    assert report["under_covered"] == expected_uncovered


def test_avg_ai_score_skips_nulls(patched_db):
    """Articles without an ai_score (still pending scoring) must not count
    toward avg_ai_score — they would pull the mean toward 0 falsely."""
    db, build = patched_db
    rows = [
        ("a1", "https://x/1", "NVDA scored", "rss", 8.0, 0),
        ("a2", "https://x/2", "NVDA unscored", "rss", None, 0),
        ("a3", "https://x/3", "NVDA scored low", "rss", 4.0, 0),
    ]
    build(rows)
    report = ticker_concentration.compute()
    # Mean of 8.0 and 4.0 only → 6.0.
    assert report["tickers"]["NVDA"]["avg_ai_score"] == pytest.approx(6.0)
    # All three count toward n_mentions, even the unscored one.
    assert report["tickers"]["NVDA"]["n_mentions"] == 3


def test_write_snapshot_round_trips(patched_db):
    db, build = patched_db
    rows = [
        ("a1", "https://x/1", "NVDA up", "rss", 6.0, 0),
        ("a2", "https://x/2", "MSFT down", "rss", 4.0, 0),
    ]
    build(rows)
    report = ticker_concentration.compute()
    out = ticker_concentration.write_snapshot(report)
    payload = json.loads(out.read_text())
    assert payload["tickers"]["NVDA"]["n_mentions"] == 1
    assert payload["tickers"]["MSFT"]["n_mentions"] == 1
    assert "generated_at" in payload
