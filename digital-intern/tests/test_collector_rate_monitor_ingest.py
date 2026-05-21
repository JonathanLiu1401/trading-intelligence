"""Regression: ``collector_rate_monitor`` synthetic SILENT alerts must reach
``ArticleStore`` instead of being dropped by ``daemon._ingest``'s 0.5 noise gate.

Bug: ``collect_rate_alerts()`` emits operations alerts whose titles carry no
portfolio tickers / financial keywords ("⚠️ COLLECTOR SILENT: [Finnhub/...] —
0 articles in 3h (avg 75/day)"). ``triage.heuristic_scorer.score_article``
returns 0.0 on them ("no_keywords"), and ``daemon._ingest`` filters
``_relevance_score < 0.5`` → every synthetic alert was silently dropped before
``store.insert_batch`` and the feature was inert in production.

Fix: collector pre-sets ``_relevance_score``; ``_ingest`` skips heuristic
scoring when a pre-set value is present (opt-in). This file pins both halves
of the contract.

No live DB access — uses the per-test ``store_factory`` redirect.
"""
from __future__ import annotations

import importlib
import sys
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triage.heuristic_scorer import score_article


def _silent_alert_dict(source: str = "Finnhub/MarketWatch",
                       daily_avg: float = 75.2) -> dict:
    """Build a SILENT-alert dict shaped exactly like
    ``collect_rate_alerts()`` emits, so the heuristic test below targets the
    actual production title/summary text."""
    title = (
        f"⚠️ COLLECTOR SILENT: [{source}] — "
        f"0 articles in 3h (avg {daily_avg:.0f}/day)"
    )
    summary = (
        f"Source '{source}' has produced 0 articles in the last 3 hours. "
        f"Its 7-day baseline is {daily_avg:.1f} articles/day. Possible "
        f"collector failure or upstream outage."
    )
    return {
        "id": "test-id",
        "link": f"internal://collector_monitor/{source.replace('/', '_')}/2026-05-20",
        "title": title,
        "summary": summary,
        "source": "collector_monitor",
        "first_seen": "2026-05-20T17:00:00Z",
        "silent_source": source,
        "daily_avg": daily_avg,
    }


# ── A. heuristic score is genuinely 0 on these synthetic titles ──────────────
# Pins the underlying root cause: if this assertion ever flips (someone adds
# "silent"/"collector"/"articles" to a heuristic tier) the regression scenario
# below no longer reproduces and the test should be revisited.
def test_heuristic_score_is_zero_for_silent_alert_title():
    art = _silent_alert_dict()
    result = score_article(
        art["title"], art["summary"], art["source"], ""
    )
    assert result["score"] == 0.0, (
        f"Heuristic score is no longer 0 for SILENT alerts: {result}. "
        "If this is intentional, update collector_rate_monitor and "
        "remove the _relevance_score pre-score (no longer needed)."
    )


# ── B. _ingest drops a NON-pre-scored synthetic alert (the bug) ──────────────
def test_ingest_drops_silent_alert_without_prescoring(store_factory, monkeypatch):
    """Reproduces the pre-fix bug: a synthetic SILENT alert dict that does
    NOT carry ``_relevance_score`` is dropped by ``_ingest`` because the
    heuristic scorer returns 0.0 and the 0.5 gate filters it out.

    Sentinel for the contract: removing the pre-score path in _ingest must
    fail this test (a future "always-rescore" refactor would re-introduce the
    bug)."""
    import daemon as daemon_mod
    store = store_factory()

    art = _silent_alert_dict()
    inserted = daemon_mod._ingest(store, [art], "collector_monitor")

    assert inserted == 0, (
        f"Synthetic SILENT alert WITHOUT pre-score should be dropped by "
        f"the 0.5 noise gate (heuristic returns 0.0); instead {inserted} "
        f"row(s) reached the store."
    )
    # Defensive: confirm no row landed in articles.db either.
    row = store.conn.execute(
        "SELECT COUNT(*) FROM articles WHERE source = ?",
        ("collector_monitor",),
    ).fetchone()
    assert row[0] == 0, "Synthetic SILENT alert should not have been inserted"


# ── C. _ingest accepts the SAME alert when collector pre-scores it ───────────
def test_ingest_keeps_silent_alert_with_prescore(store_factory):
    """The fix path: when the collector sets ``_relevance_score`` (>= 0.5),
    ``_ingest`` honors it and skips the heuristic, so the synthetic alert
    actually lands in articles.db."""
    import daemon as daemon_mod
    store = store_factory()

    art = _silent_alert_dict()
    art["_relevance_score"] = 3.0
    inserted = daemon_mod._ingest(store, [art], "collector_monitor")

    assert inserted == 1, (
        f"Pre-scored SILENT alert should reach the store; got inserted={inserted}"
    )
    row = store.conn.execute(
        "SELECT source, kw_score FROM articles WHERE source = ?",
        ("collector_monitor",),
    ).fetchone()
    assert row is not None, "Synthetic SILENT alert missing from articles.db"
    assert row[0] == "collector_monitor"
    # kw_score in DB == the _relevance_score the collector pre-set
    assert abs(row[1] - 3.0) < 1e-6, (
        f"Pre-set _relevance_score=3.0 should be persisted as kw_score; got {row[1]}"
    )


# ── D. End-to-end: collect_rate_alerts() output flows through _ingest ────────
def test_collect_rate_alerts_emits_prescored_articles(monkeypatch, tmp_path):
    """The collector's actual returned dicts must carry ``_relevance_score``
    set high enough to clear the 0.5 noise gate. This is the contract that
    pins the FIX — removing the pre-score on the collector side fails here."""
    from collectors import collector_rate_monitor as crm

    # Redirect both DBs to an isolated tmp dir so the test can't touch prod.
    fake_articles = tmp_path / "articles.db"
    fake_seen = tmp_path / "seen_articles.db"
    monkeypatch.setattr(crm, "ARTICLES_DB", fake_articles)
    monkeypatch.setattr(crm, "SEEN_DB", fake_seen)

    # Build a tiny synthetic articles.db with a high-volume source and zero
    # rows in the silent window. The query in ``_load_source_stats`` runs
    # against this DB via ``crm.ARTICLES_DB``.
    conn = sqlite3.connect(str(fake_articles))
    conn.execute("""
        CREATE TABLE articles (
            id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT,
            published TEXT, kw_score REAL, ai_score REAL, urgency INTEGER,
            full_text BLOB, first_seen TEXT, cycle INTEGER,
            time_sensitivity REAL, ml_score REAL, score_source TEXT
        )
    """)
    # 7 days of articles ~10/day = 70 over 7d, baseline ≈ 10/day. Need >=50/day.
    # Insert 400 rows in the past 7d but NOT in the last 3h, so daily_avg = ~57
    # and cnt_window = 0.
    import sqlite3 as sql
    import datetime as dt
    now = dt.datetime(2026, 5, 20, 17, 0, 0, tzinfo=dt.timezone.utc)
    rows = []
    for i in range(400):
        # Spread across 7d, all OLDER than 3h ago.
        offset_h = 4 + (i % (24 * 7 - 4))  # 4h .. ~7d-4h ago
        ts = (now - dt.timedelta(hours=offset_h)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append((
            f"id_{i}", f"http://x/{i}", f"title {i}", "Finnhub/MarketWatch",
            ts, 0.0, 0.0, 0, None, ts, 0, None, None, None,
        ))
    conn.executemany(
        "INSERT INTO articles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows,
    )
    conn.commit()
    conn.close()

    alerts = crm.collect_rate_alerts()

    # The Finnhub/MarketWatch row above had 400 rows over 7d (~57/day) and
    # zero rows in the last 3h → SILENT alert should fire.
    assert len(alerts) >= 1, (
        f"Expected at least 1 SILENT alert for the synthetic dataset; "
        f"got {len(alerts)} alerts: {alerts}"
    )
    for a in alerts:
        assert "_relevance_score" in a, (
            f"Collector output missing _relevance_score (the fix): {a}"
        )
        assert a["_relevance_score"] >= 0.5, (
            f"_relevance_score={a['_relevance_score']} below 0.5 gate — "
            f"would be dropped by daemon._ingest"
        )


# ── E. Invariant #1 (backtest isolation) preserved by the new pre-score path ─
# An attacker / future caller cannot exploit the pre-scoring opt-in to leak a
# synthetic backtest row past read filters — `_LIVE_ONLY_CLAUSE` reads on
# url/source pattern, not kw_score. Pinned by asserting that a pre-scored
# backtest:// row is still excluded from `get_unscored` / `get_unalerted_urgent`.
def test_prescore_path_does_not_break_backtest_isolation(store_factory):
    import daemon as daemon_mod
    store = store_factory()

    # Insert a synthetic backtest row with a high pre-score — simulating a
    # hostile (or careless) future caller.
    leak = {
        "id": "leak-id",
        "link": "backtest://run_999/sim/BUY/NVDA",
        "title": "Backtest leak attempt",
        "summary": "Should never reach live readers",
        "source": "backtest_run_999",
        "first_seen": "2026-05-20T17:00:00Z",
        "_relevance_score": 9.5,  # high enough to escape the noise gate
    }
    daemon_mod._ingest(store, [leak], "collector_monitor")

    # Confirm the row was inserted (pre-score worked)...
    row = store.conn.execute(
        "SELECT COUNT(*) FROM articles WHERE url = ?",
        ("backtest://run_999/sim/BUY/NVDA",),
    ).fetchone()
    assert row[0] == 1, "Pre-scored row should land in DB (insert is fine)"

    # ...but the live readers must NOT surface it. _LIVE_ONLY_CLAUSE is the
    # canonical defense and is what invariant #1 protects.
    unscored = store.get_unscored(limit=100, min_kw=0.0)
    assert not any(u.get("link", "").startswith("backtest://") for u in unscored), (
        "Backtest URL leaked into get_unscored — invariant #1 violation"
    )
    urgent = store.get_unalerted_urgent(limit=100)
    assert not any(u.get("link", "").startswith("backtest://") for u in urgent), (
        "Backtest URL leaked into get_unalerted_urgent — invariant #1 violation"
    )
