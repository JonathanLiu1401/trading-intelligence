"""Backtest isolation on the `analytics/source_diversity.py` CLI snapshot.

`source_diversity.py` writes `/home/zeph/logs/source_diversity.json` — an
analyst-facing per-ticker outlet-breadth report. Like the dashboard parity
class pinned in `test_dashboard_backtest_isolation.py`, this is a non-store
live-facing surface whose `FROM articles` read MUST apply the canonical
`_LIVE_ONLY_CLAUSE`. The shipped query carries only `source NOT LIKE
'backtest_run_%'` — which catches synthetic BUY/SELL injection rows but lets
`opus_annotation*` source rows leak through, inflating both the
`distinct_sources` count and the per-ticker mention totals on a held name
(an `opus_annotation_cycle_3` lesson titled "[Cycle 3] Good buy on NVDA"
would otherwise show as another outlet covering NVDA).

The fix mirrors `analytics/publish_lag_audit.py` / `stale_source_alerter.py` /
`ticker_concentration.py`: import `_LIVE_ONLY_CLAUSE` from
`storage.article_store` and inline it. This test pins the filter so future
edits cannot silently reopen the leak.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone


def _seed(db_path):
    """Live row + the three synthetic shapes, all mentioning NVDA + MU in the
    title so a missing filter inflates BOTH ticker counts."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT,
            published TEXT, kw_score REAL, ai_score REAL, urgency INTEGER,
            first_seen TEXT, cycle INTEGER, ml_score REAL, score_source TEXT
        )"""
    )
    now = datetime.now(timezone.utc) - timedelta(minutes=5)
    rows = [
        # Live row — the only one that should be counted.
        ("live", "https://reuters.com/x",
         "NVDA MU semis rally on AI demand",
         "rss", now.isoformat()),
        # Synthetic backtest URL — backtest_run_ source catches it under the
        # buggy filter too, but defense-in-depth: the URL itself is the canon.
        ("bt_url", "backtest://run_1/2026-01-01/BUY/NVDA",
         "NVDA MU backtest BUY winner synthetic",
         "backtest_run_1_winner", now.isoformat()),
        # The actual leak — opus_annotation source slips through the partial
        # `backtest_run_%` filter and inflates the diversity / mention counts.
        ("opus", "https://example.com/opus",
         "NVDA MU opus annotation GOOD trade lesson",
         "opus_annotation_cycle_3", now.isoformat()),
    ]
    conn.executemany(
        "INSERT INTO articles "
        "(id, url, title, source, published, kw_score, ai_score, urgency, "
        "first_seen, cycle, ml_score, score_source) "
        "VALUES (?,?,?,?,'',1.0,5.0,0,?,0,NULL,NULL)",
        [(r[0], r[1], r[2], r[3], r[4]) for r in rows],
    )
    conn.commit()
    conn.close()


def _run(tmp_path, monkeypatch):
    from analytics import source_diversity

    db = tmp_path / "articles.db"
    _seed(db)
    out = tmp_path / "source_diversity.json"
    monkeypatch.setattr(source_diversity, "DB_PATH", db)
    monkeypatch.setattr(source_diversity, "OUT_PATH", out)
    rc = source_diversity.main()
    assert rc == 0, "main() reported no in-window rows — fixture is broken"
    return json.loads(out.read_text())


def test_synthetic_rows_excluded_from_source_diversity(tmp_path, monkeypatch):
    """The shipped query was `source NOT LIKE 'backtest_run_%'`; an
    `opus_annotation*` row leaked through. The canonical clause closes both
    leaks (backtest:// URL + opus_annotation source)."""
    report = _run(tmp_path, monkeypatch)

    tickers = {r["ticker"]: r for r in report["top_diverse"]}
    # NVDA and MU should both appear with EXACTLY the live row's contribution.
    for sym in ("NVDA", "MU"):
        assert sym in tickers, f"{sym} missing from report"
        rec = tickers[sym]
        assert rec["mentions"] == 1, (
            f"{sym} mentions={rec['mentions']} — synthetic row leaked "
            f"(expected exactly 1 from the live rss row only)"
        )
        assert rec["distinct_sources"] == 1, (
            f"{sym} distinct_sources={rec['distinct_sources']} — synthetic "
            f"source(s) leaked as fake outlets"
        )
        assert rec["top_source"] == "rss", (
            f"{sym} top_source={rec['top_source']!r} — must be the live `rss` "
            f"row, not a synthetic source key"
        )


def test_uses_canonical_live_only_clause(tmp_path, monkeypatch):
    """Defense-in-depth: a future edit that hand-rolls a partial filter
    (excluding only `backtest_run_%` again) would leak `opus_annotation*` and
    `backtest://` URLs whose source carried an unexpected tag. Pin the
    canonical fragment so any drift fails here."""
    from analytics import source_diversity
    from storage.article_store import _LIVE_ONLY_CLAUSE

    # Read the module's SQL via a probe — the simplest discriminator. Patch
    # sqlite3.connect to capture the SQL the script actually executes.
    captured: list[str] = []
    real_connect = sqlite3.connect

    class _Spy:
        def __init__(self, real):
            self._real = real

        def execute(self, sql, params=()):
            captured.append(sql)
            return self._real.execute(sql, params)

        def close(self):
            return self._real.close()

    def fake_connect(*a, **kw):
        return _Spy(real_connect(*a, **kw))

    db = tmp_path / "empty.db"
    # Create the bare schema so the SELECT runs (empty result → rc=1 fine).
    bare = sqlite3.connect(str(db))
    bare.execute(
        "CREATE TABLE articles (id TEXT PRIMARY KEY, url TEXT, title TEXT, "
        "source TEXT, first_seen TEXT)"
    )
    bare.close()
    monkeypatch.setattr(source_diversity, "DB_PATH", db)
    monkeypatch.setattr(source_diversity, "OUT_PATH",
                        tmp_path / "out.json")
    monkeypatch.setattr(sqlite3, "connect", fake_connect)
    try:
        source_diversity.main()  # rc=1 on empty is fine — we only want the SQL
    except (SystemExit, sqlite3.OperationalError):
        pass

    articles_sql = [s for s in captured if "FROM articles" in s]
    assert articles_sql, "source_diversity ran no `FROM articles` query"
    sql = articles_sql[0]
    # The three canonical fragments must all be present (the test would have
    # failed against the shipped `source NOT LIKE 'backtest_run_%'`-only form).
    for fragment in (
        "url NOT LIKE 'backtest://%'",
        "source NOT LIKE 'backtest_%'",
        "source NOT LIKE 'opus_annotation%'",
    ):
        assert fragment in sql, (
            f"source_diversity SQL is missing canonical fragment {fragment!r}; "
            f"actual:\n{sql}\nExpected the full `_LIVE_ONLY_CLAUSE`:\n"
            f"{_LIVE_ONLY_CLAUSE}"
        )
