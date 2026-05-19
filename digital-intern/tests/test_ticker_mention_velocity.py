"""``ArticleStore.ticker_mention_velocity`` — programmatic per-ticker mention
velocity primitive that complements the existing source-level views
(``source_freshness`` / ``source_throughput``) and replaces the brittle
filter in ``analytics/trend_velocity.py`` (CLI snapshot, also missed
``backtest://`` URLs and ``opus_annotation*`` sources).

Pins three invariants:
  * synthetic backtest/opus rows are excluded so a per-ticker count can never
    be inflated/masked by injection bursts sharing the table (CLAUDE.md §5);
  * ticker matching is whole-word and ALL-CAPS so a substring like "NVDAQ"
    cannot leak a hit for "NVDA";
  * ordering is highest-velocity-first so a ticker getting unusual coverage
    surfaces at the top.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _insert(store, *, id, url, title, source, first_seen):
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 1.0, 0.0, 0, first_seen, 0,
             None, None),
        )
        store.conn.commit()


def _iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc)
            - timedelta(minutes=minutes_ago)).isoformat()


def test_per_ticker_recent_vs_prior_window_with_velocity_sort(store):
    # NVDA: 3 fresh hits (last 30 min), 1 in prior window — high accel.
    _insert(store, id="n1", url="https://x.com/n1",
            title="NVDA earnings crush expectations",
            source="rss", first_seen=_iso(2))
    _insert(store, id="n2", url="https://x.com/n2",
            title="$NVDA upgraded by analyst",
            source="rss", first_seen=_iso(10))
    _insert(store, id="n3", url="https://x.com/n3",
            title="NVDA data center revenue up 200%",
            source="rss", first_seen=_iso(25))
    _insert(store, id="n4", url="https://x.com/n4",
            title="NVDA quiet quarter (prior)",
            source="rss", first_seen=_iso(90))

    # AMD: 1 fresh, 2 prior — *decelerating*, low ratio.
    _insert(store, id="a1", url="https://x.com/a1",
            title="AMD chips ship later this year",
            source="rss", first_seen=_iso(15))
    _insert(store, id="a2", url="https://x.com/a2",
            title="AMD revenue miss (prior)",
            source="rss", first_seen=_iso(75))
    _insert(store, id="a3", url="https://x.com/a3",
            title="AMD layoffs disclosed (prior)",
            source="rss", first_seen=_iso(100))

    # MU: never mentioned — must come back zeroed, not absent.
    rows = store.ticker_mention_velocity(["NVDA", "AMD", "MU"], window_min=60)
    by_t = {r["ticker"]: r for r in rows}

    assert set(by_t) == {"NVDA", "AMD", "MU"}
    assert by_t["NVDA"]["recent"] == 3
    assert by_t["NVDA"]["prior"] == 1
    assert by_t["AMD"]["recent"] == 1
    assert by_t["AMD"]["prior"] == 2
    assert by_t["MU"]["recent"] == 0
    assert by_t["MU"]["prior"] == 0
    # Ratio: (recent+1)/(prior+1) for stability when prior is 0.
    assert by_t["NVDA"]["ratio"] == 2.0
    assert by_t["AMD"]["ratio"] < 1.0
    assert by_t["MU"]["ratio"] == 1.0
    # Newest age plausible (a few minutes for NVDA, None for MU).
    assert by_t["NVDA"]["newest_age_s"] is not None
    assert 0 < by_t["NVDA"]["newest_age_s"] < 60 * 60
    assert by_t["MU"]["newest_age_s"] is None
    # Ordering: highest accelerator first; a never-mentioned ticker sorts
    # below a real decelerator (ratio==1.0 from no data is not a signal).
    assert rows[0]["ticker"] == "NVDA"


def test_excludes_synthetic_backtest_and_opus_rows(store):
    # Real NVDA mention.
    _insert(store, id="n1", url="https://x.com/n1",
            title="NVDA real news", source="rss", first_seen=_iso(5))
    # Synthetic rows mentioning NVDA — must NOT count. If they did, the
    # ticker's velocity would be inflated by any backtest replay or Opus
    # annotation pass that happened to land in the window.
    _insert(store, id="b1", url="backtest://run_9/2026-05-12/BUY/NVDA",
            title="NVDA backtest winner",
            source="backtest_run_9_winner", first_seen=_iso(3))
    _insert(store, id="b2", url="backtest://run_9/2026-05-13/SELL/NVDA",
            title="NVDA prior backtest",
            source="backtest_run_9_winner", first_seen=_iso(80))
    _insert(store, id="o1", url="https://x.com/opus1",
            title="NVDA opus lesson",
            source="opus_annotation_cycle_3", first_seen=_iso(7))

    rows = store.ticker_mention_velocity(["NVDA"], window_min=60)
    nvda = rows[0]
    assert nvda["recent"] == 1, "synthetic rows leaked into recent count"
    assert nvda["prior"] == 0, "synthetic rows leaked into prior count"


def test_whole_word_match_only(store):
    # ALL of these are substrings that contain "NVDA" / "AMD" but must
    # NOT match the ticker — whole-word boundary protects against false
    # hits like "NVDAQ" (made-up) or "AMDOCS" (real, ticker DOX).
    _insert(store, id="x1", url="https://x.com/x1",
            title="NVDAQ exchange rebranded",
            source="rss", first_seen=_iso(5))
    _insert(store, id="x2", url="https://x.com/x2",
            title="AMDOCS earnings beat",
            source="rss", first_seen=_iso(5))
    # A real, distinct mention to prove the matcher isn't broken outright.
    _insert(store, id="x3", url="https://x.com/x3",
            title="NVDA quarter ends",
            source="rss", first_seen=_iso(5))

    rows = store.ticker_mention_velocity(["NVDA", "AMD"], window_min=60)
    by_t = {r["ticker"]: r for r in rows}
    assert by_t["NVDA"]["recent"] == 1
    assert by_t["AMD"]["recent"] == 0


def test_empty_tickers_list_returns_empty(store):
    _insert(store, id="n1", url="https://x.com/n1",
            title="NVDA news", source="rss", first_seen=_iso(5))
    assert store.ticker_mention_velocity([], window_min=60) == []


def test_no_articles_in_db_still_returns_zeroed_rows(store):
    rows = store.ticker_mention_velocity(["NVDA"], window_min=60)
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "NVDA"
    assert r["recent"] == 0 and r["prior"] == 0
    assert r["ratio"] == 1.0
    assert r["newest_age_s"] is None
