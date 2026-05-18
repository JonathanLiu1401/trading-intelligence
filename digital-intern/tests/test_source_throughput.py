"""ArticleStore.source_throughput — the leading-indicator companion to
source_freshness: "which collectors are SLOWING DOWN right now", detectable
before a source reads as fully stale.

Pins the operational invariants with specific numbers (not "no crash"):
  * recent-vs-prior counts and decel_pct are computed exactly;
  * synthetic backtest/opus rows are excluded (an injection burst must not
    fake or mask a real collector's rate — CLAUDE.md §5);
  * a source with no prior baseline yields decel_pct=None and sorts LAST so
    it never jumps a real slowdown;
  * sources idle in both windows are omitted (no signal);
  * ordering is most-decelerated-first.
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


def test_recent_vs_prior_counts_decel_and_ordering(store):
    # window=60: recent = [now-60, now]; prior = [now-120, now-60).
    # Decelerating source: 1 recent vs 5 prior → decel_pct = (5-1)/5*100 = 80.0
    _insert(store, id="d0", url="https://d/0", title="d recent",
            source="decel", first_seen=_iso(5))
    for i, m in enumerate((65, 75, 85, 95, 110)):
        _insert(store, id=f"dp{i}", url=f"https://d/p{i}", title="d prior",
                source="decel", first_seen=_iso(m))
    # Accelerating source: 6 recent vs 2 prior → decel_pct = (2-6)/2*100 = -200.0
    for i, m in enumerate((3, 8, 12, 20, 35, 50)):
        _insert(store, id=f"ar{i}", url=f"https://a/r{i}", title="a recent",
                source="accel", first_seen=_iso(m))
    for i, m in enumerate((70, 90)):
        _insert(store, id=f"ap{i}", url=f"https://a/p{i}", title="a prior",
                source="accel", first_seen=_iso(m))
    # Brand-new source: 3 recent, 0 prior → decel_pct = None (no baseline).
    for i, m in enumerate((4, 9, 14)):
        _insert(store, id=f"nr{i}", url=f"https://n/r{i}", title="n recent",
                source="newsrc", first_seen=_iso(m))
    # Idle in BOTH windows (200min ago) → must be omitted entirely.
    _insert(store, id="old1", url="https://o/1", title="stale",
            source="oldsrc", first_seen=_iso(200))
    # Synthetic rows must NOT be counted (would fake a live source's rate).
    _insert(store, id="b1", url="backtest://run_3/2026/BUY/AAPL",
            title="bt", source="backtest_run_3_winner", first_seen=_iso(2))
    _insert(store, id="o1", url="https://x/o1", title="opus",
            source="opus_annotation_cycle_1", first_seen=_iso(2))

    rows = store.source_throughput(window_min=60)
    by_src = {r["source"]: r for r in rows}

    assert set(by_src) == {"decel", "accel", "newsrc"}, (
        "synthetic, idle-in-both-windows sources must be excluded"
    )
    assert by_src["decel"] == {"source": "decel", "recent": 1, "prior": 5,
                               "delta": -4, "decel_pct": 80.0}
    assert by_src["accel"]["recent"] == 6 and by_src["accel"]["prior"] == 2
    assert by_src["accel"]["delta"] == 4
    assert by_src["accel"]["decel_pct"] == -200.0
    assert by_src["newsrc"]["recent"] == 3 and by_src["newsrc"]["prior"] == 0
    assert by_src["newsrc"]["decel_pct"] is None

    # Most-decelerated first; the no-baseline source sorts last.
    assert rows[0]["source"] == "decel"
    assert rows[-1]["source"] == "newsrc"


def test_empty_db_returns_empty_list(store):
    assert store.source_throughput() == []


def test_window_size_is_respected(store):
    # With window=30: a row 45min ago falls in the PRIOR window [60,30)... no:
    # recent=[now-30,now], prior=[now-60,now-30). 45min ago → prior.
    _insert(store, id="x1", url="https://x/1", title="recent",
            source="s", first_seen=_iso(10))
    _insert(store, id="x2", url="https://x/2", title="prior",
            source="s", first_seen=_iso(45))
    rows = store.source_throughput(window_min=30)
    s = {r["source"]: r for r in rows}["s"]
    assert s["recent"] == 1 and s["prior"] == 1 and s["delta"] == 0
    assert s["decel_pct"] == 0.0
