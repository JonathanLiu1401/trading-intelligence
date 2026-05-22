"""``ArticleStore.urgent_queue_health`` — the unalerted-urgent backlog view.

A ``urgency=1`` row is "scored urgent, not yet pushed". Once its ``first_seen``
ages past the 24h window ``get_unalerted_urgent`` enforces, the alert worker
can never see it and ``reap_stale_urgent`` demotes it — the push is silently
lost. ``urgent_queue_health`` surfaces that backlog *before* it is lost.

These tests assert specific counts/ages, not "no crash", and pin the
load-bearing invariant that backtest/opus rows never inflate the backlog.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _seed(store, *, aid, urgency=1, first_seen, title="Generic headline here",
          url=None, source="rss"):
    """Insert a row directly so tests can build any urgency/age state."""
    if url is None:
        url = f"https://example.com/{aid}"
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, url, title, source, "", 1.0, 0.0, urgency,
             first_seen, 0, 9.0, "ml"),
        )
        store.conn.commit()


def test_empty_queue_reports_zero(store):
    h = store.urgent_queue_health()
    assert h["queued"] == 0
    assert h["oldest_age_h"] is None
    assert h["near_reap"] == 0
    assert h["overdue"] == 0
    assert h["by_ticker"] == []


def test_counts_only_urgency_one(store):
    """urgency=0 (normal) and urgency=2 (already alerted) are NOT a backlog."""
    _seed(store, aid="q1", urgency=1, first_seen=_iso(2))
    _seed(store, aid="q2", urgency=1, first_seen=_iso(4))
    _seed(store, aid="normal", urgency=0, first_seen=_iso(1))
    _seed(store, aid="alerted", urgency=2, first_seen=_iso(1))

    h = store.urgent_queue_health()
    assert h["queued"] == 2


def test_oldest_age_reflects_oldest_row(store):
    _seed(store, aid="fresh", urgency=1, first_seen=_iso(1))
    _seed(store, aid="old", urgency=1, first_seen=_iso(12))
    _seed(store, aid="mid", urgency=1, first_seen=_iso(5))

    h = store.urgent_queue_health()
    # Oldest is the 12h-old row (allow a little wall-clock slack).
    assert 11.9 <= h["oldest_age_h"] <= 12.2


def test_near_reap_classifies_rows_close_to_deadline(store):
    """reap_age=24h, near_reap=3h → a row aged 22h is within 3h of the
    24h deadline (near_cut=21) → counted near_reap, not overdue."""
    _seed(store, aid="safe", urgency=1, first_seen=_iso(2))     # nowhere near
    _seed(store, aid="near", urgency=1, first_seen=_iso(22))    # 21<=22<24
    _seed(store, aid="edge", urgency=1, first_seen=_iso(21.5))  # 21<=21.5<24

    h = store.urgent_queue_health(reap_age_hours=24, near_reap_hours=3.0)
    assert h["near_reap"] == 2
    assert h["overdue"] == 0


def test_overdue_rows_counted_separately(store):
    """A row aged past reap_age_hours is overdue (push already lost) and is
    NOT double-counted as near_reap."""
    _seed(store, aid="lost1", urgency=1, first_seen=_iso(26))
    _seed(store, aid="lost2", urgency=1, first_seen=_iso(40))
    _seed(store, aid="near", urgency=1, first_seen=_iso(22))

    h = store.urgent_queue_health(reap_age_hours=24, near_reap_hours=3.0)
    assert h["overdue"] == 2
    assert h["near_reap"] == 1
    assert h["queued"] == 3


def test_backtest_rows_never_inflate_backlog(store):
    """Critical invariant: a synthetic urgency=1 row must not count — it is
    training data, not a missed live alert."""
    _seed(store, aid="live", urgency=1, first_seen=_iso(3))
    _seed(store, aid="bt_url", urgency=1, first_seen=_iso(3),
          url="backtest://run_1/2026-01-01/BUY/MU", source="rss")
    _seed(store, aid="bt_src", urgency=1, first_seen=_iso(3),
          source="backtest_run_42_winner")
    _seed(store, aid="opus", urgency=1, first_seen=_iso(3),
          source="opus_annotation_cycle_3")

    h = store.urgent_queue_health()
    assert h["queued"] == 1, "synthetic rows leaked into the urgent backlog"


def test_per_ticker_breakdown_for_held_names(store):
    _seed(store, aid="nv", urgency=1, first_seen=_iso(22),
          title="NVDA guidance shock hits the wire")
    _seed(store, aid="mu", urgency=1, first_seen=_iso(5),
          title="MU memory pricing update")
    _seed(store, aid="other", urgency=1, first_seen=_iso(2),
          title="Unrelated macro story with no held ticker")

    h = store.urgent_queue_health(tickers=["NVDA", "MU", "ORCL"],
                                  reap_age_hours=24, near_reap_hours=3.0)
    by = {r["ticker"]: r for r in h["by_ticker"]}
    # ORCL has no queued urgent row → omitted entirely.
    assert set(by) == {"NVDA", "MU"}
    assert by["NVDA"]["queued"] == 1
    assert by["NVDA"]["near_reap"] == 1   # 22h old, past the 21h near-cut
    assert by["MU"]["queued"] == 1
    assert by["MU"]["near_reap"] == 0     # 5h old — nowhere near the deadline


def test_per_ticker_sorted_worst_oldest_first(store):
    _seed(store, aid="mu", urgency=1, first_seen=_iso(4),
          title="MU minor update")
    _seed(store, aid="nv", urgency=1, first_seen=_iso(18),
          title="NVDA major break")

    h = store.urgent_queue_health(tickers=["MU", "NVDA"])
    # NVDA (18h) must sort before MU (4h) — closest to a silent drop first.
    assert [r["ticker"] for r in h["by_ticker"]] == ["NVDA", "MU"]


def test_backtest_row_does_not_inflate_per_ticker(store):
    """A synthetic row mentioning a held ticker must not appear in by_ticker."""
    _seed(store, aid="live", urgency=1, first_seen=_iso(3),
          title="NVDA real headline")
    _seed(store, aid="bt", urgency=1, first_seen=_iso(3),
          title="NVDA synthetic backtest row",
          url="backtest://run_1/2026-01-01/BUY/NVDA")

    h = store.urgent_queue_health(tickers=["NVDA"])
    by = {r["ticker"]: r for r in h["by_ticker"]}
    assert by["NVDA"]["queued"] == 1


def test_substring_ticker_does_not_false_match(store):
    """Whole-word matching: 'MU' must not match inside 'MUST' or 'ALUMINUM'."""
    _seed(store, aid="x", urgency=1, first_seen=_iso(2),
          title="Aluminum demand MUST recover, analysts say")

    h = store.urgent_queue_health(tickers=["MU"])
    assert h["by_ticker"] == []
