"""``ArticleStore.prefloor_pseudo_articles`` — single source of truth for the
ML-path quote-widget / recap-template pre-floor.

The helper is called by both ``ArticleStore.score_pending`` (in-store driver)
and ``daemon.scorer_worker`` (production worker loop). Before extraction, the
pre-floor lived only in ``score_pending``, and the daemon's ``scorer_worker``
silently bypassed it — ML-confident urgent pseudo-articles reached
``urgency=2`` and only got caught by the alert-path gates (after polluting
``urgent_queue_health`` and burning alert worker cycles).

A 48h live audit (2026-05-26) found 10 such bypass cases (price-glue and recap
templates like ``Here's What ...``, ``KLA (KLAC) Is Up 7.5% After ...``,
``Nvidia stock continues to struggle after earnings``). This test pins the
helper's contract so a future refactor cannot silently regress the parity.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert(store, *, id, url, title, source, kw_score=1.0):
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, 0.0, 0,
             _recent_iso(), 0, None, None),
        )
        store.conn.commit()


def _row(store, aid):
    return store.conn.execute(
        "SELECT ai_score, ml_score, score_source, urgency FROM articles WHERE id=?",
        (aid,),
    ).fetchone()


def test_quote_widget_pre_floored(store):
    """A spaceless ticker-tape pseudo-article gets ml_score=0.01 and never
    reaches scoring. The exact title from the live audit."""
    _insert(store, id="qw1", url="https://example/qw",
            # Letter-glued-to-decimal-price — the canonical quote-widget tape.
            title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            source="scraped/finance.yahoo.com")
    _insert(store, id="real1", url="https://reuters.com/x",
            title="Nvidia beats Q1 estimates on AI demand surge",
            source="reuters")

    batch = [
        {"_id": "qw1", "title": "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
         "source": "scraped/finance.yahoo.com", "link": "https://example/qw"},
        {"_id": "real1", "title": "Nvidia beats Q1 estimates on AI demand surge",
         "source": "reuters", "link": "https://reuters.com/x"},
    ]
    real, n_pre = store.prefloor_pseudo_articles(batch)

    assert n_pre == 1
    assert [a["_id"] for a in real] == ["real1"]
    # Quote-widget row got the pre-floor write
    ai, ml, src, urg = _row(store, "qw1")
    assert ai == 0
    assert ml == pytest.approx(0.01)
    assert src == "ml"
    assert urg == 0
    # Real row untouched
    ai, ml, src, urg = _row(store, "real1")
    assert ai == 0
    assert ml is None
    assert src is None


def test_recap_template_pre_floored(store):
    """A ``KLA (KLAC) Is Up 7.5% After ...`` recap template — the exact
    fingerprint that bypassed the daemon scorer in the 48h live audit."""
    _insert(store, id="recap1", url="https://example/recap",
            title="KLA (KLAC) Is Up 7.5% After Stock Split, Buyback, Dividend Hike and Earnings Beat",
            source="GoogleNews/simplywall.st")

    batch = [{
        "_id": "recap1",
        "title": "KLA (KLAC) Is Up 7.5% After Stock Split, Buyback, Dividend Hike and Earnings Beat",
        "source": "GoogleNews/simplywall.st",
        "link": "https://example/recap",
    }]
    real, n_pre = store.prefloor_pseudo_articles(batch)
    assert n_pre == 1
    assert real == []
    ai, ml, src, urg = _row(store, "recap1")
    assert ml == pytest.approx(0.01)
    assert src == "ml"
    assert urg == 0


def test_real_breaking_article_passes_through(store):
    """A real wire headline must NOT be pre-floored — defense must be tight
    enough that ground-truth breaking news always reaches inference."""
    _insert(store, id="real", url="https://reuters.com/y",
            title="Fed surprises markets with emergency 50bp rate cut",
            source="reuters")
    batch = [{
        "_id": "real",
        "title": "Fed surprises markets with emergency 50bp rate cut",
        "source": "reuters",
        "link": "https://reuters.com/y",
    }]
    real, n_pre = store.prefloor_pseudo_articles(batch)
    assert n_pre == 0
    assert [a["_id"] for a in real] == ["real"]
    # No DB write for the real row
    ai, ml, src, urg = _row(store, "real")
    assert ml is None
    assert src is None


def test_articles_without_id_skipped(store):
    """An article missing ``_id`` is silently dropped from BOTH real and
    pre-floor outputs — there's nothing to write to, so dropping prevents
    a downstream NULL-id update."""
    batch = [
        {"title": "no id here", "source": "rss", "link": "https://x/y"},
    ]
    real, n_pre = store.prefloor_pseudo_articles(batch)
    assert n_pre == 0
    assert real == []


def test_empty_batch_noop(store):
    """An empty batch must not write anything."""
    real, n_pre = store.prefloor_pseudo_articles([])
    assert n_pre == 0
    assert real == []


def test_mixed_batch_partitions_correctly(store):
    """The canonical mixed case: real + quote-widget + recap — only the
    pseudo rows get the floor, the real row passes through, and the
    counter reflects pseudo-only count."""
    # Insert three rows so the writes can land somewhere
    _insert(store, id="r", url="https://reuters.com/r",
            title="Nvidia beats Q1 estimates", source="reuters")
    _insert(store, id="q", url="https://example/q",
            title="MUMicron Technology88.43-2.10(-2.32%)", source="scraped/yahoo")
    _insert(store, id="rec", url="https://example/rec",
            title="Why MU Stock Is Trading Down Today",
            source="yfinance/Motley Fool")

    batch = [
        {"_id": "r", "title": "Nvidia beats Q1 estimates",
         "source": "reuters", "link": "https://reuters.com/r"},
        {"_id": "q", "title": "MUMicron Technology88.43-2.10(-2.32%)",
         "source": "scraped/yahoo", "link": "https://example/q"},
        {"_id": "rec", "title": "Why MU Stock Is Trading Down Today",
         "source": "yfinance/Motley Fool", "link": "https://example/rec"},
    ]
    real, n_pre = store.prefloor_pseudo_articles(batch)
    assert n_pre == 2
    assert [a["_id"] for a in real] == ["r"]
    # Both pseudo rows pre-floored
    _, ml_q, src_q, _ = _row(store, "q")
    assert ml_q == pytest.approx(0.01)
    assert src_q == "ml"
    _, ml_rec, src_rec, _ = _row(store, "rec")
    assert ml_rec == pytest.approx(0.01)
    assert src_rec == "ml"
    # Real row untouched
    _, ml_r, src_r, _ = _row(store, "r")
    assert ml_r is None
    assert src_r is None


def test_scorer_worker_calls_helper():
    """Daemon's ``scorer_worker`` must invoke ``prefloor_pseudo_articles`` so
    the production worker path inherits the same defense the in-store
    ``score_pending`` driver has. Source-grep guard (same shape as the
    existing recap-template four-surface lockstep checks); prevents a future
    refactor from silently dropping the call back to the
    pre-extraction-bug state.
    """
    import inspect
    import daemon

    src = inspect.getsource(daemon.scorer_worker)
    assert "prefloor_pseudo_articles" in src, (
        "daemon.scorer_worker must call store.prefloor_pseudo_articles before "
        "score_articles — otherwise ML-confident quote-widget / recap-template "
        "pseudo-articles bypass the pre-floor and reach urgency=1 / urgency=2 "
        "in production (10 such cases in a 48h live audit prior to extraction)."
    )


def test_pseudo_with_existing_urgency_keeps_max(store):
    """``update_ml_scores_batch`` uses ``urgency=MAX(urgency, ?)``. The
    pre-floor passes 0 for urgency, so an existing higher urgency on the row
    (e.g. left over from a prior LLM call) must be preserved — the pre-floor
    only neutralises the SCORING signal, not a separately-asserted urgency.

    Mirrors the ``update_ml_scores_batch`` test's MAX(urgency,?) discipline."""
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("q2", "https://example/q2",
             "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
             "scraped/yahoo", "", 1.0, 0.0, 2,
             _recent_iso(), 0, None, None),
        )
        store.conn.commit()

    batch = [{
        "_id": "q2",
        "title": "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
        "source": "scraped/yahoo",
        "link": "https://example/q2",
    }]
    real, n_pre = store.prefloor_pseudo_articles(batch)
    assert n_pre == 1
    assert real == []
    ai, ml, src, urg = _row(store, "q2")
    assert ml == pytest.approx(0.01)
    # urgency stays at 2 — MAX(2, 0) = 2
    assert urg == 2
