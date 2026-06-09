from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from storage.article_store import compress


def _ts(hours_ago: float = 0.1) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _insert(
    store,
    *,
    aid: str,
    title: str,
    source: str = "GN: Nvidia",
    url: str | None = None,
    kw_score: float = 1.0,
    ai_score: float = 0.0,
    ml_score: float | None = None,
    score_source: str | None = None,
    urgency: int = 0,
    first_seen: str | None = None,
    summary: str = "Local ArticleNet summary.",
):
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, "
            " ml_score, score_source, urgency, first_seen, cycle, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                aid,
                url or f"https://example.com/{aid}",
                title,
                source,
                "",
                kw_score,
                ai_score,
                ml_score,
                score_source,
                urgency,
                first_seen or _ts(),
                0,
                compress(summary),
            ),
        )
        store.conn.commit()


def test_search_articles_ranks_by_effective_score_and_reports_source(store):
    _insert(
        store,
        aid="kw",
        title="NVDA keyword-only analyst recap",
        kw_score=9.0,
        summary="Keyword-only but not locally scored.",
    )
    _insert(
        store,
        aid="ml",
        title="NVDA ArticleNet model flags accelerating demand",
        kw_score=1.0,
        ml_score=8.5,
        score_source="ml",
        summary="ArticleNet local model score should be searchable.",
    )
    _insert(
        store,
        aid="llm",
        title="NVDA LLM-vetted earnings catalyst",
        kw_score=1.0,
        ai_score=7.0,
        score_source="llm",
        summary="LLM-vetted row still beats low keyword rows.",
    )

    out = store.search_articles("NVDA", hours=24, limit=3)

    ids = [a["id"] for a in out["articles"]]
    assert ids == ["kw", "ml", "llm"]
    by_id = {a["id"]: a for a in out["articles"]}
    assert by_id["kw"]["effective_score_source"] == "keyword"
    assert by_id["ml"]["effective_score_source"] == "ml"
    assert by_id["ml"]["effective_score"] == pytest.approx(8.5)
    assert by_id["llm"]["effective_score_source"] == "llm"
    assert by_id["llm"]["snippet"]
    assert out["count"] == 3
    assert out["newest_first_seen"] is not None


def test_search_articles_filters_by_ticker_window_and_live_only(store):
    _insert(store, aid="fresh", title="$AMD demand rises", source="GN: AMD")
    _insert(
        store,
        aid="old",
        title="$AMD stale item",
        source="GN: AMD",
        first_seen=_ts(48),
    )
    _insert(
        store,
        aid="bt",
        title="$AMD synthetic backtest row",
        source="backtest_run",
        url="backtest://run/AMD",
    )
    _insert(store, aid="nvda", title="$NVDA unrelated", source="GN: Nvidia")

    out = store.search_articles(ticker="AMD", hours=24, limit=10)

    assert [a["id"] for a in out["articles"]] == ["fresh"]
    assert out["ticker"] == "AMD"
