"""News corroboration — multi-source story confirmation.

The dominant false-positive in the live feed is a wire-recap headline (e.g.
``Why <ticker> Trading Up Today``) hitting ``ai_score`` 9+ with only a single
Google-News aggregator wrapper carrying it. A desk analyst's cheap, model-
independent triage is: how many distinct sources confirm this story?

``build_news_corroboration`` greedily clusters articles by ``ml.dedup``
title-token Jaccard (same primitive the briefing's near-dup-collapse uses),
counts ``DISTINCT source`` per cluster, and returns clusters with
``n_sources >= min_sources`` ranked by corroboration count, then quality,
then freshness.

These tests pin:
  * pure / no-LLM / no-DB
  * clustering matches Jaccard threshold semantics
  * ``min_sources`` filter; default 2 elides single-source noise
  * ranking: corroboration first, then ai_score, then freshness
  * SSOT: imports from ``ml.dedup`` (no re-implemented tokenizer)
  * the route exists, returns JSON, and clamps ``hours`` / ``min_sources``
"""
from __future__ import annotations

import inspect
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.web_server import build_news_corroboration, create_app

NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _art(title, *, source="rss", ai_score=8.0, urgency=0, age_h=0.0,
         url=None):
    return {
        "title": title,
        "source": source,
        "ai_score": ai_score,
        "urgency": urgency,
        "first_seen": (NOW - timedelta(hours=age_h)).isoformat(),
        "url": url or f"https://example.com/{abs(hash(title)) % 1000000}",
    }


# ── pure-builder shape ───────────────────────────────────────────────────────


def test_empty_input_returns_well_formed_envelope():
    out = build_news_corroboration([], now=NOW)
    assert out["n_articles_scanned"] == 0
    assert out["n_clusters_formed"] == 0
    assert out["n_multi_source"] == 0
    assert out["clusters"] == []
    assert out["min_sources"] == 2
    assert "as_of" in out


def test_single_source_cluster_is_filtered_by_default():
    """A solitary wire-recap headline should not surface — the whole point."""
    arts = [
        _art("Why NVDA Trading Up Today", source="GoogleNews/Quiver"),
        _art("Why NVDA trading up today (recap)", source="GoogleNews/Quiver"),
    ]
    out = build_news_corroboration(arts, now=NOW)
    assert out["n_clusters_formed"] == 1
    assert out["n_multi_source"] == 0
    assert out["clusters"] == []


def test_two_source_cluster_surfaces():
    arts = [
        _art("Nvidia beats Q2 earnings, raises guidance", source="reuters"),
        _art("Nvidia beats Q2 earnings raises guidance", source="bloomberg"),
    ]
    out = build_news_corroboration(arts, now=NOW)
    assert out["n_multi_source"] == 1
    cluster = out["clusters"][0]
    assert cluster["n_sources"] == 2
    assert cluster["n_articles"] == 2
    assert set(cluster["sources"]) == {"reuters", "bloomberg"}


def test_clusters_collapse_syndicated_reorderings():
    """Token-set Jaccard must collapse reordered headlines."""
    arts = [
        _art("Apple beats Q2 expectations", source="reuters"),
        _art("Q2 expectations beaten by Apple", source="bloomberg"),
        _art("Apple Q2 expectations beat", source="yahoo"),
    ]
    out = build_news_corroboration(arts, now=NOW)
    assert out["n_multi_source"] == 1
    assert out["clusters"][0]["n_sources"] == 3


def test_distinct_stories_form_distinct_clusters():
    arts = [
        _art("Nvidia beats Q2 earnings", source="reuters"),
        _art("Nvidia beats Q2 earnings", source="bloomberg"),
        _art("Tesla recalls 200000 vehicles", source="reuters"),
        _art("Tesla recalls 200000 vehicles", source="cnbc"),
    ]
    out = build_news_corroboration(arts, now=NOW)
    assert out["n_clusters_formed"] == 2
    assert out["n_multi_source"] == 2


def test_ranking_corroboration_first_then_ai_score():
    """Cluster with more sources wins; ties broken by max_ai_score."""
    arts = [
        _art("Apple smartphone factory expansion", source="reuters", ai_score=9.5),
        _art("Apple smartphone factory expansion", source="bloomberg", ai_score=9.5),
        _art("Tesla vehicle recall lithium pack", source="reuters", ai_score=5.0),
        _art("Tesla vehicle recall lithium pack", source="bloomberg", ai_score=5.0),
        _art("Tesla vehicle recall lithium pack", source="yahoo", ai_score=5.0),
    ]
    out = build_news_corroboration(arts, now=NOW)
    assert [c["n_sources"] for c in out["clusters"]] == [3, 2]
    assert "Tesla" in out["clusters"][0]["headline"]


def test_ranking_ai_score_breaks_corroboration_tie():
    arts = [
        _art("Apple smartphone factory expansion", source="reuters", ai_score=3.0),
        _art("Apple smartphone factory expansion", source="bloomberg", ai_score=3.0),
        _art("Tesla vehicle recall lithium pack", source="reuters", ai_score=9.8),
        _art("Tesla vehicle recall lithium pack", source="bloomberg", ai_score=9.8),
    ]
    out = build_news_corroboration(arts, now=NOW)
    assert [c["max_ai_score"] for c in out["clusters"]] == [9.8, 3.0]
    assert "Tesla" in out["clusters"][0]["headline"]


def test_same_source_repeated_does_not_inflate_n_sources():
    """Two articles from the SAME source on the SAME story is not corroboration."""
    arts = [
        _art("Apple Q2 beat", source="rss"),
        _art("Apple Q2 beat (followup)", source="rss"),
        _art("Apple Q2 beat", source="reuters"),
    ]
    out = build_news_corroboration(arts, now=NOW)
    assert out["n_clusters_formed"] == 1
    assert out["clusters"][0]["n_sources"] == 2


def test_min_sources_param_filters_clusters():
    arts = [
        _art("Nvidia chip demand parabolic guidance", source="s1"),
        _art("Nvidia chip demand parabolic guidance", source="s2"),
        _art("Nvidia chip demand parabolic guidance", source="s3"),
    ]
    assert build_news_corroboration(
        arts, min_sources=4, now=NOW)["n_multi_source"] == 0
    assert build_news_corroboration(
        arts, min_sources=3, now=NOW)["n_multi_source"] == 1


def test_max_urgency_and_latest_first_seen_aggregated():
    arts = [
        _art("Story X", source="reuters", urgency=0, age_h=2.0),
        _art("Story X", source="bloomberg", urgency=2, age_h=0.5),
    ]
    out = build_news_corroboration(arts, now=NOW)
    c = out["clusters"][0]
    assert c["max_urgency"] == 2
    # latest_first_seen is the larger ISO string ⇔ the fresher article
    expected = (NOW - timedelta(hours=0.5)).isoformat()
    assert c["latest_first_seen"] == expected


def test_input_list_not_mutated():
    src = [_art("Story Q", source="reuters"),
           _art("Story Q", source="bloomberg")]
    before = [dict(a) for a in src]
    build_news_corroboration(src, now=NOW)
    assert src == before


def test_empty_title_does_not_break_or_cross_cluster():
    """Articles with no title-tokens form standalone clusters and silently
    fail ``min_sources`` — must not raise."""
    arts = [
        _art("", source="rss"),
        _art("   ", source="reuters"),
        _art("Real headline word here", source="reuters"),
        _art("Real headline word here", source="bloomberg"),
    ]
    out = build_news_corroboration(arts, now=NOW)
    assert out["n_multi_source"] == 1
    assert out["clusters"][0]["n_sources"] == 2


def test_no_llm_no_subprocess_no_network_purity():
    """``build_news_corroboration`` must NEVER call out — that is the whole
    point of having a survives-quota corroboration view alongside the briefing."""
    src = inspect.getsource(build_news_corroboration)
    assert "subprocess" not in src
    assert "claude_call" not in src
    assert "_claude" not in src
    assert "requests" not in src
    assert "urlopen" not in src
    assert "sqlite3" not in src


def test_uses_ml_dedup_ssot():
    """Must compose ``ml.dedup`` primitives — not a re-implemented tokenizer
    that could drift from the briefing's near-dup-collapse semantics."""
    src = inspect.getsource(build_news_corroboration)
    assert "from ml.dedup import" in src
    assert "title_tokens" in src
    assert "jaccard_similarity" in src


# ── route surface ─────────────────────────────────────────────────────────────


def _client():
    app = create_app(store=None)
    app.testing = True
    return app.test_client()


def test_route_returns_json_envelope(monkeypatch):
    # The route reads articles.db via _ro_query; force an empty result so the
    # test doesn't depend on production DB state.
    from dashboard import web_server
    monkeypatch.setattr(web_server, "_ro_query", lambda sql, params=(): [])
    c = _client()
    r = c.get("/api/news-corroboration?hours=6")
    assert r.status_code == 200
    data = r.get_json()
    assert "clusters" in data
    assert data["window_hours"] == 6
    assert data["min_sources"] == 2


def test_route_clamps_hours_window(monkeypatch):
    from dashboard import web_server
    monkeypatch.setattr(web_server, "_ro_query", lambda sql, params=(): [])
    c = _client()
    assert c.get("/api/news-corroboration?hours=0").get_json()["window_hours"] == 1
    assert c.get("/api/news-corroboration?hours=9999").get_json()["window_hours"] == 168
    assert c.get("/api/news-corroboration?hours=abc").get_json()["window_hours"] == 6


def test_route_clamps_min_sources(monkeypatch):
    from dashboard import web_server
    monkeypatch.setattr(web_server, "_ro_query", lambda sql, params=(): [])
    c = _client()
    assert c.get("/api/news-corroboration?min_sources=1"
                 ).get_json()["min_sources"] == 2
    assert c.get("/api/news-corroboration?min_sources=999"
                 ).get_json()["min_sources"] == 10


def test_route_filters_live_only_clause(monkeypatch):
    """The route's SQL must carry the backtest/opus-annotation exclusion."""
    seen = {}

    def _capture(sql, params=()):
        seen["sql"] = sql
        return []

    from dashboard import web_server
    monkeypatch.setattr(web_server, "_ro_query", _capture)
    _client().get("/api/news-corroboration")
    assert "backtest://" in seen["sql"]
    assert "backtest_" in seen["sql"]
    assert "opus_annotation" in seen["sql"]
