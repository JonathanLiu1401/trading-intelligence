"""Event-thread clustering — distinct events ranked by recency-decayed impact.

``/api/news-corroboration`` answers "what's multi-source confirmed?" and filters
single-source events out. ``/api/event-threads`` answers a different question:
"what *distinct events* happened recently, ranked by impact × recency?" — the
trader's primary scrolling read. The pin set is small but load-bearing:

  * pure / no-DB / no-LLM
  * single-article threads SURFACE (the differentiator from corroboration)
  * tickers + sectors are extracted (SSOT: same regex + _SECTOR_MAP that
    sector-pulse uses — drift between the two would scramble routing)
  * ranking is recency-decayed impact, not n_sources: a fresh max=8 thread
    must beat a 12h-stale max=10 thread (otherwise the eye lands on the
    wrong row)
  * Jaccard threshold determines clustering boundary (same primitive)
  * min_score / min_articles filters behave
  * the route exists, returns JSON, clamps its query params
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.web_server import build_event_threads, create_app

NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _art(title, *, source="rss", ai_score=8.0, urgency=0, age_h=0.0, url=None):
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
    out = build_event_threads([], now=NOW)
    assert out["n_articles_scanned"] == 0
    assert out["n_clusters_formed"] == 0
    assert out["n_threads_kept"] == 0
    assert out["threads"] == []
    assert out["min_score"] == 5.0
    assert out["min_articles"] == 1
    assert "as_of" in out


def test_non_list_input_does_not_raise():
    """Total: a None / int / dict input collapses to the empty skeleton."""
    for bad in (None, 0, {"not": "a list"}, "string"):
        out = build_event_threads(bad, now=NOW)  # type: ignore[arg-type]
        assert out["threads"] == []
        assert out["n_articles_scanned"] == 0


def test_titleless_rows_are_skipped_not_clustered():
    arts = [
        {"source": "rss", "ai_score": 9.0},  # no title
        {"title": "", "source": "rss"},       # empty title
        {"title": "   ", "source": "rss"},    # whitespace-only
        _art("Nvidia beats Q2 earnings raises guidance"),  # real one
    ]
    out = build_event_threads(arts, now=NOW)
    assert out["n_threads_kept"] == 1
    assert out["threads"][0]["anchor_title"] == "Nvidia beats Q2 earnings raises guidance"


# ── differentiator: single-article threads must surface ──────────────────────

def test_single_article_thread_surfaces_when_above_min_score():
    """Solo Reuters 8-K before the wire picks it up — corroboration view
    filters this out, event-threads view must KEEP it. The whole point."""
    arts = [_art("MU files 8-K disclosing margin pressure",
                 source="sec_edgar", ai_score=9.5)]
    out = build_event_threads(arts, now=NOW)
    assert out["n_threads_kept"] == 1
    thread = out["threads"][0]
    assert thread["n_articles"] == 1
    assert thread["n_sources"] == 1
    assert thread["max_ai_score"] == 9.5


def test_min_articles_2_drops_solo_threads():
    """Operator can opt into corroboration-style filtering."""
    arts = [
        _art("MU files 8-K disclosing margin pressure", source="sec_edgar"),
        _art("Nvidia raises guidance", source="reuters"),
        _art("Nvidia raises guidance Q2", source="bloomberg"),
    ]
    out = build_event_threads(arts, min_articles=2, now=NOW)
    titles = [t["anchor_title"] for t in out["threads"]]
    assert all("Nvidia" in t for t in titles)
    assert not any("MU files 8-K" in t for t in titles)


# ── ticker + sector enrichment (SSOT with _SECTOR_MAP) ──────────────────────

def test_tickers_extracted_via_word_boundary_uppercase():
    """Same regex sector-pulse uses: case-sensitive, word-bounded.
    'samuel' must not match MU; 'mu' (lowercase) must not match MU."""
    arts = [
        _art("Samuel and amd report no news at MU plant", ai_score=9.0),
    ]
    out = build_event_threads(arts, now=NOW)
    t = out["threads"][0]
    assert "MU" in t["tickers"]
    # lowercase 'amd' must not match (uppercase-only); 'samuel' must not match MU
    assert "AMD" not in t["tickers"]
    assert t["tickers"].count("MU") == 1


def test_tickers_from_distinct_members_are_unioned():
    """A Samsung HBM4 piece mentioning only WDC + a Reuters companion piece
    mentioning only MU should cluster as one thread spanning {MU, WDC} —
    that union IS the trader's exposure surface for the event."""
    arts = [
        _art("Samsung Begins HBM4 Shipments as WDC Strike Threatens AI Chip",
             source="techtimes", ai_score=9.0),
        _art("Samsung HBM4 shipments begin MU strike threatens AI chip",
             source="reuters", ai_score=9.5),
    ]
    out = build_event_threads(arts, now=NOW)
    assert out["n_threads_kept"] == 1
    t = out["threads"][0]
    assert set(t["tickers"]) == {"MU", "WDC"}
    assert "DRAM/Memory" in t["sectors"]


def test_unknown_ticker_does_not_appear_in_sectors():
    """Ticker not in _SECTOR_MAP must not produce a None sector."""
    arts = [_art("XYZ123 mystery company announces breakthrough",
                 ai_score=9.0)]
    out = build_event_threads(arts, now=NOW)
    assert out["threads"][0]["sectors"] == []


# ── ranking: recency-decayed impact, not n_sources ──────────────────────────

def test_fresh_lower_score_beats_stale_higher_score():
    """A 1h-old max=8 thread must outrank a 12h-old max=10 thread.
    Same recency-decay shape as sector-pulse velocity."""
    arts = [
        _art("Stale but big news on Intel chip recall",
             source="reuters", ai_score=10.0, age_h=12.0),
        _art("Fresh but smaller news on AMD product launch",
             source="bloomberg", ai_score=8.0, age_h=1.0),
    ]
    out = build_event_threads(arts, now=NOW)
    assert out["n_threads_kept"] == 2
    # Fresh AMD news (8.0 × 0.5^(1/6) ≈ 7.13) must beat stale Intel (10.0 × 0.5^2 = 2.5)
    assert "AMD" in out["threads"][0]["anchor_title"]
    assert "Intel" in out["threads"][1]["anchor_title"]
    assert out["threads"][0]["impact_score"] > out["threads"][1]["impact_score"]


def test_ranking_breaks_ties_by_n_articles_then_n_sources():
    """Two threads with identical impact_score (same age, same score) —
    deterministic tie-break order."""
    arts = [
        _art("alpha event one happens today",
             source="rss", ai_score=8.0, age_h=1.0),
        # bigger cluster (3 articles, 2 sources)
        _art("beta event happens today this morning",
             source="reuters", ai_score=8.0, age_h=1.0),
        _art("beta event happens today this morning recap",
             source="bloomberg", ai_score=8.0, age_h=1.0),
        _art("beta event happens today this morning update",
             source="reuters", ai_score=8.0, age_h=1.0),
    ]
    out = build_event_threads(arts, now=NOW)
    # beta thread (3 articles, 2 sources) must come before alpha (1 article)
    # despite identical max_ai_score and identical age.
    assert out["threads"][0]["n_articles"] == 3
    assert out["threads"][1]["n_articles"] == 1


# ── score / member-cap filters ───────────────────────────────────────────────

def test_min_score_filter_drops_noise():
    """Default min_score=5.0 elides background-noise articles."""
    arts = [
        _art("Big news high-score", ai_score=9.0),
        _art("Tiny news low-score", ai_score=2.0),
    ]
    out = build_event_threads(arts, now=NOW)
    assert out["n_threads_kept"] == 1
    assert out["threads"][0]["max_ai_score"] == 9.0


def test_min_score_zero_keeps_everything():
    arts = [
        _art("Big news high-score", ai_score=9.0),
        _art("Tiny news low-score", ai_score=2.0, source="other"),
    ]
    out = build_event_threads(arts, min_score=0.0, now=NOW)
    assert out["n_threads_kept"] == 2


def test_member_cap_per_thread():
    """Each thread surfaces at most _EVENT_THREAD_MEMBER_CAP=5 members.
    All members still count toward n_articles."""
    arts = [
        _art(f"NVDA earnings beat raises Q2 guidance update {i}",
             source=f"src{i}", ai_score=8.0 + i * 0.1)
        for i in range(8)
    ]
    out = build_event_threads(arts, now=NOW)
    assert out["n_threads_kept"] == 1
    t = out["threads"][0]
    assert t["n_articles"] == 8
    assert len(t["members"]) == 5
    # Members are highest-score first
    member_scores = [m["ai_score"] for m in t["members"]]
    assert member_scores == sorted(member_scores, reverse=True)


# ── SSOT: clustering primitive comes from ml.dedup, not a local copy ────────

def test_clustering_uses_ml_dedup_primitives():
    """Source proof: the function imports from ml.dedup. Without this, the
    clustering view could silently drift from the briefing's near-dup-collapse
    and the corroboration view's clustering — and the trader would see
    threads split inconsistently across the three reads."""
    import inspect
    from dashboard.web_server import build_event_threads as _f
    src = inspect.getsource(_f)
    assert "from ml.dedup import" in src
    assert "title_tokens" in src
    assert "jaccard_similarity" in src


# ── route: exists, returns JSON, clamps query params ────────────────────────

def test_route_exists_and_returns_json():
    app = create_app()
    client = app.test_client()
    r = client.get("/api/event-threads")
    assert r.status_code == 200
    data = r.get_json()
    assert "threads" in data
    assert "as_of" in data
    assert isinstance(data["threads"], list)


def test_route_clamps_hours_param():
    app = create_app()
    client = app.test_client()
    # 9999 must clamp to 168 (1 week ceiling — match other endpoints)
    r = client.get("/api/event-threads?hours=9999")
    assert r.status_code == 200
    assert r.get_json()["window_hours"] == 168
    # 0 / negative must clamp to 1
    r = client.get("/api/event-threads?hours=-5")
    assert r.status_code == 200
    assert r.get_json()["window_hours"] == 1


def test_route_clamps_min_score_param():
    app = create_app()
    client = app.test_client()
    r = client.get("/api/event-threads?min_score=99")
    assert r.status_code == 200
    assert r.get_json()["min_score"] == 10.0
    r = client.get("/api/event-threads?min_score=-5")
    assert r.status_code == 200
    assert r.get_json()["min_score"] == 0.0


def test_route_tolerates_garbage_params():
    app = create_app()
    client = app.test_client()
    r = client.get(
        "/api/event-threads?hours=abc&min_score=xyz&min_articles=&max_threads=NaN"
    )
    assert r.status_code == 200
    data = r.get_json()
    # Defaults must apply when parsing fails.
    assert data["window_hours"] == 24
    assert data["min_score"] == 5.0
    assert data["min_articles"] == 1
