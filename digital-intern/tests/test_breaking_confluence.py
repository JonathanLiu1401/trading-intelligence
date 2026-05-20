"""Breaking confluence — NOW-focused velocity view of multi-source clusters.

Sibling to ``/api/news-corroboration`` (whole-window trust filter ranked by
n_sources) and ``/api/event-threads`` (24h recency-decayed impact). This is
the tight-window NOW view with verdict ladder
(CONFIRMED / EMERGING / SINGLETON_HOT) and arrival velocity.

These tests pin:
  * pure / no-LLM / no-DB
  * window pre-filter drops stale articles before clustering
  * min_score pre-filter drops kw-only rows
  * Jaccard clustering matches news_corroboration semantics (SSOT)
  * verdict ladder:
      - n_sources ≥ 3 ⇒ CONFIRMED
      - n_sources == 2 AND latest within emerging window ⇒ EMERGING
      - n_sources == 2 AND latest stale ⇒ CONFIRMED
      - n_sources == 1 + hot (urg ≥ 1 AND score ≥ 9 AND fresh) ⇒ SINGLETON_HOT
      - n_sources == 1 + cold ⇒ filtered
  * velocity_per_30min math
  * ranking: verdict → recency → n_sources → max_score
  * route exists, returns JSON, clamps params
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.web_server import build_breaking_confluence, create_app

NOW = datetime(2026, 5, 18, 14, 0, 0, tzinfo=timezone.utc)


def _art(title, *, source="rss", ai_score=8.0, urgency=0, age_min=5.0,
         url=None):
    return {
        "title": title,
        "source": source,
        "ai_score": ai_score,
        "urgency": urgency,
        "first_seen": (NOW - timedelta(minutes=age_min)).isoformat(),
        "url": url or f"https://example.com/{abs(hash(title)) % 1000000}",
    }


# ── pure-builder shape ─────────────────────────────────────────────────────


def test_empty_input_returns_well_formed_envelope():
    out = build_breaking_confluence([], now=NOW)
    assert out["n_articles_scanned"] == 0
    assert out["n_articles_in_window"] == 0
    assert out["n_clusters_formed"] == 0
    assert out["n_surfaced"] == 0
    assert out["clusters"] == []
    assert out["window_minutes"] == 60
    assert "as_of" in out
    assert out["counts_by_verdict"] == {
        "CONFIRMED": 0, "EMERGING": 0, "SINGLETON_HOT": 0,
    }


# ── window pre-filter ──────────────────────────────────────────────────────


def test_articles_outside_window_are_dropped():
    arts = [
        _art("AAPL beats Q2", source="reuters", age_min=5.0),
        _art("AAPL beats Q2 results", source="bloomberg", age_min=5.0),
        # 3-hour-old; outside the 60-minute default window.
        _art("AAPL beats Q2 earnings", source="cnbc", age_min=180.0),
    ]
    out = build_breaking_confluence(arts, now=NOW)
    assert out["n_articles_scanned"] == 3
    assert out["n_articles_in_window"] == 2
    assert len(out["clusters"]) == 1
    assert out["clusters"][0]["n_sources"] == 2


def test_min_score_filter_drops_low_score_rows():
    arts = [
        _art("NVDA earnings beat", source="reuters", ai_score=6.0),
        _art("NVDA earnings beat report", source="bloomberg", ai_score=2.0),
    ]
    out = build_breaking_confluence(arts, now=NOW, min_score=5.0)
    assert out["n_articles_in_window"] == 1
    assert out["clusters"] == []


# ── verdict ladder ─────────────────────────────────────────────────────────


def test_three_sources_is_confirmed():
    arts = [
        _art("Tesla beats Q1 deliveries record", source="reuters"),
        _art("Tesla beats Q1 deliveries record yoy", source="bloomberg"),
        _art("Tesla Q1 deliveries record beat", source="cnbc"),
    ]
    out = build_breaking_confluence(arts, now=NOW)
    assert out["counts_by_verdict"]["CONFIRMED"] == 1
    assert out["clusters"][0]["verdict"] == "CONFIRMED"
    assert out["clusters"][0]["n_sources"] == 3


def test_two_sources_fresh_is_emerging():
    arts = [
        _art("AMZN Q3 cloud revenue surges", source="reuters", age_min=5.0),
        _art("AMZN cloud revenue Q3 surges", source="bloomberg", age_min=10.0),
    ]
    out = build_breaking_confluence(arts, now=NOW, emerging_window_minutes=30)
    assert out["counts_by_verdict"]["EMERGING"] == 1
    assert out["clusters"][0]["verdict"] == "EMERGING"


def test_two_sources_stale_is_confirmed_not_emerging():
    """A 2-source cluster whose latest article is 45 min old is past the
    EMERGING window — should still surface but classified CONFIRMED, since
    it has corroboration."""
    arts = [
        _art("MSFT cloud revenue Q4", source="reuters", age_min=55.0),
        _art("MSFT Q4 cloud revenue", source="bloomberg", age_min=45.0),
    ]
    out = build_breaking_confluence(arts, now=NOW, emerging_window_minutes=30)
    assert out["clusters"][0]["verdict"] == "CONFIRMED"


def test_hot_singleton_surfaces():
    """A solo Reuters 8-K with urgency 1 + ai_score 9.5 in the last
    emerging window should keep visibility as SINGLETON_HOT."""
    arts = [
        _art("META 8-K SEC filing material event",
             source="reuters", ai_score=9.5, urgency=1, age_min=5.0),
    ]
    out = build_breaking_confluence(arts, now=NOW)
    assert out["counts_by_verdict"]["SINGLETON_HOT"] == 1
    assert out["clusters"][0]["verdict"] == "SINGLETON_HOT"


def test_cold_singleton_filtered():
    """A solo cluster that doesn't meet the hot bar should be filtered —
    the corroboration discipline."""
    arts = [
        _art("Some lukewarm headline about XYZ", source="rss",
             ai_score=7.0, urgency=0, age_min=5.0),
    ]
    out = build_breaking_confluence(arts, now=NOW)
    assert out["n_surfaced"] == 0
    assert out["clusters"] == []


def test_stale_hot_singleton_filtered():
    """Even a hot singleton, if its latest_seen is past the emerging
    window, is filtered — the velocity discipline (it's no longer
    BREAKING)."""
    arts = [
        _art("AVGO 8-K material event very fresh",
             source="reuters", ai_score=9.5, urgency=1, age_min=45.0),
    ]
    out = build_breaking_confluence(arts, now=NOW, emerging_window_minutes=30)
    assert out["n_surfaced"] == 0


# ── velocity math ──────────────────────────────────────────────────────────


def test_velocity_per_30min_scales_with_window():
    """4 articles in a 60-minute window = 2.0 per 30 minutes."""
    arts = [
        _art("ORCL Q2 earnings beat", source="reuters", age_min=5.0),
        _art("ORCL beats Q2 earnings", source="bloomberg", age_min=10.0),
        _art("ORCL Q2 earnings beat estimate", source="cnbc", age_min=15.0),
        _art("ORCL Q2 earnings beat results", source="yahoo", age_min=20.0),
    ]
    out = build_breaking_confluence(arts, now=NOW, window_minutes=60)
    assert len(out["clusters"]) == 1
    assert out["clusters"][0]["n_articles"] == 4
    assert out["clusters"][0]["velocity_per_30min"] == 2.0


def test_velocity_doubles_when_window_halves():
    """Same 4 articles fit a 30-minute window if all are < 30 min old →
    velocity_per_30min == 4.0."""
    arts = [
        _art("PYPL raises full-year guidance results",
             source="reuters", age_min=5.0),
        _art("PYPL raises full-year guidance report",
             source="bloomberg", age_min=10.0),
        _art("PYPL raises full-year guidance update",
             source="cnbc", age_min=15.0),
        _art("PYPL raises full-year guidance again",
             source="yahoo", age_min=20.0),
    ]
    out = build_breaking_confluence(arts, now=NOW, window_minutes=30)
    assert out["clusters"][0]["velocity_per_30min"] == 4.0


# ── ranking ────────────────────────────────────────────────────────────────


def test_confirmed_ranks_above_emerging_above_singleton():
    arts = [
        _art("ZZZZ files 8-K material event report",
             source="reuters", ai_score=9.5, urgency=1, age_min=2.0),
        _art("XYZ beats quarterly earnings estimates",
             source="reuters", age_min=5.0),
        _art("XYZ beats quarterly earnings estimates results",
             source="bloomberg", age_min=8.0),
        _art("AAA reports record quarterly revenue surge",
             source="reuters", age_min=5.0),
        _art("AAA reports record quarterly revenue surge update",
             source="bloomberg", age_min=8.0),
        _art("AAA reports record quarterly revenue surge results",
             source="cnbc", age_min=10.0),
    ]
    out = build_breaking_confluence(arts, now=NOW)
    verdicts = [c["verdict"] for c in out["clusters"]]
    assert verdicts.index("CONFIRMED") < verdicts.index("EMERGING")
    assert verdicts.index("EMERGING") < verdicts.index("SINGLETON_HOT")


def test_recency_breaks_tie_within_same_verdict():
    """Two CONFIRMED clusters: the one whose latest article is fresher
    should rank higher."""
    arts = [
        _art("FOO Q4 beat", source="reuters", age_min=2.0),
        _art("FOO Q4 beats", source="bloomberg", age_min=3.0),
        _art("FOO Q4 results beat", source="cnbc", age_min=4.0),
        _art("BAR Q2 surprise", source="reuters", age_min=20.0),
        _art("BAR Q2 surprises", source="bloomberg", age_min=22.0),
        _art("BAR Q2 surprise win", source="cnbc", age_min=25.0),
    ]
    out = build_breaking_confluence(arts, now=NOW)
    assert "FOO" in out["clusters"][0]["headline"]
    assert "BAR" in out["clusters"][1]["headline"]


# ── max_clusters cap ───────────────────────────────────────────────────────


def test_max_clusters_caps_output():
    """Distinct stories cluster into distinct groups; max_clusters caps."""
    arts = [
        # Cluster A: PYPL guidance raise.
        _art("PYPL raises full-year guidance update",
             source="reuters", age_min=2.0),
        _art("PYPL raises full-year guidance report",
             source="bloomberg", age_min=3.0),
        # Cluster B: AMD chip lawsuit.
        _art("AMD chip patent lawsuit ruling filed",
             source="reuters", age_min=2.0),
        _art("AMD chip patent lawsuit ruling update",
             source="bloomberg", age_min=3.0),
        # Cluster C: NFLX subscriber surge.
        _art("NFLX subscriber growth surges record quarter",
             source="reuters", age_min=2.0),
        _art("NFLX subscriber growth surges record results",
             source="bloomberg", age_min=3.0),
        # Cluster D: BA aircraft order.
        _art("BA secures massive aircraft order deal closed",
             source="reuters", age_min=2.0),
        _art("BA secures massive aircraft order deal final",
             source="bloomberg", age_min=3.0),
        # Cluster E: F vehicle recall.
        _art("Ford issues vehicle recall safety notice today",
             source="reuters", age_min=2.0),
        _art("Ford issues vehicle recall safety notice update",
             source="bloomberg", age_min=3.0),
    ]
    out = build_breaking_confluence(arts, now=NOW, max_clusters=2)
    # At least 5 distinct clusters formed; surfaced count reflects the
    # pre-cap multi-source filter pass (the news_corroboration precedent
    # for n_multi_source); the actual returned list is then trimmed to
    # max_clusters.
    assert out["n_clusters_formed"] >= 5
    assert out["n_surfaced"] >= 5
    assert len(out["clusters"]) == 2


# ── route ──────────────────────────────────────────────────────────────────


def test_route_exists_and_clamps_params():
    app = create_app()
    client = app.test_client()
    # Default request — should return JSON envelope even on empty DB.
    r = client.get("/api/breaking-confluence")
    assert r.status_code in (200, 401)  # 401 if API key gate enabled
    if r.status_code == 200:
        data = r.get_json()
        assert "as_of" in data
        assert "clusters" in data
        assert "counts_by_verdict" in data
        assert data["window_minutes"] == 60
        # Verify clamp: oversize window_minutes is clamped to 720.
        r2 = client.get("/api/breaking-confluence?window_minutes=99999")
        if r2.status_code == 200:
            assert r2.get_json()["window_minutes"] == 720
        # Below-lower-bound is clamped to 5.
        r3 = client.get("/api/breaking-confluence?window_minutes=1")
        if r3.status_code == 200:
            assert r3.get_json()["window_minutes"] == 5
