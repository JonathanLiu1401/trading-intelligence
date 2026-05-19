"""Pure-helper tests for the /api/chat alert-confidence-trend enrichment.

`build_alert_confidence_trend` clusters urgent articles (urgency >= 1) in the
last 24h by title-token Jaccard similarity (reusing `ml.dedup.title_tokens`
+ `jaccard_similarity` — SSOT with the dedup module so chat / briefing /
dashboard cluster identically) and reports the unique-source count delta
between the recent half (0-6h) and the earlier half (6-24h).

`_alert_confidence_trend_chat_lines` renders the report as chat-context lines.

The surrounding chat handler is one large inline closure, so per the design
established by `_baseline_compare_chat_lines` / sibling helpers the logic
is a total/pure pair unit-tested here — no Flask, no :8090, no articles.db.

Discriminating locks:
- **clustering uses the SSOT dedup module** — a one-off rewrite would
  silently drift from /api/news-corroboration and the briefing's own
  collapse, leading to "12 → 19 sources" counts that don't match.
- **unique-source count, not article count** — a single outlet syndicating
  itself many times must NOT inflate trust.
- **STABLE / SINGLE_SOURCE collapse to silence** — chat budget is finite;
  only RISING / FADING (the actively-moving stories) surface as lines.
- **pure / total**: non-list / garbage rows / missing fields never raise.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import (  # noqa: E402
    build_alert_confidence_trend,
    _alert_confidence_trend_chat_lines,
)


_NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _art(title: str, score: float, hours_ago: float,
         source: str = "reuters") -> dict:
    ts = _NOW - timedelta(hours=hours_ago)
    return {"title": title, "ai_score": score,
            "first_seen": ts.isoformat(), "source": source}


# ── builder: clustering & trend classification ────────────────────────
def test_rising_when_more_unique_sources_in_recent_half():
    """A story with 1 earlier source then 3 new recent sources →
    delta = +2, trend RISING. Headline is the highest-ai_score title."""
    arts = [
        _art("Nvidia earnings miss guidance", 9.5, 22.0, source="finnhub"),
        _art("Nvidia earnings miss guidance", 9.8, 2.0, source="reuters"),
        _art("Nvidia earnings miss guidance", 9.5, 3.0, source="bloomberg"),
        _art("Nvidia earnings miss guidance", 9.6, 4.0, source="gdelt"),
    ]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    assert len(rep["clusters"]) == 1
    cl = rep["clusters"][0]
    assert cl["trend"] == "RISING"
    assert cl["n_earlier_sources"] == 1
    assert cl["n_recent_sources"] == 3
    assert cl["n_total_sources"] == 4
    assert cl["delta"] == 2
    assert "Nvidia" in cl["anchor_title"]
    assert cl["max_ai_score"] == pytest.approx(9.8)


def test_fading_when_more_unique_sources_in_earlier_half():
    arts = [
        _art("Fed signals rate cut imminent", 9.0, 22.0, source="reuters"),
        _art("Fed signals rate cut imminent", 9.0, 20.0, source="bloomberg"),
        _art("Fed signals rate cut imminent", 9.0, 18.0, source="finnhub"),
        _art("Fed signals rate cut imminent", 8.0, 2.0, source="gdelt"),
    ]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    cl = rep["clusters"][0]
    assert cl["trend"] == "FADING"
    assert cl["n_earlier_sources"] == 3
    assert cl["n_recent_sources"] == 1
    assert cl["delta"] == -2


def test_stable_when_recent_and_earlier_counts_match():
    arts = [
        _art("AMD launches new GPU lineup", 8.0, 20.0, source="reuters"),
        _art("AMD launches new GPU lineup", 8.0, 2.0, source="bloomberg"),
    ]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    cl = rep["clusters"][0]
    assert cl["trend"] == "STABLE"
    assert cl["delta"] == 0


def test_single_source_when_one_outlet_syndicates_itself():
    """Trust requires CORROBORATION. A single outlet posting the same
    story many times must NOT inflate the source count or be flagged
    RISING. Discriminating: 5 articles, 1 source → SINGLE_SOURCE."""
    arts = [
        _art("Tesla recalls 200k cars", 9.0, 20.0, source="reuters"),
        _art("Tesla recalls 200k cars", 9.0, 15.0, source="reuters"),
        _art("Tesla recalls 200k cars", 9.0, 10.0, source="reuters"),
        _art("Tesla recalls 200k cars", 9.0, 5.0, source="reuters"),
        _art("Tesla recalls 200k cars", 9.0, 1.0, source="reuters"),
    ]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    cl = rep["clusters"][0]
    assert cl["trend"] == "SINGLE_SOURCE"
    assert cl["n_total_sources"] == 1
    assert cl["n_articles"] == 5


def test_clustering_collapses_jaccard_near_duplicates():
    """Word-order variants of the same story must cluster together via the
    SSOT Jaccard normalizer — otherwise the trend count fragments across
    near-duplicate clusters and the analyst sees noise."""
    arts = [
        _art("Apple beats Q2 expectations on iPhone strength", 9.0, 20.0,
             source="reuters"),
        _art("Q2 expectations beaten by Apple iPhone strength", 9.0, 2.0,
             source="bloomberg"),
        _art("Apple beats Q2 expectations on iPhone strength", 9.0, 3.0,
             source="finnhub"),
    ]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    # All three should collapse into ONE cluster (Jaccard ≥ 0.6 on
    # tokens like {apple, beats, q2, expectations, iphone, strength}).
    assert len(rep["clusters"]) == 1
    cl = rep["clusters"][0]
    assert cl["n_total_sources"] == 3


def test_unrelated_stories_form_separate_clusters():
    arts = [
        _art("Nvidia earnings miss guidance", 9.0, 2.0, source="reuters"),
        _art("Nvidia earnings miss guidance", 9.0, 4.0, source="bloomberg"),
        _art("Fed signals rate cut imminent", 9.0, 2.0, source="reuters"),
        _art("Fed signals rate cut imminent", 9.0, 4.0, source="bloomberg"),
    ]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    assert len(rep["clusters"]) == 2
    titles = {c["anchor_title"] for c in rep["clusters"]}
    assert any("Nvidia" in t for t in titles)
    assert any("Fed" in t for t in titles)


def test_drops_articles_outside_24h_window():
    arts = [
        _art("MU memory crash", 9.0, 30.0, source="reuters"),
        _art("MU memory crash", 9.0, 2.0, source="bloomberg"),
    ]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    # Only the 2h-old article survives → singleton cluster below min_cluster_size
    assert rep["clusters"] == []


def test_drops_clusters_below_min_cluster_size():
    """A 'cluster' of one article isn't a corroborated story — it's a
    lone wire item. Default min_cluster_size is 2; assert the default
    drops singletons rather than reporting them as SINGLE_SOURCE."""
    arts = [_art("Lone story", 9.0, 2.0, source="reuters")]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    assert rep["clusters"] == []


def test_anchor_title_is_highest_score_member():
    """The cluster's `anchor_title` must be the highest-ai_score member —
    that's the canonical headline the analyst recognizes, not whichever
    arrived first.

    All three titles share the same dominant tokens (apple beats q2
    expectations iphone strength) so they collapse into one cluster;
    only the ai_score differs, and the high-scorer must win the anchor."""
    arts = [
        _art("Apple beats Q2 expectations on iPhone strength", 4.0, 8.0,
             source="rss"),
        _art("Apple beats Q2 expectations — Reuters scoop on iPhone strength",
             9.7, 2.0, source="reuters"),
        _art("Apple beats Q2 expectations on iPhone strength continues",
             3.5, 4.0, source="bloomberg"),
    ]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    assert len(rep["clusters"]) == 1, (
        f"expected single cluster, got {len(rep['clusters'])}: "
        f"{[c['anchor_title'] for c in rep['clusters']]}"
    )
    cl = rep["clusters"][0]
    assert cl["anchor_title"] == (
        "Apple beats Q2 expectations — Reuters scoop on iPhone strength"
    )
    assert cl["max_ai_score"] == pytest.approx(9.7)


def test_ranking_puts_rising_first_then_by_score():
    """RISING clusters surface before STABLE/FADING — the chat budget
    prioritises actionable signals."""
    arts = []
    # Fading story (3 earlier, 1 recent)
    for h in (20, 18, 16):
        arts.append(_art("Old story going quiet", 9.5, h, source=f"src-old-{h}"))
    arts.append(_art("Old story going quiet", 9.5, 2.0, source="late"))
    # Rising story (1 earlier, 3 recent)
    arts.append(_art("Breaking story spreading", 8.0, 20.0, source="early"))
    for h in (2, 3, 4):
        arts.append(_art("Breaking story spreading", 8.0, h,
                         source=f"src-new-{h}"))
    rep = build_alert_confidence_trend(arts, now=_NOW)
    assert rep["clusters"][0]["trend"] == "RISING"


# ── builder: pure / total contract ────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, object()])
def test_non_list_articles_is_empty_clusters(bad):
    rep = build_alert_confidence_trend(bad, now=_NOW)
    assert rep["clusters"] == []
    assert rep["window_hours"] == 24


def test_skips_garbage_rows_without_raising():
    arts = [
        _art("Real story 1", 9.0, 2.0, source="reuters"),
        _art("Real story 1", 9.0, 4.0, source="bloomberg"),
        {"title": None, "ai_score": 9.0,
         "first_seen": _NOW.isoformat(), "source": "x"},
        {"title": "Missing timestamp", "ai_score": 9.0, "source": "x"},
        "not a dict",
        None,
        {"title": "Bad score", "ai_score": "nope",
         "first_seen": _NOW.isoformat(), "source": "x"},
    ]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    # The 2 real stories cluster together; the garbage rows are dropped.
    assert len(rep["clusters"]) == 1
    assert rep["clusters"][0]["n_total_sources"] == 2


def test_no_source_does_not_inflate_unique_count():
    """An article with an empty source must NOT add 1 to the unique-source
    count (otherwise a blank-source row would always trip RISING)."""
    arts = [
        _art("Story alpha", 9.0, 4.0, source="reuters"),
        _art("Story alpha", 9.0, 2.0, source=""),
    ]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    cl = rep["clusters"][0]
    assert cl["n_total_sources"] == 1
    # No-source counts as cluster membership but not as corroboration.


# ── render: contract ──────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [None, "x", 42, [], object()])
def test_render_non_dict_is_silence(bad):
    assert _alert_confidence_trend_chat_lines(bad) == []


def test_render_empty_clusters_is_silence():
    assert _alert_confidence_trend_chat_lines({}) == []
    assert _alert_confidence_trend_chat_lines({"clusters": []}) == []


def test_render_stable_and_single_source_collapse_to_silence():
    """Chat budget — only actively-moving stories surface."""
    rep = {
        "clusters": [
            {"anchor_title": "Stable story", "trend": "STABLE",
             "max_ai_score": 8.0, "n_total_sources": 4,
             "n_recent_sources": 2, "n_earlier_sources": 2, "delta": 0},
            {"anchor_title": "Lone PR", "trend": "SINGLE_SOURCE",
             "max_ai_score": 8.0, "n_total_sources": 1,
             "n_recent_sources": 1, "n_earlier_sources": 0, "delta": 1},
        ],
    }
    assert _alert_confidence_trend_chat_lines(rep) == []


def test_render_rising_includes_trend_score_and_source_counts():
    rep = {
        "clusters": [
            {"anchor_title": "Nvidia earnings miss guidance",
             "trend": "RISING", "max_ai_score": 9.8,
             "n_total_sources": 4, "n_recent_sources": 3,
             "n_earlier_sources": 1, "delta": 2,
             "n_articles": 5},
        ],
    }
    out = _alert_confidence_trend_chat_lines(rep)
    assert len(out) == 1
    line = out[0]
    assert "RISING" in line
    assert "Nvidia" in line
    assert "9.8" in line
    assert "1 src" in line and "3 src" in line and "4 total" in line


def test_render_truncates_very_long_headlines():
    long_title = "X" * 200
    rep = {
        "clusters": [
            {"anchor_title": long_title, "trend": "RISING",
             "max_ai_score": 9.0, "n_total_sources": 3,
             "n_recent_sources": 3, "n_earlier_sources": 0, "delta": 3},
        ],
    }
    out = _alert_confidence_trend_chat_lines(rep)
    assert len(out) == 1
    # Chat budget — the headline must not push past ~100 chars.
    assert len(out[0]) < 200
    assert "..." in out[0]


def test_render_skips_clusters_with_blank_title():
    rep = {
        "clusters": [
            {"anchor_title": "", "trend": "RISING",
             "max_ai_score": 9.0, "n_total_sources": 3,
             "n_recent_sources": 3, "n_earlier_sources": 0, "delta": 3},
            {"anchor_title": "Real title", "trend": "RISING",
             "max_ai_score": 8.0, "n_total_sources": 2,
             "n_recent_sources": 2, "n_earlier_sources": 0, "delta": 2},
        ],
    }
    out = _alert_confidence_trend_chat_lines(rep)
    assert len(out) == 1
    assert "Real title" in out[0]


# ── end-to-end pure pipeline ──────────────────────────────────────────
def test_end_to_end_realistic_mixed_book():
    """Realistic mix: one rising story, one fading, one single-source PR.
    Render output should contain exactly 2 lines (RISING + FADING),
    RISING first."""
    arts = [
        # RISING: 1 earlier source → 3 recent sources
        _art("Apple beats Q2 expectations on iPhone strength", 9.5, 20.0,
             source="rss"),
        _art("Apple beats Q2 expectations on iPhone strength", 9.7, 2.0,
             source="reuters"),
        _art("Apple beats Q2 expectations on iPhone strength", 9.5, 3.0,
             source="bloomberg"),
        _art("Apple beats Q2 expectations on iPhone strength", 9.5, 4.0,
             source="finnhub"),
        # FADING: 3 earlier → 1 recent
        _art("Fed signals rate cut imminent", 9.0, 22.0, source="reuters"),
        _art("Fed signals rate cut imminent", 9.0, 20.0, source="bloomberg"),
        _art("Fed signals rate cut imminent", 9.0, 18.0, source="finnhub"),
        _art("Fed signals rate cut imminent", 8.0, 2.0, source="gdelt"),
        # SINGLE_SOURCE: PR spam from one outlet
        _art("Tesla recalls 200k cars", 9.0, 20.0, source="prfeed"),
        _art("Tesla recalls 200k cars", 9.0, 15.0, source="prfeed"),
        _art("Tesla recalls 200k cars", 9.0, 1.0, source="prfeed"),
    ]
    rep = build_alert_confidence_trend(arts, now=_NOW)
    out = _alert_confidence_trend_chat_lines(rep)
    assert len(out) == 2
    assert "RISING" in out[0] and "Apple" in out[0]
    assert "FADING" in out[1] and "Fed" in out[1]
    # SINGLE_SOURCE must NOT leak to the analyst.
    assert not any("Tesla" in ln for ln in out)
