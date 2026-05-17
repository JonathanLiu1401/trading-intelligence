"""Regression: get_unscored must surface published/first_seen.

`get_unscored` historically returned only {_id, link, title, source, summary}.
Two live-path consumers silently degraded because the article age was missing:

  1. watchers/urgency_scorer.score_batch derives age_hours via
     _article_age_hours(published|first_seen) — that feeds both the Sonnet
     prompt's staleness rule and the hard STALE_HOURS/STALE_SCORE_CAP clamp.
     With no date every article looked 0h old, so the entire staleness
     system was inert on the production path — stale news could still alert.

  2. ml/features.extract_features derives 5 temporal features from
     `published`. _fetch_training_data passes the real value at train time;
     omitting it at inference makes the parser fall back to now(), so those
     features became a constant — a train/serve skew on every scored article.

These tests drive the *real* path (insert_batch → get_unscored) so they fail
if the fields are ever dropped again, not just hand-built dicts.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pytest

from ml.features import extract_features_batch
from watchers import urgency_scorer


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _ingest(store, *, link, title, published, summary="body text here", kw=3.0):
    """Insert exactly as a collector would (heuristic score in _relevance_score)."""
    n = store.insert_batch([{
        "title": title,
        "link": link,
        "source": "rss",
        "published": published,
        "summary": summary,
        "_relevance_score": kw,
    }])
    assert n == 1


# ── Direct shape pin ────────────────────────────────────────────────────────
def test_get_unscored_returns_published_and_first_seen(store):
    pub = _iso(26)
    _ingest(store, link="https://x.com/a", title="MU DRAM pricing update",
            published=pub)
    rows = store.get_unscored(min_kw=0.0)
    assert len(rows) == 1
    r = rows[0]
    # The two age fields the live path depends on must be present and real.
    assert r["published"] == pub
    assert r["first_seen"]                       # ISO insert timestamp
    # Existing keys still intact.
    assert set(r) >= {"_id", "link", "title", "source", "summary"}


# ── Harm 1: staleness cap is effective on the live path ─────────────────────
class TestStalenessCapViaLivePath:
    """Thresholds are read from the live module constants so this stays
    correct if STALE_HOURS is retuned (it has been: 24h → 48h)."""

    def _score(self, store, claude_score):
        unscored = store.get_unscored(min_kw=0.0)
        assert len(unscored) == 1
        with patch.object(urgency_scorer, "claude_call",
                           return_value=json.dumps(
                               [{"index": 0, "score": claude_score,
                                 "reason": "x"}])):
            urgency_scorer.score_batch(unscored, store)
        return store.conn.execute(
            "SELECT ai_score, urgency FROM articles"
        ).fetchone()

    def test_stale_article_capped_below_urgent(self, store):
        # Comfortably past the hard-cap window. Sonnet says 9 (would be
        # urgent), but the hard STALE_SCORE_CAP must clamp it. Pre-fix this
        # failed: get_unscored dropped `published`, _article_age_hours read
        # 0h, no cap fired, ai_score=9 urgency=1 → stale news alerts.
        stale_h = urgency_scorer.STALE_HOURS + 12
        _ingest(store, link="https://x.com/stale",
                title="MU earnings beat — BREAKING today",
                published=_iso(stale_h))
        ai_score, urgency = self._score(store, 9)
        assert ai_score == pytest.approx(urgency_scorer.STALE_SCORE_CAP)
        assert urgency == 0          # capped below URGENT_THRESHOLD (8.0)

    def test_fresh_article_not_capped(self, store):
        # Control: 1h old, same Sonnet score → hard cap must NOT fire.
        _ingest(store, link="https://x.com/fresh",
                title="MU earnings beat — BREAKING today",
                published=_iso(1))
        ai_score, urgency = self._score(store, 9)
        assert ai_score == pytest.approx(9.0)
        assert urgency == 1


# ── Harm 2: train/serve temporal-feature parity ─────────────────────────────
def test_inference_features_match_training_shape(store):
    """The same article must yield identical feature rows whether it comes
    through get_unscored (inference) or the _fetch_training_data dict shape
    (training). Pre-fix the inference dict lacked `published`, so temporal
    features (indices 2-6) were computed from now() instead of publish time."""
    pub = _iso(26)
    title = "AXTI substrate capacity expansion confirmed"
    summary = "Company announced a substantial wafer fab expansion."
    _ingest(store, link="https://x.com/p", title=title, published=pub,
            summary=summary)

    inference_dict = store.get_unscored(min_kw=0.0)[0]
    # Exactly the dict shape ml/trainer._fetch_training_data builds.
    training_dict = {"title": title, "summary": summary,
                     "source": "rss", "published": pub}

    feats = extract_features_batch([training_dict, inference_dict])
    # Identical down to float noise (each row calls datetime.now() once for
    # the age delta, microseconds apart — the skew this guards against is
    # whole-feature-sized, not last-bit).
    assert np.allclose(feats[0], feats[1], atol=1e-5), (
        f"train/serve feature skew:\n train={feats[0]}\n infer={feats[1]}")
    # Sanity: a 26h-old article must NOT read as just-published. Index 6 is
    # days_since_published normalized by /30 → ~1.08/30 ≈ 0.036, not 0.
    assert feats[1][6] > 0.02
