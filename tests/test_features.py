"""Feature extractor contract — used by trainer + inference; shape must hold."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ml import features


def test_feature_dim_is_exactly_15():
    """Trainer concatenates this to TF-IDF; off-by-one in the extra-dim count
    silently changes input_dim and forces a model rebuild every cycle."""
    assert features.EXTRA_FEATURE_DIM == 15
    feats = features.extract_features({
        "title": "MU beats earnings", "summary": "good quarter",
        "source": "reuters", "published": "",
    })
    assert feats.shape == (15,)
    assert feats.dtype == np.float32


def test_batch_returns_2d_shape():
    articles = [
        {"title": "A", "summary": "x", "source": "rss", "published": ""},
        {"title": "B", "summary": "y", "source": "rss", "published": ""},
    ]
    X = features.extract_features_batch(articles)
    assert X.shape == (2, 15)
    assert X.dtype == np.float32


def test_empty_batch_returns_empty_2d():
    """Caller does np.concatenate([X_text, X_extra], axis=1); a 1-D zeros array
    here would crash with mismatched-dim. Critical contract."""
    X = features.extract_features_batch([])
    assert X.shape == (0, 15)


def test_ticker_density_zero_with_no_portfolio_mentions():
    """ticker_mention_density (idx 1) must be zero when no portfolio ticker
    appears — the trainer's strongest signal for portfolio-relevance gating."""
    feats = features.extract_features({
        "title": "weather forecast for tomorrow",
        "summary": "rain expected nationwide",
        "source": "weather.com", "published": "",
    })
    assert feats[1] == 0.0
    assert feats[12] == 0.0  # portfolio_flag
    assert feats[13] == 0.0  # distinct ticker count


def test_ticker_density_nonzero_with_portfolio_mention():
    feats = features.extract_features({
        "title": "MU beats earnings",
        "summary": "Micron earnings beat with MSFT exposure",
        "source": "reuters", "published": "",
    })
    assert feats[1] > 0.0, "ticker_mention_density must be >0 for MU/MSFT"
    assert feats[12] == 1.0
    assert feats[13] > 0.0


def test_days_since_published_zero_when_just_published():
    """A just-published article has age=0 → feature 6 must be 0."""
    now = datetime.now(timezone.utc).isoformat()
    feats = features.extract_features({
        "title": "x", "summary": "y", "source": "rss", "published": now,
    })
    # Allow ~1 second of clock skew (sub-second age).
    assert feats[6] < 1e-3, f"expected ~0 for just-published, got {feats[6]}"


def test_days_since_published_grows_with_age():
    """An article published 24h ago has age=1 day; the feature normalizes /30,
    so the expected value is ~1/30. The point of this test is that the feature
    is materially larger than just-published and monotonically grows with age."""
    dt = datetime.now(timezone.utc) - timedelta(hours=24)
    feats_24h = features.extract_features({
        "title": "x", "summary": "y", "source": "rss",
        "published": dt.isoformat(),
    })
    feats_now = features.extract_features({
        "title": "x", "summary": "y", "source": "rss",
        "published": datetime.now(timezone.utc).isoformat(),
    })
    # 24h ago should be ~1/30 (normalization), but must be strictly > now's.
    assert feats_24h[6] > feats_now[6]
    assert feats_24h[6] == pytest.approx(1.0 / 30.0, abs=0.01)


def test_days_since_published_clipped_at_one():
    """The feature is clipped to [0, 1]; ages beyond 30 days saturate."""
    dt = datetime.now(timezone.utc) - timedelta(days=90)
    feats = features.extract_features({
        "title": "x", "summary": "y", "source": "rss",
        "published": dt.isoformat(),
    })
    assert feats[6] == 1.0


def test_temporal_cyclic_features_bounded():
    """sin/cos features (idx 2..5) live in [-1, 1] — no global clip relies on
    this contract."""
    feats = features.extract_features({
        "title": "x", "summary": "y", "source": "rss",
        "published": "2026-05-15T00:00:00+00:00",
    })
    for i in (2, 3, 4, 5):
        assert -1.0 <= feats[i] <= 1.0


def test_source_credibility_known_high():
    """reuters should score significantly higher than reddit."""
    a = features.extract_features({
        "title": "x", "summary": "", "source": "reuters", "published": "",
    })
    b = features.extract_features({
        "title": "x", "summary": "", "source": "reddit", "published": "",
    })
    assert a[0] > b[0]
