"""Feature extractor contract — used by trainer + inference; shape must hold."""
from __future__ import annotations

import json
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


def test_temporal_features_timezone_invariant():
    """The SAME publishing instant expressed in different feed timezones must
    produce identical hour/dow cyclic features (indices 2..5).

    Regression: feeds emit their publish time in their own tz (Nikkei +0900,
    US wires -0500, most others UTC). Without UTC normalisation in
    _parse_published the same instant scattered across the cyclic encoding —
    a -0500 feed even landed on the previous weekday — injecting pure noise
    into 4 of 15 features and a train/serve skew (trainer + inference both
    pass raw `published` strings)."""
    # 2026-05-13 03:15:00 UTC, expressed three ways for the same instant.
    variants = [
        "Tue, 13 May 2026 03:15:00 +0000",   # UTC wire
        "Mon, 12 May 2026 22:15:00 -0500",   # US EST feed (prev local day)
        "Tue, 13 May 2026 12:15:00 +0900",   # Nikkei JST feed
        "2026-05-13T03:15:00+00:00",         # ISO 8601 UTC
        "2026-05-13T03:15:00",               # naive → assumed UTC
    ]
    ref = features.extract_features({
        "title": "x", "summary": "y", "source": "rss", "published": variants[0],
    })
    for v in variants[1:]:
        feats = features.extract_features({
            "title": "x", "summary": "y", "source": "rss", "published": v,
        })
        for i in (2, 3, 4, 5):
            assert feats[i] == pytest.approx(ref[i], abs=1e-6), (
                f"feature {i} differs for {v!r}: {feats[i]} != {ref[i]}"
            )


def test_source_credibility_known_high():
    """reuters should score significantly higher than reddit."""
    a = features.extract_features({
        "title": "x", "summary": "", "source": "reuters", "published": "",
    })
    b = features.extract_features({
        "title": "x", "summary": "", "source": "reddit", "published": "",
    })
    assert a[0] > b[0]


# ── Portfolio-ticker config loading ─────────────────────────────────────────
# LIVE_PORTFOLIO_TICKERS is the union of a hardcoded fallback with whatever
# config/portfolio.json currently holds — the operator's source of truth.

def test_load_portfolio_tickers_unions_config(tmp_path, monkeypatch):
    """positions + option underlyings + sector_watchlist from portfolio.json
    are all added (uppercased); the hardcoded fallback is preserved."""
    cfg = tmp_path / "portfolio.json"
    cfg.write_text(json.dumps({
        "positions": [{"ticker": "ZZZA"}, {"ticker": "zzzb"}],
        "options": [{"underlying": "ZZZC"}],
        "sector_watchlist": ["ZZZD", "005930.KS"],
    }))
    monkeypatch.setattr(features, "_PORTFOLIO_JSON", cfg)
    got = features._load_portfolio_tickers()
    # config tickers added, normalised to uppercase
    assert {"ZZZA", "ZZZB", "ZZZC", "ZZZD"} <= got
    # foreign/compound symbol filtered out by _TICKER_RE
    assert "005930.KS" not in got
    # union semantics: a hardcoded fallback name never disappears
    assert "MU" in got and "LITE" in got


def test_load_portfolio_tickers_falls_back_when_missing(tmp_path, monkeypatch):
    """A missing portfolio.json degrades to the fallback set — never raises."""
    monkeypatch.setattr(features, "_PORTFOLIO_JSON", tmp_path / "nope.json")
    assert features._load_portfolio_tickers() == features._FALLBACK_PORTFOLIO_TICKERS


def test_load_portfolio_tickers_falls_back_on_corrupt(tmp_path, monkeypatch):
    """A malformed portfolio.json degrades to the fallback set — never raises."""
    bad = tmp_path / "portfolio.json"
    bad.write_text("{not valid json")
    monkeypatch.setattr(features, "_PORTFOLIO_JSON", bad)
    assert features._load_portfolio_tickers() == features._FALLBACK_PORTFOLIO_TICKERS


def test_config_portfolio_positions_are_recognised():
    """Every current position/option-underlying in the live config/portfolio.json
    must register as a portfolio ticker — the regression this feature fixes was
    held names (GOOG/NVDL/COHR) silently absent from the hardcoded set."""
    cfg = json.loads(features._PORTFOLIO_JSON.read_text(encoding="utf-8"))
    held = {(p.get("ticker") or "").strip().upper()
            for p in cfg.get("positions", [])}
    held |= {(o.get("underlying") or "").strip().upper()
             for o in cfg.get("options", [])}
    held = {t for t in held if features._TICKER_RE.match(t)}
    assert held, "portfolio.json has no usable positions — fixture problem"
    missing = held - features.LIVE_PORTFOLIO_TICKERS
    assert not missing, f"held tickers not flagged portfolio-relevant: {missing}"


def test_config_held_ticker_drives_portfolio_flag():
    """A held ticker sourced only from portfolio.json (not the fallback) still
    sets portfolio_flag (idx 12) — proves the union reaches extract_features."""
    config_only = features.LIVE_PORTFOLIO_TICKERS - features._FALLBACK_PORTFOLIO_TICKERS
    if not config_only:
        pytest.skip("portfolio.json adds no tickers beyond the fallback")
    tkr = sorted(config_only)[0]
    feats = features.extract_features({
        "title": f"{tkr} jumps on upbeat guidance",
        "summary": "", "source": "reuters", "published": "",
    })
    assert feats[12] == 1.0
    assert feats[1] > 0.0
