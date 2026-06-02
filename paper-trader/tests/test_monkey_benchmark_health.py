from __future__ import annotations

from paper_trader.analytics import monkey_benchmark as mb


def test_benchmark_tickers_excludes_product_convexity_names():
    raw = [
        "NVDA", "SOXL", "TQQQ", "SQQQ", "BTC-USD", "GC=F", "^VIX",
        "SPY", "QQQ", "SMH", "AVGO", "UNG", "NVDA",
    ]

    assert mb.benchmark_tickers(raw) == ["NVDA", "SPY", "QQQ", "SMH", "AVGO"]


def test_cache_health_rejects_old_or_poisoned_cache():
    old = {
        "generated_at": "2026-06-01T00:00:00Z",
        "window": {"start": "2015-09-25", "end": "2025-09-22"},
        "n_tickers": 114,
    }

    healthy, reason = mb.cache_health(old)

    assert healthy is False
    assert reason == "old schema"


def test_cache_health_accepts_current_contract():
    current = {
        "schema_version": mb.CACHE_SCHEMA_VERSION,
        "universe_profile": mb.UNIVERSE_PROFILE,
        "generated_at": "2026-06-01T00:00:00Z",
        "window": {"start": "2015-09-25", "end": "2025-09-22"},
        "n_tickers": 42,
    }

    assert mb.cache_health(current) == (True, "ok")
