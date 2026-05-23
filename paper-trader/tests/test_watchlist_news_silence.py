"""Unit tests for paper_trader.analytics.watchlist_news_silence.

Locks the per-ticker classification (SILENT / STALE / LIVE / HOT),
the universe-level verdict (NO_DATA / BLIND_UNIVERSE / SPARSE_COVERAGE /
WELL_COVERED), the silent/hot list ordering, threshold-driven
boundary behaviour, and the never-raises discipline.
"""
from __future__ import annotations

from paper_trader.analytics.watchlist_news_silence import (
    BLIND_UNIVERSE_PCT_FLOOR,
    HOT_N,
    MIN_EVALUABLE_TICKERS,
    SPARSE_COVERAGE_PCT_FLOOR,
    STALE_HOURS_FLOOR,
    _classify_ticker,
    build_watchlist_news_silence,
)


def _summary(n, hsl, *, max_score=5.0, n_signal_grade=None):
    return {
        "n_in_window": n,
        "n_signal_grade": n if n_signal_grade is None else n_signal_grade,
        "max_score": max_score,
        "hours_since_last": hsl,
        "last_seen_iso": "2026-05-23T00:00:00+00:00" if hsl is not None else None,
    }


# ─── per-ticker classification ───────────────────────────────────────────


def test_silent_when_zero_articles():
    assert _classify_ticker({"n_in_window": 0, "hours_since_last": None}) == "SILENT"


def test_silent_when_hours_since_last_missing():
    assert _classify_ticker({"n_in_window": 5, "hours_since_last": None}) == "SILENT"


def test_stale_when_newest_older_than_floor():
    assert _classify_ticker(
        {"n_in_window": 1, "hours_since_last": STALE_HOURS_FLOOR + 0.1}
    ) == "STALE"


def test_live_below_hot_floor():
    assert _classify_ticker(
        {"n_in_window": HOT_N - 1, "hours_since_last": 1.0}
    ) == "LIVE"


def test_hot_at_threshold():
    assert _classify_ticker(
        {"n_in_window": HOT_N, "hours_since_last": 0.5}
    ) == "HOT"


# ─── universe verdict ────────────────────────────────────────────────────


def test_no_data_below_min_evaluable():
    # Fewer than MIN_EVALUABLE_TICKERS unique watchlist tickers ⇒
    # verdict withheld.
    out = build_watchlist_news_silence(["A", "B", "C"], per_ticker={})
    assert out["verdict"] == "NO_DATA"
    assert "NO_DATA" in out["headline"]


def test_blind_universe_when_all_silent():
    tickers = [f"T{i}" for i in range(10)]
    out = build_watchlist_news_silence(tickers, per_ticker={})
    assert out["verdict"] == "BLIND_UNIVERSE"
    assert out["n_silent"] == 10
    assert out["silent_pct"] == 100.0


def test_well_covered_when_no_silent():
    tickers = [f"T{i}" for i in range(8)]
    per_ticker = {t: _summary(3, 1.0) for t in tickers}
    out = build_watchlist_news_silence(tickers, per_ticker=per_ticker)
    assert out["verdict"] == "WELL_COVERED"
    assert out["n_silent"] == 0


def test_sparse_coverage_band():
    # 8 tickers; 3 silent ⇒ 37.5% — between SPARSE and BLIND floors.
    tickers = ["A", "B", "C", "D", "E", "F", "G", "H"]
    per_ticker = {t: _summary(3, 1.0) for t in tickers[:5]}
    out = build_watchlist_news_silence(tickers, per_ticker=per_ticker)
    assert out["verdict"] == "SPARSE_COVERAGE"
    assert out["n_silent"] == 3
    assert SPARSE_COVERAGE_PCT_FLOOR <= out["silent_pct"] < BLIND_UNIVERSE_PCT_FLOOR


def test_blind_universe_boundary_inclusive():
    # 10 tickers, 5 silent ⇒ exactly BLIND floor.
    tickers = [f"T{i}" for i in range(10)]
    per_ticker = {t: _summary(3, 1.0) for t in tickers[:5]}
    out = build_watchlist_news_silence(tickers, per_ticker=per_ticker)
    assert out["verdict"] == "BLIND_UNIVERSE"


def test_well_covered_just_below_sparse_floor():
    # 10 tickers, 1 silent ⇒ 10% — strictly below SPARSE floor.
    tickers = [f"T{i}" for i in range(10)]
    per_ticker = {t: _summary(3, 1.0) for t in tickers[:9]}
    out = build_watchlist_news_silence(tickers, per_ticker=per_ticker)
    assert out["verdict"] == "WELL_COVERED"


# ─── silent + hot lists ──────────────────────────────────────────────────


def test_silent_tickers_alphabetical():
    tickers = ["NVDA", "AMD", "MU", "TSM", "AAPL", "MSFT", "GOOG", "META"]
    # All silent (no per_ticker data).
    out = build_watchlist_news_silence(tickers, per_ticker={})
    assert out["silent_tickers"] == sorted(tickers)


def test_silent_tickers_capped_at_10():
    tickers = [f"T{i:02d}" for i in range(15)]
    out = build_watchlist_news_silence(tickers, per_ticker={})
    assert len(out["silent_tickers"]) == 10
    # Capped list is still alphabetical.
    assert out["silent_tickers"] == sorted(tickers)[:10]


def test_hot_storms_ordered_by_volume_desc():
    tickers = ["A", "B", "C", "D", "E", "F"]
    per_ticker = {
        "A": _summary(15, 1.0, max_score=5.0),  # HOT
        "B": _summary(20, 1.0, max_score=7.0),  # HOT, bigger n
        "C": _summary(3, 1.0),                  # LIVE
        "D": _summary(15, 1.0, max_score=9.0),  # HOT, tie n with A
        "E": _summary(3, 1.0),
        "F": _summary(3, 1.0),
    }
    out = build_watchlist_news_silence(tickers, per_ticker=per_ticker)
    storms = out["hot_storms"]
    assert [s["ticker"] for s in storms] == ["B", "D", "A"]
    assert storms[0]["n_in_window"] == 20


# ─── dedup / normalisation ───────────────────────────────────────────────


def test_duplicate_tickers_collapsed():
    tickers = ["NVDA", "NVDA", "nvda", "AMD", "AMD"]
    out = build_watchlist_news_silence(tickers, per_ticker={})
    assert out["n_total"] == 2
    assert out["silent_tickers"] == ["AMD", "NVDA"]


def test_non_string_entries_skipped():
    tickers = ["NVDA", None, 42, "", "AMD"]
    out = build_watchlist_news_silence(tickers, per_ticker={})
    assert out["n_total"] == 2


# ─── never-raises / advisory discipline ──────────────────────────────────


def test_per_ticker_not_a_dict_does_not_raise():
    tickers = [f"T{i}" for i in range(8)]
    out = build_watchlist_news_silence(tickers, per_ticker="not-a-dict")
    assert out["verdict"] == "BLIND_UNIVERSE"
    assert out["n_silent"] == 8


def test_garbage_summary_values_degrade_to_silent():
    tickers = ["A", "B", "C", "D", "E", "F"]
    per_ticker = {
        "A": "not-a-dict",
        "B": {"n_in_window": "x", "hours_since_last": "y"},
        "C": {"n_in_window": -5, "hours_since_last": None},
        "D": _summary(0, None),
        "E": _summary(1, 0.5),  # LIVE
        "F": _summary(20, 0.5),  # HOT
    }
    out = build_watchlist_news_silence(tickers, per_ticker=per_ticker)
    # A, B, C, D ⇒ SILENT; E ⇒ LIVE; F ⇒ HOT.
    assert out["n_silent"] == 4
    assert out["n_live"] == 1
    assert out["n_hot"] == 1


def test_thresholds_block_in_report():
    out = build_watchlist_news_silence([], per_ticker={})
    assert "thresholds" in out
    th = out["thresholds"]
    assert th["hot_n"] == HOT_N
    assert th["blind_universe_pct_floor"] == BLIND_UNIVERSE_PCT_FLOOR
    assert th["min_evaluable_tickers"] == MIN_EVALUABLE_TICKERS


def test_empty_watchlist_no_raise():
    out = build_watchlist_news_silence(None, per_ticker=None)
    assert out["verdict"] == "NO_DATA"
    assert out["n_total"] == 0


def test_hours_param_threaded_into_report():
    out = build_watchlist_news_silence(
        [f"T{i}" for i in range(8)], per_ticker={}, hours=48,
    )
    assert out["hours"] == 48
    # No silent → blind universe; headline must restate the window.
    assert "48h" in out["headline"]
