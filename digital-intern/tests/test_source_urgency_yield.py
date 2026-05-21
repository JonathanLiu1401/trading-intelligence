"""Tests for analytics/source_urgency_yield.py.

Pure-builder discipline: hand-built dicts, injected ``now``, hand-computed
expected verdicts. The builder is the operator panel for collector signal
quality — silent regressions (a verdict threshold flipping, a rate
miscount, a NOISY source being silently re-labeled CLEAN) would all fail
an assertion here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.source_urgency_yield import (  # noqa: E402
    DEFAULT_MIN_SAMPLES,
    _CLEAN_SUPPRESSION,
    _NOISY_SUPPRESSION,
    _URGENT_RATE_FLOOR,
    build_source_urgency_yield,
)

NOW = datetime(2026, 5, 21, 12, 30, 0, tzinfo=timezone.utc)


def _art(*, source: str = "rss", urgency: int = 0,
         age_hours: float = 1.0) -> dict:
    return {
        "source": source,
        "urgency": urgency,
        "first_seen": (NOW - timedelta(hours=age_hours)).isoformat(),
    }


def _arts(source: str, *, total: int, urgent: int = 0,
          alerted: int = 0) -> list[dict]:
    """Build ``total`` rows for one source: ``urgent`` rows with urgency=1,
    ``alerted`` rows with urgency=2, the rest urgency=0. The sum must
    not exceed ``total``."""
    assert urgent + alerted <= total, "test fixture: urgent+alerted > total"
    rows: list[dict] = []
    for _ in range(alerted):
        rows.append(_art(source=source, urgency=2))
    for _ in range(urgent):
        rows.append(_art(source=source, urgency=1))
    for _ in range(total - urgent - alerted):
        rows.append(_art(source=source, urgency=0))
    return rows


# ──────────────────── envelope & defensive cases ────────────────────────

class TestEnvelope:
    def test_empty_list_returns_no_data(self):
        rep = build_source_urgency_yield([], now=NOW)
        assert rep["state"] == "NO_DATA"
        assert rep["n_articles_kept"] == 0
        assert rep["n_sources"] == 0
        assert rep["sources"] == []
        assert rep["totals"]["urgent"] == 0
        # NO_DATA envelope must still expose every key the UI binds.
        for k in ("as_of", "headline", "n_noisy", "n_clean", "n_quiet",
                  "n_unknown", "totals", "sources"):
            assert k in rep

    def test_non_list_returns_no_data(self):
        for bad in (None, "string", 42, {"not": "a list"}):
            rep = build_source_urgency_yield(bad, now=NOW)  # type: ignore[arg-type]
            assert rep["state"] == "NO_DATA"

    def test_non_dict_rows_skipped(self):
        rows = _arts("rss", total=25, urgent=5)
        rows.extend([None, "string", 42])  # garbage
        rep = build_source_urgency_yield(rows, now=NOW)
        assert rep["n_articles_kept"] == 25
        # n_scanned counts the garbage rows the loop encountered too.
        assert rep["n_articles_scanned"] >= 25

    def test_zero_hours_returns_no_data(self):
        rep = build_source_urgency_yield(_arts("rss", total=50, urgent=5),
                                          hours=0, now=NOW)
        assert rep["state"] == "NO_DATA"

    def test_invalid_urgency_does_not_crash(self):
        rows = [_art(source="rss", urgency=2),
                {"source": "rss", "urgency": "garbage",
                 "first_seen": NOW.isoformat()}]
        rep = build_source_urgency_yield(rows, now=NOW)
        # The garbage row is silently dropped — the valid one is kept.
        assert rep["n_articles_kept"] == 1

    def test_missing_first_seen_silently_dropped(self):
        rows = [{"source": "rss", "urgency": 2, "first_seen": None}]
        rep = build_source_urgency_yield(rows, now=NOW)
        assert rep["state"] == "NO_DATA"

    def test_unknown_source_string_falls_back(self):
        rows = [{"source": None, "urgency": 0,
                 "first_seen": NOW.isoformat()}]
        rep = build_source_urgency_yield(rows, now=NOW, min_samples=1)
        assert len(rep["sources"]) == 1
        assert rep["sources"][0]["source"] == "(unknown)"


# ──────────────────────── window enforcement ────────────────────────────

class TestWindow:
    def test_rows_outside_window_dropped(self):
        in_window = _arts("rss", total=25, urgent=5)
        out_window = [_art(source="rss", urgency=2, age_hours=48.0)
                      for _ in range(10)]
        rep = build_source_urgency_yield(in_window + out_window,
                                          hours=24, now=NOW)
        # Only the 25 in-window rows count toward kept; 10 dropped on age.
        assert rep["n_articles_kept"] == 25

    def test_future_rows_dropped(self):
        in_window = _arts("rss", total=25, urgent=5)
        future = [_art(source="rss", urgency=2, age_hours=-5.0)
                  for _ in range(10)]
        rep = build_source_urgency_yield(in_window + future,
                                          hours=24, now=NOW)
        assert rep["n_articles_kept"] == 25


# ──────────────────── verdict policy (pinned) ───────────────────────────

class TestVerdict:
    def test_below_min_samples_is_unknown(self):
        # 15 rows < default min_samples=20.
        rows = _arts("low_count", total=15, urgent=10)
        rep = build_source_urgency_yield(rows, now=NOW)
        s = next(r for r in rep["sources"] if r["source"] == "low_count")
        assert s["verdict"] == "UNKNOWN"

    def test_no_urgent_is_quiet(self):
        rows = _arts("silent", total=50, urgent=0)
        rep = build_source_urgency_yield(rows, now=NOW)
        s = next(r for r in rep["sources"] if r["source"] == "silent")
        assert s["verdict"] == "QUIET"
        assert s["suppression_rate"] is None  # no urgent → undefined

    def test_below_urgent_floor_is_quiet(self):
        # 100 rows, 1 urgent total → urgent_rate=0.01, below the 0.02
        # floor — verdict must collapse to QUIET (a near-zero urgent rate
        # is the same operator signal as "no urgent flow at all", and
        # tagging it NOISY/CLEAN on a single row's outcome would be noise).
        rows = _arts("at_floor", total=100, urgent=1, alerted=0)
        rep = build_source_urgency_yield(rows, now=NOW)
        s = next(r for r in rep["sources"] if r["source"] == "at_floor")
        assert s["verdict"] == "QUIET"

    def test_at_urgent_floor_is_evaluated(self):
        # 50 rows, 1 urgent → urgent_rate=0.02 = floor exactly. The
        # implementation uses ``< floor`` so a rate exactly at the floor
        # is evaluated for suppression. Here 1 urgent + 0 alerted →
        # suppression=100% → NOISY.
        rows = _arts("at_floor", total=50, urgent=1, alerted=0)
        rep = build_source_urgency_yield(rows, now=NOW)
        s = next(r for r in rep["sources"] if r["source"] == "at_floor")
        assert s["verdict"] == "NOISY"

    def test_high_suppression_is_noisy(self):
        # 10 urgent, only 2 made it through → 80% suppression → NOISY.
        rows = _arts("noisy_src", total=100, urgent=8, alerted=2)
        rep = build_source_urgency_yield(rows, now=NOW)
        s = next(r for r in rep["sources"] if r["source"] == "noisy_src")
        assert s["verdict"] == "NOISY"
        assert s["urgent"] == 10
        assert s["alerted"] == 2
        assert s["suppression_rate"] == pytest.approx(0.8)

    def test_low_suppression_is_clean(self):
        # 10 urgent, 9 alerted → 10% suppression → CLEAN.
        rows = _arts("clean_src", total=100, urgent=1, alerted=9)
        rep = build_source_urgency_yield(rows, now=NOW)
        s = next(r for r in rep["sources"] if r["source"] == "clean_src")
        assert s["verdict"] == "CLEAN"

    def test_mid_band_is_mixed(self):
        # 10 urgent, 7 alerted → 30% suppression. _NOISY_SUPPRESSION=0.30
        # is >= threshold so it IS NOISY (>= comparison). Use 25% to land
        # in the mixed band (between 0.20 CLEAN ceil and 0.30 NOISY floor).
        rows = _arts("mid_src", total=100, urgent=4, alerted=12)
        rep = build_source_urgency_yield(rows, now=NOW)
        s = next(r for r in rep["sources"] if r["source"] == "mid_src")
        # 4 urgent + 12 alerted: suppression = 4/16 = 25%. MIXED.
        assert s["verdict"] == "MIXED"

    def test_thresholds_pinned(self):
        # Lock the public thresholds — if a future edit moves these, the
        # tests above will silently re-evaluate the wrong cases.
        assert _NOISY_SUPPRESSION == 0.30
        assert _CLEAN_SUPPRESSION == 0.20
        assert _URGENT_RATE_FLOOR == 0.02


# ─────────────────────────── rate math ──────────────────────────────────

class TestRateMath:
    def test_alerted_counted_as_urgent_too(self):
        # urgency=2 implies urgency>=1; both counters should include it.
        rows = _arts("src", total=50, urgent=0, alerted=10)
        rep = build_source_urgency_yield(rows, now=NOW, min_samples=1)
        s = next(r for r in rep["sources"] if r["source"] == "src")
        assert s["alerted"] == 10
        assert s["urgent"] == 10  # urgency=2 rows count as urgent too
        assert s["urgent_rate"] == pytest.approx(0.2)
        assert s["alerted_rate"] == pytest.approx(0.2)
        assert s["suppression_rate"] == pytest.approx(0.0)

    def test_aggregate_totals_reconcile(self):
        rows = (_arts("a", total=50, urgent=5, alerted=2)
                + _arts("b", total=30, urgent=3, alerted=3)
                + _arts("c", total=20, urgent=0))
        rep = build_source_urgency_yield(rows, now=NOW, min_samples=1)
        assert rep["totals"]["articles"] == 100
        assert rep["totals"]["urgent"] == 13   # 7 (a) + 6 (b)
        assert rep["totals"]["alerted"] == 5   # 2 + 3
        # The aggregate totals reflect every kept row, not just the
        # top_sources display rows.

    def test_suppression_rate_none_when_no_urgent(self):
        rows = _arts("idle", total=100, urgent=0)
        rep = build_source_urgency_yield(rows, now=NOW, min_samples=1)
        s = next(r for r in rep["sources"] if r["source"] == "idle")
        assert s["suppression_rate"] is None


# ──────────────────────── ranking & cap ────────────────────────────────

class TestRanking:
    def test_noisy_sources_rank_before_clean(self):
        rows = (_arts("clean_src", total=100, urgent=2, alerted=9)
                + _arts("noisy_src", total=100, urgent=8, alerted=2))
        rep = build_source_urgency_yield(rows, now=NOW)
        verdicts = [r["verdict"] for r in rep["sources"]]
        assert verdicts.index("NOISY") < verdicts.index("CLEAN")

    def test_top_sources_cap_truncates_display(self):
        rows = []
        for i in range(8):
            rows.extend(_arts(f"src_{i}", total=50, urgent=5, alerted=4))
        rep = build_source_urgency_yield(rows, now=NOW, top_sources=3)
        assert len(rep["sources"]) == 3
        # The pre-cap source count is preserved on the envelope.
        assert rep["n_sources"] == 8

    def test_alphabetical_tiebreak_for_card_stability(self):
        # Two sources with identical verdict and urgent count must order
        # alphabetically so the operator UI is byte-stable across runs.
        rows = (_arts("zeta", total=100, urgent=5, alerted=4)
                + _arts("alpha", total=100, urgent=5, alerted=4))
        rep = build_source_urgency_yield(rows, now=NOW)
        srcs = [r["source"] for r in rep["sources"]
                if r["source"] in ("alpha", "zeta")]
        assert srcs == ["alpha", "zeta"]


# ─────────────────────── headline & verdict counts ───────────────────────

class TestHeadline:
    def test_sparse_headline_when_few_articles(self):
        rows = _arts("rss", total=10, urgent=2, alerted=1)
        rep = build_source_urgency_yield(rows, now=NOW)
        assert rep["state"] == "SPARSE"
        assert "Sparse" in rep["headline"]

    def test_noisy_headline_when_any_noisy_source(self):
        rows = _arts("noisy_src", total=100, urgent=8, alerted=2)
        rep = build_source_urgency_yield(rows, now=NOW)
        assert rep["state"] == "STABLE"
        assert "NOISY" in rep["headline"]
        assert "noisy_src" in rep["headline"]

    def test_clean_headline_when_no_noisy(self):
        rows = _arts("clean_src", total=100, urgent=1, alerted=9)
        rep = build_source_urgency_yield(rows, now=NOW)
        assert rep["state"] == "STABLE"
        assert "CLEAN" in rep["headline"]

    def test_verdict_counts_tally(self):
        rows = (_arts("a_noisy", total=100, urgent=8, alerted=2)
                + _arts("b_clean", total=100, urgent=1, alerted=9)
                + _arts("c_quiet", total=100, urgent=0)
                + _arts("d_low", total=10, urgent=5))  # below min_samples
        rep = build_source_urgency_yield(rows, now=NOW)
        assert rep["n_noisy"] == 1
        assert rep["n_clean"] == 1
        assert rep["n_quiet"] == 1
        assert rep["n_unknown"] == 1


# ────────────────────────── parity guards ──────────────────────────────

class TestEnvelopeKeyStability:
    """The UI binding loads the JSON shape — silent key removal would
    break the operator panel. Mirrors the envelope-stability discipline of
    test_briefing_coverage_audit / test_news_arrival_rhythm."""

    EXPECTED = {
        "as_of", "state", "headline", "window_hours", "min_samples",
        "top_sources_cap", "n_articles_scanned", "n_articles_kept",
        "n_sources", "n_noisy", "n_clean", "n_quiet", "n_unknown",
        "totals", "sources",
    }

    def test_no_data(self):
        rep = build_source_urgency_yield([], now=NOW)
        assert set(rep.keys()) == self.EXPECTED

    def test_sparse(self):
        rep = build_source_urgency_yield(
            _arts("rss", total=5, urgent=1), now=NOW,
        )
        assert set(rep.keys()) == self.EXPECTED

    def test_stable(self):
        rep = build_source_urgency_yield(
            _arts("rss", total=100, urgent=5, alerted=4), now=NOW,
        )
        assert set(rep.keys()) == self.EXPECTED

    def test_source_row_key_stability(self):
        rep = build_source_urgency_yield(
            _arts("rss", total=100, urgent=5, alerted=4), now=NOW,
        )
        assert rep["sources"]
        row_keys = set(rep["sources"][0].keys())
        assert row_keys == {
            "source", "total", "urgent", "alerted",
            "urgent_rate", "alerted_rate", "suppression_rate", "verdict",
        }


class TestBacktestIsolationDocumented:
    """The builder itself accepts whatever the caller hands it — backtest
    isolation lives in the SQL adapter that pulls article rows. This test
    documents the contract: the builder does not filter; if a caller
    skipped ``_LIVE_ONLY_CLAUSE`` and passed a backtest row, the builder
    would happily score it. The dashboard/route layer enforces the
    invariant (verified in test_dashboard_endpoints if/when the route is
    added)."""

    def test_builder_does_not_filter_source_strings(self):
        rows = [_art(source="backtest_run_1", urgency=2)] * 25
        rep = build_source_urgency_yield(rows, now=NOW)
        # The builder counts it — it is the SQL adapter's job to exclude.
        assert rep["n_articles_kept"] == 25
        assert any(r["source"] == "backtest_run_1" for r in rep["sources"])

    def test_default_min_samples_value_pinned(self):
        # Verdict policy depends on min_samples; lock the default.
        assert DEFAULT_MIN_SAMPLES == 20
