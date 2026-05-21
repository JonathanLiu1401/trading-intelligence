"""Label-calibration line in the 5h heartbeat briefing.

The briefing already surfaces source-health and book-coverage. The
aggregate that was missing — and that the analyst persona is blind to
without it — is *how much of this window's urgent stream carried a real
LLM ground-truth label* vs only an unverified model self-prediction. Per
``ArticleStore.urgency_label_split``'s docstring + the 2026-05-19 live
finding (every urgent row in a 6h window was ``score_source='ml'``), a
near-zero ``llm_fraction`` means the Sonnet ``urgency_scorer`` is dark /
quota-throttled / flooring everything to noise — exactly the case the
per-row "[unverified — model-only urgent]" tag hedges individually but no
surface aggregated.

``daemon._format_label_calibration`` is pure and deterministic — exact
strings pinned here. Mirrors the silence-on-healthy discipline of
``_format_source_health_summary`` / ``_format_portfolio_coverage`` (quiet
or healthy → ``""``; only an actionable miscalibration emits a line).
Verdict thresholds mirror ``/api/urgent-label-split`` byte-for-byte
(``dashboard/web_server.py``) so the briefing and dashboard cannot drift.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import daemon


def _recent(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_raw(
    store, *, id, url, title, source, urgency=1, ai_score=0.0,
    ml_score=None, score_source=None, kw_score=1.0, first_seen=None,
):
    """Build any (urgency, score_source) state without going through scoring."""
    if first_seen is None:
        first_seen = _recent()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, urgency,
             first_seen, 0, ml_score, score_source),
        )
        store.conn.commit()


class TestQuietAndHealthy:
    def test_empty_store_returns_empty_string(self, store):
        """No urgent rows in window → silent (briefing stays clean)."""
        assert daemon._format_label_calibration(store, hours=5) == ""

    def test_healthy_majority_llm_vetted_returns_empty(self, store):
        """8 LLM-vetted + 2 ML-only = 80% LLM → healthy → silent."""
        for i in range(8):
            _insert_raw(
                store, id=f"l{i}", url=f"https://reuters.com/{i}",
                title=f"LLM-vetted urgent number {i}",
                source="rss", urgency=1, ai_score=9.0, score_source="llm",
            )
        for i in range(2):
            _insert_raw(
                store, id=f"m{i}", url=f"https://x.com/m{i}",
                title=f"Model-only urgent number {i}",
                source="GN: Nvidia", urgency=1, ai_score=0.0,
                ml_score=9.0, score_source="ml",
            )
        assert daemon._format_label_calibration(store, hours=5) == ""

    def test_briefing_boost_counts_as_vetted(self, store):
        """``briefing_boost`` is Opus-curated, same authority as ``llm``."""
        for i in range(3):
            _insert_raw(
                store, id=f"b{i}", url=f"https://x.com/b{i}",
                title=f"Opus briefing-boost number {i}",
                source="rss", urgency=1, ai_score=4.5,
                score_source="briefing_boost",
            )
        # 1 ML-only — far below the 50% threshold but only 1 total, so the
        # mostly_unverified gate (total >= 5) blocks the line entirely.
        _insert_raw(
            store, id="m0", url="https://x.com/m0",
            title="single ml urgent row", source="GN: x",
            urgency=1, ai_score=0.0, ml_score=9.0, score_source="ml",
        )
        # total=4, briefing_boost=3 + ml=1 → llm_fraction=0.75 → healthy.
        assert daemon._format_label_calibration(store, hours=5) == ""


class TestUnverifiedStorm:
    def test_zero_llm_total_three_emits_storm_line(self, store):
        """3+ urgent rows, 0% LLM-vetted → Sonnet-dark storm."""
        for i in range(3):
            _insert_raw(
                store, id=f"m{i}", url=f"https://x.com/m{i}",
                title=f"Model-only urgent number {i}",
                source="GN: Nvidia", urgency=1, ai_score=0.0,
                ml_score=9.0, score_source="ml",
            )
        line = daemon._format_label_calibration(store, hours=5)
        assert line == (
            "🔬 Urgent calibration: 0% LLM-vetted last 5h "
            "(3/3 ML-only) — Sonnet scorer dark"
        )

    def test_zero_llm_total_two_does_not_fire(self, store):
        """Only 2 urgent rows is below the storm threshold (>=3) → silent."""
        for i in range(2):
            _insert_raw(
                store, id=f"m{i}", url=f"https://x.com/m{i}",
                title=f"Model-only urgent number {i}",
                source="GN: Nvidia", urgency=1, ai_score=0.0,
                ml_score=9.0, score_source="ml",
            )
        assert daemon._format_label_calibration(store, hours=5) == ""


class TestMostlyUnverified:
    def test_minority_llm_total_five_emits_line(self, store):
        """1 LLM + 4 ML across 5 urgent rows → 20% LLM-vetted → emits."""
        _insert_raw(
            store, id="l0", url="https://reuters.com/0",
            title="LLM-vetted urgent", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        for i in range(4):
            _insert_raw(
                store, id=f"m{i}", url=f"https://x.com/m{i}",
                title=f"Model-only urgent number {i}",
                source="GN: Nvidia", urgency=1, ai_score=0.0,
                ml_score=9.0, score_source="ml",
            )
        assert daemon._format_label_calibration(store, hours=5) == (
            "🔬 Urgent calibration: 20% LLM-vetted last 5h "
            "(4/5 ML-only)"
        )

    def test_minority_llm_total_four_does_not_fire(self, store):
        """Only 4 rows is below the mostly_unverified threshold (>=5)
        — the dashboard verdict ladder skips this case too, matched here."""
        _insert_raw(
            store, id="l0", url="https://reuters.com/0",
            title="LLM-vetted urgent", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        for i in range(3):
            _insert_raw(
                store, id=f"m{i}", url=f"https://x.com/m{i}",
                title=f"Model-only urgent number {i}",
                source="GN: Nvidia", urgency=1, ai_score=0.0,
                ml_score=9.0, score_source="ml",
            )
        assert daemon._format_label_calibration(store, hours=5) == ""


class TestBacktestIsolation:
    def test_synthetic_rows_never_inflate_calibration(self, store):
        """Synthetic backtest:// / backtest_* / opus_annotation* rows are
        excluded by ``urgency_label_split``'s ``_LIVE_ONLY_CLAUSE`` — they
        must not appear in either the numerator or denominator.

        Without the upstream filter, the synthetic rows below would create a
        spurious '0% LLM-vetted, 6/6 ML-only' storm where the real signal is
        actually empty (no live urgent rows at all → quiet, return '').
        """
        # 6 synthetic ML-only urgent rows — across all three exclusion shapes.
        for i in range(2):
            _insert_raw(
                store, id=f"bt{i}", url=f"backtest://run_1/{i}",
                title=f"backtest synthetic {i}", source="rss",
                urgency=1, ai_score=0.0, ml_score=9.0, score_source="ml",
            )
        for i in range(2):
            _insert_raw(
                store, id=f"bs{i}", url=f"https://x.com/bs{i}",
                title=f"backtest source tag {i}",
                source=f"backtest_run_{i}",
                urgency=1, ai_score=0.0, ml_score=9.0, score_source="ml",
            )
        for i in range(2):
            _insert_raw(
                store, id=f"oa{i}", url=f"https://x.com/oa{i}",
                title=f"opus annotation {i}",
                source="opus_annotation_cycle_5",
                urgency=1, ai_score=0.0, ml_score=9.0, score_source="ml",
            )
        # Live store: no urgent rows pass the live-only filter → quiet → "".
        assert daemon._format_label_calibration(store, hours=5) == ""


class TestDegradation:
    def test_store_raising_returns_empty_string(self, store):
        """A metric-side failure must never block the briefing — fall back to
        silence so the rest of the message still posts."""
        class _BoomStore:
            def urgency_label_split(self, hours: int = 24):
                raise RuntimeError("db locked")

        assert daemon._format_label_calibration(_BoomStore(), hours=5) == ""

    def test_non_dict_total_treated_as_quiet(self):
        """A malformed return value (e.g. test stub) collapses to silence
        instead of raising or producing junk output."""
        class _NoneStore:
            def urgency_label_split(self, hours: int = 24):
                return {"total": None, "llm_fraction": None, "by_source": None}

        assert daemon._format_label_calibration(_NoneStore(), hours=5) == ""


class TestRespectsHoursArg:
    def test_hours_window_propagates_to_store(self):
        """The ``hours`` argument is passed verbatim to
        ``urgency_label_split`` — a future briefing window change (e.g. an
        adaptive lookback) must not silently still hit the default 5h."""
        calls = []

        class _ProbeStore:
            def urgency_label_split(self, hours: int = 24):
                calls.append(hours)
                return {"total": 0, "llm_fraction": 0.0,
                        "by_source": {"llm": 0, "ml": 0,
                                      "briefing_boost": 0, "null": 0}}

        daemon._format_label_calibration(_ProbeStore(), hours=24)
        assert calls == [24]


class TestCharCap:
    def test_max_chars_truncates_with_ellipsis(self, store):
        """Defense-in-depth: if a future verdict template runs over the cap,
        the line is truncated with an ellipsis (same discipline as
        ``_format_source_health_summary``)."""
        for i in range(3):
            _insert_raw(
                store, id=f"m{i}", url=f"https://x.com/m{i}",
                title=f"Model-only urgent number {i}",
                source="GN: Nvidia", urgency=1, ai_score=0.0,
                ml_score=9.0, score_source="ml",
            )
        out = daemon._format_label_calibration(store, hours=5, max_chars=40)
        assert len(out) <= 40
        assert out.endswith("…")
        assert out.startswith("🔬 Urgent calibration:")
