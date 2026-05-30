"""Tests for ``analytics.never_delivered_collector_audit``.

Pins:
  - verdict ladder transitions on the per-worker classification
    (NEVER_DELIVERED / SILENT_24H / SLOW / DELIVERING)
  - aggregate roll-up: ``HEALTHY`` / ``FEW_DEGRADED`` / ``WIDESPREAD_SILENCE``
    (the latter ALWAYS fires on any NEVER_DELIVERED hit — the high-severity
    case the freshness monitor cannot see)
  - source-prefix matching is startswith, case-sensitive
  - load-bearing invariants: the live entrypoint applies ``_LIVE_ONLY_CLAUSE``
    so backtest rows never reach the verdict; the builder is pure (no DB
    mutation)
  - envelope-keys freeze (drift-lock)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from analytics.never_delivered_collector_audit import (
    EXPECTED_COLLECTORS,
    _aggregate_verdict,
    _classify_one,
    audit,
)


# A fixed "now" so all relative-time assertions are deterministic.
NOW = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)


def _fs(hours_ago: float) -> str:
    """ISO-8601 first_seen string ``hours_ago`` hours before NOW. Matches
    the format the storage layer writes (microseconds + tz suffix)."""
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _row(source: str, hours_ago: float) -> dict:
    return {"source": source, "first_seen": _fs(hours_ago)}


# ---------------------------------------------------------------------------
# Registry / sanity
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_expected_collectors_includes_un_news(self):
        """The fixed worker is pinned at the top of the registry — the
        canonical NEVER_DELIVERED case."""
        names = {row[0] for row in EXPECTED_COLLECTORS}
        assert "un_news" in names

    def test_registry_floors_are_positive(self):
        for name, prefix, floor in EXPECTED_COLLECTORS:
            assert floor >= 1, f"min_rows_24h for {name!r} must be positive"
            assert prefix, f"empty source_prefix for {name!r}"


# ---------------------------------------------------------------------------
# Per-collector classification
# ---------------------------------------------------------------------------

class TestClassifyOne:
    def test_zero_rows_anywhere_is_never_delivered(self):
        """The exact failure shape of the un_news bug — worker registered
        but no row was ever written."""
        out = _classify_one(rows=[], prefix="un_", min_rows_24h=6,
                            cutoff_24h=NOW - timedelta(hours=24))
        assert out["verdict"] == "NEVER_DELIVERED"
        assert out["n_total"] == 0
        assert out["n_24h"] == 0
        assert out["last_first_seen"] is None

    def test_old_rows_only_is_silent_24h(self):
        """Historically delivered but quiet for >24h."""
        rows = [_row("un_econ_dev", hours_ago=72)]
        out = _classify_one(rows, prefix="un_", min_rows_24h=6,
                            cutoff_24h=NOW - timedelta(hours=24))
        assert out["verdict"] == "SILENT_24H"
        assert out["n_total"] == 1
        assert out["n_24h"] == 0

    def test_some_rows_in_window_below_floor_is_slow(self):
        """3 rows delivered in 24h, floor is 6 -> SLOW."""
        rows = [
            _row("un_econ_dev", hours_ago=1.0),
            _row("un_climate", hours_ago=2.0),
            _row("un_americas", hours_ago=10.0),
        ]
        out = _classify_one(rows, prefix="un_", min_rows_24h=6,
                            cutoff_24h=NOW - timedelta(hours=24))
        assert out["verdict"] == "SLOW"
        assert out["n_24h"] == 3

    def test_at_floor_is_delivering(self):
        """Floor is INCLUSIVE — ``n_24h >= min_rows_24h`` -> DELIVERING."""
        rows = [_row("un_climate", hours_ago=i + 0.5) for i in range(6)]
        out = _classify_one(rows, prefix="un_", min_rows_24h=6,
                            cutoff_24h=NOW - timedelta(hours=24))
        assert out["verdict"] == "DELIVERING"
        assert out["n_24h"] == 6

    def test_above_floor_is_delivering(self):
        rows = [_row("un_climate", hours_ago=i + 0.1) for i in range(15)]
        out = _classify_one(rows, prefix="un_", min_rows_24h=6,
                            cutoff_24h=NOW - timedelta(hours=24))
        assert out["verdict"] == "DELIVERING"
        assert out["n_24h"] == 15

    def test_prefix_is_startswith_not_substring(self):
        """``startswith('un_')`` matches ``un_econ_dev`` but NOT ``foo_un_bar``.
        Substring matching would false-positive on aggregator tags that
        happen to contain the prefix."""
        rows = [
            _row("un_econ_dev", hours_ago=1),
            _row("foo_un_bar", hours_ago=1),
            _row("UN_OTHER", hours_ago=1),  # different case — not a match
        ]
        out = _classify_one(rows, prefix="un_", min_rows_24h=1,
                            cutoff_24h=NOW - timedelta(hours=24))
        # Only the un_econ_dev row matches; n_24h is 1.
        assert out["n_24h"] == 1
        assert out["verdict"] == "DELIVERING"

    def test_malformed_source_does_not_raise(self):
        """A row whose ``source`` is None / int / missing must be skipped,
        not crash the audit. Defensive against future schema drift."""
        rows = [
            {"source": None, "first_seen": _fs(1)},
            {"first_seen": _fs(1)},  # no 'source' key
            {"source": 42, "first_seen": _fs(1)},  # wrong type
            _row("un_climate", hours_ago=1),
        ]
        out = _classify_one(rows, prefix="un_", min_rows_24h=1,
                            cutoff_24h=NOW - timedelta(hours=24))
        assert out["n_24h"] == 1
        assert out["verdict"] == "DELIVERING"

    def test_unparseable_first_seen_skipped_for_window_count_kept_for_total(self):
        """An unparseable ``first_seen`` (legacy / corrupt row) must NOT
        crash. It contributes to ``n_total`` but not ``n_24h``."""
        rows = [
            {"source": "un_climate", "first_seen": "not-a-date"},
            {"source": "un_climate", "first_seen": ""},
            _row("un_climate", hours_ago=1),
        ]
        out = _classify_one(rows, prefix="un_", min_rows_24h=1,
                            cutoff_24h=NOW - timedelta(hours=24))
        assert out["n_total"] == 3
        assert out["n_24h"] == 1
        assert out["verdict"] == "DELIVERING"

    def test_last_first_seen_is_max(self):
        rows = [
            _row("un_climate", hours_ago=10),
            _row("un_climate", hours_ago=2),  # most recent
            _row("un_climate", hours_ago=5),
        ]
        out = _classify_one(rows, prefix="un_", min_rows_24h=1,
                            cutoff_24h=NOW - timedelta(hours=24))
        assert out["last_first_seen"] == _fs(2)


# ---------------------------------------------------------------------------
# Aggregate verdict
# ---------------------------------------------------------------------------

class TestAggregateVerdict:
    def _mk(self, verdicts: list[str]) -> dict[str, dict]:
        return {f"w{i}": {"verdict": v} for i, v in enumerate(verdicts)}

    def test_empty_is_no_data(self):
        assert _aggregate_verdict({}) == "NO_DATA"

    def test_all_delivering_is_healthy(self):
        assert _aggregate_verdict(
            self._mk(["DELIVERING", "DELIVERING", "DELIVERING"])
        ) == "HEALTHY"

    def test_one_slow_is_few_degraded(self):
        assert _aggregate_verdict(
            self._mk(["DELIVERING", "SLOW", "DELIVERING"])
        ) == "FEW_DEGRADED"

    def test_two_degraded_is_few_degraded(self):
        assert _aggregate_verdict(
            self._mk(["DELIVERING", "SLOW", "SILENT_24H"])
        ) == "FEW_DEGRADED"

    def test_three_degraded_is_widespread(self):
        assert _aggregate_verdict(
            self._mk(["SLOW", "SLOW", "SLOW", "DELIVERING"])
        ) == "WIDESPREAD_SILENCE"

    def test_any_never_delivered_is_widespread(self):
        """The bug-this-module-pins case. A single NEVER_DELIVERED collector
        is the high-severity verdict: it cannot be seen by the freshness
        monitor, so the analyst should be told immediately."""
        assert _aggregate_verdict(
            self._mk(["NEVER_DELIVERED", "DELIVERING", "DELIVERING"])
        ) == "WIDESPREAD_SILENCE"

    def test_never_delivered_beats_few_degraded(self):
        """Even when only one collector is degraded total, NEVER_DELIVERED
        escalates above FEW_DEGRADED."""
        assert _aggregate_verdict(
            self._mk(["NEVER_DELIVERED", "DELIVERING"])
        ) == "WIDESPREAD_SILENCE"


# ---------------------------------------------------------------------------
# Full audit envelope
# ---------------------------------------------------------------------------

class TestAudit:
    REG = (
        ("un_news", "un_", 6),
        ("rss", "rss", 50),
        ("web", "scraped/", 50),
    )

    def test_empty_pool_yields_no_data_verdict(self):
        out = audit([], registry=self.REG, now=NOW)
        assert out["verdict"] == "NO_DATA"
        assert out["n_pool_rows"] == 0
        assert out["n_never_delivered"] == 0

    def test_un_news_bug_signature(self):
        """The exact shape of the live un_news bug pre-fix: rss + web
        delivering, un_news NEVER_DELIVERED. Aggregate must escalate to
        WIDESPREAD_SILENCE."""
        rows = (
            [_row("rss/seeking_alpha", i * 0.1) for i in range(60)]
            + [_row("scraped/yahoo.com", i * 0.1) for i in range(80)]
            # NO un_* rows at all.
        )
        out = audit(rows, registry=self.REG, now=NOW)
        assert out["verdict"] == "WIDESPREAD_SILENCE"
        assert out["n_never_delivered"] == 1
        assert out["by_worker"]["un_news"]["verdict"] == "NEVER_DELIVERED"
        assert out["by_worker"]["rss"]["verdict"] == "DELIVERING"
        assert out["by_worker"]["web"]["verdict"] == "DELIVERING"
        assert out["by_worker"]["un_news"]["n_24h"] == 0
        assert out["by_worker"]["rss"]["n_24h"] == 60

    def test_healthy_steady_state(self):
        """All three collectors above floor — analyst-facing HEALTHY."""
        rows = (
            [_row("un_climate", i * 1.0) for i in range(8)]
            + [_row("rss/x", i * 0.1) for i in range(60)]
            + [_row("scraped/yahoo.com", i * 0.1) for i in range(80)]
        )
        out = audit(rows, registry=self.REG, now=NOW)
        assert out["verdict"] == "HEALTHY"
        assert out["n_never_delivered"] == 0
        assert out["n_silent_24h"] == 0
        assert out["n_slow"] == 0
        assert out["n_delivering"] == 3

    def test_envelope_keys_frozen(self):
        """Drift-lock: keys consumers (operator dashboards, future
        agents) rely on must stay stable."""
        out = audit([], registry=self.REG, now=NOW)
        expected_keys = {
            "generated_at", "window_h", "n_pool_rows", "by_worker",
            "n_never_delivered", "n_silent_24h", "n_slow", "n_delivering",
            "verdict",
        }
        assert set(out.keys()) == expected_keys
        # Each by_worker block freezes its own key set as well.
        out2 = audit([_row("un_climate", 1)], registry=self.REG, now=NOW)
        per = out2["by_worker"]["un_news"]
        assert set(per.keys()) == {
            "source_prefix", "n_total", "n_24h", "min_rows_24h",
            "last_first_seen", "verdict",
        }

    def test_window_h_threads_through(self):
        """Passing window_h=1 means a row 2h ago is OUT of window."""
        rows = [_row("un_climate", hours_ago=2.0)]
        out_24h = audit(rows, registry=self.REG, now=NOW, window_h=24)
        out_1h = audit(rows, registry=self.REG, now=NOW, window_h=1)
        # In 24h: 1 row in window (SLOW: 1 < floor 6)
        assert out_24h["by_worker"]["un_news"]["n_24h"] == 1
        assert out_24h["by_worker"]["un_news"]["verdict"] == "SLOW"
        # In 1h: 0 rows in window but n_total>0 -> SILENT_24H
        assert out_1h["by_worker"]["un_news"]["n_24h"] == 0
        assert out_1h["by_worker"]["un_news"]["verdict"] == "SILENT_24H"
        assert out_1h["window_h"] == 1

    def test_audit_is_pure_no_mutation(self):
        """The builder must not mutate inputs (rows, registry). Tests that
        re-running the audit with the same inputs yields identical output."""
        rows = [_row("un_climate", 1.0), _row("rss/x", 1.0)]
        rows_before = [dict(r) for r in rows]
        a = audit(rows, registry=self.REG, now=NOW)
        b = audit(rows, registry=self.REG, now=NOW)
        # rows are unchanged
        assert rows == rows_before
        # output is deterministic
        assert a == b

    def test_naive_now_normalised_to_utc(self):
        """Passing a naive datetime must not crash; the audit normalises
        to UTC. Catches the train/serve skew that bit
        ``ml.features._parse_published``."""
        naive_now = NOW.replace(tzinfo=None)
        out = audit([_row("un_climate", 1.0)], registry=self.REG, now=naive_now)
        # Should classify the row identically to the aware-NOW path.
        assert out["by_worker"]["un_news"]["n_24h"] == 1


# ---------------------------------------------------------------------------
# Load-bearing invariants — live entrypoint
# ---------------------------------------------------------------------------

class TestLoadBearingInvariants:
    """The audit must NEVER mutate score columns and the live entrypoint
    must apply ``_LIVE_ONLY_CLAUSE`` so synthetic backtest rows are
    excluded by construction."""

    def test_no_score_mutation_in_builder(self):
        """The pure builder reads only ``source`` and ``first_seen``. Any
        change that introduced ai_score / ml_score / score_source / urgency
        writes here would be caught by an inspection on the module body —
        this test is the explicit assertion that those columns are NOT
        referenced in the module's source."""
        import inspect

        import analytics.never_delivered_collector_audit as mod

        src = inspect.getsource(mod)
        # The audit MUST NOT write any score column. Substring search
        # catches even a typo'd misuse of the score-mutation surface.
        for forbidden in (
            "update_ai_scores_batch",
            "update_ml_scores_batch",
            "mark_alerted",
            "UPDATE articles",
            "INSERT INTO articles",
        ):
            assert forbidden not in src, (
                f"never_delivered_collector_audit must not call/write "
                f"{forbidden!r} — it is a pure read-side audit"
            )

    def test_live_entrypoint_applies_live_only_clause(self):
        """The live entrypoint must reference ``_LIVE_ONLY_CLAUSE`` so
        synthetic ``backtest://`` / ``backtest_*`` / ``opus_annotation*``
        rows are excluded before they reach the builder. This is the
        documented anti-drift discipline (CLAUDE.md §5)."""
        import inspect

        import analytics.never_delivered_collector_audit as mod

        src = inspect.getsource(mod)
        assert "_LIVE_ONLY_CLAUSE" in src, (
            "never_delivered_collector_audit.audit_live must apply "
            "_LIVE_ONLY_CLAUSE to its SQL pull — synthetic backtest rows "
            "must never reach the verdict"
        )

    def test_backtest_rows_in_pool_dont_corrupt_verdict(self):
        """Even if a backtest row somehow leaks into the pool (e.g. caller
        forgot to filter), the audit must classify it under its source-tag
        prefix — and the test asserts that a leaked synthetic row WOULD
        NOT alter the verdict for any registered worker (because
        ``EXPECTED_COLLECTORS`` prefixes don't include backtest_ / opus_)."""
        rows = [
            _row("backtest_run_42_winner", 1.0),  # synthetic — should not match
            _row("opus_annotation_cycle_3", 1.0),  # synthetic — should not match
            _row("un_climate", 1.0),
        ]
        out = audit(rows, registry=(
            ("un_news", "un_", 1),  # floor 1 so the one un_ row is enough
        ), now=NOW)
        assert out["by_worker"]["un_news"]["verdict"] == "DELIVERING"
        assert out["by_worker"]["un_news"]["n_24h"] == 1
        # The two synthetic rows are in the pool but matched no prefix —
        # so they don't bump any worker's count.


# ---------------------------------------------------------------------------
# Live audit entrypoint — in-memory SQLite end-to-end
# ---------------------------------------------------------------------------

class TestAuditLive:
    """Exercises ``audit_live`` against a real in-memory ArticleStore. The
    test inserts a mix of (live, synthetic) rows and asserts:

      * synthetic backtest rows are excluded by ``_LIVE_ONLY_CLAUSE``;
      * un_news rows (via the fixed worker's collector) reach the
        DELIVERING verdict;
      * window cutoff is honoured (rows older than ``hours`` are out).
    """

    def test_live_audit_excludes_backtest_rows(self, store):
        """``store`` fixture (conftest.py) redirects the DB to a tmp path so
        we don't touch the live USB-backed production DB."""
        now = datetime.now(timezone.utc)

        # 1) Insert a healthy un_news row.
        good = {
            "title": "UN: economic outlook tightens for emerging markets",
            "link": "https://news.un.org/en/story/2026/05/abc123",
            "source": "un_econ_dev",
            "published": now.isoformat(),
            "summary": "x",
        }
        # 2) Insert a synthetic backtest_ row (must be excluded).
        bad_bt = {
            "title": "[BACKTEST] sim_2026-04-15 winner",
            "link": "backtest://run_42/sim_2026-04-15/BUY/MU",
            "source": "backtest_run_42_winner",
            "published": now.isoformat(),
            "summary": "x",
        }
        store.insert_batch([good, bad_bt])
        # Sanity: backtest row IS in the DB.
        n_bt = store.conn.execute(
            "SELECT COUNT(*) FROM articles WHERE source LIKE 'backtest_%'"
        ).fetchone()[0]
        assert n_bt == 1, "synthetic backtest row must be present in raw DB"

        report = audit_live_local(store, hours=24)
        # Pool MUST exclude the backtest row.
        assert report["n_pool_rows"] == 1
        un_block = report["by_worker"]["un_news"]
        # Floor in EXPECTED_COLLECTORS is 6, so a single row reads as SLOW
        # — exactly the calibration we want (silent enough to flag).
        assert un_block["n_24h"] == 1
        assert un_block["verdict"] == "SLOW"


def audit_live_local(store, hours=24, **kwargs):
    """Local re-binding so the test imports the live entrypoint with the
    same name everywhere. Pytest fixtures sometimes shadow module imports
    across test classes — keeping a tight inline alias avoids ambiguity."""
    from analytics.never_delivered_collector_audit import audit_live as _al
    return _al(store, hours=hours, **kwargs)
