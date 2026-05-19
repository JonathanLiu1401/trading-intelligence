"""``analytics.recap_template_audit`` — calibration view of the recap gate.

The audit answers the analyst-facing "is the recap pre-filter still working?"
question by counting recap-template-matching rows in the recent window by
their current state. The discriminating asserts:

  1. ``leaked_to_strong_pool`` (recap + score_source='llm' + ai_score>=8) is
     the load-bearing regression signal — a single such row means the
     ``urgency_scorer`` pre-filter let one through and the trainer is now
     ingesting an urgent SEO recap as ground-truth.
  2. ``floored_to_noise`` (recap + ai_score<=0.5) is the "pre-filter worked"
     counterweight — these were caught.
  3. ``leaked_urgent`` (recap + urgency>=1) catches the alert-gate
     downstream-failure case (gate failed even if Sonnet labeled correctly).
  4. Backtest-isolation: synthetic ``backtest://`` / ``backtest_*`` /
     ``opus_annotation*`` rows whose title would match a recap fingerprint
     do NOT inflate any count (the metric must measure the LIVE pool, never
     synthetic injections).
  5. ``ok`` is True iff ``leaked_to_strong_pool == 0`` — the verdict the
     daemon healthcheck / dashboard should key on.
  6. The audit's ``LIVE_ONLY_CLAUSE`` constant matches the storage layer's
     verbatim — drift here is the same class of silent regression as the
     dashboard-parity backtest-isolation tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from analytics import recap_template_audit
from storage import article_store


def _insert(
    store, *, id, title, source="rss", url=None,
    ai_score=0.0, ml_score=None, score_source=None,
    urgency=0, first_seen=None, kw_score=1.0,
):
    if first_seen is None:
        first_seen = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat()
    if url is None:
        url = f"https://news.example.com/{id}"
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


# ── Verdict shape ───────────────────────────────────────────────────────────


class TestVerdictShape:
    def test_empty_store_is_ok(self, store):
        report = recap_template_audit.audit(store)
        assert report["window_h"] == 24
        assert report["total_recap_rows"] == 0
        assert report["leaked_to_strong_pool"] == 0
        assert report["leaked_urgent"] == 0
        assert report["floored_to_noise"] == 0
        assert report["leak_fraction"] == 0.0
        assert report["ok"] is True
        # Every fingerprint key present even on empty input — dashboard
        # rendering depends on a stable shape (mirrors urgency_label_split's
        # always-4-bucket discipline).
        for name in (
            "why_trading_today", "why_did_stock", "market_today_dated",
            "earnings_call_recap", "street_thinks", "gf_value_says",
        ):
            assert name in report["by_fingerprint"]
            assert name in report["leaked_by_fingerprint"]


# ── Strong-pool leak (the load-bearing regression signal) ───────────────────


class TestStrongPoolLeak:
    def test_single_llm_urgent_recap_is_a_leak(self, store):
        """The exact live failure: Sonnet labeled a recap row ai_score=8.0
        score_source='llm'. The audit MUST surface this row in
        ``leaked_to_strong_pool`` and set ``ok=False``."""
        _insert(
            store, id="poison",
            title="Why Did Micron Stock Drop Today ? | The Motley Fool",
            source="Motley Fool", ai_score=8.0, score_source="llm",
            urgency=2,
        )
        report = recap_template_audit.audit(store)
        assert report["leaked_to_strong_pool"] == 1
        assert report["leaked_by_fingerprint"]["why_did_stock"] == 1
        assert report["ok"] is False, (
            "the regression signal the audit exists to surface was missed"
        )

    def test_clean_pool_post_prefilter_is_ok(self, store):
        """Post-fix state: the pre-filter floored every recap row to 0.01
        score_source='llm'. ``leaked_to_strong_pool`` is 0 → ``ok=True``."""
        for i, t in enumerate([
            "Why Nvidia (NVDA) Stock Is Trading Up Today",
            "Stock Market Today, May 18: Micron Falls",
        ]):
            _insert(
                store, id=f"clean{i}", title=t,
                ai_score=0.01, score_source="llm", urgency=0,
            )
        report = recap_template_audit.audit(store)
        assert report["total_recap_rows"] == 2
        assert report["floored_to_noise"] == 2, (
            "pre-filter-floored rows missing from floored_to_noise"
        )
        assert report["leaked_to_strong_pool"] == 0
        assert report["ok"] is True

    def test_briefing_boost_score_does_not_count_as_leak(self, store):
        """A row tagged ``score_source='briefing_boost'`` is an Opus
        curation nudge — those carry ai_score=4.5 and so cannot satisfy
        the >=8 leak threshold by construction. Pin this so a future
        regression that promotes a recap row through update_scores_from_labels
        doesn't quietly flip the audit verdict."""
        _insert(
            store, id="bb",
            title="Why Did AMD Stock Surge Today",
            source="rss", ai_score=4.5, score_source="briefing_boost",
            urgency=0,
        )
        report = recap_template_audit.audit(store)
        assert report["total_recap_rows"] == 1
        assert report["leaked_to_strong_pool"] == 0
        assert report["ok"] is True


# ── Leaked-urgent (alert-gate downstream failure) ───────────────────────────


class TestLeakedUrgent:
    def test_ml_only_urgent_recap_counts_as_leaked_urgent(self, store):
        """A model-flagged urgent recap (ml_score>=8, score_source='ml',
        urgency=1) is NOT a strong-pool leak (the trainer excludes 'ml'),
        but it IS an alert-gate failure — the urgency_scorer didn't catch
        it (only Sonnet path uses the pre-filter) AND the alert formatter
        gate didn't either. Surface as ``leaked_urgent``."""
        _insert(
            store, id="ml1",
            title="Lumentum (LITE) Shares Fall 8.8% -- GF Value Says S",
            source="GoogleNews/GuruFocus",
            ai_score=0.0, ml_score=9.0, score_source="ml", urgency=1,
        )
        report = recap_template_audit.audit(store)
        assert report["leaked_to_strong_pool"] == 0  # ml-tagged, not strong pool
        assert report["leaked_urgent"] == 1
        # ok keys on strong-pool only; ML-side leaks are surfaced but
        # don't flip the strong-pool verdict.
        assert report["ok"] is True


# ── Per-fingerprint counting ────────────────────────────────────────────────


class TestFingerprintCounting:
    def test_each_fingerprint_independently_counted(self, store):
        """One row per fingerprint, all in strong pool. Each must light up
        its own bucket exactly once."""
        rows = [
            ("a", "Why Nvidia (NVDA) Stock Is Trading Up Today",
             "why_trading_today"),
            ("b", "Why Did Micron Stock Drop Today", "why_did_stock"),
            ("c", "Stock Market Today, May 18: Micron Falls",
             "market_today_dated"),
            ("d", "Micron (MU) Q3 2026 Earnings Call Highlights",
             "earnings_call_recap"),
            ("e", "Here's What the Street Thinks About Micron",
             "street_thinks"),
            ("f", "AXT Inc (AXTI) Shares Fall -- GF Value Says",
             "gf_value_says"),
        ]
        for id, t, _ in rows:
            _insert(
                store, id=id, title=t, ai_score=8.0,
                score_source="llm", urgency=2,
            )
        report = recap_template_audit.audit(store)
        assert report["total_recap_rows"] == 6
        for _, _, fp in rows:
            assert report["by_fingerprint"][fp] == 1
            assert report["leaked_by_fingerprint"][fp] == 1
        assert report["leaked_to_strong_pool"] == 6
        assert report["ok"] is False

    def test_one_fingerprint_per_row_first_wins(self, store):
        """A title that COULD match two patterns counts under one only —
        defensive guarantee that ``sum(by_fingerprint) == total_recap_rows``
        always holds (so the audit's totals reconcile)."""
        # "Why Did <X> Stock <verb> Today" matches why_did_stock; the
        # discriminator is verb (drop/rise/...). "Why Nvidia Stock Is
        # Trading Up Today" matches why_trading_today. Construct a
        # title that *could* loosely look like both — first-wins applies.
        _insert(
            store, id="ambig",
            title="Why Nvidia (NVDA) Stock Is Trading Up Today",
            ai_score=8.0, score_source="llm",
        )
        report = recap_template_audit.audit(store)
        # Exactly one bucket hit, one row counted total.
        assert report["total_recap_rows"] == 1
        hits = [v for v in report["by_fingerprint"].values() if v > 0]
        assert len(hits) == 1


# ── Backtest isolation (the critical invariant) ─────────────────────────────


class TestBacktestIsolation:
    def test_backtest_url_row_never_counted(self, store):
        """A synthetic ``backtest://`` row whose title matches a recap
        pattern MUST be excluded. The metric measures the LIVE pool; an
        injection burst must not be able to fake a calibration figure."""
        _insert(
            store, id="bt",
            url="backtest://run_1/2026-01-01/BUY/MU",
            title="Why Did Micron Stock Drop Today",
            source="backtest_run_1_winner",
            ai_score=8.0, score_source=None,
        )
        report = recap_template_audit.audit(store)
        assert report["total_recap_rows"] == 0
        assert report["leaked_to_strong_pool"] == 0
        assert report["ok"] is True

    def test_backtest_source_tag_never_counted(self, store):
        _insert(
            store, id="bt2", url="https://example.com/bt2",
            title="Stock Market Today, May 18: ...",
            source="backtest_run_42_rank1",
            ai_score=8.0, score_source=None,
        )
        report = recap_template_audit.audit(store)
        assert report["total_recap_rows"] == 0

    def test_opus_annotation_source_never_counted(self, store):
        _insert(
            store, id="op", url="https://example.com/op",
            title="Why Nvidia Stock Is Trading Up Today",
            source="opus_annotation_cycle_3",
            ai_score=8.0, score_source=None,
        )
        report = recap_template_audit.audit(store)
        assert report["total_recap_rows"] == 0


# ── Window filtering ────────────────────────────────────────────────────────


class TestWindowFiltering:
    def test_old_row_outside_window_excluded(self, store):
        """A recap row older than the window MUST be excluded — the
        calibration is a recent-window measurement, not lifetime."""
        old_seen = (
            datetime.now(timezone.utc) - timedelta(hours=48)
        ).isoformat()
        _insert(
            store, id="old",
            title="Why Did Nvidia Stock Surge Today",
            ai_score=8.0, score_source="llm", first_seen=old_seen,
        )
        report = recap_template_audit.audit(store, hours=24)
        assert report["total_recap_rows"] == 0
        assert report["ok"] is True

    def test_custom_hours_widens_window(self, store):
        """A recap row 36h old is excluded at hours=24 but included at
        hours=48 — proves the window parameter is respected."""
        old_seen = (
            datetime.now(timezone.utc) - timedelta(hours=36)
        ).isoformat()
        _insert(
            store, id="m1",
            title="Why Did AMD Stock Surge Today",
            ai_score=8.0, score_source="llm", first_seen=old_seen,
        )
        r24 = recap_template_audit.audit(store, hours=24)
        r48 = recap_template_audit.audit(store, hours=48)
        assert r24["total_recap_rows"] == 0
        assert r48["total_recap_rows"] == 1
        assert r48["leaked_to_strong_pool"] == 1


# ── Anti-drift: live-only clause stays in sync with storage layer ───────────


def test_live_only_clause_in_sync_with_storage():
    """The audit duplicates ``_LIVE_ONLY_CLAUSE`` as a string constant to
    avoid pulling the writable ArticleStore graph. The two MUST stay
    byte-identical — a future change to the storage clause must propagate
    here or the audit will miscount (same drift class as the
    dashboard-parity backtest-isolation tests)."""
    assert (recap_template_audit.LIVE_ONLY_CLAUSE
            == article_store._LIVE_ONLY_CLAUSE), (
        "analytics.recap_template_audit.LIVE_ONLY_CLAUSE drifted from "
        "storage.article_store._LIVE_ONLY_CLAUSE — audit will under/over-count"
    )
