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


# ── Per-source breakdown (analyst-facing: WHICH feeds dominate the noise?) ──


class TestAuditBySource:
    """``audit_by_source`` answers the next question after ``audit()``: of
    the recap rows ingested, which sources produce most of them? Live
    evidence (2026-05-21 24h scan): 425 recap rows / 41 sources, with
    the top four producing 53% of the total — the per-source view turns
    that into a pruneable list. Pins the contract so a regression in the
    builder (counting, fingerprint attribution, sort order, leak split,
    backtest isolation, window enforcement, top_n cap) fails loudly."""

    def test_empty_store_yields_empty_list_and_ok(self, store):
        report = recap_template_audit.audit_by_source(store)
        assert report["window_h"] == 24
        assert report["by_source"] == []
        assert report["total_recap_rows"] == 0
        assert report["total_sources"] == 0
        assert report["ok"] is True

    def test_single_source_single_fingerprint(self, store):
        _insert(
            store, id="a",
            title="Micron (MU) Q3 2026 Earnings Call Highlights",
            source="Motley Fool", ai_score=4.0, score_source="llm",
            urgency=0,
        )
        report = recap_template_audit.audit_by_source(store)
        assert report["total_recap_rows"] == 1
        assert report["total_sources"] == 1
        assert len(report["by_source"]) == 1
        row = report["by_source"][0]
        assert row["source"] == "Motley Fool"
        assert row["recap_count"] == 1
        assert row["by_fingerprint"] == {"earnings_call_recap": 1}
        assert row["top_fingerprint"] == "earnings_call_recap"
        assert row["leaked_urgent"] == 0
        assert row["leaked_strong_pool"] == 0

    def test_sort_order_recap_count_desc_then_source_asc(self, store):
        """Ranking discipline (mirrors source_urgency_yield): most-recap-
        first, alphabetical tie-break. Pinned so a future builder change
        cannot silently re-order the analyst's pruning list."""
        # 3 recap rows from "GN: earnings", 2 from "Motley Fool",
        # 1 each from "Aaa" and "Zzz" (tied — Aaa must rank first).
        for i in range(3):
            _insert(
                store, id=f"gn{i}",
                title=f"NVDA Q3 2026 Earnings Call Highlights {i} extra",
                source="GN: earnings", ai_score=0.01, score_source="llm",
            )
        for i in range(2):
            _insert(
                store, id=f"mf{i}",
                title=f"AMD Q1 2027 Earnings Call Highlights {i} extra",
                source="Motley Fool", ai_score=0.01, score_source="llm",
            )
        _insert(
            store, id="aaa",
            title="MU Q3 2026 Earnings Call Highlights",
            source="Aaa", ai_score=0.01, score_source="llm",
        )
        _insert(
            store, id="zzz",
            title="TSM Q2 2026 Earnings Call Highlights",
            source="Zzz", ai_score=0.01, score_source="llm",
        )
        report = recap_template_audit.audit_by_source(store)
        ordering = [(r["source"], r["recap_count"]) for r in report["by_source"]]
        assert ordering == [
            ("GN: earnings", 3),
            ("Motley Fool", 2),
            ("Aaa", 1),     # alphabetical tie-break wins over "Zzz"
            ("Zzz", 1),
        ]

    def test_mixed_fingerprints_per_source_top_fingerprint_correct(self, store):
        """A single source with a mix of fingerprints — ``top_fingerprint``
        must be the highest-count one (alpha tie-break). Pins the
        per-source counter math: each fingerprint counted exactly once
        per row, first-wins (matches audit() precedent)."""
        rows = [
            ("a", "Why Did Micron Stock Drop Today", "why_did_stock"),
            ("b", "Why Did Nvidia Stock Surge Today", "why_did_stock"),
            ("c", "Why Did AMD Stock Plunge Today", "why_did_stock"),
            ("d", "Stock Market Today, May 18: Micron Falls",
             "market_today_dated"),
            ("e", "Stock Market Today, May 19: Nvidia Climbs",
             "market_today_dated"),
        ]
        for id, t, _ in rows:
            _insert(
                store, id=id, title=t, source="Motley Fool",
                ai_score=0.01, score_source="llm",
            )
        report = recap_template_audit.audit_by_source(store)
        assert len(report["by_source"]) == 1
        row = report["by_source"][0]
        assert row["recap_count"] == 5
        assert row["by_fingerprint"] == {
            "why_did_stock": 3,
            "market_today_dated": 2,
        }
        assert row["top_fingerprint"] == "why_did_stock"

    def test_leaked_urgent_and_strong_pool_per_source(self, store):
        """The per-source view must surface the same two regression
        signals ``audit()`` does, but attributed: which source produced
        the gate failure. Real-world action item — drop the offending
        feed or harden its credibility tier."""
        # A clean recap (floored to noise) — counts in recap_count but
        # NOT in either leak metric.
        _insert(
            store, id="ok1",
            title="MU Q3 2026 Earnings Call Highlights",
            source="Seeking Alpha Editors", ai_score=0.01,
            score_source="llm", urgency=0,
        )
        # An ml-urgent recap (alert-gate failure, NOT strong-pool leak).
        _insert(
            store, id="ml_urg",
            title="Why Did NVDA Stock Surge Today",
            source="Motley Fool", ai_score=0.0, ml_score=9.0,
            score_source="ml", urgency=1,
        )
        # A strong-pool leak (Sonnet tagged urgent — the worst case).
        _insert(
            store, id="poison",
            title="Lumentum (LITE) Shares Fall -- GF Value Says X",
            source="GoogleNews/GuruFocus", ai_score=8.0,
            score_source="llm", urgency=2,
        )
        report = recap_template_audit.audit_by_source(store)
        # Three sources, three recap rows total. ok=False because of poison.
        assert report["total_recap_rows"] == 3
        assert report["total_sources"] == 3
        assert report["ok"] is False
        by_src = {r["source"]: r for r in report["by_source"]}
        assert by_src["Seeking Alpha Editors"]["leaked_urgent"] == 0
        assert by_src["Seeking Alpha Editors"]["leaked_strong_pool"] == 0
        assert by_src["Motley Fool"]["leaked_urgent"] == 1
        assert by_src["Motley Fool"]["leaked_strong_pool"] == 0
        assert by_src["GoogleNews/GuruFocus"]["leaked_urgent"] == 1
        assert by_src["GoogleNews/GuruFocus"]["leaked_strong_pool"] == 1

    def test_backtest_rows_excluded_from_per_source_view(self, store):
        """The per-source builder shares the same ``_LIVE_ONLY_CLAUSE`` as
        ``audit()``: a synthetic backtest/opus row whose title matches a
        recap pattern must NEVER appear in the per-source list — a
        contaminated source row would mislead an analyst into pruning
        a legit feed."""
        # Live recap row — must count.
        _insert(
            store, id="live",
            title="Why Did MU Stock Surge Today",
            source="Motley Fool", ai_score=0.01, score_source="llm",
        )
        # Three synthetic recap rows — must NOT appear in by_source.
        _insert(
            store, id="bt1",
            url="backtest://run_1/2026-01-01/BUY/MU",
            title="Why Did MU Stock Surge Today",
            source="backtest_run_1_winner",
            ai_score=8.0, score_source=None,
        )
        _insert(
            store, id="bt2", url="https://x.com/bt2",
            title="Stock Market Today, May 19: Micron Falls",
            source="backtest_run_42_rank1",
            ai_score=8.0, score_source=None,
        )
        _insert(
            store, id="op", url="https://x.com/op",
            title="Why Nvidia Stock Is Trading Up Today",
            source="opus_annotation_cycle_3",
            ai_score=8.0, score_source=None,
        )
        report = recap_template_audit.audit_by_source(store)
        sources = [r["source"] for r in report["by_source"]]
        # Only the live row's source is present; backtest/opus tags must
        # NEVER appear in the analyst-facing pruning list.
        assert sources == ["Motley Fool"]
        for tag in ("backtest_run_1_winner", "backtest_run_42_rank1",
                    "opus_annotation_cycle_3"):
            assert tag not in sources
        assert report["total_recap_rows"] == 1

    def test_top_n_caps_list_but_total_counts_everything(self, store):
        """``top_n`` is a display cap on ``by_source``; the aggregate totals
        (``total_recap_rows`` / ``total_sources``) still reflect every kept
        row so a dashboard rendering only the top-N never under-states the
        scale of the noise."""
        # 5 distinct sources, 1 recap row each.
        for i, src in enumerate(["A", "B", "C", "D", "E"]):
            _insert(
                store, id=f"r{i}",
                title=f"Stock Market Today, May {18+i}: filler row text",
                source=src, ai_score=0.01, score_source="llm",
            )
        report = recap_template_audit.audit_by_source(store, top_n=2)
        assert len(report["by_source"]) == 2
        assert report["total_recap_rows"] == 5
        assert report["total_sources"] == 5

    def test_window_filtering_excludes_old_rows(self, store):
        """A recap row older than ``hours`` must NOT appear in the per-
        source breakdown — same window discipline as ``audit()``."""
        old_seen = (
            datetime.now(timezone.utc) - timedelta(hours=48)
        ).isoformat()
        _insert(
            store, id="old",
            title="Why Did Nvidia Stock Surge Today",
            source="Motley Fool", ai_score=0.01, score_source="llm",
            first_seen=old_seen,
        )
        recent_seen = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat()
        _insert(
            store, id="new",
            title="Why Did AMD Stock Plunge Today",
            source="Yahoo Finance", ai_score=0.01, score_source="llm",
            first_seen=recent_seen,
        )
        report = recap_template_audit.audit_by_source(store, hours=24)
        sources = [r["source"] for r in report["by_source"]]
        assert sources == ["Yahoo Finance"]
        assert report["total_recap_rows"] == 1

    def test_non_recap_rows_do_not_create_source_entries(self, store):
        """A source with NO recap rows must NOT appear in the per-source
        list — the analyst-facing view is recap-noise-only. Pin this so
        a future refactor doesn't silently start emitting one row per
        live source (which would bury the actual offenders)."""
        _insert(
            store, id="clean",
            title="NVDA reports Q1 revenue $81.6B beating estimates",
            source="reuters", ai_score=9.0, score_source="llm", urgency=2,
        )
        _insert(
            store, id="recap1",
            title="Why Did NVDA Stock Surge Today",
            source="Motley Fool", ai_score=0.01, score_source="llm",
        )
        report = recap_template_audit.audit_by_source(store)
        sources = [r["source"] for r in report["by_source"]]
        assert sources == ["Motley Fool"]
        assert "reuters" not in sources


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
