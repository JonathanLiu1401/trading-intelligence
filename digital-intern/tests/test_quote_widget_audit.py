"""``analytics.quote_widget_audit`` — calibration view of the widget gate.

The audit answers the analyst-facing "is the quote-widget pre-filter still
working?" question by counting widget-matching rows in the recent window by
their current state. Sibling to ``tests/test_recap_template_audit.py`` — same
shape, different noise class.

The discriminating asserts:

  1. ``leaked_to_strong_pool`` (widget + score_source='llm' + ai_score>=8) is
     the load-bearing regression signal — a single such row means the
     ``urgency_scorer`` pre-filter let one through and the trainer is now
     ingesting a price-tape pseudo-article as ground-truth.
  2. ``floored_to_noise`` (widget + ai_score<=0.5) is the "pre-filter worked"
     counterweight — these were caught.
  3. ``leaked_urgent`` (widget + urgency>=1) catches the alert-gate
     downstream-failure case (gate failed even if Sonnet labeled correctly).
  4. Backtest-isolation: synthetic ``backtest://`` / ``backtest_*`` /
     ``opus_annotation*`` rows whose title would match a widget fingerprint
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

from analytics import quote_widget_audit
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
        report = quote_widget_audit.audit(store)
        assert report["window_h"] == 24
        assert report["total_widget_rows"] == 0
        assert report["leaked_to_strong_pool"] == 0
        assert report["leaked_urgent"] == 0
        assert report["floored_to_noise"] == 0
        assert report["leak_fraction"] == 0.0
        assert report["ok"] is True
        # Every fingerprint key present even on empty input — dashboard
        # rendering depends on a stable shape (mirrors recap_template_audit's
        # discipline).
        for name in ("price_glue", "pct_paren", "listing_card", "screener_tape"):
            assert name in report["by_fingerprint"]
            assert name in report["leaked_by_fingerprint"]


# ── Strong-pool leak (the load-bearing regression signal) ───────────────────


class TestStrongPoolLeak:
    def test_single_llm_urgent_widget_is_a_leak(self, store):
        """The exact live failure (2026-05-21 30d audit): Sonnet labeled a
        widget row ai_score=8.0 score_source='llm'. The audit MUST surface
        this row in ``leaked_to_strong_pool`` and set ``ok=False``."""
        _insert(
            store, id="poison",
            title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            source="scraped/finance.yahoo.com",
            ai_score=8.0, score_source="llm", urgency=2,
        )
        report = quote_widget_audit.audit(store)
        assert report["leaked_to_strong_pool"] == 1
        # The price-glue fingerprint MUST be the one that fires on this title.
        assert report["leaked_by_fingerprint"]["price_glue"] == 1
        assert report["ok"] is False, (
            "the regression signal the audit exists to surface was missed"
        )

    def test_clean_pool_post_prefilter_is_ok(self, store):
        """Post-fix state: the urgency_scorer pre-filter floored every
        widget row to 0.01 score_source='llm'. ``leaked_to_strong_pool`` is
        0 → ``ok=True``."""
        for i, t in enumerate([
            "NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            "NQ=FNasdaq 100 Jun 2629,215.25-472.50(-1.59%)",
            "$NVIDIA (NVDA.US)$ - Moomoo",
        ]):
            _insert(
                store, id=f"clean{i}", title=t,
                source="scraped/finance.yahoo.com",
                ai_score=0.01, score_source="llm", urgency=0,
            )
        report = quote_widget_audit.audit(store)
        assert report["total_widget_rows"] == 3
        assert report["floored_to_noise"] == 3, (
            "pre-filter-floored rows missing from floored_to_noise"
        )
        assert report["leaked_to_strong_pool"] == 0
        assert report["ok"] is True

    def test_briefing_boost_score_does_not_count_as_leak(self, store):
        """A row tagged ``score_source='briefing_boost'`` (Opus curation
        nudge, ai_score=4.5) cannot satisfy the >=8 leak threshold. Pin so
        a future regression promoting a widget row through
        update_scores_from_labels doesn't quietly flip the verdict."""
        _insert(
            store, id="bb",
            title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            source="scraped/finance.yahoo.com",
            ai_score=4.5, score_source="briefing_boost", urgency=0,
        )
        report = quote_widget_audit.audit(store)
        assert report["total_widget_rows"] == 1
        assert report["leaked_to_strong_pool"] == 0
        assert report["ok"] is True


# ── Leaked-urgent (alert-gate downstream failure) ───────────────────────────


class TestLeakedUrgent:
    def test_ml_only_urgent_widget_counts_as_leaked_urgent(self, store):
        """A model-flagged urgent widget (ml_score>=8, score_source='ml',
        urgency=1) is NOT a strong-pool leak (the trainer excludes 'ml'),
        but it IS a gate failure — the urgency_scorer pre-filter blocks
        Sonnet but the ML-side urgency head still tagged urgency=1. Surface
        as ``leaked_urgent``."""
        _insert(
            store, id="ml1",
            title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            source="scraped/finance.yahoo.com",
            ai_score=0.0, ml_score=9.0, score_source="ml", urgency=1,
        )
        report = quote_widget_audit.audit(store)
        assert report["leaked_to_strong_pool"] == 0
        assert report["leaked_urgent"] == 1
        assert report["ok"] is True


# ── Per-fingerprint counting ────────────────────────────────────────────────


class TestFingerprintCounting:
    def test_each_fingerprint_independently_counted(self, store):
        """One row per fingerprint, all in strong pool. Each must light up
        its own bucket exactly once."""
        rows = [
            ("a", "NVDANVIDIA Corporation227.13-8.61(-3.65%)", "price_glue"),
            ("b", "Some random text (-3.65%)", "pct_paren"),
            ("c", "$NVIDIA (NVDA.US)$ - Moomoo", "listing_card"),
            ("d", "[YF/most_actives] MU (Micron Technology) +2.5%",
             "screener_tape"),
        ]
        for id, t, _ in rows:
            _insert(
                store, id=id, title=t,
                source="scraped/finance.yahoo.com",
                ai_score=8.0, score_source="llm", urgency=2,
            )
        report = quote_widget_audit.audit(store)
        assert report["total_widget_rows"] == 4
        for _, _, fp in rows:
            assert report["by_fingerprint"][fp] == 1, (
                f"fingerprint {fp} did not count its row"
            )
            assert report["leaked_by_fingerprint"][fp] == 1
        assert report["leaked_to_strong_pool"] == 4
        assert report["ok"] is False

    def test_one_fingerprint_per_row_first_wins(self, store):
        """A title that could match two patterns counts under one only —
        sum(by_fingerprint) == total_widget_rows always holds."""
        # "NVDANVIDIA Corporation227.13-8.61(-3.65%)" matches BOTH
        # price_glue ("n227.13") and pct_paren ("(-3.65%)"). The audit must
        # count it under price_glue only (first-wins) so totals reconcile.
        _insert(
            store, id="dual",
            title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            source="scraped/finance.yahoo.com",
            ai_score=8.0, score_source="llm", urgency=2,
        )
        report = quote_widget_audit.audit(store)
        assert report["total_widget_rows"] == 1
        assert (sum(report["by_fingerprint"].values())
                == report["total_widget_rows"]), (
            "by_fingerprint sum drifted from total_widget_rows — a row "
            "was double-counted (first-wins ordering broken)"
        )
        # The first pattern in the tuple is price_glue; assert it wins.
        assert report["by_fingerprint"]["price_glue"] == 1
        assert report["by_fingerprint"]["pct_paren"] == 0


# ── Backtest isolation ─────────────────────────────────────────────────────


class TestBacktestIsolation:
    def test_backtest_url_widget_not_counted(self, store):
        """A backtest:// row that happens to carry a widget-shaped title must
        not inflate the live audit — the metric measures the LIVE pool only.
        Same invariant as the rest of the audit family."""
        # Live widget — should count.
        _insert(
            store, id="live",
            title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            source="scraped/finance.yahoo.com",
            ai_score=8.0, score_source="llm", urgency=2,
        )
        # Backtest URL row with a widget title — must be excluded.
        _insert(
            store, id="bt",
            title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            source="rss", url="backtest://run_1/2026-01-01/BUY/NVDA",
            ai_score=8.0, score_source="llm", urgency=2,
        )
        # Backtest source row with a widget title — must be excluded.
        _insert(
            store, id="bt2",
            title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            source="backtest_run_42_winner",
            ai_score=8.0, score_source="llm", urgency=2,
        )
        # Opus annotation row with a widget title — must be excluded.
        _insert(
            store, id="opus",
            title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            source="opus_annotation_cycle_3",
            ai_score=8.0, score_source="llm", urgency=2,
        )
        report = quote_widget_audit.audit(store)
        assert report["total_widget_rows"] == 1, (
            "synthetic rows inflated the live audit — backtest isolation broken"
        )
        assert report["leaked_to_strong_pool"] == 1


def test_live_only_clause_in_sync_with_storage():
    """The audit's ``LIVE_ONLY_CLAUSE`` must match storage.article_store's
    verbatim. Drift here is a silent regression — the audit would either
    count synthetic rows (false positives) or fail to count live ones
    (false negatives). Same anti-drift discipline as the dashboard-parity
    backtest-isolation tests and recap_template_audit."""
    assert (quote_widget_audit.LIVE_ONLY_CLAUSE
            == article_store._LIVE_ONLY_CLAUSE), (
        "quote_widget_audit.LIVE_ONLY_CLAUSE has drifted from "
        "storage.article_store._LIVE_ONLY_CLAUSE — fix one of the two"
    )


# ── Window filtering ───────────────────────────────────────────────────────


class TestWindow:
    def test_old_rows_excluded(self, store):
        """A widget row older than the window must not be counted."""
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        _insert(
            store, id="old", title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            source="scraped/finance.yahoo.com",
            ai_score=8.0, score_source="llm", urgency=2,
            first_seen=old,
        )
        report = quote_widget_audit.audit(store, hours=6)
        assert report["total_widget_rows"] == 0
        assert report["ok"] is True


# ── Per-source breakdown ────────────────────────────────────────────────────


class TestAuditBySource:
    def test_empty_store_returns_empty_list(self, store):
        report = quote_widget_audit.audit_by_source(store)
        assert report["window_h"] == 24
        assert report["by_source"] == []
        assert report["total_widget_rows"] == 0
        assert report["total_sources"] == 0
        assert report["ok"] is True

    def test_per_source_aggregation_and_sort(self, store):
        """Three sources, different widget counts; result must be sorted
        most-widget-first with alphabetical tiebreak."""
        # Source A: 3 widgets
        for i in range(3):
            _insert(
                store, id=f"a{i}",
                title=f"NVDANVIDIA Corporation227.{i:02d}-8.61(-3.65%)",
                source="scraped/finance.yahoo.com",
                ai_score=0.01, score_source="llm",
            )
        # Source B: 1 widget
        _insert(
            store, id="b1",
            title="$NVIDIA (NVDA.US)$ - Moomoo",
            source="GN: Nvidia",
            ai_score=0.01, score_source="llm",
        )
        # Source C: 2 widgets (less than A, more than B)
        for i in range(2):
            _insert(
                store, id=f"c{i}",
                title=f"[YF/most_actives] X{i} (Some Co) +2.5%",
                source="YF/most_actives",
                ai_score=0.01, score_source="llm",
            )

        report = quote_widget_audit.audit_by_source(store)
        sources = [r["source"] for r in report["by_source"]]
        assert sources == [
            "scraped/finance.yahoo.com",
            "YF/most_actives",
            "GN: Nvidia",
        ], f"unexpected sort order: {sources}"
        assert report["by_source"][0]["widget_count"] == 3
        assert report["by_source"][1]["widget_count"] == 2
        assert report["by_source"][2]["widget_count"] == 1
        assert report["total_widget_rows"] == 6
        assert report["total_sources"] == 3
        assert report["ok"] is True

    def test_top_n_caps_list(self, store):
        for i in range(5):
            _insert(
                store, id=f"s{i}",
                title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
                source=f"source_{i}",
                ai_score=0.01, score_source="llm",
            )
        report = quote_widget_audit.audit_by_source(store, top_n=2)
        assert len(report["by_source"]) == 2
        assert report["total_sources"] == 5

    def test_strong_pool_leak_attributed_to_source(self, store):
        """A leaked-strong-pool row must attribute to its source's
        ``leaked_strong_pool`` count AND flip the per-source verdict to
        ok=False at the top-level."""
        _insert(
            store, id="poison",
            title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
            source="scraped/finance.yahoo.com",
            ai_score=8.0, score_source="llm", urgency=2,
        )
        report = quote_widget_audit.audit_by_source(store)
        assert report["ok"] is False
        src_row = report["by_source"][0]
        assert src_row["source"] == "scraped/finance.yahoo.com"
        assert src_row["leaked_strong_pool"] == 1
        assert src_row["leaked_urgent"] == 1
        assert src_row["top_fingerprint"] == "price_glue"


# ── Anti-drift: audit must enumerate the same fingerprints as the gate ──────


def test_audit_fingerprint_set_matches_alert_agent_gate():
    """``analytics.quote_widget_audit`` imports the SAME fingerprint tuple
    as ``alert_agent._looks_like_quote_widget`` would test (modulo URL).
    A future change adding a new fingerprint to ``_looks_like_quote_widget``
    without registering it in ``_QUOTE_WIDGET_TITLE_PATTERNS`` would create
    a silent gap — the audit would under-count the live noise. Pin the
    expected names so a drift fails this test."""
    from watchers import alert_agent
    names = {n for n, _p in alert_agent._QUOTE_WIDGET_TITLE_PATTERNS}
    assert names == {"price_glue", "pct_paren", "listing_card",
                     "screener_tape", "stocktwits_sentiment"}, (
        f"fingerprint name set drifted: {names}"
    )
