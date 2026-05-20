"""Tests for ``analytics.source_credibility_audit`` — surfaces source tags
that the credibility resolver doesn't know about (feature[0] silently flat
at DEFAULT for whole publishers).

Pinned invariants:

  * the in-line ``LIVE_ONLY_CLAUSE`` string matches
    ``storage.article_store._LIVE_ONLY_CLAUSE`` byte-for-byte (anti-drift
    discipline mirrors ``test_recap_template_audit``);
  * synthetic backtest / opus-annotation rows are excluded from BOTH sides
    of the partition (cannot inflate or mask the defaulting share);
  * an unknown tag lands in ``top_defaulting`` and a known tag does not;
  * the prefix-alias / domain rescue paths are honoured (a tag that the
    resolver moved off DEFAULT must NOT appear in the leaderboard);
  * ``defaulting_share`` is the documented ratio and ``ok`` flips when it
    crosses ``OK_THRESHOLD``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from analytics import source_credibility_audit as sca
from ml.features import DEFAULT_SOURCE_CRED


def _recent_iso(minutes_ago: int = 5) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    ).isoformat()


def _insert(store, *, id, url, title, source, first_seen=None):
    """Minimal insert bypassing the public API — same primitive
    ``tests/test_article_store.py`` uses (we only need first_seen + source
    for this audit; other columns are irrelevant)."""
    if first_seen is None:
        first_seen = _recent_iso()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 1.0, 0.0, 0,
             first_seen, 0, None, None),
        )
        store.conn.commit()


class TestLiveOnlyClauseInSync:
    def test_audit_clause_matches_storage_clause(self):
        """Anti-drift: the inline clause must be byte-identical to the
        canonical one in ``storage.article_store``. If the canonical clause
        ever gains a new synthetic pattern (e.g., a new ``some_synthetic_%``
        source prefix), this test fails until the audit's copy is updated
        too — otherwise the audit would silently include a class of
        synthetic row in its denominator."""
        from storage.article_store import _LIVE_ONLY_CLAUSE
        assert sca.LIVE_ONLY_CLAUSE == _LIVE_ONLY_CLAUSE


class TestPartitionDefaulting:
    def test_defaulting_split_separates_known_and_unknown_tags(self):
        per_source = [
            ("reuters", 100),               # 0.90 — differentiated
            ("brand-new-outlet-2027", 50),  # default
            ("rss", 20),                    # 0.65 — differentiated
            ("", 5),                        # empty tag → defaulting
        ]
        defaulting, differentiated = sca._partition_defaulting(per_source)
        assert sorted(s for s, _ in defaulting) == ["", "brand-new-outlet-2027"]
        assert sorted(s for s, _ in differentiated) == ["reuters", "rss"]

    def test_prefix_alias_rescued_tag_is_not_defaulting(self):
        """The recently-fixed aggregator prefixes (GN:, YF/, YahooFinance/)
        resolve via ``_PREFIX_ALIASES`` — they must NOT show up as
        defaulting in the audit, otherwise the audit signals a leak the
        resolver already closed."""
        per_source = [
            ("GN: Nvidia", 100),
            ("YF/most_actives", 50),
            ("YahooFinance/005930.KS", 30),
        ]
        defaulting, differentiated = sca._partition_defaulting(per_source)
        assert defaulting == []
        assert {s for s, _ in differentiated} == {
            "GN: Nvidia", "YF/most_actives", "YahooFinance/005930.KS",
        }


class TestAuditReport:
    def test_top_defaulting_lists_unknown_tags_count_desc(self, store):
        # Insert one row per unique source so counts are stable.
        _insert(store, id="r1", url="https://reuters.com/a",
                title="Real wire", source="reuters")
        _insert(store, id="r2", url="https://x.com/b",
                title="Big SEO mill", source="seo-junk-2027.example")
        _insert(store, id="r3", url="https://x.com/c",
                title="Another mill", source="seo-junk-2027.example")
        _insert(store, id="r4", url="https://x.com/d",
                title="Single mention", source="rando-outlet.example")

        report = sca.audit(store, hours=1, top=5)

        # Leaderboard order: highest count first, alphabetical tie-break.
        sources = [row["source"] for row in report["top_defaulting"]]
        assert sources[0] == "seo-junk-2027.example", (
            "highest-count defaulting source must lead the leaderboard"
        )
        assert "rando-outlet.example" in sources
        assert "reuters" not in sources, (
            "a known publisher must never appear in top_defaulting"
        )

        # Each row carries the canonical DEFAULT grade for the analyst to read.
        for row in report["top_defaulting"]:
            assert row["cred"] == pytest.approx(DEFAULT_SOURCE_CRED)

    def test_counts_and_share_are_consistent(self, store):
        for i in range(5):
            _insert(store, id=f"k{i}", url=f"https://reuters.com/{i}",
                    title=f"Known {i}", source="reuters")
        for i in range(3):
            _insert(store, id=f"u{i}", url=f"https://x.com/{i}",
                    title=f"Unknown {i}", source="brand-new-2027.example")

        report = sca.audit(store, hours=1, top=15)

        assert report["total_rows"] == 8
        assert report["differentiated_rows"] == 5
        assert report["defaulting_rows"] == 3
        assert report["defaulting_sources"] == 1
        assert report["defaulting_share"] == pytest.approx(3 / 8)
        # 3/8 = 0.375 > OK_THRESHOLD (0.25) → ok must be False
        assert report["ok"] is False

    def test_ok_true_when_share_below_threshold(self, store):
        # 9 known + 1 unknown = 10% defaulting share — comfortably below the
        # 25% bar in OK_THRESHOLD, so ok should be True.
        for i in range(9):
            _insert(store, id=f"k{i}", url=f"https://reuters.com/{i}",
                    title=f"Known {i}", source="reuters")
        _insert(store, id="u0", url="https://x.com/0",
                title="Unknown", source="brand-new-2027.example")

        report = sca.audit(store, hours=1, top=15)

        assert report["defaulting_share"] == pytest.approx(0.1)
        assert report["ok"] is True

    def test_empty_window_returns_zero_share_and_ok(self, store):
        report = sca.audit(store, hours=1)
        assert report["total_rows"] == 0
        assert report["defaulting_share"] == 0.0
        assert report["ok"] is True
        assert report["top_defaulting"] == []

    def test_synthetic_rows_excluded_from_both_sides(self, store):
        """Load-bearing invariant: ``_LIVE_ONLY_CLAUSE`` filters synthetic
        rows from BOTH the differentiated and defaulting counts. A backtest
        injection burst whose source happened to match a low-credibility
        host (e.g., the recovery test we already use for label cleanup)
        cannot inflate the defaulting share, and the differentiated side
        is equally protected — the audit operates on the live corpus only."""
        # One live unknown, plus three synthetic rows the clause should drop.
        _insert(store, id="live1", url="https://x.com/a",
                title="Live unknown", source="brand-new-2027.example")
        _insert(store, id="bt1", url="backtest://run_1/2026-01-01/BUY/MU",
                title="Synthetic", source="brand-new-2027.example")
        _insert(store, id="bt2", url="https://example.com/x",
                title="Opus annotation", source="opus_annotation_cycle_3")
        _insert(store, id="bt3", url="https://example.com/y",
                title="Backtest winner", source="backtest_run_42_winner")

        report = sca.audit(store, hours=1)

        assert report["total_rows"] == 1
        assert report["defaulting_rows"] == 1
        # The unknown's count is 1 — the synthetic copy was excluded.
        assert report["top_defaulting"][0]["count"] == 1

    def test_top_param_caps_leaderboard(self, store):
        for i in range(20):
            _insert(store, id=f"u{i}", url=f"https://x.com/{i}",
                    title=f"Unknown {i}", source=f"outlet-{i:02d}.example")
        report = sca.audit(store, hours=1, top=5)
        assert len(report["top_defaulting"]) == 5


class TestEntryPointSmoke:
    def test_format_report_is_stable_json(self):
        out = sca.format_report({
            "window_h": 24, "top": 15, "total_rows": 0,
            "differentiated_rows": 0, "defaulting_rows": 0,
            "defaulting_sources": 0, "defaulting_share": 0.0,
            "top_defaulting": [], "ok": True,
        })
        # Must be JSON the dashboard / cron job can re-parse without surprises.
        import json
        parsed = json.loads(out)
        assert parsed["ok"] is True
