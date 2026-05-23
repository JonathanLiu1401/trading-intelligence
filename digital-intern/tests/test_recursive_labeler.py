"""Recursive-labeler invariants nothing else in the suite pins.

Scope is deliberately narrow — only behaviour unique to ml/recursive_labeler:

  * The urgency-parse regression: a single malformed ``urgency`` value from
    Claude (it returns "1", "1.0", "yes", true — not a bare int) must NOT
    abort the run or discard the batch's already-collected good labels. This
    is the test that fails pre-fix and passes post-fix; the bug unwound
    _apply_labels → _run_round → run_recursive_labeling with no inner handler.
  * The 0..5 → 0..10 relevance rescale (a load-bearing magic ``* 2.0``).
  * Writes go through update_ai_scores_batch → score_source='llm' (Sonnet/Opus
    are ground-truth labelers; this is the ml-vs-ai invariant surface).
  * _fetch_round1_candidates backtest/opus exclusion — a separate WHERE filter
    than storage._LIVE_ONLY_CLAUSE, so it needs its own regression guard.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ml import recursive_labeler


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_raw(store, *, id, url, title, source="rss", ai_score=0.0,
                urgency=0, kw_score=1.0, first_seen=None):
    if first_seen is None:
        first_seen = _recent_iso()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, urgency,
             first_seen, 0, None, None),
        )
        store.conn.commit()


def _row(store, aid):
    return store.conn.execute(
        "SELECT ai_score, urgency, score_source FROM articles WHERE id=?",
        (aid,),
    ).fetchone()


class TestApplyLabels:
    def test_relevance_rescaled_and_tagged_llm(self, store):
        """relevance is 0..5 from the prompt; the store keeps 0..10. The
        labeler is an LLM path, so writes must tag score_source='llm'."""
        _insert_raw(store, id="a", url="https://x.com/a", title="t")
        articles = [{"_id": "a", "url": "https://x.com/a", "title": "t"}]
        labels = [{"url": "https://x.com/a", "relevance": 4.0, "urgency": 1}]

        n = recursive_labeler._apply_labels(store, articles, labels)

        assert n == 1
        ai_score, urgency, src = _row(store, "a")
        assert ai_score == pytest.approx(8.0), "0..5 → 0..10 rescale (×2) broken"
        assert urgency == 1
        assert src == "llm", "recursive-labeler writes are ground-truth labels"

    def test_poison_urgency_does_not_abort_or_lose_siblings(self, store):
        """THE regression test. Middle label carries a non-int urgency Claude
        commonly emits. Pre-fix: int('yes') raised ValueError that escaped
        _apply_labels (no try/except round the loop), so the in-flight
        ``updates`` list — including the perfectly good siblings a and c —
        was discarded and the rest of the 4h cycle aborted. Post-fix: the bad
        urgency degrades to 0 and every good label is still persisted."""
        for aid in ("a", "b", "c"):
            _insert_raw(store, id=aid, url=f"https://x.com/{aid}", title=aid)
        articles = [
            {"_id": "a", "url": "https://x.com/a", "title": "a"},
            {"_id": "b", "url": "https://x.com/b", "title": "b"},
            {"_id": "c", "url": "https://x.com/c", "title": "c"},
        ]
        labels = [
            {"url": "https://x.com/a", "relevance": 4.0, "urgency": 1},
            {"url": "https://x.com/b", "relevance": 3.5, "urgency": "yes"},  # poison
            {"url": "https://x.com/c", "relevance": 2.0, "urgency": 0},
        ]

        n = recursive_labeler._apply_labels(store, articles, labels)

        # All three persisted — pre-fix this was 0 (whole batch discarded).
        assert n == 3
        a = _row(store, "a")
        b = _row(store, "b")
        c = _row(store, "c")
        assert a == (pytest.approx(8.0), 1, "llm")
        # Poison sibling keeps its relevance label; urgency degrades to 0.
        assert b == (pytest.approx(7.0), 0, "llm")
        assert c == (pytest.approx(4.0), 0, "llm")

    @pytest.mark.parametrize(
        "raw_urg, expect",
        [(1, 1), ("1", 1), ("1.0", 1), (1.0, 1), (True, 1),
         (0, 0), ("0", 0), (None, 0), ("yes", 0), ("high", 0), ([], 0)],
    )
    def test_urgency_coercion_matrix(self, store, raw_urg, expect):
        """Every value below must coerce, never raise. The string/bool forms
        are exactly what Claude returns in practice."""
        _insert_raw(store, id="a", url="https://x.com/a", title="t")
        articles = [{"_id": "a", "url": "https://x.com/a", "title": "t"}]
        labels = [{"url": "https://x.com/a", "relevance": 5.0, "urgency": raw_urg}]

        n = recursive_labeler._apply_labels(store, articles, labels)

        assert n == 1
        ai_score, urgency, _ = _row(store, "a")
        assert ai_score == pytest.approx(10.0)
        assert urgency == expect

    def test_bad_relevance_skips_only_that_label(self, store):
        """A non-numeric relevance still skips its own label (the existing
        guard) without taking down siblings."""
        for aid in ("a", "b"):
            _insert_raw(store, id=aid, url=f"https://x.com/{aid}", title=aid)
        articles = [
            {"_id": "a", "url": "https://x.com/a", "title": "a"},
            {"_id": "b", "url": "https://x.com/b", "title": "b"},
        ]
        labels = [
            {"url": "https://x.com/a", "relevance": "garbage", "urgency": 1},
            {"url": "https://x.com/b", "relevance": 3.0, "urgency": 0},
        ]

        n = recursive_labeler._apply_labels(store, articles, labels)

        assert n == 1
        assert _row(store, "a") == (pytest.approx(0.0), 0, None)  # untouched
        assert _row(store, "b") == (pytest.approx(6.0), 0, "llm")

    def test_unknown_url_label_ignored(self, store):
        _insert_raw(store, id="a", url="https://x.com/a", title="t")
        articles = [{"_id": "a", "url": "https://x.com/a", "title": "t"}]
        labels = [{"url": "https://x.com/NOT-IN-BATCH", "relevance": 5.0,
                   "urgency": 1}]
        assert recursive_labeler._apply_labels(store, articles, labels) == 0
        assert _row(store, "a") == (pytest.approx(0.0), 0, None)


class TestFetchRound1Candidates:
    def test_excludes_backtest_and_opus_rows(self, store):
        """Round-1 candidate selection has its OWN backtest/opus WHERE filter
        (not storage._LIVE_ONLY_CLAUSE) — re-scoring a synthetic training row
        with the live Sonnet labeler would corrupt the backtest signal."""
        _insert_raw(store, id="live", url="https://reuters.com/x",
                    title="Live unlabeled story", source="rss", ai_score=0.0)
        _insert_raw(store, id="bt_url", url="backtest://run_1/d/BUY/MU",
                    title="Synthetic by url", source="rss", ai_score=0.5)
        _insert_raw(store, id="bt_src", url="https://x.com/y",
                    title="Synthetic by source", source="backtest_run_1_winner",
                    ai_score=0.5)
        _insert_raw(store, id="opus", url="https://x.com/z",
                    title="Opus annotation row", source="opus_annotation_cycle_2",
                    ai_score=1.0)

        out = recursive_labeler._fetch_round1_candidates(store, limit=50)
        ids = {a["_id"] for a in out}
        assert ids == {"live"}

    def test_excludes_already_strongly_labeled(self, store):
        """ai_score >= 2.0 is 'labeled enough' — only weak/zero rows recurse."""
        _insert_raw(store, id="weak", url="https://x.com/weak",
                    title="weakly labeled", ai_score=1.5)
        _insert_raw(store, id="strong", url="https://x.com/strong",
                    title="strongly labeled", ai_score=7.0)
        _insert_raw(store, id="zero", url="https://x.com/zero",
                    title="unlabeled", ai_score=0.0)

        out = recursive_labeler._fetch_round1_candidates(store, limit=50)
        ids = {a["_id"] for a in out}
        assert ids == {"weak", "zero"}
        assert "strong" not in ids

    def test_excludes_noise_floor_sentinel(self, store):
        """ai_score=0.01 is the urgency-scorer pre-floor + anti-loop sentinel.
        Re-fetching these into the recursive labeler sends already-judged-noise
        rows back to Sonnet (quota waste + re-promotion risk). The SQL filter
        excludes the exact 0.01 value while keeping ai_score=0 (genuine
        unlabeled) and ai_score>=0.5 (genuine low Sonnet labels)."""
        # The exact pre-floor sentinel — must be excluded.
        _insert_raw(store, id="qw_floor", url="https://x.com/qw",
                    title="NVDANVIDIA Corporation227.13-8.61(-3.65%)",
                    ai_score=0.01)
        _insert_raw(store, id="rt_floor", url="https://x.com/rt",
                    title="Why Nvidia (NVDA) Stock Is Trading Up Today",
                    ai_score=0.01)
        # Genuine unlabeled — must be included.
        _insert_raw(store, id="zero", url="https://x.com/zero",
                    title="Federal Reserve announces emergency rate decision",
                    ai_score=0.0)
        # Genuine low Sonnet label (1.0, 1.5) — must be included for active
        # learning re-scoring (these are the borderline cases the labeler is
        # designed to revisit).
        _insert_raw(store, id="low1", url="https://x.com/low1",
                    title="ECB rate decision week — analyst consensus split",
                    ai_score=1.0)
        _insert_raw(store, id="low15", url="https://x.com/low15",
                    title="MU outlook hazy ahead of next quarter",
                    ai_score=1.5)

        out = recursive_labeler._fetch_round1_candidates(store, limit=50)
        ids = {a["_id"] for a in out}
        # Noise-floor sentinels excluded; genuine unlabeled and low-label kept.
        assert ids == {"zero", "low1", "low15"}
        assert "qw_floor" not in ids
        assert "rt_floor" not in ids

    def test_excludes_quote_widget_fingerprints_at_ai_score_zero(self, store):
        """A quote-widget-shaped row with ai_score=0 (scorer worker hadn't
        Sonnet-routed it yet) must NOT be re-routed by the recursive labeler.
        Defense-in-depth: a fingerprint match overrides the SQL-only filter
        so the recursive labeler can never reintroduce the exact noise the
        urgency_scorer pre-floor exists to prevent."""
        _insert_raw(store, id="qw0", url="https://x.com/qw0",
                    # Quote-widget price-glue fingerprint.
                    title="BTC-USDBitcoin USD78,012.91-2,471.37(-3.07%)",
                    ai_score=0.0)
        _insert_raw(store, id="qw1", url="https://x.com/qw1",
                    # Quote-widget parenthesised-%-change fingerprint.
                    title="Bitcoin slides on rate fears (-3.07%) widget",
                    ai_score=0.0)
        _insert_raw(store, id="real", url="https://x.com/real",
                    title="Micron beats Q1 estimates on HBM demand",
                    ai_score=0.0)
        out = recursive_labeler._fetch_round1_candidates(store, limit=50)
        ids = {a["_id"] for a in out}
        # Both quote-widget rows filtered; real headline kept.
        assert "real" in ids
        assert "qw0" not in ids
        assert "qw1" not in ids

    def test_excludes_recap_template_fingerprints_at_ai_score_zero(self, store):
        """Recap-template SEO mill rows that escaped the urgency_scorer pre-floor
        (rare, but possible when the scorer worker is behind) must not be
        promoted by recursive-labeler re-scoring. Same single-source-of-truth
        recap fingerprint set as the alert path's _filter_recap_template_noise."""
        _insert_raw(store, id="rt_why_trading", url="https://x.com/rt1",
                    title="Why Micron Stock Is Trading Up Today",
                    ai_score=0.0)
        _insert_raw(store, id="rt_quick_glance", url="https://x.com/rt2",
                    title="NVIDIA Earnings: A Quick Glance at Key Metrics",
                    ai_score=0.0)
        _insert_raw(store, id="rt_market_today", url="https://x.com/rt3",
                    title="Stock Market Today, May 18: Micron Falls",
                    ai_score=0.0)
        _insert_raw(store, id="real", url="https://x.com/real",
                    title="Apple unveils new chip family at WWDC",
                    ai_score=0.0)
        out = recursive_labeler._fetch_round1_candidates(store, limit=50)
        ids = {a["_id"] for a in out}
        assert ids == {"real"}, (
            "every recap-template fingerprint must be skipped"
        )
