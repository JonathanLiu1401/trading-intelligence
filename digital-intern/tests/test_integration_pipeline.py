"""Integration tests — full pipeline flows across components.

These tests exercise multiple modules together (store + scorer + alert agent +
trainer + model). They complement the focused unit tests in test_article_store,
test_urgency_scorer, test_trainer, test_features, and test_model.

Critical system invariants verified here:
  • Backtest synthetic rows never reach the live alert path (store + agent).
  • The trainer never ingests its own model predictions (score_source='ml').
  • update_ml_scores_batch writes ml_score only — ai_score is sacred.
  • Concurrent writers do not lose articles or duplicate ids.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pytest

from ml import features as ml_features
from ml import trainer as ml_trainer
from watchers import alert_agent, urgency_scorer


# ───────────────────────── helpers ──────────────────────────


def _recent_iso(minutes_ago: int = 5) -> str:
    """first_seen inside the 24h freshness window that get_unalerted_urgent /
    get_top_for_briefing enforce. A hardcoded absolute date made these tests
    fail once wall-clock moved past it — masking that the invariant under
    test was actually fine."""
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_raw(store, *, id, url, title, source="rss", urgency=0,
                ai_score=0.0, ml_score=None, score_source=None,
                kw_score=1.0, published="", full_text=None, first_seen=None):
    """Insert a row bypassing the public API for arbitrary state setup."""
    if first_seen is None:
        first_seen = _recent_iso()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, published, kw_score, ai_score, urgency,
             first_seen, 0, ml_score, score_source, full_text),
        )
        store.conn.commit()


def _patched_claude(response):
    """Patch urgency_scorer.claude_call to return a fixed JSON-array body."""
    body = json.dumps(response)
    return patch.object(urgency_scorer, "claude_call", return_value=body)


# ─────────────── Test 1: full pipeline ingest → score → alert ───────────────


class TestFullPipeline:
    def test_article_ingest_score_alert_mark(self, store, monkeypatch):
        """Article enters via insert_batch, gets scored URGENT by Sonnet, then
        gets alerted. After alerting, it must not re-surface to the alerter.

        Mocks the Claude CLI on both legs (scorer + alert agent) and stubs the
        Discord webhook so no network call escapes. Touches: ArticleStore,
        urgency_scorer.score_batch, alert_agent.send_urgent_alert,
        discord_notifier.send."""
        # 1. Ingest via public API exactly as a collector would
        n_inserted = store.insert_batch([{
            "title": "MU earnings beat Q3 estimates significantly",
            "link": "https://reuters.com/mu-q3",
            "source": "rss",
            "published": "",
            "summary": "Micron reported Q3 revenue $8.1B vs $7.9B est.",
            "_relevance_score": 3.5,
        }])
        assert n_inserted == 1

        # 2. Pull the unscored row and route through the urgency scorer
        unscored = store.get_unscored(min_kw=0.0)
        assert len(unscored) == 1
        with _patched_claude([{"index": 0, "score": 9.5, "reason": "earnings"}]):
            urg_n = urgency_scorer.score_batch(unscored, store)
        assert urg_n == 1

        # 3. State after scoring: urgency=1, score_source='llm', ai_score=9.5
        row = store.conn.execute(
            "SELECT ai_score, urgency, score_source FROM articles"
        ).fetchone()
        assert row[0] == pytest.approx(9.5)
        assert row[1] == 1
        assert row[2] == "llm"

        # 4. Now the alerter picks it up
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1
        assert urgent[0]["title"].startswith("MU earnings beat")

        # 5. Drive send_urgent_alert with the discord webhook + claude_call mocked
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU"), \
             patch("notifier.discord_notifier.send", return_value=True):
            ok = alert_agent.send_urgent_alert(urgent, store)
        assert ok is True

        # 6. Article is now urgency=2 and must NOT re-appear in unalerted queue
        urgency_now = store.conn.execute(
            "SELECT urgency FROM articles"
        ).fetchone()[0]
        assert urgency_now == 2
        assert store.get_unalerted_urgent() == []


# ────────────── Test 2: backtest isolation across all live paths ─────────────


class TestBacktestIsolationEndToEnd:
    def test_backtest_articles_never_reach_alerter(self, store, monkeypatch):
        """Insert 5 live + 100 backtest urgent rows. Verify:

         (a) get_unalerted_urgent returns ONLY the 5 live rows (store filter).
         (b) Even if a synthetic row somehow leaks past the store, the alert
             agent's _is_synthetic defense-in-depth check drops it before it
             can reach the formatter and Discord."""
        # 5 live urgent
        for i in range(5):
            _insert_raw(store, id=f"live{i}", url=f"https://reuters.com/x{i}",
                        title=f"Live story number {i:03d} matters here please",
                        source="rss", urgency=1, ai_score=9.0 + i * 0.1)
        # 100 backtest urgent with HIGHER scores — would dominate if not filtered
        for i in range(100):
            _insert_raw(store, id=f"bt{i}",
                        url=f"backtest://run_{i}/2026-01-01/BUY/MU",
                        title=f"Synthetic backtest title number {i:04d}",
                        source=f"backtest_run_{i}_winner", urgency=1,
                        ai_score=9.9)
        # 10 opus annotation urgent
        for i in range(10):
            _insert_raw(store, id=f"opa{i}", url=f"https://x.com/opa{i}",
                        title=f"opus annotated label entry {i}",
                        source=f"opus_annotation_cycle_{i}", urgency=1,
                        ai_score=9.5)

        # (a) Store filter — only the 5 live rows
        urgent = store.get_unalerted_urgent(limit=200)
        ids = {a["_id"] for a in urgent}
        assert ids == {f"live{i}" for i in range(5)}, (
            "store filter let backtest rows through: "
            f"{ids - {f'live{i}' for i in range(5)}}"
        )

        # (b) Defense in depth — feed mixed list to the agent directly
        mixed = [
            {"_id": "bt99", "link": "backtest://run_99/d/x", "title": "bt",
             "source": "backtest_run_99_winner", "ai_score": 9.9,
             "summary": ""},
            {"_id": "opa9", "link": "https://x.com/opa9",
             "title": "opus annot", "source": "opus_annotation_cycle_9",
             "ai_score": 9.5, "summary": ""},
        ]
        # All synthetic — filtered list is empty, agent returns False without
        # touching claude_call or the webhook.
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(mixed, store)
        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()


# ───────────── Test 3: trainer never ingests its own ML predictions ─────────


class TestTrainerDataIntegrity:
    def test_fetch_training_data_excludes_ml_source(self, store):
        """Mix of LLM-labeled (good), ML-labeled (must exclude),
        and briefing-boosted (good) rows. The trainer must produce
        exactly llm + briefing_boost rows — never the ml ones."""
        for i in range(50):
            _insert_raw(store, id=f"llm{i}", url=f"https://x.com/llm{i}",
                        title=f"LLM title number {i} here ok",
                        ai_score=7.5, score_source="llm")
        for i in range(30):
            # These would create a feedback loop if trained on
            _insert_raw(store, id=f"ml{i}", url=f"https://x.com/ml{i}",
                        title=f"ML title number {i} here ok",
                        ai_score=6.0, ml_score=6.0, score_source="ml")
        for i in range(10):
            _insert_raw(store, id=f"bb{i}", url=f"https://x.com/bb{i}",
                        title=f"BB title number {i} here ok",
                        ai_score=8.0, score_source="briefing_boost")

        texts, articles, y_rel, y_urg, src = ml_trainer._fetch_training_data(store)

        # 50 (llm @7.5) + 10 (briefing_boost @8.0) = 60. Trainer also mixes
        # kw-only rows when ai_score=0 — none of our rows match that branch.
        assert len(texts) == 60
        rels = list(y_rel)
        assert rels.count(7.5) == 50, "missing some LLM rows"
        assert rels.count(8.0) == 10, "missing some briefing_boost rows"
        assert 6.0 not in rels, (
            "score_source='ml' leaked into training pool — feedback loop risk"
        )


# ─────────────── Test 4: ML score writes never touch ai_score ───────────────


class TestScoreSourceIntegrity:
    def test_update_ml_scores_does_not_touch_ai_or_source(self, store):
        """Pre-existing LLM-labeled rows must keep their ai_score AND
        score_source after the model writes its own predictions to ml_score."""
        for i in range(10):
            _insert_raw(store, id=f"a{i}", url=f"https://x.com/{i}",
                        title=f"t{i}", ai_score=8.0, score_source="llm")

        # Model writes its own predictions; should land in ml_score only
        store.update_ml_scores_batch([(f"a{i}", 0.5 + i * 0.05, 0) for i in range(10)])

        rows = list(store.conn.execute(
            "SELECT id, ai_score, ml_score, score_source FROM articles "
            "ORDER BY id"
        ).fetchall())
        assert len(rows) == 10
        for i, (rid, ai, ml, src) in enumerate(rows):
            assert ai == pytest.approx(8.0), (
                f"{rid}: ai_score clobbered by ml write ({ai} != 8.0)"
            )
            assert ml == pytest.approx(0.5 + i * 0.05)
            assert src == "llm", (
                f"{rid}: score_source clobbered ('llm' → {src!r})"
            )


# ─────────────────── Test 5: feature vector determinism ─────────────────────


class TestFeatureDeterminism:
    def test_extract_features_is_deterministic(self):
        """Same article → same feature vector (modulo wall-clock features).

        Index 6 (days_since_published) compares against now(); to make the test
        fully deterministic we publish far enough in the past that the
        feature saturates to 1.0."""
        old_dt = datetime.now(timezone.utc) - timedelta(days=100)
        art = {
            "title": "MU beats earnings, MSFT exposure cited",
            "summary": "Micron reported Q3 revenue $8.1B vs $7.9B est.",
            "source": "reuters",
            "published": old_dt.isoformat(),
        }
        v1 = ml_features.extract_features(art)
        v2 = ml_features.extract_features(art)
        np.testing.assert_array_equal(v1, v2)
        assert v1.shape == (ml_features.EXTRA_FEATURE_DIM,)

        # Dimension is also consistent across DIFFERENT articles
        other = {"title": "completely different headline",
                 "summary": "", "source": "rss",
                 "published": old_dt.isoformat()}
        v3 = ml_features.extract_features(other)
        assert v3.shape == v1.shape


# ───────────────────── Test 6: model output bounds ──────────────────────────


class TestModelOutputBounds:
    def test_outputs_bounded_over_random_inputs(self):
        """100 random batches through ArticleNetModule must each have
        relevance ∈ [0,10], urgency ∈ [0,1], finite, no NaN/Inf."""
        torch = pytest.importorskip("torch")
        from ml.model import ArticleNetModule

        net = ArticleNetModule(input_dim=64)
        net.eval()
        rng = np.random.default_rng(0xA17C1E)
        for _ in range(100):
            n = int(rng.integers(1, 32))
            # Spread inputs across reasonable scales: small, normal, large
            scale = float(rng.choice([0.01, 1.0, 10.0, 100.0]))
            x = torch.as_tensor(
                rng.standard_normal((n, 64)).astype(np.float32) * scale
            )
            with torch.no_grad():
                rel, urg, unc, tsens = net(x)
            for tag, t, lo, hi in (
                ("relevance",   rel,   0.0, 10.0),
                ("urgency",     urg,   0.0, 1.0),
                ("uncertainty", unc,   0.0, 1.0),
                ("time_sens",   tsens, 0.0, 1.0),
            ):
                a = t.cpu().numpy()
                assert np.isfinite(a).all(), f"{tag}: NaN/Inf in output"
                assert a.min() >= lo and a.max() <= hi, (
                    f"{tag}: out of range [{lo},{hi}] — actual "
                    f"[{a.min():.3f},{a.max():.3f}]"
                )


# ──────────────── Test 7: dedup across simulated collectors ─────────────────


class TestCollectorDeduplication:
    def test_same_url_from_two_collectors_inserts_once(self, store):
        """Two collectors find the same URL+title. Only one row exists, and the
        first writer's source wins (INSERT OR IGNORE)."""
        article = {
            "title": "Fed cuts rates by 25 bps in surprise move",
            "link": "https://reuters.com/fed-surprise",
            "source": "reuters_rss",
            "published": "",
            "summary": "Fed cuts.",
            "_relevance_score": 4.0,
        }
        # First collector (rss) inserts; second collector (gdelt) duplicates
        first = store.insert_batch([article])
        second_article = {**article, "source": "gdelt"}
        second = store.insert_batch([second_article])

        assert first == 1
        assert second == 0, "duplicate URL was re-inserted"
        rows = store.conn.execute(
            "SELECT id, source FROM articles WHERE url = ?",
            (article["link"],),
        ).fetchall()
        assert len(rows) == 1
        # Source from the first writer survives
        assert rows[0][1] == "reuters_rss"


# ───────────────── Test 8: urgency threshold consistency ────────────────────


class TestUrgencyThresholdConsistency:
    @pytest.mark.parametrize(
        "score, expect_urgent",
        [(1.0, False), (3.0, False), (5.0, False), (7.0, False),
         (7.9, False), (7.999, False),
         (8.0, True), (8.1, True), (9.0, True), (9.9, True), (10.0, True)],
    )
    def test_threshold_is_inclusive_at_eight(self, store, score, expect_urgent):
        """URGENT_THRESHOLD=8.0. The boundary must be ``>=`` — score=7.999 stays
        non-urgent, score=8.0 fires. A regression to ``>`` would silently delay
        every 8.0 alert by one Sonnet retry."""
        _insert_raw(store, id="x", url="https://x.com/1",
                    title="title goes here ok please now", kw_score=1.0)
        with _patched_claude([{"index": 0, "score": score, "reason": "t"}]):
            urgency_scorer.score_batch(
                [{"_id": "x", "title": "x", "summary": ""}], store,
            )
        row = store.conn.execute(
            "SELECT urgency FROM articles WHERE id='x'"
        ).fetchone()
        if expect_urgent:
            assert row[0] == 1, f"score={score} should be urgent"
        else:
            assert row[0] == 0, f"score={score} should NOT be urgent"


# ────────────────── Test 9: store thread safety on writes ───────────────────


class TestStoreThreadSafety:
    def test_concurrent_inserts_no_lost_writes(self, store):
        """Spawn 10 threads, each inserting 100 unique articles. Verify exactly
        1000 rows exist afterward and there are no SQLite threading errors.

        Each thread uses its own non-overlapping URL/title space so dedup
        cannot mask write loss."""
        N_THREADS = 10
        PER_THREAD = 100
        errors: list[BaseException] = []

        barrier = threading.Barrier(N_THREADS)

        def worker(tid: int):
            try:
                barrier.wait()  # maximize collision pressure on _write_lock
                batch = [
                    {
                        # Unique discriminator token ("id..xyz", >= 2 chars so it
                        # survives the dedup tokenizer's min-length filter)
                        # guarantees a genuinely non-overlapping title space.
                        # A bare "thread {tid}" prefix is NOT distinct: the
                        # single-digit id is dropped by tokenization, so cross-
                        # thread titles collapse and near-dedup masks write loss.
                        "title": f"thread {tid:02d} article {i:03d} id{tid:02d}{i:03d}xyz",
                        "link": f"https://x.com/{tid}/{i}",
                        "source": "rss",
                        "published": "",
                        "summary": "",
                        "_relevance_score": 1.0,
                    }
                    for i in range(PER_THREAD)
                ]
                store.insert_batch(batch)
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,))
                   for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"thread errors: {errors[:3]}"
        total = store.conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        assert total == N_THREADS * PER_THREAD, (
            f"lost writes: expected {N_THREADS * PER_THREAD}, got {total}"
        )
        # Ids must all be unique (sha256 of url||title)
        distinct = store.conn.execute(
            "SELECT COUNT(DISTINCT id) FROM articles"
        ).fetchone()[0]
        assert distinct == total, "duplicate ids generated under contention"


# ─────────────────────── Test 10: sample weight shape ───────────────────────


class TestSampleWeights:
    def test_weights_favor_high_relevance_articles(self):
        """The label-magnitude weighting formula used by trainer.py:
           w = clip((y_rel/10) ** EXP, 1e-3, inf) / mean(...)
        Mean(weights[y_rel=9]) must dominate mean(weights[y_rel=2]). All
        weights are positive, finite, and normalize to mean≈1."""
        y_rel = np.concatenate([
            np.full(20, 9.0, dtype=np.float32),  # Group A
            np.full(20, 2.0, dtype=np.float32),  # Group B
        ])
        w = np.power(
            np.clip(y_rel, 0.0, 10.0) / 10.0,
            ml_trainer.LABEL_WEIGHT_EXPONENT,
        )
        w = np.clip(w, 1e-3, None)
        w = w / w.mean()

        w_high = w[:20]
        w_low = w[20:]
        assert w_high.mean() > w_low.mean(), "high-rel group should weigh more"
        # Positive, finite, no NaN
        assert (w > 0).all(), "non-positive weight detected"
        assert np.isfinite(w).all(), "non-finite weight detected"
        # Normalized to mean ≈ 1
        assert w.mean() == pytest.approx(1.0, rel=1e-5)
