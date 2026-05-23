"""Trainer label sourcing — model must NOT train on its own predictions."""
from __future__ import annotations

import numpy as np
import pytest

from ml import trainer


def _insert(store, *, id, title, ai_score, score_source, kw_score=0.0, url=None,
            source="rss"):
    if url is None:
        url = f"https://x.com/{id}"
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, 0,
             "2026-05-15T00:00:00+00:00", 0, None, score_source),
        )
        store.conn.commit()


class TestLabelSourcing:
    def test_excludes_ml_scored_rows(self, store):
        """The trainer must never ingest score_source='ml' rows — that would
        re-feed model outputs as ground truth and collapse the loss to fit its
        own predictions instead of the LLM signal."""
        # 25 LLM-labeled rows (integer score), 25 ml-labeled rows.
        for i in range(25):
            _insert(store, id=f"llm{i}", title=f"llm title {i}",
                    ai_score=7.0, score_source="llm")
            _insert(store, id=f"ml{i}", title=f"ml title {i}",
                    ai_score=8.5, score_source="ml")

        texts, articles, y_rel, y_urg, src = trainer._fetch_training_data(store)
        # The 25 ml rows must be absent; only the 25 llm rows should remain.
        # 8.5 is the ml-only label — its absence proves the filter works.
        rels = list(y_rel)
        assert 8.5 not in rels, "ml-tagged rows leaked into training pool"
        assert rels.count(7.0) == 25

    def test_includes_synthetic_backtest_rows(self, store):
        """Backtest/opus synthetic rows have score_source=NULL and may carry
        fractional ai_score (SELL=0.5, NEUTRAL=2.5). CLAUDE.md §5 says these
        must be included in the training pool. Verify."""
        _insert(store, id="bt_win", title="backtest winner one entry here",
                ai_score=5.0, score_source=None,
                url="backtest://run_1/d/BUY/MU", source="backtest_run_1_winner")
        _insert(store, id="bt_sell", title="backtest sell row two entries",
                ai_score=0.5, score_source=None,
                url="backtest://run_1/d/SELL/MU", source="backtest_run_1_loser")
        _insert(store, id="opus", title="opus annotation entry here please",
                ai_score=2.5, score_source=None,
                url="https://x.com/opus", source="opus_annotation_cycle_1")
        # Add live LLM rows so the function doesn't bail on too-few-samples.
        for i in range(30):
            _insert(store, id=f"l{i}", title=f"llm {i}", ai_score=6.0,
                    score_source="llm")

        texts, articles, y_rel, y_urg, src = trainer._fetch_training_data(store)
        rels = list(y_rel)
        assert 5.0 in rels, "synthetic BUY winner missing"
        assert 0.5 in rels, "synthetic SELL row missing"
        assert 2.5 in rels, "opus NEUTRAL label missing"

    def test_legacy_integer_ai_scores_included(self, store):
        """Rows predating the score_source migration carry score_source=NULL
        but integer ai_score (Sonnet returned ints). They must be included."""
        for i in range(30):
            _insert(store, id=f"legacy{i}", title=f"legacy {i}",
                    ai_score=6.0, score_source=None)
        texts, _, y_rel, _, _ = trainer._fetch_training_data(store)
        assert len(texts) >= 30
        assert (y_rel == 6.0).all()


class TestBriefingSamplesBacktestIsolation:
    """``_fetch_briefing_samples`` derives extra training labels by matching
    article-title prefixes against recent Opus heartbeat briefings. Pre-fix the
    candidate scan was an unfiltered ``ORDER BY first_seen DESC LIMIT 5000`` —
    the only live-pool read in the trainer module without ``_LIVE_ONLY_CLAUSE``.

    Briefings derive from ``get_top_for_briefing`` (live-only) so the title
    prefix should only ever match live news in practice; a synthetic backtest
    row's 40-char title prefix happening to collide with briefing prose was
    rare but possible — exactly the partial-filter regression class
    ``analytics/trend_velocity.py`` violates and the rest of the codebase pins
    via defense-in-depth. If the filter is ever removed this fails by surfacing
    the synthetic row as a 4.5 training label.
    """

    def test_synthetic_titles_in_briefing_text_do_not_become_labels(self, store):
        # Save a briefing whose text contains a 40-char prefix that ALSO exists
        # as a synthetic backtest row's title. Without the live-only filter, the
        # synthetic row would be re-labeled rel=4.5 (4.5-tag re-injection on a
        # row whose REAL outcome label is 0.5 SELL-loser).
        unique_prefix = "Nvidia accelerates DRAM HBM3e supply chai"
        synth_title = unique_prefix + "n shifts after Q1"
        # 30 LLM rows so the function doesn't bail on too-few-samples.
        for i in range(30):
            _insert(store, id=f"llm{i}", title=f"llm article {i} long enough",
                    ai_score=6.0, score_source="llm")
        # Synthetic backtest row carrying its own fractional outcome label.
        _insert(store, id="bt_loser", title=synth_title, ai_score=0.5,
                score_source=None,
                url="backtest://run_77/2026-05-21/SELL/NVDA",
                source="backtest_run_77_loser")
        # Briefing that mentions the synthetic title prefix verbatim.
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO briefings (ts, text, article_count) VALUES (?,?,?)",
                ("2026-05-21T12:00:00+00:00",
                 f"Opus heartbeat. Top: {unique_prefix} matters today.",
                 1),
            )
            store.conn.commit()

        texts, articles, rels, urgs = trainer._fetch_briefing_samples(store)
        # The synthetic row's title MUST NOT appear among the briefing-derived
        # positive labels; the live-only clause excludes the row from the scan.
        for a, r in zip(articles, rels):
            assert a["title"] != synth_title, (
                f"synthetic backtest row leaked into briefing-derived training "
                f"labels (would inject rel=4.5 over its real 0.5 SELL outcome)"
            )
        # And the synthetic row's source tag must never appear in this output.
        for a in articles:
            assert not a["source"].startswith("backtest_"), (
                "backtest_ source tag in briefing-derived training pool"
            )


class TestSampleWeights:
    def test_high_relevance_weighs_more_than_low(self):
        """The fit loss is sample-weighted by (y_rel/10)^EXP — a 9.0 article
        must contribute more gradient than a 2.0 article. Confirm the formula
        used in trainer.py matches that contract."""
        y_rel = np.array([2.0, 5.0, 9.0, 9.5], dtype=np.float32)
        w = np.power(
            np.clip(y_rel, 0.0, 10.0) / 10.0,
            trainer.LABEL_WEIGHT_EXPONENT,
        )
        assert w[3] > w[2] > w[1] > w[0]
        # exp=2 → 9.0 weighs ~20x more than 2.0 pre-normalization.
        assert w[2] / w[0] > 10.0


class _StubEmbedder:
    """Deterministic, torch-free stand-in for the global TF-IDF embedder so
    train() orchestration can be exercised without fitting a real vectorizer
    or mutating the process-global singleton."""

    fitted = True

    def should_refit(self, n):
        return False

    def transform(self, texts):
        return np.zeros((len(texts), 8), dtype=np.float32)

    def fit_transform(self, texts):
        return np.zeros((len(texts), 8), dtype=np.float32)


class _StubModel:
    """Captures the X/y arrays train() hands to ArticleNet.fit and returns a
    plausible metrics dict, so the test asserts the orchestration (not torch)."""

    fitted = False

    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y_rel, y_urg, **kw):
        self.fit_calls.append((X.shape, len(y_rel)))
        return {
            "final_loss": 0.12, "val_loss": 0.15, "best_in_run": 0.15,
            "new_best": True, "epochs": kw.get("epochs", 1),
            "device": "cpu",
        }


class _CaptureModel:
    """Stub that records the y_rel array train_continuous() hands to
    ArticleNet.fit, so the label-sourcing SQL can be asserted without torch."""

    fitted = False

    def __init__(self):
        self.captured = None

    def fit(self, X, y_rel, y_urg, **kw):
        self.captured = y_rel
        return {
            "final_loss": 0.10, "val_loss": 0.12, "best_in_run": 0.12,
            "new_best": False, "epochs": kw.get("epochs", 1), "device": "cpu",
        }


class TestContinuousLabelSourcing:
    """train_continuous() carries its OWN inlined copy of the strong-label
    SQL (trainer.py ~715) — a hot-path duplicate of _fetch_training_data that
    runs every 2 min (CONTINUOUS_TRAIN_INTERVAL=120) vs the full trainer's
    3 min. TestLabelSourcing only pins _fetch_training_data; the duplicate is
    unguarded. If it ever drifts to match score_source='ml' rows, the
    continuous trainer silently re-feeds the model its own predictions as
    ground truth — the exact label-feedback loop the ml_score/ai_score split
    exists to prevent — with no exception and a healthy-looking daemon log.
    Same drift class as the dashboard-parity / vendored-signals cases.
    """

    def test_excludes_ml_and_includes_synthetic_and_llm(self, store,
                                                         monkeypatch):
        monkeypatch.setattr(trainer, "get_embedder", lambda: _StubEmbedder())
        cap = _CaptureModel()
        monkeypatch.setattr(trainer, "get_model", lambda: cap)
        monkeypatch.setattr(trainer, "_log_metrics", lambda rec: None)
        # train_continuous mutates this module global on success; pin it so the
        # test is order-independent (monkeypatch restores it afterward).
        monkeypatch.setattr(trainer, "_last_continuous_loss", float("inf"))

        # score_source='ml' self-predictions — must NEVER reach training.
        for i in range(5):
            _insert(store, id=f"ml{i}", title=f"ml self-pred row {i} here",
                    ai_score=8.5, score_source="ml")
        # Synthetic backtest winner — must be included (CLAUDE.md §5).
        _insert(store, id="bt_win", title="backtest winner one entry here",
                ai_score=5.0, score_source=None,
                url="backtest://run_9/d/BUY/MU",
                source="backtest_run_9_winner")
        # LLM ground truth — must be included.
        for i in range(30):
            _insert(store, id=f"llm{i}", title=f"llm ground truth row {i} ok",
                    ai_score=7.0, score_source="llm")

        result = trainer.train_continuous(store)

        assert result["status"] == "ok", result
        assert cap.captured is not None, "model.fit was never reached"
        rels = list(cap.captured)
        assert 8.5 not in rels, \
            "score_source='ml' row leaked into the train_continuous pool"
        assert 5.0 in rels, "synthetic backtest winner missing"
        assert rels.count(7.0) == 30, "llm ground-truth rows missing"
        # n counts only the kept rows (1 synthetic + 30 llm), never the 5 ml.
        assert result["n"] == 31, result


class TestTrainOrchestration:
    """Regression guard for the dataset-cache bug (commit 17e414b): train()
    referenced ``texts``/``articles`` after both code paths had stopped
    defining them — a NameError on EVERY cycle, so ArticleNet silently never
    retrained in production while the daemon log looked healthy. These two
    cases cover both branches; the second is the one that always raised."""

    def _patch(self, monkeypatch, tmp_path):
        model = _StubModel()
        monkeypatch.setattr(trainer, "get_embedder", lambda: _StubEmbedder())
        monkeypatch.setattr(trainer, "get_model", lambda: model)
        monkeypatch.setattr(trainer, "_log_metrics", lambda rec: None)
        # Redirect the on-disk dataset cache into the test's tmp dir so the
        # two calls genuinely exercise the write-then-reload path in isolation.
        monkeypatch.setattr(trainer, "_ML_DIR", tmp_path)
        monkeypatch.setattr(trainer, "_DATASET_CACHE", tmp_path / "dataset_cache.npz")
        monkeypatch.setattr(trainer, "_DATASET_META", tmp_path / "dataset_cache_meta.json")
        return model

    def test_fresh_path_does_not_raise_and_trains(self, store, monkeypatch,
                                                  tmp_path):
        """No cache on disk → _fetch_training_data → embed → train. Pre-fix
        this raised NameError on the leftover re-embed block."""
        model = self._patch(monkeypatch, tmp_path)
        for i in range(40):
            _insert(store, id=f"llm{i}", title=f"llm article number {i} here",
                    ai_score=7.0, score_source="llm")

        result = trainer.train(store)

        assert result["status"] == "ok", result
        assert result["n"] == 40
        # X = TF-IDF stub (8) + extra features (15) = 23 columns, 40 rows.
        assert model.fit_calls == [((40, 23), 40)]
        assert (tmp_path / "dataset_cache.npz").exists(), "cache not persisted"

    def test_cached_path_does_not_raise(self, store, monkeypatch, tmp_path):
        """Second call loads the freshly-written disk cache. This is the exact
        production path that NameError'd every 3 minutes pre-fix — the cache
        branch never binds ``texts``/``articles`` at all."""
        model = self._patch(monkeypatch, tmp_path)
        for i in range(40):
            _insert(store, id=f"llm{i}", title=f"llm article number {i} here",
                    ai_score=7.0, score_source="llm")

        first = trainer.train(store)            # writes cache
        second = trainer.train(store)           # reads cache (the bug path)

        assert first["status"] == "ok"
        assert second["status"] == "ok", second
        assert second["n"] == 40
        # Both calls reached model.fit with the same (n, dim) shape.
        assert model.fit_calls == [((40, 23), 40), ((40, 23), 40)]
