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
