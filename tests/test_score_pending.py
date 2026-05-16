"""``ArticleStore.score_pending`` — ML/LLM score separation on the in-store
inference path.

``score_pending`` is a documented public API (CLAUDE.md references it) that
runs ``ml.inference.score_articles`` and persists the result. It is the second
implementation of the model-scoring step (``daemon.scorer_worker`` is the
production path; this one is used by scripted / one-shot drivers). It is the
only model-write path with NO direct test, and a regression here silently
reintroduces the exact label-feedback bug the system is built to prevent:

  * model predictions MUST land in ``ml_score`` (never ``ai_score``) with
    ``score_source='ml'`` — the trainer reads ``ai_score`` as ground truth, so
    a model output in ``ai_score`` makes ArticleNet train on itself;
  * ``needs_llm`` articles MUST be left ``ai_score=0 / ml_score=NULL`` so the
    Sonnet path re-picks them (writing a model score here would suppress the
    LLM label entirely);
  * synthetic ``backtest://`` rows MUST never be scored (they are excluded by
    ``get_unscored``'s ``_LIVE_ONLY_CLAUSE``);
  * an urgent prediction (urgency >= 8) MUST bump ``urgency`` to 1 so the
    alert path still works, via ``MAX(urgency, ?)``.

The stub for ``score_articles`` keys off ``_id`` so the assertion is
independent of ``get_unscored``'s ``kw_score DESC`` ordering.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from ml.inference import ArticleScore


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert(store, *, id, url, title, source, kw_score=1.0):
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, 0.0, 0,
             _recent_iso(), 0, None, None),
        )
        store.conn.commit()


def _row(store, aid):
    return store.conn.execute(
        "SELECT ai_score, ml_score, score_source, urgency, time_sensitivity "
        "FROM articles WHERE id=?",
        (aid,),
    ).fetchone()


def test_score_pending_separates_ml_from_llm_and_isolates_backtest(
    store, monkeypatch
):
    # Three live rows + one synthetic backtest row.
    _insert(store, id="conf", url="https://reuters.com/a",
            title="Confident relevant article", source="rss")
    _insert(store, id="urg", url="https://reuters.com/b",
            title="Confident urgent article", source="rss")
    _insert(store, id="grey", url="https://reuters.com/c",
            title="Grey-zone article routed to Sonnet", source="rss")
    _insert(store, id="bt", url="backtest://run_1/2026-01-01/BUY/MU",
            title="Synthetic backtest row", source="backtest_run_1")

    scores = {
        # confident, not urgent → ml_score = max(rel,urg,0.01) = 7.0
        "conf": ArticleScore(relevance=7.0, urgency=2.0, rel_std=0.5,
                             urg_std=0.4, needs_llm=False,
                             confident_noise=False, time_sensitivity=0.80),
        # confident + urgent (urg>=8) → ml_score=9.0, urgency promoted to 1
        "urg": ArticleScore(relevance=6.0, urgency=9.0, rel_std=0.3,
                            urg_std=0.3, needs_llm=False,
                            confident_noise=False, time_sensitivity=0.90),
        # uncertain → routed to LLM, must stay ai_score=0 / ml_score=NULL
        "grey": ArticleScore(relevance=5.0, urgency=7.5, rel_std=0.3,
                             urg_std=0.3, needs_llm=True,
                             confident_noise=False, time_sensitivity=0.50),
    }

    def _fake_score_articles(batch):
        # batch is the get_unscored projection (dicts with "_id"); the synthetic
        # row must never reach here — assert that as a hard invariant.
        for art in batch:
            assert not art["link"].startswith("backtest://"), (
                "backtest row reached the scorer — _LIVE_ONLY_CLAUSE breached"
            )
        return [scores[art["_id"]] for art in batch]

    import ml.inference as _inf
    monkeypatch.setattr(_inf, "score_articles", _fake_score_articles)

    n = store.score_pending(batch_size=500)

    # Two model-scored rows (conf, urg); grey was skipped for the LLM.
    assert n == 2

    ai, ml, src, urg, ts = _row(store, "conf")
    assert ai == 0, "model output must NOT pollute ai_score"
    assert ml == pytest.approx(7.0)
    assert src == "ml"
    assert urg == 0
    assert ts == pytest.approx(0.80)

    ai, ml, src, urg, ts = _row(store, "urg")
    assert ai == 0
    assert ml == pytest.approx(9.0)
    assert src == "ml"
    assert urg == 1, "urgency>=8 prediction must bump urgency to 1"
    assert ts == pytest.approx(0.90)

    ai, ml, src, urg, ts = _row(store, "grey")
    assert ai == 0, "needs_llm row must stay ai_score=0 for the Sonnet path"
    assert ml is None, "needs_llm row must NOT receive a model ml_score"
    assert src is None, "needs_llm row must remain unscored (score_source NULL)"
    # time_sensitivity is still persisted for a real prediction even when the
    # relevance call is deferred to the LLM (heads are independent).
    assert ts == pytest.approx(0.50)

    ai, ml, src, urg, ts = _row(store, "bt")
    assert (ai, ml, src, urg, ts) == (0.0, None, None, 0, None), (
        "synthetic backtest row was scored — it must be invisible to score_pending"
    )

    # The grey row is still pending for the LLM path.
    pending = {a["_id"] for a in store.get_unscored(min_kw=0.0)}
    assert pending == {"grey"}


def test_score_pending_noop_when_model_unfitted(store, monkeypatch):
    """The unfitted-model sentinel (rel_std==99, needs_llm=True) must not write
    any ml_score, and must not spin: score_pending returns 0 and the row stays
    fully unscored so the LLM path can take it."""
    _insert(store, id="x", url="https://reuters.com/x",
            title="Article seen before model is fitted", source="rss")

    sentinel = ArticleScore(0, 0, 99, 99, needs_llm=True,
                            confident_noise=False, priority=1.0,
                            time_sensitivity=0.5)
    import ml.inference as _inf
    monkeypatch.setattr(_inf, "score_articles", lambda batch: [sentinel] * len(batch))

    n = store.score_pending(batch_size=500)

    assert n == 0
    ai, ml, src, urg, ts = _row(store, "x")
    assert (ai, ml, src, urg) == (0.0, None, None, 0)
    # rel_std==99 sentinel → time_sensitivity must NOT be persisted (not a real
    # prediction).
    assert ts is None
