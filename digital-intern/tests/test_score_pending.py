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


def test_score_pending_prefloor_recap_and_quote_widget(store, monkeypatch):
    """Recap-template and quote-widget pseudo-articles MUST NOT reach urgency=1
    via the ML path, even when the urgency head would have scored them >= 8.

    Live evidence (2026-05-24): ``NVIDIA earnings: A quick glance at key
    metrics - MSN`` (matches the ``_RT_QUICK_GLANCE`` recap-template regex)
    reached urgency=2 with score_source='ml' and ml_score=10.0 — the LLM path
    already pre-floors this class in ``urgency_scorer.score_batch``, but a
    model-confident urgent (``needs_llm=False``, ``sc.urgency >= 8``) bypassed
    the gate and wrote urgency=1 directly. This regression guard pins the
    matching pre-floor on the ML path: a recap-template title that the model
    would have called urgent must be floored to ``ml_score=0.01 / urgency=0``
    without ever invoking the urgency head's score.

    score_articles is monkeypatched to return ``urgency=10`` for every row in
    the batch — without the pre-floor, the recap/quote-widget rows would write
    ``ml_score=10.0 / urgency=1`` and the assertion would fail. WITH the pre-
    floor, those rows are short-circuited before the stub runs, so they are
    never present in the ``batch`` ``score_articles`` sees.
    """
    # Two genuine recap-template hits (canonical _RT_QUICK_GLANCE fingerprint
    # + canonical _RT_WHY_DID fingerprint), one quote-widget (canonical
    # _QW_PCT_PAREN fingerprint), and one real urgent control article.
    _insert(store, id="recap_qg", url="https://reuters.com/qg",
            title="NVIDIA Earnings: A Quick Glance at Key Metrics",
            source="rss")
    _insert(store, id="recap_wd", url="https://reuters.com/wd",
            title="Why Did Micron Stock Drop Today?",
            source="rss")
    _insert(store, id="quote_widget", url="https://yahoo.com/q",
            title="NVDA NVIDIA Corporation 227.13 -8.61 (-3.65%)",
            source="rss")
    _insert(store, id="real_urgent", url="https://reuters.com/r",
            title="Fed cuts rates 50bps in emergency move",
            source="rss")

    seen_ids: list[str] = []

    def _fake_score_articles(batch):
        # The pre-floor must drop the pseudo-article rows BEFORE inference.
        # If a recap/quote-widget row shows up here, the gate is broken.
        for art in batch:
            seen_ids.append(art["_id"])
            assert art["_id"] not in {"recap_qg", "recap_wd", "quote_widget"}, (
                f"recap/quote-widget row {art['_id']!r} reached score_articles "
                f"— pre-floor gate is broken"
            )
        # Every surviving row gets a confident-urgent label so we can prove the
        # gate is what stopped the pseudo-articles, not the model's own logic.
        return [
            ArticleScore(relevance=9.0, urgency=10.0, rel_std=0.1,
                         urg_std=0.1, needs_llm=False,
                         confident_noise=False, time_sensitivity=0.9)
            for _ in batch
        ]

    import ml.inference as _inf
    monkeypatch.setattr(_inf, "score_articles", _fake_score_articles)

    n = store.score_pending(batch_size=500)

    # Three pre-floored + one real model-scored = 4 writes total.
    assert n == 4

    # The pseudo-articles were short-circuited: ml_score=0.01, urgency stays 0,
    # score_source='ml'. None of them should appear in ``seen_ids`` (the gate
    # ran before inference); the stub's hard assertion above proves that, and
    # the row values below pin the floor write itself.
    for aid in ("recap_qg", "recap_wd", "quote_widget"):
        ai, ml, src, urg, _ = _row(store, aid)
        assert ai == 0, (
            f"{aid}: ai_score must remain 0 — invariant 'model never writes "
            f"ai_score' is what protects the trainer's label pool"
        )
        assert ml == pytest.approx(0.01), (
            f"{aid}: pre-floor must write ml_score=0.01 (noise floor — "
            f"matches urgency_scorer.score_batch's pre-floor semantics)"
        )
        assert src == "ml", (
            f"{aid}: pre-floor must tag score_source='ml' (the analyst can "
            f"see ml-only rows in urgency_label_split — this row is one)"
        )
        assert urg == 0, (
            f"{aid}: pre-floor MUST keep urgency=0 — this is the bug being "
            f"fixed: a 10.0 urgency head prediction on a recap-template title "
            f"would otherwise bump urgency=1 → reach alert_worker → analyst "
            f"sees a 🚨 BREAKING on retrospective recap content"
        )

    # The real urgent article goes through the full ML path unchanged: model
    # ml_score=10.0, urgency promoted to 1.
    ai, ml, src, urg, _ = _row(store, "real_urgent")
    assert ai == 0
    assert ml == pytest.approx(10.0)
    assert src == "ml"
    assert urg == 1, (
        "real urgent article must still bump urgency to 1 — the pre-floor "
        "must be precise (only the recap/quote-widget class), never blanket"
    )
