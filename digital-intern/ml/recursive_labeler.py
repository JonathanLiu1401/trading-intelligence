"""
Recursive Claude labeling pipeline — three-tier active learning.

Tiers (per run):
  Round 1 — Sonnet labels the bulk: oldest unlabeled / weakly-labeled articles
            (ai_score == 0 or ai_score < 2.0). Cheap, high volume.
  Round 2 — Train ML, then Sonnet re-labels articles where the uncertainty
            head fires (>0.7), giving higher quality on the hard cases.
  Round 3 — Opus 4.7 handles only the top 5% by uncertainty — small batch,
            highest cost, used as the gold-standard label.
  Final  — Retrain so the next inference cycle benefits from the new labels.

Triggered every 4h by the recursive_labeler_worker in daemon.py. Caps at
500 articles per run (round 1) so we never blow the Claude quota in one shot.

Outputs labels via ArticleStore.update_ai_scores_batch — the LLM relevance
(0..5) is converted to the existing 0..10 scale by `* 2.0`. ``in_briefing``
style flags are not produced; we only update ai_score + urgency.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np

from core.claude_cli import claude_call
from core.json_extract import extract_json_array
from core.logger import get_logger

log = get_logger("recursive_labeler")

SONNET_MODEL = "claude-sonnet-4-6"
OPUS_MODEL   = "claude-opus-4-7"

BATCH_SIZE   = 20            # per Claude call — keeps the JSON fits in stdout
ROUND1_CAP   = 500           # articles labeled per run (round 1 only)
ROUND2_UNC_THRESHOLD = 0.7   # uncertainty-head value above which we re-label
ROUND3_TOP_PCT = 0.05        # top 5% by uncertainty go to Opus
LLM_TIMEOUT  = 180

LABELING_PROMPT = """You are a financial signal labeler. Rate these articles for trading relevance.

For each article, return a JSON object with:
- "url": the article URL
- "relevance": float 0.0-5.0 (0=spam/noise, 5=critical market-moving news)
- "urgency": int 0 or 1 (1=needs immediate attention)
- "tickers": list of relevant ticker symbols mentioned
- "sentiment": "bullish", "bearish", or "neutral"
- "reasoning": one sentence why this score

Articles to label:
{articles}

Return a JSON array of objects, one per article. No other text."""


@dataclass
class RoundStats:
    name: str
    requested: int = 0
    labeled: int = 0
    failures: int = 0
    elapsed_s: float = 0.0


def _articles_for_prompt(articles: list[dict]) -> str:
    payload = []
    for a in articles:
        url = a.get("url") or a.get("link") or ""
        title = (a.get("title") or "")[:200]
        summary = (a.get("summary") or "")[:400]
        payload.append({"url": url, "title": title, "summary": summary})
    return json.dumps(payload, ensure_ascii=False)


def _label_batch(articles: list[dict], model: str) -> list[dict] | None:
    if not articles:
        return []
    prompt = LABELING_PROMPT.format(articles=_articles_for_prompt(articles))
    raw = claude_call(prompt, model=model, timeout=LLM_TIMEOUT)
    if raw is None:
        return None
    parsed = extract_json_array(raw)
    if parsed is None:
        log.warning(f"[recursive_labeler] failed to parse JSON ({len(raw)} chars) "
                    f"— first 200: {raw[:200]!r}")
        return None
    return parsed


def _apply_labels(store, articles: list[dict], labels: list[dict]) -> int:
    """Map LLM labels back to articles by URL; write to the store. Returns count."""
    if not labels:
        return 0
    by_url: dict[str, dict] = {}
    for art in articles:
        u = art.get("url") or art.get("link") or ""
        if u:
            by_url[u] = art

    updates: list[tuple[str, float, int]] = []
    for label in labels:
        if not isinstance(label, dict):
            continue
        url = label.get("url")
        if not url or url not in by_url:
            continue
        art = by_url[url]
        aid = art.get("_id") or art.get("id")
        if not aid:
            continue
        try:
            relevance = float(label.get("relevance", 0)) * 2.0  # 0..5 → 0..10
        except (TypeError, ValueError):
            continue
        relevance = max(0.0, min(10.0, relevance))
        # Parse urgency defensively. Claude does not reliably return a bare
        # int here (observed: "1", "1.0", "yes", true). An unguarded int()
        # raised ValueError that escaped _apply_labels → _run_round →
        # run_recursive_labeling (no inner handler on that path), aborting the
        # rest of the 4h labeling cycle AND discarding this batch's
        # already-collected good labels (the exception fires before
        # update_ai_scores_batch). Mirror the relevance guard: a junk urgency
        # degrades to 0, it never aborts and never loses the relevance label.
        try:
            urgency = 1 if int(float(label.get("urgency", 0) or 0)) >= 1 else 0
        except (TypeError, ValueError):
            urgency = 0
        updates.append((aid, relevance, urgency))

    if updates:
        store.update_ai_scores_batch(updates)
    return len(updates)


def _fetch_round1_candidates(store, limit: int) -> list[dict]:
    """Oldest unlabeled / weakly-labeled articles. Returns dicts ready for prompting.

    Excludes backtest replays and Opus annotation rows — those carry historical
    labels from offline runs and must not be re-scored against the live model.
    """
    from storage.article_store import decompress
    cur = store.conn.execute(
        "SELECT id, url, title, source, full_text, ai_score "
        "FROM articles "
        "WHERE ai_score < 2.0 "
        "AND url NOT LIKE 'backtest://%' "
        "AND source NOT LIKE 'backtest_%' "
        "AND source NOT LIKE 'opus_annotation%' "
        "ORDER BY first_seen ASC "
        "LIMIT ?",
        (limit,),
    )
    out = []
    for r in cur.fetchall():
        aid, url, title, source, blob, ai_score = r
        if not url or not title:
            continue
        out.append({
            "_id": aid, "url": url, "link": url, "title": title,
            "source": source or "",
            "summary": decompress(blob) if blob else "",
        })
    return out


def _score_with_uncertainty(articles: list[dict]) -> np.ndarray | None:
    """Run the model's uncertainty head over articles. Returns shape (N,) or None
    if the embedder/model isn't ready yet."""
    if not articles:
        return None
    from ml.embedder import get_embedder
    from ml.features import extract_features_batch
    from ml.model import get_model

    emb = get_embedder()
    model = get_model()
    if not emb.fitted or not model.fitted:
        return None
    texts = [f"{a.get('title', '')} {a.get('summary', '')}" for a in articles]
    try:
        X_text = emb.transform(texts)
    except Exception as e:
        log.warning(f"[recursive_labeler] embedder.transform failed: {e}")
        return None
    X_extra = extract_features_batch(articles)
    X = np.concatenate([X_text, X_extra], axis=1).astype(np.float32)
    _, _, unc = model.predict_with_uncertainty(X)
    return unc


def _run_round(round_name: str, articles: list[dict], store, model: str) -> RoundStats:
    stats = RoundStats(name=round_name, requested=len(articles))
    if not articles:
        return stats
    t0 = time.time()
    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i:i + BATCH_SIZE]
        labels = _label_batch(batch, model)
        if labels is None:
            stats.failures += 1
            continue
        stats.labeled += _apply_labels(store, batch, labels)
    stats.elapsed_s = round(time.time() - t0, 1)
    return stats


def _retrain(store, why: str) -> dict:
    try:
        from ml.trainer import train as ml_train
        metrics = ml_train(store)
        log.info(f"[recursive_labeler] retrain ({why}): "
                 f"n={metrics.get('n')} loss={metrics.get('final_loss')}")
        return metrics
    except Exception as e:
        log.warning(f"[recursive_labeler] retrain ({why}) failed: {e}")
        return {"status": "error", "error": str(e)}


def run_recursive_labeling(store, round1_cap: int = ROUND1_CAP) -> dict:
    """Execute the three-round labeling pipeline once. Returns aggregate stats."""
    overall_t0 = time.time()
    summary: dict = {"rounds": []}

    # ── Round 1: Sonnet labels the unlabeled bulk ───────────────────────────
    candidates = _fetch_round1_candidates(store, limit=round1_cap)
    log.info(f"[recursive_labeler] round=1 candidates={len(candidates)}")
    r1 = _run_round("round1_sonnet", candidates, store, model=SONNET_MODEL)
    log.info(f"[recursive_labeler] round=1 labeled={r1.labeled} "
             f"failures={r1.failures} elapsed={r1.elapsed_s}s")
    summary["rounds"].append(r1.__dict__)

    if r1.labeled > 0:
        summary["retrain_after_r1"] = _retrain(store, "after_round1")

    # ── Round 2: model re-scores; Sonnet re-labels uncertain ones ───────────
    uncertain_articles: list[dict] = []
    if candidates:
        unc = _score_with_uncertainty(candidates)
        if unc is not None:
            order = sorted(
                range(len(candidates)),
                key=lambda i: float(unc[i]),
                reverse=True,
            )
            uncertain_articles = [
                candidates[i] for i in order if unc[i] >= ROUND2_UNC_THRESHOLD
            ]

    log.info(f"[recursive_labeler] round=2 uncertain={len(uncertain_articles)}")
    # Cap round 2 to avoid runaway cost when many articles are uncertain.
    r2_cap = min(len(uncertain_articles), 200)
    r2 = _run_round("round2_sonnet", uncertain_articles[:r2_cap], store, model=SONNET_MODEL)
    log.info(f"[recursive_labeler] round=2 labeled={r2.labeled} "
             f"failures={r2.failures} elapsed={r2.elapsed_s}s")
    summary["rounds"].append(r2.__dict__)

    # ── Round 3: Opus on the top 5% by uncertainty ──────────────────────────
    top_n = max(1, math.ceil(len(uncertain_articles) * ROUND3_TOP_PCT))
    # Cap at a small absolute number — Opus is the expensive tier.
    top_n = min(top_n, 50)
    opus_targets = uncertain_articles[:top_n] if uncertain_articles else []
    log.info(f"[recursive_labeler] round=3 opus_targets={len(opus_targets)}")
    r3 = _run_round("round3_opus", opus_targets, store, model=OPUS_MODEL)
    log.info(f"[recursive_labeler] round=3 labeled={r3.labeled} "
             f"failures={r3.failures} elapsed={r3.elapsed_s}s")
    summary["rounds"].append(r3.__dict__)

    # ── Final retrain ───────────────────────────────────────────────────────
    if r2.labeled + r3.labeled > 0:
        summary["retrain_final"] = _retrain(store, "final")

    summary["total_labeled"] = r1.labeled + r2.labeled + r3.labeled
    summary["elapsed_s"] = round(time.time() - overall_t0, 1)
    log.info(f"[recursive_labeler] DONE total_labeled={summary['total_labeled']} "
             f"elapsed={summary['elapsed_s']}s")
    return summary
