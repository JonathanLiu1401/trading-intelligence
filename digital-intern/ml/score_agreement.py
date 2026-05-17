"""ML-vs-LLM score agreement analyzer — writes data/score_agreement.json.

The pipeline produces two independent importance scores per article:

- ``ml_score``  — the cheap local model (scores ~all articles, see model.py).
- ``ai_score``  — Sonnet's combined relevance/urgency judgement (0..10), only
  spent on the small subset the LLM was asked to grade (expensive).

Both live on the same 0..10 scale. The LLM score is the closest thing this
codebase has to ground truth, so the *gap* between the two on their overlap is
a standing model-drift / miscalibration signal: if the cheap model stops
tracking the expensive judge, the cheap model is no longer trustworthy as a
filter and the LLM budget is being spent to correct it.

This module is read-only (SELECT-only) and dependency-free — the stats are
hand-rolled so the test suite needs no numpy/scipy. ``compute_agreement`` is a
pure function over ``(ml, ai, ...)`` rows and is the unit-tested contract; the
DB plumbing around it is a thin shell.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "score_agreement.json"

# Only the overlap where both scorers actually produced a value is meaningful.
# ai_score defaults to 0 when the LLM never graded the row, so require > 0.
_MIN_AI = 0.01
# A "strong disagreement" is a gap this large on the shared 0..10 scale.
DIVERGENCE_THRESHOLD = 4.0
# Cap on how many exemplar rows we surface in each divergence direction.
TOP_N = 15


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation. Returns 0.0 for n<2 or a degenerate (flat) series."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0.0 or syy == 0.0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def _rank(values: Sequence[float]) -> list[float]:
    """Average-rank transform (ties share the mean of their rank span)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation — robust to the two scales being nonlinear."""
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    return pearson(_rank(xs), _rank(ys))


def compute_agreement(rows: Sequence[dict]) -> dict:
    """Summarize agreement between ml_score and ai_score over ``rows``.

    Each row needs ``ml_score`` and ``ai_score`` (floats on 0..10); optional
    ``title``/``source``/``first_seen`` are echoed into the exemplar lists.
    Pure and deterministic — this is the unit-tested surface.
    """
    pairs = [
        r
        for r in rows
        if r.get("ml_score") is not None and (r.get("ai_score") or 0) >= _MIN_AI
    ]
    n = len(pairs)
    if n == 0:
        return {
            "n": 0,
            "pearson": 0.0,
            "spearman": 0.0,
            "mean_abs_divergence": 0.0,
            "rmse": 0.0,
            "bias_ml_minus_ai": 0.0,
            "strong_disagreement_count": 0,
            "model_overconfident": [],
            "model_underconfident": [],
        }

    ml = [float(r["ml_score"]) for r in pairs]
    ai = [float(r["ai_score"]) for r in pairs]
    diffs = [m - a for m, a in zip(ml, ai)]

    # Model says important, LLM says it isn't (false-positive risk) and the
    # reverse (the cheap model would have filtered out something the LLM
    # flagged — the expensive miss). Surface the worst, most recent of each.
    over = sorted(
        (
            {
                "ml_score": round(float(r["ml_score"]), 3),
                "ai_score": round(float(r["ai_score"]), 3),
                "gap": round(float(r["ml_score"]) - float(r["ai_score"]), 3),
                "title": (r.get("title") or "")[:120],
                "source": r.get("source") or "",
                "first_seen": r.get("first_seen") or "",
            }
            for r in pairs
            if float(r["ml_score"]) - float(r["ai_score"]) >= DIVERGENCE_THRESHOLD
        ),
        key=lambda d: d["gap"],
        reverse=True,
    )[:TOP_N]
    under = sorted(
        (
            {
                "ml_score": round(float(r["ml_score"]), 3),
                "ai_score": round(float(r["ai_score"]), 3),
                "gap": round(float(r["ai_score"]) - float(r["ml_score"]), 3),
                "title": (r.get("title") or "")[:120],
                "source": r.get("source") or "",
                "first_seen": r.get("first_seen") or "",
            }
            for r in pairs
            if float(r["ai_score"]) - float(r["ml_score"]) >= DIVERGENCE_THRESHOLD
        ),
        key=lambda d: d["gap"],
        reverse=True,
    )[:TOP_N]

    strong = sum(1 for d in diffs if abs(d) >= DIVERGENCE_THRESHOLD)
    return {
        "n": n,
        "pearson": round(pearson(ml, ai), 4),
        "spearman": round(spearman(ml, ai), 4),
        "mean_abs_divergence": round(_mean([abs(d) for d in diffs]), 4),
        "rmse": round((_mean([d * d for d in diffs])) ** 0.5, 4),
        "bias_ml_minus_ai": round(_mean(diffs), 4),
        "strong_disagreement_count": strong,
        "strong_disagreement_pct": round(100.0 * strong / n, 2),
        "model_overconfident": over,
        "model_underconfident": under,
    }


def _db_path() -> Path:
    """Resolve the live articles.db the same way storage.article_store does."""
    from storage import article_store

    return Path(article_store._get_db_path())


def load_rows(db_path: Path, limit: int = 20000) -> list[dict]:
    """Load the most recent overlap rows (read-only) for analysis."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT ml_score, ai_score, title, source, first_seen
              FROM articles
             WHERE ml_score IS NOT NULL AND ai_score >= ?
             ORDER BY first_seen DESC
             LIMIT ?
            """,
            (_MIN_AI, limit),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def run(write: bool = True) -> dict:
    """Compute the report against the live DB and (optionally) persist it."""
    rows = load_rows(_db_path())
    report = compute_agreement(rows)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    if write:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUTPUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(OUTPUT_PATH)
    return report


if __name__ == "__main__":
    r = run()
    print(
        f"[score_agreement] n={r['n']} pearson={r['pearson']} "
        f"spearman={r['spearman']} mad={r['mean_abs_divergence']} "
        f"bias={r['bias_ml_minus_ai']} strong={r['strong_disagreement_count']}"
        f" ({r.get('strong_disagreement_pct', 0)}%)"
    )
