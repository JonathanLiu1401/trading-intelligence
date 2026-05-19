"""Per-source ML-vs-LLM score agreement breakdown.

Companion to ``ml/score_agreement.py``. That module reports a single aggregate
across the whole overlap; this one slices the same overlap *by source*. The
aggregate hides per-source heterogeneity — Sonnet's calibration on
``finnhub`` items can be excellent while ``substack`` is wildly off, and the
mean gives one number that says neither.

Why this is useful, operationally:

  * ``ml/llm_promotion_audit.py`` answers "which sources cost the most
    Sonnet budget?" — the spend side.
  * This module answers "for sources where Sonnet *was* spent, does the
    cheap ``ArticleNet`` agree with the verdict?" — the calibration side.

If a source has high promotion + low per-source agreement, the cheap model
is no longer a useful pre-filter for it: every grey-zone item gets
escalated and the LLM either confirms the model was wrong or surprises it.
That is the actionable signal a single global ``pearson=0.62`` cannot give
you.

Read-only. The stat helpers (``pearson`` / ``spearman`` / ``_mean``) are
imported verbatim from ``ml.score_agreement`` so the two reports can never
disagree on what "agreement" means.

CLI::

    python3 -m ml.per_source_agreement     # JSON report; exit 0
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ml.score_agreement import _MIN_AI, _mean, pearson, spearman
from storage.article_store import _LIVE_ONLY_CLAUSE

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "per_source_agreement.json"

# Sources with fewer overlap rows than this report ``n`` only — the Pearson /
# Spearman of <5 points is noise, not signal. The threshold is small because
# this is a sparse, slowly-filling overlap (Sonnet grades a tiny minority of
# rows); demanding 30+ would leave most sources unreported for days.
MIN_SAMPLES_FOR_CORR = 5


def compute_per_source(rows: Sequence[dict]) -> dict:
    """Group ``rows`` by ``source`` and report agreement stats per bucket.

    Each row needs ``ml_score`` (float) and ``ai_score`` (float, >= ``_MIN_AI``
    to count as actually graded by the LLM); ``source`` keys the buckets.
    Buckets with fewer than ``MIN_SAMPLES_FOR_CORR`` rows have their
    correlation fields set to ``None`` (not 0.0 — to clearly distinguish
    "insufficient data" from "uncorrelated"); ``n``/``mean_abs_divergence``
    /``bias_ml_minus_ai`` are still reported because they remain meaningful
    at small n.

    Pure and deterministic — this is the unit-tested surface. Output keys
    are sorted by descending ``n`` so the heaviest buckets read first.
    """
    pairs = [
        r
        for r in rows
        if r.get("ml_score") is not None
        and (r.get("ai_score") or 0) >= _MIN_AI
        and r.get("source")
    ]

    by_source: dict[str, list[dict]] = {}
    for r in pairs:
        by_source.setdefault(str(r["source"]), []).append(r)

    out_sources: list[dict] = []
    for src, group in by_source.items():
        ml = [float(r["ml_score"]) for r in group]
        ai = [float(r["ai_score"]) for r in group]
        diffs = [m - a for m, a in zip(ml, ai)]
        n = len(group)
        sufficient = n >= MIN_SAMPLES_FOR_CORR
        out_sources.append(
            {
                "source": src,
                "n": n,
                "pearson": round(pearson(ml, ai), 4) if sufficient else None,
                "spearman": round(spearman(ml, ai), 4) if sufficient else None,
                "mean_abs_divergence": round(_mean([abs(d) for d in diffs]), 4),
                "bias_ml_minus_ai": round(_mean(diffs), 4),
                "mean_ml": round(_mean(ml), 4),
                "mean_ai": round(_mean(ai), 4),
            }
        )

    out_sources.sort(key=lambda d: (-d["n"], d["source"]))
    return {
        "total_overlap_rows": len(pairs),
        "source_count": len(out_sources),
        "min_samples_for_corr": MIN_SAMPLES_FOR_CORR,
        "sources": out_sources,
    }


def _db_path() -> Path:
    """Resolve the live articles.db the same way storage.article_store does."""
    from storage import article_store

    return Path(article_store._get_db_path())


def load_rows(db_path: Path, limit: int = 20000) -> list[dict]:
    """Load the most recent overlap rows (read-only, live-only) for analysis.

    Uses ``mode=ro`` + a 60s ``busy_timeout`` so a concurrent writer storm
    (memory: ``di-insert-batch-lock-contention``) does not crash the audit.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
    try:
        conn.execute("PRAGMA busy_timeout=60000")
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            f"""
            SELECT ml_score, ai_score, source, first_seen
              FROM articles
             WHERE ml_score IS NOT NULL AND ai_score >= ?
               AND {_LIVE_ONLY_CLAUSE}
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
    report = compute_per_source(rows)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    if write:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUTPUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(OUTPUT_PATH)
    return report


if __name__ == "__main__":
    r = run()
    top = r["sources"][:5]
    print(
        f"[per_source_agreement] sources={r['source_count']} "
        f"overlap={r['total_overlap_rows']}"
    )
    for s in top:
        pr = s["pearson"] if s["pearson"] is not None else "n/a"
        print(
            f"  {s['source']:<28} n={s['n']:<5} pearson={pr} "
            f"mad={s['mean_abs_divergence']} bias={s['bias_ml_minus_ai']}"
        )
