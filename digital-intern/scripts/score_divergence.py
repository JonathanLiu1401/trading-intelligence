#!/usr/bin/env python3
"""Detect articles where ml_score and ai_score strongly disagree.

Surfaces probable model/scorer divergence: articles the ML model rates very
differently from the AI scorer. Read-only; output JSON to
/home/zeph/logs/score_divergence.json.
"""
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# SSOT for "the LLM actually graded this row": ai_score defaults to 0 (no LLM
# label), and urgency_scorer floors any LLM-touched score at 0.01 so a real
# LLM label is always >= _MIN_AI. Imported (not duplicated) for anti-drift —
# same discipline ml/per_source_agreement.py uses for the same threshold.
from ml.score_agreement import _MIN_AI  # noqa: E402
from storage.article_store import _LIVE_ONLY_CLAUSE  # noqa: E402

DB = REPO / "data" / "articles.db"
OUT = Path("/home/zeph/logs/score_divergence.json")
WINDOW_HOURS = 24
TOP_N = 20
MIN_GAP = 0.30  # absolute |ml_score - ai_score| threshold


def classify_divergent(rows, min_gap: float = MIN_GAP) -> list[dict]:
    """Pure: from a list of (id, title, source, ai_score, ml_score, urgency,
    first_seen) tuples, return divergent rows with gap >= ``min_gap``, sorted
    by largest gap first. ``ai_score``/``ml_score`` are coerced via ``or 0.0``
    so a stray NULL is treated as zero (consistent with the schema default).

    Callers are responsible for pre-filtering rows to those carrying a real
    LLM label (``ai_score >= _MIN_AI``); see ``load_rows`` for the SQL
    contract. Without that filter every model-scored-but-LLM-unscored row
    reports as "ml diverged upward against ai=0", which is not divergence —
    just an unlabelled row.
    """
    out: list[dict] = []
    for r in rows:
        ai = r[3] or 0.0
        ml = r[4] or 0.0
        gap = abs(ai - ml)
        if gap < min_gap:
            continue
        out.append({
            "id": r[0],
            "title": (r[1] or "")[:160],
            "source": r[2],
            "ai_score": round(ai, 3),
            "ml_score": round(ml, 3),
            "gap": round(gap, 3),
            "direction": "ml_higher" if ml > ai else "ai_higher",
            "urgency": r[5],
            "first_seen": r[6],
        })
    out.sort(key=lambda x: x["gap"], reverse=True)
    return out


def load_rows(db_path: Path = DB, window_hours: int = WINDOW_HOURS) -> list[tuple]:
    """Read the overlap (both LLM-labelled AND model-scored, live-only)
    within the recent window. Uses ``mode=ro`` so a concurrent writer storm
    cannot crash this audit; ``_LIVE_ONLY_CLAUSE`` is applied so synthetic
    backtest training rows (which carry legitimate fractional ai_score
    labels by design) do not appear as "divergent" against the live model."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        con.execute("PRAGMA busy_timeout=4000")
        return con.execute(
            "SELECT id, title, source, ai_score, ml_score, urgency, first_seen "
            "FROM articles "
            f"WHERE ai_score >= ? AND ml_score IS NOT NULL "
            f"AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
            "ORDER BY first_seen DESC LIMIT 5000",
            (_MIN_AI, cutoff),
        ).fetchall()
    finally:
        con.close()


def build_summary(divergent: list[dict], sampled: int) -> dict:
    """Pure: aggregate the classified rows into the report payload."""
    top = divergent[:TOP_N]
    n = len(divergent)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": WINDOW_HOURS,
        "sampled": sampled,
        "divergent_count": n,
        "min_gap_threshold": MIN_GAP,
        "avg_gap": round(sum(d["gap"] for d in divergent) / n, 3) if n else 0.0,
        "ml_higher_pct": round(
            100.0 * sum(1 for d in divergent if d["direction"] == "ml_higher") / n, 1
        ) if n else 0.0,
        "top": top,
    }


def main():
    rows = load_rows()
    divergent = classify_divergent(rows)
    summary = build_summary(divergent, sampled=len(rows))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"sampled={summary['sampled']} divergent={summary['divergent_count']} "
          f"avg_gap={summary['avg_gap']} ml_higher_pct={summary['ml_higher_pct']}%")
    for d in summary["top"][:5]:
        print(f"  gap={d['gap']:.2f} {d['direction']:10s} "
              f"ai={d['ai_score']:.2f} ml={d['ml_score']:.2f} | "
              f"{d['source']} | {d['title'][:80]}")


if __name__ == "__main__":
    main()
