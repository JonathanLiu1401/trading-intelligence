"""Per-source LLM-promotion + alert-yield audit (read-only).

digital-intern's two-tier scoring spends Claude budget unevenly across sources:
``ArticleNet`` scores every live article cheaply, then the *grey-zone* gate in
``ml/inference.py`` routes uncertain rows to Sonnet for an authoritative
``ai_score`` (those rows land with ``score_source='llm'``). Sonnet calls are the
single dominant cost in steady-state; how that spend distributes across the
~17 collectors decides whether the budget is buying alerts or burning on noise.

Two existing diagnostics are deliberately not this:

  * ``ml/score_agreement.py`` — agreement of ``ml_score`` vs ``ai_score`` on
    their *overlap* (Pearson/Spearman/bias). It tells you whether the cheap
    model tracks the expensive judge on items the LLM already graded; it does
    NOT tell you which sources *trigger* that LLM spend.
  * ``ml/label_audit.py`` — training-pool hygiene (column-separation +
    pre-migration heuristic trust gap). Read-side, not spend-side.

This module reports, per source over a recency window:

  * ``total`` — live articles from that source.
  * ``promoted`` — rows with ``score_source='llm'`` (Claude actually graded).
  * ``promotion_rate_pct`` — promoted / total.
  * ``mean_ai_on_promoted`` — average ``ai_score`` Claude gave that source's
    promoted rows. High = LLM consistently judges this source's items
    important. Low = LLM was asked but downgraded — wasted spend signal.
  * ``alert_yield_pct`` — fraction of promoted rows that reached
    ``urgency >= 1`` (Sonnet flagged urgent — the gate that fires Bloomberg
    alerts). The conversion from spend to actual analyst-facing alerts.

Backtest/opus_annotation rows are excluded via the canonical live-only filter
(see ``storage/article_store.py::_LIVE_ONLY_CLAUSE``) — same predicate is
inlined here so an SSOT change there will still need a paired update, matching
the discipline ``paper_trader/signals.py`` uses.

``compute_promotion_stats`` is the pure, unit-tested contract. The DB plumbing
around it is a thin read-only shell.

Run standalone::

    python3 -m ml.llm_promotion_audit              # 24h window, JSON to data/
    python3 -m ml.llm_promotion_audit --hours 6    # last 6h
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "llm_promotion_audit.json"

# Mirrors storage/article_store.py::_LIVE_ONLY_CLAUSE. Kept inline (not
# imported) for the same reason paper-trader inlines it: this module loads a
# read-only snapshot, and a stray import-time side effect from article_store
# (logger init, USB path probe) is not worth the SSOT linkage. A divergence
# here is caught by ``tests/test_llm_promotion_audit.py::test_load_rows_excludes_backtest``.
_LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

# Sources with fewer than this many live rows in the window are aggregated into
# an "_other" bucket — a 3-item source's 100% promotion rate is noise.
_MIN_SOURCE_ROWS = 5


def compute_promotion_stats(rows: Sequence[dict]) -> dict:
    """Summarize LLM-promotion + alert-yield over ``rows``.

    Each row must carry ``source`` (str), ``score_source`` (str|None — 'llm'
    means Claude graded it), ``ai_score`` (float, 0 when never graded), and
    ``urgency`` (int — 0 normal, 1 needs alert, 2 alert sent).

    Pure and deterministic. Live-only filtering is the caller's responsibility
    (do it at the SQL boundary in ``load_rows``), so this function does NOT
    re-filter — its only job is the rollup math.
    """
    total = len(rows)
    if total == 0:
        return {
            "n": 0,
            "n_promoted": 0,
            "promotion_rate_pct": 0.0,
            "n_alerted": 0,
            "overall_alert_yield_pct": 0.0,
            "by_source": [],
        }

    promoted_rows = [r for r in rows if (r.get("score_source") or "") == "llm"]
    alerted_rows = [r for r in rows if int(r.get("urgency") or 0) >= 1]
    n_promoted = len(promoted_rows)
    n_alerted = len(alerted_rows)

    # Per-source rollup. We classify a row as promoted iff score_source='llm'
    # and as alerted iff urgency>=1 (covers both the "needs alert" and
    # "alert sent" tri-states — alert_worker advances 1→2 asynchronously, so
    # using >=1 is the spend→outcome question, not the "did webhook fire").
    bucket: dict[str, dict] = {}
    for r in rows:
        src = (r.get("source") or "").strip() or "_unknown"
        b = bucket.setdefault(
            src,
            {
                "source": src,
                "total": 0,
                "promoted": 0,
                "alerted_promoted": 0,
                "_ai_sum_on_promoted": 0.0,
            },
        )
        b["total"] += 1
        if (r.get("score_source") or "") == "llm":
            b["promoted"] += 1
            b["_ai_sum_on_promoted"] += float(r.get("ai_score") or 0.0)
            if int(r.get("urgency") or 0) >= 1:
                b["alerted_promoted"] += 1

    by_source: list[dict] = []
    other = {"source": "_other", "total": 0, "promoted": 0, "alerted_promoted": 0,
             "_ai_sum_on_promoted": 0.0}
    for src, b in bucket.items():
        target = b if b["total"] >= _MIN_SOURCE_ROWS else other
        if target is other:
            other["total"] += b["total"]
            other["promoted"] += b["promoted"]
            other["alerted_promoted"] += b["alerted_promoted"]
            other["_ai_sum_on_promoted"] += b["_ai_sum_on_promoted"]
            continue
        rate = 100.0 * b["promoted"] / b["total"] if b["total"] else 0.0
        mean_ai = (
            b["_ai_sum_on_promoted"] / b["promoted"] if b["promoted"] else 0.0
        )
        yld = (
            100.0 * b["alerted_promoted"] / b["promoted"] if b["promoted"] else 0.0
        )
        by_source.append(
            {
                "source": b["source"],
                "total": b["total"],
                "promoted": b["promoted"],
                "promotion_rate_pct": round(rate, 2),
                "mean_ai_on_promoted": round(mean_ai, 3),
                "alert_yield_pct": round(yld, 2),
            }
        )

    if other["total"] > 0:
        rate = 100.0 * other["promoted"] / other["total"]
        mean_ai = (
            other["_ai_sum_on_promoted"] / other["promoted"]
            if other["promoted"] else 0.0
        )
        yld = (
            100.0 * other["alerted_promoted"] / other["promoted"]
            if other["promoted"] else 0.0
        )
        by_source.append(
            {
                "source": "_other",
                "total": other["total"],
                "promoted": other["promoted"],
                "promotion_rate_pct": round(rate, 2),
                "mean_ai_on_promoted": round(mean_ai, 3),
                "alert_yield_pct": round(yld, 2),
            }
        )

    by_source.sort(key=lambda d: (-d["promoted"], -d["total"], d["source"]))

    return {
        "n": total,
        "n_promoted": n_promoted,
        "promotion_rate_pct": round(100.0 * n_promoted / total, 2),
        "n_alerted": n_alerted,
        "overall_alert_yield_pct": (
            round(100.0 * sum(b["alerted_promoted"] for b in bucket.values())
                  / n_promoted, 2)
            if n_promoted else 0.0
        ),
        "by_source": by_source,
    }


def _db_path() -> Path:
    """Resolve the live articles.db the same way storage.article_store does."""
    from storage import article_store
    return Path(article_store._get_db_path())


def load_rows(db_path: Path, hours: int = 24) -> list[dict]:
    """Read-only snapshot of live rows in the recency window."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            f"""
            SELECT source, score_source, ai_score, urgency, first_seen
              FROM articles
             WHERE first_seen > datetime('now', ?)
               AND {_LIVE_ONLY_CLAUSE}
            """,
            (f"-{int(hours)} hours",),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def run(write: bool = True, hours: int = 24) -> dict:
    """Compute the report against the live DB and (optionally) persist it."""
    rows = load_rows(_db_path(), hours=hours)
    report = compute_promotion_stats(rows)
    report["window_hours"] = hours
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    if write:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUTPUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(OUTPUT_PATH)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hours", type=int, default=24,
                    help="recency window (default 24)")
    ap.add_argument("--no-write", action="store_true",
                    help="don't persist data/llm_promotion_audit.json")
    args = ap.parse_args()
    r = run(write=not args.no_write, hours=args.hours)
    print(
        f"[llm_promotion_audit] window={r['window_hours']}h n={r['n']} "
        f"promoted={r['n_promoted']} ({r['promotion_rate_pct']}%) "
        f"alerted={r['n_alerted']} "
        f"overall_yield={r['overall_alert_yield_pct']}%"
    )
    for b in r["by_source"][:10]:
        print(
            f"  {b['source']:<28} total={b['total']:>4} "
            f"promoted={b['promoted']:>3} "
            f"rate={b['promotion_rate_pct']:>5}% "
            f"mean_ai={b['mean_ai_on_promoted']:>5} "
            f"alert_yield={b['alert_yield_pct']:>5}%"
        )
