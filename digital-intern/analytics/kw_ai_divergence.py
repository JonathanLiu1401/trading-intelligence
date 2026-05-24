"""KW vs AI score divergence detector.

Two divergence regimes are tracked over a bounded recent window:

* **false_positive** (high kw / low ai): keyword filter fired strongly but
  the AI scorer found little signal. These rows inflated the triage queue
  without delivering intelligence value. High counts from a given source
  indicate that source's vocabulary trips keywords without substance.

* **hidden_gem** (high ai / low kw): AI found material signal that the
  keyword filter almost missed (kw_score < KW_LOW). Surfacing these tells
  the analyst which topics the keyword dictionary under-weights.

Per-source breakdowns for both regimes are included so the operator can
see which collectors are the biggest keyword-noise emitters and which
produce unrecognised high-signal content.

Design constraints (identical to all other analytics):
  * Bounded ``SCAN_LIMIT`` idx_first_seen read — never full-table scan the
    1.4 GB USB-backed DB.
  * Read-only sqlite URI — never contends with daemon writers.
  * USB-safe ``busy_timeout``.
  * ``_LIVE_ONLY_CLAUSE`` discipline — synthetic backtest/opus rows excluded.

Artifacts:
  * /home/zeph/logs/kw_ai_divergence.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# Resolve the same DB the daemon uses (USB-preferred via
# storage.article_store._get_db_path), with a fallback to the local
# data/articles.db symlink that has historically been the manual path. The
# urgency_drought sibling already uses this helper — without it, this script
# would silently target an empty local file on a host where the data/ symlink
# is absent (e.g. fresh checkout, CI sandbox) and emit a meaningless snapshot.
try:
    from storage.article_store import _get_db_path as _resolve_db_path
except Exception:
    _resolve_db_path = None

DB_PATH = (
    Path(_resolve_db_path()) if _resolve_db_path is not None
    else Path(__file__).resolve().parent.parent / "data" / "articles.db"
)
LOG_DIR = Path("/home/zeph/logs")
OUT_PATH = LOG_DIR / "kw_ai_divergence.json"

SCAN_LIMIT = 6000       # bounded idx_first_seen scan
# Both kw_score and ai_score live on the SAME 0..10 scale
# (triage/heuristic_scorer.py docstring: "Range: 0.0 – 10.0";
# articles.db.ai_score is set to Sonnet's 0..10 integer by urgency_scorer +
# bulk-updated by the trainer with the same magnitude). The previous
# AI_LOW=0.15 / AI_HIGH=0.50 were leftover from a 0..1 normalisation that
# never landed: AI_HIGH=0.5 then matched ai_score=1.0 (the bottom of
# Sonnet's "engaged at all" output) as a "hidden gem", so the hidden-gem
# list became "anything Sonnet rated >=1" — pure noise. Re-scale to the
# real ai_score range so the analyser does what the docstring says:
#   * false_positive: kw fired strongly (>=5) AND Sonnet either never
#     engaged (ai_score == 0 — unscored) OR engaged and floored to noise
#     (urgency_scorer's 0.01 anti-loop floor). AI_LOW=1.5 captures both
#     cases without sweeping in genuine "Sonnet said 2/10 relevant" rows.
#   * hidden_gem: Sonnet rated relevant (>=6 — its mid-relevance floor)
#     AND keyword barely fired (kw_score < 3.0). 6.0 is the same
#     "relevant" threshold the urgency prompt's RELEVANT band starts at.
KW_HIGH = 5.0           # kw_score threshold for "keyword fired strongly"
AI_LOW = 1.5            # ai_score ceiling for "AI found little signal"
AI_HIGH = 6.0           # ai_score floor for "AI found strong signal"
KW_LOW = 3.0            # kw_score ceiling for "keyword almost missed it"

_LIVE_ONLY = (
    "source NOT LIKE 'backtest%' "
    "AND source NOT LIKE 'opus_annotation%' "
    "AND source NOT LIKE 'backtest_run%' "
    "AND url NOT LIKE 'backtest://%'"
)


def compute() -> dict:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    try:
        rows = conn.execute(
            f"""
            SELECT source, kw_score, ai_score, title
            FROM articles
            WHERE {_LIVE_ONLY}
              AND kw_score IS NOT NULL
              AND ai_score IS NOT NULL
            ORDER BY first_seen DESC
            LIMIT ?
            """,
            (SCAN_LIMIT,),
        ).fetchall()
    finally:
        conn.close()

    total = len(rows)
    fp_by_source: dict[str, int] = defaultdict(int)
    hg_by_source: dict[str, int] = defaultdict(int)
    fp_examples: list[dict] = []
    hg_examples: list[dict] = []

    for source, kw, ai, title in rows:
        kw = kw or 0.0
        ai = ai or 0.0
        if kw >= KW_HIGH and ai <= AI_LOW:
            fp_by_source[source] += 1
            if len(fp_examples) < 5:
                fp_examples.append(
                    {"source": source, "kw": round(kw, 2), "ai": round(ai, 3),
                     "title": (title or "")[:80]}
                )
        elif ai >= AI_HIGH and kw < KW_LOW:
            hg_by_source[source] += 1
            if len(hg_examples) < 5:
                hg_examples.append(
                    {"source": source, "kw": round(kw, 2), "ai": round(ai, 3),
                     "title": (title or "")[:80]}
                )

    fp_total = sum(fp_by_source.values())
    hg_total = sum(hg_by_source.values())

    top_fp = sorted(fp_by_source.items(), key=lambda x: x[1], reverse=True)[:10]
    top_hg = sorted(hg_by_source.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_limit": SCAN_LIMIT,
        "scanned": total,
        "thresholds": {
            "false_positive": f"kw>={KW_HIGH} AND ai<={AI_LOW}",
            "hidden_gem": f"ai>={AI_HIGH} AND kw<{KW_LOW}",
        },
        "false_positives": {
            "total": fp_total,
            "rate": round(fp_total / total, 4) if total else 0.0,
            "top_sources": [{"source": s, "count": n} for s, n in top_fp],
            "examples": fp_examples,
        },
        "hidden_gems": {
            "total": hg_total,
            "rate": round(hg_total / total, 4) if total else 0.0,
            "top_sources": [{"source": s, "count": n} for s, n in top_hg],
            "examples": hg_examples,
        },
    }


def write_snapshot(report: dict, path: Path = OUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def main() -> int:
    report = compute()
    out = write_snapshot(report)
    fp = report["false_positives"]
    hg = report["hidden_gems"]
    print(
        f"kw_ai_divergence: scanned={report['scanned']} "
        f"false_positives={fp['total']} ({fp['rate']:.1%}) "
        f"hidden_gems={hg['total']} ({hg['rate']:.1%})"
    )
    if fp["top_sources"]:
        top = fp["top_sources"][0]
        print(f"  noisiest source: {top['source']} ({top['count']} fp)")
    if hg["top_sources"]:
        top = hg["top_sources"][0]
        print(f"  most hidden gems: {top['source']} ({top['count']} hg)")
    if hg["examples"]:
        ex = hg["examples"][0]
        print(f"  gem example: [{ex['source']}] kw={ex['kw']} ai={ex['ai']} \"{ex['title']}\"")
    print(f"  snapshot -> {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
