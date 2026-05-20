"""Per-source alerted-row breakdown with LLM-vs-ML calibration.

The analyst persona's recurring question: "which collectors fired BREAKING
alerts in the last N hours, and is each alert backed by an LLM ground-truth
label, or only by the local model's hunch?" Today the answer requires
hand-rolling SQL across multiple columns; this module is the queryable
primitive.

Companion to:
  * ``ArticleStore.urgency_label_split`` — aggregates the SAME calibration
    breakdown (``llm`` / ``ml`` / ``briefing_boost`` / ``null``) but across
    ALL urgent rows, with no per-source axis. Keys/semantics are kept
    byte-identical so the two reads can never drift on what "vetted" means.
  * ``analytics/daily_digest.py`` — top-N urgent rows by score; gives the
    headlines but not which collectors fed them.

Read-only. Synthetic backtest/opus-annotation rows are excluded via
``_LIVE_ONLY_CLAUSE`` — synthetic rows are inserted ``urgency=0`` today so
this is defense-in-depth in the same shape as every other live-side reader
(``get_unalerted_urgent``, ``get_top_for_briefing``, …). No DB write, no
``ai_score`` / ``ml_score`` / ``score_source`` / ``urgency`` mutation —
all four load-bearing invariants intact by construction.

CLI::

    python3 -m analytics.alert_source_breakdown        # 24h window, table
    python3 -m analytics.alert_source_breakdown --hours 6
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

BASE_DIR = Path(__file__).resolve().parents[1]
DB = BASE_DIR / "data" / "articles.db"
OUT = Path("/home/zeph/logs/alert_source_breakdown.json")

# Floor a source must clear to be reported. Two single-alert sources at the
# same hour would otherwise show up as "100% llm-vetted" / "0% llm-vetted"
# on a sample size of one — pure noise. Mirrors ``daily_digest``'s tail-
# trimming discipline.
MIN_PER_SOURCE = 1
WINDOW_HOURS = 24


def compute_breakdown(
    rows,
    min_per_source: int = MIN_PER_SOURCE,
) -> list[dict]:
    """Pure: from an iterable of ``(source, score_source)`` tuples (one per
    alerted row in the window), return a per-source breakdown.

    Each dict has:
      * ``source``         — the input ``source`` string, preserved verbatim
      * ``alerted``        — count of urgency=2 rows under this source
      * ``by_source``      — ``{"llm": N, "ml": N, "briefing_boost": N,
                              "null": N}`` — the calibration breakdown,
                              SAME keys as ``urgency_label_split.by_source``
                              (SSOT for "what a vetted label is")
      * ``llm_fraction``   — ``(llm + briefing_boost) / alerted``, 0.0 when
                              ``alerted == 0`` (mirrors urgency_label_split)

    Sources with fewer than ``min_per_source`` alerted rows are dropped.
    Output is sorted by ``alerted`` desc, then by source asc for stable ties.
    ``null`` covers legacy pre-migration rows still without an explicit
    score_source tag — same convention as urgency_label_split.
    """
    by_source: dict[str, dict[str, int]] = {}
    counts: dict[str, int] = {}
    for source, score_source in rows:
        src = (source or "").strip() or "unknown"
        slot = by_source.setdefault(
            src, {"llm": 0, "ml": 0, "briefing_boost": 0, "null": 0}
        )
        key = score_source if score_source in (
            "llm", "ml", "briefing_boost"
        ) else "null"
        slot[key] += 1
        counts[src] = counts.get(src, 0) + 1

    out: list[dict] = []
    for src, total in counts.items():
        if total < min_per_source:
            continue
        breakdown = by_source[src]
        vetted = breakdown["llm"] + breakdown["briefing_boost"]
        llm_fraction = round(vetted / total, 4) if total else 0.0
        out.append({
            "source": src,
            "alerted": total,
            "by_source": breakdown,
            "llm_fraction": llm_fraction,
        })

    out.sort(key=lambda r: (-r["alerted"], r["source"]))
    return out


def load_alerted_rows(
    db_path: Path = DB, hours: int = WINDOW_HOURS
) -> list[tuple]:
    """Read all alerted (urgency=2) rows in the window. ``mode=ro`` so a
    concurrent writer storm cannot crash this audit; ``_LIVE_ONLY_CLAUSE``
    so synthetic training rows are excluded. Returns the minimal
    ``(source, score_source)`` projection — every other column is
    irrelevant to the per-source breakdown."""
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        con.execute("PRAGMA busy_timeout=4000")
        return con.execute(
            "SELECT source, score_source FROM articles "
            f"WHERE urgency = 2 AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (cutoff,),
        ).fetchall()
    finally:
        con.close()


def build_report(breakdown: list[dict], hours: int) -> dict:
    """Pure: aggregate the per-source rows into the JSON report payload.

    ``total_alerted`` is the sum across all reported sources — same number
    ``urgency_label_split(hours).total`` would report (modulo the
    ``min_per_source`` floor; surfaced explicitly so the cap is auditable).
    ``aggregate_llm_fraction`` re-derives the calibration fraction across
    the whole window from the per-source breakdowns, again with the same
    formula urgency_label_split uses — drift-free by construction."""
    total = sum(r["alerted"] for r in breakdown)
    vetted = sum(
        r["by_source"]["llm"] + r["by_source"]["briefing_boost"]
        for r in breakdown
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": hours,
        "min_per_source": MIN_PER_SOURCE,
        "total_alerted": total,
        "aggregate_llm_fraction": round(vetted / total, 4) if total else 0.0,
        "sources": breakdown,
    }


def run(db_path: Path = DB, hours: int = WINDOW_HOURS, write: bool = True) -> dict:
    """End-to-end: read → aggregate → report (and optionally persist)."""
    rows = load_alerted_rows(db_path, hours=hours)
    breakdown = compute_breakdown(rows)
    report = build_report(breakdown, hours=hours)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(OUT)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hours", type=int, default=WINDOW_HOURS)
    p.add_argument("--db", type=Path, default=DB)
    p.add_argument("--no-write", action="store_true",
                   help="skip writing to OUT (CLI inspection only)")
    args = p.parse_args(argv)
    report = run(db_path=args.db, hours=args.hours, write=not args.no_write)
    total = report["total_alerted"]
    frac = report["aggregate_llm_fraction"]
    print(
        f"window={report['window_hours']}h  total_alerted={total}  "
        f"aggregate_llm_fraction={frac}  sources={len(report['sources'])}"
    )
    for r in report["sources"][:10]:
        bs = r["by_source"]
        print(
            f"  {r['source'][:32]:<32} alerted={r['alerted']:>4}  "
            f"llm={bs['llm']:>3} ml={bs['ml']:>3} "
            f"boost={bs['briefing_boost']:>3} null={bs['null']:>3}  "
            f"llm_frac={r['llm_fraction']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
