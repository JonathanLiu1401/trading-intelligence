"""Per-collector audit of direct-write urgent rows (kw_score-only flags).

Most collectors write articles into the store and let the scoring pipeline
(``urgency_scorer`` Sonnet path → ``score_pending`` ML path) decide urgency.
A small set of "structured signal" collectors instead direct-write
``urgency = 1`` from their own ``kw_score`` formula at insert time
(``commodity_futures``, ``dxy``, ``vix_term_structure``, ``sector_etf``,
``finra_short_volume``, ``forex_factory_calendar``, ``macro_calendar``,
``nasdaq_halts``, ``sec_13f``, ``federal_register``, ``twse_semiconductor``).
Those rows are urgent from the moment they land, well before the LLM or ML
scorer has had a chance to corroborate or contradict the call.

That direct-write path is fast and useful for true structured signals
(FOMC release, NASDAQ trading halt, USD regime flip) — but it is ALSO a
silent noise source the moment a threshold gets miscalibrated. The
2026-05-23 commodity_futures bug (kw_score cutoff 6.0 vs system-wide
URGENT_THRESHOLD 8.0) sat for weeks before the analyst noticed routine
2% oil moves firing BREAKING pushes. This module would have surfaced it
immediately: "commodity_futures fired 5 direct-write urgent rows, 0
LLM-corroborated, 0 ML-corroborated."

For each source in the window, reports:

  * ``direct_urgent``  — urgency >= 1 with ai_score=0 AND ml_score IS NULL
                          (i.e., the scoring pipeline never ran or never
                          confirmed the call before the row was alerted)
  * ``llm_urgent``     — urgency >= 1 with ai_score >= URGENT_THRESHOLD
                          (Sonnet ground truth corroborates)
  * ``ml_urgent``      — urgency >= 1 with ml_score >= URGENT_THRESHOLD
                          (the local PyTorch model corroborates)
  * ``corroborated``   — direct_urgent rows that were *later* corroborated
                          by LLM OR ML reaching URGENT_THRESHOLD
  * ``uncorroborated`` — direct_urgent rows that never got either signal
                          (the failure mode the analyst sees as noise)
  * ``uncorroborated_fraction`` — uncorroborated / direct_urgent
                          (HIGH = candidate for threshold re-tuning)

Read-only. Synthetic backtest/opus-annotation rows excluded via
``_LIVE_ONLY_CLAUSE`` (defense-in-depth — those rows are inserted with
urgency=0 today but the read-side discipline matches every other live
reader). NO DB write — no ``ai_score`` / ``ml_score`` / ``score_source``
/ ``urgency`` mutation. All four load-bearing invariants intact by
construction.

CLI::

    python3 -m analytics.collector_direct_urgent_audit          # 24h
    python3 -m analytics.collector_direct_urgent_audit --hours 168
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE
from watchers.urgency_scorer import URGENT_THRESHOLD

BASE_DIR = Path(__file__).resolve().parents[1]
DB = BASE_DIR / "data" / "articles.db"
OUT = Path("/home/zeph/logs/collector_direct_urgent_audit.json")

# Two-row floor: a single "direct_urgent=1, uncorroborated=1" entry reads as
# "100% uncorroborated" and inflates the verdict list without signal — same
# tail-trim discipline as alert_source_breakdown / daily_digest.
MIN_PER_SOURCE = 2
WINDOW_HOURS = 24
# Sources with this many or more uncorroborated direct-writes in the window
# (AND uncorroborated_fraction >= 1.0 — i.e. EVERY direct-urgent row went
# unconfirmed) are surfaced as "suspect". The commodity_futures bug landed
# 5/5 uncorroborated in 30 days; setting the floor at 3 gives an analyst-
# actionable signal without firing on a single bad cycle.
SUSPECT_MIN_UNCORROBORATED = 3


def classify_row(
    ai_score: float | None, ml_score: float | None,
    threshold: float = URGENT_THRESHOLD,
) -> dict:
    """Pure: from a single ``(ai_score, ml_score)`` pair on a urgency>=1
    row, classify whether ANY pipeline corroborated the urgent call.

    Returns a dict of bool flags:
      * ``kw_only``        — neither LLM nor ML has scored this row
                              (ai_score==0 AND ml_score IS NULL); the
                              urgency flag came from the collector
                              direct-write at insert time
      * ``llm_urgent``     — Sonnet ground truth at or above the threshold
      * ``ml_urgent``      — model prediction at or above the threshold
      * ``uncorroborated`` — urgency>=1 row that NEITHER pipeline endorsed
                              (covers both kw_only AND the post-fire-low-
                              ml-score pattern: collector wrote urgency=1,
                              alert worker fired, scorer later assigned
                              ml_score < threshold — exactly the
                              commodity_futures noise pattern from the
                              2026-05-23 audit)

    Score-source convention (see ``article_store._migrate``): any non-zero
    ai_score is an LLM (Sonnet/Opus) label; a non-NULL ml_score is an ML
    prediction. ``uncorroborated`` is the analyst-meaningful flag —
    ``kw_only`` is the diagnostic refinement (was the row never scored, or
    was it scored and rejected?).
    """
    ai = float(ai_score or 0.0)
    ml = ml_score
    has_llm = ai > 0.0
    has_ml = ml is not None
    llm_urgent = ai >= threshold
    ml_urgent = has_ml and float(ml) >= threshold

    kw_only = not has_llm and not has_ml
    uncorroborated = not (llm_urgent or ml_urgent)

    return {
        "kw_only": kw_only,
        "llm_urgent": llm_urgent,
        "ml_urgent": ml_urgent,
        "uncorroborated": uncorroborated,
    }


def compute_audit(
    rows,
    min_per_source: int = MIN_PER_SOURCE,
) -> list[dict]:
    """Pure: from an iterable of ``(source, ai_score, ml_score)`` tuples
    (one per urgency>=1 row in the window), return per-source counts.

    Each output dict has:
      * ``source``                  — verbatim source tag
      * ``urgent_total``            — total urgency>=1 rows
      * ``kw_only``                 — never LLM/ML scored (direct-write only)
      * ``llm_urgent``              — LLM-confirmed urgent
      * ``ml_urgent``               — ML-confirmed urgent
      * ``uncorroborated``          — neither LLM nor ML endorsed (covers
                                       kw_only AND ml_below_threshold rows)
      * ``uncorroborated_fraction`` — uncorroborated / urgent_total
                                       (HIGH = noise risk)

    Sorted by ``uncorroborated`` desc then source asc for stable ties so the
    "loudest collector with no LLM/ML corroboration" sits at the top.
    """
    agg: dict[str, dict[str, int]] = {}
    for source, ai_score, ml_score in rows:
        src = (source or "").strip() or "unknown"
        flags = classify_row(ai_score, ml_score)
        slot = agg.setdefault(src, {
            "urgent_total": 0,
            "kw_only": 0,
            "llm_urgent": 0,
            "ml_urgent": 0,
            "uncorroborated": 0,
        })
        slot["urgent_total"] += 1
        for k in ("kw_only", "llm_urgent", "ml_urgent", "uncorroborated"):
            if flags[k]:
                slot[k] += 1

    out: list[dict] = []
    for src, counts in agg.items():
        if counts["urgent_total"] < min_per_source:
            continue
        unc = counts["uncorroborated"]
        total = counts["urgent_total"]
        frac = round(unc / total, 4) if total else 0.0
        out.append({
            "source": src,
            **counts,
            "uncorroborated_fraction": frac,
        })
    out.sort(key=lambda r: (-r["uncorroborated"], r["source"]))
    return out


def find_suspects(
    audit: list[dict],
    min_uncorroborated: int = SUSPECT_MIN_UNCORROBORATED,
    min_fraction: float = 0.8,
) -> list[dict]:
    """Filter the per-source audit to the "suspect" list — sources where
    the urgent rows mostly went un-corroborated by either Sonnet or the
    local model. ``uncorroborated_fraction >= min_fraction`` (default 0.8)
    means 80%+ of this collector's urgent rows had NO pipeline endorsement;
    paired with the ``min_uncorroborated`` count floor this filters one-off
    blips and surfaces the structural noise sources the analyst should
    re-tune. The 0.8 default deliberately admits a small LLM corroboration
    rate (one out of five would still flag) without slamming the door on
    collectors that happen to score one true positive."""
    return [
        r for r in audit
        if r["uncorroborated"] >= min_uncorroborated
        and r["uncorroborated_fraction"] >= min_fraction
    ]


def load_urgent_rows(
    db_path: Path = DB, hours: int = WINDOW_HOURS,
) -> list[tuple]:
    """Read urgency>=1 rows in the window. ``mode=ro`` so a concurrent
    writer storm cannot crash this audit; ``_LIVE_ONLY_CLAUSE`` so synthetic
    training rows are excluded — same discipline as every other live-side
    reader. Returns the minimal ``(source, ai_score, ml_score)`` projection."""
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        con.execute("PRAGMA busy_timeout=4000")
        return con.execute(
            "SELECT source, ai_score, ml_score FROM articles "
            f"WHERE urgency >= 1 AND first_seen >= ? AND {_LIVE_ONLY_CLAUSE}",
            (cutoff,),
        ).fetchall()
    finally:
        con.close()


def build_report(audit: list[dict], hours: int) -> dict:
    """Aggregate the per-source rows into the JSON report payload, including
    the suspect filter so the analyst's top-of-pipe answer is one number:
    "how many collectors are firing un-corroborated direct-write alerts?"."""
    suspects = find_suspects(audit)
    total_kw_only = sum(r["kw_only"] for r in audit)
    total_uncorr = sum(r["uncorroborated"] for r in audit)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": hours,
        "min_per_source": MIN_PER_SOURCE,
        "suspect_min_uncorroborated": SUSPECT_MIN_UNCORROBORATED,
        "total_kw_only": total_kw_only,
        "total_uncorroborated": total_uncorr,
        "n_sources": len(audit),
        "n_suspect_sources": len(suspects),
        "suspects": suspects,
        "by_source": audit,
    }


def run(db_path: Path = DB, hours: int = WINDOW_HOURS, write: bool = True) -> dict:
    """End-to-end: read → aggregate → report (and optionally persist)."""
    rows = load_urgent_rows(db_path, hours=hours)
    audit = compute_audit(rows)
    report = build_report(audit, hours=hours)
    if write:
        try:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            tmp = OUT.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(report, indent=2))
            tmp.replace(OUT)
        except OSError:
            # Best-effort persistence; the report is still returned to the
            # caller even if the dashboard scratch path is unwritable.
            pass
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hours", type=int, default=WINDOW_HOURS)
    p.add_argument("--db", type=Path, default=DB)
    p.add_argument("--no-write", action="store_true",
                   help="skip writing to OUT (CLI inspection only)")
    args = p.parse_args(argv)
    report = run(db_path=args.db, hours=args.hours, write=not args.no_write)
    print(
        f"window={report['window_hours']}h  "
        f"kw_only={report['total_kw_only']}  "
        f"uncorroborated={report['total_uncorroborated']}  "
        f"sources={report['n_sources']}  "
        f"suspect_sources={report['n_suspect_sources']}"
    )
    if report["suspects"]:
        print("\n=== SUSPECT COLLECTORS (≥80% urgent rows with no LLM/ML corroboration) ===")
        for r in report["suspects"]:
            print(
                f"  {r['source'][:32]:<32}  "
                f"total={r['urgent_total']:>3}  "
                f"uncorr={r['uncorroborated']:>3}  "
                f"unc_frac={r['uncorroborated_fraction']}"
            )
    print("\n=== TOP 10 BY UNCORROBORATED ===")
    for r in report["by_source"][:10]:
        print(
            f"  {r['source'][:32]:<32}  "
            f"total={r['urgent_total']:>3}  kw_only={r['kw_only']:>3}  "
            f"llm={r['llm_urgent']:>3}  ml={r['ml_urgent']:>3}  "
            f"uncorr={r['uncorroborated']:>3}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
