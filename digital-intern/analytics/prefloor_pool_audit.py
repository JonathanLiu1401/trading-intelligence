"""Pre-floor accumulation audit (read-only).

The ``urgency_scorer.score_batch`` pre-filter floors quote-widget and
recap-template pseudo-articles to ``ai_score=0.01`` with
``score_source='llm'`` BEFORE calling Sonnet — saves quota and stops the
pre-filter rows from re-entering ``get_unscored`` forever. ``score_batch``
also floors any Sonnet-omitted index to ``0.01`` via the anti-loop rule
(``urgency_scorer.py`` ``Floor non-urgent ai_score at 0.01``). All such
rows end up in the trainer's strong-label pool because
``STRONG_LABEL_WHERE`` accepts ``ai_score > 0`` and ``0.01 > 0``.

That is by design — the trainer needs explicit noise labels to learn the
opposite of "high relevance". But there is no continuous view of the
**accumulation rate**: if a new noise class (e.g. a SEO mill the gates
haven't caught yet) suddenly dominates the inbound feed, the prefloor
share of the LLM-labeled pool will spike to >>50% and the model's
effective ground-truth signal collapses to "this is noise, this is also
noise". The exact same failure mode the recap-template + quote-widget
audits exist to monitor — but those count *fingerprint hits at audit
time*, not the accumulated label-pool contamination the trainer actually
sees.

Live evidence (2026-05-23 30d scan): 15,631 of 22,849 ``score_source='llm'``
rows carry ``ai_score = 0.01`` (68.4%). The pool is 7x more "noise" labels
than "real signal" labels. That is BELOW the 80% warning threshold this
audit emits, but only just — and an analyst should be able to see it on
a dashboard tile, not have to grep `articles.db` themselves.

What this audit reports:

  * ``prefloor_total`` — rows with ``score_source='llm' AND ai_score = 0.01``
  * ``real_llm_total`` — rows with ``score_source='llm' AND ai_score > 0.01``
  * ``prefloor_fraction`` — ``prefloor_total / (prefloor_total + real_llm_total)``
  * ``window_prefloor_*`` — same three figures restricted to the last
    ``window_hours`` (recent contamination rate; helps catch new SEO
    classes the gates haven't caught yet)
  * ``per_source`` — top-N sources by absolute prefloor contribution within
    the window. Surfaces WHO is generating the noise so the analyst can
    decide whether to add a fingerprint, tighten a gate, or just throttle
    the source.
  * ``verdict`` — ``HEALTHY`` (< 70% prefloor share in window) /
    ``ELEVATED`` (70-85%) / ``CONTAMINATED`` (≥ 85%). Threshold tuned to
    the current 68% steady-state baseline so the audit fires on a real
    shift, not the existing level.

Pure read-only audit. ``_LIVE_ONLY_CLAUSE`` is applied so synthetic
backtest / opus annotation rows cannot pollute the figure (they don't
carry ``score_source='llm'`` today, but the discipline is the
defense-in-depth convention every audit in this family carries — see
``ml/label_audit.py``, ``analytics/quote_widget_audit.py``).

CLI::

    python3 -m analytics.prefloor_pool_audit
    python3 -m analytics.prefloor_pool_audit --hours 6
    python3 -m analytics.prefloor_pool_audit --top 20
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path


# Verdict thresholds (window-restricted prefloor share). Tuned to the
# current ~68% steady-state baseline so a HEALTHY verdict reflects the
# normal operating regime; ELEVATED catches a 10-percentage-point shift;
# CONTAMINATED catches the trainer-killing scenario where every cycle's
# new LLM labels are virtually all noise.
PREFLOOR_HEALTHY_MAX = 0.70
PREFLOOR_ELEVATED_MAX = 0.85

# ai_score = 0.01 EXACTLY is the prefloor signature. urgency_scorer writes
# 0.01 via three distinct paths (quote-widget pre-floor, recap-template
# pre-floor, anti-loop floor for Sonnet-omitted indices) — see
# watchers/urgency_scorer.py. No live LLM-graded relevance is ever exactly
# 0.01: Sonnet returns integer 0-10 scores, the recursive labeler emits
# ``int * 2.0``, briefing_boost writes 4.5. So an exact equality test on
# 0.01 with score_source='llm' is the canonical "prefloored noise" filter.
_PREFLOOR_WHERE = (
    "score_source = 'llm' AND ai_score = 0.01 "
    f"AND {_LIVE_ONLY_CLAUSE}"
)
_REAL_LLM_WHERE = (
    "score_source = 'llm' AND ai_score > 0.01 "
    f"AND {_LIVE_ONLY_CLAUSE}"
)


def _verdict(window_share: float) -> str:
    """Map a 0..1 prefloor share within the window to a coarse verdict."""
    if window_share < PREFLOOR_HEALTHY_MAX:
        return "HEALTHY"
    if window_share < PREFLOOR_ELEVATED_MAX:
        return "ELEVATED"
    return "CONTAMINATED"


def _safe_share(num: int, denom: int) -> float:
    """Fraction with a divide-by-zero guard: returns 0.0 on an empty pool
    (a brand-new install) rather than raising. The verdict on 0.0 share is
    HEALTHY by construction (0.0 < HEALTHY_MAX), which is the right call —
    no data is not the same as contaminated data."""
    if denom <= 0:
        return 0.0
    return float(num) / float(denom)


def audit(store, window_hours: int = 24, top_sources: int = 10) -> dict:
    """Compute the prefloor accumulation audit against ``store.conn``.

    ``store.conn`` is the live shared connection (``check_same_thread=False``).
    All queries here are COUNT(*) / GROUP BY aggregates against indexed
    columns, so they are safe under writer contention (the standard
    ``_retry_on_lock`` discipline still applies if the caller wraps).

    Returns the audit dict described in the module docstring."""
    conn = store.conn

    prefloor_total = int(
        conn.execute(f"SELECT COUNT(*) FROM articles WHERE {_PREFLOOR_WHERE}").fetchone()[0]
    )
    real_llm_total = int(
        conn.execute(f"SELECT COUNT(*) FROM articles WHERE {_REAL_LLM_WHERE}").fetchone()[0]
    )

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
    window_prefloor = int(
        conn.execute(
            f"SELECT COUNT(*) FROM articles WHERE {_PREFLOOR_WHERE} AND first_seen >= ?",
            (cutoff,),
        ).fetchone()[0]
    )
    window_real_llm = int(
        conn.execute(
            f"SELECT COUNT(*) FROM articles WHERE {_REAL_LLM_WHERE} AND first_seen >= ?",
            (cutoff,),
        ).fetchone()[0]
    )

    window_total = window_prefloor + window_real_llm
    window_share = _safe_share(window_prefloor, window_total)

    # Per-source noise contribution within the window. ORDER BY count DESC
    # so the analyst sees the biggest contributors first.
    per_source_rows = conn.execute(
        f"""
        SELECT source, COUNT(*) AS n
        FROM articles
        WHERE {_PREFLOOR_WHERE} AND first_seen >= ?
        GROUP BY source
        ORDER BY n DESC
        LIMIT ?
        """,
        (cutoff, int(top_sources)),
    ).fetchall()
    per_source = [
        {"source": src or "", "prefloor_count": int(n)}
        for src, n in per_source_rows
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": int(window_hours),
        "prefloor_total": prefloor_total,
        "real_llm_total": real_llm_total,
        "prefloor_share_lifetime": round(
            _safe_share(prefloor_total, prefloor_total + real_llm_total), 4
        ),
        "window_prefloor": window_prefloor,
        "window_real_llm": window_real_llm,
        "window_prefloor_share": round(window_share, 4),
        "verdict": _verdict(window_share),
        "per_source_top": per_source,
    }


def _open_ro_store():
    """Open the live DB read-only for the CLI path. The audit function
    accepts any object with a ``.conn`` attribute, so we wrap a minimal
    shim that exposes ``conn`` — avoids importing ArticleStore (which would
    apply the schema migration and open writer locks)."""
    db = _get_db_path()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    conn.execute("PRAGMA busy_timeout=20000")

    class _Shim:
        pass
    s = _Shim()
    s.conn = conn
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=int, default=24,
                        help="window length (default: 24h)")
    parser.add_argument("--top", type=int, default=10,
                        help="how many top-noise sources to report (default: 10)")
    parser.add_argument("--json", action="store_true",
                        help="JSON only (no text summary)")
    args = parser.parse_args()

    store = _open_ro_store()
    report = audit(store, window_hours=args.hours, top_sources=args.top)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Prefloor pool audit — window: last {report['window_hours']}h")
    print(f"  Lifetime LLM-labeled pool:  {report['prefloor_total'] + report['real_llm_total']:>7d}")
    print(f"    of which prefloor (noise): {report['prefloor_total']:>7d} "
          f"({100*report['prefloor_share_lifetime']:.1f}%)")
    print(f"    of which real LLM signal:  {report['real_llm_total']:>7d}")
    print()
    print(f"  Window contamination:")
    print(f"    prefloor (noise): {report['window_prefloor']:>7d}")
    print(f"    real LLM signal:  {report['window_real_llm']:>7d}")
    print(f"    prefloor share:   {100*report['window_prefloor_share']:.1f}%  "
          f"verdict={report['verdict']}")
    if report["per_source_top"]:
        print()
        print(f"  Top noise-contributing sources in window:")
        for row in report["per_source_top"]:
            print(f"    {row['source'][:42]:<42}  {row['prefloor_count']:>5}")


if __name__ == "__main__":
    main()
