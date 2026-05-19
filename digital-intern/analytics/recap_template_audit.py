"""Recap-template noise audit (read-only).

The recap / preview / transcript-summary template class
("Why X Stock Is Trading Up Today", "Q1 2026 Earnings Call Highlights",
"Stock Market Today, May 18: ...", "Here What the Street Thinks ...",
"GF Value Says ...") is now gated on THREE surfaces — see the lockstep
parity test in ``tests/test_urgency_recap_prefilter.py``:

  1. ``watchers.urgency_scorer.score_batch`` — pre-filters BEFORE the
     Sonnet call (saves quota + stops training-pool poisoning).
  2. ``watchers.alert_agent.send_urgent_alert`` — suppresses standalone
     🚨 BREAKING pushes on caught rows.
  3. ``analysis.claude_analyst._build_payload`` — drops recap rows from
     the 5h Opus heartbeat newswire.

All three resolve fingerprints through one source of truth
(``alert_agent._looks_like_recap_template``). This module is the
*calibration view* analysts and the dashboard need to ANSWER:

  * "Is the pre-filter still working — did a new SEO template variant
    sneak through and start firing again?"
  * "How much LLM quota did the pre-filter save in the recent window?"
  * "Are recap rows in the strong-label training pool growing
    again?" (the original poisoning the fix exists to prevent)

The audit groups rows in the recent window by their CURRENT state
(score_source, ai_score band, urgency) so a regression manifests as
a nonzero ``score_source='llm' AND ai_score>=8`` row count — exactly
what the live evidence (2026-05-18/19: 10 such rows) showed before the
fix. Counterpart to ``ml/label_audit.py`` (training-pool integrity)
and ``ml/llm_promotion_audit.py`` (per-source LLM spend) — same shape,
different question.

Pure read-side (``COUNT(*)`` only). The recap fingerprint set lives in
``watchers.alert_agent`` so it can never silently drift from the live
gate. Backtest isolation is enforced via the same ``_LIVE_ONLY_CLAUSE``
fragment as the rest of the audit family — a synthetic backtest row
matching a recap title cannot inflate the calibration figure.

Run standalone::

    python3 -m analytics.recap_template_audit            # JSON report
    python3 -m analytics.recap_template_audit --hours 6  # custom window
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from watchers.alert_agent import _RECAP_TEMPLATE_PATTERNS


# Mirrors storage.article_store._LIVE_ONLY_CLAUSE — duplicated as a string
# constant rather than imported because this module deliberately does NOT
# pull the ArticleStore writer graph (we operate on a read-only connection).
# Pinned by ``test_live_only_clause_in_sync_with_storage`` in the test file.
LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)


def _count_recap_matches(
    conn,
    window_since: str,
    extra_where: str = "",
) -> dict[str, int]:
    """Per-fingerprint count of rows whose title matches a recap pattern
    AND falls in the recent window AND satisfies ``extra_where``. Pure
    Python regex (the patterns are compiled once at import time in
    ``watchers.alert_agent``) — no SQL ``REGEXP`` extension required.
    """
    # Fetch candidate rows from the indexed first_seen window first; the
    # regex scan is then bounded by that window (~minutes of articles).
    where = f"first_seen >= ? AND {LIVE_ONLY_CLAUSE}"
    if extra_where:
        where += f" AND ({extra_where})"
    rows = conn.execute(
        f"SELECT title FROM articles WHERE {where}",
        (window_since,),
    ).fetchall()
    counts = {name: 0 for name, _pat in _RECAP_TEMPLATE_PATTERNS}
    for (title,) in rows:
        if not title:
            continue
        for name, pat in _RECAP_TEMPLATE_PATTERNS:
            if pat.search(title):
                counts[name] += 1
                break  # one fingerprint per row — first wins (alert_agent precedent)
    return counts


def audit(store, hours: int = 24, now: Optional[datetime] = None) -> dict:
    """Calibration report for the recap-template gate over the last ``hours``.

    Returns::

        {
            "window_h": int,
            "by_fingerprint": {<name>: int, ...},  # all recap rows in window
            "total_recap_rows": int,
            "leaked_to_strong_pool": int,   # MUST be 0 post-fix; the regression
                                            # signal the audit exists to surface
            "leaked_urgent": int,           # urgency>=1 (some gate failed)
            "floored_to_noise": int,        # ai_score <= 0.5 (pre-filter goal)
            "leaked_by_fingerprint": {<name>: int, ...},
            "leak_fraction": float,         # leaked_to_strong_pool / total_recap_rows
            "ok": bool,                     # True iff zero strong-pool leaks
        }

    ``leaked_to_strong_pool`` is the load-bearing metric: a nonzero value
    means a recap row has score_source='llm' AND ai_score>=8 — i.e. it
    landed in the training pool tagged urgent. That is the exact poisoning
    the urgency_scorer pre-filter exists to prevent.
    """
    conn = store.conn
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(hours=hours)).isoformat()

    by_fp = _count_recap_matches(conn, since)
    total = sum(by_fp.values())

    leaked_by_fp = _count_recap_matches(
        conn, since, "score_source='llm' AND ai_score >= 8.0"
    )
    leaked_to_strong_pool = sum(leaked_by_fp.values())

    urgent_by_fp = _count_recap_matches(conn, since, "urgency >= 1")
    leaked_urgent = sum(urgent_by_fp.values())

    floored_by_fp = _count_recap_matches(
        conn, since, "ai_score > 0 AND ai_score <= 0.5"
    )
    floored = sum(floored_by_fp.values())

    leak_fraction = (
        round(leaked_to_strong_pool / total, 4) if total else 0.0
    )

    return {
        "window_h": int(hours),
        "by_fingerprint": by_fp,
        "total_recap_rows": total,
        "leaked_to_strong_pool": leaked_to_strong_pool,
        "leaked_urgent": leaked_urgent,
        "floored_to_noise": floored,
        "leaked_by_fingerprint": leaked_by_fp,
        "leak_fraction": leak_fraction,
        "ok": leaked_to_strong_pool == 0,
    }


def format_report(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


class _RoStore:
    """Read-only ``.conn``-bearing shim — mirrors ``ml/label_audit._RoStore``.
    Never opens the writable ArticleStore (which would block on busy_timeout
    under daemon writer-contention and run the score_source migration)."""

    def __init__(self, db_path) -> None:
        import sqlite3
        self.conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=15,
        )

    def close(self) -> None:
        self.conn.close()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hours", type=int, default=24,
        help="Window size in hours (default: 24)",
    )
    args = parser.parse_args(argv)

    from storage.article_store import _get_db_path

    store = _RoStore(_get_db_path())
    try:
        report = audit(store, hours=args.hours)
    finally:
        store.close()
    print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
