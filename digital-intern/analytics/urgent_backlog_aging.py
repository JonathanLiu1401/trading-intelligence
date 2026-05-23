"""Urgent-backlog aging snapshot — where in the 24h alerter window is the
unalerted urgency=1 backlog actually sitting?

Why this exists (analyst lens): the existing dashboard surfaces three numbers
about the queued (``urgency=1``, not yet pushed) urgent backlog —
``queued``, ``near_reap`` (within ~3h of being demoted), and ``overdue``
(>=24h, already lost). That tells the analyst the *worst*-case state but
not the *shape* of the queue. The diagnostic question "where in the 24h
window is the backlog concentrated?" answers a very different concern:

  * **Mass in 0-4h buckets**  → alert worker is keeping up; freshly scored
    items will be pushed on the next 20s cycle.
  * **Mass in 4-12h buckets** → alert worker is processing but slowly; a
    persistent ~8h tail means Sonnet quota throttling, recap-template gate
    suppressions, or a queue-depth backlog the alerter can't drain.
  * **Mass in 12-24h buckets** → alerter has effectively given up on these
    rows; they're about to age out. This is the "Sonnet went dark at ~12h
    ago" signature that the aggregate ``llm_fraction`` metric (which is
    monotonically integrated over the window) cannot show.
  * **Mass past 24h (overdue)** → silent missed alerts; the row crossed
    the alerter's fetch window and is awaiting reaper demotion.

Live evidence (2026-05-23 16:30Z): 81 queued urgency=1 rows, of which 22
were >24h old (already overdue, silently missed). The aggregate
``llm_fraction`` for the same 24h window was 2.3% (4 LLM-vetted of 173
urgent rows) — but the analyst could not tell from that alone whether the
problem was a recent quota throttle or a persistent baseline. Binning by
age would have made it visible: a concentration in the 25h+ overdue bucket
means the failure happened ~25h ago and the alerter has not produced a
single LLM-vetted urgent push since.

Pure read-side, single SELECT, ``_LIVE_ONLY_CLAUSE`` discipline (synthetic
backtest/opus rows never enter the queue and never inflate the counts —
the partial-filter regression class ``analytics/trend_velocity.py``
violates is what this discipline exists to prevent). NO DB write, no
ai_score / ml_score / score_source / urgency mutation. All four
load-bearing invariants intact by construction.

CLI::

    python3 -m analytics.urgent_backlog_aging
    python3 -m analytics.urgent_backlog_aging --json
    python3 -m analytics.urgent_backlog_aging --bucket-hours 2
    python3 -m analytics.urgent_backlog_aging --strict  # exit 1 on STUCK_OLD/OVERDUE_LOSS
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Iterable


REAP_AGE_HOURS = 24
# Verdict threshold: more than this fraction of queued rows sitting in the
# oldest in-window buckets (>= STUCK_OLD_AGE_H) means the alerter is not
# moving the queue and the analyst should investigate. Conservative — a
# small spike of late-arriving high-kw items is normal noise.
STUCK_OLD_AGE_H = 12.0
STUCK_OLD_FRACTION = 0.4
# Verdict threshold: any overdue (already-lost) rows are a hard signal that
# the analyst missed urgent items. Even one is worth surfacing — the alert
# was meant to push to Discord, and it didn't.
OVERDUE_THRESHOLD = 1


def _bucketize(
    ages_h: list[float],
    bucket_h: float,
    reap_age_h: float = REAP_AGE_HOURS,
) -> tuple[list[dict], int]:
    """Bin a list of row-age-in-hours values into fixed-width buckets across
    ``[0, reap_age_h]`` plus a trailing ``overdue`` bucket for ages >= reap.

    Returns ``(buckets, overdue_count)`` where each in-window bucket has::

        {"start_h": float, "end_h": float, "count": int}

    Pure function. Test-pinned so the bin edges can never silently drift.
    """
    if bucket_h <= 0:
        raise ValueError("bucket_h must be > 0")
    n_buckets = max(1, int((reap_age_h + bucket_h - 1) // bucket_h))
    buckets = [
        {"start_h": i * bucket_h,
         "end_h": min((i + 1) * bucket_h, float(reap_age_h)),
         "count": 0}
        for i in range(n_buckets)
    ]
    overdue = 0
    for age in ages_h:
        if age >= reap_age_h:
            overdue += 1
            continue
        if age < 0:
            # Defensive — a future-dated first_seen is not a real bucket.
            continue
        idx = min(int(age // bucket_h), n_buckets - 1)
        buckets[idx]["count"] += 1
    return buckets, overdue


def _verdict(
    queued: int,
    overdue: int,
    stuck_old_fraction: float,
) -> str:
    """Single-word health verdict for the backlog.

    * ``OVERDUE_LOSS`` — at least one already-lost row (>=24h, silently
      missed). Highest severity because the analyst was supposed to be
      pushed and was not.
    * ``STUCK_OLD``   — at least ``STUCK_OLD_FRACTION`` of the queue sits
      in the >=12h in-window region. The alerter is not draining; the
      backlog is shifting toward the reaper deadline.
    * ``EMPTY``        — no queued rows at all (alerter has caught up or
      no urgent events in the window).
    * ``HEALTHY``      — queue exists and is concentrated in fresh buckets.
    """
    if overdue >= OVERDUE_THRESHOLD:
        return "OVERDUE_LOSS"
    if queued == 0:
        return "EMPTY"
    if stuck_old_fraction >= STUCK_OLD_FRACTION:
        return "STUCK_OLD"
    return "HEALTHY"


def audit(
    store,
    bucket_h: float = 4.0,
    now: datetime | None = None,
) -> dict:
    """Snapshot the current queued urgent backlog by age.

    Reads only live (``_LIVE_ONLY_CLAUSE``) ``urgency=1`` rows — synthetic
    backtest/opus rows never enter the queue (inserted ``urgency=0`` by
    construction) and are filtered defensively. Pure read-side.

    Schema::

        {
            "queued":        int,              # all live urgency=1 rows
            "overdue":       int,              # subset with age >= 24h
            "in_window":     int,              # queued - overdue
            "oldest_age_h":  float | None,     # None when queued == 0
            "median_age_h":  float | None,     # None when queued == 0
            "bucket_h":      float,            # bin width (input)
            "reap_age_h":    int,              # 24 (the alerter's window)
            "buckets": [                       # in-window bins, oldest last
                {"start_h": float, "end_h": float, "count": int},
                ...
            ],
            "stuck_old_count":    int,         # in_window rows w/ age >= 12h
            "stuck_old_fraction": float,       # / queued (0.0 when queued==0)
            "verdict": "OVERDUE_LOSS" | "STUCK_OLD" | "EMPTY" | "HEALTHY",
        }

    NO DB write, no ai_score / ml_score / score_source / urgency mutation.
    All four load-bearing invariants intact by construction.
    """
    from storage.article_store import _LIVE_ONLY_CLAUSE

    if now is None:
        now = datetime.now(timezone.utc)

    rows = store.conn.execute(
        "SELECT first_seen FROM articles "
        f"WHERE urgency=1 AND {_LIVE_ONLY_CLAUSE}"
    ).fetchall()

    ages_h: list[float] = []
    for (first_seen,) in rows:
        if not first_seen:
            continue
        try:
            ts = datetime.fromisoformat(first_seen)
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (now - ts).total_seconds() / 3600.0
        if age < 0:
            continue
        ages_h.append(age)

    queued = len(ages_h)
    buckets, overdue = _bucketize(ages_h, bucket_h, REAP_AGE_HOURS)
    in_window = queued - overdue

    if queued:
        oldest_age_h = round(max(ages_h), 2)
        sorted_ages = sorted(ages_h)
        mid = queued // 2
        if queued % 2 == 0:
            median_age_h = round((sorted_ages[mid - 1] + sorted_ages[mid]) / 2, 2)
        else:
            median_age_h = round(sorted_ages[mid], 2)
    else:
        oldest_age_h = None
        median_age_h = None

    stuck_old_count = sum(1 for a in ages_h if STUCK_OLD_AGE_H <= a < REAP_AGE_HOURS)
    stuck_old_fraction = round(stuck_old_count / queued, 4) if queued else 0.0
    verdict = _verdict(queued, overdue, stuck_old_fraction)

    return {
        "queued": queued,
        "overdue": overdue,
        "in_window": in_window,
        "oldest_age_h": oldest_age_h,
        "median_age_h": median_age_h,
        "bucket_h": float(bucket_h),
        "reap_age_h": int(REAP_AGE_HOURS),
        "buckets": buckets,
        "stuck_old_count": stuck_old_count,
        "stuck_old_fraction": stuck_old_fraction,
        "verdict": verdict,
    }


def _format_report(report: dict) -> str:
    lines = [
        "Urgent-backlog aging snapshot",
        "=" * 50,
        f"Verdict:        {report['verdict']}",
        f"Queued:         {report['queued']}  "
        f"(in-window={report['in_window']}, overdue={report['overdue']})",
    ]
    if report["oldest_age_h"] is not None:
        lines.append(
            f"Oldest age:     {report['oldest_age_h']:.2f}h    "
            f"median: {report['median_age_h']:.2f}h"
        )
    lines.append(
        f"Stuck-old:      {report['stuck_old_count']}  "
        f"({report['stuck_old_fraction']*100:.1f}% of queued, "
        f"age >= {STUCK_OLD_AGE_H:.0f}h and < {REAP_AGE_HOURS}h)"
    )
    lines.append("")
    lines.append(f"Buckets ({report['bucket_h']}h wide, oldest last):")
    bar_max = max((b["count"] for b in report["buckets"]), default=0)
    bar_max = max(bar_max, report["overdue"], 1)
    for b in report["buckets"]:
        bar = "#" * int(20 * b["count"] / bar_max)
        lines.append(
            f"  [{b['start_h']:>5.1f}h - {b['end_h']:>5.1f}h]  "
            f"{b['count']:>4d}  {bar}"
        )
    if report["overdue"]:
        bar = "!" * int(20 * report["overdue"] / bar_max)
        lines.append(
            f"  [{REAP_AGE_HOURS}h+ OVERDUE      ]  "
            f"{report['overdue']:>4d}  {bar}"
        )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true",
                    help="emit the report as JSON instead of text")
    ap.add_argument("--bucket-hours", type=float, default=4.0,
                    help="age-bucket width in hours (default 4)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on STUCK_OLD or OVERDUE_LOSS (for CI gates)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    from storage.article_store import ArticleStore
    store = ArticleStore()
    report = audit(store, bucket_h=args.bucket_hours)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_report(report))

    if args.strict and report["verdict"] in ("STUCK_OLD", "OVERDUE_LOSS"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
