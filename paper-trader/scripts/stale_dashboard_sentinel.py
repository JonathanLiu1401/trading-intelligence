#!/usr/bin/env python3
"""Stale-dashboard sentinel.

The dashboard's ``/api/backtests/curves`` endpoint caps a single request at
100 ``run_ids`` and returns HTTP 400 for anything larger. The frontend was
fixed (commit b705071) to chunk requests into batches of 100, so a *current*
browser/client never trips the cap. When ``logs/runner.log`` keeps showing
``GET /api/backtests/curves?run_ids=...`` 400s with a long id list, it means a
**stale dashboard process is still serving the old un-chunked JS** — an
operational signal that the dashboard needs a restart, not a code bug.

This is a pure read-only log scanner (stdlib only, no DB, no network) so it is
safe to run from cron / hourly_review.sh without racing the live trader.

Usage:
    python3 scripts/stale_dashboard_sentinel.py            # scan default log
    python3 scripts/stale_dashboard_sentinel.py --log PATH
    python3 scripts/stale_dashboard_sentinel.py --max-age-min 60 --fail-threshold 5

Exit codes:
    0  clean (no stale-client 400s within the window, or below threshold)
    1  stale-client 400s detected at/above --fail-threshold within the window
    2  log file missing / unreadable
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "logs" / "runner.log"

# werkzeug access line, e.g.:
# 127.0.0.1 - - [17/May/2026 16:21:45] "GET /api/backtests/curves?run_ids=1,2 HTTP/1.1" 400 -
_LINE_RE = re.compile(
    r"\[(?P<ts>\d{2}/\w{3}/\d{4} \d{2}:\d{2}:\d{2})\]\s+"
    r'"GET (?P<path>/api/backtests/curves\?run_ids=(?P<ids>[0-9,]*)) HTTP/[0-9.]+"\s+'
    r"(?P<status>\d{3})"
)
_TS_FMT = "%d/%b/%Y %H:%M:%S"

# Server-side cap in dashboard.py (>100 run_ids -> 400). Kept as a constant so
# the sentinel's notion of "oversized" matches the endpoint contract.
RUN_ID_CAP = 100


def _parse_ts(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw, _TS_FMT)
    except ValueError:
        return None


def scan(log_path: Path, max_age_min: int) -> dict:
    """Return summary of stale-client curve 400s newer than max_age_min.

    A "stale-client 400" is a /api/backtests/curves request that returned 400
    AND carried more than RUN_ID_CAP ids — the exact signature of old,
    un-chunked dashboard JS. Smaller 400s (malformed/empty) are reported
    separately so a real client bug is not masked by the stale-server case.
    """
    text = log_path.read_text(errors="replace").splitlines()

    cutoff = None
    if max_age_min > 0:
        # Anchor the window to the newest timestamp in the file rather than
        # wall-clock now(): logs may lag and we want "recent within the log".
        newest = None
        for line in reversed(text):
            m = _LINE_RE.search(line)
            if m:
                ts = _parse_ts(m.group("ts"))
                if ts:
                    newest = ts
                    break
        anchor = newest or datetime.now()
        cutoff = anchor - timedelta(minutes=max_age_min)

    stale_hits: list[datetime] = []
    other_400: list[datetime] = []
    for line in text:
        m = _LINE_RE.search(line)
        if not m or m.group("status") != "400":
            continue
        ts = _parse_ts(m.group("ts"))
        if ts is None:
            continue
        if cutoff and ts < cutoff:
            continue
        ids = [s for s in m.group("ids").split(",") if s]
        if len(ids) > RUN_ID_CAP:
            stale_hits.append(ts)
        else:
            other_400.append(ts)

    stale_hits.sort()
    other_400.sort()
    return {
        "log": str(log_path),
        "window_min": max_age_min,
        "stale_client_400": len(stale_hits),
        "other_curve_400": len(other_400),
        "first": stale_hits[0].isoformat() if stale_hits else None,
        "last": stale_hits[-1].isoformat() if stale_hits else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG,
                    help=f"access log path (default: {DEFAULT_LOG})")
    ap.add_argument("--max-age-min", type=int, default=60,
                    help="only count 400s within this many minutes of the "
                         "newest log entry (0 = whole file)")
    ap.add_argument("--fail-threshold", type=int, default=3,
                    help="exit 1 if stale-client 400s >= this count")
    args = ap.parse_args(argv)

    if not args.log.exists():
        print(f"[stale-sentinel] log not found: {args.log}", file=sys.stderr)
        return 2
    try:
        summary = scan(args.log, args.max_age_min)
    except OSError as e:
        print(f"[stale-sentinel] cannot read log: {e}", file=sys.stderr)
        return 2

    n = summary["stale_client_400"]
    if n == 0:
        print(f"[stale-sentinel] OK — no stale-client curve 400s in last "
              f"{summary['window_min']}min ({summary['other_curve_400']} "
              f"other curve 400s)")
        return 0

    print(f"[stale-sentinel] STALE DASHBOARD — {n} oversized "
          f"/api/backtests/curves 400s in last {summary['window_min']}min "
          f"(first={summary['first']} last={summary['last']}). "
          f"The dashboard process is serving pre-b705071 un-chunked JS; "
          f"restart the dashboard to pick up the chunking fix.")
    return 1 if n >= args.fail_threshold else 0


if __name__ == "__main__":
    raise SystemExit(main())
