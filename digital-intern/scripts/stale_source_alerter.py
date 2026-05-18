#!/usr/bin/env python3
"""Stale source alerter / collector freshness monitor.

Reads a bounded recent slice of the articles DB (fast even on the slow
external drive: one rowid-ordered scan of small columns), then for every
source that is *active* (appeared >= MIN_ACTIVE times in the slice) computes
how long it has been since its most recent article relative to the newest
article in the slice. Sources whose lag exceeds STALE_HOURS are flagged.

Anchoring on the slice's max first_seen (not wall-clock now()) means a
delayed daemon does not produce false staleness alarms — we measure each
collector against the freshest data we actually have, not against real time.

Output: prints a human report (>=3 lines) and writes a JSON snapshot to
/home/zeph/logs/source_freshness.json. Read-only on the DB; no edits to any
shared pipeline file.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(__file__), "..", "data", "articles.db")
OUT = "/home/zeph/logs/source_freshness.json"

ROWID_WINDOW = 12000   # ~5h of data given current ingest rate; one fast scan
MIN_ACTIVE = 5         # only judge sources that actually published in the slice
STALE_HOURS = 2.0      # lag (vs newest article) past which a source is stale


def _parse(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        try:
            dt = datetime.fromisoformat(ts.replace(" ", "T"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> int:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=15)
    max_rowid = con.execute("SELECT MAX(rowid) FROM articles").fetchone()[0]
    if not max_rowid:
        print("stale_source_alerter: empty articles table", file=sys.stderr)
        return 1
    lo = max(0, max_rowid - ROWID_WINDOW)
    rows = con.execute(
        "SELECT source, first_seen FROM articles "
        "WHERE rowid > ? AND source NOT LIKE 'backtest_run_%' ORDER BY rowid",
        (lo,),
    ).fetchall()
    con.close()

    # backtest_run_* are batch backtest inserts, not live collectors; they are
    # excluded above so they never pollute the staleness signal.
    agg: dict[str, dict] = {}
    newest: datetime | None = None
    for source, fs in rows:
        dt = _parse(fs)
        if dt is None:
            continue
        if newest is None or dt > newest:
            newest = dt
        a = agg.setdefault(source, {"count": 0, "last": dt})
        a["count"] += 1
        if dt > a["last"]:
            a["last"] = dt

    if newest is None:
        print("stale_source_alerter: no parseable timestamps", file=sys.stderr)
        return 1

    active = {s: v for s, v in agg.items() if v["count"] >= MIN_ACTIVE}
    report = []
    for source, v in active.items():
        lag_h = (newest - v["last"]).total_seconds() / 3600.0
        report.append(
            {
                "source": source,
                "count": v["count"],
                "last_seen": v["last"].isoformat(),
                "lag_hours": round(lag_h, 2),
                "stale": lag_h >= STALE_HOURS,
            }
        )
    report.sort(key=lambda r: r["lag_hours"], reverse=True)
    stale = [r for r in report if r["stale"]]

    snapshot = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "slice_rows": len(rows),
        "rowid_window": [lo + 1, max_rowid],
        "anchor_newest": newest.isoformat(),
        "active_sources": len(active),
        "stale_count": len(stale),
        "stale_hours_threshold": STALE_HOURS,
        "sources": report,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(
        f"freshness: {len(rows)} rows / {len(active)} active sources "
        f"(>= {MIN_ACTIVE} pub), anchor={newest.isoformat()}"
    )
    if stale:
        print(f"STALE ({len(stale)} source(s) no data >= {STALE_HOURS}h):")
        for r in stale[:10]:
            print(
                f"  ! {r['source']}: last {r['last_seen']} "
                f"(lag {r['lag_hours']}h, {r['count']} in slice)"
            )
    else:
        print("STALE: none — all active sources fresh")
    print("freshest active sources:")
    for r in sorted(report, key=lambda r: r["lag_hours"])[:5]:
        print(
            f"  ok {r['source']}: lag {r['lag_hours']}h "
            f"({r['count']} in slice)"
        )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
