#!/usr/bin/env python3
"""Scoring backlog audit.

For each source, report how many of its most recent articles are still
unscored (ml_score IS NULL) and how stale the oldest unscored row is.
Alerts (via log line) when the backlog ratio exceeds a threshold.

Constraints (see workspace memory):
  * articles.db lives on USB and is under constant write contention.
    No full COUNT(*) — single bounded idx_first_seen scan only.
  * Read-only connection, short busy_timeout.

Artifact: /home/zeph/logs/scoring_backlog.json (overwritten each run)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "articles.db")
OUT_PATH = "/home/zeph/logs/scoring_backlog.json"
SCAN_LIMIT = 800
ALERT_RATIO = 0.40  # >40% of a source's recent rows unscored

def main() -> int:
    if not os.path.exists(DB_PATH):
        print("db missing", file=sys.stderr)
        return 0
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.execute("PRAGMA busy_timeout=10000")
    rows = conn.execute(
        "SELECT source, first_seen, ml_score FROM articles "
        "INDEXED BY idx_first_seen ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    per_source: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "unscored": 0, "oldest_unscored": None}
    )
    for source, first_seen, ml_score in rows:
        s = source or "unknown"
        d = per_source[s]
        d["total"] += 1
        if ml_score is None:
            d["unscored"] += 1
            if d["oldest_unscored"] is None or first_seen < d["oldest_unscored"]:
                d["oldest_unscored"] = first_seen

    report = []
    alerts = []
    for s, d in per_source.items():
        if d["total"] < 5:
            continue
        ratio = d["unscored"] / d["total"]
        item = {
            "source": s,
            "sampled": d["total"],
            "unscored": d["unscored"],
            "unscored_ratio": round(ratio, 3),
            "oldest_unscored": d["oldest_unscored"],
        }
        report.append(item)
        if ratio >= ALERT_RATIO:
            alerts.append(item)

    report.sort(key=lambda x: x["unscored_ratio"], reverse=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sampled_rows": len(rows),
        "alert_threshold": ALERT_RATIO,
        "alerts": alerts,
        "sources": report,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"scoring_backlog: scanned={len(rows)} sources={len(report)} alerts={len(alerts)}")
    for a in alerts[:5]:
        print(f"  ALERT {a['source']}: {a['unscored']}/{a['sampled']} unscored "
              f"({a['unscored_ratio']:.0%}) oldest={a['oldest_unscored']}")
    for r in report[:3]:
        print(f"  top {r['source']}: {r['unscored']}/{r['sampled']} "
              f"unscored ratio={r['unscored_ratio']:.2f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
