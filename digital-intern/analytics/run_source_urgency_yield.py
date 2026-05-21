"""Runner: per-source urgent-yield audit -> /home/zeph/logs/source_urgency_yield.json."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from analytics.source_urgency_yield import build_source_urgency_yield
from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT_PATH = Path("/home/zeph/logs/source_urgency_yield.json")
WINDOW_HOURS = 24
FETCH_LIMIT = 8000


def main() -> int:
    db = _get_db_path()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).isoformat(timespec="seconds")
    rows = conn.execute(
        f"""SELECT source, urgency, first_seen
              FROM articles
             WHERE {_LIVE_ONLY_CLAUSE}
               AND first_seen >= ?
             ORDER BY first_seen DESC
             LIMIT ?""",
        (cutoff, FETCH_LIMIT),
    ).fetchall()
    conn.close()

    articles = [{"source": r[0], "urgency": r[1], "first_seen": r[2]} for r in rows]
    report = build_source_urgency_yield(articles, hours=WINDOW_HOURS)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print(f"source_urgency_yield: scanned={len(rows)} state={report['state']} sources={report['n_sources']}")
    print(f"  NOISY={report['n_noisy']} CLEAN={report['n_clean']} QUIET={report['n_quiet']} UNKNOWN={report['n_unknown']}")
    print(f"  headline: {report['headline']}")
    for row in report["sources"][:5]:
        sr = row.get("suppression_rate")
        print(f"  [{row['verdict']}] {row['source']}: total={row['total']} urgent={row['urgent']} alerted={row['alerted']} suppress={sr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
