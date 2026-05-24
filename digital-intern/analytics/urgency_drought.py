"""Urgency drought monitor.

Tracks the elapsed time since the last urgency>=2 article was produced.
A long drought signals that the LLM triage / alert pipeline may have
stalled (quota exhaustion, Sonnet throttle, or scoring backlog).

Thresholds:
  * < WARN_HOURS  → status="ok"
  * >= WARN_HOURS → status="warn"
  * >= ALERT_HOURS → status="alert"

Also reports urgency=1 (queued) drought separately: a healthy system
should continuously produce urgency=1 candidates even if LLM conversion
is slow.

Design constraints:
  * No full COUNT(*) on the 1.4 GB USB-backed DB.
  * Two small ORDER BY … LIMIT 1 reads (served by idx_first_seen /
    urgency index); negligible IO.
  * Read-only connection, busy_timeout=5 000 ms.
  * _LIVE_ONLY_CLAUSE applied — synthetic backtest rows carry urgency=0
    by construction but the discipline prevents future regression.

Standalone:  python3 -m analytics.urgency_drought
Output:      /home/zeph/logs/urgency_drought.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT = Path("/home/zeph/logs/urgency_drought.json")
WARN_HOURS = 4.0
ALERT_HOURS = 12.0


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        s = str(raw).replace(" ", "T")
        if "+" not in s[10:] and not s.endswith("Z"):
            s += "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _last_ts(conn: sqlite3.Connection, min_urgency: int) -> str | None:
    row = conn.execute(
        f"SELECT first_seen FROM articles"
        f" WHERE urgency >= ? AND {_LIVE_ONLY_CLAUSE}"
        f" ORDER BY first_seen DESC LIMIT 1",
        (min_urgency,),
    ).fetchone()
    return row[0] if row else None


def compute() -> dict:
    now = datetime.now(timezone.utc)

    conn = sqlite3.connect(f"file:{_get_db_path()}?mode=ro", uri=True, timeout=10)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        last_u2_raw = _last_ts(conn, 2)
        last_u1_raw = _last_ts(conn, 1)
    finally:
        conn.close()

    def _drought(raw: str | None) -> dict:
        if raw is None:
            return {"last_seen": None, "hours_ago": None, "status": "unknown"}
        ts = _parse_ts(raw)
        if ts is None:
            return {"last_seen": raw, "hours_ago": None, "status": "unknown"}
        hours_ago = (now - ts).total_seconds() / 3600.0
        if hours_ago >= ALERT_HOURS:
            status = "alert"
        elif hours_ago >= WARN_HOURS:
            status = "warn"
        else:
            status = "ok"
        return {
            "last_seen": ts.isoformat(),
            "hours_ago": round(hours_ago, 2),
            "status": status,
        }

    u2 = _drought(last_u2_raw)
    u1 = _drought(last_u1_raw)

    # Overall status = worst of the two
    rank = {"ok": 0, "warn": 1, "alert": 2, "unknown": 1}
    overall = max(u2["status"], u1["status"], key=lambda s: rank.get(s, 0))

    payload = {
        "generated_at": now.isoformat(),
        "warn_hours": WARN_HOURS,
        "alert_hours": ALERT_HOURS,
        "status": overall,
        "urgency_2": u2,
        "urgency_1": u1,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    p = compute()
    u2 = p["urgency_2"]
    u1 = p["urgency_1"]
    print(
        f"urgency_drought: status={p['status']}"
        f"  u2_last={u2['last_seen']}  u2_ago={u2['hours_ago']}h [{u2['status']}]"
        f"  u1_last={u1['last_seen']}  u1_ago={u1['hours_ago']}h [{u1['status']}]"
    )
    if p["status"] in ("warn", "alert"):
        print(f"  *** {p['status'].upper()}: no urgency>=2 articles for {u2['hours_ago']}h ***")
    print(f"  output={OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
