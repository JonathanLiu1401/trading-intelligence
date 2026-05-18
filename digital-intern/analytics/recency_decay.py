"""Apply exponential time-decay to ml_score and surface the freshest high-signal articles."""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/recency_decayed_top.json")
HALF_LIFE_HOURS = 4.0
LOOKBACK_HOURS = 24
TOP_N = 20


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = raw.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        s2 = s.replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                dt = datetime.strptime(s2, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def compute(limit_scan: int = 5000) -> list[dict]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    conn.execute("PRAGMA busy_timeout=5000")
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=LOOKBACK_HOURS)).isoformat()
    cur = conn.execute(
        """
        SELECT id, title, source, first_seen, ml_score, urgency
        FROM articles INDEXED BY idx_first_seen
        WHERE first_seen >= ?
          AND ml_score IS NOT NULL
        ORDER BY first_seen DESC
        LIMIT ?
        """,
        (cutoff, limit_scan),
    )
    rows: list[dict] = []
    for aid, title, source, first_seen, ml_score, urgency in cur:
        ts = _parse_ts(first_seen)
        if ts is None or ml_score is None:
            continue
        age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
        decay = math.exp(-age_h / HALF_LIFE_HOURS)
        rows.append({
            "id": aid,
            "title": (title or "")[:160],
            "source": source,
            "first_seen": first_seen,
            "age_hours": round(age_h, 2),
            "ml_score": round(float(ml_score), 4),
            "decay": round(decay, 4),
            "effective_score": round(float(ml_score) * decay, 4),
            "urgency": urgency,
        })
    conn.close()
    rows.sort(key=lambda r: r["effective_score"], reverse=True)
    return rows[:TOP_N]


def main() -> int:
    top = compute()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "half_life_hours": HALF_LIFE_HOURS,
        "lookback_hours": LOOKBACK_HOURS,
        "count": len(top),
        "top": top,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"recency_decay: wrote {len(top)} rows -> {OUT_PATH}")
    for r in top[:5]:
        print(f"  eff={r['effective_score']:.3f} age={r['age_hours']:.1f}h ml={r['ml_score']:.3f} | {r['source']} | {r['title'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
