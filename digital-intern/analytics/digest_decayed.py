"""Recency-decay-aware daily digest.

Ranks the last 24h of live articles by decay-adjusted score
(ml_score * exp(-age / HALF_LIFE_HOURS)) so fresh signals surface above
stale high-scorers.  Also shows each article's rank in the raw-score ordering
so the operator can see which articles are "recent movers" vs "fading old news".

Writes a plain-text summary to /home/zeph/logs/daily_digest_decayed.txt and a
machine-readable JSON to /home/zeph/logs/daily_digest_decayed.json.

Key difference from existing tools:
  * ``daily_digest.py``   — urgency>=2 only, ranked by raw score
  * ``recency_decay.py``  — ranked by effective_score, no urgency gate, no rank-shift column
  * ``digest_decayed.py`` — all live articles, ranked by effective_score,
                            rank-shift column shows freshness lift
Standalone: python3 -m analytics.digest_decayed
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

TXT_OUT = Path("/home/zeph/logs/daily_digest_decayed.txt")
JSON_OUT = Path("/home/zeph/logs/daily_digest_decayed.json")

HALF_LIFE_HOURS = 4.0   # score halves every 4 hours; 12h-old article ~15% of original
LOOKBACK_HOURS = 24
SCAN_LIMIT = 8000
TOP_N = 10


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = raw.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    s2 = s.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt = datetime.strptime(s2, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _best_score(ml, ai, kw) -> float | None:
    for v in (ml, ai, kw):
        if v is not None:
            return float(v)
    return None


def compute() -> dict:
    db = _get_db_path()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    conn.execute("PRAGMA busy_timeout=10000")
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=LOOKBACK_HOURS)).isoformat()

    cur = conn.execute(
        f"""
        SELECT id, title, source, first_seen, ml_score, ai_score, kw_score, urgency
          FROM articles INDEXED BY idx_first_seen
         WHERE first_seen >= ?
           AND {_LIVE_ONLY_CLAUSE}
        ORDER BY first_seen DESC
        LIMIT {SCAN_LIMIT}
        """,
        (cutoff,),
    )

    rows: list[dict] = []
    for aid, title, source, first_seen, ml_score, ai_score, kw_score, urgency in cur:
        raw = _best_score(ml_score, ai_score, kw_score)
        if raw is None:
            continue
        ts = _parse_ts(first_seen)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
        decay = math.exp(-math.log(2) * age_h / HALF_LIFE_HOURS)
        rows.append({
            "id": aid,
            "title": (title or "").strip().replace("\n", " ")[:120],
            "source": source or "?",
            "first_seen": first_seen,
            "age_hours": round(age_h, 2),
            "raw_score": round(raw, 4),
            "decay": round(decay, 4),
            "effective_score": round(raw * decay, 4),
            "urgency": urgency or 0,
        })
    conn.close()

    # Build raw-score ranking for rank-shift calculation
    by_raw = sorted(rows, key=lambda r: r["raw_score"], reverse=True)
    raw_rank = {r["id"]: i + 1 for i, r in enumerate(by_raw)}

    # Sort by effective (decay-adjusted) score
    by_eff = sorted(rows, key=lambda r: r["effective_score"], reverse=True)
    for eff_rank, r in enumerate(by_eff, 1):
        r["eff_rank"] = eff_rank
        r["raw_rank"] = raw_rank[r["id"]]
        r["rank_lift"] = r["raw_rank"] - eff_rank  # positive = boosted by recency

    top = by_eff[:TOP_N]
    urgent_count = sum(1 for r in rows if r["urgency"] >= 2)

    # Build text digest
    ts_str = now.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"=== DECAY-RANKED DIGEST  {ts_str} ===",
        f"Scanned: {len(rows):,} scored articles (24h live)  |  urgent>=2: {urgent_count}",
        f"Half-life: {HALF_LIFE_HOURS}h  |  Top {TOP_N} by effective_score",
        "-" * 70,
    ]
    if not top:
        lines.append("(no scored articles in window)")
    else:
        for r in top:
            lift_str = f"+{r['rank_lift']}" if r['rank_lift'] > 0 else str(r['rank_lift'])
            urgency_tag = f" u{r['urgency']}" if r["urgency"] >= 1 else ""
            age_tag = f"{r['age_hours']:.1f}h"
            lines.append(
                f"#{r['eff_rank']:>2} [eff={r['effective_score']:.3f} raw={r['raw_score']:.3f}"
                f" lift={lift_str:>4}]{urgency_tag}  age={age_tag}"
            )
            lines.append(f"    {r['title'][:100]}  <{r['source']}>")

    TXT_OUT.parent.mkdir(parents=True, exist_ok=True)
    TXT_OUT.write_text("\n".join(lines) + "\n")

    payload = {
        "generated_at": now.isoformat(timespec="seconds"),
        "half_life_hours": HALF_LIFE_HOURS,
        "lookback_hours": LOOKBACK_HOURS,
        "scanned": len(rows),
        "urgent_count": urgent_count,
        "top": top,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2))

    return payload


def main() -> int:
    result = compute()
    print(f"digest_decayed: scanned={result['scanned']} urgent={result['urgent_count']}")
    for r in result["top"][:5]:
        lift_str = f"+{r['rank_lift']}" if r['rank_lift'] > 0 else str(r['rank_lift'])
        print(
            f"  #{r['eff_rank']} eff={r['effective_score']:.3f} raw={r['raw_score']:.3f}"
            f" lift={lift_str} age={r['age_hours']:.1f}h | {r['source']} | {r['title'][:80]}"
        )
    print(f"txt={TXT_OUT}  json={JSON_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
