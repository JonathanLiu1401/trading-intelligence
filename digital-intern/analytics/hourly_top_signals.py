"""Hourly top signals: best-scored articles from the last hour, no urgency filter.

Unlike daily_digest (which requires urgency>=2 and covers 24h), this module
surfaces the top articles by unified score from the *last hour only*, regardless
of urgency. This catches high-scored articles that haven't yet crossed the
urgency threshold — early warning signals the digest would miss.

Unified score: COALESCE(NULLIF(ml_score,0), NULLIF(ai_score,0), kw_score, 0)
— same convention used across analytics siblings.

Design constraints:
  * Bounded SCAN_LIMIT read via idx_first_seen — never full-table scan.
  * Read-only sqlite URI.
  * USB-safe busy_timeout.
  * _LIVE_ONLY_CLAUSE to exclude synthetic backtest/opus rows.

Artifacts:
  * /home/zeph/logs/hourly_top_signals.json
  * /home/zeph/logs/hourly_top_signals.txt  (human-readable)

Standalone: python3 -m analytics.hourly_top_signals
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from storage.article_store import _LIVE_ONLY_CLAUSE  # noqa: E402

DB = REPO_ROOT / "data" / "articles.db"
LOG_DIR = Path("/home/zeph/logs")
OUT_JSON = LOG_DIR / "hourly_top_signals.json"
OUT_TXT = LOG_DIR / "hourly_top_signals.txt"

TOP_N = 5
SCAN_LIMIT = 2000   # ~1-2h of live rows at typical ingest rate
WINDOW_MINUTES = 60
MIN_SCORE = 1.0     # ignore noise (score effectively 0)


def _unified_score(row: sqlite3.Row) -> float:
    ml = row["ml_score"] or 0.0
    ai = row["ai_score"] or 0.0
    kw = row["kw_score"] or 0.0
    return ml or ai or kw


def main() -> dict:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout=8000")
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        f"""
        SELECT id, title, source, url, urgency,
               ml_score, ai_score, kw_score, first_seen, score_source
          FROM articles
         WHERE {_LIVE_ONLY_CLAUSE}
           AND first_seen >= datetime('now', '-{WINDOW_MINUTES} minutes')
         ORDER BY first_seen DESC
         LIMIT {SCAN_LIMIT}
        """
    ).fetchall()
    conn.close()

    now = datetime.now(timezone.utc)
    total = len(rows)

    scored = [r for r in rows if _unified_score(r) >= MIN_SCORE]
    ranked = sorted(scored, key=_unified_score, reverse=True)[:TOP_N]

    top = []
    for r in ranked:
        score = _unified_score(r)
        top.append({
            "title": r["title"],
            "source": r["source"],
            "url": r["url"],
            "urgency": r["urgency"],
            "score": round(score, 2),
            "score_source": r["score_source"],
            "ml_score": r["ml_score"],
            "ai_score": r["ai_score"],
            "kw_score": r["kw_score"],
            "first_seen": r["first_seen"],
        })

    result = {
        "generated_at": now.isoformat(),
        "window_minutes": WINDOW_MINUTES,
        "total_articles": total,
        "scored_articles": len(scored),
        "top": top,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        f"[HOURLY TOP SIGNALS] {now.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Articles last {WINDOW_MINUTES}m: {total} total, {len(scored)} scored",
        "",
    ]
    if top:
        for i, a in enumerate(top, 1):
            urgency_tag = f" [URG={a['urgency']}]" if a["urgency"] else ""
            lines.append(f"{i}. [{a['score']:.1f}]{urgency_tag} {a['title'][:80]}")
            lines.append(f"   {a['source']} | {a['first_seen'][:16]}")
    else:
        lines.append("No scored articles in window.")

    OUT_TXT.write_text("\n".join(lines) + "\n")

    for line in lines:
        print(line)

    return result


if __name__ == "__main__":
    main()
