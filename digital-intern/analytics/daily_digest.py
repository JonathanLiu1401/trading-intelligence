"""Daily digest: top urgent articles by score over the last 24h.

Writes a plain-text digest of the highest-signal urgent articles
to /home/zeph/logs/daily_digest.txt. Excludes synthetic backtest_run
sources so the digest reflects real wire copy only.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT = Path("/home/zeph/logs/daily_digest.txt")
TOP_N = 5
WINDOW_HOURS = 24
SCAN_LIMIT = 20000


def _score(row) -> float:
    ml, ai, kw = row["ml_score"], row["ai_score"], row["kw_score"]
    for v in (ml, ai, kw):
        if v is not None:
            return float(v)
    return 0.0


def compute():
    conn = sqlite3.connect(str(DB))
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    cutoff = f"-{WINDOW_HOURS} hours"
    rows = conn.execute(
        """
        SELECT id, title, source, url, urgency, ai_score, ml_score, kw_score, first_seen
          FROM articles
         WHERE id IN (SELECT id FROM articles ORDER BY id DESC LIMIT ?)
           AND urgency >= 2
           AND source NOT LIKE 'backtest_run_%'
           AND datetime(replace(first_seen,'T',' ')) >= datetime('now', ?)
        """,
        (SCAN_LIMIT, cutoff),
    ).fetchall()

    total_real = conn.execute(
        """
        SELECT COUNT(*) FROM articles
         WHERE id IN (SELECT id FROM articles ORDER BY id DESC LIMIT ?)
           AND source NOT LIKE 'backtest_run_%'
           AND datetime(replace(first_seen,'T',' ')) >= datetime('now', ?)
        """,
        (SCAN_LIMIT, cutoff),
    ).fetchone()[0]
    conn.close()

    ranked = sorted(rows, key=_score, reverse=True)[:TOP_N]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"=== DIGITAL INTERN DAILY DIGEST  {now} ===",
        f"Real articles {WINDOW_HOURS}h (excl backtest_run): {total_real:,}   urgent>=2: {len(rows)}",
        "-" * 60,
    ]
    if not ranked:
        lines.append("(no urgent articles in window)")
    for i, r in enumerate(ranked, 1):
        title = (r["title"] or "").strip().replace("\n", " ")[:90]
        src = r["source"] or "?"
        lines.append(
            f"{i}. [u{r['urgency']} score {_score(r):.3f}] {title}  <{src}>"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    return lines


def main():
    for ln in compute():
        print(ln)
    print(f"\noutput={OUT}")


if __name__ == "__main__":
    main()
