"""Junk source detector.

Identifies collectors that flood the DB with near-identical titles (spam,
template-generated content, GDELT boilerplate). A source is flagged "junk"
when its title uniqueness ratio — unique truncated titles / total articles —
falls below JUNK_THRESHOLD.

Reads up to SCAN_LIMIT recent rows (idx_first_seen, no full-table scan).
Excludes backtest sources. Writes results to OUT_PATH; prints a summary.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/junk_source_report.json")

SCAN_LIMIT = 6000
MIN_PER_SOURCE = 10       # ignore sources with too few articles to judge
JUNK_THRESHOLD = 0.50     # uniqueness ratio below this = junk
TITLE_PREFIX_LEN = 70     # chars used to define "unique" title


def main() -> None:
    conn = sqlite3.connect(
        f"file:{DB}?mode=ro&immutable=1", uri=True
    )
    try:
        rows = conn.execute(
            f"SELECT source, title FROM articles "
            f"WHERE {_LIVE_ONLY_CLAUSE} "
            f"ORDER BY first_seen DESC LIMIT ?",
            (SCAN_LIMIT,),
        ).fetchall()
    finally:
        conn.close()

    # Aggregate per source
    source_totals: dict[str, int] = {}
    source_unique: dict[str, set] = {}
    for source, title in rows:
        src = source or "(unknown)"
        key = (title or "").lower()[:TITLE_PREFIX_LEN].strip()
        source_totals[src] = source_totals.get(src, 0) + 1
        source_unique.setdefault(src, set()).add(key)

    sources_report = []
    junk_sources = []
    for src, total in sorted(source_totals.items(), key=lambda x: -x[1]):
        if total < MIN_PER_SOURCE:
            continue
        unique_cnt = len(source_unique[src])
        ratio = unique_cnt / total
        entry = {
            "source": src,
            "total_articles": total,
            "unique_titles": unique_cnt,
            "uniqueness_ratio": round(ratio, 4),
            "is_junk": ratio < JUNK_THRESHOLD,
        }
        sources_report.append(entry)
        if ratio < JUNK_THRESHOLD:
            junk_sources.append(src)

    now = datetime.now(timezone.utc).isoformat()
    out = {
        "generated_at": now,
        "scan_limit": SCAN_LIMIT,
        "scanned": len(rows),
        "junk_threshold": JUNK_THRESHOLD,
        "junk_count": len(junk_sources),
        "junk_sources": junk_sources,
        "sources": sources_report,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print(
        f"junk_source_detector: scanned={len(rows)} sources_evaluated={len(sources_report)} "
        f"junk_count={len(junk_sources)}"
    )
    if junk_sources:
        print("  JUNK SOURCES (uniqueness_ratio < {:.0%}):".format(JUNK_THRESHOLD))
        for entry in sources_report:
            if entry["is_junk"]:
                print(
                    f"    {entry['source']}: ratio={entry['uniqueness_ratio']:.2%} "
                    f"({entry['unique_titles']}/{entry['total_articles']} unique titles)"
                )
    else:
        print("  No junk sources detected in scan window.")
    print(f"  Report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
