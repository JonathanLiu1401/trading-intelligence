"""Source debut detector.

Surfaces collector source tags that appear in the last 24 h of ``articles.db``
but were ABSENT from the prior 7 days (days 2–8 back).  These are genuinely
new source tags — new feed configs, renamed collectors, or freshly-enabled
integrations — that warrant a credibility review before the credibility scorer
defaults them to ``DEFAULT_SOURCE_CRED=0.55``.

Operational value:
  * Catches ``GN: <topic>`` / ``YahooFinance/<symbol>`` style aggregator-prefix
    spellings that ``_PREFIX_ALIASES`` may not yet recognise.
  * Flags SEO-mill sources that somehow bypassed the junk-source gate and
    registered under a new domain / tag name.
  * Acts as a "new collector went live" announcement so the operator knows
    to check the ``source_credibility_audit`` output for the tag.

Design constraints:
  * No full-table COUNT(*); two bounded ``idx_first_seen`` scans (LIMIT each).
  * Read-only connection, ``_LIVE_ONLY_CLAUSE`` applied.
  * Busy-timeout 10 s — tolerate moderate WAL write contention.

Output: /home/zeph/logs/source_debut.json

Standalone::

    python3 -m analytics.source_debut
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT = Path("/home/zeph/logs/source_debut.json")

# How many rows to pull per window. At ~500 articles/h * 24h = ~12 000 rows
# typical; 20 000 gives comfortable headroom without OOM risk on the 14 GB box.
SCAN_RECENT = 20_000   # last 24 h
SCAN_BASELINE = 40_000 # prior 7 d (days 2–8); ~84 000 theoretical max but we
                       # cap to avoid USB-I/O stall — we only need the tag set,
                       # not every row.


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw[:19].replace("T", " ")).replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def compute() -> dict:
    now = datetime.now(timezone.utc)
    cutoff_recent = now - timedelta(hours=24)
    cutoff_baseline_end = cutoff_recent           # baseline ends where recent starts
    cutoff_baseline_start = now - timedelta(days=8)

    db_path = str(_get_db_path())
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=10000")

    # --- recent window: last 24 h ---
    recent_rows = conn.execute(
        f"SELECT source, first_seen FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        f"ORDER BY first_seen DESC LIMIT ?",
        (SCAN_RECENT,),
    ).fetchall()

    # --- baseline window: days 2–8 back ---
    # Use BETWEEN on the normalised timestamp string; idx_first_seen covers it.
    baseline_start_str = cutoff_baseline_start.strftime("%Y-%m-%d %H:%M:%S")
    baseline_end_str = cutoff_baseline_end.strftime("%Y-%m-%d %H:%M:%S")
    baseline_rows = conn.execute(
        f"SELECT DISTINCT source FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        f"  AND replace(first_seen,'T',' ') BETWEEN ? AND ? "
        f"LIMIT ?",
        (baseline_start_str, baseline_end_str, SCAN_BASELINE),
    ).fetchall()
    conn.close()

    # Build sets
    recent_sources: dict[str, int] = {}  # tag -> article count in last 24 h
    for src, ts_raw in recent_rows:
        ts = _parse_ts(ts_raw)
        if ts and ts >= cutoff_recent:
            recent_sources[src] = recent_sources.get(src, 0) + 1

    baseline_set: set[str] = {row[0] for row in baseline_rows if row[0]}

    # Debut = in recent but not in baseline
    debuts = {
        src: cnt
        for src, cnt in recent_sources.items()
        if src and src not in baseline_set
    }

    # Sort by article count descending so high-volume debuts surface first
    sorted_debuts = sorted(debuts.items(), key=lambda kv: kv[1], reverse=True)

    out = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_recent_hours": 24,
        "baseline_days": "2–8 before window",
        "recent_sources_seen": len(recent_sources),
        "baseline_sources_seen": len(baseline_set),
        "debut_count": len(debuts),
        "debuts": [
            {"source": src, "articles_24h": cnt}
            for src, cnt in sorted_debuts
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    result = compute()
    debut_count = result["debut_count"]
    print(
        f"source_debut: {debut_count} new source(s) | "
        f"recent_sources={result['recent_sources_seen']} | "
        f"baseline_sources={result['baseline_sources_seen']}"
    )
    for entry in result["debuts"][:10]:
        print(f"  DEBUT  {entry['source']!r:50s}  {entry['articles_24h']} arts/24h")
    if debut_count > 10:
        print(f"  ... and {debut_count - 10} more (see {OUT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
