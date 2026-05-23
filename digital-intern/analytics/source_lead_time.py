"""Source lead-time tracker.

For each source, measures how quickly it breaks high-urgency stories
(urgency >= 2) relative to when other sources first reported the same story
(identified by Jaccard title similarity).  A negative lead_minutes means the
source is faster than the median reporter on that cluster.

Why this matters: identifies the fastest "canary" feeds so the trading engine
can weight early signals from quick-twitch sources more heavily.

Design constraints (same as score_drift_detector):
  * Never full-table scan.  We read a bounded tail of recent articles via
    idx_first_seen and work entirely in Python memory.
  * DB is ~1.4 GB on USB with contention; busy_timeout=5000, LIMIT 6000.

Output: /home/zeph/logs/source_lead_time.json
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE

DB = BASE / "data" / "articles.db"
OUT = Path("/home/zeph/logs/source_lead_time.json")
SCAN_LIMIT = 8000
JACCARD_THRESH = 0.50
MIN_CLUSTER_SIZE = 2  # need at least 2 sources to compute lead time
MIN_URGENCY = 2  # cluster must have ≥1 article this urgent to be counted

# --- title normalisation (mirrors ml/dedup.py) ----------------------------

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "by",
     "at", "as", "is", "are", "be", "was", "were", "with", "from", "after",
     "over", "amid", "into", "its", "it", "that", "this", "than", "but"}
)
_WIRE_PREFIX = re.compile(
    r"^\s*(?:(?:UPDATE|RPT|CORRECTED|BREAKING|DEVELOPING|ALERT)\s*\d*\s*[-:]\s*)+",
    re.IGNORECASE,
)


def _tokens(title: str | None) -> frozenset[str]:
    if not title:
        return frozenset()
    s = _WIRE_PREFIX.sub("", title).strip().lower()
    return frozenset(t for t in _WORD.findall(s) if len(t) >= 2 and t not in _STOPWORDS)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


# --- clustering -----------------------------------------------------------

def _cluster(rows: list[tuple]) -> list[list[tuple]]:
    """Greedy single-pass clustering on title token sets."""
    clusters: list[dict] = []
    for row in rows:
        toks = _tokens(row[2])  # title at index 2
        placed = False
        if toks:
            for cl in clusters:
                if _jaccard(toks, cl["anchor"]) >= JACCARD_THRESH:
                    cl["rows"].append(row)
                    placed = True
                    break
        if not placed:
            clusters.append({"anchor": toks, "rows": [row]})
    return [cl["rows"] for cl in clusters]


# --- main -----------------------------------------------------------------

def compute() -> dict:
    # Plain ``mode=ro`` (no ``immutable=1``): the immutable flag promises
    # SQLite the file will never change, which on the actively-written
    # production ``articles.db`` causes intermittent "database disk image is
    # malformed" errors (commit ``cdd8d4a`` fixed the same hazard in
    # score_drift_detector / source_score_drift; this script was missed).
    # ``mode=ro`` alone gives the read-only guarantee without the malformed-DB
    # risk.
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=15)
    try:
        raw = conn.execute(
            "SELECT id, source, title, urgency, first_seen "
            "FROM articles ORDER BY first_seen DESC LIMIT ?",
            (SCAN_LIMIT,),
        ).fetchall()
    finally:
        conn.close()

    # Filter out backtest/annotation rows in Python (avoids killing the index).
    rows = [
        r for r in raw
        if r[1]
        and not r[1].startswith("backtest_")
        and not r[1].startswith("opus_annotation")
        and (r[4] or "").startswith("backtest://") is False  # url check via source is enough
    ]

    if not rows:
        return {"error": "no articles found", "scan_limit": SCAN_LIMIT}

    clusters = _cluster(rows)
    # Keep only clusters with ≥2 members and ≥1 urgent article
    multi = [
        c for c in clusters
        if len(c) >= MIN_CLUSTER_SIZE
        and any(row[3] is not None and row[3] >= MIN_URGENCY for row in c)
    ]

    # Per-source: collect lead-time deltas (seconds) vs cluster-median first_seen
    lead_deltas: dict[str, list[float]] = defaultdict(list)

    for cluster in multi:
        times: list[tuple[float, str]] = []  # (epoch, source)
        for _id, source, _title, _urg, first_seen in cluster:
            try:
                fs = first_seen.replace("T", " ").split(".")[0]
                dt = datetime.strptime(fs, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                times.append((dt.timestamp(), source))
            except (ValueError, AttributeError):
                continue
        if len(times) < MIN_CLUSTER_SIZE:
            continue
        med = median(t for t, _ in times)
        for ts, src in times:
            lead_deltas[src].append(ts - med)  # negative = faster than median

    # Aggregate per source
    source_stats: list[dict] = []
    for src, deltas in sorted(lead_deltas.items(), key=lambda x: sum(x[1]) / len(x[1])):
        avg_lead = sum(deltas) / len(deltas)
        source_stats.append({
            "source": src,
            "story_count": len(deltas),
            "avg_lead_minutes": round(avg_lead / 60, 2),
            "median_lead_minutes": round(median(deltas) / 60, 2),
            "fastest_ever_minutes": round(min(deltas) / 60, 2),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_limit": SCAN_LIMIT,
        "articles_scanned": len(rows),
        "story_clusters_with_urgency": len(multi),
        "sources_ranked": source_stats,
    }


if __name__ == "__main__":
    result = compute()
    OUT.write_text(json.dumps(result, indent=2))
    print(f"Scanned: {result.get('articles_scanned', 0)}, "
          f"Clusters: {result.get('story_clusters_with_urgency', 0)}, "
          f"Sources ranked: {len(result.get('sources_ranked', []))}")
    top = result.get("sources_ranked", [])[:5]
    for s in top:
        print(f"  {s['source']}: avg lead {s['avg_lead_minutes']}m "
              f"({s['story_count']} stories)")
