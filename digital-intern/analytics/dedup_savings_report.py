"""Near-dedup savings report.

Fetches a bounded recent window of live articles and runs the Jaccard
near-dedup engine (``ml.dedup``) to estimate how many rows would be
collapsed if dedup were applied at ingestion time.  Outputs per-source
savings stats and top example duplicate clusters to
``/home/zeph/logs/dedup_savings_report.json``.

``ml.dedup`` is a pure, DB-free module explicitly designed for this kind
of caller-side batch analysis — this is the realisation of that "future
ingestion-side collapse" mentioned in its docstring.

Design constraints:
  * Bounded SCAN_LIMIT idx_first_seen scan — no full-table scan.
  * Read-only sqlite URI, busy_timeout 10 000 ms.
  * _LIVE_ONLY_CLAUSE applied — no backtest rows.
  * Groups articles into 30-min ingestion slots before deduping; within
    a slot a story can travel across all live feeds, which is the real
    unit of syndication risk.

Standalone: ``python3 -m analytics.dedup_savings_report``
Output:     /home/zeph/logs/dedup_savings_report.json
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from ml.dedup import dedupe_articles, jaccard_similarity, title_tokens
from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT_PATH = Path("/home/zeph/logs/dedup_savings_report.json")
SCAN_LIMIT = 3000
SLOT_MINUTES = 30
TOP_CLUSTERS = 5
JACCARD_THRESHOLD = 0.6


def _slot_key(first_seen: str) -> str:
    """Round first_seen ISO timestamp down to the nearest SLOT_MINUTES bucket."""
    try:
        s = first_seen.replace("T", " ")[:19]
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        slot_min = (dt.minute // SLOT_MINUTES) * SLOT_MINUTES
        return dt.strftime(f"%Y-%m-%d %H:{slot_min:02d}")
    except (ValueError, TypeError):
        return "unknown"


def _find_clusters(articles: list[dict]) -> list[dict]:
    """Return duplicate clusters (size >= 2) from a list of article dicts."""
    clusters: list[dict] = []
    assigned = [False] * len(articles)
    toks_cache = [title_tokens(a.get("title")) for a in articles]

    for i, art_i in enumerate(articles):
        if assigned[i] or not toks_cache[i]:
            continue
        cluster_idxs = [i]
        for j in range(i + 1, len(articles)):
            if assigned[j] or not toks_cache[j]:
                continue
            sim = jaccard_similarity(toks_cache[i], toks_cache[j])
            if sim >= JACCARD_THRESHOLD:
                cluster_idxs.append(j)
                assigned[j] = True
        if len(cluster_idxs) >= 2:
            assigned[i] = True
            clusters.append(
                {
                    "size": len(cluster_idxs),
                    "titles": [articles[k]["title"] for k in cluster_idxs[:4]],
                    "sources": list({articles[k]["source"] for k in cluster_idxs}),
                }
            )
    return clusters


def main() -> int:
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    conn.execute("PRAGMA query_only=ON")

    rows = conn.execute(
        f"SELECT title, source, ai_score, first_seen "
        f"FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        f"ORDER BY first_seen DESC LIMIT {SCAN_LIMIT}"
    ).fetchall()
    conn.close()

    if not rows:
        print("dedup_savings_report: no rows fetched", file=sys.stderr)
        return 1

    # Group into time-slot buckets
    slots: dict[str, list[dict]] = defaultdict(list)
    for title, source, ai_score, first_seen in rows:
        key = _slot_key(first_seen or "")
        slots[key].append({"title": title, "source": source, "ai_score": ai_score or 0.0})

    total_in = 0
    total_out = 0
    source_in: dict[str, int] = defaultdict(int)
    source_out: dict[str, int] = defaultdict(int)
    all_clusters: list[dict] = []

    for slot_key, arts in slots.items():
        total_in += len(arts)
        for a in arts:
            source_in[a["source"]] += 1

        survived = dedupe_articles(arts, threshold=JACCARD_THRESHOLD)
        total_out += len(survived)
        for a in survived:
            source_out[a["source"]] += 1

        clusters = _find_clusters(arts)
        for cl in clusters:
            cl["slot"] = slot_key
        all_clusters.extend(clusters)

    savings = total_in - total_out
    savings_pct = round(100 * savings / total_in, 1) if total_in else 0.0

    # Per-source savings table (only sources with any collapsible rows)
    per_source = []
    for src in sorted(source_in, key=lambda s: source_in[s] - source_out.get(s, source_in[s]), reverse=True):
        s_in = source_in[src]
        s_out = source_out.get(src, s_in)
        s_saved = s_in - s_out
        if s_saved > 0:
            per_source.append(
                {
                    "source": src,
                    "ingested": s_in,
                    "would_survive": s_out,
                    "saved": s_saved,
                    "savings_pct": round(100 * s_saved / s_in, 1),
                }
            )

    # Top clusters by size
    top_clusters = sorted(all_clusters, key=lambda c: c["size"], reverse=True)[:TOP_CLUSTERS]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scanned": total_in,
        "slot_minutes": SLOT_MINUTES,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "total_in": total_in,
        "total_would_survive": total_out,
        "savings": savings,
        "savings_pct": savings_pct,
        "sources_with_savings": len(per_source),
        "per_source_top": per_source[:15],
        "example_clusters": top_clusters,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))

    print(
        f"dedup_savings_report: scanned={total_in} "
        f"savings={savings} ({savings_pct}%) "
        f"sources_with_dupes={len(per_source)}"
    )
    if top_clusters:
        cl = top_clusters[0]
        print(
            f"  biggest cluster: size={cl['size']} slot={cl['slot']} "
            f"sources={cl['sources']}"
        )
        for t in cl["titles"][:2]:
            print(f"    '{t}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
