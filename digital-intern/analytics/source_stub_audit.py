#!/usr/bin/env python3
"""Stub article detector.

A "stub" is an article whose ``full_text`` decompresses to fewer than
STUB_THRESHOLD characters — typically a paywall teaser, empty RSS entry,
or scraped redirect page.  High stub rates waste ML scoring cycles on
content-free rows and dilute source quality metrics.

For each live source this reports:
  * total articles scanned
  * stub count / stub_pct
  * avg decompressed text length
  * flag: stub_heavy (stub_pct >= 50%)

Design: bounded LIMIT scan, read-only connection, zlib-safe (skips
full_text=NULL and non-zlib blobs without crashing).

Output: /home/zeph/logs/source_stub_audit.json
Standalone: ``python3 -m analytics.source_stub_audit``
"""
from __future__ import annotations

import json
import sqlite3
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/source_stub_audit.json")

SCAN_LIMIT = 6000
STUB_THRESHOLD = 150   # chars after decompression
HEAVY_STUB_PCT = 50.0  # sources above this are "stub_heavy"
MIN_ARTICLES = 3       # ignore sources with too few samples


def _decompress_safe(blob: bytes | None) -> int:
    """Return decompressed byte length, or 0 on error/NULL."""
    if not blob:
        return 0
    try:
        return len(zlib.decompress(blob))
    except Exception:
        return len(blob)  # not zlib — treat raw length as proxy


def compute() -> dict:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=10)
    try:
        rows = conn.execute(
            """
            SELECT source, full_text
            FROM articles
            WHERE source NOT LIKE 'backtest%'
              AND source NOT LIKE 'opus_annotation%'
              AND source NOT LIKE 'backtest_run%'
              AND url NOT LIKE 'backtest://%'
            ORDER BY first_seen DESC
            LIMIT ?
            """,
            (SCAN_LIMIT,),
        ).fetchall()
    finally:
        conn.close()

    buckets: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "stubs": 0, "text_lens": []}
    )

    for source, full_text in rows:
        length = _decompress_safe(full_text)
        b = buckets[source]
        b["total"] += 1
        if length < STUB_THRESHOLD:
            b["stubs"] += 1
        b["text_lens"].append(length)

    sources_out = {}
    stub_heavy: list[str] = []

    for src, b in sorted(buckets.items()):
        if b["total"] < MIN_ARTICLES:
            continue
        stub_pct = round(100.0 * b["stubs"] / b["total"], 1)
        avg_len = round(sum(b["text_lens"]) / len(b["text_lens"]))
        heavy = stub_pct >= HEAVY_STUB_PCT
        sources_out[src] = {
            "total": b["total"],
            "stubs": b["stubs"],
            "stub_pct": stub_pct,
            "avg_text_len": avg_len,
            "stub_heavy": heavy,
        }
        if heavy:
            stub_heavy.append(src)

    # Sort output by stub_pct descending
    sources_out = dict(
        sorted(sources_out.items(), key=lambda kv: kv[1]["stub_pct"], reverse=True)
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_limit": SCAN_LIMIT,
        "scanned": len(rows),
        "stub_threshold_chars": STUB_THRESHOLD,
        "sources_audited": len(sources_out),
        "stub_heavy_count": len(stub_heavy),
        "stub_heavy_sources": stub_heavy,
        "sources": sources_out,
    }


def main() -> None:
    result = compute()
    OUT_PATH.write_text(json.dumps(result, indent=2))
    heavy = result["stub_heavy_count"]
    total = result["sources_audited"]
    print(
        f"source_stub_audit: scanned={result['scanned']} sources={total} "
        f"stub_heavy={heavy} ({100*heavy//max(total,1)}%)"
    )
    if result["stub_heavy_sources"]:
        top3 = result["stub_heavy_sources"][:3]
        for src in top3:
            s = result["sources"][src]
            print(
                f"  STUB-HEAVY: {src} — {s['stub_pct']}% stubs "
                f"avg_len={s['avg_text_len']}chars n={s['total']}"
            )
    print(f"written → {OUT_PATH}")


if __name__ == "__main__":
    main()
