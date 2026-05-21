"""Collector throughput trend detector.

Compares each collector's article rate in the last 2 h against its own
baseline rate over the prior 4 h.  A source still publishing (so
``stale_source_alerter`` and ``collector_uptime`` won't catch it) but
producing at <50% of its own recent baseline is flagged as *degraded* —
the classic symptom of API throttling, slow site, or partial auth failure.

Operational gap filled: gap detectors require total silence; this catches
slow-drain degradations before they become full outages.

Design constraints (workspace memory):
  * No full COUNT(*). Single bounded ``idx_first_seen`` scan (LIMIT 15 k).
  * ``_LIVE_ONLY_CLAUSE`` applied — backtest rows excluded.
  * Read-only connection, PRAGMA busy_timeout=8 000 ms.

Output: /home/zeph/logs/collector_throughput_trend.json
Standalone: python3 -m analytics.collector_throughput_trend
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT = Path("/home/zeph/logs/collector_throughput_trend.json")
SCAN_LIMIT = 15_000
RECENT_HOURS = 2
BASELINE_HOURS = 4   # 2h..6h ago — normalised to per-2h rate for comparison
DROP_THRESHOLD = 0.50  # flag if recent/baseline_2h_equiv < 50%
MIN_BASELINE = 5       # skip sources with too few baseline articles (sparse feeds)


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    s = str(ts).replace("T", " ").split("+")[0][:19]
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def compute() -> dict:
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(hours=RECENT_HOURS)
    baseline_start = now - timedelta(hours=RECENT_HOURS + BASELINE_HOURS)

    db_path = str(_get_db_path())
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    conn.execute("PRAGMA busy_timeout=8000")

    rows = conn.execute(
        f"SELECT source, first_seen FROM articles INDEXED BY idx_first_seen "
        f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
        f"ORDER BY first_seen DESC LIMIT ?",
        (baseline_start.strftime("%Y-%m-%d %H:%M:%S"), SCAN_LIMIT),
    ).fetchall()
    conn.close()

    recent_counts: dict[str, int] = defaultdict(int)
    baseline_counts: dict[str, int] = defaultdict(int)

    for source, first_seen in rows:
        ts = _parse_ts(first_seen)
        if ts is None:
            continue
        if ts >= recent_cutoff:
            recent_counts[source] += 1
        else:
            baseline_counts[source] += 1

    results = []
    for source, baseline_raw in baseline_counts.items():
        if baseline_raw < MIN_BASELINE:
            continue
        # Normalise 4h baseline → per-2h equivalent
        baseline_2h = baseline_raw / (BASELINE_HOURS / RECENT_HOURS)
        recent = recent_counts.get(source, 0)
        ratio = recent / baseline_2h if baseline_2h > 0 else 1.0
        results.append({
            "source": source,
            "recent_2h": recent,
            "baseline_2h_equiv": round(baseline_2h, 1),
            "ratio": round(ratio, 3),
            "degraded": ratio < DROP_THRESHOLD,
        })

    results.sort(key=lambda r: (not r["degraded"], r["ratio"]))
    degraded = [r for r in results if r["degraded"]]

    out = {
        "generated_at": now.isoformat(),
        "recent_hours": RECENT_HOURS,
        "baseline_hours": BASELINE_HOURS,
        "drop_threshold": DROP_THRESHOLD,
        "scan_limit": SCAN_LIMIT,
        "scanned_rows": len(rows),
        "sources_analysed": len(results),
        "degraded_count": len(degraded),
        "degraded": degraded[:20],
        "top_healthy": [r for r in results if not r["degraded"]][:10],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    rep = compute()
    print(
        f"scanned {rep['scanned_rows']} rows | "
        f"sources: {rep['sources_analysed']} | "
        f"degraded: {rep['degraded_count']}"
    )
    for r in rep["degraded"][:5]:
        print(
            f"  DEGRADED {r['source']}: "
            f"recent={r['recent_2h']} baseline={r['baseline_2h_equiv']:.1f} "
            f"ratio={r['ratio']:.2f}"
        )
    if not rep["degraded"]:
        print("  All monitored sources at normal throughput")
    sys.exit(0)
