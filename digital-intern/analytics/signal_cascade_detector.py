"""Signal cascade detector: flags when 3+ distinct macro narrative clusters surge together.

``keyword_surge`` tracks individual keyword lift over a 1h vs 23h baseline.
``ingest_spike_detector`` watches total pipeline volume.

This module fills the gap between them: it tracks *cross-topic co-movement*.
When multiple distinct macro narrative clusters (rates, geopolitics, credit,
energy, etc.) all show surging article volume within the **same 30-minute
bucket**, that signals a systemic event — not a single-ticker story, not a
feed anomaly, but a genuine macro cascade where many topics move together.

Examples of true cascade events this would catch:
  * Bond yields spike + banking stress + dollar surge → financial-system shock
  * Oil surge + geopolitical conflict + sanctions + currency move → macro supply shock
  * Earnings miss + revenue guidance + sector rotation + credit stress → broad risk-off

Design constraints:
  * Single bounded scan via idx_first_seen (SCAN_LIMIT rows).
  * No full-table scans; all aggregation in Python.
  * Read-only SQLite connection, busy_timeout=12s.
  * Appends detected events to a JSONL log for historical record.

Output:
  /home/zeph/logs/signal_cascade.json    — latest snapshot (machine-readable)
  /home/zeph/logs/signal_cascade.jsonl   — append-only event history
Standalone: python3 -m analytics.signal_cascade_detector
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

try:
    from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path
    DB_PATH = _get_db_path()
except Exception:
    _LIVE_ONLY_CLAUSE = (
        "url NOT LIKE 'backtest://%' "
        "AND source NOT LIKE 'backtest_%' "
        "AND source NOT LIKE 'opus_annotation%'"
    )
    DB_PATH = BASE / "data" / "articles.db"

SNAPSHOT_PATH = Path("/home/zeph/logs/signal_cascade.json")
EVENTS_PATH   = Path("/home/zeph/logs/signal_cascade.jsonl")

# How far back to scan articles for cascade detection
LOOKBACK_HOURS  = 3
# Bucket width in minutes — cascade events must co-occur within a bucket
BUCKET_MINUTES  = 30
# Minimum lift (vs baseline hour) for a cluster to count as "elevated" (informational)
LIFT_THRESHOLD  = 1.5
# Minimum article count in a bucket to count a cluster as "active".
# Macro topics always have *some* coverage, so we need a meaningful floor.
# Calibrated so a cascade fires only when multiple clusters are HEAVILY covered
# in the same window — not just mentioned once or twice.
MIN_BUCKET_HITS = 7
# Number of simultaneously-active clusters that constitutes a cascade.
# 5 of 9 tracked clusters showing heavy coverage in the same 30 min = systemic event.
CASCADE_MIN_CLUSTERS = 5
# Rows to fetch from the DB (bounded scan)
SCAN_LIMIT = 6000

# ---------------------------------------------------------------------------
# Macro narrative clusters: label -> set of signal keywords
# Each keyword must appear in a headline title for that cluster to score a hit.
# Clusters are intentionally broad so a single-topic event (one cluster surging)
# does NOT trigger; only true multi-topic co-movement does.
# ---------------------------------------------------------------------------
CLUSTERS: dict[str, set[str]] = {
    "rates_bonds": {
        "yield", "yields", "treasury", "treasuries", "bond", "bonds",
        "rate", "rates", "fomc", "federal", "inflation", "cpi", "pce",
        "bps", "basis", "hike", "pivot", "taper",
    },
    "geopolitical": {
        "war", "conflict", "missile", "attack", "military", "sanctions",
        "tariff", "tariffs", "trade", "border", "invasion", "nato",
        "iran", "russia", "china", "taiwan", "ukraine", "korea",
        "nuclear", "strike", "strikes",
    },
    "credit_banking": {
        "bank", "banks", "banking", "credit", "default", "defaults",
        "bankruptcy", "debt", "bailout", "systemic", "contagion",
        "insolvency", "loans", "deposit", "deposits", "svb", "cre",
    },
    "energy_commodities": {
        "oil", "crude", "opec", "energy", "natural", "gasoline",
        "coal", "copper", "gold", "silver", "commodities",
        "pipeline", "refinery", "brent", "wti",
    },
    "tech_ai": {
        "nvidia", "semiconductor", "chips", "artificial", "intelligence",
        "openai", "microsoft", "google", "alphabet", "meta", "apple",
        "computing", "cloud", "datacenter", "hyperscaler",
    },
    "currency_dollar": {
        "dollar", "euro", "yuan", "yen", "forex", "currency", "currencies",
        "sterling", "pound", "devaluation", "exchange", "dxy",
    },
    "crypto": {
        "bitcoin", "ethereum", "crypto", "blockchain", "defi",
        "stablecoin", "altcoin", "solana", "binance", "coinbase",
    },
    "earnings_guidance": {
        "beats", "misses", "guidance", "outlook", "forecast",
        "revenue", "margin", "margins", "quarter", "quarterly",
        "beat", "miss", "raised", "lowered", "withdrawal",
    },
    "employment_macro": {
        "jobs", "unemployment", "payroll", "payrolls", "layoffs",
        "hiring", "recession", "gdp", "growth", "slowdown",
        "workforce", "wages", "labor", "labour",
    },
}

_WORD_RE = re.compile(r"\b([a-z]{3,})\b")


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _bucket_key(ts: datetime, bucket_minutes: int) -> str:
    """Round down to the nearest bucket boundary and return ISO string."""
    minutes = (ts.hour * 60 + ts.minute) // bucket_minutes * bucket_minutes
    snapped = ts.replace(
        hour=minutes // 60,
        minute=minutes % 60,
        second=0,
        microsecond=0,
    )
    return snapped.isoformat()


def _cluster_hits(title: str) -> set[str]:
    """Return the set of cluster labels this title contributes to."""
    words = set(_WORD_RE.findall((title or "").lower()))
    matched: set[str] = set()
    for label, keywords in CLUSTERS.items():
        if words & keywords:
            matched.add(label)
    return matched


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=12)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=12000")
    cur = conn.execute(
        f"SELECT first_seen, title FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        f"ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("signal_cascade: no rows", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    lookback_cut = now - timedelta(hours=LOOKBACK_HOURS)
    # 1h baseline window just before the lookback (for per-cluster baseline rate)
    baseline_cut = lookback_cut - timedelta(hours=1)

    # bucket -> cluster label -> hit count
    bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
    # baseline cluster hit counts (1h window before lookback)
    baseline_counts: Counter[str] = Counter()

    n_recent = n_base = 0
    for fs, title in rows:
        ts = _parse_ts(fs)
        if ts is None:
            continue
        clusters = _cluster_hits(title)
        if not clusters:
            continue
        if ts >= lookback_cut:
            bk = _bucket_key(ts, BUCKET_MINUTES)
            for c in clusters:
                bucket_counts[bk][c] += 1
            n_recent += 1
        elif ts >= baseline_cut:
            for c in clusters:
                baseline_counts[c] += 1
            n_base += 1

    # Baseline hourly rate per cluster (articles/hr, scaled to bucket size)
    bucket_scale = BUCKET_MINUTES / 60.0
    baseline_rate: dict[str, float] = {
        c: (baseline_counts.get(c, 0)) * bucket_scale
        for c in CLUSTERS
    }

    # Detect cascade events: buckets where CASCADE_MIN_CLUSTERS+ clusters are active.
    # "Active" = hits >= MIN_BUCKET_HITS (absolute threshold — macro topics are always
    # present in finance news so relative lift alone is a poor discriminator).
    # Lift vs baseline is computed as supplemental metadata per cluster.
    cascades: list[dict] = []
    for bk in sorted(bucket_counts):
        surging_clusters: list[dict] = []
        for label, cnt in bucket_counts[bk].items():
            if cnt < MIN_BUCKET_HITS:
                continue
            expected = baseline_rate.get(label, 0.0)
            lift = round((cnt + 1) / (expected + 1), 2)
            surging_clusters.append({
                "cluster": label,
                "hits": cnt,
                "expected": round(expected, 1),
                "lift": lift,
                "elevated": lift >= LIFT_THRESHOLD,
            })
        if len(surging_clusters) >= CASCADE_MIN_CLUSTERS:
            surging_clusters.sort(key=lambda x: x["lift"], reverse=True)
            cascades.append({
                "bucket_start": bk,
                "surging_cluster_count": len(surging_clusters),
                "clusters": surging_clusters,
            })

    payload = {
        "generated_at": now.isoformat(),
        "scan_limit": SCAN_LIMIT,
        "articles_scanned": len(rows),
        "lookback_hours": LOOKBACK_HOURS,
        "bucket_minutes": BUCKET_MINUTES,
        "lift_threshold": LIFT_THRESHOLD,
        "cascade_min_clusters": CASCADE_MIN_CLUSTERS,
        "articles_in_window": n_recent,
        "articles_in_baseline": n_base,
        "cascade_events": cascades,
        "cascade_count": len(cascades),
    }

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SNAPSHOT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(SNAPSHOT_PATH)

    # Append new cascade events to JSONL log (dedup by bucket_start + cluster set)
    existing_keys: set[str] = set()
    if EVENTS_PATH.exists():
        for line in EVENTS_PATH.read_text().splitlines():
            try:
                ev = json.loads(line)
                existing_keys.add(ev.get("bucket_start", ""))
            except json.JSONDecodeError:
                pass

    new_events = 0
    with EVENTS_PATH.open("a") as f:
        for ev in cascades:
            if ev["bucket_start"] not in existing_keys:
                f.write(json.dumps({**ev, "logged_at": now.isoformat()}) + "\n")
                new_events += 1

    # Output summary
    print(
        f"signal_cascade: scanned={len(rows)} window={n_recent} "
        f"baseline={n_base} cascades={len(cascades)} new_logged={new_events}"
    )
    if cascades:
        for ev in cascades:
            cluster_names = ", ".join(c["cluster"] for c in ev["clusters"])
            print(f"  CASCADE @ {ev['bucket_start']}: {ev['surging_cluster_count']} clusters [{cluster_names}]")
    else:
        print("  (no cascade events in lookback window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
