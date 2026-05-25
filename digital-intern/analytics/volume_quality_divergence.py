"""Volume-quality divergence detector.

Finds tickers where article *volume* and *ML score quality* diverge:

  volume_trap  — high mention count + low avg ml_score
                 (lots of noise/hype, low signal quality — avoid chasing)

  hidden_gem   — low mention count + high avg ml_score
                 (under-covered but substantive — worth attention)

For each ticker in the scan window, the module computes:
  * article count   → z-score across ticker population
  * avg ml_score    → z-score across ticker population

Divergence criteria (tunable via constants):
  * volume_trap:  z_count >= VOL_Z_THRESHOLD  AND z_score <= -SCORE_Z_THRESHOLD
  * hidden_gem:   z_count <= -VOL_Z_THRESHOLD AND z_score >=  SCORE_Z_THRESHOLD

Output: /home/zeph/logs/volume_quality_divergence.json
Standalone: python3 -m analytics.volume_quality_divergence
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analytics.trend_velocity import extract_tickers, _parse_ts
from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT_PATH = Path("/home/zeph/logs/volume_quality_divergence.json")

SCAN_WINDOW_HOURS = 6
FETCH_LIMIT = 6000
MIN_ARTICLES = 2          # skip tickers with < this many articles
VOL_HIGH_Z = 1.0          # z-score above which volume is "high" (volume trap)
VOL_LOW_PCTILE = 0.4      # count percentile below which volume is "low" (hidden gem)
SCORE_LOW_Z = -0.5        # z-score below which quality is "low" (volume trap)
SCORE_HIGH_Z = 0.6        # z-score above which quality is "high" (hidden gem)


def _fetch(conn: sqlite3.Connection) -> list[tuple[str, str, float | None]]:
    return conn.execute(
        f"SELECT first_seen, title, ml_score FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        f"ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    ).fetchall()


def _compute(rows: list[tuple[str, str, float | None]]) -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=SCAN_WINDOW_HOURS)

    counts: dict[str, int] = defaultdict(int)
    scores: dict[str, list[float]] = defaultdict(list)

    for first_seen, title, ml_score in rows:
        ts = _parse_ts(first_seen)
        if ts is None or ts < cutoff:
            continue
        for tk in set(extract_tickers(title)):
            counts[tk] += 1
            if ml_score is not None:
                val = float(ml_score)
                if math.isfinite(val):
                    scores[tk].append(val)

    tickers = [tk for tk, n in counts.items() if n >= MIN_ARTICLES]
    if not tickers:
        return {"volume_traps": [], "hidden_gems": [], "tickers_analyzed": 0}

    vol_vals = [counts[tk] for tk in tickers]
    score_avgs = {
        tk: (sum(scores[tk]) / len(scores[tk]) if scores[tk] else None)
        for tk in tickers
    }
    scored_tickers = [tk for tk in tickers if score_avgs[tk] is not None]
    if not scored_tickers:
        return {"volume_traps": [], "hidden_gems": [], "tickers_analyzed": len(tickers)}

    def _stats(vals: list[float]) -> tuple[float, float]:
        mean = sum(vals) / len(vals)
        if len(vals) < 2:
            return mean, 0.0
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return mean, math.sqrt(var)

    vol_mean, vol_std = _stats([float(counts[tk]) for tk in scored_tickers])
    sc_mean, sc_std = _stats([score_avgs[tk] for tk in scored_tickers])  # type: ignore[arg-type]

    def _z(val: float, mean: float, std: float) -> float:
        return (val - mean) / std if std > 0 else 0.0

    # For hidden-gem detection use a percentile cutoff on raw count (not z-score)
    # because the count distribution is right-skewed — a z < -1.2 is unreachable
    # when min_articles=2 and mean~5. Low-volume means bottom 40th percentile.
    sorted_counts = sorted(counts[tk] for tk in scored_tickers)
    n_sc = len(sorted_counts)
    vol_low_cutoff = sorted_counts[int(n_sc * VOL_LOW_PCTILE)] if n_sc else 0

    volume_traps: list[dict] = []
    hidden_gems: list[dict] = []

    for tk in scored_tickers:
        n = counts[tk]
        avg_sc = score_avgs[tk]
        z_vol = _z(float(n), vol_mean, vol_std)
        z_sc = _z(avg_sc, sc_mean, sc_std)  # type: ignore[arg-type]
        entry = {
            "ticker": tk,
            "article_count": n,
            "avg_ml_score": round(avg_sc, 2),  # type: ignore[arg-type]
            "z_volume": round(z_vol, 2),
            "z_score": round(z_sc, 2),
        }
        if z_vol >= VOL_HIGH_Z and z_sc <= SCORE_LOW_Z:
            volume_traps.append(entry)
        elif n <= vol_low_cutoff and z_sc >= SCORE_HIGH_Z:
            hidden_gems.append(entry)

    volume_traps.sort(key=lambda x: (-x["z_volume"], x["z_score"]))
    hidden_gems.sort(key=lambda x: (-x["z_score"], x["z_volume"]))

    return {
        "volume_traps": volume_traps,
        "hidden_gems": hidden_gems,
        "tickers_analyzed": len(scored_tickers),
        "vol_mean": round(vol_mean, 2),
        "vol_std": round(vol_std, 2),
        "score_mean": round(sc_mean, 2),
        "score_std": round(sc_std, 2),
    }


def main() -> int:
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    try:
        rows = _fetch(conn)
    finally:
        conn.close()

    result = _compute(rows)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["scan_window_hours"] = SCAN_WINDOW_HOURS
    result["fetch_limit"] = FETCH_LIMIT

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    payload = json.dumps(result, indent=2, allow_nan=False)
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(OUT_PATH)

    n_traps = len(result["volume_traps"])
    n_gems = len(result["hidden_gems"])
    n_total = result["tickers_analyzed"]
    print(
        f"volume_quality_divergence: {n_total} tickers | "
        f"volume_traps={n_traps} hidden_gems={n_gems} | "
        f"window={SCAN_WINDOW_HOURS}h"
    )
    if result["volume_traps"]:
        t = result["volume_traps"][0]
        print(f"  top trap: {t['ticker']} count={t['article_count']} avg_score={t['avg_ml_score']}")
    if result["hidden_gems"]:
        g = result["hidden_gems"][0]
        print(f"  top gem:  {g['ticker']} count={g['article_count']} avg_score={g['avg_ml_score']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
