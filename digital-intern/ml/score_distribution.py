"""ml/score_distribution.py — read-only histogram + percentiles of ai_score.

Drift surface for the ArticleNet scorer. The trainer churns model_gpu.pt on
its own schedule (see ml/trainer.py), and `/api/ml-status` already exposes
``last_trained`` and ``val_loss``. What is **not** exposed is the live
*distribution* of scores the model is emitting — and a silent regression in
the scorer typically shows up as a shifted histogram (collapse to 0, blow-up
toward 10, bimodal pile-up) long before val_loss moves on the next retrain.

This module answers two questions read-only:

  1. What does the ai_score distribution look like over the last 24h?
  2. How does it compare to the 7-day baseline (mean / median shift)?

It opens articles.db in ``mode=ro`` directly — same pattern as db_health.py
and dashboard._ro_conn — so it can run alongside the daemon without touching
the writer or article_store's evolving API. The canonical live-only clause is
duplicated verbatim (not imported) for the same reason.

Performance: the underlying DB is multi-GB and USB-backed, so we do **not**
pull rows; histogram + mean are computed by GROUP BY in SQL, and percentiles
come from a bounded SAMPLE (LIMIT) of the rows in each window. The histogram
is exact; percentiles are approximate but stable.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

_USB_PATH = Path(os.environ.get("DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"))
_LOCAL_PATH = Path(__file__).resolve().parent.parent / "data"

# ai_score is on a 0..10 scale; bucket by integer floor, top bucket includes 10.
_BIN_LABELS = [
    "[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,5)",
    "[5,6)", "[6,7)", "[7,8)", "[8,9)", "[9,10]",
]

# Cap on rows read for percentile estimation. 5000 gives stable p50/p90/p99
# without pulling millions of rows from the USB-backed DB.
_PERCENTILE_SAMPLE_CAP = 5000


def resolve_db_path() -> Path:
    usb_db = _USB_PATH / "articles.db"
    if _USB_PATH.exists() and (usb_db.exists() or _USB_PATH.is_mount()):
        return usb_db
    return _LOCAL_PATH / "articles.db"


def open_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    # Short busy timeout: we never want this monitor to wait on the writer.
    conn.execute("PRAGMA busy_timeout=2000")
    return conn


def _percentile(sorted_vals: list[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _window_stats(conn: sqlite3.Connection, hours: int) -> dict:
    """Return count, mean, percentiles, and histogram for the last `hours`.

    histogram + count + mean: exact via GROUP BY in SQL.
    percentiles: approximate, computed from a bounded sample.
    """
    where = (
        f"ai_score > 0 AND first_seen >= datetime('now', '-{int(hours)} hours') "
        f"AND {LIVE_ONLY_CLAUSE}"
    )

    # Exact count + mean in one pass.
    row = conn.execute(
        f"SELECT COUNT(*), AVG(ai_score) FROM articles WHERE {where}"
    ).fetchone()
    n = int(row[0] or 0)
    mean = float(row[1]) if row[1] is not None else None

    # Exact histogram via integer-floor bucketing in SQL.
    hist_rows = conn.execute(
        "SELECT CAST(MIN(ai_score, 9.99) AS INTEGER) AS bucket, COUNT(*) "
        f"FROM articles WHERE {where} GROUP BY bucket"
    ).fetchall()
    counts = {int(b): int(c) for b, c in hist_rows}
    histogram = [
        {"bin": _BIN_LABELS[i], "count": counts.get(i, 0)} for i in range(10)
    ]

    # Approximate percentiles from a bounded sample.
    p50 = p90 = p99 = None
    if n > 0:
        sample = [
            float(r[0]) for r in conn.execute(
                f"SELECT ai_score FROM articles WHERE {where} "
                f"ORDER BY first_seen DESC LIMIT {_PERCENTILE_SAMPLE_CAP}"
            ).fetchall() if r[0] is not None
        ]
        sample.sort()
        p50 = round(_percentile(sample, 0.50) or 0.0, 3)
        p90 = round(_percentile(sample, 0.90) or 0.0, 3)
        p99 = round(_percentile(sample, 0.99) or 0.0, 3)

    return {
        "n": n,
        "mean": round(mean, 3) if mean is not None else None,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "histogram": histogram,
        "percentile_sample_size": min(n, _PERCENTILE_SAMPLE_CAP) if n else 0,
    }


def snapshot(db_path: Optional[Path] = None) -> dict:
    """Return a drift snapshot: 24h window + 7d baseline + delta + status.

    Status thresholds (based on mean shift between 24h and 7d windows):
      - ok:       |Δmean| < 0.5  AND  n_24h ≥ 50
      - degraded: 0.5 ≤ |Δmean| < 1.5
      - drift:    |Δmean| ≥ 1.5  OR  insufficient data
    """
    db_path = db_path or resolve_db_path()
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = open_ro(db_path)
        w24 = _window_stats(conn, 24)
        w7d = _window_stats(conn, 24 * 7)
    except sqlite3.Error as exc:
        return {"error": f"db open failed: {exc}", "db_path": str(db_path)}
    finally:
        if conn is not None:
            conn.close()
    delta: Optional[float] = None
    status = "drift"
    if w24["mean"] is not None and w7d["mean"] is not None:
        delta = round(w24["mean"] - w7d["mean"], 3)
        ad = abs(delta)
        if w24["n"] < 50:
            status = "drift"
        elif ad >= 1.5:
            status = "drift"
        elif ad >= 0.5:
            status = "degraded"
        else:
            status = "ok"
    return {
        "db_path": str(db_path),
        "window_24h": w24,
        "window_7d": w7d,
        "mean_delta": delta,
        "status": status,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(snapshot(), indent=2))
