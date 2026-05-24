"""News fatigue detector: high-volume tickers where average score is declining.

A ticker is "fatigued" when it has accumulated many articles over the last
24 hours but its average score in the most-recent 6-hour window is
meaningfully lower than the prior 18-hour window.  This pattern signals a
story being "burned out" — heavy coverage that the market has already
digested, so incremental articles are generating less urgency than the
initial wave.

Operator use: avoid chasing fatigued tickers as if they are breaking news.
The story is old; wait for a new catalyst before treating fresh articles
on that ticker as high-priority.

UNIFIED SCORE (the bug this module's refactor fixes): the prior
implementation read ``ai_score`` only, which excludes ~92% of the live
scored corpus — every ML-only row carries ``score_source='ml'``,
``ml_score`` set (e.g. 9.73 on a model-confident urgent), and
``ai_score = 0``. A 2026-05-24 live audit showed: of 4915 scored rows in
the last 24h only 375 had ``ai_score > 0`` — the analyzer was operating
on 7.6% of the signal. With most ML-only rows contributing 0 to the mean
by construction, the recent-vs-prior delta was dominated by which window
happened to catch the rare LLM-vetted rows and could not surface real
fatigue. The fix is the same ``COALESCE(NULLIF(ai_score, 0), ml_score)``
unified read the rest of the codebase already standardises on
(see storage/article_store.py docstring on ``update_ml_scores_batch``).

Thresholds:
  MIN_TOTAL_24H   = 15   articles in last 24h to qualify
  MIN_RECENT_6H   = 3    articles in last 6h (ticker must still be active)
  FATIGUE_DROP    = 1.5  score points: recent_mean < prior_mean - 1.5
  SCAN_LIMIT      = 12000 rows (covers ~24h at typical ingest rate)

Output: /home/zeph/logs/news_fatigue.json
Standalone: ``python3 -m analytics.news_fatigue``
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from analytics.trend_velocity import TICKER_RE, STOP, _parse_ts
from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT_PATH = Path("/home/zeph/logs/news_fatigue.json")

SCAN_LIMIT = 12_000
TOTAL_WINDOW_HOURS = 24
RECENT_HOURS = 6       # "is the story still fresh?" window
PRIOR_HOURS = 18       # baseline window (hours 6-24 ago)
MIN_TOTAL_24H = 15     # minimum total mentions to qualify
MIN_RECENT_6H = 3      # must still have recent coverage (not dead)
FATIGUE_DROP = 1.5     # score drop to call fatigue
TOP_N = 10


def _extract_tickers(title: str) -> list[str]:
    out: list[str] = []
    for m in TICKER_RE.findall(title or ""):
        if m not in STOP and len(m) >= 2:
            out.append(m)
    return out


def _unified_score(ai_score, ml_score) -> Optional[float]:
    """``COALESCE(NULLIF(ai_score, 0), ml_score)`` — the canonical unified
    read. Returns the LLM ground-truth score if non-zero, else the model
    prediction, else None (row contributes nothing to the mean).

    A row with ``ai_score=0 AND ml_score IS NULL`` returns None — those
    are unscored rows that should not participate in either window's
    mean. Without this guard the prior implementation's ``ai_score``-only
    read was actually treating every ML-only urgent row as a zero-valued
    sample, dragging the mean down and creating phantom fatigue signal.
    """
    try:
        ai = float(ai_score) if ai_score is not None else 0.0
    except (TypeError, ValueError):
        ai = 0.0
    if ai > 0.0:
        return ai
    if ml_score is None:
        return None
    try:
        ml = float(ml_score)
    except (TypeError, ValueError):
        return None
    return ml


def compute_news_fatigue(
    rows: Iterable[tuple],
    now: Optional[datetime] = None,
    min_total_24h: int = MIN_TOTAL_24H,
    min_recent_6h: int = MIN_RECENT_6H,
    fatigue_drop: float = FATIGUE_DROP,
    top_n: int = TOP_N,
) -> dict:
    """Pure aggregator — no DB / IO. Detects fatigued tickers.

    ``rows`` is an iterable of ``(first_seen, title, ai_score, ml_score)``
    tuples — the same shape ``main()`` pulls from articles.db. The pure
    surface lets the analytics test suite assert specific output values
    without spinning up a sqlite fixture (same discipline as
    ``analytics.held_ticker_news_silence.compute_silence`` and
    ``analytics.held_alert_reaction_latency.compute_held_alert_reaction_latency``).

    Window boundaries are anchored to ``now`` (default ``datetime.now(UTC)``)
    so tests can freeze a clock. A row whose ``first_seen`` is older than
    ``TOTAL_WINDOW_HOURS`` is silently dropped; rows older than ``now``
    by less than ``RECENT_HOURS`` count toward the recent window. A row
    with an unparseable timestamp is dropped (the same row-by-row tolerance
    as ``held_ticker_news_silence``).

    Returns the same shape as the previous ``main()`` payload minus
    ``scanned_rows`` (the caller adds that):

    .. code-block:: python

        {
          "generated_at":           iso timestamp,
          "fatigue_threshold_drop": float,
          "min_total_24h":          int,
          "min_recent_6h":          int,
          "fatigued_count":         int,    # total fatigued before top_n cap
          "tickers": [
            {
              "ticker":            str,
              "total_24h":         int,
              "recent_6h_count":   int,
              "prior_18h_count":   int,
              "recent_mean_score": float (2dp),
              "prior_mean_score":  float (2dp),
              "score_drop":        float (2dp),
            },
            ...
          ],
        }
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    recent_cut = now - timedelta(hours=RECENT_HOURS)
    prior_cut = now - timedelta(hours=TOTAL_WINDOW_HOURS)

    recent_scores: dict[str, list[float]] = defaultdict(list)
    prior_scores: dict[str, list[float]] = defaultdict(list)

    for r in rows:
        if len(r) < 4:
            continue
        fs, title, ai_score, ml_score = r[0], r[1], r[2], r[3]
        ts = _parse_ts(fs)
        if ts is None or ts < prior_cut:
            continue
        tickers = _extract_tickers(title or "")
        if not tickers:
            continue
        score = _unified_score(ai_score, ml_score)
        if score is None:
            continue
        score = float(score)
        bucket = recent_scores if ts >= recent_cut else prior_scores
        for tk in tickers:
            bucket[tk].append(score)

    fatigued: list[dict] = []
    all_tickers = set(recent_scores) | set(prior_scores)
    for tk in all_tickers:
        recent = recent_scores.get(tk, [])
        prior = prior_scores.get(tk, [])
        total = len(recent) + len(prior)
        if total < min_total_24h:
            continue
        if len(recent) < min_recent_6h:
            continue
        if not prior:
            continue
        recent_mean = mean(recent)
        prior_mean = mean(prior)
        drop = prior_mean - recent_mean
        if drop >= fatigue_drop:
            fatigued.append({
                "ticker": tk,
                "total_24h": total,
                "recent_6h_count": len(recent),
                "prior_18h_count": len(prior),
                "recent_mean_score": round(recent_mean, 2),
                "prior_mean_score": round(prior_mean, 2),
                "score_drop": round(drop, 2),
            })

    fatigued.sort(key=lambda r: r["score_drop"], reverse=True)
    return {
        "generated_at": now.isoformat(),
        "fatigue_threshold_drop": fatigue_drop,
        "min_total_24h": min_total_24h,
        "min_recent_6h": min_recent_6h,
        "fatigued_count": len(fatigued),
        "tickers": fatigued[:top_n] if top_n else fatigued,
    }


def main() -> int:
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    rows = conn.execute(
        "SELECT first_seen, title, ai_score, ml_score FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    if not rows:
        print("news_fatigue: no rows", file=sys.stderr)
        return 1

    payload = compute_news_fatigue(rows)
    payload["scanned_rows"] = len(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    top = payload["tickers"]
    print(f"news_fatigue: scanned={len(rows)} fatigued={payload['fatigued_count']}")
    for r in top:
        print(
            f"  {r['ticker']}: total={r['total_24h']} "
            f"recent_avg={r['recent_mean_score']:.1f} "
            f"prior_avg={r['prior_mean_score']:.1f} "
            f"drop={r['score_drop']:.1f}"
        )
    if not top:
        print("  (no fatigued tickers in window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
