"""Ticker co-mention graph — top ticker pairs co-occurring in recent articles.

Pure builder + CLI snapshot + Flask endpoint surface. Refactored to match
the ``sentiment_reversal`` / ``ticker_velocity_runner`` template:
``build_ticker_comentions(articles, ...)`` is the pure I/O-free builder
used by both the Flask endpoint and unit tests; ``main()`` opens
``articles.db``, calls the builder, and writes the snapshot JSON for the
CLI/cron consumer.

Surfaces the *sector-axis* sibling of single-ticker velocity: when two
tickers light up TOGETHER repeatedly in a short window, it's usually a
sector ETF rip, a peer-readthrough, or an M&A pairing rather than a
single-name story. Operators answering "is this NVDA velocity
idiosyncratic or part of a semis basket move?" need this surface — the
mention-volume primitives can't separate the two.

Verdict ladder (most-severe-first):
  * ``SECTOR_BURST``    — top pair lift ≥ ``BURST_LIFT`` (=0.7) AND
                           co-count ≥ ``BURST_MIN_CO`` (=4). At least one
                           pair has the wire pairing them aggressively.
  * ``COUPLED_NAMES``   — top pair co-count ≥ ``MIN_PAIR_COUNT`` (=2). Some
                           recurring pairs but below burst.
  * ``DISCONNECTED``    — no pair reached the minimum co-count.
  * ``NO_DATA``         — no live articles in the window.

Lift is ``co_count / min(solo_a, solo_b)`` — the fraction of the rarer
ticker's mentions that ALSO carry the other ticker. 0.7 means most stories
about the rarer name include the other name; that's a strong basket signal.

Pure / total — never raises on missing keys or unparseable rows. Honours
``_LIVE_ONLY_CLAUSE`` at the endpoint layer (backtest:// URLs and
``opus_annotation*`` rows excluded).

Sibling surfaces:
  * ``ticker_velocity_runner`` — per-ticker arrival ratio (single-name axis).
  * ``ticker_score_dispersion`` — intra-window score consensus (intra-name).
  * ``sentiment_reversal`` — directional flip across windows.
  * ``sector_pulse`` — news density per sector (taxonomy-driven, not
    discovered from pair counts).

Standalone:  python3 -m analytics.ticker_comentions
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

from analytics.trend_velocity import TICKER_RE, STOP as _BASE_STOP, _parse_ts
from storage.article_store import _LIVE_ONLY_CLAUSE

# Mirror the ticker_velocity_runner STOP extension verbatim — comentions is
# the pair-axis sibling of velocity; both surfaces must agree on what
# counts as a ticker or a single "AD HOC" headline produces a phantom
# SECTOR_BURST. Subtractive (removes false positives, no new positives).
_EXTRA_STOP = frozenset({
    "DRAM", "NAND", "HDD", "SSD", "RAM", "CPU", "GPU", "PCB", "PCIe",
    "AG", "TD", "AD", "HOC", "AOL", "ATM", "TSMC",
})
_STOP = _BASE_STOP | _EXTRA_STOP


def extract_tickers(title: str) -> list[str]:
    """Local override of ``trend_velocity.extract_tickers`` that applies
    the joined STOP set. Mirrored verbatim from
    ``ticker_velocity_runner._extract_tickers`` so the velocity and
    comentions surfaces never disagree on the ticker universe."""
    return [
        m for m in TICKER_RE.findall(title or "")
        if m not in _STOP and len(m) >= 2
    ]

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/ticker_comentions.json")

WINDOW_HOURS = 2
FETCH_LIMIT = 4000
TOP_N = 10
MIN_PAIR_COUNT = 2

# Burst thresholds.
BURST_LIFT = 0.7
BURST_MIN_CO = 4


def build_ticker_comentions(
    articles: Iterable[dict],
    window_hours: int = WINDOW_HOURS,
    top_n: int = TOP_N,
    now: datetime | None = None,
) -> dict:
    """Pure builder.

    ``articles`` is any iterable of dicts with at least ``first_seen`` and
    ``title`` keys. Rows older than ``window_hours`` (or unparseable) are
    skipped (counted via ``skipped``).
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    window_hours = max(1, int(window_hours))
    top_n = max(1, int(top_n))
    cutoff = now - timedelta(hours=window_hours)

    rows_scanned = 0
    skipped = 0
    rows_in_window = 0

    pair_counts: Counter[tuple[str, str]] = Counter()
    solo_counts: Counter[str] = Counter()

    for art in articles:
        rows_scanned += 1
        ts = _parse_ts(art.get("first_seen"))
        if ts is None:
            skipped += 1
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < cutoff:
            continue
        rows_in_window += 1
        # ``set`` to dedupe a title that mentions the same ticker twice;
        # ``sorted`` so pair keys are canonical regardless of mention order.
        tix = sorted(set(extract_tickers(art.get("title") or "")))
        for t in tix:
            solo_counts[t] += 1
        if len(tix) < 2:
            continue
        for a, b in combinations(tix, 2):
            pair_counts[(a, b)] += 1

    qualified = [
        (pair, c) for pair, c in pair_counts.items() if c >= MIN_PAIR_COUNT
    ]
    qualified.sort(key=lambda kv: kv[1], reverse=True)
    top = qualified[:top_n]

    pairs_out: list[dict] = []
    for (a, b), c in top:
        denom = min(solo_counts[a], solo_counts[b])
        lift = round(c / denom, 3) if denom else 0.0
        pairs_out.append({
            "pair": [a, b],
            "co_count": int(c),
            "a_total": int(solo_counts[a]),
            "b_total": int(solo_counts[b]),
            "lift": lift,
        })

    if rows_in_window == 0:
        verdict = "NO_DATA"
        headline = "no live articles in the comention window"
    elif not pairs_out:
        verdict = "DISCONNECTED"
        headline = (
            f"DISCONNECTED: no pair reached {MIN_PAIR_COUNT}+ co-mentions "
            f"in the {window_hours}h window"
        )
    else:
        top_pair = pairs_out[0]
        is_burst = (
            top_pair["lift"] >= BURST_LIFT
            and top_pair["co_count"] >= BURST_MIN_CO
        )
        if is_burst:
            verdict = "SECTOR_BURST"
            headline = (
                f"SECTOR_BURST: {top_pair['pair'][0]}+{top_pair['pair'][1]} "
                f"co={top_pair['co_count']} lift={top_pair['lift']:.2f} "
                f"(a_total={top_pair['a_total']} b_total={top_pair['b_total']})"
            )
        else:
            verdict = "COUPLED_NAMES"
            headline = (
                f"COUPLED_NAMES: {top_pair['pair'][0]}+{top_pair['pair'][1]} "
                f"co={top_pair['co_count']} lift={top_pair['lift']:.2f}"
            )

    return {
        "generated_at": now.isoformat(),
        "window_hours": int(window_hours),
        "top_n": int(top_n),
        "rows_scanned": rows_scanned,
        "rows_in_window": rows_in_window,
        "skipped": skipped,
        "unique_pairs": len(pair_counts),
        "qualified_pairs": len(qualified),
        "min_pair_count": MIN_PAIR_COUNT,
        "burst_lift_threshold": BURST_LIFT,
        "burst_min_co": BURST_MIN_CO,
        "verdict": verdict,
        "headline": headline,
        "top": pairs_out,
    }


def _fetch_articles_from_db(
    db_path: Path,
    window_hours: int = WINDOW_HOURS,
    limit: int = FETCH_LIMIT,
) -> list[dict]:
    """Read live articles bounded by ``window_hours`` from the DB."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20)
    try:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=window_hours)
        ).isoformat()
        rows = conn.execute(
            "SELECT first_seen, title FROM articles "
            f"WHERE {_LIVE_ONLY_CLAUSE} AND first_seen >= ? "
            "ORDER BY first_seen DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
    finally:
        conn.close()
    return [{"first_seen": r[0], "title": r[1]} for r in rows]


def compute(
    now: datetime | None = None,
    db_path: Path | None = None,
    window_hours: int = WINDOW_HOURS,
) -> dict:
    """CLI entry — fetch from DB, build, write snapshot, return payload."""
    db_path = db_path or DB_PATH
    articles = _fetch_articles_from_db(db_path, window_hours=window_hours)
    payload = build_ticker_comentions(
        articles, window_hours=window_hours, now=now
    )
    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass
    return payload


def main() -> int:
    payload = compute()
    print(
        f"ticker_comentions: rows_scanned={payload.get('rows_scanned', 0)} "
        f"window={payload.get('window_hours')}h "
        f"verdict={payload.get('verdict')}"
    )
    for row in (payload.get("top") or [])[:10]:
        a, b = row["pair"]
        print(
            f"  {a}+{b}  co={row['co_count']}  "
            f"a_total={row['a_total']}  b_total={row['b_total']}  "
            f"lift={row['lift']:.2f}"
        )
    if not payload.get("top"):
        print("  (no qualifying pairs in window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
