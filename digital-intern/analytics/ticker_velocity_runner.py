"""Ticker mention velocity — top tickers' arrival-count ratio recent-vs-prior.

Pure builder + CLI snapshot + Flask endpoint surface. Refactored to match
the ``sentiment_reversal`` template: ``build_ticker_velocity(articles, ...)``
is the pure I/O-free builder used by both the Flask endpoint and unit
tests; ``compute()`` opens ``articles.db``, calls the builder, and writes
the snapshot JSON for the CLI/cron consumer.

Discovers the top-N tickers from the most-recent ``2 * window_min`` minutes
of LIVE articles (canonical ``_LIVE_ONLY_CLAUSE`` — synthetic backtest/opus
rows are excluded so per-ticker counts cannot be inflated by an injection
burst). For each discovered ticker computes:
  * ``recent``        — live mentions in [now - window_min, now]
  * ``prior``         — live mentions in [now - 2*window_min, now - window_min]
  * ``ratio``         — (recent + 1) / (prior + 1), Laplace-smoothed so a
                         prior=0 case yields a finite ratio
  * ``newest_age_s``  — seconds since the most-recent matching mention
  * ``verdict``       — per-ticker BREAKING / WARMING / QUIET (see thresholds)

Top-level ``verdict`` ladder (most-severe-first):
  * ``BREAKING``   — best ticker ratio ≥ ``BREAKING_RATIO`` AND recent ≥
                      ``BREAKING_MIN_RECENT``.
  * ``WARMING``    — best ratio ≥ ``WARMING_RATIO`` AND recent ≥
                      ``WARMING_MIN_RECENT``.
  * ``QUIET``      — every ticker below thresholds.
  * ``NO_DATA``    — no live articles in the full ``2 * window_min`` window.

Sibling surfaces and how this differs:
  * ``trend_velocity`` — same numerator/denominator shape but JSON-only and
    carries a partial live-only filter (long-standing bug; does not exclude
    ``backtest://`` URLs or ``opus_annotation*`` sources). This module uses
    the canonical ``_LIVE_ONLY_CLAUSE`` end-to-end and is the version
    addressable from the endpoint + chat block.
  * ``ticker_score_acceleration`` — slope of *ml_score* over four 30-min
    sub-windows (score-based momentum). This is arrival-count based.
  * ``ticker_comentions`` — pair co-mention graph (sector axis).
  * ``keyword_surge`` — narrative-keyword frequency, not tickers.
  * ``ArticleStore.ticker_mention_velocity`` — the DB-bound primitive whose
    counting logic this builder mirrors in pure Python so the endpoint can
    answer in a single round-trip without callers needing to pre-seed the
    ticker list.

Standalone:  python3 -m analytics.ticker_velocity_runner
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from analytics.trend_velocity import TICKER_RE, STOP as _BASE_STOP, _parse_ts
from storage.article_store import _LIVE_ONLY_CLAUSE

# Local STOP extension — noise tokens the trend_velocity base STOP doesn't
# yet catch but that consistently surface in this builder's per-ticker
# count axis (count is a volume metric, so it amplifies headline noise
# more than score-based surfaces). Subtractive only: every entry here
# removes a false-positive ticker; no new positives are introduced.
# These were carried in the pre-refactor module-local STOP and must
# survive the refactor so the live output doesn't regress.
_EXTRA_STOP = frozenset({
    "DRAM", "NAND", "HDD", "SSD", "RAM", "CPU", "GPU", "PCB", "PCIe",
    "AG", "TD", "AD", "HOC", "AOL", "ATM", "TSMC",
})
STOP = _BASE_STOP | _EXTRA_STOP

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/ticker_velocity.json")

WINDOW_MIN = 120
TOP_N = 10
FETCH_LIMIT = 6000

# Verdict thresholds. Conservative so a tiny baseline cannot fire BREAKING
# off a single new arrival; matches the silence-on-healthy precedent used
# by sentiment_reversal / ticker_score_dispersion.
BREAKING_RATIO = 4.0
BREAKING_MIN_RECENT = 5
WARMING_RATIO = 2.0
WARMING_MIN_RECENT = 3


def _extract_tickers(title: str) -> list[str]:
    """Whole-word, all-caps ticker extraction with the same STOP set as
    ``trend_velocity`` so every velocity / reversal / dispersion / comentions
    surface agrees on what counts as a "ticker"."""
    return [
        m for m in TICKER_RE.findall(title or "")
        if m not in STOP and len(m) >= 2
    ]


def _classify_ticker(ratio: float, recent: int) -> str:
    if ratio >= BREAKING_RATIO and recent >= BREAKING_MIN_RECENT:
        return "BREAKING"
    if ratio >= WARMING_RATIO and recent >= WARMING_MIN_RECENT:
        return "WARMING"
    return "QUIET"


def build_ticker_velocity(
    articles: Iterable[dict],
    window_min: int = WINDOW_MIN,
    top_n: int = TOP_N,
    now: datetime | None = None,
) -> dict:
    """Pure builder.

    ``articles`` is any iterable of dicts with at least ``first_seen`` and
    ``title`` keys. Rows with missing/unparseable ``first_seen`` are
    skipped (counted via ``skipped``).

    Returns a stable shape so the endpoint can ``jsonify`` it directly and
    chat helpers can assume every key exists.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    window_min = max(1, int(window_min))
    top_n = max(1, int(top_n))

    cutoff_recent = now - timedelta(minutes=window_min)
    cutoff_prior = now - timedelta(minutes=2 * window_min)

    rows_scanned = 0
    skipped = 0
    rows_in_window = 0

    discover_counts: Counter[str] = Counter()
    recent_counts: Counter[str] = Counter()
    prior_counts: Counter[str] = Counter()
    newest_ts: dict[str, datetime] = {}

    for art in articles:
        rows_scanned += 1
        ts = _parse_ts(art.get("first_seen"))
        if ts is None:
            skipped += 1
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < cutoff_prior:
            continue
        in_recent = ts >= cutoff_recent
        rows_in_window += 1
        for tk in _extract_tickers(art.get("title") or ""):
            discover_counts[tk] += 1
            if in_recent:
                recent_counts[tk] += 1
                cur = newest_ts.get(tk)
                if cur is None or ts > cur:
                    newest_ts[tk] = ts
            else:
                prior_counts[tk] += 1

    top_tickers = [t for t, _ in discover_counts.most_common(top_n)]

    tickers_out: list[dict] = []
    for tk in top_tickers:
        recent = recent_counts.get(tk, 0)
        prior = prior_counts.get(tk, 0)
        ratio = (recent + 1) / (prior + 1)
        newest = newest_ts.get(tk)
        newest_age_s = (
            round((now - newest).total_seconds(), 1)
            if newest is not None else None
        )
        tickers_out.append({
            "ticker": tk,
            "recent": int(recent),
            "prior": int(prior),
            "ratio": round(ratio, 3),
            "newest_age_s": newest_age_s,
            "verdict": _classify_ticker(ratio, recent),
        })

    # Sort highest ratio first; ties broken by recent count desc so
    # an actively-breaking name surfaces above an equally-rising-but-quieter
    # baseline.
    tickers_out.sort(key=lambda r: (r["ratio"], r["recent"]), reverse=True)

    n_breaking = sum(1 for r in tickers_out if r["verdict"] == "BREAKING")
    n_warming = sum(1 for r in tickers_out if r["verdict"] == "WARMING")

    if rows_in_window == 0:
        verdict = "NO_DATA"
        headline = "no live articles in the velocity window"
    elif n_breaking >= 1:
        verdict = "BREAKING"
        top = next(r for r in tickers_out if r["verdict"] == "BREAKING")
        age_str = (
            f"{top['newest_age_s']:.0f}s ago"
            if top.get("newest_age_s") is not None else "no hits"
        )
        headline = (
            f"BREAKING: {top['ticker']} {top['prior']}→{top['recent']} "
            f"(ratio {top['ratio']:.2f}, newest {age_str})"
        )
    elif n_warming >= 1:
        verdict = "WARMING"
        top = next(r for r in tickers_out if r["verdict"] == "WARMING")
        headline = (
            f"WARMING: {top['ticker']} {top['prior']}→{top['recent']} "
            f"(ratio {top['ratio']:.2f})"
        )
    else:
        verdict = "QUIET"
        if tickers_out:
            top = tickers_out[0]
            headline = (
                f"QUIET: top ticker {top['ticker']} ratio {top['ratio']:.2f}"
            )
        else:
            headline = "QUIET: no tickers discovered"

    return {
        "generated_at": now.isoformat(),
        "window_min": int(window_min),
        "top_n": int(top_n),
        "rows_scanned": rows_scanned,
        "rows_in_window": rows_in_window,
        "skipped": skipped,
        "breaking_ratio_threshold": BREAKING_RATIO,
        "breaking_min_recent": BREAKING_MIN_RECENT,
        "warming_ratio_threshold": WARMING_RATIO,
        "warming_min_recent": WARMING_MIN_RECENT,
        "verdict": verdict,
        "headline": headline,
        "n_breaking": n_breaking,
        "n_warming": n_warming,
        "tickers": tickers_out,
    }


def _fetch_articles_from_db(
    db_path: Path,
    window_min: int = WINDOW_MIN,
    limit: int = FETCH_LIMIT,
) -> list[dict]:
    """Read live articles bounded by ``2 * window_min`` minutes from the DB."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20)
    try:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=2 * window_min)
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
    window_min: int = WINDOW_MIN,
) -> dict:
    """CLI entry — fetch from DB, build, write snapshot, return payload."""
    db_path = db_path or DB_PATH
    articles = _fetch_articles_from_db(db_path, window_min=window_min)
    payload = build_ticker_velocity(articles, window_min=window_min, now=now)
    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass
    return payload


def main() -> int:
    payload = compute()
    tickers = payload.get("tickers") or []
    print(
        f"ticker_velocity: rows_scanned={payload.get('rows_scanned', 0)} "
        f"window={payload.get('window_min')}min "
        f"verdict={payload.get('verdict')}"
    )
    for row in tickers[:10]:
        age = (
            f"{row['newest_age_s']:.0f}s ago"
            if row.get("newest_age_s") is not None else "no hits"
        )
        print(
            f"  {row['ticker']:6s} {row['verdict']:8s}  "
            f"recent={row['recent']:3d}  prior={row['prior']:3d}  "
            f"ratio={row['ratio']:.2f}  ({age})"
        )
    if not tickers:
        print("  (no tickers discovered in window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
