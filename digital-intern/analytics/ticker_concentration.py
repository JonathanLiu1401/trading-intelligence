"""Ticker-concentration audit: held-book mention saturation in a recent window.

``collection_quality`` and ``publish_lag_audit`` answer source-level questions
(volume, score, freshness). This module answers a *book-level* question:
**which of our held/watchlist tickers is dominating the recent news flow, and
which are we blind to?**

Two operational failure modes this surfaces:

* **Echo-chamber dominance** — one ticker is the subject of >SATURATION_PCT of
  recent name-mentioning articles. The briefing's `_book_tickers` /
  `_rank_by_decayed_score` ranker will then over-weight that name; the
  scorer's training signal skews toward it; alert volume is one-sided.
* **Under-coverage** — a held ticker has zero mentions in the window. The
  briefing's BOOK HEAT and BOOK COVERAGE lines go silent for that name even
  when the position is open; the operator may not realise news flow has
  evaporated until they're already in a drawdown.

The matched ticker set is mirrored verbatim from
``analysis.claude_analyst._BOOK_TICKERS`` (parity pinned by
``tests/test_briefing_book_tag.py``) so the audit reports against the same
held universe the briefing ranks against — drift between the two would silently
break the read.

Output shape::

    {
      "generated_at": "<iso utc>",
      "scan_limit": 5000,
      "scanned": <int>,                        # rows pulled
      "articles_with_book_ticker": <int>,      # rows mentioning ≥1 held name
      "tickers": {
         "NVDA": {
            "n_mentions": <int>,
            "pct_share": <0..100>,             # share of mentioning articles
            "n_urgent": <int>,                 # urgency >= 2 among those
            "avg_ai_score": <float|null>,
         },
         ...
      },
      "hhi": <0..10000>,                       # Herfindahl-Hirschman index
      "over_saturated": ["NVDA", ...],         # share >= SATURATION_PCT
      "under_covered":  ["LITE", ...],         # held tickers with zero mentions
    }

Read-only sqlite (``mode=ro``) — never takes a write lock on the production DB.
The scan is bounded by ``SCAN_LIMIT`` (a recent-id slice, not a full-table
COUNT) so it stays fast against the USB-backed DB.

Standalone:   ``python3 -m analytics.ticker_concentration``
Importable:   ``from analytics.ticker_concentration import compute``
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

# Mirror of analysis.claude_analyst._BOOK_TICKERS. Kept as a local literal
# rather than imported so this module doesn't pull the analysis layer's import
# graph (and its claude_cli/subprocess weight) for a pure read-side audit.
# Parity with the briefing's source is pinned by
# tests/test_ticker_concentration.py::test_book_ticker_parity.
_BOOK_TICKERS: tuple[str, ...] = (
    "LITE", "LNOK", "MUU", "DRAM", "SNDU",
    "MU", "MSFT", "AXTI", "ORCL", "TSEM", "QBTS", "NVDA",
)

# Longest-first alternation so the regex prefers \bMUU\b over \bMU\b — same
# convention as analysis.claude_analyst._BOOK_RE / ml.features._LIVE_RE.
_BOOK_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(t) for t in sorted(set(_BOOK_TICKERS),
                                             key=len, reverse=True))
    + r")\b"
)

# Identical recency bound to scorer_skew / publish_lag_audit — sized for
# sub-second reads on the slow USB DB while still covering a meaningful window.
SCAN_LIMIT = 5000

# A ticker mentioned in ≥ this share of book-mentioning articles is flagged.
# Calibrated at 25 %: below that, mention skew is normal noise (12 tickers;
# uniform share is ~8 %, so 25 % is 3× uniform and a genuine outlier).
SATURATION_PCT = 25.0

# Snapshot lives alongside the operator's other advisory artifacts. Matches
# scorer_skew / publish_lag_audit's directory deliberately.
SNAPSHOT_PATH = Path("/home/zeph/logs/ticker_concentration.json")


def _book_hits(title: Optional[str]) -> list[str]:
    """Return the held tickers mentioned in ``title``, in canonical
    ``_BOOK_TICKERS`` order, de-duplicated. Empty when nothing matches.

    The production ``articles`` table stores title and zlib-compressed
    full_text but no separate summary column — decompressing full_text per
    row would be O(N) heavy for a recurring audit. Title-only matching is
    cheap and matches how the briefing's BOOK tag covers the bulk of named
    headlines (financial copy puts the ticker in the title).
    """
    if not title:
        return []
    hits = set(_BOOK_RE.findall(title))
    if not hits:
        return []
    return [t for t in _BOOK_TICKERS if t in hits]


def _hhi(shares_pct: list[float]) -> float:
    """Herfindahl-Hirschman index over percentage shares (each 0..100).

    Result is on the conventional 0..10000 scale (a monopoly is 10000, perfect
    distribution across N is 10000/N). 0 when no mentions.
    """
    return round(sum(s * s for s in shares_pct), 2)


def compute(now: Optional[datetime] = None, scan_limit: int = SCAN_LIMIT) -> dict:
    """Build the ticker-concentration report.

    ``now`` is recorded as ``generated_at``; the audit itself uses the
    most-recent-by-id slice (matching the convention of scorer_skew /
    publish_lag_audit), so a wall-clock argument is purely a stamp.
    """
    now = now or datetime.now(timezone.utc)
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            f"""
            SELECT title, ai_score, urgency
              FROM articles
             WHERE id IN (SELECT id FROM articles ORDER BY id DESC LIMIT ?)
               AND {_LIVE_ONLY_CLAUSE}
            """,
            (scan_limit,),
        ).fetchall()
    finally:
        conn.close()

    agg: dict[str, dict] = {}
    mentioning_articles = 0
    for title, ai_score, urgency in rows:
        hits = _book_hits(title)
        if not hits:
            continue
        mentioning_articles += 1
        for t in hits:
            s = agg.setdefault(
                t, {"n_mentions": 0, "n_urgent": 0, "ai_sum": 0.0, "ai_n": 0}
            )
            s["n_mentions"] += 1
            if urgency is not None and urgency >= 2:
                s["n_urgent"] += 1
            if ai_score is not None:
                s["ai_sum"] += ai_score
                s["ai_n"] += 1

    tickers: dict[str, dict] = {}
    shares_pct: list[float] = []
    for t, s in agg.items():
        share = (100.0 * s["n_mentions"] / mentioning_articles) if mentioning_articles else 0.0
        shares_pct.append(share)
        tickers[t] = {
            "n_mentions": s["n_mentions"],
            "pct_share": round(share, 2),
            "n_urgent": s["n_urgent"],
            "avg_ai_score": (
                round(s["ai_sum"] / s["ai_n"], 3) if s["ai_n"] else None
            ),
        }

    over_saturated = [
        t for t, info in tickers.items()
        if info["pct_share"] >= SATURATION_PCT
    ]
    over_saturated.sort(key=lambda t: tickers[t]["pct_share"], reverse=True)

    under_covered = [t for t in _BOOK_TICKERS if t not in tickers]

    return {
        "generated_at": now.isoformat(),
        "scan_limit": scan_limit,
        "scanned": len(rows),
        "articles_with_book_ticker": mentioning_articles,
        "tickers": tickers,
        "hhi": _hhi(shares_pct),
        "over_saturated": over_saturated,
        "under_covered": under_covered,
    }


def write_snapshot(report: dict, path: Path = SNAPSHOT_PATH) -> Path:
    """Persist ``report`` to ``path`` as pretty JSON. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path


def main() -> None:
    report = compute()
    out = write_snapshot(report)
    print(
        f"scanned={report['scanned']} "
        f"with_book_ticker={report['articles_with_book_ticker']} "
        f"hhi={report['hhi']}"
    )
    print(f"output={out}")
    if report["over_saturated"]:
        print(f"over_saturated: {', '.join(report['over_saturated'])}")
    if report["under_covered"]:
        print(f"under_covered:  {', '.join(report['under_covered'])}")
    ranked = sorted(
        report["tickers"].items(),
        key=lambda kv: kv[1]["n_mentions"],
        reverse=True,
    )
    for t, info in ranked[:8]:
        avg = info["avg_ai_score"]
        print(
            f"  {t:<6} n={info['n_mentions']:>4}  "
            f"share={info['pct_share']:>5.2f}%  "
            f"urgent={info['n_urgent']:>3}  "
            f"avg_ai={avg if avg is not None else '  n/a'}"
        )


if __name__ == "__main__":
    main()
