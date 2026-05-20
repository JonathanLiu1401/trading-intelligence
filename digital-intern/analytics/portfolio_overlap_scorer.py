"""Portfolio overlap scorer: rank recent articles by held-ticker overlap count.

Reads open positions from the live paper-trader DB, then scans the most recent
articles and scores each by how many distinct held tickers it mentions.
Outputs the top-N articles ranked by (overlap_count DESC, ai_score DESC) to
/home/zeph/logs/portfolio_overlap.json.

Standalone:  python3 -m analytics.portfolio_overlap_scorer
Importable:  from analytics.portfolio_overlap_scorer import compute
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

ARTICLES_DB = Path(__file__).resolve().parents[1] / "data" / "articles.db"
PAPER_TRADER_DB = Path("/media/zeph/projects/paper-trader/data/paper_trader.db")
OUT_PATH = Path("/home/zeph/logs/portfolio_overlap.json")

SCAN_LIMIT = 3000
TOP_N = 10


def _held_tickers() -> list[str]:
    """Return uppercase ticker symbols of currently open paper-trader positions."""
    candidates = [
        PAPER_TRADER_DB,
        Path("/home/zeph/trading-intelligence/paper-trader/data/paper_trader.db"),
        Path(__file__).resolve().parents[1] / "data" / "paper_trader.db",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            rows = conn.execute(
                "SELECT UPPER(ticker) FROM positions WHERE closed_at IS NULL"
            ).fetchall()
            conn.close()
            tickers = sorted({r[0] for r in rows if r[0]})
            if tickers:
                return tickers
        except Exception:
            continue
    return []


def _build_re(tickers: list[str]) -> re.Pattern:
    """Word-boundary pattern for the held ticker set, longest first."""
    alt = "|".join(re.escape(t) for t in sorted(tickers, key=len, reverse=True))
    return re.compile(rf"\b(?:{alt})\b")


def _overlap(title: str, pattern: re.Pattern, tickers: list[str]) -> list[str]:
    """Distinct held tickers found in title."""
    if not title:
        return []
    found = set(pattern.findall(title))
    return [t for t in tickers if t in found]


def compute(
    scan_limit: int = SCAN_LIMIT,
    top_n: int = TOP_N,
) -> dict:
    held = _held_tickers()
    generated_at = datetime.now(timezone.utc).isoformat()

    if not held:
        result = {
            "generated_at": generated_at,
            "held_tickers": [],
            "scanned": 0,
            "top_articles": [],
            "note": "no open positions in paper trader",
        }
        OUT_PATH.write_text(json.dumps(result, indent=2))
        return result

    pattern = _build_re(held)

    conn = sqlite3.connect(f"file:{ARTICLES_DB}?mode=ro", uri=True, timeout=20)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, source, ai_score, ml_score, urgency, first_seen "
        f"FROM articles WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (scan_limit,),
    ).fetchall()
    conn.close()

    scored = []
    for row in rows:
        hits = _overlap(row["title"] or "", pattern, held)
        if not hits:
            continue
        scored.append({
            "id": row["id"],
            "title": row["title"],
            "source": row["source"],
            "ai_score": row["ai_score"],
            "ml_score": row["ml_score"],
            "urgency": row["urgency"],
            "first_seen": row["first_seen"],
            "held_tickers_hit": hits,
            "overlap_count": len(hits),
        })

    scored.sort(key=lambda x: (-x["overlap_count"], -(x["ai_score"] or 0)))
    top = scored[:top_n]

    # Per-ticker mention counts across all matching articles
    ticker_counts: dict[str, int] = {t: 0 for t in held}
    for art in scored:
        for t in art["held_tickers_hit"]:
            ticker_counts[t] += 1

    result = {
        "generated_at": generated_at,
        "held_tickers": held,
        "scanned": len(rows),
        "matching_articles": len(scored),
        "ticker_mention_counts": ticker_counts,
        "top_articles": top,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    result = compute()
    held = result["held_tickers"]
    scanned = result["scanned"]
    matching = result["matching_articles"]
    counts = result.get("ticker_mention_counts", {})
    print(f"Held: {held}")
    print(f"Scanned {scanned} articles → {matching} overlap matches")
    for t, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {n} mentions")
    print(f"\nTop {len(result['top_articles'])} articles:")
    for art in result["top_articles"][:5]:
        hits = ",".join(art["held_tickers_hit"])
        score = art["ai_score"] or 0
        print(f"  [{hits}] score={score:.2f}  {art['title'][:70]}")
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
