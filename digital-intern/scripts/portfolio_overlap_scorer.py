#!/usr/bin/env python3
"""Rank recent articles by count of held-portfolio tickers mentioned."""
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "data" / "articles.db"
PORTFOLIO = REPO / "config" / "portfolio.json"
OUT = Path("/home/zeph/logs/portfolio_overlap.json")


def load_held_tickers() -> set[str]:
    p = json.loads(PORTFOLIO.read_text())
    held: set[str] = set()
    for pos in p.get("positions", []):
        t = (pos.get("ticker") or "").strip().upper()
        if t:
            held.add(t)
    for opt in p.get("options", []):
        t = (opt.get("underlying") or "").strip().upper()
        if t:
            held.add(t)
    return held


def score_articles(held: set[str], hours: int = 6, limit: int = 500) -> list[dict]:
    if not held:
        return []
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in held) + r")\b")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT id, title, source, ml_score, ai_score, urgency, first_seen
        FROM articles
        WHERE replace(first_seen,'T',' ') >= datetime('now', ?)
        ORDER BY first_seen DESC
        LIMIT ?
        """,
        (f"-{hours} hours", limit * 10),
    ).fetchall()
    con.close()

    scored = []
    for r in rows:
        title = r["title"] or ""
        hits = set(m.upper() for m in pattern.findall(title.upper()))
        if not hits:
            continue
        base = r["ml_score"] if r["ml_score"] is not None else (r["ai_score"] or 0)
        scored.append(
            {
                "id": r["id"],
                "title": title[:180],
                "source": r["source"],
                "first_seen": r["first_seen"],
                "ml_score": base,
                "urgency": r["urgency"],
                "held_hits": sorted(hits),
                "overlap_count": len(hits),
                "overlap_score": round(len(hits) * (1 + (base or 0)), 4),
            }
        )
    scored.sort(key=lambda x: (x["overlap_count"], x["overlap_score"]), reverse=True)
    return scored[:limit]


def main() -> int:
    held = load_held_tickers()
    ranked = score_articles(held)
    payload = {
        "held_tickers": sorted(held),
        "window_hours": 6,
        "total_matches": len(ranked),
        "top": ranked[:25],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"portfolio_overlap: held={len(held)} matches={len(ranked)} out={OUT}")
    for r in ranked[:5]:
        print(
            f"  [{r['overlap_count']}x {','.join(r['held_hits'])}] "
            f"ml={r['ml_score']} {r['source']} :: {r['title'][:100]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
