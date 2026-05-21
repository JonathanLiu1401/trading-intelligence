"""Regulatory surge detector: enforcement/regulatory keyword bursts in last 2h vs prior 2h.

Monitors specific regulatory and enforcement keywords in article titles — FINRA, SEC
enforcement, DOJ investigation, FTC antitrust, etc. — as a distinct signal class
separate from generic ticker trend velocity. A 3x+ spike in regulatory keyword
mentions often precedes market-moving enforcement actions before the full story
breaks on mainstream wires.

Output: /home/zeph/logs/regulatory_surge.json
Standalone: python3 -m analytics.regulatory_surge_detector
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/regulatory_surge.json")
WINDOW_HOURS = 2
FETCH_LIMIT = 5000
SURGE_THRESHOLD = 2.5  # ratio to flag as surge

# Regulatory / enforcement keyword groups.
# Each group represents a distinct enforcement category.
GROUPS: dict[str, list[str]] = {
    "sec_enforcement": [
        r"\bSEC\b.*(?:charge|fine|penalty|fraud|investigate|enforce|sanction|subpoena|settle)",
        r"(?:charge|fine|penalty|fraud|investigate|enforce|sanction|subpoena|settle).*\bSEC\b",
        r"\bSEC\s+(?:enforcement|action|investigation|probe|lawsuit|filing|complaint)\b",
    ],
    "finra_action": [
        r"\bFINRA\b",
    ],
    "doj_investigation": [
        r"\bDOJ\b.*(?:investigate|indict|charge|criminal|prosecute|antitrust|sue)",
        r"(?:investigate|indict|charge|criminal|prosecute|antitrust|sue).*\bDOJ\b",
        r"\bDepartment\s+of\s+Justice\b.*(?:invest|charge|indict|antitrust)",
    ],
    "ftc_antitrust": [
        r"\bFTC\b.*(?:antitrust|merger|block|sue|investigate|charge|fine|ban)",
        r"(?:antitrust|merger|block|sue|investigate|charge|fine|ban).*\bFTC\b",
        r"\bFederal\s+Trade\s+Commission\b",
    ],
    "occ_cftc": [
        r"\b(?:OCC|CFTC|FinCEN|OFAC|OFR|FDIC|OTS)\b.*(?:fine|penalty|enforce|order|action|charge)",
    ],
    "class_action": [
        r"\bclass[\s-]action\b.*(?:lawsuit|suit|filed|alleged|settle)",
        r"(?:lawsuit|suit|filed|alleged|settle).*\bclass[\s-]action\b",
    ],
    "insider_trading": [
        r"\binsider\s+trading\b",
        r"\binsider\s+information\b",
    ],
    "fraud_charges": [
        r"\b(?:securities|wire|bank|mail)\s+fraud\b",
        r"\bPonzi\b",
        r"\baccounting\s+fraud\b",
    ],
}

_COMPILED: dict[str, list[re.Pattern]] = {
    group: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for group, pats in GROUPS.items()
}


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip().replace("Z", "+00:00")
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


def _match_groups(title: str) -> list[str]:
    matched = []
    for group, patterns in _COMPILED.items():
        for pat in patterns:
            if pat.search(title):
                matched.append(group)
                break
    return matched


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")

    now = datetime.now(timezone.utc)
    cutoff_now = (now - timedelta(hours=WINDOW_HOURS)).isoformat()
    cutoff_prev = (now - timedelta(hours=WINDOW_HOURS * 2)).isoformat()

    rows = conn.execute(
        "SELECT first_seen, title, source, urgency, ml_score FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "AND first_seen >= ? "
        "ORDER BY first_seen DESC LIMIT ?",
        (cutoff_prev, FETCH_LIMIT),
    ).fetchall()
    conn.close()

    # Bucket into now-window and prev-window
    now_counts: Counter[str] = Counter()
    prev_counts: Counter[str] = Counter()
    now_articles: dict[str, list[dict]] = defaultdict(list)
    prev_articles: dict[str, list[dict]] = defaultdict(list)

    for first_seen, title, source, urgency, ml_score in rows:
        if not title:
            continue
        ts = _parse_ts(first_seen)
        if ts is None:
            continue
        groups = _match_groups(title)
        if not groups:
            continue
        is_now = ts >= _parse_ts(cutoff_now)
        for g in groups:
            if is_now:
                now_counts[g] += 1
                if len(now_articles[g]) < 5:
                    now_articles[g].append({
                        "title": title[:160],
                        "source": source,
                        "first_seen": first_seen,
                        "urgency": urgency or 0,
                        "ml_score": round(float(ml_score or 0), 4),
                    })
            else:
                prev_counts[g] += 1
                if len(prev_articles[g]) < 3:
                    prev_articles[g].append({"title": title[:120], "source": source})

    all_groups = set(now_counts) | set(prev_counts)
    results = []
    for g in sorted(all_groups):
        now_n = now_counts.get(g, 0)
        prev_n = prev_counts.get(g, 0)
        ratio = now_n / max(prev_n, 1)
        is_surge = ratio >= SURGE_THRESHOLD and now_n >= 2
        results.append({
            "group": g,
            "now_2h": now_n,
            "prev_2h": prev_n,
            "ratio": round(ratio, 2),
            "surge": is_surge,
            "sample_articles": now_articles.get(g, []),
        })

    results.sort(key=lambda r: (-r["ratio"], -r["now_2h"]))
    surges = [r for r in results if r["surge"]]

    output = {
        "generated_at": now.isoformat(),
        "window_hours": WINDOW_HOURS,
        "scanned": len(rows),
        "surge_threshold": SURGE_THRESHOLD,
        "surges_detected": len(surges),
        "groups": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"regulatory_surge_detector: scanned={len(rows)} surges={len(surges)}")
    for r in results[:8]:
        flag = " [SURGE]" if r["surge"] else ""
        print(f"  {r['group']:25s} now={r['now_2h']:3d} prev={r['prev_2h']:3d} ratio={r['ratio']:.1f}x{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
