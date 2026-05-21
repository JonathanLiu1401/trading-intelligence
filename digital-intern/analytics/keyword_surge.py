"""Keyword surge detector: headline terms gaining frequency in last 1h vs 23h baseline.

Complements trend_velocity (which tracks $TICKER symbols) by catching emerging
narrative keywords — "tariff", "default", "acquisition", "layoff", etc. — before
they cluster into high-urgency articles.

Writes /home/zeph/logs/keyword_surge.json.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/keyword_surge.json")

RECENT_HOURS = 1
BASELINE_HOURS = 23
FETCH_LIMIT = 4000
TOP_N = 10
MIN_RECENT_COUNT = 3  # require at least 3 hits in recent window to surface

# Words that carry no narrative signal
STOP = {
    "the", "a", "an", "and", "or", "of", "for", "in", "to", "on", "at", "by",
    "as", "is", "are", "was", "be", "it", "its", "with", "from", "that", "this",
    "has", "have", "had", "will", "says", "said", "after", "over", "into",
    "about", "more", "new", "top", "how", "why", "what", "when", "who",
    "year", "years", "week", "month", "day", "today", "now", "rate", "rates",
    "stock", "stocks", "market", "share", "shares", "company", "companies",
    "news", "report", "reports", "data", "high", "low", "up", "down",
    "first", "second", "third", "one", "two", "three", "four", "five",
    "inc", "llc", "ltd", "corp", "co", "plc", "group",
    "quarter", "fiscal", "earnings", "revenue",  # too generic
}

# Bigram pairs where both halves are noise (still surfaced in unigrams)
_NOISE_BIGRAMS = {"short volume", "finra high", "high short", "volume alert"}

_WORD_RE = re.compile(r"\b([a-z]{4,})\b")


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


def tokenize(title: str) -> list[str]:
    words = [w for w in _WORD_RE.findall((title or "").lower()) if w not in STOP]
    return words


def bigrams(words: list[str]) -> list[str]:
    return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]


def main() -> int:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=20)
    conn.execute("PRAGMA query_only=ON")
    cur = conn.execute(
        "SELECT first_seen, title FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (FETCH_LIMIT,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("keyword_surge: no rows", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    recent_cut = now - timedelta(hours=RECENT_HOURS)
    baseline_cut = now - timedelta(hours=RECENT_HOURS + BASELINE_HOURS)

    recent_uni: Counter[str] = Counter()
    recent_bi: Counter[str] = Counter()
    base_uni: Counter[str] = Counter()
    base_bi: Counter[str] = Counter()
    n_recent = n_base = 0

    for fs, title in rows:
        ts = _parse_ts(fs)
        if ts is None:
            continue
        words = tokenize(title)
        bgs = bigrams(words)
        if ts >= recent_cut:
            recent_uni.update(words)
            recent_bi.update(bgs)
            n_recent += 1
        elif ts >= baseline_cut:
            base_uni.update(words)
            base_bi.update(bgs)
            n_base += 1

    # Hourly-normalised baseline counts (scale 23h -> 1h equivalent)
    scale = RECENT_HOURS / BASELINE_HOURS if n_base else 0.0

    def lift(now_c: int, base_c: int) -> float:
        expected = base_c * scale
        return round((now_c + 1) / (expected + 1), 2)

    # Unigram surges
    uni_surges = []
    for word, cnt in recent_uni.most_common(200):
        if cnt < MIN_RECENT_COUNT:
            continue
        bc = base_uni.get(word, 0)
        l = lift(cnt, bc)
        if l >= 2.0:
            uni_surges.append({"term": word, "type": "unigram",
                                "now": cnt, "baseline_1h_equiv": round(bc * scale, 1), "lift": l})
    uni_surges.sort(key=lambda x: x["lift"], reverse=True)

    # Bigram surges (skip pure-noise bigrams)
    bi_surges = []
    for bg, cnt in recent_bi.most_common(200):
        if cnt < MIN_RECENT_COUNT or bg in _NOISE_BIGRAMS:
            continue
        bc = base_bi.get(bg, 0)
        l = lift(cnt, bc)
        if l >= 2.5:
            bi_surges.append({"term": bg, "type": "bigram",
                               "now": cnt, "baseline_1h_equiv": round(bc * scale, 1), "lift": l})
    bi_surges.sort(key=lambda x: x["lift"], reverse=True)

    top = (bi_surges[:TOP_N // 2] + uni_surges[:TOP_N])[:TOP_N]
    top.sort(key=lambda x: x["lift"], reverse=True)

    payload = {
        "generated_at": now.isoformat(),
        "articles_scanned": len(rows),
        "articles_recent_1h": n_recent,
        "articles_baseline_23h": n_base,
        "top_surging": top,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(OUT_PATH)

    print(f"keyword_surge: scanned={len(rows)} recent_1h={n_recent} baseline_23h={n_base}")
    for item in top[:5]:
        print(f"  [{item['type']}] '{item['term']}': now={item['now']} lift={item['lift']}x")
    if not top:
        print("  (no surging keywords in window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
