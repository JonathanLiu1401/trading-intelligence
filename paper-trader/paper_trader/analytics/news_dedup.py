"""News deduplication + urgency decay.

Two related problems on the live signal feed:

  1. **Same event, N sources.** GDELT, Reuters, Yahoo, Bloomberg all syndicate
     the same headline within minutes. Each becomes a separate article row
     with a separate ai_score and urgency. Without dedup, the trader sees the
     same story 5 times.
  2. **Stale urgency.** `urgency = 1` is set when an article is ingested and
     never decays. An 18-hour-old "URGENT" headline still surfaces as urgent
     in `get_urgent_articles()`.

We dedup by a short normalized title signature (first 8 word-tokens) and apply
an exponential decay to urgency, falling off with hours since first_seen.

Both helpers are pure functions over the article-dict shape that
`signals.py` already returns. They don't read the DB themselves.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

_DECAY_HALFLIFE_HOURS = 4.0  # urgency*0.5 after 4h, *0.25 after 8h
_DEDUP_TOKENS = 8           # how many leading title tokens form the key
_DEDUP_WINDOW_HOURS = 6     # only dedup against the last 6 hours
_WORD = re.compile(r"[a-z0-9]+")


def _norm_signature(title: str) -> str:
    """Lowercase + alpha-num tokens; collapse to the first N words.

    Crude but effective for syndicated news. "MU stock jumps 6% on earnings beat"
    from Reuters and "Micron (MU) jumps 6% after earnings beat" from Yahoo
    collide after lowercasing + first-8-token compaction.
    """
    if not title:
        return ""
    toks = _WORD.findall(title.lower())
    return " ".join(toks[:_DEDUP_TOKENS])


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def decay_factor(first_seen: str | None,
                 halflife_hours: float = _DECAY_HALFLIFE_HOURS) -> float:
    """Exponential decay factor in [0, 1] based on age since first_seen."""
    dt = _parse_iso(first_seen)
    if dt is None:
        return 1.0
    age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    if age_h <= 0:
        return 1.0
    return math.pow(0.5, age_h / max(halflife_hours, 0.1))


def decay_urgency(articles: list[dict],
                  halflife_hours: float = _DECAY_HALFLIFE_HOURS,
                  min_effective: float = 0.5) -> list[dict]:
    """Annotate each article with `urgency_decayed` and `decay_factor`.

    Filters out items whose decayed urgency falls below `min_effective`.
    Original `urgency` field is preserved unchanged.
    """
    out: list[dict] = []
    for a in articles:
        df = decay_factor(a.get("first_seen"), halflife_hours)
        orig_urg = float(a.get("urgency") or 0)
        decayed = orig_urg * df
        a2 = dict(a)
        a2["urgency_decayed"] = round(decayed, 3)
        a2["decay_factor"] = round(df, 3)
        if decayed >= min_effective or orig_urg == 0:
            out.append(a2)
    return out


def dedupe_articles(articles: list[dict],
                    window_hours: float = _DEDUP_WINDOW_HOURS) -> list[dict]:
    """Collapse syndicated duplicates by first-N-token title signature.

    Within `window_hours`, only the highest-ai_score article per signature is kept.
    Output preserves the original input order so downstream renderers still see
    "newest/highest-ranked first."
    """
    keep: dict[str, dict] = {}
    cutoff = datetime.now(timezone.utc).timestamp() - window_hours * 3600
    out_order: list[str] = []

    for a in articles:
        sig = _norm_signature(a.get("title") or "")
        if not sig:
            # No signature → no dedup candidate, pass through with random key
            sig = f"_pt::{a.get('id') or len(out_order)}"
        ts = _parse_iso(a.get("first_seen"))
        if ts and ts.timestamp() < cutoff:
            # Outside dedup window → still emit, but don't compare across boundary
            sig = f"old::{sig}::{a.get('id') or len(out_order)}"

        cur = keep.get(sig)
        if cur is None:
            keep[sig] = a
            out_order.append(sig)
        else:
            # Higher ai_score wins; tie → prefer the more urgent / newer
            cs = float(cur.get("ai_score") or 0)
            ns = float(a.get("ai_score") or 0)
            if (ns, a.get("urgency") or 0, a.get("first_seen") or "") > \
               (cs, cur.get("urgency") or 0, cur.get("first_seen") or ""):
                # Replace; track the displaced count
                a_with_dup = dict(a)
                a_with_dup["dup_count"] = int(cur.get("dup_count", 1)) + 1
                keep[sig] = a_with_dup
            else:
                cur["dup_count"] = int(cur.get("dup_count", 1)) + 1

    return [keep[s] for s in out_order if s in keep]


def dedupe_and_decay(articles: list[dict],
                     window_hours: float = _DEDUP_WINDOW_HOURS,
                     halflife_hours: float = _DECAY_HALFLIFE_HOURS,
                     min_effective: float = 0.5) -> list[dict]:
    """End-to-end helper: dedup first (so duplicate counts roll up), then decay."""
    return decay_urgency(dedupe_articles(articles, window_hours),
                         halflife_hours, min_effective)


if __name__ == "__main__":
    # smoke
    sample = [
        {"id": "a", "title": "Micron stock jumps 6% on earnings beat", "ai_score": 7.0,
         "urgency": 1, "first_seen": datetime.now(timezone.utc).isoformat()},
        {"id": "b", "title": "MU stock jumps 6% on earnings beat (Reuters)", "ai_score": 6.5,
         "urgency": 1, "first_seen": datetime.now(timezone.utc).isoformat()},
        {"id": "c", "title": "Nvidia hits new high", "ai_score": 8.0, "urgency": 2,
         "first_seen": "2024-01-01T00:00:00+00:00"},
    ]
    out = dedupe_and_decay(sample)
    for a in out:
        print(a["title"], a["ai_score"], a["urgency_decayed"], a.get("dup_count"))
