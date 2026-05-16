"""Syndication dedup for the urgent-alert batch.

A breaking story is carried within minutes by GDELT, Reuters, Yahoo, Finnhub,
Google News and half a dozen RSS feeds. Each copy lands as its own row, each
can independently cross the urgency threshold, and ``send_urgent_alert`` then
packs five near-identical headlines into one Bloomberg alert — the trader reads
the same event five times and the genuinely distinct fifth story never makes
the batch.

This module collapses syndicated duplicates *before* batching. Articles whose
titles share a normalized signature (first 8 alphanumeric tokens, lowercased)
are merged into the highest-``ai_score`` representative; the merged-away copies
are recorded on ``_dup_ids`` so the caller can still mark every one of them
``alerted`` and stop them re-firing next cycle.

Pure function over the article-dict shape returned by
``ArticleStore.get_unalerted_urgent`` — ``{_id, link, title, source, ai_score,
summary}``. It does not touch the DB.
"""
from __future__ import annotations

import re

_DEDUP_TOKENS = 8                       # leading title tokens forming the key
_WORD = re.compile(r"[a-z0-9]+")
# Headline-vs-source separators: "Nvidia beats - Reuters", "... | Bloomberg".
_SOURCE_SEP = re.compile(r"\s+[-|–—]\s+")
# Trailing attribution parenthetical: "Nvidia beats (Reuters)".
_TRAIL_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
# Leading wire-service editorial markers. Reuters/AP/AFP republish the same
# story as "UPDATE 2-...", "RPT-...", "EXCLUSIVE-...", "WRAPUP 1-...",
# "BREAKING: ..." etc., and the markers stack ("RPT-UPDATE 2-..."). Without
# stripping them the 8-token window starts on the marker, so a revision
# ("UPDATE 3-") and the bare headline get different signatures and the most
# heavily reposted wire stories — exactly what this module exists to collapse —
# dedup the least. Anchored, whitelisted, and repeated so only known prefixes
# are consumed (a real all-caps headline word is never eaten).
_WIRE_PREFIX = re.compile(
    r"^\s*(?:"
    r"(?:UPDATE|WRAPUP|WRAP|RECAST|REFILE|RPT|CORRECTED|EXCLUSIVE|TABLE|"
    r"FACTBOX|TIMELINE|ANALYSIS|INSTANT\ VIEW|PRESS\ DIGEST|BREAKINGVIEWS|"
    r"BUZZ|GRAPHIC|POLL|SCENARIOS|EXPLAINER|HIGHLIGHTS|NEWSMAKER|COLUMN|"
    r"BREAKING|DEVELOPING|JUST\ IN|LIVE|WATCH|ALERT)"
    r"\s*\d*\s*[-:]\s*"
    r")+",
    re.IGNORECASE,
)


def _signature(title: str | None) -> str:
    """Lowercased first-N-alphanumeric-token signature of a headline.

    Leading wire-service editorial markers ("UPDATE 2-", "RPT-", "BREAKING:")
    and trailing source attribution ("...blowout - Reuters", "(Bloomberg)")
    are both stripped first — otherwise the most heavily syndicated stories
    (the ones with the most revisions and attributed reposts) would dedup the
    least. Once markers and attribution are gone, verbatim reposts collide
    outright and minor suffix/revision variants collide too.
    """
    if not title:
        return ""
    head = _WIRE_PREFIX.sub("", title.strip())
    head = _SOURCE_SEP.split(head)[0]
    head = _TRAIL_PAREN.sub("", head)
    return " ".join(_WORD.findall(head.lower())[:_DEDUP_TOKENS])


def dedupe_urgent(articles: list[dict]) -> list[dict]:
    """Collapse syndicated duplicates, preserving input order of the survivors.

    Each surviving article gains:
      * ``dup_count``  — total copies it represents (1 = no duplicates)
      * ``_dup_ids``   — ``_id``s of the merged-away copies (excludes its own)

    The highest-``ai_score`` copy wins the merge; ties keep the earlier one.
    Untitled articles are never merged (each gets a unique key).
    """
    keep: dict[str, dict] = {}
    order: list[str] = []

    for idx, art in enumerate(articles):
        sig = _signature(art.get("title"))
        if not sig:
            sig = f"__uniq__{art.get('_id') or idx}"

        cur = keep.get(sig)
        if cur is None:
            merged = dict(art)
            merged["dup_count"] = 1
            merged["_dup_ids"] = []
            keep[sig] = merged
            order.append(sig)
            continue

        # Same story already seen — pick the better representative, keep a
        # running tally and collect the loser's id either way.
        cur_score = float(cur.get("ai_score") or 0)
        new_score = float(art.get("ai_score") or 0)
        if new_score > cur_score:
            winner = dict(art)
            winner["dup_count"] = cur["dup_count"] + 1
            winner["_dup_ids"] = cur["_dup_ids"] + [cur["_id"]]
            keep[sig] = winner
        else:
            cur["dup_count"] += 1
            if art.get("_id") is not None:
                cur["_dup_ids"].append(art["_id"])

    return [keep[s] for s in order]


def alerted_ids(batch: list[dict]) -> list[str]:
    """Every id that must be marked ``alerted`` for a deduped batch.

    The batch members themselves plus all copies that were merged into them —
    so a syndicated duplicate of an alerted story never re-triggers, while
    duplicates of an article still queued (not in the batch) stay urgent.
    """
    ids: list[str] = []
    for art in batch:
        if art.get("_id") is not None:
            ids.append(art["_id"])
        ids.extend(art.get("_dup_ids") or [])
    return ids


if __name__ == "__main__":  # smoke test
    # Syndication is mostly verbatim reposting of a wire headline, plus the
    # occasional trailing-suffix variant — that is what the 8-token prefix
    # signature is built to collapse.
    sample = [
        {"_id": "a", "title": "Micron shares surge after Q3 earnings blowout", "ai_score": 7.0},
        {"_id": "b", "title": "Micron shares surge after Q3 earnings blowout", "ai_score": 8.5},
        {"_id": "c", "title": "Micron shares surge after Q3 earnings blowout - Reuters", "ai_score": 6.0},
        # Wire revisions/markers of the same story must collapse into it too.
        {"_id": "f", "title": "UPDATE 2-Micron shares surge after Q3 earnings blowout", "ai_score": 4.0},
        {"_id": "g", "title": "RPT-UPDATE 3-Micron shares surge after Q3 earnings blowout (Reuters)", "ai_score": 3.0},
        {"_id": "d", "title": "Fed holds rates steady amid inflation concerns", "ai_score": 9.0},
        {"_id": "e", "title": None, "ai_score": 5.0},
    ]
    out = dedupe_urgent(sample)
    for a in out:
        print(f"{a['_id']}  score={a['ai_score']}  dup_count={a['dup_count']}  dups={a['_dup_ids']}")
    print("alerted_ids(all):", alerted_ids(out))
    assert len(out) == 3, out
    assert out[0]["_id"] == "b" and out[0]["dup_count"] == 5, out[0]
    assert sorted(out[0]["_dup_ids"]) == ["a", "c", "f", "g"], out[0]
    assert sorted(alerted_ids(out)) == ["a", "b", "c", "d", "e", "f", "g"]
    # Marking only the batch's collapsed ids — a queued story's dups stay urgent.
    assert sorted(alerted_ids(out[:1])) == ["a", "b", "c", "f", "g"]
    print("OK")
