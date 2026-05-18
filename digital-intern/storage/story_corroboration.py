"""Cross-source breaking-news corroboration detector (read-only).

A single high-urgency headline from one source is weak signal — collectors
routinely surface a lone rumour, a misread filing, or clickbait. The strong,
market-moving signal a real news desk acts on is *corroboration*: the same
story breaking near-simultaneously across many **independent** sources within
a tight window ("confirmed by multiple sources").

This module clusters recent near-duplicate headlines and ranks the resulting
stories by how many distinct sources are reporting them. It is purely
read-only over ``articles.db`` and writes nothing, so it cannot perturb the
live scoring pipeline, the ``ai_score`` / ``ml_score`` separation, or
``score_source``. It can be run ad hoc as a CLI digest, or imported by the
briefing / alert path later to add a corroboration tier.

Backtest isolation: the SQL pull applies the canonical ``_LIVE_ONLY_CLAUSE``
imported from ``article_store`` (kept in sync by import, not copy), so
``backtest://`` URLs and ``backtest_*`` / ``opus_annotation*`` sources can
never inflate a corroboration count or surface as "breaking".
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

# Tokens that carry no story identity — dropped before similarity so that
# "Fed cuts rates" and "The Fed has cut rates" collapse to the same story.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "has", "have", "had", "as",
    "at", "by", "from", "with", "its", "it", "this", "that", "after",
    "amid", "over", "into", "up", "down", "new", "says", "say", "said",
    "report", "reports", "reported", "breaking", "update", "live", "watch",
    "exclusive", "video", "photos", "u", "s", "us",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Two headlines belong to the same story when their token sets overlap at or
# above this Jaccard ratio. 0.6 is deliberately high: syndicated copies of one
# wire story share ~all content tokens, while two genuinely different stories
# about the same ticker share only the ticker.
DEFAULT_JACCARD = 0.6


def _normalize(title: str) -> frozenset[str]:
    """Lowercase → alphanumeric tokens → drop stopwords and 1-char tokens."""
    toks = {
        t for t in _TOKEN_RE.findall(title.lower())
        if len(t) > 1 and t not in _STOPWORDS
    }
    return frozenset(toks)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _parse_ts(value: str):
    """Best-effort parse of ``first_seen`` (ISO) or ``published`` (RFC822/ISO).
    Returns an aware UTC datetime or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        try:
            dt = parsedate_to_datetime(value)
        except Exception:
            return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _domain(url: str) -> str:
    try:
        net = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return net[4:] if net.startswith("www.") else net


def corroborated_breaking(
    conn: sqlite3.Connection | None = None,
    hours: float = 3.0,
    min_sources: int = 3,
    jaccard: float = DEFAULT_JACCARD,
    now: datetime | None = None,
) -> list[dict]:
    """Return stories corroborated by ``>= min_sources`` distinct sources
    within the last ``hours``, strongest corroboration first.

    Args:
        conn: an open SQLite connection (used by tests with in-memory DBs).
            When None, a fresh **read-only** connection to the live
            ``articles.db`` is opened and closed here.
        hours: look-back window measured against ``first_seen``.
        min_sources: minimum number of *distinct* ``source`` values for a
            cluster to count as corroborated.
        jaccard: headline token-set similarity threshold for clustering.
        now: override "current time" (tests); defaults to UTC now.

    Each returned dict:
        title            representative (longest) headline of the cluster
        sources          sorted list of distinct source tags reporting it
        source_count     len(sources) — the corroboration strength
        domain_count     distinct URL domains (independent outlets)
        article_count    total clustered articles (incl. exact syndications)
        first_seen       earliest first_seen in the cluster (ISO)
        latest_seen      latest first_seen in the cluster (ISO)
        span_minutes     minutes between first and last sighting
        max_ai_score     highest ai_score across the cluster
        max_urgency      highest urgency flag across the cluster
    """
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=hours)).isoformat()

    owns_conn = conn is None
    if owns_conn:
        db = _get_db_path()
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT title, source, url, ai_score, urgency, first_seen "
            "FROM articles "
            f"WHERE first_seen >= ? AND {_LIVE_ONLY_CLAUSE} "
            "ORDER BY first_seen ASC",
            (cutoff,),
        ).fetchall()
    finally:
        if owns_conn:
            conn.close()

    # Greedy single-pass clustering. Recent-window article counts are bounded
    # (a few hundred to low thousands), so O(n * clusters) is fine and avoids
    # pulling in a clustering dependency.
    clusters: list[dict] = []
    for title, source, url, ai_score, urgency, first_seen in rows:
        title = (title or "").strip()
        if not title:
            continue
        # Defensive: re-check the window in Python in case a row's first_seen
        # is stored in an off-format the lexical SQL compare mis-ordered.
        ts = _parse_ts(first_seen)
        if ts is not None and ts < (now - timedelta(hours=hours)):
            continue

        toks = _normalize(title)
        if not toks:
            continue

        best = None
        best_sim = 0.0
        for cl in clusters:
            sim = _jaccard(toks, cl["_tokens"])
            if sim >= jaccard and sim > best_sim:
                best, best_sim = cl, sim

        if best is None:
            clusters.append({
                "_tokens": toks,
                "_titles": [title],
                "sources": {source or ""},
                "domains": {_domain(url or "")},
                "article_count": 1,
                "first_seen": first_seen,
                "latest_seen": first_seen,
                "max_ai_score": float(ai_score or 0.0),
                "max_urgency": int(urgency or 0),
            })
        else:
            best["_titles"].append(title)
            best["sources"].add(source or "")
            best["domains"].add(_domain(url or ""))
            best["article_count"] += 1
            # Rows arrive ASC by first_seen, so only latest needs updating;
            # guard anyway in case a caller passes an unsorted connection.
            if first_seen and first_seen < best["first_seen"]:
                best["first_seen"] = first_seen
            if first_seen and first_seen > best["latest_seen"]:
                best["latest_seen"] = first_seen
            best["max_ai_score"] = max(best["max_ai_score"], float(ai_score or 0.0))
            best["max_urgency"] = max(best["max_urgency"], int(urgency or 0))
            # Grow the cluster's token set so later near-dupes still match the
            # accreted story rather than only its seed headline.
            best["_tokens"] = best["_tokens"] | toks

    out: list[dict] = []
    for cl in clusters:
        sources = sorted(s for s in cl["sources"] if s)
        if len(sources) < min_sources:
            continue
        domains = sorted(d for d in cl["domains"] if d)
        rep = max(cl["_titles"], key=len)
        start = _parse_ts(cl["first_seen"])
        end = _parse_ts(cl["latest_seen"])
        span_min = round((end - start).total_seconds() / 60.0, 1) if start and end else 0.0
        out.append({
            "title": rep,
            "sources": sources,
            "source_count": len(sources),
            "domain_count": len(domains),
            "article_count": cl["article_count"],
            "first_seen": cl["first_seen"],
            "latest_seen": cl["latest_seen"],
            "span_minutes": span_min,
            "max_ai_score": round(cl["max_ai_score"], 2),
            "max_urgency": cl["max_urgency"],
        })

    # Strongest corroboration first; break ties by tightest burst (a story
    # confirmed across N sources in 5 min beats the same N over 3 h).
    out.sort(key=lambda c: (-c["source_count"], c["span_minutes"]))
    return out


def format_digest(stories: list[dict], limit: int = 15) -> str:
    """Plain-text digest of the top corroborated stories."""
    if not stories:
        return "No multi-source corroborated stories in window."
    lines = []
    for s in stories[:limit]:
        srcs = ", ".join(s["sources"][:6])
        if len(s["sources"]) > 6:
            srcs += f" +{len(s['sources']) - 6} more"
        lines.append(
            f"[{s['source_count']}x src / {s['domain_count']} domains "
            f"in {s['span_minutes']:.0f}m | ai={s['max_ai_score']}] "
            f"{s['title']}\n    via {srcs}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual CLI
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--min-sources", type=int, default=3)
    ap.add_argument("--jaccard", type=float, default=DEFAULT_JACCARD)
    args = ap.parse_args()
    stories = corroborated_breaking(
        hours=args.hours, min_sources=args.min_sources, jaccard=args.jaccard
    )
    print(f"{len(stories)} corroborated stories "
          f"(>= {args.min_sources} sources, last {args.hours}h):\n")
    print(format_digest(stories))
    if stories:
        top_src = Counter()
        for s in stories:
            top_src.update(s["sources"])
        print("\nMost-corroborating sources:",
              ", ".join(f"{k}={v}" for k, v in top_src.most_common(8)))
