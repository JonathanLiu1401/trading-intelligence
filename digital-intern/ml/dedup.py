"""Order-independent near-duplicate detection for syndicated articles.

A breaking story is carried within minutes by GDELT, Reuters, Yahoo, Finnhub,
Google News and a dozen RSS feeds. ``storage.article_store`` only dedups on the
exact ``sha256(url || title)`` id, so every syndicated copy lands as its own
row, gets scored independently, and inflates the feed the news analyst reads.

``watchers.alert_dedup`` already collapses the *urgent-alert batch*, but with an
exact **first-8-token prefix signature**: fast, yet blind to word reordering
("Apple beats Q2 expectations" vs "Q2 expectations beaten by Apple") and only
applied at alert time. This module is the complementary general-purpose
detector — a tunable token-set **Jaccard** similarity over normalized titles —
suitable anywhere an order-independent fuzzy match is wanted (ingestion-side
collapse, briefing input, dashboard feed).

Design notes:
  * Pure functions over the standard article-dict shape
    (``{"title": str, "ai_score": float, ...}``). No DB, no LLM, no network,
    no mutation of the caller's list or dicts — safe to call on a read-only
    snapshot and trivially testable.
  * Title normalization reuses the same wire-prefix / source-attribution
    shapes as ``watchers.alert_dedup`` so the two stay behaviourally aligned.
  * Backtest isolation is unaffected: this never reads or writes the DB and
    operates only on rows the caller already selected.

Integration is intentionally left to the caller (e.g. a future ingestion-side
collapse in ``daemon.py`` or a briefing pre-filter). ``dedupe_articles`` is a
drop-in: ``rows = dedupe_articles(rows)`` before scoring/ranking.
"""
from __future__ import annotations

import re
from typing import Any

# --- title normalization ---------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")

# Leading wire-service editorial markers ("UPDATE 2-", "RPT-", "BREAKING:"),
# possibly stacked ("RPT-UPDATE 2-"). Anchored + whitelisted so a real all-caps
# headline word is never consumed. Mirrors watchers.alert_dedup._WIRE_PREFIX.
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
# Trailing source attribution: "...blowout - Reuters", "...blowout | Bloomberg".
_TRAIL_SEP = re.compile(r"\s+[-|–—]\s+[\w][\w .,&'/]*$")
# Trailing attribution parenthetical: "...blowout (Bloomberg)".
_TRAIL_PAREN = re.compile(r"\s*\([^()]*\)\s*$")

# Minimal, deterministic stopword set — function words that carry no
# story-identifying signal and only dilute the Jaccard ratio.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "by",
        "at", "as", "is", "are", "be", "was", "were", "with", "from", "after",
        "over", "amid", "into", "its", "it", "that", "this", "than", "but",
    }
)

_MIN_TOKEN_LEN = 2


def normalize_title(title: str | None) -> str:
    """Lowercased headline with wire prefix and trailing source attribution
    stripped and whitespace collapsed. ``None``/blank -> ``""``."""
    if not title or not title.strip():
        return ""
    s = _WIRE_PREFIX.sub("", title).strip()
    # Attribution can stack as "Headline - Reuters (Bloomberg)"; peel both.
    for _ in range(2):
        s = _TRAIL_PAREN.sub("", s)
        s = _TRAIL_SEP.sub("", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def title_tokens(title: str | None, *, min_len: int = _MIN_TOKEN_LEN) -> set[str]:
    """Set of significant tokens: normalized, alphanumeric, stopwords and
    sub-``min_len`` tokens removed. Order-independent by construction."""
    return {
        t
        for t in _WORD.findall(normalize_title(title))
        if len(t) >= min_len and t not in _STOPWORDS
    }


# --- similarity ------------------------------------------------------------


def jaccard_similarity(a: set, b: set) -> float:
    """|a ∩ b| / |a ∪ b|. Two empty sets -> 0.0 (no ZeroDivisionError, and
    two contentless titles must not be treated as a match)."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def is_near_duplicate(
    title_a: str | None,
    title_b: str | None,
    *,
    threshold: float = 0.6,
) -> bool:
    """True iff the two titles' token sets meet ``threshold`` Jaccard
    similarity. Empty/``None`` titles are never duplicates of anything."""
    ta, tb = title_tokens(title_a), title_tokens(title_b)
    if not ta or not tb:
        return False
    return jaccard_similarity(ta, tb) >= threshold


# --- batch dedup -----------------------------------------------------------


def dedupe_articles(
    articles: list[dict[str, Any]],
    *,
    threshold: float = 0.6,
    title_key: str = "title",
    score_key: str = "ai_score",
) -> list[dict[str, Any]]:
    """Collapse near-duplicate articles, keeping one representative per cluster.

    Greedy single-pass clustering: each article joins the first existing
    cluster whose anchor (first member) is a near-duplicate, else opens a new
    cluster. The surviving representative of a cluster is its highest
    ``score_key`` member (missing/non-numeric -> 0.0; ties keep the earliest).
    Survivors are emitted in order of first cluster appearance.

    The caller's list and dicts are never mutated; the returned list holds the
    original dict objects.
    """

    def _score(art: dict[str, Any]) -> float:
        try:
            return float(art.get(score_key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    clusters: list[dict[str, Any]] = []  # {anchor_tokens, best, best_score}
    for art in articles:
        toks = title_tokens(art.get(title_key))
        placed = False
        if toks:
            for cl in clusters:
                anchor = cl["anchor_tokens"]
                if anchor and jaccard_similarity(toks, anchor) >= threshold:
                    if _score(art) > cl["best_score"]:
                        cl["best"], cl["best_score"] = art, _score(art)
                    placed = True
                    break
        if not placed:
            clusters.append(
                {"anchor_tokens": toks, "best": art, "best_score": _score(art)}
            )
    return [cl["best"] for cl in clusters]
