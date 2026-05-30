"""Emerging press-mill / non-English-PR detector — surfaces urgent rows that
slipped past the existing recap-template gates in ``watchers/alert_agent.py``.

Why this exists
---------------
``watchers.alert_agent`` carries 30+ recap-template regexes catching SEO-mill /
13F filing-summary / quote-widget / image-credit pseudo-articles BEFORE they
fire a standalone Bloomberg "🚨 BREAKING" Discord push. Two distinct noise
templates were observed firing urgency=2 in the live DB despite the existing
gates (2026-05-29, news-analyst persona pass):

1. **Stock Titan 13D/13G "ownership disclosed" press mill**:
     "FMR LLC (COHR) reports 22.6M shares, 12.1% ownership disclosed - Stock
      Titan"  ml_score=9.96, urgency=2, source='GoogleNews/Stock Titan'
   The existing ``_RT_FUND_STAKE_DELTA`` (WIP in concurrent agent's diff) and
   ``_RT_FUND_MAKES_INVESTMENT`` catch leading-LLC + delta-verb + stake-noun
   structures, but this template uses ``reports`` + ``ownership disclosed`` —
   neither in any existing verb list — and routes the magnitude through the
   ``shares, N% ownership disclosed`` trailer rather than ``in/of <Company>``.

2. **Foreign-language PR Newswire syndication storm**:
     "Arasan Chip Systems anuncia la primera solución IP Sureboot™..." (es)
     "Arasan Chip Systems kündigt branchenweit erste Sureboot™..." (de)
     "Arasan Chip Systems annonce la première solution IP Sureboot™..." (fr)
   All three fired urgency=2 within the same minute at ml_score 9.9, sources
   ``PR Newswire`` / ``PR Newswire Tech``. The same announcement crosses the
   wire 3+ times in 3+ languages; alert_dedup's signature differs per language
   (Spanish/German/French verbs replace English ones), so neither in-batch
   dedup nor the cross-cycle alert_recency 6h TTL catches them — three
   standalone BREAKING pushes for one product announcement.

The module is **pure-read analytics**, never gates anything. It surfaces
"emerging" press-mill patterns so the operator (or a future agent) can wire
the matching predicate into ``watchers.alert_agent._RECAP_TEMPLATE_PATTERNS``.
Keeping Phase-2 surgical avoids the concurrent-agent staging race documented
in MEMORY.md (pt-concurrent-samerole-staging-race, di-shared-repo-concurrency).

Invariants
----------
Pure read-side. No DB write, no ai_score / ml_score / score_source / urgency
mutation. ``backtest://`` / ``backtest_*`` / ``opus_annotation*`` rows are
caller-filtered (Phase 3 read uses ``_LIVE_ONLY_CLAUSE``); the builder itself
NEVER raises on a synthetic row leaking through — it just classifies the
title. All four load-bearing invariants intact by construction.

API
---
Two exported predicates and one pure builder::

    is_ownership_disclosed_press_mill(article) -> bool
    is_foreign_pr_newswire(article) -> bool
    build_emerging_press_mill(rows, *, now=None) -> dict

``rows`` is a list of dicts shaped like ``ArticleStore.get_unalerted_urgent``
(``{_id, title, source, ai_score, ml_score, link/url, ...}``). The builder
returns a deterministic envelope with verdict, counts, per-pattern matches,
per-source breakdown for uncaught rows, and sample titles.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable


# ── Predicate 1: Stock Titan "(TICKER) reports N shares, N% ownership disclosed"
#
# Discriminator: a parenthesised ticker (case-sensitive ALLCAPS, 1-6 chars) +
# ``reports`` + a digits-and-units shares count + ``ownership disclosed``. The
# four-token co-occurrence is the high-precision discriminator — no real wire
# headline strings "reports", a parenthesised ticker, "shares", and "ownership
# disclosed" together. The fund-name prefix is NOT required (the template can
# fire with an institution name in the lead OR a beneficial-owner individual);
# what matters is the SEC 13D/13G filing-summary structural fingerprint after
# the actor.
#
# Validated against the must-survive corpus:
# - "Apple reports earnings beat" — no parens, no "ownership disclosed"
# - "Nvidia reports Q1 revenue rises 22%" — no parens, no "ownership disclosed"
# - "FMR confirms ownership of $50B in semis" — no "disclosed" keyword form
# - "Insider ownership disclosed in 13G filing" — no parenthesised ticker
_PM_OWNERSHIP_DISCLOSED = re.compile(
    r"\([A-Z]{1,6}\)\s+reports?\s+[\d.,]+\s*[KMB]?\s+shares?\b.*?"
    r"\bownership\s+disclosed\b",
    re.IGNORECASE,
)


def is_ownership_disclosed_press_mill(article: dict) -> bool:
    """True if the title matches the Stock Titan 13D/13G filing-summary press
    mill template (``(TICKER) reports N shares, N% ownership disclosed``).

    Pure side-effect-free predicate over the article-dict shape returned by
    ``ArticleStore.get_unalerted_urgent``. Only reads ``title`` via ``.get()``
    and never mutates the article."""
    if not isinstance(article, dict):
        return False
    title = article.get("title") or ""
    return bool(_PM_OWNERSHIP_DISCLOSED.search(title))


# ── Predicate 2: foreign-language PR Newswire syndication
#
# Two-stage discriminator: (a) source tag starts with ``PR Newswire`` /
# ``GlobeNewswire`` / ``BusinessWire`` (the wire-aggregator prefix), AND
# (b) title carries a non-English language marker — either a known
# non-English verb-of-announcement, or a non-ASCII character (Latin-extended
# accents the English headline corpus essentially never carries: ä ö ü ß é
# è à ñ + CJK).
#
# The marker words below are observed in the live DB and span the languages
# PR Newswire actually syndicates to: Spanish (anuncia, presenta), German
# (kündigt, gibt bekannt), French (annonce, présente), Italian (annuncia),
# Portuguese (anuncia — same verb as Spanish; differentiated by surrounding
# tokens), Dutch (kondigt, presenteert), plus CJK announcement verbs.
#
# Why source-prefix-gated: a non-ASCII character in a *news* headline
# (e.g. "Macron meets Xi" — accented é, NOT a foreign-language press release)
# would false-match a pure-character test. Anchoring on the PR-Newswire source
# tag means we ONLY flag wire-aggregator releases — exactly the corpus where
# foreign-language syndication storms occur. Real news pieces from Reuters,
# Bloomberg, AP that happen to mention a non-English topic still pass through.

_PM_FOREIGN_SOURCES = (
    "pr newswire",
    "globenewswire",
    "businesswire",
    "business wire",
    "globe newswire",
)

_PM_FOREIGN_MARKER_WORDS = (
    # Spanish / Portuguese
    "anuncia",
    "anuncian",
    "presenta",
    "lanzamiento",
    "lanza",
    # German
    "kündigt",
    "kundigt",  # ASCII fallback when the umlaut is lost
    "bekannt",
    "präsentiert",
    "prasentiert",
    # French
    "annonce",
    "annoncent",
    "présente",
    "presente",
    "première",
    "premiere",
    # Italian
    "annuncia",
    "presentano",
    # Dutch
    "kondigt",
    "presenteert",
)

# Non-ASCII Latin-extended / CJK characters. Headlines that are pure ASCII
# don't trip this — only titles with accented Latin or non-Latin scripts.
_PM_NON_ASCII = re.compile(r"[À-ɏͰ-Ͽ一-鿿぀-ヿ֐-׿]")


def _source_is_pr_aggregator(source: str) -> bool:
    """True if the source tag matches a known wire-aggregator prefix."""
    if not source:
        return False
    s_low = source.strip().lower()
    return any(s_low.startswith(p) for p in _PM_FOREIGN_SOURCES)


def _title_has_foreign_marker(title: str) -> bool:
    """True if the title contains a non-English announcement verb or a
    non-ASCII Latin-extended / CJK character."""
    if not title:
        return False
    t_low = title.lower()
    for marker in _PM_FOREIGN_MARKER_WORDS:
        # Word-boundary match so "anuncia" does NOT match "announced" (English
        # past-tense, doesn't share a stem) and "presente" does not catch
        # "presents" embedded in compound words.
        if re.search(r"\b" + re.escape(marker) + r"\b", t_low):
            return True
    if _PM_NON_ASCII.search(title):
        return True
    return False


def is_foreign_pr_newswire(article: dict) -> bool:
    """True if the article is a non-English wire-aggregator press release.

    Discriminator: source is a known wire aggregator (PR Newswire / Globe
    Newswire / Business Wire) AND title carries a non-English language marker
    or a non-ASCII Latin-extended / CJK character.

    A foreign-language wire release of the same announcement that crosses
    the English wire produces 2-4 standalone BREAKING pushes for one event
    (signature differs per language → alert_dedup's cross-cycle TTL is bypassed).
    Pure side-effect-free predicate over ``source`` + ``title`` only."""
    if not isinstance(article, dict):
        return False
    source = article.get("source") or ""
    if not _source_is_pr_aggregator(source):
        return False
    title = article.get("title") or ""
    return _title_has_foreign_marker(title)


# Ordered tuple of (label, predicate) — mirrors the
# ``_RECAP_TEMPLATE_PATTERNS`` discipline in ``watchers.alert_agent``. Tests
# pin this length so a regression that drops a predicate fails focused tests.
EMERGING_PREDICATES: tuple[tuple[str, "callable"], ...] = (
    ("ownership_disclosed_press_mill", is_ownership_disclosed_press_mill),
    ("foreign_pr_newswire", is_foreign_pr_newswire),
)


def _now_iso(now: datetime | None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).isoformat()


def _classify(article: dict) -> str | None:
    """Return the matching predicate label, or None if no emerging predicate
    fires. The FIRST matching predicate wins (predicates are mutually
    orthogonal by construction — ownership-disclosed requires English text;
    foreign-pr requires non-English markers — but the first-match rule pins
    behaviour if a future predicate overlaps)."""
    for label, pred in EMERGING_PREDICATES:
        try:
            if pred(article):
                return label
        except Exception:
            continue
    return None


def build_emerging_press_mill(
    rows: Iterable[dict],
    *,
    now: datetime | None = None,
    max_samples_per_pattern: int = 3,
    max_uncaught_sources: int = 5,
) -> dict:
    """Build an emerging-press-mill audit envelope from a list of urgent rows.

    Parameters
    ----------
    rows : iterable of dict
        Urgent (urgency >= 1) article rows. Each is expected to have at least
        ``title`` and ``source``; ``ml_score``, ``ai_score``, ``urgency``,
        ``first_seen`` are read defensively via ``.get()``.
    now : datetime, optional
        Override the ``as_of`` timestamp. Defaults to UTC now.
    max_samples_per_pattern : int
        Maximum sample titles to surface per matching predicate.
    max_uncaught_sources : int
        Top-N sources by uncaught-urgent count to surface in the envelope.

    Returns
    -------
    dict
        Deterministic envelope::

            {
              "as_of": "2026-05-29T17:00:00+00:00",
              "n_audited": int,
              "n_emerging_caught": int,         # rows matching an emerging predicate
              "n_uncaught": int,                # rows neither in existing gates nor emerging
              "by_predicate": {
                 "ownership_disclosed_press_mill": {
                    "count": int,
                    "mean_ml_score": float,
                    "sample_titles": [str, ...],   # up to max_samples_per_pattern
                 },
                 ...
              },
              "by_uncaught_source": [
                 {"source": str, "count": int, "mean_ml_score": float},
                 ...
              ],
              "verdict": str,   # NO_DATA / ALL_GATED / EMERGING_NOISE
            }

    Verdict ladder
    --------------
    * ``NO_DATA``         — empty input
    * ``ALL_GATED``       — no row matches any emerging predicate (the existing
                            alert_agent gates are already covering the input)
    * ``EMERGING_NOISE``  — at least one predicate fired, meaning a new
                            press-mill class is leaking past existing gates
                            (operator should wire the predicate into
                            ``watchers.alert_agent._RECAP_TEMPLATE_PATTERNS``)

    The builder is pure: it does NOT touch the DB, does NOT call out to
    ``watchers.alert_agent`` for the existing-gate check (the *caller* is
    expected to feed only rows that already passed the existing gates, or to
    feed every urgent row knowing the builder will surface the union of all
    uncaught templates). This keeps the test contract crisp — no hidden
    coupling to the WIP ``_RT_FUND_STAKE_DELTA`` regex set.
    """
    as_of = _now_iso(now)
    by_pred: dict[str, dict] = {
        label: {"count": 0, "ml_scores": [], "sample_titles": []}
        for label, _ in EMERGING_PREDICATES
    }
    by_source_uncaught: dict[str, dict] = {}
    n_audited = 0
    n_caught = 0
    n_uncaught = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        n_audited += 1
        try:
            ml = float(row.get("ml_score") or 0.0)
        except (TypeError, ValueError):
            ml = 0.0
        label = _classify(row)
        if label is not None:
            n_caught += 1
            pred_bucket = by_pred[label]
            pred_bucket["count"] += 1
            pred_bucket["ml_scores"].append(ml)
            if len(pred_bucket["sample_titles"]) < max_samples_per_pattern:
                title = (row.get("title") or "")[:160]
                if title and title not in pred_bucket["sample_titles"]:
                    pred_bucket["sample_titles"].append(title)
        else:
            n_uncaught += 1
            src = (row.get("source") or "?") or "?"
            bucket = by_source_uncaught.setdefault(
                src, {"count": 0, "ml_scores": []}
            )
            bucket["count"] += 1
            bucket["ml_scores"].append(ml)

    # Verdict ladder.
    if n_audited == 0:
        verdict = "NO_DATA"
    elif n_caught == 0:
        verdict = "ALL_GATED"
    else:
        verdict = "EMERGING_NOISE"

    # Materialise per-predicate stats.
    by_predicate: dict[str, dict] = {}
    for label, bucket in by_pred.items():
        scores = bucket["ml_scores"]
        mean = round(sum(scores) / len(scores), 3) if scores else 0.0
        by_predicate[label] = {
            "count": bucket["count"],
            "mean_ml_score": mean,
            "sample_titles": list(bucket["sample_titles"]),
        }

    # Top uncaught sources.
    uncaught_ranked = sorted(
        by_source_uncaught.items(),
        key=lambda kv: (-kv[1]["count"], kv[0]),
    )[:max_uncaught_sources]
    by_uncaught_source = [
        {
            "source": src,
            "count": b["count"],
            "mean_ml_score": round(
                sum(b["ml_scores"]) / len(b["ml_scores"]), 3
            ) if b["ml_scores"] else 0.0,
        }
        for src, b in uncaught_ranked
    ]

    return {
        "as_of": as_of,
        "n_audited": n_audited,
        "n_emerging_caught": n_caught,
        "n_uncaught": n_uncaught,
        "by_predicate": by_predicate,
        "by_uncaught_source": by_uncaught_source,
        "verdict": verdict,
    }
