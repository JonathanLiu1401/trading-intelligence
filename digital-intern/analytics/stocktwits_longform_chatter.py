"""Long-form Stocktwits forum-chatter detector — surfaces stocktwits posts the
existing chatter gate's 50-char cap deliberately misses.

Why this exists
---------------
``watchers.alert_agent._looks_like_stocktwits_chatter`` already gates SHORT
stocktwits forum chatter (``len(title) < _STOCKTWITS_CHATTER_TITLE_MAX = 50``)
when no news-keyword from ``_STOCKTWITS_NEWS_EXIT`` is present. The docstring
on that gate explicitly stops at 50 chars because "the 50-79 char tier is
mixed so we conservatively stop the gate there".

A live audit (2026-05-30, 7d articles.db, news-analyst persona pass) showed
the 50+ char tier is NOT mixed once you also require zero news-keyword:
**92 of 105 stocktwits rows with title length >= 50 and ml_score >= 9 (88%)
carry NO news-keyword from the existing exit regex** — pure forum chatter the
ML urgency head over-scored because the ``$TICKER`` density + held-name
features fire hard on any "$<HELD> ..." lead regardless of content.

Sample live-DB rows (urgency=2, ml_score >= 9.9, len >= 50) that the existing
50-char cap deliberately misses but a long-form gate would catch::

    $QQQ $SPY $SOXX $SOXL $TQQQ
    After latest price cuts Deepseek upto 34x cheaper than Cla...
        (176 chars, ml_score 10.0)

    $IONQ $QBTS $QUBT $RGTI $XRP.X nobody wants to buy your dumb shtcoin.
    Go knock on someone...
        (101 chars, ml_score 9.9)

    $MU joining the AI momentum is exactly why I want better alerts. Memory
    demand d...
        (200 chars, ml_score 9.9)

These all reach ``urgency=1`` and get fetched by the alert worker, decompressed,
fed through ``_fmt`` for a Sonnet call — only to be suppressed at the
``_filter_low_authority_lone`` gate (stocktwits cred=0.30 < 0.45). The
push is correctly NOT fired, but the rows still:

  * inflate the urgent backlog by occupying slots in the
    ``get_unalerted_urgent`` ``LIMIT 50`` set
  * pollute ``urgent_score_distribution`` calibration metrics into
    BORDERLINE_HEAVY when the actual Discord output is clean
  * burn ML inference cycles on rows whose ground-truth label is noise
  * displace genuine signals from the briefing's TOP SIGNALS pool

This module is **pure-read analytics**: it surfaces "is the long-form chatter
gap material right now?" so an operator (or a future agent) can wire the
predicate into the three existing pre-floor surfaces in lockstep:

  * ``storage.article_store.prefloor_pseudo_articles`` (ML pre-floor)
  * ``watchers.urgency_scorer.score_batch`` (LLM pre-floor)
  * ``watchers.alert_agent._looks_like_stocktwits_chatter`` (alert-side
    defense — by extending the existing helper, not adding a second one)

SSOT discipline
---------------
``_STOCKTWITS_NEWS_EXIT`` is imported VERBATIM from ``watchers.alert_agent``
rather than redeclared here. The AGENTS.md history is explicit on this
lockstep discipline (the ``_QW_IMAGE_CREDIT`` triple-gate, the ``_RT_*``
recap families, the documented enforcement test
``test_alert_and_briefing_recap_tuples_have_same_length``). A future agent
widening the news-exit list on the alert side automatically extends THIS
predicate's must-survive corpus — there is no other path to that consistency.

Invariants
----------
Pure read-side. No DB write, no ai_score / ml_score / score_source / urgency
mutation. ``backtest://`` / ``backtest_*`` / ``opus_annotation*`` rows are
caller-filtered (the canonical live-only-clause read at the production
callsite); the builder itself NEVER raises on a synthetic row leaking
through — it just classifies the title. All four load-bearing invariants
intact by construction.

API
---
One exported predicate + one constant tuple + one pure builder::

    is_longform_stocktwits_chatter(article) -> bool
    LONGFORM_PREDICATES                            # ((label, predicate), ...)
    build_longform_chatter_report(rows, *, now=None,
                                  max_samples_per_pattern=5,
                                  max_uncaught_sources=5) -> dict

``rows`` is a list of dicts shaped like ``ArticleStore.get_unalerted_urgent``
(``{_id, title, source, ai_score, ml_score, ...}``). The builder returns a
deterministic envelope with verdict, counts, per-predicate matches with sample
titles, and per-source breakdown for non-stocktwits rows.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable

# SSOT — the same word-bounded news-keyword regex the alert-side chatter gate
# uses (a "this is real news, do not gate" escape valve). Imported verbatim so
# a future widening on the alert side (extra keywords) automatically extends
# the must-survive corpus for THIS predicate. Same anti-drift discipline as
# the recap-template byte-identical twins documented in AGENTS.md.
from watchers.alert_agent import _STOCKTWITS_NEWS_EXIT


# ── Tunables ────────────────────────────────────────────────────────────────
# Floor matches the existing ``_STOCKTWITS_CHATTER_TITLE_MAX`` in
# ``watchers.alert_agent`` so this gate sits STRICTLY ABOVE the short-form
# gate's coverage (titles 0..49 chars → existing gate; 50..299 chars → this
# gate when no news-keyword AND no shared URL).
LONGFORM_MIN_TITLE_LEN = 50
# Ceiling chosen to avoid catching very-long real news headlines syndicated
# through stocktwits — empirically, the 7d ml>=9 set has no chatter post above
# ~200 chars (the longest samples are 176/200/200). 300 leaves headroom for
# the rare longer chatter while excluding multi-paragraph real-news blocks.
LONGFORM_MAX_TITLE_LEN = 300

# Source-tag prefix that identifies the stocktwits raw user-stream collector
# family (``stocktwits``, ``stocktwits_trending``, ...). The structured
# sentiment-digest source ``stocktwits/sentiment`` carries real signal and is
# excluded EXPLICITLY — same discipline as the alert-side gate.
_STOCKTWITS_PREFIX = "stocktwits"
_STOCKTWITS_SENTIMENT = "stocktwits/sentiment"

# URLs embedded in a stocktwits title generally mean the user shared an
# external article (Bloomberg, Investing.com, marketwirenews — see the
# has-news survivors in the 7d audit) — even if no news-keyword from the
# exit regex matched. Treat a title carrying an http(s) URL as "user shared
# real content" and survive it; one extra precision step on top of the
# news-keyword exit.
_TITLE_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _source_is_raw_stocktwits(source: str) -> bool:
    """True if ``source`` is a raw-stream stocktwits collector tag (any
    case-insensitive substring ``stocktwits`` that is NOT the structured
    sentiment digest). Matches the alert-side gate convention so the two
    surfaces never disagree on WHICH source-tags count as raw stocktwits."""
    if not source:
        return False
    s_low = source.strip().lower()
    if _STOCKTWITS_PREFIX not in s_low:
        return False
    return s_low != _STOCKTWITS_SENTIMENT


def is_longform_stocktwits_chatter(article: dict) -> bool:
    """True for a LONG raw-stocktwits forum-chatter row the existing
    short-form gate's 50-char cap deliberately misses.

    Discriminator (all five must hold — keeps precision high; the predicate
    is the candidate gate, not just an audit):

      1. ``source`` is a raw stocktwits stream (case-insensitive substring
         ``stocktwits`` AND not the structured ``stocktwits/sentiment`` digest).
      2. ``LONGFORM_MIN_TITLE_LEN <= len(title) < LONGFORM_MAX_TITLE_LEN``
         (above the existing gate, below the real-news-block ceiling).
      3. ``title`` does NOT match ``_STOCKTWITS_NEWS_EXIT`` (the SSOT
         news-keyword escape valve from ``watchers.alert_agent``).
      4. ``title`` does NOT carry an embedded ``http(s)://`` URL — a shared
         link generally means the user pointed at a real article (live
         evidence: 4 of the 13 ``has-news`` survivors in the 7d audit
         carried bloomberg.com / investing.com / marketwirenews URLs).

    Pure side-effect-free predicate over the article-dict shape returned by
    ``ArticleStore.get_unalerted_urgent``. Only reads ``source`` / ``title``
    via ``.get()`` and never mutates the article. Returns False on a malformed
    article (non-dict, missing keys) — never raises.
    """
    if not isinstance(article, dict):
        return False
    source = article.get("source") or ""
    if not _source_is_raw_stocktwits(source):
        return False
    title = (article.get("title") or "")
    if not title:
        return False
    n = len(title)
    if n < LONGFORM_MIN_TITLE_LEN or n >= LONGFORM_MAX_TITLE_LEN:
        return False
    if _STOCKTWITS_NEWS_EXIT.search(title):
        return False
    if _TITLE_URL_RE.search(title):
        return False
    return True


# Ordered tuple of (label, predicate). Mirrors
# ``analytics.emerging_press_mill.EMERGING_PREDICATES`` and the
# ``watchers.alert_agent._RECAP_TEMPLATE_PATTERNS`` discipline — tests pin
# this length so a regression that drops a predicate fails a focused test.
# Length intentionally 1 (a single predicate today); growing this tuple is
# how a future agent layers additional long-form chatter sub-templates while
# keeping the audit per-predicate bucket counters intact.
LONGFORM_PREDICATES: tuple[tuple[str, "callable"], ...] = (
    ("longform_stocktwits_chatter", is_longform_stocktwits_chatter),
)


def _now_iso(now: datetime | None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).isoformat()


def _classify(article: dict) -> str | None:
    """Return the matching predicate label, or None if no predicate fires.
    FIRST-MATCH wins — pins behaviour deterministically if a future predicate
    overlaps with an existing one."""
    for label, pred in LONGFORM_PREDICATES:
        try:
            if pred(article):
                return label
        except Exception:
            continue
    return None


def build_longform_chatter_report(
    rows: Iterable[dict],
    *,
    now: datetime | None = None,
    max_samples_per_pattern: int = 5,
    max_uncaught_sources: int = 5,
) -> dict:
    """Build a long-form chatter audit envelope from a list of urgent rows.

    Parameters
    ----------
    rows : iterable of dict
        Urgent (urgency >= 1) article rows. Each is expected to have at least
        ``title`` and ``source``; ``ml_score``, ``ai_score`` are read
        defensively via ``.get()``.
    now : datetime, optional
        Override the ``as_of`` timestamp. Defaults to UTC now.
    max_samples_per_pattern : int
        Maximum sample titles to surface per matching predicate.
    max_uncaught_sources : int
        Top-N non-stocktwits sources by uncaught count to surface.

    Returns
    -------
    dict
        Deterministic envelope::

            {
              "as_of": "2026-05-30T03:00:00+00:00",
              "n_audited": int,
              "n_chatter_caught": int,          # rows matching a chatter predicate
              "n_uncaught": int,                # everything else
              "by_predicate": {
                 "longform_stocktwits_chatter": {
                    "count": int,
                    "mean_ml_score": float,
                    "sample_titles": [str, ...],   # up to max_samples_per_pattern
                 },
              },
              "by_uncaught_source": [
                 {"source": str, "count": int, "mean_ml_score": float},
                 ...
              ],
              "verdict": str,
                # NO_DATA / NO_CHATTER / CHATTER_LEAKING_PAST_GATE
            }

    Verdict ladder (mirrors ``emerging_press_mill``'s NO_DATA / ALL_GATED /
    EMERGING_NOISE shape):

      * ``NO_DATA``                   — empty input
      * ``NO_CHATTER``                — no row matches any chatter predicate
        (the alert path's existing gates are already covering the input)
      * ``CHATTER_LEAKING_PAST_GATE`` — at least one predicate fired (the
        short-form chatter gate's 50-char cap is materially leaking right
        now; the operator should consider wiring the predicate into the
        three pre-floor surfaces named at the top of this module)

    Pure: NO DB touch, NO call into alert_agent / urgency_scorer beyond the
    SSOT predicate import. Synthetic rows (``url`` ``backtest://`` /
    ``source`` ``backtest_*`` / ``opus_annotation*``) cannot match the
    predicate (the source check requires ``stocktwits``) but the builder
    never inspects ``url`` / synthetic markers either — caller is responsible
    for live-only filtering via ``_LIVE_ONLY_CLAUSE`` at the read site, the
    same contract every analytics builder in this package follows.
    """
    as_of = _now_iso(now)
    by_pred: dict[str, dict] = {
        label: {"count": 0, "ml_scores": [], "sample_titles": []}
        for label, _ in LONGFORM_PREDICATES
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

    if n_audited == 0:
        verdict = "NO_DATA"
    elif n_caught == 0:
        verdict = "NO_CHATTER"
    else:
        verdict = "CHATTER_LEAKING_PAST_GATE"

    by_predicate: dict[str, dict] = {}
    for label, bucket in by_pred.items():
        scores = bucket["ml_scores"]
        mean = round(sum(scores) / len(scores), 3) if scores else 0.0
        by_predicate[label] = {
            "count": bucket["count"],
            "mean_ml_score": mean,
            "sample_titles": list(bucket["sample_titles"]),
        }

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
        "n_chatter_caught": n_caught,
        "n_uncaught": n_uncaught,
        "by_predicate": by_predicate,
        "by_uncaught_source": by_uncaught_source,
        "verdict": verdict,
    }
