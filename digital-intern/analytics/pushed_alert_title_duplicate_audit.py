"""Pushed-alert title-duplication audit (read-only).

Sibling analytics primitive to ``analytics.pushed_alert_event_concentration``.

``pushed_alert_event_concentration`` answers "of the recent pushes, which
(held-ticker × event-class) cells fired more than once?" via the canonical
alert_recency.db ledger and a small closed-vocabulary keyword set. It is
deliberately scoped to held tickers and a closed event vocabulary so the
verdict is actionable.

This module answers the broader, complementary question: "of all
``urgency=2`` rows in articles.db within the recent window, how many distinct
TITLES fired more than once — regardless of whether the title carries a held
ticker or not?". A duplicate-fire of "Nvidia posts record $81.6B quarter,
unveils $80B buyback - MSN" across 9 different MSN URL variants is the analyst's
single most-cited noise complaint, and it doesn't require a held-ticker tag to
matter — the analyst gets nine pushes, reads them as nine events, then
discovers they were all the same wire copy. ``pushed_alert_event_concentration``
catches it only if the title contains a recognised held-ticker token; rows
that paraphrase the underlying event with a generic name ("MSN", "Reuters",
"Bloomberg") fall outside its lens.

Live evidence (2026-05-29 24h pull): same Barron's "Micron Faces New Threat
From Samsung's Memory Chip for AI" fired a BREAKING push 3× via referrer-param
URL variants (?siteid=yhoof2&ypt=1 vs ?mod=md_home_pan_m vs the bare URL),
and "NVIDIA projects $91B Q2 revenue while outlining $80B buyback and a
$0.25 quarterly dividend - MSN" fired 10× in 7d across MSN syndication. The
recent ``fix(url-canonicalizer): strip mod/siteid/ypt referrer params`` closes
the first failure mode; this audit gives the analyst the calibration view to
confirm the fix is holding and to surface remaining patterns (cross-source
title paraphrases, MSN-internal id drift) the URL canonicaliser cannot catch.

Sibling discipline:
  * Pure read-only over articles.db (no LLM, no network, no DB write).
  * ``_LIVE_ONLY_CLAUSE`` applied so synthetic backtest/opus rows never inflate
    the audit — same string the storage layer uses, duplicated as a constant
    rather than imported (the module deliberately does NOT pull the
    ArticleStore writer graph).
  * Backtest isolation: load-bearing invariant intact by construction.
  * ml_score / ai_score / score_source / urgency state machine: untouched
    (read-only).

Title normalization: lowercase + collapse internal whitespace runs. Aggressive
enough to collapse minor spacing artefacts but conservative enough that a
genuine paraphrase ("NVIDIA Q1 beat" vs "Nvidia Q1 Beats") still groups as
DISTINCT titles — paraphrase-style duplicates are
``watchers.alert_recency.partition_paraphrase_alerted``'s responsibility, and
double-counting them here would obscure exact-title duplicates the analyst
most wants surfaced.

Run standalone::

    python3 -m analytics.pushed_alert_title_duplicate_audit          # JSON report, 24h window
    python3 -m analytics.pushed_alert_title_duplicate_audit --hours 168  # 7-day window
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Optional


# Mirrors storage.article_store._LIVE_ONLY_CLAUSE — duplicated as a string
# constant rather than imported because this module deliberately does NOT
# pull the ArticleStore writer graph (we operate on a read-only connection).
# Pinned by ``test_live_only_clause_in_sync_with_storage`` in the test file.
LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)


# Verdict thresholds. Tuned to the live-evidence pattern: in a healthy
# window, exact-title duplicates should be <5% of pushes (the
# alert_recency exact-signature gate catches most). Above 15% means a
# canonicalisation gap (URL referrer-param variants, source-tag drift)
# or a paraphrase that the existing gates' Jaccard threshold misses.
DUPLICATION_RATE_HEAVY_PCT = 15.0
DUPLICATION_RATE_LIGHT_PCT = 3.0
MIN_PUSHES_FOR_VERDICT = 8     # need a non-trivial sample to issue a verdict


_WS_RE = re.compile(r"\s+")


def _normalize_title(title: str) -> str:
    """Lowercase + collapse internal whitespace. Aggressive enough to fold
    minor spacing artefacts (extra spaces around punctuation introduced by
    GDELT/Google News re-tokenization) but conservative enough that any
    actual word-level paraphrase falls outside this fingerprint.

    Trailing publisher tags (" - MSN", " - Motley Fool") are deliberately
    NOT stripped — they ARE part of the canonical title from the analyst's
    perspective, and a future caller building a syndication-style audit
    (same headline, multiple outlets) would want them preserved.
    """
    if not title:
        return ""
    return _WS_RE.sub(" ", title.strip().lower())


def build_audit(
    rows: list[tuple],
    *,
    window_h: int,
    min_pushes_for_verdict: int = MIN_PUSHES_FOR_VERDICT,
    max_groups: int = 20,
    max_source_examples: int = 4,
) -> dict:
    """Build the audit envelope from a list of ``(title, source, first_seen)``
    tuples (already filtered to urgency=2 and the live-only clause by the
    caller).

    Pure builder — accepts rows from any source (live conn, fixture, test
    in-memory DB) so the function is trivially unit-testable. Same shape as
    ``analytics.recap_template_audit.audit_by_source`` and the rest of the
    audit family.

    Returns::

        {
            "window_h":              int,
            "n_pushes":              int,             # total urgency=2 rows in window
            "n_distinct_titles":     int,
            "n_duplicate_titles":    int,             # titles with >= 2 pushes
            "n_redundant_pushes":    int,             # sum(count-1) over dup groups
            "duplication_rate_pct":  float,           # n_redundant / n_pushes * 100
            "duplicate_groups": [
                {
                    "title":            str,         # normalized form
                    "push_count":       int,
                    "first_seen_oldest": str,
                    "first_seen_newest": str,
                    "sources":          [str,...],   # capped at max_source_examples
                },
                ...                                   # most-pushes-first; capped at max_groups
            ],
            "verdict": "HEAVY_DUPLICATION" | "LIGHT_DUPLICATION" | "NO_DUPLICATION" | "NO_DATA",
        }

    Verdict ladder (most severe first):
      * ``NO_DATA``         — fewer than ``min_pushes_for_verdict`` pushes in
        window (sample too small to draw a meaningful rate).
      * ``HEAVY_DUPLICATION`` — duplication_rate_pct > DUPLICATION_RATE_HEAVY_PCT.
      * ``LIGHT_DUPLICATION`` — duplication_rate_pct > DUPLICATION_RATE_LIGHT_PCT.
      * ``NO_DUPLICATION``   — everything else (rate at or below the light
        threshold; the existing recency/dedup gates are catching duplicates).
    """
    n_pushes = len(rows)
    # Group by normalized title.
    by_title: dict[str, dict] = {}
    for title, source, first_seen in rows:
        norm = _normalize_title(title or "")
        if not norm:
            continue
        bucket = by_title.setdefault(
            norm,
            {"push_count": 0, "first_seen_oldest": None,
             "first_seen_newest": None, "sources": []},
        )
        bucket["push_count"] += 1
        if (bucket["first_seen_oldest"] is None
                or (first_seen and first_seen < bucket["first_seen_oldest"])):
            bucket["first_seen_oldest"] = first_seen
        if (bucket["first_seen_newest"] is None
                or (first_seen and first_seen > bucket["first_seen_newest"])):
            bucket["first_seen_newest"] = first_seen
        src = source or "(unknown)"
        if src not in bucket["sources"]:
            bucket["sources"].append(src)

    n_distinct = len(by_title)
    dup_titles = [(t, b) for t, b in by_title.items() if b["push_count"] >= 2]
    n_redundant = sum(b["push_count"] - 1 for _t, b in dup_titles)

    duplication_rate = round(
        (n_redundant / n_pushes * 100.0) if n_pushes else 0.0, 2
    )

    # Most-pushes-first, alphabetical tie-break for determinism.
    dup_titles.sort(key=lambda kv: (-kv[1]["push_count"], kv[0]))
    duplicate_groups = []
    for title, bucket in dup_titles[:max_groups]:
        duplicate_groups.append({
            "title": title,
            "push_count": bucket["push_count"],
            "first_seen_oldest": bucket["first_seen_oldest"],
            "first_seen_newest": bucket["first_seen_newest"],
            # Deterministic ordering on sources too (sorted) so the test
            # corpus pins exact values.
            "sources": sorted(bucket["sources"])[:max_source_examples],
        })

    if n_pushes < min_pushes_for_verdict:
        verdict = "NO_DATA"
    elif duplication_rate > DUPLICATION_RATE_HEAVY_PCT:
        verdict = "HEAVY_DUPLICATION"
    elif duplication_rate > DUPLICATION_RATE_LIGHT_PCT:
        verdict = "LIGHT_DUPLICATION"
    else:
        verdict = "NO_DUPLICATION"

    return {
        "window_h": window_h,
        "n_pushes": n_pushes,
        "n_distinct_titles": n_distinct,
        "n_duplicate_titles": len(dup_titles),
        "n_redundant_pushes": n_redundant,
        "duplication_rate_pct": duplication_rate,
        "duplicate_groups": duplicate_groups,
        "verdict": verdict,
    }


def audit(store, hours: int = 24, **kwargs) -> dict:
    """Audit pushed-alert title duplication over the recent ``hours`` window.

    Live entrypoint: queries ``urgency=2`` rows in the window from the
    provided ``store.conn`` (read-only SELECT), applies ``LIVE_ONLY_CLAUSE``,
    and delegates to ``build_audit``. ``hours`` clamped to ``>= 1``.
    """
    hours = max(1, int(hours))
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    rows = store.conn.execute(
        "SELECT title, source, first_seen FROM articles "
        f"WHERE urgency=2 AND first_seen >= ? AND {LIVE_ONLY_CLAUSE}",
        (since,),
    ).fetchall()
    return build_audit(list(rows), window_h=hours, **kwargs)


def _cli() -> int:  # pragma: no cover - thin CLI shim
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()
    # Lazy import to keep the module itself dependency-light.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from storage.article_store import ArticleStore
    store = ArticleStore()
    print(json.dumps(audit(store, hours=args.hours), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
