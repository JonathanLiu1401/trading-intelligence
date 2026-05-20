"""Per-unheld-ticker fresh-vs-prior decayed-news-score velocity.

The mirror image of ``held_theme_decay``. ``held_theme_decay`` answers
*is the catalyst on a ticker I OWN still alive in the wire?* This
endpoint answers the complementary question: *which tickers does the
wire have a RISING catalyst on that I am NOT in?* — i.e. the
rotation-opportunity surface.

Distinct from every neighbour (invariant #10 — do not consolidate):

* ``/api/news-themes`` (``analytics/news_themes.py``) — single-window
  decayed-score snapshot of every theme (held + unheld). NO comparison
  vs an earlier window: a stale theme with a long tail looks identical
  to one that just lit up. The ``held`` boolean filters by ownership
  but the score is point-in-time, so a fading-but-still-loud unheld
  theme outranks a newly-rising one in the same scan.
* ``/api/held-theme-decay`` (``analytics/held_theme_decay.py``) — same
  fresh/prior velocity decomposition, but restricted to HELD tickers.
  Identical math intentionally; this module reuses its constants and
  ``_verdict`` rule so the two surfaces are directly comparable.
* ``/api/watchlist-opportunities`` — single-window scan over the named
  WATCHLIST tickers only (not the full wire universe), surfacing
  high-score articles. No velocity dimension, no decay weighting.
* ``/api/idle-opportunity`` — tied to a DROUGHT period (only fires when
  the bot is in a HOLD streak). Snapshot during the drought, not a
  velocity over time.
* digital-intern ``trend_velocity`` — market-wide MENTION-RATE
  gainers (Poisson on raw counts), not score-weighted decayed prominence.

The fresh window (default 6h) matches ``news_themes.DECAY_HALF_LIFE_HOURS``
and ``held_theme_decay.FRESH_WINDOW_HOURS`` so an unheld ticker's
``fresh_score`` here, a held ticker's ``fresh_score`` in
held-theme-decay, and the same ticker's ``decayed_score`` contribution
in news-themes line up — three surfaces, one decay shape.

State ladder (per-ticker, same as held-theme-decay for direct comparability):
  ``DARK``     — no qualifying articles in either window
  ``FADING``   — fresh < prior × FADE_RATIO (catalyst weakening on
                 something I don't own — not actionable, surfaced only
                 in the per-row debug list)
  ``BUILDING`` — fresh > prior × BUILD_RATIO and fresh meets
                 ``MIN_FRESH_SCORE`` — the actionable surface, sorted
                 to the top
  ``BREAKING`` — prior==0 AND fresh >= ``BREAKING_FRESH_SCORE`` — a
                 brand-new catalyst with no prior coverage at all
                 (the highest-urgency rotation signal; held-theme-decay
                 collapses this case into BUILDING since for a held
                 name "BUILDING fresh from no prior" still means the
                 catalyst is strengthening — for UNHELD names the
                 zero-prior case is qualitatively distinct: it's a
                 brand-new story, not an accelerating existing one)
  ``STABLE``   — between FADE_RATIO and BUILD_RATIO (steady coverage
                 on something I don't own — informational only)

A "non-trivial" fresh bar (``MIN_FRESH_SCORE``) prevents a 0.1 → 0.5
noise jump from claiming BUILDING. Mirrors held-theme-decay exactly.
``BREAKING_FRESH_SCORE`` (3.0) is a higher floor than ``MIN_FRESH_SCORE``
(1.0): a "brand new catalyst" verdict needs more absolute weight than
the "accelerating" verdict to be honest about it.

Held tickers are EXCLUDED entirely before any per-ticker computation
— this is the rotation-opportunity surface, so a held name showing up
in BUILDING here would be a duplicate of the held-theme-decay row
(invariant #10). The exclusion is case-insensitive (same normalization
held_theme_decay does internally).

Multi-ticker articles split their decayed weight evenly across ALL
mentioned tickers (held + unheld combined) — same anti-inflation rule
news_themes and held_theme_decay use. A 4-ticker article with one
held name does NOT contribute 1× to each unheld theme; it contributes
0.25× to each (including the held one, which is then dropped from the
output but its share is NOT redistributed to the unheld three — same
discriminator the held-theme-decay row uses, so the held and unheld
weights are directly comparable across the two endpoints).

Pure and deterministic (no clock, no IO when ``now`` is provided). Never
raises on garbage rows: defense-in-depth backtest-row drop (mirrors
``news_themes._is_synthetic``), tolerant ISO/datetime parsing,
unknown/missing tickers degrade to skipped.

**Single source of truth.** ``DECAY_HALF_LIFE_HOURS``,
``FRESH_WINDOW_HOURS``, ``MIN_FRESH_SCORE``, ``FADE_RATIO``,
``BUILD_RATIO`` and ``_verdict`` are all imported from
``analytics.held_theme_decay`` so the two velocity-decomposition
surfaces share constants. Re-tuning ``FRESH_WINDOW_HOURS`` updates
both endpoints in lockstep — a rotation-pair invariant.

**Observational, never prescriptive.** AGENTS.md invariants #2/#12:
states facts, issues no directive, imposes no cap, never gates a trade.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .held_theme_decay import (
    BUILD_RATIO,
    FADE_RATIO,
    FRESH_WINDOW_HOURS,
    MIN_FRESH_SCORE,
    _parse_ts,
    _is_synthetic,
    _verdict,
    _weighted,
)
from .news_themes import DECAY_HALF_LIFE_HOURS


# A BREAKING verdict (prior==0, brand-new catalyst) requires a higher
# absolute weight floor than BUILDING — a single 1.0-score article with
# no prior coverage shouldn't claim "brand new theme." A desk treats a
# fresh story as a real catalyst only after multiple corroborating
# articles or a single high-relevance one (~3.0 = three middle-quality
# articles or one high-relevance Sonnet-9 article at half-life age).
BREAKING_FRESH_SCORE = 3.0

# Cap how many rows we return to keep the response shape predictable
# and the UI scan-able. The aggregate counts (n_building, n_breaking,
# n_fading, n_dark, n_stable) still cover the full unheld universe.
DEFAULT_MAX_THEMES = 20


def _unheld_verdict(fresh_score: float, prior_score: float) -> str:
    """Mirror held-theme-decay._verdict, but expose the BREAKING case.

    held_theme_decay collapses prior==0 / fresh>=MIN_FRESH_SCORE into
    BUILDING (correct for a held name — "the catalyst that justifies
    my holding is strengthening from no prior coverage" is the same
    actionable signal as accelerating from an existing base).

    For an UNHELD ticker the distinction matters: BUILDING (fresh > prior
    × 1.43 on top of an existing base) is "an existing story I'm not
    in is accelerating" while BREAKING (no prior coverage at all) is
    "a brand-new story I had no chance to be in started right now."
    The operator triages those differently — BREAKING is a faster
    rotation signal; BUILDING is a slower-burn rotation candidate.
    """
    if fresh_score <= 0.0 and prior_score <= 0.0:
        return "DARK"
    if fresh_score < MIN_FRESH_SCORE and prior_score < MIN_FRESH_SCORE:
        return "DARK"
    if prior_score <= 0.0:
        # Brand-new catalyst — distinct verdict for unheld names.
        if fresh_score >= BREAKING_FRESH_SCORE:
            return "BREAKING"
        if fresh_score >= MIN_FRESH_SCORE:
            return "BUILDING"
        return "DARK"
    if fresh_score <= 0.0:
        return "FADING"
    ratio = fresh_score / prior_score
    if ratio < FADE_RATIO:
        return "FADING"
    if ratio > BUILD_RATIO and fresh_score >= MIN_FRESH_SCORE:
        return "BUILDING"
    return "STABLE"


def build_rising_unheld_themes(
    articles,
    held_tickers,
    now=None,
    fresh_window_hours: float = FRESH_WINDOW_HOURS,
    max_themes: int = DEFAULT_MAX_THEMES,
):
    """Compute per-unheld-ticker fresh-vs-prior decayed score and verdict.

    Inputs (mirrors held_theme_decay.build_held_theme_decay):
        articles: list of dicts (news-themes row shape) — must carry
            ``first_seen``, ``ai_score``, ``tickers`` (list).
            Multi-ticker articles split their score evenly across ALL
            mentioned tickers (held + unheld) so a wide-net headline
            does not inflate four themes at once.
        held_tickers: iterable of held tickers (case-insensitive). These
            are EXCLUDED from the output (use held-theme-decay for them).
        now: datetime (default UTC now).
        fresh_window_hours: width of FRESH and PRIOR windows. Default
            6h. PRIOR is the immediately preceding non-overlapping
            band of the same width.
        max_themes: cap on returned per-ticker rows (default 20).
            Aggregate counts span the full unheld universe regardless.

    Returns dict with stable shape regardless of input:
        as_of, fresh_window_hours, prior_window_hours,
        decay_half_life_hours, max_themes, state, themes,
        n_unheld_seen, n_building, n_breaking, n_fading, n_dark,
        n_stable, building_tickers, breaking_tickers, top_rising,
        headline.

    ``state``:
      ``NO_DATA`` — no qualifying articles in either window
      ``OK``     — at least one unheld ticker scored
    """
    now = now or datetime.now(timezone.utc)

    # Normalize held set (case-insensitive) for the exclusion filter.
    held_set: set[str] = set()
    for t in (held_tickers or []):
        if not t:
            continue
        u = str(t).upper().strip()
        if u:
            held_set.add(u)

    fresh_window_hours = max(0.5, float(fresh_window_hours))
    prior_window_hours = fresh_window_hours
    half_life_h = float(DECAY_HALF_LIFE_HOURS)
    fresh_cutoff = now.timestamp() - fresh_window_hours * 3600
    prior_cutoff = now.timestamp() - 2 * fresh_window_hours * 3600

    try:
        max_themes = int(max_themes)
    except (TypeError, ValueError):
        max_themes = DEFAULT_MAX_THEMES
    max_themes = max(1, min(100, max_themes))

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "fresh_window_hours": fresh_window_hours,
        "prior_window_hours": prior_window_hours,
        "decay_half_life_hours": half_life_h,
        "max_themes": max_themes,
        "state": "NO_DATA",
        "themes": [],
        "n_unheld_seen": 0,
        "n_building": 0,
        "n_breaking": 0,
        "n_fading": 0,
        "n_dark": 0,
        "n_stable": 0,
        "building_tickers": [],
        "breaking_tickers": [],
        "top_rising": None,
        "headline": "Rising-unheld themes: no qualifying articles in window.",
    }

    # Bucket per-ticker fresh / prior decayed scores. We accumulate ONLY
    # unheld tickers — held tickers don't allocate (would duplicate
    # held-theme-decay). The multi-ticker-article split denominator
    # uses ALL mentioned tickers (held+unheld) so the per-ticker weight
    # matches what held-theme-decay sees for the held names; the held
    # name's share is simply dropped, not redistributed.
    per: dict[str, dict] = {}

    saw_any_article = False
    for art in (articles or []):
        if not isinstance(art, dict):
            continue
        if _is_synthetic(art):
            continue
        ts = _parse_ts(art.get("first_seen"))
        if ts is None:
            continue
        t_sec = ts.timestamp()
        if t_sec < prior_cutoff:
            continue
        is_fresh = t_sec >= fresh_cutoff

        tickers = art.get("tickers") or []
        if not isinstance(tickers, list):
            continue
        norm = sorted({str(t).upper().strip() for t in tickers if t})

        w = _weighted(art, now, half_life_h)
        if w <= 0.0:
            continue
        # Mark "wire was alive in window" BEFORE filtering to unheld
        # tickers — the empty-per state distinguishes "wire was dead"
        # (NO_DATA) from "wire was held-only" (OK + held-only headline).
        saw_any_article = True

        unheld_mentions = [t for t in norm if t and t not in held_set]
        if not unheld_mentions:
            # Held-only or no-ticker article — nothing to surface here.
            continue
        # Multi-ticker split denominator: ALL mentioned tickers
        # (held + unheld). Same anti-inflation rule held_theme_decay
        # uses, so the unheld and held weights are directly comparable.
        split = w / max(1, len(norm))

        title = str(art.get("title") or "") or None
        url = str(art.get("url") or "") or None

        for tk in unheld_mentions:
            row = per.get(tk)
            if row is None:
                row = {
                    "ticker": tk,
                    "fresh_score": 0.0,
                    "prior_score": 0.0,
                    "fresh_n": 0,
                    "prior_n": 0,
                    "top_fresh_title": None,
                    "top_fresh_url": None,
                    "_top_fresh_weight": -1.0,
                }
                per[tk] = row
            if is_fresh:
                row["fresh_score"] += split
                row["fresh_n"] += 1
                # Surface the single highest-decayed-weight FRESH article
                # per ticker (full weight, not split — same tie-break
                # rule held-theme-decay uses).
                if w > row["_top_fresh_weight"]:
                    row["_top_fresh_weight"] = w
                    row["top_fresh_title"] = title
                    row["top_fresh_url"] = url
            else:
                row["prior_score"] += split
                row["prior_n"] += 1

    if not per:
        # No unheld tickers scored — either the wire is held-only or
        # there's no qualifying flow at all. Either way the honest
        # state is NO_DATA.
        if saw_any_article:
            base["state"] = "OK"
            base["headline"] = (
                "Rising-unheld themes: no UNHELD ticker had qualifying "
                "coverage in window (wire is held-only)."
            )
        return base

    # Classify every unheld ticker we saw, then pick the top by
    # decayed fresh-score for the surfaced theme list. Aggregate counts
    # span the full unheld universe (n_unheld_seen) even though only
    # max_themes rows are returned.
    n_building = n_breaking = n_fading = n_dark = n_stable = 0
    building_tickers: list[str] = []
    breaking_tickers: list[str] = []

    classified: list[dict] = []
    for tk, row in per.items():
        verdict = _unheld_verdict(row["fresh_score"], row["prior_score"])
        if row["prior_score"] > 0:
            ratio = round(row["fresh_score"] / row["prior_score"], 3)
        else:
            ratio = None
        classified.append({
            "ticker": tk,
            "fresh_score": round(row["fresh_score"], 4),
            "prior_score": round(row["prior_score"], 4),
            "fresh_n": int(row["fresh_n"]),
            "prior_n": int(row["prior_n"]),
            "ratio": ratio,
            "verdict": verdict,
            "top_fresh_title": row["top_fresh_title"],
            "top_fresh_url": row["top_fresh_url"],
        })
        if verdict == "BUILDING":
            n_building += 1
            building_tickers.append(tk)
        elif verdict == "BREAKING":
            n_breaking += 1
            breaking_tickers.append(tk)
        elif verdict == "FADING":
            n_fading += 1
        elif verdict == "DARK":
            n_dark += 1
        else:
            n_stable += 1

    # Sort: BREAKING first (highest-urgency rotation signal — brand-new
    # catalyst), then BUILDING (accelerating existing story), then
    # STABLE / FADING / DARK — within each bucket by descending
    # fresh_score so the loudest catalyst tops the list. The operator
    # reads top-down for actionable rotation candidates.
    order = {"BREAKING": 0, "BUILDING": 1, "STABLE": 2, "FADING": 3, "DARK": 4}
    classified.sort(key=lambda r: (order.get(r["verdict"], 9), -r["fresh_score"]))

    # Trim to max_themes. Aggregate counts already captured.
    themes = classified[:max_themes]

    # Top rising = the highest-priority actionable verdict's loudest
    # ticker. BREAKING outranks BUILDING; within each, fresh_score.
    # Returns None if nothing actionable is present.
    top_rising = None
    for r in classified:
        if r["verdict"] in ("BREAKING", "BUILDING"):
            top_rising = r
            break

    # Headline composition. Lead with BREAKING (operator's first signal
    # of a brand-new story), then BUILDING (rotation candidate), then
    # informational status if nothing actionable surfaced.
    n_unheld_seen = len(classified)
    if n_breaking > 0:
        head = (
            f"{n_breaking} BREAKING unheld theme(s) "
            f"({', '.join(breaking_tickers[:5])}) — brand-new catalyst, "
            f"no prior coverage."
        )
        if n_building > 0:
            head += f" {n_building} also BUILDING."
    elif n_building > 0:
        head = (
            f"{n_building} BUILDING unheld theme(s) "
            f"({', '.join(building_tickers[:5])}) — fresh score above "
            f"{int(BUILD_RATIO * 100)}% of prior window."
        )
    else:
        head = (
            f"No rotation candidates: {n_unheld_seen} unheld theme(s) "
            f"in window, none BUILDING or BREAKING."
        )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "fresh_window_hours": fresh_window_hours,
        "prior_window_hours": prior_window_hours,
        "decay_half_life_hours": half_life_h,
        "max_themes": max_themes,
        "state": "OK",
        "themes": themes,
        "n_unheld_seen": n_unheld_seen,
        "n_building": n_building,
        "n_breaking": n_breaking,
        "n_fading": n_fading,
        "n_dark": n_dark,
        "n_stable": n_stable,
        "building_tickers": building_tickers,
        "breaking_tickers": breaking_tickers,
        "top_rising": top_rising,
        "headline": head,
    }
