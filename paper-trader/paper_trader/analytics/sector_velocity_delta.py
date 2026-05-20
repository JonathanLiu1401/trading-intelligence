"""Per-sector-bucket news-velocity delta (fresh vs prior window).

The single question this answers: *which sector bucket is the wire
ROTATING INTO right now?* — i.e. is the news flow on memory_core
ACCELERATING or DECELERATING relative to an immediately-prior window
of equal length?

Distinct from every neighbour (invariant #10 — do not consolidate):

* ``/api/sector-heatmap`` (``analytics/sector_heatmap.py``) — PRICE
  + RSI + 24h news COUNT per bucket. Point-in-time snapshot. NO
  velocity dimension: a bucket with steady 17-art/day and one ramping
  from 2→17 look identical.
* ``/api/sector-pulse`` — per-ticker momentum + news count snapshot;
  not bucketed, not velocity.
* ``/api/sector-signal-fit`` — bucket NEWS COVERAGE vs the live
  book's bucket WEIGHT (a fit measure, not a wire-time-derivative).
* ``/api/sector-exposure`` — $-weight per bucket of the book; the
  inverse direction (book→wire, not wire-velocity).
* ``/api/news-themes`` — per-TICKER single-window snapshot.
* ``/api/held-theme-decay`` / ``/api/rising-unheld-themes`` —
  per-TICKER fresh-vs-prior decomposition; this module is the
  per-BUCKET aggregation of the same shape.

Shares constants with ``held_theme_decay`` (FRESH_WINDOW_HOURS,
DECAY_HALF_LIFE_HOURS, FADE_RATIO, BUILD_RATIO, MIN_FRESH_SCORE) so
the bucket-level verdict thresholds match the per-ticker view —
re-tuning the decay shape in one place updates all three velocity
surfaces in lockstep (the rotation-pair invariant, extended).

Bucket SSOT is ``sector_heatmap.HEATMAP_BUCKETS`` — the same bucket
definitions the heatmap UI uses, so a "memory_core BUILDING" verdict
here lines up exactly with the memory_core row in
``/api/sector-heatmap``.

Per-bucket row:
  * ``fresh_score`` / ``prior_score`` — Σ over all tickers in the
    bucket of the decayed ai_score in each window. Multi-ticker
    articles split their weight evenly across mentioned tickers
    (same anti-inflation rule news_themes / held_theme_decay use), so
    a bucket's score equals the sum of the same per-ticker weights
    those surfaces report.
  * ``fresh_n`` / ``prior_n`` — raw article counts per window
    (an article that mentions two tickers in the same bucket counts
    once for the bucket; the SPLIT is on the score, not the count).
  * ``ratio`` — fresh / prior (None when prior is 0).
  * ``verdict`` — ACCELERATING / BUILDING / STABLE / FADING /
    DECELERATING / DARK (ladder below).
  * ``top_fresh_ticker`` — the loudest single-ticker contributor to
    the bucket's fresh_score (operator's drill-down target).
  * ``top_fresh_title`` / ``top_fresh_url`` — the loudest article
    in the fresh window, full-weighted (not split) for tie-break.

Verdict ladder (per bucket):
  ``DARK``         — no qualifying articles in either window
  ``DECELERATING`` — fresh < prior × FADE_RATIO AND prior >=
                     MIN_FRESH_SCORE × N_TICKERS_IN_BUCKET
                     (sector-level "rotation OUT" signal —
                     materially-prominent bucket coverage is dropping)
  ``FADING``       — fresh < prior × FADE_RATIO but prior below the
                     bucket-prominence floor (decay on a marginal
                     bucket — informational only)
  ``STABLE``       — between FADE_RATIO and BUILD_RATIO
  ``BUILDING``     — fresh > prior × BUILD_RATIO and fresh >=
                     MIN_FRESH_SCORE
  ``ACCELERATING`` — fresh > prior × ACCEL_RATIO AND fresh >=
                     MIN_FRESH_SCORE × N_TICKERS_IN_BUCKET
                     (sector-level "rotation IN" signal — the bucket
                     is now meaningfully louder than the prior window
                     on absolute prominence, not just on ratio)

ACCELERATING and DECELERATING are STRICTER than BUILDING/FADING
because at the bucket level (sum over multiple tickers) the noise
floor is correspondingly higher — a 1.0 absolute fresh_score on a
4-ticker bucket is per-ticker average of 0.25, which is below the
single-ticker MIN_FRESH floor. The bucket-prominence floor scales the
MIN_FRESH_SCORE by bucket size so a "loud bucket" verdict really
means the whole sector is contributing.

State ladder (overall): NO_DATA / OK. ``rotating_in`` is the bucket
list with ACCELERATING; ``rotating_out`` is the bucket list with
DECELERATING. The operator's first-glance question — "what sector is
the wire moving into?" — is answered by the headline composition.

Pure and deterministic (no clock, no IO when ``now`` is provided).
Never raises on garbage rows: defense-in-depth backtest filter,
tolerant ISO/datetime parsing, unknown/missing tickers degrade to
skipped.

**Single source of truth.** ``HEATMAP_BUCKETS`` is imported from
``sector_heatmap`` so the bucket universe never drifts. Decay /
window / ratio constants come from ``held_theme_decay`` /
``news_themes`` so the rotation-pair invariant holds across all
velocity surfaces.

**Observational, never prescriptive.** AGENTS.md invariants #2/#12:
states facts, issues no directive, imposes no cap, never gates a
trade.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .held_theme_decay import (
    BUILD_RATIO,
    FADE_RATIO,
    FRESH_WINDOW_HOURS,
    MIN_FRESH_SCORE,
    _is_synthetic,
    _parse_ts,
    _weighted,
)
from .news_themes import DECAY_HALF_LIFE_HOURS
from .sector_heatmap import HEATMAP_BUCKETS


# Stricter bucket-level ratio thresholds: ACCELERATING requires not
# just BUILD_RATIO (1.43) but a sector-prominence floor (the bucket
# really moved, not just one ticker spiking). Picked so a 60% jump in
# bucket-level prominence qualifies — the magnitude a discretionary
# desk reaches for when calling a sector "rotating in."
ACCEL_RATIO = 1.6
DECEL_RATIO = FADE_RATIO  # symmetric: re-uses single-ticker floor


def _bucket_floor(n_tickers: int) -> float:
    """Bucket-prominence floor — MIN_FRESH_SCORE scaled by bucket size.

    A 4-ticker bucket needs 4 × MIN_FRESH_SCORE absolute fresh_score
    to claim ACCELERATING; otherwise the bucket-level verdict is just
    one ticker's noise relabeled as "the whole sector is rotating in."

    The floor is intentionally lenient (linear, not quadratic): a
    1-ticker bucket only needs MIN_FRESH (consistent with the per-
    ticker view); a 4-ticker bucket needs 4× as much absolute weight
    to claim the sector-level verdict.
    """
    return MIN_FRESH_SCORE * max(1, int(n_tickers))


def _verdict_bucket(
    fresh_score: float,
    prior_score: float,
    n_tickers: int,
) -> str:
    """Map (fresh, prior, n) → sector verdict. Deterministic, total."""
    floor = _bucket_floor(n_tickers)
    # DARK case — neither window has meaningful weight.
    if fresh_score <= 0.0 and prior_score <= 0.0:
        return "DARK"
    if fresh_score < MIN_FRESH_SCORE and prior_score < MIN_FRESH_SCORE:
        # Neither window even crosses the SINGLE-ticker floor —
        # honest DARK regardless of ratio.
        return "DARK"

    # No prior: anything fresh-meaningful is BUILDING (single-ticker
    # rules at bucket level; ACCELERATING needs prior context).
    if prior_score <= 0.0:
        if fresh_score >= floor:
            # No prior but loud bucket-level fresh → call it
            # ACCELERATING (a sector lighting up from nothing IS
            # the strongest rotation-in signal).
            return "ACCELERATING"
        if fresh_score >= MIN_FRESH_SCORE:
            return "BUILDING"
        return "DARK"

    if fresh_score <= 0.0:
        # Prior loud, fresh dead — DECELERATING if prior crossed the
        # bucket floor, else single-ticker FADING.
        if prior_score >= floor:
            return "DECELERATING"
        return "FADING"

    ratio = fresh_score / prior_score
    # ACCELERATING: ratio AND bucket-level absolute floor.
    if ratio > ACCEL_RATIO and fresh_score >= floor:
        return "ACCELERATING"
    # BUILDING: per-ticker-style ratio + per-ticker absolute floor.
    if ratio > BUILD_RATIO and fresh_score >= MIN_FRESH_SCORE:
        return "BUILDING"
    # DECELERATING: ratio drop AND prior was bucket-level prominent.
    if ratio < DECEL_RATIO and prior_score >= floor:
        return "DECELERATING"
    # FADING: ratio drop on a marginal bucket (single-ticker floor).
    if ratio < FADE_RATIO:
        return "FADING"
    # In-band → STABLE.
    return "STABLE"


def build_sector_velocity_delta(
    articles,
    now=None,
    fresh_window_hours: float = FRESH_WINDOW_HOURS,
    buckets: dict | None = None,
):
    """Compute per-bucket fresh-vs-prior news-velocity delta.

    Inputs:
        articles: list of dicts (news-themes row shape) — must carry
            ``first_seen``, ``ai_score``, ``tickers`` (list). Multi-
            ticker articles split decayed weight evenly across ALL
            mentioned tickers (same anti-inflation rule news_themes
            and held_theme_decay use, so bucket scores equal the
            sum of the per-ticker scores those surfaces report).
        now: datetime (default UTC now).
        fresh_window_hours: width of FRESH and PRIOR windows. Default
            6h (matches held_theme_decay). PRIOR is the immediately
            preceding non-overlapping band of the same width.
        buckets: optional bucket→tickers mapping override (default:
            ``sector_heatmap.HEATMAP_BUCKETS``). Test seam — production
            always uses the SSOT.

    Returns dict with stable shape regardless of input:
        as_of, fresh_window_hours, prior_window_hours,
        decay_half_life_hours, state, buckets, n_buckets,
        n_accelerating, n_building, n_decelerating, n_fading, n_dark,
        n_stable, rotating_in, rotating_out, top_accelerating,
        top_decelerating, headline.

    ``state``:
      ``NO_DATA`` — no qualifying articles in window
      ``OK``     — at least one bucket scored
    """
    now = now or datetime.now(timezone.utc)

    fresh_window_hours = max(0.5, float(fresh_window_hours))
    prior_window_hours = fresh_window_hours
    half_life_h = float(DECAY_HALF_LIFE_HOURS)
    fresh_cutoff = now.timestamp() - fresh_window_hours * 3600
    prior_cutoff = now.timestamp() - 2 * fresh_window_hours * 3600

    bucket_map = buckets if buckets is not None else HEATMAP_BUCKETS

    # Build ticker→buckets reverse map (a ticker may live in multiple
    # buckets — e.g. SMH/SOXX in the etf bucket only, but the data
    # structure tolerates overlap). Normalized to upper-strip.
    ticker_to_buckets: dict[str, list[str]] = {}
    bucket_tickers: dict[str, list[str]] = {}
    for bname, tlist in bucket_map.items():
        norm_t: list[str] = []
        for t in (tlist or []):
            u = str(t).upper().strip()
            if not u:
                continue
            norm_t.append(u)
            ticker_to_buckets.setdefault(u, []).append(bname)
        bucket_tickers[bname] = norm_t

    # Initialize per-bucket accumulators with stable shape.
    per: dict[str, dict] = {
        bname: {
            "name": bname,
            "tickers": list(tlist),
            "n_tickers": len(tlist),
            "fresh_score": 0.0,
            "prior_score": 0.0,
            "fresh_n": 0,
            "prior_n": 0,
            "_top_fresh_weight": -1.0,
            "_top_fresh_per_ticker": {},  # ticker -> accumulated fresh score
            "top_fresh_title": None,
            "top_fresh_url": None,
            "top_fresh_ticker": None,
        }
        for bname, tlist in bucket_tickers.items()
    }

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
        saw_any_article = True
        split = w / max(1, len(norm))

        title = str(art.get("title") or "") or None
        url = str(art.get("url") or "") or None

        # Bucket attribution: for each ticker in the article, find
        # every bucket it belongs to. A 4-ticker article with one
        # memory_core name contributes 0.25× the article's full weight
        # to memory_core; a 4-ticker article with two design names
        # contributes 0.5× to design (= 0.25 + 0.25). The article-level
        # COUNT (fresh_n / prior_n) increments once per bucket
        # regardless of multi-mention (the operator's "how many
        # articles is the wire spending on memory_core" question
        # treats a 2-ticker memory_core article as ONE article in the
        # bucket, not two).
        per_bucket_score: dict[str, float] = {}
        per_bucket_top_ticker: dict[str, tuple[str, float]] = {}
        for tk in norm:
            buckets_for_t = ticker_to_buckets.get(tk, [])
            for bname in buckets_for_t:
                per_bucket_score[bname] = per_bucket_score.get(bname, 0.0) + split
                # Track which ticker contributed most to this bucket
                # in this article (used to pick the bucket's
                # top_fresh_ticker in the FRESH window only).
                prev = per_bucket_top_ticker.get(bname)
                if prev is None or split > prev[1]:
                    per_bucket_top_ticker[bname] = (tk, split)

        for bname, bucket_w in per_bucket_score.items():
            row = per[bname]
            if is_fresh:
                row["fresh_score"] += bucket_w
                row["fresh_n"] += 1
                # Track per-ticker contributions so we can name the
                # bucket's top contributor (drill-down target).
                top_t = per_bucket_top_ticker.get(bname)
                if top_t is not None:
                    tk_name, tk_w = top_t
                    cur = row["_top_fresh_per_ticker"].get(tk_name, 0.0)
                    row["_top_fresh_per_ticker"][tk_name] = cur + tk_w
                # The bucket's single top article (for the drill-down
                # headline link) uses full article weight (not split)
                # so a single high-relevance article outranks four
                # split mentions at the same per-bucket contribution.
                if w > row["_top_fresh_weight"]:
                    row["_top_fresh_weight"] = w
                    row["top_fresh_title"] = title
                    row["top_fresh_url"] = url
            else:
                row["prior_score"] += bucket_w
                row["prior_n"] += 1

    # Resolve top_fresh_ticker per bucket from the per-ticker score
    # bag we built (max accumulated FRESH weight wins).
    for row in per.values():
        per_t = row.pop("_top_fresh_per_ticker", {})
        if per_t:
            row["top_fresh_ticker"] = max(per_t.items(), key=lambda kv: kv[1])[0]
        row.pop("_top_fresh_weight", None)

    # Classify every bucket. Aggregate counts span the full bucket
    # universe (n_buckets), not a slice.
    n_accelerating = n_building = n_decelerating = n_fading = n_dark = n_stable = 0
    rotating_in: list[str] = []
    rotating_out: list[str] = []
    classified: list[dict] = []

    for bname, row in per.items():
        verdict = _verdict_bucket(
            row["fresh_score"], row["prior_score"], row["n_tickers"]
        )
        if row["prior_score"] > 0:
            ratio = round(row["fresh_score"] / row["prior_score"], 3)
        else:
            ratio = None
        classified.append({
            "name": bname,
            "n_tickers": int(row["n_tickers"]),
            "fresh_score": round(row["fresh_score"], 4),
            "prior_score": round(row["prior_score"], 4),
            "fresh_n": int(row["fresh_n"]),
            "prior_n": int(row["prior_n"]),
            "ratio": ratio,
            "verdict": verdict,
            "top_fresh_ticker": row["top_fresh_ticker"],
            "top_fresh_title": row["top_fresh_title"],
            "top_fresh_url": row["top_fresh_url"],
        })
        if verdict == "ACCELERATING":
            n_accelerating += 1
            rotating_in.append(bname)
        elif verdict == "BUILDING":
            n_building += 1
        elif verdict == "DECELERATING":
            n_decelerating += 1
            rotating_out.append(bname)
        elif verdict == "FADING":
            n_fading += 1
        elif verdict == "DARK":
            n_dark += 1
        else:
            n_stable += 1

    # Sort: ACCELERATING > BUILDING > STABLE > FADING > DECELERATING
    # > DARK. Within each bucket, larger absolute fresh_score first.
    # DECELERATING is *higher-information* than FADING (sector-level
    # rotation OUT, not a marginal bucket cooling), so we sort it
    # below STABLE/BUILDING in the rotation-IN scan but it's
    # explicitly surfaced via rotating_out and the headline.
    order = {
        "ACCELERATING": 0, "BUILDING": 1, "STABLE": 2,
        "FADING": 3, "DECELERATING": 4, "DARK": 5,
    }
    classified.sort(key=lambda r: (order.get(r["verdict"], 9), -r["fresh_score"]))

    # top_accelerating / top_decelerating are operator drill-down
    # targets — the loudest rotation-IN bucket and the loudest
    # rotation-OUT bucket. None when nothing in that bucket exists.
    # ACCELERATING is sorted to the top of `classified` already
    # (loudest fresh_score first within the verdict bucket).
    top_accelerating = next(
        (r for r in classified if r["verdict"] == "ACCELERATING"), None
    )
    # DECELERATING: largest prior_score wins (the rotation OUT of the
    # formerly-loudest bucket is the more material signal).
    decelerating = [r for r in classified if r["verdict"] == "DECELERATING"]
    if decelerating:
        top_decelerating = max(decelerating, key=lambda r: r["prior_score"])
    else:
        top_decelerating = None

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "fresh_window_hours": fresh_window_hours,
        "prior_window_hours": prior_window_hours,
        "decay_half_life_hours": half_life_h,
        "accel_ratio": ACCEL_RATIO,
        "build_ratio": BUILD_RATIO,
        "fade_ratio": FADE_RATIO,
        "decel_ratio": DECEL_RATIO,
        "min_fresh_score": MIN_FRESH_SCORE,
        "state": "NO_DATA",
        "buckets": classified,
        "n_buckets": len(classified),
        "n_accelerating": 0,
        "n_building": 0,
        "n_decelerating": 0,
        "n_fading": 0,
        "n_dark": 0,
        "n_stable": 0,
        "rotating_in": [],
        "rotating_out": [],
        "top_accelerating": None,
        "top_decelerating": None,
        "headline": "Sector velocity: no qualifying articles in window.",
    }

    if not saw_any_article:
        # All buckets stay DARK by the verdict rules; preserve the
        # NO_DATA state for the empty-wire case (downstream callers
        # use state="NO_DATA" to short-circuit panel rendering).
        return base

    base["state"] = "OK"
    base["n_accelerating"] = n_accelerating
    base["n_building"] = n_building
    base["n_decelerating"] = n_decelerating
    base["n_fading"] = n_fading
    base["n_dark"] = n_dark
    base["n_stable"] = n_stable
    base["rotating_in"] = rotating_in
    base["rotating_out"] = rotating_out
    base["top_accelerating"] = top_accelerating
    base["top_decelerating"] = top_decelerating

    # Headline composition — surface the operator's immediate
    # rotation question. Lead with ACCELERATING (the actionable
    # rotation-in signal), then DECELERATING (rotation-out signal),
    # then a flat status line. Both rotating bands are surfaced when
    # they coexist (a market rotating from X into Y is the highest-
    # information case the operator wants).
    if n_accelerating > 0 and n_decelerating > 0:
        head = (
            f"Sector rotation: {', '.join(rotating_in)} ACCELERATING; "
            f"{', '.join(rotating_out)} DECELERATING."
        )
    elif n_accelerating > 0:
        head = (
            f"{n_accelerating} sector(s) ACCELERATING "
            f"({', '.join(rotating_in)}) — bucket-level news velocity "
            f"above {int(ACCEL_RATIO * 100)}% of prior window."
        )
    elif n_decelerating > 0:
        head = (
            f"{n_decelerating} sector(s) DECELERATING "
            f"({', '.join(rotating_out)}) — bucket-level news flow "
            f"dropped below {int(DECEL_RATIO * 100)}% of prior window."
        )
    elif n_building > 0:
        head = (
            f"{n_building} sector(s) BUILDING (individual ticker "
            f"acceleration, below sector-level rotation floor)."
        )
    else:
        head = (
            f"No sector rotation: {n_stable} STABLE, {n_dark} DARK, "
            f"{n_fading} FADING bucket(s)."
        )
    base["headline"] = head

    return base
