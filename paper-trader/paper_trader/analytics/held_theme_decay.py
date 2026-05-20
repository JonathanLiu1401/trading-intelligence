"""Per-held-ticker fresh-vs-prior decayed-news-score velocity.

The single question this answers: *for each ticker I currently own, is the
score-weighted news flow LOUDER NOW or QUIETER NOW than it was earlier?* —
i.e. is the catalyst that justifies holding it still alive in the wire, or
has the wire moved on?

Distinct from every neighbour (invariant #10 — do not consolidate):

* ``/api/news-themes`` (``analytics/news_themes.py``) — single-window
  decayed-score snapshot. NO comparison vs an earlier window, so a position
  whose theme has gone DARK looks identical to one that just lit up.
* ``/api/news-velocity`` (``analytics/news_velocity.py``) — per-held-ticker
  ARTICLE-COUNT rate vs a 168h baseline. Mention COUNT, NOT score-weighted:
  a flood of low-relevance mentions inflates the rate while a single 9.5
  Sonnet-labelled article moves it the same as a junk RSS row. Different
  signal, different question (Poisson rate vs decayed-score weight).
* ``/api/position-thesis`` — latest 24h headlines per held position with a
  bull/bear split. Single window, no velocity dimension.
* ``/api/thesis-drift`` — grades held positions against ENTRY rationale,
  not against current wire prominence.
* digital-intern ``trend_velocity`` — market-wide mention-gainers across
  the entire universe, NOT keyed to the live book.

The fresh window (default 6h) matches ``news_themes.DECAY_HALF_LIFE_HOURS``
so a held ticker's ``fresh_score`` here lines up with its contribution to
that endpoint's top-themes ranking. The prior window is the immediately
preceding non-overlapping band of the same width — answers "is the wire
moving toward this name or away from it?" with no calendar / weekend
baseline confusion.

State ladder:
  ``DARK``     — no qualifying articles in either window (likely outside
                 the wire's current focus entirely)
  ``FADING``   — fresh score < prior × FADE_RATIO (catalyst is decaying;
                 reassess the thesis)
  ``BUILDING`` — fresh score > prior × BUILD_RATIO and fresh is non-trivial
                 (catalyst is strengthening — current entry is well-timed)
  ``STABLE``   — between FADE_RATIO and BUILD_RATIO (steady-state coverage)

A "non-trivial" fresh bar (``MIN_FRESH_SCORE``) prevents a 0.1 → 0.5 jump
from claiming BUILDING — the absolute prominence must matter too. Mirrors
``news_velocity.MIN_WINDOW_FOR_SURGE``'s "both z AND absolute" gate.

**Single source of truth.** ``DECAY_HALF_LIFE_HOURS`` is imported from
``analytics.news_themes`` so the two endpoints share a decay shape — any
future re-tune updates both. Article-ticker extraction is the caller's
responsibility (it already happens at the SQL/extract layer in the
``/api/news-themes`` route — we receive rows with ``tickers``).

**Observational, never prescriptive.** Same contract as ``news_velocity``
/ ``news_themes`` (AGENTS.md invariants #2/#12): states facts, issues no
directive, imposes no cap, never gates a trade.

Pure and deterministic (no clock, no IO when ``now`` is provided). Never
raises on garbage rows: defense-in-depth backtest-row drop (mirrors the
``news_themes._is_synthetic`` SSOT shape), tolerant ISO/datetime parsing,
unknown/missing tickers degrade to skipped (not crashed).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .news_themes import DECAY_HALF_LIFE_HOURS


# Fresh-window default. 6h matches ``news_themes.DECAY_HALF_LIFE_HOURS`` so
# the fresh-score lines up with the news-themes contribution magnitude.
FRESH_WINDOW_HOURS = 6.0

# A BUILDING/FADING verdict requires this much absolute fresh score so a
# 0.1 → 0.5 ratio jump on noise doesn't claim a real surge. Matches the
# "small but real" floor a discretionary desk applies before re-acting on
# coverage. Below this floor with NO prior either ⇒ DARK.
MIN_FRESH_SCORE = 1.0

# Ratio thresholds — fresh/prior < FADE_RATIO ⇒ FADING; > BUILD_RATIO ⇒
# BUILDING; in between ⇒ STABLE. Picked so a 30% swing in either
# direction qualifies (1/0.7 ≈ 1.43 — the symmetric inverse), the same
# magnitude a desk reaches for when calling a coverage shift "material".
FADE_RATIO = 0.7
BUILD_RATIO = 1.43


def _parse_ts(ts):
    """Tolerant ISO/datetime parse — naive → UTC. None on garbage."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _is_synthetic(row):
    """Defense-in-depth backtest filter mirroring ``news_themes._is_synthetic``.

    The SSOT for this filter is the SQL ``_LIVE_ONLY_CLAUSE`` (digital-intern
    ``storage/article_store.py``) — this rejects any row that leaked past it.
    """
    url = str(row.get("url") or "")
    src = str(row.get("source") or "")
    if url.startswith("backtest://"):
        return True
    if src.startswith("backtest_"):
        return True
    if src.startswith("opus_annotation"):
        return True
    return False


def _verdict(fresh_score: float, prior_score: float) -> str:
    """Map (fresh, prior) → state. Bounded, deterministic, total."""
    if fresh_score <= 0.0 and prior_score <= 0.0:
        return "DARK"
    if fresh_score < MIN_FRESH_SCORE and prior_score < MIN_FRESH_SCORE:
        # Neither window has meaningful weight — DARK is the honest verdict
        # even if technically fresh > prior × BUILD_RATIO on noise.
        return "DARK"
    if prior_score <= 0.0:
        # No prior coverage — anything fresh-meaningful is BUILDING.
        return "BUILDING" if fresh_score >= MIN_FRESH_SCORE else "DARK"
    if fresh_score <= 0.0:
        return "FADING"
    ratio = fresh_score / prior_score
    if ratio < FADE_RATIO:
        return "FADING"
    if ratio > BUILD_RATIO and fresh_score >= MIN_FRESH_SCORE:
        return "BUILDING"
    return "STABLE"


def _weighted(art, ts_ref: datetime, half_life_h: float) -> float:
    """Decayed weight of a single article relative to ``ts_ref``.

    Mirrors ``news_themes`` arithmetic: ai_score × exp(-age_h / half_life × ln 2).
    Articles older than ts_ref get 0 (no negative-age inflation).
    """
    ts = _parse_ts(art.get("first_seen"))
    if ts is None:
        return 0.0
    age_h = (ts_ref.timestamp() - ts.timestamp()) / 3600.0
    if age_h < 0.0:
        return 0.0
    try:
        ai = float(art.get("ai_score") or 0.0)
    except Exception:
        ai = 0.0
    if ai <= 0.0:
        return 0.0
    return ai * math.exp(-age_h / half_life_h * math.log(2))


def build_held_theme_decay(
    articles,
    held_tickers,
    now=None,
    fresh_window_hours: float = FRESH_WINDOW_HOURS,
):
    """Compute per-held-ticker fresh-vs-prior decayed score and verdict.

    Inputs:
        articles: list of dicts (``/api/news-themes`` row shape) — must
            carry ``first_seen``, ``ai_score``, ``tickers`` (list).
            Multi-ticker articles split their score evenly across the
            tickers they mention — matches the ``news_themes`` rule so a
            wide-net headline does not inflate four held themes at once.
        held_tickers: iterable of held tickers (case-insensitive).
        now: datetime (default UTC now).
        fresh_window_hours: width of the FRESH and PRIOR windows. Default
            6h. The prior window is the immediately preceding band of
            the same width (``now - 2*window`` → ``now - window``).

    Returns dict with stable shape regardless of input:
        as_of, fresh_window_hours, prior_window_hours,
        decay_half_life_hours, state, holds, n_held, n_fading,
        n_building, n_dark, n_stable, fading_tickers, building_tickers,
        dark_tickers, worst_verdict, headline.

    ``state`` is the overall ladder:
      ``NO_HELD`` — no held tickers at all (collapse-to-silence path,
        the chat-enrichment SSOT precedent — never report on an empty book).
      ``OK``     — at least one held ticker scored.
    """
    now = now or datetime.now(timezone.utc)
    held_norm: list[str] = []
    seen: set[str] = set()
    for t in (held_tickers or []):
        if not t:
            continue
        u = str(t).upper().strip()
        if u and u not in seen:
            seen.add(u)
            held_norm.append(u)

    fresh_window_hours = max(0.5, float(fresh_window_hours))
    prior_window_hours = fresh_window_hours
    half_life_h = float(DECAY_HALF_LIFE_HOURS)
    fresh_cutoff = now.timestamp() - fresh_window_hours * 3600
    prior_cutoff = now.timestamp() - 2 * fresh_window_hours * 3600

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "fresh_window_hours": fresh_window_hours,
        "prior_window_hours": prior_window_hours,
        "decay_half_life_hours": half_life_h,
        "state": "NO_HELD",
        "holds": [],
        "n_held": 0,
        "n_fading": 0,
        "n_building": 0,
        "n_dark": 0,
        "n_stable": 0,
        "fading_tickers": [],
        "building_tickers": [],
        "dark_tickers": [],
        "worst_verdict": None,
        "headline": "Held-theme decay: no held positions.",
    }
    if not held_norm:
        return base

    # Bucket per-ticker fresh / prior decayed scores. Multi-ticker articles
    # split their decayed weight evenly across mentioned tickers — same
    # discriminator as news_themes (a 4-ticker headline contributes
    # 0.25× to each, not 1×).
    per: dict[str, dict] = {
        tk: {
            "ticker": tk,
            "fresh_score": 0.0,
            "prior_score": 0.0,
            "fresh_n": 0,
            "prior_n": 0,
            "top_fresh_title": None,
            "top_fresh_url": None,
            "_top_fresh_weight": -1.0,
        }
        for tk in held_norm
    }
    held_set = set(held_norm)

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
        # Window membership: FRESH (last `fresh_window`), PRIOR
        # (previous non-overlapping band). Anything in between is impossible
        # because prior_cutoff = fresh_cutoff - window.

        tickers = art.get("tickers") or []
        if not isinstance(tickers, list):
            continue
        norm = sorted({str(t).upper().strip() for t in tickers if t})
        # Only allocate weight to held tickers — the only ones we report.
        held_mentions = [t for t in norm if t in held_set]
        if not held_mentions:
            continue

        # Per-article decayed weight relative to ts_ref=now (matches
        # news_themes shape so a held ticker's fresh_score here is
        # comparable to its themes contribution).
        w = _weighted(art, now, half_life_h)
        if w <= 0.0:
            continue
        # Split full article weight evenly across ALL mentioned tickers
        # (not just held ones) so a 4-ticker headline that names one held
        # name contributes 0.25× (not 1×) — same anti-inflation rule as
        # news_themes. ``norm`` is the canonical multi-ticker denominator.
        split = w / max(1, len(norm))

        title = str(art.get("title") or "") or None
        url = str(art.get("url") or "") or None

        for tk in held_mentions:
            row = per[tk]
            if is_fresh:
                row["fresh_score"] += split
                row["fresh_n"] += 1
                # Track the single article carrying the most decayed weight
                # IN THE FRESH WINDOW — surface as the "what's making this
                # theme loud right now" headline. Use full weight (not the
                # split) for the comparison so a single-ticker 9.5 article
                # outranks a 4-way split on a 10.0 even when the split puts
                # both at the same per-ticker contribution.
                if w > row["_top_fresh_weight"]:
                    row["_top_fresh_weight"] = w
                    row["top_fresh_title"] = title
                    row["top_fresh_url"] = url
            else:
                row["prior_score"] += split
                row["prior_n"] += 1

    holds = []
    n_fading = n_building = n_dark = n_stable = 0
    fading_tickers: list[str] = []
    building_tickers: list[str] = []
    dark_tickers: list[str] = []

    for tk in held_norm:
        row = per[tk]
        fresh = round(row["fresh_score"], 4)
        prior = round(row["prior_score"], 4)
        verdict = _verdict(row["fresh_score"], row["prior_score"])
        # Ratio: prior 0 ⇒ None (don't fabricate inf); else fresh/prior.
        if row["prior_score"] > 0:
            ratio = round(row["fresh_score"] / row["prior_score"], 3)
        else:
            ratio = None
        holds.append({
            "ticker": tk,
            "fresh_score": fresh,
            "prior_score": prior,
            "fresh_n": int(row["fresh_n"]),
            "prior_n": int(row["prior_n"]),
            "ratio": ratio,
            "verdict": verdict,
            "top_fresh_title": row["top_fresh_title"],
            "top_fresh_url": row["top_fresh_url"],
        })
        if verdict == "FADING":
            n_fading += 1
            fading_tickers.append(tk)
        elif verdict == "BUILDING":
            n_building += 1
            building_tickers.append(tk)
        elif verdict == "DARK":
            n_dark += 1
            dark_tickers.append(tk)
        else:
            n_stable += 1

    # Sort holds for the UI by a "how worried should I be" ordering:
    # FADING first (catalyst decaying on a position I own), then DARK
    # (no one is talking about it), then STABLE, then BUILDING.
    order = {"FADING": 0, "DARK": 1, "STABLE": 2, "BUILDING": 3}
    holds.sort(key=lambda h: (order.get(h["verdict"], 9), -h["fresh_score"]))

    # Worst verdict = the highest-severity bucket present (FADING > DARK
    # > STABLE > BUILDING). A FADING anywhere is the operator's first
    # signal to reassess; collapse to None when nothing was scored.
    if n_fading > 0:
        worst = "FADING"
    elif n_dark > 0:
        worst = "DARK"
    elif n_stable > 0:
        worst = "STABLE"
    elif n_building > 0:
        worst = "BUILDING"
    else:
        worst = None

    # Headline composition — surface the operator's immediate decision
    # signal. FADING is the load-bearing case (active deterioration on
    # an open position), so lead with it; DARK is the secondary case;
    # an all-BUILDING/STABLE book gets a flat status line.
    if n_fading > 0:
        head = (
            f"{n_fading} of {len(held_norm)} held positions FADING "
            f"({', '.join(fading_tickers)})"
            f" — fresh score below {int((1 - FADE_RATIO) * 100)}% of prior"
            f" window; reassess thesis."
        )
    elif n_dark > 0:
        head = (
            f"{n_dark} of {len(held_norm)} held positions DARK "
            f"({', '.join(dark_tickers)})"
            f" — no meaningful news flow in either window."
        )
    elif n_building > 0:
        head = (
            f"{n_building} of {len(held_norm)} held positions BUILDING; "
            f"all coverage steady or strengthening."
        )
    else:
        head = (
            f"All {len(held_norm)} held positions STABLE — coverage "
            f"unchanged window-over-window."
        )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "fresh_window_hours": fresh_window_hours,
        "prior_window_hours": prior_window_hours,
        "decay_half_life_hours": half_life_h,
        "state": "OK",
        "holds": holds,
        "n_held": len(held_norm),
        "n_fading": n_fading,
        "n_building": n_building,
        "n_dark": n_dark,
        "n_stable": n_stable,
        "fading_tickers": fading_tickers,
        "building_tickers": building_tickers,
        "dark_tickers": dark_tickers,
        "worst_verdict": worst,
        "headline": head,
    }
