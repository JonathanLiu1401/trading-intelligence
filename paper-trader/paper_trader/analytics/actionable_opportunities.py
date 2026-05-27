"""Actionable-opportunity composite ranker — pure builder.

Composes three independent unheld-watchlist surfaces into a single ranked
list. Each input answers a DIFFERENT actionability axis and none of them
alone resolves the operator's real question — *"of the strong scorer picks,
which one is the wire ALSO talking about right now?"*.

Live evidence motivating this composite (2026-05-27 02:49 ET):

  ``/api/scorer-opportunities`` lists 46 STRONG_HOLD candidates (AMD
  +26.1%, MU +24.7%, SMH +22.1%, …) but ``/api/suggestions`` returns 0
  candidates because its input pipeline is *news-mention driven* (``ai_score
  ≥ 5``); ``/api/persistent-watchlist-opportunity`` returns 0 (no
  ``ai_score≥6`` ticker has held that level for 6h+). The book is 100%
  cash. The analyst lands on three disjoint reads that DO NOT cross-confirm
  each other — there is no single surface that answers *"AMD is a
  STRONG_HOLD AND the wire is heating up on it AND that heat has
  persisted"*.

Axes composed (each from its own SSOT endpoint / builder so taxonomies
cannot drift):

  1. ``scorer_pred_5d_pct``  — DecisionScorer-predicted 5d return
                                (from ``/api/scorer-opportunities``)
  2. ``news_burst_verdict``   — ticker-volume burst BLAZING/HOT/WARMING/...
                                (from digital-intern ``/api/ticker-news-burst``)
  3. ``persistent_hours``     — contiguous current-hot-run on ai_score≥6
                                (from ``/api/persistent-watchlist-opportunity``)

Per-ticker actionability ladder (most-actionable first; ``HIGH_CONVICTION``
means the operator should LOOK FIRST):

  * ``HIGH_CONVICTION``      — scorer_pred ≥ ``STRONG_PRED_PP`` AND
                                news_burst in {BLAZING,HOT,WARMING}
  * ``NEWS_CONFIRMED``       — scorer_pred ≥ ``MIN_PRED_PP`` AND
                                news_burst in {BLAZING,HOT,WARMING}
  * ``PERSISTENT_FOLLOWUP``  — scorer_pred ≥ ``MIN_PRED_PP`` AND
                                persistent_hours ≥ ``MIN_PERSISTENT_H``
  * ``SCORER_ONLY``          — scorer_pred ≥ ``STRONG_PRED_PP`` AND
                                news_burst in {NORMAL, COLD}
  * ``NEWS_ONLY``            — scorer_pred < ``MIN_PRED_PP`` AND
                                news_burst in {BLAZING, HOT}
  * ``WEAK``                 — otherwise

Top-level verdict ladder (silence-on-healthy-and-empty precedent):

  * ``INSUFFICIENT_DATA``      — scorer not qualified (``is_trained=False``
                                  or ``n_train < gate_threshold``).
  * ``HIGH_CONVICTION_FOUND``  — ≥1 HIGH_CONVICTION candidate.
  * ``NEWS_CONFIRMED``         — ≥1 NEWS_CONFIRMED but no HIGH_CONVICTION.
  * ``SCORER_BUT_NO_NEWS``     — ≥1 SCORER_ONLY but no news-confirmed cell
                                  (= the documented live failure mode).
  * ``NEWS_BUT_NO_SCORER``     — ≥1 NEWS_ONLY but no scorer-strong cell.
  * ``ALL_QUIET``              — every cell is WEAK.

Pure / never raises — every input degrades to "no contribution" rather
than exception. Returns a stable shape so the route can ``jsonify`` it
directly and the chat helper can rely on every key.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

# Thresholds chosen to match the existing scorer verdict ladder
# (_scorer_verdict in dashboard.py at line 9649): STRONG_HOLD fires at
# pred >= 3.0pp, HOLD at >= 1.0, so STRONG_PRED_PP (10pp) is a "well above
# STRONG_HOLD" bar — the analyst should treat HIGH_CONVICTION as a
# genuine high-bar signal, not a routine STRONG_HOLD label that fires
# 46× in the live live snapshot.
STRONG_PRED_PP = 10.0
MIN_PRED_PP = 5.0
MIN_PERSISTENT_H = 6.0

# Composite ranking weights — pred has the biggest dollar-impact axis, news
# burst is the freshness axis (high weight because a stale STRONG_HOLD is
# less actionable than a fresh+strong combo), persistent_hours is the
# time-confirmation axis (lower weight because it can lag a fast catalyst).
PRED_WEIGHT = 1.0
BURST_WEIGHT = 2.0
PERSISTENT_WEIGHT = 0.5

# Burst → numeric contribution. We clamp at HOT-equivalent so a 50× spike
# doesn't dominate a +25pp pred (those should be ranked together, not
# burst-first).
_BURST_NUMERIC = {
    "BLAZING": 6.0,
    "HOT": 4.0,
    "WARMING": 2.0,
    "NORMAL": 0.0,
    "COLD": 0.0,
}


def _classify_actionability(
    pred_pp: float,
    burst_verdict: str,
    persistent_hours: float,
) -> str:
    burst_hot_tier = burst_verdict in ("BLAZING", "HOT", "WARMING")
    burst_strong_tier = burst_verdict in ("BLAZING", "HOT")
    if pred_pp >= STRONG_PRED_PP and burst_hot_tier:
        return "HIGH_CONVICTION"
    if pred_pp >= MIN_PRED_PP and burst_hot_tier:
        return "NEWS_CONFIRMED"
    if pred_pp >= MIN_PRED_PP and persistent_hours >= MIN_PERSISTENT_H:
        return "PERSISTENT_FOLLOWUP"
    if pred_pp >= STRONG_PRED_PP:
        return "SCORER_ONLY"
    if pred_pp < MIN_PRED_PP and burst_strong_tier:
        return "NEWS_ONLY"
    return "WEAK"


def _composite_score(
    pred_pp: float, burst_verdict: str, persistent_hours: float,
) -> float:
    burst_num = _BURST_NUMERIC.get(burst_verdict, 0.0)
    return round(
        pred_pp * PRED_WEIGHT
        + burst_num * BURST_WEIGHT
        + max(0.0, persistent_hours) * PERSISTENT_WEIGHT,
        3,
    )


def _reasons(
    pred_pp: float,
    pred_verdict: str,
    burst_verdict: str,
    spike: float | None,
    persistent_hours: float,
) -> list[str]:
    out: list[str] = []
    if pred_verdict and pred_verdict != "UNKNOWN":
        out.append(
            f"scorer {pred_pp:+.1f}% predicted 5d return ({pred_verdict})"
        )
    if burst_verdict in ("BLAZING", "HOT", "WARMING"):
        if spike is not None:
            out.append(
                f"news {burst_verdict} ({spike:.1f}× baseline mention rate)"
            )
        else:
            out.append(f"news {burst_verdict}")
    if persistent_hours >= MIN_PERSISTENT_H:
        out.append(f"{persistent_hours:.1f}h contiguous news heat")
    return out


def build_actionable_opportunities(
    scorer_payload: dict | None,
    burst_payload: dict | None,
    persistent_payload: dict | None,
    *,
    now: datetime | None = None,
    top_n: int = 10,
) -> dict:
    """Compose the three input payloads into a ranked actionability list.

    Every input is treated as untrusted (might be ``None``, might be
    ``{"error": ...}``, might be a missing-key dict). The builder degrades
    each missing axis to "no contribution" — never raises.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    top_n = max(1, min(50, int(top_n)))

    # ── 1. Scorer axis ──
    scorer = scorer_payload if isinstance(scorer_payload, dict) else {}
    is_trained = bool(scorer.get("is_trained"))
    n_train = int(scorer.get("n_train") or 0)
    gate_threshold = int(scorer.get("gate_threshold") or 500)
    scorer_opps = scorer.get("opportunities") or []
    scorer_by_tk: dict[str, dict] = {}
    for row in scorer_opps:
        if not isinstance(row, dict):
            continue
        tk = str(row.get("ticker") or "").upper().strip()
        if tk:
            scorer_by_tk[tk] = row

    # Gate: scorer unqualified ⇒ INSUFFICIENT_DATA (the trained_qualified
    # gate is the SSOT for "are predictions trustworthy yet?"; mirror it).
    if not is_trained or n_train < gate_threshold:
        return {
            "generated_at": now.isoformat(),
            "verdict": "INSUFFICIENT_DATA",
            "headline": (
                f"scorer not qualified (n_train={n_train} / "
                f"gate={gate_threshold}, trained={is_trained}) — "
                f"actionability rankings withheld"
            ),
            "is_trained": is_trained,
            "n_train": n_train,
            "gate_threshold": gate_threshold,
            "n_scored": 0,
            "n_high_conviction": 0,
            "n_news_confirmed": 0,
            "by_ticker": [],
        }

    # ── 2. News burst axis ──
    burst = burst_payload if isinstance(burst_payload, dict) else {}
    burst_rows = burst.get("by_ticker") or []
    burst_by_tk: dict[str, dict] = {}
    for row in burst_rows:
        if not isinstance(row, dict):
            continue
        tk = str(row.get("ticker") or "").upper().strip()
        if tk:
            burst_by_tk[tk] = row

    # ── 3. Persistent watchlist axis ──
    persistent = persistent_payload if isinstance(persistent_payload, dict) else {}
    persistent_rows = persistent.get("opportunities") or []
    persistent_by_tk: dict[str, float] = {}
    for row in persistent_rows:
        if not isinstance(row, dict):
            continue
        tk = str(row.get("ticker") or "").upper().strip()
        if not tk:
            continue
        # Field is named `current_run_hours` in the SSOT module — see
        # analytics/persistent_watchlist_opportunity.py. Accept the alias
        # `persistent_hours` for forward compatibility / chat reuse.
        hours = (
            row.get("current_run_hours")
            or row.get("persistent_hours")
            or 0.0
        )
        try:
            persistent_by_tk[tk] = float(hours or 0.0)
        except (TypeError, ValueError):
            continue

    # ── 4. Compose ──
    out: list[dict] = []
    for tk, scorer_row in scorer_by_tk.items():
        try:
            pred_pp = float(scorer_row.get("pred_5d_return_pct") or 0.0)
        except (TypeError, ValueError):
            pred_pp = 0.0
        pred_verdict = str(scorer_row.get("verdict") or "UNKNOWN")
        off_dist = bool(scorer_row.get("off_distribution") or False)

        b = burst_by_tk.get(tk) or {}
        burst_verdict = str(b.get("verdict") or "COLD")
        spike = b.get("spike")
        try:
            spike_f: float | None = float(spike) if spike is not None else None
        except (TypeError, ValueError):
            spike_f = None
        count_window = int(b.get("count_window") or 0)
        count_baseline = int(b.get("count_baseline") or 0)

        persistent_h = persistent_by_tk.get(tk, 0.0)

        actionability = _classify_actionability(
            pred_pp, burst_verdict, persistent_h
        )
        composite = _composite_score(pred_pp, burst_verdict, persistent_h)
        reasons = _reasons(
            pred_pp, pred_verdict, burst_verdict, spike_f, persistent_h
        )

        out.append({
            "ticker": tk,
            "scorer_pred_5d_pct": round(pred_pp, 3),
            "scorer_verdict": pred_verdict,
            "scorer_off_distribution": off_dist,
            "news_burst_verdict": burst_verdict,
            "news_burst_spike": (
                round(spike_f, 2) if spike_f is not None else None
            ),
            "news_burst_count_window": count_window,
            "news_burst_count_baseline": count_baseline,
            "persistent_hours": round(persistent_h, 1),
            "composite_score": composite,
            "actionability": actionability,
            "reasons": reasons,
        })

    out.sort(
        key=lambda r: (
            -r["composite_score"],
            -r["scorer_pred_5d_pct"],
            r["ticker"],
        )
    )

    n_high = sum(1 for r in out if r["actionability"] == "HIGH_CONVICTION")
    n_news_conf = sum(1 for r in out if r["actionability"] == "NEWS_CONFIRMED")
    n_persistent = sum(1 for r in out if r["actionability"] == "PERSISTENT_FOLLOWUP")
    n_scorer_only = sum(1 for r in out if r["actionability"] == "SCORER_ONLY")
    n_news_only = sum(1 for r in out if r["actionability"] == "NEWS_ONLY")

    if not out:
        verdict = "ALL_QUIET"
        headline = "no scorer-evaluated unheld watchlist names — nothing to rank"
    elif n_high >= 1:
        verdict = "HIGH_CONVICTION_FOUND"
        top = next(r for r in out if r["actionability"] == "HIGH_CONVICTION")
        headline = (
            f"HIGH CONVICTION: {top['ticker']} — scorer "
            f"{top['scorer_pred_5d_pct']:+.1f}% AND news "
            f"{top['news_burst_verdict']}"
            + (
                f" ({top['news_burst_spike']:.1f}×)"
                if top.get("news_burst_spike") is not None else ""
            )
        )
    elif n_news_conf >= 1:
        verdict = "NEWS_CONFIRMED"
        top = next(r for r in out if r["actionability"] == "NEWS_CONFIRMED")
        headline = (
            f"NEWS-CONFIRMED: {top['ticker']} — scorer "
            f"{top['scorer_pred_5d_pct']:+.1f}% AND news "
            f"{top['news_burst_verdict']}"
        )
    elif n_persistent >= 1:
        verdict = "PERSISTENT_FOLLOWUP"
        top = next(r for r in out if r["actionability"] == "PERSISTENT_FOLLOWUP")
        headline = (
            f"PERSISTENT: {top['ticker']} — scorer "
            f"{top['scorer_pred_5d_pct']:+.1f}% AND "
            f"{top['persistent_hours']:.1f}h contiguous heat"
        )
    elif n_scorer_only >= 1:
        verdict = "SCORER_BUT_NO_NEWS"
        top = next(r for r in out if r["actionability"] == "SCORER_ONLY")
        headline = (
            f"SCORER-ONLY: {top['ticker']} — scorer "
            f"{top['scorer_pred_5d_pct']:+.1f}% but news is "
            f"{top['news_burst_verdict']}. Strong quant pick the wire "
            f"hasn't caught yet (or the catalyst is invisible to news)."
        )
    elif n_news_only >= 1:
        verdict = "NEWS_BUT_NO_SCORER"
        top = next(r for r in out if r["actionability"] == "NEWS_ONLY")
        headline = (
            f"NEWS-ONLY: {top['ticker']} — news "
            f"{top['news_burst_verdict']} but scorer "
            f"{top['scorer_pred_5d_pct']:+.1f}%. The wire is hot but "
            f"the model does not corroborate."
        )
    else:
        verdict = "ALL_QUIET"
        headline = (
            "no actionable opportunity — all scorer-evaluated unheld names "
            "fell into WEAK"
        )

    return {
        "generated_at": now.isoformat(),
        "verdict": verdict,
        "headline": headline,
        "is_trained": is_trained,
        "n_train": n_train,
        "gate_threshold": gate_threshold,
        "n_scored": len(out),
        "n_high_conviction": n_high,
        "n_news_confirmed": n_news_conf,
        "n_persistent_followup": n_persistent,
        "n_scorer_only": n_scorer_only,
        "n_news_only": n_news_only,
        "by_ticker": out[:top_n],
    }
