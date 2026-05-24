"""Opportunity-cost skill — for each HOLD_CASH / NO_DECISION decision in
the window, what did the top news-heat watchlist ticker do *after* the
bot chose to sit out?

The dashboard has a thick wall of analytics around the *behaviour* of
the bot when it does act (decision-vapor, thesis-drift, conviction-
language-skill) and the *cost of being stuck* (decision-paralysis,
capital-paralysis, idle-opportunity). What it doesn't have is the
PnL-aware question: when the bot saw a hot watchlist and chose cash,
did the cash choice EARN its keep — or did the runner Opus deliberately
passed on actually run?

``idle-opportunity`` is the closest cousin: it reports "during this
drought N watchlist signals ≥6.0 arrived" — but it stops at the
signal-arrival count. It never looks at what those tickers DID after.
A drought with 5 missed signals where every ticker fell 4% is a
``DEFENSIVE_WIN``; a drought with 5 missed signals where every ticker
ran +6% is a ``MISSED_ALPHA``. The current surface conflates the two.

Per-decision classification:

* For each HOLD-CASH / NO_DECISION row in the window, the caller
  resolves the top watchlist ticker by news heat in a short look-back
  ending at the decision timestamp (the bot's actual contextual window
  at the moment of the decision).
* For that ticker, the caller resolves the 1-day and 3-day forward
  *raw* return from the decision day's reference close.
* Per row verdict (default thresholds):
    - ``MISSED_RUNNER`` — ``fwd_3d_pct >= +5.0``
    - ``MISSED_OK``     — ``+1.0 <= fwd_3d_pct < +5.0``
    - ``NEUTRAL_HOLD``  — ``-1.0 < fwd_3d_pct < +1.0``
    - ``DEFENSIVE_HIT`` — ``fwd_3d_pct <= -1.0``
    - ``NO_FWD``        — forward return unavailable (too recent or
      yfinance gap; excluded from aggregate verdict)

Aggregate verdict (defaults):

* ``MISSED_ALPHA``  — ``(missed_runner_pct + missed_ok_pct) >= 50``
  AND ``mean_fwd_3d_pct >= +2.0``. The bot sat in cash while the
  tape it could see was running.
* ``DEFENSIVE_WIN`` — ``defensive_hit_pct >= 50``
  AND ``mean_fwd_3d_pct <= -2.0``. The HOLD CASH calls were correct;
  the missed tickers fell.
* ``NEUTRAL``       — between the two extremes (the bot's defensive
  stance neither cost nor earned material alpha).
* ``NO_DATA``       — fewer than ``MIN_DECISIONS_FOR_VERDICT`` (5)
  HOLD-CASH / NO_DECISION rows with a resolvable top ticker + forward
  return inside the window.

Pure builder. Decisions + callbacks in, dict out, never raises.
Observational only — never gates Opus, no caps (AGENTS.md #2 / #12 —
the ``decision_vapor_skill`` / ``cash_redeployment_latency_skill``
precedent).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

DEFAULT_WINDOW_HOURS = 168.0  # 7 days — enough for 3d-forward truth on most rows
DEFAULT_RUNNER_PCT_FLOOR = 5.0
DEFAULT_OK_PCT_FLOOR = 1.0
DEFAULT_DEFENSIVE_PCT_CEIL = -1.0
DEFAULT_MISSED_PCT_FLOOR = 50.0
DEFAULT_MEAN_FWD_PCT_FLOOR = 2.0
MIN_DECISIONS_FOR_VERDICT = 5
DEFAULT_SAMPLE_LIMIT = 20

_FILLED_SUFFIXES = ("FILLED", "→ FILLED")

# Action-string prefixes that count as "bot chose to sit out". The
# canonical formats are:
#   "HOLD CASH → HOLD"   — Opus actively chose to hold cash
#   "NO_DECISION"        — host saturated / parse_failed / quota / etc.
_SITOUT_PREFIXES = ("HOLD CASH", "NO_DECISION")


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def _is_sitout(action_taken: Any) -> bool:
    """A decision row is a 'sit-out' iff it's a HOLD-CASH branch or a
    NO_DECISION row. FILLED trades (BUY / SELL / option fills) and
    BLOCKED / REBALANCE rows are NOT sit-outs and never enter the
    aggregate.
    """
    if not isinstance(action_taken, str):
        return False
    s = action_taken.strip()
    if not s:
        return False
    # Active trade — anything that ends with FILLED is a real trade.
    for sfx in _FILLED_SUFFIXES:
        if s.endswith(sfx):
            return False
    for pfx in _SITOUT_PREFIXES:
        if s.startswith(pfx):
            return True
    return False


def _classify_row(
    fwd_3d_pct: float | None,
    *,
    runner_pct_floor: float,
    ok_pct_floor: float,
    defensive_pct_ceil: float,
) -> str:
    """Pure per-row verdict from a 3d forward return."""
    if fwd_3d_pct is None:
        return "NO_FWD"
    try:
        v = float(fwd_3d_pct)
    except (TypeError, ValueError):
        return "NO_FWD"
    if v != v:  # NaN
        return "NO_FWD"
    if v >= runner_pct_floor:
        return "MISSED_RUNNER"
    if v >= ok_pct_floor:
        return "MISSED_OK"
    if v <= defensive_pct_ceil:
        return "DEFENSIVE_HIT"
    return "NEUTRAL_HOLD"


def _aggregate_verdict(
    n_runner: int,
    n_ok: int,
    n_neutral: int,
    n_defensive: int,
    mean_fwd_3d_pct: float | None,
    *,
    min_decisions: int,
    missed_pct_floor: float,
    mean_fwd_pct_floor: float,
) -> tuple[str, str]:
    """Return ``(verdict, headline)``. Pure."""
    n_classified = n_runner + n_ok + n_neutral + n_defensive
    if n_classified < min_decisions:
        return (
            "NO_DATA",
            f"only {n_classified} sit-out decisions with forward returns "
            f"(need {min_decisions}) — accumulate more history",
        )
    missed_pct = 100.0 * (n_runner + n_ok) / n_classified
    defensive_pct = 100.0 * n_defensive / n_classified
    mean_str = (
        f"{mean_fwd_3d_pct:+.2f}%" if mean_fwd_3d_pct is not None else "n/a"
    )
    if (
        missed_pct >= missed_pct_floor
        and mean_fwd_3d_pct is not None
        and mean_fwd_3d_pct >= mean_fwd_pct_floor
    ):
        return (
            "MISSED_ALPHA",
            f"sit-out cost: {missed_pct:.0f}% of sit-outs preceded "
            f"a runner / ok move (mean 3d {mean_str}, n={n_classified})",
        )
    if (
        defensive_pct >= missed_pct_floor
        and mean_fwd_3d_pct is not None
        and mean_fwd_3d_pct <= -mean_fwd_pct_floor
    ):
        return (
            "DEFENSIVE_WIN",
            f"defensive sit-out paid: {defensive_pct:.0f}% of sit-outs "
            f"dodged a drawdown (mean 3d {mean_str}, n={n_classified})",
        )
    return (
        "NEUTRAL",
        f"sit-outs were neutral: {missed_pct:.0f}% missed / "
        f"{defensive_pct:.0f}% defensive, mean 3d {mean_str} (n={n_classified})",
    )


def build_opportunity_cost_skill(
    decisions: Sequence[Any] | None,
    *,
    top_ticker_at: Callable[[datetime], tuple[str, float] | None] | None = None,
    forward_returns_for: Callable[
        [str, datetime], tuple[float | None, float | None]
    ] | None = None,
    now: datetime | None = None,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    runner_pct_floor: float = DEFAULT_RUNNER_PCT_FLOOR,
    ok_pct_floor: float = DEFAULT_OK_PCT_FLOOR,
    defensive_pct_ceil: float = DEFAULT_DEFENSIVE_PCT_CEIL,
    missed_pct_floor: float = DEFAULT_MISSED_PCT_FLOOR,
    mean_fwd_pct_floor: float = DEFAULT_MEAN_FWD_PCT_FLOOR,
    min_decisions: int = MIN_DECISIONS_FOR_VERDICT,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Pure HOLD-CASH / NO_DECISION opportunity-cost classifier.

    Inputs:
      ``decisions`` — list of decision dicts ``{action_taken, timestamp, ...}``.
      ``top_ticker_at(ts)`` — callable returning ``(ticker, heat_score)`` for
        the watchlist ticker with the most news heat in the bot's look-back
        window ending at ``ts``, or ``None`` if no candidate exists.
      ``forward_returns_for(ticker, ts)`` — callable returning
        ``(fwd_1d_pct, fwd_3d_pct)`` for that ticker as of ``ts``'s session,
        either component ``None`` if unavailable.
      ``now`` — defaults to ``datetime.now(utc)``.
      ``window_hours`` — analysis window; sit-outs older than this are ignored.

    Threshold overrides exposed for tests + caller knobs.
    Never raises — a failing callback is treated as "no candidate / no
    forward return" for that row.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(0.0, window_hours))

    n_runner = 0
    n_ok = 0
    n_neutral = 0
    n_defensive = 0
    n_no_fwd = 0
    n_no_candidate = 0
    n_sitout_total = 0
    fwd_3d_sum = 0.0
    fwd_3d_count = 0
    samples: list[dict[str, Any]] = []

    for d in (decisions or []):
        if not isinstance(d, dict):
            continue
        if not _is_sitout(d.get("action_taken")):
            continue
        ts_dt = _parse_iso(d.get("timestamp"))
        if ts_dt is None:
            continue
        if ts_dt < cutoff:
            continue
        n_sitout_total += 1

        top: tuple[str, float] | None = None
        if top_ticker_at is not None:
            try:
                top = top_ticker_at(ts_dt)
            except Exception:
                top = None
        if not top or not isinstance(top, tuple) or len(top) != 2:
            n_no_candidate += 1
            continue
        ticker, heat = top
        if not isinstance(ticker, str) or not ticker:
            n_no_candidate += 1
            continue
        try:
            heat_f = float(heat) if heat is not None else 0.0
        except (TypeError, ValueError):
            heat_f = 0.0

        fwd_1d: float | None = None
        fwd_3d: float | None = None
        if forward_returns_for is not None:
            try:
                pair = forward_returns_for(ticker, ts_dt)
                if isinstance(pair, tuple) and len(pair) == 2:
                    a, b = pair
                    fwd_1d = float(a) if a is not None else None
                    fwd_3d = float(b) if b is not None else None
            except Exception:
                fwd_1d = None
                fwd_3d = None

        verdict = _classify_row(
            fwd_3d,
            runner_pct_floor=runner_pct_floor,
            ok_pct_floor=ok_pct_floor,
            defensive_pct_ceil=defensive_pct_ceil,
        )

        if verdict == "NO_FWD":
            n_no_fwd += 1
        elif verdict == "MISSED_RUNNER":
            n_runner += 1
        elif verdict == "MISSED_OK":
            n_ok += 1
        elif verdict == "DEFENSIVE_HIT":
            n_defensive += 1
        else:
            n_neutral += 1

        if fwd_3d is not None and fwd_3d == fwd_3d:  # not NaN
            fwd_3d_sum += fwd_3d
            fwd_3d_count += 1

        if len(samples) < sample_limit:
            samples.append({
                "ts": ts_dt.isoformat(),
                "action": str(d.get("action_taken") or "").strip(),
                "top_ticker": ticker,
                "top_heat": round(heat_f, 4),
                "fwd_1d_pct": (
                    round(fwd_1d, 4) if fwd_1d is not None else None
                ),
                "fwd_3d_pct": (
                    round(fwd_3d, 4) if fwd_3d is not None else None
                ),
                "verdict": verdict,
            })

    mean_fwd_3d_pct: float | None = (
        fwd_3d_sum / fwd_3d_count if fwd_3d_count else None
    )

    verdict, headline = _aggregate_verdict(
        n_runner,
        n_ok,
        n_neutral,
        n_defensive,
        mean_fwd_3d_pct,
        min_decisions=min_decisions,
        missed_pct_floor=missed_pct_floor,
        mean_fwd_pct_floor=mean_fwd_pct_floor,
    )

    n_classified = n_runner + n_ok + n_neutral + n_defensive
    return {
        "verdict": verdict,
        "headline": headline,
        "as_of": now.isoformat(),
        "window_hours": float(window_hours),
        "stats": {
            "n_sitout_total": n_sitout_total,
            "n_no_candidate": n_no_candidate,
            "n_no_fwd": n_no_fwd,
            "n_classified": n_classified,
            "n_missed_runner": n_runner,
            "n_missed_ok": n_ok,
            "n_neutral": n_neutral,
            "n_defensive": n_defensive,
            "missed_pct": (
                round(100.0 * (n_runner + n_ok) / n_classified, 2)
                if n_classified
                else None
            ),
            "defensive_pct": (
                round(100.0 * n_defensive / n_classified, 2)
                if n_classified
                else None
            ),
            "mean_fwd_3d_pct": (
                round(mean_fwd_3d_pct, 4)
                if mean_fwd_3d_pct is not None
                else None
            ),
        },
        "thresholds": {
            "runner_pct_floor": float(runner_pct_floor),
            "ok_pct_floor": float(ok_pct_floor),
            "defensive_pct_ceil": float(defensive_pct_ceil),
            "missed_pct_floor": float(missed_pct_floor),
            "mean_fwd_pct_floor": float(mean_fwd_pct_floor),
            "min_decisions": int(min_decisions),
        },
        "samples": samples,
    }


# --------------------------------------------------------------------------
# Helper exposed for the endpoint: pick top ticker by news heat in a
# window ending at ``ts``. Pure given a pre-loaded article list — separate
# from build_opportunity_cost_skill so tests can exercise both sides
# independently.
# --------------------------------------------------------------------------

def top_ticker_by_heat(
    articles: Sequence[dict],
    tickers: Sequence[str],
    ts_end: datetime,
    lookback_hours: float = 2.0,
) -> tuple[str, float] | None:
    """Top watchlist ticker by ai_score-weighted, urgency-bonused heat in
    the window ``[ts_end - lookback_hours, ts_end]``.

    ``articles`` rows must carry ``{title, ai_score, urgency, first_seen}``
    and may include any ``text`` field used for ticker detection. Caller
    has applied the live-only filter. ``tickers`` is the watchlist
    universe; the first whole-word / cashtag match wins (mirrors
    ``news_edge._resolve_ticker``).

    Heat formula: ``sum(ai_score * (1 + 0.5 * urgency))`` over matching
    articles. Ties broken by the first ticker (in ``tickers`` order)
    with the highest single ``ai_score`` in the window.

    Returns ``(ticker, heat)`` or ``None`` if no article in the window
    mentions any ``tickers`` member. Never raises.
    """
    import re

    if not articles or not tickers or ts_end is None:
        return None
    try:
        start = ts_end - timedelta(hours=max(0.0, lookback_hours))
    except Exception:
        return None

    tickers_norm = [t.upper() for t in tickers if isinstance(t, str) and t]
    if not tickers_norm:
        return None
    pats = {tk: re.compile(rf"(?:\$|\b){re.escape(tk)}\b") for tk in tickers_norm}

    heat: dict[str, float] = {}
    for a in articles:
        if not isinstance(a, dict):
            continue
        fs_dt = _parse_iso(a.get("first_seen"))
        if fs_dt is None:
            continue
        if fs_dt < start or fs_dt > ts_end:
            continue
        try:
            ai = float(a.get("ai_score") or 0.0)
            urg = float(a.get("urgency") or 0.0)
        except (TypeError, ValueError):
            continue
        if ai <= 0:
            continue
        text = " ".join(
            str(a.get(k) or "")
            for k in ("text", "title", "summary")
        ).upper()
        if not text.strip():
            continue
        contribution = ai * (1.0 + 0.5 * urg)
        for tk in tickers_norm:
            if pats[tk].search(text):
                heat[tk] = heat.get(tk, 0.0) + contribution
                # First-ticker match per article (mirrors news_edge) —
                # if NVDA + AMD both appear, NVDA (the first one in
                # WATCHLIST order) gets the heat and we stop scanning.
                break

    if not heat:
        return None
    # Tie-break: highest heat first, then watchlist order
    best_tk = max(
        heat.keys(),
        key=lambda tk: (heat[tk], -tickers_norm.index(tk)),
    )
    return (best_tk, heat[best_tk])
