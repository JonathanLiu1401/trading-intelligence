"""Cash-conviction fit — point-in-time verdict on whether the book's
cash level is appropriate given the strongest live news signal.

The other behavioural mirrors are *historical* — they walk the ledger
or the closed round-trips. They are silent when N is small. They were
silent on the live observation that motivated this endpoint:

* ``capital_paralysis`` flagged FREE (idle cash is fine, no
  involuntary alpha bleed *historically*).
* ``idle_opportunity`` flagged 5 score-9 signals — including the held
  NVDA name itself — but its headline collapses to "drought 4.5h"
  rather than "your cash is idle in spite of max conviction".
* ``position_action_brief`` produced MONITOR even with NVDA at
  ai_score 10, z-score 105, news SURGING — because the held
  position's *attention* is fresh; nothing joined cash idleness to
  signal loudness in one verdict.

None of those endpoints answers, *right now*: **is your book idle
despite the loudest live signal in the universe, or is it idle
correctly because nothing is screaming?** That's the
``IDLE_DESPITE_SURGE`` failure mode this endpoint catches. It is
point-in-time by construction — works at N=1 trade, N=1 position, or
N=0 (cold-boot). It does **not** read history.

Verdict matrix (cash_pct × top_signal_score × last_decision):

* ``IDLE_DESPITE_SURGE`` — cash_pct ≥ idle threshold AND top signal
  ≥ high-conviction floor AND last decision was HOLD / NO_DECISION.
  The book is sitting on capital while the loudest signal screams.
  Operator question: why?
* ``OVERDEPLOYED`` — cash_pct ≤ overdeployed floor AND top signal
  ≥ high-conviction floor. The book *cannot* respond to the signal
  without selling something — and the signal is strong enough that
  the inability to add is itself a constraint worth surfacing.
* ``IDLE_LOW_CONVICTION`` — cash_pct ≥ idle threshold AND top signal
  < low-conviction ceiling. Cash idleness is *correct* — nothing is
  worth deploying for. Affirmation, not a warning.
* ``BALANCED`` — none of the above. The cash level is in proportion
  to the live conviction.
* ``NO_DATA`` — missing portfolio or no signals supplied. Always
  emits the envelope; never raises.

``last_decision`` is consulted only to disambiguate
``IDLE_DESPITE_SURGE`` from ``BALANCED`` — when the bot is
*actively* deploying (last action a FILL within the recency floor),
the cash-idle reading is transient and the verdict shouldn't fire.
The recency floor is configurable.

Pure builder. Portfolio snapshot + top signals + last decision in,
dict out, never raises. Observational only — never gates Opus, no
caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

# Cash above this fraction-of-equity triggers an "idle" reading. 40%
# is the implicit cap in the bot's persona — sustained cash above
# this typically signals the bot has stopped finding fresh entries.
DEFAULT_IDLE_CASH_PCT = 40.0

# Cash below this fraction triggers an "overdeployed" reading. With
# <10% cash the book cannot add even one round-lot of a leveraged
# ETF without trimming something else.
DEFAULT_OVERDEPLOYED_CASH_PCT = 10.0

# ai_score floor for a signal to count as "high conviction" — at or
# above this the bot is supposed to act. The watchlist-opportunities
# endpoint uses 6.0 as its actionable floor; 8.0 here is the
# *surging* threshold (briefing-coverage-audit / portfolio-signals).
DEFAULT_HIGH_CONVICTION_SCORE = 8.0

# ai_score ceiling below which the live universe is materially quiet —
# nothing worth deploying for. Synonym for "the alert pipeline isn't
# firing". Articles in [low, high) are the grey zone that doesn't
# move either reading.
DEFAULT_LOW_CONVICTION_SCORE = 6.0

# Last-decision age beyond which a FILL is no longer recent enough
# to disambiguate idle-despite-surge. 30 minutes is roughly two
# market-open cycles (60s cadence) plus slack — long enough that a
# stale FILL doesn't mask a current idleness reading.
DEFAULT_RECENT_FILL_MAX_MIN = 30.0

# Verdict labels — kept as module-level constants so tests and callers
# never depend on string literals at the call site.
IDLE_DESPITE_SURGE = "IDLE_DESPITE_SURGE"
OVERDEPLOYED = "OVERDEPLOYED"
IDLE_LOW_CONVICTION = "IDLE_LOW_CONVICTION"
BALANCED = "BALANCED"
NO_DATA = "NO_DATA"

# Action-verb buckets — anything that's not an active fill counts as
# "passive" for the purposes of disambiguation.
_PASSIVE_ACTIONS = ("HOLD", "NO_DECISION", "BLOCKED")
_FILL_ACTIONS = ("BUY", "SELL", "BUY_CALL", "BUY_PUT",
                 "SELL_CALL", "SELL_PUT", "ADD", "REBALANCE")


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if x != x:  # NaN
        return None
    return float(x)


def _parse_ts(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _extract_verb(action_taken: Any) -> str | None:
    """Pull the leading verb out of an ``action_taken`` string.

    The store writes free-text like ``"BUY NVDA → FILLED"``,
    ``"HOLD NVDA → HOLD"``, or ``"NO_DECISION"``. The first
    space-separated token is the verb. ``CLAUDE.md`` #11 documents
    this convention; ``dashboard._parse_action_ticker`` is the
    canonical extractor on the read-side."""
    if not isinstance(action_taken, str):
        return None
    s = action_taken.strip()
    if not s:
        return None
    return s.split(None, 1)[0].upper()


def _top_signal(signals: Sequence[dict]) -> dict | None:
    """Pick the loudest signal by ai_score. Tie-breaks on urgency
    (higher first) then alphabetical ticker for stability. Skips
    rows without a numeric ai_score."""
    best: dict | None = None
    best_key: tuple[float, int, str] | None = None
    for s in signals or ():
        if not isinstance(s, dict):
            continue
        score = _num(s.get("ai_score"))
        if score is None:
            continue
        urg_raw = s.get("urgency")
        urg = int(urg_raw) if isinstance(urg_raw, (int, float)) and urg_raw == urg_raw else 0
        ticker = s.get("ticker") if isinstance(s.get("ticker"), str) else ""
        key = (score, urg, ticker)
        if best_key is None or key > best_key:
            best_key = key
            best = s
    return best


def build_cash_conviction_fit(
    portfolio: dict | None,
    signals: Sequence[dict] | None,
    last_decision: dict | None,
    *,
    now: datetime | None = None,
    idle_cash_pct: float = DEFAULT_IDLE_CASH_PCT,
    overdeployed_cash_pct: float = DEFAULT_OVERDEPLOYED_CASH_PCT,
    high_conviction_score: float = DEFAULT_HIGH_CONVICTION_SCORE,
    low_conviction_score: float = DEFAULT_LOW_CONVICTION_SCORE,
    recent_fill_max_min: float = DEFAULT_RECENT_FILL_MAX_MIN,
) -> dict:
    """Build the point-in-time cash-vs-conviction verdict.

    Arguments:
        portfolio: ``{cash, total_value, n_positions}`` dict from
            ``store.portfolio_snapshot`` or ``/api/portfolio``. May
            include ``cash_pct`` explicitly; recomputed if missing.
            ``None`` ⇒ NO_DATA.
        signals: live news signals — each at minimum
            ``{ticker, ai_score, urgency, source, held}``. Already
            ``_LIVE_ONLY_CLAUSE``-filtered by the caller. The "held"
            flag is purely for the rendered card — it does not
            influence the verdict. Empty / None ⇒ no top signal.
        last_decision: most recent decision row
            ``{timestamp, action_taken}`` from
            ``store.recent_decisions(limit=1)``. ``None`` is
            tolerated — the verdict still fires; the disambiguation
            falls back to "decision unknown, treat as passive".
        now: optional override (test seam).

    Returns:
        Dict with stable keys regardless of the verdict path.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # --- Portfolio normalisation -------------------------------------
    cash_usd: float | None = None
    total_value_usd: float | None = None
    cash_pct: float | None = None
    n_positions: int | None = None
    if isinstance(portfolio, dict):
        cash_usd = _num(portfolio.get("cash"))
        total_value_usd = _num(portfolio.get("total_value"))
        cash_pct = _num(portfolio.get("cash_pct"))
        if cash_pct is None and cash_usd is not None and total_value_usd is not None and total_value_usd > 0:
            cash_pct = (cash_usd / total_value_usd) * 100.0
        n_raw = portfolio.get("n_positions")
        if isinstance(n_raw, (int, float)) and n_raw == n_raw:
            n_positions = int(n_raw)

    # --- Top signal --------------------------------------------------
    top = _top_signal(signals or ())
    top_score = _num(top.get("ai_score")) if top else None
    top_ticker = top.get("ticker") if top and isinstance(top.get("ticker"), str) else None
    top_urgency = top.get("urgency") if top else None
    top_source = top.get("source") if top else None
    top_held = bool(top.get("held")) if top else False

    # --- Last decision freshness ------------------------------------
    last_verb = _extract_verb(last_decision.get("action_taken")) if isinstance(last_decision, dict) else None
    last_ts = _parse_ts(last_decision.get("timestamp")) if isinstance(last_decision, dict) else None
    last_age_min: float | None = None
    if last_ts is not None:
        last_age_min = (now - last_ts).total_seconds() / 60.0
        if last_age_min < 0:
            last_age_min = 0.0
    recent_fill = (
        last_verb is not None
        and last_verb in _FILL_ACTIONS
        and last_age_min is not None
        and last_age_min <= recent_fill_max_min
    )

    # --- Verdict logic ----------------------------------------------
    # NO_DATA: portfolio missing OR no usable signals (we still emit a
    # full envelope so the UI binding never sees missing fields).
    if cash_pct is None or top_score is None:
        verdict = NO_DATA
        if cash_pct is None and top_score is None:
            headline = "NO_DATA — portfolio + signals both missing."
        elif cash_pct is None:
            headline = "NO_DATA — portfolio snapshot missing."
        else:
            headline = "NO_DATA — no live signals (alert pipeline silent or filtered)."
    elif recent_fill:
        # Active deployment in the last recent_fill_max_min — the
        # cash-idle reading is transient by construction.
        verdict = BALANCED
        headline = (
            f"BALANCED — active fill {last_age_min:.0f}m ago "
            f"({last_verb}); cash level reflects an active loop."
        )
    elif cash_pct >= idle_cash_pct and top_score >= high_conviction_score:
        verdict = IDLE_DESPITE_SURGE
        headline = (
            f"IDLE_DESPITE_SURGE — {cash_pct:.0f}% cash idle while "
            f"{top_ticker or '(unknown)'} screams ai_score "
            f"{top_score:.1f}; last decision {last_verb or 'unknown'}"
            + (f" {last_age_min:.0f}m ago" if last_age_min is not None else "")
            + "."
        )
    elif cash_pct <= overdeployed_cash_pct and top_score >= high_conviction_score:
        verdict = OVERDEPLOYED
        headline = (
            f"OVERDEPLOYED — only {cash_pct:.0f}% cash with "
            f"{top_ticker or '(unknown)'} at ai_score {top_score:.1f}; "
            f"the book cannot add without trimming."
        )
    elif cash_pct >= idle_cash_pct and top_score < low_conviction_score:
        verdict = IDLE_LOW_CONVICTION
        headline = (
            f"IDLE_LOW_CONVICTION — {cash_pct:.0f}% cash idle; "
            f"loudest live signal only {top_score:.1f}. Cash idleness "
            f"is correct — nothing is screaming."
        )
    else:
        verdict = BALANCED
        headline = (
            f"BALANCED — cash {cash_pct:.0f}% vs top signal "
            f"{top_score:.1f}; level fits conviction."
        )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "verdict": verdict,
        "headline": headline,
        "portfolio": {
            "cash_usd": round(cash_usd, 2) if cash_usd is not None else None,
            "total_value_usd": round(total_value_usd, 2) if total_value_usd is not None else None,
            "cash_pct": round(cash_pct, 2) if cash_pct is not None else None,
            "n_positions": n_positions,
        },
        "top_signal": {
            "ticker": top_ticker,
            "ai_score": top_score if top_score is None else round(top_score, 2),
            "urgency": top_urgency,
            "source": top_source,
            "held": top_held,
        },
        "last_decision": {
            "verb": last_verb,
            "age_min": round(last_age_min, 1) if last_age_min is not None else None,
            "recent_fill": recent_fill,
        },
        "thresholds": {
            "idle_cash_pct": idle_cash_pct,
            "overdeployed_cash_pct": overdeployed_cash_pct,
            "high_conviction_score": high_conviction_score,
            "low_conviction_score": low_conviction_score,
            "recent_fill_max_min": recent_fill_max_min,
        },
    }
