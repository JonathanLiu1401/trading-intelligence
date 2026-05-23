"""Exit-intent audit — classify each closed sell by the stated *intent* in
the exit reason and roll up outcome per intent.

The desk question no panel answers. The existing neighbours each see a
different slice:

* ``/api/loser-autopsy`` — classifies LOSING trips by an *objective*
  (hold-days × magnitude) failure-mode taxonomy (KNIFE_CATCH /
  WHIPSAW / SLOW_BLEED / STOPPED_OUT). It does not look at the trader's
  *stated reason* for the sell — only the numerical shape. It also only
  looks at losers, not winners.
* ``/api/winner-autopsy`` — verbatim entry reason on winners; no
  classification of the exit reason at all.
* ``/api/round-trip-postmortem`` — was the exit timed correctly relative
  to the *next price drift*? It judges exit *quality*, not intent.
* ``/api/reasoning-themes`` — keyword themes over ALL decision reasoning
  (not just exit reasons), not joined to outcome.

This module fills the missing slot: for every *closed* round-trip, take
the verbatim exit reason (joined back from the contributing trade rows,
the same discipline as ``loser_autopsy._reason_for``), classify it into
one of a small set of **intent buckets** by deterministic substring
matching, then aggregate outcome per bucket so the operator can see, e.g.:
"the DEFENSIVE_CASH_RAISE bucket — exits with reasons like 'free cash',
'raise dry powder', 'redeploy' — has a -3.4% average return across 8
trips, while EARNINGS_CLEAR has +5.2%". That answers a question raw
``trade-asymmetry`` cannot: which exit *intentions* the trader executes
well, and which ones bleed money.

Intent matching is **observational, deterministic, additive**. A trip
matches an intent only when its exit-reason text contains an exact
case-insensitive phrase from the bucket's keyword list. A trip with no
exit-reason text, or one whose text matches no bucket, falls into
``UNCLASSIFIED``. Multi-bucket hits resolve to the bucket appearing first
in ``_INTENT_ORDER`` — the order is documented and tested, never
randomised by dict iteration.

Pure builder consuming ``round_trips.build_round_trips`` (the canonical
P&L aggregator, AGENTS.md #10). Observational only — never gates Opus
and adds no caps (#2/#12). Sample-size honesty: numerics emit from the
first round-trip; the **dominant-intent verdict** is withheld until
``n_round_trips >= STABLE_MIN_RTS``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips

# Verdict label withholding — a 5-trip dominant-intent claim is noise.
# Mirrors trade_asymmetry STABLE gate.
STABLE_MIN_RTS = 10

# Intent buckets in resolution order. A trip matches the FIRST bucket
# whose keyword list contains a substring of its exit reason. Order is
# from most-specific stated motive (an earnings clear is a clear cause)
# down to the catch-all (free cash). STOP_LOSS comes before
# DEFENSIVE_CASH_RAISE because a forced stop is a more specific event
# than a discretionary cash-raise.
_INTENT_ORDER = (
    "EARNINGS_CLEAR",
    "STOP_LOSS",
    "TARGET_HIT",
    "THESIS_FLIP",
    "DEFENSIVE_CASH_RAISE",
)

# Case-insensitive substring matches. Each phrase is tested against the
# lower-cased exit reason. Phrases were selected from verbatim exit text
# observed in the live decision ledger 2026-05-19..23 (DRAM, NVDA round
# trips) plus standard practitioner vocabulary.
_INTENT_KEYWORDS = {
    "EARNINGS_CLEAR": (
        "pre-earnings",
        "before earnings",
        "ahead of earnings",
        "into earnings",
        "into the print",
        "into the report",
        "before the print",
        "earnings risk",
        "earnings print",
        "earnings tomorrow",
        "post-earnings dip",
    ),
    "STOP_LOSS": (
        "stop loss",
        "stop-loss",
        "stopped out",
        "cut loss",
        "cutting loss",
        "cut the loss",
        "hard stop",
    ),
    "TARGET_HIT": (
        "target hit",
        "price target",
        "take profit",
        "took profit",
        "taking profit",
        "lock in",
        "locking in",
        "locked in",
        "profit target",
        "target reached",
    ),
    "THESIS_FLIP": (
        "no working thesis",
        "thesis broken",
        "thesis flipped",
        "thesis weakening",
        "thesis weakened",
        "thesis collapsed",
        "thesis no longer",
        "thesis is broken",
        "no thesis",
        "broken thesis",
    ),
    "DEFENSIVE_CASH_RAISE": (
        "raise cash",
        "raising cash",
        "raised cash",
        "free cash",
        "free up cash",
        "freeing cash",
        "free $",
        "dry powder",
        "redeploy",
        "redeploying",
        "re-deploy",
        "raise dry powder",
        "raising dry powder",
        "optionality",
    ),
}


def _classify_exit_intent(exit_reason: str | None) -> str:
    """Deterministic intent label for one exit reason. Pure, never raises.

    Returns the FIRST bucket in ``_INTENT_ORDER`` whose keyword list
    contains a case-insensitive substring of ``exit_reason``. Returns
    ``"UNCLASSIFIED"`` when ``exit_reason`` is missing / empty or matches
    no bucket. Order is load-bearing for multi-bucket text — a "raise
    cash ahead of earnings" reason resolves to EARNINGS_CLEAR, not
    DEFENSIVE_CASH_RAISE.
    """
    if not exit_reason:
        return "UNCLASSIFIED"
    s = str(exit_reason).lower()
    for intent in _INTENT_ORDER:
        if any(kw in s for kw in _INTENT_KEYWORDS[intent]):
            return intent
    return "UNCLASSIFIED"


def _reason_for(trade_ids: list, by_id: dict) -> str | None:
    """Verbatim ``reason`` of the closing (last) trade in the round-trip.

    Mirrors ``loser_autopsy._reason_for(..., pick_last=True)`` — never
    NLP-parsed for trading logic, only surfaced for classification and
    UI display. Missing id / absent row / empty string → ``None``.
    """
    if not trade_ids:
        return None
    tid = trade_ids[-1]
    row = by_id.get(tid)
    if not row:
        return None
    r = row.get("reason")
    return r if (r is not None and str(r).strip() != "") else None


def _bucket_stats(trips: list[dict]) -> dict:
    """Aggregate outcome stats for one intent bucket. Pure."""
    n = len(trips)
    if n == 0:
        return {
            "n": 0,
            "total_pnl_usd": 0.0,
            "avg_pnl_usd": None,
            "avg_pnl_pct": None,
            "win_rate_pct": None,
            "median_hold_days": None,
            "n_wins": 0,
            "n_losses": 0,
        }
    pnls = [float(t.get("pnl_usd") or 0.0) for t in trips]
    pcts = [t.get("pnl_pct") for t in trips if t.get("pnl_pct") is not None]
    holds = [t.get("hold_days") for t in trips
             if t.get("hold_days") is not None]
    n_wins = sum(1 for p in pnls if p > 0)
    n_losses = sum(1 for p in pnls if p < 0)
    n_decided = n_wins + n_losses
    median_hold = None
    if holds:
        sv = sorted(holds)
        m = len(sv) // 2
        median_hold = (sv[m] if len(sv) % 2
                       else round((sv[m - 1] + sv[m]) / 2.0, 4))
    return {
        "n": n,
        "total_pnl_usd": round(sum(pnls), 4),
        "avg_pnl_usd": round(sum(pnls) / n, 4),
        "avg_pnl_pct": (round(sum(pcts) / len(pcts), 4)
                        if pcts else None),
        "win_rate_pct": (round(n_wins / n_decided * 100.0, 2)
                         if n_decided else None),
        "median_hold_days": median_hold,
        "n_wins": n_wins,
        "n_losses": n_losses,
    }


def build_exit_intent_audit(trades: list[dict],
                            worst_n: int = 5,
                            now: datetime | None = None) -> dict:
    """Per-intent exit audit over closed round-trips. Pure, never raises.

    ``trades`` must be ``Store.recent_trades()``-shaped, oldest→newest —
    ``build_round_trips`` reads rows in sequence and does not sort.
    ``worst_n`` caps the per-bucket recent-examples list (most negative
    P&L first); aggregates are always over *all* trips in the bucket.
    """
    now = now or datetime.now(timezone.utc)
    rts = build_round_trips(trades)
    n_rts = len(rts)

    by_id: dict = {}
    for t in trades:
        tid = t.get("id")
        if tid is not None:
            by_id[tid] = t

    # Tag each round-trip with its exit reason + classified intent.
    tagged: list[dict] = []
    for rt in rts:
        reason = _reason_for(rt.get("exit_trade_ids") or [], by_id)
        intent = _classify_exit_intent(reason)
        tagged.append({
            "ticker": rt.get("ticker"),
            "type": rt.get("type"),
            "entry_ts": rt.get("entry_ts"),
            "exit_ts": rt.get("exit_ts"),
            "hold_days": rt.get("hold_days"),
            "pnl_usd": round(float(rt.get("pnl_usd") or 0.0), 4),
            "pnl_pct": rt.get("pnl_pct"),
            "exit_reason": reason,
            "intent": intent,
        })

    # Bucket round-trips by intent.
    by_intent: dict[str, list[dict]] = {
        intent: [] for intent in _INTENT_ORDER
    }
    by_intent["UNCLASSIFIED"] = []
    for c in tagged:
        by_intent[c["intent"]].append(c)

    # Per-bucket stats + worst-N examples.
    buckets: list[dict] = []
    for intent in (*_INTENT_ORDER, "UNCLASSIFIED"):
        trips = by_intent[intent]
        stats = _bucket_stats(trips)
        # Worst (most negative pnl_usd) first; ticker as deterministic
        # tie-break so the example list is stable across identical pnls.
        worst = sorted(trips, key=lambda c: (c["pnl_usd"], c["ticker"] or ""))
        buckets.append({
            "intent": intent,
            **stats,
            "examples": worst[:max(0, worst_n)],
        })

    # Dominant intent (by count). Deterministic ordering: ties broken by
    # the fixed _INTENT_ORDER sequence (UNCLASSIFIED last). A book full
    # of UNCLASSIFIED exits still picks UNCLASSIFIED as dominant — the
    # signal is "we lack classified exit language", which is itself
    # actionable.
    full_order = (*_INTENT_ORDER, "UNCLASSIFIED")
    dominant_intent = None
    if n_rts > 0:
        dominant_intent = max(
            full_order,
            key=lambda i: (len(by_intent[i]), -full_order.index(i)),
        )

    # ---- state -----------------------------------------------------------
    if n_rts == 0:
        state = "NO_DATA"
    elif n_rts >= STABLE_MIN_RTS:
        state = "STABLE"
    else:
        state = "EMERGING"

    # ---- verdict (gated to STABLE) ---------------------------------------
    # The desk question: when the DOMINANT stated intent has negative
    # average P&L, the trader is systematically losing money on the kind
    # of exits they make most often. That's the signal worth surfacing.
    verdict = None
    verdict_reason = None
    dom_stats = (next((b for b in buckets if b["intent"] == dominant_intent),
                      None) if dominant_intent else None)
    if state == "STABLE" and dom_stats and dom_stats["n"] >= 3:
        avg_pnl = dom_stats["avg_pnl_usd"]
        avg_pct = dom_stats["avg_pnl_pct"]
        if dominant_intent == "UNCLASSIFIED":
            verdict = "INTENT_UNCLEAR"
            verdict_reason = (
                f"the dominant exit category is UNCLASSIFIED — "
                f"{dom_stats['n']} of {n_rts} round-trips closed without "
                f"language matching any intent bucket; exit motives are "
                f"unstated or ad-hoc")
        elif avg_pnl is not None and avg_pnl < 0:
            verdict = "DOMINANT_INTENT_BLEED"
            pct_clause = (f" ({avg_pct:+.2f}%/trip)"
                          if avg_pct is not None else "")
            verdict_reason = (
                f"dominant exit intent is {dominant_intent} "
                f"({dom_stats['n']}/{n_rts} round-trips) but it averages "
                f"${avg_pnl:+.2f}/trip{pct_clause} — the most common "
                f"reason to sell is also a money loser")
        else:
            verdict = "DOMINANT_INTENT_HEALTHY"
            pct_clause = (f" ({avg_pct:+.2f}%/trip)"
                          if avg_pct is not None else "")
            verdict_reason = (
                f"dominant exit intent {dominant_intent} averages "
                f"${avg_pnl:+.2f}/trip{pct_clause} across "
                f"{dom_stats['n']}/{n_rts} round-trips — the most common "
                f"reason to sell is profitable on average")

    # ---- headline --------------------------------------------------------
    if state == "NO_DATA":
        headline = "No closed round-trips yet — exit-intent audit empty."
    elif state == "EMERGING":
        dom_n = dom_stats["n"] if dom_stats else 0
        headline = (
            f"Emerging — {n_rts} of {STABLE_MIN_RTS} round-trips for a "
            f"stable read. Dominant intent so far: {dominant_intent} "
            f"({dom_n}/{n_rts}) — verdict withheld.")
    elif verdict is not None:
        headline = f"{verdict} — {verdict_reason}."
    else:
        # STABLE but dominant bucket too thin for a verdict.
        dom_n = dom_stats["n"] if dom_stats else 0
        headline = (
            f"Stable exit-intent mix across {n_rts} round-trips; "
            f"dominant {dominant_intent} ({dom_n}/{n_rts}) has too few "
            f"trips for a verdict (need ≥3).")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "headline": headline,
        "n_round_trips": n_rts,
        "dominant_intent": dominant_intent,
        "buckets": buckets,
        "intent_order": list(full_order),
        "stable_min_round_trips": STABLE_MIN_RTS,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json
    from paper_trader.store import get_store

    s = get_store()
    rep = build_exit_intent_audit(list(reversed(s.recent_trades(2000))))
    print(json.dumps(rep, indent=2, default=str))
