"""Intent-followthrough skill — does the bot execute its own conditional intents?

``decision_conditionals`` surfaces STANDING intents extracted from recent
reasoning prose ("wait for cash session", "ready to trim on bounce",
"rotate into LITE/LNOK", "preserve dry powder for MRVL earnings"). It
answers *what is the bot planning to do next?* — but stops there. Nothing
in the existing analytic stack then asks the **observational** follow-up:
*did the bot actually do the thing it said it would do?*

That gap matters because every other reasoning skill panel — decision-
vapor (specificity), thesis-drift (open-position thesis), exit-intent-
audit (post-hoc sell motives), reasoning-coherence (HOLD-to-HOLD
stability) — grades the *text*, not the *bridge from text to action*.
A bot that emits crisp, specific, on-thesis "wait for X then buy Y"
statements every cycle but never actually executes Y has perfect
``decision_vapor_skill`` and zero followthrough — and only this panel
catches it.

The desk question:

  Over the last ``window_hours`` of decisions, for the actionable
  intents (watch-for, ready-to, will-if, look-for, rotate-into, if-then),
  what fraction were FOLLOWED by a matching trade within
  ``eval_window_hours`` of the stated intent? What fraction sat past the
  evaluation window with no matching trade (ABANDONED)? And separately —
  for the abstention intents (preserve-for, too-early-to), did the
  preserve get deployed and did the too-early-to actually hold?

This module is a pure builder. It takes two lists already maintained by
the store — ``recent_decisions`` and ``recent_trades`` — composes
``decision_conditionals.build_decision_conditionals`` verbatim as the
SSOT intent extractor (no regex duplication: an intent is whatever
``decision_conditionals`` says it is, AGENTS.md invariant #10), then
joins each intent against the trade tape that came after its decision_ts.

Match contract is **deterministic + conservative**:

* ``target_tickers`` for the intent = ``[intent.ticker]`` when the
  intent carries a parsed ticker (from ``action_taken``), else ``None``
  (any-ticker scope — a "watch-for the cash session" intent doesn't
  name a ticker, so any subsequent trade satisfies it).
* ``verb_hint`` is inferred from a small keyword scan over the intent's
  verbatim snippet (``add``/``buy``/``long``/``scale-in`` → BUY family;
  ``trim``/``exit``/``sell``/``scale-out``/``dump`` → SELL family).
  Hint is optional — when present it tightens the match; when absent
  any trade verb satisfies the intent.
* A match is the OLDEST trade after the intent's ``decision_ts`` AND
  within ``eval_window_hours`` AND satisfying both filters above.
  Oldest-after-intent (not newest) so a chain of intents on the same
  ticker each get credited to their first execution, not all crediting
  the same late trade.

Verdict ladder over the actionable bucket (denominator = FOLLOWED +
ABANDONED; PENDING intents are excluded because the verdict on them is
not yet in):

  * ``NO_DATA`` — no decisions in window or no intents extracted.
  * ``NO_RESOLVED`` — intents present but all PENDING (all within their
    evaluation window). The verdict on this desk is not yet decided.
  * ``DISCIPLINED`` — followthrough rate ≥ ``discipline_floor``.
  * ``DRIFTING`` — rate ∈ [drifting_floor, discipline_floor).
  * ``ABANDONED`` — rate < ``drifting_floor`` AND ≥ ``abandoned_min_n``
    abandoned intents (sample-size honesty: a single abandoned intent
    doesn't earn the ABANDONED label).

Observational only — never gates Opus, never adds caps (AGENTS.md
invariants #2 / #12). Never raises — every input degrades to a safe
NO_DATA report on garbage.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from .decision_conditionals import build_decision_conditionals

DEFAULT_WINDOW_HOURS = 24.0
DEFAULT_STALE_HOURS = 12.0
DEFAULT_EVAL_WINDOW_HOURS = 12.0
DEFAULT_DISCIPLINE_FLOOR = 0.66
DEFAULT_DRIFTING_FLOOR = 0.33
DEFAULT_ABANDONED_MIN_N = 3
DEFAULT_MAX_INTENTS = 30

# Intent kinds that imply an *action* the bot should take. The
# remaining kinds (preserve-for, too-early-to) are abstention intents —
# they're scored on a different axis and excluded from the followthrough
# rate.
_ACTIONABLE_KINDS = frozenset({
    "watch-for", "ready-to", "will-if", "look-for", "rotate-into",
    "if-then",
})
_ABSTENTION_KINDS = frozenset({"preserve-for", "too-early-to"})

# Verb hint inference. The snippet matched by a pattern is scanned for
# these keywords case-insensitively; the first hit wins. A BUY hint
# matches trade.action ∈ {BUY, BUY_CALL, BUY_PUT}; SELL hint matches
# {SELL, SELL_CALL, SELL_PUT}. ``REBALANCE`` is verb-neutral and never
# vetoed by either hint.
_BUY_VERB_KEYWORDS = (
    "add", "adding", "buy", "buying", "long", "go long",
    "scale in", "scale-in", "scaling in", "build", "building",
    "initiate", "initiating", "open", "opening", "enter", "entering",
    "rotate into", "rotating into", "accumulate",
)
_SELL_VERB_KEYWORDS = (
    "trim", "trimming", "exit", "exiting", "sell", "selling",
    "scale out", "scale-out", "scaling out", "dump", "dumping",
    "close", "closing", "reduce", "reducing", "lighten", "lightening",
    "take profit", "take profits", "harvest", "harvesting",
    "stop out", "stopped out",
)
_BUY_VERBS = frozenset({"BUY", "BUY_CALL", "BUY_PUT"})
_SELL_VERBS = frozenset({"SELL", "SELL_CALL", "SELL_PUT"})


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


def _infer_verb_hint(snippet: str) -> str | None:
    """Return ``"BUY"`` / ``"SELL"`` / ``None`` from a verbatim intent snippet.

    The scan is case-insensitive substring match in keyword-list order.
    SELL keywords are checked first so a snippet like
    "scale out of NVDA to add MU" is hinted SELL (the local action),
    not BUY. Returns ``None`` when neither family is detected.
    """
    if not isinstance(snippet, str) or not snippet:
        return None
    s = snippet.lower()
    for kw in _SELL_VERB_KEYWORDS:
        if kw in s:
            return "SELL"
    for kw in _BUY_VERB_KEYWORDS:
        if kw in s:
            return "BUY"
    return None


def _verb_matches_hint(trade_action: Any, verb_hint: str | None) -> bool:
    """A trade's action satisfies the hint when:

    * ``verb_hint is None`` (any verb satisfies); or
    * the action is in the corresponding family; or
    * the action is ``REBALANCE`` (verb-neutral, satisfies either).
    """
    if verb_hint is None:
        return True
    if not isinstance(trade_action, str):
        return False
    a = trade_action.upper().strip()
    if a == "REBALANCE":
        return True
    if verb_hint == "BUY":
        return a in _BUY_VERBS
    if verb_hint == "SELL":
        return a in _SELL_VERBS
    return False


def _trade_ts_in_window(trade_ts: datetime, intent_ts: datetime,
                        eval_window_hours: float) -> bool:
    """A trade matches when it is strictly AFTER the intent and within
    the evaluation window. Strict-after avoids crediting a trade that
    fired in the SAME microsecond as the intent's decision (the
    decision row is the *source* of the intent — the matching trade
    must come after, not at, that moment)."""
    if trade_ts <= intent_ts:
        return False
    age_after = (trade_ts - intent_ts).total_seconds()
    return age_after <= max(0.0, eval_window_hours) * 3600.0


def _match_intent_against_trades(
    intent: dict,
    trades_sorted: list[tuple[datetime, dict]],
    *,
    eval_window_hours: float,
) -> dict | None:
    """Find the OLDEST matching trade after ``intent.decision_ts``.

    ``trades_sorted`` is pre-sorted ascending by timestamp (oldest
    first) so the first match is the oldest. Returns ``None`` when no
    trade in the evaluation window satisfies the ticker + verb filters.
    """
    intent_ts = _parse_iso(intent.get("decision_ts"))
    if intent_ts is None:
        return None

    target_ticker = intent.get("ticker")  # may be None → any ticker ok
    verb_hint = _infer_verb_hint(intent.get("text") or "")

    for trade_ts, trade in trades_sorted:
        if not _trade_ts_in_window(trade_ts, intent_ts, eval_window_hours):
            # Either before/at intent (keep scanning — list might not be
            # strictly time-ordered relative to intents on different
            # decision_ts), or past the eval window. The list is sorted
            # ascending overall, so once we pass the window for THIS
            # intent we can break — but only if trade_ts > intent_ts.
            if trade_ts > intent_ts:
                break  # past the window
            continue
        if target_ticker and (trade.get("ticker") or "").upper() != target_ticker.upper():
            continue
        if not _verb_matches_hint(trade.get("action"), verb_hint):
            continue
        age_after_h = (trade_ts - intent_ts).total_seconds() / 3600.0
        return {
            "trade_id": trade.get("id"),
            "trade_ts": trade_ts.isoformat(),
            "trade_ticker": trade.get("ticker"),
            "trade_action": trade.get("action"),
            "age_after_intent_h": round(age_after_h, 2),
            "verb_hint": verb_hint,
            "verb_strict_match": verb_hint is not None,
        }
    return None


def _classify_actionable(intent: dict, match: dict | None, *,
                         now: datetime, eval_window_hours: float) -> str:
    """``FOLLOWED`` if a match was found; otherwise PENDING (intent still
    within evaluation window) or ABANDONED (window passed without match).
    """
    if match is not None:
        return "FOLLOWED"
    intent_ts = _parse_iso(intent.get("decision_ts"))
    if intent_ts is None:
        return "ABANDONED"
    age_h = max(0.0, (now - intent_ts).total_seconds() / 3600.0)
    if age_h < max(0.0, eval_window_hours):
        return "PENDING"
    return "ABANDONED"


def _classify_abstention(intent: dict, trades_sorted: list[tuple[datetime, dict]],
                         *, now: datetime, eval_window_hours: float) -> tuple[str, dict | None]:
    """Abstention-intent scoring.

    * ``preserve-for`` — success = bot actually DEPLOYED preserved cash
      with a BUY inside the evaluation window. PRESERVED (no BUY,
      still within window) / DEPLOYED (a BUY landed — preserve hit) /
      DEPLOYED_EARLY is collapsed into DEPLOYED for v1; the snippet
      itself often names the catalyst so a manual reader can judge
      timing. Past window with no BUY → DEPLOYED_NEVER (the preserve
      sat unused — the canonical "dry powder dead-weight" pattern).
    * ``too-early-to <verb>`` — success = bot HELD OFF on the named
      action. The verb hint flips polarity here: a verb-match trade in
      the window means the bot **broke its own restraint**.
      RESTRAINED (no matching action) / BROKE_RESTRAINT (matching
      action landed).

    Returns ``(status, match_dict_or_None)`` so the caller can render
    the offending trade when applicable.
    """
    intent_ts = _parse_iso(intent.get("decision_ts"))
    if intent_ts is None:
        return "UNKNOWN", None
    age_h = max(0.0, (now - intent_ts).total_seconds() / 3600.0)
    within_window = age_h < max(0.0, eval_window_hours)
    target_ticker = intent.get("ticker")  # often None for cash-preserve
    snippet = intent.get("text") or ""
    kind = intent.get("kind")

    if kind == "preserve-for":
        # Success = any BUY in the eval window deploys the preserved cash.
        verb_hint = "BUY"
        for trade_ts, trade in trades_sorted:
            if not _trade_ts_in_window(trade_ts, intent_ts, eval_window_hours):
                if trade_ts > intent_ts:
                    break
                continue
            if target_ticker and (trade.get("ticker") or "").upper() != target_ticker.upper():
                continue
            if not _verb_matches_hint(trade.get("action"), verb_hint):
                continue
            return "DEPLOYED", {
                "trade_id": trade.get("id"),
                "trade_ts": trade_ts.isoformat(),
                "trade_ticker": trade.get("ticker"),
                "trade_action": trade.get("action"),
                "age_after_intent_h": round(
                    (trade_ts - intent_ts).total_seconds() / 3600.0, 2),
            }
        return ("PRESERVED" if within_window else "DEPLOYED_NEVER"), None

    if kind == "too-early-to":
        # Polarity flips: a verb-match trade in the window MEANS the bot
        # broke its restraint. Same verb hint scan we use for actionable.
        verb_hint = _infer_verb_hint(snippet)
        for trade_ts, trade in trades_sorted:
            if not _trade_ts_in_window(trade_ts, intent_ts, eval_window_hours):
                if trade_ts > intent_ts:
                    break
                continue
            if target_ticker and (trade.get("ticker") or "").upper() != target_ticker.upper():
                continue
            if not _verb_matches_hint(trade.get("action"), verb_hint):
                continue
            return "BROKE_RESTRAINT", {
                "trade_id": trade.get("id"),
                "trade_ts": trade_ts.isoformat(),
                "trade_ticker": trade.get("ticker"),
                "trade_action": trade.get("action"),
                "age_after_intent_h": round(
                    (trade_ts - intent_ts).total_seconds() / 3600.0, 2),
            }
        return "RESTRAINED", None

    return "UNKNOWN", None


def build_intent_followthrough(
    decisions: Sequence[Any] | None,
    trades: Sequence[Any] | None,
    *,
    now: datetime | None = None,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    stale_hours: float = DEFAULT_STALE_HOURS,
    eval_window_hours: float = DEFAULT_EVAL_WINDOW_HOURS,
    discipline_floor: float = DEFAULT_DISCIPLINE_FLOOR,
    drifting_floor: float = DEFAULT_DRIFTING_FLOOR,
    abandoned_min_n: int = DEFAULT_ABANDONED_MIN_N,
    max_intents: int = DEFAULT_MAX_INTENTS,
) -> dict[str, Any]:
    """Score the bot's intent → trade followthrough.

    ``decisions`` and ``trades`` are the row lists ``store.recent_decisions``
    and ``store.recent_trades`` already return. Decisions feed the SSOT
    intent extractor (``build_decision_conditionals``); trades are
    sorted ascending and joined against each intent's eval window.

    Output (always a dict, never raises):
      ``state``: ``NO_DATA`` | ``OK``
      ``verdict``: ``NO_DATA`` | ``NO_RESOLVED`` | ``DISCIPLINED`` |
        ``DRIFTING`` | ``ABANDONED``
      ``headline``: short verbatim string
      ``n_intents``: total intents considered (pre-cap)
      ``n_actionable``: actionable intents (denominator before
        PENDING exclusion)
      ``n_followed`` / ``n_pending`` / ``n_abandoned``: per-status counts
        over actionable intents
      ``followthrough_rate``: n_followed / (n_followed + n_abandoned),
        None when the denominator is 0
      ``abstention``: ``{n_preserve_deployed, n_preserve_active,
        n_preserve_dead, n_restraint_held, n_restraint_broken}``
      ``by_kind``: ``{kind: {followed, pending, abandoned}}``
      ``intents``: list of evaluated intent dicts (newest first, capped)
      ``window_hours``, ``stale_hours``, ``eval_window_hours``,
      ``discipline_floor``, ``drifting_floor``, ``abandoned_min_n``,
      ``as_of``: echo of inputs
    """
    now = now or datetime.now(timezone.utc)

    base = build_decision_conditionals(
        decisions,
        now=now,
        window_hours=window_hours,
        stale_hours=stale_hours,
        max_intents=max(50, int(max_intents) * 2),
    )
    raw_intents = base.get("intents", []) or []

    # Build (ts, trade) list, ascending by ts. Skip rows with unparseable ts.
    trade_rows: list[tuple[datetime, dict]] = []
    for t in (trades or []):
        if not isinstance(t, dict):
            continue
        ts = _parse_iso(t.get("timestamp"))
        if ts is None:
            continue
        trade_rows.append((ts, t))
    trade_rows.sort(key=lambda r: r[0])

    evaluated: list[dict[str, Any]] = []
    n_followed = n_pending = n_abandoned = 0
    n_preserve_deployed = n_preserve_active = n_preserve_dead = 0
    n_restraint_held = n_restraint_broken = 0
    by_kind: dict[str, dict[str, int]] = {}

    for it in raw_intents:
        if not isinstance(it, dict):
            continue
        kind = it.get("kind")
        out: dict[str, Any] = {
            "decision_id": it.get("decision_id"),
            "decision_ts": it.get("decision_ts"),
            "ticker": it.get("ticker"),
            "kind": kind,
            "text": it.get("text"),
            "age_hours": it.get("age_hours"),
            "stale": it.get("stale"),
            "action_taken": it.get("action_taken"),
        }

        if kind in _ACTIONABLE_KINDS:
            match = _match_intent_against_trades(
                it, trade_rows, eval_window_hours=eval_window_hours)
            status = _classify_actionable(
                it, match, now=now, eval_window_hours=eval_window_hours)
            out["bucket"] = "ACTIONABLE"
            out["status"] = status
            out["match"] = match
            out["verb_hint"] = _infer_verb_hint(it.get("text") or "")
            slot = by_kind.setdefault(
                kind, {"followed": 0, "pending": 0, "abandoned": 0})
            if status == "FOLLOWED":
                n_followed += 1
                slot["followed"] += 1
            elif status == "PENDING":
                n_pending += 1
                slot["pending"] += 1
            elif status == "ABANDONED":
                n_abandoned += 1
                slot["abandoned"] += 1
        elif kind in _ABSTENTION_KINDS:
            status, match = _classify_abstention(
                it, trade_rows, now=now, eval_window_hours=eval_window_hours)
            out["bucket"] = "ABSTENTION"
            out["status"] = status
            out["match"] = match
            if status == "DEPLOYED":
                n_preserve_deployed += 1
            elif status == "PRESERVED":
                n_preserve_active += 1
            elif status == "DEPLOYED_NEVER":
                n_preserve_dead += 1
            elif status == "RESTRAINED":
                n_restraint_held += 1
            elif status == "BROKE_RESTRAINT":
                n_restraint_broken += 1
        else:
            # Unknown kind — surface as UNKNOWN bucket without scoring.
            out["bucket"] = "UNKNOWN"
            out["status"] = "UNKNOWN"
            out["match"] = None

        evaluated.append(out)

    # Cap to max_intents NEWEST (build_decision_conditionals already
    # returns newest first).
    capped = evaluated[: max(0, int(max_intents))]

    n_total = len(evaluated)
    n_actionable = n_followed + n_pending + n_abandoned
    denom = n_followed + n_abandoned
    followthrough_rate = (n_followed / denom) if denom > 0 else None

    # Verdict ladder
    if n_total == 0:
        state, verdict = "NO_DATA", "NO_DATA"
        headline = "no intents in window"
    elif denom == 0:
        # All actionable intents still pending — verdict not yet decided.
        state, verdict = "OK", "NO_RESOLVED"
        headline = (
            f"{n_pending} actionable intent(s) still inside the "
            f"{eval_window_hours:g}h evaluation window — verdict pending"
        )
    elif followthrough_rate is not None and followthrough_rate >= discipline_floor:
        state, verdict = "OK", "DISCIPLINED"
        headline = (
            f"{n_followed}/{denom} actionable intent(s) followed through "
            f"({100.0 * followthrough_rate:.0f}% — disciplined)"
        )
    elif (
        followthrough_rate is not None
        and followthrough_rate < drifting_floor
        and n_abandoned >= max(1, int(abandoned_min_n))
    ):
        state, verdict = "OK", "ABANDONED"
        headline = (
            f"{n_followed}/{denom} followed; {n_abandoned} abandoned "
            f"({100.0 * followthrough_rate:.0f}% followthrough — bot states "
            f"plans it does not execute)"
        )
    else:
        state, verdict = "OK", "DRIFTING"
        # When rate is None at this point we're below discipline_floor by
        # default — guard the display anyway.
        rate_str = (
            f"{100.0 * followthrough_rate:.0f}%"
            if followthrough_rate is not None else "—"
        )
        headline = (
            f"{n_followed}/{denom} followed, {n_abandoned} abandoned "
            f"({rate_str} followthrough — drifting)"
        )

    return {
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "n_intents": n_total,
        "n_actionable": n_actionable,
        "n_followed": n_followed,
        "n_pending": n_pending,
        "n_abandoned": n_abandoned,
        "followthrough_rate": followthrough_rate,
        "abstention": {
            "n_preserve_deployed": n_preserve_deployed,
            "n_preserve_active": n_preserve_active,
            "n_preserve_dead": n_preserve_dead,
            "n_restraint_held": n_restraint_held,
            "n_restraint_broken": n_restraint_broken,
        },
        "by_kind": by_kind,
        "intents": capped,
        "window_hours": float(window_hours),
        "stale_hours": float(stale_hours),
        "eval_window_hours": float(eval_window_hours),
        "discipline_floor": float(discipline_floor),
        "drifting_floor": float(drifting_floor),
        "abandoned_min_n": int(abandoned_min_n),
        "as_of": now.isoformat(),
    }


def is_followthrough_abandoned(report: dict | None) -> bool:
    """Single-bool ``operator should pay attention`` view, mirroring the
    sibling ``decision_conditionals.is_intents_stale`` shape.
    """
    if not isinstance(report, dict):
        return False
    return report.get("verdict") == "ABANDONED"


__all__ = [
    "DEFAULT_WINDOW_HOURS",
    "DEFAULT_STALE_HOURS",
    "DEFAULT_EVAL_WINDOW_HOURS",
    "DEFAULT_DISCIPLINE_FLOOR",
    "DEFAULT_DRIFTING_FLOOR",
    "DEFAULT_ABANDONED_MIN_N",
    "DEFAULT_MAX_INTENTS",
    "build_intent_followthrough",
    "is_followthrough_abandoned",
]
