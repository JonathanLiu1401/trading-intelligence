"""Decision-conditionals — surface the bot's STANDING conditional intents.

Every other reasoning analytic looks backward at what was *said* and what was
*done*: ``decision_vapor_skill`` grades reasoning specificity on FILLED
trades, ``thesis_drift`` re-tests the open-position thesis, ``exit_intent_
audit`` classifies *closed* sells by stated motive, ``reasoning_coherence``
measures HOLD-to-HOLD stability. None surface the *forward* slice — the
explicit conditional intents the bot itself stated ("wait for the cash
session", "if it holds 220 I'll add", "rotating into LITE/LNOK") and that
are still STANDING (within the freshness window) without a follow-up action.

A standing intent is the bot's own forward-looking commitment. Operators
have no view into "what is the bot planning to do next?" today; they can
only inspect *what happened*. A bot that says "wait for cash session" then
sits through five cash sessions has an unfulfilled stated plan worth
surfacing — and a bot that says "ready to trim on bounce" then never trims
has a quiet behavioural flag the analyst can act on.

This module is the FUTURE-INTENT companion to ``signal_followthrough`` (did
the signals on screen translate into action), ``exit_intent_audit`` (post-
hoc sell motives), and ``thesis_drift`` (is the open-position thesis still
intact). It answers: "what did the bot SAY it would do next, that it has
not yet done?"

Pattern extraction is regex-based against the reasoning prose (the JSON-
envelope ``decision.reasoning`` field, with raw-string fallback — the
``decision_vapor_skill._extract_reasoning_text`` precedent). Each match
yields ONE intent with a verbatim snippet (≤120 chars), a kind tag, a
ticker (parsed from the decision's ``action_taken`` field), a ``stale``
boolean, and the source decision's id/ts/age.

Dedup is applied within (ticker, kind, normalized prefix): a "wait for cash
session" said in five consecutive HOLDs surfaces ONE intent on the newest
decision, not five. This matches operator expectation that a *standing*
intent is a single line item, not a transcript.

Verdict ladder:

* ``NO_DATA`` — zero decisions in the window.
* ``NO_INTENTS`` — decisions present but no conditional patterns matched.
* ``STANDING_INTENTS`` — ≥1 intent, majority fresh (age < stale_hours).
* ``STALE_INTENTS`` — ≥1 intent, majority stale (age ≥ stale_hours). The
  bot said it would do something and aged past the freshness window
  without follow-up — a quiet discipline flag.

Pure builder. Decisions in, dict out, never raises. Advisory only — never
gates Opus, no caps (AGENTS.md #12).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

DEFAULT_WINDOW_HOURS = 24.0
DEFAULT_STALE_HOURS = 12.0
DEFAULT_MAX_INTENTS = 20
DEFAULT_SAMPLE_PER_KIND = 3

# Each pattern captures ONE conditional intent. Order matters only for
# precedence inside dedup (earlier patterns win on identical snippets) —
# the regex engine itself does not backtrack across patterns.
#
# All patterns are case-insensitive and bound their capture to ≤120 chars
# before the next sentence boundary (``,.;:!?\n``), so a single match
# cannot run away into a paragraph.
_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # "wait for cash session price action" / "watching for follow-through"
    ("watch-for", re.compile(
        r"(?i)\b(?:wait|watch|waiting|watching)\s+for\s+([^,.;:!?\n]{8,120})"
    )),
    # "if it holds 220 I'll add" / "if it breaks 200 will exit"
    ("if-then", re.compile(
        r"(?i)\bif\s+([^,.;:!?\n]{4,60}?)\s+"
        r"(?:then|will|i'?ll|i\s+will|may|might|would|breaks?|holds?|hits?|"
        r"drops?|reaches?|crosses?)\s+([^,.;:!?\n]{4,80})"
    )),
    # "ready to add on breakout" / "prepared to trim on bounce"
    ("ready-to", re.compile(
        r"(?i)\b(?:ready|prepared|positioned)\s+to\s+(\w+(?:\s+[^,.;:!?\n]{4,80})?)"
    )),
    # "will add if it holds" / "plan to exit when momentum breaks"
    ("will-if", re.compile(
        r"(?i)\b(?:will|plan(?:ning)?\s+to|going\s+to|intend\s+to)\s+"
        r"(\w+(?:\s+\w+){0,3})\s+"
        r"(?:if|on|when|once|after)\s+([^,.;:!?\n]{4,80})"
    )),
    # "looking for follow-through" / "look for entry on dip"
    ("look-for", re.compile(
        r"(?i)\blook(?:ing)?\s+for\s+([^,.;:!?\n]{8,120})"
    )),
    # "preserve cash for tomorrow's open" / "preserve dry powder to redeploy"
    ("preserve-for", re.compile(
        r"(?i)\bpreserv\w+\s+(?:cash|capital|dry\s+powder|powder)\s+"
        r"(?:for|to)\s+([^,.;:!?\n]{4,120})"
    )),
    # "premature to dump" / "too early to add"
    ("too-early-to", re.compile(
        r"(?i)\b(?:premature|too\s+early)\s+to\s+(\w+(?:\s+\w+){0,4})"
    )),
    # "rotating into LITE/LNOK" / "rotate to defensive names"
    ("rotate-into", re.compile(
        r"(?i)\brotat\w+\s+(?:into|to|toward|towards)\s+([^,.;:!?\n]{4,120})"
    )),
]

# Action verbs at the head of a decision's ``action_taken`` string. The
# canonical format is ``"BUY NVDA → FILLED"`` (AGENTS.md #11). HOLD and
# NO_DECISION are *not* trades but their reasoning still carries forward
# intent (e.g. an HOLD that says "wait for cash session" is the bot's
# clearest standing intent).
_KNOWN_VERBS = frozenset({
    "BUY", "SELL", "HOLD", "BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT",
    "REBALANCE",
})

# Pseudo-tickers that ``_parse_action_ticker`` (dashboard.py invariant #11)
# nullifies so they cannot pollute per-position panels. Mirrored here so
# this module agrees on what "no ticker" means.
_NULL_TICKERS = frozenset({"CASH", "NONE", "NULL", "N/A"})


def _parse_iso(ts: Any) -> datetime | None:
    """Coerce an ISO-8601 string into a tz-aware datetime, else None."""
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


def _extract_reasoning_text(decision: dict) -> str:
    """Pull the natural-language prose from a decision row.

    The canonical ``reasoning`` is Opus's raw JSON envelope; the prose
    lives at ``decision.reasoning``. We fall back to the raw string when
    parsing fails — a ``parse_failed:``-prefixed row still has prose worth
    scanning. Mirrors ``decision_vapor_skill._extract_reasoning_text``.
    """
    raw = decision.get("reasoning")
    if not isinstance(raw, str) or not raw:
        return ""
    try:
        env = json.loads(raw)
        if isinstance(env, dict):
            inner = env.get("decision")
            if isinstance(inner, dict):
                txt = inner.get("reasoning")
                if isinstance(txt, str) and txt:
                    return txt
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return raw


def _extract_ticker(decision: dict) -> str | None:
    """Pull the ticker from ``action_taken`` (canonical ``"VERB TICKER → STATUS"``).

    Falls back to the JSON envelope's ``decision.ticker`` if action_taken
    is missing/malformed. Returns None for CASH / NONE / NULL pseudo-
    tickers (the dashboard ``_parse_action_ticker`` invariant #11).
    """
    at = decision.get("action_taken")
    if isinstance(at, str) and at:
        # "BUY NVDA → FILLED" / "HOLD NVDA → HOLD" / "NO_DECISION"
        parts = at.split()
        if len(parts) >= 2 and parts[0].upper() in _KNOWN_VERBS:
            tk = parts[1].upper().strip(",.;:!?")
            if tk and tk not in _NULL_TICKERS:
                return tk
    raw = decision.get("reasoning")
    if isinstance(raw, str) and raw:
        try:
            env = json.loads(raw)
            if isinstance(env, dict):
                inner = env.get("decision")
                if isinstance(inner, dict):
                    tk = inner.get("ticker")
                    if isinstance(tk, str) and tk:
                        tk_u = tk.upper().strip()
                        if tk_u and tk_u not in _NULL_TICKERS:
                            return tk_u
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return None


def _normalize_snippet(s: str) -> str:
    """Collapse whitespace + lowercase a snippet for dedup-key purposes."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _clip(s: str, n: int = 120) -> str:
    """Tight clip a snippet for display (suffix ellipsis past n)."""
    s = re.sub(r"\s+", " ", s.strip())
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def extract_intents_from_text(text: str) -> list[dict[str, Any]]:
    """Run every intent pattern over a reasoning text and return the matches.

    Pure / never raises. Returns list of ``{kind, text, span}`` rows; the
    caller annotates with decision_id / ticker / age. ``text`` is the
    verbatim matched snippet, clipped to 120 chars.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(text, str) or not text:
        return out
    for kind, pat in _INTENT_PATTERNS:
        for m in pat.finditer(text):
            snippet = _clip(m.group(0))
            out.append({
                "kind": kind,
                "text": snippet,
                "span": (m.start(), m.end()),
            })
    return out


def build_decision_conditionals(
    decisions: Sequence[Any] | None,
    *,
    now: datetime | None = None,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    stale_hours: float = DEFAULT_STALE_HOURS,
    max_intents: int = DEFAULT_MAX_INTENTS,
) -> dict[str, Any]:
    """Surface STANDING conditional intents from recent decisions.

    Inputs:
      ``decisions`` — decision dicts ``{id, timestamp, action_taken,
        reasoning, ...}``. NO_DECISION / BLOCKED rows are scanned for
        intents too (a HOLD that says "wait for cash session" is the
        bot's clearest standing intent), but ``parse_failed:`` envelopes
        still get the raw-string fallback so nothing silently drops.
      ``window_hours`` — only consider decisions within this many hours
        of ``now`` (default 24h — a trading day's worth of standing
        intent).
      ``stale_hours`` — an intent is ``stale`` if its source decision is
        older than this. Majority-stale → ``STALE_INTENTS`` verdict.
      ``max_intents`` — return at most this many intents, newest first.

    Dedup: within (ticker, kind, normalized first 60 chars of snippet), the
    NEWEST occurrence wins. This collapses a "wait for cash session"
    repeated across five consecutive HOLDs into one standing intent on
    the most recent decision.

    Output (always a dict, never raises):
      ``state``: ``NO_DATA`` | ``OK``
      ``verdict``: ``NO_DATA`` | ``NO_INTENTS`` | ``STANDING_INTENTS`` |
        ``STALE_INTENTS``
      ``headline``: short verbatim string
      ``n_decisions_scanned``: int
      ``n_intents_raw``: pre-dedup count
      ``n_intents``: post-dedup, post-cap count
      ``n_stale``: how many of returned intents are stale
      ``intents``: list of ``{decision_id, decision_ts, ticker, kind,
        text, age_hours, stale, action_taken}`` rows, newest first.
      ``by_kind``: ``{kind: count}`` rollup of returned intents
      ``window_hours``, ``stale_hours``, ``as_of``: echo of inputs
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(0.0, window_hours))
    stale_cutoff_seconds = max(0.0, stale_hours) * 3600.0

    n_scanned = 0
    raw_intents: list[dict[str, Any]] = []

    for d in (decisions or []):
        if not isinstance(d, dict):
            continue
        ts_dt = _parse_iso(d.get("timestamp"))
        if ts_dt is None:
            continue
        if ts_dt < cutoff:
            continue
        n_scanned += 1
        text = _extract_reasoning_text(d)
        if not text:
            continue
        ticker = _extract_ticker(d)
        age_s = max(0.0, (now - ts_dt).total_seconds())
        matches = extract_intents_from_text(text)
        for m in matches:
            raw_intents.append({
                "decision_id": d.get("id"),
                "decision_ts": ts_dt.isoformat(),
                "ts_dt": ts_dt,           # internal, for sort + dedup
                "ticker": ticker,
                "kind": m["kind"],
                "text": m["text"],
                "age_hours": round(age_s / 3600.0, 2),
                "stale": age_s >= stale_cutoff_seconds,
                "action_taken": d.get("action_taken"),
            })

    # Dedup by (ticker, kind, normalized first 60 chars). Newest wins.
    deduped_by_key: dict[tuple, dict[str, Any]] = {}
    for it in raw_intents:
        key = (
            it["ticker"],
            it["kind"],
            _normalize_snippet(it["text"])[:60],
        )
        prev = deduped_by_key.get(key)
        if prev is None or it["ts_dt"] > prev["ts_dt"]:
            deduped_by_key[key] = it

    deduped = sorted(
        deduped_by_key.values(),
        key=lambda r: r["ts_dt"],
        reverse=True,
    )

    n_intents_raw = len(raw_intents)
    capped = deduped[: max(0, int(max_intents))]
    # Strip the internal sort key before returning
    out_intents: list[dict[str, Any]] = []
    for it in capped:
        it2 = dict(it)
        it2.pop("ts_dt", None)
        out_intents.append(it2)

    n_intents = len(out_intents)
    n_stale = sum(1 for it in out_intents if it["stale"])
    by_kind: dict[str, int] = {}
    for it in out_intents:
        by_kind[it["kind"]] = by_kind.get(it["kind"], 0) + 1

    # Verdict ladder
    if n_scanned == 0:
        state = "NO_DATA"
        verdict = "NO_DATA"
        headline = f"no decisions in last {window_hours:g}h"
    elif n_intents == 0:
        state = "OK"
        verdict = "NO_INTENTS"
        headline = (
            f"no conditional intents in last {window_hours:g}h "
            f"across {n_scanned} decision(s)"
        )
    else:
        state = "OK"
        # n_intents > 0 here
        if n_stale * 2 > n_intents:  # strict majority stale
            verdict = "STALE_INTENTS"
            headline = (
                f"{n_stale}/{n_intents} standing intent(s) stale "
                f"(≥{stale_hours:g}h old without follow-up)"
            )
        else:
            verdict = "STANDING_INTENTS"
            n_tickers = len({it["ticker"] for it in out_intents if it["ticker"]})
            kind_summary = ", ".join(
                f"{c} {k}" for k, c in sorted(by_kind.items(), key=lambda kv: -kv[1])[:3]
            )
            headline = (
                f"{n_intents} standing intent(s) "
                f"across {n_tickers} ticker(s): {kind_summary}"
            )

    return {
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "n_decisions_scanned": n_scanned,
        "n_intents_raw": n_intents_raw,
        "n_intents": n_intents,
        "n_stale": n_stale,
        "intents": out_intents,
        "by_kind": by_kind,
        "window_hours": float(window_hours),
        "stale_hours": float(stale_hours),
        "as_of": now.isoformat(),
    }


def is_intents_stale(report: dict | None) -> bool:
    """Mirror the ``is_failed_runs_hidden`` / ``is_pickle_smoke_failed``
    convention: a single-bool view of "operator should pay attention".

    Returns True ONLY on the ``STALE_INTENTS`` verdict — STANDING_INTENTS
    is normal operation (bot is forward-thinking), NO_INTENTS / NO_DATA
    are silent.
    """
    if not isinstance(report, dict):
        return False
    return report.get("verdict") == "STALE_INTENTS"


__all__ = [
    "DEFAULT_WINDOW_HOURS",
    "DEFAULT_STALE_HOURS",
    "DEFAULT_MAX_INTENTS",
    "build_decision_conditionals",
    "extract_intents_from_text",
    "is_intents_stale",
]
