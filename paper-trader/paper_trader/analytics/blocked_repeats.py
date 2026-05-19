"""Repeated-BLOCKED action audit — which (verb, ticker) is Opus trying but
the paper engine keeps refusing?

``decisions.action_taken`` is free text in the form ``"BUY NVDA → BLOCKED"``
(AGENTS.md invariant #11). When Opus keeps trying to act but the engine
keeps blocking (insufficient cash, no price, ambiguous option close,
non-numeric strike), the trader needs to know:

  * **Which (verb, ticker) is repeatedly blocked**, so they can either fund
    the trade (top up cash), re-prompt Opus (resolve the ambiguity), or
    accept that the ticker is data-blacked-out (no live price).
  * **What the blocking cause is**, so they pick the right remediation —
    a CASH block is a funding action, a DATA block is a feed problem, a
    SIZING block is a prompt-tuning concern.

Every other operator-facing surface (``/api/decision-health``,
``/api/no-decision-reasons``, the Discord ``_no_decision_reasons_line``) is
about *NO_DECISION* (Claude didn't reply). Repeated BLOCKED is the
orthogonal failure mode: Claude DID reply, the engine rejected the trade,
and nothing on the dashboard names it. This is the missing surface.

``build_blocked_repeats`` is pure (no DB, no network, never raises on
garbage input). The endpoint owns I/O — the documented
``round_trips``/``no_decision_reasons`` builder/endpoint split.
Observational only — never gates Opus, never injected into the decision
prompt, no caps (AGENTS.md #2 / #12 — the ``no_decision_reasons`` /
``capital_paralysis`` precedent).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

# Minimum count of distinct BLOCKED rows on the same (verb, ticker) to
# qualify as a "repeat" worth surfacing. A single BLOCKED is not a pattern.
MIN_REPEAT = 2

# Free-text BLOCKED reasons → operator-actionable bucket. The strategy
# emit-site for each phrase is pinned in tests (``strategy._execute``)
# so this mapping is the contract.
_CAUSE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("insufficient cash", "CASH"),
    ("no price for", "DATA"),                  # market.get_price() returned None
    ("no option price for", "DATA"),           # market.get_option_price() returned None
    ("no open", "SIZING"),                     # "no open call/put position in X to close"
    ("no matching open", "SIZING"),            # SELL_CALL/PUT path
    ("exceeds held", "SIZING"),                # SELL qty > held
    ("ambiguous", "SPECIFICATION"),            # SELL_CALL/PUT without strike+expiry
    ("missing strike/expiry", "SPECIFICATION"),
    ("strike not numeric", "SPECIFICATION"),
    ("qty not numeric", "SPECIFICATION"),
    ("qty must be > 0", "SPECIFICATION"),
    ("unknown action", "SPECIFICATION"),
)


def _classify_cause(detail: str) -> str:
    """Map a free-text BLOCKED detail to an operator-actionable bucket.

    Returns ``"OTHER"`` when no pattern matches — a future strategy
    change that introduces a new BLOCKED phrase still surfaces (count +
    raw detail) but is bucketed as OTHER until someone adds the mapping.
    """
    s = (detail or "").lower()
    for phrase, bucket in _CAUSE_PATTERNS:
        if phrase in s:
            return bucket
    return "OTHER"


def _parse_verb_ticker(action_taken: str) -> tuple[str | None, str | None]:
    """Local copy of ``dashboard._parse_action_ticker`` — duplicated to keep
    the builder importable without pulling in the heavy ``dashboard``
    module (the analytics modules are loaded under tests with NO Flask
    bound; importing dashboard would 8090-bind in setup).

    ``"BUY NVDA → BLOCKED"`` → ``("BUY", "NVDA")``.
    ``"NO_DECISION"`` / sentinel / malformed → ``(None, None)``.
    """
    if not action_taken or action_taken in ("NO_DECISION", "BLOCKED"):
        return None, None
    head = action_taken.split("→")[0].strip()
    parts = head.split()
    if not parts:
        return None, None
    verb = parts[0].upper()
    ticker = parts[1].upper() if len(parts) >= 2 else None
    if ticker in ("CASH", "NONE", ""):
        ticker = None
    return verb, ticker


def _extract_detail(reasoning: str | None) -> str:
    """Pull the human-readable ``detail`` field out of the JSON reasoning
    blob ``strategy.decide`` writes for non-NO_DECISION rows:

      ``json.dumps({"decision": {...}, "auto_exits": [...], "detail": "...",
                    "fallback_used": False})``

    Returns ``""`` on missing / parse failure (degrade-safe; the row still
    counts in the verb/ticker aggregate, just shows no cause).
    """
    if not reasoning:
        return ""
    try:
        blob = json.loads(reasoning)
    except (ValueError, TypeError):
        return ""
    if not isinstance(blob, dict):
        return ""
    d = blob.get("detail")
    return str(d) if isinstance(d, str) else ""


def _ts_age_hours(ts: str | None, now: datetime) -> float | None:
    """Hours between ``ts`` (ISO-8601 UTC) and ``now``. None on parse fail
    or empty — caller renders age token only when usable."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    secs = (now - dt).total_seconds()
    return secs / 3600.0 if secs >= 0 else 0.0


def build_blocked_repeats(decisions: list[dict],
                          now: datetime | None = None,
                          min_repeat: int = MIN_REPEAT) -> dict:
    """Aggregate repeated BLOCKED decisions by (verb, ticker).

    ``decisions`` is a ``store.recent_decisions()`` newest-first list.
    Returns a dict with at minimum:

      * ``state`` — ``"OK"`` / ``"NO_DATA"`` / ``"NO_REPEATS"``
      * ``verdict`` — ``"CLEAN"`` (no repeats), ``"REPEATING"`` (>=1 repeat)
      * ``headline`` — operator-readable summary
      * ``blocked_repeats`` — list of dicts:
          ``{verb, ticker, count, dominant_cause, latest_detail, latest_ts,
             latest_age_hours, by_cause}``

    Sorted by count DESC then latest_ts DESC. ``by_cause`` is the per-
    bucket count (so a CASH+DATA mix on the same ticker is visible).

    Pure: no DB, no network. Never raises on garbage input (the
    ``round_trips`` / ``news_source_mix`` precedent). ``now`` is
    injectable for deterministic tests.
    """
    now = now or datetime.now(timezone.utc)
    if not decisions:
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": "NO_DATA",
            "verdict": "CLEAN",
            "headline": "No decisions on record.",
            "n_blocked_total": 0,
            "n_distinct_repeats": 0,
            "blocked_repeats": [],
        }

    # Aggregate: {(verb, ticker): {count, latest_ts, latest_detail, by_cause}}.
    agg: dict[tuple[str, str], dict] = {}
    n_blocked_total = 0
    for d in decisions:
        if not isinstance(d, dict):
            continue
        action = d.get("action_taken") or ""
        # Block rows have "→ BLOCKED" suffix. The literal "BLOCKED" alone
        # would be a sentinel-style row (no verb/ticker); count it as a
        # blocked row but it can't aggregate by (verb, ticker) so it
        # silently falls through the parser to (None, None).
        if "BLOCKED" not in action:
            continue
        n_blocked_total += 1
        verb, ticker = _parse_verb_ticker(action)
        if verb is None or ticker is None:
            continue
        detail = _extract_detail(d.get("reasoning"))
        cause = _classify_cause(detail)
        key = (verb, ticker)
        row = agg.get(key)
        if row is None:
            row = {
                "verb": verb,
                "ticker": ticker,
                "count": 0,
                "latest_ts": "",
                "latest_detail": "",
                "by_cause": {},
            }
            agg[key] = row
        row["count"] += 1
        row["by_cause"][cause] = row["by_cause"].get(cause, 0) + 1
        # Decisions are newest-first; the FIRST row we see for this key is
        # the latest. Subsequent rows are older and must not overwrite
        # ``latest_ts`` / ``latest_detail``.
        if not row["latest_ts"]:
            row["latest_ts"] = d.get("timestamp") or ""
            row["latest_detail"] = detail

    # Filter to repeats.
    repeats = [r for r in agg.values() if r["count"] >= min_repeat]

    # Enrich + sort.
    for r in repeats:
        bc = r["by_cause"]
        dom_cause = max(bc.items(), key=lambda kv: kv[1])[0] if bc else "OTHER"
        r["dominant_cause"] = dom_cause
        r["latest_age_hours"] = _ts_age_hours(r["latest_ts"], now)
    # Sort key: count DESC (negate), then latest_ts DESC (newer = smaller
    # age_hours, ascending sort = newest first). A row with no latest_ts
    # falls to the end of its count group via a large sentinel age.
    def _sort_key(r):
        age = _ts_age_hours(r["latest_ts"], now)
        return (-r["count"], age if age is not None else 1e9)
    repeats.sort(key=_sort_key)

    if not repeats:
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": "NO_REPEATS",
            "verdict": "CLEAN",
            "headline": (f"No repeated BLOCKED actions "
                         f"({n_blocked_total} single block"
                         f"{'' if n_blocked_total == 1 else 's'} seen)."),
            "n_blocked_total": n_blocked_total,
            "n_distinct_repeats": 0,
            "blocked_repeats": [],
        }

    # Headline names the worst offender + how often it blocked.
    worst = repeats[0]
    headline = (f"{worst['verb']} {worst['ticker']} blocked "
                f"{worst['count']}x ({worst['dominant_cause']}); "
                f"{len(repeats)} distinct repeat"
                f"{'' if len(repeats) == 1 else 's'}.")
    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "OK",
        "verdict": "REPEATING",
        "headline": headline,
        "n_blocked_total": n_blocked_total,
        "n_distinct_repeats": len(repeats),
        "blocked_repeats": repeats,
    }
