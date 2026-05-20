"""Per-position current rationale — what was Opus's last reason for each open
position?

``position_attention`` answers *did* Opus look at this position lately
(freshness, neglect detection). ``thesis_drift`` re-tests an entry thesis
against current state. Neither surfaces the **concrete current rationale** a
trader most wants when reviewing the live book: *"why am I still holding
NVDA? what did Opus actually say last time it considered this name?"*

The data is already in ``decisions.reasoning`` (a JSON envelope written by
``strategy.decide()``: ``{"decision": {"reasoning": "...", "confidence": x},
"detail": "...", ...}``). Every operator who asks the question today has to
manually scroll the dashboard's decision feed, find the most recent row for
that ticker, and read it. This module is the per-position view that puts
the answer one read away from any caller that already has
``store.open_positions()`` + ``store.recent_decisions()``.

Each row returns the most recent **real** (non-NO_DECISION, non-BLOCKED)
decision row whose ``(verb, ticker)`` names that position. Verb selection
intentionally includes HOLD: a 9-cycle "HOLD NVDA" streak is exactly when
the trader most needs to read the current reasoning to decide whether the
thesis is still active or the bot is in a rut.

Pure, degrade-safe (the ``_safe`` contract):

* No DB, no LLM, no network. Caller passes the lists.
* Garbage / unparseable ``reasoning`` JSON degrades to ``reasoning=""`` for
  that row (the rest of the row still ships) — never raises.
* A position with no real decision since its open returns the position row
  with ``last_decision_ts=None`` so the *absence* is visible.

Observational only — never injected into the decision prompt, never gates
Opus, adds no caps (invariants #2/#12 — the ``position_attention``
precedent)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

# Cap a single reasoning excerpt so the JSON response stays compact even
# when Opus emitted a long block (the ``decisions.reasoning`` envelope is
# already capped at ``strategy.RAW_CAPTURE_CHARS=1000`` for the parse-fail
# path, but a successful reasoning string can be longer than that).
_MAX_REASON_CHARS = 600


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _parse_action_ticker(action_taken: str | None) -> tuple[str, str | None]:
    """Mirror of ``position_attention._parse_action_ticker`` so the two
    builders never disagree on the (verb, ticker) split. Kept local — same
    reason ``position_attention`` keeps a local copy: no dashboard import
    on the analytics path."""
    if not action_taken or action_taken in ("NO_DECISION", "BLOCKED"):
        return action_taken or "", None
    head = action_taken.split("→")[0].strip()
    parts = head.split()
    if not parts:
        return "", None
    verb = parts[0].upper()
    ticker = parts[1].upper() if len(parts) >= 2 else None
    if ticker in ("CASH", "NONE", ""):
        ticker = None
    return verb, ticker


def _extract_decision(reasoning_blob: str | None) -> dict:
    """Pull ``confidence``, ``reasoning`` (and ``action``) from the JSON
    envelope ``strategy.decide()`` writes into ``decisions.reasoning``.

    The envelope shape is ``{"decision": {...}, "detail": "...", ...}`` —
    not the inner decision dict directly. The legacy free-text reasoning
    (``parse_failed:``, ``retry_failed:``, ``claude returned no response``)
    is not JSON; degrade to an empty result and let the caller surface a
    bare last-decision-verb row.

    Never raises (the ``_safe`` contract): a malformed envelope returns
    ``{}`` so the rest of the row still ships."""
    if not reasoning_blob or not isinstance(reasoning_blob, str):
        return {}
    s = reasoning_blob.strip()
    if not s.startswith("{"):
        return {}
    try:
        env = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(env, dict):
        return {}
    d = env.get("decision")
    if not isinstance(d, dict):
        return {}
    out: dict = {}
    if isinstance(d.get("confidence"), (int, float)):
        out["confidence"] = float(d["confidence"])
    r = d.get("reasoning")
    if isinstance(r, str) and r.strip():
        out["reasoning"] = r.strip()[:_MAX_REASON_CHARS]
    act = d.get("action")
    if isinstance(act, str) and act.strip():
        out["action_from_reasoning"] = act.strip().upper()
    return out


def build_position_rationale(
    open_positions: list[dict],
    decisions: list[dict],
    now: datetime | None = None,
) -> dict:
    """Per-open-position most-recent rationale summary.

    Inputs:
      open_positions — ``store.open_positions()`` rows.
      decisions      — ``store.recent_decisions(limit)`` rows, newest-first.
      now            — defaults to wall clock UTC; injectable for tests.

    Output:
      {
        "as_of": iso8601,
        "n_positions": int,
        "n_with_rationale": int,
        "positions": [
          {
            "ticker", "type", "qty",
            "days_held":                  float | None
            "last_decision_ts":           iso | None
            "hours_since_last_decision":  float | None
            "last_decision_verb":         str | None  (HOLD / BUY / SELL / ...)
            "last_decision_confidence":   float | None
            "last_decision_reasoning":    str | None  (capped at 600 chars)
          }, ...
        ],
        "verdict": "OK" | "MISSING_RATIONALE" | "INSUFFICIENT_DATA",
        "note":   short operator-facing string.
      }

    Pure; never raises.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Walk decisions newest-first; first match per ticker is the most recent.
    last_by_tk: dict[str, dict] = {}
    for d in decisions:
        verb, tk = _parse_action_ticker(d.get("action_taken") or "")
        if not tk:
            continue
        if tk in last_by_tk:
            continue
        ts = _parse_ts(d.get("timestamp"))
        if ts is None:
            continue
        extra = _extract_decision(d.get("reasoning"))
        last_by_tk[tk] = {
            "ts": ts,
            "verb": verb,
            "confidence": extra.get("confidence"),
            "reasoning": extra.get("reasoning"),
        }

    rows: list[dict] = []
    n_with_rationale = 0
    for p in open_positions:
        tk = (p.get("ticker") or "").upper()
        if not tk:
            continue
        opened_at = _parse_ts(p.get("opened_at"))
        days_held = None
        if opened_at is not None:
            days_held = round((now - opened_at).total_seconds() / 86400.0, 2)

        last = last_by_tk.get(tk)
        if last is not None:
            row = {
                "ticker": tk,
                "type": p.get("type"),
                "qty": p.get("qty"),
                "days_held": days_held,
                "last_decision_ts": last["ts"].isoformat(),
                "hours_since_last_decision": round(
                    (now - last["ts"]).total_seconds() / 3600.0, 2),
                "last_decision_verb": last["verb"],
                "last_decision_confidence": last["confidence"],
                "last_decision_reasoning": last["reasoning"],
            }
            if last["reasoning"]:
                n_with_rationale += 1
        else:
            row = {
                "ticker": tk,
                "type": p.get("type"),
                "qty": p.get("qty"),
                "days_held": days_held,
                "last_decision_ts": None,
                "hours_since_last_decision": None,
                "last_decision_verb": None,
                "last_decision_confidence": None,
                "last_decision_reasoning": None,
            }
        rows.append(row)

    # Sort: positions WITHOUT a rationale to the top (they need attention),
    # then by oldest hours_since_last_decision (largest first — most stale
    # rationale is most operator-actionable).
    def _sort_key(r: dict):
        has_reason = r["last_decision_reasoning"] is not None
        h = r["hours_since_last_decision"]
        # has_reason False → 0 (top); True → 1
        # within bucket, oldest first via negative hours_since
        return (1 if has_reason else 0,
                -(h if h is not None else 1e9))
    rows.sort(key=_sort_key)

    n_pos = len(rows)
    if n_pos == 0:
        verdict = "INSUFFICIENT_DATA"
        note = "No open positions to evaluate."
    elif n_with_rationale < n_pos:
        verdict = "MISSING_RATIONALE"
        note = (f"{n_pos - n_with_rationale} of {n_pos} held position(s) "
                f"have no recent Opus rationale on file — review before "
                f"the next cycle.")
    else:
        verdict = "OK"
        note = (f"All {n_pos} held position(s) have a current Opus "
                f"rationale on file.")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "n_positions": n_pos,
        "n_with_rationale": n_with_rationale,
        "positions": rows,
        "verdict": verdict,
        "note": note,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json as _json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_position_rationale(s.open_positions(),
                                    s.recent_decisions(limit=200))
    print(_json.dumps(rep, indent=2, default=str))
