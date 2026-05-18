"""Per-position attention freshness — has Opus actually examined this holding
lately, or has it gone hours without a real look?

`analytics/decision_health.py` reports the aggregate NO_DECISION rate.
`analytics/decision_drought.py` measures portfolio-wide drift during paralysis
windows. `analytics/thesis_drift.py` re-tests an entry thesis against current
state. `analytics/hold_discipline.py` measures hold time vs the desk's
historical losing-cut time. None answer the per-ticker question:

  **"Which of my open positions has the model actually examined recently?"**

When a NO_DECISION storm hits (host saturation, the documented #1 pathology),
the live trader silently defaults to holding every open position — but those
positions are no longer being *evaluated*. After hours of NO_DECISIONs, an
open lot can sit unmonitored for a full session while the operator wrongly
assumes Opus is still watching it. This module surfaces that gap.

`build_position_attention` is pure: feed it `store.open_positions()` and
`store.recent_decisions(limit)` (newest-first) and it returns a JSON-ready
dict. No DB or network access. `now` is injectable for tests.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Per-position freshness thresholds (hours since the most recent decision row
# that named this ticker AND wasn't a NO_DECISION sentinel).
_FRESH_H = 2.0
_MONITORED_H = 6.0
_STALE_H = 24.0


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_action_ticker(action_taken: str | None) -> tuple[str, str | None]:
    """Pull (verb, ticker) out of a `decisions.action_taken` string. Mirrors
    `dashboard._parse_action_ticker` — kept local so the analytics module has
    no dashboard dependency (the round_trips/decision_forensics pattern)."""
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


def _classify(hours_since: float | None) -> str:
    if hours_since is None:
        return "NEGLECTED"
    if hours_since <= _FRESH_H:
        return "FRESH"
    if hours_since <= _MONITORED_H:
        return "MONITORED"
    if hours_since <= _STALE_H:
        return "STALE"
    return "NEGLECTED"


def build_position_attention(
    open_positions: list[dict],
    decisions: list[dict],
    now: datetime | None = None,
) -> dict:
    """Per-open-position last-real-look summary.

    Inputs:
      open_positions — store.open_positions() rows (any type).
      decisions      — store.recent_decisions(limit) rows, newest-first.
      now            — defaults to wall clock UTC; injectable for tests.

    Output schema:
      {
        "as_of": iso8601,
        "n_positions": int,
        "positions": [
          {
            "ticker", "type", "qty",
            "days_held",                          float | None
            "last_decision_ts",                   iso | None
            "hours_since_last_decision",          float | None
            "last_decision_verb",                 str | None
            "n_real_decisions_24h",               int
            "verdict",                            FRESH / MONITORED / STALE / NEGLECTED
          }, ...
        ],
        "summary": {
          "fresh": int, "monitored": int, "stale": int, "neglected": int,
        },
        "verdict": OK / STALE_BOOK / NEGLECTED_BOOK / INSUFFICIENT_DATA,
        "note": str,
      }
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Index real (non-NO_DECISION, non-BLOCKED) decisions by ticker, newest-first.
    # `decisions` is already newest-first; we keep that order so the first hit
    # per ticker IS the most recent.
    last_by_tk: dict[str, dict] = {}
    by_tk_count_24h: dict[str, int] = {}
    cutoff_24h = now.timestamp() - 24 * 3600

    for d in decisions:
        verb, tk = _parse_action_ticker(d.get("action_taken") or "")
        if not tk:
            continue
        # A real decision = one tied to a specific ticker. Implicit since
        # _parse_action_ticker returns None ticker for NO_DECISION / BLOCKED
        # and the CASH/NONE pseudo-tickers — those are filtered above.
        ts = _parse_ts(d.get("timestamp"))
        if ts is None:
            continue
        if ts.timestamp() >= cutoff_24h:
            by_tk_count_24h[tk] = by_tk_count_24h.get(tk, 0) + 1
        if tk not in last_by_tk:
            last_by_tk[tk] = {"ts": ts, "verb": verb}

    rows: list[dict] = []
    summary = {"fresh": 0, "monitored": 0, "stale": 0, "neglected": 0}

    for p in open_positions:
        tk = (p.get("ticker") or "").upper()
        if not tk:
            continue
        opened_at = _parse_ts(p.get("opened_at"))
        days_held = None
        if opened_at is not None:
            days_held = round((now - opened_at).total_seconds() / 86400.0, 2)

        last = last_by_tk.get(tk)
        last_ts_iso = None
        hours_since = None
        last_verb = None
        if last is not None:
            last_ts_iso = last["ts"].isoformat()
            hours_since = round(
                (now - last["ts"]).total_seconds() / 3600.0, 2)
            last_verb = last["verb"]

        verdict = _classify(hours_since)
        summary[verdict.lower()] += 1

        rows.append({
            "ticker": tk,
            "type": p.get("type"),
            "qty": p.get("qty"),
            "days_held": days_held,
            "last_decision_ts": last_ts_iso,
            "hours_since_last_decision": hours_since,
            "last_decision_verb": last_verb,
            "n_real_decisions_24h": by_tk_count_24h.get(tk, 0),
            "verdict": verdict,
        })

    # Sort worst-first (NEGLECTED on top, then by hours_since descending so the
    # oldest neglect is most visible). Positions with no last-look go to the top
    # since None compares poorly across types.
    def _sort_key(r: dict):
        order = {"NEGLECTED": 0, "STALE": 1, "MONITORED": 2, "FRESH": 3}
        # Within a verdict bucket, biggest hours_since first; None = infinity.
        h = r["hours_since_last_decision"]
        return (order.get(r["verdict"], 9), -(h if h is not None else 1e9))

    rows.sort(key=_sort_key)

    if not rows:
        verdict = "INSUFFICIENT_DATA"
        note = "No open positions to evaluate."
    elif summary["neglected"] > 0:
        verdict = "NEGLECTED_BOOK"
        note = (f"{summary['neglected']} of {len(rows)} held position(s) "
                f"have had no Opus look in >24h — model attention has lapsed "
                f"on them (likely passive HOLD via NO_DECISION storms).")
    elif summary["stale"] > 0:
        verdict = "STALE_BOOK"
        note = (f"{summary['stale']} of {len(rows)} held position(s) last "
                f"seen by Opus >6h ago — monitor for drift.")
    else:
        verdict = "OK"
        note = (f"All {len(rows)} held position(s) examined by Opus in the "
                f"last {_MONITORED_H:.0f}h.")

    return {
        "as_of": now.isoformat(),
        "n_positions": len(rows),
        "positions": rows,
        "summary": summary,
        "verdict": verdict,
        "note": note,
        "thresholds_hours": {
            "fresh_le": _FRESH_H,
            "monitored_le": _MONITORED_H,
            "stale_le": _STALE_H,
        },
    }
