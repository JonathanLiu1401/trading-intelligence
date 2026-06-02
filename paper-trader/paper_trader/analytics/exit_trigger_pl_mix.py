"""Realized P&L bucketed by exit-trigger type — mechanical vs discretionary.

Every closed round-trip has exactly one closing SELL. That SELL's ``reason``
field tells you *who decided to exit*:

* ``HARD_SL: price X <= threshold Y`` — the auto stop-loss check in
  ``strategy._enforce_hard_exits`` fired. Risk-management code, not Opus.
* ``HARD_TP: price X >= threshold Y`` — the auto take-profit check in the
  same function fired. Risk-management code, not Opus.
* anything else — a discretionary SELL whose reason is Opus' free-text
  thesis (the canonical case, e.g. "Rotating fresh capital into the
  cleanest setup …").

``/api/hard-exit-summary`` (in ``analytics/hard_exit_summary.py``) counts
HARD_SL/HARD_TP events and emits a discipline ratio + last-of-each
snapshot — *event-based*. ``/api/exit-intent-audit`` buckets discretionary
SELLs by Opus' intent label parsed from the prose. Neither rolls up the
**realized P&L** by trigger bucket with the same shape across all three
buckets — there's no place a reader can compare *"how does the desk's
discretionary edge compare to its mechanical SL/TP exits?"* on a like-for-
like ledger.

This module is exactly that rollup. Pure, never raises, never trains, never
writes, never has a path to ``_execute``. Mirrors the
``round_trips_derived`` grouping & walking discipline.

Run as a CLI::

    python3 -m paper_trader.analytics.exit_trigger_pl_mix
"""
from __future__ import annotations

from datetime import datetime
from statistics import median


_BUCKETS = ("HARD_SL", "HARD_TP", "DISCRETIONARY")


def _classify_reason(reason) -> str:
    """Return the bucket label for a SELL's ``reason`` field.

    The mechanical exits in ``strategy._enforce_hard_exits`` emit reasons
    prefixed ``HARD_SL:`` or ``HARD_TP:`` literally — see strategy.py:1361.
    Anything else (None, "", or Opus' thesis prose) is the discretionary
    bucket.
    """
    if not reason:
        return "DISCRETIONARY"
    s = str(reason).strip()
    if s.startswith("HARD_SL"):
        return "HARD_SL"
    if s.startswith("HARD_TP"):
        return "HARD_TP"
    return "DISCRETIONARY"


def _trade_key(t):
    """Group key — stocks have NULL option fields; options carry all three.
    Identical shape to ``round_trips_derived._trade_key`` so a round-trip
    sliced here lines up with the one that endpoint reports."""
    return (
        t.get("ticker") or "",
        (t.get("option_type") or None) if t.get("option_type") else None,
        t.get("expiry") or None,
        t.get("strike") if t.get("strike") is not None else None,
    )


def _hold_days(opened_at, closed_at):
    if not opened_at or not closed_at:
        return None
    try:
        op = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
        cl = datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    secs = (cl - op).total_seconds()
    if secs < 0:
        secs = 0.0
    return round(secs / 86400.0, 4)


def derive_trigger_round_trips(trades):
    """Walk every trade and emit one dict per closed round-trip carrying the
    closing SELL's ``reason`` (so the bucket is recoverable).

    Output rows: ``{ticker, type, opened_at, closed_at, hold_days, cost,
    proceeds, realized_pl, realized_pl_pct, close_reason, bucket}``.

    Sorted newest-closed first. Pure: never raises on missing fields.
    """
    if not trades:
        return []

    groups: dict[tuple, list[dict]] = {}
    for t in trades:
        if not isinstance(t, dict):
            continue
        if not t.get("ticker") or not t.get("action"):
            continue
        groups.setdefault(_trade_key(t), []).append(t)

    out: list[dict] = []
    for key, key_trades in groups.items():
        key_trades.sort(key=lambda r: (
            str(r.get("timestamp") or ""),
            int(r.get("id") or 0),
        ))
        ticker = key[0]
        opt = key[1]
        ptype = opt if opt in ("call", "put") else "stock"

        held = 0.0
        start_idx = 0
        for i, t in enumerate(key_trades):
            act = (t.get("action") or "").upper()
            try:
                q = float(t.get("qty") or 0.0)
            except (TypeError, ValueError):
                q = 0.0
            if act.startswith("BUY"):
                if abs(held) < 1e-6:
                    start_idx = i
                held += q
            elif act.startswith("SELL"):
                held -= q
                if abs(held) < 1e-6:
                    slice_ = key_trades[start_idx:i + 1]
                    rt = _aggregate_slice(slice_, ticker, ptype)
                    if rt is not None:
                        out.append(rt)

    out.sort(key=lambda d: str(d.get("closed_at") or ""), reverse=True)
    return out


def _aggregate_slice(slice_, ticker, ptype):
    """Sum cost / proceeds / realized P&L; opened_at = first BUY, closed_at =
    last SELL; close_reason = last SELL's reason field."""
    if not slice_:
        return None
    cost = 0.0
    proceeds = 0.0
    realized = 0.0
    opened_at = None
    closed_at = None
    close_reason = None
    buy_seen = False
    for t in slice_:
        act = (t.get("action") or "").upper()
        try:
            val = float(t.get("value") or 0.0)
        except (TypeError, ValueError):
            val = 0.0
        ts = t.get("timestamp") or None
        if act.startswith("BUY"):
            cost += val
            realized -= val
            if not buy_seen:
                opened_at = ts
                buy_seen = True
        elif act.startswith("SELL"):
            proceeds += val
            realized += val
            closed_at = ts
            close_reason = t.get("reason")

    if not buy_seen or closed_at is None:
        return None

    realized_pct = (round(realized / cost * 100.0, 2)
                    if cost > 1e-9 else None)
    bucket = _classify_reason(close_reason)
    return {
        "ticker": ticker,
        "type": ptype,
        "opened_at": opened_at,
        "closed_at": closed_at,
        "hold_days": _hold_days(opened_at, closed_at),
        "cost": round(cost, 2),
        "proceeds": round(proceeds, 2),
        "realized_pl": round(realized, 2),
        "realized_pl_pct": realized_pct,
        "close_reason": close_reason,
        "bucket": bucket,
    }


def _bucket_stats(rows):
    """One bucket's roll-up: n, wins, losses, totals, win-rate, hold stats."""
    rows = [r for r in rows if isinstance(r, dict)]
    n = len(rows)
    wins = sum(1 for r in rows if (r.get("realized_pl") or 0) > 0)
    losses = sum(1 for r in rows if (r.get("realized_pl") or 0) < 0)
    decided = wins + losses
    total_pl = round(sum((r.get("realized_pl") or 0.0) for r in rows), 2)
    total_cost = round(sum((r.get("cost") or 0.0) for r in rows), 2)
    win_rate = round(100.0 * wins / decided, 2) if decided else None
    avg_pl_pct = (round(total_pl / total_cost * 100.0, 2)
                  if total_cost > 1e-9 else None)
    holds = [r["hold_days"] for r in rows if r.get("hold_days") is not None]
    avg_hold = round(sum(holds) / len(holds), 4) if holds else None
    med_hold = round(median(holds), 4) if holds else None
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "flat": n - wins - losses,
        "win_rate_pct": win_rate,
        "total_pl_usd": total_pl,
        "total_cost_usd": total_cost,
        "avg_pl_pct": avg_pl_pct,
        "avg_hold_days": avg_hold,
        "median_hold_days": med_hold,
    }


def _headline(buckets, verdict):
    """One-line summary contrasting the discretionary path with the
    mechanical exits — the question this endpoint exists to answer."""
    disc = buckets["DISCRETIONARY"]
    sl = buckets["HARD_SL"]
    tp = buckets["HARD_TP"]
    parts = []
    if disc["n"]:
        parts.append(
            f"Discretionary {disc['n']} trips, ${disc['total_pl_usd']:+.2f} "
            f"(win {disc['win_rate_pct']}%)"
            if disc.get("win_rate_pct") is not None
            else
            f"Discretionary {disc['n']} trips, ${disc['total_pl_usd']:+.2f}"
        )
    if sl["n"]:
        parts.append(
            f"HARD_SL {sl['n']} trips, ${sl['total_pl_usd']:+.2f}"
        )
    if tp["n"]:
        parts.append(
            f"HARD_TP {tp['n']} trips, ${tp['total_pl_usd']:+.2f}"
        )
    if not parts:
        return "No closed round-trips on record yet."
    return verdict + " — " + " · ".join(parts) + "."


def _verdict(buckets, min_for_verdict=4):
    """Compare discretionary vs mechanical edge. Withhold until enough data.

    * ``MECHANICAL_DOMINANT`` — HARD_TP+HARD_SL net > 0 AND > discretionary
    * ``DISCRETIONARY_DOMINANT`` — discretionary net > 0 AND > mechanical
    * ``BOTH_NEGATIVE`` — both buckets in the red
    * ``BOTH_POSITIVE`` — both green, within a stone's throw
    * ``EMERGING`` — total trips < ``min_for_verdict``
    """
    total = sum(b["n"] for b in buckets.values())
    if total < min_for_verdict:
        return "EMERGING"
    disc_pl = buckets["DISCRETIONARY"]["total_pl_usd"]
    mech_pl = (buckets["HARD_SL"]["total_pl_usd"]
               + buckets["HARD_TP"]["total_pl_usd"])
    if disc_pl > 0 and mech_pl > 0:
        return "BOTH_POSITIVE"
    if disc_pl < 0 and mech_pl < 0:
        return "BOTH_NEGATIVE"
    if mech_pl > disc_pl:
        return "MECHANICAL_DOMINANT"
    return "DISCRETIONARY_DOMINANT"


def build(trades, min_for_verdict=4):
    """Top-level: take raw trade rows, return the bucket rollup envelope.

    Returns
    -------
    dict
        ``{verdict, headline, buckets, n_round_trips, min_for_verdict}``.
        ``buckets`` is a dict of ``{HARD_SL, HARD_TP, DISCRETIONARY}`` →
        the ``_bucket_stats`` dict. Verdict ladder defined in ``_verdict``.
        Never raises — empty trades returns the ``EMERGING`` envelope with
        zero-stat buckets.
    """
    rows = derive_trigger_round_trips(trades)
    by_bucket = {b: [] for b in _BUCKETS}
    for r in rows:
        by_bucket.setdefault(r.get("bucket", "DISCRETIONARY"), []).append(r)
    buckets = {b: _bucket_stats(by_bucket.get(b, [])) for b in _BUCKETS}
    verdict = _verdict(buckets, min_for_verdict=min_for_verdict)
    return {
        "verdict": verdict,
        "headline": _headline(buckets, verdict),
        "buckets": buckets,
        "n_round_trips": len(rows),
        "min_for_verdict": min_for_verdict,
    }


if __name__ == "__main__":
    import json
    from ..store import get_store
    store = get_store()
    trades = store.recent_trades(limit=5000)
    out = build(trades)
    print(json.dumps(out, indent=2, default=str))
