"""Per-ticker re-entry-velocity verdict — how fast does the desk re-buy a
name it just closed?

``round_trips.build_round_trips`` groups raw trades into closed round-trips
on the (ticker, type, strike, expiry) key. ``track_record`` composes the
loser/winner-autopsy narratives so the live prompt sees a name-specific
memory. ``churn`` measures *quantity* churn (size-weighted intraday turnover).
None of them answer:

  **"After closing a position, how quickly did the desk re-enter the same
  ticker — and is the median re-entry gap drifting toward fast-flip churn?"**

That gap is exactly the documented live pathology (CLAUDE.md/AGENTS.md:
observed ``avg_holding_days`` ~0.27 with the NVDA→LITE→NVDA shape and
``KNIFE_CATCH`` repeats). ``hold_discipline`` reports hold *time*, but a
seven-hour hold followed by a one-hour re-buy is a very different signal from
a seven-hour hold followed by a seven-day re-buy. This module surfaces the
re-entry interval distribution explicitly.

``build_reentry_velocity`` is pure — it composes the closed round-trips
returned by ``round_trips.build_round_trips`` and walks each key's exits to
the next same-key entry. The endpoint owns I/O (the documented
``round_trips`` / ``track_record`` builder split). Observational only —
never gates Opus, never injected into the decision prompt, no caps
(AGENTS.md #2 / #12 — the ``self_review`` / ``capital_paralysis`` precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips

# Re-entry-gap classification ladder. Pinned in tests; do not adjust
# without updating both the tests and any chat-enrichment caller that
# uses the verdict string.
_IMMEDIATE_HOURS = 1.0        # <1h — same-session flip
_SAME_DAY_HOURS = 24.0        # 1h .. 24h — same-day re-entry
_QUICK_DAYS = 3.0             # 24h .. 3d — quick re-entry
_NORMAL_DAYS = 14.0           # 3d .. 14d — normal cadence
# >14d — RARE


def _classify_gap(gap_hours: float) -> str:
    if gap_hours < _IMMEDIATE_HOURS:
        return "IMMEDIATE"
    if gap_hours < _SAME_DAY_HOURS:
        return "SAME_DAY"
    if gap_hours < _QUICK_DAYS * 24.0:
        return "QUICK"
    if gap_hours < _NORMAL_DAYS * 24.0:
        return "NORMAL"
    return "RARE"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def build_reentry_velocity(
    trades: list[dict],
    open_positions: list[dict] | None = None,
    now: datetime | None = None,
    recent_limit: int = 10,
) -> dict:
    """Compose per-key re-entry intervals from closed round-trips.

    Inputs:
      trades — oldest→newest list shaped like ``Store.recent_trades()``
        (``timestamp``, ``ticker``, ``action``, ``qty``, ``value``,
        ``strike``, ``expiry``, ``option_type``, ``id``). The function
        sorts internally so a newest-first input is also tolerated.
      open_positions — optional list of currently-open positions
        (``Store.open_positions()`` shape: ``ticker``, ``type``, ``strike``,
        ``expiry``, ``opened_at``). When supplied, an "open after close"
        gap is reported: the latency between the most recent same-key
        close and the still-open entry, so the operator can see the
        live fast-flip case that ``round_trips`` cannot (it only sees
        closed round-trips).
      now — injected clock (defaults to UTC now); only used to time-cap
        the "open after close" gaps if ``open_positions`` is supplied.
      recent_limit — how many newest-first gaps to surface in
        ``recent_gaps`` (default 10).

    Returns a JSON-ready dict:
      ``as_of`` (ISO, seconds), ``n_round_trips``, ``n_gaps``,
      ``median_gap_hours``, ``min_gap_hours``, ``buckets`` (count by
      classification), ``recent_gaps`` (newest-first list with
      ``ticker``, ``type``, ``strike``, ``expiry``, ``closed_at``,
      ``reentered_at``, ``gap_hours``, ``classification``,
      ``open_after_close`` bool), ``per_ticker`` (per-ticker summary:
      ``ticker``, ``n_gaps``, ``median_gap_hours``, ``min_gap_hours``,
      ``last_gap_hours``, ``last_classification``), ``verdict``
      (CHURN_RISK / FAST_FLIP / STABLE / SPARSE).

    Pure — no DB, no network. ``trades`` may be empty; the function
    returns an empty/SPARSE result rather than raising.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Round_trips expects oldest→newest. Sort defensively on timestamp
    # so the caller can pass either direction.
    def _sort_key(t):
        p = _parse_ts(t.get("timestamp"))
        return p or datetime.min.replace(tzinfo=timezone.utc)

    sorted_trades = sorted(trades or [], key=_sort_key)
    rts = build_round_trips(sorted_trades)

    # Bucket round-trips by (ticker, type, strike, expiry) key, then walk
    # each key's exits to the next same-key entry to compute gaps.
    by_key: dict[tuple, list[dict]] = {}
    for rt in rts:
        key = (rt["ticker"], rt["type"], rt.get("strike"), rt.get("expiry"))
        by_key.setdefault(key, []).append(rt)

    gaps: list[dict] = []
    per_ticker_acc: dict[str, list[dict]] = {}

    for key, rt_list in by_key.items():
        # build_round_trips emits in the order each round-trip *closes*,
        # which for any single key equals close-order; sort defensively.
        rt_list.sort(key=lambda r: _parse_ts(r.get("exit_ts")) or datetime.min.replace(tzinfo=timezone.utc))
        for i in range(len(rt_list) - 1):
            closed_at = _parse_ts(rt_list[i].get("exit_ts"))
            reentered_at = _parse_ts(rt_list[i + 1].get("entry_ts"))
            if closed_at is None or reentered_at is None:
                continue
            gap_h = (reentered_at - closed_at).total_seconds() / 3600.0
            if gap_h < 0:
                continue
            row = {
                "ticker": key[0],
                "type": key[1],
                "strike": key[2],
                "expiry": key[3],
                "closed_at": rt_list[i].get("exit_ts"),
                "reentered_at": rt_list[i + 1].get("entry_ts"),
                "gap_hours": round(gap_h, 4),
                "classification": _classify_gap(gap_h),
                "open_after_close": False,
            }
            gaps.append(row)
            per_ticker_acc.setdefault(key[0], []).append(row)

    # "Open after close" gaps — currently-open positions whose key has at
    # least one prior closed round-trip. The latency is close→open_at,
    # which is the live churn signal round_trips alone cannot show.
    if open_positions:
        for op in open_positions:
            tkr = op.get("ticker")
            if not tkr:
                continue
            typ = op.get("type") or "stock"
            key = (tkr, typ, op.get("strike"), op.get("expiry"))
            prior = by_key.get(key)
            if not prior:
                continue
            last_close = _parse_ts(prior[-1].get("exit_ts"))
            opened_at = _parse_ts(op.get("opened_at"))
            if last_close is None or opened_at is None:
                continue
            # The opened_at can pre-date the last close in rare interleaved
            # cases (option roll where strikes differ but ticker matches);
            # the (ticker, type, strike, expiry) key isolates legs, so a
            # negative delta is genuinely "open before close" and we skip.
            if opened_at < last_close:
                continue
            gap_h = (opened_at - last_close).total_seconds() / 3600.0
            row = {
                "ticker": tkr,
                "type": typ,
                "strike": op.get("strike"),
                "expiry": op.get("expiry"),
                "closed_at": prior[-1].get("exit_ts"),
                "reentered_at": op.get("opened_at"),
                "gap_hours": round(gap_h, 4),
                "classification": _classify_gap(gap_h),
                "open_after_close": True,
            }
            gaps.append(row)
            per_ticker_acc.setdefault(tkr, []).append(row)

    # Sort gaps newest re-entry first for the recent_gaps slice.
    gaps.sort(
        key=lambda g: _parse_ts(g["reentered_at"]) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    all_gap_h = [g["gap_hours"] for g in gaps]
    buckets = {"IMMEDIATE": 0, "SAME_DAY": 0, "QUICK": 0, "NORMAL": 0, "RARE": 0}
    for g in gaps:
        buckets[g["classification"]] = buckets.get(g["classification"], 0) + 1

    per_ticker: list[dict] = []
    for tkr, rows in per_ticker_acc.items():
        rows_sorted_newest = sorted(
            rows,
            key=lambda g: _parse_ts(g["reentered_at"]) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        gh = [r["gap_hours"] for r in rows_sorted_newest]
        per_ticker.append({
            "ticker": tkr,
            "n_gaps": len(rows_sorted_newest),
            "median_gap_hours": round(_median(gh), 4) if gh else None,
            "min_gap_hours": round(min(gh), 4) if gh else None,
            "last_gap_hours": rows_sorted_newest[0]["gap_hours"] if rows_sorted_newest else None,
            "last_classification": rows_sorted_newest[0]["classification"] if rows_sorted_newest else None,
        })
    # Surface the fastest-flipping names first so the operator's eye lands
    # on the highest-risk re-entries.
    per_ticker.sort(
        key=lambda p: (p["min_gap_hours"] if p["min_gap_hours"] is not None else 1e9)
    )

    # Verdict ladder. CHURN_RISK and FAST_FLIP are the two operator-action
    # states; everything else is fine. Median is the right summary because
    # one IMMEDIATE outlier in a thirty-gap sample shouldn't tip the verdict.
    median_h = _median(all_gap_h) if all_gap_h else None
    if not all_gap_h:
        verdict = "SPARSE"
    elif median_h is not None and median_h < _SAME_DAY_HOURS:
        verdict = "CHURN_RISK"
    elif buckets["IMMEDIATE"] + buckets["SAME_DAY"] >= max(2, len(all_gap_h) // 2):
        verdict = "FAST_FLIP"
    else:
        verdict = "STABLE"

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "n_round_trips": len(rts),
        "n_gaps": len(gaps),
        "median_gap_hours": round(median_h, 4) if median_h is not None else None,
        "min_gap_hours": round(min(all_gap_h), 4) if all_gap_h else None,
        "buckets": buckets,
        "recent_gaps": gaps[: max(0, int(recent_limit))],
        "per_ticker": per_ticker,
        "verdict": verdict,
    }
