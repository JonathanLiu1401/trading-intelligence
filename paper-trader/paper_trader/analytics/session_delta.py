"""Session delta — "what materially changed since you last looked".

Every panel across ``:8090`` and the ``:8888`` landing is a **current-state
snapshot**. ``/api/daily-recap`` is a calendar-"today" aggregate;
``/api/missed-signals`` is a per-signal news counterfactual;
``/api/command-center`` is a one-shot current-state aggregate;
``/api/decision-drought`` grades idle-cost vs SPY abstractly. None answers the
first question an operator has when they reopen the dashboard after being
away: **what materially happened while I was gone?** — which today means
scanning ~19 panels.

``build_session_delta`` is a ranked, deduplicated material-event timeline over
a parameterised look-back window, plus a one-line headline. It reads **only
``paper_trader.db`` event tables** (trades / decisions / equity_curve — full
history), so it never depends on ``articles.db`` news depth and its output
matures gracefully on a fresh book.

Distinct from its neighbours:

* ``/api/daily-recap`` — calendar-"today" aggregate (resets at UTC midnight).
* ``/api/decision-drought`` — idle-cost vs SPY, segmented into droughts.
* ``build_session_delta`` — an **arbitrary "since T" change feed**: the
  specific fills, closes, equity move (+ SPY-relative), intra-window
  drawdown, and idle-cycle fact that occurred in the window, ranked by
  materiality.

Single source of truth (AGENTS.md invariant #10): realised P&L on a
``POSITION_CLOSED`` event is consumed **verbatim** from
``round_trips.build_round_trips`` — never recomputed here. The intra-window
drawdown is a deliberately *window-scoped* metric, explicitly NOT the
book-wide ``/api/drawdown`` anatomy (the same intentional-divergence pattern
``churn`` documents for its median-hold metric).

Contracts:

* **Never raises.** Each event-derivation block is independently guarded; a
  fault drops that one event class, never the whole report ("no delta this
  cycle, never an exception" — the ``churn``/``trader_scorecard`` contract).
* **Advisory only.** Dashboard/chat surface; never injected into the live
  decision prompt, never gates Opus, adds no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .round_trips import build_round_trips

# An absolute equity move of this % over the window is escalated HIGH (a
# routine sub-1% mark-to-market drift stays MED so it does not crowd out a
# fill in the ranked feed).
MOVE_HIGH_PCT = 1.0

# A peak→trough that occurred *inside the window* of at least this % is
# surfaced as its own HIGH event — the operator wants "it dipped 3% while you
# were out" called out even if it recovered by the time you looked.
DD_PCT = 2.0

# A window with at least this many decision cycles and zero fills is reported
# as a factual INACTION event (no verdict — that would duplicate
# decision_reliability / decision_drought).
INACTION_MIN_CYCLES = 3

# Bound the returned feed; ranked, so the cut keeps the most material rows.
MAX_EVENTS = 40

_SEV_RANK = {"HIGH": 0, "MED": 1, "LOW": 2}


def _parse_ts(ts: str | None) -> datetime | None:
    """ISO-8601 → aware UTC datetime. Naive input is assumed UTC (store writes
    ``datetime.now(timezone.utc).isoformat()`` so live rows are always aware;
    this only hardens hand-built/legacy rows)."""
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def _aware(d: datetime | None, default: datetime) -> datetime:
    if d is None:
        return default
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)


def _window_label(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 3600:
        return f"{max(1, s // 60)}m"
    if s < 86400:
        h, m = divmod(s // 60, 60)
        return f"{h}h" if m == 0 else f"{h}h{m}m"
    d, rem = divmod(s, 86400)
    h = rem // 3600
    return f"{d}d" if h == 0 else f"{d}d{h}h"


def build_session_delta(
    trades: list[dict],
    decisions: list[dict],
    equity: list[dict],
    since_ts: datetime,
    now: datetime | None = None,
) -> dict:
    """Material-event timeline over ``(since_ts, now]``. Pure, never raises.

    ``trades`` must be oldest→newest (the ``/api/analytics`` / ``churn``
    convention, ``list(reversed(store.recent_trades(N)))`` — required by
    ``build_round_trips``, which reads in sequence and does not sort).
    ``decisions`` and ``equity`` are sorted defensively inside, so either
    store-native order is accepted.

    Window boundary is **strict-exclusive lower** (``ts > since_ts``) applied
    identically to every event class, so a row exactly at ``since_ts`` is
    "before you looked" and is excluded; one second later is included.
    """
    now = _aware(now, datetime.now(timezone.utc))
    since_ts = _aware(since_ts, now)
    # A since in the future (or == now) is a degenerate empty window — clamp so
    # window_seconds is never negative and the headline reads honestly.
    if since_ts > now:
        since_ts = now
    window_seconds = (now - since_ts).total_seconds()

    trades = trades or []
    decisions = decisions or []
    equity = equity or []

    events: list[dict] = []
    n_fills = 0
    n_closed = 0
    net_realized_usd = 0.0
    equity_delta_usd: float | None = None
    equity_delta_pct: float | None = None
    spy_delta_pct: float | None = None
    alpha_pp: float | None = None
    max_dd_pct: float | None = None
    anchor_fallback = False
    inaction: dict | None = None

    # ---- TRADE: every executed fill in the window ---------------------
    # The trades table only ever holds executed fills (record_trade is called
    # on fill), so every in-window row is a TRADE event.
    try:
        for t in trades:
            ts = _parse_ts(t.get("timestamp"))
            if ts is None or ts <= since_ts:
                continue
            action = (t.get("action") or "").upper()
            ticker = t.get("ticker") or "?"
            qty = float(t.get("qty") or 0.0)
            price = float(t.get("price") or 0.0)
            notional = float(t.get("value") or 0.0)
            reason = (t.get("reason") or "").strip()
            if len(reason) > 160:
                reason = reason[:160] + "…"
            n_fills += 1
            events.append({
                "kind": "TRADE",
                "ts": t.get("timestamp"),
                "severity": "HIGH",
                "ticker": ticker,
                "action": action,
                "qty": round(qty, 6),
                "price": round(price, 4),
                "notional_usd": round(notional, 2),
                "option_type": t.get("option_type"),
                "reason": reason,
                "summary": (f"{action} {ticker} {qty:g} @ ${price:,.2f} "
                            f"(${notional:,.2f})"),
            })
    except Exception:
        pass  # drop the TRADE class only; never sink the report

    # ---- POSITION_CLOSED: round-trips that closed in the window -------
    # Realised P&L is consumed verbatim from build_round_trips (SSOT #10).
    try:
        for rt in build_round_trips(trades):
            exit_dt = _parse_ts(rt.get("exit_ts"))
            if exit_dt is None or exit_dt <= since_ts:
                continue
            pnl = rt.get("pnl_usd")
            pnl_pct = rt.get("pnl_pct")
            hold = rt.get("hold_days")
            tk = rt.get("ticker") or "?"
            n_closed += 1
            if pnl is not None:
                net_realized_usd += float(pnl)
            pnl_s = "?" if pnl is None else f"{pnl:+,.2f}"
            pct_s = "" if pnl_pct is None else f" ({pnl_pct:+.1f}%)"
            hold_s = "" if hold is None else f" after {hold:.2f}d"
            events.append({
                "kind": "POSITION_CLOSED",
                "ts": rt.get("exit_ts"),
                "severity": "HIGH",
                "ticker": tk,
                "type": rt.get("type"),
                "strike": rt.get("strike"),
                "expiry": rt.get("expiry"),
                "qty": rt.get("qty"),
                "pnl_usd": pnl,
                "pnl_pct": pnl_pct,
                "hold_days": hold,
                "entry_ts": rt.get("entry_ts"),
                "summary": f"Closed {tk}: ${pnl_s}{pct_s}{hold_s}",
            })
    except Exception:
        pass

    # ---- EQUITY_MOVE: anchor (value when you last looked) → latest ----
    try:
        eq = sorted(
            ((_parse_ts(p.get("timestamp")), p) for p in equity
             if _parse_ts(p.get("timestamp")) is not None),
            key=lambda x: x[0],
        )
        if len(eq) >= 2:
            last_dt, last_p = eq[-1]
            # Anchor = most recent point at-or-before since_ts ("the value
            # when you last looked"). If none exists (since older than all
            # history) fall back to the earliest point and flag it — never
            # fabricate a baseline.
            anchor_p = None
            for d, p in eq:
                if d <= since_ts:
                    anchor_p = p
                else:
                    break
            if anchor_p is None:
                anchor_p = eq[0][1]
                anchor_fallback = True
            a_val = float(anchor_p.get("total_value") or 0.0)
            l_val = float(last_p.get("total_value") or 0.0)
            # Only a window that actually advanced past since_ts has a move to
            # report (single stored point, or all points ≤ since → omit, do
            # not emit a Δ=0 event).
            if last_dt > since_ts and anchor_p is not last_p and a_val != 0.0:
                equity_delta_usd = round(l_val - a_val, 2)
                equity_delta_pct = round((l_val - a_val) / a_val * 100.0, 4)
                a_spy = anchor_p.get("sp500_price")
                l_spy = last_p.get("sp500_price")
                spy_s = ""
                if a_spy is not None and l_spy is not None and float(a_spy) != 0.0:
                    spy_delta_pct = round(
                        (float(l_spy) - float(a_spy)) / float(a_spy) * 100.0, 4)
                    alpha_pp = round(equity_delta_pct - spy_delta_pct, 4)
                    spy_s = (f" vs SPY {spy_delta_pct:+.2f}% "
                             f"→ {alpha_pp:+.2f}pp")
                sev = ("HIGH" if abs(equity_delta_pct) >= MOVE_HIGH_PCT
                       else "MED")
                events.append({
                    "kind": "EQUITY_MOVE",
                    "ts": last_p.get("timestamp"),
                    "severity": sev,
                    "delta_usd": equity_delta_usd,
                    "delta_pct": equity_delta_pct,
                    "spy_delta_pct": spy_delta_pct,
                    "alpha_pp": alpha_pp,
                    "anchor_value": round(a_val, 2),
                    "latest_value": round(l_val, 2),
                    "anchor_fallback": anchor_fallback,
                    "summary": (f"Equity ${equity_delta_usd:+,.2f} "
                                f"({equity_delta_pct:+.2f}%){spy_s}"),
                })

            # ---- DRAWDOWN_LOW: deepest peak→trough inside the window ---
            # Window-scoped (anchor + in-window points), explicitly NOT the
            # book-wide /api/drawdown anatomy (intentional divergence).
            seq = [anchor_p] + [p for d, p in eq if d > since_ts]
            peak = None
            worst = 0.0
            worst_peak = worst_trough = None
            worst_peak_ts = worst_trough_ts = None
            cur_peak_ts = None
            for p in seq:
                v = float(p.get("total_value") or 0.0)
                if peak is None or v > peak:
                    peak = v
                    cur_peak_ts = p.get("timestamp")
                if peak and peak > 0:
                    dd = (peak - v) / peak * 100.0
                    if dd > worst:
                        worst = dd
                        worst_peak = peak
                        worst_trough = v
                        worst_peak_ts = cur_peak_ts
                        worst_trough_ts = p.get("timestamp")
            if worst >= DD_PCT and worst_peak and worst_trough is not None:
                max_dd_pct = round(worst, 4)
                events.append({
                    "kind": "DRAWDOWN_LOW",
                    "ts": worst_trough_ts,
                    "severity": "HIGH",
                    "drawdown_pct": max_dd_pct,
                    "peak_value": round(worst_peak, 2),
                    "trough_value": round(worst_trough, 2),
                    "peak_ts": worst_peak_ts,
                    "summary": (f"Intra-window drawdown -{max_dd_pct:.1f}% "
                                f"(${worst_peak:,.2f} → ${worst_trough:,.2f})"),
                })
    except Exception:
        pass

    # ---- INACTION: factual idle-window counts (no verdict) ------------
    try:
        n_dec = n_hold = n_nodec = n_blocked = n_dec_fill = 0
        for dcn in decisions:
            ts = _parse_ts(dcn.get("timestamp"))
            if ts is None or ts <= since_ts:
                continue
            n_dec += 1
            a = (dcn.get("action_taken") or "").upper()
            # action_taken is free text (CLAUDE.md #11): "BUY NVDA → FILLED" /
            # "HOLD NONE → HOLD" / "NO_DECISION" / "BLOCKED …". Match the
            # arrowed "→ FILLED" form, not a bare FILLED substring.
            if "→ FILLED" in a or "-> FILLED" in a:
                n_dec_fill += 1
            elif "NO_DECISION" in a:
                n_nodec += 1
            elif "BLOCKED" in a:
                n_blocked += 1
            else:
                n_hold += 1
        if n_dec >= INACTION_MIN_CYCLES and n_dec_fill == 0:
            inaction = {
                "n_cycles": n_dec, "n_hold": n_hold,
                "n_no_decision": n_nodec, "n_blocked": n_blocked,
            }
            events.append({
                "kind": "INACTION",
                "ts": now.isoformat(timespec="seconds"),
                "severity": "MED",
                **inaction,
                "summary": (f"{n_dec} decision cycles, 0 fills "
                            f"({n_hold} HOLD, {n_nodec} NO_DECISION, "
                            f"{n_blocked} BLOCKED) — desk idle this window"),
            })
    except Exception:
        pass

    # ---- rank: severity (HIGH→MED→LOW) then most-recent first --------
    def _sort_key(e: dict):
        d = _parse_ts(e.get("ts"))
        epoch = d.timestamp() if d is not None else 0.0
        return (_SEV_RANK.get(e.get("severity"), 9), -epoch)

    events.sort(key=_sort_key)
    events = events[:MAX_EVENTS]

    # ---- state + headline -------------------------------------------
    has_any = bool(trades or decisions or equity)
    since_disp = since_ts.strftime("%H:%M")
    wl = _window_label(window_seconds)

    if not has_any:
        state = "NO_DATA"
        headline = "No trader activity recorded yet."
    elif not events:
        state = "QUIET"
        headline = (f"Quiet since {since_disp} UTC ({wl}) — no fills, "
                    f"no material moves.")
    else:
        state = "ACTIVE"
        bits: list[str] = [f"{n_fills} fill(s)" if n_fills
                           else "no fills"]
        if n_closed:
            bits.append(f"closed {n_closed} for ${net_realized_usd:+,.2f}")
        if equity_delta_usd is not None:
            seg = f"equity ${equity_delta_usd:+,.2f} ({equity_delta_pct:+.2f}%)"
            if alpha_pp is not None:
                seg += f" vs SPY {spy_delta_pct:+.2f}% → {alpha_pp:+.2f}pp"
            bits.append(seg)
        if max_dd_pct is not None:
            bits.append(f"intra-window DD -{max_dd_pct:.1f}%")
        if inaction is not None:
            bits.append("desk idle (0 fills over "
                        f"{inaction['n_cycles']} cycles)")
        headline = f"Since {since_disp} UTC ({wl}): " + "; ".join(bits) + "."

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "since": since_ts.isoformat(timespec="seconds"),
        "window_seconds": int(window_seconds),
        "window_label": wl,
        "state": state,
        "headline": headline,
        "n_events": len(events),
        "events": events,
        "n_fills": n_fills,
        "n_closed": n_closed,
        "net_realized_usd": round(net_realized_usd, 2),
        "equity_delta_usd": equity_delta_usd,
        "equity_delta_pct": equity_delta_pct,
        "spy_delta_pct": spy_delta_pct,
        "alpha_pp": alpha_pp,
        "intra_window_drawdown_pct": max_dd_pct,
        "anchor_fallback": anchor_fallback,
        "inaction": inaction,
        "move_high_pct": MOVE_HIGH_PCT,
        "drawdown_pct_threshold": DD_PCT,
    }


if __name__ == "__main__":  # smoke against the live DB
    import json
    from datetime import timedelta

    from paper_trader.store import get_store

    s = get_store()
    _now = datetime.now(timezone.utc)
    rep = build_session_delta(
        list(reversed(s.recent_trades(2000))),
        s.recent_decisions(500),
        s.equity_curve(1000),
        _now - timedelta(hours=6),
        _now,
    )
    print(json.dumps(rep, indent=2, default=str))
