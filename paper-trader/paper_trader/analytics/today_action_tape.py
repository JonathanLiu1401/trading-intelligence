"""Today's action tape — chronological timeline of every trade and every
decision since UTC midnight.

The operator's morning question is "what did my bot do *today* — every move
and every NO-move, in order?". The existing surfaces all fall short of a
literal chronological tape:

  * ``/api/session-delta`` — arbitrary ``since=T`` window, RANKED by
    materiality, capped at 40 events, emits SYNTHESIZED rows (EQUITY_MOVE /
    DRAWDOWN_LOW / INACTION). Designed for "what materially changed", NOT
    for "show me every cycle in order".
  * ``/api/daily-recap`` — calendar-today AGGREGATE. Reports totals
    (n_fills, realized P&L) but not the per-cycle timeline.
  * ``/api/decision-forensics`` / ``/api/decision-clock`` /
    ``/api/decision-daily`` — behavioural analytics on top of the decisions
    table; no chronological tape with trade rows interleaved.

``build_today_action_tape`` returns a calendar-anchored chronological feed
that interleaves **every** ``trades`` row and **every** ``decisions`` row
(including HOLD and NO_DECISION) since ``since`` (default: today's UTC
midnight). Plus light aggregate counts. No materiality ranking, no
synthesized event rows, no cap (limited by store query size, not by
materiality). The intent is a single panel that reproduces what the bot
saw and what it did, one row per cycle.

Pure: no DB, no LLM, no network, no yfinance — only the lists the caller
passes (``store.recent_trades`` + ``store.recent_decisions`` +
``store.INITIAL_CASH`` style inputs). Never raises; per-class faults drop
that one class only (the ``session_delta`` precedent — never sink the
report). Advisory only — dashboard/chat surface; never gates Opus, adds
no caps (AGENTS.md invariants #2/#12).
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any

# Cap so a runaway decisions table can't return a multi-megabyte payload
# even if the caller forgets to limit. A 60s decision cadence × 24h is
# 1440 rows; the dashboard SWR layer is sub-50ms below this size.
_MAX_TAPE_ROWS = 2000


def _parse_ts(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def _aware(d: datetime | None, default: datetime) -> datetime:
    if d is None:
        return default
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)


def _utc_midnight(now: datetime) -> datetime:
    return datetime.combine(now.date(), time.min, tzinfo=timezone.utc)


def _truncate(s: str | None, n: int = 200) -> str:
    if not s:
        return ""
    s = str(s).strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _parse_action(action_taken: str | None) -> tuple[str, str | None]:
    """Match dashboard._parse_action_ticker contract: (verb, ticker).

    ``decisions.action_taken`` is free-text of the form ``"BUY NVDA →
    FILLED"`` / ``"HOLD"`` / ``"NO_DECISION"`` / ``"BLOCKED"``. We extract
    the verb (first token, uppercased) and the second token as ticker iff
    it looks like a ticker — pseudo-tickers (CASH, NONE) collapse to None
    so they don't pollute the tape (the dashboard SSOT).
    """
    if not action_taken:
        return ("UNKNOWN", None)
    s = str(action_taken).strip()
    if not s:
        return ("UNKNOWN", None)
    parts = s.replace("→", " ").split()
    verb = parts[0].upper() if parts else "UNKNOWN"
    tk: str | None = None
    if len(parts) >= 2:
        candidate = parts[1].strip().upper()
        if (candidate
                and candidate not in ("CASH", "NONE", "→", "FILLED",
                                       "BLOCKED")
                and candidate.replace("_", "").isalnum()
                and len(candidate) <= 8):
            tk = candidate
    return (verb, tk)


def build_today_action_tape(
    trades: list[dict],
    decisions: list[dict],
    now: datetime | None = None,
    since: datetime | None = None,
) -> dict:
    """Chronological tape of in-window trades and decisions.

    ``trades`` / ``decisions`` accepted in store-native (newest-first)
    order; we sort defensively. ``since`` defaults to today's UTC
    midnight if not given.

    Output ordering: oldest → newest (operator reads top-to-bottom in
    time). When a trade and a decision share the same timestamp (the
    live runner records the decision microseconds after the trade), the
    TRADE row sorts first within the tie — matches the
    ``recent_trades`` deterministic ``(timestamp, id) DESC`` tie-break,
    so the tape and the trades-table chronology never disagree.
    """
    now = _aware(now, datetime.now(timezone.utc))
    if since is None:
        since = _utc_midnight(now)
    since = _aware(since, now)
    if since > now:
        since = now

    trades = trades or []
    decisions = decisions or []

    tape: list[dict] = []
    n_buys = 0
    n_sells = 0
    n_holds = 0
    n_no_decisions = 0
    n_blocked = 0
    n_other_decisions = 0
    realized_proxy_usd = 0.0  # proxies the desk's "spend - proceeds" feeling
    notional_in_usd = 0.0
    notional_out_usd = 0.0

    # ---- TRADE rows ---------------------------------------------------
    try:
        for t in trades:
            ts = _parse_ts(t.get("timestamp"))
            if ts is None or ts < since or ts > now:
                continue
            action = (t.get("action") or "").upper()
            ticker = t.get("ticker") or "?"
            qty = float(t.get("qty") or 0.0)
            price = float(t.get("price") or 0.0)
            notional = float(t.get("value") or 0.0)
            opt = t.get("option_type")
            if action.startswith("BUY"):
                notional_in_usd += notional
                n_buys += 1
            elif action.startswith("SELL"):
                notional_out_usd += notional
                n_sells += 1
            tape.append({
                "kind": "TRADE",
                "ts": t.get("timestamp"),
                "ticker": ticker,
                "action": action,
                "qty": round(qty, 6),
                "price": round(price, 4),
                "notional_usd": round(notional, 2),
                "option_type": opt,
                "expiry": t.get("expiry"),
                "strike": t.get("strike"),
                "reason": _truncate(t.get("reason")),
            })
    except Exception:
        pass

    # ---- DECISION rows (incl. HOLD / NO_DECISION / BLOCKED) -----------
    try:
        for d in decisions:
            ts = _parse_ts(d.get("timestamp"))
            if ts is None or ts < since or ts > now:
                continue
            verb, tk = _parse_action(d.get("action_taken"))
            if verb == "HOLD":
                n_holds += 1
            elif verb in ("NO_DECISION", "NO"):
                n_no_decisions += 1
            elif verb == "BLOCKED":
                n_blocked += 1
            elif verb in ("BUY", "SELL", "BUY_CALL", "BUY_PUT",
                          "SELL_CALL", "SELL_PUT", "REBALANCE"):
                # An executed-action decision: the TRADE row already
                # captures the fill; count it here too for a faithful
                # decision-mix tally.
                n_other_decisions += 1
            else:
                n_other_decisions += 1
            tape.append({
                "kind": "DECISION",
                "ts": d.get("timestamp"),
                "verb": verb,
                "ticker": tk,
                "action_taken": d.get("action_taken"),
                "market_open": bool(d.get("market_open")),
                "signal_count": d.get("signal_count"),
                "portfolio_value": d.get("portfolio_value"),
                "cash": d.get("cash"),
                "reasoning": _truncate(d.get("reasoning"), 240),
            })
    except Exception:
        pass

    # Sort oldest -> newest. Tie-break: TRADE before DECISION (the runner
    # records the trade row microseconds before the decision row, but the
    # two store inserts can share a microsecond; pin the order so the
    # tape and the trades-table chronology never disagree).
    _KIND_ORDER = {"TRADE": 0, "DECISION": 1}

    def _sort_key(row: dict) -> tuple:
        ts = _parse_ts(row.get("ts")) or now
        return (ts, _KIND_ORDER.get(row.get("kind", ""), 9))

    tape.sort(key=_sort_key)

    if len(tape) > _MAX_TAPE_ROWS:
        # Keep the most recent _MAX_TAPE_ROWS rows; this only triggers on
        # a multi-day "since" override, never the default UTC-midnight.
        tape = tape[-_MAX_TAPE_ROWS:]

    realized_proxy_usd = round(notional_out_usd - notional_in_usd, 2)

    window_seconds = max(0.0, (now - since).total_seconds())
    window_minutes = window_seconds / 60.0
    n_decisions_total = (
        n_holds + n_no_decisions + n_blocked + n_other_decisions
    )

    return {
        "as_of": now.isoformat(),
        "since": since.isoformat(),
        "window_seconds": round(window_seconds, 1),
        "window_minutes": round(window_minutes, 2),
        "n_trades": n_buys + n_sells,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "n_decisions": n_decisions_total,
        "n_holds": n_holds,
        "n_no_decisions": n_no_decisions,
        "n_blocked": n_blocked,
        "n_other_decisions": n_other_decisions,
        "notional_in_usd": round(notional_in_usd, 2),
        "notional_out_usd": round(notional_out_usd, 2),
        # Headline cash-flow (proceeds - spend). Positive == net cash in.
        # NOT realized P&L — it ignores cost basis; that's what
        # /api/round-trips computes. Named so callers don't confuse it.
        "net_cash_flow_usd": realized_proxy_usd,
        "tape": tape,
    }
