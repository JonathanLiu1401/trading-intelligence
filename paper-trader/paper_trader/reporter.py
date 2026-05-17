"""Discord reporter — pushes trades, hourly summaries, and daily close to the channel."""
from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timedelta, timezone

from . import market
from .store import INITIAL_CASH, get_store

DISCORD_CHANNEL = "channel:1496099475838603324"
# Single source of truth — keep P/L baselines in lockstep with the store.
# A hardcoded copy silently desyncs every reported P/L% if INITIAL_CASH moves.
_INITIAL_EQUITY = INITIAL_CASH


def _send(message: str) -> bool:
    if not shutil.which("openclaw"):
        print(f"[reporter] openclaw not installed; would send:\n{message}")
        return False
    try:
        r = subprocess.run(
            ["openclaw", "message", "send",
             "--channel", "discord",
             "--target", DISCORD_CHANNEL,
             "--message", message],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            print(f"[reporter] openclaw failed: {r.stderr.strip()[:300]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("[reporter] openclaw timeout")
        return False
    except Exception as e:
        print(f"[reporter] openclaw exception: {e}")
        return False


def send_trade_alert(trade: dict) -> bool:
    """Post a single trade immediately."""
    t = trade
    extra = ""
    if t.get("option_type"):
        extra = f" {t['strike']}{t['option_type'][0].upper()} {t['expiry']}"
    body = (
        f"**TRADE** `{t['action']}` `{t['ticker']}`{extra}\n"
        f"qty `{t['qty']}` @ `${t['price']:.2f}` = `${t['value']:.2f}`\n"
        f"_{t.get('reason','')}_"
    )
    return _send(body)


def send_decision_log(summary: dict) -> bool:
    d = summary.get("decision") or {}
    action = d.get("action", "NO_DECISION")
    ticker = d.get("ticker", "")
    conf = d.get("confidence", "?")
    reasoning = d.get("reasoning", "")
    status = summary.get("status", "?")
    detail = summary.get("detail", "")
    auto = summary.get("auto_exits") or []
    pf = summary.get("snapshot") or {}
    total_value = float(pf.get("total_value") or 0.0)
    cash = float(pf.get("cash") or 0.0)
    pl = total_value - _INITIAL_EQUITY
    pl_pct = pl / _INITIAL_EQUITY * 100

    parts = [
        f"**Δ DECISION** `{action} {ticker}` → `{status}`",
        f"conf=`{conf}` value=`${total_value:.2f}` "
        f"P/L=`${pl:+.2f}` (`{pl_pct:+.2f}%`) cash=`${cash:.2f}`",
    ]
    if auto:
        parts.append("auto: " + "; ".join(f"`{a}`" for a in auto))
    if detail:
        parts.append(f"_{detail}_")
    if reasoning:
        parts.append(f"> {reasoning[:600]}")
    return _send("\n".join(parts))


def _behavioural_block() -> str:
    """Compose the behavioural verdict-alignment scorecard *verbatim* into a
    compact Discord block for the hourly / daily-close report.

    The trading stack has ~24 behavioural builders and ~30 endpoints, all of
    which the operator only ever sees on a dashboard they don't open. The
    operator lives in Discord. This routes the *synthesis* (the scorecard's
    own router verdict — does ≥1 independent behavioural check flag a problem,
    and do any concur on a theme) to the surface they actually read.

    Single source of truth (AGENTS.md invariant #10): it calls
    ``build_trader_scorecard`` with the exact same store reads as
    ``/api/scorecard`` and forwards the builder's *own* headline / focus /
    concordance verbatim — it re-derives no verdict. Observational only,
    never gates Opus, adds no caps (invariants #2/#12 — the ``self_review`` /
    ``scorecard`` precedent).

    Failure contract mirrors the rest of ``reporter``: any builder/store
    fault degrades to ``""`` ("no behavioural block this report"), **never**
    an exception ("no Discord summary this report"). NO_DATA / ERROR / None
    is suppressed (mirrors the unified ``_fetch_scorecard`` chat-line
    contract); a mature verdict — including ALIGNED_HEALTHY — is shown.
    """
    try:
        from .analytics.trader_scorecard import build_trader_scorecard
        store = get_store()
        sc = build_trader_scorecard(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(2000),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        )
        if not isinstance(sc, dict):
            return ""
        state = sc.get("state")
        headline = sc.get("headline")
        if state in (None, "NO_DATA", "ERROR") or not headline:
            return ""
        lines = [f"**BEHAVIOURAL** ◈ {state}", f"> {headline}"]
        focus = sc.get("focus")
        if isinstance(focus, dict) and focus.get("headline"):
            lines.append(
                f"> look first — {focus.get('name')}: {focus['headline']}"
            )
        for n in (sc.get("concordance") or [])[:2]:
            if not isinstance(n, dict):
                continue
            labels = ", ".join(n.get("labels") or [])
            lines.append(
                f"> concur — {n.get('count')} checks on "
                f"{n.get('theme')}: {labels}"
            )
        return "\n".join(lines)
    except Exception as e:
        print(f"[reporter] behavioural block skipped: {e}")
        return ""


def _classify_decision_outcome(action_taken: str | None) -> str:
    """Coarse bucket for a free-text ``decisions.action_taken`` value.

    The column is free text (AGENTS.md invariant #11): ``"BUY NVDA → FILLED"``,
    ``"HOLD MU → HOLD"``, ``"NO_DECISION"``, ``"SELL X → BLOCKED"``. Check
    order is load-bearing: a ``NO_DECISION`` row has no arrow, and a
    ``→ FILLED`` / ``→ BLOCKED`` verb line also contains its own verb — it
    must not be misread as ``hold`` because the buy/sell verb happened to
    precede the arrow.
    """
    s = (action_taken or "").upper()
    if "NO_DECISION" in s:
        return "no_decision"
    if "FILLED" in s:
        return "filled"
    if "BLOCKED" in s:
        return "blocked"
    if "HOLD" in s:
        return "hold"
    return "other"


def _activity_counts(decisions: list[dict], since_iso: str) -> dict[str, int]:
    """Tally decision outcomes whose timestamp is at-or-after ``since_iso``.

    ``decisions`` are ``store.recent_decisions()`` rows (newest-first). Both
    the row timestamp and ``since_iso`` are the store's own
    ``datetime.now(timezone.utc).isoformat()`` strings — fixed-offset UTC, so
    a lexical ``<`` orders them correctly (the same comparison pattern
    ``signals.py`` documents and relies on for ``first_seen``).
    """
    counts = {"filled": 0, "hold": 0, "no_decision": 0, "blocked": 0, "other": 0}
    for d in decisions:
        if (d.get("timestamp") or "") < since_iso:
            continue
        counts[_classify_decision_outcome(d.get("action_taken"))] += 1
    return counts


def _movers(positions: list[dict]) -> tuple[dict | None, dict | None]:
    """``(best, worst)`` open position by ``unrealized_pl``.

    Both ``None`` when no position carries a numeric mark. With a single
    position ``best is worst`` (same object) — callers use object identity
    to decide whether to render one line or two.
    """
    scored = [p for p in positions
              if isinstance(p.get("unrealized_pl"), (int, float))]
    if not scored:
        return None, None
    best = max(scored, key=lambda p: p["unrealized_pl"])
    worst = min(scored, key=lambda p: p["unrealized_pl"])
    return best, worst


def _window_delta(equity_asc: list[dict], since_iso: str) -> dict | None:
    """Portfolio %Δ and SPY %Δ from the first equity point at-or-after
    ``since_iso`` to the latest point.

    ``equity_asc`` is ``store.equity_curve()`` (ascending). Returns ``None``
    when there is no usable baseline (< 2 points, or the only point in-window
    is the latest one). ``alpha_pct`` is only set when both legs resolve, so
    a missing ``sp500_price`` degrades to portfolio-only, never a crash.
    """
    if len(equity_asc) < 2:
        return None
    last = equity_asc[-1]
    base = next((p for p in equity_asc
                 if (p.get("timestamp") or "") >= since_iso), None)
    if base is None or base is last:
        return None
    out: dict[str, float] = {}
    b_tv, l_tv = base.get("total_value"), last.get("total_value")
    if b_tv and b_tv > 0 and l_tv is not None:
        out["port_pct"] = (l_tv / b_tv - 1.0) * 100.0
    b_sp, l_sp = base.get("sp500_price"), last.get("sp500_price")
    if b_sp and b_sp > 0 and l_sp:
        out["spy_pct"] = (l_sp / b_sp - 1.0) * 100.0
    if "port_pct" in out and "spy_pct" in out:
        out["alpha_pct"] = out["port_pct"] - out["spy_pct"]
    return out or None


def _session_block(store, window_hours: float, label: str) -> str:
    """Compact "what the desk actually did this <label>" block for the
    hourly / daily-close report: the decision-activity mix (did the bot
    *do* anything, or sit on its hands?), the best/worst open mover, and
    the portfolio-vs-SPY delta over the window.

    Composed purely from existing store reads — no new state, no caps,
    observational only (the `_behavioural_block` precedent; invariants
    #2/#12). Failure contract mirrors the rest of ``reporter``: any
    store/compute fault degrades to ``""`` ("no session block this
    report"), **never** an exception ("no Discord summary this report").
    """
    try:
        since = (datetime.now(timezone.utc)
                 - timedelta(hours=window_hours)).isoformat()
        counts = _activity_counts(store.recent_decisions(limit=500), since)
        n_dec = sum(counts.values())
        lines = [
            f"**SESSION** ◈ last {label}",
            "```\n"
            f"Decisions {n_dec:>3}   filled {counts['filled']}  "
            f"hold {counts['hold']}  no-dec {counts['no_decision']}  "
            f"blocked {counts['blocked']}\n"
            "```",
        ]
        best, worst = _movers(store.open_positions())
        if best is not None:
            if worst is not None and worst is not best:
                lines.append(
                    f"Best `{best['ticker']}` "
                    f"${best['unrealized_pl']:+.2f}  ·  "
                    f"Worst `{worst['ticker']}` "
                    f"${worst['unrealized_pl']:+.2f}"
                )
            else:
                lines.append(
                    f"Only open mover `{best['ticker']}` "
                    f"${best['unrealized_pl']:+.2f}"
                )
        d = _window_delta(store.equity_curve(limit=5000), since)
        if d and "port_pct" in d:
            seg = f"Δ port `{d['port_pct']:+.2f}%`"
            if "spy_pct" in d:
                seg += (f"  spy `{d['spy_pct']:+.2f}%`  "
                        f"alpha `{d['alpha_pct']:+.2f}%`")
            lines.append(seg)
        return "\n".join(lines)
    except Exception as e:
        print(f"[reporter] session block skipped: {e}")
        return ""


def _realized_pl_today(trades_newest_first: list[dict], today: str
                       ) -> tuple[float, int, int] | None:
    """True realized P/L from round-trips that *closed* today (UTC).

    The existing "Realized P/L (today, cash flow basis)" line is a net-cash
    figure: a day where the desk only *deploys* cash (BUYs, no closes) reads
    as a large negative even though nothing was actually realized. That number
    is correct-by-disclosure (it says "cash flow basis") so it stays — this is
    an *additive* second line that answers the question a trader actually
    asks at the close: "what did I lock in today?"

    Consumes the ``build_round_trips`` single source of truth (AGENTS.md
    invariant #10) so the figure reconciles with ``/api/trade-asymmetry``,
    ``/api/churn``, ``session_delta`` and the scorecard — never a second
    hand-rolled P&L. ``build_round_trips`` reads the ledger in sequence and
    pairs BUYs→SELLs, so a round-trip that *opened* days ago but *closes*
    today is attributed to today correctly; a position merely opened today
    (still held) does not count.

    Args:
        trades_newest_first: ``store.recent_trades(N)`` (newest-first); this
            helper reverses it to the oldest→newest order build_round_trips
            requires. Pass a deep window so an old-open/today-close trip pairs.
        today: ``datetime.now(timezone.utc).date().isoformat()`` — the same
            UTC date string ``send_daily_close`` already computes.

    Returns ``(pnl_usd, n_closed, n_wins)``, or ``None`` when nothing closed
    today or on any failure (additive contract: a fault drops this one line,
    never the whole report — the ``_session_block`` / ``_behavioural_block``
    precedent).
    """
    try:
        from .analytics.round_trips import build_round_trips
        rts = [
            rt for rt in build_round_trips(list(reversed(trades_newest_first)))
            if (rt.get("exit_ts") or "").startswith(today)
        ]
        if not rts:
            return None
        pnl = sum(float(rt.get("pnl_usd") or 0.0) for rt in rts)
        wins = sum(1 for rt in rts if (rt.get("pnl_usd") or 0.0) > 0)
        return pnl, len(rts), wins
    except Exception as e:
        print(f"[reporter] realized-pl-today skipped: {e}")
        return None


def _portfolio_lines(positions: list[dict]) -> list[str]:
    lines = []
    for p in positions:
        # Additive: only positions carrying an explicit ``stale_mark`` True
        # (the enriched snapshot shape) get the flag. ``open_positions()``
        # table rows have no such key, so output is byte-identical to before
        # for the existing Discord path — a genuinely flat $0.00 P/L is not
        # falsely flagged; only a *missing-price* mark is.
        stale = "  ⚠ STALE (price unavailable; marked at cost)" if p.get("stale_mark") else ""
        if p["type"] in ("call", "put"):
            lines.append(
                f"  {p['ticker']} {p['type'].upper()}{p['strike']} {p['expiry']}  "
                f"qty {p['qty']}  P/L ${(p.get('unrealized_pl') or 0):+.2f}{stale}"
            )
        else:
            lines.append(
                f"  {p['ticker']:<6} qty {p['qty']:<8} avg ${p['avg_cost']:.2f} "
                f"now ${(p.get('current_price') or 0):.2f}  P/L ${(p.get('unrealized_pl') or 0):+.2f}{stale}"
            )
    return lines


def send_hourly_summary() -> bool:
    store = get_store()
    pf = store.get_portfolio()
    positions = store.open_positions()
    sp = market.benchmark_sp500()
    pl = pf["total_value"] - _INITIAL_EQUITY
    pl_pct = pl / _INITIAL_EQUITY * 100

    recent_trades = store.recent_trades(5)
    trade_lines = [
        f"  [{t['timestamp'][11:16]}] {t['action']} {t['qty']} {t['ticker']} @ ${t['price']:.2f}"
        for t in recent_trades
    ] or ["  (no trades yet)"]

    sp_line = f"S&P 500: {sp:.2f}" if sp else "S&P 500: N/A"

    body = (
        f"**HOURLY** ◈ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"```\n"
        f"Equity      ${pf['total_value']:.2f}\n"
        f"Cash        ${pf['cash']:.2f}\n"
        f"P/L         ${pl:+.2f} ({pl_pct:+.2f}%)\n"
        f"{sp_line}\n"
        f"```\n"
        f"**Positions**\n```\n"
        + ("\n".join(_portfolio_lines(positions)) or "  (none)")
        + "\n```\n**Recent trades**\n```\n"
        + "\n".join(trade_lines)
        + "\n```"
    )
    sx = _session_block(store, 1.0, "1h")
    if sx:
        body += "\n" + sx
    bx = _behavioural_block()
    if bx:
        body += "\n" + bx
    return _send(body)


def send_daily_close() -> bool:
    store = get_store()
    pf = store.get_portfolio()
    positions = store.open_positions()
    sp = market.benchmark_sp500()

    pl = pf["total_value"] - _INITIAL_EQUITY
    pl_pct = pl / _INITIAL_EQUITY * 100

    # daily slice
    today = datetime.now(timezone.utc).date().isoformat()
    todays_trades = [t for t in store.recent_trades(200) if t["timestamp"].startswith(today)]
    n_trades = len(todays_trades)
    pnl_real = sum(
        t["value"] if t["action"].startswith("SELL") else -t["value"]
        for t in todays_trades
    )

    # True realized P/L from round-trips closed today (additive — the
    # cash-flow line above stays). Deep window so an old-open/today-close
    # trip pairs correctly inside build_round_trips.
    rt = _realized_pl_today(store.recent_trades(5000), today)
    if rt is not None:
        rt_pnl, rt_n, rt_w = rt
        realized_rt_line = (
            f"Realized P/L (today, {rt_n} round-trip"
            f"{'' if rt_n == 1 else 's'} closed, {rt_w}W"
            f"/{rt_n - rt_w}L)  ${rt_pnl:+.2f}\n"
        )
    else:
        realized_rt_line = ""

    sp_line = f"S&P 500: {sp:.2f}" if sp else "S&P 500: N/A"

    body = (
        f"**DAILY CLOSE** ◈ {today}\n"
        f"```\n"
        f"Equity         ${pf['total_value']:.2f}\n"
        f"Cash           ${pf['cash']:.2f}\n"
        f"Total P/L      ${pl:+.2f} ({pl_pct:+.2f}%)  vs ${_INITIAL_EQUITY:.0f} start\n"
        f"Realized P/L (today, cash flow basis)  ${pnl_real:+.2f}\n"
        f"{realized_rt_line}"
        f"Trades today   {n_trades}\n"
        f"{sp_line}\n"
        f"```\n"
        f"**Open positions**\n```\n"
        + ("\n".join(_portfolio_lines(positions)) or "  (none)")
        + "\n```"
    )
    sx = _session_block(store, 24.0, "24h")
    if sx:
        body += "\n" + sx
    bx = _behavioural_block()
    if bx:
        body += "\n" + bx
    return _send(body)
