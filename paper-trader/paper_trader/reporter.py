"""Discord reporter — pushes trades, hourly summaries, and daily close to the channel."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import threading
from datetime import datetime, timedelta, timezone

from . import market
from .analytics.hold_discipline import build_hold_discipline
from .analytics.sector_exposure import classify as _sector_classify
from .analytics.stress_scenarios import _LEVERAGE_BETA, build_stress_scenarios
from .store import INITIAL_CASH, get_store

DISCORD_CHANNEL = "channel:1496099475838603324"
# Single source of truth — keep P/L baselines in lockstep with the store.
# A hardcoded copy silently desyncs every reported P/L% if INITIAL_CASH moves.
_INITIAL_EQUITY = INITIAL_CASH


# ── Discord delivery health ──────────────────────────────────────────────
# EVERY operator-facing notification (hourly, daily-close, trade alert, quota
# alarm, ONLINE ping, degraded-runner warning) flows through `_send()`. When
# `_send` silently fails — the 2026-05-17 `env node` PATH outage being the
# canonical case: openclaw resolved but its `#!/usr/bin/env node` shebang
# could not find `node` under systemd's minimal PATH — the trader looks fully
# alive (decisions log, dashboard up, equity ticks) while the operator's only
# real monitoring surface is DARK and there is no way to know from inside
# Discord (the failing channel can't report its own failure). This in-memory
# tracker records the outcome of recent `_send` attempts so a dead channel is
# *visible* on `/api/runner-heartbeat` instead of silent. Best-effort, never
# raises, intentionally NOT persisted — channel health is a property of the
# running process; a fresh process re-establishes it on its first send.
_notify_lock = threading.Lock()
_notify_state: dict = {
    "last_attempt_ts": None,    # ISO — most recent _send() call
    "last_ok_ts": None,         # ISO — most recent successful send
    "last_result": None,        # True / False / None (never attempted)
    "consecutive_failures": 0,
    "last_error": "",           # short reason for the most recent failure
}


def _record_send_outcome(ok: bool, error: str = "") -> None:
    """Best-effort update of the delivery-health tracker. Never raises — a
    monitoring side-channel must not be able to break the send path."""
    try:
        now = datetime.now(timezone.utc).isoformat()
        with _notify_lock:
            _notify_state["last_attempt_ts"] = now
            _notify_state["last_result"] = bool(ok)
            if ok:
                _notify_state["last_ok_ts"] = now
                _notify_state["consecutive_failures"] = 0
                _notify_state["last_error"] = ""
            else:
                _notify_state["consecutive_failures"] += 1
                _notify_state["last_error"] = (error or "")[:300]
    except Exception:
        pass


def notify_health() -> dict:
    """Operator snapshot of Discord delivery health. Pure read, never raises.

    ``verdict``:
      * ``UNKNOWN``  — no send attempted yet this process
      * ``HEALTHY``  — the most recent send succeeded
      * ``DEGRADED`` — the most recent send failed (the channel is dark)

    ``restart_recommended`` is True once failures persist (≥3 in a row) —
    the canonical openclaw/PATH outage is fixed only by a relaunch on the
    corrected code, so this points the operator at the actionable lever."""
    try:
        with _notify_lock:
            st = dict(_notify_state)
    except Exception:
        st = {"last_result": None, "consecutive_failures": 0,
              "last_ok_ts": None, "last_attempt_ts": None, "last_error": ""}
    res = st.get("last_result")
    n = st.get("consecutive_failures", 0) or 0
    if res is None:
        verdict = "UNKNOWN"
        headline = "no Discord message attempted yet this process"
    elif res:
        verdict = "HEALTHY"
        headline = "last Discord send succeeded"
    else:
        last_ok = st.get("last_ok_ts") or "never"
        verdict = "DEGRADED"
        headline = (
            f"Discord channel DARK — {n} consecutive send "
            f"failure{'' if n == 1 else 's'}, last OK {last_ok}; "
            f"last error: {st.get('last_error') or 'unknown'}")
    return {
        "verdict": verdict,
        "headline": headline,
        "consecutive_failures": n,
        "last_ok_ts": st.get("last_ok_ts"),
        "last_attempt_ts": st.get("last_attempt_ts"),
        "last_error": st.get("last_error") or "",
        "restart_recommended": (res is False and n >= 3),
    }


def _openclaw_fallback_candidates() -> list[str]:
    """Well-known on-disk locations for the ``openclaw`` binary when it is NOT
    on ``PATH``.

    Live failure (2026-05-17): ``openclaw`` is an npm-global living under the
    nvm node bin (``~/.nvm/versions/node/<ver>/bin/openclaw``). The live
    runner is launched by ``paper-trader.service`` as
    ``/usr/bin/python3 runner.py`` with systemd's minimal PATH
    (``~/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:…``) —
    which does NOT include the nvm bin. So ``shutil.which('openclaw')``
    returned ``None`` and **every** Discord message (trade alerts, hourly,
    daily-close, the new quota alert) was silently dropped with only a
    ``[reporter] openclaw not installed; would send:`` log line nobody reads.
    The Discord channel is the operator's primary surface; a PATH quirk in
    how the daemon happens to be launched must not blind it.
    """
    home = os.path.expanduser("~")
    cands = [
        os.path.join(home, ".local", "bin", "openclaw"),
        "/usr/local/bin/openclaw",
        "/usr/bin/openclaw",
    ]
    # npm-global under any installed nvm node version; newest version first.
    cands += sorted(
        glob.glob(os.path.join(home, ".nvm", "versions", "node", "*", "bin", "openclaw")),
        reverse=True,
    )
    return cands


def _resolve_openclaw() -> str | None:
    """Resolve the ``openclaw`` executable robustly, independent of how the
    runner was launched. Order: explicit ``OPENCLAW_BIN`` override (operator
    escape hatch) → ``PATH`` (``shutil.which``) → well-known fallback
    locations. Returns an absolute path or ``None`` when genuinely
    unresolvable (caller degrades to a logged no-op, never raises)."""
    env = os.environ.get("OPENCLAW_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    p = shutil.which("openclaw")
    if p:
        return p
    for c in _openclaw_fallback_candidates():
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _send(message: str) -> bool:
    bin_ = _resolve_openclaw()
    if not bin_:
        print(f"[reporter] openclaw not installed; would send:\n{message}")
        _record_send_outcome(False, "openclaw binary not resolvable")
        return False
    # openclaw is an npm-global Node script whose shebang is
    # ``#!/usr/bin/env node``. The live runner is launched by
    # ``paper-trader.service`` with systemd's minimal PATH (no nvm node bin),
    # so ``env node`` fails with
    # ``/usr/bin/env: 'node': No such file or directory`` — the openclaw
    # process exits non-zero and EVERY Discord message (hourly, daily-close,
    # trade alert, quota alarm, ONLINE ping, degraded-runner warning) is
    # silently dropped. This was the live failure on 2026-05-17: the binary
    # resolved fine (commit 64502ec) but its own ``node`` interpreter did
    # not. nvm / npm-global colocate ``node`` in the SAME bin/ directory as
    # the resolved ``openclaw``, so prepending that directory to PATH for the
    # subprocess makes the shebang resolve regardless of how the daemon was
    # launched. Best-effort: a binary with no usable dirname just runs with
    # the inherited PATH (today's behaviour), never raises.
    env = os.environ.copy()
    bin_dir = os.path.dirname(bin_)
    if bin_dir:
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    try:
        r = subprocess.run(
            [bin_, "message", "send",
             "--channel", "discord",
             "--target", DISCORD_CHANNEL,
             "--message", message],
            capture_output=True, text=True, timeout=60, env=env,
        )
        if r.returncode != 0:
            # The CLI sometimes writes the real error to stdout with an empty
            # stderr (the shebang/PATH failure does land on stderr, but be
            # defensive so the health tracker always has a usable reason).
            err = (r.stderr or "").strip() or (r.stdout or "").strip()
            print(f"[reporter] openclaw failed: {err[:300]}")
            _record_send_outcome(False, f"rc={r.returncode}: {err}")
            return False
        _record_send_outcome(True)
        return True
    except subprocess.TimeoutExpired:
        print("[reporter] openclaw timeout")
        _record_send_outcome(False, "openclaw timeout (60s)")
        return False
    except Exception as e:
        print(f"[reporter] openclaw exception: {e}")
        _record_send_outcome(False, f"exception: {e}")
        return False


def _hold_str_from_days(days: float | None) -> str:
    """``2.3d`` / ``5.4h`` / ``42m`` — compact hold-duration label from a
    fractional-day ``hold_days`` value (the build_round_trips emit). ``None``
    yields ``""`` so the caller can suppress the token entirely on a missing
    timestamp."""
    if days is None:
        return ""
    try:
        d = float(days)
    except (TypeError, ValueError):
        return ""
    if d < 0:
        return ""
    if d < 1 / 24.0:                   # < 1h
        return f"{int(d * 24 * 60)}m"
    if d < 1.0:                        # < 1d
        return f"{d * 24:.1f}h"
    return f"{d:.1f}d"


def _cash_delta_token(cash: float, trade: dict) -> str:
    """Compact ``cash $X.XX (-$Y.YY)`` (BUY, cash burned) or ``cash $X.XX
    (+$Y.YY)`` (SELL, cash freed) — the new cash absolute *with* the
    delta-from-pre-trade.

    A live trader's second follow-up after a fill is "how much of my cash
    did this just consume / free?". Absolute cash alone tells them what's
    *left*; the signed delta tells them what just *moved*, which is the
    sizing input for the next decision ("I just deployed 44% — am I
    over-extending?" / "I just freed $300 — what's the next setup?").
    Pure: derived from the in-hand ``trade.value`` (already qty×price for
    stock, qty×price×100 for options — ``store.record_trade``). No network,
    no extra reads. Degrade-safe: a missing/non-numeric/non-positive
    ``trade.value`` silently drops the delta and emits the bare cash token,
    byte-identical to the pre-feature path so any hand-built test trade
    without a ``value`` field still produces the same output.
    """
    bare = f"cash ${cash:.2f}"
    try:
        tv = float(trade.get("value") or 0.0)
    except (TypeError, ValueError):
        return bare
    if not (tv > 0):                        # 0 / negative / NaN → no delta
        return bare
    action = (trade.get("action") or "").upper()
    if action.startswith("BUY"):
        return f"{bare} (-${tv:.2f})"
    if action.startswith("SELL"):
        return f"{bare} (+${tv:.2f})"
    return bare


def _trade_impact_line(trade: dict, snapshot: dict | None,
                       store) -> str:
    """Compact "what did this trade just do to the book" one-liner appended
    to ``send_trade_alert``.

    A live trader's #1 follow-up question after a fill is the immediate
    consequence: for a BUY — "how big is this name now, and how much cash do
    I have left?"; for a SELL — "what did I lock in, and how long did I sit
    on it?". The hourly summary already exposes book-weight % and the daily
    close emits realized P&L by round-trip, but a trader waits up to an hour
    for the next hourly and a full day for the daily close — by then the
    next trade has already fired, and the alert is the only surface that
    pairs cause (the fill) with effect (the new book shape) at the moment
    of action.

    Pure composition over an already-marked ``snapshot`` (the post-trade
    snapshot from ``strategy.decide()``) and ``build_round_trips`` on the
    trade ledger. **No network**, no extra mark-to-market (the alert path
    must stay zero-latency — a slow alert would queue behind the next
    cycle's decision). Observational only, never gates (invariants #2/#12 —
    the reporter additive contract).

    The cash token includes a signed delta (``cash $X (-$Y)`` for BUY,
    ``(+$Y)`` for SELL) derived from ``trade.value`` so the trader sees
    *what just moved* alongside *what's left* without re-deriving — the
    sizing input for the next decision.

    Failure contract mirrors the rest of ``reporter``: any
    snapshot/store/builder fault degrades to ``""`` ("no impact line on
    this alert"), **never** an exception ("no trade alert this fill"). A
    missing ``snapshot`` or non-positive ``total_value`` returns ``""`` too
    so a flat / empty book never emits a misleading "0.0% of book" token.
    """
    if not isinstance(snapshot, dict):
        return ""
    try:
        total = float(snapshot.get("total_value") or 0.0)
    except (TypeError, ValueError):
        total = 0.0
    if total <= 0:
        return ""
    try:
        cash = float(snapshot.get("cash") or 0.0)
    except (TypeError, ValueError):
        cash = 0.0
    action = (trade.get("action") or "").upper()
    ticker = (trade.get("ticker") or "").upper()
    is_option = trade.get("option_type") in ("call", "put")

    # Find the post-trade book weight + qty-weighted cost basis of THIS lot
    # (same ticker, same stock/option side). Sum across the lot's matching
    # positions in the snapshot — for stocks there's only one row; for
    # options a single ticker can hold multiple strikes/expiries so we
    # attribute weight per contract leg, not per ticker. The cost-basis
    # accumulators feed the ``@ avg $X.XX`` token: after a BUY the trader
    # wants the NEW blended avg cost (sizing the next add/exit); after a
    # partial SELL the trader wants the REMAINING lot's avg cost (decide
    # whether to add or trim further). Computed from the post-trade
    # snapshot's own ``avg_cost`` field — single source of truth with the
    # store (no re-derivation), and the snapshot already reflects the
    # blend ``upsert_position`` just wrote.
    positions = snapshot.get("positions") or []
    same_lot_value = 0.0
    lot_qty_total = 0.0
    lot_cost_total = 0.0  # Σ qty · avg_cost across matched rows
    for p in positions:
        if (p.get("ticker") or "").upper() != ticker:
            continue
        try:
            mv = float(p.get("market_value") or 0.0)
        except (TypeError, ValueError):
            mv = 0.0
        ptype = (p.get("type") or "").lower()
        if (is_option and ptype in ("call", "put") and
                p.get("strike") == trade.get("strike") and
                p.get("expiry") == trade.get("expiry")):
            same_lot_value += mv
        elif not is_option and ptype == "stock":
            same_lot_value += mv
        else:
            continue
        try:
            q = float(p.get("qty") or 0.0)
            ac = float(p.get("avg_cost") or 0.0)
        except (TypeError, ValueError):
            q = ac = 0.0
        if q > 0 and ac > 0:
            lot_qty_total += q
            lot_cost_total += q * ac

    lot_avg_cost = (
        lot_cost_total / lot_qty_total if lot_qty_total > 0 else 0.0
    )
    avg_token = (
        f" @ avg ${lot_avg_cost:.2f}" if lot_avg_cost > 0 else ""
    )

    if action.startswith("BUY"):
        parts: list[str] = []
        # Show the lot weight (e.g. "NVDA 600C 2026-12 35% of book") rather
        # than aggregating across strikes — a trader sizing a fresh
        # contract cares about the contract leg, not the ticker stack.
        leg_pct = same_lot_value / total * 100.0 if total > 0 else 0.0
        if leg_pct >= 0.1:                      # >=0.1% to skip rounding noise
            label = ticker
            if is_option:
                strike = trade.get("strike")
                otype_l = (trade.get("option_type") or "")[:1].upper()
                expiry = trade.get("expiry") or ""
                # Format strike compactly — drop the .0 on whole strikes so
                # "600C" reads cleanly instead of "600.0C".
                try:
                    sf = float(strike) if strike is not None else None
                except (TypeError, ValueError):
                    sf = None
                if sf is not None:
                    label = (f"{ticker} {int(sf) if sf == int(sf) else sf}"
                             f"{otype_l} {expiry}")
            parts.append(f"{label} now {leg_pct:.1f}% of book{avg_token}")
        # Cash gives the trader the affordability number their NEXT decision
        # has to fit inside. The signed delta (`(-$Y)`) shows how much THIS
        # trade burned — the sizing input for the next add. Suppress delta
        # when ``trade.value`` is missing / non-positive (degrade-safe).
        parts.append(_cash_delta_token(cash, trade))
        return "post: " + " · ".join(parts) if parts else ""

    if action.startswith("SELL"):
        # Realized P&L on the round-trip(s) THIS sell closed. Re-derive
        # from the trade ledger via the single source of truth so the
        # alert and the daily close's "Realized P/L (round-trip)" line can
        # never disagree. ``build_round_trips`` emits one row per closed
        # round-trip; the SELL we just executed closes either one (full
        # close to zero) or none (partial close — held qty still > 0).
        parts = []
        if store is not None:
            try:
                from .analytics.round_trips import build_round_trips
                trades_oldest_first = list(
                    reversed(store.recent_trades(5000)))
                rts = build_round_trips(trades_oldest_first)
                # Match by exit_ts AND ticker — exit_ts is the closing-SELL's
                # own timestamp, which equals trade.timestamp for the trade
                # that just landed. A null trade.timestamp degrades to no
                # match → "partial close" fallback below.
                trade_ts = trade.get("timestamp")
                strike = trade.get("strike")
                expiry = trade.get("expiry")
                matched = None
                if trade_ts:
                    for rt in rts:
                        if (rt.get("ticker") or "").upper() != ticker:
                            continue
                        if rt.get("exit_ts") != trade_ts:
                            continue
                        # Disambiguate when options of the same ticker close
                        # the same second — match strike + expiry too.
                        if is_option and (
                            rt.get("strike") != strike
                            or rt.get("expiry") != expiry
                        ):
                            continue
                        matched = rt
                        break
                if matched is not None:
                    pnl = float(matched.get("pnl_usd") or 0.0)
                    pnl_pct = matched.get("pnl_pct")
                    if pnl_pct is not None:
                        try:
                            parts.append(
                                f"realized ${pnl:+.2f} "
                                f"({float(pnl_pct):+.1f}%)")
                        except (TypeError, ValueError):
                            parts.append(f"realized ${pnl:+.2f}")
                    else:
                        parts.append(f"realized ${pnl:+.2f}")
                    held = _hold_str_from_days(matched.get("hold_days"))
                    if held:
                        parts.append(f"held {held}")
            except Exception as e:
                # Builder/store fault → drop the realized fragment but still
                # surface the bookkeeping cash delta below.
                print(f"[reporter] trade-impact round-trip lookup failed: {e}")
        if not parts:
            # Partial close (still held >0) OR no round-trip context (the
            # snapshot path tells the trader "you still have a stake").
            if same_lot_value > 0:
                leg_pct = same_lot_value / total * 100.0
                parts.append(
                    f"partial — {ticker} still {leg_pct:.1f}% of book"
                    f"{avg_token}")
            else:
                # Full close with no round-trip available (e.g. caller did not
                # pass ``store`` or build_round_trips failed). Use a bare
                # "closed" token and let the unconditional cash append below
                # supply the cash — appending "closed — cash $X" here AND
                # falling through to the cash append produced a duplicated
                # "cash $X · cash $X" tail (no test exercised this path).
                parts.append("closed")
        # Cash absolute + the freed delta (`(+$Y)`) so the trader sees what
        # the SELL just generated alongside the running cash balance.
        parts.append(_cash_delta_token(cash, trade))
        return "post: " + " · ".join(parts)

    return ""


def send_trade_alert(trade: dict, snapshot: dict | None = None,
                      store=None) -> bool:
    """Post a single trade immediately.

    ``snapshot`` (post-trade, the same one ``strategy.decide()`` returns in
    ``summary["snapshot"]``) and ``store`` are optional; when supplied an
    extra ``post: …`` line is appended with the trade's immediate book
    impact (lot weight, realized P/L, hold time, cash). Existing callers
    that pass only ``trade`` still produce a byte-compatible body."""
    t = trade
    extra = ""
    if t.get("option_type"):
        extra = f" {t['strike']}{t['option_type'][0].upper()} {t['expiry']}"
    body = (
        f"**TRADE** `{t['action']}` `{t['ticker']}`{extra}\n"
        f"qty `{t['qty']}` @ `${t['price']:.2f}` = `${t['value']:.2f}`\n"
        f"_{t.get('reason','')}_"
    )
    impact = _trade_impact_line(trade, snapshot, store)
    if impact:
        body += f"\n{impact}"
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


def send_quota_alert(detail: str = "") -> bool:
    """One-shot alarm: the Claude CLI is rejecting every decision with a
    quota / usage-limit error, so the live trader is making NO trades and
    the portfolio is frozen at its last marks.

    This is the worst *silent* failure mode for a live trader — "I thought
    the bot was running; it hasn't traded in hours and nobody told me." The
    hourly/daily reports are independent of Claude so they keep flowing
    (often reading flat), which makes the freeze even easier to miss. The
    caller (`runner._cycle`) dedupes so this fires once per outage, not
    every cycle."""
    body = (
        "🛑 **CLAUDE QUOTA EXHAUSTED** ◈ live trader is FROZEN\n"
        "The decision engine (Opus 4.7 + Sonnet fallback) is being rejected "
        "with a usage/quota limit error. **No new trades will execute** until "
        "the quota resets or the plan is upgraded. Open positions are still "
        "marked-to-market; the book is otherwise idle."
    )
    if detail:
        body += f"\n_{detail[:300]}_"
    return _send(body)


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


def _decision_action_verb(action_taken: str | None) -> str | None:
    """Extract the action verb (``BUY`` / ``SELL`` / ``HOLD``) from a free-text
    ``decisions.action_taken`` row, or ``None`` for ``NO_DECISION`` / unknown.

    The column shape (AGENTS.md invariant #11) is ``"<ACTION> <TICKER> → <STATUS>"``
    where ACTION ∈ {BUY, SELL, BUY_CALL, BUY_PUT, SELL_CALL, SELL_PUT, HOLD,
    REBALANCE}. The action verb is the FIRST whitespace-separated token of the
    pre-arrow segment. Special row ``"NO_DECISION"`` (no arrow) returns ``None``.

    Buy variants (``BUY``, ``BUY_CALL``, ``BUY_PUT``) collapse to ``"BUY"`` —
    the trader-facing distinction is "deploying cash" vs. "freeing cash", not
    stock-vs-option (option mix is already surfaced via the trade alert). Sell
    variants collapse to ``"SELL"`` the same way.

    Pure on the input string — no store reads, never raises. The companion
    helper to ``_classify_decision_outcome``: that function buckets by STATUS
    (filled/hold/blocked/no-dec), this one buckets by ACTION.
    """
    s = (action_taken or "").upper().strip()
    if not s or s.startswith("NO_DECISION"):
        return None
    # Take the segment before the arrow (or the whole string if there isn't one),
    # then the first token. The action verb sits there in every row strategy.py
    # writes via record_decision.
    head_seg = s.split("→", 1)[0].strip() if "→" in s else s
    if not head_seg:
        return None
    head_tok = head_seg.split()[0] if head_seg.split() else ""
    if head_tok.startswith("BUY"):
        return "BUY"
    if head_tok.startswith("SELL"):
        return "SELL"
    if head_tok == "HOLD":
        return "HOLD"
    return None


def _activity_counts(decisions: list[dict], since_iso: str) -> dict[str, int]:
    """Tally decision outcomes whose timestamp is at-or-after ``since_iso``.

    ``decisions`` are ``store.recent_decisions()`` rows (newest-first). Both
    the row timestamp and ``since_iso`` are the store's own
    ``datetime.now(timezone.utc).isoformat()`` strings — fixed-offset UTC, so
    a lexical ``<`` orders them correctly (the same comparison pattern
    ``signals.py`` documents and relies on for ``first_seen``).

    Status buckets (``filled`` / ``hold`` / ``no_decision`` / ``blocked`` /
    ``other``) tell the trader what HAPPENED. The additive ``buys`` / ``sells``
    buckets sub-divide the ``filled`` count by direction so the operator can
    see at a glance whether the desk was DEPLOYING cash (buys) or FREEING it
    (sells) — three filled buys and three filled sells are very different
    states a "filled 3" line conflates. Counted only on FILLED rows; a BUY
    that BLOCKED counts as blocked, not as a buy (it never moved cash).
    """
    counts = {"filled": 0, "hold": 0, "no_decision": 0, "blocked": 0, "other": 0,
              "buys": 0, "sells": 0}
    for d in decisions:
        if (d.get("timestamp") or "") < since_iso:
            continue
        action_taken = d.get("action_taken")
        outcome = _classify_decision_outcome(action_taken)
        counts[outcome] += 1
        if outcome == "filled":
            verb = _decision_action_verb(action_taken)
            if verb == "BUY":
                counts["buys"] += 1
            elif verb == "SELL":
                counts["sells"] += 1
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


def _realized_pl_window(trades_newest_first: list[dict],
                        since_iso: str) -> tuple[float, int, int] | None:
    """True realized P/L from round-trips that **closed** at or after
    ``since_iso`` (UTC ISO-8601).

    The hourly ``_session_block`` showed decision activity, the best/worst
    open mover, and the portfolio %Δ over the window — but never the one
    number a portfolio manager actually reads first: "what did I lock in
    this hour?". ``_realized_pl_today`` answers the same question for the
    daily close (UTC-date startswith); this is the same arithmetic with a
    proper ISO comparison instead of a date-only ``startswith``, so any
    window (1h / 4h / 24h / since-last-summary) can ask "how much did the
    desk realize over THIS slice".

    Consumes ``build_round_trips`` (single source of truth, invariant #10
    — the same SSOT ``_realized_pl_today`` and ``/api/trade-asymmetry``
    feed off; this surface and the daily-close surface can never disagree
    on what counts as a closed trip). ``exit_ts`` is the closing-SELL's
    timestamp written by ``store.record_trade`` (UTC ISO-8601), so a
    lexical ``>=`` against ``since_iso`` is byte-correct (the
    ``signals.py`` first_seen precedent; both sides are fixed-offset UTC).

    Returns ``(pnl_usd, n_closed, n_wins)`` or ``None`` when nothing
    closed in the window OR on any failure (additive contract: a fault
    drops just this one line, never the whole report — the
    ``_realized_pl_today`` precedent).
    """
    try:
        from .analytics.round_trips import build_round_trips
        rts = [
            rt for rt in build_round_trips(list(reversed(trades_newest_first)))
            if (rt.get("exit_ts") or "") >= since_iso
        ]
        if not rts:
            return None
        pnl = sum(float(rt.get("pnl_usd") or 0.0) for rt in rts)
        wins = sum(1 for rt in rts if (rt.get("pnl_usd") or 0.0) > 0)
        return pnl, len(rts), wins
    except Exception as e:
        print(f"[reporter] realized-pl-window skipped: {e}")
        return None


def _session_block(store, window_hours: float, label: str) -> str:
    """Compact "what the desk actually did this <label>" block for the
    hourly / daily-close report: the decision-activity mix (did the bot
    *do* anything, or sit on its hands?), the best/worst open mover, the
    portfolio-vs-SPY delta over the window, AND the true realized P/L
    from round-trips that closed in the window.

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
        # `buys` / `sells` are an additive sub-division of `filled`, NOT a
        # separate top-level bucket — they would double-count in the total.
        # _classify_decision_outcome keys (the 5 status buckets) are the canonical
        # disjoint partition; this preserves it.
        n_dec = sum(counts[k] for k in
                    ("filled", "hold", "no_decision", "blocked", "other"))
        # "filled X (YB/ZS)" — surface the buy/sell direction split inline so
        # the trader can see at a glance whether the desk was DEPLOYING cash or
        # FREEING it. Only rendered when filled > 0 (no filled → no split to
        # render). Format matches existing inline-token style elsewhere in
        # _session_block (e.g. the closed-trip "(YW/ZL)" pattern).
        n_filled = counts["filled"]
        if n_filled > 0:
            filled_seg = (f"filled {n_filled} "
                          f"({counts['buys']}B/{counts['sells']}S)")
        else:
            filled_seg = f"filled {n_filled}"
        lines = [
            f"**SESSION** ◈ last {label}",
            "```\n"
            f"Decisions {n_dec:>3}   {filled_seg}  "
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
        # Realized P/L from round-trips that closed in this window — the one
        # number the daily close already shows that the hourly never did. A
        # fault drops just this line (the additive contract); the rest of the
        # SESSION block still ships.
        rp = _realized_pl_window(store.recent_trades(5000), since)
        if rp is not None:
            pnl, n_closed, n_wins = rp
            n_losses = n_closed - n_wins
            trip_word = "trip" if n_closed == 1 else "trips"
            lines.append(
                f"Closed {n_closed} {trip_word} ({n_wins}W/{n_losses}L) "
                f"realized `${pnl:+.2f}`"
            )
        return "\n".join(lines)
    except Exception as e:
        print(f"[reporter] session block skipped: {e}")
        return ""


def _benchmark_line(store) -> str:
    """One-line "am I beating the index?" for the hourly / daily report.

    The dashboard has ``/api/benchmark`` but the operator lives in Discord;
    this answers the *first* question every trader asks of an automated
    strategy — "would I have more money if I'd just bought the S&P and done
    nothing?" — without opening the (often slow/stale) dashboard. Composes
    ``build_benchmark`` **verbatim** (single source of truth, AGENTS.md
    invariant #10 — the headline string is the builder's, never re-derived
    here, so the Discord line, ``/api/benchmark`` and the CLI can never
    drift). Observational only, no caps (invariants #2/#12; the
    ``_session_block`` / ``_behavioural_block`` precedent). Failure contract
    mirrors the rest of ``reporter``: any store/compute fault degrades to
    ``""`` ("no benchmark line this report"), **never** an exception ("no
    Discord summary this report"). ``NO_DATA`` is suppressed (a zero-history
    book has nothing to say yet — the ``_behavioural_block`` NO_DATA
    precedent)."""
    try:
        from .analytics.benchmark import build_benchmark
        b = build_benchmark(store.equity_curve(limit=5000),
                             starting_equity=_INITIAL_EQUITY)
        if b.get("state") == "NO_DATA":
            return ""
        tag = b.get("verdict") or b.get("state")
        return ("**BENCHMARK** ◈ vs S&P 500 buy-and-hold\n"
                f"`{tag}`  {b['headline']}")
    except Exception as e:
        print(f"[reporter] benchmark line skipped: {e}")
        return ""


def _drawdown_line(store) -> str:
    """One-line "how far below my own high-water mark am I, and for how
    long?" for the hourly / daily report.

    The hourly/daily already show ``P/L  $X (Y%)`` — but that is P/L *vs the
    $1000 start*, which silently conflates two states a portfolio manager
    must never confuse: "never made money" and "made money then gave a chunk
    back". Drawdown-from-*peak* is the distinct, top-of-mind risk number
    every desk reads next to absolute P&L: how deep is the hole, how long
    underwater, how much already clawed back, and which name is dragging.
    ``/api/drawdown`` (+ its ``python -m paper_trader.analytics.drawdown``
    CLI) made this auditable on the *dashboard* — but the operator lives in
    Discord and never opens it (the exact dashboard→Discord gap
    ``_benchmark_line`` / ``_equity_integrity_line`` / ``_heartbeat_line``
    each closed, one dimension over: vs-index, then vs-own-peak, the two
    reference points a PM reads together).

    Consumes ``compute_drawdown``'s OWN computed fields verbatim — it
    re-derives no drawdown math (the ``_pos_pct_weight`` precedent: pure
    formatting of a builder's already-computed numbers; invariant #10
    governs verdict/headline single-sourcing and ``compute_drawdown`` emits
    none, so suppression keys off the builder's OWN ``at_high_water``
    boolean — never an invented threshold). Feeds it the EXACT same store
    reads ``drawdown_api`` uses (``equity_curve(limit=2000)`` +
    ``open_positions()``) and the same ``_INITIAL_EQUITY`` (==
    ``INITIAL_CASH``, invariant #12) so the Discord line and
    ``/api/drawdown`` are byte-aligned. **Pure store reads only — NO
    network** (the Discord-path discipline; adds zero latency).
    Observational only, never gates, adds no caps (invariants #2/#12 — the
    ``_benchmark_line`` precedent). Failure contract mirrors the rest of
    ``reporter``: any builder/store fault degrades to ``""`` ("no drawdown
    line this report"), **never** an exception ("no Discord summary this
    report").

    Suppression — surface ONLY when the book is off its high, so a book at a
    fresh high adds no hourly noise (the summary must never become its own
    lying green light — the ``_equity_integrity_line`` CLEAN-suppression
    precedent): ``at_high_water`` True (the builder's own
    within-1bp-of-peak flag) OR a non-dict / unusable result → silent.
    """
    try:
        from .analytics.drawdown import compute_drawdown
        dd = compute_drawdown(
            store.equity_curve(limit=2000),
            store.open_positions(),
            starting_equity=_INITIAL_EQUITY,
        )
        if not isinstance(dd, dict) or dd.get("at_high_water"):
            return ""
        try:
            dd_pct = float(dd.get("drawdown_pct") or 0.0)
            dd_abs = float(dd.get("drawdown_abs") or 0.0)
        except (TypeError, ValueError):
            return ""
        seg = f"`{dd_pct:+.2f}%` (${dd_abs:+.2f}) from peak"
        hrs = dd.get("hours_in_dd")
        try:
            if hrs is not None:
                seg += f" · {_ago(float(hrs) * 3600.0)} in DD"
        except (TypeError, ValueError):
            pass
        # Trough + the builder's own claw-back %, shown only when there was a
        # strictly deeper trough than the current draw (else it is just
        # "still at the lows" and recovery is 0 — nothing to add).
        try:
            tr_pct = float(dd.get("trough_pct") or 0.0)
            rec_pct = float(dd.get("recovery_pct") or 0.0)
        except (TypeError, ValueError):
            tr_pct = rec_pct = 0.0
        if tr_pct < dd_pct - 0.01:
            seg += f" · trough `{tr_pct:+.2f}%` (recovered {rec_pct:.0f}%)"
        # Top drag — the builder already sorted contributors most-negative
        # first; surface it only when the worst open name is actually a drag
        # (a book in DD purely from a *realized* loss has no open drag).
        contribs = dd.get("contributors") or []
        if (contribs and isinstance(contribs[0], dict)
                and contribs[0].get("drag")):
            c = contribs[0]
            try:
                seg += (f" · top drag {c.get('ticker')} "
                        f"${float(c.get('unrealized_pl') or 0.0):+.2f}")
            except (TypeError, ValueError):
                pass
        return "**DRAWDOWN** ◈ off the high-water mark\n" f"> {seg}"
    except Exception as e:
        print(f"[reporter] drawdown line skipped: {e}")
        return ""


def _realized_vs_unrealized_line(store) -> str:
    """One-line "is today's gain locked-in or one bad mark from zero?" for
    the hourly / daily report.

    ``/api/realized-vs-unrealized`` (commit ``f55d1b7``) made the
    banked-vs-paper P&L split auditable on the *dashboard* — but the
    operator lives in Discord and never opens it (the exact
    dashboard→Discord gap ``_benchmark_line`` / ``_drawdown_line`` /
    ``_cash_conviction_fit_line`` each closed, one dimension over: vs-
    index, then vs-own-peak, then cash-vs-signal, now banked-vs-paper —
    the classic give-back / paper-heavy / leak surfaces a discretionary
    desk reads alongside absolute P&L). Live evidence (2026-05-21 NVDA
    earnings-night round-trip, $1011.95 book): the desk is BANKED and the
    line stays silent; the moment a fresh BUY marks up into PAPER_HEAVY
    territory this surface fires automatically.

    Composes ``build_realized_vs_unrealized`` **verbatim** (single source
    of truth, AGENTS.md invariant #10 — the headline is the builder's,
    never re-derived here, so this Discord line and
    ``/api/realized-vs-unrealized`` can never disagree). Feeds it the
    EXACT same store reads the endpoint does
    (``recent_trades(5000)`` reversed into oldest→newest +
    ``equity_curve(limit=2000)``) and the same ``_INITIAL_EQUITY``
    (== ``INITIAL_CASH``, invariant #12) so the Discord line and the
    endpoint are byte-aligned. **Pure store reads only — NO network**
    (the Discord-path discipline; adds zero latency). Observational
    only, never gates, adds no caps (invariants #2/#12 — the
    ``_drawdown_line`` precedent). Failure contract mirrors the rest of
    ``reporter``: any builder/store fault degrades to ``""`` ("no
    banked-vs-paper line this report"), **never** an exception ("no
    Discord summary this report").

    Suppression — surface ONLY actionable verdicts. The verdict ladder
    in ``realized_vs_unrealized`` is most-specific-first, so the same
    silence discipline applies (the summary must never become its own
    lying green light — the ``_drawdown_line`` at-high-water suppression
    precedent):

      * ``LEAKING_PAPER`` — realized banked but open book undoing the
        banked gain (classic give-back) → ⚠️ fires.
      * ``DRAWING_DOWN`` — net P&L below ``-DD_PCT`` of starting →
        ⚠️ fires (catch-all; verbatim builder text covers the split).
      * ``PAPER_HEAVY`` — net positive but ≥66% is unrealized paper →
        ⚠️ fires (one bad mark and the headline evaporates).
      * ``BANKED`` / ``BALANCED`` / ``NO_DATA`` → silent (locked-in /
        neutral / nothing to say — never become a lying green light).
    """
    try:
        from .analytics.realized_vs_unrealized import (
            build_realized_vs_unrealized,
        )
        # Same store-read shape as ``/api/realized-vs-unrealized``
        # (dashboard.realized_vs_unrealized_api) — reversed into
        # oldest→newest because that endpoint and the builder both want it.
        trades = list(reversed(store.recent_trades(5000)))
        curve = store.equity_curve(limit=2000)
        rvu = build_realized_vs_unrealized(
            trades, curve, starting_value=_INITIAL_EQUITY,
        )
        if not isinstance(rvu, dict):
            return ""
        verdict = rvu.get("verdict")
        if verdict not in ("LEAKING_PAPER", "DRAWING_DOWN", "PAPER_HEAVY"):
            return ""
        headline = rvu.get("headline") or ""
        if not headline:
            return ""
        return (f"⚠️ **BANKED-vs-PAPER** ◈ {verdict}\n"
                f"> {headline}")
    except Exception as e:
        print(f"[reporter] realized-vs-unrealized line skipped: {e}")
        return ""


def _hold_discipline_line(store) -> str:
    """One-line "am I sitting on a loser past my own cut-time?" for the
    daily close.

    The desk's documented pathology is the disposition effect (16.7% win
    rate, ~0.52d median hold). ``/api/loser-autopsy`` only post-mortems
    *closed* trades; nothing tells the operator — who lives in Discord —
    that a *currently open* losing position has run past the desk's own
    empirical median losing hold *while it is still happening*. Composes
    ``build_hold_discipline`` **verbatim** (single source of truth,
    AGENTS.md invariant #10 — the headline is the builder's, never
    re-derived here, so the Discord line and ``/api/hold-discipline`` can
    never drift). Observational only, no caps (invariants #2/#12; the
    ``_benchmark_line`` / ``_session_block`` precedent). Failure contract
    mirrors the rest of ``reporter``: any store/compute fault degrades to
    ``""`` ("no hold-discipline line this report"), **never** an exception
    ("no Discord summary this report"). ``NO_DATA`` (no open book) and
    ``INSUFFICIENT`` (no empirical reference yet) are suppressed — there
    is nothing actionable to say (the ``_behavioural_block`` NO_DATA
    precedent)."""
    try:
        trades = list(reversed(store.recent_trades(2000)))
        h = build_hold_discipline(store.open_positions(), trades)
        if h.get("state") in ("NO_DATA", "INSUFFICIENT"):
            return ""
        tag = h.get("verdict") or h.get("state")
        return ("**HOLD DISCIPLINE** ◈ losers held past your own cut-time\n"
                f"`{tag}`  {h['headline']}")
    except Exception as e:
        print(f"[reporter] hold-discipline line skipped: {e}")
        return ""


def _stress_line(store) -> str:
    """One-line "what does a routine bad tape cost this book right now?"
    for the hourly / daily report.

    ``/api/tail-risk`` is the desk's downside number, but on a young book it
    correctly reads ``INSUFFICIENT`` (``<20`` daily returns) and the
    operator — who lives in Discord — gets summaries that never say what a
    −3 % tape or a single-name gap costs the *current* concentrated book.
    ``build_stress_scenarios`` answers that with **zero return history**
    (pure weight×beta arithmetic), so this is the between-history read.

    Composes ``build_stress_scenarios`` **verbatim** (single source of
    truth, AGENTS.md invariant #10 — the headline is the builder's, never
    re-derived here, so this Discord line and ``/api/stress-scenarios`` can
    never drift). Uses the **pinned** ``sector_exposure.classify`` /
    ``stress_scenarios._LEVERAGE_BETA`` copies (both CI-pinned to
    ``/api/risk``) so the Discord path never imports the ~9k-line
    dashboard. **Pure store reads only — NO network** (the Discord-path
    discipline; adds zero latency). Observational only, no caps, never
    gates (invariants #2/#12; the ``_hold_discipline_line`` precedent).
    Failure contract mirrors the rest of ``reporter``: any builder/store
    fault degrades to ``""`` ("no stress line this report"), **never** an
    exception ("no Discord summary this report"). ``NO_DATA`` (no priced
    book) is suppressed — nothing to say (the ``_hold_discipline_line``
    NO_DATA precedent)."""
    try:
        pf = store.get_portfolio()
        st = build_stress_scenarios(
            store.open_positions(),
            float(pf.get("total_value") or 0.0),
            _sector_classify,
            _LEVERAGE_BETA,
        )
        if not isinstance(st, dict) or st.get("state") in (None, "NO_DATA"):
            return ""
        return ("**FORWARD STRESS** ◈ what a routine bad tape costs this book\n"
                f"{st['headline']}")
    except Exception as e:
        print(f"[reporter] stress line skipped: {e}")
        return ""


def _source_mix_line(store) -> str:
    """One-line ECHO warning for held names whose news SURGE is
    actually one syndicated source mirrored across many feeds.

    ``/api/news-velocity`` answers *rate* (BUILDING/FADING). A SURGING
    z-score of +4 looks identical whether five distinct outlets are
    reporting or one wire is being mirrored across five feeds.
    ``build_news_source_mix`` adds the orthogonal *breadth* observable:
    the source-diversity verdict (STRONG/MODERATE/ECHO/QUIET). This line
    fires ONLY when at least one held ticker reads ECHO — the false-
    signal case (a chase risk: the operator sees the velocity spike
    in their hourly and trims/adds, not realising the surge is one
    wire). All other states are silent — the ``_capital_pulse_line``
    FREE / ``_host_pulse_line`` CLEAR suppression precedent (the
    summary must never become its own lying green light).

    Single source of truth (invariant #10): composes the builder's
    own ``headline`` verbatim so the Discord line and
    ``/api/news-source-mix`` can never drift.

    Discord-path discipline (no network): reads ONLY the articles.db
    that ``/api/news-source-mix`` reads and the held set from
    ``store.open_positions()``. No yfinance, no Claude, no remote DB
    — the documented per-call latency/hang hazard for any hot Discord
    line. Observational only, no caps, never gates
    (invariants #2/#12). Failure contract mirrors the rest of
    ``reporter``: any builder/store fault degrades to ``""``, never
    an exception."""
    try:
        from .analytics.news_source_mix import build_news_source_mix
        from .signals import _db_path as _signals_db_path
        import sqlite3

        positions = store.open_positions()
        held = []
        for p in positions:
            tk = (p.get("ticker") or "").upper().strip()
            if not tk or (p.get("type") or "stock") != "stock":
                continue
            if tk in {"CASH", "NONE", "NO_DECISION", "BLOCKED"}:
                continue
            if tk not in held:
                held.append(tk)
        if not held:
            return ""

        path = _signals_db_path()
        if not path:
            return ""

        now_utc = datetime.now(timezone.utc)
        since = (now_utc - timedelta(hours=24.0)).isoformat()
        articles: list[dict] = []
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
        try:
            like_clauses = " OR ".join(["title LIKE ?"] * len(held))
            like_params = [f"%{t}%" for t in held]
            cur = conn.execute(
                f"SELECT title, source, first_seen FROM articles "
                f"WHERE first_seen >= ? AND ({like_clauses}) "
                f"AND url NOT LIKE 'backtest://%' "
                f"AND source NOT LIKE 'backtest_%' "
                f"AND source NOT LIKE 'opus_annotation%' "
                f"ORDER BY first_seen DESC LIMIT 5000",
                [since] + like_params,
            )
            for r in cur.fetchall():
                articles.append({
                    "title": r[0] or "",
                    "source": r[1] or "",
                    "first_seen": r[2],
                })
        finally:
            conn.close()

        result = build_news_source_mix(
            articles, held, now=now_utc, window_hours=24.0,
        )
        if not isinstance(result, dict) or not result.get("any_echo"):
            # ECHO-only firing — STRONG/MODERATE/QUIET are silent.
            return ""
        # Verbatim from builder for no-drift with the endpoint.
        return ("**NEWS BREADTH** ◈ syndication warning — "
                + str(result.get("headline") or ""))
    except Exception as e:
        print(f"[reporter] source mix line skipped: {e}")
        return ""


def _earnings_shock_line(store) -> str:
    """One-line pre-earnings $-at-risk-by-position summary for the hourly /
    daily report.

    ``/api/event-calendar`` already tells the live trader WHICH held name
    reports WHEN (the prompt block, fed into Opus). But the operator who
    lives in Discord has never seen that surface — there is no event-
    calendar reporter line. So today a 44 %-of-book NVDA position into
    tomorrow's earnings is invisible to the desk's hourly/daily report,
    even though every analytics block here is built around closing exactly
    that "what's the hidden risk?" gap.

    Composes ``build_earnings_shock`` over ``build_event_calendar`` with
    ``history_provider=None`` (the **Discord-path no-network discipline**;
    the ``_stress_line`` / ``_recovery_line`` precedent — yfinance is the
    documented per-call latency/hang hazard, and a hung reporter call
    drops the WHOLE Discord summary). That makes every row read
    ``INSUFFICIENT_HISTORY`` at the σ level; the full σ figure is served
    by ``/api/earnings-shock`` (which pays the yfinance call once and
    SWR-caches 5 min) and re-surfaced in the digital-intern chat
    enrichment. Here the value-add is the **awareness + dollarized
    exposure** — even without σ, "NVDA in 0.9d ($444.70 = 44.5 % of book)"
    is the heads-up the Discord operator currently has zero of.

    Single source of truth (invariant #10): the held set + days_away come
    from ``build_event_calendar`` (the canonical earnings tier source),
    and the dollarized exposure mirrors ``stress_scenarios``'s position-
    value semantics (option ×100, price falls back avg_cost) via the
    builder's ``_position_value`` helper. Observational only, no caps,
    never gates (invariants #2/#12 — the ``_stress_line`` precedent).

    NO_DATA (no priced book) and NO_EVENTS (calendar quiet) are
    suppressed (the ``_hold_discipline_line`` no-noise precedent — the
    summary must never become its own lying green light). Failure
    contract mirrors the rest of ``reporter``: any builder/store fault
    degrades to ``""`` ("no earnings-shock line this report"), **never**
    an exception ("no Discord summary this report")."""
    try:
        from .analytics.earnings_shock import build_earnings_shock
        from .analytics.event_calendar import build_event_calendar
        pf = store.get_portfolio()
        positions = store.open_positions()
        held = {(p.get("ticker") or "").upper()
                for p in positions if p.get("ticker")}
        ec = build_event_calendar(positions, held)
        es = build_earnings_shock(
            positions,
            float(pf.get("total_value") or 0.0),
            ec,
            history_provider=None,
        )
        if not isinstance(es, dict) or es.get("state") in (None, "NO_DATA", "NO_EVENTS"):
            return ""
        events = es.get("events") or []
        if not events:
            return ""
        per_event = []
        for e in events:
            days = e.get("days_to_earnings")
            tk = e.get("ticker")
            wt = e.get("weight_pct")
            cv = e.get("current_value_usd")
            if days is None or tk is None or wt is None or cv is None:
                continue
            per_event.append(
                f"{tk} in {days:.1f}d (${cv:.2f} = {wt:.1f}% of book)"
            )
        if not per_event:
            return ""
        body = " · ".join(per_event)
        return ("**PRE-EARNINGS RISK** ◈ held names with imminent prints "
                "(σ figure in /api/earnings-shock)\n"
                f"{body}")
    except Exception as e:
        print(f"[reporter] earnings shock line skipped: {e}")
        return ""


def _recovery_line(store) -> str:
    """One-line "what does it take to get back to even?" for the hourly /
    daily report.

    ``/api/drawdown`` owns the *backward* "% of trough clawed back";
    nothing told the operator — who lives in Discord — the *forward*
    rally required to return to the $1000 start (the baseline every P/L
    line here is measured against) and the high-water peak, per name and
    for the book. ``build_recovery`` answers exactly that, scaled by THIS
    book's own realized daily vol (withheld until ``tail_risk`` reads OK
    — the young-book honesty precedent).

    Composes ``build_recovery`` over ``compute_drawdown`` +
    ``build_tail_risk`` **verbatim** (single source of truth, AGENTS.md
    invariant #10 — the headline is the builder's, never re-derived, so
    this Discord line, ``/api/recovery`` and the ``/api/analytics``
    ``recovery`` fold can never drift). **Pure store reads only — NO
    network** (the Discord-path discipline; the ``_stress_line`` /
    ``_drawdown_line`` precedent). Observational only, no caps, never
    gates (invariants #2/#12). Failure contract: any builder/store fault
    → ``""`` ("no recovery line this report"), **never** an exception
    ("no Discord summary this report"). ``NO_DATA`` (no priced book) and
    ``ABOVE_WATER`` (already at/above the start — nothing to recover) are
    suppressed (the ``_drawdown_line`` at-high-water precedent — the
    summary must never become its own lying green light)."""
    try:
        from .analytics.drawdown import compute_drawdown
        from .analytics.recovery import build_recovery
        from .analytics.tail_risk import build_tail_risk
        eq = store.equity_curve(limit=2000)
        dd = compute_drawdown(eq, store.open_positions(),
                              starting_equity=_INITIAL_EQUITY)
        rec = build_recovery(dd, build_tail_risk(eq), _INITIAL_EQUITY)
        if (not isinstance(rec, dict)
                or rec.get("state") in (None, "NO_DATA", "ABOVE_WATER")):
            return ""
        return ("**RECOVERY** ◈ the rally back to even\n"
                f"{rec['headline']}")
    except Exception as e:
        print(f"[reporter] recovery line skipped: {e}")
        return ""


def _capital_pulse_line(store) -> str:
    """One-line "is the desk capital-paralysed right now?" for the hourly /
    daily report.

    The **#2 documented live pathology** (AGENTS.md pass #14 #4; the
    ``capital_paralysis`` → ``buying_power`` lineage): a ~$972 book pinned
    near 98% deployed with ~$18 free, unable to act on a fresh signal for a
    day while involuntary NO_DECISION-storm droughts quietly bleed alpha.
    ``capital_paralysis`` synthesises this on the **dashboard** and
    ``buying_power`` now reaches the **Opus prompt** — but the operator,
    who lives in Discord, still gets hourly/daily summaries that never say
    the desk is frozen and bleeding. This routes the existing builder's own
    verdict to the surface the operator actually reads (the same
    dashboard→prompt→Discord trajectory ``buying_power`` followed).

    Composes ``build_capital_paralysis`` **verbatim** (single source of
    truth, AGENTS.md invariant #10 — the headline / unlock / verdict are
    the builder's, never re-derived here, so this Discord line,
    ``/api/capital-paralysis`` and the prompt-side ``buying_power`` can
    never drift). **Pure store reads only — NO network** (the Discord-path
    discipline; unlike ``_benchmark_line`` it adds zero latency).
    Observational only, no caps, never gates (invariants #2/#12; the
    ``_hold_discipline_line`` / ``_benchmark_line`` precedent). Failure
    contract mirrors the rest of ``reporter``: any builder/store fault
    degrades to ``""`` ("no capital pulse this report"), **never** an
    exception ("no Discord summary this report").

    Suppression — there must be nothing actionable to say:
      * ``NO_DATA`` (no book yet) → silent;
      * a genuinely ``FREE`` book whose involuntary-drought verdict is NOT
        ``BLEEDING`` → silent (it can act and is not losing alpha to the
        NO_DECISION storm — the ``_hold_discipline_line`` NO_DATA
        precedent);
      * ``PINNED`` / ``EMPTY`` are ALWAYS surfaced (the desk literally
        can't act), and a ``FREE`` book that is nonetheless ``BLEEDING``
        alpha through involuntary droughts IS surfaced (that is the whole
        point — the live 2026-05-18 state)."""
    try:
        from .analytics.capital_paralysis import build_capital_paralysis
        cp = build_capital_paralysis(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(5000),
            store.recent_decisions(limit=5000),
            store.equity_curve(limit=5000),
        )
        if not isinstance(cp, dict):
            return ""
        state = cp.get("state")
        if state in (None, "NO_DATA"):
            return ""
        para = cp.get("paralysis") or {}
        bleeding = para.get("verdict") == "BLEEDING"
        if state == "FREE" and not bleeding:
            return ""
        headline = cp.get("headline") or ""
        if not headline:
            return ""
        lines = [f"**CAPITAL** ◈ {state}", f"> {headline}"]
        rec = cp.get("recommended_unlock")
        if isinstance(rec, dict) and rec.get("ticker"):
            try:
                frees = float(rec.get("frees_usd") or 0.0)
            except (TypeError, ValueError):
                frees = 0.0
            lines.append(
                f"> unlock — sell {rec['ticker']} frees ${frees:.2f}")
        if bleeding and para.get("verdict_reason"):
            lines.append(f"> {para['verdict_reason']}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[reporter] capital-pulse line skipped: {e}")
        return ""


def _cash_conviction_fit_line(store) -> str:
    """One-line "is the book idle despite the loudest live signal screaming,
    or so deployed it cannot respond to one?" for the hourly / daily report.

    ``_capital_pulse_line`` answers a structural question: "is the desk
    *able* to act?" (PINNED / FREE / BLEEDING — derived from cash %, recent
    activity, and involuntary droughts). It cannot answer the orthogonal
    *point-in-time* question every PM asks next: *given today's loudest
    live signal*, **is my cash level appropriate?** ``capital_paralysis``
    reads FREE on a $341 / 34% cash book — historically correct — even
    while a 9.0 ai_score signal screams on an unheld name; nothing in
    Discord says "you have ~$340 sitting idle while QBTS is at 9.0".
    ``/api/cash-conviction-fit`` made this auditable on the *dashboard*
    via the verdict matrix (IDLE_DESPITE_SURGE / OVERDEPLOYED /
    IDLE_LOW_CONVICTION / BALANCED) — but the operator lives in Discord
    and never opens it (the exact dashboard→Discord gap
    ``_capital_pulse_line`` / ``_concentration_line`` /
    ``_heartbeat_line`` each closed, one dimension over).

    Composes ``build_cash_conviction_fit`` **verbatim** (single source of
    truth, AGENTS.md invariant #10 — the verdict / headline are the
    builder's, never re-derived here, so this Discord line and
    ``/api/cash-conviction-fit`` can never tell different stories). Feeds
    it the SAME shape that endpoint does: ``store.get_portfolio()`` for
    cash + total_value + cash_pct, ``signals.get_top_signals`` (which is
    already ``_LIVE_ONLY``-filtered, AGENTS.md #3) for the loudest live
    signals, and ``store.recent_decisions(limit=1)`` to disambiguate a
    transient idle reading right after a fill. **Pure store + read-only
    articles.db reads — NO network** (the Discord-path discipline; the
    ``_source_mix_line`` precedent). Observational only, never gates, adds
    no caps (invariants #2/#12 — the ``_capital_pulse_line`` precedent).
    Failure contract mirrors the rest of ``reporter``: any builder / store
    fault degrades to ``""`` ("no cash-fit line this report"), **never**
    an exception ("no Discord summary this report").

    Suppression — surface ONLY the two actionable verdicts:

      * ``IDLE_DESPITE_SURGE`` — cash idle while top signal screams: the
        operator should know the book is leaving alpha on the table.
      * ``OVERDEPLOYED``       — cash so low the book cannot respond to a
        loud signal without trimming: the operator should know.

    ``IDLE_LOW_CONVICTION`` (idle is correct — nothing is screaming) and
    ``BALANCED`` (the level fits) both stay silent. ``NO_DATA`` (missing
    portfolio or no signals) stays silent. The
    ``_capital_pulse_line`` FREE-and-not-bleeding /
    ``_concentration_line`` non-SINGLE_NAME_RISK suppression precedent —
    the summary must never become its own lying green light."""
    try:
        from .analytics.cash_conviction_fit import build_cash_conviction_fit
        from . import signals as _signals

        pf = store.get_portfolio() or {}
        cash = pf.get("cash")
        total_value = pf.get("total_value")
        cash_pct = None
        if (isinstance(cash, (int, float))
                and isinstance(total_value, (int, float))
                and total_value > 0):
            cash_pct = (cash / total_value) * 100.0
        # ``store.open_positions()`` is the SSOT for "what is currently held"
        # (qty > 0 AND closed_at IS NULL — invariant #11 / store.py:289). The
        # snapshot ``pf["positions"]`` is the lagged ``positions_json`` blob
        # last written by ``update_portfolio``; under the documented
        # equity-freshness divergence (AGENTS.md "Common failure modes") it
        # can be stale for a cycle. The held set we feed to the builder must
        # match the actual book, not a stale mirror.
        try:
            open_pos = store.open_positions() or []
        except Exception:
            open_pos = []
        portfolio = {
            "cash": cash,
            "total_value": total_value,
            "cash_pct": cash_pct,
            "n_positions": len(open_pos),
        }
        held_tickers = {
            (p.get("ticker") or "").upper()
            for p in open_pos
            if isinstance(p.get("ticker"), str) and p.get("ticker")
        }

        # Pull top live signals (already `_LIVE_ONLY`-filtered, AGENTS.md #3)
        # via signals.get_top_signals — same data path as `_idle_opportunity_line`
        # and `_source_mix_line`, so a builder/db fault here drops just this
        # line, never the whole summary.
        try:
            top_articles = _signals.get_top_signals(n=20, hours=4, min_score=4.0)
        except Exception:
            top_articles = []
        sig_list: list[dict] = []
        for a in (top_articles or []):
            if not isinstance(a, dict):
                continue
            tickers = a.get("tickers") or []
            if not tickers:
                continue
            # First extracted ticker wins — same convention the dashboard
            # endpoint uses. ai_score / urgency are already on the article.
            tk = (tickers[0] or "").upper()
            if not tk:
                continue
            sig_list.append({
                "ticker": tk,
                "ai_score": a.get("ai_score"),
                "urgency": a.get("urgency"),
                "source": a.get("source"),
                "held": tk in held_tickers,
            })

        # Last decision (any verb) — disambiguates a transient cash-idle
        # reading right after a fill (the builder's `recent_fill` gate).
        try:
            decisions = store.recent_decisions(limit=1) or []
        except Exception:
            decisions = []
        last_decision = decisions[0] if decisions else None

        result = build_cash_conviction_fit(portfolio, sig_list, last_decision)
        if not isinstance(result, dict):
            return ""
        verdict = result.get("verdict")
        # Surface ONLY the two actionable verdicts.
        if verdict not in ("IDLE_DESPITE_SURGE", "OVERDEPLOYED"):
            return ""
        headline = result.get("headline") or ""
        if not headline:
            return ""
        return f"**CASH FIT** ◈ {verdict}\n> {headline}"
    except Exception as e:
        print(f"[reporter] cash-conviction-fit line skipped: {e}")
        return ""


def _concentration_line(store) -> str:
    """One-line "is the book dangerously concentrated in one name?" for the
    hourly / daily report.

    ``/api/correlation`` exposes the SINGLE_NAME_RISK verdict (top stock-book
    weight ≥ ``DOMINANT_WEIGHT`` = 60%) on the *dashboard*, and the
    ``risk_mirror`` block surfaces the same fields to Opus in the *prompt*.
    But the operator who lives in Discord never sees this verdict directly —
    they see per-position weight %s in ``_portfolio_lines`` but no
    categorical "this is concentration risk" alarm. The 2026-05-19 live book
    sat at NVDA 75% of stock book, deep in SINGLE_NAME_RISK territory, with
    nothing in Discord saying so. This routes the correlation builder's own
    verdict to the surface the operator actually reads.

    Composes ``build_correlation`` **verbatim** (single source of truth,
    AGENTS.md invariant #10 — the verdict/headline are the builder's, never
    re-derived, so this Discord line and ``/api/correlation`` can never tell
    different stories). **Pure store reads only — NO network** (the
    Discord-path discipline; ``price_history`` is intentionally not passed
    so a per-position yfinance call is never required, mirroring the
    risk_mirror hot-path discipline). Computes ``market_value`` per
    position inline from the stored mark (avg_cost fallback for any
    missing price), then feeds the same shape ``build_correlation``
    expects. Observational only, never gates, no caps (invariants #2/#12 —
    the ``_capital_pulse_line`` / ``_stress_line`` precedent). Failure
    contract mirrors the rest of ``reporter``: any builder/store fault
    degrades to ``""`` ("no concentration line this report"), **never** an
    exception ("no Discord summary this report").

    Suppression — surface ONLY ``SINGLE_NAME_RISK`` (the actionable
    verdict, top weight ≥ 60%); MODERATE / DIVERSIFIED / INSUFFICIENT /
    NO_DATA stay silent so a balanced book adds no hourly noise (the
    ``_capital_pulse_line`` FREE-and-not-bleeding / ``_hold_discipline_line``
    NO_DATA precedent — the summary must never become its own lying green
    light). The per-position ``_portfolio_lines`` weight % continues to
    show raw weights regardless, so a non-SINGLE_NAME_RISK book is still
    fully diagnosable from the existing lines."""
    try:
        from .analytics.correlation import build_correlation
        positions = store.open_positions()
        sized: list[dict] = []
        for p in positions:
            try:
                qty = float(p.get("qty") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            try:
                cur = float(p.get("current_price") or 0.0)
            except (TypeError, ValueError):
                cur = 0.0
            if cur <= 0:
                # Mark unavailable — fall back to avg_cost so a stale-mark
                # position still contributes to the weight Herfindahl
                # (the _portfolio_snapshot stale_mark precedent). A bad
                # avg_cost coerces to 0 ⇒ weight 0 ⇒ position drops out.
                try:
                    cur = float(p.get("avg_cost") or 0.0)
                except (TypeError, ValueError):
                    cur = 0.0
            ptype = (p.get("type") or "").lower()
            mult = 100.0 if ptype in ("call", "put") else 1.0
            sized.append({**p, "market_value": cur * qty * mult})
        # ``price_history`` is a REQUIRED positional arg on ``build_correlation``
        # (the endpoint passes a real {ticker: [closes...]} dict). The
        # Discord-path discipline forbids the yfinance hop, so pass {} — the
        # builder degrades to ``state=INSUFFICIENT`` (verdict=None) but
        # ``top_weight_pct`` / ``top_weight_ticker`` are still computed from
        # ``market_value`` regardless (the ``risk_mirror`` weight-based
        # fallback precedent). So the SINGLE_NAME_RISK decision here keys off
        # the WEIGHT field directly — same threshold ``DOMINANT_WEIGHT`` the
        # builder uses for its OK-state verdict, so a future caller that
        # supplies real history would land on the same call site.
        co = build_correlation(sized, {})
        if not isinstance(co, dict):
            return ""
        from .analytics.correlation import DOMINANT_WEIGHT
        top_pct = co.get("top_weight_pct")
        top_tk = co.get("top_weight_ticker")
        try:
            top_pct_f = float(top_pct) if top_pct is not None else None
        except (TypeError, ValueError):
            top_pct_f = None
        if (top_pct_f is None or top_tk is None
                or top_pct_f < DOMINANT_WEIGHT * 100.0):
            return ""
        n = int(co.get("n_stock_positions") or 0)
        eff = co.get("effective_positions_naive")
        # If the OK-state path is reached (real price_history supplied at a
        # future call site), use the builder's own headline verbatim — single
        # source of truth. Otherwise (INSUFFICIENT, the live no-history path)
        # synthesise a one-line description from the same fields the OK
        # headline reads from, so the Discord block is meaningful without
        # the buried "verdict withheld" sentence.
        hl = co.get("headline") or ""
        if co.get("verdict") == "SINGLE_NAME_RISK" and hl:
            body = hl
        else:
            eff_clause = ""
            if isinstance(eff, (int, float)):
                eff_clause = f" — {eff:.1f} effective name(s) by weight"
            body = (f"SINGLE_NAME_RISK — {top_tk} is {top_pct_f:.0f}% of a "
                    f"{n}-name stock book{eff_clause}.")
        return f"⚠️ **CONCENTRATION** ◈ SINGLE_NAME_RISK\n> {body}"
    except Exception as e:
        print(f"[reporter] concentration line skipped: {e}")
        return ""


def _idle_opportunity_line(store) -> str:
    """One-line "what high-score signals arrived while the bot was dark?"
    for the hourly / daily report — the missing **regret** surface for a
    PARALYSIS drought.

    ``_host_pulse_line`` says WHY the desk is frozen; ``_capital_pulse_line``
    says whether the operator can act manually (cash free?). Neither names a
    specific MISSED opportunity. ``/api/idle-opportunity`` enumerates the
    loudest live watchlist articles inside the current
    ``build_decision_drought.current_drought`` window; the loudest miss is
    what the operator actually needs in Discord — "while you were dark for
    7.94h, NVDA scored 9.0 on an earnings-preview headline and you didn't
    decide on it."

    Composes ``build_idle_opportunity`` over
    ``build_decision_drought.current_drought`` **verbatim** (single source of
    truth, AGENTS.md invariant #10 — so the Discord line and the endpoint
    can never disagree on which articles count) plus a **read-only**
    articles.db scan bounded by the drought-start (typically narrows to
    hundreds of rows even on the 1.47M-rows/7d articles.db, the
    ``news_velocity`` cost discipline). Live-only clause applied
    (AGENTS.md #3). Pure store + sqlite reads — NO network (the
    Discord-path discipline; the ``_recovery_line`` / ``_stress_line``
    precedent). Observational only, no caps, never gates (invariants
    #2/#12). Failure contract mirrors the rest of ``reporter``: any
    builder / store / db fault degrades to ``""`` ("no idle-opportunity
    line this report"), **never** an exception ("no Discord summary this
    report").

    Suppression — silence-when-nothing-actionable (the
    ``_macro_calendar_chat_lines`` / ``_event_readiness_chat_lines``
    precedent):
      * ``NO_DROUGHT`` — bot is filling normally, by definition nothing
        was missed; ``""`` (the ``_capital_pulse_line`` FREE-and-not-bleeding
        suppression precedent).
      * ``NO_DATA`` — degenerate (no decisions yet); ``""``.
      * ``OK`` with ``n_opportunities == 0`` — the silence is honest
        (drought ran but no live signals arrived); ``""``.
      * ``OK`` with ``n_opportunities >= 1`` — the regret line is the
        builder's own ``headline`` (verbatim, so no drift from the
        endpoint).
    """
    try:
        import sqlite3
        from datetime import datetime, timezone

        from . import signals as _signals
        from .analytics.decision_drought import build_decision_drought
        from .analytics.idle_opportunity import build_idle_opportunity
        from .strategy import WATCHLIST as _WATCHLIST

        dd = build_decision_drought(
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        )
        cur = dd.get("current_drought") if isinstance(dd, dict) else None
        if not cur or not cur.get("ongoing"):
            return ""
        drought_start = cur.get("start")
        if not drought_start:
            return ""

        # Bounded scan — drought_start is typically hours, not days, ago.
        path = _signals._db_path()
        articles: list[dict] = []
        if path is not None:
            try:
                conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                                       timeout=5)
                try:
                    rows = conn.execute(
                        "SELECT title, ai_score, urgency, first_seen, url, source "
                        "FROM articles WHERE first_seen >= ? "
                        "AND ai_score >= ? "
                        "AND url NOT LIKE 'backtest://%' "
                        "AND source NOT LIKE 'backtest_%' "
                        "AND source NOT LIKE 'opus_annotation%' "
                        "ORDER BY ai_score DESC, first_seen DESC LIMIT 5000",
                        (drought_start, 6.0),
                    ).fetchall()
                finally:
                    conn.close()
                for r in rows:
                    articles.append({
                        "title": r[0] or "",
                        "ai_score": r[1],
                        "urgency": r[2],
                        "first_seen": r[3],
                        "url": r[4],
                        "source": r[5],
                    })
            except Exception as e:
                print(f"[reporter] idle-opportunity db read skipped: {e}")
                # Drop the line on DB read fault rather than fabricating a
                # "no opportunities" silence — but build_idle_opportunity
                # against an empty list returns OK with n=0, which we then
                # suppress below. That's the honest path: the reporter
                # said nothing because we couldn't see.

        held = [p.get("ticker") for p in store.open_positions()
                if isinstance(p, dict) and (p.get("ticker") or "").strip()
                and (p.get("type") or "").lower() == "stock"]
        result = build_idle_opportunity(
            dd, articles, list(_WATCHLIST),
            held_tickers=[t.upper() for t in held if t],
            now=datetime.now(timezone.utc),
        )
        if (not isinstance(result, dict)
                or result.get("state") != "OK"
                or not result.get("n_opportunities")):
            return ""
        headline = result.get("headline") or ""
        if not headline:
            return ""
        return f"**IDLE** ◈ regret\n> {headline}"
    except Exception as e:
        print(f"[reporter] idle-opportunity line skipped: {e}")
        return ""


def _host_pulse_line() -> str:
    """One-line "is the desk frozen because the *box* is overloaded?" for the
    hourly / daily report — the **#1 documented live pathology's** missing
    operator surface.

    The recurring multi-hour ``NO_DECISION`` PARALYSIS droughts (observed
    2026-05-18: a 27 h drought, 70/90 cycles NO_DECISION, **-5.87% alpha
    bleed**) are host saturation — the live trader's Opus call OOM-starved by
    out-of-band parallel Opus (review / backtest agents). ``host_guard``,
    ``/api/host-guard`` and ``/api/decision-drought`` all *diagnose* it, but
    the operator who lives in Discord gets hourly/daily summaries that never
    say it. Worse: ``_capital_pulse_line`` (which DOES reach Discord) reports
    the same freeze as ``CAPITAL ◈ PINNED`` — sending the operator to *sell a
    position* when the real, provable fix is killing the parallel Opus jobs
    (an OPS action; selling frees cash but the next decision still won't
    happen because Opus is still starved). This routes ``host_guard.pulse()``
    to the surface the operator actually reads — the same dashboard→Discord
    trajectory ``_capital_pulse_line`` / ``_stress_line`` each followed.

    Composes ``host_guard.pulse()`` **verbatim** (single source of truth,
    AGENTS.md invariant #10 — the state/headline are the builder's, never
    re-derived, so this line, ``/api/host-guard`` and the CLI can never
    drift). It is appended **before** ``_capital_pulse_line`` in both send
    paths so a top-down read matches the precedence (host saturation is the
    dominant, non-trading-fixable cause); both lines can be independently true
    and neither suppresses the other — the ``_OPS_ACTION`` discriminator in
    the headline is what stops the operator conflating them. Observational
    only, no caps, never gates (invariants #2/#12; the ``_capital_pulse_line``
    / ``_stress_line`` precedent). Pure ``host_guard`` reads — its own
    read-only DB probe, NO network (the Discord-path discipline).

    Suppression — there must be nothing actionable to say: ``CLEAR`` (the box
    is fine, or the probe couldn't tell — never cry wolf) → silent (the
    ``_capital_pulse_line`` ``FREE``-and-not-bleeding / ``_hold_discipline_line``
    NO_DATA precedent). ``SATURATED`` / ``STARVED`` are ALWAYS surfaced (the
    desk literally cannot get a decision out). Failure contract mirrors the
    rest of ``reporter``: any fault degrades to ``""`` ("no host pulse this
    report"), **never** an exception ("no Discord summary this report")."""
    try:
        from . import host_guard
        pl = host_guard.pulse()
        if not isinstance(pl, dict):
            return ""
        state = pl.get("state")
        if state in (None, "CLEAR"):
            return ""
        headline = pl.get("headline") or ""
        if not headline:
            return ""
        return f"**HOST** ◈ {state}\n> {headline}"
    except Exception as e:
        print(f"[reporter] host-pulse line skipped: {e}")
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


def _pos_pct_weight(p: dict, total_value: float | None) -> str:
    """Compact ``  (-11.0% · 59% bk)`` annotation for a Discord position line.

    The two numbers a portfolio manager reads *before* raw qty/avg/mark: the
    position's own return % and its weight as a share of total equity. The
    Discord summary is the operator's primary surface, yet it historically
    showed only ``qty/avg/now/P/L$`` — so a frozen book sitting e.g. 59% in a
    single −11% name (the live 2026-05-18 LITE state; single-name
    concentration is the desk's #1 documented pathology) looked the same as a
    balanced one. This surfaces both, on the surface the operator actually
    reads.

    Pure arithmetic on the position row + the portfolio total the caller
    already holds — NOT a re-derived builder verdict (invariant #10 governs
    verdict/headline single-sourcing; this is the *same* ``pl_pct`` formula
    ``strategy._mark_to_market`` already feeds Opus). Additive / degrade-safe
    (the ``stale_mark`` precedent, invariants #2/#12): any missing/garbage
    field, a stale (cost-fallback) mark, or a non-positive cost/total drops
    the offending token (or the whole annotation) — it never raises and never
    emits a misleading number.

      * P/L % is suppressed when the mark is stale (``stale_mark`` True ⇒
        mark == cost, so a "+0.0%" would lie next to the STALE flag) or when
        ``avg_cost`` / ``current_price`` is not a usable positive number.
      * weight % is shown only when ``total_value`` is a positive number and
        the position carries a usable mark — so the existing test callers
        that pass no total stay byte-compatible with the no-weight asserts.
    """
    def _num(x):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        if x != x:  # NaN
            return None
        return float(x)

    parts: list[str] = []
    avg = _num(p.get("avg_cost"))
    cur = _num(p.get("current_price"))
    qty = _num(p.get("qty"))
    is_opt = p.get("type") in ("call", "put")

    # ``cur >= 0`` (not ``> 0``): a worthless expired option settles at a real
    # ``current_price`` of $0 (not a missing mark — ``stale_mark`` stays False
    # via the deliberate ``_expired_intrinsic`` settlement in
    # ``strategy._mark_to_market``). The strict ``> 0`` previously suppressed
    # the −100% the trader needs to see, so a wiped contract showed only the
    # dollar P/L. ``avg > 0`` still guards the division; ``cur is None``
    # (missing mark) still suppresses.
    if (not p.get("stale_mark") and avg is not None and avg > 0
            and cur is not None and cur >= 0):
        parts.append(f"{(cur - avg) / avg * 100.0:+.1f}%")

    tv = _num(total_value)
    if (tv is not None and tv > 0 and cur is not None and cur > 0
            and qty is not None):
        mv = cur * qty * (100.0 if is_opt else 1.0)
        w = mv / tv * 100.0
        parts.append(f"{w:.0f}% bk" if w >= 1.0 else f"{w:.1f}% bk")

    return f"  ({' · '.join(parts)})" if parts else ""


def _pos_hold_age_token(p: dict, now: datetime | None = None) -> str:
    """Compact ``  held 3d`` token from a position's ``opened_at`` field.

    Mirrors the ``held=Xd`` annotation the Opus decision prompt already shows
    per position (``strategy._hold_age_str`` — invariants #2/#12, the
    stale_mark / pct-weight precedent: pure formatting of an existing field,
    additive on the Discord surface). The desk's #1 documented pathology is
    the disposition effect (riding losers); the hourly summary's position
    lines historically showed qty/avg/now/P/L with no idea *how long* a
    position had been held, so a 4-day-stuck loser read the same as a fresh
    fill.

    Read-only over ``opened_at`` — ``store.open_positions()`` always carries
    it, the unit-test positions and the ``portfolio.positions_json``
    snapshot cache do not (``strategy._portfolio_snapshot`` strips it on
    persist, see store.py upsert_position). A missing / unparseable field
    degrades to ``""`` so existing test callers stay byte-compatible.

    Sub-minute returns ``""`` so a just-opened lot does not flicker a noisy
    ``held 0m`` next to its own fill. A future ``opened_at`` (wall-clock
    stepped back — the documented clock-skew hazard) clamps to ``""`` too.
    ``now`` is injectable for deterministic tests (the
    ``_fmt_trade_stamp`` precedent).
    """
    opened = p.get("opened_at")
    if not opened:
        return ""
    try:
        dt = datetime.fromisoformat(str(opened).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    secs = (now - dt).total_seconds()
    if secs < 60:
        return ""
    if secs < 3600:
        return f"  held {int(secs // 60)}m"
    if secs < 86400:
        return f"  held {int(secs // 3600)}h"
    return f"  held {int(secs // 86400)}d"


def _pos_earnings_token(p: dict, events_by_ticker: dict | None) -> str:
    """Compact ``  ⚠ ER 0.7d`` / ``  ER 7.7d`` token from the earnings
    snapshot.

    Same observational-only, additive contract as ``_pos_pct_weight`` /
    ``_pos_hold_age_token`` (invariants #2/#12 — the stale_mark precedent).
    The decision prompt already gets the earnings calendar via
    ``build_event_calendar``, but the *operator's* Discord summary surfaced
    the per-position weight and the hold age with NO indication that the
    name reports in <1d. A trader scanning the hourly summary sees a 75%
    NVDA position with no signal that earnings are tomorrow — the exact
    "held into the print, blind" failure ``event_calendar`` exists to close
    on the decision side, mirrored to the operator side.

    Returns ``""`` when:
      * ``events_by_ticker`` is None / empty (callers without earnings data
        — existing unit-test callers — stay byte-compatible);
      * the position's ticker is absent from the events map;
      * ``days_away`` is missing / non-numeric (a corrupt event row degrades
        silently — the additive contract).

    ``  ⚠ ER Xd`` for tier ``HELD_IMMINENT`` (≤3d — flag, must-see) and
    ``  ER Xd`` otherwise (HELD_SOON within horizon — informational, no flag
    to keep the line compact when the print is still weeks away). The
    ``days_away`` formatter uses 1 decimal so a sub-day distance reads
    accurate (0.7d) and a same-day post-bell event (days_away < 0) reads
    explicitly as ``after close`` rather than rendering a confusing
    ``-0.1d``.
    """
    if not events_by_ticker:
        return ""
    ticker = (p.get("ticker") or "").upper()
    if not ticker:
        return ""
    ev = events_by_ticker.get(ticker)
    if not isinstance(ev, dict):
        return ""
    try:
        days = float(ev.get("days_away"))
    except (TypeError, ValueError):
        return ""
    tier = ev.get("tier") or ""
    if days < 0:
        # Same-day-post-bell — the event has just happened. Surface it
        # rather than hiding (the trader's first question after a print is
        # "did I hold through that?").
        if tier == "HELD_IMMINENT":
            return "  ⚠ ER after close"
        return "  ER after close"
    if tier == "HELD_IMMINENT":
        return f"  ⚠ ER {days:.1f}d"
    return f"  ER {days:.1f}d"


def _pos_alpha_token(p: dict, equity_asc: list[dict] | None,
                     sp500_now: float | None) -> str:
    """Compact ``  α +0.3%`` / ``  α −1.1%`` per-position alpha-vs-SPY
    annotation: the position's % return *minus* SPY's % return over the
    same hold window.

    Answers the one question per-position weight / age / earnings tokens
    can't: "is this trade contributing alpha, or just riding the index?"
    A position +1.2 % in a +1.5 %-since-entry tape is actually a 0.3-pt
    drag on alpha; per-trade P/L % alone hides that.

    Pure arithmetic over the already-stored ``equity_curve`` (which
    carries ``sp500_price`` at each tick) + the live ``sp500_now`` price
    the caller already holds for the report header — **no network**,
    same Discord-path discipline as ``_pos_pct_weight`` /
    ``_pos_hold_age_token`` / ``_pos_earnings_token`` (invariants #2/#12
    — the stale_mark precedent: pure formatting of pre-existing data).
    Looks up the FIRST equity_curve point at-or-after the position's
    ``opened_at`` (lexical ISO compare — same pattern ``_window_delta``
    uses) and reads its ``sp500_price`` as the entry-time SPY baseline.

    Failure contract — degrade to ``""`` (drop the token), never raise:
      * ``equity_asc`` missing/empty or ``sp500_now`` missing → ``""``;
      * position has no ``opened_at`` / non-ISO ``opened_at`` → ``""``;
      * no equity_curve point at-or-after the open → ``""``;
      * base point has no ``sp500_price`` (early rows before the field
        was added) → ``""``;
      * a stale-mark position (no real ``current_price``) → ``""``
        (the alpha number would lie next to the STALE flag, the
        ``_pos_pct_weight`` precedent);
      * avg_cost / current_price not usable positive numbers → ``""``.

    Sub-1bp absolute alpha is shown as ``α 0.0%`` (the trader sees the
    trade is *exactly* tracking the index, not a missing token); the
    threshold for emitting the token at all is just "we have the data".
    """
    if not equity_asc or not sp500_now:
        return ""
    if p.get("stale_mark"):
        return ""
    opened_at = p.get("opened_at")
    if not opened_at:
        return ""
    opened_str = str(opened_at)
    # ISO timestamps compare lexically when normalised; both equity_curve
    # rows and ``opened_at`` are written via ``store._now`` which is
    # always UTC ISO-8601 with offset, so a string compare is byte-correct
    # (the same precedent ``_window_delta`` and ``_realized_pl_window``
    # already rely on).
    base = None
    for pt in equity_asc:
        ts = pt.get("timestamp") or ""
        if ts >= opened_str:
            base = pt
            break
    if base is None:
        return ""
    try:
        base_spy = float(base.get("sp500_price") or 0.0)
    except (TypeError, ValueError):
        base_spy = 0.0
    if base_spy <= 0:
        return ""
    try:
        spy_pct = (float(sp500_now) / base_spy - 1.0) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        return ""
    try:
        avg = float(p.get("avg_cost") or 0.0)
        cur = float(p.get("current_price") or 0.0)
    except (TypeError, ValueError):
        return ""
    if avg <= 0 or cur <= 0:
        return ""
    pos_pct = (cur - avg) / avg * 100.0
    alpha = pos_pct - spy_pct
    # Round to the display grain before the +/- formatter so a tiny
    # floating-point residual (e.g. 1.7e-14) never renders as "-0.0%".
    alpha = round(alpha, 1)
    if alpha == 0.0:
        alpha = 0.0  # squash -0.0 → +0.0
    return f"  α {alpha:+.1f}%"


def _merge_stale_marks(open_positions: list[dict],
                       json_positions: list[dict] | None) -> list[dict]:
    """Return ``open_positions`` rows enriched with the ``stale_mark`` flag
    from the marked-to-market ``positions_json`` snapshot.

    ``store.open_positions()`` reads the ``positions`` TABLE, whose schema
    has no ``stale_mark`` column (it was added as an in-memory enrichment
    in ``strategy._mark_to_market``). The mark-to-market write path
    persists ``stale_mark`` ONLY into the ``portfolio.positions_json`` blob
    via ``store.update_portfolio``. So a caller that fetches
    ``store.open_positions()`` and feeds it to ``_portfolio_lines`` /
    ``_pos_pct_weight`` / ``_pos_alpha_token`` reads ``stale_mark`` as
    falsy on EVERY row — silently dropping the ⚠ STALE annotation and
    rendering misleading P/L% / alpha tokens for cost-fallback marks that
    look identical to a genuinely flat $0.00 position. The exact failure
    mode the ``stale_mark`` flag was introduced to expose.

    This helper merges the flag in by matching on the same
    ``(ticker, type, expiry, strike)`` key the positions UNIQUE constraint
    uses. Pure / never raises: any malformed snapshot row is skipped (the
    additive reporter contract — a bad enrichment must drop just this
    flag, never the whole position line). Returns the same list object
    with ``stale_mark`` mutated onto the matched rows so callers don't
    need to thread a copy through.
    """
    if not json_positions:
        return open_positions

    def _key(p: dict) -> tuple:
        return (
            (p.get("ticker") or "").upper(),
            (p.get("type") or "").lower(),
            p.get("expiry") or "",
            p.get("strike") or 0,
        )

    stale_keys: set[tuple] = set()
    for jp in json_positions:
        if not isinstance(jp, dict) or not jp.get("stale_mark"):
            continue
        stale_keys.add(_key(jp))
    if not stale_keys:
        return open_positions
    for p in open_positions:
        if not isinstance(p, dict):
            continue
        if _key(p) in stale_keys:
            p["stale_mark"] = True
    return open_positions


def _portfolio_lines(positions: list[dict],
                     total_value: float | None = None,
                     events_by_ticker: dict | None = None,
                     equity_asc: list[dict] | None = None,
                     sp500_now: float | None = None) -> list[str]:
    lines = []
    for p in positions:
        # Additive: only positions carrying an explicit ``stale_mark`` True
        # (the enriched snapshot shape) get the flag. ``open_positions()``
        # table rows have no such key, so output is byte-identical to before
        # for the existing Discord path — a genuinely flat $0.00 P/L is not
        # falsely flagged; only a *missing-price* mark is.
        stale = "  ⚠ STALE (price unavailable; marked at cost)" if p.get("stale_mark") else ""
        # Per-position return % + book weight %. ``total_value`` defaults to
        # None so any caller that does not pass it (the existing unit-test
        # callers) gets the no-weight form — byte-compatible with the prior
        # substring assertions; only the live hourly/daily callers, which
        # already hold ``pf['total_value']``, opt into the weight token.
        pw = _pos_pct_weight(p, total_value)
        # Hold age — same shape as the Opus prompt's per-position annotation
        # (strategy._hold_age_str). Surfaces the disposition-effect signal on
        # the Discord surface the operator actually reads. Empty when
        # opened_at is absent (existing test callers), so the existing
        # byte-compatible assertions stay locked.
        age = _pos_hold_age_token(p)
        # Per-position earnings flag — observational, additive, opt-in
        # (events_by_ticker defaults to None → token is ""). Surfaces the
        # forward earnings event the decision prompt already sees, on the
        # operator's Discord surface.
        er = _pos_earnings_token(p, events_by_ticker)
        # Per-position alpha-vs-SPY since entry — observational, additive,
        # opt-in (equity_asc / sp500_now default to None → token is "").
        # Pure arithmetic over already-stored data — no network. Byte-
        # compatible with every existing test caller that does not pass
        # the new kwargs (the events_by_ticker precedent).
        al = _pos_alpha_token(p, equity_asc, sp500_now)
        if p["type"] in ("call", "put"):
            lines.append(
                f"  {p['ticker']} {p['type'].upper()}{p['strike']} {p['expiry']}  "
                f"qty {p['qty']}  P/L ${(p.get('unrealized_pl') or 0):+.2f}{pw}{age}{er}{al}{stale}"
            )
        else:
            lines.append(
                f"  {p['ticker']:<6} qty {p['qty']:<8} avg ${p['avg_cost']:.2f} "
                f"now ${(p.get('current_price') or 0):.2f}  P/L ${(p.get('unrealized_pl') or 0):+.2f}{pw}{age}{er}{al}{stale}"
            )
    return lines


def _earnings_events_by_ticker() -> dict | None:
    """Resolve the earnings snapshot via ``build_event_calendar`` (the SAME
    source the Opus decision prompt already sees — invariant #10 single
    source of truth) and reshape it as ``{ticker: event_dict}`` for the
    per-position lookup in ``_portfolio_lines``.

    Pure filesystem read inside the builder (the ``signals.py`` /
    ``event_calendar`` precedent — NO network on the Discord path, the
    ``_concentration_line`` discipline). Wrapped end-to-end: any builder
    fault returns ``None`` so the calling line drops the earnings token but
    the whole hourly / daily summary still ships (the reporter additive
    failure contract: a side block must never take down the whole report).

    Returns ``None`` when:
      * the calendar file is missing / stale / corrupt (``source_ok=False``);
      * the builder raises (``import`` error / disk fault / future-incompat);
      * positions / in-play resolution fails.

    Returns ``{}`` when the calendar is fine but no held / in-play name has
    a print within the 14d horizon — the per-position token then renders
    empty for every position (byte-identical to the calendar-unavailable
    branch from the rendered position lines' perspective)."""
    try:
        from .analytics.event_calendar import build_event_calendar
        store = get_store()
        positions = store.open_positions()
        held = {(p.get("ticker") or "").upper()
                for p in positions if p.get("ticker")}
        # Mirror strategy.decide's scope: held ∪ FULL WATCHLIST so a name
        # the trader is *about* to add (not yet held) shows the earnings
        # flag through the position line eventually. Practically here only
        # held names matter (the line iterates positions), but build the
        # scope consistently with the prompt-side caller so a future
        # extension to /api/portfolio doesn't re-narrow the view.
        try:
            from .strategy import WATCHLIST
            in_play = held | {t.upper() for t in WATCHLIST}
        except Exception:
            in_play = held
        rep = build_event_calendar(positions, in_play)
        if not isinstance(rep, dict):
            return None
        if not rep.get("source_ok"):
            return None
        out: dict[str, dict] = {}
        for ev in rep.get("events") or []:
            if not isinstance(ev, dict):
                continue
            tk = (ev.get("ticker") or "").upper()
            if not tk:
                continue
            out[tk] = ev
        return out
    except Exception as e:
        print(f"[reporter] earnings token skipped: {e}")
        return None


def _singleton_lock_line() -> str:
    """Loud one-liner when THIS runner booted WITHOUT the single-instance
    guard (degraded — invariant #19 fail-open). A guard-less runner can be
    double-trading the same $1000 book against a properly-locked instance
    (observed live 2026-05-17/18) and was previously invisible from every
    operator surface. The operator lives in Discord, so the hourly / daily
    summary is the right surface.

    Returns ``""`` when this runner holds the lock (the normal case — no
    noise) or on ANY failure. Same additive failure contract as the other
    reporter blocks: a fault drops this one line, never the whole summary.
    The ``runner`` import is lazy (``runner`` imports ``reporter`` at module
    load — a top-level import here would be circular)."""
    try:
        from . import runner
        st = runner.singleton_lock_state()
        if not isinstance(st, dict) or not st.get("degraded"):
            return ""
        return ("⚠️ **RUNNER DEGRADED** ◈ this trader booted WITHOUT the "
                "single-instance guard — another runner may be double-trading "
                "the same paper book. Restart paper-trader so one guarded "
                "instance owns the lock.")
    except Exception as e:
        print(f"[reporter] singleton-lock line skipped: {e}")
        return ""


def _systemctl_user(verb: str) -> str:
    """``systemctl --user <verb> paper-trader`` → its one-word status, or
    ``"unknown"`` on any failure (unreadable user bus, no systemctl, …).
    Mirrors ``dashboard.supervision_api``'s probe exactly so the Discord line
    and ``/api/supervision`` feed the SAME builder identical inputs."""
    try:
        r = subprocess.run(
            ["systemctl", "--user", verb, "paper-trader"],
            capture_output=True, text=True, timeout=3,
        )
        return ((r.stdout or "").strip()
                or (r.stderr or "").strip() or "unknown")
    except Exception:
        return "unknown"


def _supervision_line() -> str:
    """Loud one-liner when this trader has NO restart safety net and/or is on
    stale code — the **#1 recurring HIGH operational finding** across review
    passes (an orphaned ``runner.py``, PPID 1, systemd unit
    ``disabled``/``inactive``, behind HEAD: the moment its git-watcher /
    deadman does ``os._exit(0)`` the trader stays DOWN permanently).

    ``/api/supervision`` made this visible on the *dashboard* — but the
    operator lives in Discord and never opens it (the exact dashboard→Discord
    gap ``_capital_pulse_line`` / ``_heartbeat_line`` / ``_singleton_lock_line``
    each closed, one dimension over). This routes the supervision builder's
    OWN verdict + recommendation to the surface the operator actually reads.

    Composes ``build_supervision`` **verbatim** (single source of truth,
    AGENTS.md invariant #10 — the verdict / recommendation strings are the
    builder's, never re-derived here, so this Discord line and
    ``/api/supervision`` can never tell different stories). The impure probes
    (pid/ppid, ``systemctl --user``, git HEAD/behind) live here — the
    established "process/network in the caller, builder is pure" split. The
    git boot/head SHAs are read from the already-imported ``dashboard``
    module so there is ONE boot-SHA source per process (``runner`` starts the
    dashboard thread at boot, so by the time an hourly/daily fires ≥1h later
    ``dashboard._BOOT_SHA`` is populated). Observational only, never gates,
    adds no caps (invariants #2/#12 — the ``_singleton_lock_line`` precedent).

    Suppression — surface ONLY when the operator must act, so a healthy
    supervised trader adds no hourly noise (the summary must never become its
    own lying green light). The actionable set is the builder's own
    ``actionable`` flag (single-sourced — the reporter never re-derives which
    verdicts matter): everything **except** HEALTHY is surfaced, incl.
    UNKNOWN (an unreadable user bus is closer to "no safety net" than to
    "healthy" — the recommendation already names the exact verify commands).

    Failure contract mirrors the rest of ``reporter``: any probe/builder/
    import fault degrades to ``""`` ("no supervision line this report"),
    **never** an exception ("no Discord summary this report"). The
    ``dashboard`` import is lazy (a top-level import would be circular —
    ``dashboard`` is heavy and ``runner`` imports ``reporter`` first)."""
    try:
        from .analytics.supervision import build_supervision
        try:
            ppid = os.getppid()
        except Exception:
            ppid = None
        boot_sha = head_sha = None
        behind = 0
        try:
            from . import dashboard
            boot_sha = dashboard._BOOT_SHA
            head_sha, behind = dashboard._head_sha_and_behind()
        except Exception as e:
            print(f"[reporter] supervision git probe skipped: {e}")
        sup = build_supervision(
            pid=os.getpid(), ppid=ppid,
            unit_active=_systemctl_user("is-active"),
            unit_enabled=_systemctl_user("is-enabled"),
            boot_sha=boot_sha, head_sha=head_sha, behind=behind,
        )
        if not isinstance(sup, dict) or not sup.get("actionable"):
            return ""
        verdict = sup.get("verdict") or "UNKNOWN"
        rec = sup.get("recommendation") or ""
        if not rec:
            return ""
        return (f"⚠️ **SUPERVISION** ◈ {verdict}\n> {rec}")
    except Exception as e:
        print(f"[reporter] supervision line skipped: {e}")
        return ""


def _equity_integrity_line(store) -> str:
    """One-line "can I trust the recorded P&L history?" for the hourly /
    daily report.

    Every headline P&L surface the operator reads — the hourly Equity/P/L
    block, ``_benchmark_line``, the dashboard ``/api/drawdown`` /
    ``/api/benchmark`` / ``/api/analytics`` Sharpe — is derived from
    ``equity_curve``. A silent corruption there (a negative-cash over-draw on
    the no-hard-cap book — invariant #12; a non-positive-equity row; a
    no-trade mismark / stale-price-unfreeze / option-settlement jump) poisons
    *all* of them with nothing in Discord saying so. ``/api/equity-integrity``
    made this auditable on the *dashboard* — but the operator lives in
    Discord and never opens it (the exact dashboard→Discord gap
    ``_heartbeat_line`` / ``_capital_pulse_line`` / ``_singleton_lock_line``
    each closed, one dimension over). This routes the integrity builder's own
    verdict to the surface the operator actually reads.

    Composes ``build_equity_integrity`` **verbatim** (single source of truth,
    AGENTS.md invariant #10 — the headline / verdict are the builder's, never
    re-derived here, so this Discord line and ``/api/equity-integrity`` can
    never tell different stories) and feeds it the EXACT same store reads the
    endpoint does (``equity_curve(limit=5000)`` + ``recent_trades(5000)``) so
    the two surfaces are byte-aligned. **Pure store reads only — NO network**
    (the Discord-path discipline; adds zero latency). Observational only,
    never gates, adds no caps (invariants #2/#12 — the ``_heartbeat_line``
    precedent). Failure contract mirrors the rest of ``reporter``: any
    builder/store fault degrades to ``""`` ("no integrity line this report"),
    **never** an exception ("no Discord summary this report").

    Suppression — surface ONLY when the recorded P&L history is NOT
    trustworthy, so a clean curve adds no hourly noise (the summary must
    never become its own lying green light — the ``_heartbeat_line``
    HEALTHY-suppression precedent):
      * ``CORRUPT`` (negative-cash / non-positive-equity) → ALWAYS surfaced
        (the headline P&L is unreliable — the whole point);
      * ``SUSPECT`` (>=1 unexplained no-trade jump) → surfaced (a likely
        mismark / settlement artifact the operator should sanity-check);
      * ``CLEAN`` / ``NO_DATA`` (and ERROR / any non-verdict) → silent
        (nothing actionable — the ``_hold_discipline_line`` NO_DATA /
        ``_heartbeat_line`` HEALTHY suppression precedent).
    """
    try:
        from .analytics.equity_integrity import build_equity_integrity
        ei = build_equity_integrity(
            store.equity_curve(limit=5000),
            store.recent_trades(5000),
        )
        if not isinstance(ei, dict):
            return ""
        verdict = ei.get("verdict")
        if verdict not in ("SUSPECT", "CORRUPT"):
            return ""
        headline = ei.get("headline") or ""
        if not headline:
            return ""
        return (f"⚠️ **EQUITY INTEGRITY** ◈ {verdict}\n> {headline}")
    except Exception as e:
        print(f"[reporter] equity-integrity line skipped: {e}")
        return ""


def _equity_freshness_line(store) -> str:
    """One-line "is the equity point my benchmark/P&L headline is computed
    from still current, or frozen behind a fresher book under load?" for the
    hourly / daily report.

    ``_equity_integrity_line`` answers "can I trust the recorded P&L history"
    (corruption *within* recorded points). This is the orthogonal,
    repeatedly-observed-live question one dimension over: under a
    host-saturation NO_DECISION storm the live ``portfolio`` table re-marks
    every cycle while the latest ``equity_curve`` point lags a whole cycle
    behind, so ``_benchmark_line`` / the hourly P/L (both derived from
    ``equity_curve``) silently misstate the true account by the divergence
    with nothing in Discord saying so (observed live 2026-05-18:
    ``/api/portfolio`` $924.13 vs ``/api/benchmark`` $928.92). The operator
    lives in Discord and never opens ``/api/equity-freshness`` — the exact
    dashboard→Discord gap ``_equity_integrity_line`` / ``_heartbeat_line`` /
    ``_capital_pulse_line`` each closed.

    Composes ``build_equity_freshness`` **verbatim** (single source of truth,
    AGENTS.md invariant #10 — the headline / verdict are the builder's, never
    re-derived here, so this Discord line and ``/api/equity-freshness`` can
    never tell different stories) and feeds it the EXACT same store reads the
    endpoint does (``get_portfolio()`` + ``equity_curve(limit=5000)``) plus
    the same ``market.is_market_open()`` cadence probe so the two surfaces are
    byte-aligned. **Pure store reads only — NO network beyond the same
    market-hours check the rest of reporter already does.** Observational
    only, never gates, adds no caps (invariants #2/#12 — the
    ``_equity_integrity_line`` precedent). Failure contract mirrors the rest
    of ``reporter``: any builder/store fault degrades to ``""`` ("no
    freshness line this report"), **never** an exception ("no Discord summary
    this report").

    Suppression — surface ONLY when the curve the headline KPIs are computed
    from is not current, so a fresh book adds no hourly noise (the summary
    must never become its own lying green light — the
    ``_equity_integrity_line`` HEALTHY-suppression precedent):
      * ``DIVERGED``    — stale AND materially off the live book → ALWAYS
        surfaced (every benchmark/drawdown/Sharpe/P&L headline is wrong by
        the divergence — the whole point);
      * ``STALE_CURVE`` — curve lagging but the book has barely moved →
        surfaced (the operator should know the loop is behind);
      * ``FRESH`` / ``NO_DATA`` (and ERROR / any non-verdict) → silent
        (nothing actionable — the ``_equity_integrity_line``
        CLEAN/NO_DATA suppression precedent).
    """
    try:
        from .analytics.equity_freshness import build_equity_freshness
        ef = build_equity_freshness(
            store.get_portfolio(),
            store.equity_curve(limit=5000),
            market.is_market_open(),
        )
        if not isinstance(ef, dict):
            return ""
        verdict = ef.get("verdict")
        if verdict not in ("DIVERGED", "STALE_CURVE"):
            return ""
        headline = ef.get("headline") or ""
        if not headline:
            return ""
        return (f"⚠️ **EQUITY FRESHNESS** ◈ {verdict}\n> {headline}")
    except Exception as e:
        print(f"[reporter] equity-freshness line skipped: {e}")
        return ""


def _heartbeat_line(store) -> str:
    """One-line "is the decision loop actually deciding, or wedged?" for the
    hourly / daily report.

    The operator lives in Discord. ``/api/runner-heartbeat`` (pass #17) made
    a brain-dead loop visible on the *dashboard* — but the hourly/daily
    summary, the surface the operator actually reads, still looked flat-green
    while the engine sat in a host-load NO_DECISION storm (the live
    2026-05-18 state: 18/20 cycles NO_DECISION, ``restart_recommended:true``,
    surfaced nowhere in Discord). ``send_quota_alert`` covers only the
    *distinct* quota-exhaustion freeze (a specific ``quota_exhausted`` flag);
    a host-load IDLE_STORM had no Discord surface at all. This routes the
    heartbeat builder's own verdict to the surface the operator reads (the
    same dashboard→Discord trajectory ``_capital_pulse_line`` /
    ``_singleton_lock_line`` followed).

    Composes ``build_runner_heartbeat`` **verbatim** (single source of truth,
    AGENTS.md invariant #10 — the headline / verdict / restart flag are the
    builder's, never re-derived here, so this Discord line and
    ``/api/runner-heartbeat`` can never tell different stories). The reporter
    owns the ``store.recent_decisions(20)`` read + ``market.is_market_open``
    + wall clock and passes the dicts to the pure builder — the exact
    "network in the caller, builder is pure" split the endpoint uses, so the
    two surfaces stay byte-aligned. Observational only, never gates, adds no
    caps (invariants #2/#12 — the ``_capital_pulse_line`` precedent). Failure
    contract mirrors the rest of ``reporter``: any builder/store fault
    degrades to ``""`` ("no heartbeat line this report"), **never** an
    exception ("no Discord summary this report").

    Suppression — surface ONLY when there is something the operator should
    act on, so a healthy deciding loop adds no hourly noise (the summary must
    never become its own lying green light):
      * ``restart_recommended`` True (STALLED liveness, or an IDLE_STORM
        decision-efficacy storm) → ALWAYS surfaced (the engine is dead or
        wedged — the whole point);
      * ``LAGGING`` liveness or ``DEGRADED`` decision-efficacy → surfaced
        (impaired throughput, the operator should know);
      * HEALTHY + PRODUCING / NO_DATA → silent (nothing actionable — the
        ``_hold_discipline_line`` DISCIPLINED/NO_DATA suppression precedent).
    """
    try:
        from .analytics.runner_heartbeat import build_runner_heartbeat
        decs = store.recent_decisions(20)
        last_ts = decs[0].get("timestamp") if decs else None
        recent_actions = [d.get("action_taken") for d in decs]
        hb = build_runner_heartbeat(
            last_ts, market.is_market_open(), recent_actions=recent_actions)
        if not isinstance(hb, dict):
            return ""
        verdict = hb.get("verdict")
        eff = hb.get("decision_efficacy")
        eff_verdict = eff.get("verdict") if isinstance(eff, dict) else None
        restart = bool(hb.get("restart_recommended"))
        actionable = (
            restart
            or verdict in ("STALLED", "LAGGING")
            or eff_verdict in ("IDLE_STORM", "DEGRADED")
        )
        if not actionable:
            return ""
        headline = hb.get("headline") or ""
        if not headline:
            return ""
        prefix = "⚠️ RESTART RECOMMENDED — " if restart else ""
        lines = [f"**RUNNER** ◈ {verdict}", f"> {prefix}{headline}"]
        # The top-level headline already folds in the IDLE_STORM clause
        # (build_runner_heartbeat appends it); only DEGRADED carries
        # additive detail not already in `headline`.
        if (isinstance(eff, dict) and eff_verdict == "DEGRADED"
                and eff.get("headline")):
            lines.append(f"> efficacy — {eff['headline']}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[reporter] heartbeat line skipped: {e}")
        return ""


def _position_attention_line(store) -> str:
    """One-line "which open positions has Opus stopped examining?" for the
    hourly / daily report.

    ``analytics/position_attention.py`` answers the **per-position** question
    ``decision_health`` (aggregate NO_DECISION rate), ``decision_drought``
    (portfolio-wide drift) and ``hold_discipline`` (hold-time vs empirical
    cut-time) do not: *which specific held lots has Opus gone hours without
    examining?*. When the documented #1 pathology (host-saturation
    NO_DECISION storms — see ``_host_pulse_line``) drags on, the live trader
    silently defaults to holding every open position while those positions
    are no longer being **evaluated**. ``/api/position-attention`` (commit
    ``f703cb2``) made this auditable on the *dashboard* — but the operator
    lives in Discord and never opens it (the exact dashboard→Discord gap
    ``_host_pulse_line`` / ``_capital_pulse_line`` / ``_singleton_lock_line``
    each closed, one dimension over: aggregate-vs-host → per-held-position).

    Composes ``build_position_attention`` **verbatim** (single source of
    truth, AGENTS.md invariant #10 — the verdict / note are the builder's,
    never re-derived, so this Discord line and ``/api/position-attention``
    can never tell different stories) and feeds it the EXACT same store
    reads the endpoint does (``open_positions()`` + ``recent_decisions``).
    **Pure store reads only — NO network** (the Discord-path discipline;
    adds zero latency). Observational only, never gates, adds no caps
    (invariants #2/#12 — the ``_host_pulse_line`` precedent). Failure
    contract mirrors the rest of ``reporter``: any builder/store fault
    degrades to ``""`` ("no attention line this report"), **never** an
    exception ("no Discord summary this report").

    Suppression — surface ONLY when ≥1 held position has gone stale, so an
    actively-evaluated book adds no hourly noise (the summary must never
    become its own lying green light — the ``_heartbeat_line`` HEALTHY
    suppression precedent):
      * ``NEGLECTED_BOOK`` (>=1 position no Opus look in >24h) → ALWAYS
        surfaced (the operator should not assume a passively-held lot is
        still under model attention — the whole point);
      * ``STALE_BOOK`` (>=1 position last seen >6h ago) → surfaced (the
        operator should know which lots are drifting unmonitored);
      * ``OK`` / ``INSUFFICIENT_DATA`` (and any non-verdict) → silent
        (nothing actionable — the ``_hold_discipline_line`` NO_DATA /
        ``_heartbeat_line`` HEALTHY suppression precedent).

    Renders up to 3 worst-first per-position lines so the operator sees the
    exact tickers to triage, not just an aggregate count.
    """
    try:
        from .analytics.position_attention import build_position_attention
        pa = build_position_attention(
            store.open_positions(),
            store.recent_decisions(limit=3000),
        )
        if not isinstance(pa, dict):
            return ""
        verdict = pa.get("verdict")
        if verdict not in ("STALE_BOOK", "NEGLECTED_BOOK"):
            return ""
        note = pa.get("note") or ""
        if not note:
            return ""
        positions = pa.get("positions") or []
        # Worst-first: NEGLECTED before STALE. The builder already sorts that
        # way, but filter so a STALE_BOOK with one neglected outlier still
        # shows the neglected one first.
        worst = [p for p in positions
                 if p.get("verdict") in ("NEGLECTED", "STALE")][:3]
        lines = [f"⚠️ **ATTENTION** ◈ {verdict}", f"> {note}"]
        for p in worst:
            tk = p.get("ticker", "?")
            hrs = p.get("hours_since_last_decision")
            v = p.get("verdict", "?")
            if hrs is None:
                lines.append(f"> `{tk:>6}` {v} — no Opus look on record")
            else:
                lines.append(f"> `{tk:>6}` {v} — {hrs:.1f}h since last look")
        return "\n".join(lines)
    except Exception as e:
        print(f"[reporter] position-attention line skipped: {e}")
        return ""


def _no_decision_reasons_line(store) -> str:
    """One-line "WHY isn't the bot deciding?" for the hourly / daily report.

    ``_heartbeat_line`` raises ``IDLE_STORM`` once consecutive
    NO_DECISION cycles cross the threshold, and its blanket
    recommendation is "restart may clear a wedged CLI". But the *real*
    cause of a storm splits into operator-distinct buckets that the
    runner already records per-row in ``decisions.reasoning``:

      * QUOTA_EXHAUSTED  — wait for reset; restart does NOTHING.
      * HOST_SATURATED   — kill parallel Opus jobs; restart does NOTHING.
      * MODEL_EMPTY      — wedged CLI; restart IS the fix.
      * PARSE_FAILED     — prompt / model regression; inspect, tweak prompt.
      * RETRY_FAILED     — JSON-nudge retry not rescuing; ditto.

    Telling the operator "restart" when the storm is actually quota or
    host saturation wastes the only Discord surface they read on a
    fix that cannot work. This routes the no_decision_reasons builder's
    dominant-cause verdict + its TARGETED recommendation to the surface
    the operator actually reads (the dashboard→Discord trajectory
    ``_heartbeat_line`` / ``_capital_pulse_line`` each followed).

    Composes ``build_no_decision_reasons`` **verbatim** (single source of
    truth, AGENTS.md invariant #10 — the headline / dominant bucket /
    recommendation are the builder's, never re-derived). **Pure store
    read only — NO network.** Observational only, never gates, adds no
    caps (invariants #2/#12 — the ``_heartbeat_line`` precedent). Failure
    contract mirrors the rest of ``reporter``: any builder/store fault
    degrades to ``""`` ("no reason-breakdown line this report"), never an
    exception ("no Discord summary this report").

    Suppression — surface ONLY a real DOMINANT cause. A MIXED storm
    (no single bucket ≥ ``DOMINANT_THRESHOLD_PCT``) and the NO_DATA
    "engine producing fine" case both stay silent: a balanced histogram
    has no single fix and a healthy book has no problem to report (the
    ``_heartbeat_line`` HEALTHY suppression precedent — the summary
    must never become its own lying green light).
    """
    try:
        from .analytics.no_decision_reasons import (
            DEFAULT_WINDOW, build_no_decision_reasons,
        )
        nr = build_no_decision_reasons(
            store.recent_decisions(DEFAULT_WINDOW),
            window=DEFAULT_WINDOW,
        )
        if not isinstance(nr, dict) or nr.get("state") != "DOMINANT":
            return ""
        bucket = (nr.get("dominant_bucket") or "?").upper()
        headline = nr.get("headline") or ""
        rec = nr.get("recommendation") or ""
        if not headline:
            return ""
        lines = [f"**NO_DECISION CAUSE** ◈ {bucket}", f"> {headline}"]
        # The headline already folds in the recommendation when DOMINANT;
        # avoid duplicating it as a second bullet.
        if rec and rec not in headline:
            lines.append(f"> {rec}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[reporter] no-decision-reasons line skipped: {e}")
        return ""


def _decision_clock_line(store) -> str:
    """One-line "is there a recurring HOUR-OF-DAY where this trader is
    consistently being starved?" for the hourly / daily report.

    ``/api/decision-clock`` (commit ``513a1c1``) made the per-hour-of-day
    NO_DECISION distribution auditable on the *dashboard* — surfacing
    e.g. "hour 20:00 ET has 80% NO_DECISION over 5 samples" as a
    HOURLY_CONCENTRATION verdict. That signal is actionable: a single
    saturation window means out-of-band concurrent jobs (review agents
    / backtest committee) consistently fire that hour and the operator
    can schedule them differently. But the operator lives in Discord
    and never opens the dashboard panel — the exact dashboard→Discord
    gap ``_host_pulse_line`` / ``_capital_pulse_line`` /
    ``_position_attention_line`` each closed, one dimension over:
    aggregate-host (host-pulse) → per-position (attention) → per-hour
    (this line).

    Composes ``build_decision_clock`` **verbatim** (single source of
    truth, AGENTS.md invariant #10 — the verdict / headline are the
    builder's, never re-derived here, so this Discord line and the
    pure builder can never tell different stories) and feeds it the
    EXACT same store read the endpoint does
    (``recent_decisions(limit=20000)``). **Pure store reads only — NO
    network** (the Discord-path discipline; adds zero latency).
    Observational only, never gates, adds no caps (invariants #2/#12
    — the ``_position_attention_line`` precedent). Failure contract
    mirrors the rest of ``reporter``: any builder/store fault degrades
    to ``""`` ("no decision-clock line this report"), **never** an
    exception ("no Discord summary this report").

    Suppression — surface ONLY HOURLY_CONCENTRATION (the actionable
    verdict — there's a real concentrated saturation window). The
    other two verdicts (``EVEN_DISTRIBUTION`` and
    ``INSUFFICIENT_DATA``) say "nothing to act on" — silent so the
    summary doesn't turn into its own lying green light (the
    ``_position_attention_line`` OK / ``_heartbeat_line`` HEALTHY
    suppression precedent).
    """
    try:
        from .analytics.decision_clock import build_decision_clock
        decisions = store.recent_decisions(limit=20000)
        dc = build_decision_clock(decisions)
        if not isinstance(dc, dict):
            return ""
        if dc.get("verdict") != "HOURLY_CONCENTRATION":
            return ""
        headline = dc.get("headline") or ""
        if not headline:
            return ""
        return f"⚠️ **DECISION CLOCK** ◈ HOURLY_CONCENTRATION\n> {headline}"
    except Exception as e:
        print(f"[reporter] decision-clock line skipped: {e}")
        return ""


def _decision_weekday_line(store) -> str:
    """One-line "is there a recurring DAY-OF-WEEK where this trader is
    consistently being starved?" for the hourly / daily report.

    The day-of-week sibling to ``_decision_clock_line`` (hour-of-day). The
    builder ``decision_weekday.build_decision_weekday`` buckets the same
    decisions by NY local weekday and emits ``WEEKDAY_CONCENTRATION`` when
    a single weekday has ≥``WEEKDAY_CONCENTRATION_PCT`` NO_DECISION over
    ≥``MIN_WORST_BUCKET_SAMPLES`` samples — surfacing e.g. a
    Friday-after-close quota slump that the hour-of-day clock cannot see
    (the same hour on the off-day washes the bucket out). ``/api/decision-
    weekday`` exposes it on the dashboard; this routes the verdict to the
    Discord surface the operator actually reads (the exact dashboard→
    Discord gap ``_decision_clock_line`` / ``_position_attention_line``
    each closed, one dimension over: hour-of-day → day-of-week).

    Composes ``build_decision_weekday`` **verbatim** (single source of
    truth, AGENTS.md invariant #10 — the verdict / headline are the
    builder's, never re-derived here, so this Discord line and
    ``/api/decision-weekday`` can never tell different stories) and feeds
    it the EXACT same store read the endpoint does
    (``recent_decisions(limit=20000)``). **Pure store reads only — NO
    network** (the Discord-path discipline; adds zero latency).
    Observational only, never gates, adds no caps (invariants #2/#12 —
    the ``_decision_clock_line`` precedent). Failure contract mirrors
    the rest of ``reporter``: any builder/store fault degrades to ``""``
    ("no decision-weekday line this report"), **never** an exception
    ("no Discord summary this report").

    Suppression — surface ONLY ``WEEKDAY_CONCENTRATION`` (the actionable
    verdict — there's a real concentrated saturation weekday). The
    other two verdicts (``EVEN_DISTRIBUTION`` and ``INSUFFICIENT_DATA``)
    say "nothing to act on" — silent so the summary doesn't turn into
    its own lying green light (the ``_decision_clock_line``
    EVEN_DISTRIBUTION / ``_heartbeat_line`` HEALTHY suppression
    precedent).
    """
    try:
        from .analytics.decision_weekday import build_decision_weekday
        decisions = store.recent_decisions(limit=20000)
        dw = build_decision_weekday(decisions)
        if not isinstance(dw, dict):
            return ""
        if dw.get("verdict") != "WEEKDAY_CONCENTRATION":
            return ""
        headline = dw.get("headline") or ""
        if not headline:
            return ""
        return f"⚠️ **DECISION WEEKDAY** ◈ WEEKDAY_CONCENTRATION\n> {headline}"
    except Exception as e:
        print(f"[reporter] decision-weekday line skipped: {e}")
        return ""


def _repeat_loser_line(store) -> str:
    """One-line "which tickers have I been losing on repeatedly?" for the
    hourly / daily report.

    ``_streak_line`` flags an aggregate ``TILT_RISK`` on a ≥4-loss run but
    never names *which* tickers carried the losses. A trader on a 4-loss
    run whose losses are all on LITE has a very different actionable
    response — "stop trading LITE" — than one whose losses are spread
    across 4 names ("general tilt → step back"). ``/api/repeat-loser`` (and
    this Discord line) close that gap with per-ticker loser-cluster
    detection.

    Composes ``build_repeat_loser`` **verbatim** (single source of truth,
    AGENTS.md invariant #10 — the headline / verdict are the builder's,
    never re-derived here, so this Discord line and ``/api/repeat-loser``
    can never tell different stories) and feeds it the EXACT same store
    read the endpoint does (``recent_trades(2000)`` reversed oldest-
    first, the ``build_repeat_loser`` contract). **Pure store reads only —
    NO network** (the Discord-path discipline; the ``_streak_line``
    precedent). Observational only, never gates, adds no caps (invariants
    #2/#12 — the ``_streak_line`` precedent). Failure contract mirrors
    the rest of ``reporter``: any builder/store fault degrades to ``""``
    ("no repeat-loser line this report"), **never** an exception ("no
    Discord summary this report").

    Suppression — surface ONLY the actionable verdict, so a balanced book
    or insufficient sample adds no hourly noise (the summary must never
    become its own lying green light — the ``_streak_line`` suppression
    precedent):
      * ``REPEAT_LOSER`` (≥1 ticker on a ≥2-loss run) → ALWAYS surfaced.
      * ``OK`` / ``NO_DATA`` (and any non-verdict) → silent.
    """
    try:
        from .analytics.repeat_loser import build_repeat_loser
        # ``build_repeat_loser`` expects oldest → newest; the store hands
        # back newest-first (same reversal the ``_streak_line`` /
        # ``_realized_pl_today`` paths use).
        trades = list(reversed(store.recent_trades(2000)))
        rl = build_repeat_loser(trades)
        if not isinstance(rl, dict):
            return ""
        verdict = rl.get("verdict")
        if verdict != "REPEAT_LOSER":
            return ""
        headline = rl.get("headline") or ""
        if not headline:
            return ""
        return f"⚠️ **REPEAT_LOSER** ◈ per-ticker tilt\n> {headline}"
    except Exception as e:
        print(f"[reporter] repeat-loser line skipped: {e}")
        return ""


def _rebuy_regret_line(store) -> str:
    """One-line "did I sell low and buy back higher?" for the hourly / daily
    report — the $ cost of premature exits followed by re-entries.

    ``_repeat_loser_line`` names tickers the desk loses on repeatedly;
    ``_streak_line`` flags loss-clusters in time. Neither answers the
    discretionary trader's hardest exit question: **when I closed a name and
    re-bought it later, did I pay UP for the re-entry?** The disposition
    effect (cutting winners early; the desk's #1 documented pathology) shows
    up here as positive net regret — sold $220, re-bought $223. ``/api/rebuy-
    regret`` quantifies this on the dashboard; the operator lives in
    Discord and never opens it (the exact dashboard→Discord gap
    ``_capital_pulse_line`` / ``_streak_line`` / ``_repeat_loser_line``
    each closed, one dimension over: capital → time → name → price).

    Composes ``build_rebuy_regret`` **verbatim** (single source of truth,
    AGENTS.md invariant #10 — the headline / verdict are the builder's,
    never re-derived here, so this Discord line and ``/api/rebuy-regret``
    can never tell different stories) and feeds it the EXACT same store
    read the endpoint does (``recent_trades(2000)`` reversed oldest-first,
    the ``build_rebuy_regret`` contract). **Pure store reads only — NO
    network** (the Discord-path discipline; the ``_repeat_loser_line``
    precedent). Observational only, never gates, adds no caps (invariants
    #2/#12 — the ``_streak_line`` precedent). Failure contract mirrors
    the rest of ``reporter``: any builder/store fault degrades to ``""``
    ("no rebuy-regret line this report"), **never** an exception ("no
    Discord summary this report").

    Suppression — surface ONLY the actionable verdict, so a balanced or
    sample-poor book adds no hourly noise (the summary must never become
    its own lying green light — the ``_streak_line`` NEUTRAL /
    ``_repeat_loser_line`` OK suppression precedent):
      * ``REGRETTING`` (net positive regret above the noise floor) → ALWAYS
        surfaced — the disposition-effect / whipsaw pattern the desk loses
        money on.
      * ``SAVINGS`` / ``NET_NEUTRAL`` / ``NO_DATA`` / ``NO_REBUYS`` (and any
        non-verdict) → silent. SAVINGS is the GOOD case (sold high,
        re-bought lower) — the trader doesn't need an hourly alert to
        celebrate it; named in the daily-close behavioural stack via the
        existing ``_session_block`` realized-P/L line instead.
    """
    try:
        from .analytics.rebuy_regret import build_rebuy_regret
        # ``build_rebuy_regret`` is direction-tolerant but oldest→newest is
        # cheaper (no re-sort inside the builder) and matches the
        # ``_streak_line`` / ``_repeat_loser_line`` precedent.
        trades = list(reversed(store.recent_trades(2000)))
        rr = build_rebuy_regret(trades)
        if not isinstance(rr, dict):
            return ""
        if rr.get("verdict") != "REGRETTING":
            return ""
        headline = rr.get("headline") or ""
        if not headline:
            return ""
        return f"⚠️ **REBUY REGRET** ◈ sold low, bought higher\n> {headline}"
    except Exception as e:
        print(f"[reporter] rebuy-regret line skipped: {e}")
        return ""


def _streak_line(store) -> str:
    """One-line "am I on a hot-hand or a tilt-risk run right now?" for the
    hourly / daily report.

    ``/api/streak`` exposes the closed-round-trip streak structure
    (HOT_HAND when a STABLE book lands a ≥4-win run, TILT_RISK on a ≥4-loss
    run) on the *dashboard* — but the operator lives in Discord and never
    opens it (the exact dashboard→Discord gap ``_capital_pulse_line`` /
    ``_hold_discipline_line`` / ``_heartbeat_line`` each closed, one
    dimension over: structural risk → process health → behavioural run).
    A trader on a 4-loss run who keeps adding risk is exactly the
    classic loss-cluster tilt this verdict exists to flag, and a 4+
    win cluster is the overconfidence trap a desk reviews before adding
    more size — neither surfaced anywhere the operator reads today.

    Composes ``build_streak`` **verbatim** (single source of truth,
    AGENTS.md invariant #10 — the headline / verdict are the builder's,
    never re-derived here, so this Discord line and ``/api/streak`` can
    never tell different stories) and feeds it the EXACT same store
    read the endpoint does (``recent_trades(2000)`` reversed oldest-
    first, the ``build_streak`` contract). **Pure store reads only —
    NO network** (the Discord-path discipline; the
    ``_capital_pulse_line`` precedent). Observational only, never gates,
    adds no caps (invariants #2/#12 — the ``_hold_discipline_line``
    precedent). Failure contract mirrors the rest of ``reporter``: any
    builder/store fault degrades to ``""`` ("no streak line this
    report"), **never** an exception ("no Discord summary this report").

    Suppression — surface ONLY the two actionable verdicts a desk acts
    on, so a balanced book or insufficient sample adds no hourly noise
    (the summary must never become its own lying green light — the
    ``_hold_discipline_line`` NO_DATA / ``_decision_clock_line``
    EVEN_DISTRIBUTION suppression precedent):
      * ``HOT_HAND``   (≥4-win cluster) → ALWAYS surfaced — the
        overconfidence trap a desk reviews before adding size.
      * ``TILT_RISK``  (≥4-loss cluster) → ALWAYS surfaced — the
        loss-cluster tilt every PM steps back from.
      * ``NEUTRAL`` / ``EMERGING`` / ``NO_DATA`` (and any non-verdict)
        → silent (nothing actionable — the ``_hold_discipline_line``
        NO_DATA / ``_decision_clock_line`` EVEN_DISTRIBUTION
        precedent).
    """
    try:
        from .analytics.streak import build_streak
        # ``build_streak`` expects oldest → newest; store hands back
        # newest-first (the canonical store contract — same reversal
        # the ``_realized_pl_today`` / ``_trade_impact_line`` paths use).
        trades = list(reversed(store.recent_trades(2000)))
        sk = build_streak(trades)
        if not isinstance(sk, dict):
            return ""
        verdict = sk.get("verdict")
        if verdict not in ("HOT_HAND", "TILT_RISK"):
            return ""
        headline = sk.get("headline") or ""
        if not headline:
            return ""
        return f"**STREAK** ◈ {verdict}\n> {headline}"
    except Exception as e:
        print(f"[reporter] streak line skipped: {e}")
        return ""


def _ago(seconds: float) -> str:
    """Compact human age: `45m` / `3h` / `2d`. Sub-minute reads `0m`."""
    seconds = max(0.0, float(seconds))
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _countdown(seconds: float) -> str:
    """Compact "in Xh Ym" / "in Nm" / "in Nd Yh" countdown label. Always
    non-negative — a negative input clamps to 0m so a tiny clock-skew never
    renders a misleading "-12m"."""
    seconds = max(0.0, float(seconds))
    if seconds < 3600:
        return f"in {int(seconds // 60)}m"
    if seconds < 86400:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"in {h}h {m}m" if m else f"in {h}h"
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    return f"in {d}d {h}h" if h else f"in {d}d"


def _next_session_line(now: datetime | None = None) -> str:
    """One-line "when does the next NYSE session start?" for the hourly
    summary.

    A trader who lives in Discord checks the hourly at any hour — including
    weekends and overnight. The body currently shows positions + stale
    prices but no orientation cue: a 2 AM ET check looks identical to a
    10 AM ET one. This adds a single line so the operator can plan ("ok,
    next open is Monday 09:30 ET, ~37h away").

    Composes ``market.next_session_open`` verbatim (single source of truth —
    the NYSE calendar lives in market.py). Pure: zero I/O, never raises.
    Suppression: emit nothing when the market is currently OPEN (the
    hourly during the session already implies "we're trading"). NY clock
    explicit in the rendered string so the operator never has to mentally
    convert.

    Same additive failure contract as the rest of reporter: any fault
    degrades to ``""`` ("no next-session line this report"), never an
    exception ("no Discord summary this report"). ``now`` is injectable
    for deterministic tests (the ``_fmt_trade_stamp`` precedent).
    """
    try:
        n = now or datetime.now(timezone.utc)
        if market.is_market_open(n):
            return ""
        nxt = market.next_session_open(n)
        if nxt is None:
            return ""
        delta = (nxt - n).total_seconds()
        nxt_ny = nxt.astimezone(market.NY)
        when = nxt_ny.strftime("%a %m-%d 09:30 ET")
        return f"**MARKET** ◈ closed — next session: {when} ({_countdown(delta)})"
    except Exception as e:
        print(f"[reporter] next-session line skipped: {e}")
        return ""


def _session_close_countdown_line(now: datetime | None = None) -> str:
    """One-line "session closes in Xh Ym ET" for the hourly summary — the
    natural complement to ``_next_session_line``.

    ``_next_session_line`` fires when the market is *closed*; this fires
    when the market is *open*. Together they give the operator a consistent
    "when is the next state transition?" cue on every hourly report,
    whichever side of the bell they happen to be reading on.

    Why this matters for a live trader:
      * **Sizing decisions near the bell.** A 10-minute-to-close hourly
        check is materially different from a 3-hour-to-close one. The
        first should bias toward NOT opening new directional exposure
        (EOD pinning / liquidity / no follow-through window); the
        second is a normal session window. The trader can already infer
        this from the wall clock, but a hourly summary read on a phone
        out-of-office may not.
      * **Half-day awareness.** On NYSE early-close half-days
        (NYSE_HALF_DAYS_2026 — day after Thanksgiving, Christmas Eve)
        the bell rings at 13:00 ET, not 16:00. A trader who only checks
        a 12:00 ET hourly on a half-day would otherwise assume the
        standard 4h-to-close. This line names the *actual* close, not
        the conventional one, so a half-day is operationally legible.

    Composes ``market.next_session_close`` + ``market.seconds_until_close``
    **verbatim** (single source of truth — the NYSE calendar lives in
    market.py, the close minute resolution and the half-day handling are
    its responsibility). The clamping inside ``seconds_until_close`` (max
    0) keeps this line non-negative under a wall-clock step-back (the
    documented clock-skew hazard the runner-state sidecar already
    hardens against).

    Pure: zero I/O, never raises by construction. Suppression mirrors
    ``_next_session_line``'s complement contract — emit nothing when the
    market is closed (let ``_next_session_line`` carry the message on
    that side; never duplicate). Both can be silent in the tiny window
    *at* the bell where ``next_session_close`` is in the past but
    ``next_session_open`` is in the future — the same atomic transition
    moment ``_next_session_line`` is silent through; this preserves that
    invariant rather than racing it.

    Same additive failure contract as the rest of ``reporter``: any
    fault degrades to ``""`` ("no session-close countdown this report"),
    **never** an exception ("no Discord summary this report"). ``now``
    is injectable for deterministic tests (the ``_next_session_line``
    precedent).
    """
    try:
        n = now or datetime.now(timezone.utc)
        if not market.is_market_open(n):
            return ""
        close_dt = market.next_session_close(n)
        if close_dt is None:
            return ""
        secs = market.seconds_until_close(n)
        if secs is None:
            return ""
        close_ny = close_dt.astimezone(market.NY)
        # 16:00 ET on a regular day, 13:00 ET on a half-day — render the
        # actual bell so a half-day reads "closes at 13:00 ET" and the
        # operator can never mistake it for the conventional 16:00.
        when = close_ny.strftime("%H:%M ET")
        return (f"**MARKET** ◈ open — closes at {when} "
                f"({_countdown(secs)})")
    except Exception as e:
        print(f"[reporter] session-close countdown line skipped: {e}")
        return ""


def _today_session_anchor_iso(now: datetime | None = None) -> str | None:
    """ISO UTC timestamp of today's NYSE session open (09:30 ET) as a string,
    or ``None`` when today is not a trading day or the session has not yet
    opened.

    Pure NYSE calendar lookup — no I/O, never raises. Mirrors
    ``market.is_market_open``'s gating (weekend / NYSE_HOLIDAYS_2026 /
    pre-09:30 ET → no session yet). Returns the anchor even POST-close so a
    late-session or after-hours hourly summary still attributes today's
    motion to today's bell (the operator at 18:00 ET still asks "what did
    today do" — that question has an answer until UTC date rolls over to
    the next session, and the resolver below shifts naturally at the next
    pre-open).
    """
    n = (now or datetime.now(timezone.utc)).astimezone(market.NY)
    if n.weekday() >= 5 or n.date() in market.NYSE_HOLIDAYS_2026:
        return None
    cur_min = n.hour * 60 + n.minute
    if cur_min < 9 * 60 + 30:                 # pre-09:30 ET → no session yet
        return None
    open_dt = datetime(n.year, n.month, n.day, 9, 30, tzinfo=market.NY)
    return open_dt.astimezone(timezone.utc).isoformat()


def _today_session_line(store, now: datetime | None = None) -> str:
    """One-line "what has TODAY's session done to the book so far?" for the
    hourly summary.

    The hourly already shows TOTAL P/L from the ``$1000`` start and a 1h
    ``_session_block`` over the trailing-hour window. Neither answers the
    trader's first morning question after the 09:30 ET bell: "what is
    today's intraday motion so far?". For a book that gained ``$30``
    yesterday then lost ``$20`` today by 11 AM ET, the TOTAL P/L still
    reads ``+$10`` (positive, looks fine) while the 1h block only shows
    the latest hour. The today-anchor line closes the gap: a single line
    anchored to today's NYSE open (09:30 ET) so any hourly answers
    "today" without waiting until the 16:05 ET daily close.

    Composes ``_window_delta`` **verbatim** (single source of truth — the
    same math the ``_session_block`` portfolio Δ uses, just with a
    different baseline, so this surface and the 1h SESSION block can
    never disagree on direction). Pure store reads only — NO network (the
    Discord-path discipline; the ``_drawdown_line`` / ``_benchmark_line``
    precedent). Observational only, never gates, adds no caps (invariants
    #2/#12).

    Suppression — silence-when-nothing-actionable (the ``_next_session_line``
    closed-only precedent): today is not a trading day, the session has
    not yet opened today, or the equity_curve has no point at-or-after
    today's 09:30 ET anchor (a runner that booted post-open with no
    pre-anchor history — the today motion is undefined).

    Same additive failure contract as the rest of ``reporter``: any
    builder/store fault degrades to ``""`` ("no today-session line this
    report"), **never** an exception ("no Discord summary this report").
    ``now`` is injectable for deterministic tests (the ``_next_session_line``
    precedent).
    """
    try:
        since = _today_session_anchor_iso(now)
        if not since:
            return ""
        eq = store.equity_curve(limit=5000)
        if not eq or len(eq) < 2:
            return ""
        last = eq[-1]
        base = next((p for p in eq if (p.get("timestamp") or "") >= since),
                    None)
        if base is None or base is last:
            return ""
        d = _window_delta(eq, since)
        if not d or "port_pct" not in d:
            return ""
        try:
            b_tv = float(base.get("total_value") or 0.0)
            l_tv = float(last.get("total_value") or 0.0)
        except (TypeError, ValueError):
            return ""
        port_abs = l_tv - b_tv
        seg = f"${port_abs:+.2f} ({d['port_pct']:+.2f}%)"
        if "alpha_pct" in d:
            seg += f" · alpha `{d['alpha_pct']:+.2f}%`"
        return ("**TODAY** ◈ since 09:30 ET NYSE open\n"
                f"> {seg}")
    except Exception as e:
        print(f"[reporter] today-session line skipped: {e}")
        return ""


def _fmt_trade_stamp(ts_iso: str | None, now: datetime | None = None) -> str:
    """Bracket label for a recent-trade line in the hourly summary.

    The block historically showed only `HH:MM` (UTC) with no date. The
    desk's #1 documented pathology is a book that freezes for many hours
    while still *looking* active — a 25h-old "BUY MU" rendered as `[09:38]`
    is read as today's fill. This makes staleness unmissable at a glance:

      * trade is on today's UTC date → ``HH:MM``                (unchanged)
      * older                        → ``MM-DD HH:MM · Nd ago``

    Pure; ``now`` injectable for tests. Any parse failure degrades to the
    original ``ts[11:16]`` slice (never raises — the reporter additive
    contract: a bad field drops detail from one line, never the report)."""
    raw = (ts_iso or "")
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        # store always writes datetime.now(utc).isoformat(), so a parse
        # failure means genuinely corrupt data — a clean sentinel beats the
        # old raw[11:16] slice (which rendered garbage like "tamp").
        return "??:??"
    now = now or datetime.now(timezone.utc)
    dt_u = dt.astimezone(timezone.utc)
    hm = dt_u.strftime("%H:%M")
    if dt_u.date() == now.astimezone(timezone.utc).date():
        return hm
    stamp = f"{dt_u.strftime('%m-%d')} {hm}"
    delta = (now - dt_u).total_seconds()
    return f"{stamp} · {_ago(delta)} ago" if delta > 0 else stamp


def send_hourly_summary() -> bool:
    store = get_store()
    pf = store.get_portfolio()
    # ``store.open_positions()`` rows lack ``stale_mark`` (no such column in
    # the positions TABLE — it's an in-memory enrichment from
    # ``strategy._mark_to_market``). Merge it in from the marked snapshot in
    # ``portfolio.positions_json`` so the ⚠ STALE annotation in
    # ``_portfolio_lines`` and the P/L%/alpha suppression in
    # ``_pos_pct_weight``/``_pos_alpha_token`` can actually fire.
    positions = _merge_stale_marks(store.open_positions(), pf.get("positions"))
    sp = market.benchmark_sp500()
    pl = pf["total_value"] - _INITIAL_EQUITY
    pl_pct = pl / _INITIAL_EQUITY * 100

    recent_trades = store.recent_trades(5)
    trade_lines = [
        f"  [{_fmt_trade_stamp(t['timestamp'])}] {t['action']} {t['qty']} {t['ticker']} @ ${t['price']:.2f}"
        for t in recent_trades
    ] or ["  (no trades yet)"]

    sp_line = f"S&P 500: {sp:.2f}" if sp else "S&P 500: N/A"
    events_by_ticker = _earnings_events_by_ticker()
    # Equity curve in ascending order — read once per report so the
    # per-position α token can pick the entry-time SPY baseline without a
    # network hop. Bounded read so a deep history doesn't bloat the path.
    try:
        equity_asc = store.equity_curve(limit=5000)
    except Exception as e:
        print(f"[reporter] equity_curve read for alpha skipped: {e}")
        equity_asc = []

    body = (
        f"**HOURLY** ◈ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"```\n"
        f"Equity      ${pf['total_value']:.2f}\n"
        f"Cash        ${pf['cash']:.2f}\n"
        f"P/L         ${pl:+.2f} ({pl_pct:+.2f}%)\n"
        f"{sp_line}\n"
        f"```\n"
        f"**Positions**\n```\n"
        + ("\n".join(_portfolio_lines(positions, pf["total_value"],
                                       events_by_ticker=events_by_ticker,
                                       equity_asc=equity_asc,
                                       sp500_now=sp)) or "  (none)")
        + "\n```\n**Recent trades**\n```\n"
        + "\n".join(trade_lines)
        + "\n```"
    )
    ns = _next_session_line()
    if ns:
        body += "\n" + ns
    # Complement to _next_session_line: when market is OPEN, surface the
    # countdown to today's close (16:00 ET regular, 13:00 ET on a half-day).
    # The two lines are mutually exclusive — at most one fires per report.
    sc = _session_close_countdown_line()
    if sc:
        body += "\n" + sc
    lk = _singleton_lock_line()
    if lk:
        body += "\n" + lk
    sv = _supervision_line()
    if sv:
        body += "\n" + sv
    hb = _heartbeat_line(store)
    if hb:
        body += "\n" + hb
    ei = _equity_integrity_line(store)
    if ei:
        body += "\n" + ei
    ef = _equity_freshness_line(store)
    if ef:
        body += "\n" + ef
    sx = _session_block(store, 1.0, "1h")
    if sx:
        body += "\n" + sx
    # TODAY sits right AFTER the 1h SESSION block — the two answer the same
    # "what has the desk done lately" question one dimension over (last 1h vs
    # since today's 09:30 ET open). Both are silence-by-default on different
    # axes (SESSION needs >=2 in-window equity points; TODAY needs the market
    # to have opened today) so neither suppresses the other.
    tsl = _today_session_line(store)
    if tsl:
        body += "\n" + tsl
    mx = _benchmark_line(store)
    if mx:
        body += "\n" + mx
    dd = _drawdown_line(store)
    if dd:
        body += "\n" + dd
    # BANKED-vs-PAPER sits right AFTER DRAWDOWN — both are P&L-shape
    # surfaces. DRAWDOWN says "how deep is the hole vs your own high".
    # BANKED-vs-PAPER says "of today's net P&L, how much is locked-in
    # vs evaporable paper" — the orthogonal give-back / paper-heavy / leak
    # diagnostic. Both can be silent independently (DRAWDOWN suppresses on
    # at-high-water; this one suppresses on BANKED/BALANCED/NO_DATA);
    # neither suppresses the other.
    rvu = _realized_vs_unrealized_line(store)
    if rvu:
        body += "\n" + rvu
    bx = _behavioural_block()
    if bx:
        body += "\n" + bx
    stx = _stress_line(store)
    if stx:
        body += "\n" + stx
    esx = _earnings_shock_line(store)
    if esx:
        body += "\n" + esx
    rcx = _recovery_line(store)
    if rcx:
        body += "\n" + rcx
    hp = _host_pulse_line()
    if hp:
        body += "\n" + hp
    # IDLE sits right AFTER HOST — host pulse names the CAUSE (saturated /
    # starved); idle-opportunity names what was MISSED while the cause held.
    # Both can be silent independently (silence-when-nothing-actionable);
    # neither suppresses the other, so a future drought without a high-score
    # miss reports HOST alone and a future drought on a clear host reports
    # IDLE alone — the same independence as HOST/CAPITAL.
    iox = _idle_opportunity_line(store)
    if iox:
        body += "\n" + iox
    # News-breadth warning — fires ONLY on ECHO (a SURGING velocity that
    # is actually one wire mirrored across many feeds). Sits right after
    # IDLE because both read articles.db and both are silence-by-default;
    # an ECHO warning on a held name catches the operator about to chase
    # a false-signal velocity spike.
    smx = _source_mix_line(store)
    if smx:
        body += "\n" + smx
    cp = _capital_pulse_line(store)
    if cp:
        body += "\n" + cp
    # CASH FIT sits right after CAPITAL — both about cash deployment, one
    # dimension over. CAPITAL says "the desk *cannot* act" (structural —
    # PINNED / FREE / BLEEDING from cash %, recent activity, droughts).
    # CASH FIT says "the desk *can* act but isn't sized to the live signal"
    # (point-in-time — IDLE_DESPITE_SURGE / OVERDEPLOYED against today's
    # loudest screaming name). Both can be silent independently (silence-
    # when-nothing-actionable); neither suppresses the other (a FREE book
    # can still be IDLE_DESPITE_SURGE on an unheld signal — the live state
    # this surface exists to catch).
    cf = _cash_conviction_fit_line(store)
    if cf:
        body += "\n" + cf
    cn = _concentration_line(store)
    if cn:
        body += "\n" + cn
    pa = _position_attention_line(store)
    if pa:
        body += "\n" + pa
    dc = _decision_clock_line(store)
    if dc:
        body += "\n" + dc
    # DECISION_WEEKDAY sits right after the hour-of-day clock — the
    # orthogonal day-of-week sibling. Both are silence-by-default; a
    # recurring Friday-after-close quota slump appears here but is
    # washed out of decision_clock by Fridays-only data.
    dw = _decision_weekday_line(store)
    if dw:
        body += "\n" + dw
    ndr = _no_decision_reasons_line(store)
    if ndr:
        body += "\n" + ndr
    # STREAK sits last among the behavioural blocks — surfaces HOT_HAND
    # (overconfidence trap) or TILT_RISK (loss-cluster bias) from
    # closed round-trip outcomes. Silent on a balanced/young book (the
    # _decision_clock / _hold_discipline suppression precedent — the
    # summary must never become its own lying green light).
    sk = _streak_line(store)
    if sk:
        body += "\n" + sk
    # REPEAT_LOSER sits right after STREAK — the per-ticker companion to
    # the aggregate run. STREAK says "you're on a 4-loss run"; REPEAT_LOSER
    # says "and 3 of those 4 are on LITE — stop adding to LITE". Both
    # surface independently (a tilt aggregate need not be one ticker, and
    # one repeat-loser need not pull the aggregate to TILT_RISK).
    rl = _repeat_loser_line(store)
    if rl:
        body += "\n" + rl
    # REBUY_REGRET sits right after REPEAT_LOSER — the same dimension (per-
    # name loss pattern) one degree sharper: not "I keep losing on LITE"
    # but "I sold LITE at $X then re-bought at $X+Δ — the *exit timing* is
    # costing me, separate from the trade picks themselves". Silent on
    # SAVINGS / NET_NEUTRAL (the silence precedent — never become a lying
    # green light), so this only fires on the actionable disposition-
    # effect / whipsaw pattern.
    rr = _rebuy_regret_line(store)
    if rr:
        body += "\n" + rr
    return _send(body)


def send_daily_close() -> bool:
    store = get_store()
    pf = store.get_portfolio()
    # See send_hourly_summary — merge stale_mark from positions_json so the
    # ⚠ STALE annotation can fire on the daily close too.
    positions = _merge_stale_marks(store.open_positions(), pf.get("positions"))
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
    events_by_ticker = _earnings_events_by_ticker()
    # Per-position α-vs-SPY since entry — same Discord-path discipline as
    # the hourly: read the equity curve once and pass it down.
    try:
        equity_asc = store.equity_curve(limit=5000)
    except Exception as e:
        print(f"[reporter] equity_curve read for alpha skipped: {e}")
        equity_asc = []

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
        + ("\n".join(_portfolio_lines(positions, pf["total_value"],
                                       events_by_ticker=events_by_ticker,
                                       equity_asc=equity_asc,
                                       sp500_now=sp)) or "  (none)")
        + "\n```"
    )
    lk = _singleton_lock_line()
    if lk:
        body += "\n" + lk
    sv = _supervision_line()
    if sv:
        body += "\n" + sv
    hb = _heartbeat_line(store)
    if hb:
        body += "\n" + hb
    ei = _equity_integrity_line(store)
    if ei:
        body += "\n" + ei
    ef = _equity_freshness_line(store)
    if ef:
        body += "\n" + ef
    sx = _session_block(store, 24.0, "24h")
    if sx:
        body += "\n" + sx
    mx = _benchmark_line(store)
    if mx:
        body += "\n" + mx
    dd = _drawdown_line(store)
    if dd:
        body += "\n" + dd
    # BANKED-vs-PAPER follows DRAWDOWN on daily close too — see
    # send_hourly_summary for the placement rationale (peak-to-trough then
    # banked-vs-paper, both P&L-shape surfaces with independent silence).
    rvu = _realized_vs_unrealized_line(store)
    if rvu:
        body += "\n" + rvu
    bx = _behavioural_block()
    if bx:
        body += "\n" + bx
    hx = _hold_discipline_line(store)
    if hx:
        body += "\n" + hx
    stx = _stress_line(store)
    if stx:
        body += "\n" + stx
    esx = _earnings_shock_line(store)
    if esx:
        body += "\n" + esx
    rcx = _recovery_line(store)
    if rcx:
        body += "\n" + rcx
    hp = _host_pulse_line()
    if hp:
        body += "\n" + hp
    # IDLE sits right AFTER HOST — host pulse names the CAUSE (saturated /
    # starved); idle-opportunity names what was MISSED while the cause held.
    # Both can be silent independently (silence-when-nothing-actionable);
    # neither suppresses the other, so a future drought without a high-score
    # miss reports HOST alone and a future drought on a clear host reports
    # IDLE alone — the same independence as HOST/CAPITAL.
    iox = _idle_opportunity_line(store)
    if iox:
        body += "\n" + iox
    smx = _source_mix_line(store)
    if smx:
        body += "\n" + smx
    cp = _capital_pulse_line(store)
    if cp:
        body += "\n" + cp
    # CASH FIT follows CAPITAL on daily close too — see send_hourly_summary
    # for the CAPITAL→CASH-FIT placement rationale (can the desk act? vs.
    # is it sized to the live signal?).
    cf = _cash_conviction_fit_line(store)
    if cf:
        body += "\n" + cf
    cn = _concentration_line(store)
    if cn:
        body += "\n" + cn
    pa = _position_attention_line(store)
    if pa:
        body += "\n" + pa
    dc = _decision_clock_line(store)
    if dc:
        body += "\n" + dc
    # DECISION_WEEKDAY follows the hour-of-day clock on daily close too —
    # see send_hourly_summary for the rationale (hour vs day-of-week).
    dw = _decision_weekday_line(store)
    if dw:
        body += "\n" + dw
    ndr = _no_decision_reasons_line(store)
    if ndr:
        body += "\n" + ndr
    # See send_hourly_summary STREAK rationale — daily close mirrors the
    # block placement so the operator sees the same surface on both reports.
    sk = _streak_line(store)
    if sk:
        body += "\n" + sk
    # REPEAT_LOSER follows STREAK on daily close too — see send_hourly_summary
    # for the rationale (aggregate vs per-ticker tilt).
    rl = _repeat_loser_line(store)
    if rl:
        body += "\n" + rl
    # REBUY_REGRET follows REPEAT_LOSER on daily close too — see
    # send_hourly_summary for the rationale (sold low, re-bought higher).
    rr = _rebuy_regret_line(store)
    if rr:
        body += "\n" + rr
    return _send(body)
