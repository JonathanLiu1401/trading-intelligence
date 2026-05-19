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

    if (not p.get("stale_mark") and avg is not None and avg > 0
            and cur is not None and cur > 0):
        parts.append(f"{(cur - avg) / avg * 100.0:+.1f}%")

    tv = _num(total_value)
    if (tv is not None and tv > 0 and cur is not None and cur > 0
            and qty is not None):
        mv = cur * qty * (100.0 if is_opt else 1.0)
        w = mv / tv * 100.0
        parts.append(f"{w:.0f}% bk" if w >= 1.0 else f"{w:.1f}% bk")

    return f"  ({' · '.join(parts)})" if parts else ""


def _portfolio_lines(positions: list[dict],
                     total_value: float | None = None) -> list[str]:
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
        if p["type"] in ("call", "put"):
            lines.append(
                f"  {p['ticker']} {p['type'].upper()}{p['strike']} {p['expiry']}  "
                f"qty {p['qty']}  P/L ${(p.get('unrealized_pl') or 0):+.2f}{pw}{stale}"
            )
        else:
            lines.append(
                f"  {p['ticker']:<6} qty {p['qty']:<8} avg ${p['avg_cost']:.2f} "
                f"now ${(p.get('current_price') or 0):.2f}  P/L ${(p.get('unrealized_pl') or 0):+.2f}{pw}{stale}"
            )
    return lines


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


def _ago(seconds: float) -> str:
    """Compact human age: `45m` / `3h` / `2d`. Sub-minute reads `0m`."""
    seconds = max(0.0, float(seconds))
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


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
    positions = store.open_positions()
    sp = market.benchmark_sp500()
    pl = pf["total_value"] - _INITIAL_EQUITY
    pl_pct = pl / _INITIAL_EQUITY * 100

    recent_trades = store.recent_trades(5)
    trade_lines = [
        f"  [{_fmt_trade_stamp(t['timestamp'])}] {t['action']} {t['qty']} {t['ticker']} @ ${t['price']:.2f}"
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
        + ("\n".join(_portfolio_lines(positions, pf["total_value"])) or "  (none)")
        + "\n```\n**Recent trades**\n```\n"
        + "\n".join(trade_lines)
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
    sx = _session_block(store, 1.0, "1h")
    if sx:
        body += "\n" + sx
    mx = _benchmark_line(store)
    if mx:
        body += "\n" + mx
    dd = _drawdown_line(store)
    if dd:
        body += "\n" + dd
    bx = _behavioural_block()
    if bx:
        body += "\n" + bx
    stx = _stress_line(store)
    if stx:
        body += "\n" + stx
    rcx = _recovery_line(store)
    if rcx:
        body += "\n" + rcx
    hp = _host_pulse_line()
    if hp:
        body += "\n" + hp
    cp = _capital_pulse_line(store)
    if cp:
        body += "\n" + cp
    pa = _position_attention_line(store)
    if pa:
        body += "\n" + pa
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
        + ("\n".join(_portfolio_lines(positions, pf["total_value"])) or "  (none)")
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
    bx = _behavioural_block()
    if bx:
        body += "\n" + bx
    hx = _hold_discipline_line(store)
    if hx:
        body += "\n" + hx
    stx = _stress_line(store)
    if stx:
        body += "\n" + stx
    rcx = _recovery_line(store)
    if rcx:
        body += "\n" + rcx
    hp = _host_pulse_line()
    if hp:
        body += "\n" + hp
    cp = _capital_pulse_line(store)
    if cp:
        body += "\n" + cp
    pa = _position_attention_line(store)
    if pa:
        body += "\n" + pa
    return _send(body)
