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
        + ("\n".join(_portfolio_lines(positions)) or "  (none)")
        + "\n```\n**Recent trades**\n```\n"
        + "\n".join(trade_lines)
        + "\n```"
    )
    lk = _singleton_lock_line()
    if lk:
        body += "\n" + lk
    hb = _heartbeat_line(store)
    if hb:
        body += "\n" + hb
    sx = _session_block(store, 1.0, "1h")
    if sx:
        body += "\n" + sx
    mx = _benchmark_line(store)
    if mx:
        body += "\n" + mx
    bx = _behavioural_block()
    if bx:
        body += "\n" + bx
    cp = _capital_pulse_line(store)
    if cp:
        body += "\n" + cp
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
    lk = _singleton_lock_line()
    if lk:
        body += "\n" + lk
    hb = _heartbeat_line(store)
    if hb:
        body += "\n" + hb
    sx = _session_block(store, 24.0, "24h")
    if sx:
        body += "\n" + sx
    mx = _benchmark_line(store)
    if mx:
        body += "\n" + mx
    bx = _behavioural_block()
    if bx:
        body += "\n" + bx
    hx = _hold_discipline_line(store)
    if hx:
        body += "\n" + hx
    cp = _capital_pulse_line(store)
    if cp:
        body += "\n" + cp
    return _send(body)
