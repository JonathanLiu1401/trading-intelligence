"""Main loop — drives the paper trader, runs the dashboard, dispatches Discord reports."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

from . import market, reporter, strategy
from .store import DB_PATH, get_store

NY = ZoneInfo("America/New_York")
OPEN_INTERVAL_S = 1800      # decide every 30 min when market is open
CLOSED_INTERVAL_S = 3600    # every 1 hour when closed
DAILY_CLOSE_HOUR_NY = 16    # report after 16:00 NY

# Auto-recovery circuit breaker: after this many consecutive cycles that
# produced no decision (Opus + Sonnet fallback both timed out / unparseable),
# kill any lingering claude subprocess so a wedged CLI can't keep starving the
# decision loop. strategy.decide() already returns status="NO_DECISION" for
# every failed cycle, so we key off summary["status"] (not decision["action"]).
CONSECUTIVE_NO_DECISION_LIMIT = 5


_daily_close_sent_for: str | None = None
_last_hourly: datetime | None = None
_consecutive_no_decisions = 0


# ── Restart-durable report markers ───────────────────────────────────────
# `_daily_close_sent_for` / `_last_hourly` are module globals — lost on every
# process restart, and the runner restarts often (a `/api/build-info` stale
# restart to apply a committed fix, systemd, the circuit breaker, an operator
# bounce). Two trader-visible failures result:
#
#   1. Hourly STARVATION. `main()` deliberately anchors `_last_hourly` to boot
#      so the first summary lands ~1h in (not alongside the online ping). But
#      a runner that bounces more often than hourly *never* sends an hourly
#      summary at all — every boot resets the 1h clock. ("I haven't gotten an
#      hourly in hours.")
#   2. Daily-close DUPLICATION. `_daily_close_sent_for` resets to None on
#      restart, so a bounce after 16:05 NY on a day the close already fired
#      re-posts a second "DAILY CLOSE" with the same numbers.
#
# A tiny JSON sidecar next to paper_trader.db makes both markers survive a
# restart. Deliberately NOT a new store.py table: SCHEMA is load-bearing
# (invariant #13) and this needs no WAL/locking — it is single-writer,
# best-effort, and a lost/corrupt file must degrade to "behave like today
# (in-memory only)", never crash the daemon loop.
_STATE_PATH = DB_PATH.parent / "runner_state.json"


def _load_runner_state() -> dict:
    """Best-effort read of the persisted report markers. Returns {} on a
    missing/corrupt/unreadable file — never raises (the daemon must boot
    even if the sidecar is garbage)."""
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_runner_state() -> None:
    """Atomically persist the current markers. Atomic (tmp + os.replace) so a
    kill mid-write — the circuit breaker / systemd / SIGKILL — can never leave
    a torn JSON that then reads back as {} and re-arms the duplicate-close
    bug. Best-effort: any IO error is swallowed (a read-only data dir must not
    take down the trade loop)."""
    payload = {
        "daily_close_sent_for": _daily_close_sent_for,
        "last_hourly_iso": (_last_hourly.isoformat()
                            if _last_hourly is not None else None),
    }
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{_STATE_PATH}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, _STATE_PATH)
    except Exception as e:
        print(f"[runner] could not persist runner state: {e}")


def _restore_runner_state() -> None:
    """Rehydrate `_daily_close_sent_for` / `_last_hourly` from the sidecar at
    boot. Called by `main()` *before* the loop. No persisted marker → leave the
    global as-is (fresh-boot behaviour: the daily flag stays None; `main()`
    still anchors `_last_hourly` to boot so a first-ever start doesn't fire an
    hourly immediately). A persisted `last_hourly` that is already >1h old
    correctly lets the first cycle send the overdue summary instead of
    swallowing another hour."""
    global _daily_close_sent_for, _last_hourly
    st = _load_runner_state()
    dcs = st.get("daily_close_sent_for")
    if isinstance(dcs, str) and dcs:
        _daily_close_sent_for = dcs
    lh = st.get("last_hourly_iso")
    if isinstance(lh, str) and lh:
        try:
            dt = datetime.fromisoformat(lh)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            _last_hourly = dt
        except ValueError:
            pass


def _maybe_hourly():
    global _last_hourly
    now = datetime.now(timezone.utc)
    if _last_hourly is not None and (now - _last_hourly).total_seconds() < 3600:
        return
    try:
        # Only advance _last_hourly on success — a transient openclaw failure
        # then retries next cycle instead of silently skipping the hour.
        if reporter.send_hourly_summary():
            _last_hourly = now
            _save_runner_state()  # survive a restart — don't starve the hour
        else:
            print("[runner] hourly send returned False, will retry next cycle")
    except Exception as e:
        print(f"[runner] hourly send failed: {e}")


def _maybe_daily_close():
    global _daily_close_sent_for
    now_ny = datetime.now(timezone.utc).astimezone(NY)
    today = now_ny.date().isoformat()
    if now_ny.weekday() >= 5:
        return  # no weekend close report
    if now_ny.date() in market.NYSE_HOLIDAYS_2026:
        # A full-holiday close is not a trading day — emitting a "DAILY CLOSE"
        # with stale marks and "Trades today 0" is misleading noise. Mirror the
        # weekend guard: bail before touching _daily_close_sent_for so the flag
        # stays where it was.
        return
    if now_ny.hour < DAILY_CLOSE_HOUR_NY or (now_ny.hour == DAILY_CLOSE_HOUR_NY and now_ny.minute < 5):
        return
    if _daily_close_sent_for == today:
        return
    try:
        # Only mark as sent on actual success — _send returns False (no exception)
        # when openclaw is missing or fails; otherwise a transient failure would
        # permanently suppress today's close.
        if reporter.send_daily_close():
            _daily_close_sent_for = today
            _save_runner_state()  # survive a post-16:05 restart — no dup close
        else:
            print("[runner] daily close: send returned False, will retry next cycle")
    except Exception as e:
        print(f"[runner] daily close failed: {e}")


def _start_dashboard():
    try:
        from . import dashboard
        threading.Thread(target=dashboard.run, daemon=True, name="dashboard").start()
        print("[runner] dashboard thread started on :8090")
    except Exception as e:
        print(f"[runner] dashboard disabled: {e}")


def _kill_stale_claude():
    """Kill the runner's *own* lingering claude subprocess. strategy._claude_call()
    launches `claude --model <model> --print ...` as a direct child of this
    process; a wedged child that survives its Python-side timeout keeps holding
    resources and can re-starve the next cycle. Match the live (Opus) model
    first, then the Sonnet fallback model so a wedged fallback zombie is also
    reaped.

    Both patterns are anchored on `claude --model <family>` because the CLI
    is always invoked as `claude --model <model> --print …` — the `--model`
    arg sits between `claude` and `--print`, so a bare `claude --print`
    pattern is never a contiguous substring of the real command line and
    would silently match nothing (the exact bug this once had: a wedged
    Sonnet fallback survived the breaker, defeating auto-recovery in the
    very Opus-timeout→Sonnet-fallback path the breaker exists for).

    SCOPED TO OUR OWN CHILDREN (`pkill -P os.getpid()`). A bare host-wide
    `pkill -f "claude --model claude-opus"` is catastrophic collateral
    damage: this box also runs the hourly self-review agents
    (`scripts/hourly_review.sh` spawns 3× `claude --model claude-opus-4-7`),
    sibling automated-review agents, and possibly an operator's interactive
    `claude` session — all of which match the same pattern. A wedged trader
    recovering by SIGTERM-ing every Claude process on the machine (including
    the review agents that keep the system healthy, and the agent that may
    have just deployed a fix) is a cure far worse than the disease. The
    decision subprocess is always a *direct* child of the runner, so `-P`
    restricts the sweep to exactly the processes this breaker is meant to
    reap and nothing else."""
    own_pid = os.getpid()
    for pattern in ("claude --model claude-opus", "claude --model claude-sonnet"):
        try:
            killed = subprocess.run(
                ["pkill", "-P", str(own_pid), "-f", pattern],
                capture_output=True,
            )
            # pkill rc: 0 = killed something, 1 = nothing matched, >1 = error.
            print(
                f"[runner] circuit breaker: pkill -P {own_pid} -f "
                f"{pattern!r} returned {killed.returncode}"
            )
        except Exception as e:
            print(f"[runner] circuit breaker: pkill {pattern!r} failed: {e}")


def _cycle():
    global _consecutive_no_decisions
    summary = strategy.decide()

    # Auto-recovery circuit breaker. strategy.decide() returns a dict whose
    # top-level "status" is "NO_DECISION" whenever Claude failed (timeout /
    # empty / unparseable, after the Sonnet fallback and JSON-only retry).
    # A genuine HOLD comes back as status="HOLD", not "NO_DECISION", so a
    # quiet market does NOT trip the breaker — only repeated Claude failures.
    status = summary.get("status", "NO_DECISION")
    if status == "NO_DECISION":
        _consecutive_no_decisions += 1
        if _consecutive_no_decisions >= CONSECUTIVE_NO_DECISION_LIMIT:
            print(
                f"[runner] WARNING: {_consecutive_no_decisions} consecutive "
                f"NO_DECISION cycles — Claude appears wedged; killing stale "
                f"claude processes to auto-recover"
            )
            _kill_stale_claude()
            _consecutive_no_decisions = 0  # reset after intervention
    else:
        _consecutive_no_decisions = 0

    try:
        if summary.get("auto_exits") or summary.get("status") == "FILLED":
            # post the trade that was just executed
            trades = get_store().recent_trades(1)
            if trades and summary.get("status") == "FILLED":
                reporter.send_trade_alert(trades[0])
            for ax in summary.get("auto_exits") or []:
                reporter._send(f"**AUTO RISK EXIT** `{ax}`")
        if summary.get("status") == "FILLED":
            reporter.send_decision_log(summary)
    except Exception as e:
        print(f"[runner] report failed: {e}")


def main():
    global _last_hourly
    print("[runner] starting paper trader")
    store = get_store()
    pf = store.get_portfolio()
    print(f"[runner] portfolio: cash=${pf['cash']:.2f} total=${pf['total_value']:.2f}")
    # Anchor the hourly clock to boot so a *first-ever* start's first summary
    # lands ~1h in, rather than firing immediately alongside the online ping.
    _last_hourly = datetime.now(timezone.utc)
    # Then rehydrate from the sidecar: on a *restart* this restores the real
    # last-hourly instant (an overdue summary fires this cycle instead of the
    # boot-anchor swallowing yet another hour) and the daily-close-sent date
    # (a post-16:05 bounce won't re-post the close). No sidecar → globals keep
    # their fresh-boot values, behaviour identical to before this change.
    _restore_runner_state()
    _start_dashboard()
    try:
        reporter._send("**PAPER TRADER ONLINE** ◈ engine booted, decision loop starting")
    except Exception:
        pass

    while True:
        try:
            _cycle()
        except Exception:
            print("[runner] cycle exception:")
            traceback.print_exc()

        _maybe_hourly()
        _maybe_daily_close()

        market_open = market.is_market_open()
        sleep_s = OPEN_INTERVAL_S if market_open else CLOSED_INTERVAL_S
        print(f"[runner] sleeping {sleep_s}s (market_open={market_open})")
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
