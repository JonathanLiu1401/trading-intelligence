"""Main loop — drives the paper trader, runs the dashboard, dispatches Discord reports."""
from __future__ import annotations

import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

from . import market, reporter, strategy
from .store import get_store

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
    """Kill any lingering claude subprocess. strategy._claude_call() launches
    `claude --model <model> --print ...`; a wedged child that survives its
    Python-side timeout keeps holding resources and can re-starve the next
    cycle. Match the live (Opus) model first, then fall back to any stray
    `claude --print` so a fallback-model zombie is also cleaned up."""
    for pattern in ("claude --model claude-opus", "claude --print"):
        try:
            killed = subprocess.run(
                ["pkill", "-f", pattern],
                capture_output=True,
            )
            # pkill rc: 0 = killed something, 1 = nothing matched, >1 = error.
            print(
                f"[runner] circuit breaker: pkill -f {pattern!r} "
                f"returned {killed.returncode}"
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
    # Anchor the hourly clock to boot so the first summary lands ~1h in,
    # rather than firing immediately alongside the online ping.
    _last_hourly = datetime.now(timezone.utc)
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
