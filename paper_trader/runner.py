"""Main loop — drives the paper trader, runs the dashboard, dispatches Discord reports."""
from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

from . import market, reporter, strategy
from .store import get_store

NY = ZoneInfo("America/New_York")
OPEN_INTERVAL_S = 60        # decide every 1 min when market is open
CLOSED_INTERVAL_S = 3600    # every 1 hour when closed
HOURLY_REPORT_S = 3600      # send hourly summary every hour
DAILY_CLOSE_HOUR_NY = 16    # report after 16:00 NY


_last_hourly = 0.0
_daily_close_sent_for: str | None = None


def _maybe_hourly(now_ts: float):
    global _last_hourly
    if now_ts - _last_hourly >= HOURLY_REPORT_S:
        try:
            reporter.send_hourly_summary()
        except Exception as e:
            print(f"[runner] hourly send failed: {e}")
        _last_hourly = now_ts


def _maybe_daily_close():
    global _daily_close_sent_for
    now_ny = datetime.now(timezone.utc).astimezone(NY)
    today = now_ny.date().isoformat()
    if now_ny.weekday() >= 5:
        return  # no weekend close report
    if now_ny.hour < DAILY_CLOSE_HOUR_NY or (now_ny.hour == DAILY_CLOSE_HOUR_NY and now_ny.minute < 5):
        return
    if _daily_close_sent_for == today:
        return
    try:
        reporter.send_daily_close()
        _daily_close_sent_for = today
    except Exception as e:
        print(f"[runner] daily close failed: {e}")


def _start_dashboard():
    try:
        from . import dashboard
        threading.Thread(target=dashboard.run, daemon=True, name="dashboard").start()
        print("[runner] dashboard thread started on :8090")
    except Exception as e:
        print(f"[runner] dashboard disabled: {e}")


def _cycle():
    summary = strategy.decide()
    try:
        if summary.get("auto_exits") or summary.get("status") == "FILLED":
            # post the trade that was just executed
            trades = get_store().recent_trades(1)
            if trades and summary.get("status") == "FILLED":
                reporter.send_trade_alert(trades[0])
            for ax in summary.get("auto_exits") or []:
                reporter._send(f"**AUTO RISK EXIT** `{ax}`")
        reporter.send_decision_log(summary)
    except Exception as e:
        print(f"[runner] report failed: {e}")


def main():
    print("[runner] starting paper trader")
    store = get_store()
    pf = store.get_portfolio()
    print(f"[runner] portfolio: cash=${pf['cash']:.2f} total=${pf['total_value']:.2f}")
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

        now_ts = time.time()
        _maybe_hourly(now_ts)
        _maybe_daily_close()

        sleep_s = OPEN_INTERVAL_S if market.is_market_open() else CLOSED_INTERVAL_S
        print(f"[runner] sleeping {sleep_s}s (market_open={market.is_market_open()})")
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
