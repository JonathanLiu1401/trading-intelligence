#!/usr/bin/env python3
"""One-shot: after Monday regular open, buy >=1 option via strategy._execute.

Jonathan requested 2026-07-18: prove live options path works after market opens
Monday. Safe to re-run - uses a done marker and will not double-buy.

Usage:
  python scripts/monday_option_proof.py            # wait until open then buy
  python scripts/monday_option_proof.py --now      # force attempt now
  python scripts/monday_option_proof.py --dry-run  # select contract only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NY = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")
DONE_PATH = ROOT / "data" / "monday_option_proof_done.json"
LOG_PATH = ROOT / "logs" / "monday_option_proof.log"

CANDIDATES = [
    # Prefer liquid, affordable contracts first under margin BP.
    ("AAPL", "2026-08-21", 230.0, "call"),
    ("AAPL", "2026-09-18", 230.0, "call"),
    ("AAPL", "2026-09-18", 240.0, "call"),
    ("AAPL", "2026-09-18", 350.0, "call"),
    ("AAPL", "2026-09-18", 340.0, "call"),
    ("AAPL", "2026-08-21", 340.0, "call"),
    ("AAPL", "2026-12-18", 350.0, "call"),
    ("MSFT", "2026-09-18", 520.0, "call"),
    ("MSFT", "2026-09-18", 540.0, "call"),
    ("SPY", "2026-09-18", 700.0, "call"),
    ("SPY", "2026-09-18", 690.0, "call"),
    ("SOXL", "2026-08-21", 45.0, "call"),
    ("SOXL", "2026-09-18", 50.0, "call"),
]


def _log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _already_done() -> dict | None:
    if not DONE_PATH.exists():
        return None
    try:
        return json.loads(DONE_PATH.read_text())
    except Exception:
        return {"raw": DONE_PATH.read_text()[:500]}


def _mark_done(payload: dict) -> None:
    DONE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DONE_PATH.write_text(json.dumps(payload, indent=2, default=str))


def _wait_until_regular_open(max_wait_s: int = 60 * 60 * 14) -> bool:
    """Block until NYSE regular session is open. Returns False on timeout."""
    from paper_trader import market

    start = time.time()
    while time.time() - start < max_wait_s:
        now_ny = datetime.now(NY)
        if market.is_market_open():
            _log(f"market OPEN (NY {now_ny.isoformat()})")
            return True
        mins = now_ny.hour * 60 + now_ny.minute
        if now_ny.weekday() >= 5:
            sleep_s = 1800
        elif mins < 9 * 60 + 25:
            sleep_s = 120
        elif mins < 9 * 60 + 35:
            sleep_s = 15
        else:
            sleep_s = 60
        _log(
            f"waiting for regular open (NY {now_ny.strftime('%a %Y-%m-%d %H:%M')}); "
            f"sleep {sleep_s}s"
        )
        time.sleep(sleep_s)
    return False


def _pick_contract(buying_power: float, cash: float | None = None) -> dict | None:
    """Pick first affordable contract against margin buying power (not cash alone)."""
    from paper_trader import market

    cash_s = f" cash ${float(cash):.2f}" if cash is not None else ""
    for ticker, expiry, strike, otype in CANDIDATES:
        px = market.get_option_price(ticker, expiry, strike, otype)
        if not px or px <= 0:
            _log(f"skip {ticker} {expiry} {strike}{otype[0].upper()}: no price")
            continue
        cost = float(px) * 100.0
        if cost + 25.0 > buying_power:
            _log(
                f"skip {ticker} {expiry} {strike}{otype[0].upper()} @ {px}: "
                f"cost ${cost:.2f} > buying_power ${buying_power:.2f}{cash_s}"
            )
            continue
        return {
            "ticker": ticker,
            "expiry": expiry,
            "strike": float(strike),
            "option_type": otype,
            "price": float(px),
            "cost": cost,
        }
    return None

def _notify(msg: str) -> None:
    try:
        from paper_trader import reporter
        reporter._send(msg)
    except Exception as e:
        _log(f"notify failed: {e}")


def run(force_now: bool = False, dry_run: bool = False) -> int:
    prior = _already_done()
    if prior and prior.get("status") == "FILLED" and not force_now:
        _log(f"already done: {prior}")
        return 0

    if not force_now:
        ok = _wait_until_regular_open()
        if not ok:
            _log("TIMEOUT waiting for market open")
            _notify(
                "**OPTIONS PROOF FAILED** ◈ timed out waiting for Monday regular open"
            )
            return 2

    from paper_trader.store import Store
    from paper_trader import strategy

    store = Store()
    pf = store.get_portfolio()
    cash = float(pf.get("cash") or 0.0)
    total = float(pf.get("total_value") or 0.0)
    # Prefer live marked snapshot BP when available (cash + 50% NW).
    if hasattr(strategy, "_portfolio_snapshot"):
        try:
            snap0 = strategy._portfolio_snapshot(store)
            total = float(snap0.get("total_value") or total)
            cash = float(snap0.get("cash") or cash)
            if snap0.get("stock_buying_power") is not None:
                buying_power = float(snap0.get("stock_buying_power"))
            else:
                buying_power = cash + max(0.0, total) * 0.50
        except Exception as e:
            _log(f"snapshot BP failed, falling back: {e}")
            buying_power = cash + max(0.0, total) * 0.50
    else:
        buying_power = cash + max(0.0, total) * 0.50
    _log(
        f"cash=${cash:.2f} total=${total:.2f} buying_power=${buying_power:.2f}"
    )

    pick = _pick_contract(buying_power, cash=cash)
    if not pick:
        msg = "no affordable option contract found for proof buy"
        _log(msg)
        _notify(
            f"**OPTIONS PROOF FAILED** ◈ {msg} "
            f"(cash ${cash:.2f}, bp ${buying_power:.2f})"
        )
        _mark_done(
            {
                "status": "FAILED",
                "reason": msg,
                "cash": cash,
                "buying_power": buying_power,
                "as_of": datetime.now(timezone.utc).isoformat(),
            }
        )
        return 3

    decision = {
        "action": "BUY_CALL" if pick["option_type"] == "call" else "BUY_PUT",
        "ticker": pick["ticker"],
        "qty": 1,
        "strike": pick["strike"],
        "expiry": pick["expiry"],
        "reasoning": (
            "Monday open options-path proof (operator one-shot). "
            "Buy >=1 option contract to verify live pricing + execute + position book."
        ),
        "confidence": 0.99,
    }
    _log(f"selected {decision} est_cost=${pick['cost']:.2f} mid={pick['price']}")

    if dry_run:
        print(json.dumps({"dry_run": True, "decision": decision, "pick": pick}, indent=2))
        return 0

    if hasattr(strategy, "_portfolio_snapshot"):
        snap = strategy._portfolio_snapshot(store)
    else:
        snap = store.get_portfolio()

    status, detail = strategy._execute(decision, snap, store)
    _log(f"execute -> {status}: {detail}")

    lots = [
        p
        for p in store.open_positions()
        if p.get("type") in ("call", "put")
        and p.get("ticker") == pick["ticker"]
        and float(p.get("strike") or 0) == pick["strike"]
        and p.get("expiry") == pick["expiry"]
        and float(p.get("qty") or 0) > 0
    ]
    payload = {
        "status": status,
        "detail": detail,
        "decision": decision,
        "pick": pick,
        "verified_lots": lots,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "ny": datetime.now(NY).isoformat(),
        "pt": datetime.now(PT).isoformat(),
    }
    _mark_done(payload)

    if status == "FILLED" and lots:
        _notify(
            "**OPTIONS PROOF FILLED** ◈ "
            f"{decision['action']} 1 {pick['ticker']} "
            f"{pick['strike']:g}{pick['option_type'][0].upper()} {pick['expiry']} "
            f"-> {detail}"
        )
        return 0

    _notify(
        f"**OPTIONS PROOF NOT FILLED** ◈ status={status} detail={detail} "
        f"lots={len(lots)}"
    )
    return 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true", help="skip wait-for-open")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="ignore done marker and buy again (still needs open unless --now)",
    )
    args = ap.parse_args()
    if args.force and DONE_PATH.exists():
        DONE_PATH.unlink()
    return run(force_now=args.now, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

