"""Portfolio-level options risk + structure opportunity scanner.

Feeds the options desk prompt with:
  - portfolio option Greeks / theta burn / premium at risk
  - collateral usage (covered calls / CSPs)
  - structure opportunity candidates from chain snapshots
"""
from __future__ import annotations

from datetime import date
from typing import Any


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _dte(expiry: str | None) -> int | None:
    if not expiry:
        return None
    try:
        exp = date.fromisoformat(str(expiry)[:10])
    except ValueError:
        return None
    return (exp - date.today()).days


def build_options_risk(snapshot: dict | None, watch_prices: dict | None = None) -> dict:
    """Aggregate open option risk. Never raises."""
    try:
        snap = snapshot or {}
        positions = list(snap.get("positions") or [])
        cash = _f(snap.get("cash"))
        equity = _f(snap.get("total_value"), cash)
        long_prem = 0.0
        short_prem = 0.0
        short_put_collateral = 0.0
        covered_call_shares = 0.0
        lots = []
        for p in positions:
            if p.get("type") not in ("call", "put"):
                continue
            qty = _f(p.get("qty"))
            if abs(qty) < 1e-9:
                continue
            mark = _f(p.get("current_price"), _f(p.get("avg_cost")))
            avg = _f(p.get("avg_cost"))
            mv = mark * qty * 100.0
            if qty > 0:
                long_prem += abs(mv)
            else:
                short_prem += abs(mv)
                if p.get("type") == "put":
                    short_put_collateral += abs(qty) * _f(p.get("strike")) * 100.0
                if p.get("type") == "call":
                    covered_call_shares += abs(qty) * 100.0
            lots.append({
                "ticker": str(p.get("ticker") or "").upper(),
                "type": p.get("type"),
                "side": "long" if qty > 0 else "short",
                "qty": qty,
                "strike": _f(p.get("strike")),
                "expiry": p.get("expiry"),
                "dte": _dte(p.get("expiry")),
                "mark": mark,
                "avg": avg,
                "market_value": mv,
                "unrealized_pl": _f(p.get("unrealized_pl")),
            })

        # Greeks if available
        greeks = {}
        try:
            from .greeks import compute_position_greeks
            g = compute_position_greeks(positions)
            greeks = g.get("totals") or {}
        except Exception as e:
            greeks = {"error": str(e)}

        prem_budget_used = long_prem
        prem_budget_pct = (prem_budget_used / equity * 100.0) if equity > 0 else 0.0
        csp_cash_pct = (short_put_collateral / cash * 100.0) if cash > 0 else 0.0
        try:
            from .friday_closer import NEW_DEBIT_CAP, leftover_option_mark, new_debit_budget_remaining
            leftover_mark = leftover_option_mark(positions)
            new_debit_left = new_debit_budget_remaining(positions)
        except Exception:
            leftover_mark = long_prem
            new_debit_left = 800.0
            NEW_DEBIT_CAP = 800.0

        flags = []
        if prem_budget_pct > 35:
            flags.append("HIGH_LONG_PREMIUM (>35% equity in leftover long option mark)")
        if csp_cash_pct > 80:
            flags.append("CSP_COLLATERAL_TIGHT (>80% cash reserved by short put strikes)")
        for lot in lots:
            dte = lot.get("dte")
            if dte is not None and dte <= 7 and lot.get("side") == "long":
                flags.append("LONG_OPTION_DTE<=7 %s" % lot["ticker"])
            if dte is not None and dte <= 5 and lot.get("side") == "short":
                flags.append("SHORT_OPTION_DTE<=5 %s" % lot["ticker"])

        lines = [
            "OPTIONS RISK (portfolio — facts only):",
            "  leftover_long_option_mark=$%.2f (%.1f%% equity; NOT the new-debit cap)  short_option_mark=$%.2f" % (
                leftover_mark, prem_budget_pct, short_prem
            ),
            "  leftover new-debit budget leftover=$%.2f of $%.0f (informational only; no NVDA/QQQ hard-send)." % (
                new_debit_left, NEW_DEBIT_CAP
            ),
            "  CSP strike collateral notionally=$%.2f (%.1f%% cash)  covered_call_shares_reserved=%.0f" % (
                short_put_collateral, csp_cash_pct, covered_call_shares
            ),
        ]
        if greeks and not greeks.get("error"):
            lines.append(
                "  greeks totals: delta=%.2f gamma=%.4f theta/day=%.2f vega=%.2f" % (
                    _f(greeks.get("delta")),
                    _f(greeks.get("gamma")),
                    _f(greeks.get("theta")),
                    _f(greeks.get("vega")),
                )
            )
        if lots:
            lines.append("  open option lots:")
            for lot in lots[:12]:
                lines.append(
                    "    - %s %s %s k=%s exp=%s dte=%s qty=%.4g mark=%.2f U-P/L=$%.2f" % (
                        lot["ticker"],
                        lot["side"],
                        str(lot["type"]).upper(),
                        lot["strike"],
                        lot["expiry"],
                        lot["dte"],
                        lot["qty"],
                        lot["mark"],
                        lot["unrealized_pl"],
                    )
                )
        else:
            lines.append("  open option lots: (none)")
        if flags:
            lines.append("  risk flags: " + "; ".join(flags))
        else:
            lines.append("  risk flags: none")
        lines.append(
            "  Budget guide: leftover option mark is already spent. There is no "
            "NVDA/QQQ Friday-weekly obligation. Diversify; do not pile the same factor."
        )

        return {
            "state": "ACTIVE" if lots else "EMPTY",
            "prompt_block": chr(10).join(lines),
            "lots": lots,
            "long_premium": long_prem,
            "short_premium": short_prem,
            "short_put_collateral": short_put_collateral,
            "covered_call_shares": covered_call_shares,
            "premium_budget_pct": prem_budget_pct,
            "greeks": greeks,
            "flags": flags,
        }
    except Exception as e:
        return {
            "state": "ERROR",
            "prompt_block": None,
            "error": str(e),
        }


def scan_structure_opportunities(chains: list, cash: float, positions=None) -> dict:
    """From chain snapshots, propose a few concrete advanced structures."""
    try:
        ideas = []
        pos = positions or []
        long_shares = {}
        for p in pos:
            if p.get("type") == "stock":
                q = _f(p.get("qty"))
                if q > 0:
                    t = str(p.get("ticker") or "").upper()
                    long_shares[t] = long_shares.get(t, 0.0) + q

        for c in chains or []:
            if c.get("error"):
                continue
            ticker = str(c.get("ticker") or "").upper()
            exp = c.get("expiry")
            dte = c.get("dte")
            spot = _f(c.get("spot"))
            calls = list(c.get("calls") or [])
            puts = list(c.get("puts") or [])
            if not calls and not puts:
                continue

            # Covered call if long stock and OTM call exists
            shares = long_shares.get(ticker, 0.0)
            if shares >= 100 and calls:
                otm_calls = [x for x in calls if _f(x.get("strike")) > spot * 1.02]
                pick = otm_calls[0] if otm_calls else calls[-1]
                k = _f(pick.get("strike"))
                mid = _f(pick.get("mid"))
                if k > 0 and mid > 0:
                    ideas.append({
                        "kind": "COVERED_CALL",
                        "ticker": ticker,
                        "expiry": exp,
                        "dte": dte,
                        "short_strike": k,
                        "credit": mid,
                        "contracts": int(shares // 100),
                        "action_hint": "SELL_CALL",
                        "note": "covered by %.0f shares" % shares,
                    })

            # CSP ATM-ish put if cash allows
            if puts and cash > 0:
                atm = min(puts, key=lambda x: abs(_f(x.get("strike")) - spot)) if spot else puts[0]
                k = _f(atm.get("strike"))
                mid = _f(atm.get("mid"))
                if k > 0 and mid > 0:
                    collat = k * 100.0 - mid * 100.0
                    if collat > 0 and cash >= collat:
                        ideas.append({
                            "kind": "CSP",
                            "ticker": ticker,
                            "expiry": exp,
                            "dte": dte,
                            "short_strike": k,
                            "credit": mid,
                            "collateral": collat,
                            "action_hint": "SELL_PUT",
                            "note": "cash-secured put 1ct collateral~$%.0f" % collat,
                        })

            # Debit call spread from two call strikes
            if len(calls) >= 2 and spot > 0:
                calls_sorted = sorted(calls, key=lambda x: _f(x.get("strike")))
                # prefer long near ATM, short higher
                long_leg = min(calls_sorted, key=lambda x: abs(_f(x.get("strike")) - spot))
                higher = [x for x in calls_sorted if _f(x.get("strike")) > _f(long_leg.get("strike")) + 0.5]
                if higher:
                    short_leg = higher[0]
                    lk = _f(long_leg.get("strike")); sk = _f(short_leg.get("strike"))
                    lpx = _f(long_leg.get("mid")); spx = _f(short_leg.get("mid"))
                    debit = lpx - spx
                    width = sk - lk
                    if debit > 0 and width > 0:
                        ideas.append({
                            "kind": "BUY_CALL_SPREAD",
                            "ticker": ticker,
                            "expiry": exp,
                            "dte": dte,
                            "long_strike": lk,
                            "short_strike": sk,
                            "debit": debit,
                            "width": width,
                            "max_gain": width - debit,
                            "cost_1ct": debit * 100.0,
                            "action_hint": "BUY_CALL_SPREAD",
                            "note": "debit call %.2f/%.2f debit=%.2f" % (lk, sk, debit),
                        })

            # Credit put spread
            if len(puts) >= 2 and spot > 0:
                puts_sorted = sorted(puts, key=lambda x: _f(x.get("strike")))
                short_leg = min(puts_sorted, key=lambda x: abs(_f(x.get("strike")) - spot * 0.98))
                lower = [x for x in puts_sorted if _f(x.get("strike")) < _f(short_leg.get("strike")) - 0.5]
                if lower:
                    long_leg = lower[-1]
                    sk = _f(short_leg.get("strike")); lk = _f(long_leg.get("strike"))
                    spx = _f(short_leg.get("mid")); lpx = _f(long_leg.get("mid"))
                    credit = spx - lpx
                    width = sk - lk
                    if credit > 0 and width > 0:
                        ideas.append({
                            "kind": "SELL_PUT_SPREAD",
                            "ticker": ticker,
                            "expiry": exp,
                            "dte": dte,
                            "short_strike": sk,
                            "long_strike": lk,
                            "credit": credit,
                            "width": width,
                            "max_loss": width - credit,
                            "credit_1ct": credit * 100.0,
                            "action_hint": "SELL_PUT_SPREAD",
                            "note": "put credit %.2f/%.2f credit=%.2f" % (sk, lk, credit),
                        })

        # de-dupe by kind+ticker+expiry keep first few
        seen = set()
        uniq = []
        for idea in ideas:
            key = (idea.get("kind"), idea.get("ticker"), idea.get("expiry"), idea.get("long_strike"), idea.get("short_strike"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(idea)
            if len(uniq) >= 8:
                break

        lines = ["STRUCTURE OPPORTUNITIES (candidates from live chains — not orders):"]
        if not uniq:
            lines.append("  (none this cycle)")
        for idea in uniq:
            lines.append(
                "  - %s %s exp=%s dte=%s | %s | hint=%s" % (
                    idea.get("kind"),
                    idea.get("ticker"),
                    idea.get("expiry"),
                    idea.get("dte"),
                    idea.get("note"),
                    idea.get("action_hint"),
                )
            )
        return {"ideas": uniq, "prompt_block": chr(10).join(lines)}
    except Exception as e:
        return {"ideas": [], "prompt_block": None, "error": str(e)}
