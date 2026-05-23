"""Bag-holding skill — $-attribution of losses by failure mode.

The single desk question this answers: *which failure mode bleeds the
account the most?* — not how many times each mode occurred, but how
many DOLLARS each mode cost. A book with one ``KNIFE_CATCH`` at -$50
and nine ``WHIPSAW`` losses at -$0.30 each has the same
"WHIPSAW dominates" verdict from ``loser_autopsy`` (mode-count basis)
but the operator's actual bleed is the one knife catch.

Distinct from every neighbour (AGENTS.md invariant #10 — do not
consolidate):

* ``/api/loser-autopsy`` — per-trade cards + COUNT of each failure mode
  + $ per TICKER. Does NOT aggregate $-loss by failure MODE, nor
  compute the bag-holding ratio.
* ``/api/trade-asymmetry`` — book-level expectancy / payoff / win-rate
  with the disposition gap (avg-winner-hold vs avg-loser-hold) as
  one aggregate scalar. Mode-blind.
* ``/api/repeat-loser`` — same-name recurrence cadence, mode-blind.
* ``/api/hold-discipline`` — winners' hold-day distribution, blind to
  losers entirely.

This module is the missing **dollar-weighted** view of the same
``loser_autopsy._classify`` taxonomy. The headline answer:

  "WHIPSAW dominates" (loser_autopsy)  vs
  "SLOW_BLEED is the bleed — 73% of total $ lost is in slow-bleed
   round-trips" (this module)

Single source of truth: ``loser_autopsy._classify`` is imported, never
reimplemented — the failure-mode taxonomy lives in exactly one place.
The round-trip ledger is ``build_round_trips`` (the same SSOT
``loser_autopsy`` consumes). Advisory only — never gates Opus, adds
no caps (AGENTS.md invariants #2/#12).

Verdict ladder:

* ``NO_DATA``        — no closed round-trips at all
* ``NO_LOSSES``      — round-trips exist but none lost money
* ``EMERGING``       — < STABLE_MIN_LOSERS losers; numerics emitted,
                       verdict withheld (the loser_autopsy precedent)
* ``BAG_HOLDER``     — SLOW_BLEED share of total $ lost ≥
                       BAG_HOLDER_RATIO. The literal "holding bags".
* ``DISCIPLINED_CUTTER`` — SLOW_BLEED share ≤ DISCIPLINED_RATIO AND
                       WHIPSAW share ≤ WHIPSAW_NOISE_RATIO. Cuts
                       losers fast without leaving them to compound.
* ``KNIFE_CATCHER``  — KNIFE_CATCH share of total $ lost ≥
                       KNIFE_CATCHER_RATIO. The entry thesis itself
                       is repeatedly wrong (not a hold-discipline
                       problem, a stock-picking problem).
* ``WHIPSAW_BLEED``  — WHIPSAW share ≥ WHIPSAW_NOISE_RATIO AND the
                       book is cutting profitable noise too tight.
* ``MIXED``          — none of the above arms triggers cleanly; the
                       bleed is spread across modes.

Pattern verdict (BAG_HOLDER, DISCIPLINED_CUTTER, KNIFE_CATCHER,
WHIPSAW_BLEED, MIXED) is only emitted at STABLE; below that the
numerics are still emitted but ``verdict`` is None.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .loser_autopsy import _classify
from .round_trips import build_round_trips

# Sample-size gate identical to loser_autopsy — a two-loss "pattern" is
# noise, surfacing the verdict before STABLE will whipsaw it on every
# new loss.
STABLE_MIN_LOSERS = 8

# Verdict thresholds on the SHARE of total loss dollars in each mode.
# These are documented module constants (tests assert exact behaviour
# at each boundary) — not tunables Opus ever sees.
BAG_HOLDER_RATIO = 0.60        # ≥ 60% of $ lost in SLOW_BLEED
DISCIPLINED_RATIO = 0.20       # ≤ 20% of $ lost in SLOW_BLEED
KNIFE_CATCHER_RATIO = 0.50     # ≥ 50% of $ lost in KNIFE_CATCH
WHIPSAW_NOISE_RATIO = 0.40     # ≥ 40% of $ lost in WHIPSAW

# Canonical ordering — drives stable JSON output, headline composition,
# and is the source-of-truth for the mode universe.
_MODES = ("KNIFE_CATCH", "SLOW_BLEED", "STOPPED_OUT", "WHIPSAW")


def _round(x: float | None, n: int = 4) -> float | None:
    if x is None:
        return None
    return round(float(x), n)


def build_bag_holding_skill(trades: list[dict],
                            now: datetime | None = None) -> dict:
    """$-weighted failure-mode attribution for closed losing round-trips.

    Args:
        trades: ``Store.recent_trades()``-shaped ledger ordered
            **oldest→newest** (the ``build_round_trips`` contract).
            Pass exactly what ``/api/analytics`` / ``/api/loser-autopsy``
            pass: ``list(reversed(store.recent_trades(2000)))``.
        now: optional clock injection for deterministic tests.

    Pure, deterministic, never raises. A malformed row degrades; the
    contract is "no skill verdict this cycle", never an exception.
    """
    now = now or datetime.now(timezone.utc)
    # build_round_trips raises KeyError on rows missing the 'ticker' key;
    # the contract is "no skill verdict this cycle", never an exception,
    # so a garbage ledger degrades to NO_DATA rather than 500-ing the
    # endpoint (the loser_autopsy / drawdown defense-in-depth precedent).
    try:
        rts = build_round_trips(trades or [])
    except Exception:
        rts = []
    n_rts = len(rts)

    # Strict < 0 loser convention — identical to loser_autopsy + #10.
    losers = [rt for rt in rts if (rt.get("pnl_usd") or 0.0) < 0]
    n_losers = len(losers)

    # Per-mode aggregator: count, sum_loss_usd, sum_loss_pct (for avg),
    # sum_hold_days (for avg), max_loss_usd (single worst), tickers list.
    by_mode: dict[str, dict] = {
        m: {
            "mode": m,
            "n": 0,
            "loss_usd": 0.0,
            "_loss_pct_sum": 0.0,
            "_loss_pct_n": 0,
            "_hold_sum": 0.0,
            "_hold_n": 0,
            "worst_loss_usd": 0.0,    # most-negative single loss seen
            "worst_ticker": None,
            "tickers": [],            # distinct tickers contributing
        }
        for m in _MODES
    }

    total_loss_usd = 0.0
    for rt in losers:
        pnl = float(rt.get("pnl_usd") or 0.0)
        pct = rt.get("pnl_pct")
        hold = rt.get("hold_days")
        mode = _classify(hold, pct)
        b = by_mode[mode]
        b["n"] += 1
        b["loss_usd"] = round(b["loss_usd"] + pnl, 4)
        if pct is not None:
            b["_loss_pct_sum"] += float(pct)
            b["_loss_pct_n"] += 1
        if hold is not None:
            b["_hold_sum"] += float(hold)
            b["_hold_n"] += 1
        if pnl < b["worst_loss_usd"]:
            b["worst_loss_usd"] = round(pnl, 4)
            b["worst_ticker"] = rt.get("ticker")
        tk = rt.get("ticker")
        if tk and tk not in b["tickers"]:
            b["tickers"].append(tk)
        total_loss_usd += pnl
    total_loss_usd = round(total_loss_usd, 4)

    # Finalize per-mode rows: avg pct/hold, $-share, drop scratch keys.
    rows: list[dict] = []
    for m in _MODES:
        b = by_mode[m]
        avg_pct = (b["_loss_pct_sum"] / b["_loss_pct_n"]
                   if b["_loss_pct_n"] else None)
        avg_hold = (b["_hold_sum"] / b["_hold_n"]
                    if b["_hold_n"] else None)
        share = (b["loss_usd"] / total_loss_usd
                 if (total_loss_usd != 0 and b["loss_usd"] < 0) else 0.0)
        rows.append({
            "mode": m,
            "n": b["n"],
            "loss_usd": b["loss_usd"],
            "share_of_loss": _round(share, 4),
            "avg_loss_pct": _round(avg_pct, 4),
            "avg_hold_days": _round(avg_hold, 4),
            "worst_loss_usd": b["worst_loss_usd"] if b["n"] else 0.0,
            "worst_ticker": b["worst_ticker"],
            "tickers": list(b["tickers"]),
        })

    # Stable order: by share descending (most painful first), with the
    # canonical ladder as the deterministic tiebreak.
    rows.sort(key=lambda r: (r["loss_usd"], _MODES.index(r["mode"])))

    # ---- state / verdict ladder ---------------------------------------
    if n_rts == 0:
        state = "NO_DATA"
    elif n_losers == 0:
        state = "NO_LOSSES"
    elif n_losers >= STABLE_MIN_LOSERS:
        state = "STABLE"
    else:
        state = "EMERGING"

    # Share lookups for verdict — by mode name (no order dependence).
    share_by_mode = {r["mode"]: (r["share_of_loss"] or 0.0) for r in rows}
    slow_share = share_by_mode.get("SLOW_BLEED", 0.0)
    knife_share = share_by_mode.get("KNIFE_CATCH", 0.0)
    whipsaw_share = share_by_mode.get("WHIPSAW", 0.0)

    verdict = None
    if state == "STABLE":
        # Verdict precedence is intentional and documented above:
        #   BAG_HOLDER > KNIFE_CATCHER > WHIPSAW_BLEED >
        #   DISCIPLINED_CUTTER > MIXED.
        if slow_share >= BAG_HOLDER_RATIO:
            verdict = "BAG_HOLDER"
        elif knife_share >= KNIFE_CATCHER_RATIO:
            verdict = "KNIFE_CATCHER"
        elif whipsaw_share >= WHIPSAW_NOISE_RATIO:
            verdict = "WHIPSAW_BLEED"
        elif (slow_share <= DISCIPLINED_RATIO
              and whipsaw_share <= WHIPSAW_NOISE_RATIO):
            verdict = "DISCIPLINED_CUTTER"
        else:
            verdict = "MIXED"

    # Dominant mode (largest $ bleed) — surfaced regardless of state for
    # the EMERGING headline. Deterministic tiebreak by canonical order.
    dominant_mode = None
    if any(r["n"] > 0 for r in rows):
        # rows are sorted most-painful first; first with n>0.
        for r in rows:
            if r["n"] > 0:
                dominant_mode = r["mode"]
                break

    # ---- headline ------------------------------------------------------
    if state == "NO_DATA":
        headline = "No closed round-trips yet — bag-holding skill unscorable."
    elif state == "NO_LOSSES":
        headline = (f"No losing round-trips across {n_rts} closed — "
                    f"nothing to attribute.")
    else:
        # Dominant-$ clause: "MODE is the bleed (NN% of $ lost over K trades)".
        dom_row = next((r for r in rows if r["mode"] == dominant_mode), None)
        if dom_row and dom_row["n"]:
            pct = (dom_row["share_of_loss"] or 0.0) * 100.0
            bleed_clause = (f"{dominant_mode} is the bleed "
                            f"({pct:.0f}% of ${abs(total_loss_usd):.2f} "
                            f"lost over {dom_row['n']} "
                            f"{'trade' if dom_row['n'] == 1 else 'trades'})")
        else:
            bleed_clause = "no dominant failure mode"

        if state == "EMERGING":
            headline = (
                f"Emerging — {n_losers} of {STABLE_MIN_LOSERS} losing "
                f"round-trips for a stable bag-holding read (verdict "
                f"withheld). Provisional: {bleed_clause}."
            )
        else:
            # STABLE — verdict-aware framing.
            if verdict == "BAG_HOLDER":
                headline = (
                    f"BAG_HOLDER — {slow_share * 100:.0f}% of "
                    f"${abs(total_loss_usd):.2f} lost is in SLOW_BLEED "
                    f"round-trips (held ≥5d through the loss). "
                    f"{bleed_clause}."
                )
            elif verdict == "KNIFE_CATCHER":
                headline = (
                    f"KNIFE_CATCHER — {knife_share * 100:.0f}% of "
                    f"${abs(total_loss_usd):.2f} lost is in KNIFE_CATCH "
                    f"round-trips (entry thesis -15%+ wrong). The "
                    f"problem is stock-picking, not hold discipline."
                )
            elif verdict == "WHIPSAW_BLEED":
                headline = (
                    f"WHIPSAW_BLEED — {whipsaw_share * 100:.0f}% of "
                    f"${abs(total_loss_usd):.2f} lost is in WHIPSAW "
                    f"round-trips (cut inside a day at a shallow loss). "
                    f"Cutting noise too tight."
                )
            elif verdict == "DISCIPLINED_CUTTER":
                headline = (
                    f"DISCIPLINED_CUTTER — only {slow_share * 100:.0f}% "
                    f"of ${abs(total_loss_usd):.2f} lost is SLOW_BLEED, "
                    f"and {whipsaw_share * 100:.0f}% is WHIPSAW. "
                    f"Cuts losers without leaving them to compound."
                )
            else:  # MIXED
                headline = (
                    f"MIXED — no single failure mode dominates "
                    f"${abs(total_loss_usd):.2f} of losses. "
                    f"{bleed_clause}."
                )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "n_round_trips": n_rts,
        "n_losers": n_losers,
        "total_loss_usd": total_loss_usd,
        "dominant_mode": dominant_mode,
        "bag_holding_ratio": _round(slow_share, 4),
        "knife_catch_ratio": _round(knife_share, 4),
        "whipsaw_ratio": _round(whipsaw_share, 4),
        "rows": rows,
        "stable_min_losers": STABLE_MIN_LOSERS,
        "thresholds": {
            "BAG_HOLDER_RATIO": BAG_HOLDER_RATIO,
            "DISCIPLINED_RATIO": DISCIPLINED_RATIO,
            "KNIFE_CATCHER_RATIO": KNIFE_CATCHER_RATIO,
            "WHIPSAW_NOISE_RATIO": WHIPSAW_NOISE_RATIO,
        },
    }


if __name__ == "__main__":  # one-screen answer; the loser_autopsy CLI precedent
    import json

    from paper_trader.store import get_store
    s = get_store()
    rep = build_bag_holding_skill(list(reversed(s.recent_trades(2000))))
    if "--json" in __import__("sys").argv:
        print(json.dumps(rep, indent=2, default=str))
    else:
        tag = rep["state"] + (f"/{rep['verdict']}" if rep["verdict"] else "")
        print(f"BAG-HOLDING SKILL  [{tag}]  {rep['headline']}")
        if rep["state"] not in ("NO_DATA", "NO_LOSSES"):
            print(f"  total loss: ${rep['total_loss_usd']:+.2f}  "
                  f"over {rep['n_losers']} losing round-trips "
                  f"(of {rep['n_round_trips']} closed)")
            print(f"  bag-holding ratio (SLOW_BLEED share): "
                  f"{(rep['bag_holding_ratio'] or 0) * 100:.1f}%")
            for r in rep["rows"]:
                if r["n"] == 0:
                    continue
                pct = (r["share_of_loss"] or 0) * 100
                avgp = (f"{r['avg_loss_pct']:+.1f}%"
                        if r["avg_loss_pct"] is not None else "n/a")
                avgh = (f"{r['avg_hold_days']:.1f}d"
                        if r["avg_hold_days"] is not None else "n/a")
                print(f"    {r['mode']:<13} "
                      f"${r['loss_usd']:+8.2f}  "
                      f"({pct:>5.1f}%)  n={r['n']:<3} "
                      f"avg={avgp}/{avgh}  worst {r['worst_ticker']}")
