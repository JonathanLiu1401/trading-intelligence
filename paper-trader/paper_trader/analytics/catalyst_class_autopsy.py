"""Catalyst-class autopsy — per-entry-thesis realised P&L of closed round-trips.

The desk question that ``/api/loser-autopsy`` and ``/api/winner-autopsy``
don't answer. Both of those classify the **exit** behaviour
(KNIFE_CATCH / WHIPSAW / SLOW_BLEED / STOPPED_OUT and HOME_RUN /
SCALP / SLOW_GRIND / TARGET_HIT respectively) — *how the trade was
closed*. Neither classifies the **entry thesis** — *which catalyst
TYPE motivated the open in the first place*.

The verbatim entry rationale is the strongest learning signal in the
ledger: a typical reason on the live book mixes multiple catalyst
sources ("ML advisor + Citi PT + RSI breakout + earnings imminent").
This builder multi-labels each closed round-trip by every catalyst
class present in its entry reason, then surfaces per-class win rate,
total realised $, average %, median hold. The operator's learning
question — *which catalyst TYPES make me money, which ones bleed?* —
gets a sample-size-honest answer that recasts each future entry
rationale as a known-edge bet.

Single source of truth (AGENTS.md invariant #10): consumes
``round_trips.build_round_trips`` verbatim and joins the entry reason
back from the contributing trade row by DB ``id`` (the same
"surface the reason verbatim, never NLP-parse it for trading logic"
discipline ``loser_autopsy`` / ``winner_autopsy`` use). Advisory
only — never gates Opus, never injected into the decision prompt,
no caps (AGENTS.md #2/#12 — the ``loser_autopsy`` /
``trade_asymmetry`` / ``winner_autopsy`` precedent).

Sample-size honesty mirrors ``trade_asymmetry`` / ``loser_autopsy``:
numerics are emitted from the first closed trip in any class, but
the per-class **verdict** (BIASED_WINNER / BIASED_LOSER / NEUTRAL)
is withheld until that class has ``STABLE_MIN_TRIPS_PER_CLASS``
closed trips. A two-trip "pattern" is noise.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from .round_trips import build_round_trips

# Stable-pattern gate: per-class win-rate-vs-baseline verdict withheld
# below this. Two trips in a class is noise even when both are losses;
# four is the smallest sample where a 100% loss rate is non-trivially
# informative against the realistic ~50% baseline. (Mirrors the
# ``loser_autopsy.STABLE_MIN_LOSERS = 8`` / ``trade_asymmetry`` STABLE
# idiom but lower because per-class N is necessarily smaller than the
# pooled losers count.)
STABLE_MIN_TRIPS_PER_CLASS = 4

# A class's realised win-rate has to beat / undershoot the pool's by
# this margin to flip BIASED. Inside the band is NEUTRAL — the class
# does not statistically distinguish itself from the operator's
# baseline disposition. Same calibration shape as
# ``decision_weekday`` / ``decision_clock``'s concentration cutoffs.
BIASED_WR_DELTA_PCT = 15.0

# Catalyst taxonomy. Pure regex, applied case-insensitively against the
# entry reason. Ordered intentionally so deterministic dominant-class
# tie-breaking has a fixed precedence (highest signal-bearing class
# wins ties — same shape as ``loser_autopsy._SEVERITY``).
#
# Each class entry: (label, compiled pattern). Patterns use word
# boundaries / cashtag prefix / explicit context so a stray substring
# can never trip a label. Two real-trade samples below:
#
#   * "Triple-stacked catalyst: Citi bullish on DRAM, HSBC/Melius PT,
#     Cramer buy signal — and ML advisor (median +143% alpha) flags
#     DRAM. NVDA earnings tomorrow may drag DRAM up sympathetically."
#       → ML_ADVISOR, ANALYST_PT, PUNDIT, EARNINGS_PLAY, SECTOR_SYMPATHY
#   * "NVDA earnings imminent with full bullish stack: RSI 59.95 + MACD
#     bullish + golden cross all aligned, HSBC beat-and-raise call,
#     biggest S&P short position (squeeze fuel), memory supercycle ($840
#     MU PT from Citi), Huang China visit at record highs, and ML
#     advisor (median +143% alpha) confirms BUY."
#       → EARNINGS_PLAY, TECHNICALS, ANALYST_PT, ML_ADVISOR
_CLASS_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Highest-signal first (deterministic dominant tie-break).
    ("ML_ADVISOR", re.compile(
        r"\b(?:ml[\s-]*advisor|decisionscorer|scorer\s+(?:gates?|nudges?|"
        r"confirms?|flags?)|median\s+[+-]?\d+%?\s*alpha|alpha\s+gate)\b",
        re.IGNORECASE)),
    ("EARNINGS_PLAY", re.compile(
        r"\b(?:earnings|earn\s+report|EPS|post[-\s]earnings|pre[-\s]earnings|"
        r"print|quarter(?:ly)?|Q[1-4]\s+(?:report|print|results)|guidance"
        r"\s+(?:cut|raise|beat))\b",
        re.IGNORECASE)),
    ("ANALYST_PT", re.compile(
        r"\b(?:price\s+target|\$\d+\s+PT|PT\s+(?:from|by|of|raise|cut|hike)|"
        r"upgrade(?:d|s)?|downgrade(?:d|s)?|reiterates?|initiat(?:e|ed|es)|"
        r"raise(?:d|s)?\s+(?:to|price)|cut\s+to\s+\$|Citi|JPM|"
        r"JPMorgan|Goldman|HSBC|BofA|Morgan\s+Stanley|Wells\s+Fargo|"
        r"Barclays|Melius|Wedbush|Piper|Bernstein|Jefferies|UBS|"
        r"Bank\s+of\s+America|Deutsche\s+Bank|Susquehanna|Mizuho|Raymond"
        r"\s+James|Stifel|Oppenheimer|Truist|Cantor|Loop\s+Capital)\b",
        re.IGNORECASE)),
    ("TECHNICALS", re.compile(
        r"\b(?:RSI|MACD|golden\s+cross|death\s+cross|moving\s+average|"
        r"\d+\s*[-]?\s*day\s+MA|MA\s*\d+|"
        r"breakout|breakdown|"
        r"support\s+(?:at|level|holding)|resistance\s+(?:at|level)|"
        r"bollinger|stochastic|"
        r"oversold|overbought|momentum\s+(?:turning|reversal|extension)|"
        r"trend\s+(?:reversal|continuation)|chart\s+(?:pattern|setup)|"
        r"52[-\s]?week\s+(?:high|low))\b",
        re.IGNORECASE)),
    ("MACRO", re.compile(
        r"\b(?:FOMC|Fed(?:eral\s+Reserve)?|Powell|rate\s+(?:cut|hike|decision|"
        r"path)|CPI|PPI|PCE|NFP|nonfarm|payrolls|jobs\s+report|inflation"
        r"\s+(?:print|reading)|treasury\s+yield|dollar\s+index|DXY|"
        r"unemployment|GDP|recession|soft\s+landing)\b",
        re.IGNORECASE)),
    ("BREAKING_NEWS", re.compile(
        r"\b(?:breaking|just\s+(?:in|crossed|broke)|urgent\s+(?:headline|"
        r"alert)|tape\s+bomb|headline\s+(?:hit|crossed)|wire\s+(?:says|"
        r"reports)|live\s+update)\b",
        re.IGNORECASE)),
    ("PUNDIT", re.compile(
        r"\b(?:Cramer|Buffett|Druckenmiller|Burry|Ackman|Tepper|Dalio|"
        r"Munger|Wood|Cathie|Loeb|Einhorn|Klarman|Marks|Howard\s+Marks|"
        r"Soros|Ichan|Carl\s+Icahn)\b",
        re.IGNORECASE)),
    ("SECTOR_SYMPATHY", re.compile(
        r"\b(?:sympathy|sympathetic(?:ally)?|peer\s+(?:strength|weakness|"
        r"move)|sector\s+(?:rotation|momentum|move|leadership)|peer\s+drag|"
        r"basket\s+(?:lift|drag)|cohort|leveraged\s+(?:cousin|peer|exposure))"
        r"\b",
        re.IGNORECASE)),
    ("CONCENTRATION", re.compile(
        r"\b(?:concentration|over[-\s]?weight(?:ed)?|under[-\s]?weight(?:ed)?|"
        r"size\s+(?:down|up|trim)|trim(?:ming)?(?:\s+exposure)?|"
        r"reduce\s+exposure|rebalanc(?:e|ing)|cash\s+raise|"
        r"raise\s+dry\s+powder|dry\s+powder|cash\s+headroom)\b",
        re.IGNORECASE)),
]

_ALL_CLASSES = tuple(label for label, _ in _CLASS_PATTERNS)

# Deterministic precedence for the "dominant catalyst" tie-break — the
# loser_autopsy._SEVERITY analogue. Higher position ⇒ wins ties when
# two classes share equal trip-count rank. Mirrors the taxonomy order
# (ML_ADVISOR is the most specific signal; CONCENTRATION is the least
# entry-thesis-y).
_DOMINANT_PRECEDENCE = list(_ALL_CLASSES)


def _classify_classes(reason: str | None) -> list[str]:
    """Multi-label classify one entry reason.

    Returns the list of catalyst classes whose pattern matched anywhere
    in the reason string. Order follows ``_CLASS_PATTERNS`` (taxonomy
    order, so a downstream "dominant" pick is deterministic).

    A ``None`` / empty / whitespace-only reason returns ``[]`` — never
    raises. An unmatched-but-non-empty reason returns ``["UNCLASSIFIED"]``
    so that *every* round-trip with a reason contributes to some bucket
    (the pool baseline win-rate stays honest — a class membership of
    zero would silently inflate the pool's WR by dropping the worst
    "all-class-miss" trades).
    """
    if not reason:
        return []
    text = str(reason).strip()
    if not text:
        return []
    matched: list[str] = []
    for label, pat in _CLASS_PATTERNS:
        if pat.search(text):
            matched.append(label)
    return matched or ["UNCLASSIFIED"]


def _reason_for_entry(trade_ids: list, by_id: dict) -> str | None:
    """Verbatim ``reason`` of the OPENING trade — mirrors
    ``loser_autopsy._reason_for(..., pick_last=False)``.

    Missing / absent / empty all degrade to ``None`` (no error).
    """
    if not trade_ids:
        return None
    tid = trade_ids[0]
    row = by_id.get(tid)
    if not row:
        return None
    r = row.get("reason")
    return r if (r is not None and str(r).strip() != "") else None


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    sv = sorted(vals)
    m = len(sv) // 2
    if len(sv) % 2:
        return round(float(sv[m]), 4)
    return round((sv[m - 1] + sv[m]) / 2.0, 4)


def _per_class_verdict(class_wr: float, pool_wr: float, n: int) -> str:
    """Sample-size-gated per-class verdict.

    NEUTRAL inside the band, BIASED above/below, UNSTABLE below the
    sample gate. Pool baseline is the operator's overall win-rate
    across all closed round-trips so a class's verdict is read against
    *this trader's* disposition (a 40% pool WR makes a 50% class WR a
    BIASED_WINNER, not a NEUTRAL one).
    """
    if n < STABLE_MIN_TRIPS_PER_CLASS:
        return "UNSTABLE"
    if class_wr >= pool_wr + BIASED_WR_DELTA_PCT:
        return "BIASED_WINNER"
    if class_wr <= pool_wr - BIASED_WR_DELTA_PCT:
        return "BIASED_LOSER"
    return "NEUTRAL"


def build_catalyst_class_autopsy(
        trades: list[dict],
        now: datetime | None = None) -> dict:
    """Per-entry-thesis-class autopsy of closed round-trips. Pure, never raises.

    ``trades`` must be a ``Store.recent_trades()``-shaped ledger ordered
    **oldest→newest** (the ``/api/analytics`` /
    ``/api/loser-autopsy`` /  ``/api/winner-autopsy`` convention —
    ``build_round_trips`` reads in sequence and does not sort).

    Returns a dict with ``state`` ∈ ``{NO_DATA, EMERGING, STABLE}``,
    a one-line ``headline``, per-class rows, and aggregate counters.
    Never raises — a malformed trade row degrades the round-trip; a
    fully-bad ledger degrades to NO_DATA.
    """
    now = now or datetime.now(timezone.utc)

    try:
        rts = build_round_trips(trades or [])
    except Exception:
        rts = []
    n_rts = len(rts)

    by_id: dict = {}
    for t in (trades or []):
        try:
            tid = t.get("id")
            if tid is not None:
                by_id[tid] = t
        except Exception:
            continue

    # Build per-trip classification + extract realised metrics ONCE,
    # so per-class aggregation is a simple bucket-fill below.
    per_trip: list[dict] = []
    for rt in rts:
        try:
            reason = _reason_for_entry(rt.get("entry_trade_ids") or [], by_id)
            classes = _classify_classes(reason)
            pnl_usd = float(rt.get("pnl_usd") or 0.0)
            pnl_pct = rt.get("pnl_pct")
            hold = rt.get("hold_days")
            is_win = pnl_usd > 0
            per_trip.append({
                "ticker": rt.get("ticker"),
                "type": rt.get("type"),
                "entry_ts": rt.get("entry_ts"),
                "exit_ts": rt.get("exit_ts"),
                "pnl_usd": round(pnl_usd, 4),
                "pnl_pct": pnl_pct,
                "hold_days": hold,
                "classes": classes,
                "is_win": is_win,
                "entry_reason": reason,
            })
        except Exception:
            # One garbage row never sinks the whole report.
            continue

    n_scored = len(per_trip)

    # Pool baseline — the operator's overall realised WR. Used as the
    # per-class verdict anchor (a class only differentiates itself
    # against THIS trader's disposition, not an external 50% prior).
    pool_wins = sum(1 for t in per_trip if t["is_win"])
    pool_wr = (pool_wins / n_scored * 100.0) if n_scored else None

    # Bucket fill: a trip with K classes contributes to K buckets.
    by_class: dict[str, dict] = {}
    for t in per_trip:
        for cls in t["classes"]:
            b = by_class.setdefault(cls, {
                "class": cls,
                "n_trips": 0,
                "n_wins": 0,
                "n_losses": 0,
                "_pnl_usds": [],
                "_pnl_pcts": [],
                "_holds": [],
                "_trips": [],
            })
            b["n_trips"] += 1
            if t["is_win"]:
                b["n_wins"] += 1
            elif t["pnl_usd"] < 0:
                b["n_losses"] += 1
            b["_pnl_usds"].append(t["pnl_usd"])
            if t["pnl_pct"] is not None:
                try:
                    b["_pnl_pcts"].append(float(t["pnl_pct"]))
                except Exception:
                    pass
            if t["hold_days"] is not None:
                try:
                    b["_holds"].append(float(t["hold_days"]))
                except Exception:
                    pass
            b["_trips"].append(t)

    # Finalise per-class rows: aggregate stats + verdict.
    rows: list[dict] = []
    for cls, b in by_class.items():
        n = b["n_trips"]
        total_pnl = round(sum(b["_pnl_usds"]), 4)
        avg_pnl_usd = round(total_pnl / n, 4) if n else None
        avg_pnl_pct = (round(sum(b["_pnl_pcts"]) / len(b["_pnl_pcts"]), 4)
                       if b["_pnl_pcts"] else None)
        median_hold = _median(b["_holds"])
        wr = (b["n_wins"] / n * 100.0) if n else 0.0
        verdict = _per_class_verdict(
            wr,
            pool_wr if pool_wr is not None else 0.0,
            n,
        )
        rows.append({
            "class": cls,
            "n_trips": n,
            "n_wins": b["n_wins"],
            "n_losses": b["n_losses"],
            "win_rate_pct": round(wr, 2),
            "total_pnl_usd": total_pnl,
            "avg_pnl_usd": avg_pnl_usd,
            "avg_pnl_pct": avg_pnl_pct,
            "median_hold_days": median_hold,
            "verdict": verdict,
        })

    # Stable sort: total_pnl_usd DESC (best-earning class first),
    # ties broken by n_trips DESC, then class name asc.
    rows.sort(key=lambda r: (-r["total_pnl_usd"], -r["n_trips"], r["class"]))

    # Best / worst classes by total_pnl (sample-size-gated to STABLE
    # rows only — a single-trip class shouldn't claim "best earner").
    stable_rows = [r for r in rows if r["n_trips"] >= STABLE_MIN_TRIPS_PER_CLASS]
    best_class = stable_rows[0]["class"] if stable_rows else None
    worst_class = (stable_rows[-1]["class"]
                   if stable_rows and stable_rows[-1]["total_pnl_usd"] < 0
                   else None)
    # Loudest BIASED class (precedence: WINNER then LOSER; ties broken
    # by n_trips and the documented precedence list).
    biased_winners = [r for r in rows if r["verdict"] == "BIASED_WINNER"]
    biased_losers = [r for r in rows if r["verdict"] == "BIASED_LOSER"]
    if biased_winners:
        bw_sorted = sorted(
            biased_winners,
            key=lambda r: (-r["n_trips"],
                           _DOMINANT_PRECEDENCE.index(r["class"])
                           if r["class"] in _DOMINANT_PRECEDENCE else 999),
        )
        top_biased_winner = bw_sorted[0]["class"]
    else:
        top_biased_winner = None
    if biased_losers:
        bl_sorted = sorted(
            biased_losers,
            key=lambda r: (-r["n_trips"],
                           _DOMINANT_PRECEDENCE.index(r["class"])
                           if r["class"] in _DOMINANT_PRECEDENCE else 999),
        )
        top_biased_loser = bl_sorted[0]["class"]
    else:
        top_biased_loser = None

    # State ladder — same shape as loser/winner_autopsy / track_record.
    if n_scored == 0:
        state = "NO_DATA"
    elif any(r["n_trips"] >= STABLE_MIN_TRIPS_PER_CLASS for r in rows):
        state = "STABLE"
    else:
        state = "EMERGING"

    # Headline.
    if state == "NO_DATA":
        headline = "No closed round-trips yet — nothing to classify."
    elif state == "EMERGING":
        n_classes = len(rows)
        headline = (
            f"Emerging — {n_scored} closed round-trip"
            f"{'' if n_scored == 1 else 's'} across {n_classes} catalyst "
            f"class{'es' if n_classes != 1 else ''}; per-class verdicts "
            f"withheld below {STABLE_MIN_TRIPS_PER_CLASS} trips each.")
    else:
        # STABLE — at least one class crossed the gate.
        if top_biased_winner and top_biased_loser:
            headline = (
                f"{top_biased_winner} biases WIN, {top_biased_loser} biases "
                f"LOSE — repeat the former's catalyst, avoid the latter's.")
        elif top_biased_winner:
            headline = (
                f"{top_biased_winner} is the earning catalyst class — "
                f"win-rate ≥{BIASED_WR_DELTA_PCT:.0f}% above pool.")
        elif top_biased_loser:
            headline = (
                f"{top_biased_loser} is the bleeding catalyst class — "
                f"win-rate ≥{BIASED_WR_DELTA_PCT:.0f}% below pool.")
        elif best_class and worst_class and best_class != worst_class:
            headline = (
                f"No class flips BIASED vs pool — best earner "
                f"{best_class}, worst {worst_class}.")
        elif best_class:
            headline = (
                f"No class flips BIASED vs pool — best earner "
                f"{best_class}, no class bleeds.")
        else:
            headline = "No class crossed the stable-sample gate."

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "headline": headline,
        "n_round_trips": n_rts,
        "n_scored": n_scored,
        "pool_win_rate_pct": (round(pool_wr, 2)
                              if pool_wr is not None else None),
        "stable_min_trips_per_class": STABLE_MIN_TRIPS_PER_CLASS,
        "biased_wr_delta_pct": BIASED_WR_DELTA_PCT,
        "best_class": best_class,
        "worst_class": worst_class,
        "top_biased_winner": top_biased_winner,
        "top_biased_loser": top_biased_loser,
        "classes": rows,
        "taxonomy": list(_ALL_CLASSES) + ["UNCLASSIFIED"],
    }


if __name__ == "__main__":  # smoke against the live DB
    import json
    from paper_trader.store import get_store
    s = get_store()
    rep = build_catalyst_class_autopsy(list(reversed(s.recent_trades(2000))))
    print(json.dumps(rep, indent=2, default=str))
