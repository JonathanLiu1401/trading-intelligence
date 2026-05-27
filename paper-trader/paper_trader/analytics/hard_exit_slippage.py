r"""Hard-exit threshold slippage — the calibration follow-up to
``hard_exit_summary``.

``/api/hard-exit-summary`` (analytics/hard_exit_summary.py) counts how
many SL/TP fires hit and which tickers are repeat-offenders. It answers
"is the mechanical discipline firing?" — it does NOT answer "are the
*thresholds themselves* well-calibrated?".

The bot's hard SL/TP machinery stamps a fixed percentage threshold at
entry (2/3% standard, 4/6% leveraged — see
``strategy._check_and_execute_hard_exits``). When price gaps past that
threshold, the fill is whatever the next mark reports — which can be
materially away from the threshold itself. Live evidence (2026-05-26):
MU TP fired at threshold $773.31 (a +3% TP off the $750.79 entry) with
an actual fill of $889.50 — the threshold captured +3% but the tape
gave +18%. The "lucky overshoot" cost the bot ~+15 pp it would have
captured by simply staying in the position another hour.

This builder fills that calibration gap. For each hard SL / TP fire in
the recent trade ledger, we parse the threshold out of the canonical
reason string emitted by the strategy:

    HARD_TP: price 889.50 >= threshold 773.31
    HARD_SL: price 750.00 <= threshold 760.00

and compute the slippage magnitude:

  * **TP_slippage_pct** = (fill - threshold) / threshold × 100 — always
    ≥ 0 by construction (fill is guaranteed ≥ threshold).
  * **SL_slippage_pct** = (threshold - fill) / threshold × 100 — always
    ≥ 0 by construction (fill is guaranteed ≤ threshold).

Slippage of ≈ 0 means the threshold caught the move at-the-tick — the
threshold was working as designed, the bot got the +/-3% it stamped in.
Material slippage means the tape gapped past the trigger, and the
threshold setting did NOT capture what the move actually paid (or cost).

Top-level verdict ladder:

  * ``NO_HARD_EXITS``      — no SL or TP fires in the ledger.
  * ``INSUFFICIENT``       — fewer than ``MIN_FOR_VERDICT`` hard exits.
  * ``LUCKY_OVERSHOOTS``   — median TP slippage ≥ ``LUCKY_TP_THRESHOLD_PCT``.
    TPs are routinely gapping past the trigger by a wide margin — the
    bot is being saved by the tape. Means the +3% TP is materially
    *under-calibrated* for the realized volatility of these names.
  * ``UNLUCKY_GAPS``       — median SL slippage ≥ ``UNLUCKY_SL_THRESHOLD_PCT``.
    SLs are routinely getting blown through by gap-downs; realized
    losses are larger than the stop intended.
  * ``CLEAN_FILLS``        — both medians are below their thresholds.
    The bot's mechanical exits are pricing at the trigger.

Pure builder: walks an already-fetched newest-first ledger payload (the
``store.recent_trades(N)`` shape ``hard_exit_summary`` consumes), never
network/store side effects, never raises (fault → valid ERROR envelope,
same contract as ``hard_exit_summary``).

Read-only — never writes, never trains, never gates Opus. Companion to
``hard_exit_summary`` (count + discipline) and ``exit_proximity`` (the
forward-looking open-position threshold view); together they answer the
three orthogonal questions: is the discipline firing? are the thresholds
calibrated? are the OPEN lots near a fire?
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median


# Substring markers strategy._check_and_execute_hard_exits writes into
# the trade reason. Locked at module level so a test (or any future
# operator tool) reads the same constants the live emitter uses —
# matches the hard_exit_summary discipline.
HARD_SL_MARKER = "HARD_SL"
HARD_TP_MARKER = "HARD_TP"


# Parse the threshold + price out of the canonical reason string. The
# emitter is strategy.py L1361:
#   f"{exit_type}: price {price:.2f} {'<=' if is_sl else '>='} threshold {threshold:.2f}"
# Float pattern allows scientific edge cases the emitter would never
# produce so the regex never silently mis-parses a future format tweak.
_REASON_RE = re.compile(
    r"(?:HARD_(?:SL|TP)).*?"
    r"price\s+(?P<price>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"\s+(?:<=|>=)\s+"
    r"threshold\s+(?P<threshold>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


# Verdict thresholds — picked so that the noise floor of a non-gappy
# name fill (the bot's typical $50–$300 stock on a 0.1% mark cadence
# tick) doesn't trigger a CLAUDE-CRYING-WOLF false positive, while real
# gap moves like the live MU +15% case fire decisively.
#
# 2.0% picked from the live data: the MU TP fire was +15.0% slippage
# (manifestly material); a 3% TP stamped at entry with a 0.5% mark-to-
# mark drift between cycles never produces 2% slippage. So 2% sits well
# below the live evidence and well above mark noise.
LUCKY_TP_THRESHOLD_PCT = 2.0
UNLUCKY_SL_THRESHOLD_PCT = 2.0

# Minimum count before a directional verdict is published — same noise
# floor as hard_exit_summary.MIN_FOR_VERDICT (a single fire never names
# a regime).
MIN_FOR_VERDICT = 3


def _classify_exit_reason(reason):
    """Return ``"HARD_SL"``, ``"HARD_TP"``, or ``None`` for a SELL trade's
    reason string. Pure, never raises. Mirrors
    ``hard_exit_summary._classify_exit_reason`` exactly so the two
    builders never disagree on what is "mechanical"."""
    if not reason:
        return None
    if HARD_SL_MARKER in reason:
        return "HARD_SL"
    if HARD_TP_MARKER in reason:
        return "HARD_TP"
    return None


def _parse_threshold(reason):
    """Return ``(price, threshold)`` floats parsed from the reason string,
    or ``(None, None)`` on any parse failure. Pure, never raises."""
    if not reason:
        return (None, None)
    m = _REASON_RE.search(str(reason))
    if not m:
        return (None, None)
    try:
        return (float(m.group("price")), float(m.group("threshold")))
    except (TypeError, ValueError):
        return (None, None)


def _slippage_pct(*, fill_price, threshold, is_sl):
    """Compute slippage % magnitude. For TP: fill above threshold.
    For SL: fill below threshold. Both branches return ≥ 0 on a
    well-formed mechanical exit. Returns ``None`` on degenerate input
    (non-positive threshold). Pure, never raises."""
    try:
        fp = float(fill_price)
        th = float(threshold)
    except (TypeError, ValueError):
        return None
    if th <= 0:
        return None
    if is_sl:
        return (th - fp) / th * 100.0
    return (fp - th) / th * 100.0


def _row_summary(t, *, slippage_pct, price, threshold, cls):
    """Compact per-trade row. Keeps the operator-facing fields a trader
    cares about plus the parsed numeric slippage for downstream
    aggregation — same compactness as hard_exit_summary._trade_summary."""
    return {
        "ticker": t.get("ticker"),
        "qty": t.get("qty"),
        "fill_price": (round(price, 4) if isinstance(price, (int, float))
                       else None),
        "threshold": (round(threshold, 4) if isinstance(threshold, (int, float))
                      else None),
        "slippage_pct": (round(slippage_pct, 4)
                         if isinstance(slippage_pct, (int, float)) else None),
        "exit_type": cls,
        "timestamp": t.get("timestamp"),
        "reason": (t.get("reason") or "")[:200],
    }


def _verdict(*, n_tp, n_sl, tp_slip_median, sl_slip_median):
    """Return ``(verdict, headline)`` given the parsed sample.

    Precedence:
      * NO_HARD_EXITS — empty.
      * INSUFFICIENT — fewer than MIN_FOR_VERDICT total mechanical
        fires (either side).
      * LUCKY_OVERSHOOTS — TP median ≥ threshold AND we have ≥ MIN_FOR_VERDICT
        TPs. Prefer this over UNLUCKY_GAPS when both fire because the
        symptom is more frequently observed on a leveraged/gappy book
        (the TP side fires more often than the SL side when the bot's
        edge is real; the asymmetry is the operator-actionable read).
      * UNLUCKY_GAPS — SL median ≥ threshold AND we have ≥ MIN_FOR_VERDICT SLs.
      * CLEAN_FILLS — otherwise.
    """
    total = n_tp + n_sl
    if total == 0:
        return ("NO_HARD_EXITS",
                "No mechanical exits in the ledger — slippage calibration "
                "untested.")
    if total < MIN_FOR_VERDICT:
        return ("INSUFFICIENT",
                f"Only {total} hard exit(s) so far — need ≥{MIN_FOR_VERDICT} "
                "before a slippage verdict.")
    lucky = (
        n_tp >= MIN_FOR_VERDICT
        and tp_slip_median is not None
        and tp_slip_median >= LUCKY_TP_THRESHOLD_PCT
    )
    unlucky = (
        n_sl >= MIN_FOR_VERDICT
        and sl_slip_median is not None
        and sl_slip_median >= UNLUCKY_SL_THRESHOLD_PCT
    )
    if lucky:
        return ("LUCKY_OVERSHOOTS",
                f"TPs gapping past the trigger: median +{tp_slip_median:.1f}% "
                f"slippage over {n_tp} fire(s) — the +3% TP threshold is "
                "materially under-calibrated for these names' realized "
                "volatility. Tape, not the threshold, is capturing the move.")
    if unlucky:
        return ("UNLUCKY_GAPS",
                f"SLs blowing through the stop: median +{sl_slip_median:.1f}% "
                f"slippage over {n_sl} fire(s) — realized losses larger than "
                "the stop intended. Names gap on the open and the SL fills at "
                "the next mark.")
    return ("CLEAN_FILLS",
            f"Mechanical fills pricing at the trigger: TP median "
            f"{(tp_slip_median or 0.0):.2f}% / SL median "
            f"{(sl_slip_median or 0.0):.2f}% over {total} fire(s). Threshold "
            "settings holding the line.")


def _percentile(values, q):
    """Return the q-th percentile (0..1) of a numeric list, or None if
    empty. Implemented locally so the module has no numpy/statistics-
    quantile dependency (matches the hard_exit_summary purity)."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    s = sorted(float(v) for v in values)
    n = len(s)
    idx = q * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def build_hard_exit_slippage(
    trades_newest_first,
    *,
    now=None,
):
    """Operator snapshot of mechanical SL/TP threshold *calibration*
    over the trade ledger we were given.

    ``trades_newest_first`` is the store-native ordering. We walk every
    SELL trade, classify by reason marker (same logic as
    hard_exit_summary), parse the threshold + price out of the reason,
    and compute the slippage magnitude. Discretionary SELLs (no marker)
    are skipped — they have no threshold to calibrate.

    Always returns a full dict shape; never raises (fault → ERROR
    envelope). ``now`` is injectable for deterministic tests."""
    now_ = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    try:
        tp_rows = []
        sl_rows = []
        per_ticker = defaultdict(lambda: {"tp": [], "sl": []})
        last_tp = None
        last_sl = None
        # Parse-failure tally so an operator can spot a strategy-format
        # change (would silently zero out the verdict otherwise — the
        # honest-deficiency precedent feed_health follows).
        parse_failed = 0

        for t in trades_newest_first or []:
            if not isinstance(t, dict):
                continue
            action = (t.get("action") or "").upper()
            if not action.startswith("SELL"):
                continue
            cls = _classify_exit_reason(t.get("reason"))
            if cls is None:
                continue
            price, threshold = _parse_threshold(t.get("reason"))
            is_sl = cls == "HARD_SL"
            slip = _slippage_pct(
                fill_price=price, threshold=threshold, is_sl=is_sl,
            )
            if slip is None:
                parse_failed += 1
                continue
            ticker = (t.get("ticker") or "").upper()
            row = _row_summary(
                t, slippage_pct=slip, price=price, threshold=threshold,
                cls=cls,
            )
            if is_sl:
                sl_rows.append(row)
                if ticker:
                    per_ticker[ticker]["sl"].append(slip)
                if last_sl is None:
                    last_sl = row
            else:
                tp_rows.append(row)
                if ticker:
                    per_ticker[ticker]["tp"].append(slip)
                if last_tp is None:
                    last_tp = row

        tp_slips = [r["slippage_pct"] for r in tp_rows]
        sl_slips = [r["slippage_pct"] for r in sl_rows]
        tp_median = median(tp_slips) if tp_slips else None
        sl_median = median(sl_slips) if sl_slips else None
        tp_p90 = _percentile(tp_slips, 0.90)
        sl_p90 = _percentile(sl_slips, 0.90)
        tp_max = max(tp_slips) if tp_slips else None
        sl_max = max(sl_slips) if sl_slips else None

        verdict, headline = _verdict(
            n_tp=len(tp_rows), n_sl=len(sl_rows),
            tp_slip_median=tp_median, sl_slip_median=sl_median,
        )

        # Per-ticker rollup — surface the worst-offending TICKER so
        # the operator can spot a single name that's distorting the
        # aggregate (a single SOXL/TQQQ fire can move the median by
        # several pp). Sort by combined fire-count desc then by max
        # slippage desc; cap at 10.
        per_ticker_out = []
        for tk, d in per_ticker.items():
            tp_l = d["tp"]
            sl_l = d["sl"]
            total = len(tp_l) + len(sl_l)
            per_ticker_out.append({
                "ticker": tk,
                "n_tp": len(tp_l),
                "n_sl": len(sl_l),
                "n_total": total,
                "max_tp_slippage_pct": (round(max(tp_l), 4) if tp_l else None),
                "max_sl_slippage_pct": (round(max(sl_l), 4) if sl_l else None),
                "median_tp_slippage_pct": (
                    round(median(tp_l), 4) if tp_l else None
                ),
                "median_sl_slippage_pct": (
                    round(median(sl_l), 4) if sl_l else None
                ),
            })
        per_ticker_out.sort(
            key=lambda r: (
                r["n_total"],
                max(
                    r["max_tp_slippage_pct"] or 0.0,
                    r["max_sl_slippage_pct"] or 0.0,
                ),
            ),
            reverse=True,
        )
        per_ticker_out = per_ticker_out[:10]

        return {
            "as_of": now_,
            "state": "OK",
            "verdict": verdict,
            "headline": headline,
            "n_hard_tp": len(tp_rows),
            "n_hard_sl": len(sl_rows),
            "n_total_hard": len(tp_rows) + len(sl_rows),
            "n_parse_failed": parse_failed,
            "tp_slippage_median_pct": (
                round(tp_median, 4) if tp_median is not None else None
            ),
            "tp_slippage_p90_pct": (
                round(tp_p90, 4) if tp_p90 is not None else None
            ),
            "tp_slippage_max_pct": (
                round(tp_max, 4) if tp_max is not None else None
            ),
            "sl_slippage_median_pct": (
                round(sl_median, 4) if sl_median is not None else None
            ),
            "sl_slippage_p90_pct": (
                round(sl_p90, 4) if sl_p90 is not None else None
            ),
            "sl_slippage_max_pct": (
                round(sl_max, 4) if sl_max is not None else None
            ),
            "last_hard_tp": last_tp,
            "last_hard_sl": last_sl,
            "per_ticker": per_ticker_out,
            # Echo live thresholds so a dashboard consumer can render
            # them next to the verdict — same echo discipline
            # hard_exit_summary follows.
            "lucky_tp_threshold_pct": LUCKY_TP_THRESHOLD_PCT,
            "unlucky_sl_threshold_pct": UNLUCKY_SL_THRESHOLD_PCT,
            "min_for_verdict": MIN_FOR_VERDICT,
        }
    except Exception as e:
        return {
            "as_of": now_,
            "state": "ERROR",
            "verdict": "ERROR",
            "headline": f"hard_exit_slippage failed: {e}",
            "error": str(e),
            "n_hard_tp": 0,
            "n_hard_sl": 0,
            "n_total_hard": 0,
            "n_parse_failed": 0,
            "tp_slippage_median_pct": None,
            "tp_slippage_p90_pct": None,
            "tp_slippage_max_pct": None,
            "sl_slippage_median_pct": None,
            "sl_slippage_p90_pct": None,
            "sl_slippage_max_pct": None,
            "last_hard_tp": None,
            "last_hard_sl": None,
            "per_ticker": [],
            "lucky_tp_threshold_pct": LUCKY_TP_THRESHOLD_PCT,
            "unlucky_sl_threshold_pct": UNLUCKY_SL_THRESHOLD_PCT,
            "min_for_verdict": MIN_FOR_VERDICT,
        }
