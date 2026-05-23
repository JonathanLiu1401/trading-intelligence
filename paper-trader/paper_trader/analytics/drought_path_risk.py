"""Drought path-risk — what the *shape* of the equity curve looked like
while the bot was frozen.

``analytics/decision_drought.py`` answers "while the bot wasn't trading, did
the portfolio drift vs. the market?" — it reports point-to-point ``port_pct``
/ ``spy_pct`` / ``alpha_pct`` for each drought. That tells you the
**endpoint** difference between drought start and drought end, but it is
**path-blind**:

* A 47h drought that ended -2.4% via a smooth -2.4% grind looks identical to
  one that bottomed at -6.5%, recovered to -2.4%, dipped again, and limped
  to the close. Endpoint alpha is the same. The desk's recovery options
  during that drought are NOT the same.
* A drought that ended FLAT (0% alpha) could have spent two hours -4%
  underwater mid-drought — a near-miss panic dodge the trader survived
  blind. Endpoint alpha doesn't see it.

This builder composes ``build_decision_drought.current_drought`` (verbatim
single source of truth — AGENTS.md #10, the ``idle_opportunity`` precedent)
and walks the ``equity_curve`` points that fall **inside** the drought window
to compute the missing path-shape view:

* ``start_equity`` / ``current_equity`` — the same anchor and tip the
  parent drought reports (here for cross-check; do not re-derive).
* ``peak_equity`` / ``peak_ts`` — the highest total_value inside the drought.
* ``trough_equity`` / ``trough_ts`` — the lowest total_value inside the
  drought.
* ``intra_drought_drawdown_pct`` — ``(trough - peak) / peak * 100`` — the
  worst peak-to-trough dip *anchored to the highest mid-drought watermark*,
  so a "drought that started high, bottomed at -6%, ended -2%" is correctly
  measured as a -6%-class event, not a -2% event. Negative number (or 0).
* ``intra_drought_max_gain_pct`` — ``(peak - start) / start * 100`` — best
  mid-drought peak above the start anchor. Positive number (or 0).
* ``range_pct`` — ``(peak - trough) / start * 100`` — total path span
  expressed against the start. Captures whipsaw-class events even when
  start≈end (a no-net-move drought that nevertheless thrashed).
* ``end_to_start_pct`` — ``(current - start) / start * 100`` — net move,
  identical convention to the parent drought's ``portfolio_pct`` (here for
  cross-check; do not re-derive).
* ``n_equity_samples`` — how many equity points fell inside the drought
  window — used as the STABLE gate.

Verdict ladder (STABLE only; ``n_equity_samples >= 3``):

| Verdict | Trigger | What it says |
|---------|---------|--------------|
| ``WHIPSAW_TRAP`` | intra_drought_drawdown ≤ -2.0% AND range ≥ 4.0% | the book got swung; a tradeable dip happened mid-drought you were blind to |
| ``LIFTED_BLIND`` | intra_drought_max_gain ≥ +2.0% | the book peaked materially above start at some point — lucky tape (whether or not it gave some back) |
| ``DODGED_DROP`` | intra_drought_drawdown ≤ -2.0% AND end_to_start ≥ -0.5% | you bottomed materially then recovered while frozen — survived blind |
| ``SLOW_BLEED`` | end_to_start ≤ -2.0% AND range < 4.0% | smooth, monotonic-ish slide; the parent ``port_pct`` captures it |
| ``QUIET_DROUGHT`` | range < 1.0% | nothing happened — the silence-when-nothing-actionable verdict |
| ``MIXED`` | none of the above arms triggers cleanly | catch-all |

LIFTED gates on ``max_gain`` (peak above start) rather than ``end_to_start``
because on a monotonic-up path ``start`` IS the trough, so DD ≈
-gain / (1+gain) — any ≥+2% net-up gain pushes DD past -2.0% into
DODGED's actionable-DD gate. ``max_gain`` is the path-shape-invariant
"did the book ever go up materially" signal. LIFTED beats DODGED in
precedence: when both fire (a V-shape that bottoms scary and recovers
beyond start), the operator's headline reading is "+net-up", not the
mid-drought scare.

Below STABLE: ``state=INSUFFICIENT``, numerics still emitted (the
``risk_adjusted_returns`` two-tier idiom — better to show a thin
intra-drought picture than a black box, and the operator already trusts
``n_equity_samples`` to gate trust).

When no ``current_drought`` is ongoing — ``state="NO_DROUGHT"`` and verdict
withheld. The silence-when-nothing-actionable precedent
(``_host_pulse_line`` / ``_macro_calendar_chat_lines``).

Observational only — never gates Opus, never injected into the decision
prompt, no caps (AGENTS.md invariants #2/#12 — the ``shadow_vs_claude`` /
``idle_opportunity`` / ``capital_paralysis`` precedent). Pure: feed it the
``decision_drought`` result + the equity_curve list; never raises on garbage
inputs. ``now`` is injectable for tests.
"""
from __future__ import annotations

from datetime import datetime, timezone

# STABLE gate — below this we still emit numerics but withhold the verdict
# (the risk_adjusted_returns EMIT_MIN_DAYS / STABLE_MIN_DAYS pattern). Three
# is the minimum to see a peak-and-trough on a path (start + one middle +
# end), so emitting a verdict on 1-2 samples would describe a line, not a
# path.
STABLE_MIN_SAMPLES = 3

# Verdict thresholds. Pegged to the parent decision_drought verdict scale
# (BLEEDING fires at involuntary_alpha_bleed_pct <= -1.0%; here we anchor
# on a 2.0% intra-drought move which is the level a desk would actively
# trade out of vs a 1% noise band). Range threshold of 4.0% picks the
# whipsaw class — twice the actionable single-leg move.
DRAWDOWN_ACTIONABLE_PCT = -2.0
RANGE_WHIPSAW_PCT = 4.0
LIFT_MATERIAL_PCT = 2.0
QUIET_RANGE_PCT = 1.0
RECOVERY_THRESHOLD_PCT = -0.5


def _parse_ts(ts) -> datetime | None:
    """Tolerate aware/naive ISO strings — same convention as
    ``decision_drought._parse_ts`` and ``idle_opportunity._parse_ts``."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).strip().replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _drought_slim(d: dict) -> dict:
    """Trim the drought block to the fields the path panel echoes."""
    return {
        "kind": d.get("kind"),
        "start": d.get("start"),
        "end": d.get("end"),
        "duration_hours": d.get("duration_hours"),
        "n_cycles": d.get("n_cycles"),
        "no_decision_pct": d.get("no_decision_pct"),
        "portfolio_pct": d.get("portfolio_pct"),
        "spy_pct": d.get("spy_pct"),
        "alpha_pct": d.get("alpha_pct"),
        "ongoing": d.get("ongoing"),
    }


def _headline(state: str, verdict: str | None, drought: dict | None,
              dd_pct: float | None, range_pct: float | None,
              end_to_start_pct: float | None,
              n_samples: int) -> str:
    """One-sentence operator headline matching the
    ``_host_pulse_line`` / ``_capital_pulse_line`` voice."""
    if state == "NO_DATA":
        return ("Drought path: no decisions recorded yet.")
    if state == "NO_DROUGHT":
        return ("Drought path: no ongoing drought — the trader is filling "
                "normally; nothing to read.")
    dur = (drought or {}).get("duration_hours")
    dur_s = f"{dur:.1f}h" if isinstance(dur, (int, float)) else "?h"
    if state == "INSUFFICIENT":
        return (
            f"Drought path: {dur_s} drought has only {n_samples} equity "
            "sample(s) — too thin to read the path shape (verdict withheld).")
    # STABLE arms
    dd_s = f"{dd_pct:.2f}%" if isinstance(dd_pct, (int, float)) else "?%"
    rng_s = f"{range_pct:.2f}%" if isinstance(range_pct, (int, float)) else "?%"
    ets_s = (f"{end_to_start_pct:+.2f}%"
             if isinstance(end_to_start_pct, (int, float)) else "?%")
    if verdict == "WHIPSAW_TRAP":
        return (
            f"Drought path: WHIPSAW — {dur_s} blind, intra-drought DD {dd_s} "
            f"with {rng_s} range; a tradeable dip happened mid-drought you "
            "couldn't act on.")
    if verdict == "DODGED_DROP":
        return (
            f"Drought path: DODGED — {dur_s} blind, bottomed at {dd_s} "
            f"intra-drought DD but recovered to {ets_s}; survived a "
            "near-miss while frozen.")
    if verdict == "LIFTED_BLIND":
        return (
            f"Drought path: LIFTED — {dur_s} blind, book peaked above start "
            f"(net {ets_s}) while paralyzed — lucky tape, not earned.")
    if verdict == "SLOW_BLEED":
        return (
            f"Drought path: SLOW BLEED — {dur_s} blind, smooth slide to "
            f"{ets_s} (range {rng_s} < 4%); the parent drought's "
            "alpha_pct captures it.")
    if verdict == "QUIET_DROUGHT":
        return (
            f"Drought path: QUIET — {dur_s} blind, range {rng_s} < 1%; "
            "nothing actionable happened.")
    # MIXED fallback
    return (
        f"Drought path: MIXED — {dur_s} blind, DD {dd_s}, range {rng_s}, "
        f"net {ets_s}; no clean pattern arm.")


def _classify(end_to_start_pct: float, dd_pct: float,
              range_pct: float, max_gain_pct: float) -> str:
    """Verdict ladder — explicit precedence (tested in
    ``TestVerdictPrecedence``).

    The order is deliberate:

    1. **WHIPSAW** dominates everything else when both gates fire — a
       big-range dip is the operator's headline regardless of where the
       path ended.
    2. **LIFTED** beats DODGED. On monotonic-up paths the start IS the
       trough so DD ≈ -gain/(1+gain) — any meaningful net gain would
       otherwise be misclassified as DODGED (DD-actionable + recovered).
       Gating LIFTED on ``max_gain_pct`` (peak above start) rather than
       ``end_to_start`` makes it path-shape-invariant: "did the book
       ever go up materially?" An operator reading a V-shape that
       bottomed -3% and ended +3% pays attention to the +3% net first.
    3. **DODGED** then catches the V-shape that recovered to roughly
       start (no material net up — LIFTED already took those).
    4. **SLOW_BLEED** catches monotonic-ish net-down with tight range.
    5. **QUIET** catches no-net-move drougths (range < 1%).
    6. **MIXED** catches everything else.
    """
    if dd_pct <= DRAWDOWN_ACTIONABLE_PCT and range_pct >= RANGE_WHIPSAW_PCT:
        return "WHIPSAW_TRAP"
    if max_gain_pct >= LIFT_MATERIAL_PCT:
        return "LIFTED_BLIND"
    if dd_pct <= DRAWDOWN_ACTIONABLE_PCT and \
            end_to_start_pct >= RECOVERY_THRESHOLD_PCT:
        return "DODGED_DROP"
    if end_to_start_pct <= DRAWDOWN_ACTIONABLE_PCT and \
            range_pct < RANGE_WHIPSAW_PCT:
        return "SLOW_BLEED"
    if range_pct < QUIET_RANGE_PCT:
        return "QUIET_DROUGHT"
    return "MIXED"


def build_drought_path_risk(decision_drought_result: dict | None,
                            equity_curve: list[dict] | None,
                            now: datetime | None = None) -> dict:
    """Intra-drought path-shape — composes ``build_decision_drought`` and
    the equity curve points inside the current-drought window.

    Pure. ``decision_drought_result`` is the dict from
    ``build_decision_drought`` (verbatim SSOT). ``equity_curve`` is a list
    of dicts with ``timestamp`` / ``total_value`` keys (the
    ``store.equity_curve`` shape). Returns a JSON-ready dict; never raises
    on garbage inputs. ``now`` is injectable for tests.
    """
    now = now or datetime.now(timezone.utc)
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "NO_DATA",
        "verdict": None,
        "headline": "Drought path: no decisions recorded yet.",
        "drought": None,
        "n_equity_samples": 0,
        "start_equity": None,
        "current_equity": None,
        "peak_equity": None,
        "peak_ts": None,
        "trough_equity": None,
        "trough_ts": None,
        "intra_drought_drawdown_pct": None,
        "intra_drought_max_gain_pct": None,
        "range_pct": None,
        "end_to_start_pct": None,
    }

    # ── 1. Gate on the parent drought block ─────────────────────────────
    if not decision_drought_result or not isinstance(
            decision_drought_result, dict):
        return out

    drought = decision_drought_result.get("current_drought")
    if drought is None:
        out["state"] = "NO_DROUGHT"
        out["headline"] = _headline("NO_DROUGHT", None, None, None, None,
                                    None, 0)
        return out
    if not isinstance(drought, dict):
        # Garbage shape — degrade to NO_DATA (same defence as the
        # unparseable-start branch). NOT NO_DROUGHT, because we couldn't
        # confirm there is no drought; we just couldn't read what's there.
        return out
    if not drought.get("ongoing"):
        out["state"] = "NO_DROUGHT"
        out["headline"] = _headline("NO_DROUGHT", None, None, None, None,
                                    None, 0)
        return out

    start_ts = _parse_ts(drought.get("start"))
    end_ts = _parse_ts(drought.get("end")) or now
    if start_ts is None:
        # Defensive: a drought block with no parseable start can't anchor a
        # window — degrade to NO_DATA. (Mirror idle_opportunity's defence.)
        return out

    out["drought"] = _drought_slim(drought)

    # ── 2. Walk equity points inside [start_ts, end_ts] ─────────────────
    inside: list[tuple[datetime, float]] = []
    if not isinstance(equity_curve, (list, tuple)):
        equity_curve = []
    for e in equity_curve:
        if not isinstance(e, dict):
            continue
        ts = _parse_ts(e.get("timestamp"))
        tv = _safe_float(e.get("total_value"))
        if ts is None or tv is None or tv <= 0:
            continue
        if start_ts <= ts <= end_ts:
            inside.append((ts, tv))
    inside.sort(key=lambda r: r[0])
    out["n_equity_samples"] = len(inside)

    if not inside:
        # Drought ongoing but the equity curve has zero samples inside the
        # window — common right at the start of a drought before the first
        # post-fill point lands. Emit INSUFFICIENT with the drought echo so
        # the operator sees the panel rendering, just no path yet.
        out["state"] = "INSUFFICIENT"
        out["headline"] = _headline("INSUFFICIENT", None, drought, None,
                                    None, None, 0)
        return out

    start_tv = inside[0][1]
    current_tv = inside[-1][1]
    peak_ts, peak_tv = max(inside, key=lambda r: r[1])
    trough_ts, trough_tv = min(inside, key=lambda r: r[1])

    out["start_equity"] = round(start_tv, 4)
    out["current_equity"] = round(current_tv, 4)
    out["peak_equity"] = round(peak_tv, 4)
    out["peak_ts"] = peak_ts.isoformat(timespec="seconds")
    out["trough_equity"] = round(trough_tv, 4)
    out["trough_ts"] = trough_ts.isoformat(timespec="seconds")

    # Path metrics. Drawdown is peak-to-trough so a path that went
    # start=100 → 105 → 95 → 98 is correctly measured as
    # (95-105)/105 = -9.52% DD, not (95-100)/100 = -5%.
    if peak_tv > 0:
        out["intra_drought_drawdown_pct"] = round(
            (trough_tv - peak_tv) / peak_tv * 100.0, 3)
    if start_tv > 0:
        out["intra_drought_max_gain_pct"] = round(
            max(0.0, (peak_tv - start_tv) / start_tv * 100.0), 3)
        out["range_pct"] = round((peak_tv - trough_tv) / start_tv * 100.0, 3)
        out["end_to_start_pct"] = round(
            (current_tv - start_tv) / start_tv * 100.0, 3)

    # ── 3. State + verdict ──────────────────────────────────────────────
    if out["n_equity_samples"] < STABLE_MIN_SAMPLES:
        out["state"] = "INSUFFICIENT"
        out["headline"] = _headline(
            "INSUFFICIENT", None, drought,
            out["intra_drought_drawdown_pct"], out["range_pct"],
            out["end_to_start_pct"], out["n_equity_samples"])
        return out

    out["state"] = "STABLE"
    verdict = _classify(
        out["end_to_start_pct"] or 0.0,
        out["intra_drought_drawdown_pct"] or 0.0,
        out["range_pct"] or 0.0,
        out["intra_drought_max_gain_pct"] or 0.0,
    )
    out["verdict"] = verdict
    out["headline"] = _headline(
        "STABLE", verdict, drought,
        out["intra_drought_drawdown_pct"], out["range_pct"],
        out["end_to_start_pct"], out["n_equity_samples"])
    return out


if __name__ == "__main__":  # smoke test against the live DB
    import json
    from paper_trader.store import get_store
    from paper_trader.analytics.decision_drought import build_decision_drought
    s = get_store()
    dr = build_decision_drought(s.recent_decisions(limit=3000),
                                s.equity_curve(limit=5000))
    rep = build_drought_path_risk(dr, s.equity_curve(limit=5000))
    print(json.dumps(rep, indent=2, default=str))
