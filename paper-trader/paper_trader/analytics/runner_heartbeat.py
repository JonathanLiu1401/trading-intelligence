"""Runner heartbeat — is the trading loop itself alive?

Every other diagnostic on the desk (``decision_health``/``-forensics``/
``-drought``/``-reliability``, ``feed_health``, ``build-info``) reasons over
rows that *exist* in ``decisions`` (or over a code SHA / article age). None
close a verdict on ``now - max(decisions.timestamp)`` vs the runner's
expected cadence — so a dead or wedged ``paper_trader.runner`` is invisible:
``decision_drought``'s ongoing drought freezes its ``duration_hours`` the
instant rows stop appearing, ``feed_health.blind_streak`` cannot grow without
new rows, and ``build-info.stale`` only catches a stale *code* SHA. This
detector closes that gap.

Pure & offline: the builder takes ``last_decision_ts`` / ``market_open`` /
``now``; the endpoint owns the ``store.recent_decisions(1)`` read and the
``market.is_market_open()`` / wall-clock calls (the ``thesis_drift``
"network in the endpoint, builder takes the dicts" split).

The module **owns** its cadence constants (the ``feed_health.STALE_HOURS``
precedent — the module is the spec; the test reads these constants so a
retune cannot false-fail it). They mirror ``runner.OPEN_INTERVAL_S`` /
``runner.CLOSED_INTERVAL_S``; deliberately not imported from ``runner`` to
keep this leaf pure and free of any import cycle.

**Advisory only.** It states a fact about loop liveness; it issues no
directive, imposes no cap, and has no path to ``_execute()``. It does *not*
violate "no hard risk limits / Opus has full autonomy" (AGENTS.md
invariants #2/#12) — that governs *gating* decisions, not *observing the
loop*; same reasoning as ``feed_health`` / ``self_review``. A mirror, not a
cage. Never raises — an unparseable timestamp degrades to ``NO_DATA``.
"""
from __future__ import annotations

from datetime import datetime, timezone

OPEN_INTERVAL_S = 1800.0    # mirrors runner.OPEN_INTERVAL_S   (market open)
CLOSED_INTERVAL_S = 3600.0  # mirrors runner.CLOSED_INTERVAL_S (market closed)
LAGGING_MULT = 1.25
STALLED_MULT = 2.0


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _humanize(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{int(round(seconds))}s"
    mins = seconds / 60.0
    if mins < 90:
        return f"{int(round(mins))}m"
    hrs = int(mins // 60)
    rem = int(round(mins - hrs * 60))
    return f"{hrs}h {rem}m" if rem else f"{hrs}h"


def build_runner_heartbeat(
    last_decision_ts: str | None,
    market_open: bool,
    now: datetime | None = None,
) -> dict:
    """Verdict on whether the decision loop is still cycling.

    Pure. ``last_decision_ts`` is the newest ``decisions.timestamp`` (as
    ``store.recent_decisions(1)[0]["timestamp"]``); ``market_open`` selects
    the expected cadence. Verdict precedence: ``NO_DATA`` (no/garbled ts) →
    ``STALLED`` (> ``STALLED_MULT`` × expected; recommends restart) →
    ``LAGGING`` (> ``LAGGING_MULT`` × expected) → ``HEALTHY``.
    """
    now = now or datetime.now(timezone.utc)
    expected = OPEN_INTERVAL_S if market_open else CLOSED_INTERVAL_S
    ctx = "market-open" if market_open else "market-closed"
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "market_open": bool(market_open),
        "expected_interval_s": expected,
        "lagging_mult": LAGGING_MULT,
        "stalled_mult": STALLED_MULT,
        "last_decision_ts": None,
        "secs_since_last_decision": None,
        "intervals_elapsed": None,
        "verdict": "NO_DATA",
        "headline": ("No decisions recorded yet — the trading loop has not "
                     "produced a single cycle."),
        "restart_recommended": False,
    }

    ts = _parse_ts(last_decision_ts)
    if ts is None:
        return out

    secs = (now - ts).total_seconds()
    out["last_decision_ts"] = ts.isoformat(timespec="seconds")
    out["secs_since_last_decision"] = round(secs, 1)
    # A future-skewed ts is a just-written decision, not a stall: clamp the
    # ratio at 0 so it can never read LAGGING/STALLED.
    out["intervals_elapsed"] = round(max(0.0, secs) / expected, 3)

    age = _humanize(secs)
    exp_h = _humanize(expected)
    if secs > STALLED_MULT * expected:
        out["verdict"] = "STALLED"
        out["restart_recommended"] = True
        out["headline"] = (
            f"STALLED — no decision in {age} (>{STALLED_MULT:g}x the {exp_h} "
            f"expected {ctx} cadence); the trading loop appears dead. "
            f"Restart paper-trader.")
    elif secs > LAGGING_MULT * expected:
        out["verdict"] = "LAGGING"
        out["headline"] = (
            f"LAGGING — last decision {age} ago (>{LAGGING_MULT:g}x the "
            f"{exp_h} {ctx} cadence); the loop is slow or a cycle overran.")
    else:
        out["verdict"] = "HEALTHY"
        out["headline"] = (
            f"HEALTHY — last decision {age} ago, within the {exp_h} {ctx} "
            f"cadence.")
    return out
