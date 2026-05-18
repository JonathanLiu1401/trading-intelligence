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

**Decision efficacy (additive, 2026-05-18).** Cadence liveness alone is a
*dangerous half-truth*: a loop that cycles perfectly on schedule but emits
``NO_DECISION`` every cycle (the documented live regime — ~60% lifetime,
runs of 5+ back-to-back under host-load timeout storms) reports a flat green
``HEALTHY … restart_recommended:false``. An operator/trader reading the
heartbeat — its primary purpose — is then *actively* reassured the engine is
fine while it is brain-dead. ``decision_health``/``-forensics`` analyse the
NO_DECISION rate, but a trader checks the *heartbeat* first for "is anything
wrong". When ``recent_actions`` is supplied (newest-first
``decisions.action_taken`` strings) the builder computes an additive
``decision_efficacy`` sub-block and, **only** on a genuine idle-storm
(``>= NO_DECISION_STORM_THRESHOLD`` consecutive NO_DECISION — the same wedge
``runner``'s auto-recovery circuit breaker targets), folds that into the
top-level ``headline`` + ``restart_recommended`` so the green light can no
longer hide a stuck engine. The liveness ``verdict`` enum is left untouched
(the documented liveness/efficacy separation; every verdict-string lock
stays green). With ``recent_actions`` omitted the output is byte-identical
to before.

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
# Mirrors runner.CONSECUTIVE_NO_DECISION_LIMIT (deliberately mirrored, not
# imported — this leaf stays pure & import-cycle-free, the OPEN_INTERVAL_S
# precedent; the test reads this constant so a retune cannot false-fail it).
# A run of this many back-to-back NO_DECISION cycles is the exact wedge the
# runner's auto-recovery breaker exists to clear, so the heartbeat
# recommends the same lever (restart) at the same threshold.
NO_DECISION_STORM_THRESHOLD = 5
# A window in which at least this fraction is NO_DECISION but the *latest*
# cycle still produced a decision: not a hard wedge, but the documented
# elevated-failure regime — surfaced informational, NO restart recommended.
NO_DECISION_ELEVATED_PCT = 50.0


def _is_no_decision(action_taken: str | None) -> bool:
    """True for a failed cycle. ``strategy.py`` records ``action_taken``
    exactly ``"NO_DECISION"`` for every cycle Claude failed to produce a
    parseable decision; a FILLED/HOLD/BLOCKED row is ``"<verb> <tk> → …"``.

    Verbatim mirror of ``decision_forensics._is_no_decision`` — the canonical
    NO_DECISION predicate (single source of truth, AGENTS.md invariant #10).
    Inlined (not imported) to keep this endpoint-path leaf free of any
    cross-analytics import; drift-locked by
    ``tests/test_runner_heartbeat.py::test_is_no_decision_mirrors_forensics``."""
    raw = (action_taken or "").strip()
    return not raw or raw == "NO_DECISION"


def _decision_efficacy(recent_actions: list[str] | None,
                       threshold: int) -> dict | None:
    """Additive efficacy sub-block from newest-first ``action_taken`` rows.

    ``None`` when ``recent_actions`` is not supplied (caller renders the
    pre-existing cadence-only output, byte-identical to before). Otherwise:

      * ``NO_DATA``    — empty window;
      * ``IDLE_STORM`` — the latest ``>= threshold`` cycles were ALL
        NO_DECISION (engine cycling but not deciding — a restart may clear a
        wedged CLI; the runner auto-recovery-breaker wedge);
      * ``DEGRADED``   — not a hard storm, but ``>= NO_DECISION_ELEVATED_PCT``
        of the window is NO_DECISION (the documented elevated regime —
        informational, no restart);
      * ``PRODUCING``  — the engine is turning out real decisions.
    """
    if recent_actions is None:
        return None
    window = len(recent_actions)
    if window == 0:
        return {
            "verdict": "NO_DATA",
            "window": 0,
            "consecutive_no_decision": 0,
            "no_decision_pct": None,
            "headline": "No decisions in the recent window — cannot assess "
                        "decision efficacy.",
        }
    consec = 0
    for a in recent_actions:               # newest-first → leading run
        if _is_no_decision(a):
            consec += 1
        else:
            break
    n_nd = sum(1 for a in recent_actions if _is_no_decision(a))
    pct = round(n_nd / window * 100.0, 1)
    if consec >= threshold:
        verdict = "IDLE_STORM"
        headline = (
            f"IDLE_STORM — the last {consec} cycles were ALL NO_DECISION "
            f"({pct:.0f}% of the last {window}); the loop is cycling but the "
            f"engine is not deciding. A restart may clear a wedged Claude "
            f"CLI.")
    elif pct >= NO_DECISION_ELEVATED_PCT:
        verdict = "DEGRADED"
        headline = (
            f"DEGRADED — {pct:.0f}% of the last {window} cycles were "
            f"NO_DECISION (latest still produced a decision); decision "
            f"throughput is impaired but the engine is not wedged.")
    else:
        verdict = "PRODUCING"
        headline = (
            f"PRODUCING — {window - n_nd}/{window} recent cycles produced a "
            f"decision; the engine is deciding, not just cycling.")
    return {
        "verdict": verdict,
        "window": window,
        "consecutive_no_decision": consec,
        "no_decision_pct": pct,
        "headline": headline,
    }


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
    recent_actions: list[str] | None = None,
    no_decision_storm_threshold: int = NO_DECISION_STORM_THRESHOLD,
) -> dict:
    """Verdict on whether the decision loop is still cycling **and deciding**.

    Pure. ``last_decision_ts`` is the newest ``decisions.timestamp`` (as
    ``store.recent_decisions(1)[0]["timestamp"]``); ``market_open`` selects
    the expected cadence. Verdict precedence: ``NO_DATA`` (no/garbled ts) →
    ``STALLED`` (> ``STALLED_MULT`` × expected; recommends restart) →
    ``LAGGING`` (> ``LAGGING_MULT`` × expected) → ``HEALTHY``.

    ``recent_actions`` (optional, newest-first ``decisions.action_taken``
    strings — the endpoint owns the ``store.recent_decisions(N)`` read, the
    thesis_drift split) adds an additive ``decision_efficacy`` sub-block. On
    a genuine idle-storm (``>= no_decision_storm_threshold`` consecutive
    NO_DECISION) the top-level ``headline`` gains a clause and
    ``restart_recommended`` becomes True even when the loop is cadence-
    HEALTHY — a loop that cycles but never decides is not actually healthy.
    The liveness ``verdict`` enum is left untouched (the documented
    liveness/efficacy separation). Omitting ``recent_actions`` ⇒ output
    byte-identical to before this parameter existed.
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

    # Additive decision-efficacy overlay. The liveness verdict above answers
    # "is the loop cycling?"; this answers "is it actually deciding?". A loop
    # that cycles perfectly but emits NO_DECISION every time is alive-but-
    # brain-dead — and the bare cadence verdict would call that HEALTHY.
    eff = _decision_efficacy(recent_actions, no_decision_storm_threshold)
    if eff is not None:
        out["decision_efficacy"] = eff
        if eff["verdict"] == "IDLE_STORM":
            # A restart is the documented lever for this exact wedge (the
            # runner auto-recovery breaker fires at the same threshold), so
            # surface it on the line the operator actually reads — never
            # mutate the liveness verdict enum (the separation contract).
            out["restart_recommended"] = True
            out["headline"] += (
                f" ⚠ but the last {eff['consecutive_no_decision']} cycles "
                f"were ALL NO_DECISION — the engine is cycling, not "
                f"deciding; a restart may clear a wedged Claude CLI.")
    return out
