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

# Sentinel for the optional ``last_real_decision_ts`` kwarg. We must
# distinguish "caller omitted the kwarg" (legacy callers ⇒ byte-identical
# output, no new keys) from "caller passed an explicit None" (the endpoint
# always passes through whatever ``store.last_real_decision()`` returns,
# including None for a fresh book ⇒ emit the keys with None values so the
# operator's JSON shape is stable across runs).
_UNSET = object()

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

# Causes for which a paper-trader RESTART is actively counter-productive, not
# merely useless: a host-saturation storm is cleared by REDUCING concurrent
# Opus jobs — restarting just adds another ~1.5GB Opus process to the storm —
# and a quota exhaustion only clears when the usage window resets. The pre-
# 2026-05-22 IDLE_STORM headline unconditionally told the operator "a restart
# may clear a wedged Claude CLI", which during the dominant live failure mode
# (host saturation — see strategy.host_guard) misdirects the trader into the
# exact action that worsens it. /api/no-decision-reasons already diagnoses the
# real cause; this makes the heartbeat agree with it.
_RESTART_INEFFECTIVE_CAUSES = ("host_saturated", "quota")


def _no_decision_cause(reason: str | None) -> str:
    """Bucket a ``decisions.reasoning`` string for a NO_DECISION row into the
    coarse cause that decides whether a restart is the right lever.

      * ``host_saturated`` — ``strategy.decide`` skipped the claude call
        because ``host_guard`` saw too many concurrent Opus subprocesses /
        high swap (reason text ``"skipped claude call — host saturated: …"``).
      * ``quota``          — the Claude CLI rejected every attempt with a
        usage/quota limit (reason ``"claude quota/usage limit exhausted …"``).
      * ``other``          — a genuine model timeout / empty response / parse
        failure (``"claude returned no response …"``, ``parse_failed:`` …):
        a restart *may* clear a wedged CLI, so the legacy advice still holds.

    Pure on the input string, never raises (the module's NO_DATA contract)."""
    r = (reason or "").lower()
    if "host saturat" in r or "skipped claude call" in r:
        return "host_saturated"
    if "quota" in r or "usage limit" in r:
        return "quota"
    return "other"


def _dominant_cause(reasons: list[str | None]) -> str:
    """Most frequent NO_DECISION cause across ``reasons`` (the leading idle
    run). Tie-break prefers ``host_saturated`` then ``quota`` then ``other`` —
    the restart-ineffective causes win a tie so a mixed storm never
    *under*-warns the operator into a useless restart. ``[]`` → ``other``
    (legacy behaviour: assume a restart could help when nothing is known)."""
    if not reasons:
        return "other"
    counts = {"host_saturated": 0, "quota": 0, "other": 0}
    for r in reasons:
        counts[_no_decision_cause(r)] += 1
    order = ("host_saturated", "quota", "other")
    return max(order, key=lambda c: (counts[c], -order.index(c)))


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
                       threshold: int,
                       recent_reasons: list[str | None] | None = None
                       ) -> dict | None:
    """Additive efficacy sub-block from newest-first ``action_taken`` rows.

    ``None`` when ``recent_actions`` is not supplied (caller renders the
    pre-existing cadence-only output, byte-identical to before). Otherwise:

      * ``NO_DATA``    — empty window;
      * ``IDLE_STORM`` — the latest ``>= threshold`` cycles were ALL
        NO_DECISION (engine cycling but not deciding — the runner
        auto-recovery-breaker wedge);
      * ``DEGRADED``   — not a hard storm, but ``>= NO_DECISION_ELEVATED_PCT``
        of the window is NO_DECISION (the documented elevated regime —
        informational, no restart);
      * ``PRODUCING``  — the engine is turning out real decisions.

    ``recent_reasons`` (optional, newest-first ``decisions.reasoning`` strings
    parallel to ``recent_actions``): when supplied, an IDLE_STORM sub-block
    additionally carries ``dominant_cause`` + ``restart_helps`` and a
    cause-specific headline so the operator is not told to restart during a
    host-saturation / quota storm — the action that worsens it. Omitting it ⇒
    the legacy generic "a restart may clear a wedged Claude CLI" headline."""
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
        storm_head = (
            f"IDLE_STORM — the last {consec} cycles were ALL NO_DECISION "
            f"({pct:.0f}% of the last {window}); the loop is cycling but the "
            f"engine is not deciding.")
        cause = restart_helps = None
        if recent_reasons is not None:
            # Diagnose the leading idle run so the restart advice is honest.
            cause = _dominant_cause(list(recent_reasons[:consec]))
            restart_helps = cause not in _RESTART_INEFFECTIVE_CAUSES
        if cause == "host_saturated":
            headline = (storm_head + " Cause: host saturation (too many "
                        "concurrent Opus jobs) — a restart will NOT help and "
                        "adds load; reduce parallel Opus jobs or wait for the "
                        "storm to clear.")
        elif cause == "quota":
            headline = (storm_head + " Cause: Claude quota/usage limit "
                        "exhausted — a restart will NOT help; wait for the "
                        "quota window to reset.")
        else:
            headline = storm_head + " A restart may clear a wedged Claude CLI."
        eff = {
            "verdict": verdict,
            "window": window,
            "consecutive_no_decision": consec,
            "no_decision_pct": pct,
            "headline": headline,
        }
        if cause is not None:
            eff["dominant_cause"] = cause
            eff["restart_helps"] = bool(restart_helps)
        return eff
    if pct >= NO_DECISION_ELEVATED_PCT:
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
    recent_reasons: list[str | None] | None = None,
    last_real_decision_ts=_UNSET,
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

    ``recent_reasons`` (optional, newest-first ``decisions.reasoning`` strings
    parallel to ``recent_actions``): when supplied, an IDLE_STORM diagnoses
    its dominant cause. A host-saturation / quota storm is NOT cleared by a
    restart — ``restart_recommended`` stays False and the headline says so,
    instead of misdirecting the operator into the action that worsens it.
    Omitting ``recent_reasons`` ⇒ the legacy unconditional "a restart may
    clear a wedged Claude CLI" headline, byte-identical to before.

    ``last_real_decision_ts`` (optional, the timestamp of the most recent
    decision row whose ``action_taken`` is NOT NO_DECISION — i.e.
    ``store.last_real_decision()["timestamp"]``): when supplied, the output
    additionally carries ``last_real_decision_ts`` /
    ``secs_since_real_decision`` / ``real_decision_age``. Under IDLE_STORM
    the bare ``last_decision_ts`` (which advances every cycle, including
    NO_DECISION) tells the operator "last decision 12m ago" while the
    engine has not produced an actual FILLED/HOLD/BLOCKED row in days —
    the exact green-light pathology AGENTS.md HYBRID pass #7 surfaced.
    Omitting ``last_real_decision_ts`` ⇒ output byte-identical to before
    these keys existed (no new fields appear on the dict at all)."""
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

    # Additive real-decision-age overlay. ``last_decision_ts`` advances every
    # cycle, including a NO_DECISION storm row — so under IDLE_STORM the
    # cadence-only output says "last decision 12m ago" while the engine has
    # not actually produced a FILLED/HOLD/BLOCKED row in days. ``store.
    # last_real_decision()`` answers that question; when its ``timestamp`` is
    # supplied via ``last_real_decision_ts`` we surface ``secs_since_real_
    # decision`` so the operator never sees a green light over a brain-dead
    # engine. Pure additive: omitting the kwarg leaves the dict unchanged
    # (no new keys), so legacy callers + the test_output_echoes_constants_
    # and_inputs pin stay green.
    if last_real_decision_ts is not _UNSET:
        real_ts = (_parse_ts(last_real_decision_ts)
                   if last_real_decision_ts is not None else None)
        if real_ts is not None:
            real_secs = (now - real_ts).total_seconds()
            out["last_real_decision_ts"] = real_ts.isoformat(timespec="seconds")
            out["secs_since_real_decision"] = round(real_secs, 1)
            out["real_decision_age"] = _humanize(real_secs)
        else:
            out["last_real_decision_ts"] = None
            out["secs_since_real_decision"] = None
            out["real_decision_age"] = None

    # Additive decision-efficacy overlay. The liveness verdict above answers
    # "is the loop cycling?"; this answers "is it actually deciding?". A loop
    # that cycles perfectly but emits NO_DECISION every time is alive-but-
    # brain-dead — and the bare cadence verdict would call that HEALTHY.
    eff = _decision_efficacy(recent_actions, no_decision_storm_threshold,
                             recent_reasons)
    if eff is not None:
        out["decision_efficacy"] = eff
        if eff["verdict"] == "IDLE_STORM":
            # Real-decision age clause for the IDLE_STORM headline — the
            # operator-actionable number under a storm. Always appended (cause-
            # diagnosed or generic) when ``last_real_decision_ts`` was supplied
            # AND parsed. ``None`` (no real decision ever) renders "never"
            # which is the right read for a fresh book whose first 24h was all
            # host-saturated. Computed once here so the three headline arms
            # below all carry the same suffix.
            real_age_clause = ""
            if last_real_decision_ts is not _UNSET:
                real_secs = out.get("secs_since_real_decision")
                if real_secs is None:
                    real_age_clause = (
                        " The engine has NEVER produced a real decision "
                        "(only NO_DECISION cycles).")
                else:
                    real_age_clause = (
                        f" Last real (FILLED/HOLD/BLOCKED) decision was "
                        f"{_humanize(real_secs)} ago.")
            # A wedged-CLI storm IS cleared by a restart (the runner
            # auto-recovery breaker fires at the same threshold); a
            # host-saturation / quota storm is NOT — restarting just adds
            # another Opus process to the storm. ``restart_helps`` is present
            # only when ``recent_reasons`` was supplied; without it, preserve
            # the legacy "restart recommended" behaviour byte-for-byte.
            n = eff["consecutive_no_decision"]
            restart_helps = eff.get("restart_helps")
            if restart_helps is False:
                # Cause-diagnosed and a restart would not help. Do NOT set
                # restart_recommended — _heartbeat_line still surfaces the
                # IDLE_STORM, just without the misleading restart directive.
                cause = eff.get("dominant_cause")
                if cause == "host_saturated":
                    out["headline"] += (
                        f" ⚠ but the last {n} cycles were ALL NO_DECISION — "
                        f"the engine is cycling, not deciding; cause is host "
                        f"saturation, a restart will NOT help (reduce "
                        f"concurrent Opus jobs).")
                else:  # quota
                    out["headline"] += (
                        f" ⚠ but the last {n} cycles were ALL NO_DECISION — "
                        f"the engine is cycling, not deciding; the Claude "
                        f"quota is exhausted, a restart will NOT help (wait "
                        f"for the quota to reset).")
            else:
                # restart_helps True (wedged CLI) OR None (legacy: no reasons
                # supplied — keep the pre-2026-05-22 string byte-identical).
                out["restart_recommended"] = True
                out["headline"] += (
                    f" ⚠ but the last {n} cycles "
                    f"were ALL NO_DECISION — the engine is cycling, not "
                    f"deciding; a restart may clear a wedged Claude CLI.")
            out["headline"] += real_age_clause
    return out
