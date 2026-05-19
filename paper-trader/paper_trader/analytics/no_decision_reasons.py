"""NO_DECISION reason breakdown — buckets the WHY of recent failed cycles.

The trader has good *aggregate* visibility into a NO_DECISION storm:
``runner_heartbeat.decision_efficacy`` raises ``IDLE_STORM`` once
``>= NO_DECISION_STORM_THRESHOLD`` cycles in a row produce no decision, and
``decision_drought`` prices the portfolio-vs-SPY drift the storm cost.
What neither answers is the question the operator asks next:

  **"WHY isn't the bot deciding? Should I restart the runner, kill the
  out-of-band Opus agents, wait for the quota to reset, or check the
  prompt?"**

Every NO_DECISION row already carries a human-readable ``reasoning``
string written by ``strategy.decide()`` (the canonical mapping):

  * ``"claude quota/usage limit exhausted (no decision)"``      → ``quota_exhausted``
  * ``"skipped claude call — host saturated: …"``               → ``host_saturated``
  * ``"claude returned no response (timeout/empty)"``           → ``model_empty``
  * ``"parse_failed: …"``                                       → ``parse_failed``
  * ``"retry_failed: …"``                                       → ``retry_failed``
  * anything else                                               → ``other``

Different buckets have *different operator actions* (a quota outage is
"wait for reset"; a host-saturation storm is "kill the parallel Opus
agents"; a model-empty storm is "restart the runner — wedged CLI"; a
parse_failed storm is "the prompt or the model is regressing"). Conflating
them as a single IDLE_STORM means the operator is told only "restart may
help" when the real fix is often elsewhere.

This builder takes the newest-first ``store.recent_decisions`` rows and
returns a histogram, the dominant bucket, and a concrete recommendation —
single source of truth for the bucketing predicate (mirrors the prefix
markers ``strategy.decide()`` uses verbatim, so a future runner change
that adds a new reason class need only register one more bucket here).

Pure & offline: the builder takes the decision dicts; the endpoint owns
the ``store.recent_decisions()`` read (the ``runner_heartbeat`` "network
in the endpoint, builder is pure" split). Never raises — garbage input
degrades to ``NO_DATA``.

**Advisory only.** Observational, no caps, never gates Opus, has no path
to ``_execute()`` (AGENTS.md invariants #2/#12 — the ``runner_heartbeat``
precedent). A mirror, not a cage.
"""
from __future__ import annotations

from collections import Counter

# Default analysis window over ``store.recent_decisions`` (newest-first).
# Lifetime ~60% NO_DECISION on the live trader (AGENTS.md), so a 50-row
# window typically holds ~30 NO_DECISION rows — enough for the dominant
# bucket to be statistically meaningful without being so deep that an
# old quota outage drowns out today's host-saturation problem.
DEFAULT_WINDOW = 50

# Dominant-share gate: a bucket holding ≥ this fraction of the NO_DECISION
# rows in the window earns the DOMINANT verdict (and the targeted
# recommendation); below it, the histogram is reported as MIXED with the
# generic "no single root cause" line. Keeps a balanced 4-way split from
# claiming a single fix.
DOMINANT_THRESHOLD_PCT = 50.0

# Bucket → concrete operator recommendation. The keys are the canonical
# bucket names emitted by ``_bucket_for`` and shown in the histogram.
_RECOMMENDATIONS: dict[str, str] = {
    "quota_exhausted": (
        "Claude usage / monthly quota is exhausted — no fix on the box. "
        "Wait for the quota to reset (or upgrade the plan); the runner "
        "will resume automatically on the next non-rejected call."
    ),
    "host_saturated": (
        "Host saturation — too many concurrent Opus subprocesses "
        "(review agents / backtest committee). Reduce parallel Opus jobs, "
        "or wait for the storm to clear; a runner restart does NOT help."
    ),
    # Sub-buckets of the legacy ``model_empty`` (the empty-response case).
    # ``strategy._claude_call`` now sets a per-call cause code which surfaces
    # as ``claude returned no response (timeout|nonzero_rc|empty_stdout|
    # cli_missing|exception)``. Each sub-cause has a different fix: a
    # ``timeout`` is the wedge case the legacy "restart the runner" line
    # was written for; ``nonzero_rc`` means the CLI itself is crashing
    # (usually a transient API error); ``empty_stdout`` is rc=0 with no
    # output (model-level blank, almost always a one-cycle blip — restart
    # is overkill); ``cli_missing`` means the ``claude`` binary is gone
    # from PATH (deploy / env regression); ``exception`` means the Popen
    # call itself raised (a system-level issue, not a model one).
    "model_timeout": (
        "Claude timed out (full request budget exhausted). Most often a "
        "wedged CLI subprocess: restart the runner so the auto-recovery "
        "breaker reaps the stale claude process and the next cycle gets a "
        "fresh one. If it persists across restarts, suspect network /"
        " upstream Anthropic latency."
    ),
    "cli_nonzero_rc": (
        "The claude CLI exited non-zero (and not a quota rejection). Usually "
        "a transient upstream API error or a temporary auth blip — wait one "
        "or two cycles. If it persists, run `claude --print --model "
        "claude-opus-4-7` by hand to read the actual error message; "
        "restarting the runner does NOT fix an upstream API problem."
    ),
    "cli_empty_stdout": (
        "The claude CLI exited cleanly but returned no text (rc=0, empty "
        "stdout). Distinct from a timeout — the model itself returned blank. "
        "Almost always a one-cycle blip; restart is overkill. If it "
        "repeats, retry with the JSON-only nudge or try the Sonnet "
        "fallback by lowering MODEL temporarily."
    ),
    "cli_missing": (
        "The ``claude`` binary is missing from PATH at decide() time. "
        "Deploy / environment regression — neither a runner restart nor "
        "waiting will help. Reinstall the Claude CLI or fix the PATH on "
        "the trader host."
    ),
    "cli_exception": (
        "Popen / communicate raised an exception (system-level failure, "
        "not a model one). Check disk space, file-handle limits, and the "
        "[strategy] claude exception: line in runner.log — restarting the "
        "runner is unlikely to help if the underlying OS issue persists."
    ),
    "model_empty": (
        "Claude returned no response (timeout/empty). Most often a wedged "
        "CLI: restart the runner so the auto-recovery breaker reaps the "
        "stale claude subprocess and the next cycle gets a fresh process."
    ),
    "parse_failed": (
        "Opus returned text that didn't parse as JSON. Repeating offenders "
        "suggest a prompt or model regression — inspect a recent reasoning "
        "excerpt from the dashboard and consider a prompt tweak."
    ),
    "retry_failed": (
        "Parse-retry failed too. Same diagnosis as parse_failed but the "
        "JSON-only nudge isn't rescuing — strong signal the prompt / "
        "system instruction needs revision."
    ),
    "other": (
        "Reason text didn't match any known bucket. Open the dashboard "
        "decision log and inspect recent reasoning excerpts directly."
    ),
}


def _bucket_for(reason: str | None) -> str:
    """Map a NO_DECISION reasoning string to its canonical bucket.

    The prefixes / phrases mirror ``strategy.decide()``'s ``reason_text``
    construction *verbatim* (single source of truth, AGENTS.md invariant
    #10): the runner writes these strings, this map reads them. A future
    runner change that adds a new bucket need only register the prefix /
    keyword here and add a recommendation.
    """
    r = (reason or "").lower()
    if not r:
        return "other"
    # Quota check first — a quota-exhausted reason can in theory contain
    # other markers (an exhaustion message echoed back from the CLI), so
    # the most operator-actionable bucket wins.
    if "quota" in r or "usage limit" in r:
        return "quota_exhausted"
    if "host saturated" in r or "skipped claude" in r:
        return "host_saturated"
    if r.startswith("parse_failed"):
        return "parse_failed"
    if r.startswith("retry_failed"):
        return "retry_failed"
    # Sub-buckets of the empty-response case, keyed off the cause code
    # ``strategy._claude_call`` writes into the parenthesised suffix. Match
    # specific causes FIRST — every one of these strings still contains
    # "no response" and would otherwise fall through to the legacy
    # ``model_empty`` bucket. The order within is irrelevant (mutually
    # exclusive substrings); the ``model_empty`` fallback below remains the
    # back-compat path for any caller that didn't set a cause (legacy DB rows
    # plus the literal "timeout/empty" default when ``_last_claude_fail`` was
    # None at decide() time).
    if "no response (timeout)" in r:
        return "model_timeout"
    if "no response (nonzero_rc)" in r:
        return "cli_nonzero_rc"
    if "no response (empty_stdout)" in r:
        return "cli_empty_stdout"
    if "no response (cli_missing)" in r:
        return "cli_missing"
    if "no response (exception)" in r:
        return "cli_exception"
    if "no response" in r or "timeout/empty" in r:
        return "model_empty"
    return "other"


def _is_no_decision(action_taken: str | None) -> bool:
    """True for a failed cycle. Verbatim mirror of the canonical predicate
    (``decision_forensics._is_no_decision`` / ``runner_heartbeat._is_no_decision``
    — single source of truth, AGENTS.md invariant #10). Inlined to keep
    this leaf free of cross-analytics imports; drift-locked by the unit
    test below.
    """
    raw = (action_taken or "").strip()
    return not raw or raw == "NO_DECISION"


def build_no_decision_reasons(
    decisions: list[dict] | None,
    window: int = DEFAULT_WINDOW,
) -> dict:
    """Bucket the NO_DECISION reasoning strings in the most recent
    ``window`` rows of ``decisions`` (newest-first) and return a
    histogram + dominant-cause verdict + targeted recommendation.

    Result shape (always present, never raises):

      * ``state``             — ``"NO_DATA"`` if window has no NO_DECISION
                                rows; ``"DOMINANT"`` if one bucket holds
                                ≥ ``DOMINANT_THRESHOLD_PCT``; else ``"MIXED"``.
      * ``window``            — the analysis window actually used (clamped).
      * ``n_decisions``       — total rows considered in the window.
      * ``n_no_decision``     — NO_DECISION rows in the window.
      * ``no_decision_pct``   — n_no_decision / n_decisions × 100, or None.
      * ``buckets``           — ``{bucket: count}`` for every present bucket
                                (zero-count buckets are omitted).
      * ``dominant_bucket``   — bucket name when ``state == "DOMINANT"``,
                                else None.
      * ``dominant_pct``      — dominant bucket's share of NO_DECISION rows
                                (DOMINANT or MIXED), else None.
      * ``headline``          — one-liner suitable for the dashboard /
                                Discord summary.
      * ``recommendation``    — concrete operator action; suppressed
                                (``""``) on NO_DATA / MIXED so a healthy
                                book never gets a misleading "restart"
                                line.
    """
    try:
        w = max(1, int(window))
    except (TypeError, ValueError):
        w = DEFAULT_WINDOW
    rows = list(decisions or [])[:w]
    nd_rows = [d for d in rows if _is_no_decision(d.get("action_taken"))]
    n_dec = len(rows)
    n_nd = len(nd_rows)

    if n_nd == 0:
        return {
            "state": "NO_DATA",
            "window": w,
            "n_decisions": n_dec,
            "n_no_decision": 0,
            "no_decision_pct": 0.0 if n_dec else None,
            "buckets": {},
            "dominant_bucket": None,
            "dominant_pct": None,
            "headline": (
                f"no NO_DECISION cycles in the last {n_dec} rows — "
                "engine is producing decisions"
                if n_dec else
                "no recent decisions on record"
            ),
            "recommendation": "",
        }

    counts: Counter[str] = Counter()
    for d in nd_rows:
        counts[_bucket_for(d.get("reasoning"))] += 1

    # Counter.most_common is stable for ties (insertion order). With ties
    # we deterministically prefer the more operator-actionable bucket by
    # sorting on (-count, _ACTION_RANK[bucket]) so e.g. a 50/50 split of
    # quota_exhausted and other never reports "other" as dominant.
    rank = {
        # Most operator-actionable first (lowest rank wins on ties).
        # Quota / host outranks model timeouts (a saturation diagnosis is
        # more actionable than a generic wedge). cli_missing outranks
        # cli_exception (PATH regression is a definite fix; an exception
        # tail is "go read the log"). Sub-buckets keep model_empty below
        # them so a tied "specific 5 vs generic 5" prefers the specific.
        "quota_exhausted": 0,
        "host_saturated": 1,
        "parse_failed": 2,
        "retry_failed": 3,
        "cli_missing": 4,
        "model_timeout": 5,
        "cli_nonzero_rc": 6,
        "cli_empty_stdout": 7,
        "cli_exception": 8,
        "model_empty": 9,
        "other": 10,
    }
    items = sorted(counts.items(), key=lambda kv: (-kv[1], rank.get(kv[0], 99)))
    top_name, top_n = items[0]
    top_pct = top_n / n_nd * 100.0
    no_dec_pct = (n_nd / n_dec * 100.0) if n_dec else None

    if top_pct >= DOMINANT_THRESHOLD_PCT:
        state = "DOMINANT"
        rec = _RECOMMENDATIONS.get(top_name, _RECOMMENDATIONS["other"])
        headline = (
            f"{n_nd}/{n_dec} cycles NO_DECISION; dominant cause: "
            f"{top_name} ({top_pct:.0f}%) — {rec}"
        )
        dom_pct = top_pct
        dom_name: str | None = top_name
    else:
        state = "MIXED"
        # Top two for the headline, comma-joined "bucket Npct%".
        mix = ", ".join(
            f"{n}={cnt} ({cnt / n_nd * 100.0:.0f}%)"
            for n, cnt in items[:3]
        )
        headline = (
            f"{n_nd}/{n_dec} cycles NO_DECISION; no single dominant cause "
            f"(mix: {mix}) — inspect dashboard decision log to triage"
        )
        rec = ""  # the operator must read the histogram themselves
        dom_pct = top_pct  # informational only — not promoted to DOMINANT
        dom_name = None

    return {
        "state": state,
        "window": w,
        "n_decisions": n_dec,
        "n_no_decision": n_nd,
        "no_decision_pct": round(no_dec_pct, 1) if no_dec_pct is not None else None,
        "buckets": dict(counts),
        "dominant_bucket": dom_name,
        "dominant_pct": round(dom_pct, 1),
        "headline": headline,
        "recommendation": rec,
    }
