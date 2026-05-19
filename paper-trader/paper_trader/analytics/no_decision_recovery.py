"""NO_DECISION recovery-time — how long do wedges last, and is the current one anomalous?

``runner_heartbeat.decision_efficacy`` answers *"are we in a storm right now?"* (binary
on the most-recent ``NO_DECISION_STORM_THRESHOLD=5`` cycles). ``no_decision_reasons``
answers *"what's the dominant cause?"*. ``decision_drought`` answers *"what's the P&L
cost?"*. None answer the question an on-call operator asks next:

  **"How long do these wedges normally last, and is the current one worse than
  typical? Should I keep waiting, or escalate now?"**

This builder walks ``store.recent_decisions`` (newest-first) and run-length encodes
the NO_DECISION sequence into:

  * **completed runs**     — closed wedges (a real decision ended them); the
                             empirical recovery-time distribution.
  * **open run**           — trailing wedge, if any; the active situation.

Then it grades the open run against the historic distribution: ``WITHIN_NORMAL`` if
≤ median, ``ELEVATED`` if > median but < p95, ``ABNORMAL_WEDGE`` if ≥ p95. Single
stale cycles (open run length 1, below ``MIN_RUN_FOR_STATS``) are classified as
``NOISE`` rather than wedge — the runner's own auto-recovery breaker only treats
``≥ NO_DECISION_STORM_THRESHOLD`` (5) consecutive cycles as wedge-worthy, so anything
shorter is qualitatively a hiccup, not a stuck state.

Pure & offline: the builder takes the decision dicts; the endpoint / CLI owns the
``store.recent_decisions()`` read (the ``no_decision_reasons`` "network in the
endpoint, builder is pure" split). Never raises — garbage input degrades to
``NO_DATA`` / ``INSUFFICIENT_HISTORY``.

**Advisory only.** Observational, no caps, never gates Opus, has no path to
``_execute()`` (AGENTS.md invariants #2/#12 — the ``runner_heartbeat`` precedent).
A mirror, not a cage.

Run as a CLI for a one-shot ops view::

    python3 -m paper_trader.analytics.no_decision_recovery
"""
from __future__ import annotations

# Default window over ``store.recent_decisions`` (newest-first). Wider than
# ``no_decision_reasons.DEFAULT_WINDOW=50`` because percentiles need enough
# closed wedges to be meaningful; 200 rows on a 60s cadence is ~3.3 hours of
# market history, enough to span several full storm-recovery cycles even at
# the documented ~60% NO_DECISION lifetime rate.
DEFAULT_WINDOW = 200

# A closed run shorter than this is "noise" and excluded from the recovery-time
# distribution: a single stale cycle isn't a wedge, it's a transient hiccup.
# Matches ``runner_heartbeat.NO_DECISION_STORM_THRESHOLD`` *only in spirit* —
# we use the lower bound (2) for stats because the storm-threshold (5) would
# leave most desks with too few completed wedges to compute a percentile.
MIN_RUN_FOR_STATS = 2

# A verdict that compares the open run to a historic distribution needs at
# least this many completed wedges to be statistically defensible. Below it
# the verdict is INSUFFICIENT_HISTORY rather than a spurious ABNORMAL/NORMAL.
MIN_HISTORY_FOR_VERDICT = 3


def _is_no_decision(action_taken: str | None) -> bool:
    """True for a failed cycle. Verbatim mirror of the canonical predicate
    (``decision_forensics._is_no_decision`` / ``runner_heartbeat._is_no_decision``
    / ``no_decision_reasons._is_no_decision`` — single source of truth,
    AGENTS.md invariant #10). Inlined to keep this leaf free of cross-analytics
    imports; drift-locked by the unit test below.
    """
    raw = (action_taken or "").strip()
    return not raw or raw == "NO_DECISION"


def _percentile(sorted_xs: list[int], q: float) -> float | None:
    """Linear-interpolated percentile (numpy-style) over a sorted list of ints.

    Returns ``None`` for the empty list. Used for median/p95 of completed-run
    lengths — kept inline so this leaf doesn't pull in numpy just for two
    quantiles."""
    if not sorted_xs:
        return None
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    k = (len(sorted_xs) - 1) * q
    f = int(k)
    c = f + 1
    if c >= len(sorted_xs):
        return float(sorted_xs[-1])
    frac = k - f
    return float(sorted_xs[f] + (sorted_xs[c] - sorted_xs[f]) * frac)


def _run_lengths(decisions_oldest_first: list[dict]) -> tuple[list[int], int]:
    """Run-length encode NO_DECISION runs in chronological order.

    Returns ``(completed_runs, open_run_length)``. A run that touches the
    newest row is "open" — separated out because the recovery time isn't
    known until a real decision closes it. Completed runs go into the
    distribution; the open run is what we *grade against* that distribution.
    """
    completed: list[int] = []
    cur = 0
    for d in decisions_oldest_first:
        if _is_no_decision(d.get("action_taken")):
            cur += 1
        else:
            if cur > 0:
                completed.append(cur)
            cur = 0
    open_run = cur  # 0 if the latest row is a real decision
    return completed, open_run


def build_no_decision_recovery(
    decisions: list[dict] | None,
    window: int = DEFAULT_WINDOW,
) -> dict:
    """Compute the NO_DECISION run-length distribution and verdict on the
    currently-open wedge (if any) against history.

    Result shape (always present, never raises):

      * ``state``                — ``"NO_DATA"`` if window is empty; else
                                   ``"OK"``.
      * ``window``               — analysis window actually used (clamped).
      * ``n_rows``               — rows considered.
      * ``open_run_length``      — length of the trailing NO_DECISION run
                                   (0 if the latest row is a real decision).
      * ``completed_runs``       — list of every closed NO_DECISION run
                                   length in the window (chronological).
      * ``n_completed_runs``     — len of the above.
      * ``n_significant_runs``   — closed runs ≥ ``MIN_RUN_FOR_STATS``;
                                   the basis for the percentile stats.
      * ``mean_run_length``      — mean over significant closed runs, or None.
      * ``median_run_length``    — median, or None.
      * ``p95_run_length``       — p95, or None.
      * ``max_run_length``       — max, or None.
      * ``verdict``              — one of ``RECOVERED`` / ``NOISE`` /
                                   ``INSUFFICIENT_HISTORY`` / ``WITHIN_NORMAL``
                                   / ``ELEVATED`` / ``ABNORMAL_WEDGE``.
      * ``verdict_detail``       — one-line explanation suitable for chat.
      * ``headline``             — dashboard / Discord one-liner.
    """
    try:
        w = max(1, int(window))
    except (TypeError, ValueError):
        w = DEFAULT_WINDOW
    rows = list(decisions or [])[:w]

    if not rows:
        return {
            "state": "NO_DATA",
            "window": w,
            "n_rows": 0,
            "open_run_length": 0,
            "completed_runs": [],
            "n_completed_runs": 0,
            "n_significant_runs": 0,
            "mean_run_length": None,
            "median_run_length": None,
            "p95_run_length": None,
            "max_run_length": None,
            "verdict": "INSUFFICIENT_HISTORY",
            "verdict_detail": "no decisions on record",
            "headline": "no decisions on record — cannot assess recovery time",
        }

    # Newest-first → oldest-first so run-length encoding produces runs in the
    # chronological order an operator reads them. The trailing run after the
    # walk is the OPEN run (touches "now").
    rows_oldest = list(reversed(rows))
    completed, open_run = _run_lengths(rows_oldest)

    significant = sorted(r for r in completed if r >= MIN_RUN_FOR_STATS)
    n_sig = len(significant)
    if n_sig:
        mean_len: float | None = sum(significant) / n_sig
        median_len: float | None = _percentile(significant, 0.5)
        p95_len: float | None = _percentile(significant, 0.95)
        max_len: int | None = max(significant)
    else:
        mean_len = median_len = p95_len = max_len = None

    # ── verdict ──────────────────────────────────────────────────
    if open_run == 0:
        verdict = "RECOVERED"
        verdict_detail = "latest decision is real — no open wedge"
    elif open_run < MIN_RUN_FOR_STATS:
        verdict = "NOISE"
        verdict_detail = (
            f"{open_run}-cycle blank streak — below {MIN_RUN_FOR_STATS}-cycle "
            "wedge threshold; not yet a stuck state"
        )
    elif n_sig < MIN_HISTORY_FOR_VERDICT:
        verdict = "INSUFFICIENT_HISTORY"
        verdict_detail = (
            f"open wedge: {open_run} cycles; only {n_sig} completed wedge(s) "
            f"in {w}-row window (need ≥ {MIN_HISTORY_FOR_VERDICT}) — cannot "
            "grade against history"
        )
    else:
        # n_sig and percentile values are non-None on this branch.
        assert median_len is not None and p95_len is not None
        if open_run >= p95_len:
            verdict = "ABNORMAL_WEDGE"
            verdict_detail = (
                f"open wedge {open_run} cycles ≥ historic p95 "
                f"({p95_len:.0f}) over {n_sig} wedges — escalate "
                "(restart runner, kill out-of-band Opus subprocesses)"
            )
        elif open_run > median_len:
            verdict = "ELEVATED"
            verdict_detail = (
                f"open wedge {open_run} cycles > historic median "
                f"({median_len:.0f}) but < p95 ({p95_len:.0f}) — watch, "
                "self-recovery still plausible"
            )
        else:
            verdict = "WITHIN_NORMAL"
            verdict_detail = (
                f"open wedge {open_run} cycles ≤ historic median "
                f"({median_len:.0f}) over {n_sig} wedges — typical, "
                "expect self-recovery"
            )

    # ── headline ─────────────────────────────────────────────────
    if n_sig and median_len is not None and p95_len is not None:
        hist = f"median/p95 over {n_sig} wedges: {median_len:.0f} / {p95_len:.0f}"
    else:
        hist = f"no comparable wedge history ({n_sig} significant)"
    headline = f"open wedge: {open_run} cycles; {hist} — {verdict}"

    return {
        "state": "OK",
        "window": w,
        "n_rows": len(rows),
        "open_run_length": open_run,
        "completed_runs": completed,
        "n_completed_runs": len(completed),
        "n_significant_runs": n_sig,
        "mean_run_length": round(mean_len, 2) if mean_len is not None else None,
        "median_run_length": median_len,
        "p95_run_length": p95_len,
        "max_run_length": max_len,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "headline": headline,
    }


def _main() -> int:
    """One-shot CLI: read the live store and pretty-print the recovery view."""
    import json
    from ..store import get_store
    out = build_no_decision_recovery(get_store().recent_decisions(DEFAULT_WINDOW))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
