"""Passive-signal-density detector — what was the news load while the engine
was sitting still?

A specific trader-grade gap no existing panel covers. The current state of
the system illustrates it sharply: 12 consecutive HOLD decisions, each one
with ``signal_count`` between 11-19 scored articles. The decision engine is
seeing rich news flow and still choosing HOLD CASH every cycle. That is a
*qualitatively different* failure mode from "I'm in HOLD because nothing is
happening": the latter is correct caution; the former is "deafening
silence" — Opus reading 19 catalysts and producing no action.

What the existing panels say (and what they miss):

  * ``/api/decision-paralysis`` reports streak LENGTH (HOLD_LOCK, IDLE_STORM,
    PASSIVE_LOOP) but not the news load DURING the streak.
  * ``/api/news-action-funnel`` is per-TICKER (DRAM has 4 articles / 0
    decisions); does not roll up to the book-wide "I had 19 signals per
    cycle and did nothing" view.
  * ``/api/decision-health.signal_count`` is an aggregate over the whole
    window, not scoped to the CURRENT passive run.
  * ``/api/capital-paralysis`` measures alpha cost; does not say whether the
    inaction was *informed* (no news → no trade) or *deafening* (lots of
    news → no trade).

The verdict ladder is sample-size-honest (the ``exit_only_streak`` /
``streak`` precedent): below the sample floor (``MIN_PASSIVE_RUN=5``) the
detector returns ``INSUFFICIENT`` and the silence-by-default reporter line
stays silent. Only the ``DEAFENING_SILENCE`` arm is actionable enough to
warrant a Discord line; the other verdicts populate the dashboard but stay
mute in the hourly summary so a quiet weekend does not add noise.

Verdict ladder:

============================  =====================================================
``NO_DATA``                   no decision rows in history
``NO_PASSIVE_RUN``            most recent decision is FILLED or BLOCKED
``INSUFFICIENT``              passive run has fewer than ``MIN_PASSIVE_RUN`` rows
``INFORMED_PASSIVE``          median ``signal_count`` ≤ ``LOW_SIGNAL_MEDIAN``
                              — quiet news + quiet trader is fine
``SIGNAL_RICH_PASSIVE``       ``LOW_SIGNAL_MEDIAN`` < median ≤ ``HIGH_SIGNAL_MEDIAN``
                              — moderate news, watching but not acting
``DEAFENING_SILENCE``         median > ``HIGH_SIGNAL_MEDIAN`` and run ≥ floor
                              — rich news flow, engine is paralysed
============================  =====================================================

Boundary semantics (pinned by tests):
  * median == ``LOW_SIGNAL_MEDIAN`` falls into ``INFORMED_PASSIVE`` (≤).
  * median == ``HIGH_SIGNAL_MEDIAN`` falls into ``SIGNAL_RICH_PASSIVE`` (≤).
  * Only ``> HIGH_SIGNAL_MEDIAN`` trips ``DEAFENING_SILENCE``.
  * n_passive == ``MIN_PASSIVE_RUN`` IS sufficient (≥, not >).

A passive run starts at the most-recent FILLED/BLOCKED decision (exclusive)
and extends to the newest row. A book that has *never* filled (a fresh-boot
trader whose entire history is HOLDs and NO_DECISIONs) treats the whole
table as the passive run — that's the most useful read for that case
(the trader needs to know "the engine has never traded; what's the news
load?"); ``state="STABLE"`` either way.

NO_DECISION cycles count as passive (Opus could not produce a decision —
that IS a form of inaction during a signal-loaded cycle); HOLD cycles count
as passive. FILLED and BLOCKED are *active* and terminate the run.

Advisory only — never gates Opus, never caps positions, never blocks a
trade. Mirrors the AGENTS.md #2 / #12 / #10 invariants (single source of
truth for the verdict + thresholds, observational-only contract).
"""
from __future__ import annotations

from datetime import datetime, timezone

# Verdict thresholds. ``3`` is the upper bound of an "essentially quiet"
# news cycle — typical scored-article counts in slow windows are 0-3
# (verified live: 23/100 decisions show signal_count ≤ 3 in the OK regime).
# ``10`` is the threshold at which a desk would say "there's real news flow
# happening" — a single big-name catalyst typically pushes the count past
# this in minutes.
LOW_SIGNAL_MEDIAN = 3
HIGH_SIGNAL_MEDIAN = 10

# Minimum passive-run length before the detector commits to a verdict.
# A 1-2 HOLD blip is statistically meaningless (e.g. one HOLD between two
# fills, or two NO_DECISION cycles inside an otherwise active book) —
# below this floor the verdict is ``INSUFFICIENT`` and the silence-by-
# default reporter line stays silent.
MIN_PASSIVE_RUN = 5

# Recent-decision tail surfaced to the consumer for quick eyeballing —
# the "recent_signal_counts" array mirrors ``exit_only_streak``'s
# ``recent_sequence`` shape so an operator can scan the actual signal-
# count rhythm.
RECENT_TAIL_LEN = 12

# Decisions whose action_taken indicates an ACTIVE decision (terminates a
# passive run). FILLED is obvious; BLOCKED is a real risk-rejected decision
# (different from NO_DECISION, which is a CLI failure). HOLD and
# NO_DECISION are passive. ``store.record_decision`` writes action_taken as
# either ``"NO_DECISION"`` / ``"BLOCKED"`` standalone OR as
# ``"<VERB> <TICKER> → <STATUS>"`` (e.g. "HOLD CASH → HOLD",
# "BUY NVDA → FILLED", "SELL AMD → BLOCKED"). We classify by the trailing
# status token (after the arrow) when present, falling back to the leading
# verb token otherwise.
_ACTIVE_STATUSES = ("FILLED", "BLOCKED")
_PASSIVE_STATUSES = ("HOLD",)


def _classify_decision(action_taken: str | None) -> str:
    """Return ``"ACTIVE"`` / ``"PASSIVE"`` / ``"UNKNOWN"`` for one row.

    Reads the trailing ``→ <STATUS>`` token when present (the canonical
    shape ``store.record_decision`` writes for resolved Opus decisions),
    falling back to the leading verb token (covers the bare
    ``"NO_DECISION"`` / ``"BLOCKED"`` shapes from earlier strategy.py
    paths). UNKNOWN rows are skipped entirely in ``build_passive_signal_density``
    — never coerced — mirroring the ``exit_only_streak._direction``
    discipline.
    """
    if not isinstance(action_taken, str):
        return "UNKNOWN"
    raw = action_taken.strip()
    if not raw:
        return "UNKNOWN"
    # Prefer the trailing → status token (canonical shape).
    if "→" in raw:
        status = raw.rsplit("→", 1)[1].strip().upper()
    else:
        # Bare label (e.g. "NO_DECISION" / "BLOCKED" / "HOLD")
        status = raw.split()[0].upper()
    if status in _ACTIVE_STATUSES:
        return "ACTIVE"
    if status in _PASSIVE_STATUSES or status.startswith("NO_DECISION"):
        return "PASSIVE"
    return "UNKNOWN"


def _coerce_signal_count(raw) -> int:
    """Best-effort int from a decisions.signal_count column value.

    Schema is ``INTEGER NOT NULL`` so a clean row never raises, but a
    historical row written before that constraint landed, or a typed-None
    in a test fixture, is degrade-safe coerced to 0. Mirrors the codebase
    pattern (e.g. ``urgent_articles`` ``ai_score or 0.0``).
    """
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return 0


def _median(values: list[int]) -> float:
    """Sample median of a non-empty int sequence.

    Pure helper — caller never passes [] (it short-circuits to the
    ``INSUFFICIENT`` verdict before this call). The return is float
    because even ints can produce a .5 median on an even count.
    """
    n = len(values)
    s = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def build_passive_signal_density(decisions: list[dict]) -> dict:
    """Compute the passive-signal-density snapshot.

    ``decisions`` must be newest → oldest (the convention
    ``store.recent_decisions`` returns by default — same orientation
    as ``decision_paralysis``'s contract). Unknown-status rows are
    silently skipped (mirrors ``exit_only_streak._direction``); they
    neither terminate the passive run nor count toward its length.

    Returned dict mirrors the existing ``exit_only_streak`` /
    ``decision_paralysis`` shape — ``as_of``, ``state``, ``verdict``,
    ``headline``, raw counters callers may want to render, plus the
    pinned thresholds for verdict-ladder transparency. Pure read;
    never raises; every field is degrade-safe.
    """
    now = datetime.now(timezone.utc)

    # Normalize to (status, signal_count, ts) tuples. Drop UNKNOWN rows
    # entirely — they neither terminate nor contribute to the run.
    classified: list[tuple[str, int, str | None]] = []
    for d in decisions or []:
        status = _classify_decision(d.get("action_taken"))
        if status == "UNKNOWN":
            continue
        sc = _coerce_signal_count(d.get("signal_count"))
        ts = d.get("timestamp")
        classified.append((status, sc, ts))

    n_total = len(classified)
    if n_total == 0:
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": "NO_DATA",
            "verdict": None,
            "headline": "No decision rows yet — passive-signal-density not available.",
            "n_passive": 0,
            "n_total_scanned": 0,
            "median_signal_count": None,
            "max_signal_count": None,
            "min_signal_count": None,
            "n_signal_rich_cycles": 0,
            "passive_run_started_ts": None,
            "passive_run_ended_ts": None,
            "most_recent_active_ts": None,
            "most_recent_active_action": None,
            "recent_signal_counts": [],
            "low_signal_median": LOW_SIGNAL_MEDIAN,
            "high_signal_median": HIGH_SIGNAL_MEDIAN,
            "min_passive_run": MIN_PASSIVE_RUN,
            "high_signal_threshold": HIGH_SIGNAL_MEDIAN,
        }

    # Walk newest-first; the passive run is the prefix of PASSIVE rows
    # before the most-recent ACTIVE row. If there is no ACTIVE row in
    # the entire table (fresh book), the prefix is everything (handled
    # naturally — most_recent_active stays None).
    most_recent_active: tuple[str, int, str | None] | None = None
    passive_run: list[tuple[str, int, str | None]] = []
    for row in classified:  # newest-first
        if row[0] == "ACTIVE":
            most_recent_active = row
            break
        passive_run.append(row)

    # The most recent decision row IS active → no current passive run.
    if classified[0][0] == "ACTIVE":
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": "STABLE",
            "verdict": "NO_PASSIVE_RUN",
            "headline": (
                f"Most recent decision was {classified[0][0]} — "
                f"book is not in a passive run."
            ),
            "n_passive": 0,
            "n_total_scanned": n_total,
            "median_signal_count": None,
            "max_signal_count": None,
            "min_signal_count": None,
            "n_signal_rich_cycles": 0,
            "passive_run_started_ts": None,
            "passive_run_ended_ts": None,
            "most_recent_active_ts": most_recent_active[2] if most_recent_active else None,
            "most_recent_active_action": (
                None  # not surfaced for the no-passive-run arm
            ),
            "recent_signal_counts": [sc for _, sc, _ in classified[:RECENT_TAIL_LEN]],
            "low_signal_median": LOW_SIGNAL_MEDIAN,
            "high_signal_median": HIGH_SIGNAL_MEDIAN,
            "min_passive_run": MIN_PASSIVE_RUN,
            "high_signal_threshold": HIGH_SIGNAL_MEDIAN,
        }

    n_passive = len(passive_run)
    counts = [sc for _, sc, _ in passive_run]
    med = _median(counts)
    mx = max(counts)
    mn = min(counts)
    n_signal_rich = sum(1 for c in counts if c >= HIGH_SIGNAL_MEDIAN)

    # passive_run is newest-first; the "started" boundary is the OLDEST
    # row in the run (the last element) and the "ended" boundary is the
    # NEWEST (the first element). When there is no most_recent_active,
    # the run started at the oldest decision in the table.
    passive_started_ts = passive_run[-1][2]
    passive_ended_ts = passive_run[0][2]
    mra_ts = most_recent_active[2] if most_recent_active else None
    # Surface a coarse, never-empty fallback for the most-recent-active
    # action (the bare status word — the action_taken string is opaque
    # otherwise; tests only assert presence).
    mra_action = most_recent_active[0] if most_recent_active else None

    # Verdict selection — boundary discipline pinned in tests.
    if n_passive < MIN_PASSIVE_RUN:
        verdict = "INSUFFICIENT"
        headline = (
            f"INSUFFICIENT — only {n_passive} passive cycle"
            f"{'s' if n_passive != 1 else ''} since the last active "
            f"decision (need {MIN_PASSIVE_RUN}+ for a verdict)."
        )
    elif med <= LOW_SIGNAL_MEDIAN:
        verdict = "INFORMED_PASSIVE"
        headline = (
            f"INFORMED_PASSIVE — {n_passive} passive cycles with median "
            f"{med:.1f} signals/cycle (≤{LOW_SIGNAL_MEDIAN}). "
            f"Engine is correctly quiet during a quiet news window."
        )
    elif med <= HIGH_SIGNAL_MEDIAN:
        verdict = "SIGNAL_RICH_PASSIVE"
        headline = (
            f"SIGNAL_RICH_PASSIVE — {n_passive} passive cycles with median "
            f"{med:.1f} signals/cycle (between {LOW_SIGNAL_MEDIAN} and "
            f"{HIGH_SIGNAL_MEDIAN}). Moderate news, watching but not acting."
        )
    else:
        verdict = "DEAFENING_SILENCE"
        headline = (
            f"DEAFENING_SILENCE — {n_passive} passive cycles with median "
            f"{med:.1f} signals/cycle (>{HIGH_SIGNAL_MEDIAN}); "
            f"{n_signal_rich}/{n_passive} cycles had ≥{HIGH_SIGNAL_MEDIAN} signals. "
            f"Rich news flow, engine produced no trade — actionable trader review."
        )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "STABLE",
        "verdict": verdict,
        "headline": headline,
        "n_passive": n_passive,
        "n_total_scanned": n_total,
        "median_signal_count": med,
        "max_signal_count": mx,
        "min_signal_count": mn,
        "n_signal_rich_cycles": n_signal_rich,
        "passive_run_started_ts": passive_started_ts,
        "passive_run_ended_ts": passive_ended_ts,
        "most_recent_active_ts": mra_ts,
        "most_recent_active_action": mra_action,
        "recent_signal_counts": [sc for _, sc, _ in classified[:RECENT_TAIL_LEN]],
        "low_signal_median": LOW_SIGNAL_MEDIAN,
        "high_signal_median": HIGH_SIGNAL_MEDIAN,
        "min_passive_run": MIN_PASSIVE_RUN,
        "high_signal_threshold": HIGH_SIGNAL_MEDIAN,
    }


if __name__ == "__main__":  # smoke test against the live DB
    import json

    from paper_trader.store import get_store
    s = get_store()
    # store.recent_decisions returns newest-first — exactly the orientation
    # build_passive_signal_density consumes.
    rep = build_passive_signal_density(s.recent_decisions(limit=500))
    print(json.dumps(rep, indent=2, default=str))
