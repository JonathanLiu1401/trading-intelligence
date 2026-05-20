"""Decision-action paralysis — consecutive HOLD streaks on the live book.

The desk has two failure modes for "the loop is alive but nothing is
happening":

* **IDLE_STORM** — runs of back-to-back ``NO_DECISION`` (Claude returned
  unparseable text). Already detected by
  ``runner_heartbeat._decision_efficacy`` and ``decision_forensics`` at the
  ``runner.CONSECUTIVE_NO_DECISION_LIMIT`` threshold (5). This is the
  "engine wedged" failure.
* **HOLD_LOCK** — runs of back-to-back ``HOLD`` decisions. Every cycle
  produced a parseable Opus decision; Opus chose to hold every time. The
  loop *is* deciding, ``decision_health`` reports HEALTHY, ``runner_heartbeat``
  reports HEALTHY, and yet the book has not moved for hours. This is the
  documented PARALYSIS failure mode — Opus is too cautious or the prompt's
  context is identical cycle after cycle so the answer is identical too.

No existing endpoint detects ``HOLD_LOCK``. ``decision_health`` aggregates
HOLD% over 24h (a 95% HOLD share looks the same whether spread across
half the day or stacked into one immovable block). ``ticker_decision_mix``
counts verbs per ticker but does not detect *contiguous* runs.
``runner_heartbeat`` only fires on ``NO_DECISION`` streaks. ``streak``
operates on closed round-trips, not on decisions.

This module fills the gap. The builder is pure: takes a newest-first list
of decision rows (matching ``store.recent_decisions(N)`` shape — a dict
per row with ``action_taken`` and ``timestamp``), returns a verdict.

The verdict ladder, **most-specific first**:

  * ``IDLE_STORM``    — current leading run is all NO_DECISION and
    ``len ≥ IDLE_STORM_THRESHOLD`` (mirrors
    ``runner_heartbeat.NO_DECISION_STORM_THRESHOLD = 5``). Re-emitted here
    so a caller composing this single panel still sees the wedge; the
    primary owner is still ``runner_heartbeat``.
  * ``HOLD_LOCK``     — current leading run is all HOLD and
    ``len ≥ HOLD_LOCK_THRESHOLD`` (10 — calibrated to ~1 market-open hour
    at the OPEN_INTERVAL_S=1800s cadence, an order of magnitude beyond a
    normal HOLD cluster).
  * ``PASSIVE_LOOP``  — current leading run is all HOLD ∪ NO_DECISION and
    ``len ≥ PASSIVE_LOOP_THRESHOLD`` (15) but neither of the more-specific
    bands fires (mixed HOLD+NO_DECISION run).
  * ``ACTIVE``        — none of the above; the loop has produced at least
    one FILLED/BLOCKED action inside the most recent
    ``PASSIVE_LOOP_THRESHOLD`` cycles.
  * ``NO_DATA``       — empty input.

Sample-size discipline mirrors ``hold_discipline`` / ``streak``: when the
total scanned window is below the smallest threshold, the verdict is
``ACTIVE`` only if a non-passive row exists; otherwise ``ACTIVE`` falls
through but the counts are still emitted so a caller can format an
emerging-mode card.

Observational only — never gates Opus, no caps (AGENTS.md #2/#12).
Mirrors the runner_heartbeat / decision_health endpoint contract:
advisory surface, never a hard restart, never injected into the prompt.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Thresholds. Module-owned (the runner_heartbeat / feed_health precedent —
# the module is the spec; tests read these constants so a retune cannot
# false-fail them).
IDLE_STORM_THRESHOLD = 5     # mirrors runner_heartbeat.NO_DECISION_STORM_THRESHOLD
HOLD_LOCK_THRESHOLD = 10     # ~1h at OPEN_INTERVAL_S=1800; 10x normal cluster
PASSIVE_LOOP_THRESHOLD = 15  # broader passive-action run

# 24h window for the secondary "longest passive run in last 24h" stat —
# bounds the scan and matches decision_health's 24h convention.
LOOKBACK_HOURS = 24.0


def _parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _classify(action_taken):
    """Map ``action_taken`` to one of HOLD / NO_DECISION / FILLED / BLOCKED
    / OTHER.

    Verbatim mirror of ``decision_health._classify``'s outcome bucket — the
    canonical predicate (AGENTS.md invariant #10). Inlined to keep this leaf
    pure and free of cross-analytics import; drift-locked by
    ``tests/test_decision_paralysis.py::test_classify_mirrors_decision_health``.
    """
    raw = (action_taken or "").strip()
    if not raw or raw == "NO_DECISION":
        return "NO_DECISION"
    outcome = raw.split("→")[-1].strip().upper() if "→" in raw else raw.upper()
    if outcome in ("FILLED", "HOLD", "BLOCKED"):
        return outcome
    return "OTHER"


def _leading_run(categories, predicate):
    """Length of the leading run (newest-first) where every element satisfies
    ``predicate``. Stops at the first non-matching element.
    """
    n = 0
    for c in categories:
        if predicate(c):
            n += 1
        else:
            break
    return n


def _longest_run(categories, predicate):
    """Longest contiguous run where ``predicate`` holds. Order-agnostic."""
    best = cur = 0
    for c in categories:
        if predicate(c):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def build_decision_paralysis(decisions, now=None):
    """Verdict on consecutive passive-action streaks.

    ``decisions`` is newest-first (matching ``store.recent_decisions(N)``).
    Each row is a dict with at least ``action_taken`` and (optionally)
    ``timestamp``. Garbage rows degrade — never raise.
    """
    now = now or datetime.now(timezone.utc)
    out = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_decisions_scanned": 0,
        "n_decisions_24h": 0,
        "current_hold_streak": 0,
        "current_no_decision_streak": 0,
        "current_passive_streak": 0,
        "longest_hold_streak_24h": 0,
        "longest_passive_streak_24h": 0,
        "last_active_action": None,
        "last_active_ts": None,
        "hold_lock_threshold": HOLD_LOCK_THRESHOLD,
        "idle_storm_threshold": IDLE_STORM_THRESHOLD,
        "passive_loop_threshold": PASSIVE_LOOP_THRESHOLD,
        "verdict": "NO_DATA",
        "headline": "No decisions recorded — cannot assess paralysis.",
    }
    if not decisions:
        return out

    cats = [_classify(r.get("action_taken")) for r in decisions]
    out["n_decisions_scanned"] = len(cats)

    # 24h-window slice for the longest-run stats. Newest-first; stop at the
    # first row older than LOOKBACK_HOURS so the bound is contiguous.
    cutoff = LOOKBACK_HOURS * 3600.0
    cats_24h = []
    for r, c in zip(decisions, cats):
        ts = _parse_ts(r.get("timestamp"))
        if ts is None:
            # Unparseable ts: include it so a fully-tsless test set isn't empty,
            # but downstream stats will mostly use the contiguous-from-newest
            # leading runs which are ts-agnostic.
            cats_24h.append(c)
            continue
        if (now - ts).total_seconds() > cutoff:
            break
        cats_24h.append(c)

    out["n_decisions_24h"] = len(cats_24h)

    is_hold = lambda c: c == "HOLD"
    is_nd = lambda c: c == "NO_DECISION"
    is_passive = lambda c: c in ("HOLD", "NO_DECISION")
    is_active = lambda c: c in ("FILLED", "BLOCKED")

    out["current_hold_streak"] = _leading_run(cats, is_hold)
    out["current_no_decision_streak"] = _leading_run(cats, is_nd)
    out["current_passive_streak"] = _leading_run(cats, is_passive)
    out["longest_hold_streak_24h"] = _longest_run(cats_24h, is_hold)
    out["longest_passive_streak_24h"] = _longest_run(cats_24h, is_passive)

    # Find the most recent FILLED/BLOCKED row for context.
    for r, c in zip(decisions, cats):
        if is_active(c):
            out["last_active_action"] = r.get("action_taken")
            ts = _parse_ts(r.get("timestamp"))
            if ts is not None:
                out["last_active_ts"] = ts.isoformat(timespec="seconds")
                out["hours_since_last_active"] = round(
                    (now - ts).total_seconds() / 3600.0, 2)
            break

    # Verdict ladder — most-specific first.
    nd_run = out["current_no_decision_streak"]
    hold_run = out["current_hold_streak"]
    passive_run = out["current_passive_streak"]

    if nd_run >= IDLE_STORM_THRESHOLD:
        out["verdict"] = "IDLE_STORM"
        out["headline"] = (
            f"IDLE_STORM — the last {nd_run} cycles produced NO_DECISION. "
            f"Engine is cycling but Claude returned unparseable text every "
            f"time; restart paper-trader may clear a wedged CLI.")
    elif hold_run >= HOLD_LOCK_THRESHOLD:
        out["verdict"] = "HOLD_LOCK"
        ago = out.get("hours_since_last_active")
        ago_str = f"{ago:.1f}h" if isinstance(ago, (int, float)) else "—"
        out["headline"] = (
            f"HOLD_LOCK — the last {hold_run} consecutive cycles were HOLD "
            f"(no FILLED/BLOCKED for {ago_str}). Opus is deciding every "
            f"cycle but never moving the book; the prompt context may be "
            f"identical cycle-to-cycle.")
    elif passive_run >= PASSIVE_LOOP_THRESHOLD:
        out["verdict"] = "PASSIVE_LOOP"
        out["headline"] = (
            f"PASSIVE_LOOP — the last {passive_run} cycles were all HOLD or "
            f"NO_DECISION (mixed; HOLD_LOCK and IDLE_STORM thresholds not "
            f"met individually). No trade activity but the book is alive.")
    else:
        out["verdict"] = "ACTIVE"
        out["headline"] = (
            f"ACTIVE — recent decision flow includes FILLED/BLOCKED "
            f"actions; the loop is not paralysed.")

    return out
