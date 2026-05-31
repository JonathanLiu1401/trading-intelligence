"""Time-windowed parse-fail breakdown — *is the deploy/restart actually helping?*

``decision_reliability`` collapses the parse-fail rate into a single
all-time-current-regime number. That answers "how bad is it on average," but the
operator's *next* question is always temporal: *did the last restart / prompt
tweak / host-saturation fix improve the rate, or am I about to ship another
inert change?* The all-time aggregate cannot answer that — by construction it
averages across the whole post-restart window, so a fresh fix's effect is
diluted to the point of invisibility for hours.

This module reports the parse-fail rate **per sliding window** (1h / 6h / 24h /
7d) plus a trend verdict comparing the freshest window (1h) to the next one
(6h). When both windows have enough samples to judge (``MIN_WIN_N=5``), a 10pp
move in either direction names the trend; otherwise it stays STABLE / withheld.

The forensics slice exposes the **most-recent N parse-fail rows** with their
mode + excerpt — the breakdown a CRITICAL aggregate can't give. Operator can
read "the last 5 failures were all HOST_SATURATED_SKIP at 14:32–14:35" and
diagnose immediately without trawling the log.

Pure composition of the canonical taxonomy helpers
(``decision_forensics.classify_failure`` / ``_is_no_decision`` / ``_parse_ts``) —
no metric is re-derived. Single source of truth. Advisory only — never gates
Opus, adds no caps (AGENTS.md invariants #2 / #12).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .decision_forensics import (
    MODES,
    _is_no_decision,
    _parse_ts,
    classify_failure,
)

# Per-window minimum decision count before a rate is judged.
# Mirrors decision_reliability.MIN_CURRENT's sample-honesty convention — a 1/1
# = 100% rate can never name the window CRITICAL.
MIN_WIN_N = 5

# Trend-naming pp gap between the 1h and 6h failure rates. Below this the
# trend stays STABLE so noise can't false-call a regression.
TREND_PP = 10.0

# Window definitions (hours back from now), oldest→newest display order.
# 1h is freshest → leftmost; 7d is the long-tail baseline. Order is deliberate:
# trend logic compares windows[0] (1h) against windows[1] (6h).
_WINDOWS = (
    ("1h", 1.0),
    ("6h", 6.0),
    ("24h", 24.0),
    ("7d", 24.0 * 7),
)

# Most-recent parse-fail rows surfaced for forensics. Operator-actionable —
# six rows comfortably fit one dashboard card without scrolling.
_FORENSICS_N = 10


def _classify_row(d: dict) -> tuple[datetime | None, bool, str | None,
                                    str | None]:
    """Return (ts, is_no_decision, mode, excerpt) for one decision row.

    Excerpt is None unless the row is a parse-fail (else there's no payload).
    """
    ts = _parse_ts(d.get("timestamp"))
    nd = _is_no_decision(d.get("action_taken"))
    if not nd:
        return (ts, False, None, None)
    cls = classify_failure(d.get("reasoning"))
    return (ts, True, cls["mode"], cls["excerpt"] or None)


def _mode_mix(failrows: list[tuple[datetime, str]]) -> list[dict]:
    """Frequency table over failure modes, descending by count then MODES order.

    Identical contract to ``decision_reliability.current_mode_mix`` — the two
    panels surface the same shape so a UI can render them with one component.
    """
    by_mode: dict[str, int] = {}
    for _, mode in failrows:
        by_mode[mode] = by_mode.get(mode, 0) + 1
    total = sum(by_mode.values())
    if not total:
        return []
    return sorted(
        ({"mode": m, "n": c, "pct": round(c / total * 100, 1)}
         for m, c in by_mode.items()),
        key=lambda r: (-r["n"],
                       MODES.index(r["mode"]) if r["mode"] in MODES else 99),
    )


def build_parse_fail_windows(decisions: list[dict],
                             now: datetime | None = None) -> dict:
    """Per-window parse-fail rates + trend verdict + recent-failure forensics.

    ``decisions`` is newest-first (as ``store.recent_decisions`` returns).
    Pure: never touches the DB. ``now`` is injectable for deterministic tests.
    Never raises — degrades to ``state='NO_DATA'`` on any pathological input.
    """
    now = now or datetime.now(timezone.utc)
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_decisions_total": 0,
        "windows": [],
        "trend": "INSUFFICIENT",
        "trend_delta_pp": None,
        "recent_failures": [],
        "min_win_n": MIN_WIN_N,
        "trend_pp": TREND_PP,
        "state": "NO_DATA",
        "headline": "No decisions recorded yet.",
    }
    if not decisions:
        return out

    # Single pass over the row list — the classifier is the only expensive
    # part (substring scan over reasoning); doing it once and reusing the
    # result across every window keeps this O(n) regardless of window count.
    classified: list[tuple[datetime | None, bool, str | None, str | None,
                           dict]] = []
    for d in decisions:
        ts, nd, mode, excerpt = _classify_row(d)
        classified.append((ts, nd, mode, excerpt, d))

    out["n_decisions_total"] = len(classified)

    # Per-window aggregates. A row with an unparseable timestamp is excluded
    # from every window (same convention as decision_reliability /
    # decision_forensics; an undated row can't honestly belong to a time
    # bucket). Such rows still count toward n_decisions_total above so the
    # operator sees they exist.
    windows_out: list[dict] = []
    for label, hours in _WINDOWS:
        cutoff = now - timedelta(hours=hours)
        in_win: list[tuple[datetime, bool, str | None]] = []
        fail_pairs: list[tuple[datetime, str]] = []
        for ts, nd, mode, _ex, _d in classified:
            if ts is None or ts < cutoff:
                continue
            in_win.append((ts, nd, mode))
            if nd and mode is not None:
                fail_pairs.append((ts, mode))
        n = len(in_win)
        f = len(fail_pairs)
        rate = round(f / n * 100, 1) if n else 0.0
        ok = n >= MIN_WIN_N
        # Per-window state mirrors decision_forensics' verdict bands so the two
        # never disagree on the same data window.
        if not ok:
            win_state = "INSUFFICIENT"
        elif rate >= 50.0:
            win_state = "CRITICAL"
        elif rate >= 25.0:
            win_state = "DEGRADED"
        else:
            win_state = "HEALTHY"
        windows_out.append({
            "label": label,
            "window_hours": round(hours, 2),
            "n_decisions": n,
            "n_failures": f,
            "failure_rate_pct": rate,
            "mode_mix": _mode_mix(fail_pairs),
            "state": win_state,
            "sufficient": ok,
        })
    out["windows"] = windows_out

    # Trend verdict — compares 1h (windows_out[0]) vs 6h (windows_out[1]). Both
    # must be sufficient; otherwise the call is honestly withheld. The 1h rate
    # is the freshest read; a 1h rate below the 6h rate by ≥ TREND_PP means
    # the most recent hour is healthier than the prior six — IMPROVING.
    short = windows_out[0]
    medium = windows_out[1]
    if short["sufficient"] and medium["sufficient"]:
        delta = round(short["failure_rate_pct"] - medium["failure_rate_pct"],
                      1)
        out["trend_delta_pp"] = delta
        if delta <= -TREND_PP:
            out["trend"] = "IMPROVING"
        elif delta >= TREND_PP:
            out["trend"] = "WORSENING"
        else:
            out["trend"] = "STABLE"
    # else: trend stays INSUFFICIENT (default)

    # Forensics: most-recent parse-fail rows (newest first; decisions is
    # already newest-first so a single linear scan suffices). Capped at
    # _FORENSICS_N so the panel stays bounded.
    recent: list[dict] = []
    for ts, nd, mode, excerpt, _d in classified:
        if not nd or ts is None:
            continue
        recent.append({
            "timestamp": ts.isoformat(timespec="seconds"),
            "mode": mode,
            "excerpt": excerpt or "",
        })
        if len(recent) >= _FORENSICS_N:
            break
    out["recent_failures"] = recent

    # Aggregate state for the headline. Drive from the 6h window (the
    # canonical "what is the system doing right now" sample — long enough to be
    # stable, short enough to reflect a recent fix). Falls back to 24h, then
    # 7d, then NO_DATA if no window has enough samples.
    canonical = None
    for w in windows_out[1:]:
        if w["sufficient"]:
            canonical = w
            break
    if canonical is None:
        # Even 7d has < MIN_WIN_N — system is brand-new or near-idle.
        out["state"] = "NO_DATA"
        out["headline"] = (
            f"INSUFFICIENT — fewer than {MIN_WIN_N} decisions in any window; "
            f"trend not yet judgeable.")
    else:
        out["state"] = canonical["state"]
        win_label = canonical["label"]
        rate = canonical["failure_rate_pct"]
        if out["trend"] == "INSUFFICIENT":
            trend_clause = " (trend withheld — 1h sample too thin)"
        else:
            delta = out["trend_delta_pp"]
            sign = "+" if (delta or 0) >= 0 else ""
            trend_clause = (
                f" — last hour {short['failure_rate_pct']}% "
                f"({sign}{delta}pp vs 6h, {out['trend']})")
        out["headline"] = (
            f"{canonical['state']} — {win_label} parse-fail {rate}% over "
            f"{canonical['n_decisions']} cycle(s){trend_clause}.")

    return out
