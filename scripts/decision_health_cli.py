#!/usr/bin/env python3
"""Headless NO_DECISION triage for the live paper trader.

Why this exists
---------------
When the live trader is healthy it records BUY/SELL/HOLD decisions; when Opus
fails to produce a usable JSON decision it records a ``NO_DECISION`` row in
``data/paper_trader.db`` with a free-text ``reasoning`` string. A raw count of
NO_DECISION rows is *deeply misleading* because the reasoning strings span
several code generations:

  * ``parse_failed: <raw>``                     — CURRENT code. Opus returned
    text but it wasn't parseable (prose-wrapped / truncated). One-shot retry
    not yet attempted, or this was the first attempt.
  * ``retry_failed: <raw>``                     — CURRENT code. The JSON-only
    retry nudge also failed to parse.
  * ``claude returned no response (timeout/empty)`` — CURRENT code. Both the
    Opus call and the Sonnet fallback returned nothing (timeout / empty stdout
    / CLI non-zero).
  * ``claude returned no parseable JSON``       — LEGACY. This string was
    emitted by a code path that was *removed* in commit 7bfc26a / 6734e19
    ("permanently eliminate NO_DECISION via 5-layer defence"). Rows carrying
    it are historical artefacts of removed code, **not** an actionable defect
    in the running system.

The dashboard's ``/api/decision-health`` reports a raw rate but does not split
historical removed-code rows from current-code failures, and it needs the
Flask server + a browser. During incident response (SSH, no browser) an
operator needs a one-shot answer to: *"Is the NO_DECISION rate a problem the
running code is causing right now, or dead records from code that no longer
exists?"* This script answers exactly that, with zero third-party deps and a
read-only WAL-safe DB open.

Usage
-----
    cd /home/zeph/paper-trader
    python3 scripts/decision_health_cli.py                 # human summary
    python3 scripts/decision_health_cli.py --json          # machine output
    python3 scripts/decision_health_cli.py --days 30       # 30-day by-day table
    python3 scripts/decision_health_cli.py --db /path.db   # explicit DB
    python3 scripts/decision_health_cli.py --selftest      # run assertions

Exit codes (so it is usable from cron / alerting):
    0  OK or WARN (or --selftest passed)
    1  --selftest failed, or DB unreadable
    2  ALERT — current code is failing to decide at a high rate
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "paper_trader.db"

# Reasoning-string taxonomy. ``legacy_*`` categories describe rows written by
# code that has since been removed — they are NOT current-system defects.
CAT_PARSE_FAILED = "parse_failed"            # current code, unparseable text
CAT_RETRY_FAILED = "retry_failed"            # current code, retry also failed
CAT_TIMEOUT_EMPTY = "timeout_empty"          # current code, Opus+Sonnet None
CAT_LEGACY_NO_JSON = "legacy_no_parseable_json"   # removed code path
CAT_LEGACY_OTHER = "legacy_other"            # any other historical string

# Categories produced by code that is still in the tree. Only these should
# drive the health verdict; legacy_* rows are inert history.
CURRENT_CODE_CATS = {CAT_PARSE_FAILED, CAT_RETRY_FAILED, CAT_TIMEOUT_EMPTY}

_LEGACY_NO_JSON_STR = "claude returned no parseable json"
_TIMEOUT_EMPTY_STR = "claude returned no response (timeout/empty)"


def is_no_decision(action_taken: str | None) -> bool:
    """The trader stores NO_DECISION as a free-text action label. Tolerate
    surrounding text (e.g. ``"NO_DECISION"`` exactly, never a ticker form)."""
    return bool(action_taken) and "NO_DECISION" in action_taken.upper()


def classify_reason(reasoning: str | None) -> str:
    """Map a NO_DECISION ``reasoning`` string onto the taxonomy above.

    The classification is prefix/exact based and deliberately conservative:
    an unrecognised string is ``legacy_other`` (history), never silently
    counted as a current-code failure that would inflate the alert rate."""
    r = (reasoning or "").strip()
    low = r.lower()
    if low.startswith("parse_failed:"):
        return CAT_PARSE_FAILED
    if low.startswith("retry_failed:"):
        return CAT_RETRY_FAILED
    if low == _TIMEOUT_EMPTY_STR:
        return CAT_TIMEOUT_EMPTY
    if low == _LEGACY_NO_JSON_STR:
        return CAT_LEGACY_NO_JSON
    return CAT_LEGACY_OTHER


def _verdict(current_recent: int, total_recent: int) -> str:
    """Health verdict from the *current-code* NO_DECISION ratio over the most
    recent decisions only. Legacy rows are excluded by the caller. Requires a
    minimum sample so a single failed hourly poll on a closed-market weekend
    does not raise a false ALERT."""
    if total_recent < 5:
        return "OK"  # too few recent decisions to judge
    ratio = current_recent / total_recent
    if ratio >= 0.5:
        return "ALERT"
    if ratio >= 0.2:
        return "WARN"
    return "OK"


def summarize(rows: list[tuple], recent_n: int = 25, days: int = 14) -> dict:
    """rows: list of (timestamp, action_taken, reasoning), newest first.

    Returns a structured report. ``recent_n`` is how many of the most recent
    decisions feed the health verdict (a rolling window robust to long-idle
    closed-market stretches that the absolute clock would distort)."""
    total = len(rows)
    nd_rows = [(ts, rsn) for ts, act, rsn in rows if is_no_decision(act)]
    cat_counts: Counter = Counter()
    for _ts, rsn in nd_rows:
        cat_counts[classify_reason(rsn)] += 1

    # Recent window: the verdict must reflect the code running *now*, so only
    # current-code categories in the most recent `recent_n` decisions count.
    recent = rows[:recent_n]
    recent_total = len(recent)
    recent_current_nd = sum(
        1 for ts, act, rsn in recent
        if is_no_decision(act) and classify_reason(rsn) in CURRENT_CODE_CATS
    )

    # By-day NO_DECISION counts (calendar days, newest first, capped).
    by_day: Counter = Counter()
    for ts, rsn in nd_rows:
        by_day[(ts or "")[:10]] += 1
    by_day_sorted = sorted(
        ((d, n) for d, n in by_day.items() if d), reverse=True
    )[:days]

    return {
        "total_decisions": total,
        "total_no_decision": len(nd_rows),
        "no_decision_rate": round(len(nd_rows) / total, 4) if total else 0.0,
        "by_category": dict(cat_counts),
        "legacy_no_decision": sum(
            cat_counts[c] for c in (CAT_LEGACY_NO_JSON, CAT_LEGACY_OTHER)
        ),
        "current_code_no_decision": sum(
            cat_counts[c] for c in CURRENT_CODE_CATS
        ),
        "recent_window": recent_total,
        "recent_current_code_no_decision": recent_current_nd,
        "verdict": _verdict(recent_current_nd, recent_total),
        "by_day": by_day_sorted,
    }


def load_rows(db_path: Path) -> list[tuple]:
    """Read decisions newest-first, read-only + WAL-safe (invariant #7:
    any non-trader reader must use ?mode=ro to avoid lock contention)."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    try:
        cur = conn.execute(
            "SELECT timestamp, action_taken, reasoning "
            "FROM decisions ORDER BY id DESC"
        )
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]
    finally:
        conn.close()


def _render(rep: dict) -> str:
    v = rep["verdict"]
    icon = {"OK": "✅", "WARN": "⚠️", "ALERT": "🚨"}.get(v, "?")
    lines = [
        "── live trader NO_DECISION triage ──",
        f"  total decisions      : {rep['total_decisions']}",
        f"  total NO_DECISION    : {rep['total_no_decision']} "
        f"({rep['no_decision_rate']:.1%})",
        f"  legacy (removed code): {rep['legacy_no_decision']}  "
        f"← historical, not actionable",
        f"  current-code failures: {rep['current_code_no_decision']}",
        "  by category:",
    ]
    for cat, n in sorted(rep["by_category"].items(), key=lambda kv: -kv[1]):
        marker = "  (legacy)" if cat.startswith("legacy_") else ""
        lines.append(f"    {cat:<26} {n}{marker}")
    lines.append(
        f"  recent {rep['recent_window']} decisions: "
        f"{rep['recent_current_code_no_decision']} current-code NO_DECISION"
    )
    if rep["by_day"]:
        lines.append("  NO_DECISION by day (newest first):")
        for d, n in rep["by_day"]:
            lines.append(f"    {d}  {n}")
    lines.append(f"  {icon} VERDICT: {v}")
    if v == "OK" and rep["legacy_no_decision"] > rep["current_code_no_decision"]:
        lines.append(
            "  note: most NO_DECISION rows are dead records from removed "
            "code paths — the running trader is healthy."
        )
    return "\n".join(lines)


def _selftest() -> int:
    """Assertion-based self-test against an in-memory DB with known rows.
    Verifies the exact taxonomy split and verdict thresholds — this is the
    real correctness check for the categorisation logic."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE decisions (id INTEGER PRIMARY KEY, timestamp TEXT, "
        "market_open INT, signal_count INT, action_taken TEXT, "
        "reasoning TEXT, portfolio_value REAL, cash REAL)"
    )
    seed = [
        # Written newest-first; inserted reversed below so that the first
        # element gets the highest id and therefore sorts first under the
        # production query's ORDER BY id DESC. The recent window must hold
        # >=5 rows (the min-sample verdict guard) with a current-code
        # NO_DECISION majority to legitimately trigger ALERT.
        ("2026-05-17T06:00:00", "NO_DECISION", _TIMEOUT_EMPTY_STR),       # cur
        ("2026-05-17T05:30:00", "NO_DECISION", "parse_failed: {oops"),    # cur
        ("2026-05-17T05:00:00", "NO_DECISION", "retry_failed: still bad"),# cur
        ("2026-05-17T04:30:00", "NO_DECISION", _TIMEOUT_EMPTY_STR),       # cur
        ("2026-05-17T04:00:00", "BUY NVDA → FILLED", "thesis ok"),        # ok
        ("2026-05-16T12:00:00", "NO_DECISION", "parse_failed: trunc"),    # cur
        ("2026-05-14T09:00:00", "NO_DECISION", "claude returned no parseable JSON"),
        ("2026-05-14T08:00:00", "NO_DECISION", "claude returned no parseable JSON"),
        ("2026-05-14T07:00:00", "NO_DECISION", "some ancient string"),    # legacy
        ("2026-05-13T10:00:00", "HOLD MU → HOLD", "holding"),             # ok
    ]
    for i, (ts, act, rsn) in enumerate(reversed(seed), 1):
        conn.execute(
            "INSERT INTO decisions (id,timestamp,action_taken,reasoning) "
            "VALUES (?,?,?,?)", (i, ts, act, rsn))
    conn.commit()
    rows = [
        (r[0], r[1], r[2]) for r in conn.execute(
            "SELECT timestamp, action_taken, reasoning "
            "FROM decisions ORDER BY id DESC")
    ]
    conn.close()

    rep = summarize(rows, recent_n=6, days=30)
    failures: list[str] = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r} want {want!r}")

    check("total_decisions", rep["total_decisions"], 10)
    # NO_DECISION rows: 4 on 05-17 + 1 on 05-16 + 3 on 05-14 = 8
    check("total_no_decision", rep["total_no_decision"], 8)
    check("cat.parse_failed", rep["by_category"].get(CAT_PARSE_FAILED), 2)
    check("cat.retry_failed", rep["by_category"].get(CAT_RETRY_FAILED), 1)
    check("cat.timeout_empty", rep["by_category"].get(CAT_TIMEOUT_EMPTY), 2)
    check("cat.legacy_no_json",
          rep["by_category"].get(CAT_LEGACY_NO_JSON), 2)
    check("cat.legacy_other", rep["by_category"].get(CAT_LEGACY_OTHER), 1)
    check("legacy_total", rep["legacy_no_decision"], 3)
    check("current_total", rep["current_code_no_decision"], 5)
    # recent_n=6 → first 6 rows by id DESC: timeout, parse_failed,
    # retry_failed, timeout, BUY(decided), parse_failed → 5 current-code
    # NO_DECISION / 6 recent → ratio .83 ≥ .5 and 6 ≥ 5 min-sample → ALERT
    check("recent_current_nd", rep["recent_current_code_no_decision"], 5)
    check("verdict", rep["verdict"], "ALERT")
    # by_day: 2026-05-17 → 4 NO_DECISION, 05-16 → 1, 05-14 → 3
    by_day = dict(rep["by_day"])
    check("by_day.2026-05-17", by_day.get("2026-05-17"), 4)
    check("by_day.2026-05-16", by_day.get("2026-05-16"), 1)
    check("by_day.2026-05-14", by_day.get("2026-05-14"), 3)

    # Pure-function spot checks.
    check("classify parse", classify_reason("parse_failed: x"),
          CAT_PARSE_FAILED)
    check("classify legacy", classify_reason("claude returned no parseable JSON"),
          CAT_LEGACY_NO_JSON)
    check("classify none", classify_reason(None), CAT_LEGACY_OTHER)
    check("is_nd true", is_no_decision("NO_DECISION"), True)
    check("is_nd false", is_no_decision("BUY NVDA → FILLED"), False)
    check("verdict low-sample", _verdict(3, 3), "OK")       # 3 < 5 guard
    check("verdict ok-ratio", _verdict(1, 10), "OK")        # .10 < .20
    check("verdict warn", _verdict(3, 10), "WARN")          # .30 in [.2,.5)
    check("verdict alert", _verdict(5, 6), "ALERT")         # .83 >= .50
    check("verdict warn-boundary", _verdict(2, 10), "WARN") # .20 inclusive

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("SELFTEST PASSED — taxonomy, verdict, and by-day aggregation OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB),
                    help=f"paper_trader.db path (default: {DEFAULT_DB})")
    ap.add_argument("--days", type=int, default=14,
                    help="rows in the by-day table (default 14)")
    ap.add_argument("--recent", type=int, default=25,
                    help="recent-decision window feeding the verdict (default 25)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of text")
    ap.add_argument("--selftest", action="store_true",
                    help="run built-in assertions and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[decision-health] DB not found: {db_path}", file=sys.stderr)
        return 1
    try:
        rows = load_rows(db_path)
    except sqlite3.Error as e:
        print(f"[decision-health] DB read failed: {e}", file=sys.stderr)
        return 1

    rep = summarize(rows, recent_n=args.recent, days=args.days)
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(_render(rep))

    return 2 if rep["verdict"] == "ALERT" else 0


if __name__ == "__main__":
    sys.exit(main())
