"""analytics/pushed_alert_gate_regret.py — retrospective gate-coverage audit
on the canonical Discord-push ledger.

Why this exists (news-analyst lens): every prior pass-style noise-gate fix
("if today's gates had been in place yesterday, that recap mill row wouldn't
have fired a 🚨 BREAKING") is described against single live failure cases in
the commit message. There is no aggregate primitive that, on demand, tells
the operator "of the N pushes the analyst received in the last 24h, how
many would TODAY's gate set retroactively have caught?". That is the
empirical measure of "how much noise reduction has the gate-engineering
work given us, retrospectively?".

Sibling surfaces and the gap each leaves:

  * ``analytics/alert_delivery_audit.py`` joins ``articles.db`` urgency=2
    rows against ``alert_recency.db`` to partition into ``delivered``
    (analyst was pushed) vs ``suppressed`` (gate marked the row alerted),
    then attributes ``suppressed`` to which fingerprint caught it. It
    grades **what queued**, not **what was pushed** — and only against the
    gate that fired AT QUEUE TIME. A push that happened BEFORE a gate
    existed is not in its scope.
  * ``analytics/recap_template_audit.py`` walks ``articles.db`` urgency>=1
    rows and counts how many match each recap fingerprint — same audit
    surface; same "what queued / scored" angle. Not push-truth.
  * ``analytics/recap_noise_by_source.py`` measures HIGH-RELEVANCE rows
    (per source) that match a recap fingerprint — "how leaky is this
    feeder?" — also at the article level, not at the push level.

This module is the missing axis: a **per-push** retrospective —
``alert_recency.db`` is the canonical record of pushes that actually fired
to Discord (``record_alerted`` is only called after ``discord_send``
succeeds), so iterating its ``alerted_sig`` rows in window and running
every saved title through the CURRENT
``watchers.alert_agent._looks_like_quote_widget`` and
``watchers.alert_agent._looks_like_recap_template`` produces exactly the
answer to "if today's gates had existed yesterday, how many of the N
pushes you got would have been suppressed?".

The audit is **strictly retrospective** — it never mutates anything,
never sends an alert, never re-classifies a row. It is a measurement
tool. Two analyst-actionable numbers come out:

  * ``would_suppress_rate`` — ``would_suppress / total``. The "noise
    pressure that newer gates would now absorb" index. Trends downward
    over time (good) as the regex set converges on the noise corpus;
    trends upward when a new SEO mill starts firing through.
  * ``would_suppress_by`` — per-fingerprint counts. Which gate (if it
    had existed earlier) would have caught the most? Same actionable
    structure as ``alert_delivery_audit.suppressed_by``.

Pure-builder design: ``build_regret_report(pushed_titles)`` is a side-
effect-free function that takes a list of ``{"title": str, "age_hours":
float}`` dicts and returns the JSON snapshot. Fully unit-testable
without SQLite. ``main()`` wires the live ``alert_recency.db`` to it.

Load-bearing invariants respected (mirrors ``alert_delivery_audit.py`` /
``recap_template_audit.py``):

  * **Backtest isolation:** ``alert_recency.db`` is Opus/Sonnet
    push-write only — the alert path's defense-in-depth ``_is_synthetic``
    re-filter (the lockstep mirror of ``_LIVE_ONLY_CLAUSE``) drops any
    ``backtest://`` URL before ``send_urgent_alert`` even runs, so the
    ledger by construction never carries a synthetic row. Defense-in-
    depth: the SQL pull is read-only; we never UPDATE/INSERT on this DB.
  * **score_source separation:** READ-only across the board; never
    touches ``ai_score`` / ``ml_score`` / ``score_source`` / ``urgency``.
  * **Read-only:** ``alert_recency.db`` opened mode=ro; cannot perturb
    the alert path or add to writer contention.

CLI: ``python3 -m analytics.pushed_alert_gate_regret [--hours 24]``
prints the JSON report; ``--pretty`` indents.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable


def build_regret_report(
    pushed_titles: Iterable[dict],
    window_h: int = 24,
) -> dict:
    """Pure-function builder. Returns a JSON-shape dict scoring each pushed
    title against the CURRENT gates.

    ``pushed_titles`` is an iterable of ``{"title": str, "age_hours":
    float}`` dicts — exactly the shape ``watchers.alert_recency.recent_alerts``
    already returns. ``age_hours`` is preserved so a UI can show the
    distribution but is not consumed by the gate evaluation.

    Returns::

        {
          "window_h": int,
          "total":          int,                 # pushes scored
          "would_suppress": int,                 # rows ANY current gate catches
          "would_keep":     int,                 # rows that survive
          "would_suppress_rate": float,          # would_suppress / total (0.0 on empty)
          "would_suppress_by": {
              "quote_widget":  int,
              # plus one key per _RECAP_TEMPLATE_PATTERNS fingerprint name
          },
          "top_offending_titles": [
              {"title": str, "fingerprint": str, "age_hours": float|None},
              ... (up to 20, newest-first when ties)
          ],
        }

    The fingerprint counts are mutually exclusive: each push is attributed
    to AT MOST one bucket (quote_widget wins over recap_template if both
    match, matching the alert path's actual execution order — quote-widget
    runs first in ``send_urgent_alert``).
    """
    # Lazy import: keeps the analytics module's import surface minimal and
    # avoids pulling watchers.alert_agent (and its ml.features transitive
    # graph) into a test that just exercises ``build_regret_report`` with
    # synthetic fingerprint lookups.
    from watchers.alert_agent import (
        _looks_like_quote_widget,
        _looks_like_recap_template,
        _RECAP_TEMPLATE_PATTERNS,
    )

    window_h = max(int(window_h), 1)

    # Pre-seed every fingerprint bucket so a quiet window still emits a
    # full-shape dict (same zero-data discipline as
    # ``urgency_label_split_trend``'s bucket-pre-seed).
    by_fingerprint: dict[str, int] = {"quote_widget": 0}
    for name, _ in _RECAP_TEMPLATE_PATTERNS:
        by_fingerprint[name] = 0

    total = 0
    would_suppress = 0
    offenders: list[dict] = []
    for row in pushed_titles:
        title = (row.get("title") or "").strip()
        if not title:
            continue
        total += 1
        # Mirror the alert-path execution order — quote-widget runs first
        # in ``send_urgent_alert``, then recap_template — so a row that
        # matches both is attributed to quote_widget exactly as the live
        # path would have suppressed it.
        art = {"title": title, "link": "", "url": ""}
        if _looks_like_quote_widget(art):
            by_fingerprint["quote_widget"] += 1
            would_suppress += 1
            offenders.append({
                "title": title[:160],
                "fingerprint": "quote_widget",
                "age_hours": row.get("age_hours"),
            })
            continue
        hit, name = _looks_like_recap_template(art)
        if hit:
            # Defensive: an unknown fingerprint name (a future RECAP
            # pattern added to the tuple without seeding by_fingerprint)
            # must NOT silently inflate "total" with no bucket — re-seed
            # on first sight so the snapshot stays self-consistent.
            if name not in by_fingerprint:
                by_fingerprint[name] = 0
            by_fingerprint[name] += 1
            would_suppress += 1
            offenders.append({
                "title": title[:160],
                "fingerprint": name,
                "age_hours": row.get("age_hours"),
            })

    would_keep = total - would_suppress
    rate = round(would_suppress / total, 4) if total else 0.0

    # Top offenders: bounded list of the actually-suppressed pushes for
    # operator review. Sort by age (newest first) — the operator wants
    # to see "what did I get pushed in the last hour that today's gates
    # would have absorbed?" front and centre. None ages sort last.
    offenders.sort(
        key=lambda r: r["age_hours"] if r["age_hours"] is not None else float("inf")
    )

    return {
        "window_h": window_h,
        "total": total,
        "would_suppress": would_suppress,
        "would_keep": would_keep,
        "would_suppress_rate": rate,
        "would_suppress_by": by_fingerprint,
        "top_offending_titles": offenders[:20],
    }


def _load_pushed_titles(db_path: Path, window_h: int) -> list[dict]:
    """Read the alert_recency.db ledger for the requested window. Opens the
    DB read-only via the ``file:...?mode=ro`` URI so we cannot perturb the
    alert path's writer. Returns the same shape ``recent_alerts`` returns,
    minus the ``sig`` column (we don't need it for the regret audit)."""
    if not db_path.exists():
        return []
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        rows = conn.execute(
            "SELECT title, last_ts FROM alerted_sig "
            "WHERE last_ts >= datetime('now', ?) "
            "ORDER BY last_ts DESC",
            (f"-{int(window_h)} hours",),
        ).fetchall()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Hours since the push — same shape as watchers.alert_recency.recent_alerts.
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    out: list[dict] = []
    for title, last_ts in rows:
        if not title:
            continue
        try:
            dt = datetime.fromisoformat(str(last_ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_h = round((now - dt).total_seconds() / 3600.0, 2)
            if age_h < 0:
                age_h = 0.0
        except (ValueError, TypeError):
            age_h = None
        out.append({"title": title, "age_hours": age_h})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrospective gate-coverage audit on the canonical "
                    "Discord-push ledger (alert_recency.db). Scores each "
                    "actually-pushed alert in the window against the CURRENT "
                    "quote-widget + recap-template gates to answer 'how many "
                    "would today's gates have suppressed?'."
    )
    parser.add_argument("--hours", type=int, default=24,
                        help="Lookback window in hours (default 24)")
    parser.add_argument("--pretty", action="store_true",
                        help="Indent JSON output")
    parser.add_argument(
        "--db",
        default=str(Path(__file__).resolve().parent.parent / "data"
                    / "alert_recency.db"),
        help="Path to alert_recency.db (default: data/alert_recency.db)",
    )
    args = parser.parse_args(argv)

    pushed = _load_pushed_titles(Path(args.db), args.hours)
    report = build_regret_report(pushed, window_h=args.hours)
    print(json.dumps(report, indent=2 if args.pretty else None,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
