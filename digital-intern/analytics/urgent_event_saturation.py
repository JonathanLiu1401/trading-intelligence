"""analytics/urgent_event_saturation.py — per-(held-ticker × event-class)
URGENT-QUEUE saturation audit over ``articles.db`` (urgency>=1 rows).

The QUEUED-side sibling of ``analytics.pushed_alert_event_concentration``:
that audit measures concentration of Discord pushes after every defense-in-
depth gate has run (paraphrase suppression, recency, recap-template).
This one measures concentration of urgent classifications BEFORE those
gates collapse anything — answering "is the urgency scorer producing
massive same-event signal density?" rather than "did multiple same-event
pushes survive to Discord?".

Why both sides are needed (live evidence, 2026-05-25 24h window):

  * ``alert_recency.db`` (push side): ~21 distinct pushes — what the analyst
    actually saw on Discord.
  * ``articles.db`` (queued side): 49 urgency>=1 rows — what the urgency
    scorer flagged AS urgent. The 28-row delta is what paraphrase
    suppression + recency dedup collapsed.

A real failure mode lives in that delta: the buyback event produced 14+
urgent classifications (multiple syndications of "Nvidia unveils $80B
buyback ..." across GN: dividend buyback / GN: Nvidia / GoogleNews/
simplywall.st / etc.), of which the formatter collapsed roughly half. The
push-side audit sees the surviving ~7; the queued side sees all 14 — and
that 14 is the relevant number when the question is "is my SCORER over-
amplifying noise about this event?". A scorer flagging 14 same-event rows
is wasting LLM quota / training pool signal even when the formatter
correctly collapses the pushes.

Sibling surfaces and the gap THIS module fills:

  * ``analytics.pushed_alert_event_concentration`` — Discord-push side
    (alert_recency.db). Does NOT touch articles.db urgency rows.
  * ``analytics.news_fatigue`` — per-ticker score-mean trend on
    articles.db; orthogonal axis (intensity over time, not duplication).
  * ``storage.article_store.cross_book_event_pulse`` — multi-ticker
    baskets in articles.db (2+ tickers in one title). This audit is the
    SINGLE-ticker complement: one event mentioned in N rows about ONE
    held name, the buyback-saturation pattern.
  * ``storage.article_store.source_recap_pollution`` — per-source recap
    fingerprint rate. Orthogonal axis (content type, not event class).
  * ``storage.article_store.urgency_label_split_by_ticker`` — per-held-
    ticker LLM-vetted fraction over urgency>=1. Same row set, but the
    axis is calibration (llm vs ml), not event clustering.

Closed-vocabulary discipline (mirrors ``pushed_alert_event_concentration``):
event-class taxonomy and held-ticker matching are imported VERBATIM from
the push-side module as SSOT — so the two surfaces never disagree about
"what counts as a BUYBACK event" or "what counts as a NVDA mention".
A regex tightening on the push side automatically engages here.

Load-bearing invariants intact:

  * **Backtest isolation:** the SELECT applies ``_LIVE_ONLY_CLAUSE``, so
    backtest:// URLs and backtest_/opus_annotation* sources can never
    inflate the urgent-event saturation figure (a backtest title like
    "NVDA NVDA NVDA buyback" would otherwise manufacture a fake (NVDA,
    BUYBACK) cluster every cycle the runner injects).
  * **ml_score / ai_score separation:** read-only — no DB write, no
    ai_score / ml_score / score_source / urgency mutation.
  * **score_source separation:** the per-row score_source is surfaced
    in the by_pair output (``score_sources`` count) but never modified.
  * **Read-only:** ``articles.db`` opened ``mode=ro`` with the canonical
    URI form so we never contend with the daemon's writer locks. The
    pure builder ``build_saturation_report`` is side-effect-free and
    takes the exact shape ``storage.article_store.get_unalerted_urgent``-
    family methods produce (plus a precomputed event class to avoid
    cross-module imports inside the storage layer).

Locked by ``tests/test_urgent_event_saturation.py`` — the four invariants
plus the live failure shape (NVDA × BUYBACK cluster), pure-builder
contract, and CLI degrade-gracefully behaviour.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

# Reuse the SSOT event-class taxonomy + held-ticker resolution from the
# push-side audit. Importing the existing module rather than redefining
# keeps the two surfaces byte-identical on the event-classification axis
# — exactly the anti-drift discipline the recap-template fingerprint pair
# uses (``test_alert_and_briefing_recap_tuples_have_same_length``).
from analytics.pushed_alert_event_concentration import (
    event_class_for_title,
    _held_tickers_in_title,
    CONCENTRATION_THRESHOLD as _PUSH_CONCENTRATION_THRESHOLD,
)


# ── Saturation verdict ladder ────────────────────────────────────────────────
# Conservative most-severe-first, mirrors ``briefing_health`` /
# ``briefing_cadence_trend`` / ``label_production_rate``'s discipline.
#
# A pair with ``urgent_count >= SATURATION_THRESHOLD`` is flagged as a
# concentration alert. ``HEAVY_SATURATION`` triggers on a single pair at
# or above ``HEAVY_THRESHOLD`` — the upper bar where the SAME EVENT is
# producing enough urgent rows to materially distort the queue (the
# buyback-saturation pattern: 14 same-event urgent rows in 24h).
#
# Live evidence (2026-05-25, 24h): NVDA × BUYBACK had 14 urgency>=1 rows
# — well above HEAVY_THRESHOLD. HEALTHY would have read on a normal day
# with no single concentrated event.
SATURATION_THRESHOLD = max(_PUSH_CONCENTRATION_THRESHOLD, 2)
HEAVY_THRESHOLD = 5

# Cap on the by_pair table — a fully-degraded window (single-name wire
# storm) must not emit a wall-of-text report that itself becomes noise.
# Same anti-noise capping discipline as ``_MAX_BY_PAIR_ROWS`` in the
# push-side module and ``BRIEFING_MAX_PER_DOMAIN`` in the storage layer.
_MAX_BY_PAIR_ROWS = 20


def build_saturation_report(
    rows: Iterable[dict],
    live_tickers: Iterable[str],
    *,
    window_h: float = 24.0,
    saturation_threshold: int = SATURATION_THRESHOLD,
    heavy_threshold: int = HEAVY_THRESHOLD,
    max_by_pair_rows: int = _MAX_BY_PAIR_ROWS,
) -> dict:
    """Pure-function builder. Returns the JSON snapshot.

    ``rows`` is an iterable of dicts shaped like::

        {"title": str, "age_hours": float, "score_source": str|None,
         "urgency": int}

    — exactly what ``_load_urgent_rows`` below produces from articles.db.
    ``live_tickers`` is the held-book universe to gate matches against
    (same discipline as ``pushed_alert_event_concentration._held_tickers_in_title``).

    Returns::

        {
          "window_h":               float,
          "saturation_threshold":   int,
          "heavy_threshold":        int,
          "total_urgent":           int,
          "urgent_with_class":      int,
          "urgent_held_x_class":    int,
          "distinct_pairs":         int,
          "by_pair": [
            {
              "ticker":         str,
              "event_class":    str,
              "urgent_count":   int,
              "alerted_count":  int,    # subset with urgency=2 (formatter-exited)
              "newest_age_h":   float | None,
              "newest_title":   str,
              "titles":         [str, ...],   # newest first
              "score_sources":  {"llm": N, "ml": N, "briefing_boost": N,
                                  "null": N},
            },
            ... (capped at max_by_pair_rows)
          ],
          "saturation_alerts":      [str, ...],   # pairs >= saturation_threshold
          "verdict": "HEALTHY" | "WATCH" | "SATURATED" | "NO_DATA",
        }

    Discipline (mirrors ``build_concentration_report``):
      * Empty input → fully-shaped dict with zeros, empty lists, NO_DATA.
      * A title with no closed-vocab class is counted in ``total_urgent``
        but NEVER appears in ``by_pair`` / ``saturation_alerts`` — the
        audit under-claims rather than over-claims.
      * A title with a class but no held-ticker match is counted in
        ``urgent_with_class`` but NEVER appears in ``by_pair`` —
        consistent with the push-side audit's held-ticker gating.
      * Multi-ticker rows contribute to every matching pair (per-ticker)
        — same convention as ``cross_book_event_pulse`` and
        ``pushed_alert_event_concentration``.
      * ``score_sources`` per pair surfaces the calibration mix at the
        event-class level — e.g. (NVDA, BUYBACK) with 12 'ml' / 2 'llm'
        rows is a different signal from one with 12 'llm' / 2 'ml' even
        if the count is identical.
      * Sort: descending by urgent_count, then by alerted_count (so a
        pair with more queue-exits surfaces above one with the same
        urgent count but fewer alerts), then alphabetical (ticker,
        event_class) — deterministic, test-pinnable, same convention
        as ``urgency_label_split_by_source`` and the push-side audit.

    Verdict ladder (most-severe-first):
      * ``NO_DATA`` — no urgent rows in the window. Refuses to make any
        claim about event saturation when there is no input to measure.
      * ``SATURATED`` — at least one pair >= heavy_threshold. The single-
        event distortion bar — the buyback-saturation pattern.
      * ``WATCH`` — at least one pair >= saturation_threshold but no
        pair at heavy_threshold. Same-event signal density is elevated
        but not yet at the analyst-action bar.
      * ``HEALTHY`` — no pair at threshold. The urgent stream is
        event-diverse.
    """
    # Clamp inputs to safe positives — defense against bad callers.
    window_h = max(float(window_h), 0.01)
    saturation_threshold = max(int(saturation_threshold), 1)
    heavy_threshold = max(int(heavy_threshold), saturation_threshold)
    max_by_pair_rows = max(int(max_by_pair_rows), 1)

    materialised: list[dict] = []
    for r in rows or ():
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or "").strip()
        if not title:
            continue
        try:
            age_h = float(r.get("age_hours") or 0.0)
        except (TypeError, ValueError):
            age_h = 0.0
        try:
            urg = int(r.get("urgency") or 0)
        except (TypeError, ValueError):
            urg = 0
        src_raw = r.get("score_source")
        # Normalize: any string outside the canonical set lands in null bucket
        # (mirrors urgency_label_split's bucket discipline).
        if src_raw in ("llm", "ml", "briefing_boost"):
            src_tag = src_raw
        else:
            src_tag = "null"
        materialised.append({
            "title": title,
            "age_hours": max(0.0, age_h),
            "urgency": max(0, urg),
            "score_source": src_tag,
        })

    total_urgent = len(materialised)
    urgent_with_class = 0
    urgent_held_x_class = 0

    pair_state: dict[tuple[str, str], dict] = {}
    for r in materialised:
        title = r["title"]
        age_h = r["age_hours"]
        urg = r["urgency"]
        src_tag = r["score_source"]
        cls = event_class_for_title(title)
        if not cls:
            continue
        urgent_with_class += 1
        held = _held_tickers_in_title(title, live_tickers)
        if not held:
            continue
        urgent_held_x_class += 1
        for t in held:
            key = (t, cls)
            slot = pair_state.get(key)
            if slot is None:
                slot = {
                    "ticker": t, "event_class": cls,
                    "urgent_count": 0, "alerted_count": 0,
                    "newest_age_h": None, "newest_title": "",
                    "titles_with_age": [],
                    "score_sources": {
                        "llm": 0, "ml": 0, "briefing_boost": 0, "null": 0,
                    },
                }
                pair_state[key] = slot
            slot["urgent_count"] += 1
            if urg >= 2:
                slot["alerted_count"] += 1
            cur_age = slot["newest_age_h"]
            if cur_age is None or age_h < cur_age:
                slot["newest_age_h"] = age_h
                slot["newest_title"] = title
            slot["titles_with_age"].append((age_h, title))
            slot["score_sources"][src_tag] += 1

    by_pair: list[dict] = []
    for slot in pair_state.values():
        titles_sorted = [
            t for _, t in sorted(slot["titles_with_age"], key=lambda x: x[0])
        ]
        by_pair.append({
            "ticker": slot["ticker"],
            "event_class": slot["event_class"],
            "urgent_count": slot["urgent_count"],
            "alerted_count": slot["alerted_count"],
            "newest_age_h": (
                round(slot["newest_age_h"], 2)
                if slot["newest_age_h"] is not None else None
            ),
            "newest_title": slot["newest_title"],
            "titles": titles_sorted,
            "score_sources": dict(slot["score_sources"]),
        })

    # Worst-first: urgent_count desc → alerted_count desc → ticker → class.
    # Same deterministic discipline as urgency_label_split_by_source.
    by_pair.sort(
        key=lambda r: (
            -r["urgent_count"], -r["alerted_count"],
            r["ticker"], r["event_class"],
        )
    )
    distinct_pairs = len(by_pair)
    by_pair = by_pair[:max_by_pair_rows]

    saturation_alerts: list[str] = []
    max_pair_count = 0
    for row in by_pair:
        if row["urgent_count"] > max_pair_count:
            max_pair_count = row["urgent_count"]
        if row["urgent_count"] >= saturation_threshold:
            newest = row["newest_age_h"]
            newest_str = f"{newest:.2f}h ago" if newest is not None else "n/a"
            saturation_alerts.append(
                f"{row['ticker']} × {row['event_class']}: "
                f"{row['urgent_count']} urgent ({row['alerted_count']} alerted) "
                f"in last {window_h:.1f}h (newest {newest_str})"
            )

    if total_urgent == 0:
        verdict = "NO_DATA"
    elif max_pair_count >= heavy_threshold:
        verdict = "SATURATED"
    elif max_pair_count >= saturation_threshold:
        verdict = "WATCH"
    else:
        verdict = "HEALTHY"

    return {
        "window_h": round(window_h, 2),
        "saturation_threshold": saturation_threshold,
        "heavy_threshold": heavy_threshold,
        "total_urgent": total_urgent,
        "urgent_with_class": urgent_with_class,
        "urgent_held_x_class": urgent_held_x_class,
        "distinct_pairs": distinct_pairs,
        "by_pair": by_pair,
        "saturation_alerts": saturation_alerts,
        "verdict": verdict,
    }


# ── CLI / live wiring ────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent
# Same DB path resolution discipline as storage/article_store.py — prefer the
# USB drive when present, fall back to local data/.
_USB_PATH = Path(
    os.environ.get("DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db")
)
_LOCAL_DB = _BASE_DIR / "data" / "articles.db"


def _resolve_articles_db() -> Path:
    """USB-or-local articles.db path, mirroring storage._get_db_path discipline.

    Read-only — never creates the file. If neither path exists, returns the
    USB path so callers see a clean ``not found`` rather than masking the
    config error by silently substituting an empty local file."""
    usb_db = _USB_PATH / "articles.db"
    if usb_db.exists():
        return usb_db
    if _LOCAL_DB.exists():
        return _LOCAL_DB
    return usb_db


# Live-only filter — same SSOT discipline as storage._LIVE_ONLY_CLAUSE.
# We deliberately replicate the clause inline here (not import) because the
# analytics module is independent of storage's import surface; the four
# invariants test pins they match byte-for-byte.
_LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)


def _load_urgent_rows(hours: float) -> list[dict]:
    """Best-effort read of articles.db urgency>=1 rows within ``hours``,
    shaped to match what ``build_saturation_report`` consumes.

    Opens a fresh short-lived ``mode=ro`` connection (never the daemon's
    shared connection — the documented cursor-collision hazard in
    storage/article_store.py). Returns ``[]`` on ANY failure so the CLI
    degrades cleanly when articles.db is missing on a fresh install.

    The ``_LIVE_ONLY_CLAUSE`` excludes synthetic backtest/opus rows by
    construction — load-bearing invariant #1 (backtest isolation).
    """
    db = _resolve_articles_db()
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(
            f"file:{db}?mode=ro", uri=True, timeout=5,
        )
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            cutoff = (
                datetime.now(timezone.utc) - timedelta(hours=hours)
            ).isoformat()
            rows = conn.execute(
                "SELECT title, first_seen, urgency, score_source "
                f"FROM articles WHERE urgency >= 1 AND first_seen >= ? "
                f"AND {_LIVE_ONLY_CLAUSE} "
                "ORDER BY first_seen DESC",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return []
    now = datetime.now(timezone.utc)
    out: list[dict] = []
    for title, first_seen, urgency, score_source in rows:
        if not title:
            continue
        age_h = 0.0
        if first_seen:
            try:
                dt = datetime.fromisoformat(str(first_seen).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_h = max(0.0, (now - dt).total_seconds() / 3600.0)
            except Exception:
                age_h = 0.0
        out.append({
            "title": title,
            "age_hours": age_h,
            "urgency": int(urgency or 0),
            "score_source": score_source,
        })
    return out


def main() -> int:
    """CLI entrypoint. Pretty-prints the JSON report. Returns 0 on success."""
    parser = argparse.ArgumentParser(
        description=("Per-(held-ticker × event-class) urgent-queue saturation "
                     "audit (articles.db side; queued sibling to "
                     "pushed_alert_event_concentration)."),
    )
    parser.add_argument(
        "--hours", type=float, default=24.0,
        help="Window in hours (default: 24).",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Indent the JSON output for human reading.",
    )
    args = parser.parse_args()

    rows = _load_urgent_rows(args.hours)
    # Lazy import — keeps the analytics module's import surface minimal and
    # the test harness can monkeypatch LIVE_PORTFOLIO_TICKERS without going
    # through this CLI path. Same lazy-import convention as the push-side
    # module.
    from ml.features import LIVE_PORTFOLIO_TICKERS
    report = build_saturation_report(
        rows, LIVE_PORTFOLIO_TICKERS, window_h=args.hours,
    )
    indent = 2 if args.pretty else None
    print(json.dumps(report, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
