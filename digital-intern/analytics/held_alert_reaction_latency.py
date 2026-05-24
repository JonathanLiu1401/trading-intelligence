"""analytics/held_alert_reaction_latency.py — per-held-ticker push reaction latency.

Why this exists (news-analyst lens): four sibling metrics already live in this
repo and each measures a different facet of "freshness" — none answers the
operator's most actionable per-position question:

  * ``storage.ingestion_latency`` — per-collector publish→first_seen lag
    ("how soon after publication did THIS feed see the article?"). Catches a
    slow collector.
  * ``analytics.publish_lag_audit`` — same axis, snapshot-and-rank form
    ("freshest vs stalest collectors right now"). Catches operationally
    drifting feeds.
  * ``analytics.alert_freshness`` — for urgency>=1 rows, the ``published``
    age at alert time, aggregated and split by ``score_source``. Catches an
    alert pipeline pushing materially-old articles.
  * ``watchers.alert_recency`` — cross-cycle dedup of REAL Discord pushes
    by canonical signature, with per-ticker breakdown.

The gap: **for each held ticker, how long had we already had an article in
``articles.db`` mentioning that ticker before the Discord push fired?** That
is the analyst's per-position reaction-latency view — a low median means
the alert pipeline reacts quickly when news about an open position appears;
a high median means we were sitting on the news for tens of minutes before
the analyst was pushed. Neither the per-collector view (``publish_lag``,
``ingestion_latency``) nor the alert-side staleness view (``alert_freshness``)
exposes this. It is exactly the failure mode "we had the NVDA buyback
headline at minute 0 but didn't push until minute 45" — invisible to every
existing freshness monitor.

This module is that slice. For each held ticker T, for every Discord push in
the window whose canonical-signature title mentions T, it finds the EARLIEST
article in ``articles.db`` (live-only) within ``mention_lookback_hours`` that
also mentions T AND was first_seen at or before the push timestamp.
``reaction_minutes = (push_ts - earliest_first_seen).total_seconds() / 60``.
Per-ticker counts, median, p90, max are aggregated.

Edge / honesty cases the pure function handles explicitly:
  * Empty inputs degrade to ``{"by_ticker": [], ...}``.
  * A push with no prior mention (article first_seen IS the push, or the
    push was on a brand-new signature carried only by the alerted row) gives
    ``reaction_minutes = 0`` — honest: we reacted as soon as we had it.
  * Mentions outside the per-push ``mention_lookback_hours`` window are
    excluded so a 23h-old NVDA recap doesn't pollute a fresh-event push.
  * Articles first_seen AFTER the push are excluded — they could not have
    informed it.
  * Multiple pushes for the same ticker fold per-push into one latency
    sample each; the per-ticker stats are over the distinct-push samples.

Load-bearing invariants respected (mirrors ``pushed_ticker_label_split.py``
/ ``alert_delivery_audit.py``):

  * **Backtest isolation:** the SQL pull carries the canonical
    ``_LIVE_ONLY_CLAUSE`` verbatim. Synthetic backtest/opus rows cannot
    contribute to either the mention pool or the push pool.
  * **score_source separation:** ``ai_score`` / ``ml_score`` / ``score_source``
    are READ only — never written. The audit derives no labels and no
    alert state.
  * **Read-only:** both DBs opened ``mode=ro`` with a short busy timeout.
    Cannot perturb the alert path or add to writer contention.
  * **urgency state machine:** never touched. ``urgency`` is read implicitly
    (mentions are pulled regardless of urgency — a row that was never
    urgent can still be the first-mention anchor) but never written.

CLI: ``python3 -m analytics.held_alert_reaction_latency [--hours 6]``
prints a JSON report; ``--pretty`` indents.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable

from watchers.alert_recency import ALERT_RECENCY_TTL_HOURS

# Canonical backtest-isolation clause — duplicated verbatim from
# storage/article_store.py::_LIVE_ONLY_CLAUSE (the documented anti-drift
# discipline; the tests pin a drift check identical to the sibling
# ``pushed_ticker_label_split`` / ``alert_delivery_audit`` analytics).
LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

_USB_PATH = Path(os.environ.get(
    "DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"))
_LOCAL_PATH = Path(__file__).resolve().parent.parent / "data"

# Default window matches the recency-store TTL exactly. A wider window would
# compare push timestamps against signatures already pruned out of
# ``alerted_sig``, silently under-reporting recent pushes.
DEFAULT_WINDOW_HOURS = ALERT_RECENCY_TTL_HOURS

# For each push, how far back to look for prior mentions of the ticker. A
# 6h ceiling matches the briefing/recency cadence: a mention >6h ago is
# very likely a different event (a recap, an unrelated wire) and would
# inflate the latency metric without reflecting actual pipeline slowness.
DEFAULT_MENTION_LOOKBACK_HOURS = 6.0


def _parse_ts(value: str | None) -> datetime | None:
    """Parse an ISO or RFC822 timestamp into a UTC datetime.

    Mirrors ``ml.features._parse_published`` and ``watchers.alert_agent.
    _article_age_hours`` so this module agrees with the alert path on what
    counts as "this many minutes ago". A failure returns ``None`` — the
    caller drops the row rather than inventing a fresh timestamp.
    """
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    dt: datetime | None = None
    try:
        dt = parsedate_to_datetime(raw)
    except Exception:
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _clean_tickers(raw_tickers: Iterable[str]) -> list[str]:
    """Uppercase, dedupe, skip <2-char entries — same hygiene as
    ``pushed_ticker_label_split._clean_tickers`` so the held-book surface
    is byte-for-byte the same."""
    out: list[str] = []
    seen: set[str] = set()
    for t in raw_tickers or []:
        if not t or not isinstance(t, str):
            continue
        u = t.strip().upper()
        if len(u) < 2 or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _percentile(samples: list[float], q: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list.

    ``q`` is in 0..1. Returns 0.0 on an empty list — callers handle the
    "no samples" case explicitly via the count.
    """
    if not samples:
        return 0.0
    if len(samples) == 1:
        return float(samples[0])
    pos = q * (len(samples) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(samples) - 1)
    frac = pos - lo
    return float(samples[lo] * (1.0 - frac) + samples[hi] * frac)


def compute_held_alert_reaction_latency(
    articles: Iterable[dict],
    alerts: Iterable[dict],
    tickers: Iterable[str],
    mention_lookback_hours: float = DEFAULT_MENTION_LOOKBACK_HOURS,
) -> dict:
    """Pure function — no DB / IO. Per-held-ticker reaction-latency over the
    distinct Discord pushes in the window.

    ``articles``: iterable of ``{"title", "summary", "first_seen"}`` dicts.
      ``first_seen`` is the canonical UTC ISO timestamp the daemon writes;
      RFC822 and ISO are both accepted (see ``_parse_ts``). Rows whose
      timestamp will not parse are dropped from the mention pool — they
      cannot be located on the timeline.

    ``alerts``: iterable of ``{"title", "last_ts"}`` dicts (the
      ``alert_recency.db.alerted_sig`` shape — ``last_ts`` is when the most
      recent push for that signature fired). Rows without a parseable
      ``last_ts`` are dropped: they cannot anchor a latency sample.

    ``tickers``: iterable of held-ticker strings (config/portfolio.json
      union with the hardcoded fallback at the CLI shell). The match is a
      whole-word case-insensitive regex on the title + summary — same
      surface as ``ml.features._LIVE_RE`` and ``alert_agent._book_tickers``
      so the analytics view and the alert path can never disagree about
      which articles touch the held book.

    ``mention_lookback_hours``: per-push, the maximum age of a candidate
      prior-mention article. Capped to 6h by default so an unrelated 23h
      recap doesn't inflate the latency on a fresh-event push.

    Returns:

      .. code-block:: python

        {
            "total_pushes": int,         # distinct pushes in the window
            "tickers_in_book": int,
            "mention_lookback_hours": float,
            "by_ticker": [               # held names with >= 1 push,
                {                        #   sorted slowest-reaction-first
                    "ticker": str,
                    "pushes": int,        # distinct pushes mentioning this ticker
                    "samples": int,       # subset that had a prior mention
                    "median_minutes": float,
                    "p90_minutes": float,
                    "max_minutes": float,
                    "instant_reactions": int,   # pushes with no prior mention
                },
                ...
            ],
            "silent_tickers": [str, ...],  # held names not mentioned by any push
        }
    """
    clean = _clean_tickers(tickers)

    # Normalise the alerts: parse the timestamp once, drop unparseable rows.
    parsed_alerts: list[tuple[str, datetime]] = []
    for a in alerts or []:
        ts = _parse_ts(a.get("last_ts"))
        if ts is None:
            continue
        title = (a.get("title") or "").strip()
        parsed_alerts.append((title, ts))
    total_pushes = len(parsed_alerts)

    if not clean:
        return {
            "total_pushes": total_pushes,
            "tickers_in_book": 0,
            "mention_lookback_hours": round(float(mention_lookback_hours), 3),
            "by_ticker": [],
            "silent_tickers": [],
        }

    # Pre-parse articles: drop unparseable timestamps once, store
    # (first_seen_utc, title+summary) tuples in a single pre-sorted list so
    # the per-push lookup is O(N) over articles (linear scan, no per-call
    # parse). A held book of 12 tickers × ~50 pushes is small enough that
    # the explicit linear scan beats building per-ticker secondary indexes.
    parsed_articles: list[tuple[datetime, str]] = []
    for art in articles or []:
        ts = _parse_ts(art.get("first_seen"))
        if ts is None:
            continue
        blob = f"{art.get('title') or ''} {art.get('summary') or ''}".strip()
        if not blob:
            continue
        parsed_articles.append((ts, blob))

    # Single compiled alternation across all uppercase tickers — one regex
    # walk per (article, push) probe. Word-boundary anchor matches
    # ``_LIVE_RE``'s convention so "AMD" never matches inside "DAMD".
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in clean) + r")\b",
        re.IGNORECASE,
    )

    # latencies[ticker] = list of minutes (one per push that mentioned T)
    latencies: dict[str, list[float]] = {t: [] for t in clean}
    # pushes_per_ticker counts the distinct pushes mentioning T regardless
    # of whether a prior mention exists — so the analyst can see push
    # frequency separately from reaction speed.
    pushes_per_ticker: dict[str, int] = {t: 0 for t in clean}
    # instant_per_ticker tallies pushes that had no prior in-window mention
    # (latency=0, the "alerted on the first mention" case). Surfaced so a
    # ticker whose median appears low can be sanity-checked against how
    # many of those zero-samples came from instant-reaction vs real fast
    # response after a prior mention.
    instant_per_ticker: dict[str, int] = {t: 0 for t in clean}

    # Pre-compute the lookback cutoff in seconds for cheap comparisons.
    lookback_secs = float(mention_lookback_hours) * 3600.0

    for push_title, push_ts in parsed_alerts:
        # Which held tickers does THIS push mention?
        push_hits = {m.upper() for m in pattern.findall(push_title)}
        if not push_hits:
            continue
        # For each ticker hit, find the earliest in-window prior mention.
        for t in push_hits:
            if t not in pushes_per_ticker:
                continue  # defensive — pattern shouldn't yield untracked
            pushes_per_ticker[t] += 1
            t_re = re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
            earliest_ts: datetime | None = None
            for art_ts, blob in parsed_articles:
                if art_ts > push_ts:
                    continue  # could not have informed the push
                delta = (push_ts - art_ts).total_seconds()
                if delta > lookback_secs:
                    continue  # too old to be the same event
                if not t_re.search(blob):
                    continue
                if earliest_ts is None or art_ts < earliest_ts:
                    earliest_ts = art_ts
            if earliest_ts is None:
                # No prior mention found: the alerted row IS the first
                # mention from the operator's POV. Latency = 0 + tallied
                # separately so the analyst can disambiguate it from a
                # genuine fast reaction.
                instant_per_ticker[t] += 1
                latencies[t].append(0.0)
            else:
                minutes = (push_ts - earliest_ts).total_seconds() / 60.0
                latencies[t].append(round(max(0.0, minutes), 2))

    materialised: list[dict] = []
    silent: list[str] = []
    for t in clean:
        if pushes_per_ticker[t] == 0:
            silent.append(t)
            continue
        samples = sorted(latencies[t])
        materialised.append({
            "ticker": t,
            "pushes": pushes_per_ticker[t],
            "samples": len(samples),
            "median_minutes": round(_percentile(samples, 0.5), 2),
            "p90_minutes": round(_percentile(samples, 0.9), 2),
            "max_minutes": round(max(samples), 2) if samples else 0.0,
            "instant_reactions": instant_per_ticker[t],
        })
    # Slowest-reaction-first (largest median), alphabetical tiebreak —
    # mirrors ``pushed_ticker_label_split``'s most-ml-first sort discipline.
    materialised.sort(key=lambda r: (-r["median_minutes"], r["ticker"]))

    return {
        "total_pushes": total_pushes,
        "tickers_in_book": len(clean),
        "mention_lookback_hours": round(float(mention_lookback_hours), 3),
        "by_ticker": materialised,
        "silent_tickers": silent,
    }


def resolve_db_paths() -> tuple[Path, Path]:
    """Resolve live ``articles.db`` (USB-preferred) and ``alert_recency.db``
    (always local — see ``watchers.alert_recency.DB_PATH``). No side effects;
    mirrors ``pushed_ticker_label_split.resolve_db_paths``."""
    usb_db = _USB_PATH / "articles.db"
    if _USB_PATH.exists() and (usb_db.exists() or _USB_PATH.is_mount()):
        articles_db = usb_db
    else:
        articles_db = _LOCAL_PATH / "articles.db"
    recency_db = _LOCAL_PATH / "alert_recency.db"
    return articles_db, recency_db


def _open_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _fetch_articles(conn: sqlite3.Connection, hours: float,
                    mention_lookback_hours: float) -> list[dict]:
    """Pull live-only articles in the (window + lookback) range. The summary
    column is decompressed so the pure helper above operates on the same
    ``title+summary`` surface as ``alert_agent._book_tickers``.

    Window = (push_window + per_push_lookback) so the earliest prior mention
    of a push at the very start of the window has a full ``lookback`` to
    look back through.
    """
    from storage.article_store import decompress
    pull_hours = float(hours) + float(mention_lookback_hours)
    since = (datetime.now(timezone.utc) - timedelta(hours=pull_hours)).isoformat()
    rows = conn.execute(
        "SELECT title, full_text, first_seen FROM articles "
        f"WHERE first_seen >= ? AND {LIVE_ONLY_CLAUSE}",
        (since,),
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        out.append({
            "title": r[0] or "",
            "summary": decompress(r[1]) if r[1] else "",
            "first_seen": r[2] or "",
        })
    return out


def _fetch_alerts(conn: sqlite3.Connection, hours: float) -> list[dict]:
    """Pull the alerted-sig rows in the push window — each becomes one push
    record. Mirrors ``alert_recency.recent_alerts`` but re-implements the
    SELECT so a future API change can't break this audit."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    rows = conn.execute(
        "SELECT title, last_ts FROM alerted_sig WHERE last_ts >= ?", (cutoff,),
    ).fetchall()
    return [{"title": r[0] or "", "last_ts": r[1] or ""} for r in rows]


def run(tickers: Iterable[str] | None = None,
        hours: float = DEFAULT_WINDOW_HOURS,
        mention_lookback_hours: float = DEFAULT_MENTION_LOOKBACK_HOURS) -> dict:
    """DB shell: open both stores read-only, pull data, compose the audit.

    ``tickers`` defaults to ``ml.features.LIVE_PORTFOLIO_TICKERS`` (the live
    held-book SSOT — config/portfolio.json union'd with the hardcoded
    fallback) so a CLI invocation works out of the box on a live host.

    ``hours`` is clamped to the recency TTL (same rationale as
    ``pushed_ticker_label_split.run`` / ``alert_delivery_audit.run_audit``).
    """
    if tickers is None:
        from ml.features import LIVE_PORTFOLIO_TICKERS
        tickers = sorted(LIVE_PORTFOLIO_TICKERS)
    if hours > ALERT_RECENCY_TTL_HOURS + 1e-6:
        hours = ALERT_RECENCY_TTL_HOURS

    articles_db, recency_db = resolve_db_paths()
    art_conn = _open_ro(articles_db)
    try:
        arts = _fetch_articles(art_conn, hours, mention_lookback_hours)
    finally:
        art_conn.close()

    try:
        rec_conn = _open_ro(recency_db)
    except sqlite3.OperationalError:
        # No recency DB yet — degrade to "nothing pushed" rather than crash
        # (same shape as ``pushed_ticker_label_split.run``).
        alerts: list[dict] = []
    else:
        try:
            alerts = _fetch_alerts(rec_conn, hours)
        finally:
            rec_conn.close()

    out = compute_held_alert_reaction_latency(
        arts, alerts, tickers,
        mention_lookback_hours=mention_lookback_hours,
    )
    out["window_h"] = round(float(hours), 3)
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Per-held-ticker push reaction latency over the recent "
                     "Discord alerts (articles.db × alert_recency.db)."))
    parser.add_argument("--hours", type=float, default=DEFAULT_WINDOW_HOURS,
                        help=(f"Push window in hours (default "
                              f"{DEFAULT_WINDOW_HOURS}, clamped to recency TTL)."))
    parser.add_argument("--lookback", type=float,
                        default=DEFAULT_MENTION_LOOKBACK_HOURS,
                        help=(f"Per-push prior-mention lookback in hours "
                              f"(default {DEFAULT_MENTION_LOOKBACK_HOURS})."))
    parser.add_argument("--pretty", action="store_true",
                        help="Indent the JSON output.")
    args = parser.parse_args()
    report = run(hours=args.hours,
                 mention_lookback_hours=args.lookback)
    if args.pretty:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
