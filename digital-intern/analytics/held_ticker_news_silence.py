"""Held-ticker news-silence audit — multi-window per-ticker coverage view.

Why this exists (news-analyst lens): the briefing's ``_book_silence_lines`` /
``_book_heat_lines`` answer "which held names did the *5h digest* touch?",
and ``ArticleStore.ticker_mention_velocity`` answers "1h-vs-1h velocity for a
ticker set". Neither answers the analyst's actual standing question:

  *For each name I hold, how is news coverage trending across 1h / 6h / 24h,
  and is any one publisher the SOLE source of it (echo) versus genuine
  cross-outlet corroboration?*

A ticker held by the book with ZERO live mentions in the last 24h is a
**DARK** name — the analyst is operating blind on it. A ticker with non-zero
mentions but only one distinct publisher is an **ECHO** name — coverage
exists but is one outlet repeating itself (the same noise pattern that
drives the briefing-side echo detection at ``ECHO_MIN_COPIES=3`` in
``analysis.claude_analyst``). Both deserve manual attention regardless of
ai_score or urgency.

Companion to:
  * ``ArticleStore.ticker_mention_velocity`` — 1h-vs-1h velocity primitive
    (used by chat enrichment and the briefing book-velocity block); shares
    the case-insensitive word-boundary match convention so a tag like
    ``$NVDA`` matches ``NVDA`` and ``NVDAQ`` does not leak (the
    ``\\b\\$?TICKER\\b`` discriminator).
  * ``analysis.claude_analyst._book_silence_lines`` — same "which held names
    are silent" question, but scoped to the 5h briefing window and emitted
    as a single Opus input line. This audit is the operator-facing CLI
    counterpart over arbitrary windows with per-source diversity.
  * ``analytics.portfolio_overlap_scorer`` — ranks recent articles by
    held-ticker mention COUNT, top-N by overlap. This audit answers the
    inverse: which held tickers have the LEAST (or no) coverage, with a
    diversity verdict.

Held-ticker set is sourced from ``ml.features.LIVE_PORTFOLIO_TICKERS`` — the
single source of truth already used by ``watchers.alert_agent._book_tickers``
(the alert ``book:`` tag), ``analysis.claude_analyst._book_tickers`` (the
briefing ``[BOOK:]`` tag), and the model's ticker-density feature. Reading
the canonical set keeps this audit drift-free with every other held-book
surface; pinned by ``TestUsesLivePortfolioTickers``.

Load-bearing invariants respected:

  * **Backtest isolation:** ``LIVE_ONLY_CLAUSE`` (byte-identical to
    ``storage.article_store._LIVE_ONLY_CLAUSE``; pinned by
    ``TestLiveOnlyClauseInSync``) — synthetic ``backtest://`` rows and
    ``backtest_*`` / ``opus_annotation*`` sources can never inflate a
    per-ticker count nor mask a real DARK verdict by adding to denominator.
  * **score_source separation / urgency state machine:** read-only — no DB
    write, no ai_score / ml_score / score_source / urgency mutation by
    construction. The DB is opened ``mode=ro``.
  * **Test-pinned:** the pure aggregator runs on synthesized tuples (no
    DB), and the inline ``_LIVE_ONLY_CLAUSE`` is drift-tested.

CLI::

    python3 -m analytics.held_ticker_news_silence
    python3 -m analytics.held_ticker_news_silence --json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from ml.features import LIVE_PORTFOLIO_TICKERS

# Canonical backtest-isolation clause. Duplicated verbatim from
# ``storage/article_store.py::_LIVE_ONLY_CLAUSE`` so this audit carries no
# import-graph dependency on the writer module's full surface. The test
# suite pins a byte-identical drift check (``TestLiveOnlyClauseInSync``).
LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

_USB_PATH = Path("/media/zeph/projects/digital-intern/db")
_LOCAL_PATH = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = Path("/home/zeph/logs/held_ticker_news_silence.json")

# Window grid the audit reports against. Ordered short-to-long so the JSON
# rendering walks the natural "is this an acceleration or a fade?" cadence.
WINDOWS_H: tuple[int, ...] = (1, 6, 24)
# The longest window also bounds the DB read — anything older is irrelevant
# to every reported counter.
SCAN_WINDOW_H = max(WINDOWS_H)


# ── Verdict ladder ──────────────────────────────────────────────────────────
# DARK    — zero mentions in the 24h window. The analyst is operating blind.
# ECHO    — mentions exist but every one is from a single distinct ``source``
#           tag. Same conceptual class as the briefing-side ``[echo]`` tag
#           (one publisher repeating itself) but at the *holding* level
#           instead of the cluster level.
# NORMAL  — mentions from 2+ distinct sources in 24h, but not unusually
#           heavy (no acceleration into the 1h window).
# HOT     — 1h count >= ``HOT_RECENT_THRESHOLD``. The bar is conservative so a
#           sleepy ticker that gets a single fresh mention does not light up.
HOT_RECENT_THRESHOLD = 3


def _build_ticker_pattern(tickers: Iterable[str]) -> re.Pattern[str]:
    """Compile one word-boundary pattern matching any ticker in ``tickers``
    with an optional leading ``$``.

    Case-insensitive (most titles are mixed case; a real ticker in a body
    like ``Nvidia (NVDA)`` and a lowercase reference like ``nvda surges``
    both count). The ``\\b\\$?`` lead is the same convention used by
    ``ArticleStore.ticker_mention_velocity`` — anti-drift with the
    velocity primitive every other held-book surface keys on.
    """
    items = sorted({t.upper() for t in tickers if t and len(t) >= 2},
                   key=len, reverse=True)
    if not items:
        return re.compile(r"(?!x)x")  # never matches
    alt = "|".join(re.escape(t) for t in items)
    return re.compile(rf"\b\$?(?:{alt})\b", re.IGNORECASE)


def _parse_first_seen(value) -> datetime | None:
    """ISO-8601 only — ``article_store`` writes ``first_seen`` in that
    format. Returns ``None`` on anything unparseable; that row is then
    skipped by the aggregator (cannot be bucketed without a clock)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_silence(
    rows: Iterable[tuple],
    tickers: Iterable[str],
    now: datetime | None = None,
    windows_h: tuple[int, ...] = WINDOWS_H,
    hot_recent_threshold: int = HOT_RECENT_THRESHOLD,
) -> list[dict]:
    """Pure: per-ticker multi-window coverage view.

    ``rows`` is an iterable of ``(title, source, first_seen)`` tuples — the
    minimal projection ``load_rows`` returns. ``tickers`` is the held set
    (sourced upstream from ``LIVE_PORTFOLIO_TICKERS``). ``now`` defaults to
    ``datetime.now(UTC)`` and is the clock against which every window cutoff
    is computed (tests pin against a fixed clock).

    Returns one dict per held ticker. Per ticker:
      * ``ticker``            — symbol, preserved verbatim
      * ``counts``            — ``{"1h": N, "6h": N, "24h": N}`` distinct
                                article matches; matching is at the title
                                level (a title that names two held tickers
                                counts for BOTH, but the same title twice in
                                the row stream counts once per ticker —
                                same convention as the briefing's
                                ``_book_tickers`` set-on-title).
      * ``distinct_sources``  — ``{"24h": N}`` count of distinct ``source``
                                tags across the 24h window. ``1`` is the
                                ECHO signal; ``0`` aligns with DARK.
      * ``verdict``           — ``DARK`` / ``ECHO`` / ``NORMAL`` / ``HOT``.

    Output is sorted by **verdict severity first** (DARK, then ECHO, then
    NORMAL, then HOT) so the analyst's eye lands on the gaps; ties by
    ticker name asc (stable, anti-flicker). Same severity-first discipline
    as ``analytics.alert_source_breakdown``'s alerted-desc sort.
    """
    now = now or datetime.now(timezone.utc)
    sorted_windows = sorted(set(windows_h))
    longest = sorted_windows[-1] if sorted_windows else 0
    cutoffs = {w: now - timedelta(hours=w) for w in sorted_windows}
    longest_cut = cutoffs[longest] if longest else now

    clean = sorted({t.upper() for t in tickers if t and len(t) >= 2})
    if not clean:
        return []

    pattern = _build_ticker_pattern(clean)

    # Per-ticker per-window counter + per-ticker 24h distinct-source set.
    counts: dict[str, dict[int, int]] = {
        t: {w: 0 for w in sorted_windows} for t in clean
    }
    sources_24h: dict[str, set[str]] = {t: set() for t in clean}

    for row in rows:
        try:
            title, source, first_seen = row
        except (TypeError, ValueError):
            # Defensive — a malformed row is skipped, not an exception.
            continue
        if not title:
            continue
        ts = _parse_first_seen(first_seen)
        if ts is None or ts < longest_cut:
            continue
        hits = {m.upper() for m in pattern.findall(title)}
        if not hits:
            continue
        src = str(source or "").strip()
        for t in hits:
            if t not in counts:
                continue  # title matched a ticker outside the held set
            for w in sorted_windows:
                if ts >= cutoffs[w]:
                    counts[t][w] += 1
            # 24h source diversity tracks the full long-window set; an empty
            # source tag is recorded as a literal "" so two empty rows
            # collapse to one distinct (same convention as the briefing's
            # ``_distinct_sources`` count).
            sources_24h[t].add(src)

    out: list[dict] = []
    for t in clean:
        per_window = counts[t]
        long_count = per_window.get(longest, 0)
        recent = per_window.get(1, 0) if 1 in sorted_windows else 0
        n_sources = len(sources_24h[t]) if long_count > 0 else 0
        if long_count == 0:
            verdict = "DARK"
        elif n_sources <= 1:
            verdict = "ECHO"
        elif recent >= hot_recent_threshold:
            verdict = "HOT"
        else:
            verdict = "NORMAL"
        out.append({
            "ticker": t,
            "counts": {f"{w}h": per_window[w] for w in sorted_windows},
            "distinct_sources": {f"{longest}h": n_sources},
            "verdict": verdict,
        })

    severity = {"DARK": 0, "ECHO": 1, "NORMAL": 2, "HOT": 3}
    out.sort(key=lambda r: (severity[r["verdict"]], r["ticker"]))
    return out


def resolve_db_path() -> Path:
    """Resolve the live ``articles.db`` (USB-preferred, else local data/).
    Mirrors ``storage.article_store._get_db_path`` semantics without
    triggering its writer-side ``mkdir`` side effects — a read-only audit
    must never materialise the fallback directory just to read."""
    usb_db = _USB_PATH / "articles.db"
    if _USB_PATH.exists() and (usb_db.exists() or _USB_PATH.is_mount()):
        return usb_db
    return _LOCAL_PATH / "articles.db"


def load_rows(
    db_path: Path | None = None, hours: int = SCAN_WINDOW_H,
) -> list[tuple]:
    """Read ``(title, source, first_seen)`` for live rows in the recent
    ``hours``. ``mode=ro`` so a concurrent writer storm cannot crash this
    audit; ``LIVE_ONLY_CLAUSE`` so synthetic rows never colour the view."""
    path = db_path or resolve_db_path()
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    try:
        con.execute("PRAGMA busy_timeout=2000")
        return con.execute(
            "SELECT title, source, first_seen FROM articles "
            f"WHERE first_seen >= ? AND {LIVE_ONLY_CLAUSE}",
            (cutoff,),
        ).fetchall()
    finally:
        con.close()


def build_report(per_ticker: list[dict], now: datetime | None = None) -> dict:
    """Pure: wrap the per-ticker rows in the JSON report payload with
    generation metadata and counts of each verdict. ``dark`` / ``echo``
    counts are the analyst's at-a-glance gaps gauge — a non-zero ``dark``
    means at least one held name has been blind all session."""
    now = now or datetime.now(timezone.utc)
    counts_by_verdict = {"DARK": 0, "ECHO": 0, "NORMAL": 0, "HOT": 0}
    for r in per_ticker:
        v = r.get("verdict")
        if v in counts_by_verdict:
            counts_by_verdict[v] += 1
    return {
        "generated_at": now.isoformat(),
        "windows_h": list(WINDOWS_H),
        "n_tickers": len(per_ticker),
        "verdict_counts": counts_by_verdict,
        "tickers": per_ticker,
    }


def run(
    db_path: Path | None = None,
    tickers: Iterable[str] | None = None,
    write: bool = True,
) -> dict:
    """End-to-end: read → aggregate → report (optionally persist)."""
    held = list(tickers) if tickers is not None else list(LIVE_PORTFOLIO_TICKERS)
    rows = load_rows(db_path)
    per_ticker = compute_silence(rows, held)
    report = build_report(per_ticker)
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(OUT_PATH)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--no-write", action="store_true",
                   help="skip writing OUT_PATH (CLI inspection only)")
    p.add_argument("--json", action="store_true",
                   help="print the full JSON report instead of the table")
    args = p.parse_args(argv)
    report = run(db_path=args.db, write=not args.no_write)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    vc = report["verdict_counts"]
    print(
        f"n_tickers={report['n_tickers']}  "
        f"dark={vc['DARK']}  echo={vc['ECHO']}  "
        f"normal={vc['NORMAL']}  hot={vc['HOT']}"
    )
    for r in report["tickers"]:
        c = r["counts"]
        n_src = r["distinct_sources"][f"{max(WINDOWS_H)}h"]
        print(
            f"  {r['ticker']:<6} {r['verdict']:<6}  "
            f"1h={c.get('1h', 0):>3}  6h={c.get('6h', 0):>3}  "
            f"24h={c.get('24h', 0):>3}  src24={n_src}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
