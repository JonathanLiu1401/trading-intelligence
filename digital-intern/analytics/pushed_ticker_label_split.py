"""analytics/pushed_ticker_label_split.py — per-held-ticker push calibration.

Why this exists (news-analyst lens): three sibling metrics already live in this
repo and EACH leaves a gap:

  * ``storage.article_store.urgency_label_split_by_ticker`` slices urgent rows
    (``urgency>=1``) per held ticker. It is gate-noise-inflated — the alert
    formatter calls ``mark_alerted_batch`` on every row a defense-in-depth gate
    absorbs (quote_widget / recap_template / low_authority / stale_published),
    so a ticker with 50 ml-only urgency=1 rows that the recap gate filtered
    reads identically to one with 50 real Discord pushes the analyst received.
  * ``watchers.alert_recency.pushed_ticker_breakdown`` IS push-correct (only
    ``record_alerted`` after ``discord_send`` succeeds writes to
    ``alert_recency.db``), but carries NO ``score_source`` dimension — so the
    analyst sees "NVDA was pushed 12 times in 6h" without knowing whether the
    pushes were Sonnet-vetted (ground truth) or model-only (unverified).
  * ``analytics.alert_delivery_audit.delivered_by_source`` joins the two and
    DOES surface ``delivered_llm_fraction`` — but only as an AGGREGATE across
    all delivered pushes. The per-held-ticker answer — "which of MY OPEN
    POSITIONS are getting good vs bad urgent vetting?" — is exactly the
    third-axis slice the existing metrics don't expose.

This module is that slice. It joins ``articles.db`` urgency=2 rows in window
against ``alert_recency.db`` alerted signatures, keeps only rows whose
signature appeared as a real push, then partitions the survivors per held
ticker by ``score_source`` — producing the answer to "of the alerts I was
pushed about NVDA, how many were LLM-vetted vs unverified ml-only?"

Load-bearing invariants respected (mirrors ``alert_delivery_audit.py``):

  * **Backtest isolation:** the SQL pull carries the canonical
    ``_LIVE_ONLY_CLAUSE`` verbatim. Synthetic backtest/opus rows cannot enter
    the per-ticker breakdown.
  * **score_source separation:** ``ai_score`` / ``ml_score`` / ``score_source``
    are READ only — never written. The audit derives no labels.
  * **Read-only:** both DBs opened ``mode=ro`` with a short busy timeout.
    Cannot perturb the alert path or add to writer contention.

CLI: ``python3 -m analytics.pushed_ticker_label_split [--hours 6]`` prints a
JSON report; ``--pretty`` indents.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from watchers.alert_dedup import _signature
from watchers.alert_recency import ALERT_RECENCY_TTL_HOURS

# Canonical backtest-isolation clause — duplicated verbatim from
# storage/article_store.py::_LIVE_ONLY_CLAUSE (the documented anti-drift
# discipline; the analytics/storage family pins a drift check in tests).
LIVE_ONLY_CLAUSE = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

_USB_PATH = Path(os.environ.get(
    "DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"))
_LOCAL_PATH = Path(__file__).resolve().parent.parent / "data"

# Default window matches the recency-store TTL exactly. A wider window would
# compare urgency=2 rows against signatures already pruned out of
# ``alerted_sig`` and would over-attribute them to "not pushed".
DEFAULT_WINDOW_HOURS = ALERT_RECENCY_TTL_HOURS


def _empty_buckets() -> dict[str, int]:
    """Zero counts in every canonical bucket — mirrors
    ``urgency_label_split`` so a UI can render every key without conditional
    branches when the window is empty."""
    return {"llm": 0, "ml": 0, "briefing_boost": 0, "null": 0}


def _bucket(art: dict) -> str:
    """Map an article row's ``score_source`` to one of the four canonical
    buckets — same convention as ``urgency_label_split`` and
    ``alert_delivery_audit._score_source_bucket``."""
    src = art.get("score_source")
    return src if src in ("llm", "ml", "briefing_boost") else "null"


def _llm_fraction(b: dict[str, int]) -> float:
    """``(llm + briefing_boost) / total``, 0.0 on an empty bucket set —
    matches ``urgency_label_split``'s definition verbatim so the per-ticker
    figure is directly comparable to the aggregate metric."""
    vetted = b["llm"] + b["briefing_boost"]
    total = sum(b.values())
    return round(vetted / total, 4) if total else 0.0


def _clean_tickers(raw_tickers: Iterable[str]) -> list[str]:
    """Uppercase, dedupe, skip <2-char entries — same hygiene as
    ``urgency_label_split_by_ticker`` / ``ticker_burst_counts``."""
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


def compute_pushed_ticker_label_split(
    urgent_rows: Iterable[dict],
    alerted_sigs: set[str],
    tickers: Iterable[str],
) -> dict:
    """Pure function — no DB / IO. Per-held-ticker breakdown of REAL Discord
    pushes by ``score_source``.

    ``urgent_rows`` carries the union of ``urgency=2`` rows in the window;
    only rows whose canonical ``_signature`` is in ``alerted_sigs`` count as
    real pushes (gate-marked rows are dropped — that is the whole point).

    For each held ticker we count the distinct PUSHES that mention it
    (title + summary, whole-word match — same surface as
    ``alert_agent._book_tickers`` and ``ml.features._LIVE_RE`` so the alert
    path and this metric can never disagree about whether a row touches the
    held book).

    Returns:

      .. code-block:: python

        {
            "total_pushes": int,        # distinct push-signatures in window
            "by_ticker": [               # held names with >= 1 push,
                {                        #   sorted most-ml-only-first
                    "ticker": str,
                    "total": int,
                    "llm": int,
                    "ml": int,
                    "briefing_boost": int,
                    "null": int,
                    "llm_fraction": float,
                },
                ...
            ],
            "silent_tickers": [str, ...],  # held names with zero pushes
        }

    Each push (one row per distinct signature) is counted ONCE per ticker it
    mentions, even if multiple urgency=2 articles share that signature
    (i.e. syndicated copies fold into one push from the analyst's POV — the
    invariant ``alert_recency.db`` itself enforces). When multiple urgent
    rows share a signature, the LLM-vetted copy wins the source attribution
    (``ai_score > 0`` → ``score_source='llm'`` / ``'briefing_boost'``); a
    ground-truth label always beats a model self-prediction. Mirrors the
    cross-product anti-drift discipline already documented on
    ``alert_delivery_audit.py``.
    """
    clean = _clean_tickers(tickers)

    # Group urgent rows by signature; keep only the one with the highest
    # severity score per signature (LLM-vetted > ML-only). This collapses
    # syndicated copies into one push the analyst's POV — same fold that
    # ``alert_recency.db`` performs at record time. Done BEFORE the
    # empty-tickers early return so ``total_pushes`` always reflects the real
    # distinct-push count, never silently zero just because the caller didn't
    # supply a held book.
    by_sig: dict[str, dict] = {}
    for art in urgent_rows:
        sig = _signature(art.get("title"))
        if not sig or sig not in alerted_sigs:
            continue
        cur = by_sig.get(sig)
        if cur is None:
            by_sig[sig] = dict(art)
            continue
        # LLM-vetted (ai_score > 0) wins over ML-only every time.
        cur_ai = cur.get("ai_score") or 0
        new_ai = art.get("ai_score") or 0
        if new_ai > cur_ai:
            by_sig[sig] = dict(art)

    if not clean:
        return {"total_pushes": len(by_sig), "by_ticker": [],
                "silent_tickers": []}

    # Single compiled alternation across all uppercase tickers — one regex
    # walk per row regardless of held-book size. Mirrors ``_LIVE_RE``'s
    # word-boundary convention so "AMD" never matches inside "DAMD".
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in clean) + r")\b",
        re.IGNORECASE,
    )

    # Per-ticker tallies. counts[ticker][bucket] = N.
    counts: dict[str, dict[str, int]] = {t: _empty_buckets() for t in clean}
    for art in by_sig.values():
        blob = f"{art.get('title') or ''} {art.get('summary') or ''}"
        if not blob.strip():
            continue
        hits = {m.upper() for m in pattern.findall(blob)}
        if not hits:
            continue
        bkt = _bucket(art)
        for t in hits:
            if t in counts:
                counts[t][bkt] += 1

    materialised: list[dict] = []
    silent: list[str] = []
    for t in clean:
        b = counts[t]
        total = sum(b.values())
        if total == 0:
            silent.append(t)
            continue
        materialised.append({
            "ticker": t,
            "total": total,
            "llm": b["llm"],
            "ml": b["ml"],
            "briefing_boost": b["briefing_boost"],
            "null": b["null"],
            "llm_fraction": _llm_fraction(b),
        })
    # Most-ml-only-first (largest ``ml`` count); alphabetical tiebreak —
    # mirrors ``urgency_label_split_by_source`` / ``_by_ticker``.
    materialised.sort(key=lambda r: (-r["ml"], r["ticker"]))

    return {
        "total_pushes": len(by_sig),
        "by_ticker": materialised,
        "silent_tickers": silent,
    }


def resolve_db_paths() -> tuple[Path, Path]:
    """Resolve live ``articles.db`` (USB-preferred) and ``alert_recency.db``
    (always local — see ``watchers.alert_recency.DB_PATH``). No side effects;
    mirrors ``alert_delivery_audit.resolve_db_paths``."""
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


def _fetch_urgent_rows(conn: sqlite3.Connection, hours: float) -> list[dict]:
    """Pull urgency=2 rows from articles.db inside the window. The summary
    column is decompressed in Python so the pure helper above operates on the
    same ``title+summary`` surface as ``alert_agent._book_tickers``."""
    from storage.article_store import decompress
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    rows = conn.execute(
        "SELECT id, title, source, full_text, ai_score, ml_score, score_source "
        "FROM articles "
        f"WHERE urgency=2 AND first_seen >= ? AND {LIVE_ONLY_CLAUSE}",
        (since,),
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        out.append({
            "_id": r[0], "title": r[1] or "", "source": r[2] or "",
            "summary": decompress(r[3]) if r[3] else "",
            "ai_score": r[4], "ml_score": r[5], "score_source": r[6],
        })
    return out


def _fetch_alerted_sigs(conn: sqlite3.Connection, hours: float) -> set[str]:
    """Mirror ``alert_delivery_audit._fetch_alerted_sigs`` — re-implements the
    SELECT rather than calling ``recent_signatures`` so a future
    alert_recency API change can't break this audit."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    rows = conn.execute(
        "SELECT sig FROM alerted_sig WHERE last_ts >= ?", (cutoff,),
    ).fetchall()
    return {r[0] for r in rows if r[0]}


def run(tickers: Iterable[str] | None = None,
        hours: float = DEFAULT_WINDOW_HOURS) -> dict:
    """DB shell: open both stores read-only, pull data, compose the audit.

    ``tickers`` defaults to ``ml.features.LIVE_PORTFOLIO_TICKERS`` (the live
    held-book SSOT — config/portfolio.json union'd with the hardcoded
    fallback) so a CLI invocation works out of the box on a live host.

    Window is clamped to the recency TTL (same rationale as
    ``alert_delivery_audit.run_audit``)."""
    if tickers is None:
        from ml.features import LIVE_PORTFOLIO_TICKERS
        tickers = sorted(LIVE_PORTFOLIO_TICKERS)
    if hours > ALERT_RECENCY_TTL_HOURS + 1e-6:
        hours = ALERT_RECENCY_TTL_HOURS

    articles_db, recency_db = resolve_db_paths()
    art_conn = _open_ro(articles_db)
    try:
        urgent = _fetch_urgent_rows(art_conn, hours)
    finally:
        art_conn.close()

    try:
        rec_conn = _open_ro(recency_db)
    except sqlite3.OperationalError:
        # No recency DB yet — degrade to "nothing pushed" rather than crash
        # (same shape as ``alert_delivery_audit.run_audit``).
        alerted: set[str] = set()
    else:
        try:
            alerted = _fetch_alerted_sigs(rec_conn, hours)
        finally:
            rec_conn.close()

    out = compute_pushed_ticker_label_split(urgent, alerted, tickers)
    out["window_h"] = round(float(hours), 3)
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--hours", type=float, default=DEFAULT_WINDOW_HOURS,
        help=f"Window in hours (default and max: {DEFAULT_WINDOW_HOURS:.1f}, "
             f"the alert_recency TTL).",
    )
    p.add_argument("--pretty", action="store_true",
                   help="Pretty-print the JSON report.")
    args = p.parse_args()
    report = run(hours=args.hours)
    print(json.dumps(report, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
