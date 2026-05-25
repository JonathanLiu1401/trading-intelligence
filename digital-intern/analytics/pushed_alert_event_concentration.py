"""analytics/pushed_alert_event_concentration.py — per-(held-ticker × event-
class) Discord-push concentration audit.

Why this exists (news-analyst lens): the analyst's #1 documented noise
complaint is duplicate / repeated BREAKING pushes about the same conceptual
event. The defense-in-depth gates that already exist collapse the easy cases:

  * ``watchers/alert_dedup.dedupe_urgent`` — same-batch exact-signature dups.
  * ``watchers.alert_recency.partition_already_alerted`` — cross-cycle exact-
    signature dups within ``ALERT_RECENCY_TTL_HOURS``.
  * ``watchers.alert_recency.partition_paraphrase_alerted`` — Jaccard ≥ 0.75
    paraphrase variants of a prior alert.

But a real failure mode SURVIVES all three: two BREAKING pushes about the
SAME held-ticker event whose canonical signatures share fewer salient tokens
than the conservative paraphrase threshold. Live evidence (2026-05-24, this
DB's ``alert_recency.db`` 24h window):

    0.93h ago: "Nvidia posts record $81.6B revenue, unveils $80B buyback plan - MSN"
    2.43h ago: "Nvidia posts $81.6B quarter, unveils $80B buyback plan - MSN"

Two pushes for the SAME NVDA-buyback event ~1.5h apart. Canonical signatures:
``nvidia posts record 81 6b revenue unveils 80b`` vs ``nvidia posts 81 6b
quarter unveils 80b buyback`` — shared salient tokens = 6, union = 10 →
Jaccard 0.60, well BELOW the documented ``PARAPHRASE_MIN_JACCARD = 0.75``
that would have caught it. Both pushed.

Sibling surfaces and the gap each leaves:

  * ``watchers.alert_recency.pushed_ticker_breakdown`` — "of the recent
    pushes, which held tickers were mentioned and how often?" Flat per-
    ticker count with no event-class axis, so it cannot distinguish two
    pushes about distinct facets of the NVDA wire (one buyback, one CFO
    memory-price commentary — genuinely separate events) from two pushes
    paraphrasing the SAME NVDA buyback.
  * ``watchers.alert_recency.ticker_burst_counts`` — same per-ticker count
    used by the alert prompt's ``burst:`` annotation. Same axis-missing
    limitation as above.
  * ``analytics.alert_delivery_audit`` — partitions urgent-row TOTALs into
    delivered vs gate-suppressed and attributes the suppressed ones to the
    fingerprint that caught them. Aggregate across all delivered pushes,
    no per-ticker × event slice.
  * ``analytics.pushed_alert_gate_regret`` — "of the pushes that fired,
    how many would TODAY's gates retroactively catch?" Measures fingerprint
    coverage drift; the event-class axis is orthogonal.
  * ``analytics.news_fatigue`` — measures repeated coverage of the same
    SUBJECT in articles.db (not pushes). Different surface entirely.

This module is the missing axis: a per-(held_ticker × event_class)
push-concentration audit over the recency window. It iterates the canonical
push ledger (``alert_recency.db``'s ``alerted_sig`` table within
``ALERT_RECENCY_TTL_HOURS``), extracts the (held_ticker, event_class) tuple
for each title via a small closed-vocabulary keyword set, and groups by
tuple. A pair with ``pushes >= CONCENTRATION_THRESHOLD`` is the noise
pattern the analyst persona complains about.

Closed-vocabulary discipline (deliberately narrow — same evidence-only
discipline as ``_LOW_AUTHORITY_DOMAINS`` and ``_RECAP_TEMPLATE_PATTERNS``):
the event-class keywords are restricted to a tight set the analyst actually
cares about as an actionable event class (EARNINGS / BUYBACK / GUIDANCE /
RATING / RATE). A title with none of these keywords resolves to no event
class and is not bucketed — the audit only counts pushes that map cleanly
to one of these explicit event categories. Real wire copy unrelated to any
closed-vocabulary class (e.g. "NVIDIA CFO ZH commentary on memory prices")
contributes nothing to the by-pair table, which is the safe direction —
the audit under-claims rather than over-claims.

Pure-builder design: ``build_concentration_report(pushed, live_tickers, …)``
is side-effect-free and takes the exact shape ``alert_recency.recent_alerts``
returns. Fully unit-testable without SQLite or live-portfolio access.
``main()`` wires the live ``alert_recency.db`` + ``ml.features.
LIVE_PORTFOLIO_TICKERS`` to it.

Load-bearing invariants respected (mirrors ``pushed_alert_gate_regret.py`` /
``alert_delivery_audit.py``):

  * **Backtest isolation:** ``alert_recency.db`` is push-write only and the
    alert path's ``_is_synthetic`` re-filter drops any ``backtest://`` row
    before ``send_urgent_alert`` ever runs, so the ledger by construction
    never carries a synthetic row. No SQL touch to ``articles.db``.
  * **score_source separation:** READ-only across the board; never touches
    ``ai_score`` / ``ml_score`` / ``score_source`` / ``urgency``.
  * **Read-only:** ``alert_recency.db`` opened ``mode=ro`` with the
    canonical short busy timeout; cannot perturb the alert path or add to
    writer contention. No ``articles.db`` access at all.

CLI: ``python3 -m analytics.pushed_alert_event_concentration [--hours 6]
[--pretty]``. ``--hours`` defaults to ``ALERT_RECENCY_TTL_HOURS`` so the
audit window matches what the live paraphrase gate already operates on (a
wider window would compare against signatures already pruned out of
``alerted_sig`` and would over-attribute them to "not pushed").
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from watchers.alert_recency import ALERT_RECENCY_TTL_HOURS


# ── Event-class taxonomy ────────────────────────────────────────────────────
# Closed-vocabulary keyword → event class. Each keyword is checked with the
# word-boundary convention shared by ``ml.features._LIVE_RE`` and
# ``watchers.alert_recency.ticker_burst_counts`` (the documented anti-drift
# discipline: an audit that scans titles must use the SAME match convention
# the live gates do, or two pushes counted "the same" by this audit may not
# be the same by what the analyst actually got via the prompt).
#
# Multi-word keys are matched with a space-tolerant pattern that allows one
# or more whitespace chars between the words. Single-word keys are
# word-boundary anchored.
#
# Mutually exclusive within a row: the FIRST class that matches wins. To
# avoid bias toward the alphabetical-first class, we evaluate classes in a
# deterministic specificity order — multi-word phrases (more specific) before
# single-word triggers (less specific). The class list itself is closed and
# small; expansion goes through evidence review the same way the recap
# fingerprint family does.
_EVENT_CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # RATE — Fed / central-bank actions. Multi-word phrases first so
    # "fed cuts" / "rate hike" cannot be mis-attributed to a single-word
    # match elsewhere.
    ("RATE", (
        "rate cut", "rate hike", "rate cuts", "rate hikes",
        "fed cuts", "fed hikes", "fed funds", "rate decision",
        "fomc",
    )),
    # RATING — analyst price-target changes / upgrades / downgrades.
    ("RATING", (
        "price target", "pt raised", "pt cut", "pt lowered",
        "upgraded", "downgraded", "upgrade", "downgrade",
        "upgrades", "downgrades",
    )),
    # GUIDANCE — forward-looking outlook revisions.
    ("GUIDANCE", (
        "guidance", "guides", "outlook", "forecast", "guided",
    )),
    # BUYBACK — capital return announcements (buyback / dividend / split).
    ("BUYBACK", (
        "buyback", "repurchase", "repurchases", "dividend",
        "share split", "stock split",
    )),
    # EARNINGS — quarterly print + beat/miss/blowout commentary. Listed
    # LAST so a title that contains both "earnings" and a more-specific
    # term (like "guidance" or "price target") is attributed to the more
    # specific class.
    ("EARNINGS", (
        "earnings", "eps", "beats estimates", "tops estimates",
        "missed estimates", "missed expectations", "blowout quarter",
        "revenue", "quarterly results",
    )),
)

# Minimum number of pushes for the SAME (held_ticker, event_class) tuple
# within the window before the pair is flagged as a concentration alert.
# 2 is the analyst's explicit "I got told the same thing twice" floor —
# this is the noise pattern the existing dedup layers were SUPPOSED to
# collapse and the live evidence (NVDA buyback ×2 above) shows leaks
# through when paraphrase Jaccard falls below 0.75.
CONCENTRATION_THRESHOLD = 2

# Cap on the by-pair table — a fully degraded window (single-name wire
# storm) must not emit a wall-of-text report that itself becomes noise.
# Same anti-noise capping discipline as ``_MAX_COVERAGE_LINES`` in
# ``analysis.claude_analyst`` and ``BRIEFING_MAX_PER_DOMAIN``.
_MAX_BY_PAIR_ROWS = 20


def event_class_for_title(title: str | None) -> str:
    """Pure: closed-vocabulary event class for a headline.

    Returns one of ``EARNINGS / BUYBACK / GUIDANCE / RATING / RATE`` or ``""``
    when no closed-vocabulary keyword fires. Multi-word phrases are matched
    with space-tolerant whitespace; single-word triggers use a word-boundary
    anchor. Class evaluation order is specificity-first (RATE / RATING /
    GUIDANCE / BUYBACK before EARNINGS) so a title with both "earnings" and
    a more-specific event noun is attributed to the more specific class.

    Conservative by construction: a title outside the closed vocabulary
    returns ``""`` and is not bucketed at all by the caller. The audit
    under-claims rather than over-claims.
    """
    if not title:
        return ""
    t_low = title.lower()
    for cls, kws in _EVENT_CLASSES:
        for kw in kws:
            if " " in kw:
                # Multi-word phrase: tolerate any whitespace run between
                # words (handles "rate  cut" double-space + "rate\tcut").
                pattern = r"\b" + r"\s+".join(re.escape(p) for p in kw.split()) + r"\b"
                if re.search(pattern, t_low):
                    return cls
            else:
                if re.search(r"\b" + re.escape(kw) + r"\b", t_low):
                    return cls
    return ""


# ── Company-name → ticker aliases (held-book name resolution) ───────────────
# Live evidence (2026-05-24, ``alert_recency.db`` 24h window): 5 of 8 pushes
# led with "Nvidia" or "NVIDIA", NOT the ticker "NVDA". An audit gated on
# the ticker spelling alone would silently miss every one of them — exactly
# the failure case this audit exists to surface. The fix is a tight,
# evidence-only alias map: each held ticker gets the company-name form(s)
# that ACTUALLY appear in the wire copy this audit consumes.
#
# Discipline (mirrors ``_LOW_AUTHORITY_DOMAINS`` / ``_RECAP_TEMPLATE_PATTERNS``):
#   * Closed list, evidence-only — no speculative additions.
#   * Each alias resolves to ONE held ticker; multi-line variants of the same
#     company collapse to the same target. The "micron technology" / "micron"
#     pair is the canonical shape — both forms appear in real headlines.
#   * Word-boundary case-insensitive match; multi-word phrases use
#     whitespace-tolerant matching (handles "Micron  Technology" double-space).
#   * Single-token aliases like "nvidia" are word-boundary anchored so
#     "nvidia.com" or "ennvidia" cannot leak.
_COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "NVDA":  ("nvidia",),
    "MSFT":  ("microsoft",),
    "MU":    ("micron", "micron technology"),
    "ORCL":  ("oracle",),
    "AXTI":  ("axt inc", "axt"),
    "LITE":  ("lumentum",),
    "QBTS":  ("d-wave", "dwave quantum", "d-wave quantum"),
    "TSEM":  ("tower semiconductor", "tower semi"),
}


def _held_tickers_in_title(
    title: str | None, live_tickers: Iterable[str]
) -> list[str]:
    """Pure: held tickers (uppercase, sorted, deduplicated) mentioned in
    a title by word-boundary case-insensitive match.

    Matches both the ticker spelling (``NVDA``) AND the held-book
    company-name aliases (``nvidia``) for tickers in ``_COMPANY_ALIASES``.
    Real wire copy leads with the company name FAR more often than the
    ticker ("Nvidia posts record $81.6B" not "NVDA posts record $81.6B"),
    so a ticker-only match would silently miss the live noise pattern this
    audit exists to surface (live: 5/8 NVDA pushes in 24h led with the
    "Nvidia" spelling). Aliases are only applied for tickers that are
    actually in the passed-in held universe — so a fresh portfolio without
    NVDA never bucketize a Nvidia mention.

    Empty or non-string titles return ``[]``. Mirrors the case-insensitive
    word-boundary convention of ``ml.features._LIVE_RE`` /
    ``watchers.alert_recency.ticker_burst_counts``.
    """
    if not title or not isinstance(title, str):
        return []
    # Normalize held-ticker list (uppercase, deduplicated, sanitized).
    norm: set[str] = set()
    for t in live_tickers or ():
        if isinstance(t, str):
            t2 = t.strip().upper()
            # Same hygiene as ml.features._TICKER_RE — A-Z0-9, 1..6.
            if t2 and re.fullmatch(r"[A-Z0-9]{1,6}", t2):
                norm.add(t2)
    if not norm:
        return []
    # Ticker spellings: anchored word-boundary alternation.
    ticker_pat = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in sorted(norm)) + r")\b",
        re.IGNORECASE,
    )
    hits = {m.upper() for m in ticker_pat.findall(title)}
    # Company-name aliases: only for tickers actually in the held set.
    # Multi-word phrases use whitespace-tolerant matching.
    for t in norm:
        for alias in _COMPANY_ALIASES.get(t, ()):
            if " " in alias or "-" in alias:
                # Multi-word / hyphenated phrase — match with whitespace and
                # hyphen tolerance ("d-wave" matches "D-Wave" or "D Wave").
                parts = re.split(r"[-\s]+", alias)
                pattern = r"\b" + r"[-\s]+".join(re.escape(p) for p in parts) + r"\b"
            else:
                pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, title, re.IGNORECASE):
                hits.add(t)
                break  # one alias hit is enough; don't multi-count the same ticker
    return sorted(hits)


def build_concentration_report(
    pushed: Iterable[dict],
    live_tickers: Iterable[str],
    *,
    window_h: float = ALERT_RECENCY_TTL_HOURS,
    concentration_threshold: int = CONCENTRATION_THRESHOLD,
    max_by_pair_rows: int = _MAX_BY_PAIR_ROWS,
) -> dict:
    """Pure-function builder. Returns the JSON snapshot.

    ``pushed`` is an iterable of ``{"title": str, "age_hours": float, ...}``
    dicts — exactly the shape ``watchers.alert_recency.recent_alerts``
    returns. ``live_tickers`` is the held-book universe to gate matches
    against (see ``_held_tickers_in_title``).

    Returns::

        {
          "window_h":              float,    # clamped to >= 0.01
          "concentration_threshold": int,    # pairs with >= this trigger an alert
          "total_pushes":          int,      # input count (after dropping titleless)
          "pushes_with_class":     int,      # subset that mapped to a closed-vocab class
          "pushes_held_x_class":   int,      # subset with BOTH a held ticker AND a class
          "distinct_pairs":        int,      # distinct (ticker, class) pairs seen
          "by_pair": [                       # sorted desc by pushes, alpha tiebreak
            {
              "ticker":          str,
              "event_class":     str,
              "pushes":          int,
              "newest_age_h":    float | None,
              "newest_title":    str,
              "titles":          [str, ...]  # all pushed titles, newest first
            },
            ... (capped at ``max_by_pair_rows``)
          ],
          "concentration_alerts": [str, ...] # human-readable per-pair lines
                                              # where pushes >= threshold
        }

    Discipline:
      * Empty input → fully-shaped dict with zeros and empty lists (same
        zero-data discipline as ``pushed_alert_gate_regret.build_regret_report``).
      * A title with no closed-vocab class is counted in ``total_pushes``
        but NEVER appears in ``by_pair`` / ``concentration_alerts``.
      * A title with a class but no held-ticker match is counted in
        ``pushes_with_class`` but NEVER appears in ``by_pair`` either —
        the audit's purpose is HELD-ticker noise concentration.
      * A title mentioning multiple held tickers is counted in every pair
        (per-ticker), reflecting that the analyst sees the push for each
        relevant position. Same per-ticker dedup discipline as
        ``ticker_burst_counts`` (a single push mentioning NVDA twice still
        counts as one for that pair).
      * The sort is descending by pushes with an alphabetical tiebreak
        (ticker then event_class) so the table is stable cycle-to-cycle —
        mirrors ``urgency_label_split_by_source``.
    """
    # Clamp window_h to a strictly-positive float so a divide-by-zero in
    # downstream display formatting cannot occur. 0.01h = 36s, well below
    # any production cadence.
    window_h = max(float(window_h), 0.01)
    concentration_threshold = max(int(concentration_threshold), 1)
    max_by_pair_rows = max(int(max_by_pair_rows), 1)

    # Materialize input so we can iterate twice (counts then build).
    rows: list[dict] = []
    for r in pushed or ():
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or "").strip()
        if not title:
            continue
        try:
            age_h = float(r.get("age_hours") or 0.0)
        except (TypeError, ValueError):
            age_h = 0.0
        rows.append({"title": title, "age_hours": max(0.0, age_h)})

    total_pushes = len(rows)
    pushes_with_class = 0
    pushes_held_x_class = 0

    # Aggregate per (ticker, class). Track newest-push age, newest-push
    # title, and the full title list (sorted newest-first by age_h).
    pair_pushes: dict[tuple[str, str], int] = {}
    pair_newest_age: dict[tuple[str, str], float | None] = {}
    pair_newest_title: dict[tuple[str, str], str] = {}
    pair_titles: dict[tuple[str, str], list[tuple[float, str]]] = {}

    for r in rows:
        title = r["title"]
        age_h = r["age_hours"]
        cls = event_class_for_title(title)
        if not cls:
            continue
        pushes_with_class += 1
        held = _held_tickers_in_title(title, live_tickers)
        if not held:
            continue
        # Count this push once per (held_ticker, class) pair — distinct
        # tickers in the same title each get a tally for that title.
        pushes_held_x_class += 1
        for t in held:
            key = (t, cls)
            pair_pushes[key] = pair_pushes.get(key, 0) + 1
            cur_age = pair_newest_age.get(key)
            if cur_age is None or age_h < cur_age:
                pair_newest_age[key] = age_h
                pair_newest_title[key] = title
            pair_titles.setdefault(key, []).append((age_h, title))

    # Build the by_pair table. Sort each pair's titles newest-first
    # (smallest age first); the pair table itself sorts pushes-desc with
    # alphabetical tiebreak.
    by_pair: list[dict] = []
    for key, count in pair_pushes.items():
        ticker, cls = key
        titles_sorted = [t for _, t in sorted(pair_titles[key], key=lambda x: x[0])]
        by_pair.append({
            "ticker": ticker,
            "event_class": cls,
            "pushes": count,
            "newest_age_h": (
                round(pair_newest_age[key], 2)
                if pair_newest_age.get(key) is not None else None
            ),
            "newest_title": pair_newest_title.get(key, ""),
            "titles": titles_sorted,
        })
    # Sort desc by pushes, then alphabetical by (ticker, event_class) for
    # deterministic cycle-to-cycle ordering.
    by_pair.sort(key=lambda r: (-r["pushes"], r["ticker"], r["event_class"]))
    distinct_pairs = len(by_pair)
    by_pair = by_pair[:max_by_pair_rows]

    # Concentration alerts: one human-readable line per pair at-or-above
    # the threshold. Ordered the same way as by_pair so the operator can
    # cross-reference position-by-position.
    concentration_alerts: list[str] = []
    for row in by_pair:
        if row["pushes"] >= concentration_threshold:
            newest = row["newest_age_h"]
            newest_str = f"{newest:.2f}h ago" if newest is not None else "n/a"
            concentration_alerts.append(
                f"{row['ticker']} × {row['event_class']}: "
                f"{row['pushes']} pushes in last {window_h:.1f}h "
                f"(newest {newest_str})"
            )

    return {
        "window_h": round(window_h, 2),
        "concentration_threshold": concentration_threshold,
        "total_pushes": total_pushes,
        "pushes_with_class": pushes_with_class,
        "pushes_held_x_class": pushes_held_x_class,
        "distinct_pairs": distinct_pairs,
        "by_pair": by_pair,
        "concentration_alerts": concentration_alerts,
    }


# ── CLI / live wiring ───────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent
_RECENCY_DB = _BASE_DIR / "data" / "alert_recency.db"


def _load_pushed(hours: float) -> list[dict]:
    """Best-effort read of ``alert_recency.db``'s alerted_sig table within
    ``hours``, shaped to match what ``recent_alerts`` returns.

    Opens a fresh short-lived ``mode=ro`` connection (never the daemon's
    shared connection — the documented cursor-collision hazard). Returns
    ``[]`` on ANY failure so the CLI degrades cleanly when the recency
    DB is missing on a fresh install.
    """
    if not _RECENCY_DB.exists():
        return []
    try:
        conn = sqlite3.connect(
            f"file:{_RECENCY_DB}?mode=ro", uri=True, timeout=5,
        )
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            from datetime import datetime, timedelta, timezone
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            rows = conn.execute(
                "SELECT title, last_ts FROM alerted_sig "
                "WHERE last_ts >= ? ORDER BY last_ts DESC",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return []
    out: list[dict] = []
    now = None
    from datetime import datetime, timezone
    base = datetime.now(timezone.utc)
    for title, last_ts in rows:
        if not title:
            continue
        try:
            dt = datetime.fromisoformat(str(last_ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_h = max(0.0, (base - dt).total_seconds() / 3600.0)
        except Exception:
            continue
        out.append({"title": title, "age_hours": age_h})
    return out


def main() -> int:
    """CLI entrypoint. Pretty-prints the JSON report. Returns 0 on success."""
    parser = argparse.ArgumentParser(
        description="Per-(held-ticker × event-class) Discord-push concentration audit.",
    )
    parser.add_argument(
        "--hours", type=float, default=ALERT_RECENCY_TTL_HOURS,
        help=f"Window in hours (default: {ALERT_RECENCY_TTL_HOURS}).",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Indent the JSON output for human reading.",
    )
    args = parser.parse_args()

    pushed = _load_pushed(args.hours)
    # Lazy import — keeps the analytics module's import surface minimal
    # and the test harness can monkeypatch LIVE_PORTFOLIO_TICKERS without
    # going through this CLI path.
    from ml.features import LIVE_PORTFOLIO_TICKERS
    report = build_concentration_report(
        pushed, LIVE_PORTFOLIO_TICKERS, window_h=args.hours,
    )
    indent = 2 if args.pretty else None
    print(json.dumps(report, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
