"""Per-position news-flow cooldown — has the news desk gone quiet on this
held ticker, or is the story still moving?

``analytics/position_attention.py`` answers "when did *Opus* last examine
this ticker." ``analytics/thesis_drift.py`` re-tests an opening rationale
against current state. ``analytics/news_velocity.py`` measures whole-book
news flow over time. None answer the per-ticker question:

  **"For each name I'm holding, when was the last live article that
  actually scored above noise?"**

The pathology this catches is *thesis decay through silence*: a position
opened on a catalyst whose news flow has dried up. The catalyst window
closes; the position sits in inventory while operator attention drifts
elsewhere. There is no error message, no NO_DECISION storm, no Discord
ping — only an absence. This module surfaces that absence per ticker and
rolls up to a portfolio-level verdict.

Design parity with the codebase:

* **Pure builder.** Takes ``open_positions`` + a pre-fetched
  ``last_news_by_ticker`` dict (the network/SQLite work belongs in the
  endpoint, mirroring ``position_attention`` / ``thesis_drift`` /
  ``correlation``). ``now`` is injectable for tests.
* **Live-only filter belongs upstream.** This module assumes the caller
  has already filtered out ``backtest://`` URLs and ``backtest_*`` /
  ``opus_annotation*`` sources (invariant #1) — same contract as
  ``signals.get_top_signals``.
* **Verdict ladder mirrors ``position_attention``** —
  ``FRESH / WARM / COOL / DARK`` per position, rolled up to
  ``OK / COOLING_BOOK / DARK_BOOK / INSUFFICIENT_DATA``. The two
  diagnostics answer different questions but share the same shape so the
  operator's eye reads them the same way.

Advisory only — never gates Opus, never injected into the decision
prompt, adds no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

from datetime import datetime, timezone

# Per-position cooldown thresholds (hours since last live article above
# ``MIN_SCORE_THRESHOLD``). FRESH ≤ 6h is one trading session; WARM ≤ 24h
# is "yesterday's news still counts"; COOL ≤ 72h is "story is aging fast";
# beyond that the catalyst window is effectively closed.
FRESH_H = 6.0
WARM_H = 24.0
COOL_H = 72.0

# Below this ai_score an article is noise and shouldn't keep a ticker out
# of DARK. The endpoint applies the same threshold when querying so the
# builder only sees scored hits.
MIN_SCORE_THRESHOLD = 4.0


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _classify(hours_since: float | None) -> str:
    if hours_since is None:
        return "DARK"
    if hours_since <= FRESH_H:
        return "FRESH"
    if hours_since <= WARM_H:
        return "WARM"
    if hours_since <= COOL_H:
        return "COOL"
    return "DARK"


def build_position_news_cooldown(
    open_positions: list[dict],
    last_news_by_ticker: dict[str, dict],
    now: datetime | None = None,
    min_score_threshold: float = MIN_SCORE_THRESHOLD,
) -> dict:
    """Per-open-position news-flow cooldown summary.

    Inputs:
      open_positions       — store.open_positions() rows (any type; options
                             are tracked under their underlying ticker).
      last_news_by_ticker  — {TICKER_UPPER: {
                                "last_first_seen": iso8601 | None,
                                "top_score": float | None,
                                "top_title": str | None,
                                "n_24h": int,
                                "n_72h": int,
                             }}
                             The endpoint pre-fetches this with the live-only
                             clause + ai_score >= min_score_threshold; missing
                             tickers are treated as DARK.
      now                  — defaults to wall clock UTC; injectable.
      min_score_threshold  — surfaced in the response so a reader can tell
                             what "no news" actually means here.

    Output schema:
      {
        "as_of": iso8601,
        "n_positions": int,
        "min_score_threshold": float,
        "positions": [
          {
            "ticker", "type", "qty",
            "last_news_ts": iso | None,
            "hours_since_last_news": float | None,
            "top_score_72h": float | None,
            "top_title_72h": str | None,
            "n_articles_24h": int,
            "n_articles_72h": int,
            "verdict": FRESH / WARM / COOL / DARK,
          }, ...
        ],
        "summary": {"fresh": int, "warm": int, "cool": int, "dark": int},
        "verdict": OK / COOLING_BOOK / DARK_BOOK / INSUFFICIENT_DATA,
        "note": str,
        "thresholds_hours": {"fresh_le": ..., "warm_le": ..., "cool_le": ...},
      }
    """
    if now is None:
        now = datetime.now(timezone.utc)

    rows: list[dict] = []
    summary = {"fresh": 0, "warm": 0, "cool": 0, "dark": 0}

    for p in open_positions or []:
        tk = (p.get("ticker") or "").upper()
        if not tk:
            continue
        news = last_news_by_ticker.get(tk) or {}
        last_ts = _parse_ts(news.get("last_first_seen"))

        if last_ts is not None:
            hours_since = round((now - last_ts).total_seconds() / 3600.0, 2)
            last_ts_iso = last_ts.isoformat()
        else:
            hours_since = None
            last_ts_iso = None

        verdict = _classify(hours_since)
        summary[verdict.lower()] += 1

        rows.append({
            "ticker": tk,
            "type": p.get("type"),
            "qty": p.get("qty"),
            "last_news_ts": last_ts_iso,
            "hours_since_last_news": hours_since,
            "top_score_72h": news.get("top_score"),
            "top_title_72h": news.get("top_title"),
            "n_articles_24h": int(news.get("n_24h") or 0),
            "n_articles_72h": int(news.get("n_72h") or 0),
            "verdict": verdict,
        })

    # Worst-first: DARK on top, then by hours_since descending (oldest
    # silence most visible). None sorts as "infinity" so never-seen-in-news
    # positions float to the very top of DARK.
    def _sort_key(r: dict):
        order = {"DARK": 0, "COOL": 1, "WARM": 2, "FRESH": 3}
        h = r["hours_since_last_news"]
        return (order.get(r["verdict"], 9), -(h if h is not None else 1e9))

    rows.sort(key=_sort_key)

    n = len(rows)
    if n == 0:
        verdict = "INSUFFICIENT_DATA"
        note = "No open positions to evaluate."
    elif summary["dark"] > 0:
        verdict = "DARK_BOOK"
        note = (f"{summary['dark']} of {n} held position(s) have had no "
                f"live article above ai_score≥{min_score_threshold:g} in "
                f">{COOL_H:.0f}h — the catalyst window is effectively "
                f"closed on these names. Reconsider thesis.")
    elif summary["cool"] > 0:
        verdict = "COOLING_BOOK"
        note = (f"{summary['cool']} of {n} held position(s) last saw "
                f"scored news >24h ago — story is aging. Monitor.")
    else:
        verdict = "OK"
        note = (f"All {n} held position(s) have scored live news inside "
                f"the last {WARM_H:.0f}h.")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "n_positions": n,
        "min_score_threshold": float(min_score_threshold),
        "positions": rows,
        "summary": summary,
        "verdict": verdict,
        "note": note,
        "thresholds_hours": {
            "fresh_le": FRESH_H,
            "warm_le": WARM_H,
            "cool_le": COOL_H,
        },
    }
