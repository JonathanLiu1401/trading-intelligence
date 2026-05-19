"""Per-held-ticker news-flow velocity — is the catalyst BUILDING or FADING?

Existing surfaces answer adjacent questions but not this one:

* ``analytics/position_thesis.py`` reports the latest 24h headlines + bull/bear
  split for each held position — a single-window snapshot.
* ``digital-intern/analytics/trend_velocity.py`` reports market-wide tickers
  gaining mentions in the last 2h vs the prior 2h — a short-horizon, *not*
  held-book-keyed surface.
* ``digital-intern/analytics/breaking_news_detector.py`` flags 3+ articles per
  ticker in a 5-min window — burst detection, not thesis evolution.

``build_news_velocity`` fills the remaining gap: for every CURRENTLY-HELD
ticker, compare the article rate over the last ``window_hours`` (default 24h)
to a non-overlapping ``baseline_hours`` (default 168h = prior 6 days) baseline,
emit a Poisson-style z-score and a state verdict (SURGING / STABLE / FADING /
INSUFFICIENT / NO_DATA). The single question it answers — *"is the news flow
on a position I actually own getting LOUDER or QUIETER?"* — is the trader's
real "should I reassess the thesis" trigger.

Sample-size honesty mirrors ``build_tail_risk`` / ``build_correlation``:
numerics are emitted whenever defined, but the per-ticker **verdict** is
withheld (``state="INSUFFICIENT"``) until ``MIN_BASELINE_N`` baseline articles
exist for that ticker. The articles.db history is observably shallow
(``project_articles_db_shallow_history`` memory: ~days deep, not 90), so
INSUFFICIENT is the common path on a new/quiet name and we treat that as
honest reporting, not a bug.

Observational only — never gates Opus, no caps (AGENTS.md #2 / #12). The
builder is pure: it takes pre-fetched article rows + held tickers + ``now``
and returns a JSON-ready dict. No DB, no network, never raises on garbage
inputs.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

# Per-ticker thresholds. INSUFFICIENT below MIN_BASELINE_N baseline articles —
# the articles.db history is shallow so a clean "needs more samples" verdict
# is the common honest answer for unknown names. SURGING requires both a high
# z-score AND a non-trivial absolute window count so a baseline of 1 article
# doesn't trip a "+inf z" sentinel.
MIN_BASELINE_N = 5
MIN_WINDOW_FOR_SURGE = 3
Z_SURGE = 2.0
Z_FADE = -1.0


def _parse_ts(ts) -> datetime | None:
    """Tolerate aware/naive ISO strings; the dashboard rows can be either."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _ticker_regex(ticker: str) -> re.Pattern:
    """``$TKR`` cashtag OR word-boundary ``TKR`` — same regex shape the rest
    of the analytics surface uses (signals.ticker_sentiments,
    dashboard._ticker_news_pulse). Case-insensitive against the upper-cased
    article body so ``AMDOCS`` does NOT alias ``AMD`` (``\\b`` boundary)."""
    return re.compile(rf"(?:\$|\b){re.escape(ticker.upper())}\b")


def _classify(z: float | None, window_count: int, baseline_count: int) -> str:
    if baseline_count < MIN_BASELINE_N:
        return "INSUFFICIENT"
    if z is None:
        return "STABLE"
    if z >= Z_SURGE and window_count >= MIN_WINDOW_FOR_SURGE:
        return "SURGING"
    if z <= Z_FADE or window_count == 0:
        return "FADING"
    return "STABLE"


def build_news_velocity(
    articles: list[dict],
    held_tickers: list[str],
    now: datetime | None = None,
    window_hours: float = 24.0,
    baseline_hours: float = 168.0,
) -> dict:
    """Per-held-ticker news velocity.

    Inputs
    ------
    articles : list[dict]
        Pre-fetched, live-only article rows spanning at least the last
        ``baseline_hours``. Each dict needs ``title`` (str), ``first_seen``
        (ISO str), and optionally ``ai_score`` (float), ``urgency`` (int),
        and ``body`` (str — full text already decoded; title-only is fine,
        but the regex will scan body too if provided).
    held_tickers : list[str]
        Tickers to bucket on. Stock-side only (option tickers reuse the
        underlying symbol). Duplicates de-duped case-insensitively.
    window_hours, baseline_hours : float
        Window vs baseline span. ``baseline_hours`` MUST exceed
        ``window_hours``; otherwise the baseline collapses and we fall
        through to NO_DATA for safety.
    now : datetime, optional
        Injectable for tests. Defaults to ``datetime.now(timezone.utc)``.

    Output
    ------
    JSON-ready dict. ``per_ticker`` is sorted by descending ``z_score``
    (None last) so the loudest catalyst surfaces first.
    """
    now = (now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "window_hours": window_hours,
        "baseline_hours": baseline_hours,
        "min_baseline_n": MIN_BASELINE_N,
        "n_held": 0,
        "n_with_data": 0,
        "per_ticker": [],
        "state": "NO_DATA",
        "headline": "News velocity: no held positions.",
    }

    if not held_tickers or baseline_hours <= window_hours:
        return base

    # De-dup case-insensitively, preserve first-seen order.
    seen: set[str] = set()
    tickers: list[str] = []
    for t in held_tickers:
        if not t:
            continue
        u = str(t).upper()
        if u in seen:
            continue
        seen.add(u)
        tickers.append(u)
    base["n_held"] = len(tickers)

    if not tickers:
        return base

    patterns = {t: _ticker_regex(t) for t in tickers}
    window_cutoff = now.timestamp() - window_hours * 3600
    baseline_cutoff = now.timestamp() - baseline_hours * 3600

    # Per-ticker accumulators.
    win_n: dict[str, int] = {t: 0 for t in tickers}
    base_n: dict[str, int] = {t: 0 for t in tickers}
    win_max_urg: dict[str, int] = {t: 0 for t in tickers}
    win_max_score: dict[str, float] = {t: 0.0 for t in tickers}
    win_top_title: dict[str, str | None] = {t: None for t in tickers}

    for a in (articles or []):
        if not isinstance(a, dict):
            continue
        ts = _parse_ts(a.get("first_seen"))
        if ts is None:
            continue
        secs = ts.timestamp()
        if secs < baseline_cutoff:
            continue
        title = str(a.get("title") or "")
        body = title + " " + str(a.get("body") or "")
        body_up = body.upper()
        ai = a.get("ai_score")
        try:
            ai_f = float(ai) if ai is not None else 0.0
        except Exception:
            ai_f = 0.0
        urg = a.get("urgency")
        try:
            urg_i = int(urg) if urg is not None else 0
        except Exception:
            urg_i = 0
        in_window = secs >= window_cutoff
        for t, pat in patterns.items():
            if not pat.search(body_up):
                continue
            if in_window:
                win_n[t] += 1
                if urg_i > win_max_urg[t]:
                    win_max_urg[t] = urg_i
                if ai_f > win_max_score[t]:
                    win_max_score[t] = ai_f
                    win_top_title[t] = title or None
            else:
                base_n[t] += 1

    # Build per-ticker rows.
    baseline_span_h = max(baseline_hours - window_hours, 1e-9)
    rows: list[dict] = []
    n_with = 0
    n_surging = 0
    n_fading = 0
    n_insufficient = 0

    for t in tickers:
        wn = win_n[t]
        bn = base_n[t]
        win_rate = wn / window_hours if window_hours > 0 else 0.0
        base_rate = bn / baseline_span_h
        expected = base_rate * window_hours
        # Poisson z. Floor sqrt(expected) at 1.0 so a tiny baseline doesn't
        # explode the score and a window count of 5 vs expected 0.5 reads as
        # "high" not "infinite".
        z: float | None
        ratio: float | None
        if bn == 0 and wn == 0:
            z = None
            ratio = None
        elif expected <= 0:
            z = None
            ratio = None
        else:
            z = (wn - expected) / max(math.sqrt(expected), 1.0)
            ratio = (win_rate / base_rate) if base_rate > 0 else None

        state = _classify(z, wn, bn)
        if state == "SURGING":
            n_surging += 1
        elif state == "FADING":
            n_fading += 1
        elif state == "INSUFFICIENT":
            n_insufficient += 1
        if wn > 0 or bn > 0:
            n_with += 1

        rows.append({
            "ticker": t,
            "state": state,
            "window_count": wn,
            "baseline_count": bn,
            "window_rate_per_h": round(win_rate, 3),
            "baseline_rate_per_h": round(base_rate, 3),
            "expected_window_count": round(expected, 2),
            "z_score": round(z, 2) if z is not None else None,
            "ratio": round(ratio, 2) if ratio is not None else None,
            "max_urgency_window": win_max_urg[t] if wn > 0 else None,
            "max_ai_score_window": round(win_max_score[t], 2) if wn > 0 else None,
            "top_window_title": win_top_title[t],
        })

    # Sort: SURGING with highest z first, then STABLE/FADING by z desc, then
    # INSUFFICIENT last. Use (rank, -z) where rank reflects state priority.
    state_rank = {"SURGING": 0, "STABLE": 1, "FADING": 1, "INSUFFICIENT": 2}
    rows.sort(key=lambda r: (
        state_rank.get(r["state"], 3),
        -(r["z_score"] if r["z_score"] is not None else -1e9),
    ))

    base["n_with_data"] = n_with
    base["per_ticker"] = rows

    if n_with == 0:
        base["state"] = "NO_DATA"
        base["headline"] = (
            f"News velocity: 0 articles matched any of "
            f"{len(tickers)} held name(s) in last {baseline_hours:.0f}h."
        )
        return base

    base["state"] = "OK"
    parts: list[str] = []
    for r in rows:
        if r["state"] == "SURGING":
            parts.append(f"{r['ticker']} SURGING (z={r['z_score']})")
        elif r["state"] == "FADING":
            parts.append(f"{r['ticker']} FADING")
    if n_insufficient and not parts:
        parts.append(f"{n_insufficient} INSUFFICIENT")
    if not parts:
        parts.append("all STABLE")
    base["headline"] = (
        f"News velocity ({window_hours:.0f}h vs {baseline_hours:.0f}h baseline): "
        + ", ".join(parts) + "."
    )
    return base
