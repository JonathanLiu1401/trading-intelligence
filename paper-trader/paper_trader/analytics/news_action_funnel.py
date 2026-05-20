"""News→Action Funnel — per-ticker live responsiveness audit.

The operator's most common question on a hot news day: *"MU has 30 high-score
articles in the last 24h — did the bot do anything?"* The 117 endpoints
already grade adjacent halves of this:

* ``signal_followthrough`` grades the **forward edge** of acted-on signals vs
  ignored ones (needs 1d/3d/5d forward bars; cannot answer right now on the
  current window).
* ``idle_opportunity`` enumerates missed watchlist news but is **drought-gated**
  by design (silent when the bot is filling normally — which is the very state
  in which a per-ticker miss can still happen on a single hot name).
* ``trade_attribution`` enumerates news preceding each FILLED trade — the
  *reverse* direction (fills → news), not (news → fills).
* ``watchlist_opportunities`` ranks unheld tickers by news heat but does
  **not** report the desk's response.

None answer the per-ticker funnel **right now**, always-on (independent of
drought state), for held ∪ top-watchlist names: in the last N hours, how many
articles ≥ floor mentioned T, how many decisions named T, how many fills
landed on T, and what verdict does that combination yield?

The verdict ladder is operator-actionable:

* ``IGNORED`` — articles ≥ ``MIN_ARTICLES_FOR_VERDICT`` AND decisions = 0.
  The loud-news / no-action pathology — the desk is staring past its newswire.
  Surfaces FIRST in the ranked rows (the operator's primary signal).
* ``DECIDED_NO_FILL`` — articles ≥ threshold AND decisions ≥ 1 AND fills = 0.
  Distinguishes "Opus saw it, evaluated, and chose to HOLD" from the truly
  ignored case. Less alarming than ``IGNORED`` but still worth examining.
* ``ACTED_WITHOUT_NEWS`` — fills ≥ 1 AND articles < threshold. The bot moved
  on a name with no fresh corroborating headline — possibly a stale thesis
  cycle, possibly genuine technicals-only conviction.
* ``RESPONSIVE`` — articles ≥ threshold AND fills ≥ 1. The desk noticed and
  acted; the headline is information-rich. The operator-happy path.
* ``QUIET`` — articles < threshold AND fills = 0. Boring rows that exist for
  completeness so the per-ticker table is exhaustive (a held position with
  no news flow is still surfaced so the operator can see it).

Sort priority puts ``IGNORED`` first (the actionable miss), then
``DECIDED_NO_FILL``, then within each tier by article count DESC. The
``RESPONSIVE`` / ``ACTED_WITHOUT_NEWS`` / ``QUIET`` rows sort to the bottom
because they are operator-OK.

``build_news_action_funnel`` is pure: no DB, no network, never raises on
garbage inputs. The caller fetches the window articles (the
``news_velocity`` / ``idle_opportunity`` precedent — endpoint owns I/O,
builder is offline and testable). Word-boundary ticker regex matches the
same shape used by ``news_velocity`` / ``trade_attribution`` /
``idle_opportunity`` so MU does NOT alias MUTUAL, AMD does NOT alias AMDOCS,
and the ``$NVDA`` cashtag still hits. NaN / Inf ``ai_score`` rejected
(digital-intern's column has been observed with stale NaNs from a half-trained
model; a NaN comparison would silently drop the row).

Observational only — never gates Opus, never injected into the decision
prompt, no caps (AGENTS.md invariants #2/#12 — the
``idle_opportunity`` / ``news_velocity`` / ``signal_followthrough``
precedent).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

# Articles ≥ this score in the window count as a "loud" headline. Same floor
# as ``idle_opportunity.DEFAULT_MIN_AI_SCORE`` — above the daemon's grey-zone
# threshold and below the urgency cutoff of 8.0.
DEFAULT_MIN_AI_SCORE = 6.0

# Window the funnel evaluates; matches the trader's headline horizon for "did
# you do anything today on this name".
DEFAULT_WINDOW_HOURS = 24.0

# Article count threshold for the verdict ladder. Below this a row is
# QUIET / ACTED_WITHOUT_NEWS regardless of decisions/fills — a single article
# isn't enough to call the desk ``IGNORED`` (one stale wire mention is not a
# regret-list event; matches the ``news_source_mix.ECHO_MIN_ARTICLES``
# sample-size honesty precedent).
MIN_ARTICLES_FOR_VERDICT = 3

# Cap rows so the panel stays readable even when the watchlist is wide.
DEFAULT_MAX_TICKERS = 25


def _parse_ts(ts) -> datetime | None:
    """Tolerate aware/naive ISO strings + datetime objects — matches the
    ``decision_drought._parse_ts`` / ``idle_opportunity._parse_ts``
    convention."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).strip().replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _ticker_regex(ticker: str) -> re.Pattern:
    """``$TKR`` cashtag OR word-boundary ``TKR`` — same shape as
    ``news_velocity._ticker_regex`` / ``trade_attribution`` /
    ``idle_opportunity._ticker_regex``. Case-insensitive (against an
    upper-cased haystack) so ``MUTUAL`` does NOT alias ``MU``, ``AMDOCS``
    does NOT alias ``AMD``, while ``$NVDA`` cashtag still hits."""
    return re.compile(rf"(?:\$|\b){re.escape(ticker.upper())}\b")


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    # Reject NaN / +/-Inf — see module docstring.
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _parse_action_ticker(action_taken: str) -> tuple[str, str | None]:
    """Pull (verb, ticker) out of a free-text ``decisions.action_taken``.

    Verbatim copy of ``dashboard._parse_action_ticker`` (AGENTS.md #10 — SSOT
    pinned by the test suite). NO_DECISION / BLOCKED / CASH / NONE all
    return ticker=None so the funnel doesn't bucket sentinel rows under a
    real ticker."""
    if not action_taken or action_taken in ("NO_DECISION", "BLOCKED"):
        return action_taken or "", None
    head = action_taken.split("→")[0].strip()
    parts = head.split()
    if not parts:
        return "", None
    verb = parts[0].upper()
    ticker = parts[1].upper() if len(parts) >= 2 else None
    if ticker in ("CASH", "NONE", ""):
        ticker = None
    return verb, ticker


def _verdict(n_articles: int, n_decisions: int, n_fills: int) -> str:
    """Operator-actionable label per ticker. Ordered so the worst case is
    checked first (IGNORED → loudly missed)."""
    loud = n_articles >= MIN_ARTICLES_FOR_VERDICT
    if loud and n_decisions == 0:
        return "IGNORED"
    if loud and n_decisions >= 1 and n_fills == 0:
        return "DECIDED_NO_FILL"
    if loud and n_fills >= 1:
        return "RESPONSIVE"
    if not loud and n_fills >= 1:
        return "ACTED_WITHOUT_NEWS"
    return "QUIET"


# Sort tiers: lower index = higher priority on the panel (IGNORED first).
_VERDICT_PRIORITY = {
    "IGNORED": 0,
    "DECIDED_NO_FILL": 1,
    "ACTED_WITHOUT_NEWS": 2,
    "RESPONSIVE": 3,
    "QUIET": 4,
}


def _headline(state: str, rows: list[dict], window_h: float,
              min_score: float) -> str:
    if state == "NO_DATA":
        return "News→action funnel: no tickers to evaluate."
    # rows are pre-sorted; worst-IGNORED case (if any) is rows[0].
    ignored = [r for r in rows if r["verdict"] == "IGNORED"]
    if ignored:
        top = ignored[0]
        held_tag = " (HELD)" if top.get("held") else ""
        return (
            f"News→action funnel: {len(ignored)} IGNORED ticker(s) over {window_h:.0f}h "
            f"≥{min_score:.1f} — loudest: {top['ticker']}{held_tag} "
            f"@ {top['n_articles']} article(s), 0 decisions."
        )
    decided_no_fill = [r for r in rows if r["verdict"] == "DECIDED_NO_FILL"]
    if decided_no_fill:
        top = decided_no_fill[0]
        held_tag = " (HELD)" if top.get("held") else ""
        return (
            f"News→action funnel: {len(decided_no_fill)} ticker(s) decided "
            f"but not filled — top: {top['ticker']}{held_tag} "
            f"@ {top['n_articles']} article(s), {top['n_decisions']} decision(s)."
        )
    responsive = [r for r in rows if r["verdict"] == "RESPONSIVE"]
    if responsive:
        return (
            f"News→action funnel: {len(responsive)} ticker(s) responsive "
            f"(news + decision + fill); no IGNORED loud rows."
        )
    return (
        f"News→action funnel: no IGNORED/DECIDED_NO_FILL rows "
        f"over {window_h:.0f}h ≥{min_score:.1f} — quiet window."
    )


def build_news_action_funnel(
    articles: list[dict] | None,
    decisions: list[dict] | None,
    trades: list[dict] | None,
    positions: list[dict] | None,
    tickers: list[str] | None,
    held_tickers: list[str] | None = None,
    now: datetime | None = None,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    min_ai_score: float = DEFAULT_MIN_AI_SCORE,
    max_tickers: int = DEFAULT_MAX_TICKERS,
) -> dict:
    """Per-ticker funnel of (loud article count → decision count → fill count
    → current unrealized P&L) over the last ``window_hours``.

    Pure. ``articles`` is a list of dicts with at minimum ``title`` /
    ``ai_score`` / ``first_seen`` (and optionally ``body`` / ``source`` /
    ``url``). ``decisions`` are ``store.recent_decisions()`` rows; ``trades``
    are ``store.recent_trades()`` rows; ``positions`` are
    ``store.open_positions()`` rows (used only to attach current
    ``unrealized_pl`` per held ticker). ``tickers`` is the universe to bucket
    against — typically ``held ∪ top-N watchlist``; the caller decides.

    Returns a JSON-ready dict — never raises on garbage inputs.
    """
    now = now or datetime.now(timezone.utc)
    window_h = float(window_hours) if window_hours and window_hours > 0 else DEFAULT_WINDOW_HOURS
    score_floor = float(min_ai_score) if min_ai_score is not None else DEFAULT_MIN_AI_SCORE
    max_tk = int(max_tickers) if max_tickers and max_tickers > 0 else DEFAULT_MAX_TICKERS

    cutoff = now - timedelta(hours=window_h)

    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "NO_DATA",
        "headline": _headline("NO_DATA", [], window_h, score_floor),
        "window_hours": window_h,
        "min_ai_score": score_floor,
        "min_articles_for_verdict": MIN_ARTICLES_FOR_VERDICT,
        "n_tickers": 0,
        "n_ignored": 0,
        "n_decided_no_fill": 0,
        "n_responsive": 0,
        "n_acted_without_news": 0,
        "n_quiet": 0,
        "tickers": [],
    }

    universe = [t.upper() for t in (tickers or [])
                if t and isinstance(t, str)]
    # Deduplicate while preserving order so the panel is stable.
    seen: set[str] = set()
    universe = [t for t in universe if not (t in seen or seen.add(t))]
    if not universe:
        return out

    held_set = {t.upper() for t in (held_tickers or [])
                if t and isinstance(t, str)}

    # ── per-ticker unrealized P&L for held names (so the operator sees the
    # current cost of an IGNORED row). Only stock lots; an option's P&L is
    # already baked into ``unrealized_pl`` so we read it directly. ──────
    pl_by_ticker: dict[str, float] = {}
    for p in positions or []:
        if not isinstance(p, dict):
            continue
        tk = (p.get("ticker") or "")
        if not isinstance(tk, str) or not tk:
            continue
        tk = tk.upper()
        pl = p.get("unrealized_pl")
        try:
            f = float(pl) if pl is not None else 0.0
        except (TypeError, ValueError):
            continue
        pl_by_ticker[tk] = pl_by_ticker.get(tk, 0.0) + f

    # ── article counting per ticker ─────────────────────────────────
    patterns = {t: _ticker_regex(t) for t in universe}
    per_ticker_articles: dict[str, dict] = {
        t: {"n_articles": 0, "top_score": None,
            "top_title": None, "top_first_seen": None,
            "top_source": None, "top_url": None}
        for t in universe
    }

    for a in articles or []:
        if not isinstance(a, dict):
            continue
        score = _safe_float(a.get("ai_score"))
        if score is None or score < score_floor:
            continue
        fs = _parse_ts(a.get("first_seen"))
        if fs is None or fs < cutoff:
            continue
        title = a.get("title") or ""
        body = a.get("body") or ""
        haystack = (title + " " + body).upper()
        for tk, pat in patterns.items():
            if not pat.search(haystack):
                continue
            slot = per_ticker_articles[tk]
            slot["n_articles"] += 1
            replace = slot["top_score"] is None or score > slot["top_score"]
            if not replace and slot["top_score"] is not None and score == slot["top_score"]:
                # Tie-break: newer first_seen wins (more plausibly causal —
                # the trade_attribution / idle_opportunity tie-break).
                prev_fs = _parse_ts(slot["top_first_seen"])
                if prev_fs is None or fs > prev_fs:
                    replace = True
            if replace:
                slot["top_score"] = score
                slot["top_title"] = title[:240]
                slot["top_first_seen"] = fs.isoformat(timespec="seconds")
                slot["top_source"] = (a.get("source") or "")[:120] or None
                slot["top_url"] = (a.get("url") or "")[:240] or None

    # ── decision counting per ticker (in-window, parsed ticker) ──────
    per_ticker_decisions: dict[str, int] = {t: 0 for t in universe}
    for d in decisions or []:
        if not isinstance(d, dict):
            continue
        ts = _parse_ts(d.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        verb, tk = _parse_action_ticker(d.get("action_taken") or "")
        if not tk:
            continue
        if tk in per_ticker_decisions:
            per_ticker_decisions[tk] += 1

    # ── fill counting per ticker (in-window) ───────────────────────
    # The paper_trader.db ``trades`` schema has no ``status`` column —
    # only executed fills land in the table. So when ``status`` is absent
    # (the live shape) every row is a fill; when it is present (test
    # fixtures or a future schema bump) we still honor it and skip
    # explicit non-FILLED rows.
    per_ticker_fills: dict[str, int] = {t: 0 for t in universe}
    for tr in trades or []:
        if not isinstance(tr, dict):
            continue
        status_raw = tr.get("status")
        if status_raw is not None and str(status_raw).upper() != "FILLED":
            continue
        ts = _parse_ts(tr.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        tk = (tr.get("ticker") or "")
        if not isinstance(tk, str) or not tk:
            continue
        tk = tk.upper()
        if tk in per_ticker_fills:
            per_ticker_fills[tk] += 1

    # ── assemble rows ───────────────────────────────────────────────
    rows: list[dict] = []
    for tk in universe:
        a_slot = per_ticker_articles[tk]
        n_articles = a_slot["n_articles"]
        n_decisions = per_ticker_decisions[tk]
        n_fills = per_ticker_fills[tk]
        verdict = _verdict(n_articles, n_decisions, n_fills)
        rows.append({
            "ticker": tk,
            "held": tk in held_set,
            "n_articles": n_articles,
            "n_decisions": n_decisions,
            "n_fills": n_fills,
            "verdict": verdict,
            "top_score": a_slot["top_score"],
            "top_title": a_slot["top_title"],
            "top_first_seen": a_slot["top_first_seen"],
            "top_source": a_slot["top_source"],
            "top_url": a_slot["top_url"],
            "unrealized_pl": round(pl_by_ticker.get(tk, 0.0), 4)
                if tk in pl_by_ticker else None,
        })

    # Sort by verdict priority (IGNORED first), then within each tier by
    # n_articles DESC then ticker ASC for deterministic ordering.
    rows.sort(key=lambda r: (
        _VERDICT_PRIORITY.get(r["verdict"], 99),
        -int(r["n_articles"]),
        -int(r["n_decisions"]),
        r["ticker"],
    ))
    rows = rows[:max_tk]

    counts = {
        "n_ignored": sum(1 for r in rows if r["verdict"] == "IGNORED"),
        "n_decided_no_fill": sum(1 for r in rows if r["verdict"] == "DECIDED_NO_FILL"),
        "n_responsive": sum(1 for r in rows if r["verdict"] == "RESPONSIVE"),
        "n_acted_without_news": sum(1 for r in rows if r["verdict"] == "ACTED_WITHOUT_NEWS"),
        "n_quiet": sum(1 for r in rows if r["verdict"] == "QUIET"),
    }
    out.update(counts)
    out["state"] = "OK"
    out["n_tickers"] = len(rows)
    out["tickers"] = rows
    out["headline"] = _headline("OK", rows, window_h, score_floor)
    return out
