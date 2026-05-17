"""Signal Follow-Through — is the trader actually *using* its own news edge?

The stack spends most of its compute generating scored news signals, then a
hedge-fund-style Opus prompt that sees them. Two endpoints already grade
*halves* of this:

* ``news_edge`` grades the **signal in isolation** — does a high ``ai_score``
  headline predict the move? — *ignoring whether the bot ever acted on it*.
* ``decision_drought`` grades **inaction cost vs SPY** over idle windows —
  *not vs the specific signals that were on the screen*.

Nothing grades the *join*: of the high-score signals that were **visible to
the trader at decision time** (an article whose ``first_seen`` fell in the
``lookback_hours`` window ending at a decision's timestamp — the exact window
``strategy.decide()`` feeds Opus via ``get_top_signals(hours=2,
min_score=4.0)``), did the trader transact that ticker that cycle, and did the
signals it **ACTED** on outperform — forward, SPY-abnormal — the ones it
**IGNORED**?

That is the trader's own question: *"am I using my intelligence, or staring
past it?"* A near-zero follow-through rate says the desk ignores its
newswire; a negative ``selection_edge`` says it acts on the duds and sits on
the winners (anti-selection).

``build_signal_followthrough`` is pure and deterministic — the caller does the
I/O (``_fetch_live_articles`` runs the canonical live-only SQL; the dashboard
hands in daily bars). Ticker resolution, calendar-day mapping and the
at-or-after bar lookup are **imported from ``news_edge``** so the two panels
can never disagree on which article belongs to which name (single source of
truth, invariant #10 spirit). Advisory only — it informs, never gates Opus and
adds no caps (invariants #2/#12).
"""
from __future__ import annotations

import re
import sqlite3
import zlib
from datetime import datetime, timedelta, timezone

from .news_edge import _index_at_or_after, _parse_date, _resolve_ticker

DEFAULT_HORIZONS = (1, 3, 5)
# A signal feed must surface at least this many resolvable signals (with a
# forward window) before any verdict label is allowed — mirrors news_edge's
# `_MIN_BAND_N` / decision_reliability's `MIN_CURRENT` sample-size honesty.
_MIN_RESOLVED = 12
# And the ACTED subset needs at least this many forward-resolved samples
# before a *selection* verdict (EXPLOITING/MISUSING) can be claimed — fewer
# and we can only report follow-through, not skill.
_MIN_ACTED = 8
# Below this follow-through %, the desk is effectively ignoring its newswire.
_IGNORE_THRESHOLD_PCT = 5.0
# Abnormal-return gap (pct) the acted vs ignored means must clear before we
# call it real selection skill rather than noise.
_EDGE_EPS = 0.25

_LIVE_ONLY_SQL = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)


def _fetch_live_articles(db_path: str, since_iso: str,
                         min_score: float = 4.0) -> list[dict]:
    """Live (non-backtest) scored articles since ``since_iso``.

    The canonical live-only clause is inlined verbatim (invariant #1 / the
    ``signals.py`` mirror): a ``backtest://`` URL, a ``backtest_*`` source, or
    an ``opus_annotation*`` source must never be scored as a real signal the
    trader saw. Returns the ``build_signal_followthrough`` article shape."""
    out: list[dict] = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=8)
    except Exception:
        return out
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT title, full_text, ai_score, urgency, first_seen "
            "FROM articles WHERE ai_score >= ? AND first_seen >= ? "
            f"AND {_LIVE_ONLY_SQL} "
            "ORDER BY first_seen DESC LIMIT 6000",
            (min_score, since_iso),
        ).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    for r in rows:
        body = ""
        try:
            if r["full_text"]:
                body = zlib.decompress(r["full_text"]).decode(
                    "utf-8", errors="replace")
        except Exception:
            body = ""
        out.append({
            "text": f"{r['title'] or ''} {body}".strip(),
            "ai_score": r["ai_score"],
            "urgency": r["urgency"],
            "first_seen": r["first_seen"],
        })
    return out


def _to_dt(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_action(action_taken: str | None) -> tuple[bool, str | None]:
    """``(was_filled, ticker)`` from a decisions.action_taken string.

    Mirrors ``dashboard._parse_action_ticker`` + ``decision_drought._classify``
    conventions (the column is free-text ``"BUY NVDA → FILLED"`` /
    ``"HOLD MU → HOLD"`` / bare ``"NO_DECISION"``). Only a FILLED transaction
    counts as the trader *acting on* that ticker's signal; the ticker is
    nulled for CASH/NONE pseudo-tickers so a ``HOLD CASH`` never reads as
    acting on a name."""
    raw = (action_taken or "").strip()
    if not raw or raw in ("NO_DECISION", "BLOCKED"):
        return False, None
    filled = "FILLED" in raw.upper()
    head = raw.split("→")[0].strip()
    parts = head.split()
    ticker = parts[1].upper() if len(parts) >= 2 else None
    if ticker in ("CASH", "NONE", ""):
        ticker = None
    return filled, ticker


def _blank(horizons: tuple[int, ...]) -> dict:
    return {h: {"n": 0, "raw_sum": 0.0, "raw_up": 0,
                "abn_n": 0, "abn_sum": 0.0, "abn_up": 0} for h in horizons}


def _finalize(acc: dict, horizons: tuple[int, ...]) -> dict:
    rows = {}
    for h in horizons:
        c = acc[h]
        n, an = c["n"], c["abn_n"]
        rows[str(h)] = {
            "n": n,
            "mean_raw_pct": round(c["raw_sum"] / n, 3) if n else None,
            "raw_up_rate": round(c["raw_up"] / n * 100, 1) if n else None,
            "n_abnormal": an,
            "mean_abnormal_pct": round(c["abn_sum"] / an, 3) if an else None,
            "abnormal_hit_rate": round(c["abn_up"] / an * 100, 1) if an else None,
        }
    return rows


def build_signal_followthrough(
    decisions: list[dict],
    articles: list[dict],
    price_history: dict[str, list[tuple[str, float]]],
    spy_history: list[tuple[str, float]],
    tickers: list[str],
    now: datetime | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    lookback_hours: float = 2.0,
    min_score: float = 4.0,
) -> dict:
    """Did the trader act on the news it saw, and did acting pay?

    Args:
      decisions: store-native rows (any order) — uses ``timestamp`` +
        ``action_taken``.
      articles: caller-prefiltered live scored articles spanning the decision
        window — ``[{text, ai_score, urgency, first_seen}]`` (use
        ``_fetch_live_articles``).
      price_history: ``{TICKER: [(YYYY-MM-DD, close), ...]}`` ascending.
      spy_history: ``[(YYYY-MM-DD, close), ...]`` ascending; ``[]`` ⇒ raw only.
      tickers: resolution universe (the live WATCHLIST), tried in order.
    """
    now = now or datetime.now(timezone.utc)
    horizons = tuple(sorted(set(int(h) for h in horizons if h > 0)))
    win = timedelta(hours=lookback_hours)

    ph: dict[str, tuple[list[str], list[float]]] = {}
    for tk, bars in (price_history or {}).items():
        s = sorted(bars, key=lambda b: b[0])
        ph[tk.upper()] = ([d for d, _ in s], [float(c) for _, c in s])
    spy_by_date = {d: float(c) for d, c in (spy_history or [])}

    # Index articles by parsed first_seen once.
    parsed_arts: list[tuple[datetime, dict]] = []
    for a in articles or []:
        ts = _to_dt(a.get("first_seen") or a.get("published"))
        if ts is not None:
            parsed_arts.append((ts, a))

    acted = _blank(horizons)
    ignored = _blank(horizons)
    rx_cache: dict[str, re.Pattern] = {}
    n_signals = n_acted = n_ignored = n_resolved = n_acted_resolved = 0
    n_decisions = 0

    for d in decisions or []:
        dt = _to_dt(d.get("timestamp"))
        if dt is None:
            continue
        n_decisions += 1
        lo = dt - win
        filled, filled_tk = _parse_action(d.get("action_taken"))
        day = _parse_date(d.get("timestamp"))

        # Resolve every high-score signal visible in this cycle's window,
        # deduped to one per ticker (max score/urgency) — the unit of
        # analysis is "did it act on the NVDA news this cycle", not spam.
        per_ticker: dict[str, dict] = {}
        for ts, a in parsed_arts:
            if ts < lo or ts > dt:
                continue
            try:
                score = float(a.get("ai_score") or 0.0)
            except (TypeError, ValueError):
                continue
            if score < min_score:
                continue
            tk = _resolve_ticker(str(a.get("text") or a.get("title") or ""),
                                 tickers, rx_cache)
            if not tk:
                continue
            cur = per_ticker.get(tk)
            if cur is None or score > cur["score"]:
                per_ticker[tk] = {"score": score,
                                  "urgency": int(a.get("urgency") or 0)}

        for tk in per_ticker:
            n_signals += 1
            is_acted = filled and filled_tk == tk.upper()
            if is_acted:
                n_acted += 1
            else:
                n_ignored += 1
            bucket = acted if is_acted else ignored

            if day is None or tk.upper() not in ph:
                continue
            dates, closes = ph[tk.upper()]
            i0 = _index_at_or_after(dates, day)
            if i0 is None:
                continue
            entry = closes[i0]
            if entry <= 0:
                continue
            entry_date = dates[i0]
            spy_entry = spy_by_date.get(entry_date)
            resolved_any = False
            for h in horizons:
                j = i0 + h
                if j >= len(dates):
                    continue
                fwd = closes[j]
                if fwd <= 0:
                    continue
                raw = (fwd / entry - 1.0) * 100.0
                cell = bucket[h]
                cell["n"] += 1
                cell["raw_sum"] += raw
                if raw > 0:
                    cell["raw_up"] += 1
                spy_fwd = spy_by_date.get(dates[j])
                if spy_entry and spy_fwd and spy_entry > 0:
                    abn = raw - (spy_fwd / spy_entry - 1.0) * 100.0
                    cell["abn_n"] += 1
                    cell["abn_sum"] += abn
                    if abn > 0:
                        cell["abn_up"] += 1
                resolved_any = True
            if resolved_any:
                n_resolved += 1
                if is_acted:
                    n_acted_resolved += 1

    acted_rows = _finalize(acted, horizons)
    ignored_rows = _finalize(ignored, horizons)

    # Adaptive reference horizon: the longest horizon whose ACTED bucket is
    # well-sampled (a 5d selection edge is the strongest claim); fall back to
    # the longest with any acted data, then any data at all, then 3 / middle.
    # Matures with history exactly like news_edge.
    ref = None
    if horizons:
        well = [h for h in horizons if acted[h]["abn_n"] >= _MIN_ACTED]
        some_a = [h for h in horizons if acted[h]["abn_n"] > 0]
        any_d = [h for h in horizons
                 if acted[h]["n"] > 0 or ignored[h]["n"] > 0]
        if well:
            ref = max(well)
        elif some_a:
            ref = max(some_a)
        elif any_d:
            ref = max(any_d)
        else:
            ref = 3 if 3 in horizons else horizons[len(horizons) // 2]

    def _abn_at(rows: dict) -> float | None:
        cell = rows.get(str(ref), {}) if ref is not None else {}
        return cell.get("mean_abnormal_pct")

    acted_abn = _abn_at(acted_rows)
    ignored_abn = _abn_at(ignored_rows)
    if acted_abn is not None:
        sel_edge = round(acted_abn - (ignored_abn or 0.0), 3)
    else:
        sel_edge = None
    follow_through = (round(n_acted / n_signals * 100, 1)
                      if n_signals else 0.0)

    verdict, reason = _judge(
        n_signals, n_resolved, n_acted_resolved, follow_through,
        acted_abn, ignored_abn, sel_edge, ref)

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "n_decisions": n_decisions,
        "n_signals": n_signals,
        "n_acted": n_acted,
        "n_ignored": n_ignored,
        "n_resolved": n_resolved,
        "follow_through_rate_pct": follow_through,
        "horizons": list(horizons),
        "reference_horizon": ref,
        "spy_adjusted": bool(spy_by_date),
        "selection_edge_pct": sel_edge,
        "acted": acted_rows,
        "ignored": ignored_rows,
        "verdict": verdict,
        "verdict_reason": reason,
    }


def _judge(n_signals: int, n_resolved: int, n_acted_resolved: int,
           follow_through: float, acted_abn: float | None,
           ignored_abn: float | None, sel_edge: float | None,
           ref: int | None) -> tuple[str, str]:
    """Verdict precedence (sample-size honest, like news_edge/trade_asymmetry).

    NO_DATA → INSUFFICIENT → IGNORING_FEED → (need acted samples) →
    MISUSING_SIGNALS / EXPLOITING_SIGNALS / NEUTRAL_USE."""
    if n_signals == 0:
        return ("NO_DATA",
                "no high-score signal was visible at any recorded decision "
                "(empty decision log, no priced/resolvable news, or the feed "
                "named no watchlist ticker)")
    if n_resolved < _MIN_RESOLVED:
        return ("INSUFFICIENT",
                f"only {n_resolved} forward-resolved signals "
                f"(need {_MIN_RESOLVED}) — accumulate more decision + news "
                "history before grading signal use")
    if follow_through < _IGNORE_THRESHOLD_PCT:
        return ("IGNORING_FEED",
                f"acted on {follow_through:.1f}% of {n_signals} high-score "
                f"signals it saw — the desk is effectively ignoring its own "
                "newswire (consistent with a HOLD-dominated book)")
    if n_acted_resolved < _MIN_ACTED or acted_abn is None or ref is None:
        return ("LOW_ACTIVITY",
                f"acted on {n_acted_resolved} forward-resolved signals "
                f"(need {_MIN_ACTED}) — follow-through is "
                f"{follow_through:.1f}% but too few acted-on signals to grade "
                "selection skill yet")
    base = (f"acted-on signals lead {acted_abn:+.2f}% abnormal at {ref}d vs "
            f"{(ignored_abn or 0.0):+.2f}% on the ignored ones "
            f"(edge {sel_edge:+.2f} pp, follow-through {follow_through:.1f}%)")
    if sel_edge is not None and sel_edge < -_EDGE_EPS:
        return ("MISUSING_SIGNALS",
                base + " — anti-selection: it acts on the duds and sits on "
                "the winners")
    if (sel_edge is not None and sel_edge > _EDGE_EPS
            and acted_abn is not None and acted_abn > 0):
        return ("EXPLOITING_SIGNALS",
                base + " — the trader picks the signals that pay")
    return ("NEUTRAL_USE",
            base + " — it acts on its feed but with no measurable selection "
            "edge over what it ignored")
