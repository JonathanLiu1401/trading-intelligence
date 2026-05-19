"""Idle opportunity cost — what high-score live signals on the watchlist
arrived *while the bot was paralyzed*?

The recurring 2026-05 live pathology has been the PARALYSIS / NO_DECISION
storm: host saturated by out-of-band Opus, the live trader's claude call
times out, and ``decisions.action_taken`` records ``NO_DECISION`` for hours
on end. Existing surfaces describe the drought from different angles:

* ``analytics/decision_drought.py`` — WHEN the drought happened, how long it
  ran, and the realized portfolio-vs-SPY drift (a backward-looking P&L cost).
* ``analytics/host_guard.py`` + ``analytics/decision_forensics.py`` — WHY the
  drought is happening (host load / dominant parse-failure mode).
* ``analytics/shadow_vs_claude.py`` — RIGHT-NOW what the deterministic
  shadow rules engine would recommend, vs the last (possibly stale) claude
  decision. **Snapshot-only by design** — does not look across the drought
  window.
* ``analytics/funded_suggestions.py`` / ``analytics/watchlist_opportunities``
  — current top BUY/ADD/TRIM cards from market state.

None answer the operator's question that this builder targets:

  **"While the bot was dark for 7.94h, did anything HIGH-SCORE actually
  arrive on a name I follow that I would have acted on?"**

The honest framing: the silence-cost of a drought is the *forward* return on
ideas the bot saw and never decided against. Computing forward returns is a
separate forensic job (``analytics/winner_autopsy.py`` / ``loser_autopsy``
own the equivalent for *executed* trades). This builder takes the cheaper
first step — **enumerate the high-scoring live-only articles on watchlist
tickers that arrived during the current drought window** and bucket them per
ticker. An empty result is itself informative ("you didn't miss anything";
``state="OK"``, ``n_opportunities=0``) — the silence-when-nothing-actionable
precedent of ``_macro_calendar_chat_lines`` / ``_event_readiness_chat_lines``
/ ``_host_pulse_line``. A non-empty result is the regret list.

``build_idle_opportunity`` is pure: it composes ``build_decision_drought``'s
canonical ``current_drought`` block (single source of truth, AGENTS.md #10 —
the SAME drought logic ``/api/decision-drought`` reports, so the two
endpoints can never disagree on what counts as an ongoing drought) and a
pre-fetched list of article rows. No DB, no network, never raises on
garbage inputs. ``now`` is injectable for tests.

Observational only — never gates Opus, never injected into the decision
prompt, no caps (AGENTS.md invariants #2/#12 — the ``shadow_vs_claude`` /
``stress_scenarios`` / ``recovery`` precedent).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

# Minimum ai_score for a live-news row to count as a "missed opportunity"
# during the drought. The articles.db ai_score is on a 0..10 scale; 6.0 is
# above the daemon's grey-zone threshold and below the urgency cutoff of
# 8.0, so it picks up "the bot would have noticed this" without flooding
# the panel with kw_score-only noise. Below this we treat the row as
# pre-filter noise (the heuristic_scorer's 0.5 cutoff is too generous for
# this surface — see digital-intern CLAUDE.md §2).
DEFAULT_MIN_AI_SCORE = 6.0

# Cap on rows emitted in the opportunities table. The drought is typically
# 4–24h with thousands of articles; we surface the loudest per-ticker.
DEFAULT_MAX_OPPORTUNITIES = 20

# Per-ticker only the single top scoring article is kept (the most plausibly
# causal one — the trader scans by ticker, not by article). The
# ``article_count`` field carries the full count so the operator can see how
# many supporting headlines existed.


def _parse_ts(ts) -> datetime | None:
    """Tolerate aware/naive ISO strings (digital-intern articles.db inserts
    aware ISO; the decisions table can be either). Same convention as
    ``decision_drought._parse_ts``."""
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
    """``$TKR`` cashtag OR word-boundary ``TKR`` — same regex shape as
    ``news_velocity._ticker_regex`` / ``trade_attribution`` / signals'
    ``ticker_sentiments``. Case-insensitive against the upper-cased article
    body so ``MUTUAL`` does NOT alias ``MU``, ``AMDOCS`` does NOT alias
    ``AMD``, while ``$NVDA`` cashtag still hits."""
    return re.compile(rf"(?:\$|\b){re.escape(ticker.upper())}\b")


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    # Reject NaN / +/-Inf — the digital-intern ai_score column has been
    # observed with stale NaNs from a half-trained model; a NaN compared with
    # min_ai_score is ALWAYS False (Python semantics) which would silently
    # drop the row. Reject explicitly so the path is obvious.
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _headline(state: str, n_opps: int, drought: dict | None,
              top: dict | None, min_ai_score: float) -> str:
    """One-sentence operator headline matching the
    ``_host_pulse_line`` / ``_capital_pulse_line`` voice."""
    if state == "NO_DATA":
        return "Idle opportunity: no decisions recorded yet."
    if state == "NO_DROUGHT":
        return ("Idle opportunity: no ongoing drought — the trader is "
                "filling normally; nothing missed by definition.")
    # OK — drought exists
    dur = (drought or {}).get("duration_hours")
    n_nd = (drought or {}).get("n_no_decision")
    dur_s = f"{dur:.1f}h" if isinstance(dur, (int, float)) else "?h"
    nd_s = f"{n_nd}" if n_nd is not None else "?"
    if n_opps == 0:
        return (
            f"Idle opportunity: drought {dur_s} ({nd_s} NO_DECISION) — "
            f"no live watchlist signals ≥{min_ai_score:.1f} arrived; "
            "the silence is honest."
        )
    # Regret case — name the loudest miss.
    if top is None:  # defensive — should not happen when n_opps > 0
        return (
            f"Idle opportunity: drought {dur_s} ({nd_s} NO_DECISION) — "
            f"{n_opps} watchlist signal(s) ≥{min_ai_score:.1f} arrived "
            "while the bot was dark."
        )
    score = top.get("top_score")
    score_s = f"{score:.1f}" if isinstance(score, (int, float)) else "?"
    held_tag = " (HELD)" if top.get("held") else ""
    return (
        f"Idle opportunity: drought {dur_s} ({nd_s} NO_DECISION) — "
        f"{n_opps} watchlist signal(s) ≥{min_ai_score:.1f} arrived; "
        f"loudest: {top.get('ticker')}{held_tag} @ ai_score {score_s}."
    )


def build_idle_opportunity(
    decision_drought_result: dict | None,
    articles: list[dict] | None,
    watchlist: list[str] | None,
    held_tickers: list[str] | None = None,
    now: datetime | None = None,
    min_ai_score: float = DEFAULT_MIN_AI_SCORE,
    max_opportunities: int = DEFAULT_MAX_OPPORTUNITIES,
) -> dict:
    """Bucket high-score watchlist articles arriving during the current
    drought into a per-ticker missed-opportunity table.

    Pure. ``decision_drought_result`` is the dict from
    ``build_decision_drought`` (composes verbatim — single source of truth,
    AGENTS.md #10; the endpoint owns the store read). ``articles`` is a list
    of dicts with at minimum ``title`` / ``ai_score`` / ``first_seen`` (and
    optionally ``url`` / ``source`` / ``body`` / ``urgency``). ``watchlist``
    is the universe of tickers to consider — typically ``strategy.WATCHLIST``
    upper-cased. ``held_tickers`` flags rows that correspond to currently-
    held positions (so the operator can see "the bot was dark on MY OWN
    position's news"). ``now`` is injectable for tests.

    Returns a JSON-ready dict — never raises on garbage inputs.
    """
    now = now or datetime.now(timezone.utc)
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "NO_DATA",
        "headline": "Idle opportunity: no decisions recorded yet.",
        "drought": None,
        "min_ai_score": float(min_ai_score),
        "n_opportunities": 0,
        "opportunities": [],
        "missed_top_score": None,
        "missed_top_ticker": None,
    }

    # ── 1. Drought gate ─────────────────────────────────────────────
    if not decision_drought_result or not isinstance(
            decision_drought_result, dict):
        # No drought block at all — degrade to NO_DATA (the underlying
        # build_decision_drought returns this when ``decisions`` was empty).
        out["headline"] = _headline("NO_DATA", 0, None, None, min_ai_score)
        return out

    drought = decision_drought_result.get("current_drought")
    if drought is None or not drought.get("ongoing"):
        # No ongoing drought — either the bot just filled or it never started
        # one. Either way nothing was missed during a drought that doesn't
        # exist. This is the operator-happy path (suppressed in reports).
        out["state"] = "NO_DROUGHT"
        out["headline"] = _headline("NO_DROUGHT", 0, None, None, min_ai_score)
        return out

    drought_start = _parse_ts(drought.get("start"))
    if drought_start is None:
        # Defensive: a drought block with no parseable start can't anchor a
        # window — treat as NO_DATA so we don't silently include the entire
        # articles table.
        out["headline"] = _headline("NO_DATA", 0, None, None, min_ai_score)
        return out

    # ── 2. Article filter — within the drought window, on a watchlist
    # ticker, at or above the score floor. ──────────────────────────
    wl = [t.upper() for t in (watchlist or []) if t and isinstance(t, str)]
    held_set = {t.upper() for t in (held_tickers or []) if t and isinstance(t, str)}
    if not wl:
        # No watchlist → nothing to bucket against. Still emit OK with empty
        # opportunities so the panel renders consistently (the drought is
        # real; the "regret" surface is just unconfigured).
        out["state"] = "OK"
        out["drought"] = _drought_slim(drought)
        out["headline"] = _headline("OK", 0, drought, None, min_ai_score)
        return out

    patterns = {t: _ticker_regex(t) for t in wl}
    score_floor = float(min_ai_score)

    # Map ticker → {top_score, top_title, top_first_seen, count, urgency_max,
    # body_excerpt, url, source}
    per_ticker: dict[str, dict] = {}

    for a in articles or []:
        if not isinstance(a, dict):
            continue
        score = _safe_float(a.get("ai_score"))
        if score is None or score < score_floor:
            continue
        fs_raw = a.get("first_seen")
        fs = _parse_ts(fs_raw)
        if fs is None or fs < drought_start:
            continue
        # Bucket the article against every watchlist ticker that appears in
        # the title (body left to the caller — we keep this surface cheap
        # so the SQL-side prefilter can omit body decompression entirely).
        title = (a.get("title") or "")
        body = (a.get("body") or "")
        # Upper-cased so the case-insensitive flag isn't needed on the
        # compiled regex — same convention as news_velocity.
        haystack = (title + " " + body).upper()
        urgency = a.get("urgency")
        try:
            urgency_i = int(urgency) if urgency is not None else 0
        except (TypeError, ValueError):
            urgency_i = 0
        for tk, pat in patterns.items():
            if not pat.search(haystack):
                continue
            prev = per_ticker.get(tk)
            if prev is None:
                per_ticker[tk] = {
                    "ticker": tk,
                    "top_score": score,
                    "top_title": title[:240],
                    "top_first_seen": _iso(fs),
                    "article_count": 1,
                    "max_urgency": urgency_i,
                    "held": tk in held_set,
                    "top_url": (a.get("url") or "")[:240] or None,
                    "top_source": (a.get("source") or "")[:120] or None,
                }
            else:
                prev["article_count"] += 1
                if urgency_i > prev["max_urgency"]:
                    prev["max_urgency"] = urgency_i
                # Tie-break: higher score wins; on equal score the NEWER
                # first_seen wins (more plausibly causal — same convention
                # as trade_attribution's tie-break).
                replace = score > prev["top_score"]
                if not replace and score == prev["top_score"]:
                    prev_fs = _parse_ts(prev.get("top_first_seen"))
                    if prev_fs is None or fs > prev_fs:
                        replace = True
                if replace:
                    prev["top_score"] = score
                    prev["top_title"] = title[:240]
                    prev["top_first_seen"] = _iso(fs)
                    prev["top_url"] = (a.get("url") or "")[:240] or None
                    prev["top_source"] = (a.get("source") or "")[:120] or None

    rows = list(per_ticker.values())
    # Sort: top_score DESC, then most-recent top_first_seen DESC (so an equal-
    # score tie surfaces the fresher catalyst first), then ticker asc for a
    # deterministic tiebreak on truly identical rows.
    rows.sort(key=lambda r: (
        -float(r["top_score"]),
        -_ts_sort_key(r.get("top_first_seen")),
        r["ticker"],
    ))
    rows = rows[:max_opportunities]

    out["state"] = "OK"
    out["drought"] = _drought_slim(drought)
    out["n_opportunities"] = len(rows)
    out["opportunities"] = rows
    if rows:
        out["missed_top_score"] = rows[0]["top_score"]
        out["missed_top_ticker"] = rows[0]["ticker"]
        out["headline"] = _headline("OK", len(rows), drought, rows[0], min_ai_score)
    else:
        out["headline"] = _headline("OK", 0, drought, None, min_ai_score)
    return out


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _ts_sort_key(ts) -> float:
    """Lexicographic ISO sort would work, but we want a sortable numeric so
    None / unparseable rows go LAST in a DESC sort. Treat None as ``-inf``."""
    dt = _parse_ts(ts)
    if dt is None:
        return float("-inf")
    return dt.timestamp()


def _drought_slim(drought: dict) -> dict:
    """Carry only the load-bearing drought fields into the response so a
    future drought-block schema bump doesn't bloat this endpoint. Composed
    verbatim from build_decision_drought's output (AGENTS.md #10 — single
    source of truth)."""
    return {
        "start": drought.get("start"),
        "end": drought.get("end"),
        "duration_hours": drought.get("duration_hours"),
        "n_cycles": drought.get("n_cycles"),
        "n_no_decision": drought.get("n_no_decision"),
        "n_hold": drought.get("n_hold"),
        "n_blocked": drought.get("n_blocked"),
        "kind": drought.get("kind"),
        "ongoing": drought.get("ongoing"),
    }
