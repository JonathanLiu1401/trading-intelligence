"""Per-trade audit chain — for the last N FILLED trades, attach the news
visible just before the trade, the forward price move, and a verdict.

The information the operator wants to answer "why did the bot do this?" is
already in the DB, but split across five surfaces:

* ``store.recent_trades`` — what the bot did + its short reason string.
* ``store.recent_decisions`` — the full Opus reasoning (longer than the
  trade ``reason`` column; carries the JSON-extracted reasoning block).
* digital-intern ``articles.db`` — the news the bot was reading.
* ``market.get_prices`` — the current price for the forward move.
* ``/api/position-thesis`` / ``/api/trade-attribution`` — partial chains.

None gives the operator the *one-pager per trade* view: ts, ticker, action,
the 3-5 loudest articles in the lookback window, the forward % move, and
whether the trade is GOOD/NEUTRAL/BAD/OPEN. This builder composes that one-
pager from the existing tables. Pure: takes pre-fetched trades + articles
(already SQL-filtered to live-only) + current_prices, never raises on
garbage inputs, never hits a DB or the network.

Distinct from every neighbour (do not consolidate, AGENTS.md invariant #10):

* ``/api/decision-context`` — FORWARD: what the bot is looking at *right
  now* to make the next decision. This is BACKWARD: per-past-trade.
* ``/api/decision-forensics`` — failure-mode taxonomy (NO_DECISION /
  parse-failure). This builder excludes those rows by definition — a
  trade row means a FILL happened.
* ``/api/trade-attribution`` — bulk article-to-trade attribution. This is
  per-trade with the OUTCOME attached.
* ``/api/position-thesis`` — open-position drift; doesn't cover closed
  trades, doesn't attach the article snapshot at decision time.
* ``/api/last-real-decision`` — most recent FILLED decision summary; n=1
  by design, no article snapshot, no verdict.

Verdict thresholds (configurable):

* ``GOOD`` — directional move ≥ ``good_pct`` in the bot's favour
  (BUY → +%, SELL → -%).
* ``BAD`` — move ≥ ``bad_pct`` AGAINST the bot.
* ``NEUTRAL`` — between the two thresholds.
* ``OPEN`` — trade newer than ``open_min_age_h`` (judgement deferred —
  too early to call).
* ``NO_PRICE`` — current_price missing OR the row is an option (BUY_CALL
  / SELL_PUT / etc) — option P/L isn't comparable to underlying %.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone


DEFAULT_N = 10
DEFAULT_LOOKBACK_HOURS = 6.0
DEFAULT_ARTICLE_TOP_K = 5
DEFAULT_GOOD_PCT = 1.0
DEFAULT_BAD_PCT = 3.0
DEFAULT_OPEN_MIN_AGE_HOURS = 24.0

_BUY_VERBS = {"BUY", "REBALANCE"}
_SELL_VERBS = {"SELL"}
_OPTION_VERBS = {"BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"}


def _parse_ts(ts) -> datetime | None:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).strip().replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _ticker_regex(ticker: str) -> re.Pattern:
    """Word-boundary or $cashtag — same shape as
    ``idle_opportunity._ticker_regex`` / ``news_velocity`` so MU ≠ MUTUAL,
    AMD ≠ AMDOCS, $NVDA cashtag still hits."""
    return re.compile(rf"(?:\$|\b){re.escape(ticker.upper())}\b")


def _normalize_action(action: str | None) -> str:
    if not action:
        return ""
    return str(action).strip().upper()


def _is_option(action: str, option_type: str | None) -> bool:
    if option_type in ("call", "put", "CALL", "PUT"):
        return True
    return action in _OPTION_VERBS


def _trade_direction(action: str) -> int:
    """+1 for buy-side (price expected up to win), -1 for sell-side, 0 unknown."""
    if action in _BUY_VERBS or action == "BUY_CALL" or action == "SELL_PUT":
        return 1
    if action in _SELL_VERBS or action == "BUY_PUT" or action == "SELL_CALL":
        return -1
    return 0


def _verdict(action: str, decision_price: float | None,
             current_price: float | None, is_option: bool,
             age_h: float | None,
             good_pct: float, bad_pct: float,
             open_min_age_h: float
             ) -> tuple[str, float | None, float | None]:
    """Return (verdict, abs_move_pct, intended_move_pct).

    ``intended_move_pct`` carries the move *signed in the bot's favour*:
    positive = the trade is making money, negative = losing. For a SELL
    where the price went DOWN, intended_move_pct is positive.

    Option rows always return NO_PRICE — the price column on an option
    trade is the option contract price, not comparable to the underlying
    quote we'd see in current_price.
    """
    if is_option:
        return "NO_PRICE", None, None
    dp = _safe_float(decision_price)
    cp = _safe_float(current_price)
    if dp is None or dp <= 0 or cp is None or cp <= 0:
        return "NO_PRICE", None, None
    raw_pct = (cp - dp) / dp * 100.0
    direction = _trade_direction(action)
    if direction == 0:
        # HOLD / unknown verbs — we don't classify these as GOOD/BAD.
        return "NEUTRAL", round(raw_pct, 3), 0.0
    intended = raw_pct * direction
    if age_h is not None and age_h < open_min_age_h:
        return "OPEN", round(raw_pct, 3), round(intended, 3)
    if intended >= good_pct:
        return "GOOD", round(raw_pct, 3), round(intended, 3)
    if intended <= -bad_pct:
        return "BAD", round(raw_pct, 3), round(intended, 3)
    return "NEUTRAL", round(raw_pct, 3), round(intended, 3)


def _pick_top_articles(articles: list[dict], pattern: re.Pattern,
                       decision_ts: datetime, lookback_h: float,
                       k: int) -> list[dict]:
    """Filter articles to those that:
      * mention the ticker (title only — body decompression is the caller's
        cost and not always worth it on a per-trade scan)
      * first_seen in [decision_ts - lookback_h, decision_ts]

    Then pick the top-k by ai_score DESC, ties broken by newer first_seen.
    """
    if k <= 0:
        return []
    window_start = decision_ts.timestamp() - lookback_h * 3600.0
    window_end = decision_ts.timestamp()
    cand: list[tuple[float, float, dict]] = []
    for a in articles or []:
        if not isinstance(a, dict):
            continue
        title = (a.get("title") or "")
        if not title:
            continue
        if not pattern.search(title.upper()):
            continue
        fs = _parse_ts(a.get("first_seen"))
        if fs is None:
            continue
        ts_f = fs.timestamp()
        if ts_f < window_start or ts_f > window_end:
            continue
        score = _safe_float(a.get("ai_score"))
        if score is None:
            continue
        cand.append((score, ts_f, a))
    cand.sort(key=lambda x: (-x[0], -x[1]))
    out: list[dict] = []
    for score, ts_f, a in cand[:k]:
        fs = _parse_ts(a.get("first_seen"))
        age_min = None
        if fs is not None:
            age_min = round((decision_ts.timestamp() - fs.timestamp()) / 60.0, 1)
        urgency_raw = a.get("urgency")
        try:
            urgency_i = int(urgency_raw) if urgency_raw is not None else 0
        except (TypeError, ValueError):
            urgency_i = 0
        out.append({
            "title": (a.get("title") or "")[:240],
            "ai_score": round(float(score), 3),
            "urgency": urgency_i,
            "source": (a.get("source") or "")[:120] or None,
            "first_seen": a.get("first_seen"),
            "age_min_at_decision": age_min,
            "url": (a.get("url") or "")[:240] or None,
        })
    return out


def _classify_action_taken_match(decisions: list[dict], trade_ts: datetime,
                                 ticker: str | None,
                                 tolerance_s: float = 90.0) -> str | None:
    """Find the decisions.reasoning corresponding to this trade by ts proximity.

    The trade and decision rows are written in the same _execute() call so
    they're usually within milliseconds, but we tolerate a small skew. Match
    requires the same ticker (extracted from action_taken) to avoid a false
    match across two concurrent trades.
    """
    if not decisions:
        return None
    tk_upper = (ticker or "").upper()
    best: tuple[float, str] | None = None
    for d in decisions:
        if not isinstance(d, dict):
            continue
        ts = _parse_ts(d.get("timestamp"))
        if ts is None:
            continue
        dt = abs(ts.timestamp() - trade_ts.timestamp())
        if dt > tolerance_s:
            continue
        a_t = (d.get("action_taken") or "")
        if tk_upper and tk_upper not in a_t.upper():
            continue
        reason = d.get("reasoning")
        if not reason:
            continue
        if best is None or dt < best[0]:
            best = (dt, str(reason))
    return best[1] if best else None


def _headline(state: str, counts: dict[str, int], n_chains: int) -> str:
    if state == "NO_DATA":
        return "Decision chain: no FILLED trades to audit yet."
    good = counts.get("GOOD", 0)
    bad = counts.get("BAD", 0)
    neutral = counts.get("NEUTRAL", 0)
    open_n = counts.get("OPEN", 0)
    no_price = counts.get("NO_PRICE", 0)
    if good + bad + neutral == 0:
        # Everything still OPEN / NO_PRICE
        if open_n > 0 and no_price == 0:
            return (f"Decision chain: {open_n}/{n_chains} trades still OPEN "
                    "(too recent to judge).")
        if no_price > 0 and open_n == 0:
            return (f"Decision chain: {no_price}/{n_chains} trades have no "
                    "comparable price (options or missing mark).")
        return (f"Decision chain: {n_chains} trades pending verdict "
                f"({open_n} OPEN, {no_price} NO_PRICE).")
    return (f"Decision chain (last {n_chains}): "
            f"{good} GOOD / {neutral} NEUTRAL / {bad} BAD"
            + (f" + {open_n} OPEN" if open_n else "")
            + (f" + {no_price} NO_PRICE" if no_price else "")
            + ".")


def build_decision_chain(
    trades: list[dict] | None,
    articles: list[dict] | None,
    current_prices: dict[str, float | None] | None,
    decisions: list[dict] | None = None,
    now: datetime | None = None,
    lookback_h: float = DEFAULT_LOOKBACK_HOURS,
    n: int = DEFAULT_N,
    article_top_k: int = DEFAULT_ARTICLE_TOP_K,
    good_pct: float = DEFAULT_GOOD_PCT,
    bad_pct: float = DEFAULT_BAD_PCT,
    open_min_age_h: float = DEFAULT_OPEN_MIN_AGE_HOURS,
) -> dict:
    """Build the per-trade audit chain.

    Args:
        trades: list of trade row dicts from ``store.recent_trades``. The
            caller controls ``limit``; we slice to ``n`` from the head.
        articles: list of article row dicts from ``articles.db`` (live-only
            clause MUST be pre-applied by the caller). At minimum:
            ``title`` / ``ai_score`` / ``first_seen``. Bucketed per-trade
            by ticker mention in the title.
        current_prices: ``{ticker: latest_mark}``. None / missing → NO_PRICE.
        decisions: optional list of decision rows from
            ``store.recent_decisions``. When provided, we join the matching
            reasoning by timestamp+ticker (longer text than ``trades.reason``);
            falls back to ``trades.reason`` if no decision row matches.
        now: injectable wall clock for tests.
        lookback_h: how many hours BEFORE each trade to look for articles.
        n: number of most-recent trades to include.
        article_top_k: top-k articles per trade to surface.
        good_pct / bad_pct: verdict thresholds (% in the bot's favour).
        open_min_age_h: trades fresher than this stay OPEN.

    Returns a JSON-ready dict. Never raises on garbage inputs.
    """
    now = now or datetime.now(timezone.utc)
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "state": "NO_DATA",
        "n_chains": 0,
        "lookback_hours": float(lookback_h),
        "good_pct": float(good_pct),
        "bad_pct": float(bad_pct),
        "open_min_age_hours": float(open_min_age_h),
        "article_top_k": int(article_top_k),
        "chains": [],
        "verdict_counts": {
            "GOOD": 0, "NEUTRAL": 0, "BAD": 0, "OPEN": 0, "NO_PRICE": 0,
        },
        "headline": "Decision chain: no FILLED trades to audit yet.",
    }

    if not trades:
        return out

    sane: list[dict] = []
    for t in trades:
        if not isinstance(t, dict):
            continue
        ts = _parse_ts(t.get("timestamp"))
        if ts is None:
            continue
        tk = (t.get("ticker") or "").strip().upper()
        if not tk:
            continue
        sane.append({**t, "_ts": ts, "_ticker": tk})

    if not sane:
        return out

    # Trades are typically newest-first from store.recent_trades; sort
    # defensively and take the n newest.
    sane.sort(key=lambda r: r["_ts"], reverse=True)
    sane = sane[: max(1, int(n))]

    cp = current_prices or {}
    counts = {"GOOD": 0, "NEUTRAL": 0, "BAD": 0, "OPEN": 0, "NO_PRICE": 0}
    chains: list[dict] = []
    # Precompute ticker → pattern (cache across trades on the same ticker).
    pat_cache: dict[str, re.Pattern] = {}

    for t in sane:
        tk = t["_ticker"]
        decision_ts = t["_ts"]
        action = _normalize_action(t.get("action"))
        opt = _is_option(action, t.get("option_type"))
        decision_price = _safe_float(t.get("price"))
        current_p = _safe_float(cp.get(tk))
        if pat_cache.get(tk) is None:
            pat_cache[tk] = _ticker_regex(tk)
        pat = pat_cache[tk]
        age_h = (now.timestamp() - decision_ts.timestamp()) / 3600.0
        verdict, abs_pct, intended_pct = _verdict(
            action=action, decision_price=decision_price,
            current_price=current_p, is_option=opt,
            age_h=age_h,
            good_pct=good_pct, bad_pct=bad_pct,
            open_min_age_h=open_min_age_h,
        )
        counts[verdict] = counts.get(verdict, 0) + 1
        top_articles = _pick_top_articles(
            articles or [], pat, decision_ts, lookback_h, article_top_k,
        )
        # Reason: prefer the matching decisions.reasoning (longer Opus text)
        # when available; fall back to the shorter trade.reason.
        long_reason = _classify_action_taken_match(
            decisions or [], decision_ts, tk,
        )
        reason_raw = long_reason or t.get("reason") or ""
        # Cap excerpt — the dashboard tile is bounded, full reasoning is in
        # decisions.reasoning if the operator needs it.
        excerpt = (str(reason_raw)[:400]).strip()
        chains.append({
            "trade_id": t.get("id"),
            "ts": decision_ts.isoformat(timespec="seconds"),
            "ticker": tk,
            "action": action or None,
            "qty": _safe_float(t.get("qty")),
            "decision_price": decision_price,
            "value": _safe_float(t.get("value")),
            "expiry": t.get("expiry"),
            "strike": _safe_float(t.get("strike")),
            "option_type": t.get("option_type"),
            "is_option": opt,
            "age_hours": round(age_h, 2),
            "reason_excerpt": excerpt,
            "reason_truncated": len(str(reason_raw)) > 400,
            "pre_decision_news": {
                "lookback_hours": float(lookback_h),
                "n_articles_returned": len(top_articles),
                "top_articles": top_articles,
            },
            "outcome": {
                "current_price": current_p,
                "abs_move_pct": abs_pct,
                "intended_move_pct": intended_pct,
                "verdict": verdict,
            },
        })

    out["state"] = "OK"
    out["n_chains"] = len(chains)
    out["chains"] = chains
    out["verdict_counts"] = counts
    out["headline"] = _headline("OK", counts, len(chains))
    return out
