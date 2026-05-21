"""Watchlist coverage — which tickers has the bot stopped attending to?

The live ``WATCHLIST`` has 48 tickers (semis, hyperscalers, sector ETFs,
leveraged ETFs, biotech, defense, gold, BTC proxies). At any given
moment the bot considers them all in the prompt, but its FILLED actions
and its reasoning text drift toward the names with active catalysts. A
desk that has not mentioned LRCX, KLAC, ASML, MRVL, SMH in 48h while
the wire is dominated by memory-chip earnings is structurally
under-fishing its own universe — opportunity cost the operator never
sees because every existing dashboard panel is *position*-centric (it
shows what was traded, not what was ignored).

The adjacent endpoints, and why none of them answer this question:

* ``/api/ticker-decision-mix`` — per-ticker decision counts but only
  for tickers that *appear* in the recent decisions; never names a
  ticker that was ignored.
* ``/api/watchlist-opportunities`` — forward-looking suggestion engine
  (news heat per ticker the bot doesn't hold); not historical.
* ``/api/rising-unheld-themes`` — news theme surface, not per-ticker
  attention.
* ``/api/repeat-loser`` / ``/api/rebuy-regret`` — focused on past
  losers, not coverage breadth.

This module fills the gap. For each watchlist ticker:

* ``last_seen_ts`` — most recent decision row that referenced the
  ticker, either via ``action_taken`` (e.g. ``BUY NVDA → FILLED``) or
  via a whole-word mention in ``reasoning``.
* ``hours_since_last_seen`` / ``never_seen`` — the staleness scalar.
* ``mentions_24h`` / ``mentions_7d`` — cadence in the recent window.

The verdict ladder:

* ``STAGNANT``      — > 50% of watchlist either never seen or stale > 7d
* ``CONCENTRATED``  — top-3 tickers receive > ``CONCENTRATED_TOP3_SHARE``
  of all watchlist mentions in 24h
* ``DIVERSIFIED``   — broad coverage, no over-concentration
* ``NO_DATA``       — empty input

Pure builder, never raises (matches the
``decision_paralysis`` / ``trade_asymmetry`` discipline). Observational
only — never gates Opus, no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

# Thresholds — module-owned so tests read constants. Re-tunable.
STAGNANT_SHARE = 0.50          # > this fraction of WL untouched in > 7d ⇒ STAGNANT
CONCENTRATED_TOP3_SHARE = 0.80  # top-3 tickers receive ≥ this share of mentions ⇒ CONCENTRATED
RECENT_HOURS = 24.0
DEEP_HOURS = 168.0              # 7 days

# A ticker is "stale" once it has not been mentioned in DEEP_HOURS.
STALE_HOURS = DEEP_HOURS


def _parse_ts(ts):
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _action_ticker(action_taken):
    """Pull the (verb, ticker) from ``action_taken`` — SSOT mirror of
    ``dashboard._parse_action_ticker``. Inlined to keep this leaf pure
    and avoid the dashboard ↔ analytics import cycle. Drift-locked by
    ``tests/test_watchlist_coverage.py::test_action_ticker_mirrors_dashboard``.
    """
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


def _compile_ticker_pattern(watchlist):
    """Build a single regex that whole-word-matches any watchlist ticker.

    Using ``\\b`` boundaries so an article phrase like "fundamental MU
    coverage" matches MU but "format" does NOT match the AT&T ticker
    "T". Sorted longest-first so ``"AMD"`` is preferred over ``"AM"``
    if both were present — the longer match always wins by leftmost-
    longest alternation order.
    """
    if not watchlist:
        return None
    valid = [t.upper() for t in watchlist if t and t.isalnum()]
    if not valid:
        return None
    valid.sort(key=len, reverse=True)
    pat = r"\b(" + "|".join(re.escape(t) for t in valid) + r")\b"
    return re.compile(pat)


def _extract_reasoning_mentions(reasoning, pattern):
    """Set of watchlist tickers whole-word-mentioned in ``reasoning``."""
    if not reasoning or pattern is None:
        return set()
    try:
        return {m.upper() for m in pattern.findall(reasoning)}
    except Exception:
        return set()


def build_watchlist_coverage(watchlist, decisions, now=None):
    """Per-ticker attention scan over the recent decision stream.

    ``watchlist`` — iterable of ticker strings (typically
    ``paper_trader.strategy.WATCHLIST``).
    ``decisions`` — newest-first list of decision rows matching
    ``store.recent_decisions(N)`` shape: each row a dict with at least
    ``timestamp``, ``action_taken``, and ``reasoning``.
    """
    now = now or datetime.now(timezone.utc)
    wl = [t.upper() for t in (watchlist or []) if t]
    out = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_watchlist": len(wl),
        "n_decisions_scanned": 0,
        "by_ticker": [],
        "n_never_seen": 0,
        "n_stale_7d": 0,
        "n_active_24h": 0,
        "top_3_share_24h": 0.0,
        "thresholds": {
            "stagnant_share": STAGNANT_SHARE,
            "concentrated_top3_share": CONCENTRATED_TOP3_SHARE,
            "recent_hours": RECENT_HOURS,
            "stale_hours": STALE_HOURS,
        },
        "verdict": "NO_DATA",
        "headline": "No watchlist or no decisions — cannot assess coverage.",
    }
    if not wl or not decisions:
        return out

    pattern = _compile_ticker_pattern(wl)
    wl_set = set(wl)
    per_ticker = {t: {
        "ticker": t,
        "last_seen_ts": None,
        "last_seen_action": None,
        "hours_since_last_seen": None,
        "never_seen": True,
        "mentions_24h": 0,
        "mentions_7d": 0,
        "action_count_7d": 0,
    } for t in wl}

    out["n_decisions_scanned"] = len(decisions)
    recent_cutoff = RECENT_HOURS * 3600.0
    deep_cutoff = DEEP_HOURS * 3600.0
    total_recent_mentions = 0
    mention_counts_24h = {}

    for row in decisions:
        ts = _parse_ts(row.get("timestamp"))
        age_s = (now - ts).total_seconds() if ts is not None else None
        verb, action_ticker = _action_ticker(row.get("action_taken"))
        seen_tickers = set()
        if action_ticker and action_ticker in wl_set:
            seen_tickers.add(action_ticker)
        reasoning_tickers = _extract_reasoning_mentions(
            row.get("reasoning"), pattern)
        seen_tickers |= (reasoning_tickers & wl_set)

        for tk in seen_tickers:
            rec = per_ticker[tk]
            # Newest-first iteration ⇒ first time we encounter ticker
            # is the most recent.
            if rec["never_seen"]:
                rec["never_seen"] = False
                rec["last_seen_ts"] = (ts.isoformat(timespec="seconds")
                                       if ts is not None else None)
                rec["last_seen_action"] = row.get("action_taken")
                rec["hours_since_last_seen"] = (
                    round(age_s / 3600.0, 2) if age_s is not None else None)
            if age_s is not None:
                if age_s <= recent_cutoff:
                    rec["mentions_24h"] += 1
                    mention_counts_24h[tk] = mention_counts_24h.get(tk, 0) + 1
                    total_recent_mentions += 1
                if age_s <= deep_cutoff:
                    rec["mentions_7d"] += 1
                    if tk == action_ticker and verb not in ("NO_DECISION", "BLOCKED", ""):
                        rec["action_count_7d"] += 1

    n_never = sum(1 for r in per_ticker.values() if r["never_seen"])
    n_stale = sum(1 for r in per_ticker.values()
                  if r["never_seen"]
                  or (r["hours_since_last_seen"] is not None
                      and r["hours_since_last_seen"] > STALE_HOURS))
    n_active_24h = sum(1 for r in per_ticker.values() if r["mentions_24h"] > 0)

    out["n_never_seen"] = n_never
    out["n_stale_7d"] = n_stale
    out["n_active_24h"] = n_active_24h

    if total_recent_mentions > 0:
        top3 = sorted(mention_counts_24h.values(), reverse=True)[:3]
        out["top_3_share_24h"] = round(
            sum(top3) / total_recent_mentions, 4)

    # Verdict ladder — most-specific first.
    stale_share = n_stale / len(wl)
    if stale_share > STAGNANT_SHARE:
        out["verdict"] = "STAGNANT"
        out["headline"] = (
            f"STAGNANT — {n_stale} of {len(wl)} watchlist tickers "
            f"({stale_share*100:.0f}%) untouched in 7d+ (never_seen: "
            f"{n_never}, stale: {n_stale - n_never}). The desk is "
            f"under-fishing its own universe; the WATCHLIST is wider "
            f"than the bot's active attention.")
    elif (out["top_3_share_24h"] >= CONCENTRATED_TOP3_SHARE
          and total_recent_mentions >= 10):
        # Concentration verdict only fires with a non-trivial mention
        # base — three mentions on three different tickers isn't a
        # concentration story.
        out["verdict"] = "CONCENTRATED"
        top_names = sorted(mention_counts_24h.items(),
                           key=lambda kv: (-kv[1], kv[0]))[:3]
        top_str = ", ".join(f"{t}({c})" for t, c in top_names)
        out["headline"] = (
            f"CONCENTRATED — top 3 ({top_str}) absorb "
            f"{out['top_3_share_24h']*100:.0f}% of last-24h watchlist "
            f"mentions across {len(wl)} ticker(s). Other names are "
            f"being talked past.")
    else:
        out["verdict"] = "DIVERSIFIED"
        out["headline"] = (
            f"DIVERSIFIED — {n_active_24h} of {len(wl)} tickers "
            f"received attention in 24h; stale (>7d): {n_stale}. "
            f"Coverage breadth healthy.")

    # Sort by_ticker most-stale-first (never_seen + biggest hours gap),
    # alphabetical tiebreak — stable & easy to render.
    rows = list(per_ticker.values())
    rows.sort(key=lambda r: (
        0 if r["never_seen"] else 1,
        -(r["hours_since_last_seen"] or 0.0),
        r["ticker"],
    ))
    out["by_ticker"] = rows
    return out
