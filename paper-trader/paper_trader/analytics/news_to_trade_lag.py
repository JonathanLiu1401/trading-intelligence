"""News-to-trade **lag distribution** — is the desk acting on fresh news?

``/api/trade-attribution`` enumerates the highest-scored articles that
plausibly preceded each FILLED trade, with a ``minutes_before_trade`` per
attributed article. What it does **not** answer is the desk's diagnostic
summary question:

  **"Across the last N hours of fills, what is the typical gap between the
  *top-scored* article on a name and the *actual fill*? Am I a fast reactor
  or am I consistently 2 hours behind the news?"**

That single distribution (median / p25 / p75 / max) plus a verdict is the
high-density operator surface — ``trade_attribution`` shows N attributions
per trade, this surface compresses them to *one* signed verdict on the
desk's reactivity.

Composes ``build_trade_attribution`` (SSOT, AGENTS.md #10). For each
attributed trade, takes the *minimum* ``minutes_before_trade`` across
its attributed articles — that's the freshest plausibly-causal signal
the trade could have been reacting to. (A trade can have multiple
attributions; the fastest is the operative lag.) Trades with zero
attributions are counted separately (``n_no_attribution``); they are
honestly *excluded* from the distribution rather than being assigned
``window_hours`` as a fake worst case (the ``recovery`` / ``loser_autopsy``
negative-space-is-data precedent — silence is not the same as ``= max``).

Verdict ladder:
  ``REACTIVE_FAST`` — median lag < 30 min (acting on hot news)
  ``REACTIVE``      — 30 .. 120 min
  ``DELAYED``       — > 120 min (consistently late on the news)
  ``NO_ATTRIBUTION`` — > half of trades have no attributed live news
  ``NO_DATA``       — no FILLED trades in window

Pure — no DB, no network, never raises. The endpoint owns I/O (the
``trade_attribution`` / ``reentry_velocity`` builder split).

Observational only — never gates Opus, never injected into the decision
prompt, no caps (AGENTS.md #2 / #12 — the ``reentry_velocity`` / ``churn``
precedent).
"""
from __future__ import annotations

from datetime import datetime, timezone


# Verdict ladder. Pinned in tests — adjust together.
_REACTIVE_FAST_MIN = 30.0
_DELAYED_MIN = 120.0

# A trade counts as "no attribution" if its trade-attribution row has
# ``n_attributed == 0``. When more than half of trades fall in that
# bucket, the desk isn't reactive to news — it's likely quant-driven
# or trading on signals not in articles.db. NO_ATTRIBUTION trumps the
# numeric verdict in that case (otherwise the median of two trades
# would whipsaw the verdict).
_NO_ATTRIBUTION_PCT_FLOOR = 50.0


def _f(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _quantile(values: list[float], q: float) -> float | None:
    """Nearest-rank quantile (the ``tail_risk`` precedent — no
    interpolation, so a single sample doesn't synthesise an in-between
    figure). Uses ``math.ceil`` so banker's rounding can't shift the
    pick by an index on integer-friendly samples."""
    import math
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    # Nearest-rank: ceil(q*n) − 1, clamped to [0, n-1].
    if q <= 0:
        return s[0]
    idx = max(0, min(n - 1, math.ceil(q * n) - 1))
    return s[idx]


def _classify_lag(min_minutes: float) -> str:
    """Single trade's reactivity class."""
    if min_minutes < _REACTIVE_FAST_MIN:
        return "REACTIVE_FAST"
    if min_minutes < _DELAYED_MIN:
        return "REACTIVE"
    return "DELAYED"


def build_news_to_trade_lag(trade_attribution_result: dict,
                            now: datetime | None = None) -> dict:
    """Compose a lag distribution from a ``build_trade_attribution`` result.

    ``trade_attribution_result`` is the dict returned by
    ``build_trade_attribution`` — has ``state``, ``trades`` (list of
    per-trade rows with ``attributed`` lists and ``n_attributed``).
    The endpoint passes this dict through verbatim; this builder reads
    ``trades`` and aggregates.

    Returns a JSON-ready dict:
      ``as_of`` (ISO seconds),
      ``state`` (NO_DATA / NO_ATTRIBUTION / OK),
      ``verdict`` (REACTIVE_FAST / REACTIVE / DELAYED / NO_ATTRIBUTION / NO_DATA),
      ``n_trades`` (total considered),
      ``n_attributed`` (trades with ≥1 attributed article),
      ``n_no_attribution`` (trades with 0 attributed articles),
      ``no_attribution_pct``,
      ``min_lag_minutes`` / ``median_lag_minutes`` / ``p25_lag_minutes`` /
      ``p75_lag_minutes`` / ``max_lag_minutes`` (None if no attributed trades),
      ``bucket_fast`` / ``bucket_reactive`` / ``bucket_delayed`` (counts),
      ``per_trade`` (newest first: ticker, action, trade_ts, top_score,
        min_lag_minutes, classification, top_title),
      ``headline``.

    Pure — never raises on garbage rows. ``None`` / non-dict input
    degrades to NO_DATA.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    def _empty(state: str, verdict: str, headline: str) -> dict:
        return {
            "as_of": now.isoformat(timespec="seconds"),
            "state": state, "verdict": verdict,
            "n_trades": 0, "n_attributed": 0, "n_no_attribution": 0,
            "no_attribution_pct": 0.0,
            "min_lag_minutes": None, "median_lag_minutes": None,
            "p25_lag_minutes": None, "p75_lag_minutes": None,
            "max_lag_minutes": None,
            "bucket_fast": 0, "bucket_reactive": 0, "bucket_delayed": 0,
            "per_trade": [], "headline": headline,
        }

    if not isinstance(trade_attribution_result, dict):
        return _empty("NO_DATA", "NO_DATA",
                      "no trade-attribution input — lag unmeasurable.")

    trades = trade_attribution_result.get("trades")
    if not isinstance(trades, list) or not trades:
        return _empty("NO_DATA", "NO_DATA",
                      "no recent FILLED trades — lag unmeasurable.")

    n_trades = 0
    n_attributed = 0
    n_no_attribution = 0
    min_lags: list[float] = []
    per_trade: list[dict] = []

    for t in trades:
        if not isinstance(t, dict):
            continue
        n_trades += 1
        attributed = t.get("attributed") or []
        if not isinstance(attributed, list) or not attributed:
            n_no_attribution += 1
            per_trade.append({
                "ticker": t.get("ticker"),
                "action": t.get("action"),
                "trade_ts": t.get("timestamp"),
                "top_score": None,
                "min_lag_minutes": None,
                "classification": "NO_ATTRIBUTION",
                "top_title": None,
            })
            continue

        # The freshest article = lowest minutes_before_trade. (A trade can
        # have multiple attributed articles; the operative lag is the
        # quickest signal the trade could have reacted to.)
        candidate_lags = []
        for a in attributed:
            if not isinstance(a, dict):
                continue
            m = _f(a.get("minutes_before_trade"))
            if m is None or m < 0:
                continue
            candidate_lags.append((m, a))
        if not candidate_lags:
            n_no_attribution += 1
            per_trade.append({
                "ticker": t.get("ticker"),
                "action": t.get("action"),
                "trade_ts": t.get("timestamp"),
                "top_score": None,
                "min_lag_minutes": None,
                "classification": "NO_ATTRIBUTION",
                "top_title": None,
            })
            continue

        candidate_lags.sort(key=lambda mp: mp[0])
        min_lag, freshest = candidate_lags[0]
        # Top score across attributed = the highest ai_score article
        # — orthogonal to "freshest", but we surface both per-trade.
        top = max(attributed,
                  key=lambda a: _f(a.get("ai_score"), 0.0) if isinstance(a, dict) else 0.0)
        n_attributed += 1
        min_lags.append(min_lag)
        per_trade.append({
            "ticker": t.get("ticker"),
            "action": t.get("action"),
            "trade_ts": t.get("timestamp"),
            "top_score": _f((top or {}).get("ai_score"), 0.0)
            if isinstance(top, dict) else 0.0,
            "min_lag_minutes": round(min_lag, 1),
            "classification": _classify_lag(min_lag),
            "top_title": (top or {}).get("title")
            if isinstance(top, dict) else None,
        })

    if n_trades == 0:
        return _empty("NO_DATA", "NO_DATA",
                      "no parseable trades — lag unmeasurable.")

    no_attribution_pct = round(n_no_attribution / n_trades * 100.0, 1)

    median_lag = _median(min_lags)
    bucket_fast = sum(1 for m in min_lags if m < _REACTIVE_FAST_MIN)
    bucket_reactive = sum(1 for m in min_lags
                          if _REACTIVE_FAST_MIN <= m < _DELAYED_MIN)
    bucket_delayed = sum(1 for m in min_lags if m >= _DELAYED_MIN)

    # NO_ATTRIBUTION trumps numeric verdict when most trades lack news;
    # the median of one or two attributed trades isn't a desk-wide signal.
    if n_attributed == 0:
        state = "NO_ATTRIBUTION"
        verdict = "NO_ATTRIBUTION"
        headline = (f"{n_trades} FILLED trade(s) but none have attributed "
                    f"live news — lag unmeasurable.")
    elif no_attribution_pct >= _NO_ATTRIBUTION_PCT_FLOOR:
        state = "OK"
        verdict = "NO_ATTRIBUTION"
        headline = (f"{n_no_attribution}/{n_trades} trade(s) "
                    f"({no_attribution_pct:.0f}%) without attributed "
                    f"news — desk is quant-driven or signals are missing.")
    else:
        state = "OK"
        if median_lag is not None and median_lag < _REACTIVE_FAST_MIN:
            verdict = "REACTIVE_FAST"
        elif median_lag is not None and median_lag < _DELAYED_MIN:
            verdict = "REACTIVE"
        else:
            verdict = "DELAYED"
        headline = (
            f"Median news-to-fill lag {median_lag:.0f}min across "
            f"{n_attributed}/{n_trades} attributed trade(s) — {verdict}."
        )

    # Newest-first per_trade so the operator sees recent reactivity first.
    def _ts_key(r):
        ts = r.get("trade_ts")
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    per_trade.sort(key=_ts_key, reverse=True)

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state, "verdict": verdict,
        "n_trades": n_trades,
        "n_attributed": n_attributed,
        "n_no_attribution": n_no_attribution,
        "no_attribution_pct": no_attribution_pct,
        "min_lag_minutes": round(min(min_lags), 1) if min_lags else None,
        "median_lag_minutes": round(median_lag, 1)
        if median_lag is not None else None,
        "p25_lag_minutes": round(_quantile(min_lags, 0.25), 1)
        if min_lags else None,
        "p75_lag_minutes": round(_quantile(min_lags, 0.75), 1)
        if min_lags else None,
        "max_lag_minutes": round(max(min_lags), 1) if min_lags else None,
        "bucket_fast": bucket_fast,
        "bucket_reactive": bucket_reactive,
        "bucket_delayed": bucket_delayed,
        "per_trade": per_trade,
        "headline": headline,
    }
