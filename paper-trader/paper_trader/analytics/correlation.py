"""Concentration honesty — do the held names actually move together?

``/api/risk`` already reports **name-level** concentration
(``concentration_top1_pct`` / ``top3_pct``) and a single 3% SPY-shock. What
it cannot see is *factor* concentration: a "2-position, 59%/41%" book reads
as merely concentrated, but if both names are high-β semis that move as one,
the operator is really running a **single bet** — a semis drawdown takes the
whole book and the SPY-shock number understates the true tail.

``build_correlation`` is the missing lens: pairwise return correlation among
the held **stock** positions, the single most-correlated pair, the
weight-Herfindahl effective-position count, and the **correlation-adjusted
effective number of independent bets** — which collapses toward 1 as the
names co-move regardless of how many tickers are on the book.

This is a *diagnostic / advisory* panel only — it never gates Opus and adds
no caps (AGENTS.md invariants #2/#12).

Design parity with the codebase:

* **Options are flagged & skipped.** Correlation of a non-linear Greeks
  payoff against a linear stock return is not meaningful — the same
  "stocks only" carve-out ``open_attribution`` / ``/api/backtests/compare``
  make (invariant #10 spirit).
* **The builder is pure; the network lives in the endpoint.** Exactly the
  ``thesis_drift`` split — the endpoint fetches daily closes from yfinance
  and hands the builder plain ``{ticker: [close, …]}`` dicts, so the core
  is offline and deterministically testable.
* **Sample-size honesty** mirrors ``trade_asymmetry`` / ``news_edge``:
  numerics emit as soon as they can be computed, but the verdict is
  withheld (``INSUFFICIENT``) until there are ≥2 correlatable names with a
  long-enough series.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Need at least this many aligned daily returns for a correlation to mean
# anything (≈ two trading weeks). Below it the pair is reported as
# uncorrelatable rather than fed a noisy ρ.
MIN_RETURNS = 10
# Verdict thresholds on the mean pairwise correlation. Documented module
# constants (tests assert behaviour at each boundary), not tunables Opus
# ever sees.
HIGH_CORR = 0.70   # ≥ ⇒ the book moves as one
MOD_CORR = 0.40    # ≥ ⇒ partially diversified
# A single name this share of the book is single-name risk first; the
# correlation read is secondary context.
DOMINANT_WEIGHT = 0.60


def _returns(closes: list[float]) -> list[float]:
    """Simple daily returns from a close series (oldest→newest).

    Non-positive / non-finite prices break the chain at that point — the
    rest of the series is still used (a single bad yfinance bar must not
    zero the whole correlation)."""
    out: list[float] = []
    prev = None
    for c in closes:
        try:
            c = float(c)
        except (TypeError, ValueError):
            prev = None
            continue
        if c != c or c <= 0:  # NaN or non-positive
            prev = None
            continue
        if prev is not None:
            out.append(c / prev - 1.0)
        prev = c
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation. ``None`` if either series is flat (zero variance)
    — a constant series has an undefined correlation, never a fabricated 0."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-18 or syy <= 1e-18:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = sxy / (sxx ** 0.5 * syy ** 0.5)
    # Float guard: keep ρ inside [-1, 1] against rounding overshoot.
    return round(max(-1.0, min(1.0, r)), 4)


def build_correlation(positions: list[dict],
                      price_history: dict,
                      now: datetime | None = None) -> dict:
    """Pairwise-correlation / effective-bets decomposition. Pure, never raises.

    ``positions`` — open positions, each ``{ticker, market_value, type}``
    (``type`` ``"stock"``/falsy = stock; anything else is an option and is
    flagged & skipped). ``price_history`` — ``{ticker: [close, …]}`` daily
    closes oldest→newest (the endpoint supplies these; missing/short series
    degrade, never error).
    """
    now = now or datetime.now(timezone.utc)

    stock_pos, skipped_options = [], []
    for p in positions or []:
        typ = (p.get("type") or "stock")
        if typ != "stock":
            skipped_options.append(p.get("ticker"))
            continue
        stock_pos.append(p)

    # Returns per stock ticker, then a common aligned tail so every pair is
    # measured over the *same* window.
    rets: dict[str, list[float]] = {}
    for p in stock_pos:
        tk = p.get("ticker")
        if tk is None or tk in rets:
            continue
        r = _returns(price_history.get(tk) or [])
        if len(r) >= MIN_RETURNS:
            rets[tk] = r
    usable = sorted(rets)
    short_series = sorted(
        p.get("ticker") for p in stock_pos
        if p.get("ticker") not in rets and p.get("ticker") is not None)

    if usable:
        k = min(len(rets[t]) for t in usable)
        aligned = {t: rets[t][-k:] for t in usable}
    else:
        aligned = {}

    # Pairwise correlations (deterministic ticker-sorted order).
    pairs: list[dict] = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            a, b = usable[i], usable[j]
            rho = _pearson(aligned[a], aligned[b])
            pairs.append({"a": a, "b": b, "corr": rho})
    corr_vals = [pp["corr"] for pp in pairs if pp["corr"] is not None]
    mean_corr = (round(sum(corr_vals) / len(corr_vals), 4)
                 if corr_vals else None)
    max_pair = None
    if corr_vals:
        mp = max((pp for pp in pairs if pp["corr"] is not None),
                 key=lambda pp: (pp["corr"], pp["a"], pp["b"]))
        max_pair = {"tickers": [mp["a"], mp["b"]], "corr": mp["corr"]}

    # Weight Herfindahl over stock positions that carry a positive value.
    wvals = [(p.get("ticker"), float(p.get("market_value") or 0.0))
             for p in stock_pos]
    tot = sum(v for _, v in wvals if v > 0)
    weights, hhi, eff_naive, top_weight, top_ticker = None, None, None, None, None
    if tot > 0:
        weights = {tk: round(v / tot, 6) for tk, v in wvals if v > 0}
        hhi = round(sum(w * w for w in weights.values()), 6)
        eff_naive = round(1.0 / hhi, 4) if hhi > 0 else None
        top_ticker, top_w = max(weights.items(), key=lambda kv: kv[1])
        top_weight = round(top_w, 4)

    # Correlation-adjusted effective independent bets:
    #   n_eff = n / (1 + (n-1)·mean_corr), clamped to [1, n].
    # mean_corr→1 ⇒ 1 bet however many tickers; →0 ⇒ n bets. n counts only
    # the correlatable names (those with a usable, aligned series).
    n_corr = len(usable)
    eff_bets = None
    if n_corr >= 1 and mean_corr is not None:
        denom = 1.0 + (n_corr - 1) * mean_corr
        if denom > 1e-9:
            eff_bets = round(max(1.0, min(float(n_corr), n_corr / denom)), 4)
    elif n_corr == 1:
        eff_bets = 1.0

    # ---- state / verdict ----------------------------------------------
    if not stock_pos:
        state, verdict = "NO_DATA", None
    elif n_corr < 2 or mean_corr is None:
        state, verdict = "INSUFFICIENT", None
    else:
        state = "OK"
        if top_weight is not None and top_weight >= DOMINANT_WEIGHT:
            verdict = "SINGLE_NAME_RISK"
        elif mean_corr >= HIGH_CORR:
            verdict = "CONCENTRATED"
        elif mean_corr >= MOD_CORR:
            verdict = "MODERATE"
        else:
            verdict = "DIVERSIFIED"

    # ---- headline ------------------------------------------------------
    if state == "NO_DATA":
        headline = "No stock positions — concentration risk undefined."
    elif state == "INSUFFICIENT":
        if n_corr < 2:
            headline = (
                f"Only {n_corr} correlatable stock name(s) "
                f"(need ≥2 with ≥{MIN_RETURNS} aligned daily returns) — "
                f"correlation verdict withheld.")
        else:
            headline = ("Held names have no overlapping return history yet — "
                        "correlation verdict withheld.")
    else:
        eff_clause = (f"{eff_bets:.2f} effective independent bet(s) across "
                      f"{n_corr} correlatable name(s)"
                      if eff_bets is not None else
                      f"{n_corr} correlatable name(s)")
        pair_clause = ""
        if max_pair is not None:
            pair_clause = (f" Most-coupled pair "
                           f"{max_pair['tickers'][0]}/"
                           f"{max_pair['tickers'][1]} ρ={max_pair['corr']:+.2f}.")
        if verdict == "SINGLE_NAME_RISK":
            headline = (
                f"SINGLE_NAME_RISK — {top_ticker} is {top_weight * 100:.0f}% "
                f"of the book; {eff_clause}, mean ρ={mean_corr:+.2f}."
                + pair_clause)
        elif verdict == "CONCENTRATED":
            headline = (
                f"CONCENTRATED — the book moves as one (mean ρ="
                f"{mean_corr:+.2f}); {eff_clause}. A correlated drawdown "
                f"hits the whole book." + pair_clause)
        elif verdict == "MODERATE":
            headline = (
                f"MODERATE — partial diversification (mean ρ="
                f"{mean_corr:+.2f}); {eff_clause}." + pair_clause)
        else:
            headline = (
                f"DIVERSIFIED — names move largely independently (mean ρ="
                f"{mean_corr:+.2f}); {eff_clause}." + pair_clause)

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "verdict": verdict,
        "headline": headline,
        "n_stock_positions": len(stock_pos),
        "n_correlatable": n_corr,
        "mean_pairwise_corr": mean_corr,
        "max_pair": max_pair,
        "pairs": pairs,
        "weight_hhi": hhi,
        "effective_positions_naive": eff_naive,
        "effective_independent_bets": eff_bets,
        "top_weight_pct": (round(top_weight * 100.0, 2)
                           if top_weight is not None else None),
        "top_weight_ticker": top_ticker,
        "weights": weights,
        "skipped_options": [t for t in skipped_options if t is not None],
        "short_series_tickers": short_series,
        "min_returns": MIN_RETURNS,
    }


if __name__ == "__main__":  # smoke test against the live DB + yfinance
    import json

    import yfinance as yf

    from paper_trader.store import get_store
    s = get_store()
    pf = s.get_portfolio()
    poss = [{"ticker": p["ticker"],
             "market_value": p.get("market_value") or 0.0,
             "type": p.get("option_type") or "stock"}
            for p in pf.get("positions", [])]
    hist: dict = {}
    for p in poss:
        try:
            h = yf.Ticker(p["ticker"]).history(period="3mo")
            hist[p["ticker"]] = [float(x) for x in h["Close"].tolist()]
        except Exception:
            hist[p["ticker"]] = []
    print(json.dumps(build_correlation(poss, hist), indent=2, default=str))
