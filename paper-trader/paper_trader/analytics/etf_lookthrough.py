"""ETF look-through: pierce leveraged-ETF positions into effective single-name
exposure (2026-05-19).

Every existing risk surface stops at the ticker boundary: ``sector_exposure``
classifies TQQQ as ``broad_lev`` and adds the full position to that sector;
``stress_scenarios`` β-amplifies the sector for an SPY shock; ``risk_mirror``
reports name-level HHI on the *line-item* tickers. None of them answer the
single most decision-relevant question a leveraged-ETF book begs:

    *"I hold $447 NVDA cash AND $148 TQQQ. TQQQ is 3x QQQ, and QQQ is ~9%
    NVDA. What is my TRUE effective NVDA exposure?"*

A back-of-envelope: ``$148 × 3 × 0.09 = ~$40`` of indirect NVDA exposure on
top of the $447 direct = $487 effective NVDA, not $447. On a 3-position
book that already reads 44% NVDA the look-through reads materially higher.
The error grows when a book stacks TQQQ + SOXL + FNGU — three "different"
tickers that all amplify NVDA.

This builder is the missing lens.

For each held position:
  * If the ticker is in ``_ETF_LOOKTHROUGH``: decompose into virtual
    single-name exposures (``position_value × leverage × weight%``). Inverse
    ETFs (SQQQ/SOXS/SPXS/FNGD/TECS) carry a NEGATIVE leverage factor — they
    SHORT their underlyings, so holding NVDA + SQQQ has effective NVDA =
    direct − indirect_from_SQQQ. That sign honesty IS the value-add.
  * Else: contribute its full market value to the underlying ticker as
    *direct* exposure.

Per underlying ticker we then surface ``direct_usd``, ``indirect_usd`` (sum
of all look-through contributions, signed), and ``effective_usd =
direct + indirect``. Sorted by ``abs(effective_usd)`` DESC so the largest
TRUE bet surfaces first. ``effective_pct = effective_usd / total_value *
100`` is the % of book the underlying truly represents.

The headline calls out **hidden concentration**: if any underlying's
``effective_pct`` exceeds ``HIDDEN_RATIO`` × its ``direct_pct``
(default 1.5x — i.e. look-through reveals ≥50% more exposure than the
line items suggest), the operator is silently more concentrated than the
``risk_mirror``/``sector_exposure`` panels show.

Design parity with the codebase
-------------------------------

* **Pure builder, no I/O.** Same contract as
  ``sector_exposure`` / ``pnl_attribution`` / ``stress_scenarios``: the
  endpoint owns any network/store reads (there are none — positions
  passed in). Never raises (the ``_safe`` discipline); a garbage row
  contributes nothing, a missing total_value degrades to ``NO_DATA``.

* **Position value mirrors ``sector_exposure`` verbatim**:
  ``(current_price or avg_cost) * qty * (100 if option else 1)``. Same
  ``analytics_api`` formula — so a book reading 22% TQQQ in
  ``sector_exposure`` and a look-through that consumes the same dollar
  basis can never disagree on starting weights.

* **Options skipped from look-through** (NOT skipped entirely — they
  still contribute to direct exposure if the option's ticker is itself
  an underlying we already know about, e.g. an NVDA call). Look-through
  on an option-on-an-ETF requires delta-adjustment that's its own
  surface; this builder is for STOCK ETF holdings.

* **Observational only**, never gates Opus, never injected into the
  decision prompt, no caps (AGENTS.md invariants #2/#12 — the
  ``risk_mirror`` / ``stress_scenarios`` / ``recovery`` precedent).

State ladder
------------

* ``NO_DATA``     — no priced book / total_value ≤ 0.
* ``NO_ETF_HELD`` — book exists but contains no leveraged ETF in the
  look-through map; look-through == direct, no hidden risk to reveal.
  Suppressed in reports (the ``decision_drought``/``no_decision_recovery``
  silence-when-nothing-actionable precedent).
* ``OK``          — at least one ETF held; look-through populated.

Honesty about the static map
----------------------------

``_ETF_LOOKTHROUGH`` weights are approximate top-10 constituents of each
ETF (sources: issuer fact sheets late-2025). The headline does not pretend
the weights are real-time — they are decision support, not VaR. Holdings
drift quarterly; a constituent weight off by 1-2pp does not change the
"hidden concentration" verdict, which is what this surface exists to
flag.
"""
from __future__ import annotations

_OPTION_TYPES = {"call", "put"}

# Threshold above which look-through reveals "meaningfully more" exposure
# than the line items show — tuned to flag a real pathology (TQQQ + NVDA
# pushing NVDA past 1.5× its direct weight) without firing on every
# single-ETF book where the look-through is by construction additive.
HIDDEN_RATIO = 1.5
# Cap the surfaced underlyings to keep the report compact (the
# ``trade_attribution`` / ``idle_opportunity`` precedent — top-N by impact).
_MAX_UNDERLYINGS = 12

# ── ETF look-through weights ────────────────────────────────────────────
# Per-ETF: ``{"leverage": signed_factor, "holdings": [(underlying, pct), …]}``.
# Inverse ETFs use a NEGATIVE leverage (they short the underlying basket).
# Weights are approximate top constituents — close to real but not pinned
# to a daily issuer feed (that would be a separate maintenance surface).
# Pct values do NOT need to sum to 100 — uncovered residual is treated as
# diffuse market exposure and dropped (the headline reads "of *known*
# look-through" — never claims completeness).
_ETF_LOOKTHROUGH: dict = {
    # QQQ family (Nasdaq-100). TQQQ is 3x, SQQQ is -3x.
    "TQQQ": {
        "leverage": 3.0,
        "holdings": [
            ("NVDA", 9.0), ("MSFT", 9.0), ("AAPL", 8.5),
            ("AMZN", 5.5), ("META", 5.0), ("AVGO", 5.0),
            ("GOOGL", 3.0), ("GOOG", 3.0), ("TSLA", 3.0),
            ("COST", 2.5), ("NFLX", 2.0),
        ],
    },
    "SQQQ": {
        "leverage": -3.0,
        "holdings": [
            ("NVDA", 9.0), ("MSFT", 9.0), ("AAPL", 8.5),
            ("AMZN", 5.5), ("META", 5.0), ("AVGO", 5.0),
            ("GOOGL", 3.0), ("GOOG", 3.0), ("TSLA", 3.0),
            ("COST", 2.5), ("NFLX", 2.0),
        ],
    },
    "QQQ": {
        "leverage": 1.0,
        "holdings": [
            ("NVDA", 9.0), ("MSFT", 9.0), ("AAPL", 8.5),
            ("AMZN", 5.5), ("META", 5.0), ("AVGO", 5.0),
            ("GOOGL", 3.0), ("GOOG", 3.0), ("TSLA", 3.0),
            ("COST", 2.5), ("NFLX", 2.0),
        ],
    },
    # Semis (SOXX). SOXL 3x / SOXS -3x.
    "SOXL": {
        "leverage": 3.0,
        "holdings": [
            ("NVDA", 10.0), ("AVGO", 9.0), ("TSM", 9.0),
            ("AMD", 7.0), ("AMAT", 5.0), ("ASML", 5.0),
            ("MU", 5.0), ("LRCX", 4.5), ("KLAC", 4.5),
            ("INTC", 4.0), ("MRVL", 3.5),
        ],
    },
    "SOXS": {
        "leverage": -3.0,
        "holdings": [
            ("NVDA", 10.0), ("AVGO", 9.0), ("TSM", 9.0),
            ("AMD", 7.0), ("AMAT", 5.0), ("ASML", 5.0),
            ("MU", 5.0), ("LRCX", 4.5), ("KLAC", 4.5),
            ("INTC", 4.0), ("MRVL", 3.5),
        ],
    },
    "SOXX": {
        "leverage": 1.0,
        "holdings": [
            ("NVDA", 10.0), ("AVGO", 9.0), ("TSM", 9.0),
            ("AMD", 7.0), ("AMAT", 5.0), ("ASML", 5.0),
            ("MU", 5.0), ("LRCX", 4.5), ("KLAC", 4.5),
            ("INTC", 4.0), ("MRVL", 3.5),
        ],
    },
    "SMH": {  # 1x semis ETF — concentrated, NVDA-heavy.
        "leverage": 1.0,
        "holdings": [
            ("NVDA", 21.0), ("AVGO", 9.0), ("TSM", 9.0),
            ("AMD", 5.0), ("AMAT", 5.0), ("ASML", 5.0),
            ("MU", 5.0), ("LRCX", 4.0), ("KLAC", 4.0),
            ("INTC", 4.0),
        ],
    },
    # FANG+ (equal-weighted 10). FNGU 3x / FNGD -3x.
    "FNGU": {
        "leverage": 3.0,
        "holdings": [
            ("NVDA", 10.0), ("AAPL", 10.0), ("MSFT", 10.0),
            ("META", 10.0), ("AMZN", 10.0), ("GOOGL", 10.0),
            ("TSLA", 10.0), ("NFLX", 10.0), ("AMD", 10.0),
            ("MU", 10.0),
        ],
    },
    "FNGD": {
        "leverage": -3.0,
        "holdings": [
            ("NVDA", 10.0), ("AAPL", 10.0), ("MSFT", 10.0),
            ("META", 10.0), ("AMZN", 10.0), ("GOOGL", 10.0),
            ("TSLA", 10.0), ("NFLX", 10.0), ("AMD", 10.0),
            ("MU", 10.0),
        ],
    },
    # XLK family (US tech). TECL 3x / TECS -3x.
    "TECL": {
        "leverage": 3.0,
        "holdings": [
            ("AAPL", 14.0), ("MSFT", 13.0), ("NVDA", 12.0),
            ("AVGO", 4.5), ("ORCL", 3.0), ("CRM", 2.5),
            ("AMD", 2.5), ("CSCO", 2.0), ("ACN", 2.0),
        ],
    },
    "TECS": {
        "leverage": -3.0,
        "holdings": [
            ("AAPL", 14.0), ("MSFT", 13.0), ("NVDA", 12.0),
            ("AVGO", 4.5), ("ORCL", 3.0), ("CRM", 2.5),
            ("AMD", 2.5), ("CSCO", 2.0), ("ACN", 2.0),
        ],
    },
    # S&P 500 family. SPXL 3x / SPXS -3x / UPRO 3x. Same underlying.
    "SPXL": {
        "leverage": 3.0,
        "holdings": [
            ("AAPL", 6.5), ("MSFT", 6.5), ("NVDA", 6.0),
            ("AMZN", 3.5), ("META", 2.5), ("GOOGL", 2.0),
            ("GOOG", 2.0), ("AVGO", 1.8), ("TSLA", 1.5),
            ("BRK.B", 1.5),
        ],
    },
    "UPRO": {
        "leverage": 3.0,
        "holdings": [
            ("AAPL", 6.5), ("MSFT", 6.5), ("NVDA", 6.0),
            ("AMZN", 3.5), ("META", 2.5), ("GOOGL", 2.0),
            ("GOOG", 2.0), ("AVGO", 1.8), ("TSLA", 1.5),
            ("BRK.B", 1.5),
        ],
    },
    "SPXS": {
        "leverage": -3.0,
        "holdings": [
            ("AAPL", 6.5), ("MSFT", 6.5), ("NVDA", 6.0),
            ("AMZN", 3.5), ("META", 2.5), ("GOOGL", 2.0),
            ("GOOG", 2.0), ("AVGO", 1.8), ("TSLA", 1.5),
            ("BRK.B", 1.5),
        ],
    },
    "SPY": {
        "leverage": 1.0,
        "holdings": [
            ("AAPL", 6.5), ("MSFT", 6.5), ("NVDA", 6.0),
            ("AMZN", 3.5), ("META", 2.5), ("GOOGL", 2.0),
            ("GOOG", 2.0), ("AVGO", 1.8), ("TSLA", 1.5),
            ("BRK.B", 1.5),
        ],
    },
    # QLD/SSO 2x QQQ/SPY — same baskets, lower leverage.
    "QLD": {
        "leverage": 2.0,
        "holdings": [
            ("NVDA", 9.0), ("MSFT", 9.0), ("AAPL", 8.5),
            ("AMZN", 5.5), ("META", 5.0), ("AVGO", 5.0),
            ("GOOGL", 3.0), ("GOOG", 3.0), ("TSLA", 3.0),
            ("COST", 2.5),
        ],
    },
    "SSO": {
        "leverage": 2.0,
        "holdings": [
            ("AAPL", 6.5), ("MSFT", 6.5), ("NVDA", 6.0),
            ("AMZN", 3.5), ("META", 2.5), ("GOOGL", 2.0),
            ("GOOG", 2.0), ("AVGO", 1.8), ("TSLA", 1.5),
        ],
    },
}


def _f(x, default: float = 0.0) -> float:
    """Float coercion; garbage degrades to ``default``, never raises
    (the ``_safe`` discipline; identical to ``sector_exposure._f``)."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _z(v, ndigits: int = 2):
    """Round; fold ``-0.0 → 0.0``; None / non-numeric → None
    (the ``pnl_attribution._z`` precedent)."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _position_value(p: dict) -> float:
    """``(current_price or avg_cost) * qty * (100 if option else 1)`` —
    the ``analytics_api`` formula, mirrored verbatim from
    ``sector_exposure._position_value`` (single source of truth #10)."""
    mult = 100 if p.get("type") in _OPTION_TYPES else 1
    price = _f(p.get("current_price")) or _f(p.get("avg_cost"))
    return price * _f(p.get("qty")) * mult


def _safe_ticker(t) -> str:
    """Normalize a ticker to uppercase string; ``None``/garbage → empty
    so the falsy check upstream drops it (never KeyErrors the map lookup)."""
    try:
        s = str(t or "").strip().upper()
        return s
    except Exception:
        return ""


def _exposure_state(direct_pct: float, effective_pct: float) -> str:
    """Classify an underlying's hidden-concentration tier. Used both for
    sorting and for the headline tag. The 1.5× floor is the
    ``HIDDEN_RATIO`` constant; an effective-pct dominated by indirect
    flow when direct is ~0 is HIDDEN regardless of ratio (a zero
    denominator can't be meaningfully scaled)."""
    abs_eff = abs(effective_pct)
    if abs_eff < 0.5:
        return "TRIVIAL"
    if abs(direct_pct) < 0.1:
        # All-indirect: an ETF-only exposure with no direct line item.
        # Flag as HIDDEN_ONLY so the operator sees the underlying they
        # never explicitly bought.
        return "HIDDEN_ONLY"
    ratio = abs_eff / abs(direct_pct)
    if ratio >= HIDDEN_RATIO:
        return "HIDDEN_AMPLIFIED"
    return "TRANSPARENT"


def build_etf_lookthrough(
    snapshot: dict,
    *,
    lookthrough_map: dict | None = None,
    hidden_ratio: float = HIDDEN_RATIO,
    max_underlyings: int = _MAX_UNDERLYINGS,
) -> dict:
    """Compose the ETF look-through panel.

    ``snapshot`` — the ``strategy._portfolio_snapshot`` dict (``cash``,
    ``total_value``, enriched ``positions``).

    ``lookthrough_map`` — override for tests; defaults to the module-level
    ``_ETF_LOOKTHROUGH`` table.

    Returns ``{state, headline, total_value, n_etfs_held, etf_positions,
    underlyings, hidden_ratio, as_of}``. Pure; never raises (the
    ``_safe`` contract).
    """
    try:
        lm = lookthrough_map if lookthrough_map is not None else _ETF_LOOKTHROUGH
        snap = snapshot or {}
        total = _f(snap.get("total_value"))
        positions = list(snap.get("positions") or [])

        if total <= 0 or not positions:
            return {
                "state": "NO_DATA",
                "headline": "no priced book to look through",
                "total_value": _z(total) or 0.0,
                "n_etfs_held": 0,
                "etf_positions": [],
                "underlyings": [],
                "hidden_ratio": hidden_ratio,
            }

        # Direct exposure per ticker (every position, ETF or not).
        direct_usd: dict = {}
        # Indirect (look-through) exposure per underlying ticker, signed.
        indirect_usd: dict = {}
        # ETF lines we actually looked through, for the report.
        etf_positions: list = []

        for p in positions:
            if not isinstance(p, dict):
                continue
            tk = _safe_ticker(p.get("ticker"))
            if not tk:
                continue
            val = _position_value(p)
            if val == 0.0:
                continue
            # Direct contribution (always — even an ETF line item is a
            # direct position in that ETF).
            direct_usd[tk] = direct_usd.get(tk, 0.0) + val

            # Look-through only applies to non-option ETF positions.
            if p.get("type") in _OPTION_TYPES:
                continue
            etf = lm.get(tk)
            if not etf:
                continue
            try:
                lev = float(etf.get("leverage", 1.0))
            except (TypeError, ValueError):
                lev = 1.0
            holdings = etf.get("holdings") or []
            etf_breakdown: list = []
            for h in holdings:
                try:
                    u_tk = _safe_ticker(h[0])
                    weight = float(h[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if not u_tk:
                    continue
                # virtual exposure = position_value × leverage × weight%
                v = val * lev * (weight / 100.0)
                indirect_usd[u_tk] = indirect_usd.get(u_tk, 0.0) + v
                etf_breakdown.append({
                    "underlying": u_tk,
                    "weight_pct": _z(weight, 2),
                    "indirect_usd": _z(v, 2),
                })
            etf_positions.append({
                "ticker": tk,
                "position_usd": _z(val, 2),
                "leverage": _z(lev, 2),
                "n_holdings_mapped": len(etf_breakdown),
                "breakdown": etf_breakdown,
            })

        if not etf_positions:
            # No ETF in our map is held → look-through is identical to direct;
            # no hidden risk to reveal. Suppress in reports.
            return {
                "state": "NO_ETF_HELD",
                "headline": ("no leveraged ETF in book — look-through "
                             "matches line items"),
                "total_value": _z(total, 2) or 0.0,
                "n_etfs_held": 0,
                "etf_positions": [],
                "underlyings": [],
                "hidden_ratio": hidden_ratio,
            }

        # Compose per-underlying rows. Underlyings include both direct and
        # indirect tickers — a ticker held only via an ETF (no direct line)
        # still surfaces as HIDDEN_ONLY.
        all_tickers = set(direct_usd) | set(indirect_usd)
        rows: list = []
        for u in all_tickers:
            d_usd = direct_usd.get(u, 0.0)
            i_usd = indirect_usd.get(u, 0.0)
            eff_usd = d_usd + i_usd
            d_pct = d_usd / total * 100.0
            i_pct = i_usd / total * 100.0
            eff_pct = eff_usd / total * 100.0
            tier = _exposure_state(d_pct, eff_pct)
            rows.append({
                "ticker": u,
                "direct_usd": _z(d_usd, 2),
                "indirect_usd": _z(i_usd, 2),
                "effective_usd": _z(eff_usd, 2),
                "direct_pct": _z(d_pct, 2),
                "indirect_pct": _z(i_pct, 2),
                "effective_pct": _z(eff_pct, 2),
                "tier": tier,
            })

        # Sort by |effective_usd| DESC — the largest TRUE bet first;
        # tie-break by direct_usd DESC (a direct position outranks a
        # same-magnitude indirect-only one for the operator's read).
        rows.sort(key=lambda r: (
            -abs(_f(r["effective_usd"])),
            -_f(r["direct_usd"]),
            r["ticker"],
        ))
        rows = rows[:max_underlyings]

        # Headline: the loudest hidden-amplified or hidden-only line.
        # Falls back to the largest effective exposure when nothing is hidden.
        headline = _build_headline(rows, hidden_ratio)

        return {
            "state": "OK",
            "headline": headline,
            "total_value": _z(total, 2) or 0.0,
            "n_etfs_held": len(etf_positions),
            "etf_positions": etf_positions,
            "underlyings": rows,
            "hidden_ratio": hidden_ratio,
        }
    except Exception:
        # The _safe contract — any unexpected fault degrades to a single
        # honest line; never propagates.
        return {
            "state": "NO_DATA",
            "headline": "look-through fault — no panel this cycle",
            "total_value": 0.0,
            "n_etfs_held": 0,
            "etf_positions": [],
            "underlyings": [],
            "hidden_ratio": hidden_ratio,
        }


def _build_headline(rows: list, hidden_ratio: float) -> str:
    """Pick the most decision-relevant single line for the headline.

    Priority: HIDDEN_AMPLIFIED with largest |effective_pct| → HIDDEN_ONLY
    largest → largest effective exposure. The ``recovery``/``stress``
    headline pattern: one sentence, no fabricated precision.
    """
    if not rows:
        return "no underlying exposure to report"

    # Search HIDDEN_AMPLIFIED first; the operator-most-actionable signal.
    for r in rows:
        if r["tier"] == "HIDDEN_AMPLIFIED":
            ticker = r["ticker"]
            eff = r["effective_pct"]
            direct = r["direct_pct"]
            return (f"hidden concentration: {ticker} effective "
                    f"{eff:+.1f}% of book vs {direct:+.1f}% direct "
                    f"(amplified ≥{hidden_ratio:.1f}× via ETF look-through)")

    for r in rows:
        if r["tier"] == "HIDDEN_ONLY":
            ticker = r["ticker"]
            eff = r["effective_pct"]
            return (f"silent exposure: {ticker} {eff:+.1f}% of book "
                    f"via ETF holdings, no direct line")

    # Nothing hidden — surface the largest TRUE bet so the operator sees
    # what's at the top of their effective book.
    top = rows[0]
    return (f"largest effective bet: {top['ticker']} "
            f"{top['effective_pct']:+.1f}% of book "
            f"(direct {top['direct_pct']:+.1f}%, "
            f"indirect {top['indirect_pct']:+.1f}%)")
