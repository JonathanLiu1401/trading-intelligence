"""Inverse-pair conflict detector — flag simultaneously held leveraged-long
+ leveraged-inverse ETFs of the same underlying family.

Why this lens. The live ``WATCHLIST`` is explicitly leveraged-ETF heavy
(``TQQQ`` / ``SQQQ`` / ``SOXL`` / ``SOXS`` / ``SPXL`` / ``SPXS`` /
``FNGU`` / ``FNGD`` / ``TECL`` / ``TECS`` …). When Opus opens both sides
of the same underlying family — e.g. TQQQ AND SQQQ — the directional
exposure largely cancels but the book continues paying the leverage
decay on BOTH sides. That carry-waste pathology is invisible to every
existing risk surface on this desk:

* ``etf_lookthrough`` computes net effective single-name exposure
  (NVDA-equivalent after the SQQQ short-sleeve cancels part of the TQQQ
  long-sleeve) — but it reports the *outcome*, not the *fact* that the
  operator is paying two decay tabs to express a near-zero bet.
* ``regime_leverage_fit_skill`` checks whether the book's leveraged %
  matches the SPY-momentum regime — but a fully-paired TQQQ+SQQQ book
  reads "high leveraged %" without distinguishing it from a clean
  one-sided leveraged bet.
* ``sector_exposure`` puts both into the same ``broad_lev`` sector with
  no opposing-sign awareness.
* ``correlation_cluster_warning`` flags a high-corr cluster of similar-
  direction names but TQQQ and SQQQ are NEGATIVELY correlated and so
  pass through unflagged.

A discretionary PM would never simultaneously hold a 3x long and a 3x
inverse on the same underlying — the carry is wrong direction, the
delta is nearly zero, and any catalyst gets paid mostly to the issuer
in management fees + volatility decay. This builder is the missing
mirror.

Pure & offline. No DB read, no network. Walks open positions, projects
each ticker onto the static ``_PAIR_FAMILIES`` map (derived from
``etf_lookthrough._ETF_LOOKTHROUGH``'s family groupings — single source
of truth), and flags any family that has at least one long-leveraged
position alongside at least one inverse-leveraged position. Identical
``_safe`` discipline to ``etf_lookthrough`` / ``sector_exposure``: a
garbage row contributes nothing, an empty book degrades to a
deterministic ``NO_BOOK`` verdict. Never raises.

Advisory only. It mints no directive, imposes no cap, and has no path
to ``_execute()`` (AGENTS.md invariants #2 / #12 — Opus has full
autonomy; this is a mirror, not a cage).

Verdict ladder (threshold-driven, exactly testable):

| Verdict               | Meaning                                                                                       |
|-----------------------|-----------------------------------------------------------------------------------------------|
| ``NO_BOOK``           | No open stock positions — nothing to evaluate                                                 |
| ``CLEAN``             | No family has both a long-leveraged AND an inverse-leveraged position                         |
| ``OPPOSING_UNLEVERED`` | At least one family has the 1x core (QQQ/SPY/SMH) plus the leveraged inverse — directional   |
|                        | offset, no leverage decay drag                                                                |
| ``CARRY_WASTE``       | At least one family has BOTH a leveraged-long ETF AND a leveraged-inverse ETF — both decay    |

Per-conflicting-family report:

* ``family``, ``family_label``                          — taxonomy key + human label
* ``long_holdings``, ``inverse_holdings``, ``core_holdings``  — ``[(ticker, usd, leverage)]``
* ``long_notional_usd``, ``inverse_notional_usd``       — raw dollars on each side
* ``long_delta_usd``, ``inverse_delta_usd``             — signed-leverage-applied dollar deltas
* ``cancelled_delta_usd``                               — min(|long_delta|, |inverse_delta|) — the
                                                          dollar-delta that nets to zero
* ``net_delta_usd``                                     — signed residual delta (long_delta + inverse_delta)
* ``daily_drag_estimate_usd``                           — practitioner ballpark daily decay drag on
                                                          BOTH sleeves combined (leverage costs are
                                                          paid regardless of whether the deltas cancel)
* ``severity``                                          — ``HIGH`` (cancelled ≥ 80% of gross),
                                                          ``MEDIUM`` (40-80%), ``LOW`` (< 40%)
"""
from __future__ import annotations

from typing import Any

_OPTION_TYPES = ("call", "put")

# Family taxonomy. Derived from ``etf_lookthrough._ETF_LOOKTHROUGH``
# (single source of truth on which tickers belong to which family + their
# leverage signs). Each family separates ``long`` (positive-leverage 2x/3x),
# ``inverse`` (negative-leverage), and ``core`` (1x baseline). 1x cores DO
# count for opposing-direction detection (holding QQQ + SQQQ has a delta
# cancel, just no extra leverage decay), but they don't escalate to
# CARRY_WASTE on their own — only a leveraged long + leveraged inverse
# does, since both sides then pay the issuer to decay against each other.
_PAIR_FAMILIES: dict[str, dict[str, Any]] = {
    "QQQ": {
        "label": "Nasdaq-100 (QQQ family)",
        "long": {"TQQQ": 3.0, "QLD": 2.0},
        "core": {"QQQ": 1.0},
        "inverse": {"SQQQ": -3.0},
    },
    "SEMIS": {
        "label": "Semis (SOXX family)",
        "long": {"SOXL": 3.0, "USD": 2.0},
        "core": {"SMH": 1.0, "SOXX": 1.0},
        "inverse": {"SOXS": -3.0, "SSG": -2.0},
    },
    "SP500": {
        "label": "S&P 500 (SPY family)",
        "long": {"SPXL": 3.0, "UPRO": 3.0, "SSO": 2.0},
        "core": {"SPY": 1.0, "IVV": 1.0, "VOO": 1.0},
        "inverse": {"SPXS": -3.0, "SPXU": -3.0, "SDS": -2.0},
    },
    "FANG": {
        "label": "FANG+ (FNGU/FNGD)",
        "long": {"FNGU": 3.0},
        "core": {},
        "inverse": {"FNGD": -3.0},
    },
    "USTECH": {
        "label": "US Tech (XLK family)",
        "long": {"TECL": 3.0},
        "core": {"XLK": 1.0},
        "inverse": {"TECS": -3.0},
    },
    "RUSSELL": {
        "label": "Russell 2000 (IWM family)",
        "long": {"TNA": 3.0, "URTY": 3.0, "UWM": 2.0},
        "core": {"IWM": 1.0},
        "inverse": {"TZA": -3.0, "SRTY": -3.0},
    },
}

# Approximate daily volatility decay (basis points) per side of a leveraged
# ETF — conservative practitioner estimates. NOT a precision claim — the
# headline reads "estimated" / "approx" so the operator treats this as a
# directional cost ballpark, not a P&L attribution. Values are intentionally
# round so the carry_drag math reads cleanly in tests.
_DAILY_DRAG_BPS = {
    3.0: 6.0,   # 3x ETFs typically lose 5-8 bps/day to vol decay on a ~1.5% vol underlying
    2.0: 3.0,   # 2x ETFs see roughly half the drag
    1.0: 0.0,   # 1x baseline ETFs have no leverage-decay tab
}

# Severity thresholds — cancelled fraction of gross-leveraged notional.
SEVERITY_HIGH_PCT = 80.0   # ≥80% of the gross-leveraged $ is cancelling itself out
SEVERITY_MEDIUM_PCT = 40.0  # 40-80% — meaningful but not pathological

# Ticker → (family_key, role, leverage). Built at import time from the
# family map so any drift between the family taxonomy and the lookup
# table is impossible by construction. Role ∈ {"long", "inverse", "core"}.
_TICKER_INDEX: dict[str, tuple[str, str, float]] = {}
for _fam_key, _fam in _PAIR_FAMILIES.items():
    for _role in ("long", "inverse", "core"):
        for _tkr, _lev in _fam[_role].items():
            _TICKER_INDEX[_tkr] = (_fam_key, _role, _lev)


def _f(x, default: float = 0.0) -> float:
    """Float coercion; garbage degrades to ``default``, never raises
    (the ``_safe`` discipline; mirror of ``etf_lookthrough._f``)."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _z(v, ndigits: int = 2):
    """Round; fold ``-0.0 → 0.0``; None / non-numeric → None
    (the ``pnl_attribution._z`` / ``etf_lookthrough._z`` precedent)."""
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
    ``etf_lookthrough._position_value`` (SSOT)."""
    ptype = (p.get("type") or "stock").lower()
    mult = 100 if ptype in _OPTION_TYPES else 1
    price = _f(p.get("current_price")) or _f(p.get("avg_cost"))
    return price * _f(p.get("qty")) * mult


def _severity(cancelled_delta: float, gross_delta: float) -> str:
    """Cancelled fraction of total gross delta → severity bucket.

    ``cancelled_delta`` = min(|long_delta|, |inverse_delta|)
    ``gross_delta``    = |long_delta| + |inverse_delta|

    A perfect cancel (long_delta == -inverse_delta) reaches 50% of gross
    on each side ⇒ ratio = cancelled/gross = 50%. We surface the ratio
    as a percentage of the cancelled side (cancelled / max(long, inv))
    so a 99-99 paired book lands at ~100% (HIGH) — the more intuitive
    "what fraction of the smaller side is cancelling" framing.
    """
    if gross_delta <= 0:
        return "LOW"
    # cancelled = min(|long_delta|, |inverse_delta|). The informative ratio
    # is ``cancelled / max(|long|, |inverse|)`` — what fraction of the
    # bigger side gets neutralised by the opposing sleeve. Since
    # ``gross = |long| + |inverse|`` and ``cancelled = min(|long|, |inverse|)``,
    # ``max = gross - cancelled``.
    bigger = gross_delta - cancelled_delta
    if bigger <= 0:
        bigger = cancelled_delta  # both sides equal — fully cancelled
    pct_of_bigger = (cancelled_delta / bigger) * 100.0 if bigger > 0 else 0.0
    if pct_of_bigger >= SEVERITY_HIGH_PCT:
        return "HIGH"
    if pct_of_bigger >= SEVERITY_MEDIUM_PCT:
        return "MEDIUM"
    return "LOW"


def _daily_drag_usd(holdings: list[tuple[str, float, float]]) -> float:
    """Practitioner-estimate daily decay drag in dollars across the given
    leveraged sleeves. Sums ``|usd| × bps(|leverage|) / 10000``. 1x cores
    contribute 0 (no leverage decay). Returns 0.0 on empty / malformed.
    """
    drag = 0.0
    for _tkr, usd, lev in holdings:
        bps = _DAILY_DRAG_BPS.get(abs(lev), 0.0)
        drag += abs(usd) * bps / 10000.0
    return drag


def build_inverse_pair_conflict(positions: Any) -> dict:
    """Pure builder. ``positions`` is the open-stock list (same shape as
    ``etf_lookthrough`` / ``sector_exposure``). Returns a deterministic
    report with the verdict ladder above. Never raises.
    """
    # Bucket every stock position into its family slot. Options are
    # excluded (a TQQQ call has an inherent leverage signature that's
    # delta-dependent — handling that requires a full greeks pass which
    # is out of scope; same convention as ``persona_book_fit``).
    family_holdings: dict[str, dict[str, list[tuple[str, float, float]]]] = {}
    n_book_positions = 0
    total_book_usd = 0.0
    for p in positions or []:
        if not isinstance(p, dict):
            continue
        ticker = (p.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        ptype = (p.get("type") or "stock").lower()
        if ptype in _OPTION_TYPES:
            continue
        usd = _position_value(p)
        if usd <= 0:
            continue
        n_book_positions += 1
        total_book_usd += usd
        if ticker not in _TICKER_INDEX:
            continue
        fam_key, role, lev = _TICKER_INDEX[ticker]
        bucket = family_holdings.setdefault(
            fam_key, {"long": [], "inverse": [], "core": []}
        )
        bucket[role].append((ticker, usd, lev))

    if n_book_positions == 0:
        return {
            "state": "NO_BOOK",
            "verdict": "NO_BOOK",
            "headline": "no open stock positions — no inverse-pair conflict to evaluate",
            "n_book_positions": 0,
            "total_book_usd": _z(0.0),
            "conflicts": [],
            "n_conflicts": 0,
            "total_cancelled_delta_usd": _z(0.0),
            "total_daily_drag_usd": _z(0.0),
        }

    # Per-family conflict eval. A conflict requires at least one position
    # with a non-zero leverage sign opposite to another. The headline
    # CARRY_WASTE escalation requires BOTH sides to be leveraged (long set
    # OR inverse set, both populated with ≥2x). OPPOSING_UNLEVERED fires
    # when the only conflict involves a 1x core (e.g. SPY + SPXS).
    conflicts = []
    any_carry_waste = False
    any_opposing_unlevered = False

    for fam_key, bucket in family_holdings.items():
        longs = bucket["long"]
        inverses = bucket["inverse"]
        cores = bucket["core"]
        has_long = bool(longs)
        has_inverse = bool(inverses)
        has_core = bool(cores)

        if not (has_inverse and (has_long or has_core)):
            # No directional conflict in this family — either no inverse
            # ETF, or no positive-direction holding at all. Either way,
            # nothing to flag.
            continue
        # The 1x-core-vs-inverse case: SPY + SPXS — delta cancel but
        # the leveraged tab is paid only on the SPXS sleeve. Worth
        # surfacing distinctly from the leveraged-vs-leveraged case
        # which pays BOTH tabs.
        if has_inverse and has_long:
            any_carry_waste = True
            classification = "CARRY_WASTE"
        else:
            any_opposing_unlevered = True
            classification = "OPPOSING_UNLEVERED"

        long_notional = sum(usd for _t, usd, _l in longs)
        inverse_notional = sum(usd for _t, usd, _l in inverses)
        core_notional = sum(usd for _t, usd, _l in cores)
        # Signed-leverage deltas. ``long_delta`` is positive (positive lev),
        # ``inverse_delta`` is negative (negative lev). The cancelled
        # portion is min(|long_delta + core_delta|, |inverse_delta|).
        long_delta = sum(usd * lev for _t, usd, lev in longs)
        core_delta = sum(usd * lev for _t, usd, lev in cores)
        positive_delta = long_delta + core_delta
        inverse_delta = sum(usd * lev for _t, usd, lev in inverses)
        cancelled_delta = min(abs(positive_delta), abs(inverse_delta))
        net_delta = positive_delta + inverse_delta
        # Daily drag: sum across BOTH sleeves (long + inverse; cores
        # contribute zero since they have no leverage decay). Cores'
        # decay tab is zero by construction.
        all_sleeves = longs + inverses
        daily_drag = _daily_drag_usd(all_sleeves)
        gross_delta = abs(positive_delta) + abs(inverse_delta)
        severity = _severity(cancelled_delta, gross_delta)

        def _fmt_holdings(rows):
            return [
                {"ticker": t, "usd": _z(u), "leverage": _z(l, 1)}
                for t, u, l in sorted(rows, key=lambda r: -r[1])
            ]

        conflicts.append({
            "family": fam_key,
            "family_label": _PAIR_FAMILIES[fam_key]["label"],
            "classification": classification,
            "severity": severity,
            "long_holdings": _fmt_holdings(longs),
            "inverse_holdings": _fmt_holdings(inverses),
            "core_holdings": _fmt_holdings(cores),
            "long_notional_usd": _z(long_notional),
            "inverse_notional_usd": _z(inverse_notional),
            "core_notional_usd": _z(core_notional),
            "long_delta_usd": _z(positive_delta),
            "inverse_delta_usd": _z(inverse_delta),
            "cancelled_delta_usd": _z(cancelled_delta),
            "net_delta_usd": _z(net_delta),
            "daily_drag_estimate_usd": _z(daily_drag, 3),
        })

    # Sort conflicts so the most severe (largest cancelled $) leads.
    conflicts.sort(
        key=lambda c: (
            0 if c["classification"] == "CARRY_WASTE" else 1,
            -(c.get("cancelled_delta_usd") or 0.0),
        ),
    )

    total_cancelled = sum((c["cancelled_delta_usd"] or 0.0) for c in conflicts)
    total_drag = sum((c["daily_drag_estimate_usd"] or 0.0) for c in conflicts)
    n_conflicts = len(conflicts)

    if any_carry_waste:
        verdict = "CARRY_WASTE"
        worst = next(c for c in conflicts if c["classification"] == "CARRY_WASTE")
        headline = (
            f"CARRY_WASTE — {worst['family_label']}: long {worst['long_notional_usd']:g} + "
            f"inverse {worst['inverse_notional_usd']:g} pays ~{worst['daily_drag_estimate_usd']:g}/day "
            f"in leverage decay on both sides while {worst['cancelled_delta_usd']:g} of delta cancels"
        )
    elif any_opposing_unlevered:
        verdict = "OPPOSING_UNLEVERED"
        worst = next(c for c in conflicts if c["classification"] == "OPPOSING_UNLEVERED")
        headline = (
            f"OPPOSING_UNLEVERED — {worst['family_label']}: 1x core opposing the inverse "
            f"sleeve — directional cancellation without compounding leverage decay"
        )
    else:
        verdict = "CLEAN"
        headline = "CLEAN — no opposing leveraged/inverse pairs in the book"

    return {
        "state": "READY",
        "verdict": verdict,
        "headline": headline,
        "n_book_positions": n_book_positions,
        "total_book_usd": _z(total_book_usd),
        "n_conflicts": n_conflicts,
        "conflicts": conflicts,
        "total_cancelled_delta_usd": _z(total_cancelled),
        "total_daily_drag_usd": _z(total_drag, 3),
    }
