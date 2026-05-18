"""Forward beta/concentration stress scenarios for the live book.

``tail_risk`` answers *"what has a bad day actually looked like?"* from the
equity curve — but on a young book it correctly reads ``INSUFFICIENT``
(``<MIN_RETURNS=20`` daily observations) and the trader gets **no number
at all** for the question every desk asks first: *if the tape drops, what
does it cost me, given how this book is actually built right now?*

``build_stress_scenarios`` fills exactly that gap and nothing else. It is
the **forward** complement to ``tail_risk``'s **backward** VaR: it needs
**zero return history** because it is pure ``Σ weight × beta × shock``
arithmetic over the *current* marked book, so it produces a real dollar
figure on day one — precisely when ``tail_risk`` is dark.

Three scenario families, each a distinct risk a concentrated book carries:

* **Market shock** — SPY moves −1 / −3 / −5 / −10 % (and a +3 % upside for
  honest symmetry). Beta-amplified per position. The −3 % market P&L is
  computed **byte-identically** to ``/api/risk``'s ``shock_usd``
  (``Σ −0.03 · βᵢ · valᵢ``) — single source of truth (AGENTS.md #10): a
  drift in either fails the cross-check test loudly.
* **Single-name gap** — the *largest* position alone gaps −10 % on its own
  news (idiosyncratic, **no beta** — it is the name's own price move). The
  risk a 60-%-of-book single name carries that a diversified book does not.
* **Sector shock** — the *most-concentrated sector* corrects −10 %
  thematically (**no beta** — a direct price shock on that cluster). For
  the documented live pathology (≈98 % in two correlated AI-datacenter
  names) this is the single most decision-relevant line.

Honesty: the betas are the deliberately-approximate sector constants from
``/api/risk``'s ``_LEVERAGE_BETA`` ("decision support, not VaR"); the
headline says so, so the figure is never mistaken for a measured VaR.

State ladder mirrors the sibling builders but has **no sample-size gate**
— that absence *is* the feature. ``NO_DATA`` only when the book is empty /
unpriceable; otherwise ``OK``.

Diagnostic / advisory only: never gates Opus, adds no caps (AGENTS.md
#2 / #12 — the ``risk_mirror`` / ``sector_exposure`` precedent). The
``prompt_block`` carries the autonomy preamble and no directive verb.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Test-pinned verbatim copy of ``dashboard._LEVERAGE_BETA`` (the sector→beta
# SSOT used by ``/api/risk``'s shock_usd). Copied — not imported — for the
# exact reason ``sector_exposure.SECTOR_MAP`` is: the live decision hot path
# (``strategy._build_payload``) must not pull the ~9k-line Flask ``dashboard``
# module. ``tests/test_stress_scenarios.py`` pins ``_LEVERAGE_BETA ==
# dashboard._LEVERAGE_BETA`` so any sibling edit that drifts the two fails CI
# (the ``test_sector_exposure`` ``SECTOR_MAP == dashboard.SECTOR_MAP``
# precedent). The dashboard endpoint passes the *real* dashboard objects, so
# ``/api/stress-scenarios`` is the true SSOT and this copy only ever serves
# the prompt / reporter side.
_LEVERAGE_BETA = {
    "broad": 1.0,
    "broad_lev": 3.0,
    "tech": 1.2,
    "tech_lev": 3.0,
    "crypto_lev": 2.5,
    "semis": 1.5,
    "semis_lev": 3.0,
    "optical": 1.4,
    "bio_lev": 3.0,
    "health_lev": 3.0,
    "fin_lev": 3.0,
    "housing_lev": 3.0,
    "util_lev": 3.0,
    "defense_lev": 3.0,
    "other": 1.0,
}

# Broad-market shock magnitudes (percent SPY move). Negative = sell-off;
# the single +3 keeps the panel honest (a beta-3x book also gaps *up*).
_MARKET_MOVES_PCT = (-1.0, -3.0, -5.0, -10.0, 3.0)

# Idiosyncratic single-name gap and thematic sector-cluster correction.
_SINGLE_NAME_GAP_PCT = -10.0
_SECTOR_SHOCK_PCT = -10.0


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round, folding -0.0 → 0.0 so the JSON never carries a signed zero."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _position_betas(positions, classify, beta_map):
    """Per-position (ticker, sector, market_value, beta) computed
    **identically** to ``dashboard.risk_api`` (AGENTS.md #10): price falls
    back avg_cost→0, options ×100 with a 3× payoff beta capped at 4 and
    negated for puts. A garbage row contributes 0, never raises."""
    out = []
    for p in positions or []:
        try:
            ptype = p.get("type") or "stock"
            mult = 100 if ptype in ("call", "put") else 1
            price = p.get("current_price") or p.get("avg_cost") or 0.0
            qty = float(p.get("qty") or 0)
            val = float(price) * qty * mult
            sec = classify(p.get("ticker") or "")
            beta = float(beta_map.get(sec, 1.0))
            if ptype in ("call", "put"):
                beta = min(beta * 3.0, 4.0)
                if ptype == "put":
                    beta = -beta
            out.append({
                "ticker": p.get("ticker"),
                "sector": sec,
                "market_value": val,
                "beta": beta,
            })
        except Exception:
            continue
    return out


def build_stress_scenarios(
    positions: list[dict],
    total_value: float,
    classify,
    beta_map: dict,
    now: datetime | None = None,
) -> dict:
    """Pure, no I/O, never raises. ``classify`` is the dashboard/strategy
    ``_classify`` (ticker→sector SSOT); ``beta_map`` is the shared
    ``_LEVERAGE_BETA`` (sector→beta SSOT with ``/api/risk``)."""
    now = now or datetime.now(timezone.utc)
    try:
        tv = float(total_value or 0.0)
    except (TypeError, ValueError):
        tv = 0.0

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_positions": 0,
        "gross_value_usd": None,
        "scenarios": [],
        "single_name_gap": None,
        "sector_shock": None,
        "prompt_block": None,
    }

    rows = _position_betas(positions, classify, beta_map)
    gross = sum(r["market_value"] for r in rows)
    base["n_positions"] = len(rows)
    base["gross_value_usd"] = _z(gross)

    if not rows or tv <= 0 or gross <= 0:
        base["state"] = "NO_DATA"
        base["headline"] = "Stress scenarios: no priced book to shock yet."
        return base

    base["state"] = "OK"

    def _pct(usd: float) -> float | None:
        return _z(usd / tv * 100.0) if tv else None

    # ── Market shocks: SPY move → beta-amplified P&L ──
    scenarios = []
    for mv in _MARKET_MOVES_PCT:
        frac = mv / 100.0
        pnl = sum(frac * r["beta"] * r["market_value"] for r in rows)
        scenarios.append({
            "label": f"SPY {mv:+.0f}%",
            "kind": "market",
            "move_pct": _z(mv),
            "pnl_usd": _z(pnl),
            "pnl_pct": _pct(pnl),
        })
    base["scenarios"] = scenarios

    # ── Single-name idiosyncratic gap on the largest position (no beta) ──
    top = max(rows, key=lambda r: r["market_value"])
    sn_pnl = _SINGLE_NAME_GAP_PCT / 100.0 * top["market_value"]
    base["single_name_gap"] = {
        "ticker": top["ticker"],
        "weight_pct": _z(top["market_value"] / tv * 100.0),
        "gap_pct": _z(_SINGLE_NAME_GAP_PCT),
        "pnl_usd": _z(sn_pnl),
        "pnl_pct": _pct(sn_pnl),
    }

    # ── Thematic sector-cluster correction on the heaviest sector (no beta) ──
    by_sector: dict[str, float] = {}
    for r in rows:
        by_sector[r["sector"]] = by_sector.get(r["sector"], 0.0) + r["market_value"]
    worst_sec = max(by_sector, key=lambda s: by_sector[s])
    sec_val = by_sector[worst_sec]
    sec_pnl = _SECTOR_SHOCK_PCT / 100.0 * sec_val
    base["sector_shock"] = {
        "sector": worst_sec,
        "weight_pct": _z(sec_val / tv * 100.0),
        "shock_pct": _z(_SECTOR_SHOCK_PCT),
        "n_names": sum(1 for r in rows if r["sector"] == worst_sec),
        "pnl_usd": _z(sec_pnl),
        "pnl_pct": _pct(sec_pnl),
    }

    # ── Worst realistic line drives the headline ──
    candidates = (
        [(s["pnl_usd"], s["pnl_pct"], s["label"]) for s in scenarios]
        + [(sn_pnl_z := base["single_name_gap"]["pnl_usd"],
            base["single_name_gap"]["pnl_pct"],
            f"{top['ticker']} −10% gap")]
        + [(base["sector_shock"]["pnl_usd"],
            base["sector_shock"]["pnl_pct"],
            f"{worst_sec} −10% sector shock")]
    )
    worst = min(candidates, key=lambda c: (c[0] if c[0] is not None else 0.0))
    base["headline"] = (
        f"Stress (beta-approx, no return history needed): worst of "
        f"{len(candidates)} scenarios is {worst[2]} → "
        f"${worst[0]:+.2f} ({worst[1]:+.2f}% of book). "
        f"Single-name −10% on {top['ticker']} "
        f"({base['single_name_gap']['weight_pct']:.1f}% of book) "
        f"= ${sn_pnl:+.2f}; {worst_sec} −10% "
        f"({base['sector_shock']['weight_pct']:.1f}% of book) "
        f"= ${sec_pnl:+.2f}."
    )

    base["prompt_block"] = (
        "FORWARD STRESS (advisory, observational — never a directive; you "
        "retain full autonomy):\n"
        f"  {base['headline']}\n"
        "  These are weight×beta arithmetic on the CURRENT book (sector "
        "betas are approximate, not measured VaR). A concentrated book's "
        "single-name and sector lines are the ones a diversified book "
        "would not carry."
    )
    return base
