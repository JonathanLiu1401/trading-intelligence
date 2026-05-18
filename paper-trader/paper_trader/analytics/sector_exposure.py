"""Live-book SECTOR concentration + marginal in-play sector impact, fed into
the live Opus decision prompt.

``risk_mirror`` closed *name*-level concentration (top weight / HHI by
ticker). The book's documented #3 pathology is exactly one dimension over:
*sector* clustering (``risk_mirror.py`` docstring — "the book 60.9% in one
name's **sector** … the dashboard already exposes both … but the decision
engine itself never saw them"). ``/api/analytics`` computes
``sector_exposure_pct`` and ``/api/risk`` per-position sector, but the
**decision path has zero sector awareness** (``grep sector
paper_trader/strategy.py`` prompt path → no hits). The marginal question a
discretionary desk checks before every order — *does this trade pile onto my
single most concentrated sector?* — was invisible at decision time.

This is the lean, prompt-facing complement to the dashboard-only
``/api/analytics`` sector breakdown — exactly the gap ``risk_mirror`` /
``event_calendar`` / ``buying_power`` were each built to close, one dimension
over.

**Single source of truth.** The book-sector % mirrors ``dashboard.py``'s
``analytics_api`` formula *verbatim* — ``price = current_price or avg_cost;
val = price*qty*(100 if option else 1); pct = val/total_value*100`` classified
by ``SECTOR_MAP`` — so ``/api/sector-exposure`` and ``/api/analytics``
``sector_exposure_pct`` can never drift. ``SECTOR_MAP``/``classify`` are a
**test-pinned verbatim copy** of ``dashboard.SECTOR_MAP``/``_classify``,
duplicated deliberately (the ``strategy._ml_live_opinion`` precedent:
importing the ~9k-line Flask ``dashboard`` module onto the live decision hot
path is a fragility a ``_safe`` wrapper should never have to catch, and a
sibling edit that broke that import would silently re-blind the desk).
``tests/test_sector_exposure.py`` asserts byte-equality with
``dashboard.SECTOR_MAP`` so any drift fails CI. ``SECTOR_HEAVY_PCT`` is pinned
to ``game_plan._SECTOR_HEAVY_PCT`` so the prompt and the dashboard game-plan
card can't disagree on "heavy".

**Observational, never prescriptive.** Same contract as ``risk_mirror`` /
``buying_power`` (AGENTS.md invariants #2/#12): it states facts (per-sector %,
sector-HHI, which in-play names sit in an already-heavy sector) and reaffirms
full autonomy in its preamble. It issues no directive, imposes no cap, and
never gates a trade — ``_execute`` still runs the only real cash check.

Pure and deterministic (no clock, no IO). Never raises — the ``_safe``
contract (the caller in ``decide()`` also wraps it).
"""
from __future__ import annotations

# ── Verbatim copy of dashboard.SECTOR_MAP (drift-locked by
#    tests/test_sector_exposure.py::test_sector_map_matches_dashboard). Do NOT
#    hand-edit one without the other — the test will fail. ────────────────────
SECTOR_MAP = {
    # Semis (cash)
    "NVDA": "semis", "AMD": "semis", "MU": "semis", "AMAT": "semis",
    "LRCX": "semis", "KLAC": "semis", "TSM": "semis", "ASML": "semis",
    "MRVL": "semis", "SMH": "semis", "SOXX": "semis",
    "DRAM": "semis", "SNDU": "semis",
    # Semis leveraged
    "SOXL": "semis_lev", "SOXS": "semis_lev", "NVDU": "semis_lev",
    "MUU": "semis_lev",
    # Optical / networking
    "LITE": "optical", "LNOK": "optical",
    # Broad market
    "SPY": "broad", "QQQ": "broad", "VOO": "broad", "VTI": "broad",
    # Broad leveraged
    "TQQQ": "broad_lev", "UPRO": "broad_lev", "SPXL": "broad_lev",
    "QLD": "broad_lev", "SSO": "broad_lev", "UDOW": "broad_lev",
    "URTY": "broad_lev", "TNA": "broad_lev",
    "SPXS": "broad_lev", "SQQQ": "broad_lev",
    # Tech / FAANG
    "AAPL": "tech", "MSFT": "tech", "META": "tech", "GOOG": "tech",
    "GOOGL": "tech", "AMZN": "tech", "TSLA": "tech", "NFLX": "tech",
    "TECL": "tech_lev", "TECS": "tech_lev", "FNGU": "tech_lev",
    "FNGD": "tech_lev", "MSFU": "tech_lev", "AMZU": "tech_lev",
    "GOOGU": "tech_lev", "METAU": "tech_lev", "TSLL": "tech_lev",
    "CONL": "crypto_lev", "BITU": "crypto_lev", "ETHU": "crypto_lev",
    # Sector leveraged
    "LABU": "bio_lev", "CURE": "health_lev",
    "FAS": "fin_lev", "DPST": "fin_lev",
    "NAIL": "housing_lev", "UTSL": "util_lev",
    "DFEN": "defense_lev",
}

# Pinned to game_plan._SECTOR_HEAVY_PCT (drift-locked by the test suite) so
# the in-prompt "heavy sector" flag and the dashboard game-plan card agree.
SECTOR_HEAVY_PCT = 60.0

# Bound the rendered marginal line so the block stays one short prompt section
# (the in-play set is already lean, but a signal-heavy cycle can widen it).
_MAX_MARGINAL_NAMES = 10
_MAX_BREAKDOWN_SECTORS = 6

_PREAMBLE = (
    "SECTOR EXPOSURE (how your book clusters by sector and which in-play "
    "names would add to an already-heavy sector — facts for risk awareness "
    "only, NOT a directive or limit; you retain complete autonomy over the "
    "next decision):"
)


def classify(ticker) -> str:
    """Verbatim mirror of ``dashboard._classify`` — drift-locked by the test
    suite. Unknown ⇒ ``"other"`` (never raises; a ``None``/garbage ticker
    coerces to the string ``"NONE"`` and falls through to ``"other"``)."""
    try:
        return SECTOR_MAP.get(str(ticker).upper(), "other")
    except Exception:
        return "other"


def _f(x, default: float = 0.0) -> float:
    """Best-effort float coercion — a garbage cell degrades to ``default``,
    never raises (the _safe contract; identical to ``buying_power._f``)."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _position_value(p: dict) -> float:
    """Book value of one open position, **mirroring ``analytics_api``
    verbatim**: ``(current_price or avg_cost) * qty * (100 if option else 1)``.

    Deliberately NOT the enriched ``market_value`` ``buying_power`` prefers —
    that builder matches ``/api/capital-paralysis``; this one matches
    ``/api/analytics`` ``sector_exposure_pct``, a different single source of
    truth. Keeping the formula identical is what makes the parity test exact.
    """
    mult = 100 if p.get("type") in ("call", "put") else 1
    price = _f(p.get("current_price")) or _f(p.get("avg_cost"))
    return price * _f(p.get("qty")) * mult


def build_sector_exposure(snapshot: dict, names_in_play) -> dict:
    """Compose the live-book sector-concentration awareness block.

    ``snapshot`` — the ``strategy._portfolio_snapshot`` dict (``cash``,
    ``total_value``, enriched ``positions``). ``names_in_play`` — the
    ``strategy._names_in_play`` set so the marginal "what would I be adding
    to" view is against the same "what matters this cycle" universe the quant
    / track-record / buying-power blocks use.

    Returns ``{state, summary, prompt_block, sector_pct, sector_usd,
    top_sector, top_sector_pct, hhi, n_sectors, cash_pct, in_play}``. Pure;
    never raises.
    """
    try:
        snap = snapshot or {}
        total = _f(snap.get("total_value"))
        cash = _f(snap.get("cash"))
        positions = list(snap.get("positions") or [])
        in_play = sorted({str(t).upper() for t in (names_in_play or set())})

        # Per-sector dollar exposure (analytics_api formula, verbatim).
        sector_usd: dict[str, float] = {}
        for p in positions:
            val = _position_value(p)
            if val == 0.0:
                continue
            sec = classify(p.get("ticker"))
            sector_usd[sec] = sector_usd.get(sec, 0.0) + val

        if total <= 0 or not sector_usd:
            # No priced book to concentrate (fresh / all-cash / marks down) —
            # one honest line, the buying_power NO_DATA pattern.
            return {
                "state": "NO_DATA",
                "summary": "no priced book to assess sector concentration",
                "prompt_block": (f"{_PREAMBLE}\n  (no priced positions this "
                                 f"cycle — no sector concentration to assess)"),
                "sector_pct": {}, "sector_usd": {}, "top_sector": None,
                "top_sector_pct": None, "hhi": None, "n_sectors": 0,
                "cash_pct": (round(cash / total * 100.0, 2)
                             if total > 0 else None),
                "in_play": [],
            }

        sector_pct = {
            s: round(v / total * 100.0, 2) for s, v in sector_usd.items()
        }
        cash_pct = round(cash / total * 100.0, 2)
        invested = sum(sector_usd.values())
        # Sector-HHI over invested weights (∑w=1): 1.0 = a single sector,
        # → 0 as exposure spreads. A distinct measure from correlation's
        # name-HHI (this is by sector, not by ticker).
        hhi = round(sum((v / invested) ** 2 for v in sector_usd.values()), 4)

        # One deterministic ranking drives BOTH the top-sector pick and the
        # breakdown line, so they can never disagree on a tie (largest %
        # first, ties broken by sector name — the buying_power/correlation
        # sort idiom).
        ranked = sorted(sector_pct.items(), key=lambda kv: (-kv[1], kv[0]))
        top_sector, top_sector_pct = ranked[0]
        n_sectors = len(sector_pct)
        concentrated = top_sector_pct >= SECTOR_HEAVY_PCT
        state = "CONCENTRATED" if concentrated else "DIVERSIFIED"

        if hhi >= 0.5:
            hhi_label = "concentrated"
        elif hhi >= 0.25:
            hhi_label = "moderate"
        else:
            hhi_label = "diversified"

        # Marginal view: each in-play name → its sector → that sector's
        # CURRENT book weight (0.0 if not yet held). No fabricated fill size —
        # Opus chooses size; the honest deterministic fact is "MU is SEMIS,
        # SEMIS is already 61% of your book", not an invented projection.
        marginal = []
        for tk in in_play:
            sec = classify(tk)
            pct = sector_pct.get(sec, 0.0)
            marginal.append({
                "ticker": tk,
                "sector": sec,
                "sector_pct": pct,
                "heavy": (pct >= SECTOR_HEAVY_PCT) or (sec == top_sector
                                                       and pct > 0.0),
            })
        # Riskiest adds first (heaviest existing sector weight on top).
        marginal.sort(key=lambda m: (-m["sector_pct"], m["ticker"]))

        # ── prompt block ──
        breakdown = ranked[:_MAX_BREAKDOWN_SECTORS]
        breakdown_str = " · ".join(
            f"{s.upper()} {p:.1f}%" for s, p in breakdown)
        head = (
            f"  Book: {n_sectors} sector(s); top = {top_sector.upper()} "
            f"{top_sector_pct:.1f}% (${sector_usd[top_sector]:.0f} of "
            f"${total:.0f}); sector-HHI {hhi:.2f} ({hhi_label}). "
            f"Cash {cash_pct:.1f}%."
        )
        if concentrated:
            head += (f" {top_sector.upper()} is past the "
                     f"{SECTOR_HEAVY_PCT:.0f}% heavy mark — a single "
                     f"{top_sector.upper()} shock moves most of the book "
                     f"together.")
        lines = [head, f"  Breakdown: {breakdown_str}"]

        if marginal:
            shown = marginal[:_MAX_MARGINAL_NAMES]
            bits = []
            for m in shown:
                tag = ""
                if m["heavy"] and m["sector"] == top_sector:
                    tag = " — your heaviest sector"
                elif m["heavy"]:
                    tag = " — already heavy"
                elif m["sector_pct"] == 0.0:
                    tag = " — diversifying"
                bits.append(
                    f"{m['ticker']}→{m['sector'].upper()} "
                    f"({m['sector_pct']:.1f}%{tag})")
            lines.append("  In-play by sector: " + " · ".join(bits))

        prompt_block = f"{_PREAMBLE}\n" + "\n".join(lines)

        summary = (f"{state} · top {top_sector.upper()} "
                   f"{top_sector_pct:.1f}% · HHI {hhi:.2f} · "
                   f"{n_sectors} sector(s)")

        return {
            "state": state,
            "summary": summary,
            "prompt_block": prompt_block,
            "sector_pct": sector_pct,
            "sector_usd": {s: round(v, 2) for s, v in sector_usd.items()},
            "top_sector": top_sector,
            "top_sector_pct": top_sector_pct,
            "hhi": hhi,
            "n_sectors": n_sectors,
            "cash_pct": cash_pct,
            "in_play": marginal,
        }
    except Exception:
        # The _safe contract: a diagnostics fault must never sink a live
        # decision cycle. One honest line, no exception.
        return {
            "state": "ERROR",
            "summary": "sector exposure unavailable",
            "prompt_block": (f"{_PREAMBLE}\n  (sector-exposure computation "
                             f"unavailable this cycle)"),
            "sector_pct": {}, "sector_usd": {}, "top_sector": None,
            "top_sector_pct": None, "hhi": None, "n_sectors": 0,
            "cash_pct": None, "in_play": [],
        }


if __name__ == "__main__":  # smoke test against the live book
    import json as _json

    from paper_trader.store import get_store
    from paper_trader.strategy import (WATCHLIST, _names_in_play,
                                       _portfolio_snapshot)

    s = get_store()
    snap = _portfolio_snapshot(s)
    rep = build_sector_exposure(
        snap,
        _names_in_play(snap.get("positions") or [], [], WATCHLIST),
    )
    print(rep["prompt_block"])
    print("\n---\n")
    print(_json.dumps({k: v for k, v in rep.items() if k != "prompt_block"},
                       indent=2, default=str))
