"""Per-held-name price-momentum vs sector-peer-median divergence.

The desk's existing sector / momentum surface answers each *bucket*
question — sector_velocity_delta (per-sector NEWS velocity),
sector_signal_fit (book exposure vs sector news density),
sector_heatmap (per-bucket price + RSI + news), sector_pulse
(per-ticker momentum + news count snapshot). None of them answer the
per-held-name follow-up an operator who is long ONE name asks:

  *Is my name moving with its sector peers, or is it ripping/lagging
  on its own?*

A held position that's idiosyncratically rallying (e.g. NVDA +5%
while the semis peer median is +0.2%) is a single-name catalyst —
fragile, news-driven, and disproportionately exposed to news reversal.
A held position that's lagging while its sector peers rally is
operationally the *opposite* — the sector tide is lifting boats but
not yours; the thesis may be broken, or the entry was wrong.

This builder is per-held-name divergence (mom_5d) against the
peer-median 5d momentum, with a per-position verdict and a top-level
roll-up over the book.

Verdict matrix (per held position):

* ``IDIOSYNCRATIC_RALLY`` — held mom_5d >= peer_median + delta
  AND held mom_5d > 0. The name is ripping ahead of peers.
* ``IDIOSYNCRATIC_DECLINE`` — held mom_5d <= peer_median - delta
  AND held mom_5d < 0. The name is dropping while peers don't.
* ``LEADING_PEERS`` — held mom_5d >= peer_median + delta but both
  are positive or both negative — directional alignment, magnitude
  outperformance. Subset of "outperformance" that isn't single-name.
* ``LAGGING_PEERS`` — held mom_5d <= peer_median - delta but both
  positive or both negative — directional alignment, magnitude
  underperformance.
* ``TRACKING_PEERS`` — |held - peer_median| < delta. Moving with the
  sector — no idiosyncratic signal.
* ``NO_PEERS`` — sector unmapped or peer momentums all missing.

Top-level verdict roll-up (across all held positions):

* ``BOOK_IDIOSYNCRATIC`` — at least one IDIOSYNCRATIC_* in the book.
  The book carries single-name risk an operator should see.
* ``BOOK_DIVERGENT`` — at least one LAGGING_PEERS or
  IDIOSYNCRATIC_DECLINE position. Either drag or thesis-broken risk.
* ``BOOK_TRACKING`` — all positions TRACKING_PEERS or LEADING_PEERS
  (no IDIOSYNCRATIC, no LAGGING).
* ``INSUFFICIENT_DATA`` — fewer than ``min_positions`` positions with
  a peer_median computed.
* ``NO_DATA`` — no held positions at all.

Pure builder. Caller pre-fetches per-ticker mom_5d quant signals AND
the sector lookup. The sector→peers map is duplicated inline from
``analytics.sector_heatmap.HEATMAP_BUCKETS`` (precedent: live vs
backtest leveraged-ETF set; the duplicate is locked by a parity test).

Never raises; observational only — never gates Opus, no caps
(AGENTS.md #2/#12).
"""
from __future__ import annotations

from statistics import median
from typing import Any, Sequence

# Per-sector peer set. Mirrors ``sector_heatmap.HEATMAP_BUCKETS``
# verbatim — the same operator-curated DRAM/semis-leaning universe.
# Locked by a parity test in ``tests/test_peer_momentum_divergence_skill.py``
# so a drift in either file fails loudly.
PEERS_BY_SECTOR: dict[str, list[str]] = {
    "memory_core":      ["MU", "WDC", "STX"],
    "semis_equipment":  ["LRCX", "AMAT", "KLAC", "ASML"],
    "foundry":          ["TSM", "GFS", "UMC"],
    "design":           ["NVDA", "AMD", "MRVL", "AVGO"],
    "memory_leveraged": ["MUU", "SOXL", "NVDU", "SOXS"],
    "optical":          ["LITE", "LNOK", "CIEN"],
    "etf":              ["SMH", "SOXX"],
}

# Reverse index: ticker → sector. First sector wins on duplicate
# (none exist in the live map, but be defensive).
TICKER_TO_SECTOR: dict[str, str] = {}
for _sec, _tks in PEERS_BY_SECTOR.items():
    for _t in _tks:
        TICKER_TO_SECTOR.setdefault(_t.upper(), _sec)

# Per-position |held - peer_median| threshold (5d momentum %).
# Below this the held name is TRACKING_PEERS — no idiosyncratic
# signal. 2% on a 5d window is "noticeable but not dramatic".
DEFAULT_DELTA_PCT = 2.0

# Minimum number of peer momentums required to compute a stable
# median. 2 is the floor — a single peer is a comparison, not a
# distribution.
DEFAULT_MIN_PEERS = 2

# Top-level INSUFFICIENT_DATA floor — book-level verdict needs at
# least one position with a computed peer median.
DEFAULT_MIN_POSITIONS = 1


def _safe_ticker(p: Any) -> str:
    if not isinstance(p, dict):
        return ""
    t = p.get("ticker")
    if not isinstance(t, str) or not t:
        return ""
    return t.upper()


def _safe_mom(qs: Any, ticker: str) -> float | None:
    """Read mom_5d (a float % move) defensively. NaN / non-numeric →
    None so it doesn't poison the median."""
    if not isinstance(qs, dict):
        return None
    row = qs.get(ticker) or qs.get(ticker.upper()) or qs.get(ticker.lower())
    if not isinstance(row, dict):
        return None
    v = row.get("mom_5d")
    if not isinstance(v, (int, float)):
        return None
    if v != v:  # NaN
        return None
    return float(v)


def sector_for_ticker(ticker: str) -> str | None:
    """Pure lookup — None on miss. NVDA → 'design', 'nvda' → 'design'."""
    if not isinstance(ticker, str) or not ticker:
        return None
    return TICKER_TO_SECTOR.get(ticker.upper())


def _classify_pair(held_mom: float | None,
                   peer_median_mom: float | None,
                   delta_pct: float) -> str:
    if held_mom is None or peer_median_mom is None:
        return "NO_PEERS"
    gap = held_mom - peer_median_mom
    abs_gap = abs(gap)
    if abs_gap < delta_pct:
        return "TRACKING_PEERS"
    # Outperforming peers (held > peers by delta)
    if gap >= delta_pct:
        # Single-name rip: held positive, peers negative or near-zero
        if held_mom > 0 and peer_median_mom <= 0:
            return "IDIOSYNCRATIC_RALLY"
        return "LEADING_PEERS"
    # Underperforming peers (held < peers by delta)
    # Single-name drop: held negative, peers positive or near-zero
    if held_mom < 0 and peer_median_mom >= 0:
        return "IDIOSYNCRATIC_DECLINE"
    return "LAGGING_PEERS"


def build_peer_momentum_divergence_skill(
    positions: Sequence[Any] | None,
    quant_signals: Any,
    *,
    delta_pct: float = DEFAULT_DELTA_PCT,
    min_peers: int = DEFAULT_MIN_PEERS,
    min_positions: int = DEFAULT_MIN_POSITIONS,
    peers_by_sector: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build the per-held-position peer-momentum-divergence card.

    Inputs:
      ``positions`` — list of open position dicts; only ``ticker`` is
        read. Closed positions filtered by caller.
      ``quant_signals`` — ``{ticker: {mom_5d: float, ...}, ...}``;
        the caller composes (live: ``get_quant_signals_live``).
      ``delta_pct`` / ``min_peers`` / ``min_positions`` — knobs.
      ``peers_by_sector`` — override the inline ``PEERS_BY_SECTOR``
        map. Tests pass alternative maps to pin behaviour.

    Returns the envelope dict. Never raises.
    """
    pmap = peers_by_sector if peers_by_sector is not None else PEERS_BY_SECTOR
    # Build reverse index locally so test overrides apply.
    rev: dict[str, str] = {}
    for sec, tks in pmap.items():
        if not isinstance(tks, list):
            continue
        for t in tks:
            if isinstance(t, str) and t:
                rev.setdefault(t.upper(), sec)

    qs = quant_signals if isinstance(quant_signals, dict) else {}

    rows: list[dict[str, Any]] = []
    pos_seen = 0
    for p in (positions or []):
        t = _safe_ticker(p)
        if not t:
            continue
        pos_seen += 1
        sector = rev.get(t)
        held_mom = _safe_mom(qs, t)
        if sector is None:
            rows.append({
                "ticker": t,
                "sector": None,
                "mom_5d": held_mom,
                "peer_median_mom_5d": None,
                "delta_pct": None,
                "n_peers": 0,
                "verdict": "NO_PEERS",
            })
            continue
        peer_tickers = [pt for pt in pmap.get(sector, []) if pt.upper() != t]
        peer_moms: list[float] = []
        for pt in peer_tickers:
            m = _safe_mom(qs, pt.upper())
            if m is not None:
                peer_moms.append(m)
        if len(peer_moms) < min_peers:
            rows.append({
                "ticker": t,
                "sector": sector,
                "mom_5d": held_mom,
                "peer_median_mom_5d": None,
                "delta_pct": None,
                "n_peers": len(peer_moms),
                "verdict": "NO_PEERS",
            })
            continue
        pm = median(peer_moms)
        gap = (held_mom - pm) if held_mom is not None else None
        verdict = _classify_pair(held_mom, pm, delta_pct)
        rows.append({
            "ticker": t,
            "sector": sector,
            "mom_5d": round(held_mom, 4) if held_mom is not None else None,
            "peer_median_mom_5d": round(pm, 4),
            "delta_pct": round(gap, 4) if gap is not None else None,
            "n_peers": len(peer_moms),
            "verdict": verdict,
        })

    # ── top-level roll-up ──────────────────────────────────────────
    n_with_peers = sum(1 for r in rows if r["verdict"] != "NO_PEERS")
    if pos_seen == 0:
        top_verdict = "NO_DATA"
        headline = "no open positions"
    elif n_with_peers < min_positions:
        top_verdict = "INSUFFICIENT_DATA"
        headline = (
            f"{pos_seen} held names; only {n_with_peers} have peer comparisons "
            f"(need ≥ {min_positions})"
        )
    else:
        has_idio = any(
            r["verdict"] in ("IDIOSYNCRATIC_RALLY", "IDIOSYNCRATIC_DECLINE")
            for r in rows
        )
        has_lag = any(
            r["verdict"] in ("LAGGING_PEERS", "IDIOSYNCRATIC_DECLINE") for r in rows
        )
        if has_idio:
            top_verdict = "BOOK_IDIOSYNCRATIC"
            idio_tickers = [
                r["ticker"] for r in rows
                if r["verdict"] in ("IDIOSYNCRATIC_RALLY", "IDIOSYNCRATIC_DECLINE")
            ]
            headline = (
                "single-name risk: " + ", ".join(idio_tickers)
                + " diverging from peer median"
            )
        elif has_lag:
            top_verdict = "BOOK_DIVERGENT"
            lag_tickers = [r["ticker"] for r in rows if r["verdict"] == "LAGGING_PEERS"]
            headline = (
                "book lagging peers: " + ", ".join(lag_tickers)
            )
        else:
            top_verdict = "BOOK_TRACKING"
            headline = (
                f"{n_with_peers} held name{'s' if n_with_peers != 1 else ''} "
                f"tracking peer momentum"
            )

    return {
        "verdict": top_verdict,
        "headline": headline,
        "n_positions": pos_seen,
        "n_with_peers": n_with_peers,
        "positions": rows,
        "thresholds": {
            "delta_pct": delta_pct,
            "min_peers": min_peers,
            "min_positions": min_positions,
        },
    }
