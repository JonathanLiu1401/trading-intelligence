"""Per-sector sentiment coherence — macro story vs idiosyncratic noise.

The existing ``/api/sector-pulse`` tells the operator which sector the news
is *concentrated in* (volume + recency velocity). It does **not** tell them
whether that concentration is *coherent*: are all DRAM headlines bullish on
one macro thesis (HBM ramp, Korean supply tightening) — in which case the
sector is the trade — or split across idiosyncratic catalysts (MU upgrade
and a WDC fraud probe in the same hour) — in which case sector-wide
positioning is exactly the wrong move?

This question lives nowhere else in the chat surface and is the structural
read a trader makes before sizing into a leveraged-sector ETF: the
difference between "buy SOXL on the wire" and "trade MU only" is whether
the wire actually agrees on a direction.

``build_sector_coherence`` answers it with a single deterministic number
per sector — ``coherence_pct = max(bull, bear) / classified × 100`` over a
configurable lookback — and an explicit ``lead_direction`` (BULL/BEAR/MIXED).
Pure / total, in the ``_aggregate_sector_pulse`` / ``build_portfolio_signals``
mould: a non-list input or a row that lacks a usable title is skipped, the
result is always a well-formed skeleton, never an exception into the chat
handler or the endpoint.

SSOT discipline (project_digital_intern_chat_enrichment_pattern):
* sector taxonomy + ticker extraction are imported VERBATIM from the
  sector-pulse builder (``dashboard.web_server._SECTOR_MAP`` /
  ``_extract_tickers``) so coherence and pulse never tag the same article
  to different sectors.
* recency-decay age uses the same ``analysis.claude_analyst._seen_age_hours``
  parser the alert pipeline and pulse already share.

The bull/bear classifier is intentionally tiny and word-bounded
(case-insensitive). Headlines that match no word stay ``neutral`` and are
**excluded from the coherence ratio** — coherence is "of the *opinionated*
headlines, how many agree on direction". A sector with 10 neutral articles
and one bullish one is ``100% coherent`` with ``1`` classified observation,
not ``10% coherent``; that's the only meaningful semantic on a sparse signal.

A *verdict* (``MACRO_BULL`` / ``MACRO_BEAR`` / ``SPLIT`` / ``IDIOSYNCRATIC``)
is withheld until ``MIN_CLASSIFIED`` opinionated articles exist for that
sector — anything less is noise. Each ``state == "OK"`` sector also carries
``lead_headline`` so the operator can read the most-likely thesis source
without leaving chat.

Observational only — this never gates Opus and adds no caps (paper-trader
CLAUDE.md #2/#12 *spirit* applies here too).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

# Below this many *classified* (non-neutral) articles per sector, the
# verdict is withheld — a 1-bull / 0-bear "100% coherence" is meaningless.
MIN_CLASSIFIED = 3

# A verdict is MACRO_* when the dominant direction crosses this share of
# classified articles; everything below is SPLIT (genuinely mixed).
MACRO_COHERENCE_PCT = 70.0

# Word-bounded, case-insensitive. Kept deliberately small and high-precision
# so a generic word doesn't poison a sector: "surge" / "rally" / "beat" are
# in; "good" / "strong" / "rise" (false-positive prone) are out. Matches
# the headline-only context (we don't classify full text). Compound terms
# rely on the regex word-boundary so "outperformed" matches inside
# "outperformed" but "underperformance" does not match "perform".
_BULL_TERMS = {
    # Verb-conjugation completeness — third-person singular ("surges",
    # "upgrades", "jump") was previously missing on the bull side while the
    # bear side had its "-s" forms. Live headlines on a single ticker
    # overwhelmingly use 3rd-person singular ("MU surges", "NVDA upgrades to
    # Buy"). The asymmetry tilted classifications neutral for genuine bull
    # news, the exact failure surfaced by test_wire_stance.
    "surge", "surges", "surged", "rally", "rallies", "rallied",
    "soar", "soars", "soared", "beat", "beats", "tops", "topped",
    "upgrade", "upgrades", "upgraded", "raised", "raises",
    "boost", "boosts", "boosted", "record", "high", "highs",
    "outperform", "outperforms", "outperformed",
    "expansion", "growth", "approves", "approved",
    "breakthrough", "wins", "win", "bullish", "buy", "overweight",
    "strong-buy", "outshines", "outshone",
    "jump", "jumps", "jumped",
}
_BEAR_TERMS = {
    "plunge", "plunges", "plunged", "tumble", "tumbles", "tumbled", "slump",
    "slumps", "slumped", "miss", "misses", "missed",
    "downgrade", "downgrades", "downgraded",
    "cut", "cuts", "lowered", "warning", "warns", "warned", "probe", "fraud",
    "lawsuit", "sues", "sued", "investigation", "recall", "recalls",
    "bankruptcy", "default", "selloff", "sell-off", "low", "lows",
    "underperform", "underperforms", "underperformed",
    "bearish", "sell", "underweight", "strong-sell",
    # "fell" is the irregular past tense of "fall" — heavily used in
    # headlines ("MU fell 5%") and previously missing on the bear side.
    "falls", "fall", "fell",
    "drop", "drops", "dropped",
    "decline", "declines", "declined", "loss", "losses",
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{1,}")


def _classify(title: str) -> str:
    """One of ``"bull"``, ``"bear"``, ``"neutral"``.

    Tie-breaker on equal hit counts is ``neutral`` — refusing to pick a
    direction is more honest than alternating on word order. The classifier
    is intentionally simple and case-insensitive; high-precision tokens
    only (see module docstring on why "good"/"rise" are excluded).
    """
    if not isinstance(title, str) or not title:
        return "neutral"
    bull = bear = 0
    for tok in _WORD_RE.findall(title.lower()):
        if tok in _BULL_TERMS:
            bull += 1
        if tok in _BEAR_TERMS:
            bear += 1
    if bull > bear:
        return "bull"
    if bear > bull:
        return "bear"
    return "neutral"


def _verdict(coh_pct: float, lead: str, n_classified: int) -> str:
    if n_classified < MIN_CLASSIFIED:
        return "INSUFFICIENT"
    if coh_pct >= MACRO_COHERENCE_PCT:
        return "MACRO_BULL" if lead == "bull" else "MACRO_BEAR"
    if coh_pct >= 55.0:
        return "TILT_BULL" if lead == "bull" else "TILT_BEAR"
    return "SPLIT"


def build_sector_coherence(
    articles: Any, window_hours: float | None = None,
    now: datetime | None = None,
) -> dict:
    """Pure: roll a list of live article dicts into a per-sector coherence
    report. ``window_hours`` is informational only — the builder trusts the
    caller to have already filtered to the lookback (matches the
    ``_aggregate_sector_pulse`` contract); pass it through so the endpoint
    can advertise the window without re-deriving it.
    """
    out = {
        "generated_at": (now or datetime.now(timezone.utc))
        .isoformat(timespec="seconds"),
        "window_hours": window_hours,
        "min_classified": MIN_CLASSIFIED,
        "macro_threshold_pct": MACRO_COHERENCE_PCT,
        "n_scanned": 0,
        "n_mapped": 0,
        "n_classified": 0,
        "sectors": [],
    }
    if not isinstance(articles, list):
        return out

    try:
        from dashboard.web_server import _SECTOR_MAP, _extract_tickers
    except Exception:  # noqa: BLE001 — fall back to empty taxonomy
        _SECTOR_MAP = {}
        def _extract_tickers(_t):  # type: ignore
            return set()

    agg: dict[str, dict] = {}
    n_scanned = 0
    n_mapped = 0
    n_classified = 0
    for art in articles:
        if not isinstance(art, dict):
            continue
        n_scanned += 1
        title = art.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        tks = _extract_tickers(title)
        sectors = {_SECTOR_MAP[t] for t in tks if t in _SECTOR_MAP}
        if not sectors:
            continue
        n_mapped += 1
        stance = _classify(title)
        if stance != "neutral":
            n_classified += 1
        try:
            ai = float(art.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            ai = 0.0
        for sec in sectors:
            s = agg.setdefault(sec, {
                "bull": 0, "bear": 0, "neutral": 0, "n": 0,
                "lead_headline": ("", -1.0, "neutral"),
            })
            s["n"] += 1
            s[stance] = s.get(stance, 0) + 1
            # Track the highest-scored opinionated headline as the lead so
            # the chat surfaces the actual thesis source, not a neutral row.
            if stance != "neutral" and ai > s["lead_headline"][1]:
                s["lead_headline"] = (title, ai, stance)

    out["n_scanned"] = n_scanned
    out["n_mapped"] = n_mapped
    out["n_classified"] = n_classified

    sectors_out = []
    for sec, s in agg.items():
        cls = s["bull"] + s["bear"]
        lead = "bull" if s["bull"] > s["bear"] else (
            "bear" if s["bear"] > s["bull"] else "mixed"
        )
        coh_pct = (max(s["bull"], s["bear"]) / cls * 100.0) if cls > 0 else 0.0
        verdict = _verdict(coh_pct, lead, cls)
        sectors_out.append({
            "sector": sec,
            "n_articles": s["n"],
            "n_bull": s["bull"],
            "n_bear": s["bear"],
            "n_neutral": s["neutral"],
            "n_classified": cls,
            "coherence_pct": round(coh_pct, 1),
            "lead_direction": lead,
            "lead_headline": s["lead_headline"][0] or None,
            "verdict": verdict,
        })

    # Sort: macro-verdict sectors first (high coherence ⇒ actionable),
    # then by classified-volume descending. Stable order matters for the
    # chat-cap below so the top-N is reproducible across cycles.
    _MACRO_RANK = {"MACRO_BULL": 0, "MACRO_BEAR": 0, "TILT_BULL": 1,
                   "TILT_BEAR": 1, "SPLIT": 2, "INSUFFICIENT": 3}
    sectors_out.sort(
        key=lambda x: (_MACRO_RANK.get(x["verdict"], 99), -x["n_classified"])
    )
    out["sectors"] = sectors_out

    # A short top-level headline keeps parity with build_portfolio_beta /
    # build_tail_risk so chat helpers can pass it through verbatim.
    macro_sectors = [
        s for s in sectors_out
        if s["verdict"] in ("MACRO_BULL", "MACRO_BEAR")
    ]
    if not sectors_out:
        out["headline"] = "Sector coherence: no sector-tagged news in window."
    elif not macro_sectors:
        # No macro story; report the tightest tilt as a soft signal so
        # the operator still sees the room temperature.
        op = sectors_out[0]
        out["headline"] = (
            f"Sector coherence: no macro story — top "
            f"{op['sector']} {op['verdict']} "
            f"({op['n_bull']}↑/{op['n_bear']}↓, "
            f"{op['coherence_pct']:.0f}% coh)."
        )
    else:
        parts = [
            f"{s['sector']} {s['verdict']} "
            f"({s['n_bull']}↑/{s['n_bear']}↓, "
            f"{s['coherence_pct']:.0f}% coh)"
            for s in macro_sectors
        ]
        out["headline"] = "Sector coherence: " + "; ".join(parts) + "."
    return out
