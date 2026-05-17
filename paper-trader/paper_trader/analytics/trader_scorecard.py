"""Trader behavioural scorecard — a verdict-alignment *router*, not a grader.

paper-trader has ~24 analytics builders, each answering one narrow
behavioural question, each with its own panel and chat line. There is no
synthesis: an operator must read a dozen panels to learn whether independent
diagnostics *agree* on a problem. When three independent builders flag the
same pathology, that concordance is the real signal.

This module **mints no new opinion** (paper-trader AGENTS.md invariants
#2/#12, and the ``self_review`` "observational, never prescriptive" precedent
it mirrors). It composes the five pure, network-free, DB-read-only behavioural
builders **verbatim** (single source of truth, invariant #10), classifies each
builder's *own* verdict into FLAG / OK / IMMATURE, counts where independent
checks concur, and forwards the builders' own headlines. No grade, no
directive, no cap, and — unlike ``self_review`` — it is **not** injected into
the live decision prompt (it is human/dashboard/chat only, like every endpoint
except self-review). The load-bearing ``strategy.decide()`` path is untouched.

Pure: feed it the same ``store`` reads ``self_review`` takes. ``now`` is
injectable for deterministic tests. ``trades`` MUST be store-native
**newest-first** (``store.recent_trades()``); the asymmetry/churn consumers
are internally fed ``list(reversed(trades))`` exactly as ``/api/analytics`` /
``/api/trade-asymmetry`` / ``/api/churn`` do, while the
liquidity/paralysis/reliability path wants the newest-first order.

Never raises: a failing constituent degrades to an ``ERROR`` check class
(never counted as a flag — an invented pathology would be worse than a known
gap). The contract is "no scorecard this cycle", never an exception.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .trade_asymmetry import build_trade_asymmetry
from .churn import build_churn
from .capital_paralysis import build_capital_paralysis
from .decision_reliability import build_decision_reliability
from .open_attribution import build_open_attribution

# Each builder reports its verdict under a different key (mirrors the
# builders' own contracts — asymmetry/churn gate a `verdict` behind a STABLE
# `state`; capital_paralysis/decision_reliability use `state`;
# open_attribution uses `status`).
_VERDICT_KEY = {
    "trade_asymmetry": "verdict",
    "churn": "verdict",
    "capital_paralysis": "state",
    "decision_reliability": "state",
    "open_attribution": "status",
}

# Documented classification table. Anything not listed → IMMATURE (fail safe:
# never invent a FLAG out of a label a builder added later). Keyed by builder
# so two builders that happen to share a label string can't collide.
_FLAG = {
    "trade_asymmetry": {"PAYOFF_TRAP", "DISPOSITION_BLEED"},
    "churn": {"CHURNING"},
    "capital_paralysis": {"PINNED", "EMPTY"},
    # STALE_LEGACY_DOMINATED carries restart_recommended=True — a real
    # operational problem, so it flags (matches the builder's own framing).
    "decision_reliability": {"CRITICAL", "DEGRADED", "STALE_LEGACY_DOMINATED"},
    "open_attribution": {"SELECTION_DRAG"},
}
_OK = {
    "trade_asymmetry": {"EDGE_POSITIVE", "FLAT"},
    "churn": {"BUY_AND_HOLD", "ACTIVE_TURNOVER"},
    "capital_paralysis": {"FREE"},
    "decision_reliability": {"HEALTHY"},
    "open_attribution": {"SELECTION_ADDING", "FLAT_VS_SPY"},
}

# Coarse theme per builder — used only to count concordance (≥2 independent
# builders flagging the same theme). No new opinion; just agreement.
_THEME = {
    "trade_asymmetry": "EXIT_DISCIPLINE",
    "churn": "EXIT_DISCIPLINE",
    "capital_paralysis": "CAPITAL_TRAP",
    "decision_reliability": "DECISION_INTEGRITY",
    "open_attribution": "SELECTION",
}

# Documented severity precedence for the single `focus` pointer. This is a
# factual ordering (same pattern as trade_asymmetry's verdict precedence and
# thesis_drift's worst-first sort) — it forwards the chosen builder's headline
# verbatim and mints no number. Lower index = higher precedence.
#  - DECISION_INTEGRITY first: if the engine isn't producing decisions,
#    nothing downstream matters.
#  - CAPITAL_TRAP next: pinned capital can't act on anything.
#  - EXIT_DISCIPLINE: PAYOFF_TRAP (negative expectancy) before
#    DISPOSITION_BLEED (net-positive but leaking) before CHURNING.
#  - SELECTION last.
_FOCUS_ORDER = [
    ("decision_reliability", None),
    ("capital_paralysis", None),
    ("trade_asymmetry", "PAYOFF_TRAP"),
    ("trade_asymmetry", "DISPOSITION_BLEED"),
    ("churn", "CHURNING"),
    ("open_attribution", None),
]


def classify_check(name: str, result: dict) -> str:
    """FLAG | OK | IMMATURE | ERROR for one builder's output. Pure."""
    if not isinstance(result, dict):
        return "IMMATURE"
    # _safe's typed error marker (self_review precedent).
    if result.get("state") == "ERROR" or result.get("status") == "ERROR":
        return "ERROR"
    label = result.get(_VERDICT_KEY.get(name, "state"))
    if label in _FLAG.get(name, set()):
        return "FLAG"
    if label in _OK.get(name, set()):
        return "OK"
    return "IMMATURE"


def _label(name: str, result: dict) -> str | None:
    if not isinstance(result, dict):
        return None
    return result.get(_VERDICT_KEY.get(name, "state"))


def _safe(fn, *args, **kwargs) -> dict:
    """Run one builder; on any failure return a typed empty marker rather than
    letting a single bad builder sink the whole scorecard (self_review
    precedent — the failure mode is 'no scorecard', never an exception)."""
    try:
        out = fn(*args, **kwargs)
        return out if isinstance(out, dict) else {}
    except Exception as e:  # pragma: no cover - exercised via monkeypatch
        return {"state": "ERROR", "status": "ERROR",
                "error": f"{type(e).__name__}: {e}"}


def build_trader_scorecard(portfolio: dict,
                           positions: list[dict],
                           trades: list[dict],
                           decisions: list[dict],
                           equity_curve: list[dict],
                           now: datetime | None = None) -> dict:
    """Compose the five pure behavioural builders into one descriptive
    verdict-alignment view. Pure; never raises."""
    now = now or datetime.now(timezone.utc)

    rev = list(reversed(trades or []))  # oldest→newest for round_trips
    raw = {
        "trade_asymmetry": _safe(build_trade_asymmetry, rev, now=now),
        "churn": _safe(build_churn, rev, now=now),
        "capital_paralysis": _safe(build_capital_paralysis, portfolio or {},
                                   positions or [], trades or [],
                                   decisions or [], equity_curve or [],
                                   now=now),
        "decision_reliability": _safe(build_decision_reliability,
                                      decisions or [], equity_curve or [],
                                      now=now),
        "open_attribution": _safe(build_open_attribution, positions or [],
                                  equity_curve or [], now=now),
    }

    checks: list[dict] = []
    for name in ("trade_asymmetry", "churn", "capital_paralysis",
                 "decision_reliability", "open_attribution"):
        res = raw[name]
        klass = classify_check(name, res)
        checks.append({
            "name": name,
            "label": _label(name, res),
            "klass": klass,
            "theme": _THEME[name],
            "headline": res.get("headline"),
        })

    flags = [c for c in checks if c["klass"] == "FLAG"]
    n_flags = len(flags)
    n_ok = sum(1 for c in checks if c["klass"] == "OK")
    n_immature = sum(1 for c in checks if c["klass"] == "IMMATURE")
    n_error = sum(1 for c in checks if c["klass"] == "ERROR")

    # Concordance: ≥2 independent builders flagging the same theme. Factual —
    # count + the builders' own verbatim labels, no synthesized judgement.
    concordance: list[dict] = []
    by_theme: dict[str, list[dict]] = {}
    for c in flags:
        by_theme.setdefault(c["theme"], []).append(c)
    for theme, group in by_theme.items():
        if len(group) >= 2:
            concordance.append({
                "theme": theme,
                "count": len(group),
                "labels": [g["label"] for g in group],
                "checks": [g["name"] for g in group],
            })
    concordance.sort(key=lambda n: (-n["count"], n["theme"]))

    # focus: the single highest-precedence flag, headline forwarded verbatim.
    focus = None
    for fname, flabel in _FOCUS_ORDER:
        match = next(
            (c for c in flags
             if c["name"] == fname and (flabel is None or c["label"] == flabel)),
            None)
        if match:
            focus = {"name": match["name"], "label": match["label"],
                     "theme": match["theme"], "headline": match["headline"]}
            break

    n_mature = n_flags + n_ok
    if n_mature == 0:
        state = "NO_DATA"
        headline = "No mature behavioural history yet."
    elif n_flags == 0:
        state = "ALIGNED_HEALTHY"
        headline = (f"All {n_ok} mature behavioural check"
                    f"{'s' if n_ok != 1 else ''} healthy.")
    else:
        state = "FLAGS_PRESENT"
        labels = ", ".join(c["label"] for c in flags if c["label"])
        headline = (f"{n_flags} of {len(checks)} behavioural checks "
                    f"flagging: {labels}.")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "state": state,
        "headline": headline,
        "n_total": len(checks),
        "n_flags": n_flags,
        "n_ok": n_ok,
        "n_immature": n_immature,
        "n_error": n_error,
        "focus": focus,
        "concordance": concordance,
        "flags": [{"name": c["name"], "label": c["label"],
                   "theme": c["theme"], "headline": c["headline"]}
                  for c in flags],
        "checks": checks,
    }


if __name__ == "__main__":  # smoke against the live DB
    import json
    from paper_trader.store import get_store
    s = get_store()
    rep = build_trader_scorecard(
        s.get_portfolio(), s.open_positions(), s.recent_trades(2000),
        s.recent_decisions(limit=3000), s.equity_curve(limit=5000))
    print(json.dumps(rep, indent=2, default=str))
