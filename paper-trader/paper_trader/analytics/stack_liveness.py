"""Stack-Liveness — the single "is this whole trading stack actually live?"
verdict.

The chronic operational pathology of this two-service stack is that
*something* always lies. ``paper-trader.service`` runs but its boot SHA is
40 commits behind HEAD (committed fixes inert until restart — the documented
``paper-trader-chronic-stale`` regime). ``digital-intern.service`` runs but
its ``/healthz`` blocks 60s on a hung worker. ``articles.db`` exists but its
newest row is 6 hours old. ``decision_scorer.pkl`` exists but a synthetic
n=39 run clobbered it (the 2026-05-23 finding-#1 footprint). Each of these
is reachable from a separate endpoint; no single surface answers the literal
first operator question: **"is anything wrong with the stack right now?"**

This is the missing single-call stoplight. It composes signals every other
panel already exposes and reduces them to one HEALTHY / DEGRADED / DARK
verdict plus a per-component breakdown. The constituent diagnostics are the
authoritative sources of truth — ``stack_liveness`` mints no new opinions
(``desk_pulse`` router precedent; AGENTS.md invariants #2/#12 — Opus has
full autonomy, this surfaces *facts*, not decisions). Never raises.

Components:

* ``trader_loop`` — newest decision row age vs the runner's expected cadence
  (composed verbatim via ``runner_heartbeat.build_runner_heartbeat``)
* ``trader_sha`` — is the live process running a SHA older than HEAD? Boot
  SHA, head SHA, behind count come from the endpoint (the canonical
  ``_head_sha_and_behind`` call).
* ``scorer_pkl`` — does the deployed ``data/ml/decision_scorer.pkl`` look
  sane (n_train ≥ threshold, pred_quantiles not collapsed to a single
  value — the documented synthetic-clobber footprint)?
* ``intern`` — is ``http://localhost:8080/api/healthz`` reachable within
  a short total timeout?
* ``articles_db`` — newest ``first_seen`` in digital-intern's ``articles.db``
  vs the documented worker cadences (most workers tick ≤ 10 min — a 30+
  minute gap is a stack-wide collector outage).

Each component → ``status`` ∈ {HEALTHY, DEGRADED, DARK, UNKNOWN}. The
overall verdict is the worst component (DARK > DEGRADED > UNKNOWN >
HEALTHY) — a single dark component dominates everything (the
``trader_scorecard._FOCUS_ORDER`` "operational concerns above behavioural"
precedent). Advisory only; no path to ``_execute()``.

Threshold-driven so verdicts are exactly testable. Tests read the
constants — a retune cannot false-fail.
"""
from __future__ import annotations

# How many commits behind HEAD before the trader SHA component goes
# DEGRADED. 0 means any drift at all is flagged; 1+ tolerates an in-flight
# commit that hasn't restarted yet. Chosen at 1 to match the
# build-info ``stale`` flag's existing semantics (any non-zero behind == stale).
TRADER_SHA_DEGRADED_BEHIND = 1

# Minimum n_train for the deployed scorer pickle. The 2026-05-23 synthetic
# clobber wrote n_train=39 — anything below the gate-threshold (500) is
# operationally meaningless and any below 100 is overtly broken.
SCORER_PKL_MIN_N_TRAIN = 100

# Articles.db freshness ceiling. The fastest collector (rss_worker) ticks
# every 30s; the median worker every 2-5min; the slowest every 30min.
# 30 minutes of NO new article is the moment every collector is dark
# (the documented chronic-dark-collectors pathology takes channels offline
# but the aggregate stream rarely goes silent more than a few minutes).
ARTICLES_DB_DEGRADED_MIN = 30.0  # minutes
ARTICLES_DB_DARK_MIN = 120.0      # minutes — full stack-wide outage


# Component status ordering: worst dominates. Strings, not an Enum, so
# the dashboard JS can branch on the raw verdict and the test reads them
# verbatim.
_STATUS_ORDER = {"HEALTHY": 0, "UNKNOWN": 1, "DEGRADED": 2, "DARK": 3}


def _num(x):
    """Coerce to float-or-None; never raise on garbage input."""
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _build_trader_sha(build_info) -> dict:
    """Reduce build_info to a component verdict.

    build_info shape (from the existing ``/api/build-info`` source of truth):
    ``{boot_sha, head_sha, behind, stale}``. ``behind`` may be None when
    git is unreachable — degrade gracefully to UNKNOWN.
    """
    if not isinstance(build_info, dict):
        return {"status": "UNKNOWN", "headline": "UNKNOWN — build_info unavailable",
                "boot_sha": None, "head_sha": None, "behind": None}
    boot = build_info.get("boot_sha")
    head = build_info.get("head_sha")
    behind = build_info.get("behind")
    stale = bool(build_info.get("stale"))

    if behind is None or head is None or boot is None:
        return {"status": "UNKNOWN",
                "headline": "UNKNOWN — could not resolve boot/head SHA",
                "boot_sha": boot, "head_sha": head, "behind": behind}
    try:
        behind_i = int(behind)
    except (TypeError, ValueError):
        behind_i = 0
    if stale and behind_i >= TRADER_SHA_DEGRADED_BEHIND:
        return {"status": "DEGRADED",
                "headline": (f"DEGRADED — trader process is {behind_i} commit(s) "
                             f"behind HEAD (boot={boot[:7] if boot else '—'}, "
                             f"head={head[:7] if head else '—'}). Restart to "
                             f"pick up committed fixes."),
                "boot_sha": boot, "head_sha": head, "behind": behind_i}
    return {"status": "HEALTHY",
            "headline": (f"HEALTHY — trader process is at HEAD "
                         f"({boot[:7] if boot else '—'})."),
            "boot_sha": boot, "head_sha": head, "behind": behind_i}


def _build_trader_loop(runner_heartbeat) -> dict:
    """Reduce a runner_heartbeat report to a single status.

    runner_heartbeat shape (from the existing builder):
    ``{verdict ∈ {HEALTHY, LAGGING, STALLED, NO_DATA, ...}, headline, ...}``.
    """
    if not isinstance(runner_heartbeat, dict):
        return {"status": "UNKNOWN",
                "headline": "UNKNOWN — runner heartbeat unavailable"}
    verdict = (runner_heartbeat.get("verdict") or "").upper()
    headline = runner_heartbeat.get("headline") or ""
    # Map the runner_heartbeat enum onto the stack-liveness enum.
    if verdict == "HEALTHY":
        status = "HEALTHY"
    elif verdict in ("NO_DATA", "INSUFFICIENT_DATA"):
        status = "UNKNOWN"
    elif verdict == "STALLED":
        status = "DARK"
    elif verdict in ("LAGGING", "IDLE_STORM", "HOLD_LOCK"):
        status = "DEGRADED"
    else:
        status = "DEGRADED"  # any unrecognised non-healthy verdict
    return {"status": status, "headline": headline or f"{verdict} (loop)",
            "verdict": verdict}


def _build_scorer_pkl(scorer_pkl_info) -> dict:
    """Reduce a parsed scorer.pkl summary to a status.

    Input dict shape (the endpoint owns the pickle load):
        ``{exists: bool, n_train: int | None, pred_collapsed: bool | None,
           error: str | None}``
    """
    if not isinstance(scorer_pkl_info, dict):
        return {"status": "UNKNOWN", "headline": "UNKNOWN — scorer pkl unavailable",
                "n_train": None}
    if scorer_pkl_info.get("error"):
        return {"status": "DEGRADED",
                "headline": (f"DEGRADED — scorer pkl error: "
                             f"{scorer_pkl_info['error']}"),
                "n_train": scorer_pkl_info.get("n_train")}
    if not scorer_pkl_info.get("exists"):
        return {"status": "DEGRADED",
                "headline": "DEGRADED — no decision_scorer.pkl on disk",
                "n_train": None}
    n_train = scorer_pkl_info.get("n_train")
    collapsed = scorer_pkl_info.get("pred_collapsed")
    try:
        n_i = int(n_train) if n_train is not None else None
    except (TypeError, ValueError):
        n_i = None
    if collapsed:
        return {"status": "DARK",
                "headline": (f"DARK — scorer pkl predictions collapsed to a "
                             f"single value (synthetic-clobber footprint; "
                             f"n_train={n_i})."),
                "n_train": n_i, "pred_collapsed": True}
    if n_i is not None and n_i < SCORER_PKL_MIN_N_TRAIN:
        return {"status": "DEGRADED",
                "headline": (f"DEGRADED — scorer n_train={n_i} (< "
                             f"{SCORER_PKL_MIN_N_TRAIN}); pickle is below the "
                             f"meaningful-prediction floor."),
                "n_train": n_i, "pred_collapsed": False}
    return {"status": "HEALTHY",
            "headline": f"HEALTHY — scorer pkl n_train={n_i}",
            "n_train": n_i, "pred_collapsed": False}


def _build_intern(intern_reachable, intern_error=None) -> dict:
    """digital-intern dashboard reachability — bool input, status output."""
    if intern_reachable is True:
        return {"status": "HEALTHY",
                "headline": "HEALTHY — digital-intern dashboard reachable"}
    if intern_reachable is False:
        return {"status": "DARK",
                "headline": ("DARK — digital-intern dashboard unreachable "
                             f"({intern_error or 'no response'}). "
                             "Live news pipeline is dark.")}
    return {"status": "UNKNOWN",
            "headline": "UNKNOWN — intern probe skipped"}


def _build_articles_db(articles_db_age_minutes) -> dict:
    """Newest row age in articles.db → status.

    Input: minutes-since-newest-first_seen, or None if unknown.
    """
    age = _num(articles_db_age_minutes)
    if age is None:
        return {"status": "UNKNOWN",
                "headline": "UNKNOWN — articles.db age unavailable",
                "age_minutes": None}
    if age >= ARTICLES_DB_DARK_MIN:
        return {"status": "DARK",
                "headline": (f"DARK — newest article is {age:.0f} min old "
                             f"(≥ {ARTICLES_DB_DARK_MIN:.0f} min). Every "
                             f"collector is dark."),
                "age_minutes": round(age, 2)}
    if age >= ARTICLES_DB_DEGRADED_MIN:
        return {"status": "DEGRADED",
                "headline": (f"DEGRADED — newest article is {age:.0f} min old "
                             f"(≥ {ARTICLES_DB_DEGRADED_MIN:.0f} min). Some "
                             f"collectors are dark."),
                "age_minutes": round(age, 2)}
    return {"status": "HEALTHY",
            "headline": f"HEALTHY — newest article is {age:.1f} min old",
            "age_minutes": round(age, 2)}


def build_stack_liveness(*, build_info=None, runner_heartbeat=None,
                          scorer_pkl_info=None, intern_reachable=None,
                          intern_error=None, articles_db_age_minutes=None):
    """Compose all component reports into a single stack-liveness payload.

    Args (all keyword-only — order-independent, easier to extend with new
    components without breaking existing callers):
        build_info: ``{boot_sha, head_sha, behind, stale}`` or None
        runner_heartbeat: output of ``runner_heartbeat.build_runner_heartbeat`` or None
        scorer_pkl_info: ``{exists, n_train, pred_collapsed, error}`` or None
        intern_reachable: bool or None (None ⇒ probe skipped)
        intern_error: optional str describing the probe failure
        articles_db_age_minutes: float minutes since newest article first_seen,
            or None when unknown

    Returns a dict with ``verdict ∈ {HEALTHY, DEGRADED, DARK, UNKNOWN}``,
    a ``headline`` (the worst component's headline), and per-component
    sub-blocks under ``components``.
    """
    components = {
        "trader_sha": _build_trader_sha(build_info),
        "trader_loop": _build_trader_loop(runner_heartbeat),
        "scorer_pkl": _build_scorer_pkl(scorer_pkl_info),
        "intern": _build_intern(intern_reachable, intern_error),
        "articles_db": _build_articles_db(articles_db_age_minutes),
    }

    # Pick the worst component to drive the top-level verdict. Tie-break
    # in a documented operational priority order: a DARK trader_loop is
    # more urgent than a DARK articles_db (you can still observe from a
    # paused intern, but a dead trader cannot trade).
    _PRIORITY = ("trader_loop", "scorer_pkl", "trader_sha",
                 "intern", "articles_db")

    def _rank(comp):
        return _STATUS_ORDER.get(comp.get("status"), 1)

    worst_rank = max((_rank(c) for c in components.values()), default=0)
    worst_name = None
    for name in _PRIORITY:
        comp = components.get(name)
        if comp and _rank(comp) == worst_rank:
            worst_name = name
            break
    if worst_name is None:
        worst_name = next(iter(components))

    verdict = {0: "HEALTHY", 1: "UNKNOWN", 2: "DEGRADED", 3: "DARK"}[worst_rank]
    headline = components[worst_name]["headline"]
    if verdict != "HEALTHY":
        headline = f"[{worst_name}] {headline}"
    else:
        headline = "HEALTHY — all components green."

    return {
        "verdict": verdict,
        "headline": headline,
        "worst_component": worst_name if verdict != "HEALTHY" else None,
        "components": components,
    }
