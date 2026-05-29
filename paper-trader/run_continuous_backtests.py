#!/usr/bin/env python3
"""Continuous backtesting loop — persona-driven, scorer-trained.

Each cycle:
  1. Runs RUNS_PER_CYCLE (5) parallel year-long backtests. Each run uses
     a distinct persona; signal differences come from per-persona ticker
     boosts and different RNG seeds.
  2. Sorts results by total_return_pct and keeps top TOP_RUNS_TO_TRAIN (3)
     positive runs (or the single best if none are positive).
  3. Appends those runs' decisions to data/winner_training.jsonl tagged
     with the cycle number. (Does NOT overwrite — accumulates forever.)
  4. Computes 5-trading-day forward returns for every BUY/SELL decision
     across ALL runs (winners and losers — losing decisions are critical
     signal for the scorer too) and appends them to
     data/decision_outcomes.jsonl, then retrains DecisionScorer.
  5. Spawns a background Opus 4.7 annotator to label the top run's
     decisions GOOD/NEUTRAL/BAD and write a trading lesson — fed back into
     ArticleNet training.
  6. Trims backtest_runs to the most recent KEEP_LAST_RUNS (500) entries.
  7. Sleeps COOLDOWN_SECONDS (60) and loops.

SIGTERM/SIGINT exits cleanly between cycles.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from paper_trader.backtest import (
    BacktestEngine,
    BacktestRun,
    ROOT,
    _get_quant_signals,
    _market_regime,
)

RUNS_PER_CYCLE = 1  # throttled to 1 — load avg 37+, reduce system pressure
TOP_RUNS_TO_TRAIN = 1  # only keep best single run per cycle when throttled
KEEP_LAST_RUNS = 500
N_MONKEY_BT_PER_CYCLE = 20  # monkey runs added to backtest.db each cycle
MAX_OUTCOMES_FOR_TRAINING = 5000  # cap decision_outcomes.jsonl tail used per retrain
COOLDOWN_SECONDS = 600  # throttled from 300s — 10 min cooldown to give box breathing room
DISCORD_CHANNEL = "channel:1496099475838603324"
WINNER_JSONL = ROOT / "data" / "winner_training.jsonl"
# winner_training.jsonl is append-only across the whole loop lifetime
# (`_append_top_decisions` + `_opus_annotate` both append, nothing trimmed).
# It had grown to ~320 MB / 860k lines on disk. `_inject_and_train` already
# only ever consumes the last `_MAX_INJECT_RECORDS` (10k) lines — older rows
# are already idempotently in articles.db (INSERT OR IGNORE) — so an unbounded
# file is pure disk waste and a latent disk-full risk (the same OSError
# [Errno 28] class of failure documented in decision_scorer.py). Trim to this
# many most-recent records, well above the 10k inject tail so the consumer is
# never starved, using the same atomic tmp+`.replace` idiom as the
# decision_outcomes / scorer_skill_log trims.
WINNER_JSONL_KEEP = 50000
# digital-intern's article DB that `_inject_and_train` writes winner rows into.
# Module-level (not a function local) so it can be redirected in tests.
DIGITAL_INTERN_ARTICLES_DB = "/home/zeph/digital-intern/data/articles.db"

# How often to run the validation suite (label audit + permutation test) on the
# current cycle's engine. Validation is *expensive* (one full backtest per
# permutation), so it runs in a background thread — every 10 cycles is enough
# to catch regressions without dominating compute.
#
# n=250: with the smoothed permutation p-value (k+1)/(n+1) the minimum
# achievable p is 1/(n+1). At n=5 that floor is 0.167 (a "p=0.000" is
# mathematically impossible / meaningless); at n=250 the floor is ~0.004,
# which is the first count low enough to ever clear a real p<0.05 bar with
# headroom. The validation runner already tolerates per-permutation crashes
# and now reports n_attempted vs n_successful, so a high N degrades
# gracefully instead of silently shrinking to ~5 successes.
VALIDATION_EVERY_N_CYCLES = 10
VALIDATION_PERMUTATIONS = 250   # statistically valid floor (smoothed p ≥ 1/(n+1))
VALIDATION_RESULTS_PATH = ROOT / "data" / "validation_results.json"
VALIDATION_RESULTS_KEEP = 50    # cap file growth

# Per-cycle scorer-skill ledger. `_train_decision_scorer` already computes
# val_rmse / oos_rmse / oos_diracc / oos_ic every cycle but only *prints* them
# to continuous.log — an ephemeral, rotated, hard-to-trend sink. A skeptical
# quant needs to see whether the scorer's out-of-sample skill is improving
# with more outcomes, holding at the documented negative-skill plateau
# (AGENTS.md: oos_rmse 13–17 > σ(target) ≈ 11.7), or degrading. This appends
# one structured JSONL row per cycle so that trend is durable and queryable.
# Module-level (not a function local) so tests can redirect it — mirrors the
# AGENTS.md "hardcoded paths must be module-level for testability" rule.
SCORER_SKILL_LOG = ROOT / "data" / "scorer_skill_log.jsonl"
SCORER_SKILL_LOG_KEEP = 2000    # cap file growth (≈ one row per cycle)

# Per-cycle trivial-baseline ledger. `scorer_skill_log.jsonl` (read by
# `skill_trend`) trends the scorer's `oos_rmse` against a *constant*
# mean-predictor. The single most economically decisive documented finding
# across ~10 ML/backtest review passes is a *different* question entirely:
# `baseline_compare` shows a one-line rule (raw `ml_score`) carries higher OOS
# rank-IC than the 17-feature MLP — `MLP_WORSE_THAN_TRIVIAL` — so the
# conviction gate (invariant #5, active every cycle once n_train≥500)
# underwrites pure sizing variance with no compensating edge. That verdict was
# only ever observable by an operator manually running
# `python3 -m paper_trader.ml.baseline_compare` — there was NO durable,
# trendable signal an unattended loop could surface, exactly the gap the
# pass-#15 `_append_scorer_skill_log` wiring fix closed for the scorer ledger.
# This appends one structured row per cycle so a skeptical quant can see
# whether the net stays net-negative-complexity, recovers, or worsens as
# `decision_outcomes.jsonl` accumulates. Module-level (not a function local)
# so tests can redirect it — the same testability rule as SCORER_SKILL_LOG.
BASELINE_SKILL_LOG = ROOT / "data" / "baseline_skill_log.jsonl"
BASELINE_SKILL_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle LLM-annotation skill ledger. ``_llm_annotate_outcomes`` writes
# ENDORSE/CONDEMN labels into ``decision_outcomes.jsonl`` rows; the trainer
# upweights ENDORSEs 3× and downweights CONDEMNs to 0.1× via the ``llm_mult``
# column. The single most-repeated documented finding for this column is
# ``sample_weight_audit`` reporting ``CURRENT_TIED`` (the LLM multiplier
# provides ZERO OOS rank skill at the trained-model level). Worse, ad-hoc
# inspection of the live ``decision_outcomes.jsonl`` reveals 0/N rows carry
# a non-zero label — the entire annotation pipeline is dark in production
# (missing API key / unreachable host / regex never matching), and the
# silent ``except Exception`` in ``_llm_annotate_outcomes`` hides it. This
# appends one structured row per cycle so a skeptical quant can see the
# darkness immediately, and so the moment it starts working again is
# observable in the trend. Module-level (not a function local) so tests can
# redirect it — the same testability rule as SCORER_SKILL_LOG /
# BASELINE_SKILL_LOG.
LLM_ANNOTATION_SKILL_LOG = ROOT / "data" / "llm_annotation_skill_log.jsonl"
LLM_ANNOTATION_SKILL_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle ``predict_calibrated`` reliability ledger. Pass #10 added
# ``DecisionScorer.predict_calibrated`` (quantile-mapped honest magnitude
# reading) — the CLAIM is that the calibrated value delivers an honest 5d
# return on the empirical label support, but until ``calibration_reliability``
# landed the claim was untested out-of-sample: the existing ``calibration``
# module bins by raw ``predict()`` and the scorer skill ledger only trends
# scalar ``oos_rmse`` / ``oos_ic``. This appends one structured JSONL row per
# cycle so a quant can trend whether the calibrated value's decile-binned
# realized-vs-predicted gap is improving, holding, or degrading — AND
# whether the calibration step measurably narrows the bias vs the raw
# ``predict()`` reading on the SAME OOS pairs (``vs_raw_bias_reduction``).
# Same testability rule as the sibling skill logs (module-level so tests can
# redirect), same best-effort discipline (a ledger write must never break
# the loop).
CALIBRATED_RELIABILITY_LOG = ROOT / "data" / "calibrated_reliability_log.jsonl"
CALIBRATED_RELIABILITY_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle stop-out skill ledger. ``stop_out_audit`` already answers,
# durably, the most economically decisive question about the inherited
# ``backtest._buy`` ``stop_loss = price * 0.92`` band (the -8% downside
# arm) — *does the stop actually save more from limited-loss trades than
# it costs in prematurely-exited recoveries, or is it variance-only chop
# the gate would do better without?* The verdict (STOP_HELPS / STOP_HURTS /
# STOP_NEUTRAL / INSUFFICIENT_DATA) consumes the 2026-05-23
# ``forward_intraperiod_min_5d`` column, but until this ledger landed the
# audit was a CLI-only signal an operator had to manually invoke. The
# historical 8753-row corpus pre-dates the intraperiod feature, so the
# *immediate* state is INSUFFICIENT_DATA on every cycle — the ledger
# surfaces THAT honestly (``stop_dark=True``) so the moment new
# post-feature outcome rows accumulate enough to flip the verdict is
# observable in the trend, not invisible. Same testability rule as the
# sibling skill logs (module-level so tests can redirect), same best-
# effort discipline (a ledger write must never break the loop), same
# atomic tmp + ``.replace`` trim idiom every other ledger uses.
STOP_OUT_SKILL_LOG = ROOT / "data" / "stop_out_skill_log.jsonl"
STOP_OUT_SKILL_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle MFE-conversion / take-profit skill ledger. Sibling to
# ``STOP_OUT_SKILL_LOG`` on the matching upside arm: ``mfe_conversion``
# answers *does the inherited ``take_profit = price * 1.15`` band
# (+15% upside arm) capture more upside than it forfeits in trades that
# would have recovered further?* The verdict (TP_HELPS / TP_HURTS /
# TP_NEUTRAL / INSUFFICIENT_DATA) consumes the matching 2026-05-23
# ``forward_intraperiod_max_5d`` column. Same wiring rationale — without
# the per-cycle ledger the take-profit's realized economic effect was a
# CLI-only one-shot answer. The mean MFE-conversion ratio (endpoint / mfe)
# is the textbook quant "did this trade's peak get captured" signal a
# skeptical quant rides; persisting it per cycle makes that trendable.
# Same testability / best-effort / atomic-trim discipline as every
# sibling ledger.
MFE_SKILL_LOG = ROOT / "data" / "mfe_skill_log.jsonl"
MFE_SKILL_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle multi-horizon stop-band sweep ledger. ``stop_band_sweep.analyze``
# answers the natural extension of ``stop_out_audit``: *across a 2-D grid
# of candidate STOP bands × horizons (5d/10d/20d), is there a (band,
# horizon) cell that measurably beats the deployed (-8%, 5d) cell on
# realized return?* The pass #42 author explicitly flagged this as the
# next step after the 2026-05-26 multi-horizon intraperiod-extremes
# feature landed — without a per-cycle ledger the verdict (CELL_BEATS_DEPLOYED
# / DEPLOYED_OPTIMAL / NO_BAND_HELPS / INSUFFICIENT_DATA) is only available
# via manual CLI invocation, exactly the unobservable-state the sibling
# ledgers were built to fix. Same testability rule as the sibling skill
# logs (module-level so tests can redirect), same best-effort discipline
# (a ledger write must never break the loop), same atomic tmp+``.replace``
# trim idiom every other ledger uses.
STOP_BAND_SWEEP_LOG = ROOT / "data" / "stop_band_sweep_log.jsonl"
STOP_BAND_SWEEP_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle gate-arm historical skill ledger. ``gate_arm_historical.analyze``
# answers the documented quant-decisive question about the conviction gate
# arms (×0.6 / ×0.85 / ×1.0 / ×1.15 / ×1.3): *do the arms the gate actually
# fired at decision time (decoded from the persisted ``gate_scorer_pred``)
# realize differentiated economic outcomes, or is the bucketing just noise?*
# This is the truth-aware sibling of ``gate_audit`` — it reads the gate's
# real then-deployed prediction rather than re-predicting with today's
# pickle, so it pinpoints the gate's actual historical economic effect
# (the quantity ``gate_pnl`` itself documents as outside its verdict
# scope). The current OOS slice verdict is ``GATE_INEFFECTIVE`` (×1.30 arm
# +3.99% vs ×0.60 arm +4.12% — spread -0.13pp, sub-tolerance) and the
# scorer's strong measured rank-IC (+0.48 OOS) does NOT translate into
# arm-bucket skill (``arm_monotone_fraction=0.5``) — exactly the kind of
# state a skeptical quant needs trended per cycle to see whether bucket
# tuning recovers economic edge. Until this wiring landed it was CLI-only
# with no durable trend. Same testability rule as the sibling skill logs
# (module-level so tests can redirect), same best-effort discipline (a
# ledger write must never break the loop), same atomic tmp + ``.replace``
# trim idiom every other ledger uses.
GATE_ARM_SKILL_LOG = ROOT / "data" / "gate_arm_skill_log.jsonl"
GATE_ARM_SKILL_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle gate economic-counterfactual ledger. ``gate_pnl.analyze`` answers
# the single quant-decisive question every existing gate diagnostic
# structurally *cannot*: *aggregated across all five arms and weighted by
# how often each fires, does the conviction-gate multiplier overlay
# (×0.6 / ×0.85 / ×1.0 / ×1.15 / ×1.3) actually ADD or SUBTRACT realized
# return versus not gating at all?* ``gate_audit`` reports the per-arm mean
# returns and a verdict driven solely by the strong-tailwind-minus-headwind
# spread — that deliberately ignores both the three middle arms AND the
# arm-frequency mix. ``gate_arm_historical`` reports per-arm means weighted
# by the gate's TRUE then-deployed prediction (not today's pickle) but
# still does not roll up to one economic number. ``gate_pnl`` rolls all
# five arms into the multiplier-weighted realized mean Σmᵢrᵢ/Σmᵢ minus
# the equal-base mean(rᵢ) — the assumption-free "did the reallocation
# pay" signal a quant deciding *whether to keep the gate* actually asks.
#
# Verdict ladder: ``INSUFFICIENT_DATA`` / ``GATE_SUBTRACTS_RETURN`` /
# ``GATE_RETURN_NEUTRAL`` / ``GATE_ADDS_RETURN`` (thresholded at the
# shared ``EDGE_TOL_PP=1.0pp`` band — equal to ``gate_audit`` for
# cross-tool comparability). Until this wiring landed the verdict was
# CLI-only with no durable per-cycle trend — exactly the same
# operator-blind state the sibling ``_append_gate_arm_skill_log``
# closed for the arms breakdown. Same testability rule as every sibling
# ledger (module-level so tests can redirect), same best-effort
# discipline (a ledger write must never break the loop), same atomic
# tmp + ``.replace`` trim idiom every other ledger uses.
GATE_PNL_SKILL_LOG = ROOT / "data" / "gate_pnl_skill_log.jsonl"
GATE_PNL_SKILL_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle per-persona decision-signal skill ledger. ``persona_skill.analyze``
# already answers, durably, the single quant-decisive question about the
# 10 personas the engine cycles through: *within each persona's own
# decisions, does a stronger signal (``ml_score``, action-aligned) actually
# rank-predict a better realized outcome — i.e. is the persona's return
# real signal skill or pure leveraged-beta dispersion?* The standalone
# ``persona_leaderboard`` aggregates run-level ``vs_spy_pct`` but AGENTS.md
# repeatedly warns that a +1000% / -80% per-run swing is leverage luck,
# NOT strategy skill. ``persona_skill`` is the decision-level honest
# answer — but until this wiring landed it was CLI-only with no durable
# per-cycle trend an unattended operator could see, and the single most
# directly operational state (``HAS_INVERTED_PERSONA`` — at least one
# persona's signal is ANTI-predictive, the data for a pruning decision)
# was visible only via manual invocation. Same testability rule as the
# sibling skill logs (module-level so tests can redirect), same best-
# effort discipline (a ledger write must never break the loop), same
# atomic tmp + ``.replace`` trim idiom every other ledger uses.
PERSONA_SKILL_LOG = ROOT / "data" / "persona_skill_log.jsonl"
PERSONA_SKILL_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle dead-feature audit ledger. ``dead_feature_audit.audit_dead_features``
# answers, durably and per-cycle, the question every existing diagnostic
# misses: *did the most-recently-retrained model actually LEARN from each of
# the 20 input features the build_features contract advertises, or are some
# slots dead-trained on constant zero?* This is the model-level complement
# to ``feature_importance``'s data-level permutation reading. The class of
# bug it catches is exactly the pass-#35 finding: a feature added to
# ``DecisionScorer.build_features`` whose values are never plumbed into
# ``_compute_decision_outcomes`` (training capture) or ``_ml_decide``
# (inference) — the StandardScaler sees a constant-zero column, divides
# by ~zero std, and L2 alpha drives every weight to *exactly* 0.0 (the
# deployed pickle had 3 such slots, mean|w|=0.000000 each, until pass #35
# closed the loop). Until this wiring landed there was NO durable per-cycle
# signal an unattended operator could trend to catch the next regression
# of this class. Same testability rule as the sibling skill logs
# (module-level so tests can redirect), same best-effort discipline (a
# ledger write must never break the loop), same atomic tmp+``.replace``
# trim idiom every other ledger uses.
DEAD_FEATURE_AUDIT_LOG = ROOT / "data" / "dead_feature_audit_log.jsonl"
DEAD_FEATURE_AUDIT_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle (persona × regime) cross-tab skill ledger. Pass #44 shipped
# the ``persona_regime_skill`` analyzer — the missing intersection of
# ``persona_skill`` (per-persona aggregate, hides regime structure) and
# ``regime_audit`` (per-regime aggregate, hides per-persona structure).
# Neither sibling can answer the actionable question
# ``persona_regime_skill`` does: *does THIS persona carry signal in THIS
# regime?* But pass #44 stopped before wiring a per-cycle ledger, so the
# verdict (``REGIME_CONDITIONAL`` / ``HAS_INVERTED_CELL`` / etc.) was
# CLI-only — an unattended operator could not trend per-cell signal
# health, and the most directly operational state
# (``HAS_INVERTED_CELL`` — a specific persona is anti-predictive in a
# specific regime, the actionable data for suppressing that
# persona-in-that-regime) was invisible. The first cycle's live verdict
# on the production corpus was ``REGIME_CONDITIONAL`` with ESG/sideways
# at +0.293 IC AND Momentum/bear at -0.137 IC — exactly the kind of
# state that decays / recovers cycle by cycle and needs trending. This
# closes the wiring gap mirroring every sibling
# ``_append_*_skill_log`` pattern: best-effort, honest-gap rows,
# atomic-bounded trim, SSOT no-drift. Module-level so tests can
# redirect — the same testability rule as SCORER_SKILL_LOG.
PERSONA_REGIME_SKILL_LOG = ROOT / "data" / "persona_regime_skill_log.jsonl"
PERSONA_REGIME_SKILL_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle conviction-sizing calibration ledger. ``conviction_calibration``
# already answers, durably, the most economically decisive question about the
# `_ml_decide` gate's sizing arm (×0.6 / ×0.85 / ×1.15 / ×1.3) on top of the
# base conviction rule (min(0.25, ml_score/20) regular, min(0.40, ml_score/15)
# leveraged-bull): *does the bot's higher-conviction (25–40% of book) call
# realize higher 5-day return than its low-conviction probes?* The CLI verdict
# on the current 1173-row OOS slice is ``MISCALIBRATED`` (spearman +0.011 —
# sizing is variance with no compensating realized edge) but until this ledger
# landed there was NO durable per-cycle trend so the most directly operational
# question about the conviction sizing rule was invisible to an unattended
# operator. Same testability rule as the sibling skill logs (module-level so
# tests can redirect), same best-effort discipline (a ledger write must never
# break the loop), same atomic tmp + ``.replace`` trim idiom every other
# ledger uses.
CONVICTION_CALIBRATION_LOG = ROOT / "data" / "conviction_calibration_log.jsonl"
CONVICTION_CALIBRATION_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

# Per-cycle bootstrap-CI ledger for the deployed scorer's OOS rank-IC,
# RMSE, and dir-acc. ``paper_trader/ml/oos_bootstrap_ci.bootstrap_ci``
# already exists as a CLI / dashboard-feeder, but **nothing trends its
# verdict per cycle**: every other OOS diagnostic in this loop reports
# point estimates (``oos_ic``, ``oos_rmse``, ``oos_dir_acc``) that can
# sit at exactly the noise floor without any signal that the value IS
# the noise floor. With OOS cycle sizes in the hundreds-to-low-thousands
# and the documented near-zero underlying skill plateau
# (``MLP_WORSE_THAN_TRIVIAL`` / ``GATE_INEFFECTIVE``), the operator-
# decisive question is *whether the rank-IC CI excludes 0 right now and
# how that has been trending across cycles* — i.e. has the OOS skill
# become statistically distinguishable from a coin flip, or are the
# headline +0.05/-0.01/+0.08 reads a sampling-noise walk? The CLI can
# answer this for one snapshot; the ledger makes it trendable per cycle.
# Same testability rule as every sibling skill log (module-level so
# tests can redirect), same best-effort discipline (a ledger write must
# never break the loop), same atomic tmp + ``.replace`` trim idiom.
BOOTSTRAP_CI_SKILL_LOG = ROOT / "data" / "bootstrap_ci_skill_log.jsonl"
BOOTSTRAP_CI_SKILL_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)
# Bootstrap resample count. The default in ``oos_bootstrap_ci`` is 1000
# for the CLI; we keep it for the per-cycle ledger too because (a) it is
# the textbook lower bound for stable 2.5%/97.5% percentile estimates,
# and (b) the per-cycle cost on the live ≈1000-row OOS slice is ~1-2s
# (Spearman sort dominates), negligible against the 10-min cooldown.
BOOTSTRAP_CI_N = 1000

# Per-cycle per-sector OOS skill ledger. ``sector_skill.analyze`` already
# answers, durably and CLI-readable, the decisive sector-level question a
# quant researcher asks AFTER seeing the headline scorer rank-IC: *is the
# scorer's rank skill UNIFORM across the watchlist universe, or carried by
# one fat sector (typically tech, ~89% of the live decision_outcomes
# corpus)?* The sibling ``persona_skill`` / ``persona_regime_skill`` /
# ``dead_feature_audit`` ledgers all close exactly this kind of
# CLI-only-state gap for an unattended operator. ``sector_skill`` was
# conspicuously the LAST major skill diagnostic NOT wired into a per-cycle
# ledger — its verdicts (``HAS_INVERTED_SECTOR`` / ``SECTOR_CONCENTRATED``
# / ``NO_SECTOR_EDGE`` / ``HEALTHY``) were only knowable by manually
# invoking ``python3 -m paper_trader.ml.sector_skill``. The single most
# operational state — ``HAS_INVERTED_SECTOR`` (a sector whose
# rank-IC ≤ -0.15: the more confident the scorer is, the WORSE the
# realized 5d outcome — gating on it is actively harmful) — was invisible
# to the loop. This wiring closes the gap mirroring every sibling
# ``_append_*_skill_log`` pattern: best-effort, honest-gap rows
# (``signal_dark=True`` when corpus < MIN_RECORDS or every sector is
# SPARSE), atomic bounded trim, SSOT no-drift (re-uses the analyzer's own
# verdicts/sectors output verbatim). Module-level so tests can redirect —
# the same testability rule as every sibling skill log path constant.
SECTOR_SKILL_LOG = ROOT / "data" / "sector_skill_log.jsonl"
SECTOR_SKILL_LOG_KEEP = 2000  # cap file growth (≈ one row per cycle)

EARLIEST_WINDOW_START = date(1993, 2, 1)  # SPY inception — ~30+ years of history
WINDOW_END_BUFFER_DAYS = 180  # never end a window within 6 months of today
MIN_WINDOW_YEARS = 1
MAX_WINDOW_YEARS = 10


def _pick_window(seed: int) -> tuple[date, date]:
    """Pick a deterministic random backtest window given a seed.

    Duration is 1–10 years; start lies between 1993-02-01 (~30yr history) and
    (today - duration - 180d) so the window always ends at least 6 months before today.
    """
    rng = random.Random(seed)
    duration_years = rng.randint(MIN_WINDOW_YEARS, MAX_WINDOW_YEARS)
    duration_days = duration_years * 365

    latest_start = date.today() - timedelta(days=duration_days + WINDOW_END_BUFFER_DAYS)
    span = (latest_start - EARLIEST_WINDOW_START).days
    if span < 0:
        # Pathological: today is within ~5.5yr of EARLIEST. Clamp.
        start = EARLIEST_WINDOW_START
    else:
        start = EARLIEST_WINDOW_START + timedelta(days=rng.randint(0, span))
    end = start + timedelta(days=duration_days)
    return start, end


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _next_run_id(engine: BacktestEngine) -> int:
    # Serialise through the store lock — a background _opus_annotate thread from
    # the previous cycle may still be using the same sqlite3 connection, and
    # concurrent use of one connection across threads corrupts cursor state.
    with engine.store._lock:
        row = engine.store.conn.execute(
            "SELECT COALESCE(MAX(run_id), 0) FROM backtest_runs"
        ).fetchone()
    return int(row[0]) + 1


def _trim_history(engine: BacktestEngine, keep: int = KEEP_LAST_RUNS) -> int:
    conn = engine.store.conn
    with engine.store._lock:
        row = conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()
        total = int(row[0])
        if total <= keep:
            return 0
        cutoff = conn.execute(
            "SELECT run_id FROM backtest_runs "
            "ORDER BY run_id DESC LIMIT 1 OFFSET ?",
            (keep,),
        ).fetchone()
        if cutoff is None:
            return 0
        cutoff_id = int(cutoff[0])
        conn.execute("DELETE FROM backtest_trades WHERE run_id <= ?", (cutoff_id,))
        conn.execute("DELETE FROM backtest_decisions WHERE run_id <= ?", (cutoff_id,))
        cur = conn.execute("DELETE FROM backtest_runs WHERE run_id <= ?", (cutoff_id,))
        conn.commit()
        return cur.rowcount or 0


def _reap_orphaned_runs(max_age_hours: float = 6.0) -> int:
    """Mark long-stale ``status='running'`` backtest rows as ``failed``.

    A run thread killed by OOM / SIGKILL never reaches ``finalize_run`` *or*
    the ``run_all`` wrapper's ``upsert_run("failed")`` — that fallback only
    fires on a *caught* Python exception, not a hard kill — so the row stays
    ``running`` forever. That is exactly the documented "Backtest dashboard
    shows running forever" symptom (CLAUDE.md §11): the dashboard renders a
    dead run as in-flight indefinitely and `/api/backtests` is polluted.

    On a fresh continuous-loop start any pre-existing ``running`` row is by
    definition orphaned (the previous process is gone). The age guard is
    defensive belt-and-braces: no real run exceeds minutes (a 10-yr window
    still finishes well under the cycle budget), so a row ``running`` for
    >``max_age_hours`` cannot be a live run even if a second loop ever ran.
    Best-effort and idempotent — a DB hiccup must never stop the loop from
    starting, and a row already ``failed`` is not matched again.

    Resolves ``BACKTEST_DB`` at call time (the AGENTS.md call-time-resolution
    rule) so the conftest tmp redirect is honoured under test.
    """
    from paper_trader.backtest import BACKTEST_DB
    if not Path(BACKTEST_DB).exists():
        return 0
    cutoff = (datetime.now(timezone.utc)
              - timedelta(hours=max_age_hours)).isoformat()
    conn = None
    try:
        conn = sqlite3.connect(str(BACKTEST_DB), timeout=15)
        cur = conn.execute(
            "UPDATE backtest_runs SET status='failed', "
            "notes=COALESCE(notes,'')||' [reaped: orphaned running row]' "
            "WHERE status='running' AND started_at < ?",
            (cutoff,),
        )
        conn.commit()
        return cur.rowcount or 0
    except Exception as e:
        print(f"[continuous] orphaned-run reap failed: {e}")
        return 0
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _append_top_decisions(engine: BacktestEngine, top_runs: list[BacktestRun],
                          cycle: int) -> int:
    """Aggregate BUY/SELL decisions from top N runs into WINNER_JSONL.

    Records are weighted by each run's return — higher-return runs contribute
    decisions with higher ai_score so the ML trainer up-weights them.
    """
    WINNER_JSONL.parent.mkdir(parents=True, exist_ok=True)
    # Normalise returns to [0.5, 1.0] weight range so even 2nd/3rd place matter
    returns = [r.total_return_pct for r in top_runs]
    max_ret = max(returns) if returns else 1.0
    min_ret = min(returns) if returns else 0.0
    span = max_ret - min_ret or 1.0

    written = 0
    with WINNER_JSONL.open("a") as fh:
        for run in top_runs:
            weight = 0.5 + 0.5 * (run.total_return_pct - min_ret) / span
            try:
                # Hold the store lock — the background _opus_annotate thread
                # may share this sqlite3 connection across threads.
                with engine.store._lock:
                    # Training-integrity invariant: only decisions that actually
                    # EXECUTED (status='FILLED') may seed winner_training.jsonl /
                    # ArticleNet. `run_one` records a terminal non-FILLED row
                    # (status='BLOCKED'/'NO_DECISION') for the last intraday
                    # decision when nothing filled that day; if that last
                    # decision was a BUY/SELL that `_execute_decision` rejected
                    # (e.g. a future position cap, a no-price ticker), it would
                    # otherwise be injected as a phantom training trade that
                    # never moved capital. Empirically 0 such rows exist today
                    # (`_ml_decide` only emits executable decisions), but the
                    # filter makes the "trained only on real fills" invariant
                    # explicit and refactor-proof rather than emergent.
                    rows = engine.store.conn.execute(
                        "SELECT action, ticker, sim_date, reasoning, qty, confidence "
                        "FROM backtest_decisions "
                        "WHERE run_id = ? AND action IS NOT NULL AND action != 'HOLD' "
                        "AND status = 'FILLED'",
                        (run.run_id,),
                    ).fetchall()
            except Exception as e:
                print(f"[continuous] run {run.run_id} read failed: {e}")
                continue
            rank = top_runs.index(run) + 1
            for row in rows:
                action = (row["action"] or "").upper()
                if action not in ("BUY", "SELL"):
                    continue
                rec = {
                    "cycle": cycle,
                    "run_id": run.run_id,
                    "rank": rank,
                    "title": f"{action} {row['ticker']} on {row['sim_date']}",
                    "source": f"backtest_cycle_{cycle}_rank{rank}",
                    "ai_score": round(weight * (5.0 if action == "BUY" else 0.5), 2),
                    "urgency": 1 if rank == 1 else 0,
                    "label": action,
                    "ticker": row["ticker"] or "",
                    "sim_date": row["sim_date"] or "",
                    "qty": row["qty"],
                    "confidence": row["confidence"],
                    "reasoning": row["reasoning"] or "",
                    "return_pct": run.total_return_pct,
                    "weight": round(weight, 3),
                }
                fh.write(json.dumps(rec) + "\n")
                written += 1
    print(f"[continuous] appended {written} records from top {len(top_runs)} runs → {WINNER_JSONL}")
    return written


def _trim_winner_jsonl(keep: int = WINNER_JSONL_KEEP) -> int:
    """Bound winner_training.jsonl growth — keep only the last `keep` records.

    Mirrors the decision_outcomes / scorer_skill_log trim idiom exactly: only
    pay the rewrite when the file is well past the cap (> 2× `keep`), and write
    a temp file then atomically `.replace` so a process kill mid-truncate can
    never leave a torn/empty training file. Older rows are already idempotently
    in articles.db (`_inject_and_train` INSERT OR IGNORE) and `_inject_and_train`
    only ever reads the last 10k lines, so dropping the prefix is lossless for
    every consumer.

    Best-effort and never raises (a trim must not break the loop — same
    discipline as `_append_scorer_skill_log`). The trim runs in the main loop
    thread; a previous cycle's `_opus_annotate` daemon may still be appending
    via its own file handle, so the rare rewrite (≈ once per `keep`/cycle-yield
    cycles) can lose the handful of annotation lines written during the
    sub-second tmp-write+replace window. That is an acceptable cost for a
    gitignored, already-DB-deduped training-augmentation file — the same
    best-effort tradeoff the sibling JSONL trims accept.

    Returns the number of lines dropped (0 if no trim was needed or on fault).
    """
    try:
        if not WINNER_JSONL.exists():
            return 0
        with WINNER_JSONL.open("r") as fh:
            n = sum(1 for ln in fh if ln.strip())
        if n <= keep * 2:
            return 0
        # Stream the tail through a bounded deque so peak memory is capped at
        # `keep` lines, not the whole (hundreds-of-MB) file.
        from collections import deque
        with WINNER_JSONL.open("r") as fh:
            kept = list(deque((ln for ln in fh if ln.strip()), maxlen=keep))
        tmp = WINNER_JSONL.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(ln.rstrip("\n") for ln in kept) + "\n")
        tmp.replace(WINNER_JSONL)
        dropped = n - len(kept)
        print(f"[continuous] trimmed winner_training.jsonl {n} → {len(kept)} "
              f"lines (dropped {dropped})")
        return dropped
    except Exception as e:
        print(f"[continuous] winner_training trim failed: {e}")
        return 0


def _parse_gate_decision(reasoning: str | None) -> tuple[float | None, bool | None]:
    """Extract the conviction gate's **actual historical decision** from a
    backtest decision's ``reasoning`` string.

    ``backtest._ml_decide`` ends a BUY's reasoning with one of:

      * `` scorer=±X.X%``                            — the *then-deployed*
        pickle's predicted 5d return the gate actually modulated conviction on
      * `` scorer=±X.X%(off-dist,gate-skipped)``     — the off-distribution
        guard fired, so the gate **abstained** (conviction left untouched)
      * `` scorer=±X.X%(gate-killed,no-skill)``      — the per-cycle no-skill
        kill-switch (`backtest._should_gate_modulate_conviction`) fired
        because the scorer's trailing OOS BUY rank-IC is at noise; the gate
        **abstained** (conviction left untouched). Same semantics as the
        off-dist abstention from any downstream analyzer's perspective —
        the gate did NOT act, so the bucket assignment cannot be
        attributed to scorer skill and the row must be dropped from
        ``gate_pnl`` / ``gate_arm_historical`` analyses.
      * *(nothing)*                                  — the cycle's scorer was
        untrained / ``n_train < 500``; no gate acted at all

    SELL/HOLD reasoning never carries ``scorer=`` (the gate is BUY-only).

    Returns ``(gate_scorer_pred, gate_off_dist)``:
      * ``gate_scorer_pred`` — the float % the gate saw, or ``None`` when no
        ``scorer=`` token was emitted (untrained / sub-gate cycle, or SELL)
      * ``gate_off_dist``    — ``True`` when EITHER abstention marker
        followed the ``scorer=`` token (``(off-dist`` from the
        off-distribution branch OR ``(gate-killed`` from the no-skill
        kill-switch). Both indicate the gate abstained, so the legacy field
        name is kept for backward-compat (existing analyzers that drop
        ``gate_off_dist=True`` rows correctly drop both abstention types).
        ``False`` if a real prediction was acted on, ``None`` when there was
        no gate decision to characterize.

    Why this matters: every existing gate diagnostic (``gate_audit``,
    ``gate_pnl``) must RE-PREDICT with **today's** deployed pickle on the
    stored features — a counterfactual ("what would the current model say"),
    provably **not** what the gate did at decision time with that cycle's own
    model (``gate_pnl`` documents the resulting reconstruction residual is
    explicitly *NOT in its verdict*). Capturing the true historical prediction
    + abstention makes the gate's realized effect *measurable* rather than
    reconstructed. Pure, total, never raises (a ledger/diagnostic-feeding
    parser must not be able to break the cycle — the ``_parse_scorer_status``
    discipline).
    """
    if not reasoning:
        return None, None
    try:
        # Negative lookbehind for word char OR hyphen so a future emission
        # like `gate-scorer=` or `prev-scorer=` cannot accidentally match
        # (a bare `\b` would fire at the hyphen→word boundary — same gap
        # `_parse_conviction_pct` documents). `scorer=` (NOT `score=`)
        # — the existing `ml_score` regex relies on `score=` not being a
        # substring of `scorer=`; this is the dual side of that documented
        # first-match disambiguation. The `:+.1f` format in `_ml_decide`
        # always emits an explicit sign, so `[+-]?` + digits/dot captures
        # `+5.2` / `-50.0` / `+0.0` exactly.
        m = re.search(r"(?<![\w-])scorer=([+-]?[0-9.]+)%", reasoning)
        if not m:
            return None, None
        try:
            pred: float | None = float(m.group(1))
        except (TypeError, ValueError):
            pred = None
        # Abstention marker only ever trails a `scorer=` token, so it is
        # only meaningful once one was emitted. Both abstention types
        # (off-distribution clamp and the no-skill kill-switch) map to
        # ``off_dist=True`` so existing analyzers that filter on this
        # field correctly drop both — the gate did NOT act in either case.
        off_dist = ("(off-dist" in reasoning) or ("(gate-killed" in reasoning)
        return pred, off_dist
    except Exception:
        return None, None


def _parse_conviction_pct(reasoning: str | None) -> float | None:
    """Extract the position-sizing **conviction** the gate emitted at decision
    time from a backtest decision's ``reasoning`` string.

    ``backtest._ml_decide`` ends every BUY's reasoning with
    ``conviction={conviction:.0%}`` (e.g. ``conviction=25%``), where
    ``conviction`` is the fraction of total portfolio value sized into that
    trade (post gate/regime/leveraged-ETF modulation). The token is already
    emitted by every BUY; capturing it as a structured field in the
    outcome row unlocks sizing-weighted realized analysis:

      * does higher conviction predict higher realized return?
        (calibration of the sizing rule itself)
      * what is the realized return of trades the gate actually sized DOWN
        (×0.6 / ×0.85 arms) vs sized UP (×1.15 / ×1.3 arms) — the GATE's
        net effect on the portfolio rather than the model's rank skill
        (the `gate_pnl` question, but with TRUE then-applied sizing rather
        than a re-predict-and-multiply reconstruction).
      * are there persona × conviction interactions worth pinning?
        (`persona_skill` joined to this column)

    Returns the conviction as a **fraction** in ``[0.0, 1.0]`` (so
    ``conviction=25%`` ⇒ ``0.25``), matching the inference-side variable's
    unit. Returns ``None`` when no ``conviction=`` token is present — SELL
    reasoning never carries one (the token is BUY-only, same scope as the
    `scorer=` gate marker), and a HOLD reasoning ("no high-conviction
    signal …") legitimately omits it. SELL outcomes ALWAYS read ``None``
    here, mirroring the ``gate_scorer_pred`` SELL convention.

    Pure, total, never raises (a ledger-feeding parser must not break the
    cycle — the ``_parse_gate_decision`` discipline). Out-of-range values
    (a malformed reasoning emitting ``conviction=900%``) are clamped to
    ``[0.0, 1.0]`` rather than silently propagating impossible sizing.
    """
    if not reasoning:
        return None
    try:
        # Negative lookbehind for word char OR hyphen so a future emission
        # like `low-conviction=…` or `high-conviction=…` cannot accidentally
        # match. The prior `\bconviction=` was insufficient: `\b` matches
        # between non-word (hyphen) and word (`c`), so it fired inside
        # `low-conviction=`. `(\d+)` covers the documented `:.0%` format
        # exactly; a non-integer would not match and degrade to None.
        m = re.search(r"(?<![\w-])conviction=(\d+)%", reasoning)
        if not m:
            return None
        try:
            pct = float(m.group(1)) / 100.0
        except (TypeError, ValueError):
            return None
        # Clamp to [0, 1]. `_ml_decide` caps conviction at 0.95 (regular) /
        # 0.40 (leveraged-bull), so a real emission is always ≤ 95. Defensive
        # against malformed reasoning that emits an impossible percentage —
        # we round-trip through training the way build_features consumers
        # would expect (a fraction in [0,1]).
        return max(0.0, min(1.0, pct))
    except Exception:
        return None


def _compute_decision_outcomes(engine: "BacktestEngine",
                               top_runs: list["BacktestRun"]) -> list[dict]:
    """Compute actual 5-trading-day forward returns for BUY/SELL decisions.

    Re-uses PriceCache for returns and _get_quant_signals for features so no
    network calls are needed. Returns a list of outcome records ready to pass
    to train_scorer().

    Uses a 5-trading-day forward window (not calendar days) so weekends and
    holidays don't introduce inconsistent windows across decisions.
    """
    import bisect

    outcomes: list[dict] = []
    _quant_cache: dict[tuple, dict] = {}
    # SPY-based regime depends only on sim_date — cache per date so a cycle of
    # 5 runs × ~250 decisions doesn't recompute 1250 identical SPY 50/200 MAs.
    _regime_cache: dict[str, str] = {}

    trading_days = engine.prices.trading_days
    if not trading_days:
        return outcomes

    def _td_index(d: date) -> int:
        # bisect for exact-match trading-day lookup; -1 if not a trading day.
        i = bisect.bisect_left(trading_days, d)
        if i < len(trading_days) and trading_days[i] == d:
            return i
        return -1

    def _fwd_ret_h(ticker: str, sim_d: date, idx: int, h: int) -> float | None:
        """Forward return over `h` trading days from `sim_d` (idx in
        trading_days), or None when the window runs past cached price history,
        either endpoint price is missing, or both walked back to the same
        prior close (a fabricated flat outcome — see `_walk_back_collides`).

        Additive multi-horizon instrumentation (feature, 2026-05-18). The
        DecisionScorer trains ONLY on `forward_return_5d` (unchanged); the
        extra 10d/20d horizons are pure read-only research signal so a
        skeptical quant can ask whether the scorer's ~0 OOS skill is a
        5d-target-noise artifact — AGENTS.md notes leveraged ETFs have noisy
        5d windows but strong multi-month returns, and every existing OOS
        diagnostic (calibration/gate_audit/skill_trend/baseline_compare) can
        only ever see the 5d label. This is best-effort: a None here NEVER
        skips or zeroes the 5d row that training depends on (the 5d path
        below is left byte-identical on purpose).
        """
        ti = idx + h
        if ti < 0 or ti >= len(trading_days):
            return None
        ed = trading_days[ti]
        sim_res = engine.prices.resolved_close_date(ticker, sim_d)
        end_res = engine.prices.resolved_close_date(ticker, ed)
        # Walk-back inversion guard. See the matching comment in the 5d
        # outcome path below: an `end_d` walk-back can land BEFORE `sim_d`
        # for a thin/foreign-calendar ticker, producing a time-reversed
        # "forward return" that silently poisons multi-horizon analytics.
        # `<=` strictly strengthens the prior collision check.
        if (sim_res is None or end_res is None
                or end_res <= sim_res):
            return None
        return round(engine.prices.returns_pct(ticker, sim_d, ed), 4)

    def _fwd_intraperiod_extremes(
        ticker: str, sim_d: date, idx: int, h: int = 5
    ) -> tuple[float | None, float | None]:
        """Return ``(min_pct, max_pct)`` realized return reached at ANY
        close between sim_d+1 and sim_d+h trading days, relative to
        sim_d's close.

        Read-only research signal (additive, 2026-05-23). The DecisionScorer
        trains on the 5d *endpoint* return — but a trade with a +5% 5d return
        that drew down -15% mid-window is a very different trade from one
        that grinded straight up to +5%. Persisting the intraperiod extremes
        unlocks risk-adjusted analytics:

          * **forward_intraperiod_min_5d** — worst realized drawdown across
            the window. A high-conviction BUY whose intraperiod min is
            below the gate's implied -8% stop (the documented stop_loss
            band from ``_buy``) would have been stopped out in live
            execution even when the endpoint reading is positive; the field
            makes that condition queryable from the corpus.
          * **forward_intraperiod_max_5d** — best realized peak across the
            window. Pairs with the min so a downstream analyzer can compute
            captured-upside ratio (``forward_return_5d / forward_intraperiod_max_5d``)
            — high-conviction calls that peak then crater have ratio ≪ 1.0.

        Same defensive contract as ``_fwd_ret_h``: a None here never blocks
        the 5d field that the scorer trains on. ``build_features`` /
        ``train_scorer`` ignore unknown dict keys, so adding these columns
        is zero-risk for the gate/trade path — pure additive instrumentation
        (the same 2026-05-18 ``forward_return_10d/20d`` precedent).

        Returns ``(None, None)`` when the sim-side resolved close is missing
        OR the entire forward window has no resolvable closes; partial
        coverage (some but not all of the 1..h days have data) is honored —
        any close that resolves contributes to the running extremes.
        Tickers with very thin/foreign calendars therefore degrade
        gracefully rather than dropping the whole outcome row.
        """
        sim_res = engine.prices.resolved_close_date(ticker, sim_d)
        if sim_res is None:
            return None, None
        sim_close = engine.prices.price_on(ticker, sim_d)
        if not sim_close or sim_close <= 0:
            return None, None
        min_pct: float | None = None
        max_pct: float | None = None
        for k in range(1, h + 1):
            ti = idx + k
            if ti < 0 or ti >= len(trading_days):
                continue
            day = trading_days[ti]
            day_res = engine.prices.resolved_close_date(ticker, day)
            # Same walk-back collision guard as the endpoint computations:
            # a day whose walk-back resolves to sim_d (or earlier) tells us
            # nothing forward, so skip it.
            if day_res is None or day_res <= sim_res:
                continue
            day_close = engine.prices.price_on(ticker, day)
            if day_close is None:
                continue
            pct = (day_close - sim_close) / sim_close * 100.0
            if min_pct is None or pct < min_pct:
                min_pct = pct
            if max_pct is None or pct > max_pct:
                max_pct = pct
        if min_pct is None or max_pct is None:
            return None, None
        return round(min_pct, 4), round(max_pct, 4)

    for run in top_runs:
        try:
            # Hold the store lock — the background _opus_annotate thread may
            # share this sqlite3 connection across threads.
            with engine.store._lock:
                # Training-integrity invariant (mirrors _append_top_decisions):
                # the DecisionScorer must learn the 5d outcome of trades that
                # ACTUALLY EXECUTED, never of a BUY/SELL that `_execute_decision`
                # blocked. A blocked decision is recorded only as `run_one`'s
                # terminal non-FILLED marker when nothing filled that day;
                # without `status='FILLED'` its forward return would be trained
                # on as if the position had been taken (a phantom outcome whose
                # blocking reason — e.g. out of cash — is itself regime-
                # correlated, so it is biased contamination, not noise).
                rows = engine.store.conn.execute(
                    "SELECT action, ticker, sim_date, reasoning "
                    "FROM backtest_decisions "
                    "WHERE run_id=? AND action IN ('BUY','SELL') "
                    "AND ticker IS NOT NULL AND ticker != '' "
                    "AND status = 'FILLED'",
                    (run.run_id,),
                ).fetchall()
        except Exception as exc:
            print(f"[outcomes] run {run.run_id} read failed: {exc}")
            continue

        for r in rows:
            ticker = r["ticker"] or ""
            sim_date_str = r["sim_date"] or ""
            if not ticker or not sim_date_str:
                continue
            try:
                sim_d = date.fromisoformat(sim_date_str)
            except ValueError:
                continue

            # 5-trading-day forward window. Skip decisions whose window extends
            # past the cached price history — otherwise price_on() falls back to
            # the latest close, which equals sim_d's close and injects fake 0%
            # outcomes into training.
            idx = _td_index(sim_d)
            if idx < 0:
                continue
            target_idx = idx + 5
            if target_idx >= len(trading_days):
                continue
            end_d = trading_days[target_idx]

            # Both price lookups must hit real cached data for this ticker —
            # AND must resolve to DIFFERENT actual close dates. A walk-back
            # collision (both endpoints fall back to the same prior close on a
            # thin/foreign-calendar ticker) silently produces a fabricated 0%
            # outcome that poisons the DecisionScorer training set; see
            # PriceCache.resolved_close_date for the full honesty rationale.
            sim_res = engine.prices.resolved_close_date(ticker, sim_d)
            end_res = engine.prices.resolved_close_date(ticker, end_d)
            # Defense-in-depth beyond the collision check. A 7-day walk-back
            # on `end_d` can theoretically land BEFORE `sim_d` for a ticker
            # with 7+ consecutive missing closes around end_d (e.g. a thin
            # ADR taking a long holiday week off). The resulting
            # `returns_pct(sim_d, end_d)` would then compute a "forward
            # return" between two endpoints in REVERSE time order — a sign-
            # inverted, time-mangled outcome that silently contaminates the
            # DecisionScorer training set the same way the documented
            # collision-fabricated 0% outcomes did. `<=` already implies
            # `==` (the collision), so this strictly STRENGTHENS the prior
            # collision guard rather than replacing it: any case the
            # previous check rejected is still rejected.
            if (sim_res is None or end_res is None
                    or end_res <= sim_res):
                continue
            fwd_ret = engine.prices.returns_pct(ticker, sim_d, end_d)

            cache_key = (sim_date_str, ticker)
            if cache_key not in _quant_cache:
                sigs = _get_quant_signals(sim_d, [ticker], engine.prices)
                _quant_cache[cache_key] = sigs.get(ticker, {})
            q = _quant_cache[cache_key]

            regime = _regime_cache.get(sim_date_str)
            if regime is None:
                regime = _market_regime(sim_d, engine.prices)
                _regime_cache[sim_date_str] = regime
            # Match _ml_decide: "unknown" is treated as neutral 1.0, not bear.
            if regime == "bull":
                regime_mult = 1.0
            elif regime == "sideways":
                regime_mult = 0.6
            elif regime == "bear":
                regime_mult = 0.3
            else:
                regime_mult = 1.0

            reasoning = r["reasoning"] or ""
            ml_score = 0.0
            # Word-boundary anchor so `score=` cannot match inside a longer
            # identifier (e.g. a future `kw_score=` / `text_score=` / accidental
            # `underscore=…` token) — mirrors the sibling `\bscorer=` /
            # `\bconviction=` discipline already used by `_parse_gate_decision`
            # and `_parse_conviction_pct`. Without `\b`, a `re.search` on
            # `"underscore=999 score=1.5"` returns `999` (first substring
            # `score=` lives INSIDE `underscore=`), poisoning the
            # DecisionScorer's `ml_score` training feature.
            m = re.search(r"\bscore=([0-9.+-]+)", reasoning)
            if m:
                try:
                    ml_score = float(m.group(1))
                except ValueError:
                    pass

            news_urgency: float | None = None
            news_article_count: float | None = None
            # Same word-boundary discipline as `\bscore=` above. The current
            # `_ml_decide` reasoning emits exactly one `news_urg=` token, but
            # a `prev_news_urg=` or `max_news_urg=` future addition would
            # silently match inside the longer identifier without `\b`.
            m_urg = re.search(r"\bnews_urg=([0-9.+-]+)", reasoning)
            if m_urg:
                try:
                    news_urgency = float(m_urg.group(1))
                except ValueError:
                    pass
            m_cnt = re.search(r"\bnews_count=(\d+)", reasoning)
            if m_cnt:
                try:
                    news_article_count = float(m_cnt.group(1))
                except ValueError:
                    pass
            # Match the inference-side convention: when there is no supporting
            # news, fall back to the build_features neutral defaults (urg=50,
            # cnt=1) by passing None. Otherwise training would see (0, 0) for
            # no-news while predict sees (50, 1) — model gets two encodings
            # of the same condition.
            if news_article_count is not None and news_article_count <= 0:
                news_urgency = None
                news_article_count = None

            # Additive gate-decision capture (read-only research signal,
            # 2026-05-18 feature — the `forward_return_10d/20d` precedent).
            # `train_scorer` reads ONLY `forward_return_5d`, so these keys are
            # inert to training and to the gate/trade path. Records the gate's
            # TRUE then-deployed prediction + off-distribution abstention so
            # future analysis can measure its realized effect instead of
            # re-predicting with today's pickle (the documented `gate_pnl`
            # reconstruction residual). None on SELL / untrained-cycle rows.
            gate_scorer_pred, gate_off_dist = _parse_gate_decision(reasoning)

            # Additive sizing capture (read-only research signal, 2026-05-21
            # feature — same `forward_return_10d/20d` precedent: scorer is
            # unchanged because `train_scorer`/`build_features` ignore unknown
            # dict keys, so no retrain required and no risk of feature-vector
            # drift). The `conviction=X%` token is already emitted by every
            # BUY reasoning; structured-field capture here unlocks
            # sizing-weighted realized analysis without a schema migration
            # — see `_parse_conviction_pct` for the documented payoff.
            # SELL/HOLD rows read None (the gate is BUY-only); the SQL
            # already filters to BUY/SELL FILLED, so a None on a SELL row
            # is the parsed-truth, not a fault.
            conviction_pct = _parse_conviction_pct(reasoning)

            # Additive persona + regime-label capture (read-only research
            # signal, 2026-05-19 feature — same `forward_return_10d/20d`
            # precedent: scorer is unchanged because `train_scorer` /
            # `build_features` ignore unknown dict keys, so no retrain
            # required and no risk of feature-vector drift). Two reasons:
            #
            # 1. `persona_skill` / `persona_leaderboard` already derive the
            #    persona via `persona_for(run_id)`. Capturing the persona
            #    NAME directly in the outcome row lets ad-hoc queries
            #    (`grep persona=Momentum decision_outcomes.jsonl | …`) and
            #    future per-persona diagnostics filter without re-importing
            #    the live `PERSONAS` dict — the dict can later add/rename
            #    personas, and old outcome rows still self-describe.
            # 2. `regime_mult` (0.3/0.6/1.0) is a STRINGLY-TYPED encoding of
            #    the `bull/sideways/bear/unknown` label that `_ml_decide`
            #    and `_compute_decision_outcomes` BOTH compute from the
            #    SPY 50/200 MA via `_market_regime`. The multiplier alone
            #    can't distinguish `bull` from `unknown` (both = 1.0), so a
            #    downstream regime-conditional cut on `regime_mult==1.0`
            #    silently lumps an SPY-pre-200d-history "unknown" cycle in
            #    with a real bull cycle. Capturing the raw label resolves
            #    that ambiguity. Existing `regime_audit` decodes the same
            #    label from `regime_mult`; this is a strict superset so
            #    that diagnostic keeps working unchanged.
            try:
                from paper_trader.backtest import persona_for as _persona_for
                _persona_name = _persona_for(run.run_id).get("name")
            except Exception:
                _persona_name = None

            # Additive intraperiod-extreme capture (read-only research signal,
            # 2026-05-23 feature — same `forward_return_10d/20d` precedent:
            # `build_features` / `train_scorer` ignore unknown dict keys, so
            # no retrain required and no risk of feature-vector drift).
            # Unlocks risk-adjusted analysis the 5d endpoint cannot answer:
            #   * Did a +5% endpoint reading mask a -15% intraperiod drawdown
            #     (the trade would have been stopped out before the endpoint)?
            #   * Did a -3% endpoint reading mask a +12% intraperiod peak (the
            #     `take_profit` field could have captured the gain before the
            #     reversal)?
            # Computed once per outcome row alongside the existing 5d field.
            #
            # Multi-horizon capture (2026-05-26 feature). The 5d field above
            # already feeds `stop_out_audit` / `mfe_conversion`, but those
            # answer only one question: "does the inherited 5d-window stop /
            # take-profit pay?". A quant researcher considering longer-window
            # stop bands (e.g. would a -10% stop with a 10d window capture
            # MORE protection than the current -8% / 5d?) had to recompute
            # extremes manually from price data because the outcome corpus
            # only carried the 5d snapshot. Adding 10d / 20d extremes
            # alongside the existing `forward_return_10d/20d` endpoint pair
            # unlocks horizon-conditional stop / take-profit analysis with
            # ZERO change to training (`build_features` / `train_scorer`
            # ignore unknown dict keys — the additive-keys precedent).
            # Each helper call loops at most `h` trading days, so cost
            # is bounded; None semantics are identical to the 5d field.
            intra_min, intra_max = _fwd_intraperiod_extremes(
                ticker, sim_d, idx, h=5)
            intra_min_10, intra_max_10 = _fwd_intraperiod_extremes(
                ticker, sim_d, idx, h=10)
            intra_min_20, intra_max_20 = _fwd_intraperiod_extremes(
                ticker, sim_d, idx, h=20)
            outcomes.append({
                "run_id": run.run_id,
                "persona": _persona_name,
                "regime_label": regime,
                "sim_date": sim_date_str,
                "ticker": ticker,
                "action": r["action"],
                "ml_score": ml_score,
                # Use only numeric quant fields; the legacy uppercase "MACD"
                # is a string label and would corrupt scorer features if it
                # leaked through via `or`-fallback when macd_signal==0.0.
                "rsi": q.get("rsi"),
                "macd": q.get("macd_signal"),
                "mom5": q.get("mom_5d"),
                "mom20": q.get("mom_20d"),
                "regime_mult": regime_mult,
                "vol_ratio": q.get("vol_ratio"),
                "bb_position": q.get("bb_position"),
                "news_urgency": news_urgency,
                "news_article_count": news_article_count,
                "forward_return_5d": round(fwd_ret, 4),
                # Additive — see _fwd_ret_h. Best-effort None when the horizon
                # window exceeds cached history; the 5d field above is the
                # only one the scorer trains on and is unchanged.
                "forward_return_10d": _fwd_ret_h(ticker, sim_d, idx, 10),
                "forward_return_20d": _fwd_ret_h(ticker, sim_d, idx, 20),
                # Additive — the gate's actual then-deployed decision (see the
                # comment above + `_parse_gate_decision`). None when the
                # cycle's scorer was untrained / sub-gate (no `scorer=` in
                # reasoning) or on SELL (the gate modulates BUY only).
                "gate_scorer_pred": gate_scorer_pred,
                "gate_off_dist": gate_off_dist,
                # Additive — the gate's actual then-applied sizing
                # (`_parse_conviction_pct`). Fraction in [0,1] (e.g. 0.25
                # for `conviction=25%`); None on SELL / HOLD rows because
                # the conviction emission is BUY-only. Unlocks sizing-
                # weighted realized analysis (does higher conviction
                # actually predict higher realized return? — the
                # calibration question existing diagnostics structurally
                # cannot answer because they only see rank skill, not
                # economic weight).
                "conviction_pct": conviction_pct,
                # Additive 52-week position signal (0=at low, 1=at high) and
                # %-from-52-week-high. Already computed by
                # `_compute_technical_indicators` and consumed by
                # `_ml_decide`'s bubble-top gate (`wk52_pos > 0.80` suppresses
                # the BUY) — yet was never persisted to outcomes. Capturing it
                # here lets downstream research tools (`baseline_compare`,
                # `regime_audit`, ad-hoc Jupyter analysis) test whether the
                # documented 52-week-high gate explanation actually
                # corresponds to a realized forward-return shift. The scorer
                # is unchanged: `train_scorer`/`build_features` ignore extra
                # dict keys, so this is purely additive — same operational
                # discipline as the `forward_return_10d/20d` precedent. None
                # when `_compute_technical_indicators` had insufficient
                # history (<60 closes) for this ticker at sim_date.
                "wk52_pos": q.get("wk52_pos"),
                "pct_from_52h": q.get("pct_from_52h"),
                # Intraperiod extremes — worst drawdown / best peak reached at
                # any close between sim_d+1 and sim_d+5 trading days, relative
                # to sim_d close (signed %). None when the window has no
                # resolvable closes (thin/foreign ticker calendar, run end);
                # partial coverage is honored (any resolving day contributes).
                # See `_fwd_intraperiod_extremes` for the full rationale.
                "forward_intraperiod_min_5d": intra_min,
                "forward_intraperiod_max_5d": intra_max,
                # Multi-horizon intraperiod extremes (10d / 20d). Same shape
                # and None semantics as the 5d pair above: None when the
                # horizon window has no resolvable closes, partial coverage
                # honored. Pairs with `forward_return_10d` /
                # `forward_return_20d` so a horizon-conditional analyzer
                # (longer-window stop / take-profit sweep, captured-upside
                # ratio over multiple holding periods) has the full pair.
                "forward_intraperiod_min_10d": intra_min_10,
                "forward_intraperiod_max_10d": intra_max_10,
                "forward_intraperiod_min_20d": intra_min_20,
                "forward_intraperiod_max_20d": intra_max_20,
                # Enhanced MACD / EMA200 features (the 3 added to build_features
                # alongside the legacy 10 numeric + 7 sector). They are computed
                # by `_compute_technical_indicators` and surfaced through
                # `_get_quant_signals`, but were never captured in the outcome
                # rows — so every training record carried None → 0.0 default and
                # every inference call defaulted them the same way. Verified by
                # introspecting the deployed scorer's first-layer weights: mean
                # |w| for these 3 input neurons was EXACTLY 0.000000 vs ~0.3-0.5
                # for every live feature (the MLP correctly learned constant-
                # zero inputs have no information). Persisting them here closes
                # the training side of that gap so the next retrain has real
                # variance to fit on. The inference-side fix in `_ml_decide`
                # closes the prediction side. None for tickers whose technical-
                # indicator window had insufficient history (the same convention
                # as the sibling `wk52_pos` / `vol_ratio` keys above).
                "ema200_above": q.get("ema200_above"),
                "hist_cross_up": q.get("hist_cross_up"),
                "macd_below_zero_cross": q.get("macd_below_zero_cross"),
                "return_pct": run.total_return_pct,
            })

    return outcomes


def _oos_multi_horizon_metrics(scorer, oos_records: list[dict],
                               horizons: tuple[int, ...] = (10, 20)) -> dict:
    """Per-longer-horizon out-of-sample directional skill of the scorer.

    Mirrors ``_oos_rank_metrics`` (same predict signature, same SELL
    sign-flip, same tie-aware Spearman via ``calibration._spearman``)
    EXCEPT the realized target is ``forward_return_{h}d`` for h in
    ``horizons``, not the 5d anchor. The scorer was trained on the 5d
    label, so a non-trivial signal at 10d or 20d is informative even
    when the 5d OOS rank-IC is at noise — AGENTS.md documents leveraged
    ETFs have noisy 5d windows but stronger multi-month returns.

    Decisions whose row carries no ``forward_return_{h}d`` (the horizon
    window ran past cached price history at outcome-compute time, so
    ``_fwd_ret_h`` returned None) are DROPPED for THAT horizon only —
    each horizon reports its own ``n`` honestly. A scorer-predict crash
    on a single row drops just that row, never the whole horizon (the
    ``_oos_rank_metrics`` partial-raiser discipline).

    Returns ``{h: {dir_acc, rank_ic, n}}`` for each requested horizon,
    with the same ``{None, None, 0}`` honest-empty sentinels
    ``_oos_rank_metrics`` uses. Never raises — outer try/except returns
    empty per-horizon sentinels on any fault so the skill ledger is
    NEVER blocked by a diagnostic crash (the AGENTS.md "scorer-train
    status must stay truthful" discipline, mirrored across siblings).
    """
    empty = {h: {"dir_acc": None, "rank_ic": None, "n": 0} for h in horizons}
    try:
        if not oos_records or not getattr(scorer, "is_trained", False):
            return empty
        import numpy as _np
        from paper_trader.ml.decision_scorer import _to_float, PRED_CLAMP_PCT
        from paper_trader.ml.calibration import _spearman

        # Predict once per record (the scorer call is the same regardless of
        # horizon — the model was trained on 5d, predictions don't shift
        # with the realized-window choice). Bucket realizations per horizon.
        per_horizon_preds: dict[int, list[float]] = {h: [] for h in horizons}
        per_horizon_acts: dict[int, list[float]] = {h: [] for h in horizons}
        # Prefer predict_with_meta so we can drop rows whose prediction COULD
        # NOT BE PRODUCED (exception path / non-finite output). The scalar
        # ``predict()`` returns 0.0 silently in both cases, which would
        # otherwise tie every failed-prediction row at zero — fake rank
        # concordance that biases the metric (rare in production but the
        # `_predict_err_logged` discipline already silences after the first
        # log, so the bug is observable as a quiet drift, not a crash).
        # Test fakes / Dummy stubs without predict_with_meta fall back to the
        # legacy predict() path so existing tests continue to work unchanged.
        _pwm_h = getattr(scorer, "predict_with_meta", None)
        _use_meta_h = callable(_pwm_h)
        for r in oos_records:
            try:
                # OOS-inference feature parity with `_ml_decide` (pass #35).
                # The 3 enhanced MACD features (ema200_above / hist_cross_up /
                # macd_below_zero_cross) are now plumbed in BOTH training
                # capture (`_compute_decision_outcomes`) AND live inference,
                # but until they were forwarded here too the OOS-rank metric
                # was computed on a degraded vector — the model's first-layer
                # weights for these slots are non-zero (live mean|w|≈0.26
                # /0.24 /0.45) so defaulting them to None → 0.0 systematically
                # biased the OOS prediction away from what the live gate
                # actually sees. Pass them through so OOS rank-IC describes
                # the same prediction the deployed gate uses on the same row.
                if _use_meta_h:
                    _meta = _pwm_h(
                        ml_score=_to_float(r.get("ml_score"), 0.0),
                        rsi=r.get("rsi"), macd=r.get("macd"),
                        mom5=r.get("mom5"), mom20=r.get("mom20"),
                        regime_mult=_to_float(r.get("regime_mult"), 1.0),
                        ticker=str(r.get("ticker") or ""),
                        vol_ratio=r.get("vol_ratio"), bb_pos=r.get("bb_position"),
                        news_urgency=r.get("news_urgency"),
                        news_article_count=r.get("news_article_count"),
                        ema200_above=r.get("ema200_above"),
                        hist_cross_up=r.get("hist_cross_up"),
                        macd_below_zero_cross=r.get("macd_below_zero_cross"),
                    )
                    # `failed=True` ⇒ the 0.0 in `pred` is a sentinel, not a
                    # real prediction; drop the row from EVERY horizon so it
                    # cannot contaminate any rank-IC computation.
                    if _meta.get("failed"):
                        continue
                    p = float(_meta.get("pred", 0.0))
                else:
                    p = scorer.predict(
                        ml_score=_to_float(r.get("ml_score"), 0.0),
                        rsi=r.get("rsi"), macd=r.get("macd"),
                        mom5=r.get("mom5"), mom20=r.get("mom20"),
                        regime_mult=_to_float(r.get("regime_mult"), 1.0),
                        ticker=str(r.get("ticker") or ""),
                        vol_ratio=r.get("vol_ratio"), bb_pos=r.get("bb_position"),
                        news_urgency=r.get("news_urgency"),
                        news_article_count=r.get("news_article_count"),
                        ema200_above=r.get("ema200_above"),
                        hist_cross_up=r.get("hist_cross_up"),
                        macd_below_zero_cross=r.get("macd_below_zero_cross"),
                    )
                    p = float(p)
                if p != p:  # NaN — defensive; predict_with_meta caller would
                    continue  # neutralize but pure predict() is unguarded
                is_sell = str(r.get("action") or "BUY").upper() == "SELL"
                for h in horizons:
                    a = _to_float(r.get(f"forward_return_{h}d"),
                                  float("nan"))
                    if is_sell:
                        a = -a
                    a = float(a)
                    if a == a:  # drop NaN per-horizon
                        # Same train_scorer-aligned clamp as the 5d path
                        # above. Rank metrics are nearly insensitive (only
                        # extreme rows tie at ±50) but the clamp keeps the
                        # 10d/20d slices coherent with the 5d view: a single
                        # extreme-week row no longer extends rank space on
                        # one horizon and not another.
                        a = max(-PRED_CLAMP_PCT, min(PRED_CLAMP_PCT, a))
                        per_horizon_preds[h].append(p)
                        per_horizon_acts[h].append(a)
            except Exception:
                continue

        out: dict[int, dict] = {}
        for h in horizons:
            preds = per_horizon_preds[h]
            actuals = per_horizon_acts[h]
            n = len(preds)
            cell = {"dir_acc": None, "rank_ic": None, "n": n}
            if n >= 2:
                ic = _spearman(_np.asarray(preds, dtype=float),
                               _np.asarray(actuals, dtype=float))
                if ic == ic:
                    cell["rank_ic"] = round(float(ic), 4)
            dir_pairs = [(p, a) for p, a in zip(preds, actuals)
                         if p != 0.0 and a != 0.0]
            if dir_pairs:
                hits = sum(1 for p, a in dir_pairs if (p > 0) == (a > 0))
                cell["dir_acc"] = round(hits / len(dir_pairs), 4)
            out[h] = cell
        return out
    except Exception:
        return empty


def _oos_rank_metrics(scorer, oos_records: list[dict]) -> dict:
    """Out-of-sample *directional* skill of the scorer on the temporal holdout.

    ``oos_rmse`` answers "how big is the error", but the ``_ml_decide``
    conviction gate only ever acts on the prediction's *sign / bucket*
    (±10 / ±5 / 0 — CLAUDE.md §6). A scorer whose OOS RMSE exceeds σ(target)
    (the documented current state) can still be gate-useful **iff it gets the
    direction right**. RMSE alone cannot tell a skeptical quant whether the
    gate carries any real edge; these two metrics measure exactly that:

    - ``dir_acc`` — fraction of held-out decisions where ``sign(pred) ==
      sign(realized)`` (a zero on either side carries no directional truth
      and is excluded).
    - ``rank_ic`` — tie-aware Spearman(pred, realized), **reusing
      ``calibration._spearman``** so this OOS metric and the in-sample
      ``ml.calibration`` diagnostic can never drift (single source of truth,
      AGENTS.md invariant #10 spirit). Tie-awareness is load-bearing: the
      scorer clamps to ±``PRED_CLAMP_PCT``, so off-distribution predictions
      tie at exactly ±50 and a naïve ``argsort(argsort)`` would fabricate
      rank skill there (a constant predictor would read 1.0).

    Mirrors ``validation.evaluate_scorer_oos``'s exact 11-kwarg ``predict``
    signature and SELL sign-flip so it describes the **same** prediction path
    the gate uses. Never raises — returns ``{dir_acc, rank_ic, n}`` with
    ``None`` metrics on any fault so a post-train diagnostic crash can't mask
    a successful train (the AGENTS.md "scorer-train status must stay
    truthful" discipline, mirrored from the separate ``oos_rmse`` guard).
    """
    # Additive per-action breakdown (2026-05-20 feature). The conviction
    # gate (#5) is BUY-only, so a researcher needs to know the GATE-RELEVANT
    # skill (BUY rank-IC) separately from the aggregate. SELL rank-IC is
    # informative as a sanity check — the model is trained on flipped SELL
    # labels, so a positive SELL rank-IC means the scorer would also help
    # short selection if the gate were ever extended to SELL. Returned as
    # additional keys; the legacy aggregate (dir_acc, rank_ic, n) is
    # unchanged so every existing caller (_train_decision_scorer status
    # string, tests) keeps working byte-identically.
    out: dict = {
        "dir_acc": None, "rank_ic": None, "n": 0,
        "buy_dir_acc": None, "buy_rank_ic": None, "buy_n": 0,
        "sell_dir_acc": None, "sell_rank_ic": None, "sell_n": 0,
        # Per-regime breakdown — bucketed by `regime_mult` (the field every
        # outcome row carries; `regime_label` was added later and is absent
        # from the historical corpus). Decode mirrors `regime_audit`:
        # 0.3→bear, 0.6→sideways, 1.0→bull(-or-unknown). An aggregate rank-IC
        # ~0 can hide real skill in one regime cancelled by an inversion in
        # another; these keys make that visible and — riding the per-cycle
        # scorer_skill_log — trendable, which the standalone `regime_audit`
        # snapshot is not.
        "regime_bull_n": 0, "regime_bull_rank_ic": None,
        "regime_sideways_n": 0, "regime_sideways_rank_ic": None,
        "regime_bear_n": 0, "regime_bear_rank_ic": None,
    }
    try:
        if not oos_records or not getattr(scorer, "is_trained", False):
            return out
        import numpy as _np
        from paper_trader.ml.decision_scorer import _to_float, PRED_CLAMP_PCT
        from paper_trader.ml.calibration import _spearman

        preds: list[float] = []
        actuals: list[float] = []
        # Per-action buckets — populated alongside the aggregate so we never
        # do a second predict pass (the scorer call is the expensive step;
        # the bucket split is free).
        buy_preds: list[float] = []
        buy_acts: list[float] = []
        sell_preds: list[float] = []
        sell_acts: list[float] = []
        # Per-regime buckets — keyed by the explicit ``regime_label`` when
        # present (the 2026-05-19 feature), with the legacy ``regime_mult``
        # decode as fallback for rows that pre-date the label field. The two
        # encodings overlap (mult=1.0 ⇔ "bull") EXCEPT for "unknown":
        # ``_market_regime`` emits "unknown" with mult=1.0 for the early days
        # of any backtest window (SPY has <200 closes), so the mult-only
        # decode silently bucketed those into "bull". Live audit of
        # ``data/decision_outcomes.jsonl`` shows 182 ``regime_label='unknown'``
        # rows out of 576 in the would-be-bull bucket (~32%) — a real silent
        # contamination of the regime-conditional skill metric. By preferring
        # ``regime_label`` we drop those into the no-bucket fall-through
        # honestly, while legacy rows (no label field, ~7400 of the corpus)
        # keep their mult-based bucket assignment unchanged.
        _REGIME_BY_MULT = {0.3: "bear", 0.6: "sideways", 1.0: "bull"}
        regime_preds: dict[str, list[float]] = {
            "bull": [], "sideways": [], "bear": []}
        regime_acts: dict[str, list[float]] = {
            "bull": [], "sideways": [], "bear": []}
        # Prefer predict_with_meta to drop rows whose prediction COULD NOT BE
        # PRODUCED (exception path / non-finite output) — see the matching
        # comment in `_oos_multi_horizon_metrics` above for the full rationale.
        # The scalar ``predict()`` returns 0.0 silently on both failure paths,
        # which would otherwise contribute fake rank ties at zero (the
        # `_predict_err_logged` discipline silences the warning after the first
        # log, so the bug is observable as quiet metric drift, not a crash).
        # Test fakes / Dummy stubs without predict_with_meta fall back to the
        # legacy predict() path so existing tests continue to work unchanged.
        _pwm = getattr(scorer, "predict_with_meta", None)
        _use_meta = callable(_pwm)
        for r in oos_records:
            try:
                # OOS-inference feature parity with `_ml_decide` (pass #35).
                # The 3 enhanced MACD features are now captured by
                # `_compute_decision_outcomes` and forwarded by the live
                # gate, but were previously stripped here — biasing OOS
                # rank-IC away from the gate's actual prediction. Same
                # rationale as `_oos_multi_horizon_metrics` above.
                if _use_meta:
                    _meta = _pwm(
                        ml_score=_to_float(r.get("ml_score"), 0.0),
                        rsi=r.get("rsi"), macd=r.get("macd"),
                        mom5=r.get("mom5"), mom20=r.get("mom20"),
                        regime_mult=_to_float(r.get("regime_mult"), 1.0),
                        ticker=str(r.get("ticker") or ""),
                        vol_ratio=r.get("vol_ratio"), bb_pos=r.get("bb_position"),
                        news_urgency=r.get("news_urgency"),
                        news_article_count=r.get("news_article_count"),
                        ema200_above=r.get("ema200_above"),
                        hist_cross_up=r.get("hist_cross_up"),
                        macd_below_zero_cross=r.get("macd_below_zero_cross"),
                    )
                    # `failed=True` ⇒ the 0.0 in `pred` is a sentinel, not a
                    # real prediction; drop the row from EVERY bucket (aggregate,
                    # buy/sell, regime) so it cannot contaminate any rank-IC.
                    if _meta.get("failed"):
                        continue
                    p = float(_meta.get("pred", 0.0))
                else:
                    p = scorer.predict(
                        ml_score=_to_float(r.get("ml_score"), 0.0),
                        rsi=r.get("rsi"), macd=r.get("macd"),
                        mom5=r.get("mom5"), mom20=r.get("mom20"),
                        regime_mult=_to_float(r.get("regime_mult"), 1.0),
                        ticker=str(r.get("ticker") or ""),
                        vol_ratio=r.get("vol_ratio"), bb_pos=r.get("bb_position"),
                        news_urgency=r.get("news_urgency"),
                        news_article_count=r.get("news_article_count"),
                        ema200_above=r.get("ema200_above"),
                        hist_cross_up=r.get("hist_cross_up"),
                        macd_below_zero_cross=r.get("macd_below_zero_cross"),
                    )
                # NaN sentinel default so a missing/non-finite forward return
                # is *dropped* by the `a == a` guard below, not silently
                # coerced to 0.0 (which would poison rank_ic with fabricated
                # flat-target ties). Mirrors persona_skill._aligned's NaN-
                # sentinel discipline so this OOS metric and that diagnostic
                # treat unparseable targets the same way.
                a_raw = _to_float(r.get("forward_return_5d"), float("nan"))
                # Mirror train_scorer / evaluate_scorer_oos: a SELL's realized
                # target sign is flipped so "good" has one consistent meaning.
                is_sell = str(r.get("action") or "BUY").upper() == "SELL"
                a = -a_raw if is_sell else a_raw
                p = float(p)
                if p == p and a == a:  # drop non-finite defensively
                    # Mirror train_scorer's symmetric label clamp + the
                    # evaluate_scorer_oos RMSE clamp so this OOS metric
                    # describes the same target space the model was trained
                    # against (apples-to-apples with val). For rank-IC this is
                    # near-no-op since Spearman operates on ranks, but the
                    # ~0.4% of rows with |fr|>50 land in ties at ±50 instead of
                    # silently extending rank space the model can never reach.
                    a = max(-PRED_CLAMP_PCT, min(PRED_CLAMP_PCT, a))
                    preds.append(p)
                    actuals.append(a)
                    if is_sell:
                        sell_preds.append(p)
                        sell_acts.append(a)
                    else:
                        buy_preds.append(p)
                        buy_acts.append(a)
                    # Regime bucket — prefer the explicit ``regime_label``
                    # (2026-05-19 outcome field). Fall back to the
                    # ``regime_mult`` decode ONLY when the label is absent
                    # (pre-feature legacy rows). An explicit ``"unknown"``
                    # label is intentionally dropped from every bucket —
                    # that is the documented unknown-regime fall-through
                    # the mult-only path silently mis-bucketed as "bull".
                    _label = r.get("regime_label")
                    if _label in regime_preds:
                        _reg = _label
                    elif _label is None:
                        _reg = _REGIME_BY_MULT.get(
                            _to_float(r.get("regime_mult"), -1.0))
                    else:
                        _reg = None  # "unknown" or any other label
                    if _reg is not None:
                        regime_preds[_reg].append(p)
                        regime_acts[_reg].append(a)
            except Exception:
                continue

        def _rank_dir(ps: list[float], acs: list[float]) -> tuple:
            """Return (rank_ic, dir_acc) for one bucket. rank_ic requires
            n>=2 (Spearman undefined on n<2); dir_acc only requires at
            least one non-zero pair (a single concordant pair is honest
            information — locked by the legacy n=1 test).
            """
            ic_v: float | None = None
            if len(ps) >= 2:
                ic = _spearman(_np.asarray(ps, dtype=float),
                               _np.asarray(acs, dtype=float))
                if ic == ic:  # not NaN
                    ic_v = round(float(ic), 4)
            dir_pairs = [(p, a) for p, a in zip(ps, acs)
                         if p != 0.0 and a != 0.0]
            dacc_v: float | None = None
            if dir_pairs:
                hits = sum(1 for p, a in dir_pairs if (p > 0) == (a > 0))
                dacc_v = round(hits / len(dir_pairs), 4)
            return ic_v, dacc_v

        out["n"] = len(preds)
        agg_ic, agg_da = _rank_dir(preds, actuals)
        out["rank_ic"] = agg_ic
        out["dir_acc"] = agg_da

        out["buy_n"] = len(buy_preds)
        buy_ic, buy_da = _rank_dir(buy_preds, buy_acts)
        out["buy_rank_ic"] = buy_ic
        out["buy_dir_acc"] = buy_da

        out["sell_n"] = len(sell_preds)
        sell_ic, sell_da = _rank_dir(sell_preds, sell_acts)
        out["sell_rank_ic"] = sell_ic
        out["sell_dir_acc"] = sell_da

        # Per-regime rank-IC — same _rank_dir path as buy/sell, no second
        # predict pass (the bucket split above was free).
        for _reg in ("bull", "sideways", "bear"):
            out[f"regime_{_reg}_n"] = len(regime_preds[_reg])
            _ric, _ = _rank_dir(regime_preds[_reg], regime_acts[_reg])
            out[f"regime_{_reg}_rank_ic"] = _ric
    except Exception:
        return {
            "dir_acc": None, "rank_ic": None, "n": 0,
            "buy_dir_acc": None, "buy_rank_ic": None, "buy_n": 0,
            "sell_dir_acc": None, "sell_rank_ic": None, "sell_n": 0,
            "regime_bull_n": 0, "regime_bull_rank_ic": None,
            "regime_sideways_n": 0, "regime_sideways_rank_ic": None,
            "regime_bear_n": 0, "regime_bear_rank_ic": None,
        }
    return out


def _train_decision_scorer(outcome_records: list[dict]) -> str:
    """Train DecisionScorer on the historical 80% of outcomes; report OOS RMSE
    on the most recent 20% (true temporal holdout — never seen during training).

    `train_scorer`'s built-in val_rmse uses a *random* 80/20 split which leaks
    future information into validation when records span time. The temporal
    split here is the trustworthy generalization metric.
    """
    if not outcome_records:
        return "no outcome records"
    try:
        from paper_trader.ml.decision_scorer import train_scorer, DecisionScorer
    except Exception as exc:
        return f"scorer err: {exc}"

    # The temporal holdout is a *diagnostic* refinement (an honest OOS RMSE),
    # NOT part of the essential operation. Previously the validation import +
    # split_outcomes_temporal sat in the same try as train_scorer, so if the
    # validation module was unavailable or the split raised on pathological
    # data, training was skipped entirely and the operator saw `scorer err:` —
    # silently wedging the per-cycle retrain invariant (CLAUDE.md §6) and
    # freezing the conviction gate (#5) for as long as the condition lasted.
    # Mirror the already-separated OOS-eval guard below: degrade to "train on
    # everything, no honest holdout" rather than "don't train at all".
    oos_records: list[dict] = []
    train_records = outcome_records
    try:
        from paper_trader.validation import split_outcomes_temporal
        train_records, oos_records = split_outcomes_temporal(
            outcome_records, oos_fraction=0.2
        )
    except Exception as exc:
        print(f"[continuous] temporal split unavailable ({exc}) — training on "
              f"all {len(outcome_records)} records, OOS holdout skipped")

    try:
        result = train_scorer(train_records)
    except Exception as exc:
        return f"scorer err: {exc}"
    val_rmse = result.get("val_rmse", float("nan"))
    val_s = f"{val_rmse:.2f}" if val_rmse == val_rmse else "n/a"

    # OOS evaluation runs AFTER train_scorer has already pickled the model to
    # SCORER_PATH. A crash here (transient pickle/IO race, validation-module
    # change, …) does NOT mean training failed — the scorer is trained and the
    # next cycle's singleton reset will deploy it. Guard it separately so a
    # post-train diagnostic failure degrades to oos_rmse=n/a instead of being
    # reported to the operator as `scorer err` (a false "scorer broken" signal
    # that would make an operator think the conviction gate never engages).
    oos_rmse_s = "n/a"
    # Per-action OOS RMSE — the conviction gate (#5) is BUY-only, so the
    # gate-relevant error magnitude is the BUY RMSE. An aggregate RMSE can
    # hide a BUY error much worse (or better) than the SELL error.
    # Additive in the status string — the legacy `oos_rmse` token is
    # unchanged so every existing parser keeps working.
    oos_buy_rmse_s = "n/a"
    oos_buy_rmse_n = 0
    oos_sell_rmse_s = "n/a"
    oos_sell_rmse_n = 0
    # σ(target) + RMSE/σ skill ratio — the canonical "predict the constant
    # mean" baseline a quant should compare a regressor against. The
    # documented MLP_NO_BETTER_THAN_TRIVIAL state means `oos_rmse_ratio`
    # is hovering at ≥ 1.0 in production right now; until this token landed
    # an operator had to compute it manually from oos_rmse and a separate
    # `baseline_compare` CLI invocation. Persisting it per cycle makes the
    # net-skill state visible in the trend (and the dashboard) automatically.
    # Same n/a discipline as every other oos_ token: a None metric (degenerate
    # zero target_std, untrained scorer, or post-train diagnostic crash)
    # renders as "n/a" rather than a fabricated number.
    oos_target_std_s = "n/a"
    oos_rmse_ratio_s = "n/a"
    if result.get("status") == "ok" and oos_records:
        try:
            from paper_trader.validation import evaluate_scorer_oos
            # Re-load the freshly pickled model from disk so OOS predictions
            # use the exact serialized state (catches any save/load bugs).
            scorer = DecisionScorer()
            oos = evaluate_scorer_oos(scorer, oos_records)
            r = oos.get("rmse")
            if r is not None and r == r:
                oos_rmse_s = f"{r:.2f}"
            # Per-action breakdown — additive; legacy `oos_rmse` token above
            # is the aggregate over all actions and is unchanged.
            oos_buy_rmse_n = int(oos.get("buy_n") or 0)
            oos_sell_rmse_n = int(oos.get("sell_n") or 0)
            br = oos.get("buy_rmse")
            if br is not None and br == br:
                oos_buy_rmse_s = f"{br:.2f}"
            sr = oos.get("sell_rmse")
            if sr is not None and sr == sr:
                oos_sell_rmse_s = f"{sr:.2f}"
            ts = oos.get("target_std")
            if ts is not None and ts == ts:
                oos_target_std_s = f"{ts:.2f}"
            ratio = oos.get("rmse_ratio")
            if ratio is not None and ratio == ratio:
                oos_rmse_ratio_s = f"{ratio:.3f}"
        except Exception as exc:
            oos_rmse_s = f"n/a (oos-eval err: {type(exc).__name__})"

    # OOS directional skill — guarded SEPARATELY from the rmse block (and from
    # the train block) so a crash here also degrades to n/a rather than a
    # false "scorer err" (the AGENTS.md "scorer-train status must stay
    # truthful" discipline). Reloads the freshly-pickled model from disk so
    # the metric describes the exact serialized state the next cycle deploys.
    oos_diracc_s = "n/a"
    oos_ic_s = "n/a"
    # Per-action breakdown — the conviction gate (#5) is BUY-only, so a
    # researcher needs to see BUY rank-IC SEPARATELY from the aggregate.
    # An aggregate rank-IC ~0 (the documented current state) could hide a
    # positive BUY skill cancelled by a SELL anti-skill, OR vice versa.
    # Additive in the status string — the legacy `oos_diracc/oos_ic`
    # tokens are unchanged so every existing parser keeps working.
    oos_buy_diracc_s = "n/a"
    oos_buy_ic_s = "n/a"
    oos_buy_n = 0
    oos_sell_diracc_s = "n/a"
    oos_sell_ic_s = "n/a"
    oos_sell_n = 0
    # Per-regime OOS rank-IC (bull/sideways/bear) — see `_oos_rank_metrics`.
    # The aggregate rank-IC can sit at noise while one regime carries real
    # edge and another is inverted; surfacing each makes that trendable.
    oos_bull_n = oos_sideways_n = oos_bear_n = 0
    oos_bull_ic_s = oos_sideways_ic_s = oos_bear_ic_s = "n/a"
    if result.get("status") == "ok" and oos_records:
        try:
            m = _oos_rank_metrics(DecisionScorer(), oos_records)
            if m["dir_acc"] is not None:
                oos_diracc_s = f"{m['dir_acc']:.2f}"
            if m["rank_ic"] is not None:
                oos_ic_s = f"{m['rank_ic']:+.2f}"
            oos_buy_n = int(m.get("buy_n") or 0)
            oos_sell_n = int(m.get("sell_n") or 0)
            if m.get("buy_dir_acc") is not None:
                oos_buy_diracc_s = f"{m['buy_dir_acc']:.2f}"
            if m.get("buy_rank_ic") is not None:
                oos_buy_ic_s = f"{m['buy_rank_ic']:+.2f}"
            if m.get("sell_dir_acc") is not None:
                oos_sell_diracc_s = f"{m['sell_dir_acc']:.2f}"
            if m.get("sell_rank_ic") is not None:
                oos_sell_ic_s = f"{m['sell_rank_ic']:+.2f}"
            oos_bull_n = int(m.get("regime_bull_n") or 0)
            oos_sideways_n = int(m.get("regime_sideways_n") or 0)
            oos_bear_n = int(m.get("regime_bear_n") or 0)
            if m.get("regime_bull_rank_ic") is not None:
                oos_bull_ic_s = f"{m['regime_bull_rank_ic']:+.2f}"
            if m.get("regime_sideways_rank_ic") is not None:
                oos_sideways_ic_s = f"{m['regime_sideways_rank_ic']:+.2f}"
            if m.get("regime_bear_rank_ic") is not None:
                oos_bear_ic_s = f"{m['regime_bear_rank_ic']:+.2f}"
        except Exception as exc:
            oos_diracc_s = oos_ic_s = f"n/a ({type(exc).__name__})"

    # Per-longer-horizon OOS rank skill (10d, 20d). The scorer was trained
    # on the 5d label, so a non-trivial rank-IC at 10d/20d is informative
    # even when 5d is at noise — AGENTS.md documents leveraged ETFs have
    # noisy 5d windows but stronger multi-month returns. Guarded
    # independently so a long-horizon diagnostic crash never masks the
    # successful train or the 5d metrics already reported (the same
    # discipline applied to oos_rmse and oos_diracc/oos_ic above). Each
    # horizon reports its own n honestly — a row that lacks
    # forward_return_{h}d (window ran past cached price history) drops
    # for THAT horizon only, never poisons the 5d view.
    oos_ic_10_s = "n/a"
    oos_diracc_10_s = "n/a"
    oos_ic_20_s = "n/a"
    oos_diracc_20_s = "n/a"
    oos_n_10 = 0
    oos_n_20 = 0
    if result.get("status") == "ok" and oos_records:
        try:
            mh = _oos_multi_horizon_metrics(DecisionScorer(), oos_records,
                                            horizons=(10, 20))
            c10 = mh.get(10, {})
            c20 = mh.get(20, {})
            oos_n_10 = int(c10.get("n") or 0)
            oos_n_20 = int(c20.get("n") or 0)
            if c10.get("dir_acc") is not None:
                oos_diracc_10_s = f"{c10['dir_acc']:.2f}"
            if c10.get("rank_ic") is not None:
                oos_ic_10_s = f"{c10['rank_ic']:+.2f}"
            if c20.get("dir_acc") is not None:
                oos_diracc_20_s = f"{c20['dir_acc']:.2f}"
            if c20.get("rank_ic") is not None:
                oos_ic_20_s = f"{c20['rank_ic']:+.2f}"
        except Exception as exc:
            oos_ic_10_s = oos_diracc_10_s = f"n/a ({type(exc).__name__})"
            oos_ic_20_s = oos_diracc_20_s = f"n/a ({type(exc).__name__})"

    # Label-clamp count surfaced so the per-cycle skill ledger can trend the
    # outlier rate of the training tail (a sudden spike correlates with
    # MSTR/leveraged-ETF crash/rip weeks polluting the corpus).
    label_clamped = result.get("n_label_clamped", 0)
    # Label-drop count surfaced too. `train_scorer` already returns it but
    # the status string previously only surfaced `n_label_clamped`. A silent
    # corruption (a malformed outcomes batch with non-finite/null
    # `forward_return_5d`) would shrink the effective training set without
    # ANY observable signal — the cycle status still reads `status=ok` and
    # the skill ledger's `train_n` is the POST-drop count, so the
    # operator could not tell whether a sudden drop in `train_n` was from
    # fewer outcomes computed this cycle or from rows being silently
    # discarded by the label-validation pass. Trending the count lets a
    # quant catch corruption immediately (a non-zero spike) instead of
    # only seeing the symptom (val_rmse drift, oos_ic noise).
    label_dropped = result.get("n_label_dropped", 0)
    return (f"scorer {result['status']} train_n={result['n']} "
            f"val_rmse={val_s} oos_n={len(oos_records)} oos_rmse={oos_rmse_s} "
            f"oos_target_std={oos_target_std_s} "
            f"oos_rmse_ratio={oos_rmse_ratio_s} "
            f"oos_buy_rmse_n={oos_buy_rmse_n} "
            f"oos_buy_rmse={oos_buy_rmse_s} "
            f"oos_sell_rmse_n={oos_sell_rmse_n} "
            f"oos_sell_rmse={oos_sell_rmse_s} "
            f"oos_diracc={oos_diracc_s} oos_ic={oos_ic_s} "
            f"oos_n_10={oos_n_10} oos_diracc_10={oos_diracc_10_s} "
            f"oos_ic_10={oos_ic_10_s} oos_n_20={oos_n_20} "
            f"oos_diracc_20={oos_diracc_20_s} oos_ic_20={oos_ic_20_s} "
            f"oos_buy_n={oos_buy_n} oos_buy_diracc={oos_buy_diracc_s} "
            f"oos_buy_ic={oos_buy_ic_s} "
            f"oos_sell_n={oos_sell_n} oos_sell_diracc={oos_sell_diracc_s} "
            f"oos_sell_ic={oos_sell_ic_s} "
            f"oos_bull_n={oos_bull_n} oos_bull_ic={oos_bull_ic_s} "
            f"oos_sideways_n={oos_sideways_n} oos_sideways_ic={oos_sideways_ic_s} "
            f"oos_bear_n={oos_bear_n} oos_bear_ic={oos_bear_ic_s} "
            f"n_label_clamped={label_clamped} "
            f"n_label_dropped={label_dropped}")


def _parse_scorer_status(status: str) -> dict:
    """Parse the formatted string `_train_decision_scorer` returns into a
    structured dict for the skill ledger.

    The status string is a stable, test-locked contract
    (`tests/test_continuous.py::TestTrainDecisionScorer` asserts the exact
    `scorer ok`/`train_n=`/`oos_rmse=` tokens), so parsing it is robust and
    avoids changing `_train_decision_scorer`'s return type (which `main()`
    prints verbatim and existing tests assert on as a string).

    Numeric tokens are captured up to the next whitespace, then float-parsed:
    a real metric (`12.45`, `+0.03`, `-0.12`) parses; every "n/a" form —
    including the `oos_rmse=n/a (oos-eval err: KeyError)` parenthetical the
    error path emits — has `n/a` as its first token, fails the float parse,
    and degrades to `None`. Never raises: a parse fault yields a row with
    `status="unparseable"` and `None` metrics rather than killing the loop.

    Returns ``{status, train_n, val_rmse, oos_n, oos_rmse, oos_dir_acc,
    oos_ic, oos_n_10, oos_dir_acc_10, oos_ic_10, oos_n_20, oos_dir_acc_20,
    oos_ic_20}`` — ints for the count fields, floats or None for the
    metrics. The 10d/20d fields are additive (see ``_oos_multi_horizon_metrics``)
    and default to None on any older status string that pre-dates the
    multi-horizon wiring — old skill-log rows therefore parse cleanly.
    """
    out: dict = {
        "status": "unparseable", "train_n": None, "val_rmse": None,
        "oos_n": None, "oos_rmse": None, "oos_dir_acc": None, "oos_ic": None,
        "oos_n_10": None, "oos_dir_acc_10": None, "oos_ic_10": None,
        "oos_n_20": None, "oos_dir_acc_20": None, "oos_ic_20": None,
        # Per-action breakdown (gate-relevant BUY vs informational SELL).
        # Defaults to None on older status strings that pre-date the wiring,
        # so historical skill-ledger rows parse cleanly.
        "oos_buy_n": None, "oos_buy_dir_acc": None, "oos_buy_ic": None,
        "oos_sell_n": None, "oos_sell_dir_acc": None, "oos_sell_ic": None,
        # Per-action OOS RMSE (2026-05-23 feature). The conviction gate is
        # BUY-only, so the gate-relevant magnitude error is BUY RMSE.
        # None on older status strings — historical ledger rows parse clean.
        "oos_buy_rmse_n": None, "oos_buy_rmse": None,
        "oos_sell_rmse_n": None, "oos_sell_rmse": None,
        # σ(target) baseline + RMSE/σ skill ratio (2026-05-25 feature). The
        # canonical "predict the constant mean" baseline a quant should
        # compare a regressor's RMSE against. `oos_rmse_ratio < 1.0` ⇒ the
        # model carries skill; `≥ 1.0` ⇒ MLP_NO_BETTER_THAN_TRIVIAL (the
        # documented production state). None on older status strings —
        # historical ledger rows parse cleanly.
        "oos_target_std": None, "oos_rmse_ratio": None,
        # Per-regime OOS rank-IC. None on older status strings predating
        # the regime-breakdown wiring — historical ledger rows parse clean.
        "oos_bull_n": None, "oos_bull_ic": None,
        "oos_sideways_n": None, "oos_sideways_ic": None,
        "oos_bear_n": None, "oos_bear_ic": None,
        "n_label_clamped": None,
        # Label-drop count from `train_scorer` (pass #16, 2026-05-23). None
        # on older status strings predating the wiring — historical
        # ledger rows therefore parse cleanly.
        "n_label_dropped": None,
    }
    try:
        s = str(status or "").strip()
        if not s:
            return out
        # `no outcome records` is the only non-`scorer …` sentinel
        # `_train_decision_scorer` returns; normalise it explicitly.
        if s.startswith("no outcome records"):
            out["status"] = "no_outcome_records"
            return out
        m = re.match(r"scorer\s+([A-Za-z_]+)", s)
        out["status"] = m.group(1) if m else "unknown"

        def _num(key: str):
            mm = re.search(rf"(?:^|\s){re.escape(key)}=(\S+)", s)
            if not mm:
                return None
            try:
                return float(mm.group(1))
            except (TypeError, ValueError):
                return None

        tn = _num("train_n")
        on = _num("oos_n")
        on10 = _num("oos_n_10")
        on20 = _num("oos_n_20")
        out["train_n"] = int(tn) if tn is not None else None
        out["oos_n"] = int(on) if on is not None else None
        out["oos_n_10"] = int(on10) if on10 is not None else None
        out["oos_n_20"] = int(on20) if on20 is not None else None
        out["val_rmse"] = _num("val_rmse")
        out["oos_rmse"] = _num("oos_rmse")
        out["oos_dir_acc"] = _num("oos_diracc")
        out["oos_ic"] = _num("oos_ic")
        out["oos_dir_acc_10"] = _num("oos_diracc_10")
        out["oos_ic_10"] = _num("oos_ic_10")
        out["oos_dir_acc_20"] = _num("oos_diracc_20")
        out["oos_ic_20"] = _num("oos_ic_20")
        # Per-action breakdown — int counts, float metrics. Old status
        # strings predating this wiring omit the tokens entirely and
        # degrade to None via the `_num` regex miss.
        obn = _num("oos_buy_n")
        osn = _num("oos_sell_n")
        out["oos_buy_n"] = int(obn) if obn is not None else None
        out["oos_sell_n"] = int(osn) if osn is not None else None
        out["oos_buy_dir_acc"] = _num("oos_buy_diracc")
        out["oos_buy_ic"] = _num("oos_buy_ic")
        out["oos_sell_dir_acc"] = _num("oos_sell_diracc")
        out["oos_sell_ic"] = _num("oos_sell_ic")
        # Per-action OOS RMSE — int counts, float rmse. The status string
        # uses `oos_buy_rmse_n` / `oos_buy_rmse` (and SELL equivalents) so
        # both keys parse cleanly even when one is "n/a".
        obrn = _num("oos_buy_rmse_n")
        osrn = _num("oos_sell_rmse_n")
        out["oos_buy_rmse_n"] = int(obrn) if obrn is not None else None
        out["oos_sell_rmse_n"] = int(osrn) if osrn is not None else None
        out["oos_buy_rmse"] = _num("oos_buy_rmse")
        out["oos_sell_rmse"] = _num("oos_sell_rmse")
        # σ(target) + RMSE/σ skill ratio — additive 2026-05-25 tokens.
        # Older status strings (cycles before this wiring) omit them and
        # the `_num` regex miss degrades cleanly to None. Floats — no
        # int cast (target_std is a real-valued std, ratio is unitless).
        out["oos_target_std"] = _num("oos_target_std")
        out["oos_rmse_ratio"] = _num("oos_rmse_ratio")
        # Per-regime — int counts, float rank-IC. Old status strings omit
        # the tokens entirely and degrade to None via the `_num` regex miss.
        for _reg in ("bull", "sideways", "bear"):
            _rn = _num(f"oos_{_reg}_n")
            out[f"oos_{_reg}_n"] = int(_rn) if _rn is not None else None
            out[f"oos_{_reg}_ic"] = _num(f"oos_{_reg}_ic")
        # Label-clamp count is an integer when reported; legacy status strings
        # (cycles before the clamp landed) omit it — degrades to None cleanly.
        nlc = _num("n_label_clamped")
        out["n_label_clamped"] = int(nlc) if nlc is not None else None
        # Label-drop count — same parse pattern. Older status strings
        # (cycles before pass #16 wired it) omit the token and degrade to
        # None via the `_num` regex miss, so historical ledger rows parse
        # cleanly under the new schema.
        nld = _num("n_label_dropped")
        out["n_label_dropped"] = int(nld) if nld is not None else None
    except Exception:
        return {
            "status": "unparseable", "train_n": None, "val_rmse": None,
            "oos_n": None, "oos_rmse": None, "oos_dir_acc": None,
            "oos_ic": None, "oos_n_10": None, "oos_dir_acc_10": None,
            "oos_ic_10": None, "oos_n_20": None, "oos_dir_acc_20": None,
            "oos_ic_20": None,
            "oos_buy_n": None, "oos_buy_dir_acc": None, "oos_buy_ic": None,
            "oos_sell_n": None, "oos_sell_dir_acc": None, "oos_sell_ic": None,
            "oos_buy_rmse_n": None, "oos_buy_rmse": None,
            "oos_sell_rmse_n": None, "oos_sell_rmse": None,
            "oos_target_std": None, "oos_rmse_ratio": None,
            "oos_bull_n": None, "oos_bull_ic": None,
            "oos_sideways_n": None, "oos_sideways_ic": None,
            "oos_bear_n": None, "oos_bear_ic": None,
            "n_label_clamped": None,
            "n_label_dropped": None,
        }
    return out


def _append_scorer_skill_log(status: str, cycle: int,
                             win_start: date, win_end: date,
                             n_train_hint: int | None = None) -> bool:
    """Append one structured row to the per-cycle scorer-skill ledger.

    Best-effort and idempotent-safe: every fault is swallowed (a ledger
    write must NEVER break the continuous loop — same discipline as
    `_post_discord` / the validation persister). Bounded growth: when the
    file exceeds 2× `SCORER_SKILL_LOG_KEEP` it is atomically rewritten to the
    last `SCORER_SKILL_LOG_KEEP` rows (the decision_outcomes trim idiom — a
    torn truncate would lose skill history, so write tmp then `.replace`).

    `n_train_hint` lets `main()` pass the deployed pickle's `n_train` (the
    gate-relevant count, ≥500 ⇒ gate active) when the status string itself
    omits it (e.g. `no outcome records`); the parsed `train_n` still wins
    when present.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        parsed = _parse_scorer_status(status)
        if parsed.get("train_n") is None and n_train_hint is not None:
            try:
                parsed["train_n"] = int(n_train_hint)
            except (TypeError, ValueError):
                pass
        # Is the gate acting on the architecture the source endorses?
        # `deploy_audit.is_deploy_stale` compares the just-(re)written
        # pickle's fitted-model hyper-params against decision_scorer.
        # MLP_CONFIG: True ⇒ the running loop predates the anti-overfit
        # retune and is gating real conviction on the memorizing net (the
        # single most-repeated documented finding); False ⇒ the deployed net
        # matches source; None ⇒ unknowable (no pickle / lstsq fallback /
        # unreadable). Best-effort — never raises (a ledger field must not
        # break the loop, same discipline as the rest of this function).
        try:
            from paper_trader.ml.deploy_audit import is_deploy_stale
            deploy_stale = is_deploy_stale()
        except Exception:
            deploy_stale = None
        # Kill-switch decision capture (2026-05-28 feature). The conviction
        # gate has TWO independent guards: (a) ``train_n >= 500`` (the
        # documented invariant #5 engagement threshold, already surfaced as
        # ``gate_active``), and (b) ``_should_gate_modulate_conviction``'s
        # trailing-IC kill-switch (the Phase-1 anti-skill fix's gate). A
        # quant trending gate health needs to see BOTH — a row with
        # ``gate_active=True`` but ``gate_killswitch_active=False`` records
        # the most operationally interesting state ("the n_train threshold
        # is met but the kill-switch suppressed the modulation this cycle"),
        # which would otherwise be invisible in any per-cycle JSONL on disk
        # and only knowable by manually re-running the kill-switch on
        # historical ledger state. Pure read; the kill-switch is itself
        # best-effort and never raises — any fault here degrades to None
        # (the documented acceptable degradation), the same discipline the
        # rest of ``_append_scorer_skill_log`` follows.
        try:
            from paper_trader.backtest import _should_gate_modulate_conviction
            killswitch_active, killswitch_reason = (
                _should_gate_modulate_conviction())
        except Exception as exc:
            killswitch_active = None
            killswitch_reason = f"kill-switch read error: {exc}"
        _gate_active_n = (parsed.get("train_n") is not None
                          and parsed["train_n"] >= 500)
        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            # Surfaces the gate state a quant cares about without re-reading
            # the pickle: the gate engages only at train_n >= 500 (#5).
            "gate_active": _gate_active_n,
            # True ⇒ gate is live on a stale (pre-retune) net — see above.
            "deploy_stale": deploy_stale,
            # Kill-switch decision (additive 2026-05-28 — see comment above):
            # the trailing-OOS-IC short-circuit's verdict on whether the
            # gate should fire THIS cycle. True ⇒ kill-switch is letting the
            # gate act; False ⇒ trailing skill below tolerance (noise OR
            # anti-predictive — Phase-1 fix), gate's modulation is
            # short-circuited; None ⇒ kill-switch read raised.
            "gate_killswitch_active": killswitch_active,
            # The reason string the kill-switch returned — captures
            # `median oos_buy_ic=…` / `skill ledger missing` / etc. so a
            # quant doesn't have to re-derive WHY a given cycle's gate
            # turned on or off.
            "gate_killswitch_reason": killswitch_reason,
            # The TRUE effective gate state — both guards must say active.
            # This is the field a researcher should trend to answer "is the
            # conviction modulation actually firing right now?", with the
            # individual `gate_active` and `gate_killswitch_active` columns
            # available to attribute a False here to the right guard.
            # None when the kill-switch read failed (degrade honestly rather
            # than fabricate either True or False from partial information).
            "gate_effectively_active": (
                _gate_active_n and killswitch_active is True
                if killswitch_active is not None else None
            ),
            **parsed,
        }
        SCORER_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SCORER_SKILL_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in SCORER_SKILL_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > SCORER_SKILL_LOG_KEEP * 2:
                kept = lines[-SCORER_SKILL_LOG_KEEP:]
                tmp = SCORER_SKILL_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(SCORER_SKILL_LOG)
        except Exception as e:
            print(f"[continuous] scorer-skill-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] scorer-skill-log append failed: {e}")
        return False


def _deployed_scorer_n_train() -> int | None:
    """Best-effort read of the *currently deployed* pickle's `n_train`.

    Used as the `n_train_hint` for skill-ledger rows on cycles that did NOT
    retrain (no outcome records / no winner): the status string carries no
    `train_n=` token there, but the gate state a quant cares about
    (`gate_active` ⇔ deployed `n_train >= 500`, invariant #5) is still
    knowable from the on-disk pickle. Never raises — a fault yields `None`
    and the ledger row simply records `train_n=None` (the documented
    acceptable degradation), exactly the discipline `_append_scorer_skill_log`
    itself follows.
    """
    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
        ds = DecisionScorer()
        # An untrained scorer (no pickle on disk yet) reports n_train==0; that
        # is "no deployed model", not "a model trained on 0 rows". Return None
        # so the ledger row records train_n=None / gate_active=False honestly
        # rather than a misleading concrete 0.
        if not ds.is_trained:
            return None
        n = ds.n_train
        return int(n) if n is not None else None
    except Exception:
        return None


def _append_baseline_skill_log(cycle: int, win_start: date, win_end: date,
                               outcomes_path: "Path | str | None" = None) -> bool:
    """Append one structured row to the per-cycle trivial-baseline ledger.

    Answers, durably and per-cycle, the single most decisive documented
    ML/backtest question: *does a one-line rule (raw ``ml_score``) carry more
    out-of-sample rank skill than the 17-feature MLP the conviction gate
    relies on?* Reuses ``baseline_compare.analyze`` verbatim — the EXACT
    read-only path `python3 -m paper_trader.ml.baseline_compare` uses (which
    in turn shares `validation.split_outcomes_temporal` + the universal SELL
    sign-flip with `calibration --oos` and the scorer ledger's OOS metrics),
    so the persisted ``mlp_rank_ic`` equals the CLI's / `calibration --oos`'s
    by construction — a built-in no-drift cross-check, never a re-derivation.

    Best-effort and idempotent-safe: every fault is swallowed (a ledger write
    must NEVER break the continuous loop — the exact discipline of the sibling
    ``_append_scorer_skill_log`` / ``_post_discord`` / the validation
    persister). An untrained scorer or a missing/short outcomes file is
    persisted **honestly** as a ``status='error' verdict='INSUFFICIENT_DATA'``
    row (the "no outcome records" sentinel precedent) rather than skipped, so
    a gap in the trend is visible, not silent. Bounded growth: when the file
    exceeds 2× ``BASELINE_SKILL_LOG_KEEP`` it is atomically rewritten to the
    last ``BASELINE_SKILL_LOG_KEEP`` rows (the decision_outcomes trim idiom —
    a torn truncate would lose skill history, so write tmp then ``.replace``).

    ``gate_active`` mirrors the scorer ledger: the conviction gate engages
    only at deployed ``n_train >= 500`` (invariant #5), so a row with
    ``verdict='MLP_WORSE_THAN_TRIVIAL'`` AND ``gate_active=True`` is the
    quant-decisive "the loop is sizing on a net the data says is worse than a
    free one-liner, right now" state.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        try:
            from paper_trader.ml import baseline_compare as _bc
            rep = _bc.analyze(outcomes_path, oos_only=True)
        except Exception as exc:
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                   "hint": f"baseline_compare unavailable: {type(exc).__name__}"}
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA"}

        mlp = rep.get("mlp") or {}
        n_train = rep.get("n_train")
        try:
            gate_active = (n_train is not None and int(n_train) >= 500)
        except (TypeError, ValueError):
            gate_active = False
        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": rep.get("verdict"),
            "slice": rep.get("slice"),
            "n": rep.get("n"),
            "n_train": n_train,
            "mlp_rank_ic": mlp.get("rank_ic"),
            "mlp_dir_acc": mlp.get("dir_acc"),
            "best_baseline": rep.get("best_baseline"),
            "best_baseline_ic": rep.get("best_baseline_ic"),
            # MLP − best one-liner rank-IC. Negative ⇒ the net is
            # net-negative complexity OOS (the documented finding).
            "ic_gap": rep.get("ic_gap"),
            "gate_active": gate_active,
        }
        BASELINE_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with BASELINE_SKILL_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in BASELINE_SKILL_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > BASELINE_SKILL_LOG_KEEP * 2:
                kept = lines[-BASELINE_SKILL_LOG_KEEP:]
                tmp = BASELINE_SKILL_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(BASELINE_SKILL_LOG)
        except Exception as e:
            print(f"[continuous] baseline-skill-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] baseline-skill-log append failed: {e}")
        return False


def _append_llm_annotation_skill_log(cycle: int, win_start: date, win_end: date,
                                     outcomes_path: "Path | str | None" = None
                                     ) -> bool:
    """Append one structured row to the per-cycle LLM-annotation skill ledger.

    Answers, durably and per-cycle, the most directly operational LLM-
    annotation question: *did the pipeline produce ANY non-zero labels this
    cycle, and if so do they predict realized returns?* Reuses
    ``llm_annotation_skill.analyze`` verbatim so the persisted verdict
    equals the CLI's by construction — a built-in no-drift cross-check.

    Best-effort and idempotent-safe (the ``_append_scorer_skill_log``
    / ``_append_baseline_skill_log`` discipline — a ledger write must NEVER
    break the continuous loop). On any fault we still emit an honest row
    with ``status='error' verdict='NO_LABELS_PRODUCED'`` so a gap in the
    trend is visible, not silent. Bounded growth: when the file exceeds
    2× ``LLM_ANNOTATION_SKILL_LOG_KEEP`` it is atomically rewritten via the
    decision_outcomes trim idiom (tmp + ``.replace`` so a torn truncate
    cannot lose skill history).

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        try:
            from paper_trader.ml import llm_annotation_skill as _las
            rep = _las.analyze(outcomes_path)
        except Exception as exc:
            rep = {"status": "error", "verdict": "NO_LABELS_PRODUCED",
                   "hint": f"llm_annotation_skill unavailable: "
                           f"{type(exc).__name__}",
                   "n_total": 0, "n_endorsed": 0, "n_condemned": 0,
                   "n_unlabeled": 0,
                   "endorsed_mean_return": None,
                   "condemned_mean_return": None,
                   "unlabeled_mean_return": None,
                   "endorsed_minus_condemned": None,
                   "rank_ic": None}
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "NO_LABELS_PRODUCED"}

        # `pipeline_dark` is the ledger-row equivalent of the scorer ledger's
        # `gate_active` / the baseline ledger's `verdict=='MLP_WORSE_THAN_TRIVIAL'`:
        # the single boolean a reading quant cares about for the *current
        # state*. True ⇒ no labels are being produced at all (the documented
        # production-live state); False ⇒ at least some endorsed-or-condemned
        # labels exist this cycle, so the pipeline has woken up.
        n_lab = int(rep.get("n_endorsed") or 0) + int(
            rep.get("n_condemned") or 0)
        pipeline_dark = (n_lab == 0)
        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": rep.get("verdict"),
            "n_total": rep.get("n_total"),
            "n_endorsed": rep.get("n_endorsed"),
            "n_condemned": rep.get("n_condemned"),
            "n_unlabeled": rep.get("n_unlabeled"),
            "endorsed_mean_return": rep.get("endorsed_mean_return"),
            "condemned_mean_return": rep.get("condemned_mean_return"),
            "unlabeled_mean_return": rep.get("unlabeled_mean_return"),
            "endorsed_minus_condemned": rep.get("endorsed_minus_condemned"),
            "rank_ic": rep.get("rank_ic"),
            "pipeline_dark": pipeline_dark,
        }
        LLM_ANNOTATION_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with LLM_ANNOTATION_SKILL_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in LLM_ANNOTATION_SKILL_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > LLM_ANNOTATION_SKILL_LOG_KEEP * 2:
                kept = lines[-LLM_ANNOTATION_SKILL_LOG_KEEP:]
                tmp = LLM_ANNOTATION_SKILL_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(LLM_ANNOTATION_SKILL_LOG)
        except Exception as e:
            print(f"[continuous] llm-annotation-skill-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] llm-annotation-skill-log append failed: {e}")
        return False


def _append_calibrated_reliability_log(cycle: int, win_start: date,
                                       win_end: date,
                                       outcomes_path: "Path | str | None" = None
                                       ) -> bool:
    """Append one structured row to the per-cycle ``predict_calibrated``
    reliability ledger.

    Answers, durably and per-cycle: *does ``predict_calibrated`` actually
    deliver an honest 5-day magnitude reading on data the calibration table
    never saw, and does it measurably narrow the bias vs raw ``predict()``?*
    Reuses ``calibration_reliability.analyze`` verbatim (OOS slice by
    default) so the persisted verdict equals the CLI's by construction —
    a built-in no-drift cross-check, the same idiom the sibling
    ``_append_baseline_skill_log`` / ``_append_llm_annotation_skill_log``
    use.

    Best-effort and idempotent-safe: every fault is swallowed (a ledger
    write must NEVER break the continuous loop — the same discipline as
    every other ``_append_*_skill_log``). An untrained scorer, a legacy
    pickle (no ``label_quantiles``), or a missing outcomes file all degrade
    to a ``status='insufficient_data' verdict='INSUFFICIENT_DATA'`` row
    rather than getting skipped — so a gap in the trend (the documented
    "deployed scorer predates pass #10, calibrated path is dark" state) is
    visible, not silent. Bounded growth: when the file exceeds
    2× ``CALIBRATED_RELIABILITY_LOG_KEEP`` it is atomically rewritten via
    the same tmp+``.replace`` idiom every sibling ledger uses.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        try:
            from paper_trader.ml import calibration_reliability as _cr
            rep = _cr.analyze(outcomes_path, oos_only=True)
        except Exception as exc:
            rep = {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "hint": f"calibration_reliability unavailable: "
                        f"{type(exc).__name__}",
                "n": 0, "spearman": None, "monotone_fraction": None,
                "mean_abs_decile_error": None,
                "raw_mean_abs_decile_error": None,
                "vs_raw_bias_reduction": None,
            }
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA"}

        # ``calibrated_dark`` is the boolean a quant cares about for the
        # current state — mirrors the LLM-annotation ledger's
        # ``pipeline_dark``. True ⇒ ``predict_calibrated`` produced ZERO
        # finite triples this cycle (legacy pickle, untrained scorer, or
        # no outcomes), so every dashboard/CLI consumer of the calibrated
        # value is rendering None for every prediction; False ⇒ the
        # calibrated path is alive and the report carries real data.
        calibrated_dark = (rep.get("n") or 0) == 0
        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": rep.get("verdict"),
            "n": rep.get("n"),
            "oos_n": rep.get("oos_n"),
            "spearman": rep.get("spearman"),
            "monotone_fraction": rep.get("monotone_fraction"),
            "mean_abs_decile_error": rep.get("mean_abs_decile_error"),
            "raw_mean_abs_decile_error": rep.get("raw_mean_abs_decile_error"),
            "vs_raw_bias_reduction": rep.get("vs_raw_bias_reduction"),
            "scorer_n_train": rep.get("scorer_n_train"),
            "outcomes_n": rep.get("outcomes_n"),
            "calibrated_dark": calibrated_dark,
        }
        CALIBRATED_RELIABILITY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with CALIBRATED_RELIABILITY_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in CALIBRATED_RELIABILITY_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > CALIBRATED_RELIABILITY_LOG_KEEP * 2:
                kept = lines[-CALIBRATED_RELIABILITY_LOG_KEEP:]
                tmp = CALIBRATED_RELIABILITY_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(CALIBRATED_RELIABILITY_LOG)
        except Exception as e:
            print(f"[continuous] calibrated-reliability-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] calibrated-reliability-log append failed: {e}")
        return False


def _append_conviction_calibration_log(cycle: int, win_start: date,
                                       win_end: date,
                                       outcomes_path: "Path | str | None" = None
                                       ) -> bool:
    """Append one structured row to the per-cycle conviction-sizing
    calibration ledger.

    Answers, durably and per-cycle: *does the gate's then-applied
    ``conviction_pct`` actually predict realized return, or is sizing
    pure variance with no compensating edge?* Reuses
    ``conviction_calibration.analyze`` verbatim so the persisted verdict
    equals what the read-only CLI reports — a built-in no-drift
    cross-check the sibling ``_append_baseline_skill_log`` /
    ``_append_llm_annotation_skill_log`` / ``_append_calibrated_reliability_log``
    pattern already establishes.

    Why this matters: the existing skill ledgers trend the SCORER's rank
    skill (does the model's prediction rank realized return?). The
    gate's economic effect is a different question entirely — its sizing
    rule scales bets from ~5% (low ml_score) to 25%-40% (leveraged ETF +
    bull regime), so a tiny rank-IC bump cannot reveal whether THE BOT
    REALLY MAKES MORE MONEY WHEN IT SIZES UP. ``conviction_calibration``
    is the analyzer that DOES answer that — its current CLI verdict on
    the live 1173-row OOS slice is ``MISCALIBRATED`` (spearman +0.011,
    top-bottom realized spread -0.15pp) — but until this wiring landed
    that was a one-shot CLI output, not a trendable per-cycle signal. A
    skeptical quant needs to know the moment the sizing rule starts to
    work (or stops working), not the moment they happen to remember to
    run the CLI.

    Best-effort and idempotent-safe: every fault is swallowed (a ledger
    write must NEVER break the continuous loop — the
    ``_append_scorer_skill_log`` discipline). On any fault we still emit
    an honest row with ``status='error' verdict='INSUFFICIENT_DATA'`` so
    a gap in the trend is visible, not silent. Bounded growth: when the
    file exceeds 2× ``CONVICTION_CALIBRATION_LOG_KEEP`` it is atomically
    rewritten via the decision_outcomes trim idiom (tmp + ``.replace`` so
    a torn truncate cannot lose skill history).

    The ``sizing_dark`` boolean mirrors the sibling ledgers' ``*_dark``
    flag (``pipeline_dark`` / ``calibrated_dark``): True when zero BUY
    rows carry a parseable ``conviction_pct`` (the documented state
    before the 2026-05-21 ``_parse_conviction_pct`` feature shipped —
    historical outcomes have no conviction field, so older corpora plus
    the sub-gate cycles look ``sizing_dark`` honestly). False means the
    analyzer actually had data to work with.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        try:
            from paper_trader.ml import conviction_calibration as _cc
            rep = _cc.analyze(outcomes_path)
        except Exception as exc:
            rep = {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "hint": f"conviction_calibration unavailable: "
                        f"{type(exc).__name__}",
                "n": 0, "spearman": None,
                "mean_conviction": None, "mean_realized": None,
                "top_minus_bottom_realized_pct": None,
                "monotone_fraction": None,
                "n_dropped_action": None, "n_dropped_conviction": None,
                "n_dropped_return": None,
            }
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA"}

        # ``sizing_dark`` is True when no BUY rows carried a parseable
        # ``conviction_pct``. Historical outcome rows (cycles before the
        # ``_parse_conviction_pct`` feature shipped) have no conviction
        # field at all and surface here honestly. Once the corpus is
        # dominated by post-feature rows the boolean flips on its own —
        # no manual reset needed.
        n_obs = int(rep.get("n") or 0)
        sizing_dark = (n_obs == 0)
        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": rep.get("verdict"),
            "n": n_obs,
            "spearman": rep.get("spearman"),
            "mean_conviction": rep.get("mean_conviction"),
            "mean_realized": rep.get("mean_realized"),
            "top_minus_bottom_realized_pct":
                rep.get("top_minus_bottom_realized_pct"),
            "monotone_fraction": rep.get("monotone_fraction"),
            "n_dropped_conviction": rep.get("n_dropped_conviction"),
            "sizing_dark": sizing_dark,
        }
        CONVICTION_CALIBRATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with CONVICTION_CALIBRATION_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in
                     CONVICTION_CALIBRATION_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > CONVICTION_CALIBRATION_LOG_KEEP * 2:
                kept = lines[-CONVICTION_CALIBRATION_LOG_KEEP:]
                tmp = CONVICTION_CALIBRATION_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(CONVICTION_CALIBRATION_LOG)
        except Exception as e:
            print(f"[continuous] conviction-calibration-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] conviction-calibration-log append failed: {e}")
        return False


def _append_bootstrap_ci_skill_log(cycle: int, win_start: date,
                                    win_end: date,
                                    outcomes_path: "Path | str | None" = None,
                                    n_bootstrap: int = BOOTSTRAP_CI_N,
                                    ci_level: float = 0.95,
                                    seed: int = 42) -> bool:
    """Append one structured row to the per-cycle bootstrap-CI ledger.

    Answers, durably and per-cycle, the operator-decisive question every
    point-estimate OOS diagnostic structurally cannot: *is the deployed
    scorer's OOS rank-IC statistically distinguishable from a coin flip
    at this n_oos?* Reuses ``oos_bootstrap_ci.bootstrap_ci`` verbatim —
    the EXACT read-only path the CLI uses — so the persisted CI bounds
    equal the CLI's by construction (a built-in no-drift cross-check,
    the sibling ``_append_baseline_skill_log`` / ``_append_calibrated_reliability_log``
    pattern). The OOS slice is the same temporal holdout
    ``_train_decision_scorer`` uses via ``split_outcomes_temporal``, so
    the CI bounds describe the EXACT records the trustworthy generalization
    metrics in ``scorer_skill_log.jsonl`` are computed on — operators can
    cross-read ``oos_ic`` (point) and ``rank_ic_ci_low/_high`` (uncertainty)
    on the SAME cycle row by joining on ``cycle``.

    Verdict ladder:
      * ``NOT_TRAINED``       — deployed scorer has no pickle / failed to load
      * ``INSUFFICIENT_DATA`` — fewer than ``MIN_PAIRS_FOR_CI`` (30) valid OOS pairs
      * ``SKILL_DETECTED``    — rank-IC 95% CI strictly above 0
      * ``NO_SKILL_DETECTED`` — rank-IC 95% CI straddles or sits at/below 0
      * ``ERROR``             — best-effort fallback (the diagnostic raised)

    Why this matters as a separate ledger and not extra columns in
    ``scorer_skill_log.jsonl``: the bootstrap itself is a *post-train
    diagnostic* (loads the just-pickled scorer + the trim-bounded
    outcomes tail) that costs ~1-2s — the existing scorer log is built
    inside the synchronous train path where adding the bootstrap would
    couple two unrelated operations. Keeping the CI in its own ledger
    means the (now slower) CI computation can never block / poison the
    point-estimate metrics ``_train_decision_scorer`` writes, and a
    bootstrap-CI consumer can join on ``cycle`` to align both views
    without forcing a schema migration on every existing consumer
    (``skill_trend`` / ``gate_audit`` / ``sector_skill`` / etc.).

    Best-effort and idempotent-safe: every fault is swallowed (a ledger
    write must NEVER break the continuous loop — the
    ``_append_scorer_skill_log`` discipline). On any fault we still emit
    an honest row with ``status='error' verdict='ERROR'`` so a gap in
    the trend is visible, not silent. Bounded growth: when the file
    exceeds 2× ``BOOTSTRAP_CI_SKILL_LOG_KEEP`` it is atomically rewritten
    via the decision_outcomes trim idiom (tmp + ``.replace`` so a torn
    truncate cannot lose skill history).

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        # Default-empty result so EVERY exit path below emits a row — a
        # gap in the trend is what dies in the dark, never an honest
        # error-keyed row. Mirrors `_append_conviction_calibration_log`'s
        # fallback dict construction.
        result: dict = {
            "status": "error", "n": 0, "n_bootstrap": 0,
            "ci_level": float(ci_level),
            "rmse": {"value": None, "ci_low": None, "ci_high": None},
            "dir_acc": {"value": None, "ci_low": None, "ci_high": None},
            "rank_ic": {"value": None, "ci_low": None, "ci_high": None},
        }
        verdict = "ERROR"
        hint: str | None = None
        try:
            from paper_trader.ml.decision_scorer import DecisionScorer
            from paper_trader.ml.oos_bootstrap_ci import (
                bootstrap_ci, MIN_PAIRS_FOR_CI)
            from paper_trader.validation import split_outcomes_temporal

            # Load outcomes tail bounded to MAX_OUTCOMES_FOR_TRAINING so the
            # CI describes the same records the trainer / `scorer_skill_log`
            # see — older rows describe a stale signal regime per the
            # documented MAX_OUTCOMES_FOR_TRAINING rationale.
            outcomes: list[dict] = []
            p = Path(outcomes_path) if not isinstance(
                outcomes_path, Path) else outcomes_path
            if p.exists():
                lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
                lines = lines[-MAX_OUTCOMES_FOR_TRAINING:]
                for ln in lines:
                    try:
                        outcomes.append(json.loads(ln))
                    except Exception:
                        continue

            # Same temporal split `_train_decision_scorer` uses so the CI
            # describes the trustworthy generalization slice. A split failure
            # (validation module missing, malformed corpus) degrades to "no
            # OOS records" honestly, NOT to "evaluate on all records"
            # (which would silently include the train fold and fabricate
            # apparent skill).
            oos_records: list[dict] = []
            try:
                _, oos_records = split_outcomes_temporal(
                    outcomes, oos_fraction=0.2)
            except Exception as split_exc:
                hint = (f"temporal split unavailable: "
                        f"{type(split_exc).__name__}")

            scorer = DecisionScorer()
            # bootstrap_ci handles all degenerate cases internally and
            # returns a well-formed dict — never raises.
            result = bootstrap_ci(
                scorer, oos_records,
                n_bootstrap=int(n_bootstrap),
                ci_level=float(ci_level),
                seed=int(seed),
            )
            status = result.get("status", "error")
            if status == "ok":
                rk = result.get("rank_ic") or {}
                ic_lo = rk.get("ci_low")
                # CI excluding 0 ⇒ statistically distinguishable skill.
                # `ic_lo is not None` guards against a degenerate
                # bootstrap (constant resamples) where bootstrap_ci returns
                # CIs as None — treat as no detectable skill, mirroring the
                # `skill_uncertainty` verdict-ladder discipline.
                if ic_lo is not None and float(ic_lo) > 0.0:
                    verdict = "SKILL_DETECTED"
                else:
                    verdict = "NO_SKILL_DETECTED"
            elif status == "scorer_not_trained":
                verdict = "NOT_TRAINED"
            elif status == "insufficient_data":
                verdict = "INSUFFICIENT_DATA"
            elif status == "empty":
                # No outcome records on disk yet (fresh continuous-loop start
                # before the first cycle accumulated outcomes). Honestly
                # surfaced as INSUFFICIENT_DATA — same semantics from a
                # downstream consumer's perspective.
                verdict = "INSUFFICIENT_DATA"
            else:
                verdict = "ERROR"
        except Exception as exc:
            hint = f"{type(exc).__name__}: {exc}"

        rk = result.get("rank_ic") or {}
        rm = result.get("rmse") or {}
        da = result.get("dir_acc") or {}
        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": result.get("status"),
            "verdict": verdict,
            "n_oos": int(result.get("n") or 0),
            "n_bootstrap": int(result.get("n_bootstrap") or 0),
            "ci_level": result.get("ci_level"),
            "rank_ic_point": rk.get("value"),
            "rank_ic_ci_low": rk.get("ci_low"),
            "rank_ic_ci_high": rk.get("ci_high"),
            "rmse_point": rm.get("value"),
            "rmse_ci_low": rm.get("ci_low"),
            "rmse_ci_high": rm.get("ci_high"),
            "dir_acc_point": da.get("value"),
            "dir_acc_ci_low": da.get("ci_low"),
            "dir_acc_ci_high": da.get("ci_high"),
        }
        if hint:
            row["hint"] = hint
        BOOTSTRAP_CI_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with BOOTSTRAP_CI_SKILL_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in
                     BOOTSTRAP_CI_SKILL_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > BOOTSTRAP_CI_SKILL_LOG_KEEP * 2:
                kept = lines[-BOOTSTRAP_CI_SKILL_LOG_KEEP:]
                tmp = BOOTSTRAP_CI_SKILL_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(BOOTSTRAP_CI_SKILL_LOG)
        except Exception as e:
            print(f"[continuous] bootstrap-ci-skill-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] bootstrap-ci-skill-log append failed: {e}")
        return False


def _append_stop_out_skill_log(cycle: int, win_start: date,
                                win_end: date,
                                outcomes_path: "Path | str | None" = None
                                ) -> bool:
    """Append one structured row to the per-cycle stop-out skill ledger.

    Answers, durably and per-cycle: *is the inherited ``backtest._buy``
    ``stop_loss = price * 0.92`` band (the -8% downside arm) a real
    defensive arm — saving more from limited-loss trades than it costs
    in prematurely-exited recoveries — or is it variance-only chop?*
    Reuses ``stop_out_audit.analyze`` verbatim so the persisted verdict
    equals the read-only CLI's by construction — a built-in no-drift
    cross-check, the same idiom every sibling
    ``_append_*_skill_log`` / ``_append_*_calibration_log`` uses.

    Best-effort and idempotent-safe: every fault is swallowed (a ledger
    write must NEVER break the continuous loop — the documented
    ``_append_scorer_skill_log`` discipline). On any fault we still emit
    an honest row with ``status='error' verdict='INSUFFICIENT_DATA'`` so
    a gap in the trend is visible, not silent. Bounded growth: when the
    file exceeds 2× ``STOP_OUT_SKILL_LOG_KEEP`` it is atomically
    rewritten via the decision_outcomes trim idiom (tmp + ``.replace``
    so a torn truncate cannot lose skill history).

    The ``stop_dark`` boolean mirrors the sibling ledgers' ``*_dark``
    flag (``pipeline_dark`` / ``calibrated_dark`` / ``sizing_dark``):
    True when zero BUY rows carry a finite ``forward_intraperiod_min_5d``
    (the documented current state — the 8753-row historical corpus
    pre-dates the 2026-05-23 intraperiod feature, so older corpora
    look ``stop_dark`` honestly). False means the analyzer actually had
    intraperiod data to work with.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        try:
            from paper_trader.ml import stop_out_audit as _soa
            rep = _soa.analyze(outcomes_path)
        except Exception as exc:
            rep = {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "hint": f"stop_out_audit unavailable: {type(exc).__name__}",
                "stop_pct": None, "n_buys": 0, "n_with_intraperiod": 0,
                "n_stop_triggered": 0, "pct_stop_triggered": None,
                "mean_realized_return_pct": None,
                "mean_stop_protected_return_pct": None,
                "stop_benefit_pct": None,
                "median_realized_return_pct": None,
                "median_stop_protected_return_pct": None,
            }
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA"}

        # ``stop_dark`` is True when zero BUYs carry a finite
        # ``forward_intraperiod_min_5d`` — the documented "older
        # corpora pre-date the 2026-05-23 feature" state. Once the
        # rolling 5000-record training tail is dominated by post-feature
        # rows the boolean flips on its own — no manual reset needed.
        n_intra = int(rep.get("n_with_intraperiod") or 0)
        stop_dark = (n_intra == 0)
        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": rep.get("verdict"),
            "stop_pct": rep.get("stop_pct"),
            "n_buys": rep.get("n_buys"),
            "n_with_intraperiod": n_intra,
            "n_stop_triggered": rep.get("n_stop_triggered"),
            "pct_stop_triggered": rep.get("pct_stop_triggered"),
            "mean_realized_return_pct": rep.get("mean_realized_return_pct"),
            "mean_stop_protected_return_pct":
                rep.get("mean_stop_protected_return_pct"),
            "stop_benefit_pct": rep.get("stop_benefit_pct"),
            "median_realized_return_pct":
                rep.get("median_realized_return_pct"),
            "median_stop_protected_return_pct":
                rep.get("median_stop_protected_return_pct"),
            "stop_dark": stop_dark,
        }
        STOP_OUT_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with STOP_OUT_SKILL_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in
                     STOP_OUT_SKILL_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > STOP_OUT_SKILL_LOG_KEEP * 2:
                kept = lines[-STOP_OUT_SKILL_LOG_KEEP:]
                tmp = STOP_OUT_SKILL_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(STOP_OUT_SKILL_LOG)
        except Exception as e:
            print(f"[continuous] stop-out-skill-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] stop-out-skill-log append failed: {e}")
        return False


def _append_mfe_skill_log(cycle: int, win_start: date, win_end: date,
                          outcomes_path: "Path | str | None" = None
                          ) -> bool:
    """Append one structured row to the per-cycle MFE-conversion /
    take-profit skill ledger.

    Answers, durably and per-cycle: *does the inherited ``backtest._buy``
    ``take_profit = price * 1.15`` band (the +15% upside arm) capture
    more upside than it forfeits in trades that would have recovered
    further? And what fraction of the intraperiod peak (MFE — Maximum
    Favorable Excursion) does the 5d endpoint actually retain?* The
    mean-conversion-ratio question is the textbook quant signal for
    "peak then crater" trade shape — a low ratio means a TP would have
    captured peaks the bot let revert. Reuses ``mfe_conversion.analyze``
    verbatim so the persisted verdict / ratio equal the CLI's by
    construction — same SSOT cross-check pattern every sibling ledger
    follows.

    Same defensive contract as ``_append_stop_out_skill_log``: best-
    effort, honest gap rows when the intraperiod corpus is empty
    (``tp_dark=True``), atomic bounded trim, never breaks the loop.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        try:
            from paper_trader.ml import mfe_conversion as _mfe
            rep = _mfe.analyze(outcomes_path)
        except Exception as exc:
            rep = {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "hint": f"mfe_conversion unavailable: {type(exc).__name__}",
                "tp_pct": None, "n_buys": 0, "n_with_intraperiod": 0,
                "n_tp_triggered": 0, "pct_tp_triggered": None,
                "n_positive_mfe": 0, "n_reverted": 0, "pct_reverted": None,
                "mean_realized_return_pct": None,
                "mean_tp_protected_return_pct": None,
                "tp_benefit_pct": None,
                "median_realized_return_pct": None,
                "median_tp_protected_return_pct": None,
                "mean_mfe_pct": None, "median_mfe_pct": None,
                "mean_conversion_ratio": None,
                "median_conversion_ratio": None,
            }
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA"}

        # ``tp_dark`` mirrors the stop-out ledger's ``stop_dark``. True
        # when zero BUYs carry a finite ``forward_intraperiod_max_5d``
        # — the documented current state until post-feature rows
        # dominate the training tail.
        n_intra = int(rep.get("n_with_intraperiod") or 0)
        tp_dark = (n_intra == 0)
        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": rep.get("verdict"),
            "tp_pct": rep.get("tp_pct"),
            "n_buys": rep.get("n_buys"),
            "n_with_intraperiod": n_intra,
            "n_tp_triggered": rep.get("n_tp_triggered"),
            "pct_tp_triggered": rep.get("pct_tp_triggered"),
            "n_positive_mfe": rep.get("n_positive_mfe"),
            "n_reverted": rep.get("n_reverted"),
            "pct_reverted": rep.get("pct_reverted"),
            "mean_realized_return_pct": rep.get("mean_realized_return_pct"),
            "mean_tp_protected_return_pct":
                rep.get("mean_tp_protected_return_pct"),
            "tp_benefit_pct": rep.get("tp_benefit_pct"),
            "median_realized_return_pct":
                rep.get("median_realized_return_pct"),
            "median_tp_protected_return_pct":
                rep.get("median_tp_protected_return_pct"),
            "mean_mfe_pct": rep.get("mean_mfe_pct"),
            "median_mfe_pct": rep.get("median_mfe_pct"),
            "mean_conversion_ratio": rep.get("mean_conversion_ratio"),
            "median_conversion_ratio": rep.get("median_conversion_ratio"),
            "tp_dark": tp_dark,
        }
        MFE_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with MFE_SKILL_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in MFE_SKILL_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > MFE_SKILL_LOG_KEEP * 2:
                kept = lines[-MFE_SKILL_LOG_KEEP:]
                tmp = MFE_SKILL_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(MFE_SKILL_LOG)
        except Exception as e:
            print(f"[continuous] mfe-skill-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] mfe-skill-log append failed: {e}")
        return False


def _append_stop_band_sweep_log(cycle: int, win_start: date, win_end: date,
                                outcomes_path: "Path | str | None" = None
                                ) -> bool:
    """Append one structured row to the per-cycle multi-horizon stop-band
    sweep ledger.

    Answers, durably and per-cycle: *across a 2-D grid of candidate STOP
    bands (3-20%) × horizons (5d/10d/20d), is there a (band, horizon)
    cell that measurably beats the deployed (-8%, 5d) cell on realized
    return?* This is the multi-horizon extension of the sibling
    ``_append_stop_out_skill_log`` — the same question with a wider
    instrumentation lens. Verdict ladder mirrors the analyzer:
    ``INSUFFICIENT_DATA`` / ``NO_BAND_HELPS`` / ``DEPLOYED_OPTIMAL`` /
    ``CELL_BEATS_DEPLOYED``.

    Why this matters: the pass #42 author shipped the 10d/20d
    forward_intraperiod_min/max fields specifically so a multi-horizon
    sweep would become possible. Until this wiring landed the verdict
    was a CLI-only one-shot — a quant could not see the moment a
    longer-window or tighter band starts to dominate the deployed cell.
    The ``best_cell`` keys flatten the answer to one line per cycle so
    the trend is immediately readable (``best_stop_pct`` /
    ``best_horizon`` / ``best_benefit_pct``).

    Best-effort and idempotent-safe (the ``_append_scorer_skill_log``
    discipline — a ledger write must NEVER break the continuous loop).
    On any fault we still emit an honest row with ``status='error'
    verdict='INSUFFICIENT_DATA'`` so a gap in the trend is visible, not
    silent. Bounded growth: when the file exceeds 2× ``STOP_BAND_SWEEP_LOG_KEEP``
    it is atomically rewritten via the tmp+``.replace`` idiom every
    sibling ledger uses.

    The ``sweep_dark`` boolean mirrors the sibling ledgers' ``*_dark``
    flags: True when the deployed horizon's intraperiod coverage is
    below ``MIN_BUYS`` (the documented state while the post-2026-05-23
    intraperiod corpus is still warming up). False means the analyzer
    actually had enough deployed-horizon data to evaluate.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        try:
            from paper_trader.ml import stop_band_sweep as _sbs
            rep = _sbs.analyze(outcomes_path)
        except Exception as exc:
            rep = {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "hint": f"stop_band_sweep unavailable: {type(exc).__name__}",
                "n_buys": 0,
                "n_with_intraperiod_per_horizon": {},
                "baseline_no_stop_mean_pct_per_horizon": {},
                "best_cell": None,
                "deployed_cell_benefit_pct": None,
            }
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA"}

        # ``sweep_dark`` is True when verdict is INSUFFICIENT_DATA — the
        # deployed-horizon coverage hasn't yet cleared the min_buys
        # threshold. Once enough post-2026-05-23 outcome rows accumulate
        # the boolean flips on its own.
        verdict = rep.get("verdict") or "INSUFFICIENT_DATA"
        sweep_dark = (verdict == "INSUFFICIENT_DATA")

        # Flatten the best cell to top-level columns — `best_cell` itself
        # is a nested dict, but a JSONL consumer wants single-column
        # access for `jq '.best_benefit_pct'` style queries.
        best = rep.get("best_cell") or {}
        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": verdict,
            "deployed_stop_pct": rep.get("deployed_stop_pct"),
            "deployed_horizon": rep.get("deployed_horizon"),
            "deployed_cell_benefit_pct": rep.get("deployed_cell_benefit_pct"),
            "best_stop_pct": best.get("stop_pct"),
            "best_horizon": best.get("horizon"),
            "best_benefit_pct": best.get("benefit_pct"),
            "best_n_triggered": best.get("n_triggered"),
            "best_pct_triggered": best.get("pct_triggered"),
            "best_mean_protected_return_pct":
                best.get("mean_protected_return_pct"),
            "n_buys": rep.get("n_buys"),
            "n_with_intraperiod_per_horizon":
                rep.get("n_with_intraperiod_per_horizon"),
            "baseline_no_stop_mean_pct_per_horizon":
                rep.get("baseline_no_stop_mean_pct_per_horizon"),
            "sweep_dark": sweep_dark,
        }
        STOP_BAND_SWEEP_LOG.parent.mkdir(parents=True, exist_ok=True)
        with STOP_BAND_SWEEP_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in STOP_BAND_SWEEP_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > STOP_BAND_SWEEP_LOG_KEEP * 2:
                kept = lines[-STOP_BAND_SWEEP_LOG_KEEP:]
                tmp = STOP_BAND_SWEEP_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(STOP_BAND_SWEEP_LOG)
        except Exception as e:
            print(f"[continuous] stop-band-sweep-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] stop-band-sweep-log append failed: {e}")
        return False


def _append_gate_arm_skill_log(cycle: int, win_start: date, win_end: date,
                               outcomes_path: "Path | str | None" = None
                               ) -> bool:
    """Append one structured row to the per-cycle gate-arm historical skill
    ledger.

    Answers, durably and per-cycle: *do the conviction gate's arms
    (×0.6 / ×0.85 / ×1.0 / ×1.15 / ×1.3) — bucketed by the gate's TRUE
    then-deployed prediction (``gate_scorer_pred``, not a counterfactual
    re-predict with today's pickle) — realize differentiated economic
    outcomes, or is the bucketing just noise?* Reuses
    ``gate_arm_historical.analyze`` verbatim so the persisted verdict
    equals the read-only CLI's by construction — a built-in no-drift
    cross-check, the same idiom every sibling ``_append_*_skill_log``
    uses.

    Why this matters: the deployed scorer carries strong measured OOS
    rank-IC (+0.48 on the live OOS tail) — but ``gate_arm_historical``
    reports ``GATE_INEFFECTIVE`` because the actual bucket assignment
    only captures ~1% of that rank skill (the ×1.30 arm realizes
    +3.99% vs the ×0.60 arm's +4.12%, spread -0.13pp). That gap is the
    most economically decisive *unmonitored* fact about the gate: the
    model ranks but the buckets don't extract. A skeptical quant needs
    to know if/when bucket tuning recovers economic edge — and the
    moment historic data sees the arms diverge (a real ``GATE_EFFECTIVE``
    cycle) — without manually invoking the CLI. The
    ``arm_monotone_fraction`` (1.0 = perfectly ordered, 0.5 = coin
    flip) is the single best one-shot signal of bucket health.

    Best-effort and idempotent-safe: every fault is swallowed (a ledger
    write must NEVER break the continuous loop — the same discipline as
    every other ``_append_*_skill_log``). On any fault we still emit an
    honest row with ``status='error' verdict='INSUFFICIENT_DATA'`` so a
    gap in the trend is visible, not silent. Bounded growth: when the
    file exceeds 2× ``GATE_ARM_SKILL_LOG_KEEP`` it is atomically
    rewritten via the tmp+``.replace`` idiom every sibling ledger uses.

    The ``gate_dark`` boolean mirrors the sibling ledgers' ``*_dark``
    flags (``pipeline_dark`` / ``calibrated_dark`` / ``sizing_dark`` /
    ``stop_dark`` / ``tp_dark``): True when zero BUY rows carry a
    parseable ``gate_scorer_pred`` (the documented state before the
    2026-05-18 ``_parse_gate_decision`` feature shipped, or any cycle
    where ``n_train<500`` so the gate never acted). False means the
    analyzer actually had real then-deployed-prediction data to
    bucket.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        try:
            from paper_trader.ml import gate_arm_historical as _gah
            rep = _gah.analyze(outcomes_path, oos_only=True)
        except Exception as exc:
            rep = {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "hint": f"gate_arm_historical unavailable: "
                        f"{type(exc).__name__}",
                "n": 0, "arms": [],
                "strong_tailwind_minus_headwind_pp": None,
                "arm_monotone_fraction": None,
                "n_dropped_no_gate_pred": 0,
                "n_dropped_off_dist": 0,
                "n_dropped_no_return": 0,
            }
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA"}

        # ``gate_dark`` is True when zero BUYs carried a parseable
        # ``gate_scorer_pred`` — the historical corpus pre-dates the
        # 2026-05-18 capture, or every recent cycle ran sub-gate
        # (``n_train<500``). Once post-capture rows accumulate the
        # boolean flips on its own. Same logic as sibling ``*_dark``.
        n_obs = int(rep.get("n") or 0)
        gate_dark = (n_obs == 0)

        # Extract per-arm realized returns into flat columns so a JSON
        # consumer can query without parsing the nested ``arms`` list.
        # The list still ships intact for completeness; the flat columns
        # are the ergonomic surface (``mean_x06`` / ``mean_x13`` etc.).
        # Use 2-decimal mapped keys per multiplier to keep the schema
        # stable even if ``_ARM_MULT`` ever extends.
        arms = rep.get("arms") or []
        _MULT_TO_KEY = {0.6: "x06", 0.85: "x085", 1.0: "x10",
                        1.15: "x115", 1.3: "x13"}
        per_arm_mean: dict[str, float | None] = {}
        per_arm_n: dict[str, int] = {}
        for a in arms:
            try:
                mult = float(a.get("multiplier") or 0.0)
            except (TypeError, ValueError):
                continue
            key = _MULT_TO_KEY.get(round(mult, 2))
            if key is None:
                continue
            mr = a.get("mean_realized")
            try:
                per_arm_mean[f"mean_{key}"] = (None if mr is None
                                               else round(float(mr), 4))
            except (TypeError, ValueError):
                per_arm_mean[f"mean_{key}"] = None
            try:
                per_arm_n[f"n_{key}"] = int(a.get("n") or 0)
            except (TypeError, ValueError):
                per_arm_n[f"n_{key}"] = 0

        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": rep.get("verdict"),
            "n": n_obs,
            "slice": rep.get("slice"),
            "outcomes_n": rep.get("outcomes_n"),
            "strong_tailwind_minus_headwind_pp":
                rep.get("strong_tailwind_minus_headwind_pp"),
            "arm_monotone_fraction": rep.get("arm_monotone_fraction"),
            "n_dropped_no_gate_pred": rep.get("n_dropped_no_gate_pred"),
            "n_dropped_off_dist": rep.get("n_dropped_off_dist"),
            "n_dropped_no_return": rep.get("n_dropped_no_return"),
            "gate_dark": gate_dark,
            **per_arm_mean,
            **per_arm_n,
        }
        GATE_ARM_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with GATE_ARM_SKILL_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in
                     GATE_ARM_SKILL_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > GATE_ARM_SKILL_LOG_KEEP * 2:
                kept = lines[-GATE_ARM_SKILL_LOG_KEEP:]
                tmp = GATE_ARM_SKILL_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(GATE_ARM_SKILL_LOG)
        except Exception as e:
            print(f"[continuous] gate-arm-skill-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] gate-arm-skill-log append failed: {e}")
        return False


def _append_gate_pnl_skill_log(cycle: int, win_start: date, win_end: date,
                               outcomes_path: "Path | str | None" = None
                               ) -> bool:
    """Append one structured row to the per-cycle gate economic-counterfactual
    ledger.

    Answers, durably and per-cycle: *aggregated across all five conviction
    gate arms (×0.6 / ×0.85 / ×1.0 / ×1.15 / ×1.3) and weighted by how
    often each fires, does the multiplier overlay net ADD or SUBTRACT
    realized return on the sized fills, or is the reallocation pure
    variance with no economic edge?* Reuses ``gate_pnl.analyze`` verbatim
    (OOS slice by default — the trustworthy, generalization-relevant view
    every sibling ledger uses) so the persisted verdict equals the
    read-only CLI's by construction — a built-in no-drift cross-check,
    the same SSOT idiom every sibling ``_append_*_skill_log`` follows.

    Why this matters and how it differs from sibling gate ledgers:

      * ``gate_arm_skill_log`` (``gate_arm_historical``) reports per-arm
        mean realized returns and ``arm_monotone_fraction``, bucketed by
        the gate's TRUE then-deployed prediction. That answers
        "do the buckets separate?", NOT "does the reallocation pay?". A
        gate can score ``GATE_INEFFECTIVE`` on bucket spread while the
        portfolio-level *summed* effect is meaningfully positive (or
        negative) — what matters for the keep/kill decision.
      * ``gate_audit`` (CLI-only) reports the two-extreme spread and
        ignores both the three middle arms AND how often each fires.
      * Existing ``oos_ic`` / ``oos_rmse`` ledgers measure rank skill,
        not realized economic effect. Strong rank can co-exist with a
        gate that hurts realized return.

    ``gate_pnl`` is the assumption-free roll-up: equal-base
    ``Σ mᵢrᵢ / Σ mᵢ`` minus ``mean(rᵢ)``. Verdict ladder is
    ``INSUFFICIENT_DATA`` / ``GATE_SUBTRACTS_RETURN`` /
    ``GATE_RETURN_NEUTRAL`` / ``GATE_ADDS_RETURN`` at
    ``EDGE_TOL_PP=1.0pp``. Captures both the verdict-driving
    ``equal_weight_gate_contribution_pp`` and the secondary,
    reconstruction-approximate ``sized_gate_contribution_pp`` (the
    AGENTS.md "never folded into verdict" honesty pattern).

    Best-effort and idempotent-safe: every fault is swallowed (a ledger
    write must NEVER break the continuous loop — the documented
    ``_append_scorer_skill_log`` / ``_post_discord`` / validation
    persister discipline). An untrained scorer, a missing outcomes file,
    or a record below ``MIN_TOTAL=30`` all degrade to
    ``status='error'/INSUFFICIENT_DATA`` with ``gate_pnl_dark=True`` so
    a gap in the trend is visible, not silent. Bounded growth: when the
    file exceeds 2× ``GATE_PNL_SKILL_LOG_KEEP`` it is atomically
    rewritten via the tmp+``.replace`` idiom every sibling ledger uses.

    The ``gate_pnl_dark`` boolean mirrors the sibling ledgers' ``*_dark``
    flags (``pipeline_dark`` / ``calibrated_dark`` / ``sizing_dark`` /
    ``stop_dark`` / ``tp_dark`` / ``gate_dark`` /
    ``signal_dark``). True when ``gate_pnl.analyze`` returned zero
    usable ``(pred, realized)`` pairs — the documented
    pre-``n_train>=500`` cycles, or when no outcomes file exists yet.
    False means the analyzer actually had data to roll up.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        try:
            from paper_trader.ml import gate_pnl as _gp
            rep = _gp.analyze(outcomes_path, oos_only=True)
        except Exception as exc:
            rep = {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "hint": f"gate_pnl unavailable: {type(exc).__name__}",
                "n": 0, "gate_off_mean_pct": None,
                "gate_on_mean_pct": None,
                "equal_weight_gate_contribution_pp": None,
                "sized_gate_contribution_pp": None,
                "sized_n": 0, "avg_gate_multiplier": None,
            }
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                   "n": 0,
                   "equal_weight_gate_contribution_pp": None,
                   "sized_gate_contribution_pp": None,
                   "sized_n": 0}

        n_obs = int(rep.get("n") or 0)
        # ``gate_pnl_dark`` is True when the analyzer had no usable pairs to
        # roll up — pre-``n_train>=500`` cycles, missing outcomes file, or
        # an entirely sub-MIN_TOTAL slice. Mirrors sibling ``*_dark`` flags.
        gate_pnl_dark = (n_obs == 0)
        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": rep.get("verdict"),
            "n": n_obs,
            "slice": rep.get("slice"),
            "n_records_considered": rep.get("n_records_considered"),
            "n_train": rep.get("n_train"),
            "gate_off_mean_pct": rep.get("gate_off_mean_pct"),
            "gate_on_mean_pct": rep.get("gate_on_mean_pct"),
            # The verdict-driving headline — equal-base contribution in
            # percentage points. Negative ⇒ the overlay net sizes toward
            # losers and not gating would have realized more on these
            # sized fills.
            "equal_weight_gate_contribution_pp":
                rep.get("equal_weight_gate_contribution_pp"),
            # Secondary, reconstruction-approximate. Reported but NEVER
            # folded into the verdict (the gate_pnl honesty pattern).
            "sized_gate_contribution_pp":
                rep.get("sized_gate_contribution_pp"),
            "sized_n": rep.get("sized_n"),
            "avg_gate_multiplier": rep.get("avg_gate_multiplier"),
            "gate_pnl_dark": gate_pnl_dark,
        }
        GATE_PNL_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with GATE_PNL_SKILL_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in
                     GATE_PNL_SKILL_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > GATE_PNL_SKILL_LOG_KEEP * 2:
                kept = lines[-GATE_PNL_SKILL_LOG_KEEP:]
                tmp = GATE_PNL_SKILL_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(GATE_PNL_SKILL_LOG)
        except Exception as e:
            print(f"[continuous] gate-pnl-skill-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] gate-pnl-skill-log append failed: {e}")
        return False


def _append_persona_skill_log(cycle: int, win_start: date, win_end: date,
                              outcomes_path: "Path | str | None" = None
                              ) -> bool:
    """Append one structured row to the per-cycle per-persona
    decision-signal skill ledger.

    Answers, durably and per-cycle, the decisive question
    ``persona_leaderboard`` *cannot*: within each persona's own decisions,
    does the persona's signal actually rank-predict realized outcomes, or
    is the persona's return pure leveraged-beta noise? Reuses
    ``persona_skill.persona_skill`` verbatim so the persisted verdict
    equals the read-only CLI's by construction — a built-in no-drift
    cross-check the sibling ``_append_*_skill_log`` pattern already
    establishes.

    Why this matters: the ``persona_leaderboard`` ledger aggregates
    run-level ``vs_spy_pct`` but a single persona routinely posts +1000%
    on one window and -80% on the next — that dispersion is leveraged
    beta, NOT demonstrated signal skill. The decision-level rank-IC the
    ``persona_skill`` analyzer computes IS the honest signal: it asks
    whether a higher signal precedes a better realized outcome on the
    persona's OWN choices. The single most-decisive state — at least
    one persona is ANTI-predictive (``HAS_INVERTED_PERSONA``: a persona
    where the stronger its signal, the WORSE its 5d outcome) — is the
    actionable red flag the operator needs trended, not buried in a CLI
    output an unattended loop never invokes.

    Best-effort and idempotent-safe: every fault is swallowed (a ledger
    write must NEVER break the continuous loop — the same discipline
    every sibling ``_append_*_skill_log`` follows). On any fault we
    still emit an honest row with ``status='error'
    verdict='INSUFFICIENT_DATA'`` so a gap in the trend is visible, not
    silent. Bounded growth: when the file exceeds
    2× ``PERSONA_SKILL_LOG_KEEP`` it is atomically rewritten via the
    tmp+``.replace`` idiom every sibling ledger uses.

    The ``signal_dark`` boolean mirrors the sibling ledgers' ``*_dark``
    flags (``pipeline_dark`` / ``calibrated_dark`` / ``sizing_dark`` /
    ``stop_dark`` / ``tp_dark`` / ``gate_dark``): True when zero
    personas have a stable IC (``n_personas == 0`` — every persona has
    fewer than ``MIN_OUTCOMES_PER_PERSONA`` rows, or the corpus has
    fewer than ``MIN_RECORDS`` aligned outcomes overall). False means
    the analyzer actually had per-persona data to bucket.

    Captures three flat top-line fields a consumer can query without
    parsing the nested ``personas`` list: ``top_persona`` /
    ``top_score_ic`` (the best-IC persona — the leader on the decision
    signal) and ``n_inverted`` (count of anti-predictive personas — the
    actionable red flag). The full ``personas`` list still ships as a
    nested field for forensics, mirroring the gate-arm ledger's
    ``arms`` precedent.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        outcomes_path = Path(outcomes_path)
        try:
            from paper_trader.ml.persona_skill import (
                persona_skill as _persona_skill,
                _load_outcomes as _load_persona_outcomes,
            )
            recs = _load_persona_outcomes(outcomes_path)
            rep = _persona_skill(recs)
        except Exception as exc:
            rep = {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "hint": f"persona_skill unavailable: {type(exc).__name__}",
                "n_records": 0, "n_personas": 0,
                "personas": [], "inverted_personas": [],
            }
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                   "n_records": 0, "n_personas": 0,
                   "personas": [], "inverted_personas": []}

        # ``signal_dark`` is True when no persona meets the stable-sample
        # bar — same logic as sibling ``*_dark`` flags.
        n_personas = int(rep.get("n_personas") or 0)
        signal_dark = (n_personas == 0)

        # Pull out the top persona (highest score_ic) as a flat top-line
        # field so a JSON consumer doesn't need to parse the nested
        # ``personas`` list to answer "who's leading on signal skill".
        # ``personas`` is already sorted by ``score_ic`` desc with
        # INSUFFICIENT entries sunk last (per ``persona_skill``).
        top_persona: str | None = None
        top_score_ic: float | None = None
        personas_list = rep.get("personas") or []
        for p in personas_list:
            v = p.get("verdict") if isinstance(p, dict) else None
            if v == "INSUFFICIENT":
                continue
            try:
                top_persona = str(p.get("persona") or "") or None
                top_score_ic = (None if p.get("score_ic") is None
                                else round(float(p.get("score_ic")), 4))
            except (TypeError, ValueError):
                top_persona = None
                top_score_ic = None
            break

        inverted_list = rep.get("inverted_personas") or []
        n_inverted = (len(inverted_list)
                      if isinstance(inverted_list, list) else 0)

        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": rep.get("verdict"),
            "n_records": rep.get("n_records"),
            "n_personas": n_personas,
            "top_persona": top_persona,
            "top_score_ic": top_score_ic,
            "n_inverted": n_inverted,
            "inverted_personas": inverted_list
                if isinstance(inverted_list, list) else [],
            "personas": personas_list
                if isinstance(personas_list, list) else [],
            "signal_dark": signal_dark,
        }
        PERSONA_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PERSONA_SKILL_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in
                     PERSONA_SKILL_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > PERSONA_SKILL_LOG_KEEP * 2:
                kept = lines[-PERSONA_SKILL_LOG_KEEP:]
                tmp = PERSONA_SKILL_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(PERSONA_SKILL_LOG)
        except Exception as e:
            print(f"[continuous] persona-skill-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] persona-skill-log append failed: {e}")
        return False


def _append_persona_regime_skill_log(cycle: int, win_start: date,
                                     win_end: date,
                                     outcomes_path: "Path | str | None" = None
                                     ) -> bool:
    """Append one structured row to the per-cycle (persona × regime)
    cross-tab decision-signal skill ledger.

    Answers, durably and per-cycle, the actionable question neither
    sibling ``persona_skill`` (persona-axis aggregate) nor
    ``regime_audit`` (regime-axis aggregate) can answer:
    *does THIS persona carry signal in THIS regime?* The aggregate
    diagnostics hide regime-conditional structure (a persona +0.25 IC in
    bull but -0.10 IC in bear shows ~0 in the aggregate); this ledger
    surfaces it.

    Reuses ``persona_regime_skill.analyze`` verbatim so the persisted
    verdict equals the read-only CLI's by construction — the same
    no-drift SSOT cross-check every sibling ``_append_*_skill_log``
    enforces. Captures three flat top-line fields a JSONL consumer can
    query without parsing the nested ``cells`` list:

      * ``best_cell_*`` — the best stable (persona, regime, score_ic, n)
        cell — the leader on signal skill in a specific regime.
      * ``worst_cell_*`` — the worst stable cell — surfaces an
        anti-predictive regime cell when one exists.
      * ``n_inverted_cells`` — count of cells where signal is
        anti-predictive (verdict==``INVERTED``). The single most
        directly operational state for the persona-suppression
        decision, mirroring the sibling persona ledger's
        ``n_inverted`` convention.

    The full ``cells`` / ``inverted_cells`` lists ship as nested fields
    for forensics, mirroring the gate-arm / persona ledgers'
    "flat summary + nested forensics" precedent.

    Best-effort and idempotent-safe: every fault is swallowed (a ledger
    write must NEVER break the continuous loop — the documented
    ``_append_scorer_skill_log`` / ``_post_discord`` / validation
    persister discipline). On any fault we still emit an honest row
    with ``status='error' verdict='INSUFFICIENT_DATA'`` so a gap in
    the trend is visible, not silent. Bounded growth: when the file
    exceeds 2× ``PERSONA_REGIME_SKILL_LOG_KEEP`` it is atomically
    rewritten via the tmp + ``.replace`` idiom every sibling ledger uses.

    The ``signal_dark`` boolean mirrors the sibling ledgers' ``*_dark``
    flags (``pipeline_dark`` / ``calibrated_dark`` / ``sizing_dark`` /
    ``stop_dark`` / ``tp_dark`` / ``gate_dark`` / ``signal_dark`` from
    persona_skill). True when the analyzer found ZERO stable cells
    (every (persona, regime) bucket below ``MIN_PER_CELL``); False when
    at least one cell has demonstrable data to evaluate.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        outcomes_path = Path(outcomes_path)
        try:
            from paper_trader.ml.persona_regime_skill import (
                analyze as _prs_analyze,
                _load_outcomes as _prs_load_outcomes,
            )
            recs = _prs_load_outcomes(outcomes_path)
            rep = _prs_analyze(recs)
        except Exception as exc:
            rep = {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "hint": f"persona_regime_skill unavailable: "
                        f"{type(exc).__name__}",
                "n_records": 0, "n_cells": 0, "n_stable_cells": 0,
                "cells": [], "inverted_cells": [],
                "best_cell": None, "worst_cell": None,
                "n_dropped_unknown_regime": 0,
            }
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                   "n_records": 0, "n_cells": 0, "n_stable_cells": 0,
                   "cells": [], "inverted_cells": [],
                   "best_cell": None, "worst_cell": None,
                   "n_dropped_unknown_regime": 0}

        # ``signal_dark`` is True when no (persona, regime) cell has
        # enough rows to be evaluable — mirrors sibling ``*_dark`` flags.
        n_stable = int(rep.get("n_stable_cells") or 0)
        signal_dark = (n_stable == 0)

        # Flatten ``best_cell`` / ``worst_cell`` to top-level columns so a
        # JSONL consumer can query without parsing the nested dict. Both
        # are None on INSUFFICIENT_DATA / no-stable-cell runs; degrade to
        # None safely in that case.
        best = rep.get("best_cell") or {}
        worst = rep.get("worst_cell") or {}
        inverted_list = rep.get("inverted_cells") or []
        n_inverted = (len(inverted_list)
                      if isinstance(inverted_list, list) else 0)

        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": rep.get("verdict"),
            "n_records": rep.get("n_records"),
            "n_cells": rep.get("n_cells"),
            "n_stable_cells": n_stable,
            "n_dropped_unknown_regime": rep.get("n_dropped_unknown_regime"),
            # Best stable cell — the operator-readable "who leads on
            # signal skill in which regime" answer at a glance.
            "best_persona": best.get("persona") if isinstance(best, dict) else None,
            "best_regime": best.get("regime") if isinstance(best, dict) else None,
            "best_score_ic": best.get("score_ic") if isinstance(best, dict) else None,
            "best_n": best.get("n") if isinstance(best, dict) else None,
            # Worst stable cell — surfaces the actionable red flag when
            # one cell is anti-predictive.
            "worst_persona": worst.get("persona") if isinstance(worst, dict) else None,
            "worst_regime": worst.get("regime") if isinstance(worst, dict) else None,
            "worst_score_ic": worst.get("score_ic") if isinstance(worst, dict) else None,
            "worst_n": worst.get("n") if isinstance(worst, dict) else None,
            # Inverted-cell summary (count) + full list for forensics.
            "n_inverted_cells": n_inverted,
            "inverted_cells": inverted_list
                if isinstance(inverted_list, list) else [],
            # Full cells list — forensic detail mirroring the persona
            # ledger's "personas" nested field. Bounded by the analyzer's
            # natural bucketing (≤ N_personas × 3 regimes = 30 cells max),
            # so size is fine inline.
            "cells": rep.get("cells")
                if isinstance(rep.get("cells"), list) else [],
            "signal_dark": signal_dark,
        }
        PERSONA_REGIME_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PERSONA_REGIME_SKILL_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in
                     PERSONA_REGIME_SKILL_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > PERSONA_REGIME_SKILL_LOG_KEEP * 2:
                kept = lines[-PERSONA_REGIME_SKILL_LOG_KEEP:]
                tmp = PERSONA_REGIME_SKILL_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(PERSONA_REGIME_SKILL_LOG)
        except Exception as e:
            print(f"[continuous] persona-regime-skill-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] persona-regime-skill-log append failed: {e}")
        return False


def _append_sector_skill_log(cycle: int, win_start: date, win_end: date,
                             outcomes_path: "Path | str | None" = None
                             ) -> bool:
    """Append one structured row to the per-cycle per-sector OOS skill ledger.

    Answers, durably and per-cycle, the decisive sector-level question the
    aggregate scorer ledger cannot: *is the scorer's rank skill carried
    uniformly across the watchlist universe, or by ONE fat sector (the live
    corpus is ~89% tech)?* The single most actionable state is
    ``HAS_INVERTED_SECTOR`` — a sector whose ``rank_ic ≤ -IC_GOOD``: the
    more confident the scorer is, the WORSE the realized 5d outcome — so
    gating on it is actively harmful. Until this wiring landed, the
    sector-level verdict was CLI-only via ``python3 -m
    paper_trader.ml.sector_skill``; an unattended operator could not trend
    per-sector signal health and ``HAS_INVERTED_SECTOR`` was invisible.

    Mirrors the sibling ``_append_persona_skill_log`` /
    ``_append_persona_regime_skill_log`` discipline exactly:

    * SSOT no-drift — reuses ``sector_skill.analyze`` (the same
      ``_load_outcomes`` + ``split_outcomes_temporal`` path the CLI uses),
      so the persisted verdict equals the read-only CLI's by construction.
    * Honest gap rows — on any fault we still emit a row with
      ``status='error' verdict='INSUFFICIENT_DATA' signal_dark=True`` so a
      gap in the trend is visible, not silent.
    * Bounded growth — when the file exceeds
      2× ``SECTOR_SKILL_LOG_KEEP`` it is atomically rewritten via the
      tmp+``.replace`` idiom.
    * Never raises — a ledger write must NEVER break the continuous loop.

    The ``signal_dark`` boolean mirrors every sibling ``*_dark`` flag
    (``signal_dark`` / ``gate_dark`` / ``stop_dark`` / etc.): True when
    the analyzer could not produce a stable per-sector reading — either
    the corpus had < ``MIN_RECORDS`` aligned OOS outcomes
    (``INSUFFICIENT_DATA``) OR every sector that surfaced is ``SPARSE``
    (n_oos < ``MIN_OUTCOMES_PER_SECTOR``). False means at least one
    sector reached a non-SPARSE verdict the analyzer could classify.

    Captures three flat top-line fields a JSON consumer can query without
    parsing the nested ``sectors`` list:
      * ``top_sector`` / ``top_rank_ic`` — the best non-SPARSE sector
        (the leader on the rank-IC bar — the headline number a quant
        skims first).
      * ``n_inverted`` — count of anti-predictive sectors (the actionable
        red flag).
    The full ``sectors`` list still ships as a nested field for forensics
    (≤ 7 sectors max — N_SECTORS-bounded), and the ``inverted_sectors``
    name list ships intact so a grep-trend on a specific sector still
    works.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        if outcomes_path is None:
            outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
        outcomes_path = Path(outcomes_path)
        try:
            from paper_trader.ml.sector_skill import analyze as _sector_skill_analyze
            rep = _sector_skill_analyze(outcomes_path=outcomes_path)
        except Exception as exc:
            rep = {
                "status": "error", "verdict": "INSUFFICIENT_DATA",
                "hint": f"sector_skill unavailable: {type(exc).__name__}",
                "n_train": 0, "n_oos": 0,
                "concentrated_sector": None,
                "sectors": [], "inverted_sectors": [],
            }
        if not isinstance(rep, dict):
            rep = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                   "n_train": 0, "n_oos": 0,
                   "concentrated_sector": None,
                   "sectors": [], "inverted_sectors": []}

        sectors_list = rep.get("sectors") or []
        if not isinstance(sectors_list, list):
            sectors_list = []
        # ``signal_dark`` mirrors sibling ``*_dark`` flags: True when no
        # sector reached a stable (non-SPARSE) sample. The analyzer
        # ``sectors_out`` is sorted (SPARSE sinks to the bottom), so the
        # presence of ANY non-SPARSE entry means we have a sector to read.
        non_sparse_n = sum(
            1 for s in sectors_list
            if isinstance(s, dict) and s.get("verdict") != "SPARSE"
        )
        signal_dark = (non_sparse_n == 0)

        # Top sector — the leader on rank_ic, skipping SPARSE entries the
        # same way ``_append_persona_skill_log`` skips ``INSUFFICIENT``.
        # ``sectors`` is already sorted by ``rank_ic`` desc with SPARSE
        # sunk last (per ``sector_skill``).
        top_sector: str | None = None
        top_rank_ic: float | None = None
        for s in sectors_list:
            if not isinstance(s, dict):
                continue
            if s.get("verdict") == "SPARSE":
                continue
            try:
                top_sector = str(s.get("sector") or "") or None
                top_rank_ic = (None if s.get("rank_ic") is None
                               else round(float(s.get("rank_ic")), 4))
            except (TypeError, ValueError):
                top_sector = None
                top_rank_ic = None
            break

        inverted_list = rep.get("inverted_sectors") or []
        if not isinstance(inverted_list, list):
            inverted_list = []
        n_inverted = len(inverted_list)

        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "status": rep.get("status"),
            "verdict": rep.get("verdict"),
            "n_train": rep.get("n_train"),
            "n_oos": rep.get("n_oos"),
            "concentrated_sector": rep.get("concentrated_sector"),
            "top_sector": top_sector,
            "top_rank_ic": top_rank_ic,
            "n_inverted": n_inverted,
            "inverted_sectors": inverted_list,
            "sectors": sectors_list,
            "signal_dark": signal_dark,
        }
        SECTOR_SKILL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SECTOR_SKILL_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in
                     SECTOR_SKILL_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > SECTOR_SKILL_LOG_KEEP * 2:
                kept = lines[-SECTOR_SKILL_LOG_KEEP:]
                tmp = SECTOR_SKILL_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(SECTOR_SKILL_LOG)
        except Exception as e:
            print(f"[continuous] sector-skill-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] sector-skill-log append failed: {e}")
        return False


def _append_dead_feature_audit_log(cycle: int, win_start: date,
                                    win_end: date) -> bool:
    """Append one structured row to the per-cycle dead-feature-audit ledger.

    Calls ``dead_feature_audit.audit_dead_features`` against the just-
    retrained deployed pickle and persists the verdict (OK / HAS_DEAD /
    NOT_TRAINED / SHAPE_MISMATCH / UNKNOWN_MODEL / ERROR), the count of
    dead features, and (when HAS_DEAD) the list of feature names whose
    first-layer ``mean |w| ≤ DEAD_EPS``. The flat ``n_features_dead`` and
    list-shaped ``dead_features`` mirror the sibling ledgers' two-tier
    "summary + forensics" pattern (``gate_arm_skill_log`` /
    ``persona_skill_log``).

    Why this matters: the pass-#35 finding was that 3 of 20 input slots
    were dead-trained on constant zero for an unknown number of cycles —
    invisible to every existing diagnostic. With this ledger, the moment a
    feature plumbing regression happens (or a fresh build_features slot is
    added without backfilling the outcomes JSONL) the verdict flips to
    ``HAS_DEAD`` on the NEXT retrain and an operator sees it in the trend.

    Best-effort and idempotent-safe: every fault is swallowed (a ledger
    write must NEVER break the continuous loop — same discipline every
    sibling ``_append_*_skill_log`` follows). Bounded growth: when the file
    exceeds 2× ``DEAD_FEATURE_AUDIT_LOG_KEEP`` it is atomically rewritten
    via the tmp+``.replace`` idiom every sibling ledger uses.

    Returns True on a successful append, False on any handled fault.
    """
    try:
        try:
            from paper_trader.ml.dead_feature_audit import audit_dead_features
            rep = audit_dead_features()
        except Exception as exc:
            rep = {
                "verdict": "ERROR",
                "method": None,
                "n_train": 0,
                "n_features_total": None,
                "n_features_dead": 0,
                "dead_features": [],
                "error": f"audit_dead_features unavailable: "
                         f"{type(exc).__name__}",
            }
        if not isinstance(rep, dict):
            rep = {"verdict": "ERROR", "method": None,
                   "n_features_dead": 0, "dead_features": []}
        row = {
            "cycle": cycle,
            "timestamp": _now(),
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "verdict": rep.get("verdict"),
            "method": rep.get("method"),
            "n_train": rep.get("n_train"),
            "n_features_total": rep.get("n_features_total"),
            "n_features_dead": rep.get("n_features_dead"),
            # Flat boolean for trend cuts: True ⇒ at least one input
            # slot of the deployed model was dead-trained on constant
            # zero. Mirrors the sibling ledgers' ``*_dark`` flag
            # convention (``signal_dark`` / ``stop_dark`` / etc.).
            "has_dead": (rep.get("verdict") == "HAS_DEAD"),
            "dead_features": (rep.get("dead_features") or []),
            "eps": rep.get("eps"),
        }
        if rep.get("error"):
            row["error"] = rep["error"]
        DEAD_FEATURE_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with DEAD_FEATURE_AUDIT_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        # Bounded growth — only pay the rewrite when well past the cap.
        try:
            lines = [ln for ln in
                     DEAD_FEATURE_AUDIT_LOG.read_text().splitlines()
                     if ln.strip()]
            if len(lines) > DEAD_FEATURE_AUDIT_LOG_KEEP * 2:
                kept = lines[-DEAD_FEATURE_AUDIT_LOG_KEEP:]
                tmp = DEAD_FEATURE_AUDIT_LOG.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(kept) + "\n")
                tmp.replace(DEAD_FEATURE_AUDIT_LOG)
        except Exception as e:
            print(f"[continuous] dead-feature-audit-log trim failed: {e}")
        return True
    except Exception as e:
        print(f"[continuous] dead-feature-audit-log append failed: {e}")
        return False


def _parse_published_date(published) -> date | None:
    """Parse a `published` value (ISO or RFC822) into a date; None if unparseable."""
    if not published:
        return None
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(published)
        if dt is not None:
            return dt.date()
    except Exception:
        pass
    try:
        return date.fromisoformat(str(published)[:10])
    except Exception:
        return None


def _query_news_context(ticker: str, sim_date_str: str, n: int = 4) -> list[str]:
    """Fetch recent article titles from digital-intern DB near sim_date for ticker."""
    DB = ROOT.parent / "digital-intern" / "data" / "articles.db"
    if not DB.exists():
        return []
    try:
        d = date.fromisoformat(sim_date_str)
    except ValueError:
        return []
    lo = d - timedelta(days=3)
    hi = d + timedelta(days=1)
    conn = None
    try:
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
        # `published` in articles.db is stored in mixed formats — ISO for some
        # sources, RFC822 ("Wed, 14 May 2026 ...") for RSS. A SQL
        # `published BETWEEN` range filter silently drops every RFC822 row
        # (their leading weekday letter lex-sorts after any ISO date string),
        # so it would have excluded most live news. Fetch a generous candidate
        # set ordered by ai_score and apply the date window in Python after
        # parsing each timestamp robustly.
        rows = conn.execute(
            "SELECT title, published FROM articles "
            "WHERE (title LIKE ? OR title LIKE ?) "
            "AND (url IS NULL OR url NOT LIKE 'backtest://%') "
            "AND (source IS NULL OR (source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%')) "
            "ORDER BY ai_score DESC LIMIT ?",
            (f"%{ticker}%", f"%{ticker.lower()}%", max(n * 20, 40)),
        ).fetchall()
    except Exception:
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    out: list[str] = []
    for title, published in rows:
        if not title:
            continue
        pub_d = _parse_published_date(published)
        # Drop rows that parse to a date outside the window; keep unparseable
        # ones (can't prove they leak) so the context isn't emptied entirely.
        if pub_d is not None and not (lo <= pub_d <= hi):
            continue
        out.append(title)
        if len(out) >= n:
            break
    return out


def _opus_annotate(engine: "BacktestEngine", top_runs: list[BacktestRun],
                   cycle: int, outcome_records: list[dict] | None = None) -> int:
    """Ask Opus 4.7 to annotate ALL decisions (BUY, SELL, HOLD) in the winner run.

    Enhanced over previous version:
    - Covers every decision, not just trades, so HOLDs can also be critiqued
    - Attaches actual 5-day forward returns so Opus sees what happened after each call
    - Pulls relevant scraped news articles from articles DB near each decision date
    - Outcome records (from _compute_decision_outcomes) included as context when available

    Annotations are appended to WINNER_JSONL. Returns number of records written.
    """
    if not shutil.which("claude"):
        print("[opus_annotate] claude CLI not found — skipping annotation")
        return 0
    if not top_runs:
        return 0

    winner = top_runs[0]
    try:
        # This runs in a background thread that overlaps _trim_history and the
        # next cycle's run threads — all writing through the SAME sqlite3
        # connection. Concurrent use of one connection across threads corrupts
        # cursor state, so serialise this read through the store lock.
        with engine.store._lock:
            rows = engine.store.conn.execute(
                "SELECT action, ticker, sim_date, reasoning, qty, total_value "
                "FROM backtest_decisions WHERE run_id=? ORDER BY sim_date",
                (winner.run_id,),
            ).fetchall()
    except Exception as e:
        print(f"[opus_annotate] DB read failed: {e}")
        return 0

    # Build outcome lookup: (sim_date, ticker) -> forward_return_5d
    # Use `.get()` everywhere: a missing key from a future record-shape change
    # would otherwise KeyError out of this background annotation thread (no
    # outer catch wraps lines 2074-2105) and silently drop the entire
    # cycle's Opus annotations. A None lookup result is already filtered
    # downstream (`if fwd is not None`), so .get() degrades gracefully.
    outcome_lookup: dict[tuple, float] = {}
    for o in (outcome_records or []):
        if o.get("run_id") == winner.run_id:
            sim_date_o = o.get("sim_date")
            ticker_o = o.get("ticker")
            fwd_o = o.get("forward_return_5d")
            if sim_date_o and ticker_o and fwd_o is not None:
                outcome_lookup[(sim_date_o, ticker_o)] = fwd_o

    # Build enriched decision log — all actions, not just BUY/SELL
    decision_lines = []
    for r in rows:
        action = r["action"] or "HOLD"
        ticker = r["ticker"] or ""
        sim_date_str = r["sim_date"] or ""
        fwd_str = ""
        if ticker and sim_date_str:
            fwd = outcome_lookup.get((sim_date_str, ticker))
            if fwd is not None:
                fwd_str = f" →5d={fwd:+.1f}%"
            # Fetch scraped news snippets for this ticker/date
            news = _query_news_context(ticker, sim_date_str, n=2)
            news_str = " | NEWS: " + "; ".join(news[:2]) if news else ""
        else:
            news_str = ""
        qty_str = f" qty={r['qty']}" if r["qty"] else ""
        val_str = f" portfolio=${r['total_value']:.0f}" if r["total_value"] else ""
        reasoning_short = str(r["reasoning"] or "")[:100]
        decision_lines.append(
            f"  {sim_date_str} {action} {ticker}{qty_str}{val_str}{fwd_str}"
            f" | {reasoning_short}{news_str}"
        )

    if not decision_lines:
        return 0

    other_returns = " / ".join(f"run{r.run_id}={r.total_return_pct:+.1f}%" for r in top_runs[1:])
    prompt = f"""You are a quantitative trading analyst reviewing a backtest run for ML training purposes.

Backtest run #{winner.run_id} achieved {winner.total_return_pct:+.2f}% return over a 1-year simulation
using ML article sentiment + RSI/MACD/momentum signals. No live Claude calls were used — decisions
are pure quantitative signals. Other top runs this cycle: {other_returns or "none"}

FULL DECISION LOG (including HOLDs):
Format: date ACTION TICKER qty portfolio →5d_actual_return | reasoning | NEWS_CONTEXT
{chr(10).join(decision_lines[:60])}

Your task:
1. For EVERY decision (BUY, SELL, and HOLD), assign quality: GOOD / NEUTRAL / BAD
   - GOOD: the decision led to profit or correctly avoided loss (5d return confirms it)
   - BAD: the decision lost money or missed a clear profitable opportunity
   - NEUTRAL: outcome was mixed or the 5d return was near zero
   - For HOLDs: was holding the right call? Did a missed trade (5d return > +2%) mean BAD HOLD?
2. For BAD decisions: specify what signal should have triggered differently
3. For GOOD decisions: identify the specific signal that made it right
4. Provide an OVERALL LESSON as a concise trading rule derived from this run's outcomes

Respond as JSON with this schema (no markdown fences):
{{
  "trade_labels": [
    {{
      "sim_date": "YYYY-MM-DD",
      "action": "BUY/SELL/HOLD",
      "ticker": "...",
      "quality": "GOOD/NEUTRAL/BAD",
      "rationale": "...",
      "forward_return_5d": <number or null>,
      "signal_fix": "what signal should have changed this decision (if BAD or missed opportunity)"
    }}
  ],
  "overall_lesson": "...",
  "key_patterns": ["pattern1", "pattern2"],
  "improvement_suggestions": ["specific change to ML scoring or thresholds"]
}}"""

    try:
        r = subprocess.run(
            ["claude", "--model", "claude-sonnet-4-6", "--print",
             "--permission-mode", "bypassPermissions"],
            input=prompt, capture_output=True, text=True, timeout=240,
            env={**os.environ, "HOME": "/home/zeph"},
        )
    except subprocess.TimeoutExpired:
        print("[opus_annotate] timeout")
        return 0
    except Exception as e:
        print(f"[opus_annotate] subprocess error: {e}")
        return 0

    if r.returncode != 0 or not r.stdout.strip():
        print(f"[opus_annotate] claude rc={r.returncode} stderr={r.stderr.strip()[:200]!r}")
        return 0

    raw = r.stdout.strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        print("[opus_annotate] no JSON in response")
        return 0
    try:
        annotation = json.loads(m.group(0))
    except Exception as e:
        print(f"[opus_annotate] JSON parse error: {e}")
        return 0

    written = 0
    WINNER_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with WINNER_JSONL.open("a") as fh:
        # Opus occasionally emits JSON null for list/string fields — dict.get
        # returns None in that case, so use `or` to fall back to safe defaults.
        lesson = annotation.get("overall_lesson") or ""
        patterns = annotation.get("key_patterns") or []
        suggestions = annotation.get("improvement_suggestions") or []
        if lesson:
            fh.write(json.dumps({
                "cycle": cycle,
                "run_id": winner.run_id,
                "type": "opus_lesson",
                "title": f"Lesson run {winner.run_id} ({winner.total_return_pct:+.1f}%): {lesson[:120]}",
                "source": f"opus_annotation_cycle_{cycle}",
                "ai_score": 5.0,
                "urgency": 1,
                "label": "LESSON",
                "return_pct": winner.total_return_pct,
                "reasoning": lesson,
                "key_patterns": patterns,
                "improvement_suggestions": suggestions,
                "weight": 1.0,
            }) + "\n")
            written += 1

        quality_score = {"GOOD": 5.0, "NEUTRAL": 2.5, "BAD": 0.5}
        for tl in (annotation.get("trade_labels") or []):
            q = tl.get("quality", "NEUTRAL")
            action = tl.get("action", "HOLD")
            fh.write(json.dumps({
                "cycle": cycle,
                "run_id": winner.run_id,
                "type": "opus_trade_label",
                "title": f"{action} {tl.get('ticker','')} {tl.get('sim_date','')} [{q}]",
                "source": f"opus_annotation_cycle_{cycle}",
                "ai_score": quality_score.get(q, 2.5),
                "urgency": 1 if q == "GOOD" else 0,
                "label": action,
                "ticker": tl.get("ticker", ""),
                "sim_date": tl.get("sim_date", ""),
                "reasoning": tl.get("rationale", ""),
                "signal_fix": tl.get("signal_fix", ""),
                "forward_return_5d": tl.get("forward_return_5d"),
                "return_pct": winner.total_return_pct,
                "quality": q,
                "weight": 1.0 if q == "GOOD" else (0.5 if q == "NEUTRAL" else 0.1),
            }) + "\n")
            written += 1

    print(f"[opus_annotate] wrote {written} annotation records for run {winner.run_id} "
          f"({len(decision_lines)} decisions reviewed)")
    return written


def _inject_and_train() -> str:
    """Inject winner JSONL into article store then retrain. Returns short status string."""
    import hashlib
    import zlib

    DB_PATH = DIGITAL_INTERN_ARTICLES_DB

    def _compress(text: str) -> bytes:
        return zlib.compress(text.encode("utf-8", errors="replace"), level=6)

    def _aid(url: str, title: str) -> str:
        return hashlib.sha256(f"{url}||{title}".encode()).hexdigest()[:20]

    if not WINNER_JSONL.exists():
        return "no jsonl"

    # Cap the JSONL read to the most recent records — older ones are already
    # in articles.db (INSERT OR IGNORE de-dups by id), so re-reading them every
    # cycle wastes memory and IO as the file grows without bound.
    _MAX_INJECT_RECORDS = 10000
    # winner_training.jsonl accumulates forever; read_text() would pull the whole
    # (eventually multi-hundred-MB) file into memory every cycle. Stream it line
    # by line through a bounded deque so peak memory is capped at the tail we use.
    try:
        from collections import deque
        with WINNER_JSONL.open("r") as _fh:
            recent = list(deque((ln for ln in _fh if ln.strip()),
                                maxlen=_MAX_INJECT_RECORDS))
    except Exception as e:
        return f"jsonl read err: {e}"
    # Per-line parse so a single corrupt line doesn't drop the whole batch
    records: list[dict] = []
    for l in recent:
        try:
            records.append(json.loads(l))
        except Exception:
            pass

    now = datetime.now(timezone.utc).isoformat()

    # Build the INSERT param tuples up front — a PURE pass with no DB handle
    # so it cannot fail on a lock and need not be replayed on a retry.
    prepared: list[tuple] = []
    for rec in records:
        # `.get(k, default)` only substitutes the default when the key is
        # ABSENT — an explicit JSON `null` value still returns None, and
        # `float(None)` raises TypeError. A single such line in
        # winner_training.jsonl would abort the whole injection batch via
        # the outer `except` (returning "inject err: …"), so ArticleNet
        # never retrains that cycle. `or` coerces None/0/"" to the safe
        # default, matching the hardening idiom already used in
        # backtest._ml_decide and _opus_annotate.
        ai = float(rec.get("ai_score") or 0.0)
        w = float(rec.get("weight") or 1.0)
        eff = min(10.0, ai * w)
        title = rec.get("title", "")
        ticker = rec.get("ticker", "")
        reasoning = rec.get("reasoning", "")
        sim_date = rec.get("sim_date", "")
        label = rec.get("label", "")
        run_id = rec.get("run_id", 0)
        if not title:
            continue
        url = f"backtest://run_{run_id}/{sim_date}/{label}/{ticker}"
        aid = _aid(url, title)
        full_text = f"[{ticker}] {title}. {reasoning}"
        prepared.append(
            (aid, url, title, f"backtest_run_{run_id}", sim_date or now[:10],
             eff, eff, 0, now, rec.get("cycle", 0), _compress(full_text)))

    # digital-intern's daemon is a heavy concurrent writer to this same
    # ~1.4 GB articles.db. Observed live (continuous.log: 7 cycles): the
    # daemon held the write lock longer than sqlite3 connect `timeout=`, so
    # `execute(INSERT…)`/`commit()` raised `OperationalError: database is
    # locked` and the WHOLE ArticleNet feedback batch (CLAUDE.md §5 step 5)
    # was dropped for that cycle with NO retry. Retry connect→write→commit a
    # few times with backoff on a *transient* lock only; INSERT OR IGNORE
    # makes the replay idempotent (an uncommitted partial attempt is rolled
    # back on close and `inserted` is recomputed from scratch each attempt).
    # A non-lock OperationalError or any other exception falls through to
    # the original single-shot "inject err:" path immediately — no pointless
    # backoff on a real bug.
    _LOCK_RETRY_SLEEPS = (3.0, 8.0, 15.0)  # one initial try + 3 retries
    inserted = 0
    last_lock_err: Exception | None = None
    for _attempt in range(len(_LOCK_RETRY_SLEEPS) + 1):
        aconn = None
        try:
            aconn = sqlite3.connect(DB_PATH, timeout=15)
            # Explicit busy handler — defensive belt-and-braces over the
            # connect `timeout=` (a write-lock wait on a WAL db is governed
            # by busy_timeout; some builds honour only the PRAGMA form).
            aconn.execute("PRAGMA busy_timeout=15000")
            inserted = 0
            for row in prepared:
                aconn.execute(
                    "INSERT OR IGNORE INTO articles "
                    "(id,url,title,source,published,kw_score,ai_score,"
                    "urgency,first_seen,cycle,full_text) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    row,
                )
                if aconn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
            aconn.commit()
            last_lock_err = None
            break
        except sqlite3.OperationalError as e:
            _m = str(e).lower()
            if "locked" not in _m and "busy" not in _m:
                return f"inject err: {e}"
            last_lock_err = e
            if aconn is not None:
                try:
                    aconn.rollback()
                except Exception:
                    pass
        except Exception as e:
            return f"inject err: {e}"
        finally:
            if aconn is not None:
                try:
                    aconn.close()
                except Exception:
                    pass
        # Sleep+retry only while attempts remain; the final iteration falls
        # straight through to the post-loop lock-error return (no dead wait).
        if _attempt < len(_LOCK_RETRY_SLEEPS):
            time.sleep(_LOCK_RETRY_SLEEPS[_attempt])
    else:
        return (f"inject err: database locked after "
                f"{len(_LOCK_RETRY_SLEEPS) + 1} attempts ({last_lock_err})")

    # Now trigger actual training
    try:
        r = subprocess.run(
            ["python3", "-c",
             "import sys; sys.path.insert(0,'.'); "
             "from storage.article_store import ArticleStore; "
             "from ml.trainer import train; "
             "s=ArticleStore(); res=train(s,force=True); "
             "print(f\"trainer n={res.get('n',0)} loss={res.get('final_loss',0):.4f} "
             "val={res.get('val_loss',0):.4f}\")"],
            cwd="/home/zeph/digital-intern",
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            out = (r.stdout or "").strip().splitlines()
            return f"injected {inserted} new | {out[-1] if out else 'ok'}"
        return f"trainer rc={r.returncode} injected={inserted}"
    except subprocess.TimeoutExpired:
        return f"trainer timeout (injected {inserted})"
    except Exception as e:
        return f"trainer exc: {type(e).__name__}"


def _try_train_ml() -> str:
    return _inject_and_train()


def _llm_annotate_outcomes(
    engine,
    winner: "BacktestRun",
    loser: "BacktestRun",
    outcome_records: list[dict],
    cycle: int,
) -> list[dict]:
    """Call LLM to annotate best/worst run trades with quality labels.

    Endorsed trades get llm_quality_label=+1 (3x training weight).
    Condemned trades get llm_quality_label=-1 (0.1x training weight).
    Unlabeled records get llm_quality_label=0 (1x weight).

    Returns outcome_records with llm_quality_label filled in.
    """
    try:
        import anthropic
    except ImportError:
        return outcome_records

    for r in outcome_records:
        r.setdefault("llm_quality_label", 0)

    def _summarize_run(run, label: str, max_trades: int = 5) -> str:
        trades = []
        run_records = [r for r in outcome_records if r.get("run_id") == run.run_id][:max_trades]
        for r in run_records:
            # `r.get('forward_return_5d', 0)` only defaults on a MISSING
            # key — an explicit None value (a malformed record, or a future
            # record-shape change) hits `None:.1f` → TypeError, which is
            # caught by the outer except at line 2469 and silently drops
            # the WHOLE cycle's LLM annotations for that batch. `or 0`
            # coerces None to a safe numeric. Same hardening idiom used
            # in `_inject_and_train` for ai_score / weight.
            ml_score_v = r.get("ml_score") or 0
            fwd_v = r.get("forward_return_5d") or 0
            trades.append(
                f"  {r.get('ticker','?')} {r.get('action','BUY')}: "
                f"ml_score={ml_score_v:.1f}, "
                f"rsi={r.get('rsi') or '?'}, "
                f"5d_return={fwd_v:.1f}%"
            )
        return f"{label} (total return: {run.total_return_pct:.1f}%):\n" + "\n".join(trades or ["  (no trades)"])

    winner_summary = _summarize_run(winner, "BEST RUN")
    loser_summary = _summarize_run(loser, "WORST RUN")

    prompt = f"""You are reviewing trades from a paper-trading backtest system.

{winner_summary}

{loser_summary}

For each trade listed above, output one line:
TICKER ACTION: ENDORSE or CONDEMN [one sentence reason based on whether this trade reflects good news analysis and momentum alignment]

Be concise. Only output the labeled lines, no intro text."""

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        annotation_text = resp.content[0].text.strip()
        print(f"[continuous] LLM annotation cycle {cycle}:\n{annotation_text}")

        # The LLM only reviewed trades from the best and worst runs (see the
        # prompt above). Restrict label application to those two run_ids —
        # matching on (ticker, action) alone would leak a verdict derived from
        # one run's trade onto identically-named trades in the three unreviewed
        # middle runs, corrupting their training sample weights.
        allowed_run_ids = {winner.run_id, loser.run_id}
        for line in annotation_text.splitlines():
            # [\w\-]* (not +) so single-letter tickers like V are not dropped.
            m = re.match(r"(\w[\w\-]*)\s+(BUY|SELL|HOLD)[:\s]+(ENDORSE|CONDEMN)", line.upper())
            if not m:
                continue
            ticker, action, verdict = m.group(1), m.group(2), m.group(3)
            label = 1 if verdict == "ENDORSE" else -1
            for r in outcome_records:
                if r.get("run_id") not in allowed_run_ids:
                    continue
                if (str(r.get("ticker", "")).upper() == ticker and
                        str(r.get("action", "")).upper() == action):
                    r["llm_quality_label"] = label

    except Exception as e:
        print(f"[continuous] LLM annotation failed: {e}")

    return outcome_records


def _post_discord(message: str) -> None:
    """Best-effort Discord post via openclaw. Silent on failure — never raise."""
    if not shutil.which("openclaw"):
        return
    try:
        subprocess.run(
            ["openclaw", "message", "send",
             "--channel", "discord",
             "--target", DISCORD_CHANNEL,
             "--message", message],
            capture_output=True, timeout=20,
        )
    except Exception as e:
        print(f"[discord] post failed: {e}")


def _run_validation_async(engine, cycle: int, win_start: date, win_end: date,
                          articles_db: str | None) -> None:
    """Run the full validation suite (label audit + permutation test) and
    persist results to ``data/validation_results.json``.

    Designed to be invoked from a background daemon thread — a permutation
    test runs ``VALIDATION_PERMUTATIONS`` full backtests serially, which can
    take 20+ minutes. Running this synchronously would block the next
    backtest cycle indefinitely.

    The function never raises — every step has a best-effort try/except so
    a validation failure cannot kill the loop.
    """
    print(f"[validation] cycle {cycle} starting (this runs in background)")
    out: dict = {
        "cycle": cycle,
        "timestamp": _now(),
        "window": f"{win_start}→{win_end}",
        "permutation_test": None,
        "label_audit": None,
    }

    # 1. Label contamination audit — fast, just SQL.
    try:
        from paper_trader.validation import audit_label_contamination
        if articles_db:
            audit = audit_label_contamination(articles_db, win_start, win_end)
            out["label_audit"] = audit
            # Only alert on real Claude-label hindsight risk. A purely
            # RETROACTIVE_COLLECTION verdict is architectural (historical
            # articles are always scraped long after publication) and uses
            # ML/heuristic scores, not Claude labels — alerting on it would
            # spam every historical window with a false alarm.
            if audit.get("verdict") == "HIGH_CONTAMINATION":
                llm_n = audit.get("llm_contaminated_count", 0)
                total_n = audit.get("total_articles", 0)
                _post_discord(
                    f"WARN: high label contamination — "
                    f"{llm_n}/{total_n} Claude-labeled articles collected "
                    f"with hindsight in window {win_start}→{win_end}. "
                    f"Backtest returns may be inflated."
                )
    except Exception as e:
        out["label_audit"] = {"error": str(e)}

    # 2. Permutation test — slow.
    try:
        from paper_trader.validation import run_permutation_test
        import tempfile
        with tempfile.TemporaryDirectory(prefix="perm_cycle_") as tmp:
            perm = run_permutation_test(
                engine,
                seed=cycle,
                n_permutations=VALIDATION_PERMUTATIONS,
                isolated_db_path=Path(tmp) / "perm.db",
            )
        # Policy override: if the label audit shows the window's articles
        # were >=50% collected with hindsight, the permutation test ran on
        # contaminated inputs and its verdict (however significant) is not
        # trustworthy — the "signal" may just be future leakage. Keep this
        # in the policy layer (run_permutation_test stays pure math).
        try:
            _la = out.get("label_audit") or {}
            _cr = _la.get("contamination_rate")
            if (
                isinstance(perm, dict)
                and _cr is not None
                and float(_cr) >= 0.5
            ):
                perm["verdict_raw"] = perm.get("verdict")
                perm["verdict"] = (
                    "CONTAMINATED_DATA — permutation result invalid"
                )
                perm["contamination_rate"] = float(_cr)
        except Exception:
            pass
        out["permutation_test"] = perm
        v = perm.get("verdict")
        if v == "WORSE_THAN_RANDOM":
            _post_discord(
                f"ALERT: permutation test cycle {cycle} — strategy WORSE than "
                f"random signal ordering "
                f"(p={perm.get('p_value', 0):.2f}, "
                f"z={perm.get('z_score', 0):.2f}). "
                f"Signals may not carry real predictive value."
            )
        elif v == "SIGNIFICANT":
            _post_discord(
                f"OK: permutation test cycle {cycle} PASSED — "
                f"p={perm.get('p_value', 0):.3f}, "
                f"z={perm.get('z_score', 0):.1f}. "
                f"Signal time-ordering carries real value."
            )
    except Exception as e:
        out["permutation_test"] = {"error": str(e)}

    # 3. Persist (capped tail).
    try:
        VALIDATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if VALIDATION_RESULTS_PATH.exists():
            try:
                existing = json.loads(VALIDATION_RESULTS_PATH.read_text())
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        existing.append(out)
        existing = existing[-VALIDATION_RESULTS_KEEP:]
        # Atomic write — torn JSON would break the dashboard's /api/validation.
        tmp_p = VALIDATION_RESULTS_PATH.with_suffix(".json.tmp")
        tmp_p.write_text(json.dumps(existing, indent=2))
        tmp_p.replace(VALIDATION_RESULTS_PATH)
        print(f"[validation] cycle {cycle} done — wrote {len(existing)} entries")
    except Exception as e:
        print(f"[validation] persist failed: {e}")


_STOP = False


def _handle_sig(_signum, _frame) -> None:
    global _STOP
    _STOP = True
    print(f"\n[continuous] {_now()} signal received — stopping after current cycle")


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    print(f"[continuous] {_now()} starting ENSEMBLE-COMMITTEE loop "
          f"({RUNS_PER_CYCLE} runs/cycle, keep last {KEEP_LAST_RUNS}, "
          f"cooldown {COOLDOWN_SECONDS}s, variable {MIN_WINDOW_YEARS}–{MAX_WINDOW_YEARS}yr "
          f"windows in {EARLIEST_WINDOW_START}–present)")

    # A fresh process means any pre-existing 'running' row is orphaned from a
    # dead prior process (hard-killed before finalize_run / the failed-marker).
    # Sweep them once at startup so the dashboard stops rendering dead runs
    # as in-flight (CLAUDE.md §11). Runs before any new run is launched —
    # single-threaded, no race with this process's own runs.
    try:
        reaped = _reap_orphaned_runs()
        if reaped:
            print(f"[continuous] reaped {reaped} orphaned 'running' "
                  f"run(s) → failed")
    except Exception as e:
        print(f"[continuous] reap dispatch failed: {e}")

    cycle = 0
    while not _STOP:
        cycle += 1
        # Each cycle picks its own random window. Engine is recreated because
        # PriceCache and the per-window volume cache are window-keyed; reusing
        # the previous engine would silently mix cache state from a different
        # date range.
        cycle_seed = int(time.time()) ^ (cycle * 2654435761) & 0xFFFFFFFF
        win_start, win_end = _pick_window(cycle_seed)
        win_years = (win_end - win_start).days / 365.0
        print(f"\n[continuous] {_now()} ─── cycle {cycle} window: "
              f"{win_start} → {win_end} ({win_years:.1f}yr, seed={cycle_seed}) ───")

        try:
            engine = BacktestEngine(start=win_start, end=win_end)
        except Exception as e:
            print(f"[continuous] engine init failed for {win_start}→{win_end}: {e}")
            traceback.print_exc()
            # Sleep briefly then move to next cycle; a yfinance hiccup shouldn't
            # kill the loop.
            time.sleep(30)
            continue

        # Optional pre-warmer for historical news. Background by default so
        # backtests proceed on quant signals while news fills in.
        # `tickers=None` lets the collector pick its own narrow SEC ticker set
        # (~17 names). Passing the full 117-ticker watchlist would issue an
        # SEC request for each on every cycle, wasting rate budget on names
        # that aren't tracked by the signal pipeline anyway.
        try:
            from paper_trader.historical_collector import prewarm_window
            prewarm_window(win_start, win_end, tickers=None, background=True)
        except Exception as e:
            # Pre-warmer is best-effort — failure must not stop a cycle.
            print(f"[continuous] prewarm dispatch failed: {e}")

        # Reap orphaned 'running' rows every cycle, not only at startup. A run
        # thread hard-killed mid-cycle (OOM / SIGKILL) never reaches
        # finalize_run OR run_all's caught-exception 'failed' marker, so its
        # row stays 'running' forever — observed live: 15 rows stuck 'running'
        # for 35h while ~170 newer runs completed (the exact "dashboard shows
        # running forever" symptom, CLAUDE.md §11). The startup-only reap never
        # fires for a long-lived loop. `_reap_orphaned_runs` is idempotent,
        # best-effort, and 6h-age-guarded — no real run exceeds minutes even on
        # a 10-yr window, so this can never touch a live run from the current
        # or previous cycle (both << 6h old). Cheap (one short-lived UPDATE).
        try:
            mid_reaped = _reap_orphaned_runs()
            if mid_reaped:
                print(f"[continuous] mid-loop reaped {mid_reaped} orphaned "
                      f"'running' run(s) → failed")
        except Exception as e:
            print(f"[continuous] mid-loop reap failed: {e}")

        start_id = _next_run_id(engine)
        t0 = time.time()
        print(f"[continuous] cycle {cycle} runs {start_id}..{start_id + RUNS_PER_CYCLE - 1}")

        # Refresh local article cache so the engine sees news digital-intern
        # has written since the last engine init. (Engine is fresh, but this
        # also covers articles written during the current cycle's lifetime.)
        try:
            n_arts = engine.refresh_local_articles()
            print(f"[continuous] refreshed local_news: {n_arts} articles")
        except Exception as e:
            print(f"[continuous] refresh_local_articles failed: {e}")

        results: list[BacktestRun] = []
        try:
            results = engine.run_all(RUNS_PER_CYCLE, start_run_id=start_id) or []
        except Exception as e:
            print(f"[continuous] {_now()} cycle {cycle} crashed: {e}")
            traceback.print_exc()

        winner = None
        top_runs: list[BacktestRun] = []
        # Set only when the scorer is actually retrained this cycle; left None
        # otherwise so exactly one skill-ledger row is written per cycle below
        # (real status when trained, "no outcome records" sentinel when not).
        scorer_status: str | None = None
        if results:
            sorted_results = sorted(results, key=lambda r: r.total_return_pct, reverse=True)
            # Only include runs that beat a flat 0% return (filter out pure losers)
            top_runs = [r for r in sorted_results[:TOP_RUNS_TO_TRAIN]
                        if r.total_return_pct > 0]
            if not top_runs:
                top_runs = sorted_results[:1]  # always train on best even if negative
            winner = top_runs[0]
            try:
                _append_top_decisions(engine, top_runs, cycle)
            except Exception as e:
                print(f"[continuous] top-runs append failed: {e}")

            # Compute 5d forward return outcomes for every BUY/SELL decision
            # across ALL runs (winners and losers) so the scorer learns from
            # losing decisions too — training only on top runs caused survivorship
            # bias and an overly optimistic model.
            outcome_records: list[dict] = []
            try:
                outcome_records = _compute_decision_outcomes(engine, sorted_results)
                print(f"[continuous] computed {len(outcome_records)} decision outcomes "
                      f"from {len(sorted_results)} runs")
            except Exception as e:
                print(f"[continuous] outcome compute failed: {e}")

            # LLM annotation: endorse/condemn individual trades to improve training signal
            if outcome_records and winner and sorted_results:
                loser = sorted_results[-1]
                try:
                    outcome_records = _llm_annotate_outcomes(
                        engine, winner, loser, outcome_records, cycle
                    )
                    endorsed = sum(1 for r in outcome_records if r.get("llm_quality_label") == 1)
                    condemned = sum(1 for r in outcome_records if r.get("llm_quality_label") == -1)
                    print(f"[continuous] LLM labels: {endorsed} endorsed, {condemned} condemned")
                except Exception as e:
                    print(f"[continuous] LLM annotation outer failed: {e}")

            # Train DecisionScorer on accumulated outcomes (accumulate across cycles)
            _all_outcomes_path = ROOT / "data" / "decision_outcomes.jsonl"
            if outcome_records:
                try:
                    _all_outcomes_path.parent.mkdir(parents=True, exist_ok=True)
                    with _all_outcomes_path.open("a") as _of:
                        for _o in outcome_records:
                            _of.write(json.dumps(_o) + "\n")
                except Exception as e:
                    print(f"[continuous] outcome append failed: {e}")

            # Load most recent outcomes and retrain scorer.
            # Capped at MAX_OUTCOMES_FOR_TRAINING — older outcomes describe a stale
            # signal regime and the file would otherwise grow unbounded.
            try:
                all_lines: list[str] = []
                if _all_outcomes_path.exists():
                    all_lines = [l for l in _all_outcomes_path.read_text().splitlines() if l.strip()]
                # Trim the file on disk when it grows past 2× the training cap so
                # it doesn't accumulate indefinitely across cycles. The model only
                # ever sees the tail anyway.
                if len(all_lines) > MAX_OUTCOMES_FOR_TRAINING * 2:
                    kept = all_lines[-MAX_OUTCOMES_FOR_TRAINING:]
                    # Atomic rewrite: a torn write (process killed mid-truncate)
                    # would corrupt or empty the accumulated outcomes file —
                    # permanently losing the scorer's training history. Write to
                    # a temp file then atomically replace.
                    _tmp = _all_outcomes_path.with_suffix(".jsonl.tmp")
                    _tmp.write_text("\n".join(kept) + "\n")
                    _tmp.replace(_all_outcomes_path)
                    print(f"[continuous] trimmed outcomes file "
                          f"{len(all_lines)} → {len(kept)} lines")
                    all_lines = kept
                all_outcomes: list[dict] = []
                for _line in all_lines:
                    try:
                        all_outcomes.append(json.loads(_line))
                    except Exception:
                        pass
                all_outcomes = all_outcomes[-MAX_OUTCOMES_FOR_TRAINING:]
                scorer_status = _train_decision_scorer(all_outcomes)
                print(f"[continuous] {scorer_status}")
                # Reset the singleton under its lock so next cycle reloads the
                # freshly-trained scorer. Bare assignment races with any backtest
                # thread mid-call to _get_decision_scorer().
                import paper_trader.backtest as _bt
                with _bt._DECISION_SCORER_LOCK:
                    _bt._DECISION_SCORER = None
            except Exception as e:
                print(f"[continuous] scorer train failed: {e}")

            # Opus 4.7 annotation in background thread — don't block next cycle
            import threading as _threading
            _threading.Thread(
                target=_opus_annotate, args=(engine, top_runs, cycle, outcome_records),
                daemon=True, name=f"opus-annotate-{cycle}"
            ).start()

        # Durable per-cycle scorer-skill ledger. `_train_decision_scorer`
        # already computes val_rmse / oos_rmse / oos_diracc / oos_ic every
        # cycle but only *printed* them to continuous.log (ephemeral, rotated,
        # un-trendable). Append exactly one structured JSONL row per cycle so a
        # skeptical quant can trend whether OOS skill is improving, holding at
        # the documented negative-skill plateau, or degrading. When the scorer
        # was NOT retrained this cycle (no results / no outcome records),
        # `scorer_status` is None — record the "no outcome records" sentinel
        # with a deployed-pickle `n_train` hint so the row's `gate_active`
        # (invariant #5) stays truthful even on a non-training cycle.
        # `_append_scorer_skill_log` is best-effort by construction — a ledger
        # write must never break the loop.
        if scorer_status is not None:
            _append_scorer_skill_log(scorer_status, cycle, win_start, win_end)
        else:
            _append_scorer_skill_log(
                "no outcome records", cycle, win_start, win_end,
                n_train_hint=_deployed_scorer_n_train(),
            )

        # Durable per-cycle trivial-baseline ledger (the decisive
        # MLP_WORSE_THAN_TRIVIAL question). Wired here for the same reason —
        # and in the same place — as the scorer-skill ledger above: it was a
        # CLI-only signal with no durable trend an unattended operator could
        # see. Exactly one row per cycle, best-effort by construction (it
        # never raises), reading the just-retrained deployed pickle vs the
        # accumulated outcomes tail — the same view `baseline_compare`'s CLI
        # and `calibration --oos` report, so the numbers can never drift.
        _append_baseline_skill_log(cycle, win_start, win_end)

        # Durable per-cycle LLM-annotation skill ledger. Mirrors the wiring
        # rationale exactly: ``llm_annotation_skill.analyze`` reports whether
        # ENDORSE/CONDEMN labels exist AND whether they predict realized
        # returns, but until now an operator had to manually run the CLI to
        # see either. The most directly operational state — `pipeline_dark`
        # (zero non-zero labels across the corpus) — was invisible without
        # grepping ``decision_outcomes.jsonl``. The empty ``except Exception``
        # in ``_llm_annotate_outcomes`` swallows missing-API-key and
        # connection failures silently every cycle; the persisted row makes
        # that darkness obvious in the trend so a quant doesn't pour weight-
        # policy tuning effort into a column that is never populated. Best-
        # effort by construction; never breaks the loop.
        _append_llm_annotation_skill_log(cycle, win_start, win_end)

        # Durable per-cycle ``predict_calibrated`` reliability ledger.
        # ``calibration_reliability.analyze`` reports whether the quantile-
        # mapped calibrated value (pass #10) actually delivers honest 5d
        # magnitudes on the temporal-OOS holdout AND whether it measurably
        # narrows the bias vs raw ``predict()`` on the same OOS pairs
        # (``vs_raw_bias_reduction``). Until now an operator had to run the
        # CLI to see either. The most directly operational state —
        # ``calibrated_dark`` (the deployed pickle predates pass #10, so
        # every consumer of the calibrated value reads None for every
        # prediction) — was invisible without inspecting the pickle. This
        # row makes that darkness obvious in the trend and surfaces the
        # cycle-by-cycle bias-reduction signal so a quant can tell whether
        # the calibration step is a real improvement or cosmetic. Best-
        # effort by construction; never breaks the loop.
        _append_calibrated_reliability_log(cycle, win_start, win_end)

        # Durable per-cycle conviction-sizing calibration ledger. The
        # ``_ml_decide`` gate scales bets from ~5% (low ml_score) to
        # 25–40% (leveraged-bull) and then ×0.6/×0.85/×1.15/×1.3 modulates
        # on the deployed scorer's prediction (invariant #5). Whether that
        # economic weight actually realizes higher 5d return is the most
        # directly operational ML question — but until this wiring landed
        # it was CLI-only, with no durable trend an unattended loop could
        # surface. The current CLI verdict on the live 1173-row OOS slice
        # is ``MISCALIBRATED`` (spearman +0.011 — sizing is variance with
        # no compensating realized edge, the ``GATE_INEFFECTIVE`` shape).
        # This row makes that state visible in the trend and surfaces the
        # cycle-by-cycle ``top_minus_bottom_realized_pct`` so a quant can
        # tell whether the sizing rule starts to work or stays flat. Best-
        # effort by construction; never breaks the loop.
        _append_conviction_calibration_log(cycle, win_start, win_end)

        # Durable per-cycle bootstrap-CI ledger. Answers the operator-
        # decisive question every point-estimate OOS diagnostic ducks: *is
        # the deployed scorer's OOS rank-IC statistically distinguishable
        # from a coin flip at this n_oos?* Reuses
        # ``oos_bootstrap_ci.bootstrap_ci`` verbatim against the SAME
        # temporal OOS holdout ``_train_decision_scorer`` evaluates with,
        # so a quant can join ``scorer_skill_log.cycle`` to
        # ``bootstrap_ci_skill_log.cycle`` and read the point estimate
        # AND its 95% CI on the same row. Verdict ladder
        # SKILL_DETECTED / NO_SKILL_DETECTED / INSUFFICIENT_DATA /
        # NOT_TRAINED gives an operator an immediate yes/no rather than a
        # decimal that could be noise. Best-effort by construction; never
        # breaks the loop. Wired adjacent to ``_append_scorer_skill_log``
        # because it consumes the SAME OOS slice that ledger reports on —
        # CI bounds belong next to the point estimates they qualify.
        _append_bootstrap_ci_skill_log(cycle, win_start, win_end)

        # Durable per-cycle stop-out skill ledger. ``stop_out_audit``
        # answers whether the inherited ``_buy`` ``stop_loss = price * 0.92``
        # band actually saves more from limited-loss trades than it costs
        # in prematurely-exited recoveries — the single most directly
        # operational risk-management question about the gate. Until this
        # wiring landed it was CLI-only, with no durable trend an
        # unattended loop could surface; the historical 8753-row corpus
        # pre-dates the 2026-05-23 intraperiod feature so the immediate
        # state is INSUFFICIENT_DATA (``stop_dark=True``) on every cycle.
        # The ledger surfaces THAT honestly so the moment new post-feature
        # outcome rows accumulate enough to flip the verdict is visible in
        # the trend. Best-effort by construction; never breaks the loop.
        _append_stop_out_skill_log(cycle, win_start, win_end)

        # Durable per-cycle MFE-conversion / take-profit skill ledger.
        # Sibling to the stop-out ledger on the matching upside arm:
        # answers whether the inherited ``take_profit = price * 1.15``
        # band captures more upside than it forfeits AND what fraction
        # of the intraperiod peak the 5d endpoint actually retains (the
        # textbook quant "peak then crater" signal). Same operational
        # rationale — CLI-only until this wiring, and the immediate
        # state is ``tp_dark=True`` on every cycle while the corpus is
        # dominated by pre-feature rows. Best-effort; never breaks the
        # loop.
        _append_mfe_skill_log(cycle, win_start, win_end)

        # Durable per-cycle multi-horizon stop-band sweep ledger. Sibling
        # of ``_append_stop_out_skill_log`` with a wider lens: instead of
        # one deployed (band, horizon) pair, it sweeps a 2-D grid of
        # candidate STOP bands × horizons (5d/10d/20d) and emits
        # ``CELL_BEATS_DEPLOYED`` / ``DEPLOYED_OPTIMAL`` / ``NO_BAND_HELPS``.
        # Pass #42 (2026-05-26) added the 10d/20d intraperiod fields and
        # explicitly flagged this sweep as the next step; until this
        # wiring landed the verdict was CLI-only. Best-effort by
        # construction; never breaks the loop.
        _append_stop_band_sweep_log(cycle, win_start, win_end)

        # Durable per-cycle gate-arm historical skill ledger.
        # ``gate_arm_historical.analyze`` answers the documented
        # quant-decisive question: *do the conviction gate's arms
        # (×0.6 / ×0.85 / ×1.0 / ×1.15 / ×1.3) — bucketed by the gate's
        # TRUE then-deployed prediction — realize differentiated
        # economic outcomes, or is the bucketing just noise?* The
        # current OOS verdict is ``GATE_INEFFECTIVE``: the deployed
        # scorer carries strong OOS rank-IC (+0.48) but the bucket
        # assignment captures only a -0.13pp spread between the ×1.30
        # arm and the ×0.60 arm. Until this wiring landed that gap was
        # CLI-only with no durable trend — a quant could not see the
        # moment bucket tuning recovers real arm divergence. Best-effort
        # by construction; never breaks the loop.
        _append_gate_arm_skill_log(cycle, win_start, win_end)

        # Durable per-cycle gate economic-counterfactual ledger. The
        # sibling roll-up to ``_append_gate_arm_skill_log``:
        # ``gate_pnl.analyze`` answers the keep-or-kill question every
        # per-arm view structurally cannot — *aggregated across all five
        # arms and weighted by how often each fires, does the multiplier
        # overlay net ADD or SUBTRACT realized return vs not gating?* A
        # gate can read ``GATE_INEFFECTIVE`` (per-arm view) while the
        # *summed* effect is meaningfully positive or negative — which
        # is what matters for the keep-or-kill decision. Verdict ladder:
        # ``INSUFFICIENT_DATA`` / ``GATE_SUBTRACTS_RETURN`` /
        # ``GATE_RETURN_NEUTRAL`` / ``GATE_ADDS_RETURN`` at
        # ``EDGE_TOL_PP=1.0pp``. Until this wiring landed the verdict
        # was CLI-only with no durable trend — same operator-blind
        # state ``_append_gate_arm_skill_log`` closed for the arms
        # breakdown. Best-effort by construction; never breaks the loop.
        _append_gate_pnl_skill_log(cycle, win_start, win_end)

        # Durable per-cycle per-persona decision-signal skill ledger.
        # ``persona_skill.persona_skill`` answers the decisive question
        # ``persona_leaderboard`` *cannot*: within each persona's own
        # decisions, does the persona's signal actually rank-predict
        # realized outcomes, or is the per-persona return pure leveraged
        # beta? Until this wiring landed it was CLI-only with no durable
        # trend — and the single most operational state
        # (``HAS_INVERTED_PERSONA`` — at least one persona whose signal
        # is ANTI-predictive, the actionable red flag) was invisible to
        # an unattended operator. Best-effort by construction; never
        # breaks the loop.
        _append_persona_skill_log(cycle, win_start, win_end)

        # Durable per-cycle (persona × regime) cross-tab skill ledger. Pass
        # #44 shipped ``persona_regime_skill`` — the missing intersection
        # of ``persona_skill`` and ``regime_audit`` — but skipped per-cycle
        # wiring, so the verdict (the actionable
        # ``HAS_INVERTED_CELL`` / ``REGIME_CONDITIONAL`` state) was
        # CLI-only. This closes the wiring gap so a quant can trend
        # per-cell signal health and catch INVERTED (persona, regime)
        # pairs the moment they emerge — the same operator-blind state
        # every sibling ``_append_*_skill_log`` closed for its own
        # analyzer. Wired immediately after ``_append_persona_skill_log``
        # because it is the natural sibling: same data file, same SELL
        # double-flip, same SSOT (``_spearman`` from calibration,
        # ``persona_for`` from backtest). Best-effort by construction;
        # never breaks the loop.
        _append_persona_regime_skill_log(cycle, win_start, win_end)

        # Durable per-cycle per-sector OOS skill ledger. ``sector_skill``
        # already answers per-sector, on the SAME temporal holdout the
        # scorer-skill ledger uses, *is the scorer's rank skill UNIFORM
        # across the universe or carried by one fat sector?* — the live
        # corpus is ~89% tech, and an inverted non-tech sector
        # (``rank_ic ≤ -0.15``) means gating on that sector is actively
        # harmful. Until this wiring landed the verdict
        # (``HAS_INVERTED_SECTOR`` / ``SECTOR_CONCENTRATED`` /
        # ``NO_SECTOR_EDGE`` / ``HEALTHY``) was CLI-only with no durable
        # trend; this closes the wiring gap mirroring every sibling
        # ``_append_*_skill_log`` pattern. Wired immediately after the
        # persona/regime ledger because it is the natural sector-axis
        # sibling (same outcomes file, same temporal split, same SSOT
        # via ``_spearman``). Best-effort by construction; never breaks
        # the loop.
        _append_sector_skill_log(cycle, win_start, win_end)

        # Durable per-cycle dead-feature audit of the just-retrained model.
        # Catches the pass-#35 class of bug systematically: a feature added
        # to ``DecisionScorer.build_features`` whose values are never plumbed
        # into ``_compute_decision_outcomes`` (training capture) or
        # ``_ml_decide`` (inference) trains on constant zero, the
        # StandardScaler scales it to ~0, and L2 alpha drives every weight to
        # exactly 0.0. Existing diagnostics (``feature_importance``
        # permutation reading, ``calibration``, ``gate_audit``) don't surface
        # this — they all read DOWNSTREAM (model output / metrics), not the
        # model's own first-layer weights. Until this wiring landed, the 3
        # dead enhanced-MACD slots were invisible to the unattended loop for
        # an unknown number of cycles. Best-effort by construction; never
        # breaks the loop (the ``_append_*`` discipline every sibling
        # ledger follows). Runs AFTER the persona ledger and BEFORE
        # ``_try_train_ml`` so the audit sees the freshly-retrained pickle
        # this cycle just wrote.
        _append_dead_feature_audit_log(cycle, win_start, win_end)

        ml_status = _try_train_ml() if winner else "no winner"
        print(f"[continuous] ml: {ml_status}")

        # Bound winner_training.jsonl disk growth. Runs AFTER `_try_train_ml`
        # (which has finished reading the tail it needs) and after this cycle's
        # `_append_top_decisions`, so the trim never races the consumer.
        _trim_winner_jsonl()

        # Backtest results are silent — check the dashboard at :8090

        # IMPORTANT: trim history BEFORE dispatching validation. The
        # validation thread mutates `engine.store` (swaps in an isolated
        # store for permutation runs), so anything that operates on the
        # real backtest.db via `engine.store` must run first. Validation
        # is the *last* thing scheduled on `engine` per cycle.
        try:
            deleted = _trim_history(engine, keep=KEEP_LAST_RUNS)
            if deleted:
                print(f"[continuous] trimmed {deleted} old runs "
                      f"(keeping last {KEEP_LAST_RUNS})")
        except Exception as e:
            print(f"[continuous] trim failed: {e}")

        # Validation suite — runs in a background thread so the next cycle
        # isn't blocked by the ~25-min permutation test. Must be the LAST
        # thing scheduled on `engine` because the validation function
        # mutates `engine.store` (swaps in an isolated store for permutation
        # runs); any subsequent code reading `engine.store` would silently
        # read the empty isolated DB.
        if cycle % VALIDATION_EVERY_N_CYCLES == 0:
            try:
                from paper_trader.backtest import LOCAL_ARTICLES_DB
                articles_db = (str(LOCAL_ARTICLES_DB)
                               if LOCAL_ARTICLES_DB.exists() else None)
                import threading as _threading
                _threading.Thread(
                    target=_run_validation_async,
                    args=(engine, cycle, win_start, win_end, articles_db),
                    daemon=True, name=f"validation-{cycle}",
                ).start()
                print(f"[continuous] validation cycle {cycle} dispatched (background)")
            except Exception as e:
                print(f"[continuous] validation dispatch failed: {e}")

        # 🐒 Monkey-benchmark auto-refresh — once per day, in background.
        # Validates whether AI runs are beating a 10k-random-trader baseline.
        # Always non-fatal: the continuous loop never depends on it.
        try:
            from paper_trader.analytics.monkey_benchmark import (
                load_cached as _monkey_load_cached,
                run_and_cache as _monkey_run_and_cache,
                default_window as _monkey_default_window,
            )
            from paper_trader.backtest import (
                WATCHLIST as _MONKEY_WATCHLIST,
                BACKTEST_DB as _MONKEY_BACKTEST_DB,
            )
            import datetime as _dt
            cached = _monkey_load_cached()
            stale = cached is None
            if not stale:
                try:
                    # `monkey_benchmark.run_and_cache` writes
                    # `datetime.utcnow().isoformat() + "Z"`, a naive ISO
                    # string with a trailing Z. Parse it as timezone-aware
                    # UTC so the subtraction below uses a single timezone
                    # discipline and avoids the deprecated
                    # `datetime.utcnow()` (slated for removal in a future
                    # Python release).
                    gen_at_raw = (cached.get("generated_at") or "")
                    gen_at = gen_at_raw.rstrip("Z")
                    gen_dt = _dt.datetime.fromisoformat(gen_at)
                    if gen_dt.tzinfo is None:
                        gen_dt = gen_dt.replace(tzinfo=_dt.timezone.utc)
                    age_h = ((_dt.datetime.now(_dt.timezone.utc) - gen_dt)
                             .total_seconds() / 3600)
                    stale = age_h > 24
                except Exception:
                    stale = True
            if stale:
                m_start, m_end = _monkey_default_window()
                m_ai_returns: list[float] = []
                try:
                    _mc = sqlite3.connect(str(_MONKEY_BACKTEST_DB), timeout=5)
                    _mr = _mc.execute(
                        "SELECT total_return_pct FROM backtest_runs "
                        "WHERE status='complete' AND total_return_pct IS NOT NULL "
                        "AND start_date = ? AND end_date = ? "
                        "ORDER BY run_id DESC",
                        (m_start, m_end),
                    ).fetchall()
                    _mc.close()
                    m_ai_returns = [r[0] for r in _mr if r[0] is not None]
                except Exception as _e:
                    print(f"[monkey] AI-returns fetch failed: {_e}")
                import threading as _threading
                _threading.Thread(
                    target=_monkey_run_and_cache,
                    args=(_MONKEY_WATCHLIST, m_start, m_end),
                    kwargs={"ai_returns": m_ai_returns},
                    daemon=True, name=f"monkey-{cycle}",
                ).start()
                print(f"[monkey] refresh triggered in background "
                      f"({m_start} → {m_end}, {len(m_ai_returns)} matching AI runs)")
        except Exception as e:
            print(f"[monkey] background refresh failed (non-fatal): {e}")

        # Persist a small batch of monkey simulations to backtest.db each
        # cycle so the dashboard can show them next to the AI runs as a
        # baseline. Uses `_monkey_default_window()` (the most data-rich cached
        # window) rather than this cycle's random window — `_load_prices`
        # requires a price-cache file that fully covers `[start, end]`, and
        # `_pick_window` ranges 1993→present so most cycles wouldn't have
        # one. Always non-fatal: a missing cache or DB hiccup must never stop
        # the loop. The function trims to ≤500 monkey rows in FIFO order
        # itself, so unbounded accumulation is impossible.
        # Seed differs across processes (epoch-keyed XOR with cycle) so a
        # loop restart that resets `cycle` to 0 still produces fresh runs.
        try:
            from paper_trader.analytics.monkey_benchmark import (
                run_monkey_backtests_to_db, default_window as _mk_default_window,
            )
            from paper_trader.backtest import (
                WATCHLIST as _MK_WATCHLIST,
                BACKTEST_DB as _MK_BACKTEST_DB,
            )
            mk_start, mk_end = _mk_default_window()
            mk_seed_offset = (cycle_seed + cycle) & 0xFFFFFFFF
            n_inserted = run_monkey_backtests_to_db(
                tickers=_MK_WATCHLIST,
                start_date=mk_start,
                end_date=mk_end,
                backtest_db_path=str(_MK_BACKTEST_DB),
                n_runs=N_MONKEY_BT_PER_CYCLE,
                seed_offset=mk_seed_offset,
            )
            print(f"[monkey_bt] inserted {n_inserted} monkey runs "
                  f"({mk_start}→{mk_end}) into backtest.db")
        except Exception as e:
            print(f"[monkey_bt] backtest insertion failed (non-fatal): {e}")

        elapsed = time.time() - t0
        if winner:
            print(f"[continuous] {_now()} cycle {cycle} done in {elapsed/60:.1f}min. "
                  f"Best run {winner.run_id} {winner.total_return_pct:+.2f}%")
        else:
            print(f"[continuous] {_now()} cycle {cycle} done in {elapsed/60:.1f}min")

        if _STOP:
            break

        print(f"[continuous] sleeping {COOLDOWN_SECONDS}s before cycle {cycle + 1}")
        slept = 0
        while slept < COOLDOWN_SECONDS and not _STOP:
            chunk = min(2, COOLDOWN_SECONDS - slept)
            time.sleep(chunk)
            slept += chunk

    print(f"[continuous] {_now()} loop stopped after {cycle} cycle(s)")
    sys.exit(0)


if __name__ == "__main__":
    try:
        os.nice(10)
    except Exception:
        pass
    main()
