# digital-intern — agent guide

This guide is for AI coding agents working on this repo. CLAUDE.md has the long-form architecture
reference; this file is the operational summary plus the invariants you can break by mistake.

---

## Architecture at a glance

`daemon.py` is the production entry point. It spins up ~30 independent worker threads — one per
data source, plus the scoring/alerting/training pipeline — and supervises them. Article flow:

```
collectors/* → _ingest (heuristic_scorer) → ArticleStore.insert_batch
                                                │
                                                ▼
              scorer_worker → ml.inference.score_articles → either
                                                              ├─ ml.update_ml_scores_batch  (confident model)
                                                              └─ watchers.urgency_scorer    (uncertain → Sonnet → update_ai_scores_batch)
                                                │
                                                ▼
                              alert_worker → watchers.alert_agent.send_urgent_alert → Discord
                                                │
                                                ▼
                              heartbeat_worker (5h) → analysis.claude_analyst.analyze → Discord
                                                │
                                                ▼
                              ml.trainer.train (3min) — pulls ai_score=llm/briefing_boost rows + synthetic backtest rows
```

`storage/article_store.py` owns the SQLite layer. The same DB is read by paper-trader at
`/home/zeph/paper-trader/`, which both reads (live signals — must filter synthetic rows) and writes
(synthetic backtest training rows — kept in DB, hidden from live).

---

## Critical invariants — read this before touching the data path

### 1. Backtest isolation
Rows with `url LIKE 'backtest://%'` or `source LIKE 'backtest_%'` or `source LIKE 'opus_annotation%'`
are training-only artifacts injected by paper-trader. They **must never** reach:

- the live alert formatter (`watchers/alert_agent.py`)
- the heartbeat briefing (`analysis/claude_analyst.py`)
- the urgency scorer (`watchers/urgency_scorer.py`)

The canonical filter lives in `storage/article_store.py::_LIVE_ONLY_CLAUSE`. Every read path on the
live pipeline applies it:

- `get_unscored` — for ML inference + Sonnet routing
- `get_unalerted_urgent` — for the alert worker
- `get_top_for_briefing` — for the 5h Opus briefing
- `count_unscored`, `stats` — for monitoring
- `update_scores_from_labels` — *write* path: the only producer of
  `score_source='briefing_boost'` (read by the trainer as strong ground truth).
  Its label list is derived from the already-live-only `get_top_for_briefing`,
  but it carries the clause as defense-in-depth so a future change to the
  briefing-label path can't promote a synthetic SELL-loser's `0.5` outcome
  label to `4.5` and poison the training pool. Pinned by
  `tests/test_briefing_boost.py::TestBriefingBoostBacktestIsolation`.

`send_urgent_alert` has a defense-in-depth re-filter so a future caller that bypasses the store
can't leak synthetic rows into Discord. Tests in `tests/test_article_store.py::TestBacktestIsolation`
gate this.

Training paths (`ml/trainer.py::_fetch_training_data`, `train_continuous`) deliberately include
synthetic rows — that's the whole point of the backtest replay loop. They exclude `score_source='ml'`
instead, to avoid the label-feedback loop.

### 2. ml_score vs ai_score separation
- `ai_score` — LLM ground-truth labels (`score_source` ∈ {`llm`, `briefing_boost`}). Trainer reads
  this as truth.
- `ml_score` — the model's own predictions (`score_source='ml'`). Never read by the trainer.

`update_ml_scores_batch` writes `ml_score` and tags `score_source = COALESCE(score_source, 'ml')` —
so an LLM-tagged row stays LLM-tagged even after a later model inference pass. Tests in
`TestScoreSourceSeparation` enforce this.

Readers that need a unified "effective score" use
`COALESCE(NULLIF(ai_score, 0), ml_score, 0)` (see `get_unalerted_urgent`,
`get_top_for_briefing`).

### 3. Urgency state machine
`urgency` is tri-state: `0` = normal, `1` = needs alert, `2` = alerted. All score-writing paths
use `MAX(urgency, ?)` so a fresh Sonnet rescore can never regress an alerted article back to `1`
(which would re-fire the alert). Tested in `TestAlertedMarking::test_subsequent_llm_rescore_does_not_un_alert`
and `TestPreservesAlerted::test_rescore_does_not_unalert`.

### 4. Train/serve feature parity (article age must reach the live path)
`ArticleStore.get_unscored` MUST return `published` and `first_seen`, not just
`id/title/source/summary`. Two live consumers derive article age from those fields and
silently degrade if they are absent:

- `ml/features.py::extract_features` builds 5 temporal features (hour/dow cyclic
  encodings + `days_since_published`) from `published`. The trainer
  (`_fetch_training_data`) passes the real value; if inference omits it the parser
  falls back to `now()` and those 5 features collapse to a constant — a train/serve
  skew on **every** scored article (not an error, just a quietly worse model).
- `watchers/urgency_scorer.score_batch` computes each article's `age_hours`
  (`_article_age_hours` reads `published`/`first_seen`). That value feeds *both* the
  Sonnet prompt's staleness rule *and* the hard `STALE_HOURS`/`STALE_SCORE_CAP`
  clamp. With every article looking 0h old, the entire staleness defense is inert on
  the live path and >`STALE_HOURS` news can still fire urgent alerts.

This was the failure: `get_unscored` had been trimmed to the minimal projection and
the regression is invisible (no exception, model still trains, alerts still fire —
just on stale items with skewed features). Any future edit to the `get_unscored`
projection MUST keep both age columns. Pinned by
`tests/test_get_unscored_age_fields.py` (drives the real `insert_batch → get_unscored`
path, not hand-built dicts, and asserts feature-row parity between the training and
inference dict shapes). Note `STALE_HOURS` has been retuned (24h → 48h); tests read
the live module constants rather than hardcoding the window.

---

## Running the daemon

Production (systemd):
```sh
systemctl --user start digital-intern
journalctl --user -fu digital-intern
```

Foreground (development):
```sh
cd /home/zeph/digital-intern
python3 daemon.py
```

Health probe:
```sh
bash /home/zeph/digital-intern/healthcheck.sh
```

The daemon takes a singleton lock at `data/daemon.lock` — a second process waits for the first to
exit. Workers are supervised: 3+ crashes in 5 min → degraded (slow respawn); 10+ → disabled for 30
min. Discord alerts fire on state transitions only.

Dashboard runs on `:8080` (Flask). `WEB_SERVER_PORT` env overrides the bind port.

---

## Running tests

```sh
cd /home/zeph/digital-intern && python3 -m pytest tests/ -v
```

Tests use in-memory-ish SQLite via a `tmp_path`-redirected store fixture (`tests/conftest.py`).
External calls (Claude CLI, network) are patched. No GPU required for the model tests — they
exercise the `ArticleNetModule` directly on CPU.

**Phantom failures from a stale pytest cache.** pytest's assertion-rewrite
bytecode (`**/__pycache__/*.pyc`) can lag behind an edited test file and
surface as a failure that no longer exists in the source (observed:
`test_source_health_stale.py` showing an old `monkeypatch.setattr` body that
had already been replaced with a behavioral version). If a failure's traceback
does not match the current file content, clear the caches and re-run:
`find . -name __pycache__ -type d -exec rm -rf {} + && rm -rf .pytest_cache`.
This is a dev-loop hazard, not a code bug — don't "fix" code chasing it.

**Fixture convention — `first_seen` must be time-relative.** `get_unalerted_urgent` and
`get_top_for_briefing` enforce a 24h `first_seen` freshness window. Test `_insert*` helpers
default `first_seen` to ~5 min ago (`datetime.now(timezone.utc) - timedelta(minutes=5)`), not a
hardcoded date. A literal date silently breaks every backtest-isolation test 24h later — a
green-looking invariant test that fails on a calendar boundary, not on a real regression. Pass an
explicit `first_seen=` only when a test specifically targets the staleness cutoff.

Suites:

- `test_article_store.py` — backtest isolation, alerted-marking, ml/llm score separation, CRUD.
- `test_urgency_scorer.py` — classification at the 8.0 threshold, partial Sonnet responses,
  alerted-state preservation.
- `test_features.py` — exactly 15 extra dims, ticker density, days-since-published normalization
  (`min(age,30)/30` → ~1/30 at 24h, saturates 1.0 at ≥30d; this is intended ML feature scaling,
  not a bug), cyclic feature bounds.
- `test_model.py` — output bounds (relevance 0..10, urgency 0..1, no NaN on zero input).
- `test_trainer.py` — `score_source='ml'` exclusion, synthetic-row inclusion, sample weighting,
  and `TestTrainOrchestration` — regression guard that `train()` runs end-to-end on both the
  fresh and disk-cache paths (see ML training pipeline note below).
- `test_integration_pipeline.py` — cross-module flows (ingest→score→alert, end-to-end backtest
  isolation, concurrent-writer safety).
- `test_retrain_guard.py` — `core/retrain_guard.py` escalation policy: fires exactly at the
  consecutive-failure threshold and on every multiple after, never below it, never on a
  non-positive count or misconfigured threshold (see ML pipeline note below).
- `test_alert_dedup.py` / `test_logger_rotation.py` — syndication dedup signature/merge rules and
  size-rotation of `logs/structured.jsonl`.
- `test_recursive_labeler.py` — `_apply_labels` defensive urgency parse (a non-int urgency from
  Claude must not abort the run or discard the batch's good labels), 0..5→0..10 relevance rescale,
  `score_source='llm'` on writes, and `_fetch_round1_candidates` backtest/opus exclusion (a
  separate `WHERE` filter than `_LIVE_ONLY_CLAUSE`).
- `test_dashboard_backtest_isolation.py` — backtest isolation on the two *non-store*
  live-facing surfaces (see "dashboard parity" below): `dashboard/server.py::_articles_payload`
  / `_articles_per_hour_24h` (the standalone uvicorn dashboard) and
  `ml/sentiment_trends.py::compute_trends` (the per-ticker panel) must filter synthetic rows
  the same way the store paths and `dashboard/web_server.py` do.
- `test_paper_trader_signals_isolation.py` — cross-system backtest isolation on the
  vendored `paper_trader/signals.py` snapshot. `get_top_signals`, `get_urgent_articles`,
  `get_ticker_sentiment` and `ticker_sentiments` read the shared `articles.db` for the
  live trader; all four must inline the `_LIVE_ONLY_CLAUSE` fragment (see "Cross-system
  contract" below — the vendored copy had drifted out of sync with the authoritative
  source and was leaking synthetic rows; this suite pins it).
- `test_inference_grey_zone.py` — `ml/inference.py::score_articles` LLM-routing
  decision. Pins that `needs_llm` keys the grey band on the **urgency** head, not
  relevance (see "Inference routing" below), that wide relevance variance forces
  the LLM regardless, that `confident_noise` suppresses routing, and the
  unfitted-model `rel_std==99` sentinel. Stubs the embedder/model so the decision
  is deterministic without a checkpoint.
- `test_published_staleness.py` — `storage.article_store::_published_older_than`,
  the authoritative 24h briefing-staleness gate. Asserts the exact regression it
  defeats: an old RFC822 date that lex-sorts *after* the ISO cutoff (so the SQL
  `published >= ?` pre-filter keeps it) is still correctly flagged stale; plus
  ISO/`Z`-suffix/naive-UTC parsing and the keep-on-unparseable policy.
- `test_get_unscored_age_fields.py` — invariant #4 above. `get_unscored` must
  surface `published`/`first_seen`. Drives the real `insert_batch → get_unscored`
  path so it fails if the projection is ever trimmed again: (a) a >`STALE_HOURS`
  article scored 9 by a mocked Sonnet is hard-capped to `STALE_SCORE_CAP` with a
  fresh-article control; (b) the same article routed through `get_unscored` vs the
  `_fetch_training_data` dict shape yields identical `extract_features_batch` rows
  (catches the temporal train/serve skew). Reads `STALE_HOURS`/`STALE_SCORE_CAP`
  from the live module so a retune doesn't false-fail it.
- `test_alert_agent.py` — the live alert formatter's own guards
  (`watchers/alert_agent.py::send_urgent_alert`), asserted at the agent
  boundary rather than only end-to-end. Pins that a >24h `published` row
  returned by `get_unalerted_urgent` (whose SQL filters `first_seen`, not
  `published`) is dropped before Claude/Discord; that an unparseable-date
  batch and a missing-webhook config both short-circuit *before* the Sonnet
  call (no wasted quota, no POST to an empty URL); that the happy path marks
  exactly the alerted id `urgency=2` (cannot re-fire); and that a failed
  Discord POST leaves the row `urgency=1` (re-queued, never silently lost).
  `TestSyntheticDefenseInDepth` additionally pins the formatter's *own*
  `_is_synthetic` re-filter (invariant #1 defense-in-depth): synthetic dicts
  handed to `send_urgent_alert` directly, bypassing the store's
  `_LIVE_ONLY_CLAUSE` — `backtest://` URL, `backtest_*` source, and
  `opus_annotation*` source — are dropped before any Claude/Discord call and
  never marked alerted; a mixed batch alerts only the live row.
- `test_score_pending.py` — `storage.article_store::ArticleStore.score_pending`
  (the in-store model-scoring driver; `daemon.scorer_worker` is the parallel
  production path) was the only model-write path with no direct test. Pins
  invariants #1 and #2 on it: model predictions land in `ml_score` /
  `score_source='ml'` (never `ai_score`); a `needs_llm` row is left
  `ai_score=0 / ml_score=NULL` for the Sonnet path; an `urgency>=8` prediction
  bumps `urgency` to 1 via `MAX`; synthetic `backtest://` rows stay invisible
  (excluded by `get_unscored`); and the unfitted-model `rel_std==99` sentinel
  writes nothing (no `ml_score`, no `time_sensitivity`) and returns 0 without
  spinning. Stubs `ml.inference.score_articles` keyed by `_id` so the result
  is independent of `get_unscored`'s `kw_score DESC` ordering.
- `test_backoff.py` — `core/backoff.Backoff`, the retry throttle every collector
  worker in `daemon.py` (~20 call sites) shares. First real suite (was inline
  `__main__`-only). Pins the *actual* contract, not the prose: `peek()` is
  non-mutating; the exponent is clamped at 32 so a permanently-failing worker
  can't `OverflowError` on `2 ** failures`; jitter is applied **after** the cap
  by design (anti-thundering-herd), so the realized sleep is
  `min(cap, base*2**failures)*(1 ± jitter)` and may sit slightly *above* `cap` —
  this is intentional, do not "fix" the code to make `cap` a hard ceiling; the
  0.5s floor; and `sleep(should_continue)` polling out early on shutdown. The
  module docstring was tightened to state this explicitly (code is the spec).

---

## Worker roles (one line each)

| Worker | Interval | Job |
|--------|----------|-----|
| `gdelt`, `rss`, `web`, `reddit`, `ticker`, `sec_edgar`, `sec_edgar_ft`, `google_news`, `nitter`, `substack`, `finnhub`, `alphavantage`, `polygon`, `massive`, `newsapi`, `yahoo_ticker_rss`, `wikipedia` | varies | Collectors. Each polls its source, calls `_ingest`. |
| `scorer` | 30 s | Pulls `get_unscored`, runs `ArticleNet` inference, routes uncertain to Sonnet, writes `ml_score` or queues for LLM. |
| `alert` | 20 s | `get_unalerted_urgent` → `send_urgent_alert` → Discord + TTS. |
| `heartbeat` | 5 h | Opus 4.7 long-form briefing → Discord. Re-labels included articles at 4.5 for training. |
| `ml_trainer` | 3 min | Full ArticleNet retrain (100 epochs). |
| `continuous_trainer` | 2 min | Lightweight 40-epoch fine-tune to keep GPU warm. |
| `recursive_labeler` | 4 h | Sonnet bulk-labels → Opus reviews disagreements → active-learning queue. |
| `price_alert` | 5 min | Discord ping on \|%\| ≥ 3% portfolio move. |
| `purge` | 6 h | Delete rows older than `RETENTION_DAYS=90`; WAL checkpoint. |
| `portfolio_pl`, `sentiment_trends`, `export`, `stats`, `web_server` | varies | Dashboard inputs + Flask server. |

Supervisor state is in `logs/supervisor_state.json` (atomic-rename written every 5 min, consumed
by the dashboard).

---

## ML training pipeline

Label sources, in priority order:

1. **Opus heartbeat-derived labels** — ai_score 4.5, `score_source='briefing_boost'`. Highest signal
   quality; ~50 articles per 5h.
2. **Sonnet urgency_scorer labels** — ai_score from the Sonnet score (clamped 0.01..10),
   `score_source='llm'`.
3. **Backtest synthetic rows** — `score_source=NULL`, fractional ai_score (BUY winner=5.0,
   SELL loser=0.5, opus NEUTRAL=2.5, BAD=0.5). Allowed because they encode trade outcomes.
4. **kw_score weak labels** — bootstrap only; capped at 50% of LLM-labeled corpus or 2000 rows.

The trainer concatenates TF-IDF (15k dims) + 15 extra features. Sample-weighted MSE on relevance
(high-score articles dominate gradient) + 0.5·BCE on urgency + 0.2·BCE on uncertainty + 0.3·BCE on
time_sensitivity.

The model writes its predictions to `ml_score`. The trainer never reads `ml_score` — that's how the
label-feedback loop stays closed.

**Early stopping.** `ArticleNet.fit` takes `early_stop_patience` (default 6, the `ml_trainer`/
`continuous_trainer` callers leave it at the default). It only engages when a held-out val set
exists (`n >= 100`): after that many consecutive val checks fail to beat the running best by
`min_delta` (1e-4), training halts. Best-epoch weights are restored regardless, so early stop only
trims wasted overfitting epochs — it never changes which checkpoint is saved or the reported
`val_loss`. The metrics dict gains `epochs_run` (actual) and `stopped_early` (bool); `epochs`
stays the configured budget. `patience=0` disables it (fixed-budget back-compat). Pinned by
`tests/test_model.py::test_early_stop_triggers_on_plateau` /
`test_early_stop_disabled_runs_full_budget`.

**Dataset prep is single-pass.** `train()` builds the feature matrix exactly once, via one of two
branches: a disk-cache hit (`data/ml/dataset_cache.npz`, reused while the labeled count drifts
<5%), or a fresh `_fetch_training_data` → embed → cache-write. The fresh branch `del`s the raw
`texts`/`articles` lists to cap peak RAM before GPU training; the cache branch never builds them.
Anything after that point operates on `X / y_rel / y_urg / y_time` only — re-embedding there (the
pre-cache code shape) raises `NameError` on every cycle and ArticleNet silently stops retraining
while the daemon log still looks healthy. `TestTrainOrchestration` covers both branches.

**Retrain-failure escalation (safety net for the above).** The `NameError` blind spot was invisible
because `ml_trainer_worker` swallows retrain exceptions as `WARNING`, and the hourly healthcheck only
greps `ERROR`/`CRITICAL`. `ml_trainer_worker` now keeps a `consec_fail` counter (reset on any
successful or *skipped* train — a too-few-samples skip is not a failure) and routes the
escalate-or-not decision to the pure, unit-tested `core/retrain_guard.py::should_alert`. It fires a
Discord `is_alert=True` ping at the threshold (`ML_RETRAIN_FAIL_ALERT_THRESHOLD=3`) and re-pings on
every further multiple (6, 9, …) so a persistently broken trainer can't go stale silently again
without flooding the channel. `core/retrain_guard.py` owns the policy precisely so it stays testable
in isolation from the GPU/daemon machinery; `tests/test_retrain_guard.py` pins it.

---

## Inference routing (grey zone)

`ml/inference.py::score_articles` decides per article whether the local model's
score stands or the article is escalated to Sonnet (`needs_llm=True`). The
decision is:

```
confident_noise = rel < LLM_ZONE_CLEAR_NOISE and rel_std < UNCERTAINTY_REL
in_grey         = LLM_ZONE_MID_LO <= urg <= LLM_ZONE_MID_HI      # URGENCY head
uncertain       = rel_std > UNCERTAINTY_REL or urg_std > UNCERTAINTY_URG
needs_llm       = (in_grey or uncertain) and not confident_noise
```

**`in_grey` keys on the urgency head, not relevance.** The urgency head is a
sigmoid probability scaled to 0..10; `LLM_ZONE_MID_LO..HI` (7.0, 8.5) straddles
the 8.0 urgent threshold, so an urgency estimate near the alert boundary is what
gets escalated for an urgent/not-urgent call. CLAUDE.md's glossary and
`ml/model.py`'s docstring loosely call this the "relevance grey zone" — that
wording is imprecise; **the code is the spec.** Repointing `in_grey` at `rel`
silently changes which articles burn a Sonnet call. Pinned by
`tests/test_inference_grey_zone.py`; do not "fix" the code to match the prose.

`scorer_worker` also force-routes a narrow `3.8 <= max(rel,urg) <= 4.3` band to
the LLM independently of this. The unfitted model returns the `rel_std==99`
sentinel, which makes every article `needs_llm` and is also the value
`scorer_worker` checks before persisting `time_sensitivity`.

---

## Common failure modes

| Symptom | Likely cause | Fix |
|--------|--------------|-----|
| `database is locked` retries | High writer contention with `purge_worker`'s `wal_checkpoint(TRUNCATE)`. | `_retry_on_lock` decorator handles 5 attempts with jitter. Persistent failures → check `lock_metrics()`. |
| `score_pending` returns 0 in a loop | `_INFER_LOCK` held, or model not yet fitted. | Wait for first `ml_trainer` cycle; then `[score_pending] N scored so far...` should appear. |
| Sonnet alerts missing | `DISCORD_WEBHOOK_URL` empty, or `claude` CLI not authenticated. | Check `.env`; run `claude --version`. |
| Backtest articles leaking into Discord | A new query forgot `_LIVE_ONLY_CLAUSE`. | Grep for `FROM articles WHERE` in the store; verify every live-read path filters. Re-run `tests/test_article_store.py::TestBacktestIsolation`. |
| Model trains on its own predictions | A new write path put model output into `ai_score`. | Use `update_ml_scores_batch` for predictions, `update_ai_scores_batch` for LLM labels. Re-run `tests/test_article_store.py::TestScoreSourceSeparation`. |
| `val_loss` flat forever, model never improves, `[ml_trainer] Retrain error: name 'texts' is not defined` in `daemon.log` | A code path after dataset prep re-references `texts`/`articles` (deleted/never-built). `train()` raises every cycle but the worker swallows it as a WARNING, so the daemon looks alive. | Keep all post-prep code on `X/y_rel/y_urg/y_time`. Re-run `tests/test_trainer.py::TestTrainOrchestration`. |
| Discord `🚨 ML TRAINER STUCK: N consecutive retrain failures` | Any persistent `ml_train` exception (the `texts` NameError above, a corrupt `dataset_cache.npz`, GPU driver fault). `core/retrain_guard.py` escalates so the WARNING-only blind spot can't recur silently. | Read the `Last error:` in the alert; tail `[ml_trainer] Retrain error (#N)` in `daemon.log` for the full traceback. Counter resets on the next successful/skipped cycle. |
| LLM batch returns fewer scored items than sent (e.g. `[urgency] batch=120 scored=83`) | Claude hit its output-token limit and the JSON array came back truncated mid-element. `core/json_extract.py::extract_json_array` now salvages the complete leading elements instead of discarding the whole batch (returning `None`); the unrecovered tail stays unscored and drains over the next 1–2 cycles. | Expected/benign for very large batches. To eliminate it, lower the batch size in `watchers/urgency_scorer.py` / `ml/recursive_labeler.py`. Pinned by `tests/test_json_extract.py::TestTruncationSalvage`. |
| Heartbeat briefing posts placeholder text | `claude_analyst.analyze` returned `[analyst] No response from Claude.` | `heartbeat_worker` detects this and retries in 5 min instead of waiting the full 5 h. |
| Articles permanently stuck unscored | Sonnet returned an empty or partial response | `score_batch` floors unscored items at 0.01 when Sonnet returned at least one valid entry; the queue must drain over 1–2 cycles. |
| GPU OOM | Concurrent `_inject_and_train` from paper-trader during `ml_trainer_worker` retrain. | `_TRAIN_LOCK` serializes; lower paper-trader's `RUNS_PER_CYCLE`. `_handle_memory_error` clears CUDA cache. |
| Duplicate daemons fighting over port 8080 | Stale process didn't release the singleton lock. | The new daemon waits via blocking `flock`. Check `data/daemon.lock` for the holder PID. |
| `recursive_labeler` worker logs one WARNING per 4h cycle and `total_labeled=0`, model stops gaining gold labels | A label batch from Claude carried a non-int `urgency` (`"1"`, `"1.0"`, `"yes"`, `true`). The unguarded `int()` in `_apply_labels` used to raise, unwinding the whole pipeline and discarding the in-flight batch's good labels. Now degraded to `urgency=0` so the relevance label still lands. | Fixed in `_apply_labels` (defensive `int(float(...))` with `(TypeError, ValueError)` fallback). Pinned by `tests/test_recursive_labeler.py::TestApplyLabels::test_poison_urgency_does_not_abort_or_lose_siblings`. |
| Backtest titles/URLs visible in the standalone dashboard feed (`:8765`) or skewing the per-ticker sentiment panel, while the `:8080` daemon dashboard looks clean | **Dashboard parity.** Backtest isolation is enforced in three independent SQL spots: the store paths (`_LIVE_ONLY_CLAUSE`), `dashboard/web_server.py` (`_LIVE_ONLY_SQL`), and — newly — `dashboard/server.py` + `ml/sentiment_trends.py`. The standalone uvicorn dashboard (`dashboard.service`) and the sentiment-trends aggregator were two parallel reads of `articles` that did not filter synthetic rows, so they rendered training data as live news. | All three now use the canonical clause (`dashboard/server.py` / `ml/sentiment_trends.py` import `_LIVE_ONLY_CLAUSE` from `storage.article_store`). Any new `FROM articles` read that surfaces to a user MUST filter. Pinned by `tests/test_dashboard_backtest_isolation.py`. |
| Stale (>`STALE_HOURS`) news still firing urgent alerts, and/or the model quietly underperforming for no obvious reason | **Article age never reached the live path.** `get_unscored` projected only `id/title/source/summary`, dropping `published`/`first_seen`. `_article_age_hours` then read 0h for every article: the Sonnet staleness rule and the hard `STALE_SCORE_CAP` clamp were both inert, and `extract_features` fell back to `now()` so 5 temporal features were train/serve-skewed. No exception, model still trained, alerts still fired — invisible. | `get_unscored` now returns both age columns (invariant #4). Verify any `get_unscored` projection edit keeps `published`/`first_seen`. Re-run `tests/test_get_unscored_age_fields.py`. |

---

## Where new code goes

| Task | Where |
|------|-------|
| Add a news source | New file in `collectors/` returning `list[dict]` with `{title, link, source, published, summary}`; register worker in `daemon.py::main`. |
| Change heuristic scorer | `triage/heuristic_scorer.py`. |
| Tune ArticleNet | `ml/model.py` (architecture), `ml/trainer.py` (loss / labels), `ml/inference.py` (uncertainty thresholds). |
| Change alert format | `watchers/alert_agent.py::ALERT_PROMPT`. |
| Change briefing format | `analysis/claude_analyst.py::SYSTEM_PROMPT`. |
| New per-article ML feature | `ml/features.py` — bump `EXTRA_FEATURE_DIM` and the test in `tests/test_features.py`. |
| New dashboard panel | `dashboard/` Flask app + JSON endpoint reading `articles.db` / `data/*.json`. |

---

## Cross-system contract with paper-trader

`articles.db` is shared, read-only from paper-trader's live trader (`paper_trader/signals.py`),
read/write from `run_continuous_backtests.py::_inject_and_train`.

If a paper-trader read query is added against `articles.db`, it MUST inline the same SQL fragment
as `_LIVE_ONLY_CLAUSE`. Symptom of a violation: backtest titles appearing in the live trader's
prompt context.

`paper_trader/` here is a **vendored snapshot** of `/home/zeph/paper-trader/paper_trader/`; the
authoritative file is the one the live trader actually runs. The snapshot can silently drift —
`paper_trader/signals.py` was found missing the backtest filter on all four live-read queries
(`get_top_signals`, `get_urgent_articles`, `get_ticker_sentiment`, `ticker_sentiments`) while the
authoritative copy already carried it. Re-synced (filter only) and pinned by
`tests/test_paper_trader_signals_isolation.py`. When updating the vendored snapshot, never copy it
wholesale — port only the change you intend, and keep the `_LIVE_ONLY_CLAUSE` filter on every
`articles` read.
