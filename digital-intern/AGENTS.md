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
  `TestTrainOrchestration` — regression guard that `train()` runs end-to-end on both the
  fresh and disk-cache paths (see ML training pipeline note below) — and
  `TestContinuousLabelSourcing`, which pins the **inlined duplicate** of the
  strong-label SQL inside `train_continuous` (trainer.py ~715). `TestLabelSourcing`
  only covers `_fetch_training_data`; the duplicate is a separate copy on the
  *hotter* path (every 2 min vs 3 min) that can silently drift to match
  `score_source='ml'` rows and reopen the label-feedback loop with no exception
  and a healthy-looking daemon log. Drives the real `train_continuous` (stubbed
  model/embedder, mutation-verified) and asserts an `'ml'` row never reaches
  `model.fit` while synthetic-backtest and `'llm'` rows do — same drift class as
  the dashboard-parity / vendored-`signals.py` cases.
- `test_briefing_boost.py` — `ArticleStore.update_scores_from_labels`, the sole writer of
  `score_source='briefing_boost'` (5h Opus heartbeat → strong training label). Pins the
  `MAX(ai_score, 4.5)` formula (never downgrades a stronger LLM label, never under-labels an
  unscored mention at 0.3), the `score_source` CASE (an `'llm'` row stays `'llm'`; a `None`/`'ml'`
  row becomes `'briefing_boost'`), and backtest isolation on this write path. The
  `test_model_scored_row_promoted_off_ml_into_training_pool` case specifically guards the
  `'ml' → 'briefing_boost'` promotion: the trainer's strong pool excludes `'ml'`, so if the CASE
  ever regressed to preserving any non-NULL source an Opus-curated model row would silently never
  train. Every other case here uses `score_source` of `None`/`'llm'`; this is the only `'ml'`
  exercise.
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
- `test_claude_analyst.py` — `analysis/claude_analyst.py`, the 5h heartbeat
  payload builder (previously zero direct coverage). Pins the three bug classes
  its source comments call out: `_fmt_ticker` must not raise on a present-but-
  `None` ticker/price/pct (the `or` guards, since `dict.get()` only defaults a
  *missing* key); `_build_payload`'s article cap is **60, not 50** (the caller
  prepends up to 2 synthetic snapshot rows to a 50-item top list, so `[:50]`
  silently truncates real articles); and `analyze` returns the
  `[analyst] No response…` sentinel (which `heartbeat_worker` retries on) for
  both a `None` and an empty Claude response, never `None`.
- `test_web_scraper.py` — `collectors/web_scraper.py` pure helpers (previously
  zero direct coverage). Pins `_is_article_url`'s SKIP_PATTERNS denylist and
  the `len(path)>10 and path.count('/')>=2` heuristic, and `_extract_articles`'
  15-char title floor, relative-URL resolution against the base, per-page
  dedup, the `source = "scraped/<netloc>"` tag (ml/features credibility keys
  on it), 200-char title truncation, and graceful `[]` on a parser failure
  (the worker must never raise into the daemon thread).
- `test_seen_db_hardening.py` — fleet-wide parity pin: all **11**
  `data/seen_articles.db` writers (`rss`, `gdelt`, `finnhub`, `polygon`,
  `newsapi`, `sec_edgar`, `massive`, `yahoo_ticker_rss`, `wikipedia`,
  `alphavantage`, `google_news`) must open the connection with the canonical
  `timeout=30` + `PRAGMA journal_mode=WAL` + `PRAGMA busy_timeout=30000`
  hardening (see "shared seen_articles.db" below). Parameterized over every
  collector so the pattern can't silently drift back out on any single one;
  `google_news` is included as the canonical reference (the 76f9baa origin)
  so this file is the single source of truth. Also re-asserts the dedup
  contract survives the change (a written `seen_articles` row is durable
  across connections). Same drift class as the backtest-isolation parity
  suites.
- `test_rss_collector.py` — `collectors/rss_collector.py`, two concerns,
  previously zero direct coverage. (a) The unbounded-hang fix: `_fetch_feed`
  now routes through `requests.get(url, timeout=FETCH_TIMEOUT, headers={UA})`
  + `feedparser.parse(resp.content)` instead of `feedparser.parse(url)`
  (which fetched with **no timeout** — one hung feed pinned a worker
  forever). Pins that the bounded timeout + browser UA are actually passed,
  and that an HTTP error / network exception / missing url degrade to `[]`
  (never raise into the daemon thread). (b) The dedup contract: `collect_rss`
  collapses duplicate `(link,title)` within a pass (`seen_in_run`) and across
  passes (persistent `seen_articles`).
- `test_chat_session_delta.py` — `dashboard/web_server.py::api_chat`'s
  session-delta context block (previously zero chat coverage). Every other
  context stream the chat assembles is a current-state snapshot; this is the
  one "what materially changed since you last looked" view (sub-fetched from
  paper-trader `:8090/api/session-delta`, 4s). Pins via the Flask test client
  (memory: not a `__main__` smoke against a different DB): an ACTIVE payload
  is injected after the PAPER TRADER LIVE STATE block (headline + ranked
  event summaries); an unreachable `:8090` degrades silently — the section is
  omitted and the chat still answers 200 (the sibling sub-fetch contract,
  never raises into chat); a QUIET/NO_DATA window is suppressed (ACTIVE-only,
  matching the unified `:8888` chat's `_fetch_session_delta` so the two
  conversational surfaces stay consistent).

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
label-feedback loop stays closed. **Two code paths enforce this independently:** the strong-label
`WHERE` clause is inlined verbatim in both `_fetch_training_data` (full retrain) and
`train_continuous` (the 2-min fine-tune). They must stay byte-identical — editing one without the
other lets the continuous trainer ingest `score_source='ml'` rows silently. Both are now pinned
(`TestLabelSourcing`, `TestContinuousLabelSourcing`).

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
| `database is locked` retries (on `articles.db`) | High writer contention with `purge_worker`'s `wal_checkpoint(TRUNCATE)`. | `_retry_on_lock` decorator handles 5 attempts with jitter. Persistent failures → check `lock_metrics()`. |
| `[<collector>_worker] error: database is locked; backing off Ns` (on `seen_articles.db`) — a whole collector pass lost per event | **Shared seen_articles.db.** All 11 dedup collectors write the *same* `data/seen_articles.db` file from their own worker threads. A bare `sqlite3.connect()` defaults `busy_timeout=0`, so any transient cross-writer lock raises `OperationalError` immediately; the collector's broad `except` then returns `[]` and the worker trips its 5–300s backoff, dropping the entire fetched batch. `google_news` was hardened first (76f9baa); the other 10 carried the identical bug. | All 11 `_ensure_db` now use the canonical `timeout=30` + `PRAGMA journal_mode=WAL` + `PRAGMA busy_timeout=30000` (mirrors `article_store`/`source_health`). Any new `seen_articles.db` writer MUST copy it. Pinned fleet-wide by `tests/test_seen_db_hardening.py`. |
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

**`_db_path()` freshness fix ported (2026-05-16).** The authoritative copy's
`_db_path()` was existence-first (`USB-if-exists`), so when this daemon falls back to writing the
**LOCAL** copy (USB mount unavailable for writes) the live trader silently read the day-stale USB
mirror while every LOCAL-first surface read fresh news — a split-brain that was *detected* but
never root-fixed. It is now freshness-aware: it picks the candidate whose newest **live** article
(same `_LIVE_ONLY_CLAUSE` so an injected `backtest://` batch can't make a stale mirror win) is most
recent; USB still wins a tie. The resolver (only) was ported into this vendored snapshot; behavioral
parity — fresh-LOCAL beats stale-USB, USB-on-tie, synthetic-row exclusion — is pinned by the two new
cases in `tests/test_paper_trader_signals_isolation.py`. Operator CLI on the authoritative side:
`python3 -m paper_trader.signals --check-freshness` (exit 3 = a stale trader process is reading the
old USB; RESTART it — the on-disk fix only applies on next start).

---

## Review log

- **2026-05-16** — Full review pass over `daemon.py`, `storage/article_store.py`,
  `watchers/alert_agent.py`, `watchers/urgency_scorer.py`, `ml/trainer.py`, `ml/model.py`,
  `ml/features.py`, `ml/inference.py`, `collectors/web_scraper.py`, `analysis/claude_analyst.py`.
  No new bugs. Re-verified the four load-bearing invariants hold and are pinned by tests:
  backtest isolation (every live `FROM articles` read carries `_LIVE_ONLY_CLAUSE` or the inlined
  equivalent; `send_urgent_alert` keeps its `_is_synthetic` defense-in-depth re-filter),
  ml_score/ai_score separation (`update_ml_scores_batch` tags `score_source=COALESCE(...,'ml')`;
  `update_ai_scores_batch` tags `'llm'`; trainer strong-label SQL excludes `'ml'` in both
  `_fetch_training_data` and the `train_continuous` duplicate), the `MAX(urgency, ?)` state
  machine, and `get_unscored` train/serve age-field parity. Suite: **261 passed**
  (verified after a `__pycache__`/`.pytest_cache` clear — a stale assertion-rewrite
  cache reports a lower count, the phantom hazard documented under "Running tests").

- **2026-05-16 (post-`b0f858d`)** — Re-review covering the only production-code change since
  the entry above: `b0f858d` added three `EVENT_PATTERNS` to `triage/heuristic_scorer.py`
  (`distress` bankruptcy/default 2.7, `legal` SEC/DOJ/FTC probe + securities/accounting fraud +
  restatement 2.6, `exec_change` CEO/CFO departure 2.0, both word orders). No new bugs. The
  three regexes are correctly placed *after* the `if kw == 0.0:` early-return, so the multiplier
  only ever scales an already-domain-relevant article up (gate pinned by
  `test_heuristic_scorer.py::test_distress_is_gated_behind_domain_keywords`); residual heuristic
  imprecision (`prob\w+`→"problem", `exits?`→"…exit strategy") is bounded by the `kw>0` gate and
  the `max(event_bonus, multiplier)` ceiling and is *not* a correctness bug — per the standing
  "code is the spec, do not tune heuristics to prose" rule. All four task-critical invariant
  assertions spot-verified present and value-asserting (not no-crash): `get_unalerted_urgent`
  backtest exclusion, `update_ml_scores_batch`→`score_source='ml'`, `EXTRA_FEATURE_DIM == 15`,
  `_fetch_training_data` `score_source='ml'` exclusion. Suite: **265 passed** (`b0f858d` shipped
  +4 dedicated pattern tests; no test gap remained, so none added — adding duplicates would
  violate the no-redundant-coverage discipline). Note: a large unrelated `config/sources.json`
  working-tree delta and two `config/sources.json.bak.*` files predate this session and were
  deliberately **not** committed (config data churn, out of scope for a code-review commit).

- **2026-05-16 (post-`bb1e79c`)** — Re-review covering the four production-code changes since
  the entry above (each shipped with its own dedicated tests; all four correctness-clean):
  `e190e99` `ml/features.py::_parse_published` now normalizes every parsed `published` datetime
  to UTC (`dt.astimezone(timezone.utc)`, naive→UTC assumed) — kills the per-source train/serve
  skew in the 4 cyclic temporal features (a -0500 feed previously produced a different
  `hour_sin`/`dow_sin` for the same instant); `f1d9288` `discord_notifier.send` adds an explicit
  "gave up on chunk … — chunk dropped" log + `_MAX_ATTEMPTS` constant (the 429-storm path now
  reaches it; the definitive-4xx path still `sent=True`/`ok=False`-breaks by design, so no
  re-fire); `ab62331` `heuristic_scorer` multi-catalyst compounding (`+15%`/extra distinct
  category, capped 3.5) — verified placed **after** the `kw==0.0` and blacklist early-returns,
  so it only ever scales an already-domain-relevant article and the `n_distinct==1 → 1.0`
  single-event invariant holds; `76f9baa` `google_news._ensure_db` WAL + `busy_timeout=30000`
  matching the canonical `article_store`/`source_health` hardening (no leaked connection — one
  per `collect_google_news()`, closed in the same call). The four task-critical invariant
  assertions re-spot-verified present and value-asserting (not no-crash): `get_unalerted_urgent`
  `backtest://` exclusion, `update_ml_scores_batch`→`score_source='ml'`, `EXTRA_FEATURE_DIM==15`,
  `_fetch_training_data` `score_source='ml'` exclusion. Suite: **273 passed** (clean
  `__pycache__`/`.pytest_cache`). Known-benign deferral: `datetime.utcnow()` (deprecated in
  modern Python) appears in **12** collector dedup-write sites including `google_news.py:119` —
  it only writes the `seen_articles.first_seen` column, which is **write-only** (the dedup path
  reads `WHERE id=?` exclusively, never parses `first_seen`), so it is not a correctness bug;
  surfaced as a pytest `DeprecationWarning` only because `76f9baa`'s new test exercises the
  write path. A 12-site sweep is cross-cutting churn out of scope for a surgical review commit
  (same disposition as the prior config-churn deferral) — flagged here so the next reviewer
  doesn't re-derive it. The `config/sources.json` delta + `.bak` files still predate the session
  and remain deliberately uncommitted.

- **2026-05-16 (seen_articles.db fleet hardening)** — Full review pass over `daemon.py`,
  `storage/article_store.py`, `watchers/alert_agent.py`, `watchers/urgency_scorer.py`,
  `ml/trainer.py`, `ml/model.py`, `ml/features.py`, `ml/inference.py`,
  `collectors/web_scraper.py`, `analysis/claude_analyst.py`. The four load-bearing invariants
  re-verified present and value-asserting (backtest isolation, ml_score/ai_score separation,
  `MAX(urgency,?)` state machine, `get_unscored` age-field parity) — no new bugs in those.
  **One real systemic bug found and fixed:** the working-tree `rss_collector.py` change
  (`feedparser.parse(url)` → `requests.get(timeout=FETCH_TIMEOUT, UA)` + `parse(resp.content)`,
  a correct unbounded-hang fix) drew attention to `_ensure_db`, which was still the **bare**
  `sqlite3.connect()` pattern. Audit showed **10 of 11** collectors that share the single
  `data/seen_articles.db` file (`rss`, `gdelt`, `finnhub`, `polygon`, `newsapi`, `sec_edgar`,
  `massive`, `yahoo_ticker_rss`, `wikipedia`, `alphavantage`) carried the identical
  `busy_timeout=0` bug that 76f9baa fixed for `google_news` alone — i.e. the canonical
  hardening was applied to one collector while the *shared-file contention* it defends against
  is fleet-wide (rss is the hottest writer at 30s). This is **not** the `datetime.utcnow()`
  12-site deferral class (that was write-only, benign): this drops whole fetched batches on any
  transient cross-writer lock and trips the worker backoff. Ported the canonical `timeout=30` +
  `WAL` + `busy_timeout=30000` verbatim to all 10 (no happy-path behavior change; external
  reader sweep confirmed nothing outside `collectors/` reads `seen_articles.db`). Added
  `tests/test_seen_db_hardening.py` (parameterized fleet-wide pin, all 11 incl. google_news as
  source-of-truth reference) and `tests/test_rss_collector.py` (the requests/UA/timeout fix +
  the in-run/cross-run dedup contract — both previously zero-coverage). Suite: **302 passed**
  (275 prior + 27 new; clean `__pycache__`/`.pytest_cache`). Known-benign deferral unchanged:
  `datetime.utcnow()` write-only sites (now surfaced as a `DeprecationWarning` from the new
  rss test exercising the dedup write — same disposition as the documented 12-site sweep, not
  a correctness bug). `config/sources.json` delta + `.bak` files still predate the session and
  remain deliberately uncommitted (config churn, out of scope for a code-review commit).

- **2026-05-16 (datetime.utcnow() deferral retired)** — The standing `datetime.utcnow()`
  write-only deferral (re-derived and re-shelved across the two prior entries) is now
  **resolved as its own focused commit**, which is the correct vehicle for it (it was only
  ever "out of scope for a *review* commit", never wrong to do). All 12 sites across 10
  collector modules (`yahoo_ticker_rss`, `google_news`, `polygon`, `massive`, `alphavantage`,
  `rss`, `wikipedia`, `finnhub`, `newsapi`, `sec_edgar` ×3 incl. the 2 non-DB EFTS date-range
  params) migrated `datetime.utcnow()` → `datetime.now(timezone.utc)`; `timezone` added to the
  8 imports that lacked it (`finnhub`/`newsapi` already had it). Safety re-verified
  *independently of the prior AGENTS claim*: a full-tree `first_seen` grep confirms
  `seen_articles.first_seen` has **zero** read/parse sites (every reference is `CREATE TABLE`
  or `INSERT`; dedup is `WHERE id=?` only) — the parsed `first_seen` consumers
  (`paper_trader/signals.py:51`, dashboard `>= datetime('now',…)`, SQL range filters) all read
  `articles.first_seen`, written by `storage/article_store.py`, untouched here. The new
  `+00:00`-bearing aware ISO format is therefore unobservable to any consumer; pinned anyway
  via `tests/test_collector_tz_aware.py` (10 parametrized static no-`utcnow` guards + a
  round-trip format assertion through the canonical `signals._age_hours` parse expression).
  `sec_edgar`'s `.date().isoformat()` EFTS params verified format-identical
  (`datetime.now(timezone.utc).date()` == `datetime.utcnow().date()`). Concrete pass
  criterion met: the `DeprecationWarning` the prior entry flagged from
  `test_rss_collector.py`'s dedup-write path is **gone** under `-W error::DeprecationWarning`.
  Suite: **313 passed** (302 prior + 11 new). This deferral is now closed — a future reviewer
  should *not* re-derive it.

- **2026-05-16 (independent full re-review @ `d847789`)** — Fresh end-to-end pass over the
  nine task-critical files (`daemon.py`, `storage/article_store.py`, `watchers/alert_agent.py`,
  `watchers/urgency_scorer.py`, `ml/trainer.py`, `ml/model.py`, `ml/features.py`,
  `collectors/web_scraper.py`, `analysis/claude_analyst.py`) plus `ml/inference.py`. **No
  bugs found.** HEAD is `d847789` — *identical* to the commit the entry above closed, i.e.
  **zero production-code delta** since the last review (`git diff d847789` is only
  `config/sources.json` data churn + `logs/daemon.log.*` rotation; the standing config/`.bak`
  deferral is unchanged and remains deliberately uncommitted — config data, out of scope for
  a code-review commit). All four load-bearing invariants independently re-traced and hold:
  (1) backtest isolation — every live `FROM articles` read carries `_LIVE_ONLY_CLAUSE` or its
  inlined twin (`article_store` get_unscored/get_unalerted_urgent/get_top_for_briefing/
  count_unscored/stats/stats_since/update_scores_from_labels; `recursive_labeler._fetch_round1_candidates`
  separate-WHERE form; `dashboard/server.py`; vendored `paper_trader/signals.py`), and
  `alert_agent._is_synthetic` keeps its defense-in-depth re-filter; training paths
  (`_fetch_training_data`, `train_continuous`, `_fetch_briefing_samples`) intentionally omit
  it; (2) ml_score/ai_score separation — no code path routes model output into `ai_score`
  (`scorer_worker` + `score_pending` → `update_ml_scores_batch`; Sonnet/Opus →
  `update_ai_scores_batch`/`update_scores_from_labels`); (3) `MAX(urgency, ?)` state machine
  intact on every score-write; (4) `get_unscored` train/serve age-field parity intact. The
  task-specified test assertions were checked present **and value-asserting** (not no-crash)
  and **already exist** — no tests added (adding duplicates would violate the standing
  no-redundant-coverage discipline): `test_article_store.py`
  (`test_get_unalerted_urgent_excludes_backtest_urls`, `test_mark_alerted_removes_from_unalerted`,
  `TestScoreSourceSeparation` ml-vs-llm, CRUD), `test_urgency_scorer.py` (score 9.5 urgent /
  3.0 not / alerted-state preserved), `test_features.py` (`EXTRA_FEATURE_DIM == 15`, zero
  ticker density, days-since-published), `test_model.py` (relevance∈[0,10], urgency∈[0,1], no
  NaN on zero input), `test_trainer.py` (`score_source='ml'` excluded, high-rel weighted
  harder). **Spec-vs-prose note for the next reviewer:** the brief asks for
  `days_since_published` ≈ "1 for one published 24h ago" — that contradicts the *intended* ML
  scaling. Feature 6 is `min(age_days,30)/30`, so 24h ≈ **1/30 ≈ 0.033**, saturating 1.0 only
  at ≥30d. `test_days_since_published_grows_with_age` correctly asserts ~1/30; this is
  documented scaling, **not a bug — do not "fix" code or test to the prose** (standing
  "code is the spec" rule). Suite: **313 passed** (clean `__pycache__`/`.pytest_cache`),
  imports OK.

- **2026-05-17 (Agent 4, feature-dev — session-delta surfaced on chat + landing)** —
  Shipped the two deferred high-value increments from
  `docs/superpowers/specs/2026-05-16-session-delta-design.md`'s "Out of scope" list.
  Both reuse the already-tested `paper_trader/analytics/session_delta.py` builder +
  its `:8090/api/session-delta` endpoint (no core change), additive, never gate Opus.
  **(B, this repo)** `dashboard/web_server.py::api_chat` gained a `session_delta_block`
  sub-fetch (`:8090/api/session-delta?minutes=360`, 4s) injected after the PAPER
  TRADER LIVE STATE block — the only temporal-change stream in an otherwise
  all-current-state context. Mirrors the existing greeks/analytics/heatmap/earnings
  siblings *verbatim* (network-guarded, never raises into chat; a missing-webhook /
  unreachable `:8090` degrades to section-omitted). ACTIVE-only, matching the unified
  `:8888` chat's `_fetch_session_delta` so the two conversational surfaces stay
  consistent. New `tests/test_chat_session_delta.py` (4 cases, Flask test client) —
  the chat had zero prior coverage. **(A, local-only `/home/zeph` repo)** the
  `:8888` command-center landing card (the spec's named follow-up) — `/api/session-delta`
  added to `_build_command_center`'s fan-out + SWR payload, a `#sess-card` mirroring
  the `:8090` palette, degraded-upstream surfaced honestly (never a faked QUIET).
  Suite: **317 passed** (313 prior + 4 new; clean caches), imports OK.
  *Operational:* digital-intern `:8080` will not serve the chat block until
  `systemctl --user restart digital-intern` (the chronic-stale pattern); `:8090`
  `/api/session-delta` is current so the `:8888` card renders live now.
  *Pre-existing, not this work:* the `/home/zeph` `tests/test_unified_dashboard.py`
  suite has 2 failures (`test_decision_health_alerts_above_threshold`,
  `test_aq_decision_health_alert_exact_numbers`) — the decision-health `", 24h window"`
  string is committed at HEAD but those 2 tests were not updated by whoever shipped
  it; my session-delta diff contains zero decision-health hunks (verified). Left for
  that change's owner per the standing "don't weaken another change's tests" rule.
