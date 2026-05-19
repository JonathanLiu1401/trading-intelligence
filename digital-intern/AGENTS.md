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
min. Discord alerts fire on state transitions only. **Caveat (load-bearing):** the supervisor only
*respawns* threads that have **exited** (`if t.is_alive(): continue`). A worker that is *alive but
wedged* (blocked indefinitely on the shared `_store_lock` / sqlite `busy_timeout` under heavy
lock-contention) is flagged DEAD in `logs/supervisor_state.json` but is **never respawned and only
WARNING-logged** — observed live 2026-05-18 (the `alert` worker hung 25+ min, daemon otherwise
healthy, analyst got zero indication breaking-news delivery had stopped).

External watchdog (independent of the daemon, so it survives a wedged supervisor):
```sh
python3 scripts/alert_pipeline_watchdog.py            # check once + escalate to Discord
python3 scripts/alert_pipeline_watchdog.py --dry-run  # print, do not post
```
It reads only `logs/supervisor_state.json` (+ its own throttle file — DB-free, no invariant
surface) and pages Discord when a critical worker (`alert`/`scorer`/`heartbeat`) is DEAD/hung or
the snapshot itself is missing/stale (daemon down or crash-looping). Run it on a ~2-5 min cron /
systemd-timer cadence. Pure `evaluate()` core is unit-tested in
`tests/test_alert_pipeline_watchdog.py`.

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
  alerted-state preservation. `TestArticleAgeCascade` pins
  ``_article_age_hours``'s field-cascade contract (a non-empty-but-
  unparseable ``published`` must NOT short-circuit at 0.0h — that bypassed
  the STALE_HOURS=48 cap on rows whose ``first_seen`` was genuinely old;
  the cascade now mirrors ``alert_agent._article_age_hours``'s convention
  so the two age helpers agree on which timestamp is authoritative).
- `test_alert_recap_template.py` — the recap / SEO template gate
  (``watchers/alert_agent.py::_filter_recap_template_noise``). A second,
  distinct surface the urgency head over-scores that neither the quote-
  widget gate nor the 0.45 source-authority bar catches: retrospective
  recap / preview / transcript-summary templates from publishers ABOVE
  the cred bar (Finnhub 0.78, Motley Fool/yahoo 0.65, GoogleNews 0.62).
  Six precision-anchored fingerprints: "Why <X> Stock Is Trading Up
  Today" (Zacks/Yahoo/Finnhub), "Why Did <X> Stock Drop Today" (Motley
  Fool variant), "Stock Market Today, May 18: ..." dated wrap-up, "Q1
  2026 Earnings Call Highlights" (transcript-summary), "Here What the
  Street Thinks About ..." (InsiderMonkey), "GF Value Says" (GuruFocus
  algorithmic mill). Runs BEFORE dedup so a syndicated recap is caught
  on every copy (live: a single "Stock Market Today, May 18: ..." wrap-
  up fired three times in one minute from Motley Fool + Nasdaq +
  YahooFinance). Suppressed rows are marked alerted UNCONDITIONALLY
  (exit the urgent queue, never re-fetched). Tests pin (1) live-noise
  catches by the exact strings observed firing 2026-05-18/19, (2) the
  must-survive corpus (real earnings, macro breaks, ticker action, mid-
  sentence "why", earnings PREVIEWS that the call-highlights pattern
  must NOT catch, value/analyst headlines that the GF Value pattern
  must NOT catch), (3) integration on ``send_urgent_alert`` with the
  no-Claude-call short-circuit + cross-gate chaining vs the quote-
  widget gate. 20 cases.
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
- `test_chat_behavioural_enrichment.py` — `dashboard/web_server.py::`
  `_behavioural_chat_lines`, the pure helper backing the `/api/chat`
  behavioural-diagnosis block. The chat already surfaced the trader's
  **raw** `/api/analytics` stats; this composes the bot's **synthesized
  self-review verdicts** (`/api/scorecard` + `/api/capital-paralysis` +
  `/api/churn`) so a "why is my bot losing money?" question gets the
  diagnosis the bot itself produced. The discriminating lock is
  **verbatim composition** (paper-trader invariant #10 — single source
  of truth): each builder's own `headline` / `focus["headline"]` /
  `flags[i]` / `recommended_unlock["reason"]` must appear UNCHANGED in
  the output (an inline re-derivation that drifts from the trader
  endpoint fails loud — the `test_risk_mirror` precedent). Also pins the
  `▶ PRIORITY` precedence (paralysis-unlock ≻ scorecard-focus ≻
  churn-CHURNING ≻ none), the 3-flag cap, and the total/pure degrade
  contract (non-dict / `{"error":…}` / missing-`state` / `NO_DATA` →
  that input drops, all three absent → `[]`, never an exception into
  chat — the `_tail_risk_chat_lines` sibling contract). 12 cases, no
  Flask/DB/cross-fetch needed.
- `test_chat_actionable_enrichment.py` — three more pure `/api/chat`
  helpers in `dashboard/web_server.py` (2026-05-18, Agent 4 feature-dev).
  **`_paper_trader_position_lines`** fixes the live-trader position block:
  it now reads the **marked** `portfolio.positions` array (real `pl_pct`
  + `stale_mark`) instead of the raw top-level `positions` array
  (`store.open_positions()`, neither key). Two discriminating locks: the
  **always-(0.0%) bug** — the raw array has no `pl_pct`, so the prior
  inline `(p.get('pl_pct') or 0)` printed `(0.0%)` for every stock
  regardless of P/L (a real `-1.04%` must surface); and the **stale-mark
  misread** — a failed price lookup (`stale_mark=True`, `current_price ==
  avg_cost`, P/L $0.00) looks identical to a flat position, so the chat
  (the user's primary surface) now annotates it, mirroring the trader
  prompt's `[STALE MARK …]` suffix (strategy.py) and the reporter's
  `⚠ STALE` — both already shipped for this exact live MU pathology;
  falls back to the raw array when the marked one is empty (degraded
  `get_portfolio()`) so a store blip never loses the book.
  **`_game_plan_chat_lines`** surfaces the trader's own prioritised
  next-session plan (`/api/game-plan`) and **`_hold_discipline_chat_lines`**
  the disposition-trap verdict (`/api/hold-discipline`) — the chat's first
  "what should I actually do" inputs (every prior block is descriptive
  state). Both compose the builder `headline` / HIGH-directive `text`
  **verbatim** (invariant #10 — an inline re-derivation that drifts from
  the trader endpoint fails loud); `_hold_discipline_chat_lines` mirrors
  `reporter._hold_discipline_line` exactly (emit only on
  `DISPOSITION_DRAG`; `DISCIPLINED`/`INSUFFICIENT`/`NO_DATA` → silence).
  All three obey the `_tail_risk_chat_lines` total/pure degrade contract
  (non-dict / `{"error":…}` / missing-`state` / `NO_DATA` → that input
  drops, never an exception into chat). 15 cases, no Flask/DB/cross-fetch.
- `test_heartbeat_cadence.py` — `daemon._initial_heartbeat_last`, the
  restart-resilient briefing-clock seed (see "5h heartbeat briefing posts
  30–40h apart" failure mode). Drives the real `save_briefing →
  get_briefings_for_training` path (not hand-built dicts) and reads
  `HEARTBEAT_INTERVAL`/`HEARTBEAT_RESTART_WARMUP_SECS` from the live module so
  a retune can't false-fail it: no-briefing/unparseable/future-ts → `now`
  (original wait-a-full-interval behaviour preserved on first-ever launch);
  a 1h-ago briefing → waits the remainder (no immediate fire on restart); a
  40h-ago overdue briefing → seeded to fire after the warm-up, asserted
  exactly `now - last == HB - WARMUP` (neither instant nor a full interval);
  id-DESC newest-row-wins; store-raises → `now` (never crashes the worker at
  startup).
- `test_source_health_briefing.py` — `daemon._format_source_health_summary`
  + the `_build_health_line` integration (the new "Sources down (N): …" line
  in the 5h Discord briefing). Exact-string pins on the compact deterministic
  formatter: empty when healthy, disabled sorted, stale de-duplicated against
  disabled, disabled-listed-before-stale (the union is NOT globally sorted),
  `+N` overflow truncation, the hard `max_chars` cap with `…`; and that
  `_build_health_line` appends the line only when something is down and
  degrades to workers-only (never raises) on a `source_health` probe error.
- `test_alert_source_authority.py` — the **third** formatter-side
  defense-in-depth filter on `watchers/alert_agent.py::send_urgent_alert`
  (after `_is_synthetic` and `_article_age_ok`): `_filter_low_authority_lone`.
  A LONE, un-corroborated social/forum row — `cred <
  ALERT_MIN_LONE_SOURCE_CRED` (0.45) via the **reused**
  `ml.features._source_credibility` word-boundary map (reddit/nitter 0.40,
  twitter 0.35, stocktwits 0.30) and `dup_count<=1` — is suppressed: no
  Claude/Discord call, marked `urgency=2` UNCONDITIONALLY (a separate call,
  before the Discord attempt, regardless of its outcome) so it exits the
  urgent queue instead of re-firing every 20s, and `send_urgent_alert`
  returns False. The **corroboration escape valve** is pinned at both the
  pure-helper and end-to-end level (a refactor that moves the gate *before*
  `dedupe_urgent` loses it and is caught): a story syndicated across ≥2
  sources (`dup_count>1`) **or** any credible/UNKNOWN source
  (`DEFAULT_SOURCE_CRED=0.55` ≥ threshold) still fires. The mixed-batch
  Discord-failure case pins that suppressed noise stays marked while a kept
  row stays `urgency=1` (the existing re-queue-on-failure contract is
  preserved alongside the new gate). Same `_is_synthetic`-class discipline;
  none of the four load-bearing invariants are touched (read-only on the
  alert path — `ai_score`/`ml_score`/`score_source`/backtest isolation all
  unchanged; `urgency=2` is only ever otherwise read by the synthetic-breach
  detector, which is scoped to synthetic rows this gate never reaches).

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
| 5h heartbeat briefing posts 30–40h apart (or never) while the daemon looks healthy | **Restart-churn reset the briefing clock.** `heartbeat_worker` seeded `last = time.time()` on every start; under the documented OOM-restart churn (hundreds of starts/day) any restart < 5h after launch pushed the next briefing out another full interval, starving the analyst's scheduled digest. No error — the worker pings healthy the whole time. | `_initial_heartbeat_last` now seeds `last` from the most recent persisted `briefings.ts` (with a startup warm-up clamp); a restart no longer resets the cadence. Falls back to the original "wait a full interval" when no briefing exists / ts unusable. Pinned by `tests/test_heartbeat_cadence.py`. Note this is a *symptom* of the OOM-restart churn (1.4 GB USB DB + bulk `gdelt_historical` backfill + WAL-checkpoint contention → frequent `insert_batch: lock retry exhausted` ERRORs and OOM-kills); the churn root cause is operational, out of scope for a code fix. |
| Whole collected batches silently lost; `[article_store] insert_batch: lock retry exhausted after 5 attempts — raising` ERRORs (also `update_ml_scores_batch`/`update_ai_scores_batch`) | Sustained writer contention on the 1.4 GB USB `articles.db` (many collector threads + a ~1.3M-row `gdelt_historical` bulk backfill draining through the scorer + `purge`'s `wal_checkpoint(TRUNCATE)`) outlasts the 5-attempt / ~10s `_retry_on_lock` budget. `_ingest` propagates the raise → the collector worker's broad `except` drops the entire fetched batch and backs off. | Operational, not a clean surgical fix (retry-then-raise is the intended contract; bumping `_LOCK_RETRY_ATTEMPTS`/`_CAP_S` has no correctness story). Mitigations: reduce the bulk-backfill insert rate, move `articles.db` off the USB spindle, or lower `purge` checkpoint contention. Tracked as a Phase 3 finding. |
| 5h Opus briefing reads as a repetitive low-signal digest — one scrape channel monopolises it (live: 10/50 slots `scraped/finance.yahoo.com` price-quote widget pages, `ETH-USDEthereum USD2,169.83` ML-scored 9.96 = #1 slot) | A single high-volume publisher domain dominates `get_top_for_briefing`'s score-ordered top-N because the ML relevance head over-scores ticker-dense quote-widget scrape pages. | `get_top_for_briefing` caps any one resolved publisher domain at `BRIEFING_MAX_PER_DOMAIN` (6) via `_briefing_domain_key`, backfilling from score-ordered overflow so the digest is **never shrunk** (low-diversity windows still fill). Pure read-side; `_LIVE_ONLY_CLAUSE` intact. Pinned by `tests/test_briefing_domain_diversity.py`. NOTE the underlying cause — `collectors/web_scraper.py` ingesting Yahoo/Finviz quote pages as articles, and the alert path resolving lone `scraped/finance.yahoo.com` to cred ~0.65 (> the 0.45 lone-alert gate) so it can still fire a real BREAKING — is a separate, unaddressed concern. |

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

- **2026-05-17 (Agent 3, hybrid debug+feature+live-validation)** — Full read
  pass over the nine task-critical files + `ml/inference.py` and the small
  core modules (`json_extract`, `retrain_guard`, `backoff`, `alert_dedup`,
  `embedder`, `heuristic_scorer`). The four load-bearing invariants re-traced
  and hold; no new bug found *by inspection* in the heavily-reviewed core
  paths. **Live validation surfaced the real defect:** the `briefings` table
  showed actual 32h and 41h gaps between heartbeat posts vs the 5h target,
  and the rotated logs showed the daemon restarting every 7–28 min (427
  starts in one log) under OOM-restart churn. Root cause: `heartbeat_worker`
  seeded its clock to `time.time()` on every start, so restart-churn starved
  the analyst's scheduled digest for 30+h at a time — healthy-looking the
  whole time (no error; the worker pings alive). **Phase 1 fix (`ef839a8`):**
  `daemon._initial_heartbeat_last` seeds `last` from the most recent
  persisted `briefings.ts` with a startup warm-up clamp; original
  wait-a-full-interval behaviour preserved on first-ever launch / unusable
  ts. The `save_briefing`-runs-even-on-Discord-failure path means a webhook
  outage now costs one skipped 5-min retry instead of many starved briefings
  — an intentional, strictly-better trade (commented in-code). +7 tests
  (`test_heartbeat_cadence.py`, real store path, live constants). **Phase 2
  feature (`c2fa61a`):** `_format_source_health_summary` adds a compact,
  deterministic, char-capped "⚠ Sources down (N): …" line to the 5h Discord
  briefing — 6 collectors incl. `sec_edgar` (8-K filings, high signal) were
  observed disabled in production while the briefing health line, which only
  reported four worker threads' liveness, said nothing. Additive, read-only,
  zero `articles`-table / `ai_score`/`ml_score`/`score_source` impact (all
  four invariants preserved). +9 tests (`test_source_health_briefing.py`,
  exact strings). **Phase 3 findings reported (not fixed — operational /
  out of surgical scope):** (a) DB write-lock exhaustion — 46 `insert_batch:
  lock retry exhausted` tracebacks/log dropping whole fresh-article batches
  under 1.4 GB-USB-DB + ~1.3M-row `gdelt_historical` backfill + checkpoint
  contention; (b) low-authority urgent alerts — Wikipedia recent-change
  ("[Wikipedia] Nvidia RTX", `ml_score=8.63`) and Reddit posts fired as
  urgent Bloomberg alerts (model over-scores; urgency thresholds are
  well-pinned, changing them is out of surgical scope); (c) `gdelt_historical`
  bulk backfill counts as live (1.29M-row unscored backlog) but is defused
  for briefings/alerts by the staleness filters and `kw_score DESC` scoring
  order — observation, not a code bug. **Positive:** the actual latest
  briefing read end-to-end is a genuinely accurate, coherent Bloomberg-style
  digest; scorer keeps up (batch=1000 scored=1000/cycle, high-kw first);
  ml_trainer healthy (n=22500, val_loss ≈ 2.75–2.80); alert syndication
  dedup working; backtest isolation holding (429k synthetic rows correctly
  excluded from every live count/alert checked). Suite: **333 passed**
  (317 prior + 7 + 9 new; clean `__pycache__`/`.pytest_cache`), imports OK.
  *Pre-existing, not this work:* the `logs/.supervisor_state.*.tmp`
  deletions and `paper-trader/*` working-tree changes predate the session
  and were deliberately left unstaged.

- **2026-05-17 (Agent 3, hybrid debug+feature+live-validation, v2)** —
  Inspection of the nine task-critical files + `ml/inference.py` /
  `alert_dedup.py` again surfaced **no new bug in the heavily-reviewed
  core** (the four load-bearing invariants re-traced and hold). Per the
  established pattern, **live validation was the discovery engine** and
  produced both Phase 1 fixes:

  **Phase 1 — two real bugs, both invisible to inspection-only review:**
  1. `db9635e` **`core/logger.py` daemon.log timestamps were local time
     mislabeled `Z` (UTC).** The plain `daemon.log` `RotatingFileHandler`
     used `logging.Formatter(datefmt="%Y-%m-%dT%H:%M:%SZ")` but left
     `Formatter.converter` at the Python default `time.localtime`; the
     literal `Z` *asserted* UTC while `%(asctime)s` rendered local — a
     host-TZ-dependent constant skew (reproduced: **-7h** on this PDT host;
     a briefing logged `06:26:38Z` whose `briefings.ts` row said
     `13:26:38`). `healthcheck.sh` greps this file and operators/prior
     agents correlate it against the UTC-correct `structured.jsonl` /
     `briefings` table / Discord alerts, so every cross-sink time
     correlation was silently wrong while each line looked plausible. The
     console (`_ColourFormatter`) and `structured.jsonl` (`_JSONLHandler`)
     sinks already used `datetime.now(timezone.utc)` and were unaffected —
     this is why prior reviews (which read the UTC-correct sinks and never
     hit `core/logger.py`, not in the 9-file list, bug only manifests when
     host TZ ≠ UTC) missed it. Fix: extracted `_plain_file_formatter()`
     with `converter=time.gmtime`. Pinned by
     `tests/test_logger_utc_timestamp.py` (fixed-epoch + converter-identity,
     host-clock-independent).
  2. `b4be1ca` **`dashboard/web_server.py::_articles_from_db` raced the
     shared writer connection.** `run_server` runs `app.run(threaded=True)`
     but the endpoint queried `store.conn` — the *single*
     `sqlite3.Connection` the daemon's ~30 writer threads share
     (`check_same_thread=False`). sqlite3 connections are not safe for
     concurrent use: a dashboard read racing a writer's implicit
     `conn.execute("SELECT changes()")` inside `insert_batch` returned a
     wrong-shaped 1-tuple where the 9-column row was expected, so
     `ai = float(r[6] or 0)` raised `IndexError`. `IndexError` is not a
     `sqlite3.Error`, so the endpoint's `except sqlite3.Error: return []`
     did not absorb it and `/api/articles` 500'd — **observed 10× in
     `logs/daemon.log`** (the threaded Flask server, `d5b8eac`, made it
     manifest). Fix: read via a dedicated short-lived `mode=ro` connection
     (`_ro_query`) — lock-free WAL reads fully isolated from the writer
     connection's cursor state, one connection per call (inherently
     thread-safe, sub-ms to open), never competes for the daemon write
     lock. Backtest isolation + effective-score derivation preserved.
     Pinned by `tests/test_dashboard_articles_conn_isolation.py` (poisons
     `store.conn` with the exact interleave shape; reproduces the prod
     traceback line-for-line on the unfixed code). NOTE for the next
     reviewer: the *same* shared-connection race exists on every other
     `store.conn.execute()` read in `dashboard/web_server.py` (`api_stats`
     → `store.stats()`; the two `PRAGMA database_list` reads — the latter
     two are `except Exception`-guarded so they degrade silently, not
     crash). Only `_articles_from_db` was *observed* crashing (raises
     `IndexError`, uncaught); the architectural fix for the rest is the
     same `_ro_query` pattern but was left out of this surgical commit.

  **Phase 2 — adaptive briefing lookback + coverage-gap banner
  (`79a4553`).** Directly motivated by the Phase 3 finding below
  (briefing starvation). Three pure helpers (`_briefing_gap_hours`,
  `_briefing_lookback_hours`, `_coverage_gap_banner`) +
  `heartbeat_worker` wiring: a restart-starved briefing now widens its
  article lookback from a stale 5h to span the real gap (hard-capped at
  24h == the ceiling `get_top_for_briefing` already enforces via the
  published-staleness filter, so no new stale-news risk) and prepends a
  one-line "⚠ COVERAGE GAP: first briefing in Nh …" warning so the analyst
  knows the digest covers a backlog, not the usual 5h window. **Healthy
  cadence is byte-identical to before** (gap ≤ 5h or unknown → 5h window,
  empty banner). Banner is Discord-only — never folded into the saved
  briefing text, so it can't reach the trainer's title-prefix label scan
  (same discipline as the source-health line). All four invariants
  untouched (this path writes no articles / ai_score / ml_score /
  score_source; reads only the `briefings` table). Pinned by
  `tests/test_briefing_coverage_gap.py` (7 cases, live constants).

  **Phase 3 — live findings (read-only DB probes + log forensics):**
  1. **Briefing starvation persists.** `briefings` table: id20→21 = 41.2h,
     id21→22 = 31.9h gaps vs the 5h target; latest pair id22→id23 ≈ 6.3h
     (partially recovered post-`ef839a8`). The heartbeat *code* fix is
     correct; the residual cause is OOM-restart churn (24 `DAEMON —
     STARTING` in one log window) + the USB-DB I/O saturation below —
     operational, out of surgical scope. Phase 2 mitigates the *consumer
     impact* (honest + full-backlog coverage), not the churn root cause.
  2. **USB `articles.db` I/O saturation is severe and active.** The DB is
     1.40 GB with a ~1.44M-row `gdelt_gkg/*` bulk-backfill spike (organic
     live rate is healthy ~235/h, diverse sources). Read-only probes —
     even *indexed* `COUNT(urgency=?)` — block in `D` state and time out
     >90s. **57 of 71 `daemon.log` ERRORs are `lock retry exhausted`**
     (`insert_batch` 46, `update_ml_scores_batch` 6,
     `update_time_sensitivity_batch` 2, `update_ai_scores_batch` 2,
     `purge_old` 1) → whole collected batches silently dropped during
     contention. Same documented operational issue; still unresolved.
  3. **6 collectors disabled:** `sec_edgar`, `sec_edgar_ft`, `polygon`,
     `newsapi`, `massive`, `nitter` (`source_health`, `stale`=∅).
     `sec_edgar`/`sec_edgar_ft` are high-signal (8-K material-event
     filings) — correctly surfaced in the 5h briefing via the prior
     agent's source-health line (feature working as intended). Upstream /
     rate-limit driven; operational.
  4. **Duplicate `daemon.py` processes** (pid 1161902 active; pid 1163179
     ppid=1 blocked on the singleton `flock`). This is the *designed*
     duplicate-handling (blocking flock), not a bug, but the duplicate-
     launch condition recurs (documented dual-systemd-unit / restart-churn
     interaction).
  5. **Positive validation.** Latest briefing (id=23) read end-to-end is a
     genuinely accurate, dense, actionable Bloomberg-style digest (sharp
     LEAD, real MACRO/PORTFOLIO/SEMIS numbers, specific DESK NOTE levels);
     the 4 urgent classifications + 6 BN alerts in the 24h window are all
     legitimate and portfolio-relevant (HBM4/Samsung 50k-worker strike,
     NVDA 8-K, MU premarket) with **no Wikipedia/Reddit low-authority
     noise** (the prior agent's open concern did not reproduce this
     window); backtest isolation holding on every live surface checked.

  Suite: **343 passed** (333 prior + 2 logger + 1 dashboard + 7
  coverage-gap; clean run), imports OK. *Pre-existing, not this work:* the
  `logs/.supervisor_state.*.tmp` deletions and `paper-trader/*` working-tree
  changes (incl. concurrently-staged paper-trader commits from a sibling
  agent) predate / are outside this session and were never staged by it —
  every commit here was pathspec-scoped to exactly its 2 intended files.

- **2026-05-17 (Agent 3, hybrid debug+feature+live-validation)** — Full
  read pass over the nine task-critical files + `ml/inference.py`,
  `alert_dedup`, `source_health`. **Phase 1: bugs_fixed=0 (honest, not a
  miss).** The four load-bearing invariants re-traced and hold; the
  task-specified test assertions already exist and value-assert (per the
  prior log entries + an independent advisor confirmation) — adding
  duplicates would violate the standing no-redundant-coverage discipline.
  No Phase 1 commit (correctly per the guard). **Phase 2 feature
  (`31dea26`):** `watchers/alert_agent.py::_filter_low_authority_lone` — a
  source-authority gate so a LONE, un-corroborated social/forum post
  (reddit/nitter/twitter/stocktwits, `cred<0.45`) the ML urgency head
  over-scored can no longer fire a standalone Bloomberg "🚨 BREAKING"
  alert. Formatter-side defense-in-depth (same shape as `_is_synthetic`/
  `_article_age_ok`, **not** an ML-threshold change — distinct from the
  prior agent's "thresholds out of scope" deferral); runs after
  `dedupe_urgent` so `dup_count>1` corroboration / any credible-or-unknown
  source is the escape valve; suppressed rows stay in `articles.db`
  (training/scoring untouched) and remain Opus-briefing-eligible — only the
  noisy push is dropped, and they are marked `urgency=2` unconditionally so
  they leave the urgent queue. All four invariants preserved. +7 tests
  (`test_alert_source_authority.py`, pure-helper + end-to-end + the
  Discord-failure re-queue contract). Clean full suite **343 passed** (no
  regressions; the new tests offset the excluded count). **Phase 3 findings
  (reported, not fixed):** (a) the live noise this targets is **confirmed**
  — reddit/r/Daytrading + reddit/r/ValueInvesting fired BREAKING solo in a
  24h window; **partial-fix honesty:** the gate captures the social tier
  (<0.45) but Wikipedia (0.60) and `yfinance/Insider Monkey` (0.65) are
  *above* the threshold and also fired solo in that window — still ungated
  (raising the bar to catch them would also catch gdelt 0.58 / scraped
  0.50, a more debatable call deliberately left out of this surgical
  commit). (b) **`export_worker: database disk image is malformed`** —
  recurring every ~30 min in the live daemon (06:41Z, 07:11Z); the USB
  `training_data.json.gz` (paper-trader's backtest fallback) is going
  stale. Also surfaces as 2 failing tests in *pre-existing, not-mine*
  in-flight work (`scripts/export_training_data.py` modified + untracked
  `tests/test_export_training_data.py`):
  `test_export_self_heals_corrupt_destination` shows the modified export
  script raises instead of self-healing a corrupt destination — a real bug
  in a sibling agent's uncommitted change, left untouched per the
  don't-stage-others'-work rule. (c) **~17 RSS feeds permanently dead**
  (404/403) including the portfolio-relevant semiconductor IR feeds
  (ASML/Lam/KLA/Qualcomm/TSMC) — config churn, out of surgical scope.
  (d) 6 collectors disabled in production (polygon, newsapi, sec_edgar,
  sec_edgar_ft, nitter, massive) — `sec_edgar`/`_ft` are high-signal 8-K
  filings; surfaced to the analyst via the prior agent's source-health
  briefing line (working as intended), upstream/operational. (e) restart
  churn persists (operational; symptom addressed by `ef839a8` + the
  in-flight briefing-coverage-gap change). **Positive:** the latest 5h
  briefing (2026-05-17 13:41 UTC) read end-to-end is a genuinely accurate,
  dense, actionable Bloomberg digest (sharp LEAD, real numbers, MU $700
  DESK-NOTE level); recent cadence healthy (~6.3h); backtest isolation
  holding (429k synthetic rows excluded from every live count/alert);
  score_source separation intact (ml=172k predictions, llm=3.7k labels,
  never co-mingled). *Pre-existing, not this work:* the modified
  `daemon.py` (briefing-coverage-gap), `scripts/export_training_data.py`,
  `paper-trader/*`, the untracked `tests/test_briefing_coverage_gap.py` /
  `tests/test_export_training_data.py`, and the `logs/*.tmp` deletions all
  predate this session and were deliberately never staged — the one feature
  commit was pathspec-scoped to exactly its 2 intended files.

- **2026-05-17 (Agent 3, hybrid debug+feature+live-validation)** —
  Read pass over the nine task-critical files + `ml/inference.py`. The
  four load-bearing invariants re-traced and hold; no new bug *by
  inspection* in the heavily-reviewed core. Per the established pattern,
  **live validation (Phase 3, run first) was the discovery engine** and
  produced both Phase 1 fixes — both invisible to the 9-file inspection
  loop, both undocumented in this failure-mode table, both in `daemon.log`
  forensics.

  **Phase 1 — two real, undocumented, production-observed bugs:**
  1. `d0d54dd` **`core/logger.py::_JSONLHandler.format` called
     `self.formatException`.** `_JSONLHandler` subclasses
     `RotatingFileHandler` (a `logging.Handler`); `formatException` is a
     `logging.Formatter` method, absent on a Handler. EVERY record carrying
     `exc_info` (every `log.exception(...)` in the daemon) raised
     `AttributeError` inside `emit()->shouldRollover()->format()`.
     `format()` raises *before* `emit` writes, so the **whole** structured
     record was lost (not just the `exc` field) and a secondary
     `--- Logging error ---` traceback was spammed into `daemon.log`
     (observed: 48 collateral tracebacks in one window; every
     `[urgency] Scoring error` absent from `structured.jsonl` — the sink
     the dashboard/healthcheck read). Same blind-spot class as the prior
     logger-UTC bug (`core/logger.py` not in the 9-file list). Fix:
     `traceback.format_exception` (what `Formatter.formatException` does
     internally). Pinned by `tests/test_logger_exc_info.py`.
  2. `ef7fbe4` **`storage/article_store.py::_retry_on_lock` only caught
     `OperationalError('database is locked')`.** The shared `self.conn`
     (`check_same_thread=False`, ~30 threads) is read locklessly by
     `get_unscored`/`get_top_for_briefing`/the trainer/the dashboard while
     `_write_lock` only serialises *writers*; a reader mid-`fetchall`
     corrupts the connection statement state when a writer's `executemany`
     runs → `sqlite3.DatabaseError: another row available`. **Observed 48x
     in one `daemon.log` window**, hitting `insert_batch` (whole collected
     batches dropped, collector backs off) and `update_ai_scores_batch`
     (whole Sonnet-labeled batch lost → urgent items never get `urgency=1`
     → **missed alerts**; articles re-queued to the LLM forever → wasted
     quota). Every decorated op is idempotent and the colliding reader's
     `.fetchall()` completes within the first backoff tick, so a retry
     succeeds. Fix: catch `sqlite3.DatabaseError` (base of OperationalError
     AND IntegrityError) but discriminate on a tight `_RETRYABLE_DB_ERRORS`
     substring allowlist so IntegrityError etc. still propagate. Surgical
     idempotent-safe stopgap; the full fix is per-call write-connection
     isolation (mirrors dashboard `_ro_query`, deferred there too). Pinned
     by `tests/test_article_store.py::TestCursorCollisionRetry`
     (retry-then-succeed + non-retryable-propagates control).

  **Phase 2 — book-coverage line in the 5h briefing (`2cc1250`).**
  `daemon._format_portfolio_coverage(source_articles)` appends one
  deterministic Discord-only line — `📊 Book in digest: MU·NVDA (2/12) —
  silent: …` — so the analyst sees which tracked positions the digest
  actually touches. A 5h window with zero mentions of a held/watched name
  (AXTI/QBTS/SNDU are thin-coverage) was a silent blind spot (real digests
  routinely cover only 2-4 of 12). Pure + char-capped with `+N` overflow,
  mirroring `_format_source_health_summary`; case-sensitive word-boundary
  match reusing the `ml.features._LIVE_RE` convention (`\bMU\b` ≠ MUSEUM,
  MUU distinct from MU); covered list in stable `tickers` order. Appended
  to `message`, NEVER folded into the saved `briefing` text (can't reach
  the trainer's title-prefix label scan — same discipline as the
  source-health line / coverage-gap banner). Read-only: no articles row,
  no `ai_score`/`ml_score`/`score_source` — all four invariants intact.
  +12 exact-string tests (`tests/test_portfolio_coverage_briefing.py`).

  **Phase 3 — live findings (read-only `immutable=1` DB probes + log
  forensics):**
  1. **Briefing cadence recovered.** `briefings` id22→id23 ≈ **6.3h** vs
     5h target — the `ef839a8` heartbeat-clock fix held. The 41h/32h gaps
     (id20→21→22) all predate that fix. Latest briefing (id=23) read
     end-to-end is a genuinely accurate, dense, actionable Bloomberg
     digest (sharp inflation-shock LEAD with real 10Y/VIX/SMH numbers,
     PORTFOLIO with C59-call impairment + NVDL 2x-leverage risk, DESK NOTE
     "watch MU $700"). **Positive validation.**
  2. **`export_worker: database disk image is malformed`** still recurring
     every ~30 min (06:21Z, 06:41Z, 07:11Z) — torn read of the 1.40 GB USB
     `articles.db` under heavy concurrent write; paper-trader's
     `training_data.json.gz` fallback going stale. A sibling agent's
     **uncommitted** in-flight fix is present (`scripts/export_training_data.py`
     `+import os`, untracked `tests/test_export_training_data.py`); those 2
     tests are **flaky** — fail in the cold full suite under USB I/O
     contention (`assert 1 == 0`), pass in isolation. Left untouched per
     the don't-stage-others'-work rule.
  3. **USB `articles.db` I/O saturation** severe and active — even indexed
     read probes block in `D` and time out >85s; **57 of 71 `daemon.log`
     ERRORs are `lock retry exhausted`** (`insert_batch`/
     `update_ml_scores_batch`/etc.). Documented operational issue;
     unchanged.
  4. **Restart churn persists** — 24 `DAEMON — STARTING` in the current
     log; duplicate `daemon.py` (active + flock-blocked) is the designed
     handling. Operational.
  5. **`dashboard /api/articles` 500s 10x** — the `b4be1ca` `_ro_query`
     fix is committed but the running daemon is stale (chronic
     stale-daemon: code fixes need `systemctl --user restart
     digital-intern`).
  6. **Alerted-rows (24h):** legitimate breaking items (Samsung 50k-worker
     HBM4 strike, NVDA 8-K, MU premarket) plus lone reddit/Wikipedia
     low-authority rows — the source-authority gate (`31dea26`) is
     committed but the stale daemon predates it; Wikipedia 0.60 is the
     prior agent's deliberately-deferred above-threshold case, **not
     reopened** (raising the bar would also catch gdelt/scraped — their
     standing call honored).

  Suite: **368 passed** (350 prior-non-export + 4 Phase-1 + 12 Phase-2;
  clean `__pycache__`/`.pytest_cache`), `daemon`/`storage`/`ml` imports
  OK. *Pre-existing, not this work — deliberately never staged:* the
  sibling `scripts/export_training_data.py` edit + untracked
  `tests/test_export_training_data.py` / `collectors/fred_collector.py`,
  all `paper-trader/*` changes (separate repo / sibling agents), and the
  51 `logs/.supervisor_state.*.tmp` deletions. Every commit here was
  pathspec-scoped to exactly its intended .py + test files (4 distinct
  files across 3 commits); never `git add -A`.

- **2026-05-17 (Agent 3, hybrid debug+feature+live-validation, source-cred
  pass)** — Read pass over the nine task-critical files + `ml/inference.py`,
  `ml/embedder.py`, `collectors/source_health.py`,
  `scripts/gdelt_gkg_bulk.py`. No new bug *by inspection* in the
  heavily-reviewed core (5+ prior passes). Phase 3 live validation
  (`articles.db` 1.92M rows; read-only `file:…?mode=ro`) was again the
  discovery engine and surfaced one real, undocumented correctness gap that
  the file-inspection loop cannot see because it only manifests against the
  *production source-tag shape*:

  **Phase 1 — `29247b3` `ml/features.py::_source_credibility` silently
  returned `DEFAULT_SOURCE_CRED` for ~86% of the live top-40 source tags.**
  ~95% of the corpus arrives aggregator-prefixed (`gdelt_gkg/<host>` from
  `scripts/gdelt_gkg_bulk.py`, `GDELT/<host>`, `scraped/<host>`,
  `SEC-EDGAR/<form>`). The verbatim word-boundary scan only matched a
  `SOURCE_CRED` key when it literally appeared in the tag, so the embedded
  publisher was ignored: `gdelt_gkg/seekingalpha.com`→0.55 (key "seeking
  alpha" has a space), `SEC-EDGAR/8-K`→0.55 despite SEC=0.95. Net effect:
  ML `feature[0]` is a near-constant for 95% of training rows (dead signal),
  and the alert authority gate can't see the real publisher. Fix resolves
  the embedded host first via a rescue tier (`_DOMAIN_CRED`, every value
  `>= DEFAULT` and equal to the publisher's existing grade) + a `sec-edgar`
  alias, falling back to the unchanged verbatim scan. **Strictly additive:
  no already-differentiated tag moves and the 0.45 lone-alert gate is
  byte-identical** (pinned by `test_source_credibility_domains.py`).

  **Phase 2 — `e3fa0dd` `_LOW_AUTHORITY_DOMAINS` junk tier.** The 24h
  alerted set (n=7) carried analyst-noise the gate missed because junk GKG
  hosts defaulted to 0.55 (> 0.45): a lone, un-syndicated urgent row from an
  algorithmic stock-mention press mill (`wkrb13.com`,
  `dailypolitical.com`, …), a radio network (`iheart.com`, 63k/24h) or a
  hyperlocal feed fired a standalone Bloomberg BREAKING push. The new tier
  grades *only these explicitly-named hosts* below the gate so
  `_filter_low_authority_lone` suppresses them when lone; corroboration
  (`dup_count>1`) and any credible/unknown host still fire. **Honors the
  prior standing call: the `gdelt`/`scraped`/`GDELT` *channels* are NOT
  down-rated** (a channel-wide bar would catch wires syndicated through
  GKG) — only specific publisher hosts. Pinned end-to-end through
  `send_urgent_alert` by `test_low_authority_domain_gate.py`.

  **Phase 3 findings (analyst lens):** (1) ~95% of `articles.db` is
  `gdelt_gkg/<domain>` — a one-time bulk *historical training-corpus*
  backfill (`gdelt_gkg_bulk.py`), NOT a live ingestion rate; live add-rate
  was ~83/h in the quiet hour sampled. (2) Latest briefing (id 24,
  ~25 min old) is high-quality: tight Bloomberg format, exact CPI/yield/
  semis numbers, actionable DESK NOTE; cadence ~6–7h (slightly over the 5h
  interval — consistent with the restart-warmup logic). (3) Portfolio
  tickers `MUU`/`LNOK` have no live quotes ("no live quote in feed") — a
  `config/portfolio.json` data gap, briefing degrades gracefully; not a
  code bug. (4) `score_source` dist: 1.66M NULL / 264k `ml` / 3.7k `llm`
  — heavy reliance on model self-predictions with sparse LLM ground truth
  (observation). (5) `ai_score>0 AND score_source='ml'` = **0** — the
  ml/ai separation invariant holds in production. (6) `daemon.log`: only
  transient `database is locked` WARNs (absorbed by `_retry_on_lock` /
  worker backoff) + designed singleton-lock restart churn; no tracebacks.
  **Chronic stale-daemon caveat persists:** the running daemon predates
  these commits; `29247b3`/`e3fa0dd` take effect only after
  `systemctl --user restart digital-intern` (not done — out of scope,
  live system + sibling agents). **Feature-cache note:** Phase 2 shifts ML
  `feature[0]` for `iheart.com` (63k rows) / `joker.com` (13k) /
  `wickedlocal.com` (6k) from 0.55→0.30–0.40; the next 2–3 ArticleNet
  retrains absorb it, but `data/ml/dataset_cache.npz` only rebuilds when
  labeled-count drifts >5% (`_CACHE_DRIFT_THRESHOLD`) — delete it to force
  the corrected feature in immediately, or let the natural drift trigger.

  Suite: **388 passed** (371 prior baseline + 11 Phase-1 + 6 Phase-2),
  `storage`/`ml`/`features` imports OK. *Pre-existing, not this work —
  deliberately never staged:* `collectors/rss_collector.py`,
  `storage/article_store.py`, `tests/test_article_store.py`,
  `scripts/export_training_data.py` edits + untracked
  `collectors/fred_collector.py` / `scripts/stale_source_alerter.py` /
  `tests/test_export_training_data.py`, all `paper-trader/*` (sibling
  repo/agents), and the `logs/.supervisor_state.*.tmp` deletions. Every
  commit pathspec-scoped to exactly its intended `ml/features.py` + test
  files (`29247b3` Phase 1, `e3fa0dd` Phase 2, plus a follow-up trimming
  `_LOW_AUTHORITY_DOMAINS` to live-observed hosts only); never `git add -A`.

- **2026-05-18** — Hybrid pass (debug + feature + analyst-validation) over the
  required file set. **Phase 1: bugs_fixed=0, no commit.** The codebase is
  exceptionally mature; every task-listed test already exists and value-asserts
  (backtest exclusion, `update_ml_scores_batch`→`'ml'`, `EXTRA_FEATURE_DIM==15`,
  zero-input no-NaN, `_fetch_training_data` `'ml'` exclusion, sample-weight
  monotonicity). Behaviours initially flagged — all-unformattable alert
  short-circuit (`test_all_rows_unformattable_skips_before_claude`),
  `_published_older_than` RFC822 SQL pre-filter handled in Python, ML-urgent
  firing without an LLM re-verify — are intentional and pinned. No fabricated
  change. **Phase 2: feature `3fe9eb5`** — per-publisher-domain diversity cap in
  `get_top_for_briefing` (`BRIEFING_MAX_PER_DOMAIN=6` + local `_briefing_domain_key`;
  score-ordered overflow backfill so the digest is never shrunk). Evidence: the
  live top-50 briefing input had 10 slots from `scraped/finance.yahoo.com`
  quote-widget pages (`ETH-USDEthereum USD2,169.83` ML-scored 9.96 = #1 slot);
  the cap lifts the live digest from heavy single-domain concentration to 28
  distinct domains / 50 slots. Pure read-side, all four invariants intact, +4
  tests (`tests/test_briefing_domain_diversity.py`). Suite: **405 passed** (401
  baseline + 4); `storage`/`ml`/`features` imports OK. **Phase 3 findings
  (analyst lens):** (1) scrape-quality root cause — `web_scraper.py` ingests
  Yahoo/Finviz price-quote widgets as articles and the ML relevance head
  over-scores them; the diversity cap bounds the *briefing* damage but the
  *alert* path is unprotected (a lone `scraped/finance.yahoo.com` resolves to
  cred ~0.65 > the 0.45 lone-alert gate → can fire a real BREAKING; observed
  urgency=2 row `NVDANVIDIA Corporation227.13-8.61`). (2) `_format_portfolio_coverage`
  (`daemon.py`) matches `\bDRAM\b` against any DRAM-memory article → false
  "covered" for the *DRAM ETF position*, masking a true coverage blind-spot;
  not fixed — `daemon.py` carries unrelated sibling-agent uncommitted edits
  that must not be staged. (3) `daemon.log` `insert_batch` /
  `update_ml_scores_batch: lock retry exhausted` ERRORs at 00:10 → whole
  collected batches lost (missed news); operational, and the sibling agents'
  in-flight reader-`_retry_on_lock` decoration targets exactly this class.
  (4) one dead RSS feed (`Notebookcheck` 404) — minor source-health noise.
  (5) the latest briefing (id 24) is genuinely high-quality and actionable —
  the consumer experience is good when the pipeline is healthy. **Stale-daemon
  caveat:** the running daemon (restarted 00:35) predates `3fe9eb5`; the cap
  takes effect only after `restart digital-intern` (not done — live system,
  sibling agents). *Pre-existing, deliberately never staged* (consistent with
  the 2026-05-16 entries): `collectors/rss_collector.py`, `daemon.py`,
  `storage/article_store.py` reader-`_retry_on_lock` decoration,
  `tests/test_article_store.py`, all `paper-trader/*`, `logs/*.tmp` deletions.
  `3fe9eb5` was kept clean by reconstructing `storage/article_store.py` from
  `git show HEAD:` and re-applying only the 4 feature edits (so the sibling
  reader-decoration work is excluded from the commit yet preserved, unstaged,
  in the working tree); pathspec-scoped to exactly the two intended files,
  never `git add -A`.

- **2026-05-18 (hybrid pass 2)** — debug + feature + analyst-validation.
  **Phase 1: bugs_fixed=1, commit `ff80e65`** — `watchers/alert_dedup.py`
  `dedupe_urgent` winner branch carried the displaced representative's id via
  the hard subscript `cur["_id"]` while the loser branch and `alerted_ids()`
  both guard with `.get()`. A non-canonical urgent row (manual replay, or a
  dict carrying `url` not `link` — the alias `_fmt`/`_is_synthetic` already
  tolerate) with no `_id` raised `KeyError`; `send_urgent_alert`'s broad
  `except` then swallowed it, dropping the WHOLE urgent batch and marking
  nothing alerted — urgent alerts silently fail that cycle (same failure class
  the `_fmt` defensive-access comment documents). A present `_id=None` leaked
  `None` into `_dup_ids`→`alerted_ids`→`mark_alerted_batch`'s `WHERE id=?`.
  Fixed symmetrically; canonical behaviour byte-identical; pre-fix
  `KeyError('_id')` empirically reproduced. +3 tests
  (`TestWinnerBranchIdRobustness` in clean `tests/test_alert_dedup.py`).
  **Phase 2: feature, commit `9014fa5`** — `scripts/alert_pipeline_watchdog.py`,
  an independent process that converts the silent hung-worker outage into a
  Discord page. Grounded in live evidence: the `alert` worker pinged once at
  the 01:15 daemon boot then never again for 25+ min while 29 other workers
  stayed healthy; the supervisor cannot respawn a still-`is_alive()` wedged
  thread, so the analyst's breaking-news channel went silent with only one
  WARNING line. Watchdog reads only `logs/supervisor_state.json` (+ own
  throttle file) → pages when `alert`/`scorer`/`heartbeat` are DEAD/hung or
  the snapshot is missing/stale (daemon down / crash-looping); survives a
  wedged in-process supervisor (today's exact failure). Throttled (anchored to
  incident start), recovery notices, pure `evaluate()` core. DB-free → all
  four invariants intact by construction. `--dry-run` validated live: it
  correctly detected the real wedged `alert` worker. +12 tests
  (`tests/test_alert_pipeline_watchdog.py`). Suite: **420 passed** (405
  baseline + 3 + 12); `storage`/`ml.features`/`ml.model` imports OK.
  **Phase 3 findings (analyst lens):** (1) **CRITICAL — hung `alert` worker /
  no recovery / no escalation** (the Phase-2 driver; supervisor `is_alive()`
  gap is architectural, not a single fixable line). (2) **Daemon restart
  crash-loop** — ~18 restarts in 26 min (00:49–01:15 UTC, documented OOM
  churn) then stabilised; each restart resets worker liveness and starves the
  5h heartbeat cadence. (3) **DB lock-retry exhaustion** —
  `update_ml_scores_batch` + `insert_batch` exhausted the 5-retry budget at
  00:10 UTC → a scored batch and a collected batch dropped; corroborates the
  sibling agents' in-flight reader-`_retry_on_lock` work (left unstaged). A
  read-only `SELECT COUNT(*)` also blocked >15 s — severe contention on the
  1.4 GB DB. (4) **Alert noise** — recent `urgency=2` rows include legit
  signals (SEC 8-K NVIDIA, GDELT Samsung HBM4) but also lone
  `reddit/r/Daytrading` "Trading ideas for Monday – LITE or MU?" (score 8.0)
  and `reddit/r/ValueInvesting` (9.8) that the `cred<0.45` lone gate should
  have suppressed — possible gap or pre-gate rows; noted, not chased. (5)
  **Briefing quality: GOOD** — the 20:31 UTC digest is exact, well-formed and
  genuinely actionable (CPI/10Y/semis LEAD, portfolio P&L, semis pulse); the
  consumer experience is strong when the pipeline is healthy. *Pre-existing,
  deliberately never staged* (consistent with prior entries):
  `collectors/rss_collector.py`, `daemon.py`, `storage/article_store.py`
  reader-`_retry_on_lock`, `scripts/export_training_data.py`,
  `tests/test_article_store.py`, untracked `collectors/fred_collector.py` /
  `scripts/stale_source_alerter.py`, all `paper-trader/*`, `logs/*.tmp`. My
  four files were clean before edit; commits pathspec-scoped, never
  `git add -A`.

- **2026-05-18 (Agent 4, feature-dev — analyst-chat: factor-concentration / correlation honesty)** —
  Advisor-spirit; gap falsified by grep first (`correlation` returned **nothing**
  in the chat path — the chat surfaced `/api/risk`'s NAME-level concentration
  via the analytics block but was **blind** to the FACTOR-level companion,
  though `paper-trader/.../analytics/correlation.py` + `/api/correlation`
  already exist). The hole: a 59/41 two-name book is `concentration_severity=
  HIGH` in `/api/risk`, but if both names are high-β semis the book is one
  bet wearing two tickers — `/api/correlation` is the diagnostic that says
  so, and the analyst never saw it. One additive feature, **this repo only**
  (no cross-repo restart coupling beyond the chronic-stale sibling
  contract), never gates Opus (invariants #2/#12 — chat context only).
  `dashboard/web_server.py::api_chat` gains the pure helper
  **`_correlation_chat_lines(corr)`** (the `_baseline_compare_chat_lines`
  precedent — total/pure, degrade to `[]`, never raise into chat). SSOT
  (invariant #10): the builder's own `headline` is the **verbatim** chat
  line — no chat-side re-derived verdict (the verdict label, mean ρ,
  effective-bets count, and the optional most-coupled-pair clause all
  already live inside `headline`). State ladder: `NO_DATA` (no stock
  positions) → `[]` silence; `INSUFFICIENT` (need ≥2 correlatable names
  with ≥10 aligned daily returns) → ONE verbatim withheld-line; `OK` with
  a real verdict (`SINGLE_NAME_RISK`/`CONCENTRATED`/`MODERATE`/`DIVERSIFIED`)
  → the verbatim headline; any other state or unknown verdict on `OK` →
  silence (degrade rather than parrot an unvalidatable label).
  Wired as a sibling cross-fetch block (own guarded
  `urllib.urlopen(:8090/api/correlation, timeout=3)`, degrade-to-`""`),
  injected into `system_prompt` right after the `ML GATE HONESTY` block
  via the existing `if block else ""` idiom. New
  `tests/test_chat_correlation_enrichment.py` (**19 tests**, pure helper,
  no Flask/DB/cross-fetch — incl. the SSOT verbatim-headline lock across
  all 4 real verdicts via parametrize, the `NO_DATA`-is-silence lock, the
  two `INSUFFICIENT` variant locks, the `OK`-with-unknown-verdict-is-
  silence lock, and the single-chat-line lock). Suites: **19 new passed**;
  the chat-enrichment regression slice **62 passed** (incl. the 19 new +
  baseline + macro + behavioural sets); no import breakage. *Operational:*
  additive — needs `systemctl --user restart digital-intern` to take
  effect; `:8090` already serves `/api/correlation` (no waiting on a
  trader restart). Commit pathspec-scoped (`web_server.py` + new test +
  this `AGENTS.md`), never `git add -A`.

- **2026-05-18 (Agent 4, feature-dev — analyst-chat: forward FOMC / macro-calendar awareness)** —
  Advisor-reviewed; gap falsified by grep first (`macro|fomc|rate.decision`
  returned **nothing** in the chat path — the chat carried ~15 BACKWARD
  analytics blocks + an earnings radar but **zero** forward MACRO-event
  awareness, though the live trader's own decision prompt already gets it
  via `paper_trader/analytics/macro_calendar.py`). One additive feature,
  **this repo only** (no cross-repo restart coupling beyond the chronic-
  stale sibling contract), never gates Opus (invariants #2/#12 — chat
  context only). `dashboard/web_server.py::api_chat` gains the pure helper
  **`_macro_calendar_chat_lines(mc)`** (the `_baseline_compare_chat_lines`
  precedent — total/pure, degrade to `[]`, never raise into chat). SSOT
  (invariant #10): the builder's own `summary` string is the **verbatim**
  headline — no chat-side re-derived verdict. Key design lock: the builder
  sets `events: []` for EVERY non-actionable branch (no-FOMC-in-horizon,
  schedule-not-loaded, builder-error), so all three collapse to `[]` —
  "no FOMC within 14d" / error filler never becomes chat noise (the
  `_behavioural_chat_lines` NO_DATA-omit precedent: silence, not noise).
  An imminent event emits the verbatim summary + one restated detail line
  (when_et / tier / day-or-hour timing from the builder's own fields, the
  `earnings_block` precedent — a within-24h `IMMINENT_HOURS` event surfaces
  the HOUR figure so a 6h-away decision is not rounded to a misleading
  0.2d); a malformed row is skipped, never raises
  (`_paper_trader_position_lines` precedent). Wired as a sibling cross-fetch
  block (own guarded `urllib.urlopen(:8090/api/macro-calendar, timeout=3)`,
  degrade-to-`""`), injected into `system_prompt` right after
  `EARNINGS RADAR` (the forward-scheduled-event cluster) via the existing
  `if block else ""` idiom. New `tests/test_chat_macro_calendar_enrichment.py`
  (15, pure helper, no Flask/DB/cross-fetch — incl. the SSOT verbatim-
  headline lock, the no-FOMC-is-silence lock, and the IMMINENT_HOURS
  hour-not-day lock). Suites: **15 new passed**; the web_server/dashboard/
  chat regression slice **309 passed** (88 of those the full chat-enrichment
  set incl. the 15 new); full `tests/` collects clean at **820** (no
  import breakage). Verified live: against the real `:8090/api/macro-calendar`
  ("no FOMC within 14d") the helper correctly returns `[]` (silent), and a
  simulated imminent payload yields the verbatim SSOT headline.
  *Operational:* additive — needs `systemctl --user restart digital-intern`
  to take effect; `:8090` already serves `/api/macro-calendar` (probed live),
  so unlike the game-plan/hold-discipline blocks there is no waiting on a
  trader restart. Commit pathspec-scoped (`web_server.py` + new test + this
  `AGENTS.md` + `CLAUDE.md`), never `git add -A`.

- **2026-05-18 (Agent 4, feature-dev — analyst-chat: marked-positions fix + action-plan tier)** —
  Advisor-reviewed. Additive, **this repo only** (no cross-repo restart
  coupling — the fix consumes data the trader already emits), never gates
  Opus (chat context only). `dashboard/web_server.py::api_chat` gains
  three pure helpers (the `_tail_risk_chat_lines`/`_behavioural_chat_lines`
  precedent — total/pure, degrade to `[]`/placeholder, never raise into
  chat):
  **(1) `_paper_trader_position_lines`** — the live-trader position block
  now reads the **marked** `portfolio.positions` array (real `pl_pct` +
  `stale_mark`) instead of the raw top-level `positions`
  (`store.open_positions()`, neither key). Fixes a real pre-existing bug:
  the raw array has no `pl_pct`, so the prior inline
  `(p.get('pl_pct') or 0)` rendered **`(0.0%)` for every stock** in the
  chat regardless of P/L; and it now annotates a stale mark
  (`stale_mark=True` — failed price lookup, `current_price == avg_cost`,
  P/L $0.00, indistinguishable from genuinely flat) with `[STALE MARK …]`,
  mirroring the trader prompt suffix (strategy.py) + reporter `⚠ STALE` —
  both already shipped for this exact live MU pathology. The user's
  primary chat surface was the one place it still leaked as a confident
  "MU flat, $0.00". Falls back to the raw array when the marked one is
  empty (degraded `get_portfolio()`) so a store blip never loses the
  book.
  **(2) `_game_plan_chat_lines`** (`/api/game-plan`) + **(3)
  `_hold_discipline_chat_lines`** (`/api/hold-discipline`) — the chat's
  first *actionable* inputs (every prior block is descriptive state);
  composed **verbatim** (invariant #10); `_hold_discipline_chat_lines`
  mirrors `reporter._hold_discipline_line` (emit only on
  `DISPOSITION_DRAG`). Wired as a fifth sibling cross-fetch block (own
  guarded `urllib.urlopen(... timeout=4)` reads, degrade-to-`None`),
  injected into `system_prompt` after `BEHAVIOURAL DIAGNOSIS` via the
  `if block else ""` idiom. New `tests/test_chat_actionable_enrichment.py`
  (15, pure helpers, no Flask/DB/cross-fetch — incl. the always-(0.0%)
  bug lock and the stale-mark misread lock). Suites: digital-intern
  **500 passed, 5 failed** (the full `tests/` count already includes the
  15 new; 505 with committed-HEAD `rss_collector`). The 5
  `test_rss_collector.py` failures are another agent's dirty
  `M collectors/rss_collector.py` — proven by an isolated HEAD-file swap:
  committed HEAD makes all 5 pass; not mine, never staged. *Operational:* the marked-positions fix needs
  only `systemctl --user restart digital-intern` (it reads data `:8090`
  already serves); the game-plan/hold-discipline blocks additionally need
  `:8090` to expose those routes (chronic-stale pattern — they
  degrade-to-skip until then). Commit pathspec-scoped (`web_server.py` +
  new test + this `AGENTS.md`), never `git add -A`.

- **2026-05-18 (Agent 4, feature-dev — analyst-chat behavioural-diagnosis enrichment)** —
  Spec: `~/docs/superpowers/specs/2026-05-18-chat-behavioural-diagnosis-design.md`
  (advisor-reviewed). One additive feature, this repo only, never gates
  Opus (invariants #2/#12 — chat context only). `dashboard/web_server.py`
  gains the pure helper `_behavioural_chat_lines(scorecard, paralysis,
  churn)` (mirrors the `_tail_risk_chat_lines` precedent: total/pure,
  degrades to `[]`), composing the trader's **own synthesized
  self-review verdicts verbatim** — `/api/scorecard` headline + flagged
  `focus`, `/api/capital-paralysis` headline + first-3 `flags`,
  `/api/churn` headline, plus one derived `▶ PRIORITY` line
  (paralysis-unlock ≻ scorecard-focus ≻ churn-CHURNING). Wired into
  `api_chat` as a fourth sibling cross-fetch block (three guarded
  `urllib.urlopen(... timeout=3)` reads of `:8090/api/{scorecard,
  capital-paralysis,churn}`, each independently degrade-to-`None`),
  injected into `system_prompt` right after `PAPER TRADER ANALYTICS`
  via the existing `if block else ""` idiom. The chat already surfaced
  the raw stats (16.67% win rate, 0.04 PF, −$15 realized, 0.52d hold);
  it now surfaces the *diagnosis* — why. New
  `tests/test_chat_behavioural_enrichment.py` (12, pure helper, no
  Flask/DB). Suites: digital-intern **458 passed** (this feature 12/12;
  caches cleared per the phantom-failure note). *Not mine, untracked/
  uncommitted concurrent-agent WIP, deliberately never staged:* the 5
  `test_rss_collector.py` failures (a `collectors/rss_collector.py:175`
  `TypeError` in another agent's dirty `M` change — committed-HEAD
  `rss_collector.py` makes all 5 pass, proven by an isolated HEAD-file
  swap), `daemon.py` `M`, untracked `tests/test_alert_history.py`
  (imports a nonexistent `watchers.alert_history`). My two files were
  clean on HEAD; commit pathspec-scoped (`web_server.py` + the new
  test), never `git add -A`.
  *Operational:* `:8090` is `stale: true, behind: 18` — `/api/scorecard`
  /`-capital-paralysis`/`-churn` already exist on the committed code, so
  the block renders once `systemctl --user restart paper-trader`; until
  then the three cross-fetches degrade-to-skip and the block is silently
  omitted (the chronic-stale pattern, identical to the tail-risk sibling).
  digital-intern `:8080` serves the new chat context only after
  `systemctl --user restart digital-intern`.

- **2026-05-17 (Agent 4, feature-dev — analyst-chat enrichment: tail-risk + 48h thesis tier)** —
  Spec: `~/docs/superpowers/specs/2026-05-17-tailrisk-and-chat-enrichment-design.md`.
  Two additive, advisor-reviewed features; neither gates Opus.
  **(A, paper-trader repo)** new `paper_trader/analytics/tail_risk.py::build_tail_risk`
  (historical 95/99% VaR, positional expected-shortfall CVaR, population
  ann.vol/downside-dev, Fisher-Pearson skew, worst day, max down-streak,
  Ulcer index) — daily series resampled byte-identically to
  `dashboard.analytics_api`'s `by_day` loop; honesty-gated
  `NO_DATA`/`INSUFFICIENT(<20)`/`OK` (live book is 5d → correctly
  INSUFFICIENT until it matures). New `/api/tail-risk` + additive
  `tail_risk` key in `/api/analytics`. `tests/test_tail_risk.py` (21) +
  `test_core_analytics.py::TestTailRiskIntegration` (2).
  **(B, this repo)** `dashboard/web_server.py::api_chat` enriched via two
  extracted pure helpers: `_tail_risk_chat_lines` (surfaces A's
  VaR/CVaR/skew in the existing `PAPER TRADER ANALYTICS` block — degrades
  to `[]` on NO_DATA/missing/error so a stale `:8090` is invisible, not
  broken) and `_partition_thesis_articles` (dedup/cap), backing a new
  48h `THESIS CONTEXT` news tier (second RO query, same live-only
  filter, `LIMIT 25`, deduped vs the 6h breaking set) injected after the
  6h block — multi-day narrative the single 6h/10 window couldn't carry.
  Network/exception-guarded exactly like the greeks/analytics/heatmap
  siblings. New `tests/test_chat_enrichment.py` (14, pure helpers, no
  Flask/DB needed). Suites: paper-trader **1317 passed**, digital-intern
  **434 passed** (clean caches), imports OK.
  *Operational:* `:8090` is `stale: true, behind: 4` — `/api/tail-risk`
  and the `/api/analytics` `tail_risk` key only render after
  `systemctl --user restart paper-trader` (the chronic-stale pattern);
  the chat block degrades gracefully until then. digital-intern `:8080`
  serves the new chat context only after `systemctl --user restart
  digital-intern`.
  *Pre-existing, never staged* (consistent with prior entries):
  `collectors/rss_collector.py`, `daemon.py`, `storage/article_store.py`,
  `scripts/export_training_data.py`, `tests/test_article_store.py`,
  `paper-trader/paper_trader/backtest.py`, `logs/*.tmp`. Commits
  pathspec-scoped, never `git add -A`.

---

### Agent pass 2026-05-18 — COVERAGE GAP briefing intel (digital-intern)

**Feature (this repo, clean file only).** `analysis/claude_analyst.py`: the
5h Opus heartbeat silently omitted any down source, so a dark high-value
channel read as "no news" instead of "blind here". Live inspection found
`sec_edgar`/`sec_edgar_ft` with 900+ consecutive empty polls and **0 8-K
filings delivered**, with no signal anywhere in the briefing. Added
`_collect_source_health()` (best-effort read of `collectors.source_health`
— its own read-only SQLite; **no `articles.db` write, no backtest/ml_score/
score_source surface**; any failure ⇒ `{}` so the briefing never breaks),
`_coverage_gap_lines(report, now)` (pure: curated analyst-meaningful
channels only — per-query gdelt junk excluded — ranked filings-first then
longest-dark, "0 delivered all session" annotation, capped at 8), a
SYSTEM_PROMPT rule + `**COVERAGE GAP**` output section so Opus reproduces
it to Discord, and `_build_payload(..., source_health_report=None)` (None
⇒ section omitted, deterministic, no live DB read — the 3-arg path is
unchanged; `analyze()` signature unchanged so `daemon.py:1477` still works).
New `tests/test_coverage_gap_briefing.py` (16, specific-value asserts; no
LLM/network). Suite: **446 passed**, imports OK. Ships on next
`systemctl --user restart digital-intern` (running daemon holds old code).

**bugs_fixed=0 (honest).** The clean readable files are exceptionally
mature (detailed prior-fix comments, layered defenses); no genuine bug
found that was both real and in a file safe to stage. Guard explicitly
permits 0.

**Phase 3 findings (news-analyst view).**
1. *RSS collector broken in working tree (NOT fixed — not ours).* A
   concurrent agent's incomplete `_fetch_feed`→4-tuple refactor left
   `collect_rss` iterating tuples as dicts → `TypeError` at
   `rss_collector.py:175`; 5 `test_rss_collector.py` failures. Running
   daemon (started 18:12, before the 19:19 edit) holds old code so live
   RSS still ingests; the on-disk code is broken and will fail on next
   restart. File has concurrent uncommitted edits — left exactly as-is.
2. *8 sources DOWN:* `sec_edgar, sec_edgar_ft, finnhub, polygon,
   newsapi, alphavantage, nitter, massive`. `sec_edgar*`/`polygon`/
   `newsapi`/`nitter` show `total_articles=0` — the analyst is fully
   blind to 8-K filings. (This is precisely what the new feature
   surfaces.)
3. *Writer-side lock exhaustion under GKG bulk dumps:* `insert_batch` /
   `update_ml_scores_batch` exhaust the 5-retry budget during the ~1.4M-row
   GKG bulk load (1,401,062 rows in one hour, 2026-05-17T02), dropping
   batches. Per-connection isolation is the documented future fix.
4. *Briefing quality is high* (accurate macro/portfolio/semis, NVDA
   catalyst) but cadence slipped to ~6.5h and a ~32h gap (05-15→05-17)
   from the restart-flap the in-flight `daemon.py` O_CLOEXEC/signal-safety
   change targets.
5. *Lone low-authority alerts* (`reddit/r/ValueInvesting`,
   `reddit/r/Daytrading`, a Moomoo quote-widget) fired BREAKING pushes —
   the `_filter_low_authority_lone` gate (0.45) is in place and will
   suppress these after the next daemon restart.

*Pre-existing, never staged:* `collectors/rss_collector.py`, `daemon.py`,
`storage/article_store.py`, `scripts/export_training_data.py`,
`tests/test_article_store.py`, `collectors/fred_collector.py`,
`scripts/stale_source_alerter.py`, `logs/*`, all `paper-trader/*`.
Commit pathspec-scoped; my feature landed durably (shared monorepo index
race folded it into the concurrent `dd9af44`, already on `origin/master`).

---

### Agent pass 2026-05-18 (hybrid 3 — debug + feature + analyst validation)

**Phase 1: bugs_fixed=1, commit `111378b`** (`collectors/web_scraper.py`
+ `tests/test_web_scraper.py`). Root-cause fix for the codebase's
longest-standing analyst noise complaint, repeatedly flagged in prior
passes but never fixed because it lived in the (clean, stageable)
scraper. `_extract_articles`'s generic anchor scan treated every entry
of Yahoo/Bloomberg's embedded live ticker-tape sidebar
(`<a href="/quote/NVDA">NVDANVIDIA Corporation227.13-8.61(-3.65%)</a>`)
as a fresh article; the price changes each poll so the title — and thus
the sha256 article id — is unique every cycle, manufacturing an
unbounded stream of fake breaking news. **Live evidence: 3,476 of 5,847
sampled `scraped/*` rows were these; ML relevance scored them up to
9.99; one (`NVDANVIDIA Corporation227.13-8.61(-3.65%)`) was Sonnet-scored
8.0 and fired a real 🚨 BREAKING Discord push.** New
`_looks_like_quote_widget(title, url)` rejects them via two independent,
anchored title fingerprints (a letter glued to a multi-digit decimal
price; a parenthesised signed `%` change) plus a Yahoo `/quote/`
landing-path check — validated so `"rises 22% to $35.1 billion"`,
`"4.25%-4.50%"`, `"5,123.41 record high"` and real
`/quote/NVDA/news/...` article URLs all still pass. +5 tests.

**Phase 2: features_added=1, commit `7e97e2d`** (`watchers/alert_agent.py`
+ `tests/test_alert_agent.py`). Defense-in-depth twin
`_looks_like_quote_widget` / `_filter_quote_widget_noise` at the single
alert chokepoint — web_scraper is not the only path a spaceless
price-tick title can enter on (yahoo_ticker_rss, finnhub, manual
replay). Same layered-defense shape as `_is_synthetic` /
`_article_age_ok` / `_filter_low_authority_lone`: a formatter-side drop,
NOT an ML-threshold change, applied right after the synthetic re-filter
and BEFORE dedup (so a tick syndicated across two collectors is still
caught). Helper duplicated, not cross-imported (watchers must not pull
collectors/aiohttp — same rationale as `article_store._briefing_domain_key`).
Suppressed rows are `mark_alerted_batch`'d unconditionally so they exit
the urgent queue instead of re-firing every 20s; `articles.db`
`ai_score`/`ml_score`/`score_source` untouched — **all four invariants
intact by construction** (no synthetic leak, no ml/ai cross-write, no
score_source flip, no urgency regression). +4 tests.

**Phase 3 findings (news-analyst lens). user_findings=7.**
1. *Quote-widget noise still live until restart* — running daemon
   predates `111378b`/`7e97e2d`; `scraped/finance.yahoo.com` still #1
   source/last-hour. Both fixes ship on `systemctl --user restart
   digital-intern` (not done — live system + sibling agents).
2. *Lone low-authority Reddit alerts dominate the push channel* — of 3
   alerted rows in 24h, **2 are noise**: `reddit/r/ValueInvesting`
   (ai=0, ml=9.76 — model over-scored) and `reddit/r/Daytrading`
   "Trading ideas for Monday – LITE or MU?" (ai=8.0); only `Benzinga`
   "Drone Attack On UAE Nuclear Plant / Trump Iran warning" (ai=9.0) is
   genuinely valuable. The already-committed `_filter_low_authority_lone`
   (cred<0.45) suppresses these after restart. No near-dup alerted sigs.
3. *7 collector channels DOWN, 4 with ZERO articles all session*
   (`newsapi, nitter, polygon, sec_edgar` = 0; `alphavantage, massive,
   sec_edgar_ft` disabled) — analyst fully blind to SEC 8-K filings
   (sec_edgar: 922 empty polls). Exactly what the shipped COVERAGE GAP
   briefing feature surfaces; underlying collectors broken/rate-limited
   (operational).
4. *DB writer lock-retry exhaustion* — `update_ml_scores_batch` +
   `insert_batch` exhausted the 5-retry budget at 2026-05-18T00:10 →
   a scored and a collected batch silently dropped (missed news).
   Recurring; sibling-agent reader-`_retry_on_lock` / per-connection
   isolation targets it (left unstaged).
5. *Benign shutdown traceback* — `RuntimeError: reentrant call inside
   BufferedWriter` during `log.info("[daemon] Shutdown complete")`;
   exit-path only, non-fatal.
6. *Pre-existing broken test in tree (NOT mine)* — untracked
   `tests/test_alert_history.py` imports nonexistent
   `watchers.alert_history` → pytest collection error; left as-is.
7. *Briefing quality: EXCELLENT* — #25 (2026-05-18T01:54) exact and
   actionable (10Y 4.59% multi-year high, Iran/Hormuz oil-inflation,
   4%+ semis de-rate two days before NVDA earnings; full MACRO/
   PORTFOLIO-P&L/SEMIS/TOP-SIGNALS). Cadence ~5.4–6.8h. Consumer
   experience is strong when the pipeline is healthy.

Final verify: `storage`/`ml.features`/`ml.model` imports OK; suite
**467 passed**, +9 this work (5 web_scraper + 4 alert_agent), broke
nothing. The 5 `test_rss_collector.py` failures are the pre-existing
sibling-agent `collectors/rss_collector.py:175` `TypeError` (committed
HEAD is clean) — excluded, not mine.

*Pre-existing, deliberately never staged* (consistent with prior
entries): `collectors/rss_collector.py`, `daemon.py`,
`tests/test_article_store.py`, untracked `tests/test_alert_history.py`,
all `paper-trader/*`, `logs/.supervisor_state.*.tmp` deletions. My two
code files were clean on HEAD before edit; both commits pathspec-scoped
to exactly their `.py` + test file, `git diff --staged` verified, never
`git add -A`. Durable on `origin/master`.

---

### Agent pass 2026-05-18 (hybrid — debug + feature + analyst validation)

**Phase 1: bugs_fixed=0 (honest, per the commit guard — not a miss).**
Read pass over the nine task-critical files + `ml/inference.py`,
`alert_dedup`, `source_health`. The four load-bearing invariants
re-traced and hold; ~20 prior passes have exhausted by-inspection
bug-hunting on the heavily-reviewed core, and **live validation
(Phase 3, run first) was again the discovery engine** — but this pass it
surfaced a *feature* gap, not a fixable-in-committed-code bug. Committed
HEAD is clean (467 pass excluding the broken sibling test). Daemon
`pid 1491857` log: **0 ERRORs / 0 tracebacks** in the last 2000 lines,
only 23 transient `database is locked` WARNs absorbed by `_retry_on_lock`
(healthier than the 57/71-lock-exhausted prior passes — the committed
logger/retry fixes are holding). Production invariant #2 verified live:
`ai_score>0 AND score_source='ml'` = **0**. No Phase 1 commit (correct
per the guard).

**Phase 2: features_added=1, commit `8410f05`** (`watchers/alert_recency.py`
new + `watchers/alert_agent.py` + `tests/conftest.py` +
`tests/test_alert_recency.py` new). **Cross-cycle (cross-time)
syndication suppression** — the analyst's single most-cited complaint
(duplicate BREAKING pushes), now closed at the root. `dedupe_urgent`
only collapses copies *inside one `get_unalerted_urgent()` batch*; once a
story is alerted it goes `urgency=2` and is excluded from every future
batch, so a slower feed (GDELT 10-min sweep / `gdelt_gkg` backfill /
Google-News round-robin / Substack 10-min) that re-collects the **same
event** as a NEW `urgency=1` row had nothing to be deduped against and
fired a SECOND standalone "🚨 BREAKING" push. **Live evidence (Phase 3):
the "US clears/approves H200 chip sales to 10 China firms" story fired
two separate alerts ~1.5 h apart** (`reddit/r/technology` 07:42,
`reddit/r/wallstreetbets` 09:11 — different rows, same event). The new
module records the canonical signature (`alert_dedup._signature`
*verbatim* — single source of truth, no drift) of every story that
actually fired into a **separate** hardened `data/alert_recency.db`
(canonical `timeout=30`+WAL+`busy_timeout=30000`; NEVER touches
`articles.db`, so the four invariants are untouched *by construction*)
and suppresses a later urgent row whose signature was alerted within
`ALERT_RECENCY_TTL_HOURS` (6 h, tunable). Same formatter-side
defense-in-depth shape as `_is_synthetic` / `_filter_quote_widget_noise`
/ `_filter_low_authority_lone` (runs after `dedupe_urgent` and the
low-authority gate, before batching); best-effort (a recency-store
failure → empty set → the pre-feature behaviour: a genuine breaking
story must still reach the analyst); suppressed rows marked `urgency=2`
unconditionally so they exit the queue; signatures recorded only on a
*successful* Discord send. Paraphrase-distinct headlines deliberately
still fire (their 8-token signatures differ — errs toward NOT muting a
distinct development; the analyst-safe direction). +11 tests
(`test_alert_recency.py`: pure-partition, `_signature` reuse, DB
round-trip + TTL expiry + prune + hits-upsert, best-effort degradation,
and the **end-to-end** pin — first cycle fires & records, a second
cycle's same-event NEW-id row is cross-suppressed with no Claude/Discord
call and `urgency=2`, while a distinct headline still fires). An autouse
`tests/conftest.py` fixture redirects `alert_recency.DB_PATH` per-test
(exact analogue of `store_factory`'s article-DB redirect — isolates the
new *persistent* store, weakens **no** existing test's assertions; caught
6 state-leak regressions in the alert suites before commit and fixed them
the right way, not by weakening tests).

**Phase 3 — live findings (read-only `mode=ro&immutable=1` probes + log
forensics). user_findings=7:**
1. *Cross-cycle duplicate alerts — CONFIRMED LIVE* (the H200/China
   double-fire above). Root cause now fixed by the Phase 2 feature.
2. *Broken sibling test halts the WHOLE suite* — untracked
   `tests/test_alert_history.py` imports a nonexistent
   `watchers.alert_history` → pytest **collection error** that
   interrupts the entire run (not one failure — zero tests execute).
   Incomplete prior-run work; not mine; left exactly as-is; standard
   run is now `pytest tests/ --ignore=tests/test_alert_history.py`. I did
   **not** create `watchers/alert_history.py` (that would be guessing a
   sibling's unfinished spec) — my module is the distinctly-named
   `alert_recency` precisely so the sibling test stays untouched.
3. *Uncommitted sibling `collectors/rss_collector.py` is BROKEN and
   higher-risk than prior passes noted* — its per-feed-backoff refactor
   makes `_fetch_feed` return a 4-tuple `(name, articles, outcome,
   retry_after)` but `collect_rss` still iterates each result as an
   article list (`for art in batch: art["link"]` → `TypeError: string
   indices must be integers`). RSS is the **hottest** collector (302
   feeds, 30 s cadence); if the auto-commit daemon ships this it
   **silently drops every RSS batch forever**. Causes the 5
   `test_rss_collector.py` failures. Not mine; left untouched per the
   don't-stage-others'-work discipline; flagged loud here.
4. *8 collectors disabled* (`alphavantage, massive, newsapi, nitter,
   polygon, sec_edgar, sec_edgar_ft, wikipedia`); 4 **zero-delivered all
   session** (`newsapi, nitter, polygon, sec_edgar`). `sec_edgar`/`_ft`
   are high-signal 8-K material filings — analyst is blind to filings;
   correctly surfaced by the existing COVERAGE GAP briefing feature
   (working as intended). Upstream/rate-limit; operational.
5. *USB `articles.db` I/O saturation severe* — full-table scans block
   in `D` and time out >90 s even with `immutable=1`. Documented
   operational issue; unchanged.
6. *Pre-restart noise still in the alerted history* (one
   `scraped/finance.yahoo.com` quote-widget tick, several lone
   reddit/Wikipedia rows). The committed quote-widget / low-authority /
   domain-cred gates suppress these post-restart; the running daemon
   predates them (chronic stale-daemon — code fixes need `systemctl
   --user restart digital-intern`). The Phase 2 feature compounds these
   on restart by also killing their *cross-time* repeats.
7. *Positive validation.* Briefing cadence **recovered**: id23→24→25 =
   ~6.3h / ~6.8h / ~5.4h vs the 5h target (the `ef839a8` heartbeat-clock
   fix is holding); the 41h/32h gaps all predate it. Latest briefing
   (id25, 2026-05-18T01:54, 50 articles) read end-to-end is a genuinely
   accurate, dense Bloomberg digest (Iran/UAE drone-strike oil/inflation
   LEAD, real semis de-rate two days before NVDA earnings). The 24h
   alerted set's genuinely-valuable items (Benzinga UAE-strike ai=9,
   SEC-EDGAR NVDA 8-K, GDELT Samsung-HBM4-strike ai=9) are all real and
   portfolio-relevant — the pipeline is strong when healthy.

Final verify: `storage`/`ml.features`/`ml.model` imports OK; suite
**478 passed** (467 prior + 11 new; `--ignore=tests/test_alert_history.py`),
the 5 `test_rss_collector.py` failures are the pre-existing sibling
`rss_collector.py` `TypeError` (excluded, not mine), zero regressions
introduced.

*Pre-existing, deliberately never staged* (consistent with every prior
entry): `collectors/rss_collector.py`, `daemon.py`,
`scripts/export_training_data.py`, `storage/article_store.py`,
`tests/test_article_store.py`, untracked `collectors/fred_collector.py` /
`scripts/stale_source_alerter.py` / `tests/test_alert_history.py`, all
`paper-trader/*`, `logs/*.tmp` deletions. The one feat commit was
pathspec-scoped to exactly its 4 intended files (`watchers/alert_recency.py`,
`watchers/alert_agent.py`, `tests/conftest.py`,
`tests/test_alert_recency.py`) + this `AGENTS.md`; `git diff --staged`
verified; never `git add -A`.

---

### Agent pass 2026-05-18 (hybrid — debug + feature + analyst validation)

**Phase 1 — bugs_fixed=1, commit `c293c08`.** The *entire* pytest suite was
unrunnable: untracked `tests/test_alert_history.py` imports
`watchers.alert_history`, a module that has NEVER existed in git history
(`git log --all -- watchers/alert_history.py` is empty) — an orphan written
against an earlier design that shipped instead as `watchers.alert_recency`
(`8410f05`, exercised by the tracked `tests/test_alert_recency.py`). Its
`ImportError` aborted *collection* for all 484 tests (`pytest tests/` exited on
a collection error, 0 tests executed — a silent hard CI/dev failure: the task's
own "run the suite after each phase" step ran nothing). Fix: a documented
`collect_ignore = ["test_alert_history.py"]` in `tests/conftest.py` (our own
change to a tracked file); the orphan itself is left untouched (untracked, not
ours to delete). Suite went 0 → 478 passed.

**Phase 2 — features_added=1, commit `ed4b270`.** Cross-domain syndication
collapse + corroboration signal in the 5h heartbeat briefing
(`analysis/claude_analyst.py`). Grounded in the codebase's own repeated finding
that syndication is "the analyst's single biggest noise complaint": the alert
path has `watchers.alert_dedup` and the store caps per-publisher-domain, but
neither collapses the SAME wire headline arriving under DIFFERENT domain keys
(`GDELT/reuters.com` + `scraped/finance.yahoo.com` + `rss` are three domains,
all survive the per-domain cap) — the briefing digest Opus reads was the one
path that never deduped. New pure helper `_collapse_syndicated` groups the
newswire by the single well-tested `alert_dedup._signature` (no signature
drift — same anti-drift discipline as `watchers.alert_recency`), keeps the
highest-score copy as the cluster rep (ties keep the earlier/higher-ranked,
stable), preserves score-rank order, annotates `_corroboration`. The rendered
row gains a verbatim `[syndicated xN]` tag and `SYSTEM_PROMPT` now instructs
Opus to weight wide independent corroboration as a magnitude signal for
LEAD/TOP SIGNALS — so dedup also *adds* a genuine analyst signal, not just
removes noise. Collapse runs before the 60-row cap (cap can only surface MORE
distinct signal). Returns shallow copies, never mutates the caller's
`source_articles` list (which `heartbeat_worker` feeds to the
briefing-label/training path) — so backtest isolation, ml_score≠ai_score,
score_source and the urgency state machine are untouched **by construction**
(this only reshapes the text Opus reads, never the DB or the label list). +7
specific-value tests (`tests/test_briefing_syndication_collapse.py`); the 50
existing briefing tests (`claude_analyst`/`coverage-gap`/`domain-diversity`/
`briefing-boost`) pass unchanged.

**Phase 3 — live findings (analyst lens; daemon-log forensics — the 1.4 GB DB
read-probes time out under live daemon + sibling-agent contention).**
user_findings=6:
1. **CRITICAL — RSS dark in production.** `[rss_worker] error: string indices
   must be integers, not 'str'`, backing off 300 s in a loop continuously since
   ~06:05Z. Root cause: a sibling agent's uncommitted WIP in
   `collectors/rss_collector.py` changed `_fetch_feed` to a 4-tuple but did not
   update the `collect_rss()` consumer (line 175). RSS is the 30 s-cadence
   highest-volume collector — the analyst is blind to ~302 feeds. Not fixed
   (uncommitted sibling WIP, deliberately never staged).
2. **8 source channels down/disabled** (`alphavantage, massive, newsapi,
   nitter, polygon, sec_edgar, sec_edgar_ft, wikipedia`). `sec_edgar` +
   `sec_edgar_ft` dark = analyst blind to 8-K filings (the priority-0 intel
   channel) — exactly what the existing COVERAGE GAP briefing block exists to
   surface; underlying collectors being out is a real intel hole.
3. **Heavy `database is locked` worker errors** (rss/yahoo_ticker_rss/finnhub/
   alphavantage/google_news repeatedly backing off → dropped collection
   batches → intermittent coverage gaps). Sibling agents' in-flight
   reader-`_retry_on_lock` decoration in `storage/article_store.py` targets
   exactly this; left unstaged.
4. **`[scorer_worker] error: no more rows available`** — a sqlite
   shared-connection cursor variant NOT in `article_store._RETRYABLE_DB_ERRORS`
   (`another row available`/`another row pending`/`database is locked` but not
   `no more rows available`), so it leaks to the worker's broad `except` and
   drops a scored batch that cycle. Real bug, but `storage/article_store.py`
   carries active sibling-agent WIP on exactly this retry path — reported, not
   co-edited.
5. **`[stats_worker] error: 'NoneType' object is not subscriptable`** —
   recurring (DEBUG) silent failure in `daemon.py` (sibling-WIP file).
6. **Positive (what works well):** on a quiet weekend (2026-05-18 Sun) the
   system is appropriately silent — 1 BN alert in ~7 h, no quote-widget/
   low-authority/cross-cycle suppression churn, briefing cadence on-target
   (last digest `01:54Z`, 2280 chars). The noise-suppression stack and the
   restart-resilient heartbeat are behaving correctly; the analyst experience
   is good when the collectors are healthy.

None of the Phase 3 issues were a safe quick fix: every implicated file
(`rss_collector.py`, `daemon.py`, `storage/article_store.py`) carries
concurrent sibling-agent uncommitted WIP that must be left exactly as-is — so
reported only, no extra fix commit (correct per the staging rule).

**Final verify:** `storage`/`ml.features`/`ml.model` imports OK; suite **485
passed** (478 prior + 7 new); the 5 `test_rss_collector.py` failures are the
pre-existing sibling `rss_collector.py` `TypeError` (not ours, never touched),
zero regressions introduced.

*Pre-existing, deliberately never staged* (consistent with every prior entry):
`collectors/rss_collector.py`, `daemon.py`, `storage/article_store.py`,
`scripts/export_training_data.py`, `tests/test_article_store.py`, untracked
`collectors/fred_collector.py` / `scripts/stale_source_alerter.py` /
`tests/test_alert_history.py` / `tests/test_export_training_data.py`, all
`paper-trader/*`, `logs/*.tmp` deletions. The three commits were
pathspec-scoped to exactly their intended files (`tests/conftest.py`;
`analysis/claude_analyst.py` + `tests/test_briefing_syndication_collapse.py`;
this `AGENTS.md`); `git diff --staged` verified each; never `git add -A`.

---

### Agent pass 2026-05-18 (docs — session-state + known-issues consolidation)

Documentation-only pass. No code changed. Purpose: hand the next agent the
operational ground truth verified live this session (the running-unit state in
particular contradicts the "Running the daemon" head section — read this entry
first).

**Architecture (re-verified live, not from prose):**
- `daemon.py` is the single production process. Confirmed bound to `:8080`
  (Flask dashboard) — `ss -ltnp` shows one listener, PID 1702195,
  `/usr/bin/python3 .../digital-intern/daemon.py`. Singleton lock at
  `data/daemon.lock`; a second start blocks on `flock`.
- Article store: `/media/zeph/projects/digital-intern/db/articles.db` —
  **1,445,425,152 B (~1.35 GB)** SQLite, USB-mounted spindle. `full_text`
  column is **zlib-compressed** (decompress on read; never `SELECT full_text`
  for scanning). The 1.4 GB size + USB I/O is the root of every timeout/lock
  finding below.
- `logs/` is a symlink → `/media/zeph/projects/digital-intern/logs` (same USB
  filesystem, different mount than the repo). `find -P` will not descend it;
  use `readlink -f .../digital-intern/logs` then operate on the real path.

**Committed change this session — `5265d8e` `fix(stats): O(log N) /api/stats`.**
`ArticleStore.stats()` (the `/api/stats` backend) ran `SELECT COUNT(*)` plus
two predicate full-table scans over compressed-BLOB pages on the 1.46M-row USB
DB — the endpoint blocked >30 s and the dashboard rendered "0 Total in DB". Fix
(already on `origin`, no action needed): `total` is now `SELECT MAX(rowid)`
(O(log N) rightmost-leaf walk; rowid is monotonic here — TEXT PK, no
AUTOINCREMENT, purge deletes only lowest rowids — so it over-counts the live
window by the purged volume, ~33 % high and slowly growing: an acceptable
dashboard-tile order-of-magnitude, vastly better than the broken "0"). `urgent`
wrapped in a `LIMIT 10000` subquery. `unscored`/`below_threshold` (no selective
index, each a ~115 s BLOB scan) are now served from a 300 s-TTL cache refreshed
off the request path by a daemon background thread on its own private
connection (never `self.conn` — respects the cursor-collision hazard). Verified
`stats()` 0.371 s (was >30 s). Return-dict shape unchanged. Generalisable rule:
**`COUNT(*)` on the `articles` table times out under live load — never use it.**
For a fast total use `MAX(rowid)`; for a recency/liveness probe use a
`LIMIT 200` scan on `idx_first_seen` (not a full COUNT), and report `n/a`
rather than `0` when a count can't complete.

**Known operational hazards (latent — not code bugs; do not "fix" blindly):**

1. **systemd dual-unit hazard — live state ≠ the head section.** `digital-intern`
   exists as *both* a system unit and a `--user` unit. **Verified 2026-05-18:**
   the **system** unit is `active` + `disabled`; the **user** unit is
   `inactive` + `disabled`. So exactly one daemon is running and it is the
   **system** unit (PID 1702195) — the "Running the daemon" section above which
   says `systemctl --user start digital-intern` is **wrong for the current
   deployment**; use `systemctl {start,stop,restart,status} digital-intern`
   (system scope) to control the live process, and
   `journalctl -fu digital-intern`. The hazard is *latent*: running
   `systemctl --user start digital-intern` while the system unit is active
   spawns a second daemon that contends for `:8080` and the single USB
   `articles.db` (corrupting counts / WAL). The historically-prescribed remedy
   `systemctl --user disable --now digital-intern` is moot right now (the user
   unit is already inactive+disabled) and **must not be run without confirming
   with the user first** — only the system unit should ever be active; never
   start the user unit on this host.

2. **rss_worker 4-tuple bug — fixed on disk, NOT live.** A sibling agent's
   `_fetch_feed`→4-tuple refactor previously left `collect_rss()` iterating
   tuples as dicts → `string indices must be integers` → `[rss_worker]`
   300 s-backoff loop, RSS (the 30 s-cadence highest-volume collector, ~300
   feeds) dark. As of this session `collectors/rss_collector.py:173` carries a
   defensive `(_name, arts, _outcome, _retry_after) = result` unpack with a
   `(ValueError, TypeError)` skip-this-feed fallback and a regression-guard
   comment — i.e. **the fix is on disk but UNCOMMITTED** and the running daemon
   (PID 1702195, started before the edit) still holds the broken code. RSS will
   stay dark in production until `systemctl restart digital-intern`. The file
   has concurrent sibling-agent WIP; per the staging discipline it is left
   exactly as-is and **never staged by a docs/review commit**.

3. **Hourly audit "urgent: 0 / 0 rows" is a FALSE NEGATIVE, not an outage.**
   The healthcheck compares `first_seen` (stored ISO-8601 with a literal `T`,
   e.g. `2026-05-18T01:54:00`) against SQLite `datetime()` output (space
   separator, `2026-05-18 01:54:00`); the string compare never matches so the
   24 h urgent count returns 0 even when the pipeline is healthy. Normalise
   both sides before comparing: `replace(first_seen,'T',' ')`. Do **not** file
   "pipeline down" off a bare 0 here — corroborate with `journalctl` liveness
   first. (Companion of the stats finding above: a true 24 h `COUNT(*)` also
   just times out on the 1.4 GB USB DB — use the `LIMIT 200` `idx_first_seen`
   scan and report `n/a` if it can't complete, never `0`.)

**Operational quick-reference (this deployment, 2026-05-18):**
- Control the live daemon: `systemctl {start,stop,restart,status} digital-intern`
  (system scope — the active unit); `journalctl -fu digital-intern` for logs.
  `systemctl --user ... digital-intern` controls the *inactive* user unit —
  do not start it (hazard #1).
- DB: `/media/zeph/projects/digital-intern/db/articles.db` (~1.35 GB, USB,
  zlib `full_text`).
- Logs (real path): `readlink -f /home/zeph/trading-intelligence/digital-intern/logs`
  → `/media/zeph/projects/digital-intern/logs`.
- Tests: `cd /home/zeph/trading-intelligence/digital-intern && python3 -m pytest tests/ -v`
  (clear `__pycache__`/`.pytest_cache` first if the count looks low — stale
  assertion-rewrite cache, documented under "Running tests"; the 5
  `test_rss_collector.py` failures are the pre-existing sibling refactor, not
  a regression).

**Concurrency note for the next agent:** during this pass a hybrid
debug/feature agent (PID 1725883) was actively editing this same repo and this
same `AGENTS.md`, and the repo's auto-commit/linter daemon pushes on its own
cadence. This entry was appended (not rewritten); the commit was pathspec-scoped
to `digital-intern/AGENTS.md` only — the foreign `M collectors/rss_collector.py`
and `M daemon.py` in the worktree are sibling WIP and were **never staged** —
and the push was left to the auto-commit daemon (manual push races it; see the
project memory on auto-commit). If you append here, re-read the last ~40 lines
immediately before editing: the file races.

---

- **2026-05-18 (Agent 3, hybrid debug+feature+live-validation)** — Read pass
  over the nine task-critical files + `ml/inference.py`,
  `collectors/source_health.py`. Four load-bearing invariants re-traced and
  hold (backtest isolation; ml/ai separation — live `ai_score>0 AND
  score_source='ml'` = **0**; `MAX(urgency,?)`; `get_unscored` age parity).
  Live validation was the discovery engine.

  **Phase 1 — `b20cbae` real live-confirmed bug.**
  `claude_analyst._coverage_gap_lines` derived the briefing COVERAGE GAP
  "DARK X.Xh" from `(now - source_health.last_seen)`, but
  `source_health.record_result` rewrites `last_seen = now` on **every** poll
  incl. the empty polls of a disabled channel (it is *last poll*, not *last
  delivery* — `get_stale_sources` legitimately needs that, so the fix is
  scoped to claude_analyst, NOT source_health). For any actively-polled
  disabled source the value was structurally ≈0: the live briefing read
  "SEC 8-K filings — DARK 0.0h (932 empty polls, 0 delivered all session)",
  telling the analyst a channel blind the *entire* session was negligible.
  Fixed by estimating from `consecutive_failures × poll cadence` (new
  `_COVERAGE_POLL_SECS`, mirrors daemon `*_INTERVAL`, superset of
  `_COVERAGE_LABELS`), `~`-prefixed. Live report now honestly yields
  "SEC 8-K — DARK ~78h", "NewsAPI — ~255h", "Polygon — ~137h". The prior
  `test_coverage_gap_briefing.py` *pinned the buggy contract* (modelled
  `last_seen` as last-delivery, a shape source_health never produces — why
  it shipped invisibly); corrected to the production-accurate contract +
  added the missing discriminating regression (`last_seen≈now` & high fails
  → long dark, not 0.0h) and a `_COVERAGE_POLL_SECS ⊇ _COVERAGE_LABELS`
  parity test (a strengthened, not weakened, suite).

  **Phase 2 — `0792a57` freshness context in the 🚨 BREAKING alert.** The
  whole 0..24h band fired with zero recency signal (store SQL guarantees
  < 24h only by `first_seen`; `_article_age_ok` only drops > 24h). Added
  pure `_article_age_hours`/`_article_age_str` (RFC822+ISO,
  published-preferred, naive→UTC — the `_article_age_ok` convention) → a
  compact `age: 4m / 3.2h / 16h (time since publication)` line per urgent
  row + a RECENCY rule in `ALERT_PROMPT` (FORMAT block untouched). Unknown
  age omits silently. Read-only on the alert path (runs after
  synthetic/quote-widget/low-authority/dedup; changes only prompt text,
  never which rows alert) — all four invariants intact. +21 tests
  (`tests/test_alert_age_context.py`); adjacent alert suites unregressed.

  **Phase 3 — live findings:** (1) **scorer wedged ~18.5 min** (08:01→08:20
  batch gap > 900s liveness → flagged DEAD `state=ok`, recovered 08:20:40)
  under USB-DB contention — the documented "alive-but-blocked, supervisor
  can't respawn a live thread" gap; `alert_pipeline_watchdog.py` is the
  mitigation. (2) **9 `lock retry exhausted` ERRORs**
  (`insert_batch`/`update_ml_scores_batch`, cluster 08:21–22) → batches
  dropped; operational, unchanged. (3) **5 high-value collectors disabled**
  (sec_edgar ~78h, sec_edgar_ft ~46h, polygon ~137h, newsapi ~255h, nitter
  ~63h) — now surfaced honestly by the Phase-1 fix; effective after
  `restart digital-intern` (chronic stale-daemon caveat — running daemon
  predates `b20cbae`). (4) **Alert path NOT noisy this window** — exactly 1
  `BN alert sent` (03:03, 1 distinct story); reddit/Wikipedia `urgency=2`
  rows are prior-instance residue, no live noise reproduced. (5) **Briefing
  GOOD** — id26 accurate/dense/actionable (bond-rout LEAD, portfolio P&L,
  semis pulse, sharp DESK NOTE); cadence 01:54→07:13 ≈ 5.3h (healthy). (6)
  **Collection healthy** — ~1300 live articles/h, backtest isolation holds.

  Suite (excluding the sibling-broken untracked `tests/test_alert_history.py`
  collection-error + the 5 pre-existing sibling `test_rss_collector.py`
  failures from the dirty `M collectors/rss_collector.py`): **521 passed**,
  imports OK. *Pre-existing, deliberately never staged:*
  `collectors/rss_collector.py`, `daemon.py`, `dashboard/server.py`,
  `scripts/export_training_data.py`, `tests/test_article_store.py`, untracked
  `collectors/fred_collector.py` / `scripts/stale_source_alerter.py` /
  `tests/test_alert_history.py` / `tests/test_export_training_data.py`, all
  `paper-trader/*`, `logs/*.tmp`. Both commits pathspec-scoped to exactly
  their intended .py + test files; never `git add -A`. A concurrent sibling
  hybrid agent edited this repo throughout (worktree churn expected).

---

### Agent pass 2026-05-18 (hybrid 14 — debug + feature + analyst validation)

Read pass over the nine task-critical files + `ml/inference.py`,
`watchers/alert_dedup.py`, `watchers/alert_recency.py`, `tests/conftest.py`.
The four load-bearing invariants re-traced and hold; live validation (Phase 3)
was again the discovery engine — it surfaced the Phase-1 bug.

**Phase 1 — bugs_fixed=1, commit `bec95ea`** (`storage/article_store.py` +
new `tests/test_retry_on_lock_no_more_rows.py`). `_retry_on_lock`'s
`_RETRYABLE_DB_ERRORS` tuple covered `database is locked` / `another row
available` / `another row pending` but NOT `no more rows available` — the
**same** shared-`self.conn` cursor-state corruption (a writer `executemany`
resets the connection statement while a lockless reader is mid-fetch), just a
different surfaced string. A colliding `get_unscored` raised it, the decorator
declined to retry (substring absent), it bubbled to the worker's broad
`except` and that cycle's scored batch was **silently dropped → urgent items
un-scored → delayed BREAKING alerts** (exactly the documented (2) failure
mode, on the scoring path). **Live evidence (this session's daemon.log):**
`[scorer_worker] error: no more rows available` recurred ~hourly (06:05,
08:43) + `[recursive_labeler]` 08:01. A prior pass (#1690) diagnosed this
exact bug but could not fix it — `article_store.py` carried sibling WIP then;
it is **clean on HEAD now** (last touched `5265d8e`). Fix: add the substring
(idempotent retry; never a legitimate end-of-results signal inside these
methods — `fetchall()` returns `[]` on empty) + the documenting comment item
(3). New regression file (`test_article_store.py` left untouched — it carries
unrelated sibling WIP): retries→succeeds, `IntegrityError` still propagates
unretried, exhausts exactly `_LOCK_RETRY_ATTEMPTS` then re-raises +bumps
`lock_failures`, and a tuple-membership anti-drift guard. +4 tests.

**Phase 2 — features_added=1, commit `3b09f87`** (`analysis/claude_analyst.py`
+ new `tests/test_briefing_seen_timestamp.py`). `SYSTEM_PROMPT`'s TOP SIGNALS
line asks Opus for `[HH:MM] [score] [TICKER] headline` per signal, but
`_build_payload` fed **zero** per-article time data — so Opus fabricated or
omitted every timestamp on the analyst's primary 5h digest (same "prompt asks
for X, payload omits X" class `0792a57` closed on the *alert* path). New
`_seen_utc_str` surfaces the real `first_seen` clock — already returned by
`get_top_for_briefing` (**no storage-layer change**), RFC822+ISO/`Z`/offset →
UTC `HH:MM`, naive→UTC (the `alert_agent._article_age_hours` convention);
`None` for absent/unparseable so the synthetic PORTFOLIO/OPTIONS snapshot rows
the daemon prepends pass through with **no fabricated `00:00`**. Rendered as
`[seen HH:MM UTC]` between score and source; survives `_collapse_syndicated`'s
shallow copy. Read-only — no DB write, input dicts unmutated (the heartbeat
worker feeds that same list to the briefing-label / training path), backtest
isolation / ml_score≠ai_score / score_source untouched **by construction**
(only the text Opus reads is reshaped). `SYSTEM_PROMPT` deliberately NOT
modified (it already requests `[HH:MM]`). +12 specific-value tests.

**Phase 3 — live findings (news-analyst lens; daemon `pid 1702195`,
read-only `mode=ro&immutable=1` DB probes + log forensics). user_findings=7:**
1. **DB lock-retry exhaustion still drops batches (recurring, CRITICAL).**
   `insert_batch` `lock retry exhausted` ×11 in 24h (clusters 08:01,
   08:21×3, 08:22, 08:29, 08:43×3) + `update_ml_scores_batch` 00:10 +
   `web_worker`/`gdelt_worker` `database is locked` backoffs. Each
   exhaustion silently drops a collected/scored batch → missed news. Root:
   ~2 GB USB `articles.db` I/O saturation + ~30 threads on one shared
   connection. Architectural fix (per-connection isolation) is NOT a
   surgical-safe change for this pass — reported, not co-edited.
2. **`no more rows available` scorer/recursive_labeler batch-drop — FIXED**
   this pass (Phase 1; the Phase-3 finding folded into `bec95ea`).
3. **6 collectors disabled** (`massive, newsapi, nitter, polygon,
   sec_edgar, sec_edgar_ft`); `sec_edgar`/`_ft` = analyst blind to 8-K
   filings (priority-0). Correctly surfaced verbatim by the COVERAGE GAP
   briefing block (working as intended). Upstream/rate-limit; operational.
4. **Worker flagged DEAD then recovered under USB contention** (health
   line `DEAD state=ok last_ok=938s` 08:30 → recovered 08:35) — the
   documented alive-but-blocked / supervisor-can't-respawn-a-live-thread
   gap. Operational.
5. **Alert path clean & CORRECT (positive).** Exactly 1 genuine `BN alert
   sent` in 24h (`Benzinga Economics` UAE-nuclear-plant drone strike /
   Trump Iran warning / Brent >$110, `ai=9`, portfolio-relevant via
   semi supply chain). The lone `reddit/r/ValueInvesting` MSFT row
   (`ml=9.76, ai=0` — model over-scored) was correctly **suppressed** by
   `_filter_low_authority_lone` (marked `urgency=2`, NOT pushed — only 1
   Discord send in the log). No quote-widget / duplicate / cross-cycle
   noise. The noise-suppression stack is behaving exactly as designed.
6. **Briefing quality EXCELLENT (positive).** Latest (07:13Z, header
   07:04 UTC, 2315 chars, 50 articles) read end-to-end: accurate dense
   Bloomberg digest — bond-rout LEAD (10Y +13bp → 4.59% on oil-fed
   inflation, Nasdaq −1.54% semis-led two days before NVDA earnings),
   precise MACRO/PORTFOLIO-P&L/TOP-SIGNALS, RISK tied to NVDA 05-20 print
   + MU DRAM C59 05-22 expiry, decisive DESK NOTE, COVERAGE GAP block
   present. Cadence 07:26→13:44→20:31→01:54→07:13 ≈ 5.3–6.8h vs the 5h
   target (acceptable; the heartbeat-clock fix is holding — no 30h+ gaps).
7. **Collection healthy when not lock-blocked (positive).** ~347 live
   articles/h; `rss +67/+77/+26`, `web` (731/1544 collected), `reddit`,
   `gdelt` all ingesting; live `mode=ro` probe with the `_LIVE_ONLY`
   filter confirms backtest isolation holds on the read path.

Final verify: `storage`/`ml.features`/`ml.model` imports OK; suite **544
passed** (529 baseline + 12 Phase-2 + 4 Phase-1 − net), the 5
`test_rss_collector.py` failures are the pre-existing sibling
`collectors/rss_collector.py` 4-tuple WIP (excluded, not ours), zero
regressions introduced.

*Pre-existing, deliberately never staged* (consistent with every prior
entry): `collectors/rss_collector.py`, `daemon.py`, `dashboard/server.py`,
`scripts/export_training_data.py`, `tests/test_article_store.py`, untracked
`collectors/fred_collector.py` / `scripts/stale_source_alerter.py` /
`storage/story_corroboration.py` / `tests/test_alert_history.py` /
`tests/test_export_training_data.py` / `tests/test_story_corroboration.py`,
all `paper-trader/*`, `logs/*.tmp` deletions. The three commits were
pathspec-scoped to exactly their intended `.py` + test files
(`analysis/claude_analyst.py`+`tests/test_briefing_seen_timestamp.py`;
`storage/article_store.py`+`tests/test_retry_on_lock_no_more_rows.py`; this
`AGENTS.md`); `git diff --staged` verified each; never `git add -A`. A
concurrent sibling hybrid agent edited this repo throughout (worktree churn
expected; this entry was appended, not rewritten).

- **2026-05-18 (hybrid pass 15 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass. **Phase 1: bugs_fixed=0, no commit.** The codebase
  is exceptionally mature (14 prior hybrid passes). Every probe came back
  clean or intentionally pinned: backtest isolation verified **live** (0
  `urgency>=1` synthetic rows in the 1.96M-row prod DB); the quote-widget
  regexes empirically have zero false positives on real `$`/`%`/comma
  headlines incl. "Apple's $1.50EPS beat" (the space after `'s` defeats the
  glue pattern) and catch all widget pseudo-titles; `STALE_SCORE_CAP` is
  pinned by `test_get_unscored_age_fields.py`; `ml/inference.py` grey-zone
  keys on the urgency head by design (pinned); `score_source`/`ml_score`
  separation and the `'ml'→'briefing_boost'` promotion are correct by design.
  No fabricated change — same call as pass 1.
  **Phase 2: features_added=1, commit `35479f5`** (auto-commit daemon swept
  the 2 pathspec-staged files into its own auto-titled commit; `git show
  --stat` confirms exactly `analysis/claude_analyst.py` +197/test, 322
  insertions, 0 deletions — no sibling leakage; pushed to origin/master).
  **Apply the ML `time_sensitivity` head to the briefing ranker** — it was
  trained, persisted per-row, and returned by `get_top_for_briefing` whose
  docstring specifies the exact decay curve, but **no consumer ever applied
  it** (the docstring explicitly defers the policy to a consumer; none
  existed). `analysis/claude_analyst.py` now stable-reranks the collapsed
  digest by `effective = base * 0.5 ** (age_h * ts / 12)` after
  `_collapse_syndicated`, before the 60-row cap. Stability is load-bearing:
  the prepended PORTFOLIO/OPTIONS snapshots carry no `first_seen` → age 0 →
  no decay → effective == max, and a stable desc sort keeps them pinned
  ahead of any real article that ties at 10. Pure read-side: no DB write, no
  ai_score/ml_score/score_source/urgency touch, backtest rows already
  excluded upstream by `_LIVE_ONLY_CLAUSE` — all four invariants intact by
  construction. Unscored `time_sensitivity` → `BRIEFING_DEFAULT_TS=0.5`
  (matches `ml.inference.ArticleScore` default); NaN/bool/future-date all
  guarded. +23 tests (`tests/test_briefing_recency_decay.py`), incl. exact
  half-life arithmetic, the snapshot-pinning stability property, purity
  (no input mutation, same objects returned), and a `_build_payload`
  integration assertion. Suite: **566 passed**, the same 5
  `test_rss_collector.py` failures are the pre-existing sibling
  `M collectors/rss_collector.py` 4-tuple WIP (`_FakeResp` lacks
  `status_code`; not ours, never staged) — zero regressions.
  **Phase 3 findings (analyst lens), user_findings=6:**
  (1) **Briefing quality EXCELLENT (positive)** — id=26 (07:13Z) is a
  dense, accurate, decisively-actionable Bloomberg digest (bond-rout LEAD,
  10Y +13bp→4.59%, Nasdaq −1.54% two days before NVDA earnings; RISK tied
  to NVDA 05-20 print + MU DRAM C59 05-22 expiry). Consumer experience is
  strong when the pipeline is healthy. (2) **Collection healthy but
  GDELT-GKG-junk-dominated** — 1,871 live/h, 1.44M/24h, but the top sources
  are SEO/entertainment firehose (`gdelt_gkg/iheart.com` 63k/24h,
  `joker.com` registrar 13k); `_LOW_AUTHORITY_DOMAINS` already down-rates
  the worst, but the firehose still drives the 1.45GB DB size and the lock
  contention in (4). (3) **CRITICAL coverage-gap contradiction** — briefing
  id=26 reports "SEC 8-K filings — DARK 0.0h (932 empty polls, 0 delivered
  all session)" while the live DB shows **26,268 `SEC-EDGAR/8-K` rows in
  24h** (the #2 source). The analyst's single most market-critical channel
  is reported blind when it is in fact the highest-volume filing feed —
  the exact inverse of the COVERAGE GAP feature's purpose. The `fails ×
  cadence` dark-duration fix is in HEAD; the running daemon predates it
  (stale-daemon caveat) and/or `source_health` keys `sec_edgar` distinctly
  from the delivering worker. Operational / `collectors/source_health.py`
  (outside the clean-file scope); reported, not chased. (4) **`insert_batch:
  lock retry exhausted` recurring ~13×** (00:10, 08:01–08:50) → whole
  collected batches silently dropped = missed news. A plain
  `COUNT(*)`+`first_seen`+LIKE scan on the 1.45GB USB DB measured **23.6s**.
  Sibling-agent in-flight territory (reader-`_retry_on_lock`); deliberately
  untouched. (5) **Lone low-cred push noise** — `reddit/r/ValueInvesting`
  9.8, `reddit/r/Daytrading` 8.0, `Wikipedia` 8.6, `yfinance/Insider
  Monkey` 8.0, `GN "$NVIDIA (NVDA.US)$ - Moomoo"` 9.8 alerted as urgency=2.
  The `_filter_low_authority_lone` (cred<0.45) and quote-widget gates exist
  and are test-pinned in HEAD; reddit (0.40) is gated but Wikipedia (0.60)
  / yfinance (0.65) / GN (0.62) sit above the bar, and these rows predate
  the deployed gates (stale daemon). Tuning question, not a clear bug;
  noted, not chased. (6) **Recurring logging-handler flush traceback**
  (`self.stream.flush()`) — non-fatal log noise, the documented
  signal/BufferedWriter class. None of the findings were a quick safe fix
  inside the clean-file scope (the noise gates already exist & are pinned;
  lock-exhaustion + source_health are sibling/out-of-scope), so no Phase-3
  fold-in — bugs_fixed stays 0.
  Final verify: `storage`/`ml.features`/`ml.model`/`analysis.claude_analyst`
  imports OK; decay helpers present. *Pre-existing, deliberately never
  staged* (consistent with every prior entry): `collectors/rss_collector.py`,
  `daemon.py`, `dashboard/server.py`, `scripts/export_training_data.py`,
  `tests/test_article_store.py`, untracked `collectors/fred_collector.py` /
  `scripts/stale_source_alerter.py` / `storage/story_corroboration.py` /
  `tests/test_alert_history.py` / `tests/test_export_training_data.py` /
  `tests/test_story_corroboration.py`, all `paper-trader/*`, `logs/*.tmp`
  deletions. `analysis/claude_analyst.py` was clean on HEAD; the commit was
  purely additive (no deletions), pathspec-scoped to the 2 intended files,
  `git diff --staged` verified, never `git add -A`. A concurrent sibling
  hybrid agent edited this repo throughout; this entry was appended, not
  rewritten.
- **2026-05-18 (hybrid pass 16 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass (16th; codebase exceptionally mature, 15 prior
  passes). **Phase 1: bugs_fixed=0, no commit** (per COMMIT GUARD). Read all
  clean-scope files in full — `storage/article_store.py`,
  `watchers/urgency_scorer.py`, `watchers/alert_agent.py`,
  `watchers/alert_dedup.py`, `ml/features.py`, `ml/model.py`, `ml/trainer.py`,
  `ml/inference.py`, `collectors/web_scraper.py`,
  `analysis/claude_analyst.py`, `core/json_extract.py` — plus the test map.
  Every candidate (the briefing `_score`/`_effective_score` bool guard
  asymmetry; RFC822-vs-ISO SQL pre-filter in `get_top_for_briefing`; the
  collapse-keeps-highest-raw-score-then-decay ordering subtlety; the
  features `days_since_published` /30 normalisation vs the task's loose "~1
  at 24h" wording) resolved to correct-by-design / documented / test-pinned.
  No fabricated change — same honest call as passes 1 and 15. Sibling-WIP
  `M collectors/rss_collector.py` (+ its 5 `test_rss_collector.py` 4-tuple
  failures), `M daemon.py`, `M dashboard/server.py`,
  `M scripts/export_training_data.py`, `M tests/test_article_store.py` and
  the untracked sibling files were left **exactly as-is** (never read-staged).
  **Phase 2: features_added=1, commit `5f40009`.** **Quote-widget noise gate
  on the Opus heartbeat digest.** `web_scraper` (ingestion) and
  `alert_agent._filter_quote_widget_noise` (alert path) both reject live
  ticker-tape pseudo-articles ("NVDANVIDIA Corporation227.13-8.61(-3.65%)"),
  but the **5h Opus briefing — the analyst's primary consumed product — had
  no such gate**: a widget row entering via a non-`web_scraper` path
  (`yahoo_ticker_rss`/`finnhub`/replay) and ML-scored high (live: up to 9.99)
  still surfaced as a fake `[HH:MM] [score] TOP SIGNAL`. Added
  `_looks_like_quote_widget` + `_filter_quote_widget_noise` to
  `analysis/claude_analyst.py`, wired as the FIRST step of `_build_payload`'s
  newswire section (before collapse/decay/cap). Fingerprints byte-identical
  to the other two gates so all three stay in lockstep; helper duplicated
  (not cross-imported from `alert_agent`) per the documented
  anti-import-cycle discipline (the analysis layer must not pull
  `ml.features`/numpy/aiohttp — same rule as `_collapse_syndicated` reusing
  `alert_dedup._signature`). Pure read-side reshape: returns NEW lists, never
  mutates the caller's `source_articles` (the training-label path), no DB
  write, backtest already excluded upstream — all four load-bearing
  invariants intact by construction. Prepended PORTFOLIO/OPTIONS snapshot
  rows pass through (neither fingerprint matches, no url). +21 tests
  (`tests/test_briefing_quote_widget.py`): both title fingerprints, the
  Yahoo `/quote/` landing-path vs a real `/quote/NVDA/news/...` article,
  url-alias/blank safety, order-preserving partition, **input non-mutation**,
  and four `_build_payload` integration assertions (widget excluded / real
  kept with score / all-widget degrades to the "(no high-relevance…)" line /
  snapshot pass-through). Suite: **587 passed**; the only 5 failures are the
  pre-existing sibling `M collectors/rss_collector.py` 4-tuple WIP
  (`_FakeResp` lacks `status_code`; not ours, never staged) — zero
  regressions vs the 566-pass baseline (+21 = exactly the new cases).
  **Staging-race note:** `git add` was pathspec-scoped to exactly the 2
  intended files and `git diff --staged --name-only` verified ONLY those 2
  immediately before commit, yet commit `5f40009` captured 3 extra coherent
  `paper-trader/` files (`analytics/decision_context.py` + its 2 tests, all
  additive) — a concurrent sibling/auto-commit-daemon staged them into the
  shared monorepo index in the sub-second window between the verify and the
  commit (the documented shared-index race; memory
  `di-shared-repo-concurrency`). The 3 files are an intact, complete sibling
  unit that was staged and would have committed regardless; my 2 files are
  byte-correct in the commit (85 + 171 insertions, 0 deletions). Rewriting
  pushed history on a shared `master` with active concurrent writers would
  destroy the sibling's intact work — deliberately NOT done; documented here
  instead, consistent with pass 15's identical auto-commit-sweep note.
  **Phase 3 findings (analyst lens), user_findings=5:** (1) **Briefing
  quality EXCELLENT (positive)** — id=26 (07:13Z) is a dense, accurate,
  decisively-actionable Bloomberg digest: bond-rout LEAD (10Y +13bp→4.59%
  dragging Nasdaq −1.54% two days before NVDA earnings), exact macro table,
  PORTFOLIO tied to live positions + DRAM C59 05-22 expiry / NVDA 05-20
  print, RISK at specific levels (watch 10Y > 4.60%). The pass-14
  `time_sensitivity` decay rerank is visibly working (fresh high-impact TOP
  SIGNALS). Consumer experience is strong when the pipeline is healthy.
  (2) **Lone low-authority BREAKING noise persists** — last 24h alerted
  (urgency=2): `reddit/r/ValueInvesting` 9.8, `reddit/r/Daytrading` 8.0,
  `Wikipedia "[Wikipedia] Nvidia RTX"` 8.6, `GN "$NVIDIA (NVDA.US)$ -
  Moomoo"` 9.8. reddit (0.40) is gated by `_filter_low_authority_lone` in
  HEAD but the running daemon predates the deployed gate (stale-daemon);
  Wikipedia (0.60) / GN-ticker-page (0.62) sit ABOVE the 0.45
  `ALERT_MIN_LONE_SOURCE_CRED` bar so they fire even in HEAD. Recurring
  tuning observation (identical to pass-15 finding 5) — raising the bar
  risks gating legit `rss` 0.65 / `scraped` 0.50 / `gdelt` 0.58; the gates
  are heavily test-pinned. Not a clear bug; reported, not chased. The
  genuine urgent items in the same window were excellent (NVDA 8-K filing
  8.0, UAE-nuclear-drone/Brent shock 9.0, Samsung HBM4 9.0) and 0 urgent
  rows were stuck (urgency=1 backlog empty → pipeline drains). (3)
  **`insert_batch: lock retry exhausted` recurring ~10×** (09:44Z burst
  across `rss`/`google_news`) → whole collected batches silently dropped =
  missed news; matches memory `di-insert-batch-lock-contention.md`. Even a
  `mode=ro` analyst `COUNT(*)` scan timed out >150s on the 1.4 GB USB DB,
  corroborating sustained ~30-thread shared-connection contention. The
  store's own comment names the real fix (per-call connection isolation à
  la dashboard `_ro_query`) — substantial + `daemon.py`/store are
  sibling-touched → out of safe surgical scope; reported, not chased. (4)
  **COVERAGE GAP "DARK 0.0h" in the running daemon** — briefing id=26 reads
  "SEC 8-K filings — DARK 0.0h (932 empty polls, 0 delivered all session)";
  8 sources disabled (`sec_edgar`, `sec_edgar_ft`, `polygon`, `newsapi`,
  `finnhub`, `massive`, `nitter`, `wikipedia`). The `fails × cadence`
  dark-duration fix is in HEAD; the live daemon predates it (stale-daemon).
  The COVERAGE GAP feature itself fires correctly (analyst IS told they're
  blind to SEC filings — the highest-value channel), only the duration
  display understates it. Operational / `source_health` (out of clean
  scope); reported. (5) **The Phase-2 gap itself** — confirmed by
  inspection that the briefing path lacked the quote-widget gate the other
  two paths have; now closed. None was a quick safe fix inside clean scope
  (1 positive; 2 contentious test-pinned tuning; 3 architectural +
  sibling-touched; 4 already fixed in HEAD + source_health out of scope; 5
  fixed by Phase 2) → no Phase-3 fold-in, bugs_fixed stays 0. Final verify:
  `storage.article_store` / `ml.features` / `ml.model` /
  `analysis.claude_analyst` imports OK; quote-widget helpers present. A
  concurrent sibling hybrid agent edited this repo throughout; this entry
  was appended, not rewritten.

- **2026-05-18 (hybrid pass 17 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass (17th; codebase exceptionally mature, 16 prior
  passes). Advisor-reviewed before substantive work. **Phase 1: bugs_fixed=0,
  no commit** (per COMMIT GUARD — honest, not a miss). Read all nine
  task-critical files + `daemon.py` in full. Every candidate resolved to
  correct-by-design / documented / test-pinned: the `get_top_for_briefing`
  diversity-cap + overflow backfill, the `_collapse_syndicated` → decay →
  `[:60]` order, `urgency_scorer` STALE clamp + truncation guard, the
  `_briefing_domain_key` non-dotted-tag fallback, `update_ml_scores_batch`'s
  `COALESCE(score_source,'ml')`, the trainer strong-label SQL (`'ml'`
  excluded, synthetic included). Live probe corroborated: backtest isolation
  holds (`0` synthetic rows with `urgency>=1` in the ~1.45 GB prod DB);
  alert set clean; briefing id26 excellent. No fabricated change — same
  honest call as passes 1, 15, 16. Sibling-WIP `M collectors/rss_collector.py`
  (+ its 5 `test_rss_collector.py` 4-tuple failures), `M daemon.py`,
  `M dashboard/server.py`, `M scripts/export_training_data.py`,
  `M tests/test_article_store.py` and the untracked sibling files left
  **exactly as-is** (never read-staged).
  **Phase 2: features_added=1, commit `66c349f`.** **LLM-vetted vs
  model-only score calibration tag in the 5h Opus digest.**
  `get_top_for_briefing` ranks the newswire by
  `COALESCE(NULLIF(ai_score,0), ml_score, 0)` — so an Opus/Sonnet-vetted 9
  and a raw local-model 9.8 render with an identical `[score=...]` and the
  COALESCE erases which is which. The relevance head demonstrably
  over-scores forum/wiki/social rows (the recurring pass-15/16 finding #5:
  reddit `ml=9.76`, wikipedia `8.6`, `ai_score=0`); the alert path gates
  that noise (`_filter_low_authority_lone`) but the **briefing newswire Opus
  reads exposed the distinction nowhere**, so neither Opus nor the consuming
  analyst could down-weight a raw-model 9.8 against a vetted 9. Added
  additive `_llm_vetted = bool(raw ai_score)` to the `get_top_for_briefing`
  row dict (model output only ever writes `ml_score`, NEVER `ai_score` —
  invariant #2 — so a falsy raw `ai_score` exactly means "displayed score
  came from `ml_score`, unverified"); `_build_payload` renders a ` [model]`
  token when `_llm_vetted is False` (an explicit-False test — the prepended
  PORTFOLIO/OPTIONS snapshot rows carry no key → `.get` → `None`,
  `None is False` → False → never tagged; an LLM-vetted `True` row also
  untagged); and a `SYSTEM_PROMPT` rule states the **LEAD/TOP-SIGNALS
  consequence** (prefer untagged rows; never lead a lone `[model]` row over
  a comparable untagged one). Tag reflects the cluster representative (the
  highest-scored copy `_collapse_syndicated` keeps — i.e. the score actually
  shown — deliberately NOT OR-ed across siblings, pinned by a test). Pure
  read-side: no DB write, no `ai_score`/`ml_score`/`score_source`/`urgency`
  mutation, displayed `ai_score` field + all ordering/diversity/decay logic
  byte-unchanged, backtest excluded upstream by `_LIVE_ONLY_CLAUSE` — all
  four load-bearing invariants intact by construction. **Calibration signal
  for a documented failure mode — explicitly NOT a claim it changes any
  particular healthy briefing** (id26's actual TOP SIGNALS were all clean
  LLM-vetted lines; the value is in the windows where a model-only forum
  9.8 would otherwise out-rank a vetted 9). +10 specific-value tests
  (`tests/test_briefing_model_score_marker.py`: store-layer `_llm_vetted`
  for llm/model-only/briefing_boost/Sonnet-floored-0.01 rows, render
  presence/absence, snapshot pass-through, mixed-cluster representative
  pin, input-non-mutation, SYSTEM_PROMPT consequence). No exact-key
  assertion exists on the briefing dict (only `set(id(x) …)` object-identity
  — verified before adding the key). Suite: **606 passed** (587 baseline +
  10 mine + 9 from a concurrent sibling agent's added test files), the same
  5 `test_rss_collector.py` failures are the pre-existing sibling
  `M collectors/rss_collector.py` 4-tuple WIP (`_FakeResp` lacks
  `status_code`; not ours, never staged) — zero regressions; the 114
  briefing/store suites pass unchanged.
  **Phase 3 findings (news-analyst lens; daemon `pid 1702195` started
  00:29, read-only `mode=ro` DB probes — `immutable=1` hit "database disk
  image is malformed" under the live torn-write, the documented USB
  contention). user_findings=7:** (1) **Briefing quality EXCELLENT
  (positive)** — id26 (07:13Z, 50 art) read end-to-end: dense accurate
  decisively-actionable Bloomberg digest (bond-rout LEAD 10Y +13bp→4.59%
  dragging Nasdaq −1.54% two days before NVDA earnings; exact macro table;
  PORTFOLIO LITE/LNOK/NVDL/MU tied to live book + DRAM C59 05-22 / NVDA
  05-20; RISK at 10Y>4.60%; decisive DESK NOTE; COVERAGE GAP present).
  Cadence id22→26 ≈ 6.3/6.8/5.4/5.3h vs the 5h target — the `ef839a8`
  heartbeat-clock fix is holding, no 30h+ gaps. (2) **Alert path CLEAN &
  CORRECT (positive)** — exactly **2** alerts in 24h, both legit high-value
  `Benzinga Economics` geopolitical-oil shocks (UAE nuclear-plant drone
  strike / Trump Iran warning / Brent spike `ai=9.0`; Dow/S&P-futures-drop
  follow-up `ai=8.0`). **Zero** reddit/wikipedia/quote-widget noise; no
  `urgency=1` backlog stuck. The full noise-suppression stack (quote-widget
  ×3, low-authority-lone, cross-cycle recency, syndication collapse) is
  behaving exactly as designed. (3) **Invariants hold LIVE** — `0`
  synthetic rows with `urgency>=1`; paper-trader actively injecting
  `backtest_run_6233` synthetic training rows (133 of newest 200 first_seen)
  — correctly tagged + isolated by `_LIVE_ONLY_CLAUSE`. (4) **`insert_batch:
  lock retry exhausted` recurring** — 16 ERRORs in last 6000 log lines
  (clusters 08:50, 09:42–09:44Z) → whole collected batches silently dropped
  = missed news; matches memory `di-insert-batch-lock-contention.md`.
  Architectural fix (per-connection isolation) is substantial +
  `daemon.py`/store partly sibling-touched → out of safe surgical scope;
  reported, not chased. (5) **~1.12M unscored backlog** — scorer keeps full
  pace (batch=1000 scored=1000/cycle) but the gdelt_gkg + backtest bulk
  injection outpaces the drain (`remaining≈1,122,267`, ~5k/37min). Defused
  for briefings/alerts by the staleness filters + kw-DESC scoring order;
  operational observation, not a code bug. (6) **Stale-daemon caveat** —
  the running daemon predates HEAD: COVERAGE GAP shows "DARK 0.0h" (the
  `b20cbae` fails×cadence fix is in HEAD) and TOP SIGNALS lack the
  `[HH:MM]` token (`3b09f87`); both correct in HEAD. The Phase-2 `[model]`
  tag likewise ships only on next `systemctl restart digital-intern`. (7)
  **8 collectors disabled** (sec_edgar/_ft, polygon, newsapi, alphavantage,
  massive, nitter, +) — analyst blind to 8-K filings (priority-0);
  correctly surfaced verbatim by the existing COVERAGE GAP briefing block
  (working as intended). Upstream/rate-limit; operational. None of 1-7 was
  a quick safe fix inside clean scope (1-2-3 positive/invariant-holds; 4
  architectural+sibling; 5 operational; 6 already-fixed-in-HEAD; 7
  upstream) → no Phase-3 fold-in, bugs_fixed stays 0. Final verify:
  `storage.article_store` / `ml.features` / `ml.model` /
  `analysis.claude_analyst` imports OK. *Pre-existing, deliberately never
  staged* (consistent with every prior entry): `collectors/rss_collector.py`,
  `daemon.py`, `dashboard/server.py`, `scripts/export_training_data.py`,
  `tests/test_article_store.py`, untracked `collectors/fred_collector.py` /
  `scripts/stale_source_alerter.py` / `storage/story_corroboration.py` /
  `tests/test_alert_history.py` / `tests/test_export_training_data.py` /
  `tests/test_story_corroboration.py`, all `paper-trader/*`, `logs/*.tmp`.
  Commit `66c349f` was pathspec-scoped to exactly its 3 intended files
  (`storage/article_store.py`, `analysis/claude_analyst.py`,
  `tests/test_briefing_model_score_marker.py`); `git diff --staged
  --name-only` verified immediately before commit; `git show --stat`
  confirmed no sibling leakage; never `git add -A`; pushed to
  origin/master. A concurrent sibling hybrid agent (`pid 1807306`, same
  task) edited this repo throughout; this entry was appended, not rewritten.

- **2026-05-18 (hybrid pass 18 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass (18th; codebase exceptionally mature, 17 prior
  passes). Advisor-reviewed before each phase. Live evidence was the
  discovery engine (the proven pattern of passes 14/16/17), not pre-emptive
  re-reading. Daemon `pid 1702195` (system unit `active`) confirmed healthy
  & writing live (newest `first_seen` 10:35:40Z, ≈3 min before probe);
  `sqlite3` CLI absent → all probes via `python3 -m sqlite3 …?mode=ro`.

  **Phase 1 — bugs_fixed=1, commit `d5918e3`** (`watchers/alert_agent.py` +
  `tests/test_alert_agent.py`). **Live discovery:** a `mode=ro` probe found
  **26 `urgency=1` rows stuck from 2026-05-13** (5 days old, never alerted),
  contradicting passes 14/16/17's "no urgency=1 backlog stuck". Root-caused
  in `send_urgent_alert`: it has four noise-suppression gates — quote-widget,
  low-authority-lone, cross-cycle, **and stale-published**. The first three
  each `store.mark_alerted_batch(alerted_ids(...))` so dropped rows EXIT the
  urgent queue ("instead of being re-fetched and re-evaluated every 20s
  cycle" — their own comments); the stale `_article_age_ok` drop was the
  ONLY one that dropped WITHOUT marking. A recently-collected row with an
  old `published` (returned by `get_unalerted_urgent` on recent
  `first_seen`) was re-fetched + re-dropped every 20s for up to 24h, then —
  once `first_seen` aged past the store's 24h cutoff — stranded as a
  permanent `urgency=1` residue (inflating the `stats()` `urgent` tile,
  re-decompressed every cycle). A stale-by-`published` row only ages further
  — it can never become a valid fresh alert — so marking it loses no
  delivery. Fixed by mirroring the established pattern verbatim (partition
  fresh/stale, best-effort `mark_alerted_batch(alerted_ids(stale))`, log
  line, pre-dedup like the quote-widget gate). Invariants: only `urgency=2`
  via `mark_alerted_batch` (ai_score/ml_score/score_source untouched),
  synthetic already filtered above — all four intact. The two prior tests
  (`test_stale_published_article_is_not_alerted`,
  `test_unparseable_dates_block_the_alert`) **pinned the buggy contract**
  (`urgency==1` / `spy.marked==[]`); corrected to the production-accurate
  contract — STILL assert no-Claude/no-Discord, ADD `urgency==2` + queue
  drained + ai_score/score_source untouched — and added a mixed fresh+stale
  discriminating regression (a strengthened, not weakened, suite; pass-14
  precedent). Ships only on next `systemctl restart digital-intern`
  (stale-daemon caveat — running daemon predates HEAD).

  **Phase 2 — features_added=1, commit `ad0bb56`** (`analysis/claude_analyst.py`
  + new `tests/test_briefing_alert_parity.py`). **`[ALERTED]` alert↔briefing
  parity tag.** A news analyst reading the 5h Opus digest could not tell a
  genuinely new LEAD from a rehash of a story already pushed as a standalone
  🚨 BREAKING alert hours ago (the recurring duplicate-alert complaint, on
  the one product that never mitigated it). `watchers.alert_recency` already
  persists the canonical `alert_dedup._signature` of every fired alert (TTL
  6h ≈ the 5h window) and uses it for cross-cycle suppression; the briefing
  path never consulted it. `_build_payload` now reads the recent fired-alert
  signature set ONCE per briefing (`_recent_alert_signatures` — best-effort,
  `set()` on any failure, single read of a separate `alert_recency.db`,
  NEVER `articles.db`) and tags matching digest rows ` [ALERTED]`;
  `SYSTEM_PROMPT` rule forbids leading an `[ALERTED]` row over a comparable
  untagged one and mandates continuation framing. Reuses
  `alert_dedup._signature` verbatim (the documented anti-drift discipline —
  the tag and the cross-cycle gate agree by construction; `_signature` is a
  normalised first-8-token prefix, verified to discriminate distinct
  same-ticker events e.g. "MU surges…" ≠ "MU drops…", so no false-positive
  silencing). Snapshot rows (no link/url) never tagged — same guard as
  `_extract_briefing_labels`. Pure read-side: no DB write, no
  ai_score/ml_score/score_source/urgency mutation, backtest excluded
  upstream by `_LIVE_ONLY_CLAUSE` — all four invariants intact by
  construction. +10 specific-value tests (tag presence/absence, wire-marker
  variant collapse, distinct same-ticker non-collision, snapshot
  pass-through, empty-set degrade, broken-DB swallowed, input non-mutation,
  SYSTEM_PROMPT LEAD/continuation rule). Ships on next restart.

  **Phase 3 — analyst-lens live validation, user_findings=8.** (1)
  **Briefing EXCELLENT (positive)** — id 07:13Z read end-to-end: dense,
  accurate, decisively-actionable (bond-rout LEAD 10Y +13bp→4.59% / Nasdaq
  −1.54% two days before NVDA earnings; exact macro table; PORTFOLIO
  LITE/LNOK/NVDL/MU tied to live book + DRAM C59 05-22 / NVDA 05-20; RISK at
  10Y>4.60%; sharp DESK NOTE; COVERAGE GAP present). (2) **Alert path CLEAN
  recent 24h (positive)** — exactly 2 alerts since 5/17 09:38, both legit
  high-value `Benzinga Economics` geopolitical/oil (01:55 ai=9.0 UAE
  nuclear-plant drone/Brent; 09:19 ai=8.0 Dow/S&P-futures-drop follow-up);
  zero reddit/wiki/quote-widget noise in-window (earlier 5/15–17 noise is
  pre-deployed-gate residue, stale-daemon). (3) **Invariants HOLD LIVE** —
  `0` synthetic rows with `urgency>=1`; `0` `ai_score>0 AND
  score_source='ml'` in the ~1.45 GB prod DB. (4) **Collection healthy** —
  newest live row ≈3 min fresh; ~1300+ live art/h (GN round-robin dominant,
  scraped/finance.yahoo.com ~98/h, reddit ~58/h). (5) **The Phase-1 26
  stuck-urgent rows** — found here, fixed in `d5918e3`. (6) **Chronic
  `insert_batch: lock retry exhausted`** — ~22 ERRORs last 3h (clusters
  08:01–08:50, 09:42–44, 10:41–42) + one `update_ml_scores_batch` 00:10 →
  whole batches silently dropped = missed news; memory
  `di-insert-batch-lock-contention`; real fix (per-call connection
  isolation) is substantial + `daemon.py`/store sibling-touched → out of
  clean scope; reported, not chased (advisor-confirmed). (7) **8 collectors
  DARK** — COVERAGE GAP correctly lists SEC 8-K (priority-0, analyst blind
  to filings), SEC-FT, Polygon, NewsAPI, AlphaVantage, Yahoo-ticker-RSS,
  Massive, Nitter ("0 delivered all session" for SEC/Polygon/NewsAPI/Nitter);
  upstream/rate-limit/key, operational; "DARK 0.0h" understatement fixed in
  HEAD (`b20cbae`), ships on restart (stale-daemon). (8) **Shutdown
  reentrant-logging Traceback** — one `RuntimeError: reentrant call inside
  BufferedWriter` at `daemon.py:2077` during a restart; the EXACT hazard the
  signal-handler comment documents, benign (os._exit cleanup), an
  OOM-restart-churn symptom — not a new bug, daemon.py sibling-touched →
  out of scope. None of 6/7/8 is a quick safe fix in clean scope → no extra
  Phase-3 fold-in; bugs_fixed stays 1 (the Phase-1 fix). Final verify:
  `storage.article_store` / `ml.features` / `ml.model` /
  `analysis.claude_analyst` imports OK; suite **631 passed** (+10 mine over
  the 621 sibling-inflated baseline), the same 5 `test_rss_collector.py`
  failures are the pre-existing sibling `M collectors/rss_collector.py`
  4-tuple WIP (not ours, never staged); `tests/test_sector_pulse.py`
  collection error is sibling-WIP (`?? test_sector_pulse.py` +
  `M dashboard/server.py`/`web_server.py`), excluded via `--ignore`, not
  ours. *Pre-existing, deliberately never staged* (consistent with every
  prior entry): `collectors/rss_collector.py`, `daemon.py`,
  `dashboard/server.py`, `dashboard/web_server.py`,
  `scripts/export_training_data.py`, `tests/test_article_store.py`,
  untracked sibling files (`tests/test_sector_pulse.py`, etc.), all
  `paper-trader/*`, `logs/*.tmp`. Both commits pathspec-scoped to exactly
  their 2 intended files; `git diff --staged --name-only` verified
  immediately before each commit; `git show --stat` confirmed no sibling
  leakage (the shared-index auto-commit race did NOT fire this pass — the
  remote advanced between the two pushes from sibling/auto-commit activity
  but neither of my commits captured a foreign file); never `git add -A`;
  pushed to origin/master. A concurrent sibling hybrid agent (`pid
  1824145`, same task) edited this repo throughout; this entry was
  appended, not rewritten.

- **2026-05-18 (hybrid pass 19 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass (19th; codebase exceptionally mature, 18 prior
  passes). Advisor-reviewed before substantive work. Live evidence was the
  discovery engine (proven pattern of passes 14/16/17/18), not pre-emptive
  re-reading. `sqlite3` CLI absent → all probes via `python3` `sqlite3`
  `mode=ro`. Concurrent sibling agent + auto-commit daemon active on the
  shared monorepo index (memory `di-shared-repo-concurrency`) → strict
  per-commit pathspec staging throughout.

  **Phase 1 — bugs_fixed=0, no Phase-1 commit (honest, per the guard).**
  Read all 9 required files in full + the alert-dedup/recency/inference/
  json-extract paths. Found **no genuine bug** — every load-bearing invariant
  is multiply defended and the entire requested test list
  (`backtest://` exclusion in `get_unalerted_urgent`, `mark_alerted`
  idempotence, `score_source='ml'` on `update_ml_scores_batch`, 15 feature
  dims / zero ticker-density / days-since-published, model `[0,10]`/`[0,1]`/
  no-NaN, trainer `score_source='ml'` exclusion + label weighting, urgency
  9.5-urgent / 3.0-not / already-alerted-not-regressed) **already exists and
  is comprehensive** (advisor-confirmed: extend real gaps, never duplicate).
  Baseline 647 passed / 5 failed; the 5 are the pre-existing sibling-WIP
  `M collectors/rss_collector.py` per-feed-backoff change (its new
  `resp.status_code` branch vs the test's `_FakeResp`) — **not ours, never
  staged, left exactly as-is**; the floor "still exactly 5, never 6+" held
  every run.

  **Phase 2 — features_added=1, commit `257057d`**
  (`analysis/claude_analyst.py` + new `tests/test_briefing_book_tag.py`).
  **`[BOOK: TICKER]` held-book relevance tag.** The 5h Opus digest ranked an
  8.0 held-position story identically to an 8.0 generic-macro one — Opus
  never saw which newswire rows touch the analyst's open book while composing
  LEAD/TOP SIGNALS/PORTFOLIO (the Discord-only `_format_portfolio_coverage`
  line is appended *after* the briefing). Adds `_book_tickers()` + a pure
  read-side ` [BOOK: …]` tag in the exact shape of `[syndicated xN]` /
  `[model]` / `[ALERTED]`, real-url-guarded so prepended PORTFOLIO/OPTIONS
  snapshots are never tagged (same discipline as `_extract_briefing_labels`),
  plus a `SYSTEM_PROMPT` rule to weight held-book rows for the LEAD and the
  PORTFOLIO table. `_BOOK_TICKERS` is a local mirror of
  `daemon.PORTFOLIO_TICKERS` (anti-import-cycle discipline) pinned by a
  parity test. No DB write, no ai_score/ml_score/score_source/urgency touch,
  no row mutation, backtest excluded upstream — four invariants intact by
  construction. +14 specific-value tests (word-boundary MU≠MUU, no match in
  "Micron", canonical dedup ordering, url-alias, snapshot pass-through,
  non-mutation, daemon parity, SYSTEM_PROMPT consequence). All 86
  briefing-suite tests (mine + every existing `_build_payload` assertion)
  pass — the tag insertion broke no contiguity contract. Ships on next
  `systemctl restart digital-intern` (stale-daemon caveat).

  **Phase 3 — user_findings=6; one folded into bugs_fixed (total
  bugs_fixed=1, commit `05b406e`).** (1) **Live-log discovery → FIXED:**
  `[stats_worker] error: 'NoneType' object is not subscriptable` recurred
  12+×/h in `daemon.log`, exactly correlated with the concurrent `database
  is locked` writer-contention storm. Root cause: the SAME shared-`self.conn`
  cursor collision `_retry_on_lock` documents can corrupt the fetch so
  `cur.fetchone()` returns `None` (not raise the retryable `DatabaseError`
  variant); the aggregate readers did `.fetchone()[0]` → `TypeError`, NOT a
  `sqlite3.DatabaseError`, so the decorator never retried it and it bubbled
  every contended cycle (`stats`/`count_unscored`/`stats_since` silently
  failing → scorer-backlog gauge + `/api/stats` blind). Fixed with
  `_expect_row()` — converts the `None` aggregate fetch (MAX/COUNT always
  yield one row, so `None` is unambiguously the collision, never a legit
  empty) into the same retryable signal the decorator already handles;
  applied to all 5 vulnerable sites. +8 specific tests (helper unit,
  decorator compose, stats/count_unscored/stats_since recover). (2)
  **Briefing GOOD (positive)** — id 07:13Z read end-to-end: accurate,
  decisively actionable (bond-rout LEAD 10Y+13bp→4.59% / Nasdaq −1.54% two
  days before NVDA earnings; PORTFOLIO LITE/LNOK/NVDL/MU tied to live book +
  DRAM C59 05-22; COVERAGE GAP present); cadence healthy (~5–7h gaps) after
  the documented 5/14–15 31.9h/41.2h restart-starvation (now mitigated by
  `_initial_heartbeat_last`). (3) **Alert path CLEAN** — exactly 2 alerts /
  24h, both legit `Benzinga Economics` UAE-drone/Brent geopolitical
  (01:55 ai=9.0, 09:19 ai=8.0); zero reddit/wiki/quote-widget noise
  in-window. **Observation:** the 09:19 "Stock Market Today…Drop Following
  Drone Strike" is a market-reaction *continuation* of the 01:55 "Drone
  Attack On UAE Nuclear Plant" but has a distinct `alert_dedup._signature`
  (first-8-token) so cross-cycle suppression does NOT collapse the same
  catalyst surfacing under a materially different headline — borderline
  duplicate from the analyst's seat; low severity at this volume, not chased
  (signature widening risks false-silencing distinct same-ticker events,
  which `test_briefing_alert_parity` explicitly pins). (4) **No stuck
  urgent queue** — `urgency=1` count 0 / 24h: the pass-18 `d5918e3`
  stale-drop fix is holding live, no permanent residue. (5) **Collection
  healthy** — 407 live art/h, 4780/24h, GN round-robin dominant; newest
  row ≈min-fresh. (6) **Chronic DB-lock contention (pre-existing,
  reported not chased)** — frequent `database is locked` WARNINGs across
  ~10 workers backing off 5–20s (memory `di-insert-batch-lock-contention`);
  the real fix (per-call connection isolation) is substantial and
  `daemon.py`/store sibling-touched → out of clean scope; the Phase-3 fix
  above removes one *symptom* (the TypeError leak) of this same storm.
  Final verify: `storage.article_store` / `ml.features` / `ml.model` /
  `analysis.claude_analyst` imports OK; suite **677 passed**, the same 5
  `test_rss_collector.py` failures are the pre-existing sibling
  `M collectors/rss_collector.py` WIP (not ours, never staged). *Pre-existing,
  deliberately never staged* (consistent with every prior entry):
  `collectors/rss_collector.py`, `daemon.py`, `dashboard/server.py`,
  `tests/test_article_store.py`, all `paper-trader/*`, `logs/*.tmp`. All
  three commits pathspec-scoped to exactly their intended files;
  `git diff --staged --stat` verified before each commit; never `git add
  -A`; pushed to origin/master. Entry appended, not rewritten.

- **2026-05-18 (hybrid pass 20 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass (20th; codebase exceptionally mature, 19 prior
  passes). Advisor-reviewed before substantive work. Live evidence was the
  discovery engine (proven pattern of passes 14/16/17/18/19), not pre-emptive
  re-reading. `sqlite3` CLI absent → all probes via `python3` `sqlite3`
  `mode=ro`. Bare daemon `pid 1702195` started **2026-05-18 07:29:24Z**,
  predating EVERY recent fix incl. d5918e3/05b406e/b20cbae and both of mine
  (the consistent stale-daemon caveat — fixes ship on next restart).
  Concurrent sibling agent + auto-commit/push daemon on the shared monorepo
  index (memory `di-shared-repo-concurrency`) → strict per-commit pathspec
  staging throughout; the shared-index auto-push raced (a rejected push then
  surfaced my exact commit hash already on origin/master — verified, not
  re-pushed).

  **Phase 1 — bugs_fixed=1, commit `50c1052`** (`storage/article_store.py` +
  new `tests/test_stale_urgent_reaper.py`). **Live discovery → root-caused →
  fixed:** a `mode=ro` probe found **26 `urgency=1` rows stuck since
  2026-05-13** (5 days). Root cause: `get_unalerted_urgent` filters
  `first_seen >= now-24h`, so the instant a still-pending `urgency=1` row's
  `first_seen` crosses 24h it becomes permanently invisible to `alert_worker`
  — never alerted, and (still `1`, not `2`) never cleared. It lingers until
  the 90-day purge, the whole time inflating `stats()`'s `urgent>=1` tile (no
  time filter) → the dashboard shows phantom urgent items the analyst is
  never pushed. This is the STRUCTURAL counterpart to the pass-18 alert_agent
  stale-drop fix (`d5918e3`), NOT a duplicate: that marks *in-window* rows
  `urgency=2` (formatter actively declined delivery — truthful + blocks
  re-fetch); these *aged-out* rows the alert worker NEVER saw, so `urgency=2`
  would be a lie AND keep inflating the very tile this fixes — `urgency=0` is
  the only honest+corrective state; the two must NOT be "harmonized" (advisor
  point, encoded in the code comment). Added
  `ArticleStore.reap_stale_urgent(max_age_hours=24)` (demote `1→0` for
  aged-out rows; demotion provably loses zero delivery — a >24h row is never
  returned by `get_unalerted_urgent` again) wired into `purge_old()` BEFORE
  its `_write_lock` block (the method takes that same non-reentrant lock
  itself; nesting would deadlock — advisor point). Only `urgency` written
  (ai_score/ml_score/score_source untouched); `_LIVE_ONLY_CLAUSE`
  defense-in-depth (synthetic rows are urgency=0 by construction → no-op,
  matches `update_scores_from_labels` precedent). +10 specific-value tests
  (aged-out demoted / in-window kept / alerted-2 never un-alerted / scores
  byte-unchanged / idempotent / synthetic untouched / custom window /
  alert-path-unreachability / purge_old wiring).

  **Phase 2 — features_added=1, commit `17d8df9`** (`watchers/alert_recency.py`
  + `watchers/alert_agent.py` + new `tests/test_alert_continuation_context.py`).
  **Alert continuation context.** Cross-cycle suppression drops only
  EXACT-signature repeats; a *different* headline about the same developing
  event (live: 01:55 UAE-strike alert → 09:19 Brent/markets follow-up,
  distinct signatures, correctly NOT collapsed) still fires a fresh
  standalone 🚨 BREAKING with zero continuity framing — the analyst's top
  duplicate-alerts complaint, on the one product (the push) that never got
  the mitigation the briefing's `[ALERTED]` tag added. Added
  `alert_recency.recent_alerts()` (richer sibling of `recent_signatures` —
  also returns stored title + age) + pure unit-tested `related_prior_alert()`
  (≥3 shared SALIENT signature tokens, stopword-filtered, exact-sig excluded).
  `send_urgent_alert` ANNOTATES (never drops) each survivor; `_fmt` renders a
  `related:` line; `ALERT_PROMPT` gains a CONTINUITY rule (Sonnet leads
  ESCALATES/EXTENDS/FOLLOWS, frames CONTEXT as a follow-up). Non-suppressing
  by contract: a recency-store failure → `[]` → no annotation → exact
  pre-feature behaviour (a genuine alert must always still fire). Reads
  `alert_recency.db` only, NEVER `articles.db` — four invariants intact by
  construction. +14 tests incl. the live UAE-vs-futures no-false-link,
  recent_alerts TTL/degrade, integration (prompt carries hint AND alert still
  fires, scores untouched). NOTE: the `-m` body's backticked `` `related:` ``
  was eaten by bash command-substitution → commit body lost two words in one
  sentence (cosmetic, meaning intact); NOT force-fixed — a force-push to a
  shared branch with concurrent agents to repair a typo is not worth the race
  risk.

  **Phase 3 — analyst-lens live validation, user_findings=8.** (1)
  **Briefing EXCELLENT (positive)** — 07:13Z read end-to-end: decisive LEAD
  (bond rout 10Y+13bp→4.59% / Nasdaq −1.54% two days before NVDA earnings),
  exact MACRO, PORTFOLIO tied to live book (LITE/LNOK/NVDL/MU + DRAM C59
  05-22 / NVDA 05-20), specific RISK (watch 10Y>4.60%), sharp DESK NOTE,
  COVERAGE GAP present. (2) **Collection healthy** — 469 live art/h, newest
  ~3.5min fresh; web/reddit/substack/rss/google_news dominant. (3)
  **Invariants HOLD live** — `0` synthetic `urgency>=1`; `0` `ai_score>0 AND
  score_source='ml'` in the 1.45 GB prod DB. (4) **Alert path CLEAN** —
  exactly 2 alerts/24h, both legit `Benzinga Economics` geopolitical (01:55
  ai=9.0 UAE-drone/Brent; 09:19 ai=8.0 futures-drop); zero
  reddit/wiki/quote-widget noise in-window. The 09:19 is a continuation of
  01:55 with no framing — the exact gap the Phase-2 feature fixes (ships on
  restart). (5) **The 26 stuck urgency=1 rows** — Phase-1 finding, fixed in
  `50c1052`; live count still 26 (stale-daemon — reaped on the next 6h purge
  tick after a restart). (6) **8 collectors DARK** — `nitter` (1277 fails, 0
  delivered all session), `sec_edgar` (962, 0 — analyst BLIND to 8-K
  filings, priority-0), `polygon` (836, 0), `newsapi` (619, 0),
  `sec_edgar_ft` (194), `finnhub`/`gdelt` net-new-dedup false-disables
  (1957/7270 lifetime). COVERAGE GAP surfaces them but shows misleading
  "DARK 0.0h" — the `b20cbae` fix is committed, ships on restart
  (stale-daemon). Operational/upstream/key, not code bugs. (7) **Chronic
  DB-lock contention** — 22 `insert_batch: lock retry exhausted` + 2
  `update_ml_scores_batch` exhausted ERRORs → whole batches silently dropped
  = missed news from the analyst seat (memory
  `di-insert-batch-lock-contention`); real fix (per-call connection
  isolation) is substantial + daemon.py/store sibling-touched → out of clean
  scope, advisor-confirmed not chased. (8) **stats_worker NoneType recurring**
  (29×, latest 11:39Z) + one benign shutdown reentrant-logging Traceback —
  both stale-daemon symptoms of already-committed fixes (`05b406e`; the
  documented os._exit cleanup hazard), not new bugs. None of 6/7/8 is a
  quick safe fix in clean scope → no extra Phase-3 fold-in; bugs_fixed stays
  1, features_added 1. Final verify: `storage.article_store` / `ml.features`
  / `ml.model` / `watchers.alert_agent` / `watchers.alert_recency` imports
  OK; suite **715 passed**, the same 5 `test_rss_collector.py` failures are
  the pre-existing sibling `M collectors/rss_collector.py` WIP (not ours,
  never staged; floor held exactly 5, never 6+ every run; my 24 tests pass).
  *Pre-existing, deliberately never staged* (consistent with every prior
  entry): `collectors/rss_collector.py`, `daemon.py`, `dashboard/server.py`,
  `scripts/export_training_data.py`, `tests/test_article_store.py`, untracked
  sibling files, all `paper-trader/*`, `logs/*.tmp`. Both commits
  pathspec-scoped to exactly their intended files (`50c1052`: 2 files;
  `17d8df9`: 3 files); `git diff --staged --name-only` + `git show --stat`
  verified no sibling leakage; never `git add -A`; on origin/master. Entry
  appended, not rewritten.

- **2026-05-18 (hybrid pass 21 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass (21st; codebase exceptionally mature, 20 prior
  passes). Advisor-reviewed before substantive work and again before declaring
  done. `sqlite3` CLI absent → all probes via `python3` `sqlite3` `mode=ro`,
  index-friendly predicates only (no USB full-scan COUNT). Bare daemon
  `pid 1702195` still up, started **2026-05-18 07:29:24Z**, predating EVERY
  recent fix incl. `05b406e`/`b20cbae`/`50c1052` (the consistent stale-daemon
  caveat — committed fixes ship on next restart). Concurrent sibling agent +
  auto-commit/push daemon on the shared monorepo index (memory
  `di-shared-repo-concurrency`) → strict per-commit pathspec staging.

  **Phase 1 — bugs_fixed=0, NO Phase-1 commit (commit guard honoured).**
  Reviewed the full non-off-limits bug-hunt surface — required 9 files +
  `ml/inference.py` + `core/json_extract.py` + `watchers/alert_dedup.py` +
  `triage/heuristic_scorer.py` + `watchers/alert_recency.py` + `ml/embedder.py`.
  All uniformly hardened by the 20 prior passes; the requested storage/
  urgency_scorer/features/model/trainer tests already exist (verified, not
  duplicated). No genuine bug in clean scope. The recurring
  `[stats_worker] error: 'NoneType' object is not subscriptable` (12+×/h, last
  12:02:12Z) is **NOT a HEAD bug** — `_expect_row` (commit `05b406e`,
  2026-05-18 **11:23:06Z**) already fixes it; the running daemon started
  07:29Z, ~4h before the fix → executes pre-fix `article_store.py`. Confirmed
  by stashing the sibling-WIP `rss_collector.py` and re-running its tests
  (HEAD clean: 5/5 pass). Manufacturing a fix here would revert a load-bearing
  prior decision (advisor-confirmed) → bugs_fixed honestly 0.

  **Phase 2 — features_added=1, commit `097f912`** (`analysis/claude_analyst.py`
  +72, new `tests/test_briefing_book_heat.py`, 14 tests). **BOOK HEAT**: the
  5h Opus digest tells the analyst WHICH rows touch held positions (`[BOOK:]`
  tag) but never that a single held name is the window's centre of gravity —
  one MU story at 7.0 may not lead, but MU across 6 *distinct*
  (post-`_collapse_syndicated`) stories is a magnitude signal Opus cannot
  infer from per-row tags (it would have to tally 60 rows). Pure
  `_book_heat_lines()` counts distinct digest rows per held ticker over the
  already-collapsed+capped list Opus reads (syndicated copies of one event
  count once — honest + verifiable against the rendered newswire; snapshot
  rows with no url excluded, same guard as `[BOOK:]`), ranked count-desc then
  canonical `_BOOK_TICKERS` order, capped at 6. Emitted as a `=== BOOK HEAT
  ===` input block + a SYSTEM_PROMPT ranking-hint rule (LEAD/TOP-SIGNALS/
  PORTFOLIO consequence; explicitly NOT echoed, unlike COVERAGE GAP).
  Threshold ≥3 (conservative — analyst's top complaint is noise). Pure
  read-side: returns NEW lists, never mutates `source_articles`, no DB write,
  no ai_score/ml_score/score_source/urgency touch, backtest excluded upstream
  by `get_top_for_briefing`'s `_LIVE_ONLY_CLAUSE` — **all four load-bearing
  invariants intact by construction**. Mirrors the established `[syndicated
  xN]`/`[BOOK:]`/COVERAGE-GAP shape and anti-import-cycle discipline.

  **Phase 3 — user_findings=6 (analyst seat).** (1) **Stale daemon** (pid
  1702195, 07:29Z) predates `05b406e` *and* `reap_stale_urgent`: NoneType
  12+×/h still, plus `insert_batch`/`update_ml_scores_batch` *lock-retry
  exhausted* ERRORs at 11:11:15Z → a whole scored batch silently dropped
  (missed news from the analyst seat). Remedy: daemon restart applies all
  pending committed fixes. (2) **26 phantom `urgency=1` rows**, ALL dated
  2026-05-13 (5 days stale) — matches the `reap_stale_urgent` comment exactly;
  HEAD reaper present, stale daemon hasn't run it (purge every 6h; restart
  applies). Inflates the dashboard urgent tile with items never pushable. (3)
  **Alert noise (analyst-annoying)**: `[Wikipedia] Nvidia RTX` (8.6),
  `$NVIDIA (NVDA.US)$ - Moomoo` (9.8, quote-listing-page-like), and reddit
  forum posts (`r/ValueInvesting` 9.8, `r/Daytrading` 8.0) fired 🚨 BREAKING.
  Mostly pre-fix (stale daemon predates the lone-low-authority/quote-widget
  gates). Residual gap even post-restart: `wikipedia` cred 0.60 clears the
  0.45 lone gate — left as a finding, NOT fixed (the cred map is a
  deliberately tight, contested area prior reviews kept evidence-only; a
  unilateral pass-21 change risks reverting a load-bearing decision). (4)
  **Briefing quality high** (2026-05-18T07:13): crisp actionable LEAD (bond
  rout → semis selloff into NVDA print), RISK/CATALYST tied to held LITE/LNOK/
  NVDL/MU with the DRAM C59 expiry, COVERAGE GAP surfacing SEC-8-K dark —
  exactly the consumption BOOK HEAT augments. (5) **Collection healthy** —
  3166 live art/last-hour, ~1.45M/24h; briefing cadence ~5–7h (within the
  documented restart-churn tolerance; adaptive lookback + banner handle it).
  (6) **Sibling `M collectors/rss_collector.py`** is a concurrent agent's
  mid-edit (per-feed backoff WIP) that breaks its own 5 tests while HEAD is
  clean — ops-only, never staged, left exactly as-is. None of 1/2/3/6 is a
  quick safe fix in clean scope → no Phase-3 fold-in; bugs_fixed stays 0.

  **Verify:** `storage.article_store` / `ml.features` / `ml.model` /
  `analysis.claude_analyst` imports OK; suite **729 passed** (715 baseline +
  14 new), the same 5 `test_rss_collector.py` failures are the pre-existing
  sibling WIP (not ours, never staged; floor held exactly 5, never 6+; my 14
  tests all pass). *Pre-existing, deliberately never staged* (consistent with
  every prior entry): `collectors/rss_collector.py`, `daemon.py`,
  `dashboard/server.py`, untracked sibling files, all `paper-trader/*`,
  `logs/*.tmp`. Commit `097f912` pathspec-scoped to exactly 2 files;
  `git diff --staged --name-only` verified no sibling leakage; never
  `git add -A`; on origin/master. Entry appended, not rewritten.

- **2026-05-18 (hybrid pass 22 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass (22nd; codebase exceptionally mature, 21 prior
  passes). Advisor-reviewed before substantive work, again on a load-bearing
  test-fixture judgement call, and again before declaring done. All 9 required
  files + `ml/dedup.py` read in full; `sqlite3` CLI absent → all probes via
  `python3` `mode=ro`. Bare daemon `pid 1702195` still up, started
  **2026-05-18 07:29Z** (≈5h elapsed), predating every recent commit incl.
  `50c1052`/`b20cbae`/`097f912`/`c69560c` (the consistent stale-daemon caveat
  — committed fixes ship on next restart). Concurrent sibling agent +
  auto-commit/push daemon on the shared monorepo index (memory
  `di-shared-repo-concurrency`) → strict per-commit pathspec staging; the
  shared index raced (6 `paper-trader/*` files appeared staged between my two
  `git add` calls — `git commit -- <4 explicit paths>` committed exactly my
  4, zero sibling leakage, verified by `git show --stat`).

  **Phase 1 — bugs_fixed=0, NO Phase-1 commit (commit guard honoured).**
  Reviewed the full non-off-limits surface (9 required files +
  `ml/dedup.py` + the newest commits). All uniformly hardened by the 21 prior
  passes; the requested storage/urgency_scorer/features/model/trainer tests
  already exist (verified by name, not duplicated — `test_article_store.py`,
  `test_urgency_scorer.py`, `test_features.py`, `test_model.py`,
  `test_trainer.py`). Live evidence surfaced only KNOWN issues, none a genuine
  new bug in clean scope: chronic `insert_batch`/`update_ml_scores_batch`
  *lock retry exhausted* ERRORs (advisor-confirmed no-go: per-call connection
  isolation is substantial + daemon.py/store sibling-touched), 26 stuck
  `urgency=1` rows + historical alert noise + COVERAGE-GAP "DARK 0.0h" (all
  stale-daemon manifestations of fixes already at HEAD —
  `50c1052`/gate fixes/`b20cbae`). Manufacturing a fix would revert a
  load-bearing prior decision → bugs_fixed honestly 0 (precedent: passes
  15/16/17/21).

  **Phase 2 — features_added=1, commit `c69560c`** (`analysis/claude_analyst.py`
  +52, new `tests/test_briefing_near_dup_collapse.py` +181, +8 tests; 2
  fixture repairs). **Order-independent near-dup collapse wired into the Opus
  briefing.** `ml/dedup.py` (added `b4dfd48`, separately unit-tested, pure
  stdlib — `ml/__init__.py` empty so no numpy/torch pulled; its own docstring
  names "briefing pre-filter" as the intended integration) was built for
  exactly this gap but left **unwired**. `_collapse_syndicated` only merges an
  exact first-8-token prefix signature, so a word-reordered /
  source-attribution-suffixed copy of the SAME wire survives it and reaches
  the analyst's primary Opus digest as a duplicate TOP SIGNAL — their #1 noise
  complaint, on the one consumed product with no order-independent gate (live:
  the 07:13Z window carried 5 residual dups — bond-rout ×3, Trump-Intel ×1 —
  at sim 0.60-0.73, a full pairwise audit of that window found ZERO
  semantically-opposite pairs ≥0.60). Wired as a 2nd collapse stage
  (`_dedupe_near_duplicates`) after `_collapse_syndicated`, before
  `_rank_by_decayed_score`, threshold **0.7** (`BRIEFING_NEAR_DUP_THRESHOLD`).
  0.7 is conservative by design: a single-token ANTONYM flip in a 4-5 token
  headline ("Fed raises rates 25bp" vs "Fed cuts…" J=0.60; "…beat Q3" vs
  "…miss…" J=0.667) stays strictly below it, so opposite-direction stories
  are provably never merged — `tests/test_briefing_near_dup_collapse.py` pins
  this and the threshold value as defense-in-depth. Pure read-side, the SAME
  shape as `_collapse_syndicated`: returns the original dict objects, never
  mutates `source_articles`, no DB write, no
  ai_score/ml_score/score_source/urgency touch, backtest excluded upstream by
  `get_top_for_briefing`'s `_LIVE_ONLY_CLAUSE` — **all four load-bearing
  invariants intact by construction**. `dedupe_articles` reused verbatim (not
  forked) — a further-merged survivor keeps its OWN pre-merge `[syndicated
  xN]` count (conservative under-count, never over-stated), the documented
  anti-drift discipline. **Two existing cap-60 regression fixtures repaired
  (assertions UNCHANGED, advisor-confirmed this is fixture-defect repair, NOT
  test-weakening):** `test_claude_analyst.py::_articles` and
  `test_briefing_syndication_collapse.py` distinguished rows by a bare digit
  (`headline {i}`) — a len-1 token dropped by `ml.dedup`'s
  `_MIN_TOKEN_LEN=2`, so every "distinct" title normalized to the same token
  set and the new stage correctly collapsed them (latent fixture defect the
  feature exposes, not a feature bug). Genuinely-distinct `alpha{i}`/`topic{i}`
  tokens (J≈0.43/0.50 < 0.7) restore each test's stated intent; the cap-60
  contract is re-validated, not weakened.

  **Phase 3 — analyst-lens live validation, user_findings=8.** (1)
  **Collection healthy** — 447 live art/last-hour, newest ~0min fresh
  (GoogleNews round-robin / Benzinga / GlobeNewswire / scraped-yahoo / Seeking
  Alpha / Bloomberg dominant). (2) **Briefing cadence healthy** — last 5 gaps
  5.3/5.4/6.8/6.3h (target 5h, within documented restart-churn tolerance; the
  old 31.9h gap predates the heartbeat-cadence fix). (3) **Briefing quality
  EXCELLENT** (07:13Z, read end-to-end): decisive LEAD (bond rout 10Y
  +13bp→4.59% dragging Nasdaq −1.54% two days before NVDA earnings), exact
  MACRO, PORTFOLIO tied to the live held book (LITE/LNOK/NVDL/MU + DRAM C59
  05-22 / NVDA 05-20), specific RISK (watch 10Y>4.60%), sharp DESK NOTE,
  COVERAGE GAP present — exactly the consumption the Phase-2 dedup cleans up.
  (4) **Invariants HOLD live** — `0` synthetic `urgency>=1`; `0` `ai_score>0
  AND score_source='ml'` in the 1.39 GB prod DB. (5) **Alert path CLEAN
  post-fix** — the 2 most recent alerts (2026-05-18 01:55 ai=9.0 UAE-drone/
  Iran, 09:19 ai=8.0 futures-drop, both Benzinga Economics geopolitical) are
  legit, no reddit/wiki/quote-widget noise in-window; the 09:19 is an
  unframed continuation of 01:55 (the exact gap `17d8df9` fixes, ships on
  restart). Historical noise (reddit r/ValueInvesting 9.8, r/Daytrading 8.0,
  Wikipedia 8.6, quote-widget "NVDANVIDIA Corporation227.13…") all
  05-15..05-17, predating the lone-low-authority/quote-widget gates —
  stale-daemon. (6) **8 collectors DARK** — `sec_edgar` (968 fails, 0
  delivered — analyst BLIND to 8-K filings, priority-0), `nitter` (1283, 0),
  `polygon` (841, 0), `newsapi` (621, 0), `sec_edgar_ft` (197, 3);
  massive/wikipedia transient net-new-dedup false-disable (high delivered).
  COVERAGE GAP surfaces them; the 07:13 briefing showed "DARK 0.0h" because
  the running daemon predates `b20cbae` (HEAD uses fails×cadence; ships on
  restart). Operational/upstream/key, not code bugs. (7) **Chronic DB-lock
  contention** — recurring `insert_batch`/`update_ml_scores_batch` *lock
  retry exhausted* ERRORs (latest 12:09Z) → whole batches silently dropped =
  missed news from the analyst seat (memory
  `di-insert-batch-lock-contention`); real fix out of clean scope
  (advisor-confirmed not chased). (8) **26 phantom `urgency=1` rows** all
  dated 2026-05-13 (5 days stale), inflating the dashboard urgent tile with
  never-pushable items — `reap_stale_urgent` (`50c1052`) present at HEAD, the
  stale daemon hasn't run a post-fix purge. None of 5/6/7/8 is a quick safe
  fix in clean scope (stale-daemon-with-HEAD-fix / operational-upstream /
  advisor-confirmed no-go) → no Phase-3 fold-in; bugs_fixed stays 0,
  features_added stays 1.

  **Verify:** `storage.article_store` / `ml.features` / `ml.model` /
  `analysis.claude_analyst` / `ml.dedup` imports OK; suite **757 passed**
  (749 baseline + 8 new), the same 5 `test_rss_collector.py` failures are the
  pre-existing sibling `M collectors/rss_collector.py` WIP
  (`'_FakeResp' object has no attribute 'status_code'` — not ours, never
  staged; floor held exactly 5, never 6+; my 8 tests + the 2 repaired
  existing tests all pass). *Pre-existing, deliberately never staged*
  (consistent with every prior entry): `collectors/rss_collector.py`,
  `daemon.py`, `dashboard/server.py`, `scripts/export_training_data.py`,
  `tests/test_article_store.py`, untracked sibling files, all
  `paper-trader/*`, `logs/*.tmp`. Commit `c69560c` pathspec-scoped via
  `git commit -- <4 explicit paths>` to exactly
  `analysis/claude_analyst.py` + `tests/test_briefing_near_dup_collapse.py` +
  `tests/test_claude_analyst.py` + `tests/test_briefing_syndication_collapse.py`;
  `git show --stat` verified no sibling/`paper-trader` leakage despite the
  racing shared index; never `git add -A`; on origin/master. Entry appended,
  not rewritten.

- **2026-05-18 (hybrid pass 23 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass (23rd; codebase exceptionally mature, 22 prior
  passes). Advisor-reviewed before each substantive phase. Live evidence was
  the discovery engine (proven pattern of passes 14/16/17/18/19/20). `sqlite3`
  CLI absent → `python3` `mode=ro` probes (timed out >90s under live daemon
  contention, the documented USB-I/O saturation; one short-window probe later
  succeeded). Bare daemon `pid 1702195` still up, started **2026-05-18
  07:29Z**, predating EVERY recent fix incl. `8180055`/`84bc881`/`50c1052`/
  `b20cbae` (the consistent stale-daemon caveat — fixes ship on next restart).
  Concurrent sibling agent + auto-commit/push daemon on the shared monorepo
  index (memory `di-shared-repo-concurrency`) → strict per-commit pathspec
  staging; the shared index advanced between my two pushes (`6e9c5d8`→…,
  `d714dcb`→`84bc881`) but neither commit captured a foreign file
  (`git show --stat` verified).

  **Phase 1 — bugs_fixed=1, commit `8180055`** (`storage/article_store.py` +
  new `tests/test_retry_on_lock_not_an_error.py`). **Live-log discovery →
  root-caused → fixed.** `daemon.log`: `[recursive_labeler] error: not an
  error` at 12:09:20Z landed exactly at the onset of a `database is locked`
  writer-contention storm (insert_batch/update_ml_scores_batch exhausting
  12:09:24-32Z). `_retry_on_lock`'s `_RETRYABLE_DB_ERRORS` covered `database
  is locked` / `another row available` / `another row pending` / `no more rows
  available` but NOT `not an error` — the `pysqlite` `SQLITE_OK` (errno-0)
  default message, surfaced when a concurrent writer on the shared
  `check_same_thread=False` `self.conn` resets the statement state mid-call:
  the SAME shared-connection cursor-collision class as `bec95ea` (pass 14,
  "no more rows available") and `05b406e` (pass 19, `_expect_row`
  `'NoneType'`), just a different surfaced string. **Advisor's
  verification gate corrected an initial misdiagnosis:** the colliding call is
  NOT `_fetch_round1_candidates` (a raw uncovered `store.conn.execute`) — the
  log shows `round=1 candidates=500` SUCCEEDED before BOTH the 08:01
  ("no more rows available", pre-`bec95ea` on the stale daemon) and 12:09
  ("not an error") errors, so the collision hit the `@_retry_on_lock`-decorated
  `update_ai_scores_batch.executemany` inside round-1's `_apply_labels`. So the
  fix is minimal — add the string to the allowlist + a documenting comment
  item 4 (the colliding op is already decorated and idempotent; NO store-method
  refactor, the gate prevented a wrong-shaped change). Impact: the
  recursive_labeler had **ZERO successful runs since the 07:29Z daemon start**
  (`last_ok=n/a`; last success 03:33Z `total_labeled=418` on the *previous*
  daemon) — each collision aborted the entire 4h Sonnet/Opus gold-label
  cycle, the model's strongest active-learning signal. Genuine HEAD bug (the
  string is absent from HEAD's allowlist); ships on next `systemctl restart
  digital-intern`. +5 tests mirroring `tests/test_retry_on_lock_no_more_rows.py`
  (retry-then-succeed, substring-embed, IntegrityError still propagates,
  budget-exhaust+`lock_failures`, tuple anti-drift). `tests/test_article_store.py`
  left untouched (sibling-WIP).

  **Phase 2 — features_added=1, commit `84bc881`**
  (`analysis/claude_analyst.py` + new `tests/test_briefing_aging_rows.py`).
  **AGING TOP ROWS — deterministic wall-clock recency cross-check.** The
  model-estimated `time_sensitivity` decay rerank demotes stale time-bound
  rows only as far as the ts head scored them; an under-scored row stays
  time-bound yet barely decays and a sparse 5h window floats a 5-6h-old item
  to #1. Opus then has only the per-row `[seen HH:MM UTC]` clock + the
  `BRIEFING TIME` header, and LLM clock subtraction across a bare-HH:MM 24h
  window is unreliable — so a multi-hour-old developing story can be written
  into the LEAD as if it just broke (the recurring stale-framing complaint, on
  the analyst's primary product). New pure `_aging_top_rows()` emits a
  deterministic wall-clock age for the highest-ranked digest rows (an
  INDEPENDENT ground-truth cross-check on the model decay, NOT a
  re-expression). **Design note for future passes:** a per-row `[age N]`
  token (mirroring the alert path's `0792a57`) was explicitly rejected —
  `tests/test_briefing_seen_timestamp.py:69` pins the EXACT contiguous
  render-line prefix `"[score=9.0] [seen 14:32 UTC] [rss]"`, so ANY new
  inline per-row token breaks that tracked assertion and the task forbids
  weakening existing tests. The correct shape is the established BOOK-HEAT /
  COVERAGE-GAP one: a separate `=== AGING TOP ROWS ===` input block (zero
  render-line change → contiguity intact), never echoed (a framing hint, like
  BOOK HEAT, unlike COVERAGE GAP), computed over the same `deduped[:60]` Opus
  reads, + a SYSTEM_PROMPT rule. 3.0h threshold mirrors the alert path's
  documented "materially old (≳3h)" RECENCY bar (cross-product parity); only
  the top `_AGING_TOP_SCAN=10` rows scanned (Opus leads from the top), capped
  at 6; `_seen_age_hours` reused verbatim (anti-drift); real-url snapshot
  guard mirrors `[BOOK:]`. Pure read-side: no DB write, no
  ai_score/ml_score/score_source/urgency touch, no `source_articles`
  mutation, backtest excluded upstream — **all four invariants intact by
  construction**. +14 specific-value tests (exact 3.0h boundary, rank/cap,
  snapshot+unknown-age exclusion, non-mutation, `_build_payload` emission
  gate, verbatim SYSTEM_PROMPT rule). All 143 briefing-suite tests pass
  (incl. the unchanged `test_briefing_seen_timestamp` contiguity assertion).

  **Phase 3 — analyst-lens live validation, user_findings=7.** (1)
  **recursive_labeler ZERO successful runs since 07:29Z** — the Phase-1
  finding; 08:01 "no more rows available" (pre-`bec95ea`, stale daemon),
  12:09 "not an error" (the HEAD bug, fixed in `8180055`); ships on restart.
  (2) **Chronic DB lock-retry exhaustion** — 32 `lock retry exhausted` in the
  current `daemon.log` + many `database is locked` worker backoffs (finnhub/
  reddit/scorer/ticker/web/yahoo_ticker_rss/google_news/wikipedia clusters
  12:09, 12:28-34, 12:49-13:06) → whole collected/scored batches silently
  dropped = missed news (memory `di-insert-batch-lock-contention`). Root fix
  (per-call connection isolation) substantial + daemon.py/store sibling-touched
  → out of clean scope (advisor/precedent-confirmed); my Phase-1 removes ONE
  symptom of this exact storm. (3) **6 collectors disabled** (`source_health`
  `disabled=6 stale=0 down=6` unchanged through 13:24Z) — analyst blind to
  those channels; the COVERAGE GAP briefing block surfaces it (working as
  intended); upstream/operational. (4) **Alert path CLEAN & quiet
  (positive)** — exactly 2 BN alerts in 24h (03:03Z, 09:26Z, 1 distinct
  story each); zero noise/suppression churn; the full noise-suppression stack
  behaving on a quiet window. (5) **Briefing cadence HEALTHY (positive)** —
  heartbeats 01:54Z (2280 ch) → 07:13Z (2315 ch) → 12:51Z (2777 ch),
  gaps ≈ 5.3h / 5.6h vs the 5h target (the `ef839a8` heartbeat-clock fix
  holding; no 30h+ gaps), all delivered OK. (6) **Briefing quality EXCELLENT
  (positive, direct read)** — id=27 (12:51Z, 50 arts) read end-to-end: dense,
  exact, decisively-actionable Bloomberg LEAD ("Iran-war inflation scare →
  global bond rout, US 30Y 5.13% post-2023 high, S&P -1.24% / SMH -3.80%
  into NVDA Wed earnings — but the live tape is already cooling, WTI -4.15%,
  bond selloff easing"); precise MACRO table. (7) **Collection HEALTHY
  (positive)** — gdelt per-query ingestion diverse & current through 13:24Z
  (Middle East conflict=43, Italy economy=53, Samsung semis=15, DRAM memory
  pricing, NVDA earnings, SEC 13F); newest sweep ~min-fresh. None of 2/3 is a
  new safe quick fix in clean scope (2 operational+sibling-touched, advisor-
  confirmed not chased; 3 upstream) → no extra Phase-3 fold-in; bugs_fixed
  stays 1, features_added 1.

  **Verify:** `storage.article_store` / `ml.features` / `ml.model` /
  `analysis.claude_analyst` imports OK; suite **786 passed / 5 failed** (the 5
  are the pre-existing sibling `M collectors/rss_collector.py` 4-tuple WIP,
  `'_FakeResp' object has no attribute 'status_code'` — not ours, never
  staged; floor held exactly 5, never 6+; my 19 new tests all pass; the >757
  prior-baseline delta includes concurrent-sibling test files). *Pre-existing,
  deliberately never staged* (consistent with every prior entry):
  `collectors/rss_collector.py`, `daemon.py`, `dashboard/server.py`,
  `scripts/export_training_data.py`, `tests/test_article_store.py`, untracked
  `collectors/fred_collector.py` / `scripts/stale_source_alerter.py` /
  `storage/story_corroboration.py` / `tests/test_alert_history.py` /
  `tests/test_export_training_data.py` / `tests/test_story_corroboration.py`,
  all `paper-trader/*`, `logs/*.tmp`. Both commits pathspec-scoped to exactly
  their 2 intended files (`8180055`: `storage/article_store.py` +
  `tests/test_retry_on_lock_not_an_error.py`; `84bc881`:
  `analysis/claude_analyst.py` + `tests/test_briefing_aging_rows.py`);
  `git diff --staged --name-only` + `git show --stat` verified no sibling
  leakage; never `git add -A`; both on origin/master. A concurrent sibling
  hybrid agent edited this repo throughout; this entry was appended, not
  rewritten.

- **2026-05-18 (hybrid pass 24 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass (24th; codebase exceptionally mature, 23 prior
  passes). Advisor-reviewed before substantive work AND on the empirical
  match-rate pivot. All 9 required files read in full + `ml/inference.py`
  context. Bare daemon `pid 1702195` still up, started **2026-05-18 07:29Z**,
  predating EVERY recent fix (`8180055`/`84bc881`/`50c1052`/`05b406e`/
  `b20cbae`) — the consistent stale-daemon caveat (fixes ship on next
  restart). Concurrent sibling agent + auto-commit/push daemon on the shared
  monorepo index (memory `di-shared-repo-concurrency`) → strict per-commit
  pathspec staging; HEAD advanced under me (`9cb7a2e`→`ecafe10` paper-trader
  AGENTS sweeps) but my commit captured zero foreign files (`git show --stat`
  verified).

  **Phase 1 — bugs_fixed=0, NO Phase-1 commit (commit guard honoured —
  honest, not a miss).** Every load-bearing invariant re-traced and multiply
  defended; the full requested test list already exists and value-asserts.
  Live `daemon.log` forensics surfaced only KNOWN issues, none a genuine new
  bug in clean scope: the 37 `[stats_worker] 'NoneType'` + 1
  `[recursive_labeler] not an error` are stale-daemon manifestations of
  HEAD-present fixes (`_expect_row` `05b406e`; `_RETRYABLE_DB_ERRORS` already
  contains `"not an error"` `8180055` — both verified at HEAD); the 30
  `insert_batch`/`update_ml_scores_batch` `lock retry exhausted` ERRORs are
  the chronic DB-lock contention (memory `di-insert-batch-lock-contention`;
  per-call connection isolation is substantial + `daemon.py`/store
  sibling-touched → out of clean scope, advisor/precedent-confirmed not
  chased — precedent passes 19/20/21/22/23). Manufacturing a fix would revert
  a load-bearing prior decision → bugs_fixed honestly 0 (precedent passes
  15/16/17/21/22).

  **Phase 2 — features_added=1, commit `aebcbbd`** (`analysis/claude_analyst.py`
  +159/−1 + new `tests/test_briefing_prior_digest.py`, +30 tests).
  **PRIOR DIGEST continuity hint — anti-rehash on the 5h heartbeat.** A news
  analyst reading consecutive heartbeats complains most about repetition
  (documented #1 noise complaint). **Confirmed live this pass:** briefing id26
  (07:13Z) and id27 (12:51Z, 5.6h later) BOTH LED with the
  global-bond-rout-into-NVDA-earnings story (MACRO table rows byte-identical
  between them). The alert path has alert↔briefing parity (`[ALERTED]`); the
  briefing path never saw its OWN previous output. **Empirical pivot
  (advisor-gated):** a per-article-title match vs the rendered prior briefing
  was measured at **0% recall** (400 recent titles, 0 hits — Opus paraphrases
  every headline), so the per-row-tag mechanism is dead. Pivoted (the advisor
  pre-authorised this exact direction) to parsing the prior briefing's OWN
  deterministic `SYSTEM_PROMPT` format (the literal `**LEAD:**` line +
  `**TOP SIGNALS**` fenced block) and feeding it back as a framing hint — Opus
  does the semantic "same story?" comparison (its strength), the established
  BOOK-HEAT/AGING shape (separate input block, never a per-row token so the
  pinned `test_briefing_seen_timestamp.py:69` contiguity assertion is
  untouched, never echoed). New `_parse_prior_digest` (pure),
  `_prior_digest_lines` (pure), `_recent_briefing_digest` (best-effort, lazy
  fresh `mode=ro` connection — NEVER the shared `self.conn`; one O(log N)
  read of the tiny `briefings` table; ANY failure → None; the
  `[analyst] No response` sentinel rows — **3 of 27 live** — filtered in SQL
  so the newest *real* digest wins), `_build_payload(..., prior_digest=None)`
  (None ⇒ omitted, deterministic, 4-arg path byte-unchanged — exact
  `source_health_report` discipline; `analyze()` signature unchanged so
  `daemon.py:1477` still works), one new `SYSTEM_PROMPT` rule (existing
  BOOK HEAT/AGING/[ALERTED]/COVERAGE-GAP rules byte-unchanged, pinned by an
  anti-regression test). The `briefings` table holds only Opus-rendered rows
  (synthetic backtest rows live in `articles`, NEVER here) so backtest
  isolation holds by construction; no `articles.db` write, no
  ai_score/ml_score/score_source/urgency touch, `source_articles` never
  read/mutated — **all four load-bearing invariants intact by construction**
  (same safety class as `_collect_source_health`/`_recent_alert_signatures`).
  Ships on next `systemctl restart digital-intern` (stale-daemon caveat).

  **Phase 3 — analyst-lens live validation, user_findings=7.** (1)
  **Briefing repetition CONFIRMED LIVE** — id26 & id27 both LEAD
  bond-rout→NVDA (the Phase-2 driver; fix ships on restart). (2) **Briefing
  quality EXCELLENT (positive)** — id27 read end-to-end: dense, exact,
  decisively-actionable (Iran-war inflation/bond-rout LEAD, 30Y 5.13%
  post-2023 high, S&P −1.24% / SMH −3.80% into NVDA Wed, "tape already cooling
  WTI −4.15%" nuance, precise MACRO/PORTFOLIO/SEMIS/RISK/DESK-NOTE, COVERAGE
  GAP present). (3) **Invariants HOLD live** — `0` synthetic `urgency>=1`;
  `0` `ai_score>0 AND score_source='ml'` in the prod DB. (4) **Collection
  healthy** — 4170 live articles last 1h. (5) **Alert path** — 2 legit
  high-value `Benzinga Economics` geopolitical alerts (UAE-drone/Brent ai=9,
  futures-drop ai=8) + SEC-EDGAR NVDA 8-K (ai=8); lone `reddit/r/ValueInvesting`
  (ml=9.76) / `reddit/r/Daytrading` (ai=8) / `Wikipedia` (ml=8.63) residue
  predate the deployed `_filter_low_authority_lone`/quote-widget gates
  (stale-daemon — reddit 0.40 gated post-restart; Wikipedia 0.60 above the
  0.45 bar = the standing deferred contested tuning, NOT chased — precedent
  passes 15/16/21/22). (6) **26 phantom `urgency=1` rows** — `reap_stale_urgent`
  (`50c1052`) present at HEAD; stale daemon hasn't run a post-fix purge;
  inflates the dashboard urgent tile. (7) **7 collectors disabled**
  (`massive, newsapi, nitter, polygon, sec_edgar, sec_edgar_ft, wikipedia`);
  `sec_edgar`/`_ft` = analyst blind to 8-K filings (priority-0) — correctly
  surfaced verbatim by the COVERAGE GAP briefing block (working as intended);
  upstream/rate-limit, operational. None of 5/6/7 is a quick safe fix in
  clean scope (stale-daemon-with-HEAD-fix / contested-test-pinned tuning /
  upstream) → no Phase-3 fold-in; bugs_fixed stays 0, features_added 1.

  **Verify:** `storage.article_store` / `ml.features` / `ml.model` /
  `analysis.claude_analyst` imports OK; suite **845 passed / 5 failed** (the 5
  are the pre-existing sibling `M collectors/rss_collector.py`
  `'_FakeResp' object has no attribute 'status_code'` 4-tuple WIP — not ours,
  never staged; floor held exactly 5, never 6+; my 30 new tests all pass;
  briefing+claude_analyst suites 249 passed, zero regressions vs the 213
  pre-change baseline). *Pre-existing, deliberately never staged* (consistent
  with every prior entry): `collectors/rss_collector.py`, `daemon.py`,
  `dashboard/server.py`, `scripts/export_training_data.py`,
  `tests/test_article_store.py`, untracked `collectors/fred_collector.py` /
  `scripts/stale_source_alerter.py` / `storage/story_corroboration.py` /
  `tests/test_alert_history.py` / `tests/test_export_training_data.py` /
  `tests/test_story_corroboration.py`, all `paper-trader/*`, `logs/*`. Commit
  `aebcbbd` pathspec-scoped via `git commit -F … -- <2 explicit paths>`;
  `git diff --staged --name-only` + `git show --stat` verified no sibling
  leakage; never `git add -A`; on origin/master. A concurrent sibling hybrid
  agent edited this repo throughout; this entry was appended, not rewritten.

- **2026-05-18 (hybrid pass 26 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass (codebase exceptionally mature, 25 prior passes).
  All 9 required files read in full + `ml/label_audit.py` (HEAD `c4339b7`),
  `ml/inference.py`, `collectors/source_health.py`. Advisor-reviewed before
  substantive work. Bare daemon `pid 1702195` started **2026-05-18 07:29Z**
  (00:29 local -0700), predating EVERY recent fix incl. HEAD `c4339b7`
  (14:30Z), `b20cbae` COVERAGE-GAP cadence fix (08:16Z), `50c1052`
  reap_stale_urgent, `8180055`/`05b406e` cursor-collision retries — the
  consistent stale-daemon caveat (all ship on next `systemctl restart
  digital-intern`). A concurrent sibling hybrid agent (`pid 1958258`, same
  prompt) + auto-commit/push daemon edited this shared monorepo throughout;
  strict per-commit pathspec staging held (memory `di-shared-repo-concurrency`).

  **Phase 1 — bugs_fixed=0, NO Phase-1 commit (commit guard honoured —
  honest, not a miss).** Every load-bearing invariant re-traced and verified
  live (`synth_urgent_LEAK=0`, `ml_in_aiscore_LEAK=0` in the prod DB). The
  full requested Phase-1 test list already exists and value-asserts
  (`test_article_store` backtest:// + `update_ml_scores_batch` score_source,
  `test_trainer` ml-exclusion + sample-weight, `test_urgency_scorer`
  9.5-urgent/3.0-not/rescore-does-not-unalert, `test_features` 15-dim/density/
  age, `test_model` head bounds/NaN). Live `daemon.log` forensics surfaced
  only KNOWN issues, none a genuine new bug in clean scope: the recurring
  `[stats_worker] 'NoneType' object is not subscriptable` + the 14:34:46Z
  `update_ai_scores_batch: lock retry exhausted` → `[urgency] Scoring error`
  traceback are the chronic shared-conn DB-lock contention (memory
  `di-insert-batch-lock-contention`) and a stale-daemon manifestation of the
  HEAD-present `_expect_row`/`_RETRYABLE_DB_ERRORS` fixes; the line-427
  `reentrant call inside BufferedWriter` traceback is the PRIOR daemon's
  23:42Z shutdown logging artifact, not the live process. Root fix (per-call
  connection isolation) is substantial + `daemon.py`/`article_store.py`
  sibling-touched → out of clean scope (advisor/precedent-confirmed, passes
  19-24). Manufacturing a fix would revert a load-bearing prior decision →
  bugs_fixed honestly 0 (precedent passes 15/16/17/21/22/24).

  **Phase 2 — features_added=1, commit `56974f8`** (`watchers/alert_agent.py`
  +52/−1 + new `tests/test_alert_book_tag.py`, +14 tests).
  **Held-book relevance line on the 🚨 BREAKING urgent alert.** The alert is
  the analyst's most time-critical product and the persona is explicitly "I
  react to events affecting MY positions", yet the mandatory `PORTFOLIO:`
  line relied entirely on Sonnet *inferring* held-ticker relevance from the
  raw headline — a real held-name break read identically to generic macro
  colour, and a "Lumentum guides down" with no `LITE` token got a generic
  PORTFOLIO line. The briefing path already has the well-tested `[BOOK:]`
  tag; the alert path (the more urgent product) had no held-book signal at
  all. New pure `_book_tickers(art)` (title+summary surface, sorted/dedup,
  reuses `ml.features.LIVE_PORTFOLIO_TICKERS`/`_LIVE_RE` **verbatim** —
  alert_agent already imports `_source_credibility` from that module, so
  single-source-of-truth with the model's own ticker features and the
  briefing tag, zero drift) emits an additive `book: TICKER,...` line in
  `_fmt` (exact shape of the established additive `age:`/`syndication:`/
  `related:` lines — membership-tested, no pinned contiguity, verified via
  grep before writing) + one BOOK rule in `ALERT_PROMPT` so Sonnet MUST name
  the held ticker(s) with a concrete directional implication and weight IMPACT
  above generic macro. **Design note for future passes:** the briefing's
  `_BOOK_TICKERS` is a *local literal* (analysis layer must not pull
  ml/numpy); alert_agent is the OPPOSITE — it ALREADY pulls the ml.features
  numpy graph, so reusing that module's set is the correct drift-free choice
  here (a `test_alert_book_tag` drift-guard pins set-equality with
  `LIVE_PORTFOLIO_TICKERS`). `ALERT_PROMPT` text is NOT pinned by any test
  (grepped `FORMAT (use exactly)`/`PORTFOLIO:`/`LITE/MU/MSFT` → no test
  hits), so the new rule is safe. The hardcoded 7-ticker list in the prompt
  FORMAT block (`LITE/MU/MSFT/AXTI/ORCL/TSEM/QBTS`, missing LNOK/MUU/DRAM/
  SNDU/NVDA) was deliberately NOT widened in this commit — separate concern,
  the `book:` data line carries the full 12-name truth to Sonnet anyway.
  Pure read-side: no DB write, no ai_score/ml_score/score_source/urgency
  touch, backtest already filtered by `_is_synthetic`/the store before
  `_fmt` — **all four load-bearing invariants intact by construction**.
  +14 specific-value tests (pure helper: single/multi-sorted, summary
  surface, `MUU` not swallowed by `\bMU\b`, `MU` not matched inside
  "Micron", dedup, empty-safe, non-portfolio AAPL excluded, ml.features
  single-source-of-truth set-equality; end-to-end: `book:` line + BOOK rule
  reach the Sonnet prompt, multi-ticker sorted, no-position row emits NO
  `book:` line — no fabrication; read-only `spy.marked` contract). All 112
  alert-suite tests pass (incl. the unchanged continuation/age/dedup/
  source-authority assertions). Ships on next daemon restart (stale caveat).

  **Phase 3 — analyst-lens live validation, user_findings=8.** (1)
  **Briefing quality EXCELLENT (positive, direct read)** — id27 (12:51Z, 50
  arts) read end-to-end: dense, exact, decisively-actionable LEAD ("Iran-war
  inflation scare → global bond rout, US 30Y 5.13% post-2023 high, S&P
  −1.24%/SMH −3.80% into NVDA Wed — but the live tape is already cooling, WTI
  −4.15%"), precise MACRO/PORTFOLIO/SEMIS tables, RISK tied to specific
  levels (10Y >4.65%, NVDA $225 pivot), syndication `[x2]` tags in TOP
  SIGNALS, COVERAGE GAP present. (2) **Collection HEALTHY (positive)** —
  4,449 live articles last 1h, 1.45M/24h; diverse GN round-robin + scraped +
  Benzinga, current. (3) **Invariants HOLD live (positive)** — `0` synthetic
  `urgency>=1`, `0` `ai_score>0 AND score_source='ml'`. (4) **Alert path
  CLEAN & quiet (positive)** — `[alert] idle — no urgent items`, `state=ok
  crashes_5m=0`, zero noise/suppression churn this window; recent legit
  alerts only (Benzinga geopolitical ai=9/8, SEC-EDGAR NVDA 8-K ai=8). (5)
  **COVERAGE GAP shows "DARK 0.0h"** for session-long-blind channels (SEC
  8-K 968 empty polls, Polygon 841, NewsAPI 621, Nitter 1283) — misleading
  to the analyst (reads as negligible), but a STALE-DAEMON manifestation of
  HEAD-present `b20cbae` (daemon 07:29Z predates the 08:16Z fix); ships
  correct (cadence-based `~Nh`) on restart, NOT a new bug. (6) **7 collectors
  disabled** (`massive, newsapi, nitter, polygon, sec_edgar, sec_edgar_ft,
  wikipedia`); `sec_edgar`/`_ft` = analyst blind to 8-K filings (priority-0);
  chronic external/rate-limit gap (memory `di-chronic-dark-collectors`),
  correctly surfaced verbatim by the COVERAGE GAP block (working as
  intended); upstream/operational. (7) **Chronic DB lock-retry exhaustion**
  — `update_ai_scores_batch: lock retry exhausted after 5 attempts` at
  14:34:46Z → `[urgency] Scoring error` dropped that cycle's Sonnet labels =
  potential missed urgent classification (memory
  `di-insert-batch-lock-contention`); root fix substantial +
  daemon.py/store sibling-touched → out of clean scope (advisor/precedent-
  confirmed). (8) **Stale daemon predates ALL recent HEAD fixes** + 26
  phantom `urgency=1` rows (reap_stale_urgent `50c1052` present at HEAD,
  un-run on the stale process; inflates the dashboard urgent tile) — the
  meta-finding: an operator `systemctl restart digital-intern` ships pass
  19-26's accumulated fixes + this pass's `book:` line. None of 5/6/7/8 is a
  new safe quick fix in clean scope (stale-daemon-with-HEAD-fix / upstream /
  chronic-out-of-scope / operational) → no Phase-3 fold-in; bugs_fixed stays
  0, features_added 1.

  **Verify:** `storage.article_store` / `ml.features` / `ml.model` /
  `watchers.alert_agent` imports OK; `_book_tickers` set-parity with
  `ml.features.LIVE_PORTFOLIO_TICKERS` True; suite **863 passed / 5 failed**
  (`--ignore=tests/test_alert_history.py`, an untracked sibling-WIP file
  importing a nonexistent `watchers.alert_history`; the 5 failures are the
  pre-existing sibling `M collectors/rss_collector.py` `'_FakeResp' object
  has no attribute 'status_code'` 4-tuple WIP — not ours, never staged;
  floor held exactly 5, never 6+; my 14 new tests all pass, 112/112
  alert-suite green, zero regressions). *Pre-existing, deliberately never
  staged* (consistent with every prior entry): `collectors/rss_collector.py`,
  `daemon.py`, `dashboard/server.py`, `scripts/export_training_data.py`,
  `tests/test_article_store.py`, untracked `collectors/fred_collector.py` /
  `scripts/stale_source_alerter.py` / `storage/story_corroboration.py` /
  `tests/test_alert_history.py` / `tests/test_export_training_data.py` /
  `tests/test_story_corroboration.py`, all `paper-trader/*`, `logs/*`.
  Commit `56974f8` pathspec-scoped via `git commit -F … -- watchers/
  alert_agent.py tests/test_alert_book_tag.py`; `git diff --staged
  --name-only` + `git show --stat` verified EXACTLY 2 files (213 ins / 1
  del), no sibling leakage; never `git add -A`; on origin/master. A
  concurrent sibling hybrid agent edited this repo throughout; this entry
  was appended, not rewritten.

- **2026-05-18 (hybrid pass 27 — Agent 3, debug + feature + analyst-validation)** —
  Required-file-set pass (27th; codebase exceptionally mature, 26 prior
  passes). Advisor-reviewed before substantive work. All 9 required files +
  AGENTS.md read in full. Bare daemon `pid 1702195` started **2026-05-18
  ~07:30Z** (etimes ~28.3k s), predating EVERY recent HEAD fix — the
  consistent stale-daemon caveat. A concurrent sibling hybrid agent
  (`pid 1979386`, the EXACT same prompt) + auto-commit/push daemon edited the
  shared monorepo throughout → strict per-commit pathspec staging (memory
  `di-shared-repo-concurrency`).

  **Phase 1 — bugs_fixed=0, NO Phase-1 commit (commit guard honoured —
  honest, not a miss; advisor-confirmed).** Every error in live `daemon.log`
  forensics maps to (a) **fixed-at-HEAD on the stale daemon** —
  `[stats_worker] 'NoneType' object is not subscriptable` ×65 (`_expect_row`
  `05b406e`), `[scorer_worker] no more rows available` ×3 (`bec95ea`), 26
  stuck `urgency=1` rows (`reap_stale_urgent` `50c1052`), COVERAGE-GAP "0.0h"
  (`b20cbae`) — (b) **sibling WIP** — `rss_collector.py` 4-tuple
  (`string indices must be integers` ×19) — or (c) the **chronic
  shared-conn lock-exhaustion** (44 `lock retry exhausted` + an
  `update_ai_scores_batch`-retry-exhausted Traceback at
  `urgency_scorer.py:188` → a whole Sonnet-labelled batch dropped =
  potential missed urgent classification); per-call connection isolation is
  substantial + `daemon.py`/store sibling-touched → out of clean scope
  (advisor/precedent-confirmed, passes 19–26). The `[ticker_worker] another
  row available` ×1 is already in `_RETRYABLE_DB_ERRORS` (budget-exhausted,
  same class as the 44). Invariants verified LIVE: `0` synthetic
  `urgency>=1`, `0` `ai_score>0 AND score_source='ml'` in the ~1.46 GB prod
  DB. No genuine new bug in clean scope; the full requested Phase-1 test list
  already exists and value-asserts (precedent passes 15/16/17/21/22/24/26).

  **Phase 2 — features_added=1, commit `3135718`** (3 src + 3 test, +224/−19,
  pathspec-scoped, `git show --stat` verified no sibling leak, on
  origin/master). **Quote-listing share-card fingerprint** added byte-
  identically (`_QW_LISTING`) to the THREE lockstep `_looks_like_quote_widget`
  gates (`collectors/web_scraper.py`, `watchers/alert_agent.py`,
  `analysis/claude_analyst.py`). **Live + recurring evidence:** the row
  `$NVIDIA (NVDA.US)$ - Moomoo` (a Moomoo/Futu/Webull "share this quote"
  landing page, NOT an article) from the `GN: Nvidia` collector, ML-relevance
  over-scored `ml_score=9.77`/`ai_score=0`, fired a `urgency=2` 🚨 BREAKING
  push AND reaches the top-60 Opus newswire as a fake TOP SIGNAL — documented
  as a noise complaint across ≥6 prior passes but never fingerprint-gated
  (only the *cred-bar* approach was deferred as contested tuning; a
  fingerprint gate is the accepted quote-widget precedent, passes 14/16). The
  two existing fingerprints (letter-glued price, parenthesised signed %) +
  Yahoo `/quote/` path miss this distinct surface. Fingerprint =
  `^\s*\$[^$\n]{0,60}\(SYM.EXCH\)\$` (leading "$" share-card lead glued to a
  `(SYMBOL.EXCH)$` close); bounded so no catastrophic backtracking; **offline-
  and live-validated ZERO false positives** against the real $+paren headline
  corpus (`$NVDA breaks out (NYSE)`, `$MU upgraded to Buy (price target
  $150.00)`, `Zscaler (NASDAQ:ZS) … $223.00`). Ships to BOTH consumed
  products (alert push + 5h Opus digest; the pass-16 "every consumed product
  gets the gate" precedent — advisor-directed not to scope alert-only),
  reusing the existing `_filter_quote_widget_noise` suppression machinery
  (suppressed rows marked `urgency=2`, kept in `articles.db` for training).
  Pure read-side: no DB write, no ai_score/ml_score/score_source/urgency
  mutation, backtest already filtered upstream — **all four load-bearing
  invariants intact by construction**. +23 specific-value tests across the 3
  lockstep gate test files (helper True/False incl. the FP corpus,
  end-to-end suppression, mixed-batch, `_build_payload` integration). Ships
  on next `systemctl restart digital-intern` (stale-daemon caveat).

  **Phase 3 — analyst-lens live validation, user_findings=8.** (1) **Phase-2
  driver CONFIRMED LIVE** — `$NVIDIA (NVDA.US)$ - Moomoo` (GN: Nvidia,
  ml=9.77, ai=0) in the live `urgency=2` set (fixed by `3135718`, ships on
  restart). (2) **Stale daemon predates ALL recent HEAD fixes** (the
  meta-finding: an operator `systemctl restart digital-intern` ships passes
  19–27's accumulated fixes incl. this one). (3) **26 phantom `urgency=1`
  rows** dated 2026-05-13 (5 days) — `reap_stale_urgent` at HEAD, stale
  daemon hasn't run a post-fix purge; inflates the dashboard urgent tile.
  (4) **Chronic DB-lock contention** — 44 `lock retry exhausted` + an
  `update_ai_scores_batch`-retry-exhausted Traceback (whole Sonnet batch
  dropped = potential missed urgent classification); memory
  `di-insert-batch-lock-contention`; advisor/precedent-confirmed out of
  clean scope. (5) **RSS dark in production** — sibling-WIP
  `collectors/rss_collector.py` 4-tuple bug (`string indices must be
  integers` ×19; the 5 `test_rss_collector.py` `_FakeResp` failures); not
  mine, never staged. (6) **6 collectors disabled** (`sec_edgar`/`_ft`,
  `polygon`, `newsapi`, `nitter`, `massive`) — analyst blind to 8-K filings
  (priority-0); COVERAGE GAP surfaces it; upstream/operational
  (`di-chronic-dark-collectors`). (7) **Alert path otherwise CLEAN & quiet
  (positive)** — exactly 2 legit BN alerts/24h (Benzinga geopolitical
  ai=9/8); recurring reddit/Wikipedia `urgency=2` residue is
  pre-deployed-gate (stale daemon); Wikipedia 0.60 above the 0.45 lone bar =
  the standing deferred contested *cred-map* tuning, NOT chased (distinct
  from this pass's *fingerprint* gate). (8) **Briefing EXCELLENT + cadence
  HEALTHY (positive)** — id27 (12:51Z, 50 arts) read end-to-end: dense,
  accurate, decisively-actionable (Iran-war/bond-rout LEAD 30Y 5.13%, exact
  MACRO/PORTFOLIO/SEMIS, syndication `[x2]` tags, COVERAGE GAP present);
  cadence gaps 5.3/5.4/5.7/6.8/6.3h vs 5h target (the `ef839a8`
  heartbeat-clock fix holding). None of 2–8 is a quick safe fix in clean
  scope (stale-daemon-with-HEAD-fix / advisor-confirmed out-of-scope /
  upstream / contested-cred-tuning) → no Phase-3 fold-in; bugs_fixed stays 0,
  features_added 1.

  **Verify:** `storage.article_store` / `ml.features` / `ml.model` /
  `watchers.alert_agent` / `analysis.claude_analyst` /
  `collectors.web_scraper` imports OK; suite **886 passed / 5 failed**
  (`--ignore=tests/test_alert_history.py`; the 5 are the pre-existing
  sibling `M collectors/rss_collector.py` `'_FakeResp' object has no
  attribute 'status_code'` 4-tuple WIP — not ours, never staged; floor held
  exactly 5, never 6+; my +23 new tests all pass; the 405-test alert/
  briefing/analyst/web_scraper slice green, zero regressions). *Pre-existing,
  deliberately never staged* (consistent with every prior entry):
  `collectors/rss_collector.py`, `daemon.py`, `dashboard/server.py`,
  `scripts/export_training_data.py`, `tests/test_article_store.py`, untracked
  sibling files, all `paper-trader/*`, `logs/*`. Commit `3135718`
  pathspec-scoped via `git commit -F … -- <6 explicit paths>`;
  `git diff --staged --name-only` + `git show --stat` verified EXACTLY 6
  files, no sibling leakage; never `git add -A`; pushed to origin/master
  (`318dfe4..3135718`). A concurrent sibling hybrid agent edited this repo
  throughout; this entry was appended, not rewritten.

- **2026-05-18 (hybrid pass 28 — Agent 3, debug + feature + analyst-validation)** —
  All 9 required files + AGENTS.md read in full. Stale daemon caveat
  applies: same operator-tuned `daemon.py` (ML_TRAIN_INTERVAL 180→1800,
  CONTINUOUS_TRAIN_INTERVAL 120→600, both bootstrap sleeps 30/45→300) sits
  uncommitted, indicating active operator tuning + recent restarts — purge
  worker (6h cadence) has fired 10+ times today per the log but produced no
  `Purged` lines, so it has likely been killed mid-startup-sleep on every
  cycle (memory `di-stale-manual-daemon`). Per the same memory note +
  `di-shared-repo-concurrency`, strict per-commit pathspec staging held;
  every concurrent-agent / operator change (`daemon.py`,
  `dashboard/web_server.py`, untracked `collectors/fda_collector.py`,
  `collectors/seekingalpha_collector.py`, `tests/test_chat_correlation_
  enrichment.py`, all `paper-trader/*`, `logs/`) deliberately never staged.

  **Phase 1 — bugs_fixed=1, commit `868dc91`** (1 test file,
  pathspec-scoped, `git show --stat` verified no sibling leak, on
  origin/master `536d932..868dc91`). The 5 long-failing
  `tests/test_rss_collector.py` cases pinned the *pre-7729638* `_fetch_feed`
  contract (returned a list); the production refactor (`7729638 — Fix
  rss_collector 4-tuple refactor`) changed the contract to
  `(name, articles, outcome, retry_after)` so the caller can drive per-feed
  backoff (404=permanent, 429=ratelimited+Retry-After, network=transient,
  ok=articles+ok). The author updated `collect_rss` but left the tests
  pinned to the old contract: they have failed EVERY suite run since
  7729638 (`'_FakeResp' object has no attribute 'status_code'` ×4 plus
  one collect_rss empty-result mismatch). This is exactly the pre-existing
  failure mode every prior pass enumerated as "not ours, never staged" —
  closing it here. Updates the `_FakeResp` shim to mirror the
  `requests.Response` surface `_fetch_feed` ACTUALLY consumes (`status_code`,
  `headers` for Retry-After, `content`, `raise_for_status`), unpacks the
  4-tuple at every call site, AND adds two new branch-coverage tests
  (`test_fetch_feed_404_is_permanent`, `test_fetch_feed_429_returns_
  ratelimited_with_retry_after`) that pin the previously-untested 404 +
  429 paths. Suite 911→918 pass after Phase 1.

  **Phase 2 — features_added=1, commit `84dff1a`** (1 src + 1 test,
  +346/−1, pathspec-scoped via explicit paths, `git show --stat` verified
  no sibling leak, on origin/master `8e170fa..84dff1a`). **THROUGHPUT
  DEGRADATION** — the early-warning complement to COVERAGE GAP. The latter
  only surfaces sources the FAILURE_THRESHOLD has already pushed to
  `disabled` (a binary, late signal); a live source can be quietly losing
  most of its throughput (e.g. an RSS feed delivering 40/h yesterday, 3/h
  now) without ever crossing that bar. `ArticleStore.source_throughput`
  already detects this — CLAUDE.md §6, `tests/test_source_throughput.py`,
  per-source `recent`/`prior`/`decel_pct` over rolling windows — but until
  now had **NO consumer**: a fully-implemented detector blind to the
  briefing that the consuming analyst's "stale sources" complaint applies
  to. Three coordinated pieces in `analysis/claude_analyst.py`:
  (a) `_collect_source_throughput` opens a fresh `mode=ro` connection
  (never the daemon's shared `self.conn` — the documented cursor-collision
  hazard, same discipline as `_collect_source_health` /
  `_recent_briefing_digest`), best-effort → `[]` on any failure so the 5h
  briefing is never broken or delayed; (b) `_throughput_degradation_lines`
  is a pure renderer with **conservative thresholds** (`prior >= 10` so a
  5→0 drop never produces noise even though it's 100% decel; `decel_pct >=
  60%` so mild fluctuation stays silent), sorted by absolute loss desc with
  prior-magnitude tiebreak (a 50→0 source matters more than a 20→0 source
  even when both are 100% decel), capped at 6 lines so this section can
  never itself become noise; (c) wired into `_build_payload` as a new
  optional input block + `SYSTEM_PROMPT` rule directly under COVERAGE GAP,
  with the same "omit when absent" discipline. Read-only by construction:
  no DB write, no ai_score/ml_score/score_source/urgency touch, never
  mutates source_articles, backtest already excluded upstream by
  `_LIVE_ONLY_CLAUSE` — **all four load-bearing invariants intact**.
  +14 specific-value tests pin: threshold gates (min_prior tiny-baseline
  exclusion, min_decel_pct mild-slowdown exclusion, `decel_pct=None`
  no-baseline exclusion, accelerating-source exclusion), the
  significant-degradation flagship case with exact formatted output,
  sort order (largest absolute loss first, prior tiebreak),
  `_MAX_DEGRADATION_LINES` cap, empty/malformed-row robustness,
  `_build_payload` wiring (emit/omit/empty/all-below-threshold/no-arg
  byte-determinism), SYSTEM_PROMPT coverage. Suite 918→951 pass after
  Phase 2 (the +33 includes my 14 plus other tests previously gated by
  conftest collection that now run; my new file's 14 all green; zero
  regressions). Ships on next `systemctl restart digital-intern` (stale
  daemon caveat).

  **Phase 3 — analyst-lens live validation, user_findings=5.**
  (1) **Collection HEALTHY (positive)** — 379 live articles/last 1h,
  7398/24h, diverse GN round-robin + GDELT + scraped + Finnhub + Yahoo +
  Bloomberg + Block + Nikkei + Korea Herald flowing. (2) **Alerts firing
  on-book (positive)** — 14+ legit BN alerts/24h, all portfolio-relevant
  or memory-complex: LITE -8.8% insider selling (GN: Nasdaq, ai=9.6); AXTI
  +650% YTD (GN/TradingView, ai=9.9) and -14% today (GN/Quiver, ai=9.0);
  NVDA earnings prep ×3 (ai=8.0–9.3); MU -X% ×3 (ai=8.0–9.0); CXMT
  revenue +700% (Finnhub/Yahoo, ai=9.9); NVDA China-market commentary
  (Finnhub/Yahoo, ai=9.6); Samsung labor dispute → memory threat (ai=8.0).
  Exact persona match — these are the alerts an analyst holding the SAO
  semis book WOULD react to. (3) **Briefings firing on cadence
  (positive)** — id26 (07:13Z), id27 (12:51Z), id28 (18:05Z) ≈5h apart,
  50 articles each, with LEAD lines materially actionable
  ("Memory/storage complex crushed — STX…", "Iran-war inflation…", "Global
  bond rout deepens — 10Y UST +…"). The `_recent_briefing_digest`
  anti-rehash gate (passes 24+) is live. (4) **26 phantom `urgency=1`
  rows from 2026-05-13 (5.6 days)** — `reap_stale_urgent` exists at HEAD
  but `purge_worker` has fired 10+ times in `daemon.log` without producing
  a single `Purged` line, meaning every fire was inside the 6h startup-
  sleep cooldown (operator restarts faster than that interval, so the
  reaper never gets a chance). Inflates the dashboard urgent tile. Not a
  new code bug — the fix is deployed; the cure is a single uninterrupted
  6h+ daemon run (or a one-shot `store.reap_stale_urgent()` from a manual
  Python invocation). Deliberately did NOT touch the live production DB
  this pass (write to prod is a risky-action class — same discipline as
  every prior pass, even though the call is well-tested and idempotent).
  (5) **Active "another row available" cursor-collision retries +
  `[google_news_worker] database is locked; backing off`** in the live log
  this minute — the chronic shared-`self.conn` lock contention (memory
  `di-insert-batch-lock-contention`); the retry decorator absorbed the
  reader collisions successfully (`stats: transient DB error …; retrying
  in 0.29s` ×N → no exception escape), so the dashboard `/api/stats`
  endpoint did NOT 500. The google_news write path is on Backoff/5s →
  10s, recoverable. Per-call connection isolation is substantial +
  `daemon.py`/store sibling-touched → out of clean scope
  (advisor/precedent-confirmed across passes 19–27). 6 disabled channels
  observed (`alphavantage`, `newsapi`, `nitter`, `polygon`, `sec_edgar`,
  `sec_edgar_ft`) — chronic external/rate-limit gap (memory
  `di-chronic-dark-collectors`), correctly surfaced by COVERAGE GAP in the
  briefing; not in scope. None of 4/5 is a quick safe fix in clean scope
  → no Phase-3 fold-in; bugs_fixed stays 1, features_added 1.

  **Verify:** `from storage import article_store; from ml import features,
  model; from analysis import claude_analyst` imports OK; suite **951
  passed** (`tests/`, my 14 new throughput tests + 7 RSS tests all green,
  zero regressions). Commits `868dc91` (Phase 1) and `84dff1a` (Phase 2)
  pathspec-scoped via explicit `git add <files>`; `git diff --staged
  --stat` + `git show --stat` verified EXACTLY the intended files
  (1 + 2 respectively), zero sibling leakage; never `git add -A`; both
  pushed to origin/master. A concurrent sibling hybrid agent + operator
  edited this repo throughout the session (uncommitted `daemon.py`,
  `dashboard/web_server.py`, untracked `collectors/fda_collector.py`,
  `collectors/seekingalpha_collector.py`,
  `tests/test_chat_correlation_enrichment.py`); this AGENTS.md entry was
  appended, not rewritten.

- **2026-05-19 (hybrid pass 29 — Agent 3, debug + feature + analyst-validation)** —
  All 9 required files + AGENTS.md read in full. Concurrent sibling hybrid
  agents (`pid 1979386` finishing as `pid 2291376` started) committed/pushed
  `6018347 feat(dashboard): /api/scorer-portfolio-attribution` mid-session;
  strict per-commit pathspec staging held throughout (memory
  `di-shared-repo-concurrency`). Stale daemon (pid 2124003, etimes ≈4h+) was
  still running unrestarted, so phantom-row evidence persisted into this
  pass.

  **Phase 1 — bugs_fixed=1, commit `a27109f`** (1 src + 1 test,
  +95/−4, pathspec-scoped, `git show --stat` verified EXACTLY 2 files, on
  origin/master `6018347..a27109f`). **purge_worker startup reap.** Live
  evidence (2026-05-18 → 19): 26 rows STILL stuck at `urgency=1` since
  2026-05-13 — 6 days, never alerted — even though the well-tested
  `ArticleStore.reap_stale_urgent` exists at HEAD. Root cause: reap is
  called ONLY inside `purge_old`, which fires on a 6h cadence after a
  manually-initialised `last_purge = time.time()` (so the FIRST purge is 6h
  after worker start). The operator-restart cycle is shorter than 6h
  (memory `di-stale-manual-daemon`), so on every daemon run the reaper
  never gets a turn — phantom rows accumulate indefinitely, inflating the
  dashboard `urgent` tile and re-fetched/re-decompressed by the alert
  worker every cycle. Fix: a one-shot `_purge_worker_startup_reap(store)`
  call at the top of `purge_worker` (BEFORE the 5-min health-ping loop).
  Idempotent + cheap (one indexed UPDATE), identically invariant-safe to
  the existing in-`purge_old` call: only `urgency` is mutated, never
  ai_score/ml_score/score_source/synthetic rows. Best-effort wrapper —
  any store exception is logged and swallowed so the 5-min liveness ping
  loop still starts. +4 specific-value tests pin: aged-row demotion (6d
  phantom → urgency=0), no-op when nothing stale (fresh row + already-
  alerted row both untouched), exception swallowing (custom `_Boom` mock),
  synthetic-row defense-in-depth (backtest:// row with urgency=1 stays
  urgency=1, the live row in the same call is reaped). Suite 960→964 pass
  after Phase 1.

  **Phase 2 — features_added=1, commit `cef83f2`** (1 src + 1 test,
  +399/−1, pathspec-scoped via explicit `git add <files>`,
  `git show --stat` verified EXACTLY 2 files, on origin/master
  `3e24437..cef83f2`). **ALERT VELOCITY — BREAKING-wire firing-rate
  magnitude hint.** The 🚨 BREAKING alert path is the analyst's most
  time-critical product, and its raw firing rate over a 5h window vs the
  prior 5h carries a magnitude signal NO individual story score can
  express: 24 alerts vs 8 prior tells Opus the wire is materially hot (a
  real macro event under way — Fed surprise, geopolitical escalation,
  broad selloff) and stories should be weighted with cumulative gravity;
  2 vs 12 means the wire is unusually quiet so a lone BREAKING-tagged
  story deserves closer scrutiny than the same score in a busy window.
  Until now the briefing composed LEAD/TOP SIGNALS with ZERO awareness
  of the standalone-push channel's firing rate.

  Same shape as COVERAGE GAP / THROUGHPUT DEGRADATION (operational-status
  family): three coordinated pieces in `analysis/claude_analyst.py` —
  (a) `_collect_alert_velocity(window_hours=5)` opens a fresh `mode=ro`
  connection (never the daemon's shared `self.conn` — the documented
  cursor-collision hazard, same discipline as the family), best-effort →
  None on any failure so the 5h briefing is never broken or delayed;
  (b) `_alert_velocity_lines` is a pure renderer with conservative
  thresholds (`recent+prior >= 5` AND `|delta_pct| >= 50%`, plus two
  special-case branches for the previously-dark and newly-silent edges
  that bypass the percentage gate because the ratio is undefined / -100%);
  (c) wired into `_build_payload` as a new optional input block +
  SYSTEM_PROMPT rule under THROUGHPUT DEGRADATION, with the same "omit
  when absent" byte-determinism discipline as `source_throughput` /
  `source_health_report` / `prior_digest`. Counts only `urgency=2` (the
  actually-fired state); `urgency=1` is the queued/phantom state (whose
  reap I fixed in Phase 1) and is correctly excluded. `_LIVE_ONLY_CLAUSE`
  applied — backtest isolation invariant.

  Pure read-side by construction: no DB write, no ai_score / ml_score /
  score_source / urgency mutation, never reads or mutates source_articles,
  backtest already excluded upstream — **all four load-bearing invariants
  intact**. +18 specific-value tests pin: empty/non-dict input,
  below-min-total / below-min-delta silence, hot-wire exact rendered
  message, cooling-wire exact rendered message, newly-lit / newly-silent
  edges, below-min-total special cases stay silent, doubling at threshold
  emits, window_hours reflected in text, malformed dict (non-numeric /
  negative / zero window) → [], `_build_payload` wiring (emit/omit/
  none-vs-explicit-none byte-equality), SYSTEM_PROMPT coverage rule.
  **Live verification before commit:** current 5h window reads "32 alerts
  vs 17 prior (+88%) — wire materially hot"; current 2h window reads
  "7 vs 15 (-53%) — cooling". Both pass the magnitude bar with real DB
  data, confirming the feature produces a real operational signal on next
  briefing run. Suite 964→982 pass after Phase 2 (zero regressions).

  **Phase 3 — analyst-lens live validation, user_findings=8.**
  (1) **Collection HEALTHY (positive)** — 414/h GN: Nasdaq, ~3-4k articles/h
  aggregate across GN round-robin + GDELT + scraped + Finnhub + Yahoo +
  Benzinga + DigiTimes; well within expected rates.
  (2) **Alerts on-book and actionable (positive)** — LITE -8.83%
  (ai=9.71, insider distribution), AXTI -14.46% (ai=9.0, +650% YTD
  profit-take), TSEM -9.46% (ai=9.63), MU -5.95% (continuation),
  NVDA Culper Research short (ai=9.33, "tip of iceberg" China problem),
  NVIDIA Huang/Dell parabolic-demand quote (ai=8.0). Exact persona match
  — these are the alerts the SAO semis analyst WOULD react to.
  (3) **Recap-headline noise (negative)** — `Why Nvidia (NVDA) Stock Is
  Trading Up Today` fired BREAKING twice (StockStory + YahooFinance/NVDA,
  ml=8.6/9.4) — these are post-hoc price-move recaps, not breaking news.
  Contested ML-tuning territory (per the cred-bar precedent, deferred);
  the fingerprint pattern is "Why <TICKER> ... Today" but its FP rate on
  legitimate "Why semis are crashing today" explainers is unmeasured,
  out of clean scope this pass.
  (4) **GDELT GKG SEO-mill noise** — `Here What the Street Thinks About
  ​NVIDIA Corporation` (note zero-width space U+200B between space-and-N
  in "​NVIDIA" — SEO content from insidermonkey.com via GDELT, ml=8.57).
  Distinct surface from existing junk-domain map; not in the
  _LOW_AUTHORITY_DOMAINS list. Worth a future evidence-driven addition.
  (5) **Briefing id29 (23:13Z) is EXCELLENT** — read end-to-end: LEAD
  ties LITE/AXTI/TSEM/MU together as broadened book pain ahead of NVDA
  print; PORTFOLIO table has exact prices/%/notes for every held name;
  TOP SIGNALS carry [seen HH:MM] timestamps with continuation framing.
  Highest-quality briefing observed across recent passes.
  (6) **26 phantom urgency=1 rows STILL in live DB** — daemon hasn't
  restarted to pick up Phase 1 fix; ships on next `systemctl restart
  digital-intern`. Confirmed live root cause matches my fix discipline.
  (7) **7 disabled collectors** (alphavantage, massive, newsapi, nitter,
  polygon, sec_edgar, sec_edgar_ft) — chronic external/rate-limit gap
  (memory `di-chronic-dark-collectors`), correctly surfaced by COVERAGE
  GAP in the briefing. Operational, not a code bug.
  (8) **Live alert wire is HOT (positive — feature validated)** — 32
  alerts/5h vs 17 prior = +88% confirmed against the live DB. The new
  ALERT VELOCITY feature would correctly flag this to Opus, weighting
  the LEAD with the cumulative-gravity context the prior briefing
  composed without. None of 3/4/6/7 is a quick safe fix in clean scope
  → no Phase-3 fold-in; bugs_fixed stays 1, features_added 1.

  **Verify:** `from storage import article_store; from ml import
  features, model; from analysis import claude_analyst` imports OK;
  suite **982 passed** (`tests/`, my +4 reaper-startup tests + 18
  alert-velocity tests all green, zero regressions). Commits `a27109f`
  (Phase 1) and `cef83f2` (Phase 2) pathspec-scoped via explicit
  `git add <files>`; `git diff --staged --stat` + `git show --stat`
  verified EXACTLY 2 + 2 = 4 intended files, zero sibling leakage; never
  `git add -A`; both pushed to origin/master. Concurrent sibling
  committed `6018347 feat(dashboard): /api/scorer-portfolio-attribution`
  mid-session (separate file domain, no collision); untracked
  `collectors/fda_collector.py`, `collectors/nasdaq_ipo_calendar.py`,
  `collectors/seekingalpha_collector.py` + all `paper-trader/*`
  deliberately never staged. This AGENTS.md entry was appended, not
  rewritten.
