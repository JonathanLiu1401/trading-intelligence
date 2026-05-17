# paper-trader — onboarding & system reference

This repo is a **live Claude-driven paper trading engine** plus a **continuous backtesting loop** that trains an
ML "decision scorer" against historical outcomes. It depends on `digital-intern` (see
`/home/zeph/digital-intern/CLAUDE.md`) as its news pipeline.

A new engineer or agent should be able to read this doc, then jump straight into debugging, extending, or
shipping a change. **No claim here is invented** — every file/function reference is what's actually on disk
under `/home/zeph/paper-trader/` as of writing.

---

## 1. What this system does

Three loops run concurrently:

1. **Live paper trader** (`paper_trader/runner.py`) — every 60s when the NYSE is open (3600s when closed),
   it asks **Claude Opus 4.7** to make one trading decision against a $1000 paper portfolio, then executes
   the JSON decision via `paper_trader/strategy.py`. Hourly summaries and a daily close summary are pushed
   to a Discord channel via `openclaw` and the Flask dashboard at `:8090`.

2. **Continuous backtests** (`run_continuous_backtests.py`) — runs 5 parallel year-long historical
   simulations per cycle, each driven by `paper_trader/backtest.py::BacktestEngine`. Each run is a
   committee of 10 trading personas. After every cycle the top runs' decisions and 5-day forward returns
   are appended to `data/decision_outcomes.jsonl`, then a small MLP (`paper_trader/ml/decision_scorer.py`)
   retrains on the accumulated outcomes. Once the scorer has ≥500 training records, it gates new buys in
   subsequent backtests.

3. **Hourly Opus self-review** (`scripts/hourly_review.sh`) — three Opus 4.7 agents run in parallel
   reviewing paper-trader core, paper-trader ML/backtests, and digital-intern. They fix bugs surgically
   and log results to Discord + `data/run_log.md`.

The live trader and backtest engine share the same prompt style, watchlist concepts, and JSON decision
schema, but **are otherwise independent**: live signals come from the digital-intern articles DB, backtest
"signals" come from the local-articles-DB snapshot + GDELT + Alpha Vantage caches + a heuristic scorer.

---

## 2. Architecture at a glance

```
                                    ┌───────────────────────────────────────────────┐
                                    │ digital-intern (separate process / repo)      │
                                    │   collectors → triage → ML score → SQLite     │
                                    │   /home/zeph/digital-intern/data/articles.db  │
                                    └────────────────────┬──────────────────────────┘
                                                         │ read-only (mode=ro)
                                                         │ filtered to live rows
                                                         │ (NOT LIKE 'backtest://%')
                                                         ▼
┌──────────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────┐
│ paper_trader │     │paper_    │     │ paper_       │     │paper_    │     │paper_    │
│  /runner.py  │────▶│ trader/  │────▶│ trader/      │────▶│ trader/  │────▶│ trader/  │
│  loop        │     │signals.py│     │ strategy.py  │     │market.py │     │reporter  │
│              │     │(news)    │     │(Opus 4.7)    │     │(yfinance)│     │(Discord) │
└──────┬───────┘     └──────────┘     └──────┬───────┘     └──────────┘     └──────────┘
       │                                     │
       │                                     ▼
       │                              ┌──────────────┐
       │                              │paper_trader/ │
       │                              │  store.py    │
       │                              │data/paper_   │
       │                              │  trader.db   │
       │                              └──────────────┘
       │
       ▼
┌──────────────┐    Flask dashboard at :8090
│paper_trader/ │    /            → live portfolio + trades
│ dashboard.py │    /backtests   → backtest runs + equity curves
│              │    /api/portfolio — consumed by digital-intern dashboard at :8080
└──────────────┘

╭─────────────────────────────── backtest side ────────────────────────────────╮
│                                                                              │
│  run_continuous_backtests.py   ──┐                                           │
│    cycle loop (60s cooldown)     │                                           │
│                                  ▼                                           │
│                          paper_trader/backtest.py                            │
│                            BacktestEngine                                    │
│                              ├─ PriceCache (yfinance, on-disk)               │
│                              ├─ _load_local_articles() ← digital-intern DB   │
│                              ├─ GDELTFetcher (disk-cached, rate-limited)     │
│                              ├─ AlphaVantageNewsFetcher (quota-capped)       │
│                              ├─ _ml_decide() ← quant + sentiment scorer      │
│                              │     uses paper_trader/ml/decision_scorer.py   │
│                              │     to nudge conviction (only ≥500 samples)   │
│                              └─ Opus 4.7 _claude_call() for live mode prompts│
│                                                                              │
│  After each cycle:                                                           │
│   1. Top runs' decisions → data/winner_training.jsonl (accumulated forever)  │
│   2. Forward 5d returns computed → data/decision_outcomes.jsonl              │
│   3. DecisionScorer.train_scorer(last 5000 outcomes) → data/ml/scorer.pkl    │
│   4. Opus annotates top run with GOOD/NEUTRAL/BAD labels + a lesson          │
│   5. JSONL re-injected into articles.db as `backtest_*` rows for ArticleNet  │
│   6. trim_history keeps the last KEEP_LAST_RUNS=500 runs                     │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 3. File map

### Top-level

| File | 1-line purpose |
|------|----------------|
| `runner.py` | One-line entry-point: `from paper_trader.runner import main; main()` |
| `run_backtests.py` | One-shot launcher: `BacktestEngine().run_all(10)` |
| `run_continuous_backtests.py` | Long-lived cycle loop: runs 5 backtests, trains scorer, retrains ArticleNet, posts Discord, sleeps 60s, repeats |
| `backfill_news.py` | Resumable historical news backfill — pumps GDELT articles into digital-intern's `articles.db` with a correct `published` date |
| `paper-trader.service` | systemd unit for the live trader (`python3 -m paper_trader.runner`) |
| `backtest.db` | SQLite store for `backtest_runs`, `backtest_trades`, `backtest_decisions` |
| `continuous.log` / `backtest.log` | Tailable logs from the continuous loop and one-shot backtests |

### `paper_trader/`

| File | 1-line purpose |
|------|----------------|
| `__init__.py` | Marks the package |
| `runner.py` | Main loop: `_cycle()` calls `strategy.decide()`, posts trade alerts, kicks hourly + daily Discord summaries, starts dashboard thread |
| `strategy.py` | The live decision engine — builds a context payload from signals + technicals + portfolio + market data, calls Opus 4.7, parses JSON, executes via `_execute()`. Watchlist + system prompt live here. On parse failure: conditional one-shot retry with a JSON-only suffix (`RETRY_TIMEOUT_S=45`, only when Claude returned non-empty unparseable text — see `_should_retry_parse`). When the final parse still fails, the raw response (capped at `RAW_CAPTURE_CHARS=1000`) is persisted to `decisions.reasoning` with a `parse_failed:` / `retry_failed:` tag so operators can diagnose silent failures from the dashboard |
| `signals.py` | Read-only queries against digital-intern's `articles.db` (USB if mounted else local). `get_top_signals()`, `get_urgent_articles()`, `ticker_sentiments()`, `get_ml_predictions()` (delegates to digital-intern's `ml.inference`). **All live queries filter out `backtest://` URLs and `backtest_*` / `opus_annotation*` sources.** |
| `market.py` | yfinance wrapper — `get_price`, `get_prices`, `get_option_price`, `get_options_chain`, `get_futures_price`, `is_market_open` (NYSE 2026 holiday calendar) |
| `store.py` | SQLite store for the live portfolio (`paper_trader.db`): `portfolio`, `trades`, `positions`, `decisions`, `equity_curve`. Initial cash = `$1000` |
| `reporter.py` | Discord output — `send_trade_alert`, `send_hourly_summary`, `send_decision_log`, `send_daily_close`. All routed via `openclaw message send` to `DISCORD_CHANNEL = "channel:1496099475838603324"` |
| `dashboard.py` | Flask app on `:8090`. `/` = live trader, `/backtests` = backtest grid. API endpoints: `/api/state`, `/api/portfolio`, `/api/backtests`, `/api/backtests/<id>`, `/api/backtests/<id>/trades`, `/api/backtests/<id>/decisions`, `/api/model-progress`, `/api/risk` (includes `concentration_warning` + `concentration_severity` flags), `/api/disagreement` (scorer-vs-Opus disagreement panel — flags positions where the DecisionScorer says EXIT/TRIM while Opus is still long), `/api/decision-health`, `/api/source-edge` (per-collector predictive-edge leaderboard — which of digital-intern's ~17 news sources' scored headlines actually precede the SPY-abnormal move; see AGENTS.md endpoint table for the full contract). Note: the full live-trader endpoint surface (analytics, behavioural-mirror, news/source-edge, …) is documented exhaustively in `AGENTS.md` — that table is canonical |
| `backtest.py` | The backtesting engine. Contains `BacktestEngine`, `BacktestStore`, `PriceCache`, `GDELTFetcher`, `AlphaVantageNewsFetcher`, persona dict (`PERSONAS`), watchlist (`WATCHLIST`), heuristic scorer (`score_article`), `_ml_decide` (quant+sentiment decision function), `_market_regime`, `_get_quant_signals` (RSI/MACD/BB/momentum), `_train_ml_from_winners` (writes winner JSONL). `START_DATE = 2025-05-01`, `END_DATE = 2026-05-13` |
| `ml/decision_scorer.py` | Tiny MLP (`MLPRegressor` 32→16 via sklearn, numpy-lstsq fallback) — features: `[ml_score, rsi, macd, mom5, mom20, regime_mult]` + 7-way sector one-hot → predicted 5-day forward return %. Pickle persisted at `data/ml/decision_scorer.pkl` |

### `scripts/`

| File | 1-line purpose |
|------|----------------|
| `hourly_review.sh` | Cron/systemd target — fires three parallel Opus 4.7 agents that audit and bug-fix the codebases, appends a timestamp line to `data/run_log.md` |

### `data/`

| Path | Purpose |
|------|---------|
| `paper_trader.db` | Live trader state — portfolio, positions, trades, equity_curve, decisions |
| `decision_outcomes.jsonl` | Accumulated `(ticker, sim_date, features → forward_return_5d)` rows — training data for `DecisionScorer` |
| `winner_training.jsonl` | Decisions of top backtest runs per cycle + Opus annotations + lessons. Re-injected into `articles.db` as `backtest_*` rows so ArticleNet (digital-intern) can train on them |
| `backtest_articles.db` | Optional pre-baked article DB used by some backtest tooling |
| `backtest_cache/prices.json` | yfinance OHLCV cache for the full backtest window across all watchlist tickers |
| `backtest_cache/volumes.json` | Volume series cache used for `vol_ratio` quant signal |
| `backtest_cache/gdelt/<date>_<hash>.json` | Per-(date, keyword) GDELT response cache |
| `backtest_cache/alphavantage/<date>_<ticker>.json` | Alpha Vantage NEWS_SENTIMENT cache |
| `backtest_cache/av_quota.json` | Cross-restart Alpha Vantage daily quota tracker (cap 22/day) |
| `ml/decision_scorer.pkl` | Trained MLP (pickled: `{model, scaler, n_train}`) |
| `run_log.md` | Hourly review cycle log — see §10 |

---

## 4. Live data flow (one cycle, every 60s/3600s)

```
runner._cycle()
    │
    ├─▶ strategy.decide()
    │     │
    │     ├─ market.is_market_open()
    │     ├─ _portfolio_snapshot(store)         ── marks every open position to market via market.get_prices / get_option_price; writes back to DB
    │     ├─ signals.get_top_signals(20, hours=2, min_score=4.0)    ── reads articles.db, filters out backtest:// rows
    │     ├─ signals.get_urgent_articles(minutes=30)                ── same filter
    │     ├─ signals.ticker_sentiments(WATCHLIST, hours=4)
    │     ├─ market.get_prices(WATCHLIST)
    │     ├─ market.get_futures_price(ES=F, NQ=F, CL=F, GC=F)
    │     ├─ market.benchmark_sp500()                               ── ^GSPC
    │     ├─ get_quant_signals_live(QUANT_TICKERS_LIVE + held)      ── RSI/MACD/MA/BB/mom_5d/mom_20d/wk52_pos
    │     ├─ _build_payload(...)                                    ── string-encodes everything for the prompt
    │     ├─ _claude_call(SYSTEM_PROMPT + payload)                  ── subprocess: claude --model claude-opus-4-7 --print --permission-mode bypassPermissions
    │     ├─ _parse_decision(raw)                                   ── strips ```json fences, extracts first JSON object
    │     ├─ _enforce_risk_pre_trade(decision, snapshot)            ── only blocks SELLs that exceed held qty; no other limits
    │     ├─ _execute(decision, snapshot, store)                    ── BUY / SELL / BUY_CALL / BUY_PUT / SELL_CALL / SELL_PUT / HOLD / REBALANCE
    │     │     ├─ market.get_price / get_option_price
    │     │     ├─ store.record_trade(...)
    │     │     ├─ store.upsert_position(...)
    │     │     └─ store.update_portfolio(...)
    │     ├─ store.record_decision(...)
    │     └─ store.record_equity_point(total, cash, sp500)
    │
    ├─▶ reporter.send_trade_alert(...)         ── if status == FILLED
    ├─▶ reporter.send_decision_log(summary)    ── if status == FILLED
    └─▶ runner._maybe_hourly() / _maybe_daily_close()
```

Hourly: `reporter.send_hourly_summary()` posts equity, cash, P/L, S&P benchmark, recent trades.
Daily after 16:05 NY (weekdays only): `reporter.send_daily_close()` adds realized P/L for the day.

---

## 5. Backtest loop (one cycle of `run_continuous_backtests.py`)

```
main()
  while not _STOP:
    cycle += 1
    start_id = MAX(run_id) + 1     # backtest_runs table

    results = engine.run_all(RUNS_PER_CYCLE=5, start_run_id=start_id)
    # ─── inside run_all: launches 5 threads, each runs BacktestEngine.run_one
    #     ┌─ run_one(run_id) ────────────────────────────────────────────────┐
    #     │ for sim_date in trading_days:                                    │
    #     │   _enforce_risk_exits(...)        # daily SL/TP scan             │
    #     │   signals = _fetch_signals(...)   # local DB > yfinance > GDELT │
    #     │   for _ in range(MAX_DECISIONS_PER_DAY=10):                      │
    #     │     decision = _ml_decide(...)    # PURE quant — no Opus call    │
    #     │     status = _execute_decision(...)                              │
    #     │     if HOLD or no fill: break                                    │
    #     │   record equity point                                            │
    #     └──────────────────────────────────────────────────────────────────┘
    #     Note: only the continuous loop uses _ml_decide. The Opus prompt
    #     path (_build_prompt + _claude_call) still exists in backtest.py
    #     but the active run loop now calls _ml_decide() exclusively.

    top_runs = top N by total_return_pct (filtered to positive returns)
    _append_top_decisions(...)         → data/winner_training.jsonl
    outcomes  = _compute_decision_outcomes(top_runs)
    append outcomes → data/decision_outcomes.jsonl (capped to last 5000 used per retrain)
    _train_decision_scorer(outcomes)   → data/ml/decision_scorer.pkl
    reset _DECISION_SCORER singleton   → next cycle reads new model

    # background thread (non-blocking):
    _opus_annotate(top_runs, cycle, outcomes)
      → asks claude-opus-4-7 to label every decision GOOD/NEUTRAL/BAD with rationale
      → writes annotations + an overall lesson to winner_training.jsonl

    _try_train_ml() == _inject_and_train()
      → injects WINNER_JSONL into digital-intern's articles.db as backtest_* / opus_annotation rows
      → invokes /home/zeph/digital-intern with `ml.trainer.train(force=True)`
        retrains ArticleNet on the augmented dataset

    _trim_history(keep=500)            # backtest_runs / _trades / _decisions
    sleep 60s
```

---

## 6. ML components — two distinct models

There are **two independent models** in this system. Don't confuse them.

| | **ArticleNet** | **DecisionScorer** |
|---|---|---|
| Lives in | `/home/zeph/digital-intern/ml/model.py` | `/home/zeph/paper-trader/paper_trader/ml/decision_scorer.py` |
| Framework | PyTorch (GPU) | sklearn `MLPRegressor` (CPU) with numpy lstsq fallback |
| Input | TF-IDF + 26 extras of (title + summary) of an article | `[ml_score, rsi, macd, mom5, mom20, regime_mult]` + 7-way sector one-hot |
| Output | `(relevance, urgency, uncertainty, time_sensitivity)` — multi-head | scalar — predicted 5-day forward return % |
| Trained on | Sonnet-labeled articles + Opus heartbeat labels + backtest winner injections | `decision_outcomes.jsonl` — actual 5d outcomes of past backtest BUY/SELLs |
| Used by | digital-intern's `score_pending()` → fills `ai_score`, `urgency`, `time_sensitivity` in `articles.db`. paper-trader reads these via `signals.py` | Used inside `paper_trader/backtest.py::_ml_decide` to **gate** buys when `is_trained AND _n_train >= 500` |
| Retrain | Every 3 min (full) + every 2 min (continuous) in digital-intern daemon, plus `_inject_and_train()` once per backtest cycle | After every backtest cycle (in `run_continuous_backtests.py`) |

`DecisionScorer.predict()` returns `0.0` (a no-op nudge) until trained. The gate in
`_ml_decide` **modulates BUY conviction only — it never cancels a trade** (an earlier
HOLD-blocking version oscillated leveraged-ETF strategies; see the comment in
`_ml_decide`). The current arms (`paper_trader/backtest.py::_ml_decide`, authoritative):

```python
if _scorer.is_trained and _scorer_n >= 500:
    if scorer_pred < -10.0:                 # strong headwind, still buys
        conviction *= 0.6
    elif scorer_pred < 0.0:                 # mild headwind
        conviction *= 0.85
    elif scorer_pred > 10.0:                # strong tailwind
        conviction = min(conviction * 1.3, 0.95)
    elif scorer_pred > 5.0:                 # mild tailwind
        conviction = min(conviction * 1.15, 0.95)
    # 0 ≤ scorer_pred ≤ 5 → unchanged
```

So the scorer only starts modulating real trades after ~500 BUY/SELL outcomes accumulate in
`data/decision_outcomes.jsonl`. (This block previously documented a `p < -5 → HOLD` /
`p < 0 → ×0.7` blocking gate; that was removed because blocking sabotaged SOXL/TQQQ — see
invariant #5 and the AGENTS.md "How the DecisionScorer works" table, the authoritative
companion reference.)

---

## 7. Key configs (and where to change them)

| Constant | Default | Location | Effect |
|----------|---------|----------|--------|
| `START_DATE`, `END_DATE` | `2025-05-01`, `2026-05-13` | `paper_trader/backtest.py` top | Backtest window. **Must match yfinance data availability**; bumping `END_DATE` past today means PriceCache will fall back to last known close |
| `INITIAL_CASH` | `1000.0` | `paper_trader/backtest.py` and `paper_trader/store.py` | Per-run starting capital |
| `SAMPLE_EVERY_N_DAYS` | `1` | `paper_trader/backtest.py` | Trading-day stride. `1` = daily decisions |
| `MAX_DECISIONS_PER_DAY` | `10` | `paper_trader/backtest.py` | Intra-day `_ml_decide` call cap |
| `_CLAUDE_SEM` | semaphore(3) | `paper_trader/backtest.py` | **Hard cap on concurrent claude subprocesses (OOM defense at 14 GB RAM)** |
| `RUNS_PER_CYCLE` | `5` | `run_continuous_backtests.py` | Parallel runs per cycle |
| `TOP_RUNS_TO_TRAIN` | `3` | `run_continuous_backtests.py` | How many top runs contribute to winner_training.jsonl |
| `KEEP_LAST_RUNS` | `500` | `run_continuous_backtests.py` | Backtest history retention |
| `MAX_OUTCOMES_FOR_TRAINING` | `5000` | `run_continuous_backtests.py` | Tail of `decision_outcomes.jsonl` used per scorer retrain (older outcomes describe a stale signal regime) |
| `COOLDOWN_SECONDS` | `60` | `run_continuous_backtests.py` | Sleep between cycles |
| DecisionScorer gate threshold | `_n_train >= 500` | `paper_trader/backtest.py::_ml_decide` | Don't gate trades until the scorer has seen enough outcomes |
| `OPEN_INTERVAL_S` / `CLOSED_INTERVAL_S` | `60` / `3600` | `paper_trader/runner.py` | Live trader decision cadence |
| `MODEL` | `"claude-opus-4-7"` | `paper_trader/strategy.py` | **Live trader always uses Opus 4.7** |
| `WATCHLIST` (live) | 50+ tickers incl. leveraged ETFs | `paper_trader/strategy.py` | What Opus sees in the prompt |
| `WATCHLIST` (backtest) | broader, ~120 tickers incl. ADRs, sector ETFs, leveraged | `paper_trader/backtest.py` | Universe for the historical sim |
| `GDELT_RATE_LIMIT_S` | `5.5` | `paper_trader/backtest.py` | GDELT's actual limit is ~1 req/5s |
| `AV_MAX_DAILY` | `22` | `paper_trader/backtest.py` | Alpha Vantage free tier is 25/day; leave headroom |
| `DISCORD_CHANNEL` | `channel:1496099475838603324` | `paper_trader/reporter.py`, `run_continuous_backtests.py`, `scripts/hourly_review.sh` | Where every notification goes |
| Dashboard port | `8090` | `paper_trader/dashboard.py::run` | digital-intern dashboard at `:8080` cross-fetches `/api/portfolio` |

---

## 8. Invariants and gotchas

These rules are load-bearing — breaking any one of them causes silent corruption of the live trade
loop or training data.

1. **Backtest articles must never reach live signals.** Every read in `signals.py` includes:
   ```sql
   AND url NOT LIKE 'backtest://%'
   AND source NOT LIKE 'backtest_%'
   AND source NOT LIKE 'opus_annotation%'
   ```
   Mirror this in any new query that goes to `articles.db`. The corresponding clause in digital-intern
   is named `_LIVE_ONLY_CLAUSE` in `storage/article_store.py` — see that file's CLAUDE.md entry.

2. **Backtest articles always have `urgency = 0`.** Even Opus annotations that mark a trade GOOD don't
   bump urgency — alerts come from live news, not training synthesis. `_inject_and_train` writes
   `urgency=0` unconditionally.

3. **The live trader always calls Claude Opus 4.7** (`MODEL = "claude-opus-4-7"` in `strategy.py`).
   Do not downgrade to Sonnet for cost reasons without an explicit decision — the entire system prompt
   is tuned around Opus's reasoning depth.

4. **The continuous backtest loop runs `_ml_decide()` (quant-only)**, not `_claude_call()`, for trading
   decisions. The Opus prompt path in `backtest.py` (`SYSTEM_PROMPT`, `_build_prompt`, `_claude_call`,
   `COMMITTEE_BRIEF`) is intact but currently unused by `run_continuous_backtests.py`. **Opus is still
   invoked in `_opus_annotate` to label completed runs for ML training.**

5. **DecisionScorer only gates after `_n_train >= 500`.** Below that threshold its predictions are too
   noisy to trust. If you change the gate threshold, change it in both `_ml_decide` and (ideally)
   surface it as a module-level constant.

6. **Concurrent `claude` subprocesses are capped at 3** via `_CLAUDE_SEM = threading.Semaphore(3)`.
   Each claude process eats ~1.5 GB; 14 GB RAM total means >3 = OOM kill. Don't raise this without
   either more RAM or a serialized claude proxy.

7. **`paper_trader.db` uses WAL.** Any external reader must use `PRAGMA journal_mode=WAL` or
   `?mode=ro` to avoid lock contention with the live trader's writer.

8. **GDELT cache is keyed by `(date, md5(keywords)[:8])`** in `data/backtest_cache/gdelt/`. If you
   change `KEYWORD_GROUPS` the old caches stay on disk but are invisible to the new code — that's
   intentional, not a leak.

9. **The Alpha Vantage daily quota persists across restarts** in `data/backtest_cache/av_quota.json`.
   Deleting it does not reset AV's server-side counter; it only resets the client's view.

10. **`_train_ml_from_winners()` (in `backtest.py`) is dead code in the continuous loop.** It still
    writes `data/winner_training.jsonl` but the continuous loop now does its own per-cycle append in
    `_append_top_decisions()`. Don't rely on the in-engine version.

11. **The `decisions.action_taken` column in `paper_trader.db` is a free-text string** of the form
    `"BUY NVDA → FILLED"`. Tools that parse it must tolerate `"NO_DECISION"` and `"BLOCKED"`. See
    `dashboard._parse_action_ticker()` for the canonical (verb, ticker) extractor — it nullifies
    `CASH` / `NONE` pseudo-tickers so they don't pollute per-position panels.

12. **Live trader's risk model is "no hard limits"** by design — the system prompt is explicit that
    Opus has full autonomy. `_enforce_risk_pre_trade` only checks that SELLs don't exceed held
    quantity. Don't add silent position caps; if you need them, change the system prompt.

13. **`SCHEMA` in `paper_trader/store.py` uses `CREATE TABLE IF NOT EXISTS`** — it doesn't ALTER on
    column additions. To add a column, do `ALTER TABLE` from a one-off script first, then add the
    column to the SCHEMA constant for fresh-DB compatibility.

---

## 9. How to run

### Live trader
```bash
cd /home/zeph/paper-trader
python3 -m paper_trader.runner             # foreground
# or
systemctl --user start paper-trader        # see paper-trader.service
```

The dashboard auto-starts in a thread on `:8090`. First-boot posts a `**PAPER TRADER ONLINE**` ping
to Discord.

### One-shot backtests (10 parallel year-long runs)
```bash
cd /home/zeph/paper-trader
python3 run_backtests.py                    # equivalent to BacktestEngine().run_all(10)
```

### Continuous backtest loop
```bash
cd /home/zeph/paper-trader
python3 run_continuous_backtests.py
# logs continuously; SIGTERM/SIGINT exits cleanly between cycles
```

### Backfill historical news
```bash
cd /home/zeph/paper-trader
python3 backfill_news.py --status                       # show current coverage
python3 backfill_news.py                                # fill START_DATE..END_DATE (~8h)
python3 backfill_news.py --from 2025-09-01 --to 2025-12-31
```

### Hourly self-review (3 parallel Opus agents)
```bash
bash /home/zeph/paper-trader/scripts/hourly_review.sh
# Drives 3 claude --model claude-opus-4-7 sessions in parallel.
# Logs to /tmp/review_logs/agent{1,2,3}_<ts>.log
# Appends a header to data/run_log.md
```

### Dashboard only (without the trader)
```python
from paper_trader.dashboard import run; run(host="0.0.0.0", port=8090)
```

---

## 10. Run log

A markdown file at `data/run_log.md` carries one entry per hourly review cycle. The format is
established in §10 of this doc and is also written by `scripts/hourly_review.sh`. New agents that run
out-of-band work (manual bug fixes, schema migrations, model experiments) should also append an
entry with the same format. The file is human-readable and intended as the historical narrative of
the system.

See the file for the schema; the canonical template lives at the top of `data/run_log.md`.

---

## 11. Debugging cheat sheet

| Symptom | Likely cause | Where to look |
|---------|--------------|---------------|
| Live trader posts `NO_DECISION` repeatedly | Claude returned malformed JSON or timed out (`DECISION_TIMEOUT_S = 180`, raised from 120 in 6227cd5) | tail the runner stdout; check `_parse_decision` in `strategy.py` |
| Live trader stuck on a `BLOCKED` SELL | `_enforce_risk_pre_trade` rejected — qty exceeds held position, or option strike/expiry don't match an open position | `strategy.py::_enforce_risk_pre_trade` |
| Hourly summary missing | `runner._maybe_hourly()` ran during a `_cycle` exception; `_last_hourly` doesn't advance until the report succeeds | check `[runner] hourly send failed:` in stdout |
| `signals.py` returns `[]` | Either `articles.db` isn't where expected (USB unmounted) or filter is too strict | confirm `_db_path()` resolves, then run `signals.py` as a script — it has a `__main__` that dumps top signals |
| Backtest runs all return `0.00%` | PriceCache failed to download — check `data/backtest_cache/prices.json` exists and isn't empty. yfinance often rate-limits a cold cache build | delete `prices.json` + retry with debug prints in `PriceCache._load` |
| Backtest dashboard shows `running` forever | A run thread died but `BacktestStore.upsert_run("failed")` never wrote, or the engine crashed before `finalize_run` | check `continuous.log` for `[engine] RUN N CRASHED:` traces |
| Backtest dashboard shows wildly different equity curves than the JSON | Browser cached an old `/api/backtests` response — Flask sends no-cache headers but proxies may | hard reload, or hit `/api/backtests` directly |
| `_inject_and_train()` reports `trainer rc≠0` | digital-intern's `ml.trainer.train()` raised — usually GPU OOM if scoring is still draining the GPU | reduce `RUNS_PER_CYCLE`, restart the daemon, or check `/home/zeph/digital-intern/logs/daemon.log` |
| DecisionScorer never gates | `data/decision_outcomes.jsonl` has fewer than 500 rows (`wc -l`) | accumulate more cycles; or temporarily lower the `>= 500` threshold in `_ml_decide` for testing |
| Dashboard shows the live trader and digital-intern numbers disagreeing on portfolio | Cross-origin fetch from `:8080` is reading `/api/portfolio` from `:8090` — make sure both processes are running, both DBs are reachable, and CORS headers are present (`dashboard.py::_cors`) | curl `http://localhost:8090/api/portfolio` directly |
| Discord posts stop | `openclaw` binary missing or auth expired; `reporter._send` prints `[reporter] openclaw not installed; would send:` | `which openclaw`, re-login if needed |
| Opus annotation never runs | `_opus_annotate` is launched in a daemon thread — if the main loop sleeps past 60s into the next cycle the thread may still be running. Check with `ps -fT` | search `continuous.log` for `[opus_annotate]` |
| `paper_trader.db is locked` | Two writers (live trader + a dashboard helper script) without `?mode=ro` | open read-only for any non-trader process |

---

## 12. Where new code typically goes

| You want to… | Add code in |
|--------------|-------------|
| Change the live system prompt or watchlist | `paper_trader/strategy.py` (`SYSTEM_PROMPT`, `WATCHLIST`) |
| Add a new technical indicator | `paper_trader/strategy.py::get_quant_signals_live` (live), `paper_trader/backtest.py::_compute_technical_indicators` (backtest), and `paper_trader/ml/decision_scorer.py::build_features` if you want the scorer to learn it |
| Add a new news source | digital-intern side — `collectors/<source>.py` + register a worker in `daemon.py` |
| Change how backtest decisions are made | `paper_trader/backtest.py::_ml_decide` |
| Add a new persona | `paper_trader/backtest.py::PERSONAS` (and the `_PERSONA_BOOSTS` table inside `_ml_decide`) |
| Add a new Discord report type | `paper_trader/reporter.py` (use `_send` helper); call from `runner._cycle` or `runner._maybe_hourly` |
| Change the dashboard | `paper_trader/dashboard.py` — `TEMPLATE` (HTML), then add a `/api/*` route |
| Train the DecisionScorer differently | `paper_trader/ml/decision_scorer.py::train_scorer` — the sklearn `MLPRegressor` config is right there |
| Change the cadence of the continuous loop | `RUNS_PER_CYCLE`, `COOLDOWN_SECONDS` in `run_continuous_backtests.py` |

---

## 13. External services

| Service | What it's used for | Auth |
|---------|-------------------|------|
| yfinance | All equity / option / futures / index prices, both live and backtest | None |
| GDELT (via `gdeltdoc`) | Historical news for backtests | None — rate-limited |
| Alpha Vantage `NEWS_SENTIMENT` | Optional ticker news enrichment in backtests | `ALPHA_VANTAGE_KEY` env var, 25/day free |
| digital-intern `articles.db` | Live news for the live trader | File-system read |
| `openclaw` | Discord posting | Pre-configured user session |
| Anthropic Claude (Opus 4.7) | Live decisions, backtest committee prompt, Opus annotation, hourly self-review | `claude` CLI in PATH; `claude --print --permission-mode bypassPermissions` |

---

## 14. Glossary

- **Backtest run** — one full year-long simulation. Identified by `run_id` in `backtest.db`.
- **Cycle** — one iteration of `run_continuous_backtests.py::main` — currently 5 runs + training.
- **Persona** — one of 10 trading styles (`PERSONAS` dict). Each `run_id` maps to one persona via
  `((run_id - 1) % 10) + 1`.
- **Committee** — the conceptual framing in the (currently unused) Opus prompt path: all 10 personas
  vote and a consensus trade is executed.
- **Quant signal** — an RSI/MACD/BB/momentum/vol_ratio/wk52_pos dict keyed by ticker.
- **Regime** — bull/sideways/bear from SPY 50/200 MA cross + slope (`_market_regime`).
- **Live-only clause** — the SQL filter that strips out `backtest://` URLs and synthetic sources from
  any query that feeds the live trader or alerts.
- **Forward return** — `(close[sim_date + 7d] - close[sim_date]) / close[sim_date] * 100`. Used as
  the DecisionScorer target.

---

## 15. ML advisor gate (live trader)

When the backtest ML model has demonstrated a consistent edge over the benchmark, its
recommendation is surfaced to Opus inside the live decision prompt as an **advisory opinion**.
This is gated, read-only, and never blocks or overrides a trade.

- **Qualification:** `_ml_is_qualified()` in `paper_trader/strategy.py` queries `backtest.db`
  read-only for the last `ML_QUALIFY_MIN_RUNS=20` *qualifying* runs
  (`status='complete' AND vs_spy_pct IS NOT NULL AND n_trades >= 5`, ordered by `run_id` DESC).
  The ML becomes an advisor only when the **median `vs_spy_pct`** of those 20 runs exceeds
  `ML_QUALIFY_MEDIAN_ALPHA=0.0` (%). Fewer than 20 qualifying runs ⇒ not qualified.
- **Gate cadence:** the qualification result is cached for `ML_QUALIFY_TTL_S=3600.0`s
  (module-level `_ml_qualify_cache`), so the gate is re-evaluated from `backtest.db` at most
  once per hour regardless of the 60s decision cadence.
- **Opinion generation:** `_ml_live_opinion()` runs the same quant+news scoring shape as the
  backtest engine — sentiment from a self-contained bullish/bearish word list, keyword→ticker
  mapping, RSI/MACD/momentum/BB quant adjustments, SPY-20d-momentum regime multiplier, then
  picks the single best-scoring watchlist ticker. It uses **only the live data already fetched
  by `decide()`** (`merged` articles, `quant_sigs`, `snap`, `watch_px`) and emits a
  BUY/HOLD action + reasoning string.
- **Self-contained, no backtest.py import:** `_ml_live_opinion` and its vocabulary tables
  (`_BULLISH_WORDS_LIVE`, `_BEARISH_WORDS_LIVE`, `_WORD_TO_TICKER_LIVE`, `_LEVERAGED_ETFS_LIVE`)
  are duplicated in `strategy.py` deliberately — importing from `backtest.py` would create a
  circular dependency. They mirror the backtest scorer closely enough for an advisory.
- **Opus retains full autonomy:** when qualified, the opinion is appended to the prompt under
  an `ML ADVISOR:` section that explicitly states it is advisory only. Opus sees the ML's
  recommended action and reasoning but makes the final call. The ML cannot execute trades and
  cannot veto Opus — it has no path to `_execute()`.
- **Non-fatal by construction:** every part of this path (gate query, opinion generation,
  prompt wiring) is wrapped so any failure degrades to "no ML advisor block this cycle",
  never "no decision this cycle". A qualification-check error returns `(False, ...)` and the
  prompt is built exactly as before.

Constants and code live in `paper_trader/strategy.py`: `ML_QUALIFY_MIN_RUNS`,
`ML_QUALIFY_MEDIAN_ALPHA`, `ML_QUALIFY_TTL_S`, `_ml_qualify_cache`, `_ml_is_qualified()`,
`_ml_live_opinion()`, wired into `decide()` immediately before `_build_payload()`.

---

*Last revised: 2026-05-16*
