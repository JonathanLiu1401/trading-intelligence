# Paper Trader

A Claude Opus 4.7-driven paper trading engine with continuous backtesting, ML decision scoring, and a real-time analytics dashboard.

## What it does

Three concurrent loops run simultaneously:

1. **Live paper trader** — every 30 min when NYSE is open, Opus 4.7 decides what to buy/sell/hold from a $1,000 portfolio and executes through paper trade plumbing. Results posted to Discord hourly + at daily close.

2. **Continuous backtests** — 5 parallel year-long historical simulations per cycle. Each run is a committee of 10 trading personas. Forward returns are recorded and used to train a small MLP ("decision scorer"). Once trained on ≥500 outcomes, the scorer's advice is injected into Opus's live prompt as an *advisory opinion* (Opus retains full autonomy).

3. **Hourly Opus self-review** — three Opus 4.7 agents review paper-trader core, paper-trader ML, and digital-intern; fix bugs surgically; log results to Discord.

## Architecture

```
digital-intern (separate process)
  └── articles.db ──(read-only)──► signals.py → strategy.py
                                                    │
                                         Claude Opus 4.7 (180s timeout)
                                                    │
                                         JSON decision: action/ticker/qty/thesis
                                                    │
                                    ┌───────────────┴────────────────┐
                                    │           store.py             │
                                    │       backtest.db (SQLite)     │
                                    └───────────────┬────────────────┘
                                                    │
                                    ┌───────────────▼────────────────┐
                                    │    analytics/ (25 modules)     │
                                    │    dashboard.py (:8090)        │
                                    │    reporter.py (Discord)       │
                                    └────────────────────────────────┘
```

## Directory Structure

```
paper-trader/
├── runner.py                    # entrypoint: python3 runner.py
├── paper_trader/
│   ├── runner.py               # main loop — market hours gate, hourly/daily reports
│   ├── strategy.py             # Opus 4.7 prompt builder + JSON decision parser
│   ├── signals.py              # reads articles.db, computes news signals
│   ├── market.py               # NYSE calendar, price fetching, position management
│   ├── store.py                # SQLite store wrapper (backtest.db)
│   ├── backtest.py             # BacktestEngine — year-long simulations
│   ├── reporter.py             # Discord summaries (hourly + daily close)
│   ├── dashboard.py            # Flask dashboard on :8090
│   ├── validation.py           # input validation utilities
│   ├── historical_collector.py # price history collector
│   ├── ml/
│   │   └── decision_scorer.py  # MLP trained on (quant+news) → trade outcome
│   └── analytics/              # 25 analytics modules (see below)
├── run_backtests.py            # one-shot backtest run
├── run_continuous_backtests.py # continuous backtest loop
├── backfill_news.py            # news backfill utility
├── scripts/
│   └── audit_overfitting.py   # checks for backtest data leakage
└── tests/                     # pytest suite (~65 test files)
```

## Analytics Modules

The dashboard (`paper_trader/dashboard.py`) pulls from 25 analytics modules:

| Module | What it tracks |
|---|---|
| `calibration` | ML model calibration curve |
| `capital_paralysis` | idle cash / under-deployment detection |
| `churn` | excessive position turnover |
| `decision_drought` | gaps between decisions |
| `decision_forensics` | per-decision P&L attribution |
| `decision_health` | overall decision quality score |
| `decision_reliability` | Opus parse success rate |
| `drawdown` | max drawdown + recovery time |
| `feed_health` | news source freshness |
| `funded_suggestions` | what the ML would buy with available cash |
| `greeks` | options greeks monitoring |
| `liquidity` | bid/ask spread awareness |
| `news_edge` | which news sources predict moves |
| `news_dedup` | duplicate article detection |
| `open_attribution` | open position P&L by entry signal |
| `position_thesis` | current hold reasoning |
| `round_trips` | completed trade P&L analysis |
| `scorer_confidence` | ML confidence distribution |
| `sector_heatmap` | exposure by GICS sector |
| `self_review` | Opus self-critique of recent decisions |
| `session_delta` | intraday delta tracking |
| `signal_followthrough` | did signals predict returns? |
| `source_edge` | per-source alpha attribution |
| `thesis_drift` | position thesis changes over time |
| `trade_asymmetry` | win/loss size asymmetry |
| `trader_scorecard` | composite performance grade |
| `correlation` | factor / position concentration honesty |
| `loser_autopsy` | per-loss post-mortem attribution |
| `decision_health` (CLI: `scripts/decision_health_cli.py`) | headless NO_DECISION triage |

(28 modules total in `paper_trader/analytics/`; the table lists the dashboard-surfaced set.)

## Decision Schema

Opus returns JSON per decision cycle:

```json
{
  "action": "BUY" | "SELL" | "HOLD",
  "ticker": "NVDA",
  "qty": 1.5,
  "thesis": "...",
  "confidence": 0.82
}
```

If Opus times out (180s) or returns unparseable JSON, Sonnet 4.6 is tried as fallback (60s timeout). After 5 consecutive `NO_DECISION` cycles, any lingering Claude subprocess is killed.

## Backtest Engine

`paper_trader/backtest.py` (`BacktestEngine`) runs year-long historical simulations:

- **Seed** — each run takes a deterministic seed so a run is reproducible (persona ordering, any stochastic tie-breaks).
- **Window** — a one-year forward window; entry decisions are made on data available *at that point in time only* (no look-ahead). `scripts/audit_overfitting.py` checks for data leakage.
- **Persona committee** — each run is a committee of 10 trading personas (different risk/horizon styles). Their decisions are aggregated; forward returns of the resulting trades are recorded as `decision_outcomes`.
- **Continuous loop** — `run_continuous_backtests.py` runs ~5–10 parallel year-long runs per cycle (the `continuous-backtests` user service), feeding the ML training set.
- ML training data is the accumulated `(quant features, news features) → realized forward return` from all completed backtest round-trips.

## ML Advisor Gate

The backtest ML model's advice is injected into the Opus live prompt only when:
- ≥20 qualifying backtest runs exist (`ML_QUALIFY_MIN_RUNS`)
- Median alpha vs. SPY > 0% over those runs (`ML_QUALIFY_MEDIAN_ALPHA`)

The gate rechecks every hour. When active, the ML's `(quant, news)` recommendation appears in the prompt with the explicit note that Opus retains full autonomy over the final call. Per project policy the *live* paper trader always uses Opus 4.7; only backtests use the ML-only path.

## Watchlist

Tracked universe (~80 tickers):
- Semis: NVDA, AMD, MU, AMAT, LRCX, KLAC, TSM, ASML, MRVL
- Mega-cap: AAPL, MSFT, GOOGL, AMZN, META, TSLA
- ETFs: SPY, QQQ, SMH, SOXX
- Leveraged ETFs: TQQQ, UPRO, SOXL, TECL, FNGU (and inverse: SQQQ, SPXS, SOXS)
- Financials: JPM, BAC, GS, MS
- Macro: GC=F (gold), BTC-USD

## Running

```bash
# Install dependencies
pip install -r requirements.txt   # or: pip install yfinance requests flask anthropic

# Start the live paper trader
python3 runner.py

# Or via systemd
systemctl start paper-trader

# Run a one-shot backtest
python3 run_backtests.py

# Run continuous backtest loop
python3 run_continuous_backtests.py

# Dashboard at http://localhost:8090
```

## Configuration

Paper trader reads configuration from `paper_trader/strategy.py`:

| Constant | Default | Description |
|---|---|---|
| `OPEN_INTERVAL_S` | 1800 | decision interval when market is open (s) |
| `CLOSED_INTERVAL_S` | 3600 | decision interval when market is closed (s) |
| `DECISION_TIMEOUT_S` | 180 | Opus timeout per decision cycle |
| `FALLBACK_TIMEOUT_S` | 60 | Sonnet fallback timeout |
| `INITIAL_CASH` | 1000.0 | starting portfolio cash (backtests) |
| `ML_QUALIFY_MIN_RUNS` | 20 | runs needed before ML gate activates |
| `ML_QUALIFY_MEDIAN_ALPHA` | 0.0 | min median alpha (%) for ML gate |

## API Endpoints (dashboard :8090)

Flask dashboard (`paper_trader/dashboard.py`). Selected endpoints:

| Endpoint | Returns |
|---|---|
| `GET /api/state` | Live portfolio state, cash, positions |
| `GET /api/portfolio` | Portfolio P&L summary |
| `GET /api/decisions` (`/decisions`) | Recent Opus decisions + raw responses |
| `GET /api/trades` (`/trades`) | Completed trades / round-trips |
| `GET /api/backtests` (`/backtests`) | Backtest run index |
| `GET /api/backtests/<id>` | Single run detail |
| `GET /api/backtests/curves` | Equity curves |
| `GET /api/backtests/compare` | Run-vs-run comparison |
| `GET /api/model-progress` | ML scorer training progress |
| `GET /api/scorer-confidence` | ML confidence distribution |
| `GET /api/scorer-predictions` | Latest ML `(quant,news)` recommendations |
| `GET /api/calibration` | Calibration curve |
| `GET /api/decision-health` | Decision quality score |
| `GET /api/decision-reliability` | Opus parse success rate |
| `GET /api/decision-drought` | Gaps between decisions |
| `GET /api/decision-forensics` | Per-decision P&L attribution |
| `GET /api/loser-autopsy` | Per-loss post-mortem |
| `GET /api/correlation` | Factor concentration |
| `GET /api/drawdown` | Max drawdown + recovery |
| `GET /api/risk` | Risk summary |
| `GET /api/sector-heatmap` / `/api/sector-pulse` | Sector exposure |
| `GET /api/news-edge` / `/api/source-edge` | Source alpha attribution |
| `GET /api/feed-health` / `/api/data-feed` | digital-intern feed freshness |
| `GET /api/self-review` | Opus self-critique |
| `GET /api/scorecard` | Composite trader scorecard |
| `GET /api/build-info` | Running code build / stale-code check |

(Full list ~45 endpoints; one per analytics module plus core state/backtest routes.)

## Database

`backtest.db` (SQLite, WAL mode) stores:

- `decisions` — every trading decision (action, ticker, qty, thesis, raw Opus response)
- `positions` — current open positions with entry price and thesis
- `prices` — yfinance price cache to avoid redundant fetches
- `decision_outcomes` — completed round-trips with forward returns (ML training data)

## Tests

```bash
pytest tests/ -x -q
```

Key test areas: backtest isolation from live DB, ML scorer seams, signal followthrough, trade asymmetry, calibration, capital paralysis detection.

## Environment Reference

The live `paper-trader` service reads `digital-intern/.env` (shared); `continuous-backtests` reads `paper-trader/.env`.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` / Claude CLI auth | Opus 4.7 live decisions + Sonnet fallback |
| `DISCORD_WEBHOOK_URL` | Hourly + daily-close report posting |
| `DIGITAL_INTERN` (code constant) | Path to digital-intern for `articles.db` read access |

Tuning constants live in `paper_trader/strategy.py` (see Configuration table above).

## Dependencies

- `yfinance` — price data, corporate actions
- `requests` — news fetching
- `flask` — dashboard web server
- `anthropic` / `claude` CLI — Opus 4.7 trading decisions
- `torch`, `scikit-learn` — ML decision scorer

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Repeated `NO_DECISION` cycles | Opus returned unparseable JSON or timed out. Sonnet 4.6 fallback (60s) is tried; after 5 consecutive, lingering Claude subprocesses are killed. Inspect with `scripts/decision_health_cli.py`. |
| Opus timeout every cycle | `DECISION_TIMEOUT_S` (180s) exceeded — usually Claude CLI quota/latency. Check `/api/decision-reliability`; the loop degrades gracefully (HOLD) rather than crashing. |
| Stale code alert on dashboard | `/api/build-info` detects the running process is on older code than disk. Restart the service (`sudo systemctl restart paper-trader`). |
| `possibly delisted; no price data found` log spam | Benign yfinance noise for futures/odd tickers (`ES=F`, `GOOGU`, …); not a path or startup failure. |
| Backtests not feeding ML | `continuous-backtests` user service down — `systemctl --user status continuous-backtests`. ML gate stays inactive until ≥20 qualifying runs. |
| ML advice not in Opus prompt | Gate not satisfied (need ≥20 runs AND median alpha > 0%). Check `/api/model-progress`. |
| Live trader using ML instead of Opus | Should never happen — live path is always Opus 4.7 by policy; ML is advisory-only and backtest-only. |

## Related

- **digital-intern** at `../digital-intern/` (monorepo sibling) — news pipeline that populates `articles.db`
- Monorepo root: [`../README.md`](../README.md)
- Dashboard: `http://localhost:8090`
- Discord reports sent via `openclaw` CLI
