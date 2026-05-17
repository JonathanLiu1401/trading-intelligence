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

## ML Advisor Gate

The backtest ML model's advice is injected into the Opus live prompt only when:
- ≥20 qualifying backtest runs exist
- Median alpha vs. SPY > 0% over those runs

The gate rechecks every hour. When active, the ML's `(quant, news)` recommendation appears in the prompt with the note that Opus retains full autonomy over the final call.

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

## Dependencies

- `yfinance` — price data, corporate actions
- `requests` — GDELT API, news fetching
- `flask` — dashboard web server
- `anthropic` / `claude` CLI — Opus 4.7 trading decisions
- `torch`, `scikit-learn` — ML decision scorer

## Related

- **digital-intern** at `/home/zeph/digital-intern/` — news pipeline that populates `articles.db`
- Dashboard: `http://localhost:8090`
- Discord reports sent via `openclaw` CLI
