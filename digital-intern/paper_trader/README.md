# Paper Trader

Opus 4.7-driven virtual trading engine that consumes signals from Digital Intern
and manages a simulated $1000 portfolio. Lives alongside the Digital Intern
codebase but runs as its own systemd service (`paper-trader.service`) on
port 8090.

## What it does

- Polls Digital Intern's article store for high-score signals
- Feeds those signals into an LLM-driven strategy layer (`strategy.py`)
- Places virtual stock and option trades against a local market data feed
  (`market.py`)
- Persists portfolio, trades, decisions, and equity history to SQLite
  (`store.py`)
- Serves a live dashboard at `http://<host>:8090/` (`dashboard.py`)
- Cross-links to the Digital Intern dashboard at `:8080`

## Layout

| File | Purpose |
| --- | --- |
| `runner.py` | Main loop — invoked by the systemd unit |
| `strategy.py` | Opus 4.7 decision layer |
| `signals.py` | Bridge between Digital Intern signals and trade ideas |
| `market.py` | Quote/price fetching |
| `store.py` | SQLite persistence |
| `dashboard.py` | Flask UI + JSON API (`/api/state`, `/api/portfolio`, `/api/backtests`) |
| `backtest.py` | Replay engine for historical strategy validation |
| `reporter.py` | Periodic summary builders |

Top-level helpers in the repo root:

- `paper_trader_runner.py` — shim that boots the runner with `sys.path` set
- `run_backtests.py` — kicks off independent year-long backtest runs
- `paper-trader.service` — systemd unit file

## Running

```bash
python3 paper_trader_runner.py           # live paper trading
python3 run_backtests.py                 # run backtest sweep
```

Or install the systemd unit:

```bash
sudo cp paper-trader.service /etc/systemd/system/
sudo systemctl enable --now paper-trader
```

The service reads its env from `digital-intern/.env` (Anthropic API key, etc.).

## Dashboard

- `/` — live trader (equity curve, positions, trades, decisions)
- `/backtests` — historical run summary + per-run trade log
- `/api/portfolio` — compact JSON snapshot consumed by the Digital Intern
  dashboard's "Live Paper Trader" panel
