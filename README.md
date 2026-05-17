# Trading Intelligence

AI-powered financial intelligence daemon plus an Opus 4.7-driven paper trading engine, unified into a single monorepo.

## What this is

**Trading Intelligence** is a two-stage automated trading research stack. The first stage, **digital-intern**, is a continuous financial-news intelligence daemon: it runs ~20 parallel worker threads pulling from 21 collector sources, scores every article with a local GPU model (ArticleNet) plus LLM escalation, persists everything to a multi-million-row SQLite database, and fires Discord alerts and Opus 4.7 briefings on market-moving events.

The second stage, **paper-trader**, consumes that intelligence. It reads digital-intern's `articles.db` read-only, builds a news+quant signal frame, and asks Claude Opus 4.7 to make buy/sell/hold decisions for a paper portfolio every 30 minutes while the NYSE is open. In parallel it runs continuous historical backtests (committees of trading personas) that train a small ML "decision scorer"; once that model proves itself, its recommendation is injected into Opus's live prompt as a non-binding advisory.

## Architecture

```
┌──────────────────────────── digital-intern ────────────────────────────┐
│  21 collectors ─► heuristic triage ─► articles.db (SQLite, WAL, USB)    │
│        │                                      │                         │
│        │                              ArticleNet (GPU inference)        │
│        │                              + Sonnet 4.6 grey-zone labeling   │
│        │                              + Opus 4.7 hardest-case labeling  │
│        ▼                                      ▼                         │
│  Discord alerts                       Opus 4.7 5h briefing              │
│  Flask dashboard :8080                                                  │
└────────────────────────────────┬───────────────────────────────────────┘
                                  │  articles.db  (read-only cross-process)
                                  ▼
┌──────────────────────────── paper-trader ──────────────────────────────┐
│  signals.py (reads articles.db) ─► strategy.py ─► Claude Opus 4.7       │
│                                                       │                 │
│                              JSON decision {action,ticker,qty,thesis}   │
│                                                       │                 │
│  continuous backtests ─► decision_scorer (MLP) ──advisory──┘            │
│       (persona committees)                                              │
│                                                                         │
│  store.py ─► backtest.db    analytics/ (28 modules)                     │
│  reporter.py ─► Discord     Flask dashboard :8090                       │
└─────────────────────────────────────────────────────────────────────────┘
```

The cross-process contract is a single file: digital-intern owns and writes `digital-intern/data/articles.db` (a symlink to a USB-mounted volume at `/media/zeph/projects/digital-intern/db/articles.db`); paper-trader opens it read-only.

## Directory structure

```
trading-intelligence/
├── digital-intern/    # News intelligence daemon — collectors, ML, alerts, :8080 dashboard
├── paper-trader/      # Opus 4.7 trading engine — backtests, analytics, :8090 dashboard
├── README.md          # This file
└── .github/           # CI / repo metadata
```

| Path | Description |
|---|---|
| `digital-intern/daemon.py` | Entrypoint; spins up ~20 worker threads |
| `digital-intern/collectors/` | 21 source collectors (RSS, GDELT, Finnhub, SEC, Reddit, …) |
| `digital-intern/ml/` | ArticleNet model, trainer, recursive LLM labeler |
| `digital-intern/data/articles.db` | Symlink → USB SQLite DB (~1.8M+ articles) |
| `paper-trader/runner.py` | Entrypoint; live decision loop + report scheduler |
| `paper-trader/paper_trader/strategy.py` | Opus 4.7 prompt builder + JSON decision parser |
| `paper-trader/paper_trader/analytics/` | 28 analytics modules feeding the dashboard |
| `paper-trader/run_continuous_backtests.py` | Continuous backtest loop feeding the ML scorer |

## Quick start

Both projects run as systemd services on the host.

```bash
# digital-intern + paper-trader run as SYSTEM units (require sudo)
sudo systemctl start digital-intern      # news daemon  → dashboard :8080
sudo systemctl start paper-trader        # trading loop → dashboard :8090

# continuous-backtests + the 5-minute watchdog run as USER units
systemctl --user start continuous-backtests
systemctl --user start trading-watchdog.timer

# Check health
sudo systemctl status digital-intern paper-trader
systemctl --user status continuous-backtests trading-watchdog.timer
```

Run directly for development:

```bash
cd digital-intern && python3 daemon.py     # http://localhost:8080
cd paper-trader   && python3 runner.py     # http://localhost:8090
```

Both services read environment from `digital-intern/.env`. `continuous-backtests` additionally reads `paper-trader/.env`.

## Subproject documentation

- [`digital-intern/README.md`](digital-intern/README.md) — collectors, DB schema, ML pipeline, bulk historical collection, API endpoints, troubleshooting
- [`paper-trader/README.md`](paper-trader/README.md) — decision schema, analytics modules, backtest engine, ML advisor gate, API endpoints, troubleshooting

## Operational notes

- **Service scope:** `digital-intern` and `paper-trader` are *system* services (`/etc/systemd/system/`); `continuous-backtests` and `trading-watchdog.timer` are *user* services (`~/.config/systemd/user/`). The watchdog restarts a service every 5 minutes if it goes down.
- **Storage:** `articles.db` lives on a USB-mounted drive; the systemd `usb.conf` drop-in adds `RequiresMountsFor=/media/zeph/projects` so the daemon will not start before the mount is ready.
- **Cron:** heartbeat, healthcheck, and hourly-review jobs are installed in the user crontab and point at the monorepo paths.
</content>
</invoke>
