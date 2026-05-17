# Digital Intern

A continuous financial news intelligence daemon that collects, scores, and alerts on market-moving articles in real time, feeding into a paper trading engine.

## What it does

Digital Intern runs ~20 parallel worker threads pulling from 17+ sources, scores every article with a local GPU model, and fires Discord alerts when something is urgent. Every 5 hours an Opus 4.7 briefing distills the market landscape and posts it to Discord.

```
Collectors (17+ sources) → Heuristic triage → SQLite (articles.db)
       → ArticleNet (GPU inference) → Urgency scorer (Sonnet 4.6)
       → Alert agent (Discord) / Heartbeat briefing (Opus 4.7)
```

## Sources (21 collectors)

Every file in `collectors/` is one source adapter. The full set:

| Collector | Source | What it pulls |
|---|---|---|
| `rss_collector` | 100+ financial RSS feeds | Reuters, WSJ, MarketWatch, CNBC, Bloomberg, etc. every 30s |
| `google_news` | Google News RSS | Per-ticker query feeds, every 2 min |
| `yahoo_ticker_rss` | Yahoo Finance RSS | Per-ticker company news |
| `ticker_news` | yfinance `.news` | Ticker-attached headlines, every 60s |
| `newsapi_collector` | NewsAPI.org | Keyword + business-category articles |
| `gdelt_collector` | GDELT v2 DOC API | Global news event stream, sweep every 10 min |
| `finnhub_collector` | Finnhub company-news | Per-ticker news, 50 req/min |
| `alphavantage_collector` | Alpha Vantage NEWS_SENTIMENT | News + sentiment scores |
| `polygon_collector` | Polygon.io | Market news + ticker reference |
| `reddit_collector` | Reddit (r/investing, r/stocks, r/wallstreetbets, …) | Top/new posts every 45s |
| `nitter_collector` | Nitter (Twitter/X mirror) | Finance accounts / cashtags |
| `substack_collector` | Substack finance newsletters | New posts |
| `sec_edgar` | SEC EDGAR | Real-time 8-K filing RSS, every 5 min |
| `wikipedia_collector` | Wikipedia | Company/event page change signals |
| `web_scraper` | 100+ financial sites | Full-text scrape every 60s |
| `massive_collector` | Massive API | Bulk article enrichment |
| `stock_data` | yfinance | Price/quote context for scoring |
| `options_monitor` | Options chains | Unusual options activity signals |
| `earnings_calendar` | Earnings calendar feed | Upcoming earnings dates |
| `portfolio_pnl` | yfinance + `config/portfolio.json` | Portfolio P&L snapshots |
| `source_health` | Internal | Per-source freshness/error tracking |

## ML Pipeline

### ArticleNet architecture
- PyTorch MLP over a 512-dim text feature vector (`ml/embedder.py` → `ml/features.py`)
- Hidden layers with ReLU + dropout; two heads: a relevance head (0–5 regression) and an urgency head (0–1 sigmoid)
- Trained with MSE/BCE on LLM-labeled examples; runs batched GPU inference (RTX 3060) in `ml/inference.py`

### Training loop
- `ml/trainer.py` runs the full retrain on accumulated labels every ~3 min (worker W10)
- `continuous_trainer` (W12) does a lightweight 40-epoch GPU pass every ~2 min for fast adaptation
- `core/retrain_guard.py` is a circuit breaker: if labeling/training fails repeatedly (e.g., Claude org usage limit), it backs off instead of hot-looping. A run of `failures=N labeled=0` from the recursive labeler indicates the Claude CLI usage cap, not a code bug.

### Recursive labeling (three tiers)
1. **Heuristics** — `triage/heuristic_scorer.py` keyword scoring (0–5), cheap, runs on every article
2. **Sonnet 4.6** — grey-zone articles (`needs_llm=1`, model uncertain) batch-escalated via `watchers/urgency_scorer.py`
3. **Opus 4.7** — hardest/ambiguous cases via `ml/recursive_labeler.py`, every ~4h

### Calibration
- Predicted urgency is calibrated against realized LLM labels so the alert threshold maps to a stable precision; the alert agent fires only above the calibrated urgency cutoff.

## Architecture

```
daemon.py
├── W1  gdelt_worker          — GDELT sweep every 10 min
├── W2  rss_worker            — 100+ RSS feeds every 30s
├── W3  web_worker            — 100+ sites scraped every 60s
├── W4  reddit_worker         — Reddit every 45s
├── W5  ticker_worker         — yfinance news every 60s
├── W6  scorer_worker         — ArticleNet GPU inference
├── W7  alert_worker          — Discord alerts on urgent items
├── W8  heartbeat_worker      — Opus 4.7 briefing every 5h
├── W9  purge_worker          — cleanup every 6h
├── W10 ml_trainer_worker     — retrain every 3 min
├── W11 price_alert_worker    — >3% portfolio move alerts
├── W12 continuous_trainer    — lightweight 40-epoch GPU pass every 2 min
├── W12b recursive_labeler    — Sonnet+Opus labeling every 4h
├── W13 sec_edgar_worker      — SEC 8-K RSS every 5 min
├── W14 google_news_worker    — Google News per ticker every 2 min
├── W15 portfolio_pl_worker   — P&L snapshot every 5 min
├── W16 sentiment_trends      — trends JSON every 10 min
└── W17 web_server_worker     — Flask dashboard on :8080
```

## Directory Structure

```
digital-intern/
├── daemon.py               # main entrypoint — spins up all workers
├── collectors/             # one file per source (17 collectors)
├── triage/                 # heuristic_scorer.py — keyword scoring 0–5
├── storage/
│   └── article_store.py   # SQLite wrapper, INSERT OR IGNORE dedup by SHA256(url+title)
├── ml/
│   ├── model.py           # ArticleNet definition
│   ├── trainer.py         # training loop (GPU)
│   ├── inference.py       # batch scoring
│   ├── embedder.py        # text → 512-dim feature vectors
│   ├── features.py        # feature engineering
│   ├── recursive_labeler.py  # LLM label pipeline
│   └── sentiment_trends.py
├── watchers/
│   ├── urgency_scorer.py  # Sonnet 4.6 batch scoring for grey-zone articles
│   └── alert_agent.py     # Discord alert dispatch
├── analysis/
│   └── claude_analyst.py  # Opus 4.7 heartbeat briefing
├── notifier/
│   ├── discord_notifier.py
│   └── tts.py             # text-to-speech for voice alerts
├── core/
│   ├── logger.py          # structured logging + metrics
│   ├── backoff.py         # exponential backoff
│   ├── claude_cli.py      # Claude CLI subprocess wrapper
│   └── retrain_guard.py   # ML retrain circuit breaker
├── dashboard/             # Flask dashboard on :8080
├── scheduler/             # cron-style pipeline runner
├── scripts/               # bulk historical collection (see below)
├── tests/                 # pytest suite (~40 test files)
├── config/
│   ├── portfolio.json     # current positions + watchlist
│   └── watchlist.json     # extended ticker universe
└── data/
    └── articles.db        # SQLite DB (WAL mode, ~750K+ articles)
```

## Bulk Historical Collection

Scripts to backfill years of articles for ML training:

| Script | Source | Rate | Coverage |
|---|---|---|---|
| `scripts/gdelt_gkg_bulk.py` | GDELT GKG v2 daily ZIPs | 20 parallel, no limit | 2015–present, ~100–200 articles/file |
| `scripts/gdelt_historical_sweep.py` | GDELT v2 DOC API | 1 req/5.1s | 2015–present, yearly windows |
| `scripts/finnhub_historical_news.py` | Finnhub company-news | 50 req/min | 2018–present, 54 tickers |
| `scripts/sec_edgar_bulk.py` | SEC EDGAR full-index | 5 req/sec | 1994–present, 8-K filings |
| `scripts/bulk_collect.sh` | All four in parallel | — | orchestrator with progress monitor |

```bash
# Run all four in parallel (recommended)
bash scripts/bulk_collect.sh 2015

# Or individually
python3 scripts/gdelt_gkg_bulk.py 2018
python3 scripts/sec_edgar_bulk.py 2015
```

### Expected counts & timing

| Window | Source | Approx. articles | Wall time (this host) |
|---|---|---|---|
| 1 year, all 4 in parallel | `bulk_collect.sh` | ~250K–500K | ~6–12 h |
| 2015–present GKG | `gdelt_gkg_bulk.py` | ~1.5M–2M | multi-day (resumable) |
| 2018–present, 54 tickers | `finnhub_historical_news.py` | ~150K–300K | ~3–6 h (50 req/min cap) |
| 1994–present 8-Ks | `sec_edgar_bulk.py` | ~400K–600K | ~8–16 h (5 req/s cap) |

The live `articles.db` on this host is ~1.8M+ rows. All scripts are checkpoint-resumable (JSON checkpoint files in `data/`, atomic writes) — safe to kill and restart.

## API Endpoints (dashboard :8080)

Flask dashboard served by worker W17 (`dashboard/`):

| Endpoint | Returns |
|---|---|
| `GET /healthz` | Liveness probe |
| `GET /api/health` | Component health summary |
| `GET /api/stats` | Aggregate counts (articles, sources, labels) |
| `GET /api/articles?limit=N` | Most recent scored articles |
| `GET /api/articles_per_hour` | Ingest rate histogram |
| `GET /api/volume-history` | Article volume over time |
| `GET /api/briefings` | Recent Opus 4.7 heartbeat briefings |
| `GET /api/trends` | Sentiment trend series |
| `GET /api/metrics` | Internal metrics (worker timings, queue depths) |
| `GET /api/ml-status` | ArticleNet training state, last loss, label counts |
| `GET /api/collector-health` | Per-source freshness / error rates |
| `GET /api/portfolio` | Current portfolio P&L snapshot |
| `GET /api/portfolio/config` | Portfolio + watchlist config |
| `GET /api/earnings` | Upcoming earnings calendar |
| `GET /api/invariants` | Runtime invariant checks |
| `GET /api/logs` | Recent structured log lines |
| `POST /api/chat` | Chat-with-the-daemon endpoint (`/chat` UI) |
| `POST /api/restart` | Trigger a graceful daemon restart |

## Configuration

Environment is loaded from `.env` (also consumed by paper-trader via systemd `EnvironmentFile`).

| Variable | Purpose |
|---|---|
| `FINNHUB_API_KEY` | Finnhub company-news collector |
| `ALPHA_VANTAGE_KEY` | Alpha Vantage NEWS_SENTIMENT collector |
| `POLYGON_API_KEY` | Polygon.io collector |
| `NEWSAPI_KEY` | NewsAPI.org collector |
| `SEC_USER_AGENT` | Required UA string for SEC EDGAR (`App contact@domain`) |
| `DISCORD_WEBHOOK_URL` | Alert + briefing webhook (watchdog also reads this) |
| `ANTHROPIC_API_KEY` / Claude CLI auth | Sonnet/Opus labeling + briefings |

Portfolio positions and watchlist live in `config/portfolio.json` and `config/watchlist.json`. Source list (the "blitz" feed set) lives in `config/sources.json`.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Daemon won't start, exits immediately | USB mount not ready. The `usb.conf` drop-in requires `/media/zeph/projects`. Check `mount \| grep projects`; the symlink `data/articles.db` must resolve. |
| `recursive_labeler: failures=25 labeled=0` | Claude CLI org usage limit reached (not a code bug). `retrain_guard` circuit-breaks; resumes when quota resets. |
| Disk space warnings / large `logs/` | `daemon.log.{1,2,3,4}` and `structured.jsonl.*` are RotatingFileHandler backups (gitignored). Safe to truncate old `.log.N` / `.jsonl.N`. |
| ML trainer not advancing (`/api/ml-status` flat loss) | No new labels accumulating — check labeler tier 2/3 (quota) and that articles have `needs_llm` set. |
| Rate-limit / 429 from a source | Per-collector backoff (`core/backoff.py`) handles this; persistent failures show in `/api/collector-health`. |
| High memory (multi-GB peak) | Expected during GPU retrain bursts; the watchdog restarts if the process dies. Memory peak ~2–5 GB is normal on this host. |
| Daemon orphaned after restart | A previous `daemon.py` may survive a systemd restart. Kill any non-MainPID `daemon.py` (`pgrep -af daemon.py`). |

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Start the daemon
python3 daemon.py

# Or via systemd
systemctl start digital-intern

# Dashboard at http://localhost:8080
```

## Database Schema

`articles.db` (SQLite, WAL mode):

```sql
CREATE TABLE articles (
    article_id   TEXT PRIMARY KEY,   -- SHA256(url || "||" || title)
    url          TEXT,
    title        TEXT,
    source       TEXT,
    published    TEXT,               -- ISO 8601 UTC
    summary      TEXT,
    full_text    BLOB,               -- zlib-compressed
    kw_score     REAL,               -- heuristic 0–5
    relevance    REAL,               -- ArticleNet output
    urgency      REAL,               -- ArticleNet output
    needs_llm    INTEGER,            -- 1 = route to Sonnet
    llm_label    INTEGER,            -- LLM verdict
    inserted_at  TEXT
);
```

## Tests

```bash
pytest tests/ -x -q
```

## Related

- **paper-trader** at `../paper-trader/` (monorepo sibling) — reads `articles.db` (read-only) to drive trading decisions
- Monorepo root: [`../README.md`](../README.md)
- Dashboard: `http://localhost:8080`
