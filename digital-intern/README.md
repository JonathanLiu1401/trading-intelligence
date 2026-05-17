# Digital Intern

A continuous financial news intelligence daemon that collects, scores, and alerts on market-moving articles in real time, feeding into a paper trading engine.

## What it does

Digital Intern runs ~20 parallel worker threads pulling from 17+ sources, scores every article with a local GPU model, and fires Discord alerts when something is urgent. Every 5 hours an Opus 4.7 briefing distills the market landscape and posts it to Discord.

```
Collectors (17+ sources) → Heuristic triage → SQLite (articles.db)
       → ArticleNet (GPU inference) → Urgency scorer (Sonnet 4.6)
       → Alert agent (Discord) / Heartbeat briefing (Opus 4.7)
```

## Sources

| Category | Sources |
|---|---|
| News feeds | RSS (100+ feeds), Google News, Yahoo RSS, NewsAPI |
| Financial data | GDELT v2, Finnhub, Alpha Vantage, Polygon |
| Social | Reddit (r/investing, r/stocks, etc.), Nitter (Twitter/X), Substack |
| SEC filings | EDGAR 8-K real-time RSS |
| Web | 100+ financial sites scraped every 60s, Wikipedia |
| Data enrichment | Web scraper, Massive API |

## ML Pipeline

- **ArticleNet** — PyTorch MLP trained on LLM-labeled examples; outputs relevance (0–5) and urgency (0–1) per article
- **Recursive labeler** — Three-tier LLM labeling: heuristics → Sonnet 4.6 → Opus 4.7 for the hardest cases
- **Continuous training** — Model retrains every 2–3 minutes on GPU (RTX 3060) as new labels accumulate
- **Grey-zone routing** — Articles where the model is uncertain (`needs_llm=True`) are escalated to Sonnet 4.6

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

All scripts are checkpoint-resumable (JSON files in `data/`).

## Configuration

Copy `.env.example` to `.env` (or set env vars directly):

```env
FINNHUB_API_KEY=...
ALPHA_VANTAGE_KEY=...
SEC_USER_AGENT=YourApp contact@yourapp.com
DISCORD_WEBHOOK_URL=...
```

Portfolio positions and watchlist live in `config/portfolio.json` and `config/watchlist.json`.

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

- **paper-trader** at `/home/zeph/paper-trader/` — reads `articles.db` (read-only) to drive trading decisions
- Dashboard: `http://localhost:8080`
