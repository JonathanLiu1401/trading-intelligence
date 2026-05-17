"""Pipeline orchestrator: all collectors in parallel -> filter -> stocks -> earnings -> analyze -> notify."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from collectors.rss_collector import collect_rss
from collectors.gdelt_collector import collect_gdelt
from collectors.ticker_news import collect_ticker_news
from collectors.reddit_collector import collect_reddit
from collectors.stock_data import get_stock_data
from collectors.earnings_calendar import get_earnings
from triage.local_filter import filter_articles
from analysis.claude_analyst import analyze
from notifier.discord_notifier import send as discord_send


def _log(step: str, msg: str = ""):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] [{step}] {msg}", flush=True)


def _merge_dedupe(batches: list[list]) -> list:
    """Merge multiple article lists, deduplicate by URL."""
    seen_urls = set()
    merged = []
    for batch in batches:
        for art in batch:
            url = art.get("link", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(art)
            elif not url:
                merged.append(art)  # keep urlless items (Reddit selfposts)
    return merged


def run_pipeline():
    t0 = time.time()
    _log("start", "Pipeline cycle starting")

    # Step 1: run all collectors in parallel
    _log("collect", "Launching all collectors in parallel (RSS + GDELT + ticker news + Reddit)...")
    collector_results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(collect_rss): "rss",
            executor.submit(collect_gdelt): "gdelt",
            executor.submit(collect_ticker_news): "ticker",
            executor.submit(collect_reddit): "reddit",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                collector_results[name] = result
                _log("collect", f"{name}: {len(result)} articles")
            except Exception as e:
                _log("collect", f"{name}: FAILED — {e}")
                collector_results[name] = []

    all_articles = _merge_dedupe([
        collector_results.get("rss", []),
        collector_results.get("gdelt", []),
        collector_results.get("ticker", []),
        collector_results.get("reddit", []),
    ])
    _log("collect", f"Total unique articles: {len(all_articles)}")

    # Step 2: filter
    _log("filter", f"Running triage filter on {len(all_articles)} articles...")
    filtered = filter_articles(all_articles)
    _log("filter", f"Forwarding {len(filtered)} relevant articles to Claude")

    # Step 3: stock data + earnings in parallel
    _log("market", "Fetching market data and earnings calendar...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        stocks_future = executor.submit(get_stock_data)
        earnings_future = executor.submit(get_earnings)
        stocks = stocks_future.result()
        earnings = earnings_future.result()

    macro_count = len(stocks.get("macro", []))
    equity_count = len(stocks.get("equities", []))
    _log("market", f"Stocks: {macro_count} macro + {equity_count} equities | Earnings: {len(earnings)} in 48h")

    # Step 4: Claude analysis
    _log("analyze", "Generating Claude Opus briefing...")
    briefing = analyze(filtered, stocks, earnings)
    _log("analyze", f"Briefing: {len(briefing)} chars")

    # Step 5: post to Discord
    _log("notify", "Posting to Discord...")
    ok = discord_send(briefing)
    _log("notify", f"Discord send {'ok' if ok else 'FAILED'}")

    elapsed = time.time() - t0
    _log("done", f"Cycle complete in {elapsed:.1f}s")
    return briefing


if __name__ == "__main__":
    run_pipeline()
