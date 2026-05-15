"""GDELT 2.0 DOC API collector — 7-day window, parallel queries, SQLite deduplication."""
import hashlib
import requests
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250       # GDELT API hard limit per query
TIMESPAN = "10080"      # 7 days in minutes — maximise coverage; SQLite dedupes repeats
REQUEST_TIMEOUT = 20
MAX_WORKERS = 30

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

# -------------------------------------------------------------------------
# Query list — breadth > depth; each query returns up to 250 unique articles
# -------------------------------------------------------------------------
QUERY_GROUPS = [
    # --- Memory core ---
    "DRAM memory pricing", "NAND flash pricing", "HBM memory AI chips",
    "Micron Technology DRAM", "Micron earnings revenue",
    "SK Hynix DRAM HBM", "SK Hynix earnings",
    "Samsung memory chips", "Samsung semiconductor",
    "Kioxia NAND flash IPO", "Western Digital NAND storage",
    "memory chip supply demand", "DRAM ASP pricing",
    "memory oversupply glut", "HBM3E production",
    "wafer starts production", "bit growth demand",
    # --- Semis equipment ---
    "ASML EUV lithography", "Lam Research wafer",
    "Applied Materials AMAT earnings", "KLA Corporation KLAC",
    "Tokyo Electron semiconductor", "TSMC N2 fab",
    "TSMC advanced packaging", "Intel foundry IFS",
    "semiconductor equipment orders",
    # --- AI / GPU demand ---
    "Nvidia GPU AI demand", "Nvidia earnings revenue",
    "AMD AI GPU data center", "AMD earnings",
    "Qualcomm chips mobile", "Marvell AI networking",
    "Broadcom AI ASIC", "AI chip supply shortage",
    "AI data center capex", "hyperscaler AI spending",
    "Google TPU AI chip", "Microsoft Azure AI chips",
    "Meta AI infrastructure", "Amazon AWS AI chips",
    "generative AI semiconductor demand",
    # --- Portfolio tickers ---
    "Lumentum LITE photonics", "AXT semiconductor AXTI",
    "Tower Semiconductor TSEM", "D-Wave quantum computing QBTS",
    "Oracle cloud ORCL earnings", "Microsoft MSFT earnings cloud",
    # --- US macro ---
    "Federal Reserve rate decision", "FOMC minutes statement",
    "US inflation CPI report", "PCE deflator inflation",
    "nonfarm payroll employment", "US unemployment jobless",
    "US GDP growth quarter", "US recession risk",
    "US treasury yield curve", "10 year treasury bond",
    "S&P 500 earnings season", "S&P 500 market rally selloff",
    "Nasdaq technology stocks", "Russell 2000 small cap",
    "VIX volatility fear index", "stock market correction",
    # --- Fed / rates ---
    "Fed pivot rate cuts 2026", "Fed balance sheet QT",
    "US dollar DXY strength", "dollar index Fed",
    # --- China / Asia ---
    "China economy stimulus", "China GDP growth",
    "China export controls chips", "US China trade war",
    "China property market Evergrande", "PBOC rate cut",
    "China consumption retail", "China manufacturing PMI",
    "Japan Bank of Japan yield", "Japan yen dollar",
    "Nikkei 225 Japan stocks", "Japan inflation CPI",
    "South Korea KOSPI economy", "Korea exports semiconductor",
    "Korea SK Hynix Samsung earnings",
    "Taiwan semiconductor TSMC geopolitical",
    "Taiwan strait military tension",
    # --- Europe ---
    "ECB rate decision Europe", "European Central Bank inflation",
    "Germany DAX economy recession", "UK FTSE economy",
    "European economy GDP", "Euro dollar exchange rate",
    "France CAC 40 market", "Italy economy",
    # --- India / Emerging ---
    "India Nifty economy market", "India semiconductor",
    "Brazil Bovespa economy", "emerging markets stocks",
    "MSCI EM emerging market fund",
    # --- Commodities / Energy ---
    "oil price OPEC production", "crude oil WTI Brent",
    "gold price inflation hedge", "silver commodities",
    "natural gas LNG price", "copper demand China",
    "lithium battery EV demand", "rare earth China export",
    # --- Crypto ---
    "Bitcoin price rally", "Bitcoin ETF institutional",
    "Ethereum price DeFi", "crypto regulation SEC",
    "stablecoin market", "crypto market correction",
    "Solana crypto ecosystem",
    # --- Geopolitical ---
    "US tariff trade policy 2026", "semiconductor export ban BIS",
    "Middle East conflict oil", "Russia sanctions economy",
    "BRICS dollar alternatives",
    # --- Earnings season ---
    "tech earnings beat revenue", "earnings miss guidance cut",
    "analyst upgrade price target", "analyst downgrade sell rating",
    "Q1 2026 earnings results", "quarterly earnings season",
    # --- Specific news types ---
    "IPO listing stock market", "merger acquisition deal",
    "stock buyback dividend", "short squeeze gamma squeeze",
    "insider buying selling", "SEC filing 13F",
    "private equity LBO", "venture capital AI funding",
    # --- Options / derivatives ---
    "options expiry gamma", "put call ratio sentiment",
    "VIX options hedging", "implied volatility earnings",
    # --- Global finance ---
    "IMF World Bank global", "sovereign debt crisis",
    "currency devaluation FX", "carry trade yen",
    "global supply chain disruption", "shipping freight rates",
    # --- Global macro (expanded) ---
    "central bank interest rates inflation global",
    "ECB Federal Reserve BOJ rate decision",
    "currency exchange forex dollar euro yen",
    "oil crude gold commodity price",
    "emerging markets GDP growth",
    # --- Asian markets (expanded) ---
    "Nikkei Hang Seng Shanghai stock market",
    "TSMC Samsung ASML earnings",
    "China economy property real estate",
    "Japan Bank of Japan stimulus",
    "Korea semiconductor memory",
    # --- European markets (expanded) ---
    "FTSE DAX CAC European stock market",
    "ECB Draghi rate decision euro",
    "LVMH Volkswagen ASML SAP earnings",
    "UK economy Brexit trade",
    # --- LatAm / MENA (expanded) ---
    "Brazil Bovespa Petrobras commodity",
    "Saudi Arabia oil OPEC production cut",
    "India Sensex Nifty earnings growth",
    # --- Crypto / macro (expanded) ---
    "Bitcoin Ethereum crypto blockchain DeFi",
    "stablecoin regulation SEC crypto",
    # --- Commodities (expanded) ---
    "gold silver copper lithium battery supply",
    "natural gas energy price winter",
    # --- FX (expanded) ---
    "dollar index DXY strong weak",
    "yuan renminbi devaluation",
    # --- Supply chain (deep) ---
    "container shipping rates freight",
    "port congestion supply chain",
    "semiconductor supply chain shortage",
    "lithium battery supply chain",
    "rare earth minerals China processing",
    "cobalt nickel battery materials",
    "Red Sea shipping disruption",
    "Panama Canal water levels shipping",
    "drewry container freight index",
    "Baltic dry index shipping",
    # --- Earnings / analyst actions (deep) ---
    "earnings beat analyst estimates",
    "earnings miss guidance cut",
    "buy rating initiation coverage",
    "guidance raised full year forecast",
    "guidance lowered cut full year",
    "preannouncement preliminary results",
    "consensus estimate revisions",
    "investor day analyst day guidance",
    # --- M&A / corporate ---
    "hostile takeover bid premium",
    "private equity buyout LBO",
    "spin-off divestiture corporate",
    "strategic review alternatives sale",
    "antitrust investigation FTC DOJ",
    "merger arbitrage spread",
    "going private take-private deal",
    "corporate restructuring chapter 11",
    # --- Macro (deep) ---
    "inflation expectations breakeven five-year",
    "yield curve inversion recession",
    "commercial real estate office vacancy",
    "bank credit conditions tightening SLOOS",
    "small business sentiment NFIB",
    "consumer sentiment confidence index",
    "credit card delinquency default rate",
    "auto loan delinquency subprime",
    "mortgage rate 30-year fixed",
    "housing starts permits building",
    "ISM manufacturing services PMI",
    "JOLTS job openings quits",
    "initial jobless claims weekly",
    "retail sales consumer spending",
    "industrial production capacity utilization",
    # --- Geopolitical / trade (deep) ---
    "CHIPS Act semiconductor subsidy",
    "tariff trade war retaliation",
    "sanctions Russia oil price cap",
    "sanctions China entity list",
    "South China Sea shipping incident",
    "North Korea missile test",
    "Iran oil sanctions enforcement",
    "Ukraine Russia war oil grain",
    "BIS entity list semiconductor",
    "FDPR foreign direct product rule",
    # --- AI / data center capex (deep) ---
    "GPU server AI infrastructure order",
    "hyperscaler cloud capex spending forecast",
    "sovereign AI investment fund",
    "AI energy consumption power grid",
    "nuclear power AI data center deal",
    "Stargate AI data center project",
    "Coreweave AI cloud GPU",
    "Lambda Labs AI cloud GPU",
    "Anthropic OpenAI Google compute deal",
    "Blackwell GB200 NVL72 deployment",
    "HBM3 HBM3E supply qualification",
    "CoWoS advanced packaging capacity",
    # --- Portfolio-specific (deep) ---
    "Micron MU analyst price target HBM",
    "Micron LPDDR5 HBM3E revenue",
    "Nvidia data center revenue guidance",
    "Nvidia Blackwell ramp shipments",
    "AMD MI300 MI350 GPU sales",
    "AMD Instinct AI accelerator",
    "ASML EUV order backlog high-NA",
    "Intel foundry IFS customer Microsoft",
    "Samsung HBM yield production qualification",
    "SK Hynix HBM3E HBM4 roadmap",
    "TSMC N2 N3 advanced node yields",
    "Lumentum LITE photonics datacom",
    "AXT compound semiconductor InP",
    "Oracle Cloud Infrastructure AI revenue",
    "Tower Semiconductor TSEM analog foundry",
    "D-Wave Quantum QBTS annealing",
    # --- Options / derivatives flow ---
    "unusual options activity call sweep",
    "dark pool prints institutional",
    "0DTE options volume gamma",
    "skew put-call ratio sentiment",
    # --- Energy / commodities (deep) ---
    "OPEC+ production cut quota",
    "Strategic Petroleum Reserve release refill",
    "natural gas storage EIA inventory",
    "uranium price spot enrichment",
    "copper inventory LME shanghai",
]

# Multi-language queries — GDELT v2 supports sourcelang=<Language> filter.
# Each entry: (query_string, sourcelang) — sourcelang None means English/default.
MULTILANG_QUERIES = [
    ("中国股市 科技股 半导体", "Chinese"),
    ("日経 株価 決算", "Japanese"),
    ("DAX Aktien Zinsen", "German"),
    ("CAC bourse taux", "French"),
    ("bolsa mercados acciones", "Spanish"),
    ("أسواق المال النفط", "Arabic"),
    ("주식 반도체 삼성", "Korean"),
]


def _ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS seen_articles "
        "(id TEXT PRIMARY KEY, link TEXT, title TEXT, source TEXT, first_seen TEXT)"
    )
    conn.commit()
    return conn


def _article_id(url: str, title: str) -> str:
    return hashlib.sha256(f"{url}||{title}".encode()).hexdigest()


def _fetch_query(keyword_query: str, sourcelang: str | None = None) -> list:
    params = {
        "query": keyword_query,
        "mode": "artlist",
        "maxrecords": MAX_RECORDS,
        "format": "json",
        "timespan": TIMESPAN,
        "sort": "DateDesc",
    }
    if sourcelang:
        params["sourcelang"] = sourcelang
    try:
        r = requests.get(GDELT_URL, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
        articles = data.get("articles") or []
        return [
            {
                "title": a.get("title", "").strip(),
                "link": a.get("url", ""),
                "summary": f"[{a.get('sourcecountry', '')}] {a.get('seendate', '')}",
                "published": a.get("seendate", ""),
                "source": f"GDELT/{a.get('domain', 'unknown')}",
                "_query": keyword_query,
            }
            for a in articles
            if a.get("title") and a.get("url")
        ]
    except Exception:
        return []


def collect_gdelt() -> list:
    """Run all queries in parallel; skip already-seen articles via SQLite."""
    total_queries = len(QUERY_GROUPS) + len(MULTILANG_QUERIES)
    print(f"[gdelt] Starting {total_queries} parallel queries "
          f"({len(QUERY_GROUPS)} EN + {len(MULTILANG_QUERIES)} multilang; "
          f"7-day window, max {MAX_RECORDS} each)...")
    t0 = time.time()

    conn = _ensure_db()
    all_articles = []
    seen_urls: set[str] = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_query, q): q for q in QUERY_GROUPS}
        for ml_q, lang in MULTILANG_QUERIES:
            futures[executor.submit(_fetch_query, ml_q, lang)] = f"{ml_q} [{lang}]"
        for future in as_completed(futures):
            for art in future.result():
                url = art["link"]
                title = art["title"]
                if not url or url in seen_urls:
                    continue
                aid = _article_id(url, title)
                cur = conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,))
                if cur.fetchone():
                    continue  # already processed in a previous cycle
                seen_urls.add(url)
                all_articles.append(art)
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles VALUES (?,?,?,?,?)",
                    (aid, url, title, art["source"], datetime.now(timezone.utc).isoformat()),
                )

    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    print(f"[gdelt] {len(all_articles)} new unique articles in {elapsed:.1f}s")
    return all_articles


if __name__ == "__main__":
    articles = collect_gdelt()
    print(f"Total new: {len(articles)}")
    for a in articles[:5]:
        print(f"  [{a['source']}] {a['title'][:80]}")
