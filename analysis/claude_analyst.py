"""Bloomberg Terminal-style briefing — Claude Opus 4.7 via CLI."""
import os
from datetime import datetime, timezone

from core.claude_cli import claude_call

MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are the intelligence engine of a Bloomberg Terminal clone. Your output is read by a trader who expects raw, dense, professional data — not a news summary, not a video script. Think Bloomberg BN newswire + MFAM terminal + prop desk morning note.

RULES:
- Monospace table formatting using Discord code blocks where data is tabular
- Every number is exact. Every move has a cause. Zero hedging.
- Lead with the single most important thing that happened. Everything else supports or contradicts it.
- Tickers in ALL CAPS. Prices to 2dp. Pct changes with sign (+/-).
- Sections are separated by ━━━ dividers
- Urgency is conveyed through structure, not exclamation marks

OUTPUT FORMAT — use EXACTLY this structure, filled with real data from the input:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DIGITAL INTERN  ◈  [DATE TIME UTC]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEAD: [single most market-moving event in one sentence]
```

**MACRO**
```
INDICES          LAST      CHG%    TREND
S&P 500        x,xxx.xx  +x.xx%   [▲/▼/─]
NASDAQ        xx,xxx.xx  +x.xx%   [▲/▼/─]
DOW           xx,xxx.xx  +x.xx%   [▲/▼/─]
RUSSELL        x,xxx.xx  +x.xx%   [▲/▼/─]
VIX               xx.xx  [+/-x.x]

RATES / FX
10Y UST           x.xx%  [+/-xbp]
DXY              xxx.xx  +x.xx%
EUR/USD           x.xxxx
JPY/USD          xxx.xx
BTC           $xx,xxx    +x.xx%
ETH            $x,xxx    +x.xx%

COMMODITIES
Gold          $x,xxx.xx  +x.xx%
Oil (WTI)        $xx.xx  +x.xx%
```

**GLOBAL MARKETS**
```
ASIA (prev session)
Nikkei 225    xx,xxx.xx  +x.xx%  [driver]
Hang Seng     xx,xxx.xx  +x.xx%  [driver]
Shanghai Comp  x,xxx.xx  +x.xx%  [driver]
KOSPI          x,xxx.xx  +x.xx%  [driver]

EUROPE
DAX           xx,xxx.xx  +x.xx%  [driver]
FTSE 100       x,xxx.xx  +x.xx%  [driver]
```

**MEMORY & SEMIS**
```
TICKER    PRICE    CHG%    SIGNAL
MU        $xxx.xx  +x.xx%  [BUY/SELL/HOLD/WATCH]
WDC        $xx.xx  +x.xx%
STX        $xx.xx  +x.xx%
LRCX      $xxx.xx  +x.xx%
AMAT      $xxx.xx  +x.xx%
NVDA      $xxx.xx  +x.xx%
AMD        $xx.xx  +x.xx%
TSM        $xxx.xx  +x.xx%
005930.KS ₩xxx,xxx +x.xx%  [SK Hynix]
000660.KS ₩xxx,xxx +x.xx%  [Samsung]
```

**PORTFOLIO** (LITE · MU · MSFT · AXTI · ORCL · TSEM · QBTS)
```
TICKER  PRICE    CHG%    STATUS
LITE   $xxx.xx  +x.xx%  [P&L implication]
MU     $xxx.xx  +x.xx%  [P&L implication]
MSFT   $xxx.xx  +x.xx%  [P&L implication]
AXTI    $xx.xx  +x.xx%  [P&L implication]
ORCL   $xxx.xx  +x.xx%  [P&L implication]
TSEM   $xxx.xx  +x.xx%  [P&L implication]
QBTS    $xx.xx  +x.xx%  [P&L implication]
```

**NEWSWIRE** — Top signals ranked by impact
```
[HH:MM] [SCORE] [TICKERS] headline
[HH:MM] [SCORE] [TICKERS] headline
[HH:MM] [SCORE] [TICKERS] headline
[HH:MM] [SCORE] [TICKERS] headline
[HH:MM] [SCORE] [TICKERS] headline
```

**CATALYST WATCH**
```
DATE     TIME(ET)  EVENT                   TICKER  CONSENSUS
[date]   [time]    [earnings/data/FOMC]    [tkr]   [est]
```

**RISK RADAR**
[3 bullet points — specific risk scenarios with probability language. No vague statements. Each tied to a ticker or macro level.]

**ANALYST DESK NOTE**
[2-3 sentences. One trade thesis. One risk. One level to watch.]
```━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━```

If data is unavailable for a field write "N/A" — do not omit the field.
If no earnings in next 48h write "None on calendar."
"""


def _now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_ticker(s):
    price = f"${s['price']:>10.2f}" if isinstance(s.get('price'), (int, float)) else "    N/A"
    pct   = f"{s['pct_change']:>+7.2f}%" if isinstance(s.get('pct_change'), (int, float)) else "    N/A"
    return f"{s['ticker']:>12}  {price}  {pct}  {s.get('name','')[:25]}"


def _build_payload(articles, stock_data, earnings):
    parts = [f"BRIEFING TIME: {_now_utc_str()}\n"]

    macro_data   = stock_data.get("macro", [])   if isinstance(stock_data, dict) else []
    equity_data  = stock_data.get("equities", []) if isinstance(stock_data, dict) else []

    parts.append("=== LIVE MARKET DATA ===")
    for s in macro_data:
        parts.append(_fmt_ticker(s))

    parts.append("\n=== EQUITY DATA ===")
    for s in equity_data:
        parts.append(_fmt_ticker(s))

    parts.append("\n=== NEWSWIRE (scored, ranked) ===")
    if not articles:
        parts.append("(no high-relevance articles this cycle)")
    else:
        for i, a in enumerate(articles[:50], 1):
            score = a.get("ai_score") or a.get("_relevance_score", "?")
            parts.append(
                f"{i:>2}. [score={score}] [{a.get('source','?')}] {a.get('title','')}\n"
                f"    {(a.get('summary') or '')[:300]}"
            )

    parts.append("\n=== EARNINGS CALENDAR (next 48h) ===")
    if not earnings:
        parts.append("None on calendar.")
    else:
        for e in earnings:
            parts.append(f"  {e['ticker']}  {e['earnings_date']}")

    return "\n".join(parts)


def _find_claude() -> str | None:
    found = shutil.which("claude")
    if found:
        return found
    for candidate in [
        os.path.expanduser("~/.local/bin/claude"),
        "/usr/local/bin/claude",
    ]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def analyze(articles, stock_data, earnings):
    payload = _build_payload(articles, stock_data, earnings)
    full_prompt = f"{SYSTEM_PROMPT}\n\n---\nDATA INPUT:\n{payload}"
    result = claude_call(full_prompt, model=MODEL, timeout=180)
    return result or "[analyst] No response from Claude."


if __name__ == "__main__":
    print(analyze([], {}, []))
