"""Bloomberg Terminal-style briefing — Claude Opus 4.7 via CLI."""
from datetime import datetime, timezone

from core.claude_cli import claude_call

MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are a financial intelligence briefing engine. Output is posted directly to Discord. Format must render cleanly there.

RULES:
- Every number exact. Every move has a cause. Zero hedging.
- Tickers in ALL CAPS. Prices to 2dp. Pct changes with sign (+/-).
- Each table in its own code block. Section headers as plain **bold** outside code blocks.
- Total output must fit in 1800 characters. Be ruthlessly concise. Cut low-signal rows.
- No nested backticks. No backtick dividers. Dividers are plain ━━━ lines outside code blocks.

OUTPUT FORMAT — use EXACTLY this, filled with real data:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**DIGITAL INTERN** ◈ [DATE TIME UTC]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**LEAD:** [single most market-moving event, one sentence]

**MACRO**
```
INDEX        LAST       CHG%
S&P 500    x,xxx.xx   +x.xx%
NASDAQ    xx,xxx.xx   +x.xx%
VIX           xx.xx   [+/-x.x]
10Y UST        x.xx%  [+/-xbp]
BTC        $xx,xxx    +x.xx%
Gold       $x,xxx     +x.xx%
Oil (WTI)    $xx.xx   +x.xx%
```

**PORTFOLIO** (SAO — LITE · LNOK · MUU · DRAM CALL C59)
```
TICKER       PRICE     CHG%   NOTE
LITE       $x,xxx.xx  +x.xx%  [implication]
LNOK          $xx.xx  +x.xx%  [implication]
MUU          $xxx.xx  +x.xx%  [implication]
MU (watch)   $xxx.xx  +x.xx%  [DRAM call driver]
```

**SEMIS PULSE**
```
NVDA  $xxx  +x.xx%  |  MU  $xxx  +x.xx%  |  TSM  $xxx  +x.xx%
AMD   $xxx  +x.xx%  |  AMAT $xxx +x.xx%  |  SMH  $xxx  +x.xx%
```

**TOP SIGNALS**
```
[HH:MM] [score] [TICKER] headline — one line each, max 5
```

**RISK / CATALYST**
- [risk 1 — specific, tied to ticker/level]
- [risk 2]
- [upcoming catalyst with date and ticker]

**DESK NOTE:** [1-2 sentences. One thesis. One level to watch.]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If data unavailable write N/A. Omit empty sections entirely.
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


def analyze(articles, stock_data, earnings):
    payload = _build_payload(articles, stock_data, earnings)
    full_prompt = f"{SYSTEM_PROMPT}\n\n---\nDATA INPUT:\n{payload}"
    result = claude_call(full_prompt, model=MODEL, timeout=180)
    return result or "[analyst] No response from Claude."


if __name__ == "__main__":
    print(analyze([], {}, []))
