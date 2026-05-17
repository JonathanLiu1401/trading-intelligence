"""Opus 4.7 trading strategy — packages context, asks Claude for a JSON decision,
executes it through paper trade plumbing. No hard risk limits — Opus has full autonomy."""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import statistics
import subprocess
import time
from datetime import date, datetime, timezone
from pathlib import Path

from . import market, signals
from .store import Store, get_store

MODEL = "claude-opus-4-7"
FALLBACK_MODEL = "claude-sonnet-4-6"
FALLBACK_TIMEOUT_S = 60
DECISION_TIMEOUT_S = 180
# Retry uses a shorter budget so a parse-failure rescue can't blow past the
# next 60s open-market cycle. Worst case adds DECISION_TIMEOUT_S + RETRY = 225s.
RETRY_TIMEOUT_S = 45
# Cap the raw-response excerpt we write back into decisions.reasoning. Long
# enough to diagnose JSON / prose / truncation, short enough to keep the DB lean.
RAW_CAPTURE_CHARS = 1000

# ML advisor gate — when the backtest ML model's median alpha consistently
# beats SPY, its (quant+news) recommendation is injected into the Opus prompt
# as an *advisory* opinion. Opus retains full autonomy over the final call.
ML_QUALIFY_MIN_RUNS = 20       # qualifying runs needed
ML_QUALIFY_MEDIAN_ALPHA = 0.0  # median vs_spy_pct must beat this (%)
ML_QUALIFY_TTL_S = 3600.0      # recheck every hour

_ml_qualify_cache: tuple[bool, str, float] | None = None

# Tracks the most recent claude subprocess so a new _claude_call can kill a
# lingering one from a prior cycle. Overlapping calls compete for the same API
# quota and cause *both* to time out. Process-local — does not guard against a
# second orphaned runner process, but covers stacked calls within one process
# (decision → retry → Sonnet fallback, or this cycle's call vs. the next).
_active_claude_proc: subprocess.Popen | None = None

WATCHLIST = [
    "LITE", "LNOK", "MUU", "DRAM", "SNDU",  # current real-account interests
    "NVDA", "AMD", "MU", "AMAT", "LRCX", "KLAC", "TSM", "ASML", "MRVL",  # semis
    "SMH", "SOXX", "SPY", "QQQ",  # ETFs
    # Leveraged ETFs — 3x Bull
    "TQQQ", "UPRO", "SPXL", "UDOW", "URTY",
    "SOXL", "TECL", "FNGU", "CURE", "LABU",
    "NAIL", "DFEN", "DPST", "FAS", "TNA", "UTSL",
    # Leveraged ETFs — 2x Bull
    "QLD", "SSO", "NVDU", "MSFU", "AMZU", "GOOGU", "METAU",
    "TSLL", "CONL", "BITU", "ETHU",
    # Leveraged Bear / Hedge
    "SQQQ", "SPXS", "SOXS", "TECS", "FNGD",
]

# Subset used for live quant indicator computation. Mix of mega-caps + leveraged.
QUANT_TICKERS_LIVE = [
    "SPY", "QQQ", "NVDA", "AMD", "MU", "TSM", "AAPL", "MSFT", "META",
    "TQQQ", "SOXL", "LITE",
]

FUTURES = ["ES=F", "NQ=F", "CL=F", "GC=F"]

SYSTEM_PROMPT = """You are managing a paper trading portfolio with $1000 starting capital.
Your ONLY goal is maximum profit. You have complete freedom over position sizing,
risk, leverage, and timing. There are NO enforced limits. You can:
- Put 100% of portfolio into one trade if you have high conviction
- Hold options through expiry if you believe in the thesis
- Go all-in on a single ticker
- Let losers run if you expect reversal
- Take leveraged ETF positions (MUU, LNOK, etc.)

LEVERAGE INSTRUMENTS AVAILABLE:
- Leveraged ETFs 3x Bull: TQQQ (QQQ), UPRO/SPXL (SPY), UDOW (Dow), URTY (Russell), SOXL (semis), TECL (tech), FNGU (FANGs), CURE (healthcare), LABU (biotech), NAIL (homebuilders), DPST (banks), FAS (financials), DFEN (defense), TNA (small-cap), UTSL (utilities)
- Leveraged ETFs 2x Bull: QLD (QQQ 2x), SSO (SPY 2x), NVDU (NVDA), MSFU (MSFT), AMZU (AMZN), GOOGU (GOOG), METAU (META), TSLL (TSLA), CONL (COIN), LNOK (Nokia), BITU (BTC), ETHU (ETH)
- Leveraged Bear/Hedge: SQQQ/SPXS (3x short index), SOXS (3x short semis), TECS (3x short tech), FNGD (3x short FANGs)
- For high-conviction directional trades, consider 2-3x leveraged ETFs instead of the underlying
- For options-equivalent exposure: buy deep ITM LEAPS calls (delta >0.80) to simulate leveraged long
- Risk: leveraged ETFs decay in sideways markets; best for strong trending moves only

POSITION SIZING GUIDANCE:
- High conviction (RSI+MACD+MA all aligned): up to 40% portfolio
- Medium conviction (2/3 signals aligned): 15-25%
- Low conviction / leveraged ETF: max 10%
- Never go 100% into one leveraged ETF (decay risk)

THINK LIKE A HEDGE FUND MANAGER WHO WANTS ASYMMETRIC RETURNS.
Small, safe trades will not outperform. Take calculated risks.
High conviction = large size. Low conviction = stay cash.

Respond with a SINGLE JSON object — no prose, no markdown fences. Schema:

{
  "action": "BUY" | "SELL" | "BUY_CALL" | "BUY_PUT" | "SELL_CALL" | "SELL_PUT" | "HOLD" | "REBALANCE",
  "ticker": "NVDA",
  "qty": 0.5,
  "strike": 900,             // only for option actions
  "expiry": "2026-05-30",    // only for option actions, YYYY-MM-DD
  "confidence": 0.85,
  "reasoning": "1-3 sentences why"
}

Return JSON with your decision. No limits on qty, strike, or cash used.
For SELL/SELL_CALL/SELL_PUT, ticker must match an open position (and strike/expiry for options).

TECHNICAL SIGNAL INTERPRETATION (use alongside news, not in isolation):
- RSI > 70 = overbought — avoid new longs, consider reducing; RSI < 30 = oversold — potential
  long opportunity if news/thesis supports it.
- MACD signal crossovers confirm momentum: positive macd_signal with rising price is bullish
  confirmation; negative macd_signal with falling price is bearish confirmation.
- Bollinger Band squeezes (bb_position near 0 after a tight range) often precede breakouts;
  bb_position approaching +2 or -2 signals stretched conditions and elevated reversal risk.
- Require volume confirmation for breakout trades: only trust a breakout when vol_ratio > 1.2.
  Low-volume breakouts often fail.
- Weight technical signals alongside news — neither alone is sufficient. A strong news catalyst
  with confirming technicals is high-conviction; a news catalyst that contradicts technicals
  (e.g. "beat earnings" on a stock at RSI 80 with bb_position +2) is a lower-conviction setup.

Return JSON ONLY.
"""


def _ema_live(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def _rsi_live(closes: list[float], period: int = 14) -> float | None:
    if len(closes) <= period:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_g = gains / period
    avg_l = losses / period
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        g = diff if diff > 0 else 0.0
        l = -diff if diff < 0 else 0.0
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_live(closes: list[float]) -> str | None:
    if len(closes) < 35:
        return None
    ema12 = _ema_live(closes, 12)
    ema26 = _ema_live(closes, 26)
    if not ema12 or not ema26:
        return None
    offset = len(ema12) - len(ema26)
    macd_line = [ema12[i + offset] - ema26[i] for i in range(len(ema26))]
    if len(macd_line) < 9:
        return None
    signal = _ema_live(macd_line, 9)
    if not signal:
        return None
    return "bullish" if macd_line[-1] > signal[-1] else "bearish"


_QUANT_CACHE: dict[str, tuple[dict, float]] = {}
_QUANT_TTL = 300.0  # 5 min — indicators change slowly intraday


def _stdev_live(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def get_quant_signals_live(tickers: list[str]) -> dict[str, dict]:
    """Fetch ~1y of daily closes from yfinance for each ticker and compute
    RSI(14), MACD bullish/bearish, 50/200 MA cross, plus expanded signals:
    rsi, macd_signal, bb_position, mom_5d, mom_20d, vol_ratio, wk52_pos.
    Cached 5 minutes per ticker."""
    import time as _time
    import yfinance as yf
    out: dict[str, dict] = {}
    for t in tickers:
        cached = _QUANT_CACHE.get(t)
        if cached and _time.time() - cached[1] < _QUANT_TTL:
            out[t] = cached[0]
            continue
        try:
            hist = yf.Ticker(t).history(period="1y", auto_adjust=False)
            if hist is None or hist.empty:
                continue
            closes = [float(c) for c in hist["Close"].tolist() if c == c]
            if len(closes) < 60:
                continue
            last = closes[-1]
            rsi = _rsi_live(closes, 14)
            macd_label = _macd_live(closes)
            if len(closes) >= 200:
                ma50 = sum(closes[-50:]) / 50
                ma200 = sum(closes[-200:]) / 200
                ma_cross = "golden" if ma50 > ma200 else "death"
            elif len(closes) >= 50:
                ma50 = sum(closes[-50:]) / 50
                ma_cross = "above50" if last > ma50 else "below50"
            else:
                ma_cross = None
            hi_52 = max(closes[-252:]) if len(closes) >= 252 else max(closes)
            lo_52 = min(closes[-252:]) if len(closes) >= 252 else min(closes)
            pct_h = (last - hi_52) / hi_52 * 100 if hi_52 else 0.0
            pct_l = (last - lo_52) / lo_52 * 100 if lo_52 else 0.0
            vol_ratio = None
            try:
                vols = [float(v) for v in hist["Volume"].tolist() if v == v]
                if len(vols) >= 21 and vols[-1] > 0:
                    avg20 = sum(vols[-21:-1]) / 20
                    if avg20 > 0:
                        vol_ratio = round(vols[-1] / avg20, 2)
            except Exception:
                pass

            # Expanded signals (lowercase keys per spec)
            macd_signal_val = None
            try:
                if len(closes) >= 35:
                    e12 = _ema_live(closes, 12)
                    e26 = _ema_live(closes, 26)
                    if e12 and e26:
                        offset = len(e12) - len(e26)
                        macd_line = [e12[i + offset] - e26[i] for i in range(len(e26))]
                        if len(macd_line) >= 9:
                            sig = _ema_live(macd_line, 9)
                            if sig:
                                macd_signal_val = round(sig[-1], 2)
            except Exception:
                macd_signal_val = None

            bb_position = None
            try:
                if len(closes) >= 20:
                    window20 = closes[-20:]
                    sma20 = sum(window20) / 20
                    sd20 = _stdev_live(window20)
                    if sd20 > 0:
                        raw = (last - sma20) / (2 * sd20)
                        bb_position = round(max(-2.0, min(2.0, raw)), 2)
            except Exception:
                bb_position = None

            mom_5d = None
            try:
                if len(closes) >= 6 and closes[-6] > 0:
                    mom_5d = round((last - closes[-6]) / closes[-6] * 100, 2)
            except Exception:
                mom_5d = None
            mom_20d = None
            try:
                if len(closes) >= 21 and closes[-21] > 0:
                    mom_20d = round((last - closes[-21]) / closes[-21] * 100, 2)
            except Exception:
                mom_20d = None

            wk52_pos = None
            try:
                if hi_52 > lo_52:
                    wk52_pos = round((last - lo_52) / (hi_52 - lo_52), 2)
            except Exception:
                wk52_pos = None

            rec = {
                "RSI": round(rsi, 1) if rsi is not None else None,
                "MACD": macd_label,
                "MA_cross": ma_cross,
                "vol_ratio": vol_ratio,
                "pct_from_52h": round(pct_h, 1),
                "pct_from_52l": round(pct_l, 1),
                # Expanded fields per spec
                "rsi": round(rsi, 2) if rsi is not None else None,
                "macd_signal": macd_signal_val,
                "bb_position": bb_position,
                "mom_5d": mom_5d,
                "mom_20d": mom_20d,
                "wk52_pos": wk52_pos,
            }
            _QUANT_CACHE[t] = (rec, _time.time())
            out[t] = rec
        except Exception as e:
            print(f"[strategy] quant signal fetch failed {t}: {e}")
    return out


def _format_quant_signals(sigs: dict[str, dict]) -> str:
    if not sigs:
        return "  (no quant signals available)"
    def _v(x):
        return "?" if x is None else x
    def _pct(x):
        return "?" if x is None else f"{x}%"
    return "\n".join(
        f"  {tk}: rsi={_v(q.get('rsi'))}  macd={_v(q.get('MACD'))}/{_v(q.get('macd_signal'))}  "
        f"ma_cross={_v(q.get('MA_cross'))}  bb_position={_v(q.get('bb_position'))}  "
        f"vol_ratio={_v(q.get('vol_ratio'))}  mom_5d={_pct(q.get('mom_5d'))}  "
        f"mom_20d={_pct(q.get('mom_20d'))}  "
        f"wk52_pos={_v(q.get('wk52_pos'))}  52h={_pct(q.get('pct_from_52h'))}  52l={_pct(q.get('pct_from_52l'))}"
        for tk, q in sorted(sigs.items())
    )


def _claude_call(prompt: str, timeout_s: int = DECISION_TIMEOUT_S,
                 model: str = MODEL) -> str | None:
    global _active_claude_proc
    if not shutil.which("claude"):
        print("[strategy] claude CLI not found")
        return None
    # A claude call still alive from a prior cycle competes for the same API
    # quota as this one and makes *both* time out. Kill the stale one first.
    if _active_claude_proc is not None and _active_claude_proc.poll() is None:
        print("[strategy] killing stale claude subprocess before new call")
        try:
            _active_claude_proc.kill()
            _active_claude_proc.wait(timeout=5)
        except Exception as e:
            print(f"[strategy] failed killing stale claude proc: {e}")
    _active_claude_proc = None
    try:
        proc = subprocess.Popen(
            ["claude", "--model", model, "--print",
             "--permission-mode", "bypassPermissions"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _active_claude_proc = proc
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            print(f"[strategy] claude timeout after {timeout_s}s")
            return None
        finally:
            _active_claude_proc = None
        if proc.returncode != 0:
            # The claude CLI frequently exits non-zero with an EMPTY stderr and
            # the real error written to stdout, which produced useless blank
            # "[strategy] claude err:" lines in runner.log (the operator could
            # not tell why decisions were being skipped). Log the returncode
            # plus a stdout tail so the failure is actually diagnosable.
            err = (stderr or "").strip()[:300]
            if not err:
                err = f"(empty stderr) stdout={(stdout or '').strip()[:200]!r}"
            print(f"[strategy] claude err (rc={proc.returncode}): {err}")
            return None
        return (stdout or "").strip() or None
    except Exception as e:
        print(f"[strategy] claude exception: {e}")
        _active_claude_proc = None
        return None


_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. "
    "Reply with the JSON decision object ONLY — no prose, no markdown fences, "
    "no commentary before or after. Start your response with `{` and end with `}`."
)


def _should_retry_parse(raw: str | None) -> bool:
    """Retry only when Claude actually returned text we couldn't parse.

    A None response means timeout / CLI error / empty stdout — retrying the
    same prompt would just hit the same wall. A non-empty raw that fails to
    parse suggests prose-wrapping or truncation, which a stronger JSON-only
    nudge can often rescue."""
    if not raw:
        return False
    return "{" not in raw or _parse_decision(raw) is None


def _parse_decision(raw: str) -> dict | None:
    if not raw:
        return None
    # strip ```json fences if model ignored instructions
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Walk to the first '{' and use raw_decode so trailing text after the
    # JSON object doesn't break parsing (greedy regex was over-matching).
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    print(f"[strategy] JSON parse failed; raw: {text[:300]}")
    return None


def _option_expired(expiry: str | None, today: date | None = None) -> bool:
    """True if the option's expiry date is strictly before today (UTC). An
    option is still live *on* its expiry date, so the comparison is `<`."""
    if not expiry:
        return False
    try:
        exp = date.fromisoformat(str(expiry)[:10])
    except (TypeError, ValueError):
        return False
    return exp < (today or datetime.now(timezone.utc).date())


def _expired_intrinsic(ticker: str, otype: str, strike: float) -> float:
    """Cash-settlement value (per share) of an *expired* option: its intrinsic
    value against the current underlying. 0.0 when out-of-the-money or the
    underlying price is unavailable. An expired option is never worth its
    purchase premium — falling back to avg_cost would mark a worthless
    contract at full cost forever and silently inflate equity."""
    try:
        und = market.get_price(ticker)
    except Exception:
        und = None
    if not und or und <= 0:
        return 0.0
    try:
        k = float(strike)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, und - k) if otype == "call" else max(0.0, k - und)


def _portfolio_snapshot(store: Store) -> dict:
    """Mark-to-market every open position, write back to DB, return summary."""
    positions = store.open_positions()
    stock_tickers = sorted({p["ticker"] for p in positions if p["type"] == "stock"})
    prices = market.get_prices(stock_tickers) if stock_tickers else {}

    marks: dict[int, tuple[float, float]] = {}
    enriched = []
    open_value = 0.0
    for p in positions:
        if p["type"] in ("call", "put"):
            multiplier = 100
            # An expired contract has no live chain — yfinance returns nothing,
            # so settle it at intrinsic against the underlying instead of
            # letting the avg_cost fallback below mark it at full premium.
            if _option_expired(p["expiry"]):
                cur = _expired_intrinsic(p["ticker"], p["type"], p["strike"])
            else:
                cur = market.get_option_price(p["ticker"], p["expiry"], p["strike"], p["type"])
        else:
            cur = prices.get(p["ticker"])
            multiplier = 1
        # `is not None`, not `or`: a legitimate 0.0 (expired worthless option)
        # must survive — `cur or avg_cost` would clobber it back to premium.
        cur = cur if cur is not None else p["avg_cost"]
        pl = (cur - p["avg_cost"]) * p["qty"] * multiplier
        pl_pct = ((cur - p["avg_cost"]) / p["avg_cost"]) * 100 if p["avg_cost"] else 0.0
        marks[p["id"]] = (cur, pl)
        enriched.append({**p, "current_price": cur, "unrealized_pl": pl, "pl_pct": pl_pct,
                         "market_value": cur * p["qty"] * multiplier})
        open_value += cur * p["qty"] * multiplier

    if marks:
        store.update_position_marks(marks)

    pf = store.get_portfolio()
    total = pf["cash"] + open_value
    store.update_portfolio(pf["cash"], total, [
        {k: v for k, v in pos.items() if k != "opened_at"} for pos in enriched
    ])
    return {
        "cash": pf["cash"],
        "total_value": total,
        "open_value": open_value,
        "positions": enriched,
    }


def _build_payload(snapshot: dict, top_signals: list[dict], sentiments: list[dict],
                   watch_prices: dict[str, float | None],
                   futures_prices: dict[str, float | None],
                   sp500: float | None, market_open: bool,
                   quant_signals: dict[str, dict] | None = None,
                   self_review_block: str | None = None) -> str:
    now = datetime.now(timezone.utc).isoformat()
    pos_lines = []
    for p in snapshot["positions"]:
        if p["type"] in ("call", "put"):
            pos_lines.append(
                f"  {p['ticker']} {p['type'].upper()} {p['strike']} {p['expiry']}: "
                f"qty={p['qty']} avg={p['avg_cost']:.2f} mark={p['current_price']:.2f} "
                f"P/L=${p['unrealized_pl']:.2f} ({p['pl_pct']:.1f}%)"
            )
        else:
            pos_lines.append(
                f"  {p['ticker']} {p['type']}: qty={p['qty']} avg={p['avg_cost']:.2f} "
                f"mark={p['current_price']:.2f} P/L=${p['unrealized_pl']:.2f} ({p['pl_pct']:.1f}%)"
            )

    sig_lines = []
    for s in top_signals[:10]:
        sig_lines.append(
            f"  [{s['ai_score']:.1f}] urg={s['urgency']} {s['title'][:120]}"
            + (f"  tickers={','.join(s['tickers'][:5])}" if s['tickers'] else "")
        )

    # Cap the quant block to keep the prompt lean: only tickers that are
    # (a) held, (b) mentioned in the top signals, or (c) top-5 watchlist
    # priority (list order is the priority signal — first entries are the
    # current real-account interests). Drops the long tail of curated
    # tickers that aren't actionable this cycle. Mirrors the existing n>0
    # sentiment filter below — shrink the rendered string, not the fetch.
    if quant_signals:
        held = {p["ticker"] for p in snapshot["positions"]}
        mentioned = {
            t for s in top_signals[:10] for t in (s.get("tickers") or [])
        }
        priority = set(WATCHLIST[:5])
        keep = held | mentioned | priority
        quant_signals = {
            tk: q for tk, q in quant_signals.items() if tk in keep
        }

    sent_lines = [
        f"  {r['ticker']:>6}: avg={r['avg_score']:.1f} n={r['n']} urgent={r['urgent']}"
        for r in sentiments if r["n"] > 0
    ]

    px_lines = [f"  {t}: {p:.2f}" if p else f"  {t}: N/A" for t, p in watch_prices.items()]
    fut_lines = [f"  {t}: {p:.2f}" if p else f"  {t}: N/A" for t, p in futures_prices.items()]

    sp = f"{sp500:.2f}" if sp500 else "N/A"

    # Behavioural mirror — observational only (advisory; never gates). Placed
    # right after PORTFOLIO so the trader sees its own track record next to its
    # current book, before market data biases it.
    review_section = f"\n{self_review_block}\n" if self_review_block else ""

    return f"""TIME (UTC): {now}
MARKET_OPEN: {market_open}
S&P 500 BENCHMARK: {sp}

PORTFOLIO:
  cash: ${snapshot['cash']:.2f}
  open positions value: ${snapshot['open_value']:.2f}
  total value: ${snapshot['total_value']:.2f}
  positions:
{chr(10).join(pos_lines) if pos_lines else '  (none)'}
{review_section}
WATCHLIST PRICES:
{chr(10).join(px_lines)}

FUTURES:
{chr(10).join(fut_lines)}

TECHNICAL SIGNALS (RSI/MACD/MA cross/vol ratio/52w proximity):
{_format_quant_signals(quant_signals or {})}

TICKER SENTIMENT (last 4h, from scored news):
{chr(10).join(sent_lines) if sent_lines else '  (no scored mentions)'}

TOP SCORED SIGNALS (last 2h, ai_score >= 4.0):
{chr(10).join(sig_lines) if sig_lines else '  (no high-score signals)'}

NO RISK LIMITS — full autonomy. Size by conviction.

Return JSON only."""


def _build_fallback_payload(snap: dict, merged: list[dict],
                            quant_sigs: dict[str, dict]) -> str:
    """Condensed (~200-word) context for the Sonnet fallback when Opus times
    out: cash + total value + top 3 positions, top 5 article headlines with
    ai_score, and RSI/MACD for up to 3 held positions. Deliberately omits
    summaries, watchlist prices, futures, sentiment and self-review so Sonnet
    can answer well inside the short fallback timeout."""
    positions = snap.get("positions", [])
    top_pos = sorted(
        positions,
        key=lambda p: abs(p.get("market_value") or 0.0),
        reverse=True,
    )[:3]
    pos_lines = []
    for p in top_pos:
        if p["type"] in ("call", "put"):
            tk = f"{p['ticker']} {p['type'].upper()} {p['strike']} {p['expiry']}"
        else:
            tk = p["ticker"]
        pos_lines.append(
            f"  {tk}: qty={p['qty']} value=${(p.get('market_value') or 0.0):.2f}"
        )

    art_lines = []
    for a in merged[:5]:
        try:
            score = float(a.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        art_lines.append(f"  [{score:.1f}] {(a.get('title') or '')[:120]}")

    held = {p["ticker"] for p in positions}
    quant_lines = []
    for tk in sorted(held):
        q = quant_sigs.get(tk)
        if not q:
            continue
        quant_lines.append(
            f"  {tk}: RSI={q.get('rsi', '?')} MACD={q.get('MACD', '?')}"
        )
        if len(quant_lines) >= 3:
            break

    return f"""PORTFOLIO:
  cash: ${snap.get('cash', 0.0):.2f}
  total value: ${snap.get('total_value', 0.0):.2f}
  top positions:
{chr(10).join(pos_lines) if pos_lines else '  (none)'}

TOP SIGNALS (title + ai_score):
{chr(10).join(art_lines) if art_lines else '  (no high-score signals)'}

QUANT (held positions — RSI + MACD direction):
{chr(10).join(quant_lines) if quant_lines else '  (none held / no signals)'}

Return JSON only."""


def _enforce_risk_pre_trade(decision: dict, snapshot: dict) -> tuple[bool, str]:
    """Basic sanity only — can't sell more than you own. No position/option/cash caps."""
    action = decision.get("action", "HOLD")
    if action == "HOLD":
        return True, ""

    ticker = (decision.get("ticker") or "").upper()
    qty = float(decision.get("qty") or 0)
    if qty <= 0 and action != "REBALANCE":
        return False, "qty must be > 0"

    if action in ("SELL", "SELL_CALL", "SELL_PUT"):
        opt_type = "call" if action == "SELL_CALL" else "put" if action == "SELL_PUT" else "stock"
        matches = [
            p for p in snapshot["positions"]
            if p["ticker"] == ticker and p["type"] == opt_type
        ]
        if not matches:
            return False, f"no open {opt_type} position in {ticker} to close"
        held = sum(p["qty"] for p in matches)
        if qty > held + 1e-6:
            return False, f"sell qty {qty} exceeds held {held} for {ticker} {opt_type}"
    return True, ""


def _execute(decision: dict, snapshot: dict, store: Store) -> tuple[str, str]:
    """Apply the decision against the paper book. Returns (status, detail)."""
    action = decision.get("action", "HOLD")
    if action == "HOLD":
        return "HOLD", decision.get("reasoning", "")

    if action == "REBALANCE":
        return "HOLD", "REBALANCE not yet implemented; treated as HOLD"

    ticker = (decision.get("ticker") or "").upper()
    # Claude can emit a non-numeric qty (e.g. "all", "half"). Coerce defensively
    # so a bad field yields a recorded BLOCKED decision instead of an uncaught
    # ValueError that aborts the whole cycle with no decision/equity point logged.
    try:
        qty = float(decision.get("qty") or 0)
    except (TypeError, ValueError):
        return "BLOCKED", f"qty not numeric: {decision.get('qty')!r}"
    reason = decision.get("reasoning", "")

    ok, why = _enforce_risk_pre_trade(decision, snapshot)
    if not ok:
        return "BLOCKED", why

    if action in ("BUY", "SELL"):
        price = market.get_price(ticker)
        if not price:
            return "BLOCKED", f"no price for {ticker}"
        notional = price * qty
        if action == "BUY":
            if snapshot["cash"] - notional < 0:
                return "BLOCKED", f"insufficient cash (have ${snapshot['cash']:.2f}, need ${notional:.2f})"
            store.record_trade(ticker, "BUY", qty, price, reason)
            store.upsert_position(ticker, "stock", qty, price)
            store.update_portfolio(snapshot["cash"] - notional, snapshot["total_value"], snapshot["positions"])
            return "FILLED", f"BUY {qty} {ticker} @ {price:.2f}"
        else:
            store.record_trade(ticker, "SELL", qty, price, reason)
            store.upsert_position(ticker, "stock", -qty, price)
            store.update_portfolio(snapshot["cash"] + notional, snapshot["total_value"], snapshot["positions"])
            return "FILLED", f"SELL {qty} {ticker} @ {price:.2f}"

    if action in ("BUY_CALL", "BUY_PUT"):
        otype = "call" if action == "BUY_CALL" else "put"
        strike = decision.get("strike")
        expiry = decision.get("expiry")
        if not (strike and expiry):
            return "BLOCKED", "option trade missing strike/expiry"
        opt_px = market.get_option_price(ticker, expiry, float(strike), otype)
        if not opt_px:
            return "BLOCKED", f"no option price for {ticker} {expiry} {strike} {otype}"
        notional = opt_px * qty * 100
        if snapshot["cash"] - notional < 0:
            return "BLOCKED", f"insufficient cash (have ${snapshot['cash']:.2f}, need ${notional:.2f})"
        store.record_trade(ticker, action, qty, opt_px, reason, expiry=expiry,
                           strike=float(strike), option_type=otype)
        store.upsert_position(ticker, otype, qty, opt_px, expiry=expiry, strike=float(strike))
        store.update_portfolio(snapshot["cash"] - notional, snapshot["total_value"], snapshot["positions"])
        return "FILLED", f"{action} {qty} {ticker} {strike}{otype[0].upper()} {expiry} @ {opt_px:.2f}"

    if action in ("SELL_CALL", "SELL_PUT"):
        otype = "call" if action == "SELL_CALL" else "put"
        strike = decision.get("strike")
        expiry = decision.get("expiry")
        candidates = [p for p in snapshot["positions"]
                      if p["ticker"] == ticker and p["type"] == otype
                      and (not strike or p["strike"] == float(strike))
                      and (not expiry or p["expiry"] == expiry)]
        if not candidates:
            return "BLOCKED", f"no matching open {otype} for {ticker}"
        # If strike/expiry are unspecified and multiple contracts match, refuse
        # to pick — silently closing the "first" contract could exit the wrong
        # leg and lose intended exposure.
        if len(candidates) > 1 and (not strike or not expiry):
            legs = ", ".join(f"{p['strike']}{otype[0].upper()} {p['expiry']}" for p in candidates)
            return "BLOCKED", f"ambiguous {otype} close for {ticker}; specify strike+expiry (open: {legs})"
        match = candidates[0]
        # Cash flow must be bounded by what's actually held in the matched
        # contract — pre-trade check sums across all strikes/expiries and
        # would otherwise let qty over-credit cash here.
        if qty > match["qty"] + 1e-6:
            return "BLOCKED", (
                f"sell qty {qty} exceeds held {match['qty']} for "
                f"{ticker} {match['strike']}{otype[0].upper()} {match['expiry']}"
            )
        live_px = market.get_option_price(ticker, match["expiry"], match["strike"], otype)
        if live_px is not None:
            opt_px = live_px
        elif _option_expired(match["expiry"]):
            # Closing an expired contract settles at intrinsic, never at the
            # avg_cost breakeven the old `or match["avg_cost"]` produced.
            opt_px = _expired_intrinsic(ticker, otype, match["strike"])
        else:
            opt_px = match["avg_cost"]
        notional = opt_px * qty * 100
        store.record_trade(ticker, action, qty, opt_px, reason,
                           expiry=match["expiry"], strike=match["strike"], option_type=otype)
        store.upsert_position(ticker, otype, -qty, opt_px,
                              expiry=match["expiry"], strike=match["strike"])
        store.update_portfolio(snapshot["cash"] + notional, snapshot["total_value"], snapshot["positions"])
        return "FILLED", f"{action} {qty} {ticker} {match['strike']}{otype[0].upper()} {match['expiry']} @ {opt_px:.2f}"

    return "BLOCKED", f"unknown action {action}"


def _ml_is_qualified() -> tuple[bool, str]:
    """Return (qualified, reason). Cached ML_QUALIFY_TTL_S seconds.
    ML is qualified when median vs_spy_pct over last ML_QUALIFY_MIN_RUNS
    qualifying runs exceeds ML_QUALIFY_MEDIAN_ALPHA."""
    global _ml_qualify_cache
    now = time.time()
    if _ml_qualify_cache and now - _ml_qualify_cache[2] < ML_QUALIFY_TTL_S:
        return _ml_qualify_cache[0], _ml_qualify_cache[1]
    try:
        _db = Path(__file__).resolve().parent.parent / "backtest.db"
        conn = sqlite3.connect(f"file:{_db}?mode=ro", uri=True, timeout=5)
        try:
            rows = conn.execute(
                "SELECT vs_spy_pct FROM backtest_runs "
                "WHERE status='complete' AND vs_spy_pct IS NOT NULL AND n_trades >= 5 "
                "ORDER BY run_id DESC LIMIT ?",
                (ML_QUALIFY_MIN_RUNS,)
            ).fetchall()
        finally:
            conn.close()
        n = len(rows)
        if n < ML_QUALIFY_MIN_RUNS:
            result = (False, f"only {n}/{ML_QUALIFY_MIN_RUNS} qualifying runs")
        else:
            alphas = [r[0] for r in rows]
            median_a = statistics.median(alphas)
            if median_a > ML_QUALIFY_MEDIAN_ALPHA:
                result = (True, f"median alpha {median_a:+.1f}% over last {n} runs")
            else:
                result = (False, f"median alpha {median_a:+.1f}% ≤ threshold")
    except Exception as e:
        result = (False, f"qualification check error: {e}")
    _ml_qualify_cache = (result[0], result[1], now)
    return result[0], result[1]


# Self-contained sentiment / ticker-mapping tables for the ML advisory opinion.
# Deliberately NOT imported from backtest.py (circular-dependency risk) — these
# mirror the backtest engine's scorer vocabulary closely enough for an advisory.
_BULLISH_WORDS_LIVE = {
    "surges", "surge", "rally", "rallies", "gains", "gain", "beats", "beat",
    "rises", "rise", "jumps", "jump", "soars", "soar", "strong", "record",
    "bullish", "outperforms", "boosts", "boost", "breakout",
}
_BEARISH_WORDS_LIVE = {
    "falls", "fall", "drops", "drop", "plunges", "plunge", "slumps", "slump",
    "misses", "miss", "cuts", "cut", "warns", "warn", "weak", "bearish",
    "disappoints", "crashes", "crash", "declines", "decline",
}
_WORD_TO_TICKER_LIVE: dict[str, str] = {
    "nvidia": "NVDA", "amd": "AMD", "apple": "AAPL", "microsoft": "MSFT",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "tesla": "TSLA", "intel": "INTC", "micron": "MU", "broadcom": "AVGO",
    "qualcomm": "QCOM", "spy": "SPY", "qqq": "QQQ",
    "semiconductor": "SOXL", "chip": "SOXL", "chips": "SOXL",
    "nasdaq": "TQQQ", "ai": "TQQQ", "artificial intelligence": "TQQQ",
    "oil": "USO", "crude": "USO", "energy": "XLE", "opec": "USO",
    "federal reserve": "TLT", "fed rate": "TLT", "treasury": "TLT",
    "gold": "GLD", "bitcoin": "BTC-USD", "crypto": "COIN",
    "defense": "DFEN", "biotech": "LABU",
}
_LEVERAGED_ETFS_LIVE = {
    "TQQQ", "UPRO", "SPXL", "UDOW", "URTY", "SOXL", "TECL", "FNGU",
    "CURE", "LABU", "NAIL", "DPST", "FAS", "DFEN", "TNA", "UTSL",
    "QLD", "SSO", "NVDU", "MSFU", "AMZU", "TSLL", "CONL", "BITU", "ETHU",
    "SQQQ", "SPXS", "SOXS", "TECS", "FNGD",
}


def _ml_live_opinion(
    articles: list[dict],
    quant_sigs: dict[str, dict],
    snap: dict,
    watch_px: dict,
) -> dict | None:
    """Pure ML+quant advisory opinion using live data. Returns action/ticker/reasoning dict.
    Never raises — failure returns None."""
    try:
        # Score tickers from articles
        ticker_scores: dict[str, float] = {}
        ticker_article_count: dict[str, int] = {}
        ticker_max_urgency: dict[str, float] = {}
        for a in articles:
            # Live articles (signals.get_top_signals / get_urgent_articles)
            # carry "ai_score" — NOT "score". Reading only "score" here meant
            # raw_score was always 0.0, so EVERY article was skipped and the
            # news-sentiment half of this advisory was silently dead (it
            # degraded to quant-only, contradicting CLAUDE.md §15). Prefer the
            # live key; keep "score" as a fallback for any backtest-shaped input.
            try:
                raw_score = float(a.get("ai_score") or a.get("score") or 0.0)
            except (TypeError, ValueError):
                raw_score = 0.0
            if raw_score < 1.0:
                continue
            title = (a.get("title") or "").lower()
            words = set(title.split())
            bull = sum(1 for w in words if w in _BULLISH_WORDS_LIVE)
            bear = sum(1 for w in words if w in _BEARISH_WORDS_LIVE)
            total_sent = bull + bear
            sentiment = ((bull - bear) / total_sent) if total_sent else 0.0
            tickers = list(a.get("tickers") or [])
            for keyword, sym in _WORD_TO_TICKER_LIVE.items():
                if keyword in title and sym not in tickers:
                    tickers.append(sym)
            try:
                a_urg = float(a.get("urgency", 0.0) or 0.0)
            except (TypeError, ValueError):
                a_urg = 0.0
            for tk in tickers:
                if tk not in WATCHLIST:
                    continue
                ticker_scores[tk] = ticker_scores.get(tk, 0.0) + raw_score * sentiment
                ticker_article_count[tk] = ticker_article_count.get(tk, 0) + 1
                if a_urg > ticker_max_urgency.get(tk, 0.0):
                    ticker_max_urgency[tk] = a_urg

        # Quant adjustments
        for tk, q in quant_sigs.items():
            if tk not in WATCHLIST:
                continue
            adj = 0.0
            rsi = q.get("rsi")
            macd = q.get("macd_signal")
            mom5 = q.get("mom_5d")
            mom20 = q.get("mom_20d")
            bb = q.get("bb_position")
            if isinstance(rsi, (int, float)):
                if rsi < 33: adj += 1.5
                elif rsi < 45: adj += 0.5
                elif rsi > 67: adj -= 1.5
                elif rsi > 55: adj -= 0.5
            if isinstance(macd, (int, float)):
                adj += 0.5 if macd > 0 else -0.5
            if isinstance(mom5, (int, float)):
                adj += min(1.0, max(-1.0, mom5 / 3.0))
            if isinstance(mom20, (int, float)):
                adj += min(0.5, max(-0.5, mom20 / 10.0))
            if isinstance(bb, (int, float)):
                adj -= bb * 0.5
            ticker_scores[tk] = ticker_scores.get(tk, 0.0) + adj

        # Market regime from SPY 20d momentum
        spy_mom20 = quant_sigs.get("SPY", {}).get("mom_20d")
        if isinstance(spy_mom20, (int, float)):
            if spy_mom20 > 3.0:
                regime, regime_mult = "bull", 1.0
            elif spy_mom20 < -3.0:
                regime, regime_mult = "bear", 0.3
            else:
                regime, regime_mult = "sideways", 0.6
        else:
            regime, regime_mult = "unknown", 1.0

        # Pick best ticker above threshold
        buy_ticker: str | None = None
        best_score = 1.0
        for tk, s in ticker_scores.items():
            adj_s = s * regime_mult
            px = watch_px.get(tk)
            if adj_s > best_score and px and px > 0:
                best_score = adj_s
                buy_ticker = tk

        if not buy_ticker:
            return {"action": "HOLD", "ticker": "",
                    "reasoning": f"ML+quant: no high-conviction signal; regime={regime}"}

        q_buy = quant_sigs.get(buy_ticker, {})
        news_count = ticker_article_count.get(buy_ticker, 0)
        news_urg = ticker_max_urgency.get(buy_ticker, 0.0)
        if buy_ticker in _LEVERAGED_ETFS_LIVE and regime in ("bull", "sideways"):
            conviction = min(0.40, best_score / 15.0)
        else:
            conviction = min(0.25, best_score / 20.0)
        return {
            "action": "BUY",
            "ticker": buy_ticker,
            "reasoning": (
                f"ML+quant: {buy_ticker} score={best_score:.2f} regime={regime} "
                f"RSI={q_buy.get('rsi', 'N/A')} news_count={news_count} "
                f"news_urg={news_urg:.1f} conviction={conviction:.0%}"
            ),
        }
    except Exception as e:
        print(f"[strategy] _ml_live_opinion error: {e}")
        return None


def decide() -> dict:
    """Run one decision cycle. Returns summary dict for logging."""
    store = get_store()
    market_open = market.is_market_open()

    snap = _portfolio_snapshot(store)
    auto_exits: list[str] = []  # disabled — Opus has full autonomy

    top = signals.get_top_signals(20, hours=2, min_score=4.0)
    urgent = signals.get_urgent_articles(minutes=30)
    sents = signals.ticker_sentiments(WATCHLIST, hours=4)
    watch_px = market.get_prices(WATCHLIST)
    fut_px = {f: market.get_futures_price(f) for f in FUTURES}
    sp500 = market.benchmark_sp500()

    # Quant signals (RSI/MACD/MA cross) — include held positions + curated subset.
    held_tickers = sorted({p["ticker"] for p in snap["positions"]})
    quant_tickers = sorted(set(QUANT_TICKERS_LIVE) | set(held_tickers))
    try:
        quant_sigs = get_quant_signals_live(quant_tickers)
    except Exception as e:
        print(f"[strategy] quant signals failed: {e}")
        quant_sigs = {}

    # include urgent items at the top
    seen_ids = {s["id"] for s in top}
    merged = [a for a in urgent if a["id"] not in seen_ids] + top

    # Behavioural self-review — feed the trader its own track record (payoff
    # ratio, disposition gap, capital-paralysis state, open-book alpha) so it
    # can self-correct, exactly as a desk reviews its P&L before trading.
    # Advisory only; composes the existing pure builders (single source of
    # truth). Wrapped so a diagnostics failure NEVER blocks a trade — the
    # failure mode is "no mirror this cycle", never "no decision this cycle".
    self_review_block: str | None = None
    try:
        from .analytics.self_review import build_self_review
        sr = build_self_review(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(2000),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        )
        self_review_block = sr.get("prompt_block")
    except Exception as e:
        print(f"[strategy] self-review failed (non-fatal): {e}")

    # ML advisor: when model consistently beats SPY, include its opinion in prompt
    ml_opinion_block: str | None = None
    ml_qualified, ml_qual_reason = _ml_is_qualified()
    if ml_qualified:
        try:
            ml_op = _ml_live_opinion(merged, quant_sigs, snap, watch_px)
            if ml_op:
                ml_opinion_block = (
                    f"ML MODEL OPINION ({ml_qual_reason}):\n"
                    f"  Action: {ml_op['action']}"
                    + (f" {ml_op['ticker']}" if ml_op.get("ticker") else "")
                    + f"\n  Reasoning: {ml_op['reasoning']}\n"
                    "This is an advisory opinion only. You retain full autonomy over the final decision."
                )
                print(f"[strategy] ML advisor: {ml_op['action']} {ml_op.get('ticker', '')}")
        except Exception as e:
            print(f"[strategy] ML advisor failed (non-fatal): {e}")

    payload = _build_payload(snap, merged, sents, watch_px, fut_px, sp500, market_open,
                             quant_signals=quant_sigs,
                             self_review_block=self_review_block)
    prompt = f"{SYSTEM_PROMPT}\n\n---\nCONTEXT:\n{payload}"
    if ml_opinion_block:
        prompt += f"\n\n---\nML ADVISOR:\n{ml_opinion_block}"

    raw = _claude_call(prompt)
    decision = _parse_decision(raw) if raw else None
    retried = False
    fallback_used = False

    # True timeout (raw is None, so _should_retry_parse is False): Opus blew
    # past its budget. Retrying the full prompt on Opus would just stall again,
    # so fall back to a faster model with a condensed prompt instead of
    # recording a NO_DECISION.
    if raw is None:
        print("[strategy] Opus timeout — trying Sonnet fallback")
        fb_payload = _build_fallback_payload(snap, merged, quant_sigs)
        fb_prompt = f"{SYSTEM_PROMPT}\n\n---\nCONTEXT (condensed):\n{fb_payload}"
        raw = _claude_call(fb_prompt, timeout_s=FALLBACK_TIMEOUT_S,
                           model=FALLBACK_MODEL)
        if raw:
            fallback_used = True
            print("[strategy] Sonnet fallback returned response")
            decision = _parse_decision(raw)

    # Conditional one-shot retry: Claude returned text but it wasn't parseable.
    # A None response (timeout / empty stdout) won't be rescued by a retry —
    # same prompt, same failure — so we skip retrying in that case.
    if not decision and not fallback_used and _should_retry_parse(raw):
        retried = True
        print("[strategy] parse failed; retrying with JSON-only nudge")
        retry_raw = _claude_call(prompt + _RETRY_SUFFIX, timeout_s=RETRY_TIMEOUT_S)
        retry_decision = _parse_decision(retry_raw) if retry_raw else None
        if retry_decision:
            raw, decision = retry_raw, retry_decision
        else:
            # Keep first raw for diagnostics if retry also failed but the first
            # was actually empty; otherwise prefer the more recent attempt so
            # operators see what the retry looked like.
            if retry_raw:
                raw = retry_raw

    summary = {
        "market_open": market_open,
        "signal_count": len(merged),
        "auto_exits": auto_exits,
        "decision": decision,
        "raw": raw,
        "snapshot": snap,
        "status": "NO_DECISION",
        "detail": "",
        "retried": retried,
        "fallback_used": fallback_used,
    }

    if not decision:
        # Capture an excerpt of what Claude actually returned so we can
        # diagnose parse failures from the dashboard / DB instead of staring
        # at a generic "no parseable JSON" line.
        if raw:
            excerpt = raw[:RAW_CAPTURE_CHARS].replace("\x00", "")
            tag = "retry_failed" if retried else "parse_failed"
            reason_text = f"{tag}: {excerpt}"
        else:
            reason_text = "claude returned no response (timeout/empty)"
        store.record_decision(market_open, len(merged), "NO_DECISION",
                              reason_text,
                              snap["total_value"], snap["cash"])
        store.record_equity_point(snap["total_value"], snap["cash"], sp500)
        return summary

    status, detail = _execute(decision, snap, store)
    summary["status"] = status
    summary["detail"] = detail

    action_label = f"{decision.get('action','?')} {decision.get('ticker','')}".strip()
    if fallback_used:
        # Make it visible in the DB / dashboard that this decision came from
        # the Sonnet fallback (condensed context), not the full Opus pass.
        decision = {**decision,
                    "reasoning": f"{decision.get('reasoning', '')} [sonnet-fallback]"}
    store.record_decision(
        market_open,
        len(merged),
        f"{action_label} → {status}",
        json.dumps({"decision": decision, "auto_exits": auto_exits,
                    "detail": detail, "fallback_used": fallback_used}),
        snap["total_value"],
        snap["cash"],
    )
    # final mark + equity point
    final = _portfolio_snapshot(store)
    store.record_equity_point(final["total_value"], final["cash"], sp500)
    summary["snapshot"] = final
    return summary


if __name__ == "__main__":
    import pprint
    pprint.pp(decide())
