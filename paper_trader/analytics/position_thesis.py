"""Position Thesis Cards — one integrated view per open holding.

For every open stock position we surface a single dict combining:
  - days held + unrealized P/L
  - the strongest news the model has seen on the ticker in the last 24h
  - bull/bear article counts (split by ai_score >= 5.0 vs < 5.0)
  - DecisionScorer's predicted 5-day return
  - live quant snapshot (RSI, MACD, mom_5d, mom_20d)
  - latest decision the bot made that involved this ticker (action + Opus confidence)
  - a coarse verdict pill (STRONG_HOLD / HOLD / WATCH / TRIM / EXIT) and a one-line thesis

The verdict is intentionally simple and explainable — it's not a fresh prediction,
just a structured roll-up of inputs the trader already glances at across five
different panels. Saves a few seconds per position and surfaces stale theses.
"""
from __future__ import annotations

import json
import re
import sqlite3
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DI_DB = Path("/home/zeph/digital-intern/data/articles.db")
USB_DB = Path("/media/zeph/projects/digital-intern/db/articles.db")


def _db_path() -> Path:
    return USB_DB if USB_DB.exists() else DI_DB


def _connect_ro() -> sqlite3.Connection | None:
    p = _db_path()
    if not p.exists():
        return None
    try:
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10)
        c.row_factory = sqlite3.Row
        return c
    except Exception:
        return None


def _decompress(blob: bytes | None) -> str:
    if not blob:
        return ""
    try:
        return zlib.decompress(blob).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _days_held(opened_at: str | None) -> float:
    if not opened_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
        return round((datetime.now(timezone.utc) - dt).total_seconds() / 86400.0, 2)
    except Exception:
        return 0.0


def _ticker_news(ticker: str, hours: int = 24, limit: int = 3) -> dict:
    """Top headlines + bull/bear split for a ticker over the last N hours."""
    out = {"headlines": [], "bull": 0, "bear": 0, "n": 0, "avg_score": 0.0, "max_score": 0.0}
    conn = _connect_ro()
    if conn is None:
        return out
    since = (datetime.now(timezone.utc).timestamp() - hours * 3600)
    since_iso = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()
    pat = re.compile(rf"(?:\$|\b){re.escape(ticker.upper())}\b")
    try:
        rows = conn.execute(
            "SELECT title, source, ai_score, urgency, first_seen, full_text "
            "FROM articles WHERE first_seen >= ? AND ai_score > 0 "
            "AND url NOT LIKE 'backtest://%' AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%' "
            "ORDER BY ai_score DESC LIMIT 600",
            (since_iso,),
        ).fetchall()
    finally:
        conn.close()
    matched = []
    scores = []
    for r in rows:
        body = f"{r['title']} {_decompress(r['full_text'])}".upper()
        if pat.search(body):
            matched.append(r)
            sc = float(r["ai_score"] or 0.0)
            scores.append(sc)
            if sc >= 5.0:
                out["bull"] += 1
            else:
                out["bear"] += 1
    out["n"] = len(matched)
    if scores:
        out["avg_score"] = round(sum(scores) / len(scores), 2)
        out["max_score"] = round(max(scores), 2)
    for r in matched[:limit]:
        out["headlines"].append({
            "title": (r["title"] or "")[:180],
            "source": r["source"] or "",
            "score": float(r["ai_score"] or 0.0),
            "urgency": int(r["urgency"] or 0),
            "first_seen": r["first_seen"],
        })
    return out


def _latest_decision(decisions: list[dict], ticker: str) -> dict | None:
    """Find the most recent decision row that touched this ticker."""
    tk = ticker.upper()
    for d in decisions:
        action = (d.get("action_taken") or "").upper()
        # action looks like "BUY MU → FILLED" or "HOLD NONE → HOLD"
        if f" {tk} " in f" {action} ":
            conf = None
            try:
                blob = json.loads(d.get("reasoning") or "{}")
                inner = blob.get("decision") or {}
                conf = inner.get("confidence")
                reason = inner.get("reasoning") or ""
            except Exception:
                reason = (d.get("reasoning") or "")[:200]
            return {
                "timestamp": d.get("timestamp"),
                "action": action,
                "confidence": conf,
                "reasoning": (reason or "")[:280],
            }
    return None


def _verdict(scorer_pred: float, rsi: float | None, mom_5d: float | None,
             news_avg: float, news_n: int, has_bear_majority: bool) -> tuple[str, str]:
    """Combine inputs into a coarse verdict + one-line thesis."""
    reasons = []
    score = 0.0
    if scorer_pred is not None:
        score += float(scorer_pred)
        if scorer_pred >= 2.0:
            reasons.append(f"scorer +{scorer_pred:.1f}%")
        elif scorer_pred <= -2.0:
            reasons.append(f"scorer {scorer_pred:.1f}%")
    if mom_5d is not None:
        score += float(mom_5d) * 0.4
        if abs(mom_5d) > 2.0:
            reasons.append(f"mom_5d {mom_5d:+.1f}%")
    if rsi is not None:
        if rsi >= 75:
            score -= 1.5
            reasons.append(f"RSI {rsi:.0f} overbought")
        elif rsi <= 30:
            score += 1.0
            reasons.append(f"RSI {rsi:.0f} oversold")
    if news_n >= 2:
        if news_avg >= 6.0:
            score += 1.5
            reasons.append(f"news avg {news_avg:.1f}")
        elif has_bear_majority and news_avg < 4.0:
            score -= 1.0
            reasons.append(f"news avg {news_avg:.1f} (bearish)")
    if news_n == 0:
        reasons.append("no news 24h")

    if score >= 3.5:
        verdict = "STRONG_HOLD"
    elif score >= 1.0:
        verdict = "HOLD"
    elif score >= -1.0:
        verdict = "WATCH"
    elif score >= -3.0:
        verdict = "TRIM"
    else:
        verdict = "EXIT"
    thesis = "; ".join(reasons) if reasons else "no strong signals"
    return verdict, thesis


def build_thesis_cards(
    open_positions: list[dict],
    decisions: list[dict],
    scorer_preds: list[dict],
    quant: dict,
) -> dict:
    """Build the thesis card list from upstream data.

    Inputs are pulled by the caller (the dashboard) so this module stays free
    of yfinance / claude / DB-import side effects when used from tests."""
    preds_by_tk = {p["ticker"]: p for p in (scorer_preds or [])}
    cards = []
    for pos in open_positions:
        if pos.get("type") != "stock":
            continue
        tk = (pos.get("ticker") or "").upper()
        if not tk:
            continue
        days = _days_held(pos.get("opened_at"))
        upl = float(pos.get("unrealized_pl") or 0.0)
        cost = float(pos.get("avg_cost") or 0.0) * float(pos.get("qty") or 0.0)
        pl_pct = round(upl / cost * 100, 2) if cost > 0 else 0.0

        news = _ticker_news(tk)
        pred_row = preds_by_tk.get(tk) or {}
        q = quant.get(tk) or {}

        scorer_pred = pred_row.get("pred_5d_return_pct")
        verdict, thesis = _verdict(
            scorer_pred if isinstance(scorer_pred, (int, float)) else 0.0,
            q.get("RSI"),
            q.get("mom_5d"),
            news["avg_score"],
            news["n"],
            news["bear"] > news["bull"],
        )

        cards.append({
            "ticker": tk,
            "qty": pos.get("qty"),
            "avg_cost": round(float(pos.get("avg_cost") or 0.0), 2),
            "current_price": round(float(pos.get("current_price") or 0.0), 2),
            "unrealized_pl": round(upl, 2),
            "pl_pct": pl_pct,
            "days_held": days,
            "verdict": verdict,
            "thesis": thesis,
            "scorer_pred_5d": scorer_pred,
            # Honesty flag — True ⇒ the scorer extrapolated past its
            # empirical label support and ``scorer_pred_5d`` is a clamped
            # ±50 floor/ceiling, not a confident point estimate. The unified
            # conviction board decays its ML axis off this flag.
            "off_distribution": bool(pred_row.get("off_distribution", False)),
            "raw_pred_5d_return_pct": pred_row.get("raw_pred_5d_return_pct"),
            "scorer_verdict": pred_row.get("verdict"),
            "rsi": q.get("RSI"),
            "macd": q.get("MACD"),
            "mom_5d": q.get("mom_5d"),
            "mom_20d": q.get("mom_20d"),
            "vol_ratio": q.get("vol_ratio"),
            "news": news,
            "last_decision": _latest_decision(decisions, tk),
        })
    # Sort: EXIT/TRIM first (action items at top), then by abs P/L
    order = {"EXIT": 0, "TRIM": 1, "WATCH": 2, "HOLD": 3, "STRONG_HOLD": 4}
    cards.sort(key=lambda c: (order.get(c["verdict"], 9), -abs(c["unrealized_pl"])))
    return {
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_positions": len(cards),
        "cards": cards,
    }
