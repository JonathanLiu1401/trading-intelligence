"""AI catalyst-cycle monitor.

Detects the pattern Jonathan described in chat:

    fresh catalyst -> chase -> no follow-up -> profit-taking/dip -> next catalyst

The monitor is intentionally lightweight and read-only. It scans recent article
signals, scores AI/semiconductor tickers for fresh catalyst opportunity or stale
profit-taking risk, writes a JSON report, and can post only newly-throttled
events to Discord.

Standalone:
    python3 -m analytics.catalyst_cycle_monitor --dry-run
    python3 -m analytics.catalyst_cycle_monitor
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from analytics.catalyst_classifier import classify_spike
from analytics.trend_velocity import STOP, TICKER_RE, _parse_ts
from analytics.news_fatigue import _unified_score
from ml.features import _source_credibility
from storage.article_store import _LIVE_ONLY_CLAUSE, _get_db_path

OUT_PATH = Path("/home/zeph/logs/catalyst_cycle_monitor.json")
STATE_PATH = BASE / "logs" / ".catalyst_cycle_monitor_state.json"

SCAN_LIMIT = 1_000
LOOKBACK_HOURS = 24
RECENT_MINUTES = 30
PRIOR_HOURS = 6
URGENT_THROTTLE_S = 45 * 60
WATCH_THROTTLE_S = 2 * 60 * 60
MAX_SEND_PER_RUN = 3

DISCORD_CHANNEL = os.environ.get(
    "CATALYST_MONITOR_DISCORD_CHANNEL", "channel:1496099475838603324"
)
JONATHAN_DISCORD_USER_ID = os.environ.get(
    "JONATHAN_DISCORD_USER_ID", "454961974048980992"
)
SAO_DISCORD_USER_ID = os.environ.get("SAO_DISCORD_USER_ID", "702863115276124211")
OPENCLAW_CLI = os.environ.get("OPENCLAW_CLI", "")

# AI/semiconductor/watchlist names that tend to trade as narrative momentum.
# Keep this intentionally broad: ticker extraction is noisy, so this also acts
# as a guardrail against alerting on random all-caps words.
AI_TICKERS = {
    "NVDA", "AMD", "AVGO", "MU", "LITE", "AXTI", "ADBE", "MSFT", "ORCL",
    "ARM", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "SMCI", "MRVL", "QCOM",
    "INTC", "SNPS", "CDNS", "CRDO", "ANET", "DELL", "HPE", "VRT", "ETN",
    "GRID", "DRAM", "SNDU", "QBTS", "TSEM",
}

AI_CONTEXT_RE = re.compile(
    r"\b(ai|artificial intelligence|hbm|dram|gpu|accelerator|data center|"
    r"datacenter|semiconductor|chip|memory|optical|silicon|computex|gtc|"
    r"blackwell|cuda|inference|local llm|llm|server|superchip)\b",
    re.IGNORECASE,
)

# Fresh timestamps alone are not enough. These are usually recaps, analyst
# churn, previews, or post-earnings commentary where the market already knows
# the story and a repost should not page people.
KNOWN_CONSENSUS_RE = re.compile(
    r"\b("
    r"price\s*target|maintained\s+by|raised\s+to|analysts?\s+have\s+lifted|"
    r"upgrad(?:e|ed|es|ing)|downgrad(?:e|ed|es|ing)|"
    r"set\s+to\s+report|to\s+report\s+.*earnings|earnings\s+preview|"
    r"ahead\s+of\s+earnings|q[1-4]\s+earnings|on\s+.*earnings|"
    r"after\s+earnings|post.?earnings|earnings\s+call|"
    r"ceo\s+on|nobody\s+wants\s+to\s+be\s+left\s+behind|"
    r"trading\s+\d+(?:\.\d+)?%\s+higher|trading\s+higher|"
    r"stock\s+moves|why\s+.*stock|profit\s+from\s+others'? speculation|"
    r"intrinsic\s+value|stick\s+to\s+intrinsic\s+value|"
    r"s&p\s*500|nasdaq|dow\s+futures?|record\s+close|"
    r"futures?\s+(ease|rise|fall)|markets?\s+in\s+focus|in\s+focus|"
    r"microsoft\s+build|\bmsn\b"
    r")\b",
    re.IGNORECASE,
)

LOW_QUALITY_TRADE_CHATTER_RE = re.compile(
    r"\b(drop\s+to|crash|shorts?|puts?|bearish|sell|support|resistance|"
    r"checked\s+the\s+shorts?|checkmated)\b",
    re.IGNORECASE,
)

URGENT_CATALYSTS = {
    "EARNINGS",
    "EARNINGS_PRE",
    "PRODUCT",
    "ANALYST",
    "M&A",
    "REGULATORY",
    "GOVERNMENT",
    "MACRO",
    "SHORT_SQUEEZE",
}

CATALYST_WEIGHT = {
    "EARNINGS": 2.0,
    "EARNINGS_PRE": 1.6,
    "PRODUCT": 1.4,
    "ANALYST": 1.1,
    "M&A": 1.8,
    "REGULATORY": 1.4,
    "MACRO": 1.0,
    "GOVERNMENT": 1.2,
    "SHORT_SQUEEZE": 1.0,
    "TECHNICAL": 0.4,
    "UNKNOWN": 0.0,
}


def _openclaw_cli_path() -> str | None:
    if OPENCLAW_CLI:
        return OPENCLAW_CLI
    found = shutil.which("openclaw")
    if found:
        return found
    fallback = "/home/zeph/.nvm/versions/node/v24.15.0/bin/openclaw"
    return fallback if os.path.exists(fallback) else None


def _extract_tickers(title: str | None) -> set[str]:
    if not title:
        return set()
    return {
        m for m in TICKER_RE.findall(title)
        if m not in STOP and len(m) >= 2 and (m in AI_TICKERS or f"${m}" in title)
    }


def _is_ai_context(title: str | None) -> bool:
    return bool(AI_CONTEXT_RE.search(title or ""))


def _is_known_consensus_coverage(title: str | None) -> bool:
    return bool(KNOWN_CONSENSUS_RE.search(title or ""))


def _is_low_quality_trade_chatter(title: str | None) -> bool:
    return bool(LOW_QUALITY_TRADE_CHATTER_RE.search(title or ""))


def _source_family(source: str | None) -> str:
    """Canonical source family for catalyst corroboration.

    Aggregator labels such as ``yfinance/Stocktwits`` and
    ``GoogleNews/Stocktwits`` are still Stocktwits chatter for alert-quality
    purposes. Counting those wrappers as independent sources made a one-feed
    forum thread look like a corroborated two-source catalyst.
    """
    s = (source or "").strip().lower()
    if not s:
        return ""
    if "stocktwits" in s:
        return "stocktwits"
    if s.startswith("yahoofinance/") or s.startswith("yf/"):
        return "yahoo"
    if s.startswith("gn:"):
        return "googlenews"
    return s.split("/", 1)[0].split(":", 1)[0].strip()


def _catalyst_source_credibility(source: str | None) -> float:
    """Source credibility for catalyst alerts.

    ``ml.features._source_credibility`` intentionally treats
    ``yfinance/...`` / ``GN: ...`` as the aggregator's publisher grade for ML
    features. For paging, a Stocktwits item republished through those wrappers
    is still low-authority Stocktwits content, so override that family first.
    """
    if _source_family(source) == "stocktwits":
        return 0.30
    return _source_credibility(source or "")


def _score_value(ai_score, ml_score) -> float | None:
    return _unified_score(ai_score, ml_score)


def build_cycle_events(
    rows: Iterable[tuple],
    now: datetime | None = None,
    watch_tickers: set[str] | None = None,
) -> dict:
    """Pure catalyst-cycle scorer.

    ``rows`` shape: ``(first_seen, title, source, url, ai_score, ml_score)``.
    Returns a report containing ticker-level events, each with ``level``:
    ``urgent`` for fresh high-confidence catalyst, ``watch`` for active but less
    proven catalyst or stale/profit-taking risk.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    watch_tickers = watch_tickers or AI_TICKERS
    recent_cut = now - timedelta(minutes=RECENT_MINUTES)
    prior_cut = now - timedelta(hours=PRIOR_HOURS)
    lookback_cut = now - timedelta(hours=LOOKBACK_HOURS)

    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for first_seen, title, source, url, ai_score, ml_score in rows:
        ts = _parse_ts(first_seen)
        if ts is None or ts < lookback_cut:
            continue
        score = _score_value(ai_score, ml_score)
        tickers = _extract_tickers(title)
        if not tickers and _is_ai_context(title):
            # No explicit ticker means it is useful context, but not alertable.
            continue
        for ticker in tickers:
            if ticker not in watch_tickers:
                continue
            by_ticker[ticker].append({
                "ts": ts,
                "title": title or "",
                "source": source or "",
                "url": url or "",
                "score": score,
                "ai_context": _is_ai_context(title),
            })

    events: list[dict] = []
    for ticker, items in by_ticker.items():
        items.sort(key=lambda r: r["ts"], reverse=True)
        recent = [r for r in items if r["ts"] >= recent_cut]
        prior = [r for r in items if prior_cut <= r["ts"] < recent_cut]
        scored_recent = [float(r["score"]) for r in recent if r["score"] is not None]
        scored_prior = [float(r["score"]) for r in prior if r["score"] is not None]
        latest = items[0]
        latest_age_min = (now - latest["ts"]).total_seconds() / 60.0
        recent_titles = [r["title"] for r in recent] or [latest["title"]]
        catalyst, catalyst_conf = classify_spike(ticker, recent_titles)
        max_recent_score = max(scored_recent) if scored_recent else 0.0
        avg_recent = mean(scored_recent) if scored_recent else 0.0
        avg_prior = mean(scored_prior) if scored_prior else None
        score_drop = (avg_prior - avg_recent) if avg_prior is not None else 0.0
        ai_context_count = sum(1 for r in recent if r["ai_context"])
        all_ai_context_count = sum(1 for r in items if r["ai_context"])
        recent_source_families = {
            _source_family(r["source"]) for r in recent
            if _source_family(r["source"])
        }
        source_count = len(recent_source_families)
        non_stocktwits_source_count = sum(
            1 for f in recent_source_families if f != "stocktwits"
        )
        max_source_cred = max(
            (_catalyst_source_credibility(r["source"]) for r in recent if r["source"]),
            default=0.0,
        )
        all_source_families = {
            _source_family(r["source"]) for r in items
            if _source_family(r["source"])
        }
        all_source_count = len(all_source_families)
        non_stocktwits_all_source_count = sum(
            1 for f in all_source_families if f != "stocktwits"
        )
        max_any_source_cred = max(
            (_catalyst_source_credibility(r["source"]) for r in items if r["source"]),
            default=0.0,
        )
        catalyst_strength = CATALYST_WEIGHT.get(catalyst, 0.0) * catalyst_conf
        source_ok = non_stocktwits_source_count >= 2 or max_source_cred >= 0.6
        any_source_ok = (
            non_stocktwits_all_source_count >= 2 or max_any_source_cred >= 0.6
        )
        consensus_latest = _is_known_consensus_coverage(latest["title"])
        noisy_trade_chatter = _is_low_quality_trade_chatter(latest["title"])

        fresh_score = (
            max_recent_score
            + min(len(recent), 5) * 0.35
            + min(source_count, 3) * 0.25
            + catalyst_strength
            + min(ai_context_count, 3) * 0.2
        )

        if (
            recent and latest_age_min <= RECENT_MINUTES
            and fresh_score >= 9.0 and source_ok and catalyst in URGENT_CATALYSTS
            and catalyst_conf >= 0.5 and not consensus_latest and not noisy_trade_chatter
        ):
            level = "urgent"
            kind = "fresh_ai_catalyst"
            reason = "fresh high-score catalyst"
        elif (
            recent and latest_age_min <= RECENT_MINUTES
            and fresh_score >= 7.5 and source_ok and catalyst in URGENT_CATALYSTS
            and catalyst_conf >= 0.5 and not consensus_latest and not noisy_trade_chatter
        ):
            level = "watch"
            kind = "fresh_ai_catalyst_watch"
            reason = "fresh catalyst forming"
        elif (
            len(items) >= 8 and not recent and latest_age_min >= 90
            and any_source_ok and all_ai_context_count >= 2
            and avg_prior is not None and avg_prior >= 7.5
        ):
            level = "watch"
            kind = "stale_catalyst_profit_taking_risk"
            reason = "older AI coverage with no fresh follow-up; profit-taking/dip risk"
        elif len(recent) >= 2 and score_drop >= 1.5 and source_ok and ai_context_count >= 1:
            level = "watch"
            kind = "catalyst_fatigue"
            reason = "coverage still active but recent signal quality is fading"
        else:
            continue

        events.append({
            "ticker": ticker,
            "level": level,
            "kind": kind,
            "reason": reason,
            "fresh_score": round(fresh_score, 2),
            "latest_age_min": round(latest_age_min, 1),
            "recent_mentions_30m": len(recent),
            "mentions_24h": len(items),
            "source_count_30m": source_count,
            "max_source_cred": round(max_source_cred, 2),
            "max_recent_score": round(max_recent_score, 2),
            "avg_recent_score": round(avg_recent, 2) if scored_recent else None,
            "avg_prior_score": round(avg_prior, 2) if avg_prior is not None else None,
            "score_drop": round(score_drop, 2),
            "catalyst": catalyst,
            "catalyst_confidence": catalyst_conf,
            "title": latest["title"],
            "source": latest["source"],
            "url": latest["url"],
        })

    events.sort(key=lambda e: (e["level"] != "urgent", -e["fresh_score"], e["latest_age_min"]))
    return {
        "generated_at": now.isoformat(),
        "lookback_hours": LOOKBACK_HOURS,
        "recent_minutes": RECENT_MINUTES,
        "events": events,
    }


def _event_key(event: dict) -> str:
    return f"{event.get('kind')}:{event.get('ticker')}"


def _shorten_text(value: str | None, limit: int = 170) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def select_new_events(
    events: list[dict],
    throttle_state: dict | None,
    now_epoch: float | None = None,
    max_send: int = MAX_SEND_PER_RUN,
    min_level: str = "urgent",
) -> tuple[list[dict], dict]:
    """Throttle event sends by ticker/kind."""
    now_epoch = time.time() if now_epoch is None else now_epoch
    old = dict(throttle_state or {})
    new_state: dict[str, float] = {
        k: v for k, v in old.items()
        if isinstance(v, (int, float)) and now_epoch - float(v) < 24 * 3600
    }
    selected: list[dict] = []
    for event in events:
        if min_level == "urgent" and event.get("level") != "urgent":
            continue
        key = _event_key(event)
        throttle_s = URGENT_THROTTLE_S if event.get("level") == "urgent" else WATCH_THROTTLE_S
        last_raw = old.get(key)
        last = float(last_raw) if isinstance(last_raw, (int, float)) else None
        if last is not None and now_epoch - last < throttle_s:
            continue
        selected.append(event)
        new_state[key] = now_epoch
        if len(selected) >= max_send:
            break
    return selected, new_state


def format_event(event: dict) -> str:
    ping = ""
    if event.get("level") == "urgent":
        ping = f"<@{JONATHAN_DISCORD_USER_ID}> <@{SAO_DISCORD_USER_ID}> "
    label = "URGENT" if event.get("level") == "urgent" else "WATCH"
    kind = event.get("kind") or ""
    if event.get("level") == "urgent":
        action = "BUY WATCH"
    elif "profit_taking" in kind or "fatigue" in kind:
        action = "SELL RISK"
    else:
        action = "HOLD"
    url = event.get("url") or ""
    title = _shorten_text(event.get("title") or "(untitled)")
    source = event.get("source") or "unknown"
    score = event.get("fresh_score")
    age = event.get("latest_age_min")
    mentions = event.get("recent_mentions_30m")
    sources = event.get("source_count_30m")
    catalyst = event.get("catalyst")
    catalyst_conf = event.get("catalyst_confidence", 0)
    lines = [
        f"{ping}**${event['ticker']} - {action}**  [{label}]",
        f"Stats: score {score} | {age}m old | {mentions} mentions, {sources} sources | {catalyst} {catalyst_conf:.0%}",
        f"Source: {source}",
        f"Latest: \"{title}\"",
    ]
    if event.get("score_drop", 0) > 0:
        lines.append(f"Risk: signal quality dropped {event['score_drop']} vs prior window")
    if url:
        lines.append(f"<{url}>")
    return "\n".join(lines)[:1900]


def send_discord(message: str) -> bool:
    cli = _openclaw_cli_path()
    if not cli:
        print("[catalyst_cycle_monitor] openclaw CLI not found; cannot send")
        return False
    try:
        proc = subprocess.run(
            [
                cli, "message", "send",
                "--channel", "discord",
                "--target", DISCORD_CHANNEL,
                "--message", message,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        print(f"[catalyst_cycle_monitor] discord send failed: {exc}")
        return False
    if proc.returncode == 0:
        return True
    err = (proc.stderr or proc.stdout or "").strip()
    print(f"[catalyst_cycle_monitor] discord send rc={proc.returncode}: {err[:300]}")
    return False


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def _write_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def run_once(dry_run: bool = False, min_level: str = "urgent") -> dict:
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    rows = conn.execute(
        "SELECT first_seen, title, source, url, ai_score, ml_score "
        "FROM articles "
        f"WHERE {_LIVE_ONLY_CLAUSE} "
        "ORDER BY first_seen DESC LIMIT ?",
        (SCAN_LIMIT,),
    ).fetchall()
    conn.close()

    report = build_cycle_events(rows)
    events = report["events"]
    if min_level == "urgent":
        events = [e for e in events if e.get("level") == "urgent"]
    state = _load_state()
    selected, new_state = select_new_events(events, state, min_level=min_level)
    report["selected_for_send"] = selected
    report["dry_run"] = dry_run
    report["scanned_rows"] = len(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))

    for event in selected:
        msg = format_event(event)
        if dry_run:
            print(f"[dry-run] would send:\n{msg}\n")
        else:
            send_discord(msg)

    if not dry_run:
        _write_state(new_state)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="do not post to Discord or update throttle state")
    ap.add_argument(
        "--min-level",
        choices=("watch", "urgent"),
        default=os.environ.get("CATALYST_MONITOR_MIN_LEVEL", "urgent"),
        help="minimum level to send; report JSON still includes all events",
    )
    args = ap.parse_args(argv)
    report = run_once(dry_run=args.dry_run, min_level=args.min_level)
    urgent = sum(1 for e in report["events"] if e["level"] == "urgent")
    watch = sum(1 for e in report["events"] if e["level"] == "watch")
    sent = len(report["selected_for_send"])
    print(
        f"catalyst_cycle_monitor: scanned={report['scanned_rows']} "
        f"urgent={urgent} watch={watch} selected={sent} -> {OUT_PATH}"
    )
    for e in report["events"][:5]:
        print(
            f"  {e['level'].upper()} ${e['ticker']} {e['kind']} "
            f"score={e['fresh_score']} age={e['latest_age_min']}m :: {e['title'][:90]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
