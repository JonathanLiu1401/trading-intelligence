"""
Urgent alert agent — Bloomberg BN newswire style, immediate Discord post.
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from core.claude_cli import claude_call
from watchers.alert_dedup import alerted_ids, dedupe_urgent

try:
    from core.logger import get_logger
    _log = get_logger("alert_agent")
except Exception:
    _log = logging.getLogger("alert_agent")

SONNET_MODEL = "claude-sonnet-4-6"
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")

ALERT_PROMPT = """You are a Bloomberg BN terminal newswire alert system. A high-urgency financial event has been detected.

Write a Discord alert in Bloomberg newswire style — dense, exact, no filler. Max 1800 chars.

Current UTC time (use this verbatim in the timestamp slot — do NOT guess): {now_utc}

FORMAT (use exactly):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 BREAKING  ◈  [CATEGORY]  ◈  {now_utc} UTC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[ONE LINE HEADLINE IN CAPS — what happened]

TICKERS:   [affected symbols]
IMPACT:    [BUY/SELL/WATCH] — [one sentence on direction]
CONTEXT:   [one sentence of background]
PORTFOLIO: [specific implication for LITE/MU/MSFT/AXTI/ORCL/TSEM/QBTS]
SOURCE:    [source name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
Then on a new line after the code block: [article url]

Categories: EARNINGS | RATING CHANGE | MACRO SHOCK | SUPPLY CHAIN | REGULATORY | FED | CRYPTO | M&A | GEOPOLITICAL

Urgent articles detected:
{articles_text}

Output ONLY the alert message."""


ALERT_BATCH_SIZE = 5


def _article_age_ok(art: dict) -> bool:
    """Return True if the article is less than 24 hours old."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    for field in ("published", "first_seen"):
        raw = (art.get(field) or "").strip()
        if not raw:
            continue
        try:
            # Try RFC 2822 (RSS/Atom)
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt >= cutoff
        except Exception:
            pass
        try:
            # Try ISO 8601
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt >= cutoff
        except Exception:
            pass
    # No parseable date in either field — block rather than risk stale alert.
    # Articles without any date were already pre-filtered by first_seen >= 24h
    # in get_unalerted_urgent, so reaching here means both fields are corrupt.
    _log.warning("[alert] article has no parseable date — dropping to be safe")
    return False


def _is_synthetic(art: dict) -> bool:
    """True for backtest/opus-annotation rows that must never reach the live
    Bloomberg formatter. Mirrors storage.article_store._LIVE_ONLY_CLAUSE.

    The store's get_unalerted_urgent already excludes synthetic rows, but the
    invariant is load-bearing enough that we re-check at the formatter — a
    future caller that bypasses the store filter (e.g., a manual replay) must
    not leak training rows into Discord."""
    url = (art.get("link") or art.get("url") or "")
    source = (art.get("source") or "")
    if url.startswith("backtest://"):
        return True
    if source.startswith("backtest_") or source.startswith("opus_annotation"):
        return True
    return False


def send_urgent_alert(urgent_articles: list, store) -> bool:
    if not urgent_articles:
        return False
    if not DISCORD_WEBHOOK:
        _log.warning("[alert] No DISCORD_WEBHOOK_URL — skipping")
        return False

    # Defense-in-depth: synthetic backtest/opus-annotation rows must never
    # reach the live alert formatter. The store filter is the primary defense;
    # this is a second line.
    filtered = [a for a in urgent_articles if not _is_synthetic(a)]
    n_dropped = len(urgent_articles) - len(filtered)
    if n_dropped:
        _log.warning(
            f"[alert] dropped {n_dropped} synthetic rows leaked from upstream"
        )
    if not filtered:
        return False

    # Drop articles older than 24 hours — stale news must not fire as breaking.
    fresh = [a for a in filtered if _article_age_ok(a)]
    n_stale = len(filtered) - len(fresh)
    if n_stale:
        _log.info(f"[alert] dropped {n_stale} stale article(s) (>24h old)")
    if not fresh:
        _log.info("[alert] all urgent articles are stale — skipping alert")
        return False
    filtered = fresh

    # Collapse syndicated duplicates first: one breaking story carried by GDELT
    # + Reuters + Yahoo + RSS would otherwise eat the whole 5-slot batch and
    # show the trader the same event five times. After dedup the batch holds
    # five DISTINCT stories; each survivor knows the ids of the copies it
    # absorbed (``_dup_ids``) so all of them can still be marked alerted.
    deduped = dedupe_urgent(filtered)

    # Only the first ALERT_BATCH_SIZE feed the prompt — and only those (plus the
    # duplicates they absorbed) get marked alerted. Marking the entire urgent
    # list would silently drop the tail (it'd never be picked up next cycle),
    # so we cap both ends.
    batch = deduped[:ALERT_BATCH_SIZE]

    def _fmt(a: dict) -> str | None:
        # Defensive field access. The rest of this pipeline (_is_synthetic,
        # dedupe_urgent) reads every key through .get(); _fmt used to be the
        # one place with hard subscripts (a['link'], a['ai_score'], ...). A
        # single dict from a non-canonical caller (manual replay, or a row
        # carrying `url` instead of `link` — the exact alias _is_synthetic
        # already tolerates) raised KeyError, the broad except below swallowed
        # it, the WHOLE batch was dropped, nothing was marked alerted, and
        # urgent alerts silently failed every cycle. Skip one bad row instead
        # of unwinding the batch; only the headline is truly required.
        title = (a.get("title") or "").strip()
        if not title:
            _log.warning("[alert] skipping urgent row with no title (id=%s)",
                         a.get("_id"))
            return None
        try:
            score = float(a.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        link = a.get("link") or a.get("url") or ""
        source = (a.get("source") or "unknown").strip() or "unknown"
        block = (
            f"[score={score:.0f}] {title}\n"
            f"source: {source}\nurl: {link}"
        )
        dup_count = int(a.get("dup_count") or 1)
        if dup_count > 1:
            # Tell the alert LLM how broadly the story is being carried — wide
            # syndication is itself a signal of how big the event is.
            block += f"\nsyndication: reported by {dup_count} sources"
        summary = (a.get("summary") or "").strip()
        if summary:
            block += f"\nbody: {summary[:600]}"
        return block

    # Filter the batch to formattable rows BEFORE building the prompt AND
    # before alerted_ids(batch) — marking a skipped row alerted would silently
    # drop it forever; keeping it in batch would re-fire the whole cycle.
    formatted = [(a, _fmt(a)) for a in batch]
    batch = [a for a, t in formatted if t is not None]
    if not batch:
        _log.warning("[alert] no formattable urgent rows in batch — skipping")
        return False
    articles_text = "\n\n".join(t for _, t in formatted if t is not None)

    # Full date+time so Discord history is unambiguous across day boundaries.
    # Template already appends " UTC", so don't include it here.
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    prompt = ALERT_PROMPT.format(articles_text=articles_text, now_utc=now_utc)

    try:
        message = claude_call(prompt, model=SONNET_MODEL, timeout=60)
        if not message:
            _log.warning("[alert] No response from Claude — skipping")
            return False

        # post via discord_notifier which also fires TTS
        from notifier.discord_notifier import send as discord_send
        ok = discord_send(message, is_alert=True)

        if ok:
            # Bulk-mark in one transaction; previous code took the write lock
            # N times (5 round-trips for the default batch size). alerted_ids
            # includes the syndicated copies merged into the batch, so they
            # never re-fire — duplicates of still-queued stories stay urgent.
            mark_ids = alerted_ids(batch)
            store.mark_alerted_batch(mark_ids)
            collapsed = len(mark_ids) - len(batch)
            tail = len(deduped) - len(batch)
            notes = []
            if collapsed > 0:
                notes.append(f"{collapsed} syndicated dupes folded in")
            if tail > 0:
                notes.append(f"{tail} more queued")
            note = f" ({'; '.join(notes)})" if notes else ""
            _log.info(f"[alert] BN alert sent ({len(batch)} distinct stories){note}")
        else:
            _log.warning("[alert] Discord POST failed")
        return ok

    except Exception:
        _log.exception("[alert] Error sending urgent alert")
        return False
