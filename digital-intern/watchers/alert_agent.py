"""
Urgent alert agent — Bloomberg BN newswire style, immediate Discord post.
"""
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from core.claude_cli import DEFAULT_LLM_MODEL, claude_call
# Reuse the *well-tested* word-boundary source-credibility lookup (pins the
# "ap matched inside snap" class of bug — tests/test_features.py) rather than
# duplicating the 40-entry SOURCE_CRED map here, which would silently drift
# (the recurring dashboard-parity / vendored-signals.py failure class).
from ml.features import _source_credibility, LIVE_PORTFOLIO_TICKERS, _LIVE_RE
from watchers import alert_recency
from watchers.alert_dedup import alerted_ids, dedupe_urgent

try:
    from core.logger import get_logger
    _log = get_logger("alert_agent")
except Exception:
    _log = logging.getLogger("alert_agent")

SONNET_MODEL = DEFAULT_LLM_MODEL
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
JONATHAN_DISCORD_USER_ID = os.environ.get(
    "JONATHAN_DISCORD_USER_ID", "454961974048980992"
)
SAO_DISCORD_USER_ID = os.environ.get("SAO_DISCORD_USER_ID", "702863115276124211")
SAO_BREAKING_DM_ENABLED = (
    os.environ.get("SAO_BREAKING_DM_ENABLED", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)
# Prefix every BREAKING channel alert with Discord user pings so urgent items
# actually notify Jonathan (and Sao) instead of sitting silently in #claw.
BREAKING_PING_ENABLED = (
    os.environ.get("BREAKING_PING_ENABLED", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)
OPENCLAW_CLI = os.environ.get("OPENCLAW_CLI", "")

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
PORTFOLIO: [specific implication for any of the analyst's held positions: {held_book}]
SOURCE:    [source name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
Then on a new line after the code block: [article url]

Categories: EARNINGS | RATING CHANGE | MACRO SHOCK | SUPPLY CHAIN | REGULATORY | FED | CRYPTO | M&A | GEOPOLITICAL

RECENCY: Each article below carries `age` = elapsed time since publication. Reflect it honestly — an item several hours old is a developing/continued story, NOT one that "just" broke; never imply a multi-hour-old item happened moments ago. {now_utc} is the alert send time, not the event time. If an item is materially old (≳3h), make that explicit in CONTEXT (e.g. "first reported ~Nh ago"). EARNINGS is stricter: only page fresh prints near release. Never frame a multi-hour-old earnings recap as a just-released print, and never write "Reported ~Nh ago" for an EARNINGS BREAKING alert — if it is already hours old it should not be in this batch.

CALIBRATION: An article tagged "[unverified — model-only urgent]" was flagged urgent by the local relevance/urgency model with NO LLM ground-truth label (raw ai_score=0; the displayed score came from ml_score alone). The model demonstrably over-scores recap/SEO/forum/wiki rows ("Why X Stock Is Trading Up Today" templates, "Here What the Street Thinks ..." mill content). This is a lower-confidence call. CONTEXT must explicitly hedge — "model flagged as urgent, no LLM relevance label" — and IMPACT must NOT state magnitude as confirmed (use "WATCH" rather than "BUY"/"SELL" unless other rows in the batch corroborate). Do NOT lead the alert HEADLINE on a lone unverified row when a non-unverified one is present in the batch.

CONTINUITY: If an article carries a `related:` line, a standalone 🚨 BREAKING alert on a related developing story ALREADY fired to this analyst within the last few hours — they have already been told the headline event. Frame THIS alert explicitly as a continuation/update of it: lead the HEADLINE with a development verb (ESCALATES / EXTENDS / WIDENS / FOLLOWS), and in CONTEXT state it follows the earlier alert (e.g. "follows ~Nh-ago alert on <prior event>"). Do NOT present it as the first time this story broke. This is what stops the analyst seeing what reads as a duplicate BREAKING for an event they are already tracking.

BOOK: If an article carries a `book:` line, it names tickers from the live portfolio/watchlist set ({held_book}). Treat `book_open:` as true open risk (positions/options with non-zero qty) and `book_watch:` as watchlist-only — never claim the analyst HOLDS a watchlist-only name. For open names, PORTFOLIO MUST give a concrete directional implication. For watchlist-only names, PORTFOLIO may note watchlist relevance but must not say "held". Absence of a `book:` line means the event does not touch the book — keep PORTFOLIO short (sector read-through only, no invented position).

BOOK VELOCITY: If a `book_velocity:` line ALSO appears on a `book:` alert, it names how many other distinct articles mentioned the same held ticker in the last 60 minutes — the wire is materially CONCENTRATING on that name (momentum / cluster of related developments), so the IMPACT magnitude should reflect that: prefer BUY/SELL over WATCH and state magnitude with more confidence than for a lone event. A `book:` line WITHOUT a `book_velocity:` line means this is the only recent mention — frame it as an isolated headline (use WATCH unless the body itself is unambiguous). Absence of `book_velocity:` is silent (never reproduced as a section).

BURST WIRE: If a `burst:` line appears, the analyst has already received N other 🚨 BREAKING alerts mentioning the named held ticker(s) in the last few hours — different headlines / different facets, but the SAME wire-concentrated event series (e.g. earnings night: revenue beat → guidance → buyback → segment colour). The analyst has seen N prior pushes for this name; this is the (N+1)th. Frame THIS alert explicitly as the next development in an active wire, NOT a fresh break: lead the HEADLINE with a development verb (DETAILS / ADDS / NOW / FOLLOWS / EXTENDS), and in CONTEXT make the burst explicit (e.g. "Nth NVDA wire today — adds to prior beats/guidance/buyback alerts"). PORTFOLIO must still name the held ticker. Do NOT under-state magnitude (the wire is genuinely active) — but DO honestly tell the analyst this is part of an ongoing series so they don't read it as a separate event needing a separate trade.

Urgent articles detected:
{articles_text}

Output ONLY the alert message."""


ALERT_BATCH_SIZE = 5

GROK_URGENCY_GATE_PROMPT = """You are the final urgency gate for a trading desk Discord pager.

Decide whether ANY article below is truly page-now urgent for a human trader who only wants rare, high-signal interruptions.

Default to urgent=false. Most political and market color is NOT urgent.

PAGE (urgent=true) only if ALL are true:
1) Material market-moving catalyst with NEW information (earnings surprise/guidance change, major M&A, regulatory/FDA action, Fed decision/path change, sudden tariff/sanctions policy change, war/geopolitical shock with immediate market impact, supply-chain break, major forced selling/buyback/capital-return event).
2) Fresh enough to act on now (not a recap, not a daily open, not multi-hour-old color).
3) Specific and actionable — not SEO/recap/forum/wiki/quote-widget noise.
4) Preferably touches the held/watchlist book, unless it is a true market-wide shock.

Do NOT page for:
- Trump speeches, campaign optics, teleprompter/staff/stage anecdotes, or routine political theater that happens often
- "CNBC Daily Open" / morning wraps / Asia open color without a hard catalyst
- "Asian markets lower/higher" style wraps
- yield quote pages / price tickers / screener tapes
- non-English syndication copies
- analyst price-target churn without a new catalyst
- "why stock is up today" recaps
- model-only SEO mill content
- duplicate facets of a story already known
- soft geopolitics / Kremlin comments / generic inflation fear without a new print or policy action

Articles:
{articles_text}

Return ONLY compact JSON, no markdown:
{{"urgent": true|false, "reason": "one short sentence", "keep_ids": ["id1","id2"], "headline": "ONE LINE CAPS HEADLINE IF URGENT ELSE EMPTY"}}
If urgent=false, keep_ids must be [].
If urgent=true, keep_ids must include only the article ids that justify paging (max 2).
"""

# Deterministic rejects before spending a Grok call. These are daily-color
# patterns that must never page even if the local model scores them 9+.
_HARD_NON_URGENT_TITLE_RE = re.compile(
    r"(?i)("
    r"daily open|morning wrap|markets? wrap|asia open|overnight wrap|"
    r"teleprompter|takes the stage|without his longtime|"
    r"why .{0,40}stock is (up|down|trading)|"
    r"stocks? making the biggest moves|"
    r"what analysts want to see|"
    r"here.?s what (happened|it means)|"
    r"most undervalued stocks to buy"
    r")"
)
_POLITICAL_COLOR_RE = re.compile(
    r"(?i)\b(trump|biden|white house|campaign|rally|speech|press conference)\b"
)
_HARD_CATALYST_RE = re.compile(
    r"(?i)("
    r"tariff|sanction|executive order|fed (cut|hike|hold|decision|minutes)|fomc|"
    r"earnings|guidance|acquire|acquisition|merger|fda|approval|recall|"
    r"bankruptcy|default|halted|halt |cyberattack|missile|invasion|ceasefire collapse"
    r")"
)


def _looks_like_hard_non_urgent(art: dict) -> str | None:
    """Return a reject reason if the row is obviously not page-worthy."""
    title = (art.get("title") or "").strip()
    summary = (art.get("summary") or "").strip()
    blob = f"{title}\n{summary}"
    if not title:
        return "empty title"
    if _HARD_NON_URGENT_TITLE_RE.search(title):
        return "daily-color / recap template"
    # Political personality/optics with no hard catalyst = everyday noise.
    if _POLITICAL_COLOR_RE.search(blob) and not _HARD_CATALYST_RE.search(blob):
        return "political color without hard catalyst"
    return None


def _article_id(art: dict):
    return art.get("_id") or art.get("id")


def _gate_articles_text(batch: list[dict]) -> str:
    lines = []
    for a in batch:
        aid = _article_id(a)
        title = (a.get("title") or "(untitled)").strip()
        source = (a.get("source") or "unknown").strip()
        link = (a.get("link") or a.get("url") or "").strip()
        score = a.get("ai_score") or a.get("ml_score") or "?"
        src_kind = a.get("score_source") or "?"
        book = ",".join(_book_tickers(a)) or "-"
        summary = (a.get("summary") or "").strip()[:280]
        lines.append(
            f"id={aid} | score={score} score_source={src_kind} | book={book} | source={source}\n"
            f"title={title}\nurl={link}\nsummary={summary}"
        )
    return "\n\n".join(lines)


def _parse_grok_urgency_gate(raw: str | None, batch: list[dict]) -> dict:
    """Parse Grok gate JSON. Fail closed (not urgent) on any ambiguity."""
    if not raw or not str(raw).strip():
        return {"urgent": False, "reason": "empty gate response", "keep": [], "headline": ""}
    text = str(raw).strip()
    # Strip markdown fences if the model ignores instructions.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        # Prefer first JSON object.
        m = re.search(r"\{[\s\S]*\}", text)
        payload = json.loads(m.group(0) if m else text)
    except Exception:
        return {"urgent": False, "reason": "unparseable gate response", "keep": [], "headline": ""}
    if not isinstance(payload, dict):
        return {"urgent": False, "reason": "gate payload not object", "keep": [], "headline": ""}
    urgent = bool(payload.get("urgent") is True)
    reason = str(payload.get("reason") or "").strip()[:240]
    headline = str(payload.get("headline") or "").strip()[:180]
    keep_ids = payload.get("keep_ids") or payload.get("ids") or []
    if not isinstance(keep_ids, list):
        keep_ids = []
    keep_ids_norm = {str(x) for x in keep_ids}
    id_map = {str(_article_id(a)): a for a in batch if _article_id(a) is not None}
    keep = [id_map[i] for i in keep_ids_norm if i in id_map]
    if urgent and not keep:
        # Fail closed if model says urgent but names no valid ids.
        return {"urgent": False, "reason": "urgent without valid keep_ids", "keep": [], "headline": ""}
    if not urgent:
        keep = []
        headline = ""
    return {"urgent": urgent, "reason": reason, "keep": keep[:2], "headline": headline}


def _grok_urgency_gate(batch: list[dict]) -> dict:
    """Hard final gate: every Discord page must be approved by Grok first."""
    # Deterministic rejects first so daily Trump/politics color never pages
    # and does not burn a model call.
    survivors = []
    rejected_reasons = []
    for a in batch:
        reason = _looks_like_hard_non_urgent(a)
        if reason:
            rejected_reasons.append(reason)
            continue
        survivors.append(a)
    if not survivors:
        why = rejected_reasons[0] if rejected_reasons else "hard non-urgent filter"
        return {"urgent": False, "reason": why, "keep": [], "headline": ""}

    prompt = GROK_URGENCY_GATE_PROMPT.format(articles_text=_gate_articles_text(survivors))
    raw = claude_call(prompt, model=SONNET_MODEL, timeout=45)
    return _parse_grok_urgency_gate(raw, survivors)


def _format_gated_breaking_message(keep: list[dict], headline: str, reason: str, now_utc: str) -> str:
    """Deterministic BN body for Grok-approved pages only."""
    lead = (headline or (keep[0].get("title") if keep else "") or "URGENT MARKET EVENT").strip().upper()
    held = _held_book_phrase()
    tickers = sorted({t for a in keep for t in _book_tickers(a)})
    ticker_line = "/".join(tickers) if tickers else "MARKET"
    source = (keep[0].get("source") if keep else "unknown") or "unknown"
    link = ((keep[0].get("link") or keep[0].get("url")) if keep else "") or ""
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🚨 BREAKING  ◈  WIRE  ◈  {now_utc} UTC",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        lead,
        "",
        f"TICKERS:   {ticker_line}",
        f"IMPACT:    WATCH — {reason or 'Grok-confirmed urgent catalyst'}",
        f"CONTEXT:   Grok urgency gate approved this page after local filters.",
        f"PORTFOLIO: book={held}",
        f"SOURCE:    {source}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    if link:
        lines.append(link)
    # Include secondary approved item titles if present.
    if len(keep) > 1:
        lines.append("")
        lines.append(f"ALSO: {(keep[1].get('title') or '')[:160]}")
        l2 = (keep[1].get('link') or keep[1].get('url') or '').strip()
        if l2:
            lines.append(l2)
    return "\n".join(lines).strip()[:1850]



def _breaking_ping_prefix() -> str:
    """Discord user mentions for urgent BREAKING channel alerts.

    Catalyst-cycle urgent events already ping Jonathan + Sao. Standalone
    BREAKING alerts historically did not, so high-urgency wires could land in
    #claw without paging anyone. Controlled by BREAKING_PING_ENABLED and the
    same JONATHAN/SAO Discord user id env vars as the catalyst monitor.
    """
    if not BREAKING_PING_ENABLED:
        return ""
    pings = []
    if JONATHAN_DISCORD_USER_ID:
        pings.append(f"<@{JONATHAN_DISCORD_USER_ID}>")
    if SAO_DISCORD_USER_ID:
        pings.append(f"<@{SAO_DISCORD_USER_ID}>")
    return (" ".join(pings) + "\n") if pings else ""


def _is_llm_vetted_urgent(art: dict) -> bool:
    """True only when an LLM (not the local model alone) labeled the row urgent.

    Model-only rows (`score_source=ml`, ai_score~0) demonstrably over-score
    recap/SEO/GDELT/quote noise. The full BN formatter can hedge those; the
    deterministic fallback cannot. Never page on model-only when LLM is down.
    """
    try:
        ai = float(art.get("ai_score") or 0.0)
    except (TypeError, ValueError):
        ai = 0.0
    src = (art.get("score_source") or "").strip().lower()
    return ai >= 8.0 or src == "llm"


def _filter_fallback_pageable(batch: list[dict]) -> tuple[list[dict], list[dict]]:
    """Strict pageable subset for LLM-outage fallback.

    Keep only rows that are:
      1) LLM-vetted urgent (not model-only),
      2) English (non-English PR/GDELT copies are noise),
      3) book-touching (held/watchlist ticker in title/summary).

    Everything else is returned as ``dropped`` so the caller can mark it
    alerted without Discord — drain the queue, do not page.
    """
    try:
        from watchers.non_english_filter import looks_non_english
    except Exception:
        def looks_non_english(_art: dict) -> bool:  # type: ignore
            return False

    pageable: list[dict] = []
    dropped: list[dict] = []
    for a in batch:
        if not _is_llm_vetted_urgent(a):
            dropped.append(a)
            continue
        if looks_non_english(a):
            dropped.append(a)
            continue
        if not _book_tickers(a):
            dropped.append(a)
            continue
        pageable.append(a)
    return pageable, dropped


def _fallback_breaking_message(batch: list[dict], now_utc: str) -> str:
    """Deterministic BREAKING alert when the LLM formatter is unavailable.

    Only used after ``_filter_fallback_pageable`` — so this is a short,
    book-touching, LLM-vetted dump, not a raw 5-story model-score spam page.
    """
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🚨 BREAKING  ◈  WIRE  ◈  {now_utc} UTC",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "LLM FORMATTER UNAVAILABLE — BOOK-RELEVANT FALLBACK",
        "",
    ]
    # Cap hard: fallback is a page, not a digest.
    for i, a in enumerate(batch[:2], 1):
        title = (a.get("title") or "(untitled)").strip()
        source = (a.get("source") or "unknown").strip()
        link = (a.get("link") or a.get("url") or "").strip()
        score = a.get("ai_score") or a.get("ml_score") or "?"
        book_hits = _book_tickers(a)
        book = f" | book: {', '.join(book_hits)}" if book_hits else ""
        lines.append(f"{i}. {title}")
        lines.append(f"   source={source} score={score}{book}")
        if link:
            lines.append(f"   {link}")
        lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    body = "\n".join(lines).strip()
    # Leave room for the ping prefix and Discord's 2000-char limit.
    return body[:1850]


def _alert_impact_side(message: str) -> str:
    """Parse IMPACT side from a composed BN alert body.

    Returns BUY / SELL / WATCH when present, else "".
    """
    text = message or ""
    m = re.search(r"(?im)^\s*IMPACT:\s*(BUY|SELL|WATCH)\b", text)
    if m:
        return m.group(1).upper()
    # Fallback for compacted / fence-stripped bodies.
    m = re.search(r"(?i)\bIMPACT:\s*(BUY|SELL|WATCH)\b", text)
    return m.group(1).upper() if m else ""


# Levered / wrapper symbols → cash underlyings for open-book paging.
# Sector watchlist alone is intentionally NOT enough to page on WATCH colour
# (PLD can sit on XLRE watch without being a desk-page event).
_OPEN_BOOK_UNDERLYING_MAP = {
    "MUU": "MU",
    "ADBG": "ADBE",
    "LNOK": "NOK",
    "DRAM": "MU",
    "SNDU": "MU",
}


def _open_book_tickers() -> set[str]:
    """Tickers that justify paging a WATCH alert: open qty only.

    Sources ``config/portfolio.json`` positions/options with non-zero qty,
    plus known levered-wrapper underlyings. Falls back to a small hard set if
    the file is missing so paging still works for core names.
    """
    open_syms: set[str] = set()
    path = Path(__file__).resolve().parent.parent / "config" / "portfolio.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            pf = json.load(f)
        for pos in pf.get("positions", []) or []:
            try:
                qty = float((pos or {}).get("qty") or 0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty == 0:
                continue
            sym = str((pos or {}).get("ticker") or "").strip().upper()
            if sym:
                open_syms.add(sym)
                und = _OPEN_BOOK_UNDERLYING_MAP.get(sym)
                if und:
                    open_syms.add(und)
        for opt in pf.get("options", []) or []:
            try:
                qty = float((opt or {}).get("qty") or 0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty == 0:
                continue
            und = str((opt or {}).get("underlying") or "").strip().upper()
            if und:
                open_syms.add(und)
            sym = str((opt or {}).get("symbol") or "").strip().upper()
            # "NOK CALL" → also keep NOK if underlying missing
            if not und and sym:
                tok = sym.split()[0]
                if tok:
                    open_syms.add(tok)
    except Exception:
        # Degrade to current open-ish core names rather than paging on the
        # entire sector watchlist.
        open_syms = {"MUU", "MU", "ADBG", "ADBE", "NOK"}
    return open_syms


def _batch_has_open_book_hit(batch: list | None) -> bool:
    """True when any approved article mentions an open held name."""
    open_syms = _open_book_tickers()
    if not open_syms:
        return False
    for art in batch or []:
        try:
            hits = set(_book_tickers(art if isinstance(art, dict) else {}))
        except Exception:
            continue
        if hits & open_syms:
            return True
        # Also scan raw text for open symbols not in LIVE_PORTFOLIO_TICKERS
        # (should be rare; LIVE set already unions portfolio.json).
        blob = f"{(art or {}).get('title') or ''} {(art or {}).get('summary') or ''}".upper()
        for sym in open_syms:
            if re.search(rf"\b{re.escape(sym)}\b", blob):
                return True
    return False


def _should_page_breaking(message: str, batch: list | None = None) -> bool:
    """Decide whether a BREAKING channel post should @-page + Sao-DM.

    Channel still gets the alert either way. Pings are reserved for:
      1) open held names (position/option qty > 0, plus levered underlyings), or
      2) actionable IMPACT sides (BUY / SELL).

    Sector-watchlist-only WATCH colour (e.g. PLD/Segro M&A while PLD is only
    on the REIT watchlist) posts quietly — no Discord mention storm, no Sao
    DM, no alert-priority TTS. News-desk "breaking" ≠ "page the desk."
    """
    if not BREAKING_PING_ENABLED:
        return False
    if _batch_has_open_book_hit(batch):
        return True
    side = _alert_impact_side(message)
    return side in {"BUY", "SELL"}


def _with_breaking_ping(message: str, *, page: bool | None = None,
                        batch: list | None = None) -> str:
    msg = (message or "").strip()
    if not msg:
        return msg
    if page is None:
        page = _should_page_breaking(msg, batch)
    if not page:
        return msg
    prefix = _breaking_ping_prefix()
    if not prefix:
        return msg
    # Avoid double-prefix if the LLM already mentioned the same user ids.
    if JONATHAN_DISCORD_USER_ID and f"<@{JONATHAN_DISCORD_USER_ID}>" in msg:
        return msg
    return f"{prefix}{msg}"


def _openclaw_cli_path() -> str | None:
    if OPENCLAW_CLI:
        return OPENCLAW_CLI
    found = shutil.which("openclaw")
    if found:
        return found
    mac_fallback = "/usr/local/bin/openclaw"
    if os.path.exists(mac_fallback):
        return mac_fallback
    fallback = "/home/zeph/.nvm/versions/node/v24.15.0/bin/openclaw"
    return fallback if os.path.exists(fallback) else None


def _send_sao_breaking_dm(message: str) -> bool:
    """Best-effort fanout of successful BREAKING alerts to Sao's Discord DM."""
    if not SAO_BREAKING_DM_ENABLED or not SAO_DISCORD_USER_ID:
        return False
    if not (message or "").strip():
        return False
    cli = _openclaw_cli_path()
    if not cli:
        _log.warning("[alert] Sao DM skipped: openclaw CLI not found")
        return False

    env = os.environ.copy()
    env["PATH"] = (
        "/usr/local/bin:/opt/homebrew/bin:"
        f"{os.path.expanduser('~')}/.local/bin:"
        f"{os.path.expanduser('~')}/.npm-global/bin:"
        + env.get("PATH", "")
    )
    try:
        proc = subprocess.run(
            [
                cli,
                "message",
                "send",
                "--channel",
                "discord",
                "--target",
                f"user:{SAO_DISCORD_USER_ID}",
                "--message",
                message,
            ],
            capture_output=True,
            text=True,
            # Match discord_notifier OpenClaw timeout: CLI plugin load is slow.
            timeout=90,
            check=False,
            env=env,
        )
    except Exception:
        _log.exception("[alert] Sao DM delivery failed")
        return False

    if proc.returncode == 0:
        _log.info("[alert] Sao DM sent for BREAKING alert")
        return True

    err = (proc.stderr or proc.stdout or "").strip()
    _log.warning("[alert] Sao DM send failed rc=%s %s", proc.returncode, err[:500])
    return False


def _held_book_phrase() -> str:
    """Slash-joined, sorted held/watched tickers for the BREAKING alert prompt's
    PORTFOLIO + BOOK slots.

    Reads ``ml.features.LIVE_PORTFOLIO_TICKERS`` (config/portfolio.json's
    positions + option underlyings + sector_watchlist, unioned with the
    hardcoded fallback) so Sonnet writes PORTFOLIO implications and resolves
    `book:` tags against the book the analyst *actually* holds, not a frozen
    literal. The PORTFOLIO/BOOK slots historically hardcoded a held set in the
    prompt body, so a position added in the trading UI was invisible to the
    Bloomberg alert formatter — its PORTFOLIO line read "no holdings affected"
    even on a fresh held name, the exact mirror of the urgency_scorer drift
    fix. ``sorted`` gives a deterministic, test-pinnable order. Degrades to a
    minimal semiconductor default only if the live set is somehow empty — an
    alert prompt must never go out with a blank held-positions slot. Same
    SSOT (LIVE_PORTFOLIO_TICKERS) as ``urgency_scorer._portfolio_ticker_line``
    so the two prompts can never disagree on what counts as held."""
    tickers = sorted(t for t in LIVE_PORTFOLIO_TICKERS if t)
    return "/".join(tickers) if tickers else "MU/NVDA/MSFT"

# Minimum number of recent (within ``alert_recency.ALERT_RECENCY_TTL_HOURS``)
# BREAKING alerts mentioning a held ticker before the alert prompt is annotated
# with a ``burst:`` line. The analyst's noise complaint is the (N+1)th alert
# about an active wire (e.g. NVDA earnings night) reading identically to the
# 1st; below the threshold the wire is a normal lone or low-velocity event
# and no annotation is added. Conservative — three prior PUSHES (not three
# articles in the DB) means the analyst has already SEEN three Discord pushes
# for this name, well above any false-positive bar.
BURST_MIN_PRIOR_ALERTS = 3

# Minimum source credibility for a LONE (un-syndicated) article to fire a
# standalone urgent Bloomberg "🚨 BREAKING" alert. Below this is the
# social/forum tier the ML urgency head demonstrably over-scores:
#   reddit 0.40 · nitter 0.40 · twitter 0.35 · stocktwits 0.30
# (per ml.features.SOURCE_CRED). DEFAULT_SOURCE_CRED is 0.55, so an unknown /
# brand-new source is NEVER gated — only the explicitly-known low tier is.
# Everything legitimate clears it: rss 0.65, scraped 0.50, gdelt 0.58,
# wikipedia 0.60, google/yahoo 0.62-0.65, reuters/bloomberg 0.90, sec 0.95.
# Corroboration is the escape valve — a story syndicated across ≥2 sources
# (dedup ``dup_count`` > 1) bypasses this entirely. See
# ``_filter_low_authority_lone``. Observed live (24h): reddit/r/Daytrading and
# reddit/r/ValueInvesting each fired a BREAKING alert solo — exactly the noise
# this gate removes from the analyst's push channel.
ALERT_MIN_LONE_SOURCE_CRED = 0.45

# Earnings prints must page near release, not hours later. A 24h general
# staleness gate is correct for most wires, but an "BREAKING" EARNINGS
# push that says "Reported ~1.6h ago" is already late for a trader. Keep the
# earnings page window tight so only fresh prints reach Discord as BREAKING;
# older earnings recaps still exist in articles.db for briefings, they just
# must not @-page as a fresh release.
ALERT_EARNINGS_MAX_AGE_HOURS = 0.5  # 30 minutes


def _filter_low_authority_lone(
    deduped: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition a *post-dedup* urgent list into ``(kept, suppressed)``.

    ``suppressed`` = a single, un-corroborated copy (``dup_count`` <= 1) from a
    known low-credibility social/forum source
    (``cred < ALERT_MIN_LONE_SOURCE_CRED``). Everything else is kept:

      - anything syndicated across ≥2 sources (``dup_count`` > 1) — independent
        corroboration is itself the signal the event is real;
      - anything from a credible *or unknown* source (≥ the threshold).

    Pure function — no DB / IO. Must run AFTER ``dedupe_urgent`` so
    ``dup_count`` reflects cross-source corroboration; running it before would
    suppress a genuinely breaking story that a low-cred feed merely happened to
    surface first."""
    kept: list[dict] = []
    suppressed: list[dict] = []
    for a in deduped:
        try:
            dup_count = int(a.get("dup_count") or 1)
        except (TypeError, ValueError):
            dup_count = 1
        if dup_count > 1:
            kept.append(a)
            continue
        cred = _source_credibility(a.get("source") or "")
        if cred < ALERT_MIN_LONE_SOURCE_CRED:
            suppressed.append(a)
        else:
            kept.append(a)
    return kept, suppressed


def _article_age_ok(art: dict) -> bool:
    """Return True if the article is fresh enough to page as BREAKING.

    Default: < 24h (general wire freshness).
    Earnings prints: < ``ALERT_EARNINGS_MAX_AGE_HOURS`` (on-release only).
    A multi-hour-old earnings recap must never fire as BREAKING — the
    trader wanted the print at release, not a late "Reported ~Nh ago" ping.
    """
    hours = _article_age_hours(art)
    if hours is None:
        # No parseable date in either field — block rather than risk stale alert.
        # Articles without any date were already pre-filtered by first_seen >= 24h
        # in get_unalerted_urgent, so reaching here means both fields are corrupt.
        _log.warning("[alert] article has no parseable date — dropping to be safe")
        return False
    max_h = (
        ALERT_EARNINGS_MAX_AGE_HOURS
        if _is_earnings_article(art)
        else 24.0
    )
    return hours <= max_h


_EARNINGS_TITLE_RE = re.compile(
    r"\b("
    r"earnings|eps\b|revenue\s+(beat|miss|results?)|"
    r"quarterly\s+results|q[1-4]\s*(?:fy)?\s*\d{2,4}|"
    r"fiscal\s+(?:q[1-4]|year)|results\s+(?:beat|miss|topped|top)|"
    r"beats?\s+(?:estimates?|expectations?|consensus)|"
    r"misses?\s+(?:estimates?|expectations?|consensus)|"
    r"guidance\s+(?:raise[ds]?|cut|lowered|raised|boosted)|"
    r"reports?\s+(?:q[1-4]|quarterly|earnings)|"
    r"posted\s+(?:q[1-4]|quarterly)|"
    r"earnings\s+(?:call|release|print|report)"
    r")\b",
    re.IGNORECASE,
)


def _is_earnings_article(art: dict) -> bool:
    """True when the row is an earnings print / results wire.

    Used only for the tighter on-release freshness gate. Prefer explicit
    category/tags when present; otherwise title+summary keyword match.
    Pure read-side — never mutates the article.
    """
    cat = str(art.get("category") or art.get("event_type") or "").strip().upper()
    if "EARNINGS" in cat:
        return True
    tags = art.get("tags") or art.get("labels") or []
    if isinstance(tags, str):
        tags = [tags]
    for t in tags:
        if "EARNINGS" in str(t or "").upper():
            return True
    blob = f"{art.get('title') or ''} {art.get('summary') or ''}"
    return bool(_EARNINGS_TITLE_RE.search(blob))


def _article_age_hours(art: dict) -> float | None:
    """Hours since the article was published — ``published`` preferred, else
    ``first_seen``. ``None`` when neither field parses (the caller then simply
    omits the age line; this NEVER blocks an alert — >24h staleness is already
    enforced by ``_article_age_ok``). First parseable field wins, RFC822 + ISO,
    naive→UTC: the exact convention ``_article_age_ok``/``urgency_scorer`` use,
    so the displayed age is consistent with the staleness gate."""
    now = datetime.now(timezone.utc)
    for field in ("published", "first_seen"):
        raw = (art.get(field) or "").strip()
        if not raw:
            continue
        dt = None
        try:
            dt = parsedate_to_datetime(raw)
        except Exception:
            dt = None
        if dt is None:
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except Exception:
                dt = None
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (now - dt).total_seconds() / 3600.0)
    return None


def _article_age_str(art: dict) -> str | None:
    """Compact, analyst-readable freshness label for an urgent row —
    ``4m`` / ``3.2h`` / ``16h``. ``None`` when the age is unknown so the
    caller omits the line entirely (silent, never a fabricated "0m")."""
    h = _article_age_hours(art)
    if h is None:
        return None
    if h < 1.0:
        m = int(round(h * 60))
        return f"{m}m" if m >= 1 else "<1m"
    if h < 10.0:
        return f"{h:.1f}h"
    return f"{int(round(h))}h"


# ── Quote-widget noise gate (defense-in-depth) ───────────────────────────────
# collectors/web_scraper now rejects Yahoo/Bloomberg live ticker-tape entries
# ("NVDANVIDIA Corporation227.13-8.61(-3.65%)") at ingestion, but web_scraper
# is not the only path a spaceless price-tick title can enter on
# (yahoo_ticker_rss, finnhub, a manual replay), and the urgency head
# demonstrably over-scores them (live: ml up to 9.99; one was Sonnet-scored
# 8.0 and fired a real 🚨 BREAKING push — the analyst's single biggest noise
# complaint). This is the same layered-defense shape as _is_synthetic /
# _article_age_ok / _filter_low_authority_lone: a formatter-side drop, NOT an
# ML-threshold change, at the single chokepoint every alert funnels through.
# The helper is duplicated from web_scraper rather than cross-imported — the
# watchers layer must not pull the collectors/aiohttp import graph (same
# rationale as article_store._briefing_domain_key duplicating ml.features).
_QW_PRICE_GLUE = re.compile(r"[A-Za-z]\$?\d{1,4}[.,]\d{2,3}")
_QW_PCT_PAREN = re.compile(r"\([+-]?\d{1,3}(?:\.\d+)?%\)")
_QW_QUOTE_PATH = re.compile(r"/quote/[^/]+/?$", re.I)
# Quote-aggregator share-card / listing-page pseudo-article — a DISTINCT
# surface the two title fingerprints above don't catch. Google News indexes
# the Moomoo/Futu/Webull "share this quote" landing pages whose title is the
# rendered card: "$NVIDIA (NVDA.US)$ - Moomoo" / "$Tencent (00700.HK)$ - Futu".
# Live evidence (2026-05-18, recurring across ≥6 prior passes): the exact row
# "$NVIDIA (NVDA.US)$ - Moomoo" from the `GN: Nvidia` collector was
# ML-relevance-scored 9.77 (ai_score 0) and fired a 🚨 BREAKING urgent alert
# (urgency=2) — the consuming analyst's recurring noise complaint, never
# fingerprint-gated (only the cred-bar approach was deferred as contested
# tuning; a fingerprint gate is the accepted quote-widget precedent). The
# fingerprint = leading "$" share-card lead glued to a "(SYMBOL.EXCH)$" close;
# bounded ({0,60}) so no catastrophic backtracking; validated zero false
# positives on the live $+paren headline corpus
# ("$NVDA breaks out (NYSE)", "Zscaler (NASDAQ:ZS) ... $223.00").
# Byte-identical to collectors.web_scraper / analysis.claude_analyst (lockstep).
_QW_LISTING = re.compile(
    r"^\s*\$[^$\n]{0,60}\([A-Za-z0-9.\-]{1,8}\.[A-Za-z]{1,4}\)\$"
)
# Yahoo Finance screener-tape pseudo-article — distinct fingerprint the three
# regexes above don't catch (no glued price, no parenthesised %, no $share-card
# lead). ``collectors/market_movers.py`` emits screener entries with a leading
# ``[YF/<bucket>]`` tag, e.g.
#   ``[YF/most_actives] MU (Micron Technology, Inc.) +2.5% @ $698.74 | vol 6``
#   ``[YF/day_gainers] AXTI (AXT Inc) +6.6% @ $112.88 | vol 9.6M (0.8x avg)``
# Live evidence (2026-05-19, last 2h of urgency=2 rows): 4 of 12 BREAKING
# alerts fired by the analyst's standalone-push channel were YF screener
# entries — ml_score 9.9 score_source='ml' (urgency head over-scores them
# because the title looks "extreme": signed %, large vol number, ticker dollar
# price). They are NOT breaking news — they describe the CURRENT market state
# (this ticker is one of today's top movers), and the 30-min per-(symbol,
# screener) cooldown in market_movers.py dampens repetition but cannot
# down-rank the urgency itself. The defense-in-depth drop here is the only
# surface that suppresses the standalone push without breaking the collector
# (it still emits one row per surge that survives in the digest if Opus picks
# it up).
#
# Fingerprint anchoring: ``^\s*\[YF/<lowercase_underscore>\]\s+[A-Z]``. The
# real publisher tag convention is ``GDELT/reuters.com`` / ``scraped/finance.
# yahoo.com`` / ``GN: Nvidia`` — never bracketed. The bucket-token character
# class ``[a-z_]+`` excludes the ``.com`` in ``[GDELT/reuters.com]`` should one
# ever appear, and the trailing ``\s+[A-Z]`` requires a ticker-like next token
# so a bare ``[YF/...]`` paragraph break is not matched.
_QW_SCREENER_TAPE = re.compile(
    r"^\s*\[YF/[a-z_]+\]\s+[A-Z]"
)
# StockTwits sentiment pseudo-article — distinct fingerprint the four above
# don't catch (no glued price, no parenthesised % paren, no $share-card, no
# bracketed YF screener). ``collectors/stocktwits_sentiment.py`` emits
# extreme-sentiment summary rows whose title is structured data, not a news
# headline:
#   ``[StockTwits Sentiment] NVDA Bullish: 53% Bullish / 3% Bearish (16↑ 1↓ of 30 msgs)``
#   ``[StockTwits Sentiment] LITE Bullish: 33% Bullish / 0% Bearish (10↑ 0↓ of 30 msgs)``
# Live evidence (2026-05-21, last 5h): 130 such rows, 45 ML-scored >=5, several
# at the 10.0 ceiling (the urgency head over-scores them because the title is
# dense with held tickers and "Bullish:"/percent figures the model has learned
# correlate with high relevance — model artefact). The stocktwits credibility
# tier (0.30 < ALERT_MIN_LONE_SOURCE_CRED 0.45) already suppresses LONE pushes,
# but the briefing's per-domain cap admits up to 6 of them into the 50-row
# top pool every cycle, displacing real news in TOP SIGNALS. The drop here is
# the alert-path lockstep for the briefing-path drop in
# analysis.claude_analyst._QW_STOCKTWITS_SENTIMENT, same anti-drift discipline
# as the four fingerprints above (byte-identical regex across the three
# defense layers: alert_agent / claude_analyst / web_scraper-pattern).
#
# Fingerprint anchoring: ``^\s*\[StockTwits Sentiment\]\s+[A-Z]``. The real
# publisher tag convention is unbracketed (``GDELT/reuters.com`` /
# ``scraped/finance.yahoo.com`` / ``GN: Nvidia``), so the leading bracket is
# the canonical pseudo-article-tag pattern (mirrors ``[YF/<bucket>]``). The
# trailing ``\s+[A-Z]`` requires a ticker-like next token so a bare
# ``[StockTwits Sentiment]`` paragraph break is not matched. Validated zero
# false positives on the live corpus — no real headline leads with this
# bracketed marker.
_QW_STOCKTWITS_SENTIMENT = re.compile(
    r"^\s*\[StockTwits\s+Sentiment\]\s+[A-Z]"
)
# Image-credit pseudo-article — lockstep mirror of the sixth fingerprint added
# to ``collectors.web_scraper._QW_IMAGE_CREDIT`` and
# ``analysis.claude_analyst._QW_IMAGE_CREDIT``. Live evidence (2026-05-21
# 16:30:49Z, alert_recency.db): "Angela Weiss/AFP/Getty Images" fired a real
# 🚨 BREAKING push from ``scraped/www.bloomberg.com`` (cred=0.90 — above the
# 0.45 lone-source bar; the source-authority gate cannot catch this, content
# type IS the failure). The ML urgency head scored it 10.0 because the
# bloomberg.com URL + proper-noun tokens triggered high-relevance pattern
# recognition. The bug: news pages wrap the hero image inside the article's
# own <a> link, so the web scraper's anchor-text fallback picks up the photo
# credit line beneath the image as the article title. Other live samples in
# articles.db (lower-scored, no push): "Tomohiro Ohsumi/Getty Images",
# "Timorthy A. Clary/AFP/Getty Images".
#
# Discriminator: anchored ^...$ so the WHOLE title is the credit (real
# headlines never end with this no-space ``/Agency`` structure). Title-Case
# photographer name (≥2 tokens, allowing initials like ``A.``), then one or
# more ``/Agency`` slugs with no space around the slash, ending in a
# recognised image agency from a closed list. Validated zero false positives
# against the must-survive corpus: "Reuters/Yahoo Finance reports earnings",
# "Sam Altman/OpenAI says GPT-5 coming", "MU drops 5%/Yahoo", "AFP/Getty
# Images launches new service" all do NOT match. Byte-identical to
# collectors.web_scraper / analysis.claude_analyst — the documented
# triple-gate lockstep (anti-import-cycle: the watchers layer must not pull
# the collectors/aiohttp graph and the analysis layer must not pull the
# watchers/ml graph).
_QW_IMAGE_CREDIT = re.compile(
    # Name-token alternation: a hyphenated form (I-Hwa, O-Lin, Jean-Pierre)
    # OR a plain Capitalized run (Tomohiro, Cheng). The hyphenated branch
    # exists because Asian / French given-name conventions place the hyphen
    # *inside* the first token, and the prior `[A-Z][a-zA-Z]+` requirement
    # silently missed every such name — live evidence (2026-05-23 urgency=2
    # set): "I-Hwa Cheng/Bloomberg" reached alerted state un-suppressed
    # because the name-token regex demanded ≥1 letter immediately after the
    # capital and hit `-` first. The hyphen branch is anchored on a
    # second uppercase letter so a stray "I-foo" prose token cannot match.
    r"^\s*(?:[A-Z][a-zA-Z]*(?:-[A-Z][a-zA-Z]+)+|[A-Z][a-zA-Z]+)"
    r"(?:\s+(?:[A-Z]\.?|(?:[A-Z][a-zA-Z]*(?:-[A-Z][a-zA-Z]+)+|[A-Z][a-zA-Z]+)))+"
    r"(?:/(?:AFP|Reuters|Getty\s+Images|AP|Bloomberg|EPA|TASS|"
    r"WireImage|Shutterstock|Polaris|Bloomberg\s+News))+"
    r"\s*$"
)

# Single source of truth for the title-fingerprint set, mirrored by
# ``_RECAP_TEMPLATE_PATTERNS`` below. analytics.quote_widget_audit imports this
# tuple so per-fingerprint counts in the audit cannot silently drift from what
# ``_looks_like_quote_widget`` actually catches — the same anti-drift
# discipline the recap-audit module follows. The URL-based ``_QW_QUOTE_PATH``
# is NOT included here: it inspects the link, not the title, and the audit
# operates on title-only counts (matching how the rows enter the training
# pool — by title-derived ai_score, not URL).
_QUOTE_WIDGET_TITLE_PATTERNS = (
    ("price_glue", _QW_PRICE_GLUE),
    ("pct_paren", _QW_PCT_PAREN),
    ("listing_card", _QW_LISTING),
    ("screener_tape", _QW_SCREENER_TAPE),
    ("stocktwits_sentiment", _QW_STOCKTWITS_SENTIMENT),
    ("image_credit", _QW_IMAGE_CREDIT),
)


def _looks_like_quote_widget(art: dict) -> bool:
    """True for a live quote-tape / quote-listing / structured-data-summary /
    image-credit entry masquerading as an urgent article.

    Six independent title fingerprints (a letter glued to a decimal price; a
    parenthesised signed % change; a "$NAME (SYMBOL.EXCH)$" share-card listing
    page; a ``[YF/<bucket>]`` screener-tape lead from ``market_movers``; a
    ``[StockTwits Sentiment]`` extreme-sentiment summary row from
    ``stocktwits_sentiment``; a ``Photographer Name/Agency/Getty Images``
    photo credit the web scraper picked up as a title) plus a Yahoo /quote/
    landing path. All are anchored so real headlines with $/%/comma numbers
    ("rises 22% to $35.1 billion", "5,123.41 record high"), real "$TICKER ..."
    prose ("$MU upgraded to Buy"), real quote-scoped article URLs, and real
    headlines that happen to contain agency names ("Reuters/Yahoo Finance
    reports") are never caught. Mirrors
    collectors.web_scraper._looks_like_quote_widget."""
    title = art.get("title") or ""
    if (_QW_PRICE_GLUE.search(title) or _QW_PCT_PAREN.search(title)
            or _QW_LISTING.search(title) or _QW_SCREENER_TAPE.search(title)
            or _QW_STOCKTWITS_SENTIMENT.search(title)
            or _QW_IMAGE_CREDIT.search(title)):
        return True
    url = art.get("link") or art.get("url") or ""
    try:
        if _QW_QUOTE_PATH.search(urlparse(url).path):
            return True
    except Exception:
        pass
    return False


def _filter_quote_widget_noise(
    arts: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition urgent rows into ``(kept, suppressed)``. ``suppressed`` is the
    quote-tape pseudo-articles; everything else is kept. Pure — no DB / IO.
    Runs BEFORE dedup so a price tick syndicated across two collectors
    (yahoo + finnhub surfacing the same NVDA tick) is still caught."""
    kept: list[dict] = []
    suppressed: list[dict] = []
    for a in arts:
        (suppressed if _looks_like_quote_widget(a) else kept).append(a)
    return kept, suppressed


# ── Recap / SEO template title gate (defense-in-depth) ──────────────────────
# A second, distinct surface the urgency head over-scores is the *recap /
# preview / transcript-summary* template — content that is inherently
# retrospective ("trading up TODAY", "Q1 Earnings Call Highlights", a date-
# stamped "Stock Market Today, May 18:" wrap-up) or algorithmic mill output
# ("(LITE) Shares Fall 8.8% -- GF Value Says ..."). These NEVER warrant a
# standalone 🚨 BREAKING push: by the time the recap was written the move was
# already in the market, the call was already over, the wire already printed.
#
# Live evidence (2026-05-18/19, recurring across multiple alert cycles
# inspected from the live articles.db urgency=2 set):
#   - "Why Nvidia (NVDA) Stock Is Trading Up Today" — fired 22:34 and 00:12
#     from Finnhub/Yahoo and YahooFinance/NVDA (two separate BREAKING pushes)
#   - "Why Did Micron Stock Drop Today ? | The Motley Fool" — fired 00:50
#   - "Stock Market Today, May 18: Micron Falls as Memory Concerns Test AI
#      Rally" — fired THREE times at 22:52 from YahooFinance/005930.KS,
#     Motley Fool, and Nasdaq Markets (three separate BREAKING pushes for
#     one wrap-up)
#   - "D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights: Surging
#      Bookings Amid Revenue Decline" — fired twice ~14 min apart
#   - "Lumentum Holdings Inc (LITE) Shares Fall 8.8% -- GF Value Says ..."
#     and "AXT Inc (AXTI) Shares Fall 14.3% -- GF Value Says Still
#     Overvalued" — GuruFocus algorithmic stock-mention press mill
#   - "Here What the Street Thinks About NVIDIA Corporation (NVDA)" —
#     opinion mill recap
#
# These all came from publishers ABOVE the ``ALERT_MIN_LONE_SOURCE_CRED``
# 0.45 bar (Finnhub 0.78, Motley Fool/yahoo ~0.65, GoogleNews 0.62) so the
# existing source-authority gate doesn't catch them — the failure is the
# *content type*, not the publisher's credibility tier.
#
# Same shape as ``_filter_quote_widget_noise``: a small, evidence-backed set
# of title fingerprints, anchored so real breaking headlines never match;
# runs BEFORE dedup so a recap syndicated across multiple feeds is
# suppressed once (not after dedup picks one survivor); suppressed rows are
# marked alerted UNCONDITIONALLY by ``send_urgent_alert`` so they exit the
# urgent queue. Pure read-side — no DB write, no ai_score/ml_score/
# score_source mutation, backtest already filtered upstream by
# ``_is_synthetic`` — all four load-bearing invariants intact.

# "Why <Co|TICKER> Stock Is Trading {Up|Down|Higher|Lower} Today" — the
# canonical Zacks/Yahoo/Finnhub recap template. Anchored ``^Why`` so a
# headline that USES "why" mid-sentence is unaffected. The "Stock Is
# Trading" + "Today" co-occurrence is the discriminator: a real headline
# like "Why investors are bullish on Nvidia" or "Why MU beat estimates"
# does NOT match.
_RT_WHY_TRADING = re.compile(
    r"^\s*why\s+.+?\s+stock\s+is\s+trading\s+(?:up|down|higher|lower)\s+today\b",
    re.IGNORECASE,
)
# "Why Did <X> Stock {Drop|Rise|Surge|Fall|Climb|Plunge|Soar|Jump} Today"
# (Motley Fool / Zacks variant — past-tense recap of an intraday move).
_RT_WHY_DID = re.compile(
    r"^\s*why\s+did\s+.+?\s+stock\s+"
    r"(?:drop|rise|surge|fall|climb|plunge|soar|jump|tumble)\b",
    re.IGNORECASE,
)
# "Stock Market Today, May 18: ..." — date-stamped daily market wrap-up
# (Motley Fool / Nasdaq / Yahoo). Anchored ^ + month-name + 1-2 digit day.
# Real headlines do not lead with this exact "Stock Market Today, <Month>
# <day>" pattern.
_RT_MARKET_TODAY = re.compile(
    r"^\s*stock\s+market\s+today\s*[,:]\s*"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}\b",
    re.IGNORECASE,
)
# "Q1 2026 Earnings Call Highlights" / "Q1 Earnings Call Highlights" /
# "Q1 2027 Earnings Transcript" — GuruFocus / Seeking Alpha / Globe-and-Mail
# transcript-summary template. The call already happened; this is recap, not
# breaking. Substring (not anchored) — the template appears mid-headline
# ("D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights: ...").
#
# Live evidence (2026-05-20, NVDA earnings day): the prior form REQUIRED
# both a year ``20\d{2}`` *and* the literal ``call`` between earnings and
# the recap noun, so two recap variants leaked through and fired BREAKING:
#   - "NVIDIA Q1 Earnings Call Highlights"            (no year)
#   - "Nvidia (NVDA) Q1 2027 Earnings Transcript - The Globe and Mail"
#                                                     (no "Call")
# Year and the "call " bridge are now BOTH optional; the discriminator stays
# the recap-noun list ``highlights|recap|takeaways|transcript|summary|review``.
# Validated against the must-survive corpus: forward-looking titles
# ("Q1 Earnings Preview", "Q3 2026 earnings preview", "Nvidia Q1 beats
# estimates", "NVDA Q2 2026 earnings call begins at 5pm ET") are NOT caught
# because they lack any recap-noun terminator.
_RT_EARNINGS_CALL = re.compile(
    r"\bq[1-4](?:\s*(?:fy\s*)?20\d{2})?\s+earnings\s+(?:call\s+)?"
    r"(?:highlights|recap|takeaways|transcript|summary|review)\b",
    re.IGNORECASE,
)
# "Here['s|is] What the Street Thinks About <X>" — InsiderMonkey opinion-
# mill recap template. The "Street thinks" framing IS the recap signature.
_RT_STREET_THINKS = re.compile(
    r"^\s*here(?:'?s|\s+is)?\s+what\s+the\s+street\s+thinks\b",
    re.IGNORECASE,
)
# "(TICKER) Shares Fall 8.8% -- GF Value Says ..." / "Shares Fall 3.8% -- What
# GF Score Offers" — GuruFocus algorithmic press-mill templates, posted on
# every fractional stock-price move. Both ``GF Value Says`` and ``GF Score``
# are GuruFocus-specific marketing brands (no other publisher uses either
# phrase as part of a recap title), so the combined pattern is high-precision.
#
# Live evidence (2026-05-26, last 24h urgency=2 set on the consumer's
# articles.db): GoogleNews/GuruFocus's "Lumentum Holdings Inc (LITE) Shares
# Fall 3.8% -- What GF Score O[ffers] - GuruFocus" reached urgency=2 with
# ml_score=9.66, source_source='ml' — a real BREAKING push fired on a held
# ticker (LITE) for a pure recap-mill stock-move attribution. The original
# ``\bgf\s+value\s+says\b`` discriminator required the ``Value Says`` exact
# phrase, so the ``GF Score`` sibling variant slipped through. 5 such rows
# landed in the 7d window before this widening (1 alerted, 4 queued or
# gate-suppressed at earlier stages). The widened alternation ``(?:value\s+
# says|score)`` catches both family members while keeping the GuruFocus-
# unique ``GF `` anchor — real headlines never lead with this phrase.
_RT_GF_VALUE = re.compile(
    r"\bgf\s+(?:value\s+says|score)\b",
    re.IGNORECASE,
)
# "Why <X> Stock {Just|Now|Today|Finally|Suddenly|...} {Popped|Surged|...}"
# (Motley Fool variant — same retrospective shape as _RT_WHY_DID, but the
# subject does the moving past-tense without "Did" between Why and the
# subject). Live evidence (2026-05-19): "Why Micron Stock Just Popped
# Again" was Sonnet-scored urgent=8 and fired a 🚨 BREAKING alert at
# 19:49Z; the recap gate would have caught it but the regex required
# "Did", so the row leaked all the way to Discord. The cross-cycle
# dedup later suppressed three syndicated copies of the same headline
# (yfinance/Motley Fool, scraped/finance.yahoo.com, YahooFinance/MU at
# 19:57-20:13Z) — the analyst still got the first push.
#
# Discriminator: an adverb between "Stock" and the past-tense verb is
# required so a real forward-looking headline ("Why Microsoft Stock
# Drop Continues", "Why MU Stock Pop Could Continue") is NOT caught. The
# verb list is past-tense ONLY, so "Pop" / "Surge" / "Drop" as nouns
# (forward-looking) also do not match. Validated against the live noise
# set and the must-survive corpus (the question-form headlines that
# legitimately discuss the future).
_RT_WHY_JUST_MOVED = re.compile(
    r"^\s*why\s+.+?\s+stock\s+"
    r"(?:just|now|today|finally|suddenly|then|recently|already)\s+"
    r"(?:popped|surged|jumped|soared|crashed|tumbled|plunged|sank|fell|"
    r"dropped|climbed|spiked|slid|slipped|rallied|tanked|plummeted|"
    r"nosedived|hammered|skyrocketed|rocketed|rebounded)\b",
    re.IGNORECASE,
)
# "X (TICKER) Reports Earnings Tomorrow: What To Expect" — the FinancialContent
# / StockStory / MSN / TradingView SEO-mill earnings-preview template. By
# definition NOT breaking ("tomorrow"), heavily syndicated, and the trailing
# "What To Expect" is the SEO-mill discriminator that distinguishes it from a
# real earnings preview (which says "Q1 Earnings Preview" or "ahead of
# earnings"). Live evidence (2026-05-19/20, 36h articles.db scan, all
# urgency=2): 6 distinct hits — DECK + SCVL (not held, pure SEO spam) at
# 03:57Z / 04:12Z 2026-05-20 fired BREAKING pushes today, plus NVDA syndicated
# 4× across FinancialContent / StockStory / MSN / TradingView at 03:21Z /
# 05:16Z / 05:42Z / 14:51Z 2026-05-19. Same retrospective-template class as
# _RT_EARNINGS_CALL (which catches POST-earnings "Q1 2026 Earnings Call
# Highlights"); this catches the PRE-earnings preview variant the existing
# gate's "highlights|recap|takeaways|transcript|summary" verb list explicitly
# excludes (real previews like "Q1 Earnings Preview" must NOT match).
#
# Discriminator: "Reports Earnings" + "Tomorrow" + ":" + "What To Expect" all
# four must appear in that order with the canonical separators. The colon-
# bounded "What To Expect" suffix is the SEO-mill tell — real wire copy
# announces an earnings date without that trailer ("NVIDIA Earnings Today:
# Wall Street Expects EPS to Jump..." has the colon but no "what to expect").
# Catches all 6 live cases (mixed-case, optional space around the colon, both
# `(NVDA)` and `( NVDA )` spacings handled by the regex's whitespace
# tolerance). Validated against the 12h Discord-push corpus that any
# legitimate NVDA-earnings-day push ("Nvidia Earnings Are Hours Away. Here Are
# 3 Things to Watch.", "Stock futures edge higher ahead of Nvidia earnings",
# "Nvidia stock erases early losses ahead of earnings", "NVIDIA Earnings
# Today: ...") does NOT match.
_RT_EARNINGS_TOMORROW = re.compile(
    r"\breports?\s+earnings\s+tomorrow\s*:\s*what\s+to\s+expect\b",
    re.IGNORECASE,
)
# "These Stocks Are Today's Movers: Nvidia, Micron, Intel, ..." — Barron's
# daily column heavily syndicated across yfinance/Barrons.com,
# scraped/www.barrons.com, Finnhub/Yahoo, YahooFinance/<TICKER> (the per-
# ticker RSS picks up the SAME column for every ticker it mentions). Same
# retrospective-recap class as ``_RT_MARKET_TODAY`` (the date-stamped daily
# wrap-up); this one is the *same-day movers* sibling — by definition a
# retrospective list of names that already moved, not a forward-looking
# breaking story.
#
# Live evidence (2026-05-20, last 4h urgency=1 queue scan): the exact title
# "These Stocks Are Today's Movers: Nvidia, Micron, Intel, Meta" appeared
# from 3 distinct sources (yfinance/Barrons.com, scraped/www.barrons.com,
# Finnhub/Yahoo) all ML-flagged urgency=1, score_source='ml', ml_score~9.x,
# and "These Stocks Are Today's Movers: Micron, Intel, Lowe's, Nvidia" from
# 5 more sources (Finnhub/Yahoo, YahooFinance/NVDA, yfinance/Barrons.com,
# scraped/www.barrons.com, GoogleNews variants). The cross-cycle alert-
# recency suppression collapses repeated copies of the SAME signature, but
# the analyst still gets one push per distinct movers-list (which changes
# composition daily as different tickers move) — pure SEO-mill recap, never
# breaking, and the ML urgency head systematically over-scores it because
# the title is dense with held tickers (NVDA + MU concentration trips the
# model's portfolio_flag/ticker_count/ticker_density features).
#
# Discriminator: "These Stocks Are Today's Movers" + ":" — the colon-bounded
# list is the SEO-mill signature. Anchored ``^`` so a mid-sentence "today's
# movers" reference in a real headline ("Why Some Of Today's Movers Could
# Run Higher Tomorrow", forward-looking analysis) is NOT caught. The
# possessive apostrophe is optional (``today'?s``) to handle the curly
# Unicode apostrophe / no-apostrophe variants live feeds occasionally emit.
# Validated catches all live-noise copies and does NOT catch the must-
# survive corpus (real "today's high" / "today's session" / mid-sentence
# uses; forward-looking "tomorrow's movers" / "next week's movers"; legit
# headlines that don't lead with the bracketed list pattern).
_RT_TODAYS_MOVERS = re.compile(
    # ['’]? handles ASCII (U+0027), curly Unicode (U+2019), or no apostrophe.
    # The Barron's column live-feeds curly while the GoogleNews/Finnhub
    # republished copies sometimes carry the ASCII or none at all.
    r"^\s*these\s+stocks\s+are\s+today['’]?s\s+(?:top\s+|biggest\s+)?movers\s*:",
    re.IGNORECASE,
)
# "Is <X> a Buy After <Earnings|Q1|Results|Report|Quarter>" — the Motley Fool /
# Yahoo / TipRanks / InsiderMonkey post-event valuation question template. By
# the time someone writes "Is Nvidia a Buy After Their Latest Earnings Report?",
# the print has already crossed the wire, the price has moved, and the analyst
# has either traded the print or missed it — a standalone 🚨 BREAKING push on
# this content is by definition retrospective recap, not breaking news.
#
# Live evidence (2026-05-21, alert_recency.db pushed-alert audit; the canonical
# record of REAL Discord pushes, distinct from urgency=2 in articles.db which
# also captures gate-suppressed rows): "Is Nvidia a Buy After Their Latest
# Earnings Report?" fired a real BREAKING push at 04:46:07Z from
# `yfinance/Motley Fool` and was repeated by `YahooFinance/NVDA` ml_score 9.79.
# Source-credibility tier 0.65-0.68 → above the 0.45 ALERT_MIN_LONE_SOURCE_CRED
# bar so the authority gate doesn't catch it; content type IS the failure.
#
# Discriminator: leading `^Is` + `a (Buy|Sell|Hold)` + the `\bafter\b` bridge +
# a recap-noun terminator (`earnings|results|report|quarter|Q[1-4]`). The
# `after` requirement is what makes this retrospective — a forward-looking
# pre-earnings question ("Is NVDA a Buy Before Earnings?" / "Is AMD a Buy")
# does NOT match. The recap-noun list bounds it so "Is X a Buy after the
# crash" without an earnings/results context (a real macro question) is not
# auto-suppressed either. Validated against the must-survive corpus: real
# analyst raises ("Bank of America raises NVDA price target", "Wedbush says
# Nvidia likely to top estimates"), forward-looking previews, and macro
# headlines all do NOT match because none lead with `^Is ... a Buy/Sell/Hold`.
_RT_IS_BUY_AFTER = re.compile(
    # Two leading-anchor variants: bare "Is X a Buy..." (the live failure case)
    # AND the "Subject Is [Still|Now|It] a Buy..." variant where the ticker
    # name precedes the verb ("Tesla Is Still a Buy After Q1 Beat, Says
    # Wedbush"). Both are POST-event valuation questions — the `after` +
    # earnings-noun terminator is the discriminator.
    r"^\s*(?:\S+\s+)?is\s+(?:\w+\s+){0,2}a\s+(?:buy|sell|hold)\b.*?"
    r"\bafter\b.*?\b(?:earnings|results|report|quarter|q[1-4])\b",
    re.IGNORECASE,
)
# "Why Is <X> Down/Up N.N% Since Last <Earnings|Report|Quarter>?" — the
# Zacks / Seeking Alpha / TipRanks post-event price-attribution template. By
# definition retrospective ("since" anchors the move BEFORE this article was
# written), with an explicit percent figure that names the move that already
# happened. Same retrospective-recap class as `_RT_WHY_DID` ("Why Did X Stock
# Drop Today") but uses present-tense `Is` instead of past-tense `Did`, and
# requires the `% since` discriminator to avoid catching real ongoing-move
# coverage ("Why is the rally fading?", "Why is Tesla down?").
#
# Live evidence (2026-05-21, alert_recency.db pushed-alert audit): "Why Is
# AGNC Investment (AGNC) Down 7.2% Since Last Earnings Report?" fired a real
# BREAKING push at 05:19:12Z. Source-credibility tier above the 0.45 bar so
# the authority gate doesn't catch it; the content is pure SEO-mill retro.
#
# Discriminator: leading `^Why\s+Is` + `\b(up|down|higher|lower)\b` +
# `\d+(?:\.\d+)?%` (the explicit move magnitude, REQUIRED) + `\bsince\b` (the
# explicit temporal anchor placing the move BEFORE the article — REQUIRED).
# Both `%` and `since` together is the discriminator — either alone is too
# broad. Validated against the must-survive corpus: "Why is the rally
# fading?" (no % no since), "Why investors are bullish on Nvidia" (no Is),
# "MU down 7% since earnings" (no leading Why Is anchor), "Why MU beat Q3
# estimates" (no up/down/higher/lower + % + since trio) all do NOT match.
_RT_WHY_IS_PCT_SINCE = re.compile(
    r"^\s*why\s+is\s+.+?\s+(?:up|down|higher|lower)\s+"
    r"\d+(?:\.\d+)?\s*%\s+since\b",
    re.IGNORECASE,
)
# "Why <X> Stock Is [adverb]? <present-state-action> After <Earnings|Q1|Results|...>"
# — the Barron's / MSN / Yahoo post-event price-attribution recap template.
# Distinct from the existing why-recap variants:
#   - _RT_WHY_TRADING requires "trading up/down today" (today + direction)
#   - _RT_WHY_DID requires "Did" between Why and subject
#   - _RT_WHY_JUST_MOVED requires past-tense verb with adverb (just / now / ...)
#   - _RT_WHY_IS_PCT_SINCE requires explicit "% since" trio
# This shape is "Why X Stock Is <state-verb> After <event>" — present-tense,
# retrospective via the "after" + earnings-noun anchor placing the move BEFORE
# the article was written. None of the existing four caught it.
#
# Live evidence (2026-05-21 NVDA earnings night, alerted urgency=2 rows
# 10:37:16Z + 10:50:41Z): "Why Nvidia Stock Is Barely Moving After Earnings
# Crushed Expectations" fired BREAKING twice on Barron's + MSN syndication
# (the cross-cycle dedup caught the THIRD copy at 10:59:00Z but the analyst
# had already received two pushes). Pure SEO/recap post-event explainer.
#
# Discriminator: leading `^Why\s+...\s+Stock\s+Is` + (adverb)? + a
# present-state action verb from a closed list + `\bafter\b` + recap-noun
# (earnings|results|report|quarter|q[1-4]|beat|miss|guidance). The closed
# action-verb list is what makes this safe: "Why X Stock Is the Best Buy"
# does NOT match because "the" is not a present-state action verb; "Why X
# stock could rise after earnings" does NOT match because "could" is not in
# the verb list. The "after" + recap-noun terminator is REQUIRED so a real
# ongoing question ("Why X stock is moving") doesn't match.
_RT_WHY_STOCK_IS_AFTER = re.compile(
    r"^\s*why\s+.+?\s+stock\s+is\s+"
    # Optional qualifying adverb (live evidence: "barely"; plausible extensions
    # the same SEO template uses: still / now / finally / just / currently /
    # actually / suddenly / hardly / so / really).
    r"(?:still\s+|barely\s+|now\s+|finally\s+|just\s+|currently\s+|"
    r"actually\s+|suddenly\s+|hardly\s+|so\s+|really\s+)?"
    # Closed list of present-state action verbs / state adjectives. Each is a
    # past-or-present description of what the stock IS doing — never a future-
    # tense "could/may/might" or a non-action filler. Matches the live failure
    # case (Barely Moving) plus the same retrospective shape with synonymous
    # verbs the template family uses.
    r"(?:moving|trading|sliding|sinking|tumbling|crashing|plunging|"
    r"jumping|surging|soaring|rising|falling|climbing|dropping|"
    r"rallying|spiking|tanking|skyrocketing|nosediving|"
    r"up|down|higher|lower|flat|stuck|red|green|bid|offered)"
    r"\b.*?"
    # Required "after" + recap-noun terminator — the retrospective anchor.
    r"\bafter\b.*?\b"
    r"(?:earnings|results|report|quarter|q[1-4]|beat|miss|guidance)\b",
    re.IGNORECASE,
)
# "Why <X> Is <up|down|higher|lower> N.N% After <event>" — the Zacks /
# StockStory / MSN post-event price-attribution recap template that lacks an
# explicit ``Stock`` token (so ``_RT_WHY_STOCK_IS_AFTER`` doesn't anchor) and
# uses ``after`` rather than ``since`` (so ``_RT_WHY_IS_PCT_SINCE`` doesn't
# trigger). Same retrospective shape as both, distinct phrasing.
#
# Live evidence (2026-05-21, alert_recency.db pushed-alert audit): "Why AXT
# (AXTI) Is Down 14.2% After Betting Big On AI-Focused Indium Phosphide
# Expansion" fired a real BREAKING push at 11:14:35Z. Source-credibility tier
# above the 0.45 bar so the authority gate doesn't catch it; content type IS
# the failure. The 14.2% move was already in the market by the time the recap
# was written and the "After Betting Big ..." terminator names the prior event
# the move is being retrospectively attributed to.
#
# Discriminator: leading ``^Why\s+`` + arbitrary subject (``.+?``) + auxiliary
# (``is/are/was/were``) + direction (``up/down/higher/lower``) + explicit move
# (``\d+(?:\.\d+)?\s*%``) + ``after\b``. The auxiliary + direction + % + after
# quad is the discriminator — a real ongoing question ("Why is Tesla down?")
# lacks the % digit; a real earnings beat ("Nvidia Q1 revenue rises 22%")
# lacks the ``^Why ... Is ... after`` anchor; a forward-looking question
# ("Why NVDA Stock Could Rise 10% After Q1") uses "could" not "is". The
# AGNC ``% since`` variant is already caught by ``_RT_WHY_IS_PCT_SINCE`` so
# this pattern deliberately requires ``after`` not ``since`` — the two
# fingerprints are mutually exclusive on their terminator.
_RT_WHY_PCT_AFTER = re.compile(
    r"^\s*why\s+.+?\s+(?:is|are|was|were)\s+"
    r"(?:up|down|higher|lower)\s+"
    r"\d+(?:\.\d+)?\s*%\s+"
    r"after\b",
    re.IGNORECASE,
)

# "<Company> Earnings: A Quick Glance at Key Metrics" — the Zacks Investment
# Research post-earnings recap-mill template. By definition retrospective: the
# print has already crossed the wire and the "Quick Glance" summary is written
# AFTER the fact. NOT breaking news — a standalone 🚨 BREAKING push on it tells
# the analyst nothing they can trade.
#
# Live evidence (2026-05-21 NVDA earnings night, articles.db urgency=2 set):
# "NVIDIA Earnings: A Quick Glance at Key Metrics" reached urgency=2 THREE
# times — YahooFinance/NVDA (ml_score 9.9), yfinance/Zacks (ml_score 9.7), and
# GN: Nvidia (ai_score 9.0 — Sonnet itself over-scored it, so the urgency_
# scorer pre-floor missed it before this gate existed). All three publishers
# are above the 0.45 ALERT_MIN_LONE_SOURCE_CRED bar so the source-authority
# gate does not catch it; content type IS the failure, same class as
# `_RT_EARNINGS_CALL` (post-earnings "Q1 2026 Earnings Call Highlights").
#
# Discriminator: the verbatim Zacks signature phrase "a quick glance at
# [key] [financial] metrics". Substring (not anchored ^) — the phrase appears
# mid-headline after the "<Company> Earnings:" lead, exactly like
# `_RT_EARNINGS_CALL`. No real breaking headline contains this phrase, so the
# `key`/`financial` qualifiers are optional without false-positive risk.
# Validated against the must-survive corpus: real earnings movers ("Nvidia Q1
# revenue rises 22%", "MU earnings blow past estimates") do NOT match.
_RT_QUICK_GLANCE = re.compile(
    r"\ba\s+quick\s+glance\s+at\s+(?:key\s+)?(?:financial\s+)?metrics\b",
    re.IGNORECASE,
)

# "<headline>. Here's What Happened" — the Motley Fool / MarketBeat /
# tickerreport.com SEO retrospective tail. By definition past-tense ("What
# HAPPENED"), and the phrase itself is the discriminator: a real wire copy
# names the event in the headline, NEVER trails with a generic "Here's What
# Happened" hook.
#
# Live evidence (2026-05-23, 24h articles.db scan): the Motley Fool variant
# "Nvidia Just Crushed Earnings Estimates, but the Stock Fell. Here's What
# Happened (and What Comes Next)" reached urgency=1 syndicated across 6
# sources (Motley Fool, yfinance/Motley Fool, scraped/finance.yahoo.com,
# YahooFinance/NVDA, GN: earnings, GDELT/fool.com — all above the 0.45 lone-
# source bar, so the authority gate cannot catch it). ML scored 9.22-9.41
# and would have fired a 🚨 BREAKING push on the next alert cycle.
#
# MarketBeat / tickerreport.com variant: "Costco Wholesale (NASDAQ:COST)
# Stock Price Down 2.1% - Here's What Happened - MarketBeat" — same
# retrospective shape, also syndicated across GN: + GDELT. These pre-event
# rows were urgency=0 in the snapshot (ml_score < 8.0) but are the same
# template class; pre-emptively gating them costs nothing.
#
# Discriminator: \bhere(?:[s'’]+|\s+is)?\s+what\s+happened\b. Three apostrophe
# forms covered (ASCII straight ', curly ’, no apostrophe with bare s),
# plus the longer "Here is What Happened" form. Past-tense "happened" is
# REQUIRED — "Here's What's Happening" (present continuous) is a different
# template (live market wraps, often forward-looking) and is NOT matched.
# Substring (not anchored) since the phrase appears mid-headline, exactly
# like ``_RT_EARNINGS_CALL`` / ``_RT_QUICK_GLANCE`` / ``_RT_GF_VALUE``.
# Validated against the must-survive corpus: "Fed surprises with 50bp cut",
# "MU earnings blow past estimates", "Nvidia Q1 revenue rises 22%", "Trump
# signs executive order on chips" — none contain the trailing "Here's What
# Happened" SEO hook, so none match.
_RT_HERES_WHAT_HAPPENED = re.compile(
    r"\bhere(?:[s'’]+|\s+is)?\s+what\s+happened\b",
    re.IGNORECASE,
)

# "<headline>. Here's What It Means [for X]" / "Here's What That Really Means" —
# present-tense SEO trailer sibling of ``_RT_HERES_WHAT_HAPPENED`` (which only
# catches the past-tense "happened" form). Same SEO retrospective-trailer
# template family, distinct regex.
#
# Live evidence (2026-05-25, alert_recency.db pushed-alert audit — the
# canonical record of REAL Discord pushes): two distinct titles fired
# standalone 🚨 BREAKING pushes within a 90-minute window of NVDA earnings
# afterglow:
#   - "Nvidia's Board Just Authorized an Additional $80 Billion Buyback.
#      Here's What That Really Means" at 00:17:28Z (GN: dividend buyback,
#      ml_score=9.75 score_source='ml')
#   - "Jensen Huang just made a surprise announcement. Here's what it means
#      for Nvidia investors." at 02:03:48Z (GN: Nvidia, ai_score=8.0
#      score_source='llm' — Sonnet ALSO mis-labeled this as urgent because
#      the news lead before the SEO trailer was real, even though the rest
#      of the title adds nothing actionable)
# Both publishers sit above the 0.45 ``ALERT_MIN_LONE_SOURCE_CRED`` bar so
# the source-authority gate doesn't catch them; content type IS the failure.
# By the time a headline trails with "Here's what it means", the lead has
# already named the event and the trailer is filler — the SEO trailer adds
# zero new information the analyst can trade on.
#
# Discriminator: ``\bhere(?:[s'’]+|\s+is)?\s+what\s+(?:it|that|this)(?:\s+really)?
# \s+means\b``. FOUR required tokens (``here`` + ``what`` + closed pronoun set +
# ``means``) — the closed pronoun set (``it``/``that``/``this``) is what makes
# this safe against real news that uses "means" mid-sentence ("Bank of America
# says rate cut means recession risk eases"). The leading ``here`` anchor
# excludes leading-``what`` constructions ("What it means when the Fed pauses",
# "What does this mean for traders?"). The plural ``means`` (verb) is REQUIRED
# — singular ``mean`` (noun) is a different word class and doesn't appear in
# this template ("Powell on what tariffs mean for the labor market" is NOT
# caught because it uses ``mean`` not ``means``). Three apostrophe forms
# covered for the Here['s|s|is] alternation (ASCII straight ', curly ’, no
# apostrophe), mirroring ``_RT_HERES_WHAT_HAPPENED``.
#
# Validated against the must-survive corpus: "Powell explained what the cut
# means for inflation" (no leading ``here``), "Analysts weigh in on what the
# buyback means" (no leading ``here``), "What it means when the Fed pauses"
# (leading ``what`` not ``here``), "Bank of America says rate cut means
# recession risk eases" (no ``here what`` lead-in), "What does this mean for
# traders?" (no ``means`` — uses ``mean``), "Powell on what tariffs mean for
# the labor market" (no ``means`` — uses ``mean``). All survive.
#
# Defense-in-depth: a byte-identical twin lives in
# ``analysis.claude_analyst._BRIEFING_RT_HERES_WHAT_MEANS`` — the documented
# lockstep, enforced structurally by
# ``test_alert_and_briefing_recap_tuples_have_same_length``.
_RT_HERES_WHAT_MEANS = re.compile(
    r"\bhere(?:[s'’]+|\s+is)?\s+what\s+(?:it|that|this)(?:\s+really)?\s+means\b",
    re.IGNORECASE,
)

# "<headline>. Here's What [It|That|This] Really Signals [to|for] X" —
# variant-verb sibling of ``_RT_HERES_WHAT_MEANS`` (same SEO-trailer template
# family with the verb ``signals`` instead of ``means``). The Global &
# Mail / MSN financial-content syndicator uses the "Signals" verb as a
# substitute for "Means" on the same post-event explainer template — the
# trailer adds nothing actionable beyond the news lead. Discovered live
# in a 24h scan run alongside the ``_RT_HERES_WHAT_MEANS`` rollout.
#
# Live evidence (2026-05-25, articles.db 24h urgent scan with the
# heres_what_means gate active): two NEW rows reached urgency=1 with
# ml_score 9.5-9.8 carrying the same SEO trailer template but with
# ``signals`` instead of ``means``:
#   - "Nvidia's Board Just Authorized an Additional $80 Billion Buyback.
#      Here's What That Really Signals to Investors. - The Globe and Mail"
#      (ml_score=9.83, score_source='ml', GN: dividend buyback)
#   - "Nvidia's board just authorized an additional $80 billion buyback.
#      Here's what that really signals to investors. - MSN"
#      (ml_score=9.53, score_source='ml', GN: Nvidia)
# Both publishers above the 0.45 ``ALERT_MIN_LONE_SOURCE_CRED`` bar — the
# source-authority gate doesn't catch them; content type IS the failure.
#
# Discriminator: identical structure to ``_RT_HERES_WHAT_MEANS`` but with
# the verb ``signals`` — FOUR required tokens (``here`` + ``what`` + closed
# pronoun set + ``signals``). The plural ``signals`` (verb form) is REQUIRED
# — singular ``signal`` (noun: "rate cut signal") is a different word class
# and is NOT matched.
#
# Must-survive corpus (validated): "Powell signals more cuts after rate
# decision" (no ``here`` + leading lead, not template), "Fed signals dovish
# tilt after CPI miss" (no ``here what``), "What this signals for the AI
# trade" (no ``here``), "Earnings results signal continued growth"
# (singular ``signal``, no ``here what``). All survive.
#
# Defense-in-depth: byte-identical twin in
# ``analysis.claude_analyst._BRIEFING_RT_HERES_WHAT_SIGNALS`` — same lockstep
# discipline enforced by ``test_alert_and_briefing_recap_tuples_have_same_length``.
_RT_HERES_WHAT_SIGNALS = re.compile(
    r"\bhere(?:[s'’]+|\s+is)?\s+what\s+(?:it|that|this)(?:\s+really)?\s+signals\b",
    re.IGNORECASE,
)

# "<Fund Name> LLC Makes New $X Million/Billion Investment in <Company>" —
# MarketBeat / americanbankingnews / AlphaVantage 13F press-mill SIBLING of
# ``_RT_HOLDINGS_BY_FUND`` and ``_RT_SHARES_BOUGHT_BY``. The same 13F press
# pipeline rephrases the institutional disclosure as a "<Fund LLC> Makes
# New $X Investment in <Company>" headline, which neither existing gate
# catches because:
#   - ``_RT_HOLDINGS_BY_FUND`` requires "Holdings <verb> by <fund> LLC" —
#     the trailing-LLC structure; this template leads with the LLC.
#   - ``_RT_SHARES_BOUGHT_BY`` requires "Shares (in|of) ... bought by
#     <fund> LLC" — the "Shares ... by" prefix; this template uses
#     "Makes/Acquired ... Investment in" instead.
#
# Live evidence (2026-05-25, articles.db 24h urgent scan): "Torren
# Management LLC Makes New $1.86 Million Investment in NVIDIA Corporation
# $NVDA" reached urgency=1 (ml_score=9.71, score_source='ml',
# AlphaVantage/MarketBeat) — pure 13F filing recap, by definition
# retrospective (the SEC filing was already public weeks before the
# headline was generated). The MarketBeat credibility tier (0.68) sits
# above the 0.45 lone-source bar so the source-authority gate doesn't
# catch it; content type IS the failure, identical class to the two
# existing 13F-mill gates.
#
# Discriminator: leading "<Fund> LLC" (the LLC is the announcer, NOT the
# trailer) + an action verb (``Makes|Made|Acquires|Acquired|Takes|Took``)
# + optional ``New`` + a dollar-prefixed magnitude (``$X[.X][KMB]``) +
# magnitude qualifier (``Million|Billion``)? + a stake-noun
# (``Position|Investment|Stake|Shares|Holdings``) + ``in|of``. The LLC
# anchor at the LEAD + the dollar-magnitude figure together are the
# discriminator — real news doesn't lead with "<Fund> LLC Makes
# $X Investment in <Co>". Validated against the must-survive corpus:
# "Berkshire Hathaway takes new stake in Apple" (no LLC), "Saudi fund
# makes $5B investment in semis" (no LLC), "Tesla insiders bought 100,000
# shares" (different structure), "BlackRock Holdings now 12.3M shares of
# NVDA" (different structure). All survive.
#
# Defense-in-depth: byte-identical twin in
# ``analysis.claude_analyst._BRIEFING_RT_FUND_MAKES_INVESTMENT`` — same
# lockstep discipline enforced by
# ``test_alert_and_briefing_recap_tuples_have_same_length``.
_RT_FUND_MAKES_INVESTMENT = re.compile(
    r"\S+(?:\s+\S+){0,5}\s+LLC\s+"
    r"(?:Makes|Made|Acquires|Acquired|Takes|Took)\s+"
    r"(?:New\s+)?\$?[\d,.]+(?:\s*[KMB])?\s+"
    r"(?:Million\s+|Billion\s+)?"
    r"(?:Position|Investment|Stake|Shares?|Holdings?)\s+"
    r"(?:in|of)\b",
    re.IGNORECASE,
)

# "<Fund> LLC (Has|Owns|Grows|Trims|Sells N Shares of|Buys N Shares of|...)
# (Stock )?(Holdings|Shares|Position|Stake) in <Co>" — the 13F holdings-CHANGE
# sibling family of ``_RT_FUND_MAKES_INVESTMENT``. The existing gate catches the
# NEW-position template ("Makes/Acquires/Takes ... $X Investment/Position/Stake
# in"); this catches the much larger volume of QUARTERLY-CHANGE templates that
# the MarketBeat / AlphaVantage 13F press mill emits for every existing-position
# delta. Same retrospective non-event as the existing gates: the SEC filing was
# already public weeks before the press-mill headline was generated.
#
# Live evidence (2026-05-28..29, 24h articles.db urgency>=1 scan):
#   - "Kingsview Wealth Management LLC Has $31.47 Million Stock Holdings in
#      Oracle Corporation $O" (ml_score=9.0, urgency=1, GoogleNews/MarketBeat)
#   - "Leeward Financial Partners LLC Grows Stock Holdings in Oracle
#      Corporation $ORCL" (ml_score=9.82, urgency=2 — FIRED A REAL BREAKING
#      ALERT, GoogleNews/MarketBeat)
#   - "Jackson Creek Investment Advisors LLC Sells 3,338 Shares of Lam
#      Research Corporation $LRCX" (ml_score=9.64, urgency=1,
#      GoogleNews/MarketBeat)
# The MarketBeat credibility tier (0.68) sits ABOVE the 0.45
# ``ALERT_MIN_LONE_SOURCE_CRED`` bar so the source-authority gate doesn't catch
# them; content type IS the failure, same class as the three sibling 13F-mill
# gates (``_RT_FUND_MAKES_INVESTMENT`` / ``_RT_HOLDINGS_BY_FUND`` /
# ``_RT_SHARES_BOUGHT_BY``).
#
# Discriminator: leading ``<Fund> LLC`` (up to 6 tokens) + a closed
# delta-action verb list (distinct from ``_RT_FUND_MAKES_INVESTMENT``'s
# initial-position verbs) + OPTIONAL dollar-magnitude figure + a stake-noun
# (Position/Investment/Stake/Shares/Holdings/Stock Holdings) + in/of. The
# dollar/magnitude middle is OPTIONAL so plain delta phrasings like "Grows
# Stock Holdings in" / "Trims Position in" / "Buys 1,200 Shares of" all match,
# while the leading-LLC anchor and the stake-noun + in/of terminator together
# keep precision: real news doesn't combine all four pieces.
#
# Validated against the must-survive corpus (per
# ``test_alert_and_briefing_gates_agree_on_must_survive_corpus`` plus the
# ``test_alert_fund_makes_investment`` survivors): no LLC ("Berkshire takes
# stake in Apple", "Saudi fund makes $5B investment in semis", "Tesla insiders
# bought 100k shares"), LLC mid-sentence without a verb in the closed list
# ("Apple Inc reports earnings beat; major LLC holders take note" — "holders"
# not in verb list), and forward-looking ("Apple unveils $100B buyback
# program" — no LLC). Mutually orthogonal with ``_RT_FUND_MAKES_INVESTMENT``:
# the verb sets are disjoint, so a single title is fingerprinted by exactly
# one of the two siblings.
#
# Defense-in-depth: byte-identical twin in
# ``analysis.claude_analyst._BRIEFING_RT_FUND_STAKE_DELTA`` — same lockstep
# discipline enforced by ``test_alert_and_briefing_recap_tuples_have_same_length``.
_RT_FUND_STAKE_DELTA = re.compile(
    r"\S+(?:\s+\S+){0,5}\s+LLC\s+"
    r"(?:Has|Have|Hold|Holds|Own|Owns|Grows|Grew|Trims|Trimmed|"
    r"Boosts|Boosted|Cuts|Cut|Raises|Raised|Lowers|Lowered|"
    r"Maintains|Increases|Increased|Decreases|Decreased|Reduces|Reduced|"
    r"Hikes|Hiked|Slashes|Slashed|Sells|Sold|Buys|Bought|Purchases|Purchased|"
    r"Adds|Added|Removes|Removed|Liquidates|Liquidated|Initiates|Initiated)\s+"
    r"(?:(?:New\s+)?\$?[\d,.]+(?:\s*[KMB])?\s+(?:Million\s+|Billion\s+)?)?"
    r"(?:Position|Investment|Stake|Shares?|(?:Stock\s+)?Holdings?)\s+"
    r"(?:in|of)\b",
    re.IGNORECASE,
)
# Note: ``Disposes/Disposed`` is intentionally NOT in the verb set above —
# the live press-mill template for it is "LLC Disposed of N Shares in <Co>"
# (verb + ``of`` + number + Shares), which has a different syntactic shape
# (the ``of`` precedes the number rather than terminating the phrase). The
# existing ``_RT_SHARES_BOUGHT_BY`` already handles the trailing-LLC
# "Shares ... disposed by <fund> LLC" form; the alternative leading-LLC
# "disposed of" form is currently not observed in the live noise sample, so
# adding it would be speculative widening rather than evidence-driven.

# "[Wikipedia] <article title>" — the canonical title prefix emitted by
# ``collectors.wikipedia_collector`` for recent-changes mainspace edits. By
# definition encyclopedic reference content, NOT breaking news: a Wikipedia
# article describes a stable subject and is editable by anyone, so a fresh
# revision is rarely tied to a market-moving event. The ML urgency head
# demonstrably over-scores them because the (often ticker-shaped) title plus
# semis-keyword summary tokens trip its high-relevance pattern recognition.
#
# Live evidence (2026-05-23, 7-day articles.db scan): two rows reached
# urgency=2 with ``score_source='ml'``:
#   - ``[Wikipedia] DRAM (musician)`` at ml_score=10.0 — a pure musician
#     disambiguation page, not even semiconductor-related; the urgency head
#     scored max because "DRAM" is a learned semis keyword. The Wikipedia
#     credibility tier (0.60) sits ABOVE the 0.45 ``ALERT_MIN_LONE_SOURCE_CRED``
#     bar so the authority gate doesn't catch it; content type IS the failure.
#   - ``[Wikipedia] Nvidia RTX`` at ml_score=8.6 — a long-standing GPU-product
#     reference page (not a fresh product launch), same over-scoring class.
#
# Discriminator: anchored ``^\s*\[Wikipedia\]\s+`` matches the
# wikipedia_collector's exact tag convention. Critically, the SIBLING
# ``collectors.wikipedia_pageviews`` collector — which IS a useful predictive
# signal ("a 2.5σ surge on $NVDA's Wikipedia article reliably tracks/precedes
# breaking news") — emits titles in a DIFFERENT shape:
# ``"Wiki pageview SURGE NVDA (NVIDIA_Corporation): 12,345 vs 4,567 baseline
#   (z=+3.2, x2.7) 2026-05-23"``. No leading ``[Wikipedia]`` bracket, so the
# pageview-spike signal is preserved verbatim and only the encyclopedic
# recent-changes content is dropped.
#
# Validated against the must-survive corpus: real wire headlines that happen
# to mention Wikipedia mid-sentence ("Wikipedia adds new NVDA reference page
# after IPO", "MU references Wikipedia in 10-K filing") are NOT matched
# because they lack the leading bracketed-source tag.
_RT_WIKIPEDIA_REF = re.compile(
    r"^\s*\[Wikipedia\]\s+",
)

# "<TICKER> Holdings (Raised|Cut|Lowered|Increased|Trimmed|Boosted) by <Fund> LLC"
# — the MarketBeat / americanbankingnews.com / tickerreport.com institutional-
# 13F press mill. Triggered automatically for EVERY 13F filing change against
# any tracked ticker; ramps up dramatically during 13F filing season (Q1/Q2/
# Q3/Q4 +45 days). Pure boilerplate: the "fund X moved its position in
# ticker Y" snippet is generated programmatically with no editorial judgment
# and is by definition retrospective (the filing was already made public on
# the SEC site weeks ago).
#
# Live evidence (2026-05-23, 48h articles.db scan): 4 such rows reached
# urgency=2 with ``score_source='ml'`` — e.g.
#   - "Applied Materials, Inc. $AMAT Holdings Raised by Global Retirement
#      Partners LLC - MarketBeat" (ml_score 9.x)
# The MarketBeat credibility tier (0.68) sits ABOVE the 0.45
# ``ALERT_MIN_LONE_SOURCE_CRED`` bar so the authority gate doesn't catch it;
# content type IS the failure. The ML urgency head systematically over-scores
# them because the title is dense with held-ticker mentions, dollar-sign
# prefixes, and "Holdings"/"Boosted"/"Trimmed" tokens it has learned to
# associate with high relevance.
#
# Discriminator: ``\bholdings\s+(raised|cut|lowered|increased|trimmed|boosted|
# reduced|decreased|sold|acquired)\s+by\s+\S+(?:\s+\S+){0,5}\s+LLC\b``. The
# combination of "Holdings <verb> by" + a multi-word fund name + an LLC
# terminator is the discriminator. The "LLC" anchor is what makes this
# high-precision — real news mentioning "holdings" doesn't end in "by <fund>
# LLC". Validated against the must-survive corpus: "Berkshire Hathaway trims
# AAPL holdings", "Saudi fund increases Tesla stake", "Fund's holdings now
# 12.3M shares" — none match because none have the "<verb> by <fund> LLC"
# trailer.
_RT_HOLDINGS_BY_FUND = re.compile(
    # Original MarketBeat variant: "<TICKER> Holdings <verb> by <fund> LLC".
    r"\bholdings\s+(?:raised|cut|lowered|increased|trimmed|boosted|reduced|"
    r"decreased|sold|acquired)\s+by\s+\S+(?:\s+\S+){0,5}\s+LLC\b",
    re.IGNORECASE,
)
# Sibling MarketBeat variant: "<N> Shares (in|of) <Company> $TICKER (Bought|
# Sold|Acquired|Disposed) by <fund> LLC" — same 13F-press-mill template, just
# rephrased around "Shares Bought" instead of "Holdings Boosted". Live
# evidence (2026-05-23, urgency=1 queue): "100,000 Shares in Oracle
# Corporation $ORCL Bought by Mizuho Markets Americas LLC" (AlphaVantage/
# MarketBeat channel). Same retrospective non-event: a 13F filing change
# already published to SEC weeks ago. Discriminator: "Shares (in|of) ...
# (Bought|Sold|Acquired|Disposed) by ... LLC" — the LLC terminator anchors
# precision (validated zero false positives on the must-survive corpus:
# "AAPL shares bought heavily by institutions", "Berkshire bought 100k MU
# shares", "Insider buys 500k shares of NVDA" all lack the trailing "<fund>
# LLC" structure).
_RT_SHARES_BOUGHT_BY = re.compile(
    r"\bshares\s+(?:in|of)\s+\S+(?:\s+\S+){0,5}\s+"
    r"(?:bought|sold|acquired|disposed|purchased)\s+by\s+"
    r"\S+(?:\s+\S+){0,5}\s+LLC\b",
    re.IGNORECASE,
)

# "Why Are Stock Market Futures Down Today, M/D/YY?" — the TipRanks daily
# pre-market futures-state recap mill template. Same retrospective shape as
# ``_RT_WHY_TRADING`` ("Why X Stock Is Trading Up Today") but the subject is
# the market index futures, not a single ticker. By definition retrospective:
# describes the CURRENT futures state at the time of writing, not a forward-
# looking event the analyst can act on. The date-stamped header is the SEO
# mill signature.
#
# Live evidence (2026-05-23, 48h scan): "Why Are Stock Market Futures Down
# Today, 5/21/26? - TipRanks" reached urgency=2 with ``score_source='ml'``
# (ml_score 9.x). TipRanks credibility tier ~0.65 sits ABOVE the 0.45 bar so
# the authority gate doesn't catch it; content type IS the failure. Same SEO
# mill template family as ``_RT_TODAYS_MOVERS`` ("These Stocks Are Today's
# Movers: ...") and ``_RT_MARKET_TODAY`` ("Stock Market Today, May 18: ...")
# — daily, date-stamped, retrospective.
#
# Discriminator: ``^why\s+are\s+stock\s+market\s+futures\s+(up|down|higher|
# lower|mixed|moving|moved|moving)\s+today\b``. Anchored ``^`` so a mid-
# sentence "futures" reference is unaffected. The trailing "today" is the
# retrospective signature. Validated against the must-survive corpus: "Stock
# futures edge higher ahead of NVDA earnings", "Pre-market: Futures rally
# 0.8%", "Fed minutes due; futures rise" all do NOT match.
_RT_FUTURES_WHY_TODAY = re.compile(
    r"^\s*why\s+are\s+stock\s+market\s+futures\s+"
    r"(?:up|down|higher|lower|mixed|moving|moved|sliding|rising|falling)\s+"
    r"today\b",
    re.IGNORECASE,
)

# "Gold/Petrol/Silver Rate Today in <City> Nth <Month> 2026 : ..." — the
# Business Today (India) daily city-by-city commodity-price feed. Posted
# automatically for every Indian metro every day, naming the commodity, the
# city, the date, and the per-gram or per-litre rate. Not even US-market
# relevant. Pure noise from the analyst's perspective (held book is
# semiconductors / mega-caps, not Indian retail gold).
#
# Live evidence (2026-05-23, 48h scan): 4 such rows reached urgency=2 with
# ``score_source='ml'`` — e.g.
#   - "Gold Rate Today in Kolkata 21st May 2026 : 22 & 24 Carat, Gold Price
#      in Kolkata - Business Today"
#   - "Gold Rate Today in Kanpur 21st May 2026 : 22 & 24 Carat, Gold Price
#      in Kanpur - Business Today"
# Business Today credibility tier (~DEFAULT 0.55) sits ABOVE the 0.45 bar.
# The ML urgency head over-scores these because the title is dense with "Gold
# Rate Today" + price/carat tokens, and the "Today" + date stamp pattern that
# triggered ``_RT_MARKET_TODAY``'s sibling template family.
#
# Discriminator: ``^(gold|silver|petrol|diesel|crude\s+oil)\s+(rate|price)\s+
# today\s+in\s+\S``. The leading commodity-name + "Rate/Price Today in <city>"
# is the SEO mill signature. Anchored ``^`` so a mid-sentence reference is
# unaffected. Validated against the must-survive corpus: real macro/wire
# copy ("Gold prices rally on Fed minutes", "Oil futures gap higher",
# "Silver surges 8% to 14-year high") does NOT match — none lead with the
# bracketed "<commodity> Rate Today in <city>" SEO header.
_RT_DAILY_PRICE_CITY = re.compile(
    r"^\s*(?:gold|silver|petrol|diesel|crude\s+oil)\s+(?:rate|price)\s+"
    r"today\s+in\s+\S",
    re.IGNORECASE,
)

# "What is/'s Next After <X> {Trounces|Crushes|Beats|...} Expectations/Earnings"
# — Baystreet.ca / Investorideas / GuruFocus post-event valuation-question SEO
# mill. By definition retrospective: the print is named ("After NVIDIA Trounces
# Expectations") and the article asks what comes next; the move was already in
# the market when this was written. Same retrospective shape as
# ``_RT_IS_BUY_AFTER`` (post-event valuation question) and
# ``_RT_HERES_WHAT_HAPPENED`` (post-event explainer).
#
# Live evidence (2026-05-23 17:39:55Z, articles.db urgency=2 set): the exact
# row "Baystreet . ca - What is Next After NVIDIA Trounces Expectations" from
# the ``GDELT/baystreet.ca`` collector reached urgency=2 with
# ``score_source='ml'``, ml_score=10.0. Baystreet credibility tier (DEFAULT
# 0.55) sits ABOVE the 0.45 ``ALERT_MIN_LONE_SOURCE_CRED`` bar so the
# source-authority gate doesn't catch it; content type IS the failure. The
# same Baystreet "What is Next After NVIDIA Trounces Expectations" was scored
# repeatedly across the NVDA earnings-week window — a recurring SEO syndication
# the ML urgency head over-scores because "NVIDIA" + "Expectations" + a
# question-form title trip its high-relevance pattern recognition.
#
# Discriminator: ``\bwhat(?:'s|\s+is)?\s+next\s+after\b`` (the "next after"
# bridge is the post-event anchor — a real ongoing question without "after"
# is unaffected) + a post-event terminator from a closed list (``trounces|
# crushes|beats|beat|beating|expectations|earnings|results|report|quarter|
# q[1-4]|miss(?:es|ed)?|ipo|guidance``). The "after" requirement is what
# makes this safe — "What's next for the rally?" / "What's Next for Apple"
# do NOT match because they lack the retrospective anchor. Validated against
# the must-survive corpus: real macro questions ("What's next for the chip
# cycle?", "What comes next after Q3?", "Fed cuts rates 25bp; Powell
# signals more") all do NOT match because none combine the leading
# "What's/Is Next After" bridge with a post-event terminator. The "What
# is Next After Q1 Earnings" form fires, which is also legitimately
# retrospective — Q1 earnings already happened.
#
# Substring (not anchored ^) so the GDELT-style prefix "Baystreet . ca - ..."
# is matched — the actual live failure form. Validated against the live
# noise corpus and the must-survive corpus.
_RT_WHATS_NEXT_AFTER = re.compile(
    r"\bwhat(?:'s|\s+is)?\s+next\s+after\b.*?\b"
    r"(?:trounces?|trounced|crushes?|crushed|beats?|beating|"
    r"expectations?|earnings|results|report|quarter|q[1-4]|"
    r"miss(?:es|ed)?|ipo|guidance)\b",
    re.IGNORECASE,
)

# "Earnings Release: Here's Why Analysts <Cut|Raised|Lowered|...> Their
# <Company> Price Target..." — the SimplyWallSt / scraped-Yahoo-syndication
# SEO-mill post-earnings analyst-action recap template. By definition
# retrospective: the print already happened ("Earnings Release:"), the
# analyst-action has been announced, and the article is summarising what
# the Street did AFTER the fact — there is no forward-looking trade the
# analyst can make from this.
#
# Live evidence (2026-05-24, 30-day articles.db scan): 9 such rows landed
# in the DB; the canonical noisiest copy was
# "Earnings Release: Here's Why Analysts Cut Their The Home Depot, Inc.
#  (NYSE:HD) Price Target To US$378.32" — one reached urgency=2 at
# ``ml_score=9.83`` via the scraped/finance.yahoo.com channel (cred=0.65 —
# well above the 0.45 ``ALERT_MIN_LONE_SOURCE_CRED`` bar, so the
# source-authority gate cannot catch it; content type IS the failure).
# Sibling templates for TELA, KROS, CuriosityStream (CURI), Intuit (INTU)
# show the SimplyWallSt SEO mill posts the same shape for every quarterly
# print across the entire small-cap universe — a recurring, evidence-backed
# noise class.
#
# Discriminator: anchored ``^Earnings Release\s*:\s*Here(?:'s|s)?\s+Why\s+
# Analysts\b`` + analyst-action verb + ``Price Target`` terminator. The
# colon-bounded "Earnings Release:" lead + "Here's Why Analysts" bridge +
# "Price Target" trailer is the SimplyWallSt SEO mill's exact triple
# signature. Validated against the must-survive corpus: real wire copy
# does not use this prefix — "MU price target raised 15% by Citi",
# "Analysts cut their PT on Tesla after Q1 miss", "Q1 Earnings Release
# shows record revenue", "Earnings preview: HD reports tomorrow" all
# do NOT match because none combine the ``^Earnings Release:`` lead
# with the ``Here's Why Analysts <verb>`` bridge AND the ``Price Target``
# terminator.
_RT_EARNINGS_RELEASE_PT = re.compile(
    r"^\s*earnings\s+release\s*:\s*"
    r"here(?:[s'’]+|\s+is)?\s+why\s+analysts\s+"
    r"(?:cut|raised|lowered|increased|boosted|reduced|trimmed|hiked|"
    r"slashed|maintained|reiterated|downgraded|upgraded)\s+.+?"
    r"\bprice\s+target\b",
    re.IGNORECASE,
)

# "<Subject> (TICKER) (Is|Are|Shares|Stock Is) (Up|Down|Higher|Lower) N.N%
# After <event>" — the SUBJECT-LED price-attribution recap mill (MarketBeat /
# simplywall.st / bloomingbit / Yahoo). Sibling of ``_RT_WHY_PCT_AFTER`` and
# ``_RT_WHY_STOCK_IS_AFTER`` which both require a leading ``^Why`` anchor; this
# is the same retrospective shape WITHOUT the ``Why`` lead — the price-attribution
# is stated as fact, not as a question.
#
# Live evidence (2026-05-22..24, 7d articles.db urgency=2 scan):
#   - "D-Wave Quantum (QBTS) Is Up 44.5% After $100 Million Planned Federal
#      Equity Investment - simplywall.st" — Sonnet-scored ai=9.0, fired BREAKING
#   - "NVIDIA (NASDAQ:NVDA) Shares Down 1.9% After Analyst Downgrade -
#      MarketBeat" — ml=9.98 score_source='ml', urgency=2
#   - "Lumentum (NASDAQ:LITE) Shares Down 8.8% After Insider Selling -
#      MarketBeat" — ml=9.63 score_source='ml', urgency=2 (held position!)
#   - "MakeMyTrip (MMYT) Is Down 7.8% After Mixed FY26 Earnings Under Travel
#      Disruptions And Ongoing Investments" — ml=9.87 score_source='ml',
#      urgency=2 (not held, pure SEO mill spam)
#
# All four sources are above the 0.45 ``ALERT_MIN_LONE_SOURCE_CRED`` bar
# (MarketBeat 0.68, simplywall.st defaults to 0.55, Google News 0.62) so the
# authority gate does not catch them; content type IS the failure. By the time
# the headline says "Shares Down 1.9% After Downgrade", the downgrade and the
# move are both in the rear-view — exactly the post-hoc recap shape
# ``_RT_WHY_PCT_AFTER`` catches in its ``^Why`` variant.
#
# Discriminator (the QUINT): subject lead (NOT ``^Why`` — that is the existing
# sibling's territory) ≤6 tokens + state-verb bridge (Shares/Stock + optional
# Is/Was/Now + Up/Down/Higher/Lower OR bare Is/Are/Was/Were + Up/Down/Higher/
# Lower) + explicit ``\d+(?:\.\d+)?%`` move + ``\bafter\b`` retrospective
# anchor. The ≤6 token subject limit is what makes this safe against a real
# news headline that happens to mention "Shares Up N% After Hours" as a footer:
# "Nvidia Beats Estimates With $81.62 Billion in Q1 Revenue; Shares Up 0.2%
# After Hours - bloomingbit" requires 9 leading tokens to reach "Shares" and
# does NOT match — the BREAKING news verb ("Beats Estimates") is at the front,
# and the price-tick clause is the trailer, exactly the shape the analyst
# wants to keep.
#
# Validated against the must-survive corpus: "Nvidia Beats Estimates ...; Shares
# Up 0.2% After Hours" (9 tokens to verb — too far), "Fed cuts rates 50bp;
# stocks up 2% after surprise announcement" (semicolon-separated commentary,
# not subject-verb), "MU revenue rises 22% after Q1 beats consensus" (no
# Is/Are/Shares verb bridge), "Tesla shares jump after earnings beat" (no
# digit %), "Stock futures edge higher ahead of Nvidia earnings" (no digit %),
# "Why Nvidia Stock Is Down 3% After Q1 Earnings" (caught by sibling
# ``_RT_WHY_STOCK_IS_AFTER`` — the ``^Why`` negative-lookahead anchors that
# division of labour) — none match.
#
# Pure read-side: no DB write, no ai_score/ml_score/score_source/urgency
# mutation. All four load-bearing invariants intact by construction.
_RT_SUBJECT_PCT_AFTER = re.compile(
    # NEGATIVE lookahead: not leading with "Why" — those are caught by
    # _RT_WHY_PCT_AFTER / _RT_WHY_STOCK_IS_AFTER. Keeps the gate strictly
    # orthogonal to the existing Why-led siblings.
    r"^(?!\s*why\b)"
    # Subject lead — 1..6 tokens. Lazy quantifier finds the verb bridge as
    # early as possible so a forward-news headline with a trailing footer
    # ("... ; Shares Up 0.2% After Hours") with the verb 9+ tokens in does
    # NOT match.
    r"\s*\S+(?:\s+\S+){0,5}?\s+"
    # State-verb bridge — either:
    #   "<TICKER> Shares Down" / "Stock Is Up" / "Stock Was Higher" / "Stock Now Lower"
    #   "Stock Is Now Up" / "Shares Was Currently Down" — 0..2 modifier tokens
    #   "<Subject> Is Up" / "Are Down" / "Was Higher" / "Were Lower"
    r"(?:"
        r"(?:shares|stock)(?:\s+(?:is|was|now|currently)){0,2}\s+"
        r"(?:up|down|higher|lower)"
        r"|"
        r"(?:is|are|was|were)\s+(?:up|down|higher|lower)"
    r")\s+"
    # Explicit move magnitude required — distinguishes recap from forward news.
    r"\d+(?:\.\d+)?\s*%\s+"
    # Retrospective anchor — same as the Why-led sibling.
    r"after\b",
    re.IGNORECASE,
)

# "<Subject> (TICKER) (Reports|Posts) <Adjective> Earnings/Quarter/Results/Q[1-4]"
# AND "<Subject> Stock Faces Setback Despite ..." AND "<Subject> Exceeds Earnings
# Expectations ..." — the GuruFocus / "GoogleNews/GuruFocus" algorithmic
# post-earnings-recap mill. Triggered automatically for every quarterly print
# across the tracked universe, with a templated qualitative adjective the wires
# never use ("Reports Robust Earnings", "Stock Faces Setback Despite Strong
# Earnings Report"). Same retrospective-recap class as ``_RT_EARNINGS_CALL``
# (post-earnings transcript summary) and ``_RT_HERES_WHAT_HAPPENED`` (post-event
# explainer) — the print has crossed the wire and the move is already in the
# market by the time the GuruFocus mill posts the qualitative summary.
#
# Live evidence (2026-05-23, 7d articles.db urgency=2 scan — all NVDA earnings
# night syndication on the ``GoogleNews/GuruFocus`` / ``GN: Nvidia`` /
# ``GN: earnings`` channels, all above the 0.45 lone-source bar so the
# authority gate doesn't catch them; content type IS the failure):
#   - "NVIDIA (NVDA) Reports Robust Earnings While Valuation Appears At -
#      GuruFocus"           ml=9.98 score_source='ml' (urgency=2)
#   - "NVIDIA (NVDA) Reports Robust Earnings While Valuation Appears At -
#      GuruFocus" (dup)     ml=9.98 score_source='ml' (urgency=2)
#   - "NVIDIA (NVDA) Reports Strong Earnings Amid AI Investment Surge -
#      GuruFocus"           ml=9.26 score_source='ml' (urgency=2)
#   - "NVIDIA (NVDA) Stock Faces Setback Despite Strong Earnings Report -
#      GuruFocus"           ml=9.83 score_source='ml' (urgency=2)
#   - "NVIDIA (NVDA) Exceeds Earnings Expectations with Strong Future O -
#      GuruFocus"           ai=8.00 score_source='llm' (Sonnet over-scored
#                                                       — same SEO mill template)
#
# Discriminator: three orthogonal sub-templates the GuruFocus mill cycles
# through, unified into one regex so all three count as the same fingerprint:
#
#   1. ``\bReports\s+(Robust|Strong|Solid|Mixed|Weak|Modest|Disappointing)\s+
#       (Earnings|Quarter|Results|Q[1-4])\b``
#      Real wires don't use qualitative adjectives — they cite specifics
#      ("Reports Record $81.6B Revenue", "Beats Q1 Estimates by $5B"). The
#      closed adjective list is the SEO-mill signature.
#   2. ``\bStock\s+Faces\s+Setback\s+Despite\b``
#      "Faces Setback Despite" is editorial framing wire copy avoids — real
#      news says "stock falls", "shares slip", "stock drops on weak guidance".
#   3. ``\b(Exceeds|Outperforms?)\s+Earnings\s+Expectations\b``
#      "Exceeds Earnings Expectations" is unusually formal — real wires say
#      "beats estimates", "tops EPS forecast", "blows past consensus".
#
# Substring (not anchored ^) so the trailing "- GuruFocus" attribution and
# any subject lead are tolerated — exactly the live failure shape. Validated
# against the must-survive corpus: "Nvidia Q1 revenue rises 22% to $35.1
# billion", "NVIDIA Reports Record $81.62 Billion Revenue", "Nvidia tops Q1
# earnings", "Nvidia Beats Estimates With $81.62 Billion in Q1 Revenue",
# "Apple shares fall after weak guidance", "AMD reports record Q1 revenue",
# "Lumentum delivers strong quarter, beats EPS estimates" — none match,
# because none use the templated qualitative adjective + earnings-noun pair,
# the "Stock Faces Setback Despite" editorial framing, or the formal
# "Exceeds Earnings Expectations" phrase.
#
# Pure read-side: no DB write, no ai_score/ml_score/score_source/urgency
# mutation. All four load-bearing invariants intact by construction.
_RT_GURUFOCUS_RECAP = re.compile(
    r"\b("
    r"Reports\s+(?:Robust|Strong|Solid|Mixed|Weak|Modest|Disappointing)\s+"
    r"(?:Earnings|Quarter|Results|Q[1-4])"
    r"|"
    r"Stock\s+Faces\s+Setback\s+Despite"
    r"|"
    r"(?:Exceeds|Outperforms?)\s+Earnings\s+Expectations"
    r")\b",
    re.IGNORECASE,
)

# "<Subject> stock (continues|keeps|stays|remains) ... after <event>" — the
# Invezz / CryptoRank / MSN / TradingView post-event continued-state recap
# template. Sibling of ``_RT_SUBJECT_PCT_AFTER`` (which requires an explicit
# move magnitude) — this catches the SAME retrospective shape WITHOUT a
# percent: the title states the stock is in an ongoing post-event state, so
# the move already happened and the article is summarising the lingering
# reaction. By definition retrospective: the "after earnings/guidance/..."
# anchor places the original event BEFORE the article and the state-
# continuation verb names what the stock IS still doing AFTER it.
#
# Live evidence (2026-05-22..25, 7d articles.db scan + alert_recency.db push
# audit): the exact title "Nvidia stock continues to struggle after earnings,
# but analysts remain firmly bullish" fired 4 distinct Discord BREAKING pushes
# in 3 days across 4 syndication channels (Invezz, CryptoRank, MSN,
# TradingView) — all ``score_source='ml'`` ml_score 9.81-9.99 (urgency head
# over-scored the headline because the NVDA + "earnings" + "bullish" tokens
# trip its high-relevance pattern). Each push was the same retrospective
# recap of an already-priced-in move, and ``alert_recency.db`` confirms a
# real push at 2026-05-25T00:57 — exactly the noise complaint the existing
# recap-gate family was built to solve, on a fingerprint no sibling caught:
#
#   * ``_RT_SUBJECT_PCT_AFTER`` requires ``\d+(?:\.\d+)?\s*%`` — this title
#     has no explicit percent, so the gate didn't fire.
#   * ``_RT_WHY_STOCK_IS_AFTER`` requires a leading ``^Why`` — this title
#     leads with the subject (no Why).
#   * ``_RT_WHATS_NEXT_AFTER`` requires "what's/is next after" — different
#     phrasing entirely.
#
# Empirical false-positive bar (2026-05-25 audit, 55,488 titles in last 7
# days): ZERO false positives. The full 30-day urgency=2 set (1340 titles)
# matched only the 4 canonical "Nvidia stock continues to struggle after
# earnings" syndication copies — exactly the target failure, nothing else.
#
# Discriminator (the QUAD): subject lead (≤5 tokens, NOT "Why" — that is the
# existing siblings' territory) + literal ``stock`` token + state-continuation
# verb from a closed list (``continues|keeps|stays|remains``) + ``\bafter\b``
# anchor + earnings/event noun terminator from the established recap-family
# list (``earnings|results|report|quarter|q[1-4]|beat|miss|guidance|downgrade|
# upgrade|announcement|filing|merger|sec|fcc|fda``). The state-continuation
# verb is what distinguishes recap from forward news: real wire copy says
# "stock nears", "stock extends rally", "stock could soar", "stock to watch"
# — none of those verbs are in the closed list.
#
# Validated must-survive corpus (all the recent real pushes that share
# similar token patterns but are forward-looking or describe a fresh event):
#   * "JPMorgan lifts Nvidia target to $280 after record quarter" — analyst
#     action verb ``lifts``, not a state-continuation verb. NO MATCH.
#   * "Bank of America revamps Nvidia stock price target after earnings" —
#     after "stock" comes "price", not a state-continuation verb. NO MATCH.
#   * "All eyes on NVDA stock as Q1 report looms" — after "stock" comes
#     "as", forward-looking. NO MATCH.
#   * "Nvidia stock continues rallying on AI demand" — no "after" + earnings
#     anchor. NO MATCH.
#   * "Apple Stock Could Soar After Earnings" — ``Could Soar`` not in the
#     state-continuation verb list. NO MATCH.
#   * "Apple stock still has room to run after Q1 earnings" — ``still`` is
#     deliberately NOT in the verb list (too ambiguous between forward
#     "still has room" and retrospective "still falling"). NO MATCH.
#
# Pure read-side: no DB write, no ai_score/ml_score/score_source/urgency
# mutation. All four load-bearing invariants intact by construction.
_RT_STOCK_CONTINUES_AFTER = re.compile(
    # NEGATIVE lookahead — leave the ``^Why ...`` form to _RT_WHY_STOCK_IS_AFTER.
    r"^(?!\s*why\b)"
    # Subject lead — 1..5 tokens before the literal ``stock`` token. Lazy so
    # a forward-news headline with a trailing footer doesn't false-match.
    r"\s*\S+(?:\s+\S+){0,4}\s+stock\s+"
    # State-continuation verb — closed list. ``still`` deliberately excluded
    # (too ambiguous; see must-survive corpus comment above).
    r"(?:continues|keeps|stays|remains)\b.*?"
    # Retrospective anchor + earnings/event noun terminator.
    r"\bafter\b.*?\b"
    r"(?:earnings|results|report|quarter|q[1-4]|beat|miss|guidance|"
    r"downgrade|upgrade|announcement|filing|merger|sec|fcc|fda)\b",
    re.IGNORECASE,
)

# "The Zacks Analyst Blog Highlights <TICKERS>" / "Zacks.com featured highlights
# include <TICKERS>" — the Zacks Investment Research recurring SEO blog wrap-up
# template. By definition retrospective: a "Blog Highlights" / "featured
# highlights" post lists tickers the Zacks editorial team is covering THIS WEEK
# / THIS DAY — there is no underlying breaking event, just an editorial roundup.
# Cross-syndicated by every Zacks-republishing channel (yfinance/Zacks,
# YahooFinance/<TICKER>, Finnhub/Yahoo, GoogleNews/* variants) so the same
# template surfaces 4-6 times for the same blog post within hours.
#
# Live evidence (2026-05-26, articles.db 30-day urgency scan): 4 distinct
# urgency=2 (alerted state) rows from this template family today alone, with
# 101 additional urgency=0 rows leaking the same template in 30 days:
#   - 2026-05-26T10:35:20Z ``Finnhub/Yahoo`` "Zacks.com featured highlights
#     include Micron Technology, Murphy USA and Vertiv" ml_score=9.9 urgency=2
#   - 2026-05-26T09:40:31Z ``YahooFinance/NVDA`` "The Zacks Analyst Blog
#     Highlights NVDA, FTEC, VGT, SMH, IYW and XLK" ml_score=9.9 urgency=2
#   - 2026-05-26T09:35:40Z ``yfinance/Zacks`` (same NVDA blog) ml_score=9.83
#     urgency=2
#   - 2026-05-26T08:54:55Z ``yfinance/Zacks`` "Zacks.com featured highlights
#     include Micron Technology, Murphy USA and Vertiv" ml_score=9.5 urgency=2
# All four publishers above the 0.45 ``ALERT_MIN_LONE_SOURCE_CRED`` bar so the
# source-authority gate doesn't catch them; the ML urgency head over-scores
# them because the title is dense with held tickers (NVDA + MU + KLA + LITE
# appear in nearly every Zacks blog highlights post — the model's
# portfolio_flag/ticker_count/ticker_density features fire hard). The
# cross-cycle dedup doesn't merge them because the source-attribution suffix
# (" - The Globe and Mail", " - Yahoo Finance") varies between syndications.
#
# Discriminator: two anchored leads (the Zacks SEO template has two canonical
# title forms; both must be caught):
#   1. ``^The\s+Zacks\s+Analyst\s+Blog\s+Highlights\b`` — the "Analyst Blog
#      Highlights" prefix is the Zacks-specific signature.
#   2. ``^Zacks(?:\.com)?\s+featured\s+highlights\b`` — the "Zacks featured
#      highlights" / "Zacks.com featured highlights" prefix is the alternate
#      Zacks SEO template form. The ``.com`` is optional to handle both
#      published variants.
# Anchored ``^`` so a mid-sentence reference ("Bank of America says Zacks
# highlights NVDA in their year-end note") is NOT caught. Validated against
# the must-survive corpus: real wire copy never leads with this Zacks-specific
# SEO header — "Nvidia reports record Q1", "Zacks raises MU price target to
# $200", "MU added to Zacks Rank 1 list" all do NOT match because none lead
# with "The Zacks Analyst Blog Highlights" or "Zacks(.com) featured highlights".
#
# Defense-in-depth: a byte-identical twin lives in
# ``analysis.claude_analyst._BRIEFING_RT_ZACKS_HIGHLIGHTS`` — the documented
# lockstep, enforced structurally by
# ``test_alert_and_briefing_recap_tuples_have_same_length``.
_RT_ZACKS_HIGHLIGHTS = re.compile(
    r"^\s*(?:"
    r"the\s+zacks\s+analyst\s+blog\s+highlights"
    r"|"
    r"zacks(?:\.com)?\s+featured\s+highlights"
    r")\b",
    re.IGNORECASE,
)

_RECAP_TEMPLATE_PATTERNS = (
    ("why_trading_today", _RT_WHY_TRADING),
    ("why_did_stock", _RT_WHY_DID),
    ("why_just_moved", _RT_WHY_JUST_MOVED),
    ("why_is_pct_since", _RT_WHY_IS_PCT_SINCE),
    # ``why_stock_is_after`` is the strictly more-specific sibling of
    # ``why_pct_after`` (the title has a ``Stock`` token AND a state verb AND
    # an earnings-noun terminator) — must run first so a title like
    # "Why NVDA Stock Is Down 3% After Q1 Earnings" gets the more-precise
    # fingerprint name. The two gates are otherwise mutually compatible:
    # whichever fires, the row is suppressed identically.
    ("why_stock_is_after", _RT_WHY_STOCK_IS_AFTER),
    ("why_pct_after", _RT_WHY_PCT_AFTER),
    # SUBJECT-led sibling of the two Why-led ``...PCT_AFTER`` gates — same
    # retrospective price-attribution shape WITHOUT the leading ``Why`` anchor
    # (MarketBeat / simplywall.st / bloomingbit / Yahoo recap mill). Strictly
    # orthogonal: the regex's ``(?!\s*why\b)`` negative lookahead keeps it
    # mutually exclusive with the Why-led siblings, so whichever the title
    # leads with, exactly one fingerprint fires.
    ("subject_pct_after", _RT_SUBJECT_PCT_AFTER),
    ("market_today_dated", _RT_MARKET_TODAY),
    ("earnings_call_recap", _RT_EARNINGS_CALL),
    ("quick_glance_metrics", _RT_QUICK_GLANCE),
    ("heres_what_happened", _RT_HERES_WHAT_HAPPENED),
    # Present-tense SEO-trailer sibling of ``heres_what_happened``. Distinct
    # regex, distinct fingerprint name so per-template audit counters
    # (``analytics.recap_template_audit`` etc.) bucket each variant separately.
    # See ``_RT_HERES_WHAT_MEANS`` for live evidence.
    ("heres_what_means", _RT_HERES_WHAT_MEANS),
    # Variant-verb sibling — present-tense ``signals`` form of the same SEO
    # trailer template. See ``_RT_HERES_WHAT_SIGNALS`` for live evidence.
    ("heres_what_signals", _RT_HERES_WHAT_SIGNALS),
    # MarketBeat / AlphaVantage 13F press-mill — leading "<Fund> LLC Makes
    # New $X Investment in <Co>" sibling of holdings_by_fund / shares_bought_by.
    ("fund_makes_investment", _RT_FUND_MAKES_INVESTMENT),
    # MarketBeat / AlphaVantage 13F press-mill holdings-CHANGE sibling — same
    # leading-LLC shape as ``fund_makes_investment`` but with the closed set of
    # delta-action verbs (Grows/Trims/Sells/Buys/Has ...) instead of the
    # initial-position verbs (Makes/Acquires/Takes). The two are mutually
    # orthogonal (disjoint verb sets), so a single title is fingerprinted by
    # exactly one of them. Three live ml_score 9+ urgency=1/2 fires in 24h.
    ("fund_stake_delta", _RT_FUND_STAKE_DELTA),
    ("wikipedia_ref", _RT_WIKIPEDIA_REF),
    ("earnings_tomorrow_preview", _RT_EARNINGS_TOMORROW),
    ("todays_movers_list", _RT_TODAYS_MOVERS),
    ("is_buy_after", _RT_IS_BUY_AFTER),
    ("street_thinks", _RT_STREET_THINKS),
    ("gf_value_says", _RT_GF_VALUE),
    ("holdings_by_fund", _RT_HOLDINGS_BY_FUND),
    ("shares_bought_by", _RT_SHARES_BOUGHT_BY),
    ("futures_why_today", _RT_FUTURES_WHY_TODAY),
    ("daily_price_city", _RT_DAILY_PRICE_CITY),
    ("whats_next_after", _RT_WHATS_NEXT_AFTER),
    ("earnings_release_pt", _RT_EARNINGS_RELEASE_PT),
    ("gurufocus_recap", _RT_GURUFOCUS_RECAP),
    # SUBJECT-led post-event continued-state recap — sibling of
    # ``subject_pct_after`` (which requires explicit %); catches the same
    # retrospective shape WITHOUT a percent. The state-continuation verb
    # (continues|keeps|stays|remains) is the discriminator that distinguishes
    # this from forward news with similar tokens. See _RT_STOCK_CONTINUES_AFTER
    # for live evidence (the same "Nvidia stock continues to struggle after
    # earnings" recap pushed 4× across syndication channels in 3 days).
    ("stock_continues_after", _RT_STOCK_CONTINUES_AFTER),
    # Zacks SEO blog-highlights mill — distinct from every other recap gate
    # (no Why-led prefix, no PCT, no earnings-noun terminator). The two
    # canonical lead forms (``The Zacks Analyst Blog Highlights ...`` /
    # ``Zacks(.com) featured highlights include ...``) are the Zacks-specific
    # signature. See ``_RT_ZACKS_HIGHLIGHTS`` for the full live evidence
    # (4 urgency=2 fires today + 101 leaks in 30d).
    ("zacks_highlights", _RT_ZACKS_HIGHLIGHTS),
)


def _looks_like_recap_template(art: dict) -> tuple[bool, str]:
    """``(True, fingerprint_name)`` for a recap/preview/transcript-summary or
    algorithmic-mill title — these are inherently retrospective and must not
    fire a standalone 🚨 BREAKING push. ``(False, "")`` for everything else.

    Pure, side-effect-free; reads only ``title``. Six independent
    fingerprints, anchored so real breaking headlines are never caught
    (validated against the live noise set on 2026-05-18/19 and the
    must-survive corpus: "Nvidia Q3 revenue rises 22%...", "Fed cuts rates",
    "MU earnings blow past estimates", "Why investors are bullish on
    Nvidia", "MU shares halted")."""
    title = (art.get("title") or "").strip()
    if not title:
        return False, ""
    for name, pat in _RECAP_TEMPLATE_PATTERNS:
        if pat.search(title):
            return True, name
    return False, ""


def _filter_recap_template_noise(
    arts: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition urgent rows into ``(kept, suppressed)``. ``suppressed`` is
    the recap/preview/transcript-summary/algorithmic-mill rows the urgency
    head over-scored; everything else is kept. Pure — no DB / IO.

    Runs BEFORE ``dedupe_urgent`` so a recap syndicated across multiple
    feeds ("Stock Market Today, May 18: ..." carried by Motley Fool +
    Nasdaq + YahooFinance) is caught on every copy and the dedup layer
    isn't asked to discriminate against a real story with a similar prefix.

    The suppressed rows are tagged ``_recap_fingerprint`` for log clarity
    so an operator can see WHICH template fired — without dumping the full
    title, mirrors the syndication-dedup tagging discipline."""
    kept: list[dict] = []
    suppressed: list[dict] = []
    for a in arts:
        hit, name = _looks_like_recap_template(a)
        if hit:
            a = dict(a)  # shallow copy — never mutate caller's row
            a["_recap_fingerprint"] = name
            suppressed.append(a)
        else:
            kept.append(a)
    return kept, suppressed


# ── Stocktwits forum chatter gate (defense-in-depth) ───────────────────────
# Raw stocktwits user posts ("$MU lol", "$MU yum", "$MU mooooooooo 🚀") that
# the ML urgency head over-scores into urgency=1 despite carrying no actionable
# news. Distinct surface from quote-widget (which is a price-tape pseudo-row)
# and recap-template (which is a SEO blog template) — this is FORUM CHATTER
# from a single source family.
#
# Live evidence (2026-05-27, articles.db 24h scan):
#   - 2444 raw ``stocktwits`` rows collected (top source by 6x)
#   - 162 reached urgency=1 (95 with title length < 30 chars: "$MU lol",
#     "$MU yum", "$MU mooooooooo", "$MU 1k 😅", "$MU LETS V")
#   - ml_score commonly 9.95–9.97 — the urgency head trained on $TICKER +
#     held-name density features fires hard on any "$MU <anything>" lead
#
# The alert-side ``_filter_low_authority_lone`` gate suppresses the Discord
# push (stocktwits cred=0.30 < 0.45 ALERT_MIN_LONE_SOURCE_CRED), but only
# AFTER the row has reached urgency=1, been fetched by the alert worker, and
# decompressed — burning alert worker cycles, inflating urgent_queue_health
# and ``urgent_score_distribution`` calibration metrics into BORDERLINE_HEAVY
# even when the analyst's actual Discord output is clean. Pre-flooring at the
# ML-scoring stage exits these rows BEFORE they reach urgency=1.
#
# Discriminator: BOTH source AND title-shape conditions must hold —
#   1. ``source`` matches a stocktwits raw-stream family — case-insensitive
#      substring "stocktwits" but NOT the structured ``stocktwits/sentiment``
#      digest source (which carries real signal — see _QW_STOCKTWITS_SENTIMENT,
#      a SEPARATE gate for the sentiment "$NVDA: extreme bullish sentiment ..."
#      rollup pseudo-articles, not the raw user stream).
#   2. ``title`` is short (< 50 chars — chosen against the live noise corpus:
#      95 of the 157 urgency=1 rows in the 24h window were < 30 chars and the
#      30-49 char tier is overwhelmingly more chatter; the 50-79 char tier is
#      mixed so we conservatively stop the gate there).
#   3. ``title`` does NOT contain any real-news keyword (price target /
#      analyst attribution / earnings verb / etc.) — the escape hatch that
#      lets a short "$MU upgraded by Barclays" survive even at length 30.
#
# Same shape and discipline as ``_filter_quote_widget_noise`` /
# ``_filter_recap_template_noise``: pure read-side — no DB write, no
# ai_score / ml_score / score_source / urgency mutation, backtest already
# filtered upstream by _is_synthetic / the store. All four load-bearing
# invariants intact by construction.
_STOCKTWITS_CHATTER_TITLE_MAX = 50
_STOCKTWITS_NEWS_EXIT = re.compile(
    r"\b(?:earnings|beats?|misses?|raised|lowered|upgrad\w*|downgrad\w*|"
    r"price\s+target|guidance|fda|merger|acquir\w+|dividend|"
    r"buyback|recall|investigation|lawsuit|partner(?:ship)?|appointed|"
    r"resign(?:ed|s)?|reuters|bloomberg|cnbc|wsj|barclays|"
    r"goldman|jpmorgan|analyst|reiterat\w+|halted?|approval|approved|"
    r"sec\s+filing|8-?k|10-?k|10-?q|files?\s+|filed\s+)\b",
    re.IGNORECASE,
)


def _looks_like_stocktwits_chatter(art: dict) -> bool:
    """True for a raw stocktwits forum-chatter row — short, unstructured user
    post that the ML urgency head over-scores into urgency=1.

    Source-scoped to the raw stocktwits stream (any tag containing
    ``stocktwits`` case-insensitively) EXCEPT ``stocktwits/sentiment`` which is
    the structured sentiment-digest source carrying real signal. Pure,
    side-effect-free; reads only ``source`` and ``title`` via ``.get()``."""
    source = (art.get("source") or "").strip().lower()
    if "stocktwits" not in source:
        return False
    if source == "stocktwits/sentiment":
        return False
    title = (art.get("title") or "").strip()
    if not title or len(title) >= _STOCKTWITS_CHATTER_TITLE_MAX:
        return False
    if _STOCKTWITS_NEWS_EXIT.search(title):
        return False
    return True


def _filter_stocktwits_chatter(
    arts: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition rows into ``(kept, suppressed)`` where ``suppressed`` is the
    raw stocktwits forum chatter. Mirrors ``_filter_quote_widget_noise`` /
    ``_filter_recap_template_noise`` shape so the three pre-floor surfaces
    behave identically."""
    kept: list[dict] = []
    suppressed: list[dict] = []
    for a in arts:
        (suppressed if _looks_like_stocktwits_chatter(a) else kept).append(a)
    return kept, suppressed


# ── Held-book relevance (the analyst's open positions) ───────────────────────
# The 🚨 BREAKING alert is the analyst's most time-critical product, and the
# persona is explicitly "I depend on this to react to events affecting MY
# positions". Yet the prompt's mandatory PORTFOLIO line relied entirely on
# Sonnet *inferring* held-ticker relevance from the raw headline — a lone
# "Lumentum guides Q4 down" with no "LITE" in the text got a generic PORTFOLIO
# line, and a real held-name break read identically to generic macro colour.
# This surfaces the held tickers explicitly in the alert input, exactly like
# the briefing's well-tested ``[BOOK: ...]`` tag (tests/test_briefing_book_tag),
# so the two consumed products judge "touches the book" the same way (the
# documented cross-product anti-drift discipline). Reuses ml.features'
# LIVE_PORTFOLIO_TICKERS / _LIVE_RE verbatim — alert_agent already imports
# _source_credibility from that module, so this is a single source of truth
# with the model's own ticker features and can never drift from a local copy.
# Pure read-side: no DB write, no ai_score/ml_score/score_source/urgency touch,
# backtest rows already filtered by _is_synthetic / the store — only the prompt
# text Sonnet reads is enriched. All four load-bearing invariants intact.
def _book_tickers(art: dict) -> list[str]:
    """Held/watchlist tickers mentioned in an urgent row's title+summary.

    Sorted (deterministic, stable cycle-to-cycle), de-duplicated; ``[]`` when
    the row touches no held name. Matches on ``title + summary`` — the SAME
    surface as the briefing's ``_book_tickers`` and ml.features ticker density,
    so an alert and the 5h digest never disagree about whether a wire touches
    the book. Pure, side-effect-free; reads only ``title``/``summary`` via
    ``.get()`` and never mutates the article."""
    blob = f"{art.get('title') or ''} {art.get('summary') or ''}"
    if not blob.strip():
        return []
    hits = {m.upper() for m in _LIVE_RE.findall(blob)}
    if not hits:
        return []
    # hits ⊆ LIVE_PORTFOLIO_TICKERS by construction (the regex only matches
    # those literals); sorted() gives a stable, test-pinnable order.
    return sorted(hits)


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
    # Emergency kill switch. Default is enabled; set URGENT_ALERTS_FORCE_OFF=1
    # to hard-stop paging without removing the alert worker.
    if os.environ.get("URGENT_ALERTS_FORCE_OFF", "0").strip().lower() not in {"0", "false", "no", "off", ""}:
        _log.warning("[alert] URGENT_ALERTS_FORCE_OFF active — refusing to page")
        return False
    if not urgent_articles:
        return False
    if not DISCORD_WEBHOOK and not _openclaw_cli_path():
        _log.warning("[alert] No DISCORD_WEBHOOK_URL and no OpenClaw CLI — skipping")
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

    # Quote-widget noise gate (defense-in-depth, same shape as the synthetic
    # re-filter above). Runs before dedup. Suppressed rows are marked alerted
    # UNCONDITIONALLY here — regardless of whether Discord later fires — so a
    # spaceless price-tick that the urgency head over-scored exits the urgent
    # queue instead of being re-fetched and re-evaluated every 20s cycle. They
    # stay in articles.db untouched (training reads ai_score; ml_score /
    # score_source unchanged) — only the noisy push is dropped.
    filtered, qw_suppressed = _filter_quote_widget_noise(filtered)
    if qw_suppressed:
        try:
            store.mark_alerted_batch(alerted_ids(qw_suppressed))
        except Exception:
            _log.exception(
                "[alert] failed to mark quote-widget rows alerted"
            )
        srcs = ", ".join(
            f"{(a.get('source') or '?')}:{(a.get('title') or '')[:40]}"
            for a in qw_suppressed[:5]
        )
        _log.info(
            f"[alert] suppressed {len(qw_suppressed)} quote-widget pseudo-"
            f"article(s) (live ticker-tape, not breaking news) — {srcs}"
        )
    if not filtered:
        _log.info("[alert] all urgent rows were quote-widget noise — skipping")
        return False

    # Recap / SEO template gate (defense-in-depth, same shape as the quote-
    # widget gate above). "Why X Stock Is Trading Up Today" / "Stock Market
    # Today, May 18:" / "Q1 2026 Earnings Call Highlights" / GuruFocus's
    # "GF Value Says ..." are inherently retrospective or algorithmic mill
    # output the urgency head over-scores — the move was already in the
    # market by the time the recap was written. Suppressed rows are marked
    # alerted UNCONDITIONALLY so they exit the urgent queue instead of being
    # re-fetched every 20s cycle. They stay in articles.db untouched (ai_score
    # / ml_score / score_source unchanged) — only the noisy push is dropped.
    # Runs BEFORE dedup so a recap syndicated across multiple feeds ("Stock
    # Market Today, May 18: ..." live-evidenced three copies at the same
    # minute from Motley Fool + Nasdaq + YahooFinance) is caught on every
    # copy. Best-effort store mark — a store failure must never block a
    # genuine fresh alert in the same batch.
    filtered, recap_suppressed = _filter_recap_template_noise(filtered)
    if recap_suppressed:
        try:
            store.mark_alerted_batch(alerted_ids(recap_suppressed))
        except Exception:
            _log.exception(
                "[alert] failed to mark recap-template rows alerted"
            )
        fps = ", ".join(
            f"{a.get('_recap_fingerprint','?')}:{(a.get('title') or '')[:60]}"
            for a in recap_suppressed[:5]
        )
        _log.info(
            f"[alert] suppressed {len(recap_suppressed)} recap-template "
            f"row(s) (retrospective recap / SEO mill / transcript summary — "
            f"never breaking) — {fps}"
        )
    if not filtered:
        _log.info(
            "[alert] all urgent rows were recap-template noise — skipping"
        )
        return False

    # Drop articles older than 24 hours — stale news must not fire as breaking.
    # Suppressed (stale / un-datable) rows are marked alerted UNCONDITIONALLY
    # here — identical shape to the quote-widget gate above and the
    # low-authority / cross-cycle gates below — so a row get_unalerted_urgent
    # legitimately returns (recent first_seen) but whose `published` is >24h
    # old EXITS the urgent queue instead of being re-fetched and re-dropped
    # every 20s cycle until its first_seen ages out, then lingering forever as
    # a permanent urgency=1 residue (observed live 2026-05-18: 26 such rows
    # stuck 5 days, inflating the dashboard `urgent` tile and re-decompressed
    # every cycle). A stale-by-`published` row only ages further — it can
    # never become a valid fresh alert — so marking it loses no delivery; a
    # no-parseable-date row likewise can never pass _article_age_ok, so it too
    # must exit rather than churn forever. articles.db ai_score / ml_score /
    # score_source are untouched (mark_alerted_batch only sets urgency=2) and
    # synthetic rows were already filtered above — all four load-bearing
    # invariants intact. Best-effort mark (a store failure must never block a
    # genuine fresh alert in the same batch).
    fresh: list[dict] = []
    stale: list[dict] = []
    for a in filtered:
        (fresh if _article_age_ok(a) else stale).append(a)
    if stale:
        try:
            store.mark_alerted_batch(alerted_ids(stale))
        except Exception:
            _log.exception("[alert] failed to mark stale rows alerted")
        srcs = ", ".join(
            f"{(a.get('source') or '?')}:{(a.get('title') or '')[:40]}"
            for a in stale[:5]
        )
        _log.info(
            f"[alert] dropped {len(stale)} stale article(s) "
            f"(>24h general / >{ALERT_EARNINGS_MAX_AGE_HOURS:g}h earnings) — {srcs}"
        )
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

    # Source-authority gate (defense-in-depth, same shape as _is_synthetic /
    # _article_age_ok — a formatter-side drop, NOT an ML-threshold change). A
    # lone, un-corroborated social/forum post the urgency head over-scored must
    # not fire a standalone Bloomberg "🚨 BREAKING" alert: that is the single
    # biggest noise complaint from the analyst consuming this channel. A
    # syndicated story (dup_count>1) or any credible/unknown source still
    # fires. Suppressed rows are NOT lost — they stay in articles.db (training
    # reads ai_score, untouched; score_source/ml_score untouched) and Opus can
    # still surface them in the curated 5h briefing — only the noisy *push* is
    # dropped. They are marked alerted UNCONDITIONALLY (separate call, before
    # the Discord attempt, regardless of its outcome) so they exit the urgent
    # queue instead of being re-fetched and re-evaluated every 20s cycle.
    deduped, suppressed = _filter_low_authority_lone(deduped)
    if suppressed:
        try:
            store.mark_alerted_batch(alerted_ids(suppressed))
        except Exception:
            _log.exception(
                "[alert] failed to mark suppressed low-authority rows alerted"
            )
        srcs = ", ".join(
            f"{(a.get('source') or '?')}:{(a.get('title') or '')[:40]}"
            for a in suppressed[:5]
        )
        _log.info(
            f"[alert] suppressed {len(suppressed)} lone low-authority urgent "
            f"row(s) (cred<{ALERT_MIN_LONE_SOURCE_CRED}, no syndication) — "
            f"{srcs}"
        )
    if not deduped:
        _log.info(
            "[alert] all urgent rows were lone low-authority noise — skipping"
        )
        return False

    # Cross-cycle syndication suppression (defense-in-depth, same
    # formatter-side shape as _is_synthetic / _filter_quote_widget_noise /
    # _filter_low_authority_lone — NOT an ML-threshold change). dedupe_urgent
    # only collapses copies *inside this batch*; a slower feed re-collecting
    # an already-alerted event as a NEW row (urgency=1, the old copies are
    # urgency=2 and excluded from get_unalerted_urgent) would otherwise fire a
    # second standalone BREAKING push for an event the analyst was already
    # told about, possibly hours later (live: the H200/China story alerted
    # twice ~1.5h apart from different sources). Runs AFTER dedupe_urgent so a
    # signature is computed once per collapsed survivor, and after the
    # low-authority gate so a row suppressed there is never recorded. Best-
    # effort: a recency-store failure yields an empty set → no-op (the old
    # behaviour) — a genuine breaking story must still reach the analyst.
    # Suppressed rows are marked alerted UNCONDITIONALLY (separate call,
    # before any Discord attempt) so they leave the urgent queue instead of
    # being re-fetched every 20s. articles.db ai_score/ml_score/score_source
    # are untouched — only the noisy duplicate push is dropped.
    try:
        recent_sigs = alert_recency.recent_signatures()
    except Exception:
        recent_sigs = set()
    deduped, cross_suppressed = alert_recency.partition_already_alerted(
        deduped, recent_sigs
    )
    if cross_suppressed:
        try:
            store.mark_alerted_batch(alerted_ids(cross_suppressed))
        except Exception:
            _log.exception(
                "[alert] failed to mark cross-cycle duplicate rows alerted"
            )
        srcs = ", ".join(
            f"{(a.get('source') or '?')}:{(a.get('title') or '')[:40]}"
            for a in cross_suppressed[:5]
        )
        _log.info(
            f"[alert] suppressed {len(cross_suppressed)} cross-cycle "
            f"duplicate(s) (signature alerted within "
            f"{alert_recency.ALERT_RECENCY_TTL_HOURS:.0f}h) — {srcs}"
        )
    if not deduped:
        _log.info(
            "[alert] all urgent rows were cross-cycle duplicates — skipping"
        )
        return False

    # Paraphrase-tolerant cross-cycle suppression (defense-in-depth, runs AFTER
    # the exact-signature suppression above and BEFORE the continuation
    # annotation below — same formatter-side shape as every other alert gate).
    # Live evidence (2026-05-20, alert_recency.db 12h audit): the "Union calls
    # strike at South Korea" wire fired a "S. Korea" abbreviated variant FIRST
    # at 04:26Z, then the "South Korea" spelling at 05:28Z — Jaccard 0.86
    # between the two canonical sigs, but the exact-sig mismatch let the
    # second standalone 🚨 BREAKING push through to the analyst. partition_
    # paraphrase_alerted catches this with a strict ≥0.75 Jaccard + ≥4 shared
    # salient tokens bar — well above the antonym-flip ceiling (0.50-0.667)
    # so an opposite-direction story is provably never merged. Suppressed
    # rows are marked alerted UNCONDITIONALLY (separate call, before any
    # Discord attempt) so they leave the urgent queue instead of being
    # re-fetched every 20s. articles.db ai_score/ml_score/score_source are
    # untouched — only the noisy paraphrase push is dropped. Best-effort:
    # any failure in paraphrase_match degrades silently to the prior
    # (exact-sig-only) behaviour — a missed alert is far worse than a
    # duplicate.
    try:
        para_recent = alert_recency.recent_alerts()
    except Exception:
        para_recent = []
    if para_recent:
        try:
            deduped, para_suppressed = alert_recency.partition_paraphrase_alerted(
                deduped, para_recent
            )
        except Exception:
            _log.exception(
                "[alert] paraphrase suppression failed — degrading to exact-sig only"
            )
            para_suppressed = []
    else:
        para_suppressed = []
    if para_suppressed:
        try:
            store.mark_alerted_batch(alerted_ids(para_suppressed))
        except Exception:
            _log.exception(
                "[alert] failed to mark paraphrase duplicate rows alerted"
            )
        # Log shows WHICH paraphrase fired so an operator can audit the gate.
        notes = []
        for a in para_suppressed[:5]:
            m = a.get("_paraphrase_match") or {}
            notes.append(
                f"{(a.get('source') or '?')}:{(a.get('title') or '')[:40]} "
                f"≈{m.get('jaccard','?')} of '{(m.get('title') or '')[:40]}'"
            )
        _log.info(
            f"[alert] suppressed {len(para_suppressed)} paraphrase "
            f"duplicate(s) (Jaccard >= "
            f"{alert_recency.PARAPHRASE_MIN_JACCARD} of a prior alert "
            f"within {alert_recency.ALERT_RECENCY_TTL_HOURS:.0f}h) — "
            f"{'; '.join(notes)}"
        )
    if not deduped:
        _log.info(
            "[alert] all urgent rows were paraphrase duplicates — skipping"
        )
        return False

    # Only the first ALERT_BATCH_SIZE feed the prompt — and only those (plus the
    # duplicates they absorbed) get marked alerted. Marking the entire urgent
    # list would silently drop the tail (it'd never be picked up next cycle),
    # so we cap both ends.
    batch = deduped[:ALERT_BATCH_SIZE]

    # Continuation context (non-suppressing). Cross-cycle suppression above
    # already dropped EXACT-signature repeats; what survives can still be a
    # *different* headline about a story the analyst was already pushed (live:
    # the UAE-strike alert at 01:55 then a Brent/markets follow-up — distinct
    # signatures, correctly NOT collapsed). With no hint the LLM writes the
    # follow-up as if it just broke → the analyst's top "duplicate alerts"
    # complaint, on the one product (the standalone push) that never got the
    # mitigation the briefing's [ALERTED] tag gave. Annotate (never drop) each
    # survivor with the related prior alert so the prompt's CONTINUITY rule
    # frames it as a developing update. Best-effort: a recency-store failure
    # yields [] → no annotation → exact pre-feature behaviour (a genuine alert
    # must always still fire). Read-only: alert_recency.db only, NEVER
    # articles.db — no ai_score/ml_score/score_source/urgency touch, backtest
    # already filtered above. All four invariants intact by construction.
    try:
        _recent = alert_recency.recent_alerts()
    except Exception:
        _recent = []
    if _recent:
        for a in batch:
            try:
                rel = alert_recency.related_prior_alert(a.get("title"), _recent)
            except Exception:
                rel = None
            if rel:
                a["_related_prior"] = rel

    # Per-held-ticker mention velocity. For each held ticker mentioned by any
    # row in the batch, count distinct article mentions in the last 60min via
    # the canonical ``ArticleStore.ticker_mention_velocity`` primitive (already
    # ``_LIVE_ONLY_CLAUSE``-scoped — synthetic backtest/opus rows can never
    # inflate the count, CLAUDE.md §5). The analyst's persona is "react to
    # events affecting MY positions"; today the ``book:`` line names WHICH
    # held tickers touch this wire but not "is this part of a multi-mention
    # surge or a lone event?". A standalone alert on a held name with 4 other
    # recent mentions is materially different (the wire is concentrating on
    # that name — bigger event, momentum trade) from one with zero other
    # mentions (an isolated headline). Single batched call covering every
    # held ticker in the batch (not per-row) so DB cost is one query per
    # alert cycle. Best-effort: any failure (mock store without the method,
    # locked DB, …) degrades to no annotation — never blocks an alert. Pure
    # read-side: no DB write, no ai_score / ml_score / score_source /
    # urgency mutation. Pinned by ``tests/test_alert_book_velocity.py``.
    velocity_map: dict[str, dict] = {}
    all_book_tickers = sorted({t for a in batch for t in _book_tickers(a)})
    if all_book_tickers:
        try:
            v_rows = store.ticker_mention_velocity(
                all_book_tickers, window_min=60
            )
            velocity_map = {
                r["ticker"]: r for r in v_rows if isinstance(r, dict)
            }
        except Exception:
            _log.exception(
                "[alert] ticker_mention_velocity failed — degrading"
            )
            velocity_map = {}

    # Per-held-ticker BREAKING-alert burst counts. ``velocity_map`` above counts
    # COLLECTED article mentions (every wire copy in articles.db); the analyst
    # cares about how many BREAKING pushes they've already received for the
    # same name — a different and more important signal. Pure read of
    # ``alert_recency.db`` (same source ``_related_prior`` and paraphrase
    # suppression already use, so adding this carries the same import-safety
    # profile and never touches articles.db / the four load-bearing
    # invariants). Best-effort: a recency-DB failure yields {} → no annotation
    # → exact pre-feature behaviour (a genuine alert must still fire). The
    # threshold (>= ``BURST_MIN_PRIOR_ALERTS``) is conservative — a single
    # prior is silently a lone event, two is borderline, three or more is the
    # wire-concentration pattern the BURST WIRE prompt rule frames. Pinned by
    # ``tests/test_alert_ticker_burst.py``.
    burst_counts: dict[str, int] = {}
    if all_book_tickers:
        try:
            burst_counts = alert_recency.ticker_burst_counts(
                _recent or [], all_book_tickers
            )
        except Exception:
            _log.exception(
                "[alert] ticker_burst_counts failed — degrading"
            )
            burst_counts = {}

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
        # Verified-vs-model-only calibration tag. ``_llm_vetted`` is set by
        # ``ArticleStore.get_unalerted_urgent``: True = a real Opus/Sonnet
        # ai_score, False = the displayed score came from ml_score only (an
        # UNVERIFIED local-model urgent call; the urgency head demonstrably
        # over-scores recap-template/forum/wiki rows that the LLM would label
        # noise). Only an explicit False tags — a row from a non-canonical
        # caller missing the key (.get → None, ``is False`` → False) is NEVER
        # tagged, mirroring the briefing's [model] tag discipline (66c349f).
        # The tag does NOT suppress: this row already passed every gate and is
        # going to fire; it only augments the prompt context so Sonnet hedges
        # CONTEXT/IMPACT honestly instead of stating an unverified urgent call
        # as confirmed magnitude. Read-only — no DB write, no
        # ai_score/ml_score/score_source/urgency touch, backtest already
        # filtered upstream by _is_synthetic — invariants intact.
        verified_tag = (
            " [unverified — model-only urgent]"
            if a.get("_llm_vetted") is False else ""
        )
        block = (
            f"[score={score:.0f}]{verified_tag} {title}\n"
            f"source: {source}\nurl: {link}"
        )
        # Freshness context: a news analyst reacting to "🚨 BREAKING" must be
        # able to tell a 4-minute-old 8-K (act now) from a 16-hour-old reused
        # headline (already priced in). The 0..24h band reaching here is wide
        # — the Sonnet prompt's RECENCY rule turns this into honest framing.
        age = _article_age_str(a)
        if age:
            block += f"\nage: {age} (time since publication)"
        dup_count = int(a.get("dup_count") or 1)
        if dup_count > 1:
            # Tell the alert LLM how broadly the story is being carried — wide
            # syndication is itself a signal of how big the event is.
            block += f"\nsyndication: reported by {dup_count} sources"
        # Held-book relevance — drives the prompt's BOOK rule so the mandatory
        # PORTFOLIO line names the analyst's actual open risk instead of Sonnet
        # guessing ticker relevance from the headline alone. Same shape as the
        # additive age/syndication/related lines; same concept as the briefing's
        # [BOOK:] tag (cross-product parity). Read-only — see _book_tickers.
        book = _book_tickers(a)
        if book:
            try:
                open_syms = set(_open_book_tickers())
            except Exception:
                open_syms = set()
            open_hits = [t for t in book if t in open_syms]
            watch_hits = [t for t in book if t not in open_syms]
            block += f"\nbook: {','.join(book)}"
            if open_hits:
                block += (
                    f"\nbook_open: {','.join(open_hits)} — OPEN held risk; "
                    f"PORTFOLIO MUST give a concrete directional implication"
                )
            if watch_hits:
                block += (
                    f"\nbook_watch: {','.join(watch_hits)} — watchlist only; "
                    f"do NOT claim these are held positions"
                )
            if open_hits and not watch_hits:
                block += (
                    " — analyst HOLDS these; the PORTFOLIO line MUST give a "
                    "concrete directional implication for them"
                )
            # Per-held-ticker mention velocity hint. Multiple recent
            # mentions = a wire concentrating on that name (momentum / event
            # cluster), so the IMPACT magnitude should reflect that. Single
            # mention (no velocity line) means this is a lone event. Pure
            # read-side annotation; the velocity_map was computed above.
            # >=2 mentions in 60min is the conservative discriminator —
            # one mention is THIS alert itself, two means at least one other
            # recent article hit the same name (the analyst persona's "is
            # this part of a wire surge?" question is genuinely answered).
            velocity_notes: list[str] = []
            for t in book:
                v = velocity_map.get(t)
                if not isinstance(v, dict):
                    continue
                try:
                    recent = int(v.get("recent") or 0)
                except (TypeError, ValueError):
                    continue
                if recent >= 2:
                    velocity_notes.append(
                        f"{t}: {recent} mentions in last 60min"
                    )
            if velocity_notes:
                block += (
                    f"\nbook_velocity: {'; '.join(velocity_notes)} — "
                    f"weight IMPACT magnitude accordingly (wire is "
                    f"concentrating on these held names)"
                )
            # Cross-cycle BREAKING-alert burst hint. Drives the prompt's
            # BURST WIRE rule. Only the held tickers on THIS row that have
            # already triggered >= BURST_MIN_PRIOR_ALERTS standalone pushes
            # appear here; rows below the bar emit no burst line (silent —
            # the BOOK / BOOK VELOCITY framing stays unchanged for a normal
            # lone or low-velocity event). Pure read of the pre-computed
            # burst_counts map (alert_recency.db); the row itself is never
            # mutated and articles.db / the four invariants are untouched.
            burst_notes: list[str] = []
            for t in book:
                c = burst_counts.get(t, 0)
                if c >= BURST_MIN_PRIOR_ALERTS:
                    burst_notes.append(
                        f"{t}: {c} prior BREAKING alerts in last "
                        f"{int(round(alert_recency.ALERT_RECENCY_TTL_HOURS))}h"
                    )
            if burst_notes:
                block += (
                    f"\nburst: {'; '.join(burst_notes)} — analyst has "
                    f"already been pushed these; frame THIS as the next "
                    f"development in an active wire (DETAILS/ADDS/NOW/"
                    f"FOLLOWS/EXTENDS), not a fresh break"
                )
        rel = a.get("_related_prior")
        if isinstance(rel, dict) and (rel.get("title") or "").strip():
            # Drives the prompt's CONTINUITY rule — a related 🚨 alert already
            # fired, so this is a developing update, not a fresh break.
            rh = float(rel.get("age_hours") or 0.0)
            if rh < 1.0:
                rage = f"{int(round(rh * 60))}m"
            elif rh < 10.0:
                rage = f"{rh:.1f}h"
            else:
                rage = f"{int(round(rh))}h"
            block += (
                f"\nrelated: a 🚨 BREAKING alert fired ~{rage} ago on a "
                f"related developing story — \"{rel['title'][:140]}\" "
                f"(frame THIS as a continuation/update, not a fresh break)"
            )
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
    prompt = ALERT_PROMPT.format(
        articles_text=articles_text,
        now_utc=now_utc,
        held_book=_held_book_phrase(),
    )

    try:
        # HARD RULE: every page must pass a Grok urgency gate first.
        # No deterministic fallback paging. If Grok is down or says no, drain
        # and do not ping.
        gate = _grok_urgency_gate(batch)
        if not gate.get("urgent"):
            try:
                store.mark_alerted_batch(alerted_ids(batch))
            except Exception:
                _log.exception("[alert] failed to drain Grok-rejected batch")
            _log.info(
                "[alert] Grok gate rejected page — drained %d row(s) (%s)",
                len(batch),
                gate.get("reason") or "not urgent",
            )
            return True

        keep = gate.get("keep") or []
        if not keep:
            try:
                store.mark_alerted_batch(alerted_ids(batch))
            except Exception:
                _log.exception("[alert] failed to drain empty keep set")
            return True

        # Drop non-approved siblings from the queue so they cannot re-spam.
        rejected = [a for a in batch if a not in keep]
        if rejected:
            try:
                store.mark_alerted_batch(alerted_ids(rejected))
            except Exception:
                _log.exception("[alert] failed to drain Grok-rejected siblings")

        # Prefer Grok BN formatter on the approved subset; if formatter fails,
        # still page using the gate headline/reason (already Grok-approved).
        keep_text = "\n\n".join(
            t for a, t in ((a, _fmt(a)) for a in keep) if t is not None
        )
        message = None
        if keep_text:
            fmt_prompt = ALERT_PROMPT.format(
                articles_text=keep_text,
                now_utc=now_utc,
                held_book=_held_book_phrase(),
            )
            message = claude_call(fmt_prompt, model=SONNET_MODEL, timeout=60)
        if not message:
            message = _format_gated_breaking_message(
                keep,
                headline=gate.get("headline") or "",
                reason=gate.get("reason") or "",
                now_utc=now_utc,
            )
        batch = keep
        if not message:
            _log.warning("[alert] Grok-approved but empty message body — skipping send")
            return False

        page = _should_page_breaking(message, batch)
        message = _with_breaking_ping(message, page=page, batch=batch)
        if not page:
            _log.info(
                "[alert] quiet BREAKING (no page): off-book non-actionable "
                f"impact={_alert_impact_side(message) or 'n/a'}"
            )

        # post via discord_notifier which also fires TTS
        # is_alert=True only when paging — quiet WATCH colour should not
        # trigger alert-priority TTS on the laptop.
        from notifier.discord_notifier import send as discord_send
        ok = discord_send(message, is_alert=page)

        if ok:
            if page:
                try:
                    _send_sao_breaking_dm(message)
                except Exception:
                    _log.exception("[alert] Sao DM fanout failed")
            else:
                _log.info("[alert] Sao DM skipped: quiet non-page BREAKING")
            # Bulk-mark in one transaction; previous code took the write lock
            # N times (5 round-trips for the default batch size). alerted_ids
            # includes the syndicated copies merged into the batch, so they
            # never re-fire — duplicates of still-queued stories stay urgent.
            mark_ids = alerted_ids(batch)
            store.mark_alerted_batch(mark_ids)
            # Record the canonical signature of every story that actually
            # fired so a later slower-feed copy of the same event is
            # cross-cycle-suppressed above. Best-effort — a failure here only
            # means a future duplicate is not muted (never worse than the
            # pre-feature behaviour); it must never undo a sent alert.
            try:
                alert_recency.record_alerted(batch)
            except Exception:
                _log.exception("[alert] alert_recency.record_alerted failed")
            collapsed = len(mark_ids) - len(batch)
            tail = len(deduped) - len(batch)
            notes = []
            if collapsed > 0:
                notes.append(f"{collapsed} syndicated dupes folded in")
            if tail > 0:
                notes.append(f"{tail} more queued")
            notes.append("grok-gated")
            note = f" ({'; '.join(notes)})" if notes else ""
            _log.info(f"[alert] BN alert sent ({len(batch)} distinct stories){note}")
        else:
            _log.warning("[alert] Discord POST failed")
        return ok

    except Exception:
        _log.exception("[alert] Error sending urgent alert")
        return False
