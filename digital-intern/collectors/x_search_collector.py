"""Live X/Twitter intern source via xAI x_search (no official X API).

Nitter mirrors are dead (0 tweets). This collector is the live path: it
POSTs to the xAI Responses API with the server-side ``x_search`` tool,
scoped to ``config/sources.json`` ``twitter_accounts``, and emits the same
article dicts other collectors feed into ``daemon._ingest``.

Auth is the shared intern Grok token (env or OpenClaw SQLite). Never log it.
"""
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_PATH = BASE_DIR / "config" / "sources.json"
SOURCE_TAG = "twitter_xsearch"
USER_AGENT = "DigitalInternXSearch/1.0 (+local paper research desk)"

XAI_API_BASE = os.environ.get(
    "DIGITAL_INTERN_XAI_API_BASE",
    os.environ.get("PAPER_TRADER_XAI_API_BASE", "https://api.x.ai/v1"),
).rstrip("/")
MODEL = os.environ.get(
    "DIGITAL_INTERN_X_SEARCH_MODEL",
    os.environ.get("DIGITAL_INTERN_LLM_MODEL", "grok-4.6"),
)
HTTP_TIMEOUT_S = int(os.environ.get("DIGITAL_INTERN_X_SEARCH_TIMEOUT_S", "180"))
MAX_TWEETS = int(os.environ.get("DIGITAL_INTERN_X_SEARCH_MAX_TWEETS", "24"))
LOOKBACK_HOURS = int(os.environ.get("DIGITAL_INTERN_X_SEARCH_LOOKBACK_HOURS", "24"))

_TWEET_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:x\.com|twitter\.com)/"
    r"(?:i/web/status|i/status|[^/\s]+/status)/(\d+)",
    re.IGNORECASE,
)
_HANDLE_FROM_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:x\.com|twitter\.com)/([^/\s]+)/status/\d+",
    re.IGNORECASE,
)

_DEFAULT_ACCOUNTS = [
    "KobeissiLetter",
    "Forbes",
    "business",
    "Reuters",
    "zerohedge",
    "WSJmarkets",
]


def _ssl_context():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def normalize_handle(raw: str) -> str:
    handle = (raw or "").strip()
    if handle.startswith("@"):
        handle = handle[1:]
    handle = handle.split("/")[-1].strip()
    return handle


def load_twitter_accounts(path: Path | None = None) -> list[str]:
    """Read ``twitter_accounts`` from sources.json; fall back to the known set."""
    src = Path(path) if path is not None else SOURCES_PATH
    accounts: list[str] = []
    try:
        raw = json.loads(src.read_text())
        listed = raw.get("twitter_accounts") if isinstance(raw, dict) else None
        if isinstance(listed, list):
            accounts = [normalize_handle(str(x)) for x in listed]
    except Exception as e:
        print(f"[x_search] sources.json unreadable ({src}): {e}")
    accounts = [a for a in accounts if a and re.fullmatch(r"[A-Za-z0-9_]{1,15}", a)]
    if not accounts:
        accounts = list(_DEFAULT_ACCOUNTS)
    # x_search allowed_x_handles max is 20
    seen: set[str] = set()
    out: list[str] = []
    for a in accounts:
        key = a.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
        if len(out) >= 20:
            break
    return out


def _lookback_date() -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=max(1, LOOKBACK_HOURS))
    return dt.date().isoformat()


def build_x_search_payload(accounts: list[str], *, from_date: str | None = None) -> dict:
    handles = [normalize_handle(a) for a in accounts if normalize_handle(a)]
    if not handles:
        handles = list(_DEFAULT_ACCOUNTS)
    mentioned = ", ".join(f"@{h}" for h in handles)
    prompt = (
        f"Latest market-moving X posts from {mentioned} in the last "
        f"{LOOKBACK_HOURS} hours. Return ONLY a JSON array of objects with "
        "handle, text, url, published, tickers. JSON only."
    )
    return {
        "model": MODEL,
        "input": [
            {"role": "user", "content": prompt},
        ],
        "tools": [
            {
                "type": "x_search",
                "allowed_x_handles": handles,
                "from_date": from_date or _lookback_date(),
            }
        ],
    }


def _extract_response_text(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()
    chunks: list[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {
                        "output_text",
                        "text",
                    }:
                        val = (part.get("text") or "").strip()
                        if val:
                            chunks.append(val)
            elif isinstance(content, str) and content.strip():
                chunks.append(content.strip())
            text_val = item.get("text")
            if isinstance(text_val, str) and text_val.strip():
                chunks.append(text_val.strip())
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def _collect_citation_urls(data: dict) -> list[str]:
    urls: list[str] = []
    if not isinstance(data, dict):
        return urls
    citations = data.get("citations")
    if isinstance(citations, list):
        for c in citations:
            if isinstance(c, str):
                urls.append(c)
            elif isinstance(c, dict):
                u = c.get("url") or c.get("uri")
                if isinstance(u, str):
                    urls.append(u)
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                for ann in part.get("annotations") or []:
                    if isinstance(ann, dict) and isinstance(ann.get("url"), str):
                        urls.append(ann["url"])
    return urls


def _parse_json_array(text: str) -> list[dict]:
    if not text:
        return []
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    return []


def _tweet_id(url: str) -> str | None:
    m = _TWEET_URL_RE.search(url or "")
    return m.group(1) if m else None


def _handle_from_url(url: str) -> str:
    m = _HANDLE_FROM_URL_RE.search(url or "")
    if not m:
        return ""
    handle = m.group(1)
    if handle.lower() in {"i", "intent", "share"}:
        return ""
    return handle


def _article_from_fields(
    *,
    handle: str,
    text: str,
    url: str,
    published: str = "",
    tickers: list[str] | None = None,
) -> dict | None:
    text = (text or "").strip()
    url = (url or "").strip()
    handle = normalize_handle(handle) or _handle_from_url(url)
    tid = _tweet_id(url)
    if not tid:
        return None
    if not url.startswith("http"):
        return None
    # Canonicalize nitter/mobile variants onto x.com
    url = f"https://x.com/{handle or 'i'}/status/{tid}"
    title_body = text[:200] if text else f"tweet {tid}"
    title = f"@{handle}: {title_body}" if handle else title_body
    return {
        "title": title[:400],
        "link": url,
        "summary": text[:1500],
        "published": (published or "")[:80],
        "source": f"{SOURCE_TAG}/@{handle}" if handle else SOURCE_TAG,
        "tickers": [str(t).upper() for t in (tickers or []) if str(t).strip()],
    }


def articles_from_response(data: dict) -> list[dict]:
    """Turn an xAI Responses payload into intern article dicts."""
    found: list[dict] = []
    seen_ids: set[str] = set()

    def _add(art: dict | None) -> None:
        if not art:
            return
        tid = _tweet_id(art.get("link") or "")
        if not tid or tid in seen_ids:
            return
        seen_ids.add(tid)
        found.append(art)

    for raw in _parse_json_array(_extract_response_text(data)):
        _add(
            _article_from_fields(
                handle=str(raw.get("handle") or raw.get("user") or raw.get("username") or ""),
                text=str(raw.get("text") or raw.get("summary") or raw.get("title") or ""),
                url=str(raw.get("url") or raw.get("link") or ""),
                published=str(raw.get("published") or raw.get("created_at") or ""),
                tickers=raw.get("tickers") if isinstance(raw.get("tickers"), list) else None,
            )
        )

    # Citations are always returned; harvest tweet URLs the model may not
    # have serialized into the JSON array.
    text_blob = _extract_response_text(data)
    extra_urls = list(_collect_citation_urls(data))
    extra_urls.extend(m.group(0) for m in _TWEET_URL_RE.finditer(text_blob))
    for url in extra_urls:
        if not _tweet_id(url):
            continue
        handle = _handle_from_url(url)
        _add(
            _article_from_fields(
                handle=handle,
                text="",
                url=url,
            )
        )
        if len(found) >= MAX_TWEETS:
            break
    return found[:MAX_TWEETS]


def _xai_responses(payload: dict) -> dict | None:
    """POST /v1/responses. Token is never logged."""
    from core.claude_cli import _load_xai_access_token

    token = _load_xai_access_token()
    if not token:
        print("[x_search] xAI auth missing; cannot call x_search")
        return None
    req = urllib.request.Request(
        f"{XAI_API_BASE}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S, context=_ssl_context()) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        # Never print Authorization or token-shaped strings.
        snippet = (err_body or str(e))[:200].replace("\n", " ")
        print(f"[x_search] xAI HTTP {e.code}: {snippet}")
        return None
    except Exception as e:
        print(f"[x_search] xAI call failed: {type(e).__name__}: {e}")
        return None


def collect_x_search(*, accounts: list[str] | None = None) -> list[dict]:
    """Pull recent tweets for configured accounts via xAI x_search."""
    handles = accounts if accounts is not None else load_twitter_accounts()
    print(f"[x_search] {len(handles)} accounts via xAI x_search model={MODEL}")
    payload = build_x_search_payload(handles)
    data = _xai_responses(payload)
    if not data:
        return []
    articles = articles_from_response(data)
    print(f"[x_search] {len(articles)} tweets")
    return articles


if __name__ == "__main__":
    items = collect_x_search()
    print(f"Total: {len(items)}")
    for a in items[:10]:
        print(f"  [{a['source']}] {a['title'][:100]}")
