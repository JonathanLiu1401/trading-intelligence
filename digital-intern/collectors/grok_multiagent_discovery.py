"""Experimental Grok 4.20 multi-agent discovery collector for ArticleNet.

Goal: surface market/news articles that existing RSS/GDELT/Google News
collectors are missing. Uses the xAI multi-agent Grok model via OpenClaw
xAI OAuth login, validates candidate URLs, and inserts only novel rows into
the local ArticleStore.

Safety rails:
- host-load gate
- hard cap on candidates / inserts per run
- URL validation required before insert
- explicit experimental source tag for later keep/remove audit
"""
from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs" / "grok_multiagent_discovery"
STATE_PATH = LOG_DIR / "state.json"
TRIALS_PATH = LOG_DIR / "trials.jsonl"

MODEL = os.environ.get(
    "GROK_DISCOVERY_MODEL",
    "grok-4.20-multi-agent-0309",
)
XAI_API_BASE = os.environ.get("PAPER_TRADER_XAI_API_BASE", "https://api.x.ai/v1").rstrip("/")
XAI_AUTH_PROFILES_PATH = Path(
    os.environ.get(
        "PAPER_TRADER_XAI_AUTH_PROFILES",
        str(Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"),
    )
)
XAI_AUTH_PROFILE = os.environ.get(
    "PAPER_TRADER_XAI_AUTH_PROFILE",
    "xai:artintel1110@gmail.com",
)
CURSOR_API_BASE = os.environ.get(
    "PAPER_TRADER_CURSOR_API_BASE", "http://127.0.0.1:4646/v1"
).rstrip("/")
CURSOR_MODEL = os.environ.get(
    "GROK_DISCOVERY_CURSOR_MODEL",
    os.environ.get("PAPER_TRADER_CURSOR_MODEL", "cursor-grok-4.6-xhigh"),
)
CURSOR_FALLBACK = os.environ.get(
    "GROK_DISCOVERY_CURSOR_FALLBACK",
    os.environ.get("PAPER_TRADER_CURSOR_FALLBACK", "1"),
).strip().lower() not in {"0", "false", "no", "off"}
MAX_CANDIDATES = int(os.environ.get("GROK_DISCOVERY_MAX_CANDIDATES", "24"))
MAX_INSERTS = int(os.environ.get("GROK_DISCOVERY_MAX_INSERTS", "16"))
LOAD_LIMIT = float(os.environ.get("GROK_DISCOVERY_LOAD_LIMIT", "8"))
HTTP_TIMEOUT_S = int(os.environ.get("GROK_DISCOVERY_HTTP_TIMEOUT_S", "8"))
LLM_TIMEOUT_S = int(os.environ.get("GROK_DISCOVERY_LLM_TIMEOUT_S", "420"))
# REST multi-agent fanout: reasoning.effort high/xhigh => 16 agents; low/medium => 4.
AGENT_EFFORT = os.environ.get("GROK_DISCOVERY_AGENT_EFFORT", "high").strip().lower()
ENABLE_WEB_SEARCH = os.environ.get("GROK_DISCOVERY_ENABLE_WEB_SEARCH", "1").strip().lower() not in {"0", "false", "no", "off"}
ENABLE_X_SEARCH = os.environ.get("GROK_DISCOVERY_ENABLE_X_SEARCH", "1").strip().lower() not in {"0", "false", "no", "off"}
SOURCE_TAG = "grok_multiagent_discovery"
USER_AGENT = "DigitalInternGrokDiscovery/0.1 (+local paper research desk)"

_TOPIC_SEEDS = [
    "US equities, mega-cap tech, semis/AI infrastructure catalysts from the last 12 hours",
    "macro rates, oil, geopolitics, China/Asia market-moving catalysts from the last 12 hours",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_avg() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except Exception:
        return None


def _ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _load_xai_token() -> str | None:
    for key in ("PAPER_TRADER_XAI_API_KEY", "XAI_API_KEY"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    try:
        raw = json.loads(XAI_AUTH_PROFILES_PATH.read_text())
    except Exception as e:
        print(f"[grok_discovery] auth profiles unreadable: {e}")
        return None
    profiles = raw.get("profiles") if isinstance(raw, dict) else None
    if not isinstance(profiles, dict):
        return None
    preferred = [XAI_AUTH_PROFILE, "xai:default", "xai"]
    candidates: list[tuple[str, Any]] = []
    for key in preferred:
        if key and key in profiles:
            candidates.append((key, profiles[key]))
    for key, value in profiles.items():
        if str(key).startswith("xai:") or (
            isinstance(value, dict) and str(value.get("provider", "")).lower() == "xai"
        ):
            if (key, value) not in candidates:
                candidates.append((key, value))
    now_ms = int(time.time() * 1000)
    for key, value in candidates:
        if not isinstance(value, dict):
            continue
        token = (
            value.get("access")
            or value.get("accessToken")
            or value.get("apiKey")
            or value.get("api_key")
            or value.get("token")
        )
        if not token:
            continue
        expires = value.get("expires")
        try:
            exp_i = int(expires) if expires is not None else None
        except (TypeError, ValueError):
            exp_i = None
        if exp_i is not None and exp_i < now_ms:
            print(f"[grok_discovery] expired auth profile: {key}")
            continue
        return str(token)
    return None


def _load_cursor_api_key() -> str | None:
    """Optional bearer for the local Cursor CLI OpenAI proxy."""
    for key in (
        "GROK_DISCOVERY_CURSOR_API_KEY",
        "PAPER_TRADER_CURSOR_API_KEY",
        "CURSOR_API_KEY",
    ):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    env_path = Path.home() / ".cursor-agent-api" / "env"
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export CURSOR_API_KEY="):
                line = line[len("export ") :]
            if line.startswith("CURSOR_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'").strip('"')
    except Exception:
        pass
    return None


def _is_quota_like_error(text: str) -> bool:
    low = (text or "").lower()
    return any(
        s in low
        for s in ("429", "quota", "rate limit", "usage limit", "credit", "exhausted")
    )


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _cursor_chat(prompt: str, *, timeout_s: int = LLM_TIMEOUT_S) -> str | None:
    """Fallback: Cursor CLI proxy Grok (chat only; no xAI multi-agent tools)."""
    if not CURSOR_FALLBACK:
        return None
    model_id = (CURSOR_MODEL or "cursor-grok-4.6-xhigh").strip()
    system = (
        "You are a financial news discovery harness (Cursor Grok fallback). "
        "Find real, recent public market/news articles with working URLs. "
        "Return ONLY a JSON array of objects. No markdown. "
        "Web/X search tools are unavailable on this fallback path — "
        "use known public sources and precise URLs."
    )
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    token = _load_cursor_api_key()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{CURSOR_API_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    print(
        f"[grok_discovery] Cursor Grok fallback model={model_id} "
        f"base={CURSOR_API_BASE}",
        flush=True,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=_ssl_context()) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        text = _extract_response_text(data)
        if text:
            print("[grok_discovery] Cursor Grok fallback OK", flush=True)
        return text or None
    except Exception as e:
        print(f"[grok_discovery] Cursor fallback failed: {e}", flush=True)
        return None


def _extract_response_text(data: dict) -> str | None:
    """Normalize chat.completions or responses API payloads to text."""
    if not isinstance(data, dict):
        return None
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if text:
            return text
    output = data.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                        val = (part.get("text") or "").strip()
                        if val:
                            chunks.append(val)
            elif isinstance(content, str) and content.strip():
                chunks.append(content.strip())
            text_val = item.get("text")
            if isinstance(text_val, str) and text_val.strip():
                chunks.append(text_val.strip())
        if chunks:
            return "\n".join(chunks).strip()
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()
    return None


def _xai_chat(prompt: str, *, model: str = MODEL, timeout_s: int = LLM_TIMEOUT_S) -> str | None:
    """Primary xAI Grok path; Cursor Grok is fallback-only on miss/failure."""
    token = _load_xai_token()
    if not token:
        print("[grok_discovery] missing xAI token; trying Cursor Grok fallback")
        return _cursor_chat(prompt, timeout_s=timeout_s)
    use_responses = "multi-agent" in (model or "").lower()
    system = (
        "You are a multi-agent financial news discovery harness. "
        "Find real, recent public market/news articles with working URLs. "
        "Return ONLY a JSON array of objects. No markdown."
    )
    if use_responses:
        effort = AGENT_EFFORT if AGENT_EFFORT in {"low", "medium", "high", "xhigh"} else "high"
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            # high/xhigh => 16 agents; low/medium => 4 agents (xAI multi-agent REST mapping)
            "reasoning": {"effort": effort},
        }
        tools = []
        if ENABLE_WEB_SEARCH:
            tools.append({"type": "web_search"})
        if ENABLE_X_SEARCH:
            tools.append({"type": "x_search"})
        if tools:
            payload["tools"] = tools
        endpoint = f"{XAI_API_BASE}/responses"
        tool_types = [t.get("type") for t in tools]
        print(
            f"[grok_discovery] multi-agent call model={model} effort={effort} "
            f"tools={tool_types}",
            flush=True,
        )
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        endpoint = f"{XAI_API_BASE}/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=_ssl_context()) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        text = _extract_response_text(data)
        if text:
            return text
        print("[grok_discovery] xAI returned empty; trying Cursor Grok fallback")
    except Exception as e:
        print(f"[grok_discovery] xAI call failed: {e}")
    # Fallback ONLY: local Cursor CLI proxy (still Grok, never primary).
    return _cursor_chat(prompt, timeout_s=timeout_s)


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
    for i, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            for key in ("articles", "items", "results", "candidates"):
                val = obj.get(key)
                if isinstance(val, list):
                    return [x for x in val if isinstance(x, dict)]
    return []


def _topic_prompt(topic: str, known_titles: list[str]) -> str:
    known = "\n".join(f"- {t}" for t in known_titles[:12]) or "- (none)"
    return (
        f"Topic focus: {topic}\n\n"
        "Find at least 15 and up to 20 real public web articles published in the last 24 hours that "
        "are useful for equity/macro trading research and are likely NOT already in a "
        "generic Google News / GDELT / Yahoo RSS feed set.\n"
        "Use the multi-agent team and web search to verify live working URLs. Prefer primary sources, reputable financial outlets, company IR/press, "
        "regulator pages, or high-signal specialty finance sites.\n"
        "Avoid social posts, empty homepage shells, paywalled-only blurbs without a real article URL, "
        "and pure opinion blogs with no market facts.\n\n"
        "Already known recent titles to avoid repeating:\n"
        f"{known}\n\n"
        "Return a JSON array only. Each object must have:\n"
        '  "title": string,\n'
        '  "url": https URL,\n'
        '  "source": publisher/domain,\n'
        '  "summary": 1-2 sentence factual summary,\n'
        '  "published": ISO-8601 if known else "",\n'
        '  "tickers": array of ticker symbols if obvious else [],\n'
        '  "why_novel": short reason this may be missing from generic scrapers\n'
    )


def _recent_known_titles(limit: int = 20) -> list[str]:
    try:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        rows = store.conn.execute(
            "SELECT title FROM articles "
            "WHERE first_seen >= datetime('now', '-6 hours') "
            "ORDER BY first_seen DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows if r and r[0]]
    except Exception as e:
        print(f"[grok_discovery] known-title probe failed: {e}")
        return []


def _normalize_url(url: str) -> str | None:
    url = (url or "").strip()
    if not url:
        return None
    if url.startswith("//"):
        url = "https:" + url
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    host = parsed.netloc.lower()
    if any(bad in host for bad in ("localhost", "127.0.0.1", "example.com")):
        return None
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path or "/", "", parsed.query, "")
    )


def _validate_url(url: str) -> tuple[bool, str]:
    """Best-effort URL validation with short timeouts.

    Prefer live GET. Soft-accept known reputable publishers on bot walls/timeouts
    so the experiment is not starved by anti-bot HTML endpoints.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    host = (urllib.parse.urlparse(url).netloc or "").lower()
    reputable = any(
        host == d or host.endswith("." + d)
        for d in (
            "reuters.com", "cnbc.com", "bloomberg.com", "wsj.com", "ft.com",
            "nytimes.com", "apnews.com", "bbc.com", "bbc.co.uk", "marketwatch.com",
            "barrons.com", "finance.yahoo.com", "seekingalpha.com", "sec.gov",
            "federalreserve.gov", "prnewswire.com", "businesswire.com",
            "globenewswire.com", "theverge.com", "techcrunch.com", "semianalysis.com",
        )
    )
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S, context=_ssl_context()) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            final = resp.geturl() or url
            body = resp.read(2048)
            if code and int(code) >= 400:
                if reputable and int(code) in {401, 403, 429}:
                    return True, final
                return False, f"http {code}"
            if not body and not reputable:
                return False, f"empty body ({code})"
            return True, final
    except urllib.error.HTTPError as e:
        if reputable and e.code in {401, 403, 429}:
            return True, url
        return False, f"http {e.code}"
    except Exception as e:
        if reputable:
            return True, url
        return False, str(e)[:120]


def _url_exists_in_store(url: str) -> bool:
    try:
        from storage.article_store import ArticleStore
        store = ArticleStore()
        row = store.conn.execute(
            "SELECT 1 FROM articles WHERE url = ? LIMIT 1",
            (url,),
        ).fetchone()
        return bool(row)
    except Exception:
        return False


def _candidate_from_model(raw: dict) -> dict | None:
    title = str(raw.get("title") or "").strip()
    url = _normalize_url(str(raw.get("url") or raw.get("link") or ""))
    if not title or not url:
        return None
    source = str(raw.get("source") or urllib.parse.urlparse(url).netloc or SOURCE_TAG).strip()
    summary = str(raw.get("summary") or raw.get("snippet") or "").strip()
    published = str(raw.get("published") or "").strip()
    why = str(raw.get("why_novel") or "").strip()
    tickers = raw.get("tickers") if isinstance(raw.get("tickers"), list) else []
    tickers = [str(t).upper() for t in tickers if str(t).strip()]
    if why:
        summary = (summary + f" [why_novel: {why}]").strip()
    return {
        "title": title[:400],
        "link": url,
        "source": f"{SOURCE_TAG}/{source}"[:120],
        "summary": summary[:1500],
        "published": published[:80],
        "tickers": tickers,
        "_relevance_score": 6.0,
    }


def discover_candidates() -> list[dict]:
    known = _recent_known_titles()
    # One multi-agent call for the whole batch. Sequential multi-agent topic
    # fanout is too slow/expensive on this host.
    topic = (
        "US equities, mega-cap tech, semis/AI infrastructure, macro rates, oil, "
        "geopolitics, and China/Asia market-moving catalysts from the last 12-24 hours"
    )
    raw = _xai_chat(_topic_prompt(topic, known))
    items = _parse_json_array(raw or "")
    print(f"[grok_discovery] multi-agent raw_items={len(items)}", flush=True)
    found: list[dict] = []
    seen_urls: set[str] = set()
    for item in items:
        cand = _candidate_from_model(item)
        if not cand:
            continue
        url = cand["link"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        found.append(cand)
        if len(found) >= MAX_CANDIDATES:
            break
    return found


def filter_and_validate(candidates: list[dict]) -> tuple[list[dict], dict]:
    stats = {
        "candidates": len(candidates),
        "already_in_db": 0,
        "invalid_url": 0,
        "validated": 0,
    }
    out: list[dict] = []
    for cand in candidates:
        url = cand.get("link") or ""
        if _url_exists_in_store(url):
            stats["already_in_db"] += 1
            continue
        ok, final_or_err = _validate_url(url)
        if not ok:
            stats["invalid_url"] += 1
            print(f"[grok_discovery] reject {url} :: {final_or_err}")
            continue
        if isinstance(final_or_err, str) and final_or_err.startswith("http"):
            cand = {**cand, "link": final_or_err}
            if _url_exists_in_store(cand["link"]):
                stats["already_in_db"] += 1
                continue
        out.append(cand)
        stats["validated"] += 1
        if len(out) >= MAX_INSERTS:
            break
    return out, stats


def insert_articles(articles: list[dict]) -> int:
    if not articles:
        return 0
    from storage.article_store import ArticleStore
    store = ArticleStore()
    return int(store.insert_batch(articles, cycle=0) or 0)


def append_trial(record: dict) -> None:
    _ensure_dirs()
    with TRIALS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    STATE_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def run_once(*, force: bool = False) -> dict:
    """Run one discovery cycle. Returns metrics dict."""
    _ensure_dirs()
    started = time.time()
    load = _load_avg()
    result = {
        "ts": _now_iso(),
        "model": MODEL,
        "agent_effort": AGENT_EFFORT,
        "agent_count_target": 16 if AGENT_EFFORT in {"high", "xhigh"} else 4,
        "load_1m": load,
        "status": "ok",
        "candidates": 0,
        "validated": 0,
        "already_in_db": 0,
        "invalid_url": 0,
        "inserted": 0,
        "elapsed_s": 0.0,
        "error": "",
        "sample_titles": [],
    }
    if not force and load is not None and load > LOAD_LIMIT:
        result["status"] = "skipped_load"
        result["error"] = f"load {load:.2f} > {LOAD_LIMIT}"
        append_trial(result)
        print(f"[grok_discovery] skipped: {result['error']}", flush=True)
        return result

    try:
        candidates = discover_candidates()
        validated, stats = filter_and_validate(candidates)
        inserted = insert_articles(validated)
        result.update(
            {
                "candidates": stats["candidates"],
                "validated": stats["validated"],
                "already_in_db": stats["already_in_db"],
                "invalid_url": stats["invalid_url"],
                "inserted": inserted,
                "sample_titles": [a.get("title", "")[:120] for a in validated[:5]],
                "status": "ok",
            }
        )
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:300]
        print(f"[grok_discovery] error: {e}", flush=True)

    result["elapsed_s"] = round(time.time() - started, 2)
    append_trial(result)
    print(
        "[grok_discovery] "
        f"status={result['status']} candidates={result['candidates']} "
        f"validated={result['validated']} inserted={result['inserted']} "
        f"dup={result['already_in_db']} bad={result['invalid_url']} "
        f"elapsed={result['elapsed_s']}s",
        flush=True,
    )
    return result


def summarize_trials(path: Path = TRIALS_PATH) -> dict:
    if not path.exists():
        return {
            "n_trials": 0,
            "inserted_total": 0,
            "validated_total": 0,
            "avg_inserted": 0.0,
            "ok_rate": 0.0,
            "recommendation": "REMOVE",
            "reason": "no trial data",
        }
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return {
            "n_trials": 0,
            "inserted_total": 0,
            "validated_total": 0,
            "avg_inserted": 0.0,
            "ok_rate": 0.0,
            "recommendation": "REMOVE",
            "reason": "no parseable trials",
        }
    n = len(rows)
    inserted = sum(int(r.get("inserted") or 0) for r in rows)
    validated = sum(int(r.get("validated") or 0) for r in rows)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    avg_inserted = inserted / n
    if n >= 3 and avg_inserted >= 1.0 and ok / n >= 0.5:
        rec = "KEEP"
        reason = (
            f"{n} trials, avg_inserted={avg_inserted:.2f}, "
            f"ok_rate={ok/n:.0%}, validated_total={validated}"
        )
    elif n >= 3 and inserted == 0:
        rec = "REMOVE"
        reason = f"{n} trials produced 0 inserts"
    else:
        rec = "REMOVE"
        reason = (
            f"weak yield: n={n}, avg_inserted={avg_inserted:.2f}, "
            f"ok_rate={ok/n:.0%}, inserted_total={inserted}"
        )
    return {
        "n_trials": n,
        "inserted_total": inserted,
        "validated_total": validated,
        "avg_inserted": round(avg_inserted, 3),
        "ok_rate": round(ok / n, 3),
        "recommendation": rec,
        "reason": reason,
        "latest": rows[-1],
    }


if __name__ == "__main__":
    import pprint
    pprint.pp(run_once(force=True))
