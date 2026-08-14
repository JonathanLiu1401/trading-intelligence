"""Durable order + lesson memory for paper-trader decision cycles."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .claude_mem_trades import recent_trade_memory, _default_memory_dir, _iso_now


def _memory_paths(memory_dir: Path | None = None) -> dict[str, Path]:
    directory = memory_dir or _default_memory_dir()
    return {
        "dir": directory,
        "lessons": directory / "lessons.jsonl",
        "lessons_md": directory / "LESSONS.md",
        "orders_md": directory / "RECENT_ORDERS.md",
        "standing_orders": directory / "OPERATOR_STANDING_ORDERS.md",
    }


def load_operator_standing_orders(memory_dir: Path | None = None, max_chars: int = 4500) -> str | None:
    """Jonathan/operator standing orders — injected every cycle when present."""
    path = _memory_paths(memory_dir)["standing_orders"]
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _load_all_trade_rows(memory_dir: Path | None = None, max_files: int = 14) -> list[dict]:
    directory = memory_dir or _default_memory_dir()
    if not directory.exists():
        return []
    files = sorted(directory.glob("trades-*.jsonl"))[-max_files:]
    rows: list[dict] = []
    for path in files:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        except OSError:
            continue
    return rows


def _short_reason(text: Any, cap: int = 140) -> str:
    s = " ".join(str(text or "").split())
    if not s:
        return ""
    if len(s) > cap:
        return s[: cap - 1].rstrip() + "…"
    return s


def _order_line(row: dict) -> str:
    ts = str(row.get("ts") or "")[:19].replace("T", " ")
    status = str(row.get("status") or "?")
    action = str(row.get("action") or "?")
    ticker = str(row.get("ticker") or "?")
    qty = row.get("qty")
    detail = _short_reason(row.get("detail"), 80)
    reason = _short_reason(row.get("reasoning"), 120)
    opt = ""
    if row.get("is_option") or row.get("strike") or row.get("expiry"):
        opt = " strike=%s expiry=%s" % (row.get("strike"), row.get("expiry"))
    bits = [ts, status, f"{action} {ticker}{opt}", f"qty={qty}"]
    if detail:
        bits.append(detail)
    if reason:
        bits.append("why:\"%s\"" % reason)
    return "- " + " | ".join(str(b) for b in bits if b not in (None, ""))


def detect_lessons_from_orders(rows: list[dict], lookback: int = 40) -> list[dict]:
    recent = [r for r in rows[-lookback:] if str(r.get("status") or "").upper() == "FILLED"]
    lessons: list[dict] = []
    by_ticker: dict[str, list[dict]] = {}
    for r in recent:
        t = str(r.get("ticker") or "").upper()
        if not t:
            continue
        by_ticker.setdefault(t, []).append(r)

    for ticker, hist in by_ticker.items():
        if len(hist) < 2:
            continue
        for i in range(1, len(hist)):
            a = hist[i - 1]
            b = hist[i]
            aa = str(a.get("action") or "").upper()
            bb = str(b.get("action") or "").upper()
            pair = {aa, bb}
            if not (
                ("BUY" in pair and "SELL" in pair)
                or ("BUY_CALL" in pair and "SELL_CALL" in pair)
                or ("BUY_PUT" in pair and "SELL_PUT" in pair)
                or ("SHORT" in pair and "COVER" in pair)
            ):
                continue
            try:
                ta = datetime.fromisoformat(str(a.get("ts")).replace("Z", "+00:00"))
                tb = datetime.fromisoformat(str(b.get("ts")).replace("Z", "+00:00"))
            except Exception:
                continue
            hours = abs((tb - ta).total_seconds()) / 3600.0
            if hours > 36:
                continue
            lessons.append({
                "ts": _iso_now(),
                "kind": "lesson",
                "code": "same_name_flip",
                "ticker": ticker,
                "severity": 0.8 if hours <= 12 else 0.6,
                "summary": (
                    f"{ticker}: flipped {aa}->{bb} in {hours:.1f}h. "
                    "Do not re-enter the same name on the next impulse without a new catalyst."
                ),
                "evidence": [
                    _short_reason(a.get("detail") or a.get("text"), 100),
                    _short_reason(b.get("detail") or b.get("text"), 100),
                ],
            })

        blocked = [
            r for r in rows[-lookback:]
            if str(r.get("ticker") or "").upper() == ticker
            and str(r.get("status") or "").upper() == "BLOCKED"
        ]
        if len(blocked) >= 2:
            last = blocked[-1]
            lessons.append({
                "ts": _iso_now(),
                "kind": "lesson",
                "code": "repeated_block",
                "ticker": ticker,
                "severity": 0.5,
                "summary": (
                    f"{ticker}: repeated BLOCKED orders ({len(blocked)} recent). "
                    "Respect the block reason instead of retrying the same action."
                ),
                "evidence": [_short_reason(last.get("detail"), 120)],
            })

    dedup: dict[tuple[str, str], dict] = {}
    for les in lessons:
        key = (str(les.get("code")), str(les.get("ticker")))
        dedup[key] = les
    out = list(dedup.values())
    out.sort(key=lambda x: float(x.get("severity") or 0.0), reverse=True)
    return out[:12]


def refresh_lessons_log(memory_dir: Path | None = None) -> Path:
    paths = _memory_paths(memory_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    rows = _load_all_trade_rows(memory_dir=memory_dir)
    lessons = detect_lessons_from_orders(rows)
    if lessons:
        with paths["lessons"].open("a", encoding="utf-8") as f:
            for les in lessons:
                f.write(json.dumps(les, ensure_ascii=False) + chr(10))

    recent = rows[-25:]
    order_lines = ["# RECENT ORDERS (durable paper-trader memory)", ""]
    if not recent:
        order_lines.append("_No orders logged yet._")
    else:
        for row in reversed(recent):
            order_lines.append(_order_line(row))
    paths["orders_md"].write_text(chr(10).join(order_lines) + chr(10), encoding="utf-8")

    latest_lessons: dict[tuple[str, str], dict] = {}
    if paths["lessons"].exists():
        try:
            for line in paths["lessons"].read_text(encoding="utf-8").splitlines()[-200:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                key = (str(row.get("code")), str(row.get("ticker")))
                latest_lessons[key] = row
        except OSError:
            pass
    ranked = sorted(latest_lessons.values(), key=lambda x: float(x.get("severity") or 0.0), reverse=True)[:15]
    md = [
        "# DURABLE TRADE LESSONS",
        "",
        "Auto-distilled from paper-trader order memory. These are observations,",
        "not hard blocks. Prefer not repeating the same mistake without new evidence.",
        "",
    ]
    if not ranked:
        md.append("_No lessons distilled yet._")
    else:
        for les in ranked:
            md.append(
                "- [%s] %s (sev=%.2f)"
                % (les.get("code") or "lesson", les.get("summary") or "", float(les.get("severity") or 0.0))
            )
    paths["lessons_md"].write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    return paths["lessons_md"]


def load_latest_lessons(limit: int = 8, memory_dir: Path | None = None) -> list[dict]:
    paths = _memory_paths(memory_dir)
    if not paths["lessons"].exists():
        try:
            refresh_lessons_log(memory_dir=memory_dir)
        except Exception:
            return []
    latest: dict[tuple[str, str], dict] = {}
    try:
        for line in paths["lessons"].read_text(encoding="utf-8").splitlines()[-300:]:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            key = (str(row.get("code")), str(row.get("ticker")))
            latest[key] = row
    except OSError:
        return []
    ranked = sorted(latest.values(), key=lambda x: float(x.get("severity") or 0.0), reverse=True)
    return ranked[:limit]


def build_durable_memory_prompt_block(
    *,
    order_limit: int = 12,
    lesson_limit: int = 6,
    memory_dir: Path | None = None,
    names_in_play: set[str] | list[str] | None = None,
) -> str | None:
    rows = recent_trade_memory(limit=max(order_limit * 3, 30), memory_dir=memory_dir)
    standing_only = load_operator_standing_orders(memory_dir=memory_dir)
    if not rows and not load_latest_lessons(limit=1, memory_dir=memory_dir) and not standing_only:
        return None

    names = None
    if names_in_play is not None:
        names = {str(x).upper() for x in names_in_play if str(x).strip()}

    ordered = list(reversed(rows[-order_limit:]))
    if names:
        preferred = [r for r in reversed(rows) if str(r.get("ticker") or "").upper() in names][: max(4, order_limit // 2)]
        seen = {id(x) for x in preferred}
        tail = [r for r in ordered if id(r) not in seen]
        ordered = (preferred + tail)[:order_limit]

    lessons = load_latest_lessons(limit=lesson_limit, memory_dir=memory_dir)
    if names:
        lessons = sorted(
            lessons,
            key=lambda les: (0 if str(les.get("ticker") or "").upper() in names else 1, -float(les.get("severity") or 0.0)),
        )[:lesson_limit]

    standing = load_operator_standing_orders(memory_dir=memory_dir)
    lines: list[str] = []
    if standing:
        lines.extend(
            [
                "OPERATOR STANDING ORDERS (hard operator mandate — outranks "
                "construction underweights, drought-fallback busywork, and "
                "generic 'stay invested' pressure when they conflict):",
                standing,
                "",
            ]
        )

    lines.extend([
        "DURABLE TRADE MEMORY (your own prior orders + distilled lessons — "
        "observations only, NOT hard blocks; use them so you do not amnesia-trade "
        "the same name every cycle):",
        "Recent orders (newest first):",
    ])
    if ordered:
        for row in ordered:
            lines.append("  " + _order_line(row)[2:])
    else:
        lines.append("  (none yet)")

    lines.append("Active lessons:")
    if lessons:
        for les in lessons:
            lines.append(
                "  - [%s/%s] %s"
                % (les.get("code") or "lesson", les.get("ticker") or "?", les.get("summary") or "")
            )
    else:
        lines.append("  (none distilled yet)")

    lines.append(
        "If a lesson conflicts with fresh catalyst evidence, explain why the new "
        "evidence overrides the old memory before re-entering."
    )
    return chr(10).join(lines)
