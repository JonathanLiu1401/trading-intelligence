"""Flask dashboard at :8090 — portfolio chart, trade log, positions, decisions, backtests."""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from .store import INITIAL_CASH, get_store

app = Flask(__name__)

# ── Code-freshness probe ─────────────────────────────────────────────────
# Long-running daemons silently serve pre-deploy bytecode: the scorer-clamp
# fix landed while this :8090 process was already up, so it kept emitting
# ±700% "predictions" for hours. /api/build-info exposes the git SHA the
# process booted with vs the on-disk HEAD so an operator (and the unified
# dashboard's banner) can see "you're running stale code — restart".
_REPO_DIR = str(Path(__file__).resolve().parent.parent)


def _git_sha(repo_dir: str, ref: str = "HEAD") -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", ref],
            capture_output=True, text=True, timeout=3,
        )
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except Exception:
        return None


_BOOT_SHA = _git_sha(_REPO_DIR)


def _head_sha_and_behind() -> tuple[str | None, int]:
    """Current on-disk HEAD short SHA + how many commits it is ahead of the
    SHA this process booted with (0 if in sync or indeterminable)."""
    head = _git_sha(_REPO_DIR)
    behind = 0
    if head and _BOOT_SHA and head != _BOOT_SHA:
        try:
            r = subprocess.run(
                ["git", "-C", _REPO_DIR, "rev-list", "--count",
                 f"{_BOOT_SHA}..HEAD"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                behind = int(r.stdout.strip() or 0)
        except Exception:
            behind = 0
    return head, behind


@app.route("/api/build-info")
def build_info_api():
    """{boot_sha, head_sha, behind, stale} — stale ⇒ restart to apply
    committed fixes (e.g. the DecisionScorer clamp)."""
    head, behind = _head_sha_and_behind()
    stale = bool(_BOOT_SHA and head and head != _BOOT_SHA)
    return jsonify({
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "service": "paper_trader",
        "boot_sha": _BOOT_SHA,
        "head_sha": head,
        "behind": behind,
        "stale": stale,
    })


# Static sector classification for analytics + sector-pulse cards.
# Keyed by the symbols we actually use in the watchlist + portfolio.
SECTOR_MAP = {
    # Semis (cash)
    "NVDA": "semis", "AMD": "semis", "MU": "semis", "AMAT": "semis",
    "LRCX": "semis", "KLAC": "semis", "TSM": "semis", "ASML": "semis",
    "MRVL": "semis", "SMH": "semis", "SOXX": "semis",
    "DRAM": "semis", "SNDU": "semis",
    # Semis leveraged
    "SOXL": "semis_lev", "SOXS": "semis_lev", "NVDU": "semis_lev",
    "MUU": "semis_lev",
    # Optical / networking
    "LITE": "optical", "LNOK": "optical",
    # Broad market
    "SPY": "broad", "QQQ": "broad", "VOO": "broad", "VTI": "broad",
    # Broad leveraged
    "TQQQ": "broad_lev", "UPRO": "broad_lev", "SPXL": "broad_lev",
    "QLD": "broad_lev", "SSO": "broad_lev", "UDOW": "broad_lev",
    "URTY": "broad_lev", "TNA": "broad_lev",
    "SPXS": "broad_lev", "SQQQ": "broad_lev",
    # Tech / FAANG
    "AAPL": "tech", "MSFT": "tech", "META": "tech", "GOOG": "tech",
    "GOOGL": "tech", "AMZN": "tech", "TSLA": "tech", "NFLX": "tech",
    "TECL": "tech_lev", "TECS": "tech_lev", "FNGU": "tech_lev",
    "FNGD": "tech_lev", "MSFU": "tech_lev", "AMZU": "tech_lev",
    "GOOGU": "tech_lev", "METAU": "tech_lev", "TSLL": "tech_lev",
    "CONL": "crypto_lev", "BITU": "crypto_lev", "ETHU": "crypto_lev",
    # Sector leveraged
    "LABU": "bio_lev", "CURE": "health_lev",
    "FAS": "fin_lev", "DPST": "fin_lev",
    "NAIL": "housing_lev", "UTSL": "util_lev",
    "DFEN": "defense_lev",
}

# Sector-pulse card focuses on the user's actual interest areas.
SECTOR_PULSE_TICKERS = [
    "MU", "NVDA", "AMD", "TSM", "AMAT", "LRCX", "KLAC", "MRVL", "ASML",
    "SMH", "SOXX", "SOXL",
    "LITE", "LNOK", "DRAM", "SNDU", "MUU",
]


def _classify(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper(), "other")


@app.after_request
def _cors(resp):
    # Cross-port fetch from Digital Intern dashboard (8080 → 8090).
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, OPTIONS")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
    return resp


TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Paper Trader</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='5' fill='%230d0d0d'/%3E%3Cline x1='7' y1='15' x2='7' y2='18' stroke='%2300d4ff' stroke-width='1.5'/%3E%3Crect x='5.5' y='18' width='3' height='7' rx='0.5' fill='%2300d4ff'/%3E%3Cline x1='7' y1='25' x2='7' y2='27' stroke='%2300d4ff' stroke-width='1.5'/%3E%3Cline x1='15' y1='12' x2='15' y2='15' stroke='%23ff3c4c' stroke-width='1.5'/%3E%3Crect x='13.5' y='15' width='3' height='6' rx='0.5' fill='%23ff3c4c'/%3E%3Cline x1='15' y1='21' x2='15' y2='24' stroke='%23ff3c4c' stroke-width='1.5'/%3E%3Cline x1='23' y1='5' x2='23' y2='8' stroke='%2300ff9f' stroke-width='1.5'/%3E%3Crect x='21.5' y='8' width='3' height='12' rx='0.5' fill='%2300ff9f'/%3E%3Cline x1='23' y1='20' x2='23' y2='23' stroke='%2300ff9f' stroke-width='1.5'/%3E%3Cpolyline points='7,21 15,17 23,11' stroke='%23ffd700' stroke-width='1.2' fill='none' stroke-dasharray='2,1.5'/%3E%3C/svg%3E">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Outfit:wght@400;500;600;700&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');
    :root {
      color-scheme: dark;
      --bg: #0c0d0f;
      --bg-panel: #111316;
      --bg-elevated: #17191d;
      --bg-hover: #1c1f24;
      --bg-input: #0e1012;
      --border: rgba(255,255,255,0.07);
      --border-strong: rgba(255,255,255,0.13);
      --text: #dde1e7;
      --text-secondary: #8b929d;
      --text-muted: #50565f;
      --amber: #f0b429;
      --amber-dim: rgba(240,180,41,0.12);
      --cyan: #0acdff;
      --cyan-dim: rgba(10,205,255,0.12);
      --green: #00c896;
      --green-dim: rgba(0,200,150,0.12);
      --red: #ff4455;
      --red-dim: rgba(255,68,85,0.12);
      --blue: #4d9eff;
      --blue-dim: rgba(77,158,255,0.12);
      --yellow: #fbbf24;
      --yellow-dim: rgba(251,191,36,0.12);
      --pink: #f472b6;
      --font-sans: 'Outfit', system-ui, sans-serif;
      --font-mono: 'DM Mono', 'JetBrains Mono', monospace;
      --font-display: 'Syne', system-ui, sans-serif;
      --radius: 8px;
      --radius-sm: 5px;
    }
    * { box-sizing: border-box; }
    html { overflow-x: hidden; max-width: 100%; }
    body { overflow-x: hidden; }
    body {
      margin: 0; padding: 0;
      font-family: var(--font-sans);
      background: var(--bg); color: var(--text);
      font-size: 15px; line-height: 1.5;
    }
    .brand, h1, h2, h3 { font-family: var(--font-display); }
    .page-content { padding: 24px; max-width: 1600px; width: 100%; }
    .topbar {
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border);
      padding: 0 20px; height: 48px;
      display: flex; align-items: center; gap: 2px;
      position: sticky; top: 0; z-index: 100; margin: 0;
      overflow: hidden; max-width: 100%;
    }
    .brand {
      font-weight: 700; color: var(--amber);
      font-size: 13px; letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-right: 16px; flex-shrink: 0;
    }
    .topbar a {
      color: var(--text-secondary); text-decoration: none;
      font-size: 13px; font-weight: 500;
      padding: 5px 12px; border-radius: var(--radius-sm);
      transition: color 0.15s, background 0.15s;
      white-space: nowrap;
    }
    .topbar a:hover { color: var(--text); background: var(--bg-hover); }
    .topbar a.active { color: var(--amber); background: var(--amber-dim); }
    h1 { margin: 0 0 4px; font-size: 22px; font-weight: 600; color: var(--text); }
    .sub { color: var(--text-secondary); font-size: 13px; margin-bottom: 20px; }
    nav.tabs {
      display: flex; gap: 2px; margin-bottom: 18px;
      border-bottom: 1px solid var(--border);
      overflow-x: auto; -webkit-overflow-scrolling: touch; flex-wrap: nowrap;
    }
    nav.tabs a {
      padding: 8px 16px; color: var(--text-secondary); text-decoration: none;
      border-bottom: 2px solid transparent; font-size: 13px; font-weight: 500;
      cursor: pointer; transition: color 0.15s; margin-bottom: -1px;
    }
    nav.tabs a.active { color: var(--amber); border-bottom-color: var(--amber); }
    nav.tabs a:hover { color: var(--text); }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }
    .grid {
      display: grid; gap: 18px;
      grid-template-columns: 1fr 1fr;
    }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--bg-panel); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 18px 20px;
      overflow-x: auto; -webkit-overflow-scrolling: touch;
    }
    .card h2 {
      margin: 0 0 14px; font-size: 11px; font-weight: 600;
      color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em;
    }
    .stat-row { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 12px; }
    .stat { flex: 1 1 120px; }
    .stat .v {
      font-family: var(--font-mono);
      font-size: 24px; color: var(--text); font-weight: 500;
      font-variant-numeric: tabular-nums;
      min-width: 0; max-width: 100%;
    }
    .stat .l { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }
    .pos, .pl { color: var(--green); }
    .neg { color: var(--red); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th {
      text-align: left; padding: 0 10px 10px;
      font-size: 11px; font-weight: 600; color: var(--text-muted);
      text-transform: uppercase; letter-spacing: 0.08em;
      border-bottom: 1px solid var(--border-strong);
    }
    td {
      padding: 8px 10px; border-bottom: 1px solid var(--border);
      font-size: 13px;
    }
    td.num {
      text-align: right;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }
    tr:hover td { background: var(--bg-hover); }
    .muted { color: var(--text-secondary); }
    canvas { max-width: 100%; max-height: 280px; }
    .pill {
      display: inline-flex; align-items: center;
      padding: 2px 8px; border-radius: 4px;
      background: var(--bg-elevated); color: var(--text-secondary);
      font-size: 11px; font-weight: 500; letter-spacing: 0.04em;
      font-family: var(--font-sans);
    }
    .pill.buy { background: var(--green-dim); color: var(--green); }
    .pill.sell { background: var(--red-dim); color: var(--red); }
    .pill.hold { background: var(--bg-elevated); color: var(--text-secondary); }
    .pill.run { background: var(--blue-dim); color: var(--blue); }
    .pill.status-running  { background: var(--blue-dim); color: var(--blue); }
    .pill.status-complete { background: var(--green-dim); color: var(--green); }
    .pill.status-failed   { background: var(--red-dim); color: var(--red); }
    .pill.status-pending  { background: var(--bg-elevated); color: var(--text-secondary); }
    .spinner {
      display: inline-block; width: 10px; height: 10px;
      border: 2px solid var(--border-strong); border-top-color: var(--cyan);
      border-radius: 50%; animation: spin 0.8s linear infinite;
      vertical-align: middle; margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .progress-wrap {
      margin: 8px 0; height: 4px; background: var(--bg-elevated);
      border-radius: 4px; overflow: hidden;
    }
    .progress-bar {
      height: 100%; background: linear-gradient(90deg, var(--amber), var(--cyan));
      transition: width 0.4s ease;
    }
    .progress-label { font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
    tr.bt-row { cursor: pointer; }
    tr.bt-row:hover td { background: var(--bg-hover); }
    tr.bt-row.best td { background: var(--green-dim); }
    tr.bt-row.beat td:first-child { border-left: 2px solid var(--green); }
    tr.bt-row.miss td:first-child { border-left: 2px solid var(--red); }
    #bt-trades { margin-top: 14px; display: none; }
    #bt-trades.show { display: block; }
    .bt-headline {
      display: flex; gap: 28px; flex-wrap: wrap; margin-bottom: 12px;
    }
    .bt-headline .stat .v { font-size: 22px; }
    .bt-layout {
      display: grid; grid-template-columns: 240px 1fr; gap: 14px; align-items: start;
    }
    @media (max-width: 980px) { .bt-layout { grid-template-columns: 1fr; } }
    .bt-sidebar { position: sticky; top: 62px; max-height: calc(100vh - 78px); overflow-y: auto; }
    .bt-sidebar h2 { margin: 0; }
    .bt-legend-row {
      display: flex; align-items: center; gap: 8px; padding: 6px 4px;
      border-bottom: 1px solid var(--border); cursor: pointer; user-select: none;
      transition: background 0.15s;
    }
    .bt-legend-row:hover { background: var(--bg-hover); }
    .bt-legend-row.selected { background: var(--bg-elevated); }
    .bt-legend-row.hidden-run { opacity: 0.35; }
    .bt-legend-row input[type=checkbox] { accent-color: var(--cyan); margin: 0; }
    .bt-swatch {
      width: 12px; height: 12px; border-radius: 3px; flex: 0 0 12px;
    }
    .bt-legend-row .name { flex: 1; font-size: 13px; color: var(--text); }
    .bt-legend-row .ret { font-size: 11px; font-variant-numeric: tabular-nums; font-family: var(--font-mono); }
    .bt-btn {
      background: var(--bg-elevated); color: var(--text);
      border: 1px solid var(--border-strong); border-radius: var(--radius-sm);
      padding: 3px 8px; font-size: 11px; cursor: pointer;
      text-transform: uppercase; letter-spacing: 0.5px;
      font-family: var(--font-sans);
    }
    .bt-btn:hover { background: var(--bg-hover); }
    .bt-filter-chip {
      background: var(--bg-elevated); color: var(--text-secondary);
      border: 1px solid var(--border); border-radius: 99px;
      padding: 3px 10px; font-size: 11px; cursor: pointer;
      font-family: var(--font-sans); transition: all 0.15s;
    }
    .bt-filter-chip:hover { border-color: var(--cyan); color: var(--text); }
    .bt-filter-chip.active {
      background: rgba(10,205,255,0.12); border-color: var(--cyan);
      color: var(--cyan); font-weight: 600;
    }
    .bt-tabs {
      display: flex; gap: 2px; margin-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }
    .bt-tabs a {
      padding: 8px 14px; color: var(--text-secondary); cursor: pointer; font-size: 13px;
      border-bottom: 2px solid transparent; font-weight: 500;
    }
    .bt-tabs a.active { color: var(--amber); border-bottom-color: var(--amber); }
    .bt-subpane { display: none; }
    .bt-subpane.active { display: block; }
    tr.bt-row.selected td { background: var(--bg-elevated) !important; }
    .pill.status-running { animation: pulse 1.5s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.55;} }
    .live-dot {
      display: inline-block; width: 7px; height: 7px; border-radius: 50%;
      background: var(--green); margin-right: 6px; animation: pulse 1.5s infinite;
    }
    th.sortable-h { cursor: pointer; user-select: none; }
    th.sortable-h:hover { color: var(--text); }
    th.sortable-h.sort-asc::after  { content: " ▲"; font-size: 9px; }
    th.sortable-h.sort-desc::after { content: " ▼"; font-size: 9px; }
    select, input[type="text"], input[type="number"] {
      background: var(--bg-input); color: var(--text);
      border: 1px solid var(--border-strong); border-radius: var(--radius-sm);
      padding: 6px 10px; font-size: 13px; font-family: var(--font-sans);
    }
    button, .btn {
      background: var(--bg-elevated); color: var(--text);
      border: 1px solid var(--border-strong); border-radius: var(--radius-sm);
      padding: 6px 14px; font-size: 13px; font-family: var(--font-sans);
      cursor: pointer; transition: background 0.15s;
    }
    button:hover, .btn:hover { background: var(--bg-hover); }
    button.primary, .btn-primary {
      background: var(--amber-dim);
      border-color: rgba(240,180,41,0.3);
      color: var(--amber);
    }
    /* === Mobile-first responsive additions ============================== */
    .nav-hamburger {
      display: none; flex-direction: column; justify-content: space-between;
      width: 32px; height: 22px; background: none; border: none; cursor: pointer;
      padding: 0; margin-left: auto;
    }
    .nav-hamburger span {
      display: block; height: 2px; background: var(--text); border-radius: 2px;
      transition: all 0.2s;
    }
    .nav-drawer {
      position: fixed; top: 0; left: -280px; width: 280px; height: 100vh;
      background: var(--bg-panel); border-right: 1px solid #1e2028;
      z-index: 1000; transition: left 0.25s ease; overflow-y: auto; padding: 20px 0;
    }
    .nav-drawer.open { left: 0; }
    .nav-drawer-header {
      font-family: var(--font-display); font-weight: 700; color: var(--amber);
      font-size: 13px; letter-spacing: 0.1em; padding: 0 20px 20px;
      border-bottom: 1px solid #1e2028; margin-bottom: 8px;
    }
    .nav-drawer a {
      display: block; padding: 12px 20px; color: var(--text-secondary);
      text-decoration: none; font-size: 14px; transition: all 0.15s;
    }
    .nav-drawer a:hover, .nav-drawer a.active {
      color: var(--text); background: var(--bg-elevated);
    }
    .nav-overlay {
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,0.6); z-index: 999;
    }
    .nav-overlay.open { display: block; }
    .bottom-nav {
      display: none; position: fixed; bottom: 0; left: 0; right: 0; height: 64px;
      background: var(--bg-panel); border-top: 1px solid #1e2028;
      grid-template-columns: repeat(5, 1fr); z-index: 200; align-items: stretch;
    }
    .bottom-tab {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 4px; color: var(--text-secondary);
      text-decoration: none; font-size: 10px; min-height: 44px; transition: color 0.15s;
    }
    .bottom-tab svg { width: 20px; height: 20px; }
    .bottom-tab.active, .bottom-tab:hover { color: var(--amber); }
    .table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .table-scroll table { min-width: 500px; }
    /* Responsive 2-col grid that stacks on mobile */
    .grid-2col {
      display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
    }
    @media (max-width: 768px) {
      .topbar-nav { display: none; }
      .nav-hamburger { display: flex; }
      body { font-size: 14px; }
      button, .btn, a.btn, [role="button"] { min-height: 44px; min-width: 44px; }
      /* Prevent any fixed-width grid column from overflowing on narrow screens */
      .grid-2col { grid-template-columns: 1fr; }
    }
    @media (max-width: 480px) {
      body { padding-bottom: 72px; }
      .bottom-nav { display: grid; }
      .topbar { padding: 0 16px; }
      .page-content { padding: 14px; }
      .card { min-height: auto !important; padding: 14px 16px; }
      .grid, .grid-2, .grid2, .grid-2col { grid-template-columns: 1fr !important; }
      .bt-layout { grid-template-columns: 1fr !important; }
      .stat-row { gap: 12px; }
      .stat .v { font-size: 18px; }
      [style*="max-height: 520px"],
      [style*="max-height:520px"] { max-height: 60vh !important; }
      table { font-size: 12px; }
      th, td { padding: 8px 10px; }
    }
  </style>
</head>
<body>
  <nav class="topbar">
    <span class="brand">◈ TRADING STACK</span>
    <span class="topbar-nav" style="display:flex;align-items:center;gap:2px;">
      <a href="/">Command Center</a>
      <a href="/intern/">Digital Intern</a>
      <a href="/trader/" class="{% if initial_tab != 'backtests' %}active{% endif %}">Paper Trader</a>
      <a href="/trader/backtests" class="{% if initial_tab == 'backtests' %}active{% endif %}">Backtests</a>
      <a href="/backtests/compare">Compare</a>
      <a href="/journal">Journal</a>
      <a href="/ops/">Ops View</a>
      <a href="/intern/chat">Chat</a>
      <a href="/system/">System</a>
    </span>
    <button class="nav-hamburger" id="navToggle" aria-label="Menu">
      <span></span><span></span><span></span>
    </button>
  </nav>
  <!-- ─── Global stale-process banner (new 2026-05-16, agent 4) ───
       Always-on, page-wide. Per-panel fetchMaybeStale only degrades the
       endpoints a stale boot is missing; nothing told the operator the
       whole process is behind HEAD — so the self-review mirror silently
       not being injected (exactly the live state on 2026-05-16) was
       invisible from the trader page. Polls /api/build-info. -->
  <div id="global-stale-banner" style="display:none;background:#b71c1c;color:#fff;
       padding:9px 16px;font-size:13px;font-weight:600;text-align:center;
       letter-spacing:0.2px;border-bottom:1px solid #7f0000;">
    <span id="global-stale-text">⚠ Paper-trader is running stale code — restart to apply committed fixes.</span>
  </div>
  <div class="nav-drawer" id="navDrawer">
    <div class="nav-drawer-header">◈ TRADING STACK</div>
    <a href="/">Command Center</a>
    <a href="/intern/">Digital Intern</a>
    <a href="/trader/" class="{% if initial_tab != 'backtests' %}active{% endif %}">Paper Trader</a>
    <a href="/trader/backtests" class="{% if initial_tab == 'backtests' %}active{% endif %}">Backtests</a>
    <a href="/backtests/compare">Compare</a>
    <a href="/journal">Journal</a>
    <a href="/ops/">Ops View</a>
    <a href="/intern/chat">Chat</a>
    <a href="/system/">System</a>
  </div>
  <div class="nav-overlay" id="navOverlay"></div>

  <div class="page-content">
  <h1>Paper Trader</h1>
  <div class="sub" id="hb">loading…</div>

  <!-- ─── Live news data feed (Digital Intern collector pulse) ─── -->
  <div id="data-feed-widget"
       style="display:flex;flex-wrap:wrap;align-items:center;gap:14px;
              background:#11141a;border:1px solid #1f2126;border-radius:6px;
              padding:8px 12px;margin-bottom:14px;font-size:12px;color:#8b929d;">
    <span style="display:inline-flex;align-items:center;gap:6px;color:#dde1e7;">
      <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#00c896;"></span>
      <b style="font-weight:600;letter-spacing:0.04em;">DATA FEED</b>
    </span>
    <span>last 1h: <b id="df-1h" style="color:#dde1e7;">—</b></span>
    <span>24h: <b id="df-24h" style="color:#dde1e7;">—</b></span>
    <span id="df-sources" style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">—</span>
    <span class="muted" id="df-asof" style="font-size:11px;">—</span>
  </div>

  <div class="card" style="margin-bottom:18px;">
    <h2 style="display:flex;justify-content:space-between;align-items:center;">
      <span>Signal Feed — Digital Intern</span>
      <a href="/intern/" style="font-size:11px;color:#0acdff;text-decoration:none;text-transform:none;letter-spacing:normal">View All Signals →</a>
    </h2>
    <ul id="signal-feed" style="margin:0;padding:0;list-style:none;font-size:12px;">
      <li class="muted">loading…</li>
    </ul>
  </div>

  <nav class="tabs">
    <a id="tab-trader-link"    onclick="showTab('trader')">Trader</a>
    <a id="tab-backtests-link" onclick="showTab('backtests')">Backtests</a>
  </nav>

  <!-- ────── Trader pane ────── -->
  <div id="tab-trader" class="tab-pane">

    <!-- ─── Equity Curve (pinned top) ─── -->
    <div class="card" style="margin-bottom:18px;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
        <h2 style="margin:0;">Live portfolio</h2>
        <div style="display:flex;gap:3px;font-size:11px;">
          <button class="bt-filter-chip active" id="eq-range-all" onclick="setEqRange('all')">All</button>
          <button class="bt-filter-chip" id="eq-range-24h" onclick="setEqRange('24h')">24h</button>
          <button class="bt-filter-chip" id="eq-range-7d" onclick="setEqRange('7d')">7d</button>
        </div>
      </div>
      <div class="stat-row" style="margin-bottom:10px;">
        <div class="stat"><div class="l">total value</div><div class="v" id="tv">—</div></div>
        <div class="stat"><div class="l">cash</div><div class="v" id="cash">—</div></div>
        <div class="stat"><div class="l">return vs start</div><div class="v" id="pl">—</div></div>
        <div class="stat"><div class="l">vs SPY (same period)</div><div class="v" id="vs-spy-live">—</div></div>
        <div class="stat"><div class="l">max drawdown</div><div class="v" id="live-maxdd">—</div></div>
        <div class="stat"><div class="l">cash deployed</div><div class="v" id="live-deployed">—</div></div>
      </div>
      <div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;">
        <span style="color:#0acdff;">●</span> Portfolio (% from start) &nbsp;
        <span style="border-top:2px dashed rgba(255,183,77,0.7);display:inline-block;width:16px;vertical-align:middle;"></span> SPY (% from same start) &nbsp;
        <span style="color:var(--text-muted);">↑ buy &nbsp; ↓ sell</span>
      </div>
      <div style="position:relative;height:280px;"><canvas id="eq"></canvas></div>
      <div style="margin-top:6px;">
        <div style="font-size:10px;color:var(--text-muted);margin-bottom:3px;">Drawdown from peak (%)</div>
        <div style="position:relative;height:80px;"><canvas id="eq-dd"></canvas></div>
      </div>
    </div>

    <!-- ─── Daily Briefing (futures + market countdown + urgent news) ─── -->
    <div class="card" id="briefing-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span><span id="briefing-dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#8b929d;margin-right:8px;"></span>Daily briefing</span>
        <span class="muted" id="briefing-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div id="briefing-status" style="font-size:14px;color:#dde1e7;margin-bottom:12px;">loading…</div>
      <div id="briefing-futures" style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:14px;font-size:13px;"></div>
      <div style="font-size:11px;color:#8b929d;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Urgent overnight news</div>
      <ul id="briefing-urgent" style="margin:0;padding:0;list-style:none;font-size:13px;"></ul>
    </div>

    <!-- ─── Session Delta — what materially changed since you last looked (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="sess-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Since you last looked <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— material events in the window, ranked (no snapshot scanning)</span></span>
        <span id="sess-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div style="display:flex;gap:4px;font-size:11px;margin-bottom:10px;">
        <button class="bt-filter-chip" id="sess-w-60" onclick="setSessWindow(60)">1h</button>
        <button class="bt-filter-chip active" id="sess-w-360" onclick="setSessWindow(360)">6h</button>
        <button class="bt-filter-chip" id="sess-w-1440" onclick="setSessWindow(1440)">24h</button>
      </div>
      <div class="muted" id="sess-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <table id="sess-events" style="font-size:12px;">
        <thead><tr>
          <th style="width:62px;">when</th><th style="width:120px;">event</th><th>detail</th>
        </tr></thead>
        <tbody><tr><td colspan="3" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Trade Suggestions (co-pilot) ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Trade suggestions <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— co-pilot, not auto-executed</span></span>
        <span class="muted" id="sug-meta" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div id="sug-summary" style="font-size:12px;color:#8b929d;margin-bottom:10px;">loading…</div>
      <table id="sug-tbl" style="font-size:13px;">
        <thead><tr>
          <th>action</th><th>ticker</th><th class="num">conv.</th>
          <th class="num">price</th><th class="num">qty</th>
          <th class="num">news</th><th class="num">RSI</th>
          <th>reasons</th><th>headline</th>
        </tr></thead><tbody><tr><td colspan="9" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Risk panel (concentration / leverage / age / shock) ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2>Risk panel</h2>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">concentration top1</div><div class="v" id="risk-top1">—</div></div>
        <div class="stat"><div class="l">top3 weight</div><div class="v" id="risk-top3">—</div></div>
        <div class="stat"><div class="l">leveraged %</div><div class="v" id="risk-lev">—</div></div>
        <div class="stat"><div class="l">SPY -3% shock</div><div class="v" id="risk-shock">—</div></div>
        <div class="stat"><div class="l">median age (d)</div><div class="v" id="risk-age">—</div></div>
        <div class="stat"><div class="l">stale positions</div><div class="v" id="risk-stale-n">—</div></div>
      </div>
      <div id="risk-stale-list" style="font-size:12px;color:#dde1e7;"></div>
    </div>

    <!-- ─── Earnings Risk ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Earnings radar <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— scheduled gap risk on holdings &amp; watchlist</span></span>
        <span class="muted" id="er-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="er-meta" style="font-size:11px;margin-bottom:8px;">—</div>
      <ul id="er-list" style="margin:0;padding:0;list-style:none;font-size:13px;">
        <li class="muted">loading…</li>
      </ul>
    </div>

    <!-- ─── Portfolio Greeks (options exposure) ─── -->
    <div class="card" id="greeks-card" style="margin-bottom:18px;display:none;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Portfolio Greeks <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— Black-Scholes, live IV from yfinance</span></span>
        <span class="muted" id="gk-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">net delta</div><div class="v" id="gk-delta">—</div></div>
        <div class="stat"><div class="l">net gamma</div><div class="v" id="gk-gamma">—</div></div>
        <div class="stat"><div class="l">theta / day</div><div class="v" id="gk-theta">—</div></div>
        <div class="stat"><div class="l">vega / 1% IV</div><div class="v" id="gk-vega">—</div></div>
        <div class="stat"><div class="l">gross $ notional</div><div class="v" id="gk-notional">—</div></div>
        <div class="stat"><div class="l">delta % of port</div><div class="v" id="gk-deltapct">—</div></div>
      </div>
      <table id="gk-tbl" style="font-size:13px;">
        <thead><tr>
          <th>ticker</th><th>type</th><th class="num">qty</th>
          <th class="num">expiry / strike</th><th class="num">IV</th>
          <th class="num">Δ delta</th><th class="num">Γ</th>
          <th class="num">Θ / day</th><th class="num">ν / 1%</th>
        </tr></thead><tbody><tr><td colspan="9" class="muted">no option positions</td></tr></tbody>
      </table>
    </div>

    <!-- ─── DecisionScorer per-position predictions ─── -->
    <div class="card" id="scorer-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>ML scorer · per-position outlook <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— predicted 5-day forward return from DecisionScorer MLP</span></span>
        <span class="muted" id="sc-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="sc-meta" style="font-size:11px;margin-bottom:8px;">loading…</div>
      <table id="sc-tbl" style="font-size:13px;">
        <thead><tr>
          <th>ticker</th>
          <th class="num">pred 5d</th>
          <th>verdict</th>
          <th class="num">RSI</th>
          <th class="num">MACD</th>
          <th class="num">mom 5d</th>
          <th class="num">mom 20d</th>
          <th class="num">news</th>
        </tr></thead>
        <tbody><tr><td colspan="8" class="muted">no open stock positions</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Position Thesis Cards (new 2026-05-15) ─── -->
    <div class="card" id="thesis-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Position thesis <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— per-holding integrated view: news, scorer, technicals, last decision, verdict</span></span>
        <span class="muted" id="th-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="th-meta" style="font-size:11px;margin-bottom:10px;">loading…</div>
      <div id="th-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(min(420px,100%),1fr));gap:12px;">
        <div class="muted">loading…</div>
      </div>
    </div>

    <!-- ─── Drawdown Anatomy (new 2026-05-15) ─── -->
    <div class="card" id="dd-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Drawdown anatomy <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— current DD from peak with per-position contribution</span></span>
        <span class="muted" id="dd-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">current equity</div><div class="v" id="dd-current">—</div></div>
        <div class="stat"><div class="l">peak equity</div><div class="v" id="dd-peak">—</div></div>
        <div class="stat"><div class="l">drawdown</div><div class="v" id="dd-pct">—</div></div>
        <div class="stat"><div class="l">trough</div><div class="v" id="dd-trough">—</div></div>
        <div class="stat"><div class="l">time in DD</div><div class="v" id="dd-hours">—</div></div>
        <div class="stat"><div class="l">recovered</div><div class="v" id="dd-rec">—</div></div>
      </div>
      <div style="font-size:13px;color:#dde1e7;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">Per-position contribution</div>
      <table id="dd-tbl" style="font-size:13px;">
        <thead><tr>
          <th>ticker</th><th class="num">qty</th><th class="num">cost</th>
          <th class="num">px</th><th class="num">P/L $</th><th class="num">P/L %</th>
          <th>drag</th>
        </tr></thead>
        <tbody><tr><td colspan="7" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Confidence Calibration + Signal Attribution (new 2026-05-15) ─── -->
    <div class="card" id="cal-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Confidence calibration &amp; signal attribution <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— do high-confidence trades actually win? which signal types pay?</span></span>
        <span class="muted" id="cal-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="cal-meta" style="font-size:11px;margin-bottom:10px;">loading…</div>
      <div class="grid-2col">
        <div>
          <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">By Opus confidence</div>
          <table id="cal-conf-tbl" style="font-size:13px;">
            <thead><tr>
              <th>bucket</th><th class="num">n</th><th class="num">win %</th>
              <th class="num">avg ret</th><th class="num">avg conf</th>
            </tr></thead>
            <tbody><tr><td colspan="5" class="muted">loading…</td></tr></tbody>
          </table>
        </div>
        <div>
          <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">By signal source</div>
          <table id="cal-src-tbl" style="font-size:13px;">
            <thead><tr>
              <th>source</th><th class="num">n</th><th class="num">win %</th>
              <th class="num">avg ret</th><th class="num">best / worst</th>
            </tr></thead>
            <tbody><tr><td colspan="5" class="muted">loading…</td></tr></tbody>
          </table>
        </div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-top:14px;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Recent realized trades</div>
      <table id="cal-recent-tbl" style="font-size:12px;">
        <thead><tr>
          <th>buy → sell</th><th>ticker</th><th class="num">return</th>
          <th class="num">conf</th><th>source</th><th>reasoning</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Capital Deployment & Liquidity (new 2026-05-15, agent 4) ─── -->
    <div class="card" id="liq-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Capital deployment &amp; liquidity <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— is the book pinned with no dry powder?</span></span>
        <span id="liq-status" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="liq-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:12px;">
        <div class="stat"><div class="l">cash</div><div class="v" id="liq-cash">—</div></div>
        <div class="stat"><div class="l">deployed</div><div class="v" id="liq-deployed">—</div></div>
        <div class="stat"><div class="l">positions</div><div class="v" id="liq-npos">—</div></div>
        <div class="stat"><div class="l">top weight</div><div class="v" id="liq-top">—</div></div>
        <div class="stat"><div class="l">unrealized P/L</div><div class="v" id="liq-upl">—</div></div>
        <div class="stat"><div class="l">last entry</div><div class="v" id="liq-entry">—</div></div>
      </div>
      <div id="liq-bar" style="display:flex;height:18px;border-radius:6px;overflow:hidden;background:#0d1117;border:1px solid #1f2126;margin-bottom:6px;"></div>
      <div class="muted" id="liq-bar-legend" style="font-size:11px;margin-bottom:12px;">—</div>
      <div id="liq-flags" style="font-size:12px;color:#dde1e7;"></div>
    </div>

    <!-- ─── Decision Pipeline Health (new 2026-05-15, agent 4) ─── -->
    <div class="card" id="dh-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Decision pipeline health <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— is the live Opus trader actually deciding? NO_DECISION = parse failure</span></span>
        <span id="dh-verdict" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="dh-reason" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">cycles (24h)</div><div class="v" id="dh-total">—</div></div>
        <div class="stat"><div class="l">parse-fail (24h)</div><div class="v" id="dh-fail">—</div></div>
        <div class="stat"><div class="l">fills (24h)</div><div class="v" id="dh-fills">—</div></div>
        <div class="stat"><div class="l">avg confidence</div><div class="v" id="dh-conf">—</div></div>
        <div class="stat"><div class="l">since last fill</div><div class="v" id="dh-lastfill">—</div></div>
        <div class="stat"><div class="l">signals / cycle</div><div class="v" id="dh-sigs">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Action mix (all-time)</div>
      <div id="dh-mix" style="margin-bottom:14px;"><div class="muted">loading…</div></div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Recent cycles</div>
      <table id="dh-tape" style="font-size:12px;">
        <thead><tr>
          <th>time</th><th>outcome</th><th>action</th>
          <th class="num">conf</th><th class="num">signals</th>
        </tr></thead>
        <tbody><tr><td colspan="5" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Decision Failure Forensics (new 2026-05-15, agent 4) ─── -->
    <div class="card" id="df-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Decision failure forensics <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— WHY a cycle produced no decision, with the raw Opus excerpt</span></span>
        <span id="df-verdict" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="df-reason" style="font-size:12px;margin-bottom:6px;">loading…</div>
      <div id="df-hint" style="font-size:12px;color:#ffd479;margin-bottom:12px;"></div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">failures</div><div class="v" id="df-nfail">—</div></div>
        <div class="stat"><div class="l">rate (24h)</div><div class="v" id="df-rate">—</div></div>
        <div class="stat"><div class="l">retry-exhausted</div><div class="v" id="df-retry">—</div></div>
        <div class="stat"><div class="l">dominant mode</div><div class="v" id="df-dom" style="font-size:14px;">—</div></div>
        <div class="stat"><div class="l">open mkt fail%</div><div class="v" id="df-open">—</div></div>
        <div class="stat"><div class="l">closed mkt fail%</div><div class="v" id="df-closed">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Failure modes</div>
      <div id="df-mix" style="margin-bottom:12px;"><div class="muted">loading…</div></div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Hourly parse-fail (last 24h)</div>
      <div id="df-hourly" style="display:flex;align-items:flex-end;gap:3px;height:46px;margin-bottom:14px;"><div class="muted">—</div></div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Recent failures — raw Opus excerpt</div>
      <table id="df-tape" style="font-size:12px;">
        <thead><tr>
          <th>time</th><th>mode</th><th>mkt</th><th>excerpt</th>
        </tr></thead>
        <tbody><tr><td colspan="4" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Decision Drought Drift (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="drought-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Decision drought drift <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— what the bot's <em>inaction</em> cost: portfolio vs S&amp;P while it wasn't trading</span></span>
        <span id="dd-verdict" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="dd-reason" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">fills / cycles</div><div class="v" id="dd-fills">—</div></div>
        <div class="stat"><div class="l">droughts</div><div class="v" id="dd-n">—</div></div>
        <div class="stat"><div class="l">paralysis droughts</div><div class="v" id="dd-npar">—</div></div>
        <div class="stat"><div class="l">involuntary alpha bleed</div><div class="v" id="dd-bleed">—</div></div>
      </div>
      <div id="drought-current" style="font-size:12px;background:#0d1117;border:1px solid #1f2126;border-radius:6px;padding:10px 12px;margin-bottom:14px;color:#8b929d;">loading…</div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Droughts (newest first) — alpha = portfolio% − S&amp;P% over the idle window</div>
      <table id="dd-tape" style="font-size:12px;">
        <thead><tr>
          <th>start</th><th class="num">hrs</th><th class="num">cyc</th><th>kind</th>
          <th class="num">ND%</th><th class="num">port%</th><th class="num">spy%</th><th class="num">alpha%</th>
        </tr></thead>
        <tbody><tr><td colspan="8" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── News Edge (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="ne-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>News edge <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— does a high ai_score headline actually predict the move? (SPY-abnormal)</span></span>
        <span id="ne-verdict" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="ne-reason" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">lookback</div><div class="v" id="ne-days">—</div></div>
        <div class="stat"><div class="l">articles</div><div class="v" id="ne-narts">—</div></div>
        <div class="stat"><div class="l">resolved</div><div class="v" id="ne-nres">—</div></div>
        <div class="stat"><div class="l">tickers priced</div><div class="v" id="ne-ntk">—</div></div>
        <div class="stat"><div class="l">ref horizon</div><div class="v" id="ne-ref">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Forward return by ai_score band — mean abnormal % (raw in muted)</div>
      <table id="ne-bands" style="font-size:12px;margin-bottom:14px;">
        <thead><tr>
          <th>ai_score band</th><th class="num">n@ref</th>
          <th class="num">1d abn</th><th class="num">3d abn</th><th class="num">5d abn</th>
          <th class="num">ref hit%</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Urgent vs normal — 3d abnormal %</div>
      <div id="ne-urg" style="font-size:12px;color:#8b929d;">—</div>
    </div>

    <!-- ─── Scorer Reliability + Confidence Intervals (new 2026-05-15, agent 4) ─── -->
    <div class="card" id="scrl-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Scorer reliability <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— how far DecisionScorer predictions actually land from reality</span></span>
        <span class="muted" id="scrl-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="scrl-meta" style="font-size:11px;margin-bottom:10px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">directional accuracy</div><div class="v" id="scrl-dir">—</div></div>
        <div class="stat"><div class="l">mean abs error</div><div class="v" id="scrl-mae">—</div></div>
        <div class="stat"><div class="l">90% residual band</div><div class="v" id="scrl-band">—</div></div>
        <div class="stat"><div class="l">replay samples</div><div class="v" id="scrl-n">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Held positions — prediction with empirical band</div>
      <table id="scrl-pos" style="font-size:13px;margin-bottom:16px;">
        <thead><tr>
          <th>ticker</th><th class="num">pred 5d</th><th class="num">likely range</th>
          <th>verdict</th><th class="num">band hit %</th><th>trust</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Calibration by prediction band</div>
      <table id="scrl-cal" style="font-size:12px;">
        <thead><tr>
          <th>predicted band</th><th class="num">n</th><th class="num">mean actual</th>
          <th class="num">residual P10/P90</th><th class="num">MAE</th><th class="num">dir. acc.</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Scorer ⇄ Opus Disagreement (new 2026-05-16, agent 4 feature-dev) ─── -->
    <div class="card" id="dis-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Scorer ⇄ Opus disagreement <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— where the ML safety net and Opus are fighting on held positions</span></span>
        <span class="muted" id="dis-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="dis-meta" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">high conflict</div><div class="v" id="dis-high">—</div></div>
        <div class="stat"><div class="l">medium</div><div class="v" id="dis-med">—</div></div>
        <div class="stat"><div class="l">aligned</div><div class="v" id="dis-aln">—</div></div>
        <div class="stat"><div class="l">positions</div><div class="v" id="dis-n">—</div></div>
      </div>
      <table id="dis-tbl" style="font-size:13px;">
        <thead><tr>
          <th>ticker</th><th>scorer verdict</th><th class="num">pred 5d</th>
          <th>last Opus action</th><th>conflict</th><th>read</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Portfolio Analytics ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2>Portfolio analytics</h2>
      <div class="stat-row" style="margin-bottom:18px;">
        <div class="stat"><div class="l">today's P/L</div><div class="v" id="an-daily">—</div></div>
        <div class="stat"><div class="l">max drawdown</div><div class="v" id="an-dd">—</div></div>
        <div class="stat"><div class="l">sharpe (ann.)</div><div class="v" id="an-sharpe">—</div></div>
        <div class="stat"><div class="l">win rate</div><div class="v" id="an-winrate">—</div></div>
        <div class="stat"><div class="l">avg winner</div><div class="v" id="an-avgw">—</div></div>
        <div class="stat"><div class="l">avg loser</div><div class="v" id="an-avgl">—</div></div>
        <div class="stat"><div class="l">realized P/L</div><div class="v" id="an-realized">—</div></div>
      </div>
      <div class="stat-row" style="margin-bottom:18px;">
        <div class="stat"><div class="l">profit factor</div><div class="v" id="an-pf">—</div></div>
        <div class="stat"><div class="l">sortino (ann.)</div><div class="v" id="an-sortino">—</div></div>
        <div class="stat"><div class="l">calmar</div><div class="v" id="an-calmar">—</div></div>
        <div class="stat"><div class="l">S&amp;P β</div><div class="v" id="an-beta">—</div></div>
        <div class="stat"><div class="l">S&amp;P corr</div><div class="v" id="an-corr">—</div></div>
        <div class="stat"><div class="l">avg hold</div><div class="v" id="an-hold">—</div></div>
      </div>
      <div style="font-size:13px;color:#dde1e7;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">Sector exposure</div>
      <div id="an-sector-bar" style="display:flex;height:22px;border-radius:6px;overflow:hidden;background:#0d1117;border:1px solid #1f2126;margin-bottom:6px;"></div>
      <div id="an-sector-legend" style="display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:#dde1e7;"></div>
    </div>

    <!-- ─── Trade Asymmetry / Behavioural Edge (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="ta-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Behavioural edge <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— payoff ratio, breakeven win-rate, the disposition effect</span></span>
        <span id="ta-verdict" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="ta-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">expectancy / trade</div><div class="v" id="ta-exp">—</div></div>
        <div class="stat"><div class="l">payoff ratio</div><div class="v" id="ta-payoff">—</div></div>
        <div class="stat"><div class="l">win-rate (actual)</div><div class="v" id="ta-wr">—</div></div>
        <div class="stat"><div class="l">breakeven win-rate</div><div class="v" id="ta-be">—</div></div>
        <div class="stat"><div class="l">realized P/L</div><div class="v" id="ta-real">—</div></div>
      </div>
      <div class="stat-row">
        <div class="stat"><div class="l">round-trips (W/L)</div><div class="v" id="ta-n">—</div></div>
        <div class="stat"><div class="l">avg winner</div><div class="v" id="ta-avgw">—</div></div>
        <div class="stat"><div class="l">avg loser</div><div class="v" id="ta-avgl">—</div></div>
        <div class="stat"><div class="l">winner / loser hold</div><div class="v" id="ta-hold">—</div></div>
        <div class="stat"><div class="l">disposition gap</div><div class="v" id="ta-disp">—</div></div>
      </div>
    </div>

    <!-- ─── Loser autopsy (per-closed-losing-trade post-mortem) ─── -->
    <div class="card" id="lautopsy-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Loser autopsy <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— why each closed trade lost: verbatim thesis, hold, failure mode</span></span>
        <span id="lautopsy-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="lautopsy-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">total realised loss</div><div class="v" id="lautopsy-total">—</div></div>
        <div class="stat"><div class="l">losing round-trips</div><div class="v" id="lautopsy-n">—</div></div>
        <div class="stat"><div class="l">avg loss</div><div class="v" id="lautopsy-avg">—</div></div>
        <div class="stat"><div class="l">median hold</div><div class="v" id="lautopsy-hold">—</div></div>
        <div class="stat"><div class="l">dominant mode</div><div class="v" id="lautopsy-mode">—</div></div>
      </div>
      <table id="lautopsy-tbl" style="font-size:12px;width:100%;">
        <thead><tr>
          <th>ticker</th><th class="num">P/L $</th><th class="num">P/L %</th>
          <th class="num">hold d</th><th>mode</th><th>opening thesis</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Concentration honesty (do the held names move together?) ─── -->
    <div class="card" id="pcorr-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Concentration honesty <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— pairwise return ρ &amp; effective independent bets</span></span>
        <span id="pcorr-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="pcorr-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">mean pairwise ρ</div><div class="v" id="pcorr-meanrho">—</div></div>
        <div class="stat"><div class="l">effective bets</div><div class="v" id="pcorr-effbets">—</div></div>
        <div class="stat"><div class="l">naive eff. positions</div><div class="v" id="pcorr-effnaive">—</div></div>
        <div class="stat"><div class="l">top weight</div><div class="v" id="pcorr-topw">—</div></div>
        <div class="stat"><div class="l">most-coupled pair</div><div class="v" id="pcorr-maxpair">—</div></div>
      </div>
      <table id="pcorr-tbl" style="font-size:12px;width:100%;">
        <thead><tr><th>pair</th><th class="num">ρ</th></tr></thead>
        <tbody><tr><td colspan="2" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Capital Paralysis & Unlock Ladder (wired 2026-05-16, agent 4) ─── -->
    <div class="card" id="cp-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Capital paralysis <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— the trap, its cost, and the single sale that unlocks it</span></span>
        <span id="cp-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="cp-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">cash</div><div class="v" id="cp-cash">—</div></div>
        <div class="stat"><div class="l">deployed</div><div class="v" id="cp-dep">—</div></div>
        <div class="stat"><div class="l">can act?</div><div class="v" id="cp-canact">—</div></div>
        <div class="stat"><div class="l">cycles since fill</div><div class="v" id="cp-stuck">—</div></div>
        <div class="stat"><div class="l">alpha bled</div><div class="v" id="cp-bleed">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Unlock ladder — desk cut-priority (biggest loser first)</div>
      <table id="cp-ladder" style="font-size:12px;">
        <thead><tr>
          <th>ticker</th><th class="num">weight%</th><th class="num">P/L%</th>
          <th class="num">frees $</th><th class="num">cash if sold alone</th><th>unlocks?</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Open-Book Alpha — selection vs market (wired 2026-05-16, agent 4) ─── -->
    <div class="card" id="oa-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Open-book alpha <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— how much of the open P&amp;L is selection vs just SPY</span></span>
        <span id="oa-status" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="oa-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">book alpha vs SPY</div><div class="v" id="oa-alpha">—</div></div>
        <div class="stat"><div class="l">net excess $</div><div class="v" id="oa-excess">—</div></div>
        <div class="stat"><div class="l">unrealized $</div><div class="v" id="oa-unreal">—</div></div>
        <div class="stat"><div class="l">SPY-equiv $</div><div class="v" id="oa-spyeq">—</div></div>
        <div class="stat"><div class="l">anchored names</div><div class="v" id="oa-n">—</div></div>
      </div>
      <table id="oa-rows" style="font-size:12px;">
        <thead><tr>
          <th>ticker</th><th class="num">pos %</th><th class="num">SPY %</th>
          <th class="num">alpha %</th><th class="num">excess $</th>
        </tr></thead>
        <tbody><tr><td colspan="5" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Overtrading / re-entry churn (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="churn-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Overtrading &amp; churn <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— how often it re-buys a name it just closed, and how fast</span></span>
        <span id="churn-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="churn-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">fast re-entries</div><div class="v" id="churn-reentry">—</div></div>
        <div class="stat"><div class="l">round-trips / day</div><div class="v" id="churn-rtpd">—</div></div>
        <div class="stat"><div class="l">median hold</div><div class="v" id="churn-hold">—</div></div>
        <div class="stat"><div class="l">sub-day trips</div><div class="v" id="churn-subday">—</div></div>
        <div class="stat"><div class="l">loss in &lt;1d trips</div><div class="v" id="churn-lossconc">—</div></div>
      </div>
      <div style="font-size:12px;color:#dde1e7;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px;">Fastest same-name re-entries</div>
      <table id="churn-events" style="font-size:12px;">
        <thead><tr>
          <th>ticker</th><th class="num">gap (d)</th><th class="num">prior P/L $</th><th>closed → re-bought</th>
        </tr></thead>
        <tbody><tr><td colspan="4" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Thesis drift — entry rationale vs reality (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="tdrift-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Thesis drift <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— is the reason each position was opened for still true?</span></span>
        <span id="tdrift-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="tdrift-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <table id="tdrift-rows" style="font-size:12px;">
        <thead><tr>
          <th>ticker</th><th>health</th><th class="num">P/L %</th>
          <th class="num">held (d)</th><th>entry rationale → current drift</th>
        </tr></thead>
        <tbody><tr><td colspan="5" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Signal-feed health — is the trader even seeing news? (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="fh-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Signal-feed health <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— is the live trader receiving any news, or flying blind?</span></span>
        <span id="fh-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="fh-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">blind streak (0-signal cycles)</div><div class="v" id="fh-streak">—</div></div>
        <div class="stat"><div class="l">newest live article age</div><div class="v" id="fh-age">—</div></div>
        <div class="stat"><div class="l">live articles (2h / 24h)</div><div class="v" id="fh-live">—</div></div>
        <div class="stat"><div class="l">split-brain DB</div><div class="v" id="fh-split">—</div></div>
      </div>
      <div class="muted" id="fh-path" style="font-size:12px;word-break:break-all;">—</div>
    </div>

    <!-- ─── Decision reliability — true current-regime parse-fail rate (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="dr-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Decision reliability <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— headline NO_DECISION % vs the true post-restart rate</span></span>
        <span id="dr-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="dr-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">current-regime fail</div><div class="v" id="dr-cur">—</div></div>
        <div class="stat"><div class="l">headline fail (legacy-incl)</div><div class="v" id="dr-head">—</div></div>
        <div class="stat"><div class="l">current sample / total</div><div class="v" id="dr-n">—</div></div>
        <div class="stat"><div class="l">legacy dead rows</div><div class="v" id="dr-legacy">—</div></div>
        <div class="stat"><div class="l">dead cycles / day</div><div class="v" id="dr-dead">—</div></div>
      </div>
      <div class="muted" id="dr-mode" style="font-size:12px;">—</div>
    </div>

    <!-- ─── Funded suggestions — which idea is fundable, and the sale that funds it (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="fund-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Funded suggestions <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— which BUY/ADD idea is fundable, and the sale that unlocks it</span></span>
        <span id="fund-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="fund-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">actionable ideas</div><div class="v" id="fund-n">—</div></div>
        <div class="stat"><div class="l">funded now</div><div class="v" id="fund-funded">—</div></div>
        <div class="stat"><div class="l">unlockable via sale</div><div class="v" id="fund-unlock">—</div></div>
        <div class="stat"><div class="l">unfundable</div><div class="v" id="fund-unfund">—</div></div>
        <div class="stat"><div class="l">pairing</div><div class="v" id="fund-pair">—</div></div>
      </div>
      <table id="fund-rows" style="font-size:12px;">
        <thead><tr>
          <th>idea</th><th class="num">conv</th><th class="num">notional $</th>
          <th>fundability</th><th>sell to fund</th><th class="num">frees $</th>
        </tr></thead>
        <tbody><tr><td colspan="6" class="muted">loading…</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Signal follow-through — is the trader using its own news edge? (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="sft-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Signal follow-through <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— did it act on the news it saw, and did acting pay (vs SPY)?</span></span>
        <span id="sft-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="sft-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <div class="stat-row" style="margin-bottom:14px;">
        <div class="stat"><div class="l">follow-through</div><div class="v" id="sft-ft">—</div></div>
        <div class="stat"><div class="l">acted / ignored</div><div class="v" id="sft-ai">—</div></div>
        <div class="stat"><div class="l">selection edge (ref)</div><div class="v" id="sft-edge">—</div></div>
        <div class="stat"><div class="l">acted abn% @ref</div><div class="v" id="sft-acted">—</div></div>
        <div class="stat"><div class="l">ignored abn% @ref</div><div class="v" id="sft-ign">—</div></div>
        <div class="stat"><div class="l">resolved / signals</div><div class="v" id="sft-n">—</div></div>
      </div>
      <div class="muted" id="sft-meta" style="font-size:12px;">—</div>
    </div>

    <!-- ─── News source edge — which collector is worth trusting? (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="se-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>News source edge <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— which of ~17 collectors' scored headlines actually precede the move (vs SPY)?</span></span>
        <span id="se-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="se-headline" style="font-size:12px;margin-bottom:12px;">loading…</div>
      <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <thead><tr style="text-align:left;color:#8b929d;">
          <th style="padding:4px 6px;">collector</th>
          <th style="padding:4px 6px;">abn% @ref</th>
          <th style="padding:4px 6px;">hit</th>
          <th style="padding:4px 6px;">resolved</th>
          <th style="padding:4px 6px;">verdict</th>
        </tr></thead>
        <tbody id="se-rows"><tr><td colspan="5" class="muted" style="padding:6px;">—</td></tr></tbody>
      </table>
      <div class="muted" id="se-meta" style="font-size:12px;margin-top:10px;">—</div>
    </div>

    <!-- ─── Behavioural scorecard — verdict-alignment router (new 2026-05-16, agent 4) ─── -->
    <div class="card" id="score-card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Behavioural scorecard <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— do the independent behavioural checks agree on a problem? (no grade, just concordance)</span></span>
        <span id="score-state" style="font-size:12px;padding:3px 10px;border-radius:4px;background:#1f2126;color:#8b929d;">—</span>
      </h2>
      <div class="muted" id="score-headline" style="font-size:12px;margin-bottom:10px;">loading…</div>
      <div id="score-focus" style="font-size:12px;margin-bottom:8px;"></div>
      <div id="score-concordance" style="font-size:12px;margin-bottom:12px;"></div>
      <table style="font-size:12px;width:100%;">
        <thead><tr style="text-align:left;color:#8b929d;">
          <th style="padding:4px 6px;">check</th><th style="padding:4px 6px;">verdict</th><th style="padding:4px 6px;">what it says</th>
        </tr></thead>
        <tbody id="score-rows"><tr><td colspan="3" class="muted" style="padding:6px;">—</td></tr></tbody>
      </table>
    </div>

    <!-- ─── Sector Pulse ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Sector pulse — semis &amp; optical</span>
        <span class="muted" id="sp-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div id="sp-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;">
        <div class="muted">loading…</div>
      </div>
    </div>

    <!-- ─── DRAM / Semis Sector Heatmap ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>DRAM / semis heatmap <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— 5d momentum &amp; news pulse</span></span>
        <span class="muted" id="hm-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="hm-bench" style="font-size:11px;margin-bottom:10px;">SOXX baseline: —</div>
      <div id="hm-grid"><div class="muted">loading…</div></div>
    </div>

    <!-- ─── Deduped News Feed ─── -->
    <div class="card" style="margin-bottom:18px;">
      <h2 style="display:flex;justify-content:space-between;align-items:center;">
        <span>Deduped signals <span class="muted" style="font-size:11px;text-transform:none;letter-spacing:normal;font-weight:normal;">— syndication collapsed, urgency decayed (halflife 4h)</span></span>
        <span class="muted" id="nd-asof" style="font-size:11px;text-transform:none;letter-spacing:normal;">—</span>
      </h2>
      <div class="muted" id="nd-meta" style="font-size:11px;margin-bottom:8px;">—</div>
      <ul id="nd-list" style="margin:0;padding:0;list-style:none;font-size:13px;">
        <li class="muted">loading…</li>
      </ul>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Open positions</h2>
        <div class="table-scroll">
        <table id="pos-tbl">
          <thead><tr>
            <th>ticker</th><th>type</th><th class="num">qty</th>
            <th class="num">avg</th><th class="num">now</th>
            <th class="num">total $</th><th class="num">% port</th>
            <th class="num">P/L</th>
          </tr></thead><tbody></tbody>
        </table>
        </div>
      </div>
      <div class="card">
        <h2>Recent trades</h2>
        <div class="table-scroll">
        <table id="trades-tbl">
          <thead><tr>
            <th>time</th><th>action</th><th>ticker</th>
            <th class="num">qty</th><th class="num">price</th><th>reason</th>
          </tr></thead><tbody></tbody>
        </table>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:18px;">
      <h2>Decision log</h2>
      <div class="table-scroll">
      <table id="dec-tbl">
        <thead><tr>
          <th>time</th><th>open?</th><th class="num">signals</th>
          <th>action</th><th class="num">equity</th><th>reasoning</th>
        </tr></thead><tbody></tbody>
      </table>
      </div>
    </div>
  </div>

  <!-- ────── Backtests pane ────── -->
  <div id="tab-backtests" class="tab-pane">
    <div class="bt-layout">
      <aside class="bt-sidebar card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
          <h2 style="margin:0;">Runs</h2>
          <div style="display:flex;gap:4px;">
            <button class="bt-btn" onclick="btToggleAll(true)">all</button>
            <button class="bt-btn" onclick="btToggleAll(false)">none</button>
          </div>
        </div>
        <div id="bt-legend"></div>
      </aside>

      <div class="bt-main">
        <div class="card" style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;">
            <div>
              <h2 style="margin:0 0 4px;">Backtest equity curves</h2>
              <div class="progress-label" id="bt-progress-label">—</div>
            </div>
            <div style="text-align:right;font-size:12px;color:#8b929d;">
              <div id="bt-live-indicator"></div>
              <div id="bt-last-updated">last update: —</div>
            </div>
          </div>
          <div class="progress-wrap" style="margin:8px 0 14px"><div class="progress-bar" id="bt-progress-bar" style="width:0%"></div></div>
          <div class="bt-headline">
            <div class="stat"><div class="l">avg annualized</div><div class="v" id="bt-avg-ann">—</div></div>
            <div class="stat"><div class="l">avg total %</div><div class="v" id="bt-avg">—</div></div>
            <div class="stat"><div class="l">best</div><div class="v" id="bt-best">—</div></div>
            <div class="stat"><div class="l">worst</div><div class="v" id="bt-worst">—</div></div>
            <div class="stat"><div class="l">beat SPY</div><div class="v" id="bt-beat">—</div></div>
            <div class="stat"><div class="l">win rate</div><div class="v" id="bt-winrate">—</div></div>
            <div class="stat"><div class="l">filtered runs</div><div class="v" id="bt-filtered-count">—</div></div>
          </div>
          <!-- Filter + mode bar -->
          <div style="display:flex;align-items:center;gap:8px;margin:10px 0 6px;flex-wrap:wrap;font-size:12px;">
            <span style="color:var(--text-secondary);">Window:</span>
            <div id="bt-win-filter" style="display:flex;gap:4px;flex-wrap:wrap;">
              <button class="bt-filter-chip active" data-min="0" data-max="99" onclick="setBtWinFilter(this)">All</button>
              <button class="bt-filter-chip" data-min="0" data-max="1.5" onclick="setBtWinFilter(this)">≤1yr</button>
              <button class="bt-filter-chip" data-min="1.5" data-max="2.5" onclick="setBtWinFilter(this)">2yr</button>
              <button class="bt-filter-chip" data-min="2.5" data-max="3.5" onclick="setBtWinFilter(this)">3yr</button>
              <button class="bt-filter-chip" data-min="3.5" data-max="5.5" onclick="setBtWinFilter(this)">4–5yr</button>
              <button class="bt-filter-chip" data-min="5.5" data-max="99" onclick="setBtWinFilter(this)">6–10yr</button>
            </div>
            <div style="display:flex;gap:3px;margin-left:auto;">
              <button id="mode-agg" class="bt-filter-chip active" onclick="setChartMode('aggregate')">Distribution</button>
              <button id="mode-ind" class="bt-filter-chip" onclick="setChartMode('individual')">Individual</button>
            </div>
          </div>
          <!-- Aggregate mode legend / Individual mode limit control -->
          <div id="agg-legend" style="font-size:11px;color:var(--text-secondary);margin-bottom:6px;">
            <span style="display:inline-flex;align-items:center;gap:4px;margin-right:10px;">
              <span style="display:inline-block;width:24px;height:3px;background:#0acdff;border-radius:2px;"></span>Median
            </span>
            <span style="display:inline-flex;align-items:center;gap:4px;margin-right:10px;">
              <span style="display:inline-block;width:24px;height:8px;background:rgba(10,205,255,0.25);border-radius:2px;"></span>P25–P75
            </span>
            <span style="display:inline-flex;align-items:center;gap:4px;margin-right:10px;">
              <span style="display:inline-block;width:24px;height:8px;background:rgba(10,205,255,0.08);border-radius:2px;"></span>P5–P95
            </span>
            <span style="display:inline-flex;align-items:center;gap:4px;margin-right:10px;">
              <span style="display:inline-block;width:24px;height:2px;background:rgba(180,180,180,0.7);border-radius:2px;border-top:2px dashed rgba(180,180,180,0.7);"></span>Actual SPY median
            </span>
            <span id="agg-n-label" style="color:var(--text-muted);"></span>
          </div>
          <div id="ind-controls" style="display:none;font-size:12px;color:var(--text-secondary);margin-bottom:6px;">
            Show last
            <input id="bt-chart-limit" type="range" min="5" max="500" step="5" value="100"
              style="width:80px;cursor:pointer;accent-color:#0acdff;vertical-align:middle;"
              oninput="document.getElementById('bt-chart-limit-val').textContent=this.value; redrawChart()">
            <span id="bt-chart-limit-val">100</span> runs · X = day offset · Y = % return from start
          </div>
          <!-- Main equity chart -->
          <div style="position:relative;height:380px;"><canvas id="bt-chart"></canvas></div>
          <!-- Drawdown sub-chart (aggregate mode only) -->
          <div id="bt-drawdown-wrap" style="position:relative;height:120px;margin-top:8px;">
            <div style="font-size:10px;color:var(--text-muted);margin-bottom:4px;">Max drawdown distribution (% below peak, by day from start)</div>
            <canvas id="bt-dd-chart"></canvas>
          </div>
        </div>

        <!-- ── Multi-dimensional analysis ── -->
        <div class="card" style="margin-bottom:14px;">
          <h2 style="margin:0 0 2px;">Multi-dimensional analysis</h2>
          <div style="color:var(--text-secondary);font-size:12px;margin-bottom:14px;">
            Duration × era × return — three ways to read the same 500+ runs simultaneously.
          </div>

          <!-- Row 1: Scatter (duration vs annualized, colored by era) -->
          <div style="margin-bottom:20px;">
            <div style="font-size:12px;font-weight:600;color:var(--text);margin-bottom:4px;letter-spacing:0.02em;">
              Duration vs annualized return
              <span style="font-weight:400;color:var(--text-muted);font-size:11px;margin-left:6px;">each dot = one run · click to drill in · color = market era</span>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;font-size:11px;" id="bt-era-legend"></div>
            <div style="position:relative;height:320px;"><canvas id="bt-scatter"></canvas></div>
          </div>

          <!-- Row 2: Era × Duration heatmap -->
          <div>
            <div style="font-size:12px;font-weight:600;color:var(--text);margin-bottom:4px;letter-spacing:0.02em;">
              Era × duration performance heatmap
              <span style="font-weight:400;color:var(--text-muted);font-size:11px;margin-left:6px;">avg annualized return % per cell · (n = run count)</span>
            </div>
            <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">
              <div id="bt-heatmap"></div>
            </div>
          </div>
        </div>

        <div class="card" style="margin-bottom:14px;">
          <h2 style="margin:0 0 4px;">Model progress — return by cycle</h2>
          <div style="color:#8b929d;font-size:12px;margin-bottom:10px;">Best / avg / worst return per cycle of 5 runs. Upward trend = model improving.</div>
          <div style="position:relative;height:220px;"><canvas id="mp-chart"></canvas></div>
        </div>

        <div class="card" id="validation-card" style="margin-bottom:14px;">
          <h2 style="margin:0 0 4px;">Signal Integrity</h2>
          <div style="color:#8b929d;font-size:12px;margin-bottom:14px;">
            Permutation test + label contamination audit. Runs every 10 backtest cycles in the background.
            <br>SIGNIFICANT (p&lt;0.05) means signal time-ordering carries real predictive value, not random noise.
          </div>
          <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div style="min-width:220px;">
              <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.05em;">Permutation Test</div>
              <div id="val-perm-verdict" style="font-weight:600;font-size:18px;margin-top:4px;">—</div>
              <div style="font-size:12px;color:var(--text-secondary);margin-top:2px;">
                <span id="val-perm-pvalue">p=—</span> · <span id="val-perm-zscore">z=—</span>
              </div>
              <div id="val-perm-original" style="font-size:12px;margin-top:6px;"></div>
              <div id="val-perm-shuffled" style="font-size:12px;color:var(--text-secondary);"></div>
            </div>
            <div style="min-width:220px;">
              <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.05em;">Label Contamination</div>
              <div id="val-contam-rate" style="font-weight:600;font-size:18px;margin-top:4px;">—</div>
              <div style="font-size:12px;color:var(--text-secondary);margin-top:6px;">
                High = Claude labels carry hindsight<br>(retroactively-collected articles)
              </div>
              <div id="val-contam-detail" style="font-size:12px;margin-top:6px;color:var(--text-secondary);"></div>
            </div>
            <div style="min-width:220px;">
              <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.05em;">Last Validation</div>
              <div id="val-last-cycle" style="font-weight:600;font-size:18px;margin-top:4px;">—</div>
              <div id="val-last-window" style="font-size:12px;color:var(--text-secondary);margin-top:2px;"></div>
              <div id="val-last-when" style="font-size:11px;color:var(--text-muted);margin-top:6px;"></div>
            </div>
          </div>
        </div>

        <div class="card" style="margin-bottom:14px;">
          <h2>Runs table — click a row to highlight</h2>
          <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">
          <table id="bt-tbl" class="sortable">
            <thead><tr>
              <th data-k="run_id">#</th>
              <th data-k="status">status</th>
              <th data-k="total_return_pct" class="num">total %</th>
              <th data-k="annualized_return_pct" class="num">ann. %/yr</th>
              <th data-k="vs_spy_pct" class="num">vs SPY</th>
              <th data-k="start_date">window</th>
              <th data-k="duration_days" class="num">dur.</th>
              <th data-k="n_trades" class="num">trades</th>
              <th data-k="n_decisions" class="num">signals</th>
            </tr></thead><tbody></tbody>
          </table>
          </div>
        </div>

        <div class="card" id="bt-detail" style="display:none;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <h2 style="margin:0;">Run <span id="bt-detail-id">—</span> detail</h2>
            <button class="bt-btn" onclick="closeDetail()">close</button>
          </div>
          <div id="bt-detail-meta" class="muted" style="font-size:13px;margin-bottom:12px;"></div>
          <div class="bt-tabs">
            <a id="bt-tab-trades-link" class="active" onclick="showBtSubtab('trades')">Trades</a>
            <a id="bt-tab-decisions-link" onclick="showBtSubtab('decisions')">Decisions</a>
          </div>
          <div id="bt-tab-trades" class="bt-subpane active">
            <table id="bt-trades-tbl">
              <thead><tr>
                <th>date</th><th>action</th><th>ticker</th>
                <th class="num">qty</th><th class="num">price</th>
                <th class="num">value</th><th>reason</th>
              </tr></thead><tbody></tbody>
            </table>
          </div>
          <div id="bt-tab-decisions" class="bt-subpane">
            <table id="bt-decisions-tbl">
              <thead><tr>
                <th>date</th><th>action</th><th>ticker</th>
                <th>status</th><th>detail</th><th class="num">portfolio $</th>
              </tr></thead><tbody></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
const fmt = (n, d=2) => (n == null ? "—" : Number(n).toLocaleString(undefined, {minimumFractionDigits:d, maximumFractionDigits:d}));
const dollar = n => (n == null ? "—" : "$" + fmt(n));
const dt = s => s ? s.replace("T", " ").slice(0,16) : "";

const INITIAL_TAB = "{{ initial_tab }}";
const API_PREFIX = "{{ api_prefix }}";
const RUN_COLORS = [
  "#00d4ff","#ff6b35","#7fff00","#ff3cac","#ffd700",
  "#00ff9f","#ff1744","#e040fb","#40c4ff","#ff9100"
];
const SPY_COLOR = "#888888";

function showTab(name) {
  document.querySelectorAll(".tab-pane").forEach(el => el.classList.remove("active"));
  document.querySelectorAll("nav.tabs a").forEach(el => el.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  document.getElementById("tab-" + name + "-link").classList.add("active");
  if (name === "backtests" && !btLoaded) loadBacktests();
  // Update URL without reload
  if (history.replaceState) history.replaceState(null, "", name === "trader" ? "/" : "/backtests");
}

// ───────── Trader pane ─────────
let chart;
let ddChart;
let eqRange = "all";   // "all" | "24h" | "7d"
let _lastEquity = [];  // cache for range filtering
let _lastTrades = [];

function setEqRange(r) {
  eqRange = r;
  ["all","24h","7d"].forEach(k => {
    const el = document.getElementById("eq-range-"+k);
    if (el) el.classList.toggle("active", k === r);
  });
  drawEquityChart(_lastEquity, _lastTrades);
}

function _filterEqByRange(eq) {
  if (eqRange === "all" || !eq.length) return eq;
  const cutMs = eqRange === "24h" ? 86400000 : 7*86400000;
  const cutoff = new Date(Date.now() - cutMs).toISOString();
  return eq.filter(p => p.timestamp >= cutoff);
}

function drawEquityChart(eq, trades) {
  const filtered = _filterEqByRange(eq);
  if (!filtered.length) return;

  // Normalize: portfolio and SPY both as % from first point in filtered range
  const baseVal = filtered[0].total_value || 1000;
  const baseSpy = filtered[0].sp500_price || 1;
  const labels = filtered.map(p => p.timestamp.replace("T"," ").slice(0,16));
  const portPct = filtered.map(p => ((p.total_value / baseVal) - 1) * 100);
  const spyPct  = filtered.map(p => p.sp500_price ? ((p.sp500_price / baseSpy) - 1) * 100 : null);
  const cashPct = filtered.map(p => p.cash != null ? (p.cash / p.total_value) * 100 : null);

  // Rolling drawdown from peak
  let peak = 0;
  const ddPct = portPct.map(v => {
    if (v > peak) peak = v;
    return peak > 0 ? v - peak : (v < 0 ? v : 0);
  });

  // Trade markers — find trades within the filtered time range
  const t0 = filtered[0].timestamp;
  const t1 = filtered[filtered.length-1].timestamp;
  const visibleTrades = (trades||[]).filter(t => t.timestamp >= t0 && t.timestamp <= t1);

  // Map each trade to nearest label index for scatter overlay
  const buyX = [], sellX = [], buyY = [], sellY = [];
  visibleTrades.forEach(tr => {
    const ts = tr.timestamp.replace("T"," ").slice(0,16);
    let idx = labels.indexOf(ts);
    if (idx < 0) {
      // Find closest label
      idx = labels.reduce((best, lbl, i) => Math.abs(lbl.localeCompare(ts)) < Math.abs(labels[best].localeCompare(ts)) ? i : best, 0);
    }
    const isBuy = tr.action && tr.action.startsWith("BUY");
    if (isBuy) { buyX.push(idx); buyY.push(portPct[idx] ?? 0); }
    else { sellX.push(idx); sellY.push(portPct[idx] ?? 0); }
  });

  // Summary stats
  const finalPct = portPct[portPct.length-1];
  const spyFinalPct = spyPct[spyPct.length-1];
  const maxDd = Math.min(...ddPct);
  const deployed = 100 - (cashPct[cashPct.length-1] || 0);
  const vsSpyEl = document.getElementById("vs-spy-live");
  if (vsSpyEl && spyFinalPct != null) {
    const vs = finalPct - spyFinalPct;
    vsSpyEl.textContent = (vs>=0?"+":"")+vs.toFixed(2)+"% vs SPY";
    vsSpyEl.className = "v " + (vs>=0?"pos":"neg");
  }
  const ddEl = document.getElementById("live-maxdd");
  if (ddEl) { ddEl.textContent = maxDd.toFixed(2)+"%"; ddEl.className = "v " + (maxDd < -5 ? "neg" : ""); }
  const depEl = document.getElementById("live-deployed");
  if (depEl) depEl.textContent = deployed.toFixed(1)+"%";

  const mkDataset = (xs, ys, color, label, offset) => ({
    type: "scatter",
    label,
    data: xs.map((x,i) => ({ x, y: ys[i] + offset })),
    backgroundColor: color,
    borderColor: color,
    pointRadius: 7,
    pointStyle: label.includes("Buy") ? "triangle" : "triangle",
    rotation: label.includes("Sell") ? 180 : 0,
    showLine: false,
    order: 0,
  });

  const datasets = [
    {
      label: "Portfolio %",
      data: portPct,
      borderColor: "#0acdff",
      backgroundColor: "rgba(10,205,255,0.07)",
      fill: true, tension: 0.15, borderWidth: 2,
      pointRadius: 0, pointHoverRadius: 4, order: 2,
    },
    {
      label: "SPY %",
      data: spyPct,
      borderColor: "rgba(255,183,77,0.7)",
      backgroundColor: "transparent",
      borderDash: [5,4], borderWidth: 1.5,
      pointRadius: 0, fill: false, order: 3,
    },
  ];
  if (buyX.length)  datasets.push(mkDataset(buyX,  buyY,  "#00c896", "Buy ↑", 0.5));
  if (sellX.length) datasets.push(mkDataset(sellX, sellY, "#ff4455", "Sell ↓", -0.5));

  if (!chart) {
    chart = new Chart(document.getElementById("eq"), {
      type: "line",
      data: { labels, datasets },
      options: {
        animation: false,
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(15,20,28,0.95)", borderColor: "#2a3a4f", borderWidth: 1,
            titleColor: "#dde1e7", bodyColor: "#dde1e7", padding: 8, boxPadding: 3,
            callbacks: {
              label: ctx => {
                if (ctx.dataset.type === "scatter") return null;
                const v = ctx.parsed.y;
                return `${ctx.dataset.label}: ${v>=0?"+":""}${v.toFixed(2)}%`;
              },
            },
          },
        },
        scales: {
          x: { ticks: { color: "#8b929d", maxTicksLimit: 8 }, grid: { color: "#1f2126" }},
          y: {
            ticks: { color: "#dde1e7", callback: v => (v>=0?"+":"")+v.toFixed(1)+"%" },
            grid: { color: "#1f2126" },
          },
        },
      },
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets = datasets;
    chart.update("none");
  }

  // Drawdown sub-chart
  if (!ddChart) {
    ddChart = new Chart(document.getElementById("eq-dd"), {
      type: "line",
      data: { labels, datasets: [{
        label: "Drawdown %",
        data: ddPct,
        borderColor: "rgba(239,83,80,0.7)",
        backgroundColor: "rgba(239,83,80,0.12)",
        fill: true, tension: 0.15, borderWidth: 1.5,
        pointRadius: 0,
      }]},
      options: {
        animation: false,
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: { display: false },
          y: {
            ticks: { color: "#8b929d", font: { size: 9 }, callback: v => v.toFixed(1)+"%" },
            grid: { color: "#1f2126" }, max: 0,
          },
        },
      },
    });
  } else {
    ddChart.data.labels = labels;
    ddChart.data.datasets[0].data = ddPct;
    ddChart.update("none");
  }
}

async function refresh() {
  const r = await fetch(API_PREFIX + "/api/state").then(r => r.json());
  document.getElementById("hb").textContent = "updated " + (r.now || "");
  document.getElementById("tv").textContent = dollar(r.portfolio.total_value);
  document.getElementById("cash").textContent = dollar(r.portfolio.cash);
  const startVal = (r.equity && r.equity[0]) ? r.equity[0].total_value : 1000;
  const pl = r.portfolio.total_value - startVal;
  const plPct = (r.portfolio.total_value / startVal - 1) * 100;
  const plEl = document.getElementById("pl");
  plEl.textContent = (plPct >= 0 ? "+" : "") + plPct.toFixed(2) + "%";
  plEl.className = "v " + (plPct >= 0 ? "pos" : "neg");
  const _spEl = document.getElementById("sp"); if (_spEl) _spEl.textContent = r.sp500 ? fmt(r.sp500) : "—";
  _lastEquity = r.equity || [];
  _lastTrades = r.all_trades || r.trades || [];
  drawEquityChart(_lastEquity, _lastTrades);

  const posBody = document.querySelector("#pos-tbl tbody");
  const portTotal = r.portfolio.total_value || 0;
  posBody.innerHTML = r.positions.map(p => {
    const cls = (p.unrealized_pl || 0) >= 0 ? "pos" : "neg";
    const label = p.type === "stock" ? p.type :
                  `${p.type.toUpperCase()} ${p.strike}/${p.expiry}`;
    const mult = (p.type === "call" || p.type === "put") ? 100 : 1;
    const totalVal = (p.current_price || 0) * (p.qty || 0) * mult;
    const pctPort = portTotal > 0 ? (totalVal / portTotal * 100) : 0;
    return `<tr><td>${p.ticker}</td><td>${label}</td>
      <td class="num">${fmt(p.qty,4)}</td>
      <td class="num">${fmt(p.avg_cost)}</td>
      <td class="num">${fmt(p.current_price)}</td>
      <td class="num">${dollar(totalVal)}</td>
      <td class="num">${fmt(pctPort,1)}%</td>
      <td class="num ${cls}">${fmt(p.unrealized_pl)}</td></tr>`;
  }).join("") || `<tr><td colspan="8" class="muted">no positions</td></tr>`;

  const trBody = document.querySelector("#trades-tbl tbody");
  trBody.innerHTML = r.trades.map(t => {
    const cls = t.action.startsWith("SELL") ? "sell" : "buy";
    return `<tr><td>${dt(t.timestamp)}</td>
      <td><span class="pill ${cls}">${t.action}</span></td>
      <td>${t.ticker}</td>
      <td class="num">${fmt(t.qty,4)}</td>
      <td class="num">${fmt(t.price)}</td>
      <td class="muted">${(t.reason||"").slice(0,80)}</td></tr>`;
  }).join("") || `<tr><td colspan="6" class="muted">no trades</td></tr>`;

  const dBody = document.querySelector("#dec-tbl tbody");
  dBody.innerHTML = r.decisions.map(d => {
    let reason = "";
    try {
      const j = JSON.parse(d.reasoning || "{}");
      reason = (j.decision && j.decision.reasoning) || j.detail || "";
    } catch (_) { reason = d.reasoning || ""; }
    return `<tr><td>${dt(d.timestamp)}</td>
      <td>${d.market_open ? "yes" : "no"}</td>
      <td class="num">${d.signal_count}</td>
      <td>${(d.action_taken||"").slice(0,40)}</td>
      <td class="num">${fmt(d.portfolio_value)}</td>
      <td class="muted">${reason.slice(0,140)}</td></tr>`;
  }).join("") || `<tr><td colspan="6" class="muted">no decisions yet</td></tr>`;

}

// ───────── Backtests pane ─────────
let btLoaded = false;
let btChart;
let btScatter;
let btRuns = [];
let btPollTimer = null;
let btSelectedRunId = null;
let btHiddenRuns = new Set();
let btLastUpdated = null;
let btSortKey = "run_id", btSortDir = -1;
let btDetailSubtab = "trades";
let btSpyBaseline = null;
let btCurvesCache = {};              // run_id → normalized curve array
let btWinMinYears = 0, btWinMaxYears = 99; // window-length filter

function btRunColor(runId, idx) { return RUN_COLORS[idx % RUN_COLORS.length]; }
function hexToRgba(hex, a) {
  const h = hex.replace("#","");
  const r = parseInt(h.slice(0,2),16), g = parseInt(h.slice(2,4),16), b = parseInt(h.slice(4,6),16);
  return `rgba(${r},${g},${b},${a})`;
}

// Window-length filter chip handler
function setBtWinFilter(el) {
  document.querySelectorAll("#bt-win-filter .bt-filter-chip").forEach(b => b.classList.remove("active"));
  el.classList.add("active");
  btWinMinYears = parseFloat(el.dataset.min);
  btWinMaxYears = parseFloat(el.dataset.max);
  // Scatter + heatmap always show all runs (for cross-dimension visibility),
  // only the equity curve chart and table respect the filter.
  renderLegend();
  renderTable();
  redrawChart();
}

// Returns runs passing the current window-length filter
function filteredRuns() {
  return btRuns.filter(r => {
    if (!r.duration_days) return btWinMinYears === 0;
    const yrs = r.duration_days / 365.25;
    return yrs >= btWinMinYears && yrs < btWinMaxYears;
  });
}

// Lazily fetch curves for an array of run_ids not yet in cache.
// After fetching, calls callback().
async function ensureCurves(runIds, callback) {
  const missing = runIds.filter(id => !btCurvesCache[id]);
  if (missing.length) {
    try {
      const data = await fetch(API_PREFIX + "/api/backtests/curves?run_ids=" + missing.join(",")).then(r => r.json());
      Object.keys(data).forEach(k => { btCurvesCache[parseInt(k)] = data[k]; });
    } catch(e) { console.error("curves fetch:", e); }
  }
  if (callback) callback();
}

let mpChart;
async function loadModelProgress() {
  try {
    const d = await fetch(API_PREFIX + "/api/model-progress").then(r => r.json());
    const cycles = d.cycles || [];
    if (!cycles.length) return;
    // cycle label is now a run_id range string e.g. "#1491-#1495"
    const labels = cycles.map(c => c.cycle);
    const best  = cycles.map(c => c.best);
    const avg   = cycles.map(c => c.avg);
    const worst = cycles.map(c => c.worst);
    const totalRuns = d.total_runs || cycles.length * 5;
    const ctx = document.getElementById("mp-chart");
    if (!ctx) return;
    // Update subtitle with total run count
    const sub = ctx.closest(".card")?.querySelector("div.sub,div[style*='78909c']");
    if (sub) sub.textContent = `Best / avg / worst return per cycle of 5 runs (${totalRuns} total). Upward trend = model improving.`;
    if (mpChart) mpChart.destroy();
    mpChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Best %",  data: best,  borderColor: "#00c896", backgroundColor: "rgba(76,175,80,0.08)",  tension: 0.3, pointRadius: 3, fill: false },
          { label: "Avg %",   data: avg,   borderColor: "#0acdff", backgroundColor: "rgba(66,165,245,0.08)", tension: 0.3, pointRadius: 3, fill: false },
          { label: "Worst %", data: worst, borderColor: "#ff4455", backgroundColor: "rgba(239,83,80,0.08)",  tension: 0.3, pointRadius: 3, fill: false },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#dde1e7", font: { size: 11 } } },
          tooltip: { callbacks: { label: c => c.dataset.label + ": " + c.raw.toFixed(1) + "%" } }
        },
        scales: {
          x: {
            ticks: { color: "#8b929d", maxTicksLimit: 20, maxRotation: 45 },
            grid: { color: "rgba(255,255,255,0.05)" }
          },
          y: { ticks: { color: "#8b929d", callback: v => v.toFixed(0) + "%" }, grid: { color: "rgba(255,255,255,0.05)" } }
        }
      }
    });
  } catch(e) { console.error("model-progress:", e); }
}

async function loadBacktests() {
  try {
    const r = await fetch(API_PREFIX + "/api/backtests").then(r => r.json());
    btRuns = r.runs || [];
    btSpyBaseline = r.spy_baseline != null ? r.spy_baseline : null;
    btLastUpdated = Date.now();
    btLoaded = true;
    renderBacktests();
    loadModelProgress();
  } catch (e) {
    console.error(e);
  } finally {
    if (btPollTimer) clearTimeout(btPollTimer);
    // 60s poll — list is now metadata-only (<50KB) so fast, but no need to hammer
    btPollTimer = setTimeout(loadBacktests, 60_000);
  }
}

function renderBacktests() {
  const total = btRuns.length;
  const running = btRuns.filter(x => x.status === "running");
  const failed = btRuns.filter(x => x.status === "failed");
  const completed = btRuns.filter(x => x.status === "complete");
  const pctDone = total ? (completed.length / total) * 100 : 0;
  document.getElementById("bt-progress-bar").style.width = pctDone + "%";
  let lbl = `${completed.length}/${total || 10} runs complete`;
  if (running.length)  lbl += ` · ${running.length} running`;
  if (failed.length)   lbl += ` · ${failed.length} failed`;
  document.getElementById("bt-progress-label").textContent = lbl;
  document.getElementById("bt-live-indicator").innerHTML =
    running.length ? `<span class="live-dot"></span>live` : `<span style="color:#00c896;">●</span> idle`;

  // Stats over the filtered set of completed runs
  const vis = filteredRuns().filter(r => r.status === "complete");
  document.getElementById("bt-filtered-count").textContent = vis.length ? `${vis.length}` : "—";
  if (vis.length) {
    const avg = vis.reduce((a,b) => a + (b.total_return_pct||0), 0) / vis.length;
    const annVals = vis.map(r => r.annualized_return_pct).filter(v => v != null);
    const avgAnn = annVals.length ? annVals.reduce((a,b)=>a+b,0)/annVals.length : null;
    const best = vis.reduce((a,b) => (b.annualized_return_pct||0) > (a.annualized_return_pct||0) ? b : a);
    const worst = vis.reduce((a,b) => (b.annualized_return_pct||0) < (a.annualized_return_pct||0) ? b : a);
    const beat = vis.filter(x => x.vs_spy_pct != null && x.vs_spy_pct > 0).length;
    const wins = vis.filter(x => (x.total_return_pct||0) > 0).length;

    const avgEl = document.getElementById("bt-avg");
    avgEl.textContent = (avg >= 0 ? "+" : "") + fmt(avg) + "%";
    avgEl.className = "v " + (avg >= 0 ? "pos" : "neg");

    const avgAnnEl = document.getElementById("bt-avg-ann");
    if (avgAnn != null) {
      avgAnnEl.textContent = (avgAnn >= 0 ? "+" : "") + fmt(avgAnn) + "%/yr";
      avgAnnEl.className = "v " + (avgAnn >= 0 ? "pos" : "neg");
    } else { avgAnnEl.textContent = "—"; avgAnnEl.className = "v"; }

    const bestAnn = best.annualized_return_pct;
    document.getElementById("bt-best").innerHTML =
      `<span class="${(bestAnn||0)>=0?'pos':'neg'}">${bestAnn!=null?(bestAnn>=0?"+":"")+fmt(bestAnn)+"%/yr":"—"}</span> <span class="muted" style="font-size:11px;">#${best.run_id}</span>`;
    const worstAnn = worst.annualized_return_pct;
    document.getElementById("bt-worst").innerHTML =
      `<span class="${(worstAnn||0)>=0?'pos':'neg'}">${worstAnn!=null?(worstAnn>=0?"+":"")+fmt(worstAnn)+"%/yr":"—"}</span> <span class="muted" style="font-size:11px;">#${worst.run_id}</span>`;
    document.getElementById("bt-beat").textContent = `${beat} / ${vis.length}`;
    const winrateEl = document.getElementById("bt-winrate");
    winrateEl.textContent = `${wins} / ${vis.length}`;
    winrateEl.className = "v " + (wins >= vis.length/2 ? "pos" : "neg");
  } else {
    ["bt-avg","bt-avg-ann","bt-best","bt-worst","bt-beat","bt-winrate"].forEach(id =>
      (document.getElementById(id).textContent = "—", document.getElementById(id).className = "v"));
  }

  renderLegend();
  renderTable();
  redrawChart();
  drawScatterChart();
  renderEraHeatmap();
  tickLastUpdated();
}

function renderLegend() {
  const wrap = document.getElementById("bt-legend");
  const vis = filteredRuns();
  const limitEl = document.getElementById("bt-chart-limit");
  const limit = limitEl ? parseInt(limitEl.value, 10) : 20;
  // Show the most-recent `limit` runs (sorted by run_id desc)
  const chartRuns = [...vis].sort((a,b) => b.run_id - a.run_id).slice(0, limit);
  wrap.innerHTML = chartRuns.map((r, i) => {
    const color = btRunColor(r.run_id, i);
    const hidden = btHiddenRuns.has(r.run_id);
    const selected = btSelectedRunId === r.run_id;
    const ann = r.annualized_return_pct;
    const retCls = (ann || 0) >= 0 ? "pos" : "neg";
    const retTxt = ann != null ? ((ann >= 0 ? "+" : "") + fmt(ann) + "%/yr") : "—";
    const durYrs = r.duration_days ? (r.duration_days / 365.25).toFixed(1) + "yr" : "";
    return `<div class="bt-legend-row${hidden ? ' hidden-run' : ''}${selected ? ' selected' : ''}" onclick="selectRun(${r.run_id})">
      <input type="checkbox" ${hidden ? '' : 'checked'} onclick="event.stopPropagation();toggleRun(${r.run_id})">
      <span class="bt-swatch" style="background:${color};"></span>
      <span class="name">#${r.run_id} <span class="muted" style="font-size:10px;">${durYrs}</span>${r.status === 'running' ? ' <span class="spinner" style="width:8px;height:8px;border-width:1px;margin:0 0 0 4px;"></span>' : ''}</span>
      <span class="ret ${retCls}">${retTxt}</span>
    </div>`;
  }).join("") || `<div class="muted" style="font-size:12px;">no runs match filter</div>`;
}

function renderTable() {
  const tbody = document.querySelector("#bt-tbl tbody");
  document.querySelectorAll("#bt-tbl thead th").forEach(th => {
    th.classList.add("sortable-h");
    th.classList.remove("sort-asc","sort-desc");
    if (th.dataset.k === btSortKey) th.classList.add(btSortDir > 0 ? "sort-asc" : "sort-desc");
    th.onclick = () => {
      const k = th.dataset.k;
      if (btSortKey === k) btSortDir = -btSortDir; else { btSortKey = k; btSortDir = -1; }
      renderTable();
    };
  });
  const vis = filteredRuns();
  const sorted = [...vis].sort((a,b) => {
    const va = a[btSortKey], vb = b[btSortKey];
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "number") return (va - vb) * btSortDir;
    return String(va).localeCompare(String(vb)) * btSortDir;
  });
  tbody.innerHTML = sorted.map(r => {
    const isRunning = r.status === "running";
    const isComplete = r.status === "complete";
    const retCls = (r.total_return_pct || 0) >= 0 ? "pos" : "neg";
    const annCls = (r.annualized_return_pct || 0) >= 0 ? "pos" : "neg";
    const vsCls  = (r.vs_spy_pct || 0) >= 0 ? "pos" : "neg";
    const selected = btSelectedRunId === r.run_id;
    const retCell = r.total_return_pct == null
      ? `<span class="muted">—</span>`
      : `<span class="${retCls}">${(r.total_return_pct >= 0 ? "+" : "") + fmt(r.total_return_pct)}%</span>`;
    const annCell = r.annualized_return_pct == null
      ? `<span class="muted">—</span>`
      : `<span class="${annCls}">${(r.annualized_return_pct >= 0 ? "+" : "") + fmt(r.annualized_return_pct)}%</span>`;
    const vsCell = isComplete && r.vs_spy_pct != null
      ? `<span class="${vsCls}">${(r.vs_spy_pct >= 0 ? "+" : "") + fmt(r.vs_spy_pct)}%</span>`
      : `<span class="muted">—</span>`;
    const win = formatWindow(r.start_date, r.end_date);
    const era = classifyEra(r.start_date, r.end_date);
    const eraPill = era ? `<span class="pill" style="background:${era.bg};color:${era.fg};font-size:10px;">${era.tag}</span>` : "";
    const winCell = win
      ? `<div style="font-size:11px;line-height:1.5;">${r.start_date} → ${r.end_date}<br>${eraPill}</div>`
      : `<span class="muted">—</span>`;
    const durCell = r.duration_days
      ? `<span style="font-size:11px;">${(r.duration_days/365.25).toFixed(1)}yr</span>`
      : "—";
    return `<tr class="bt-row${selected ? ' selected' : ''}" onclick="selectRun(${r.run_id})">
      <td><span class="pill" style="font-size:11px;">#${r.run_id}${isRunning?'<span class="spinner" style="width:7px;height:7px;border-width:1px;margin-left:3px;"></span>':''}</span></td>
      <td><span class="pill status-${r.status || 'pending'}">${r.status || 'pending'}</span></td>
      <td class="num">${retCell}</td>
      <td class="num">${annCell}</td>
      <td class="num">${vsCell}</td>
      <td>${winCell}</td>
      <td class="num">${durCell}</td>
      <td class="num">${r.n_trades || 0}</td>
      <td class="num">${r.n_decisions || 0}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="9" class="muted">no runs match the current filter</td></tr>`;
}

// ───────── Backtest era classification (frontend) ─────────
// Classifies a date range into a market era for at-a-glance context.
function classifyEra(startStr, endStr) {
  if (!startStr) return null;
  const s = startStr, e = endStr || startStr;
  const sy = parseInt(s.slice(0,4),10), ey = parseInt(e.slice(0,4),10);
  // Pre-2008
  if (ey < 2008) return { tag: "Pre-GFC", bg: "rgba(96,96,128,0.18)", fg: "#9a9ec0" };
  // GFC overlap
  if (sy <= 2009 && ey >= 2008) return { tag: "GFC", bg: "rgba(239,83,80,0.22)", fg: "#ff6b6b" };
  // COVID Q1 2020 inclusion (treat any range touching Jan–Apr 2020 as COVID crash)
  if (s <= "2020-04-30" && e >= "2020-01-20") return { tag: "COVID crash", bg: "rgba(239,83,80,0.28)", fg: "#ff7676" };
  // 2020–2021 recovery (only if start ≥ 2020-04 and end ≤ 2021-12)
  if (s >= "2020-04-01" && e <= "2021-12-31") return { tag: "Recovery", bg: "rgba(0,200,150,0.18)", fg: "#00c896" };
  // 2022 rate hike bear
  if (sy === 2022 && ey === 2022) return { tag: "Rate hike bear", bg: "rgba(255,140,0,0.22)", fg: "#ffb74d" };
  // 2023–2024 AI bull
  if (sy >= 2023 && ey <= 2024) return { tag: "AI bull", bg: "rgba(10,205,255,0.18)", fg: "#0acdff" };
  // 2025+
  if (sy >= 2025) return { tag: "Recent", bg: "rgba(127,255,0,0.18)", fg: "#7fff00" };
  // 2010–2019 fallback bull market
  if (sy >= 2010 && ey <= 2019) return { tag: "Bull market", bg: "rgba(0,200,150,0.15)", fg: "#5dd9b3" };
  // Spans multiple eras
  return { tag: "Multi-era", bg: "rgba(155,155,155,0.18)", fg: "#cfd2da" };
}

function formatWindow(startStr, endStr) {
  if (!startStr) return null;
  const e = endStr || "…";
  // Years span for the "(Nyr)" suffix
  let yrs = "";
  try {
    const sd = new Date(startStr);
    const ed = endStr ? new Date(endStr) : new Date();
    const days = (ed - sd) / 86400000;
    if (isFinite(days) && days > 0) {
      const y = days / 365.25;
      yrs = y >= 1 ? ` (${y.toFixed(y >= 5 ? 0 : 1)}yr)` : ` (${Math.round(days/30)}mo)`;
    }
  } catch (_) {}
  return `${startStr} → ${e}${yrs}`;
}

function dataSourcesForWindow(startStr, endStr) {
  if (!startStr) return [];
  const s = startStr, e = endStr || startStr;
  // GDELT public coverage starts 2015-02-19; SEC EDGAR full-text goes back to 1994.
  const gdeltOk = e >= "2015-02-19";
  return [
    { label: "GDELT news", ok: gdeltOk, hint: gdeltOk ? "coverage since 2015-02-19" : "pre-2015 — not available" },
    { label: "SEC EDGAR filings", ok: true, hint: "back to 1994" },
    { label: "Price / quant signals", ok: true, hint: "yfinance OHLCV" },
    { label: "Historical articles labeled by Claude", ok: true, hint: "Opus winner annotations + backtest injections" },
  ];
}

// Normalized chart: X = day-index from run start (so all window lengths compare),
// Y = % gain from start (so 1yr and 10yr runs are on the same scale).
async function drawBacktestChart() {
  const limitEl = document.getElementById("bt-chart-limit");
  const limit = limitEl ? parseInt(limitEl.value, 10) : 20;

  const vis = filteredRuns();
  // Most recent `limit` runs by run_id
  const chartRuns = [...vis].sort((a,b) => b.run_id - a.run_id).slice(0, limit);

  // Fetch any missing curves first, then render
  const needIds = chartRuns.map(r => r.run_id);
  await ensureCurves(needIds, null);

  // Build day-index label set (union across all visible runs)
  const daySet = new Set([0]);
  chartRuns.forEach(r => {
    const curve = btCurvesCache[r.run_id] || [];
    curve.forEach(p => { if (p.day_index != null) daySet.add(p.day_index); });
  });
  const labels = Array.from(daySet).sort((a,b) => a-b);

  const hasSelection = btSelectedRunId != null;

  const datasets = chartRuns.map((r, i) => {
    const curve = btCurvesCache[r.run_id] || [];
    const lookup = {};
    curve.forEach(p => { if (p.day_index != null) lookup[p.day_index] = p.value_pct; });
    let last = 0;
    const data = labels.map(d => {
      if (lookup[d] != null) { last = lookup[d]; return lookup[d]; }
      // forward-fill, but only within the run's duration
      const maxDay = r.duration_days || 9999;
      return d <= maxDay ? last : null;
    });
    const isRunning = r.status === "running";
    const color = btRunColor(r.run_id, i);
    const isHidden = btHiddenRuns.has(r.run_id);
    const isSelected = btSelectedRunId === r.run_id;
    const dim = hasSelection && !isSelected;
    const durYrs = r.duration_days ? (r.duration_days/365.25).toFixed(1)+"yr" : "";
    const ann = r.annualized_return_pct;
    const annTxt = ann != null ? ` (${(ann>=0?"+":"")+ann.toFixed(1)}%/yr ann.)` : "";
    return {
      label: `#${r.run_id} ${durYrs}${annTxt}`,
      data,
      runId: r.run_id,
      kind: "run",
      borderColor: dim ? hexToRgba(color, 0.18) : color,
      backgroundColor: hexToRgba(color, 0.04),
      borderWidth: isSelected ? 3.5 : (dim ? 0.8 : 1.5),
      borderDash: isRunning ? [5, 4] : [],
      pointRadius: 0, pointHoverRadius: 5,
      tension: 0.15, fill: false,
      hidden: isHidden,
      spanGaps: false,
    };
  });

  // SPY benchmark: average annualized SPY return for S&P ~10.7%/yr.
  // We draw an "average SPY" reference line that grows at ~10.7%/yr.
  // If a specific per-window SPY is available from btSpyBaseline we use it scaled to max duration.
  const maxDur = chartRuns.reduce((m, r) => Math.max(m, r.duration_days||0), 0);
  if (maxDur > 0) {
    const spyAnnPct = 10.7; // long-run S&P average annualized %
    const spyData = labels.map(d => {
      const yrs = d / 365.25;
      return ((1 + spyAnnPct/100)**yrs - 1) * 100;
    });
    datasets.push({
      label: `SPY avg (~${spyAnnPct}%/yr)`,
      data: spyData,
      kind: "benchmark",
      borderColor: hasSelection ? "rgba(180,180,180,0.2)" : "rgba(180,180,180,0.7)",
      borderWidth: 2,
      borderDash: [6, 3],
      pointRadius: 0,
      tension: 0,
      fill: false,
      order: -1,
    });
  }

  if (btChart) { btChart.destroy(); btChart = null; }
  const canvas = document.getElementById("bt-chart");
  if (!canvas) return;
  btChart = new Chart(canvas, {
    type: "line",
    data: { labels, datasets },
    options: {
      animation: false,
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      onClick: (evt, els, chart) => {
        if (els && els.length) {
          for (const el of els) {
            const ds = chart.data.datasets[el.datasetIndex];
            if (ds && ds.kind === "run") { selectRun(ds.runId); return; }
          }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: "index", intersect: false,
          backgroundColor: "rgba(15,20,28,0.95)",
          borderColor: "#2a3a4f", borderWidth: 1,
          titleColor: "#dde1e7", bodyColor: "#dde1e7",
          padding: 10, boxPadding: 4,
          itemSort: (a,b) => b.parsed.y - a.parsed.y,
          filter: (item) => item.parsed.y != null,
          callbacks: {
            title: (items) => {
              const d = items[0]?.parsed?.x;
              if (d == null) return "";
              const yrs = (d/365.25).toFixed(1);
              return `Day ${d} (year ${yrs})`;
            },
            label: (ctx) => {
              const v = ctx.parsed.y;
              if (v == null) return null;
              return `${ctx.dataset.label}: ${v>=0?"+":""}${v.toFixed(1)}%`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Days from start", color: "#50565f", font: { size: 10 } },
          ticks: {
            color: "#8b929d", maxTicksLimit: 10,
            callback: v => v >= 365 ? (v/365).toFixed(1)+"yr" : "d"+v,
          },
          grid: { color: "#1f2126" },
        },
        y: {
          title: { display: true, text: "Return from start (%)", color: "#50565f", font: { size: 10 } },
          ticks: {
            color: "#dde1e7",
            callback: v => (v>=0?"+":"") + v.toFixed(0) + "%",
          },
          grid: { color: "#1f2126" },
        },
      },
    },
  });
}

// ───────── Chart mode toggle ─────────
let btChartMode = "aggregate";
let btDdChart = null;

function setChartMode(mode) {
  btChartMode = mode;
  document.getElementById("mode-agg").classList.toggle("active", mode === "aggregate");
  document.getElementById("mode-ind").classList.toggle("active", mode === "individual");
  document.getElementById("agg-legend").style.display = mode === "aggregate" ? "" : "none";
  document.getElementById("ind-controls").style.display = mode === "individual" ? "" : "none";
  document.getElementById("bt-drawdown-wrap").style.display = mode === "aggregate" ? "" : "none";
  redrawChart();
}

function redrawChart() {
  if (btChartMode === "aggregate") {
    drawAggregateChart();
  } else {
    drawBacktestChart();
    if (btDdChart) { btDdChart.destroy(); btDdChart = null; }
  }
}

// ───────── Aggregate chart: percentile bands ─────────
// X = day-offset, Y = % return. Shows median/P25-P75/P5-P95 across all completed
// runs in the current window filter. SPY overlay uses actual per-run spy_return_pct.
async function drawAggregateChart() {
  const vis = filteredRuns().filter(r => r.status === "complete" && r.duration_days);
  const sampleRuns = vis;

  // Fetch curves we don't have yet
  await ensureCurves(sampleRuns.map(r => r.run_id), null);

  const nLabel = document.getElementById("agg-n-label");
  if (nLabel) nLabel.textContent = `(${sampleRuns.length} runs)`;

  // Build day → [value_pct] map; every run contributes 0% at day 0
  const byDay = { 0: sampleRuns.map(() => 0) };
  sampleRuns.forEach(r => {
    const curve = btCurvesCache[r.run_id] || [];
    curve.forEach(p => {
      if (p.day_index == null || p.day_index === 0) return;
      if (!byDay[p.day_index]) byDay[p.day_index] = [];
      byDay[p.day_index].push(p.value_pct);
    });
  });

  // Build day → [drawdown_pct] map (from curve peaks)
  const ddByDay = {};
  sampleRuns.forEach(r => {
    const curve = btCurvesCache[r.run_id] || [];
    let peak = 0;
    curve.forEach(p => {
      if (p.day_index == null) return;
      if (p.value_pct > peak) peak = p.value_pct;
      const dd = peak > 0 ? (p.value_pct - peak) : (p.value_pct < 0 ? p.value_pct : 0);
      if (!ddByDay[p.day_index]) ddByDay[p.day_index] = [];
      ddByDay[p.day_index].push(dd);
    });
  });

  const pct = (arr, p) => {
    const s = [...arr].sort((a,b) => a-b);
    const idx = (p/100) * (s.length - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return s[lo] + (s[hi] - s[lo]) * (idx - lo);
  };

  const MIN_N = 5;
  const days = Object.keys(byDay).map(Number).sort((a,b) => a-b);
  const labels = days.filter(d => byDay[d].length >= MIN_N);

  const P5=[], P25=[], P50=[], P75=[], P95=[], DD_P50=[], DD_P75=[];
  labels.forEach(d => {
    const v = byDay[d];
    P5.push({ x: d, y: pct(v, 5) });
    P25.push({ x: d, y: pct(v, 25) });
    P50.push({ x: d, y: pct(v, 50) });
    P75.push({ x: d, y: pct(v, 75) });
    P95.push({ x: d, y: pct(v, 95) });
    const dd = ddByDay[d] || [0];
    DD_P50.push({ x: d, y: pct(dd, 50) });
    DD_P75.push({ x: d, y: pct(dd, 75) });
  });

  // Median actual SPY growth curve using per-run spy_return_pct annualized
  const spyAnns = sampleRuns
    .filter(r => r.spy_return_pct != null && r.duration_days > 30)
    .map(r => Math.pow(1 + r.spy_return_pct / 100, 365.25 / r.duration_days) - 1);
  spyAnns.sort((a,b) => a-b);
  const medSpyAnn = spyAnns.length ? spyAnns[Math.floor(spyAnns.length/2)] : 0.107;
  const spyLine = labels.map(d => ({ x: d, y: (Math.pow(1 + medSpyAnn, d/365.25) - 1) * 100 }));
  const spyPctLabel = (medSpyAnn * 100).toFixed(1);

  // Zero line
  const zeroLine = labels.map(d => ({ x: d, y: 0 }));

  const chartOpts = (yLabel, yFmt, minY) => ({
    animation: false,
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: "index", intersect: false,
        backgroundColor: "rgba(15,20,28,0.95)", borderColor: "#2a3a4f", borderWidth: 1,
        titleColor: "#dde1e7", bodyColor: "#dde1e7", padding: 8, boxPadding: 3,
        filter: item => item.dataset.showInTooltip !== false,
        callbacks: {
          title: items => {
            const d = items[0]?.parsed?.x;
            return d != null ? `Day ${d} (${(d/365.25).toFixed(1)} yr from start)` : "";
          },
          label: ctx => {
            const v = ctx.parsed.y;
            if (v == null) return null;
            return `${ctx.dataset.label}: ${yFmt(v)}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        ticks: { color: "#8b929d", maxTicksLimit: 10, callback: v => v>=365?(v/365).toFixed(1)+"yr":"d"+v },
        grid: { color: "#1f2126" },
      },
      y: {
        title: { display: true, text: yLabel, color: "#50565f", font: { size: 10 } },
        min: minY,
        ticks: { color: "#dde1e7", callback: yFmt },
        grid: { color: "#1f2126" },
      },
    },
  });

  // ── Main equity chart ──
  if (btChart) { btChart.destroy(); btChart = null; }
  const canvas = document.getElementById("bt-chart");
  if (canvas) {
    btChart = new Chart(canvas, {
      type: "scatter",
      data: {
        datasets: [
          // Outer band (P5→P95): fill from P5 up to P95
          { label: "P5",  data: P5,  borderColor:"transparent", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, showInTooltip:false },
          { label: "P95 outer", data: P95, borderColor:"rgba(10,205,255,0.08)", backgroundColor:"rgba(10,205,255,0.06)", pointRadius:0, showLine:true, fill:"-1", borderWidth:1, showInTooltip:false },
          // Inner band (P25→P75)
          { label: "P25", data: P25, borderColor:"transparent", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, showInTooltip:false },
          { label: "P75 inner", data: P75, borderColor:"rgba(10,205,255,0.22)", backgroundColor:"rgba(10,205,255,0.18)", pointRadius:0, showLine:true, fill:"-1", borderWidth:1, showInTooltip:false },
          // Median
          { label: "Median", data: P50, borderColor:"#0acdff", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, borderWidth:2.5, tension:0.15 },
          // SPY actual median
          { label: `SPY (${spyPctLabel}%/yr actual median)`, data: spyLine, borderColor:"rgba(200,200,200,0.65)", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, borderWidth:1.5, borderDash:[6,3], tension:0 },
          // Zero reference
          { label: "0%", data: zeroLine, borderColor:"rgba(255,255,255,0.08)", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, borderWidth:1, borderDash:[2,4], showInTooltip:false },
        ],
      },
      options: {
        ...chartOpts("Return from start (%)", v => (v>=0?"+":"")+v.toFixed(0)+"%", null),
        onClick: () => {},
      },
    });
  }

  // ── Drawdown sub-chart ──
  if (btDdChart) { btDdChart.destroy(); btDdChart = null; }
  const ddCanvas = document.getElementById("bt-dd-chart");
  if (ddCanvas && DD_P50.length) {
    btDdChart = new Chart(ddCanvas, {
      type: "scatter",
      data: {
        datasets: [
          { label: "DD P75", data: DD_P75, borderColor:"rgba(239,83,80,0.12)", backgroundColor:"rgba(239,83,80,0.10)", pointRadius:0, showLine:true, fill:"origin", borderWidth:1 },
          { label: "Median DD", data: DD_P50, borderColor:"rgba(239,83,80,0.7)", backgroundColor:"transparent", pointRadius:0, showLine:true, fill:false, borderWidth:1.5 },
        ],
      },
      options: chartOpts("Drawdown (%)", v => v.toFixed(0)+"%", null),
    });
  }
}

// ───────── Era definitions (shared for scatter + heatmap) ─────────
const ERA_DEFS = [
  { key: "Pre-GFC",       start: "1900-01-01", end: "2007-12-31", color: "#9a9ec0" },
  { key: "GFC",           start: "2008-01-01", end: "2009-12-31", color: "#ff6b6b" },
  { key: "Bull 2010s",    start: "2010-01-01", end: "2019-12-31", color: "#5dd9b3" },
  { key: "COVID crash",   start: "2020-01-01", end: "2020-06-30", color: "#ff7676" },
  { key: "Recovery",      start: "2020-07-01", end: "2021-12-31", color: "#00c896" },
  { key: "Rate-hike bear",start: "2022-01-01", end: "2022-12-31", color: "#ffb74d" },
  { key: "AI bull",       start: "2023-01-01", end: "2024-12-31", color: "#0acdff" },
  { key: "Recent",        start: "2025-01-01", end: "2099-12-31", color: "#7fff00" },
];

// Assign an era to a run based on the midpoint of its window
function runEra(r) {
  if (!r.start_date || !r.duration_days) return null;
  const startMs = new Date(r.start_date).getTime();
  const midMs = startMs + (r.duration_days / 2) * 86400000;
  const midStr = new Date(midMs).toISOString().slice(0, 10);
  for (const e of ERA_DEFS) {
    if (midStr >= e.start && midStr <= e.end) return e;
  }
  return { key: "Other", color: "#555" };
}

// ───────── Scatter: duration (X) vs annualized return (Y), colored by era ─────────
function drawScatterChart() {
  const completed = btRuns.filter(r => r.status === "complete" && r.duration_days && r.annualized_return_pct != null);
  if (!completed.length) return;

  // Group into era datasets
  const byEra = {};
  completed.forEach(r => {
    const era = runEra(r) || { key: "Other", color: "#555" };
    if (!byEra[era.key]) byEra[era.key] = { color: era.color, points: [] };
    byEra[era.key].points.push({ x: r.duration_days / 365.25, y: r.annualized_return_pct, runId: r.run_id });
  });

  // Era legend
  const legendEl = document.getElementById("bt-era-legend");
  if (legendEl) {
    legendEl.innerHTML = Object.entries(byEra).map(([key, v]) =>
      `<span style="display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:99px;background:${hexToRgba(v.color,0.15)};border:1px solid ${hexToRgba(v.color,0.4)};color:${v.color};font-size:10px;">
        <span style="width:6px;height:6px;border-radius:50%;background:${v.color};display:inline-block;"></span>${key} (${v.points.length})
      </span>`
    ).join("");
  }

  const datasets = Object.entries(byEra).map(([key, v]) => ({
    label: key,
    data: v.points,
    backgroundColor: hexToRgba(v.color, 0.7),
    borderColor: v.color,
    borderWidth: 1,
    pointRadius: 5,
    pointHoverRadius: 8,
  }));

  if (btScatter) { btScatter.destroy(); btScatter = null; }
  const canvas = document.getElementById("bt-scatter");
  if (!canvas) return;
  btScatter = new Chart(canvas, {
    type: "scatter",
    data: { datasets },
    options: {
      animation: false,
      responsive: true, maintainAspectRatio: false,
      onClick: (evt, els) => {
        if (els && els.length) {
          const pt = els[0];
          const ds = btScatter.data.datasets[pt.datasetIndex];
          const runId = ds?.data[pt.index]?.runId;
          if (runId != null) selectRun(runId);
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(15,20,28,0.95)", borderColor: "#2a3a4f", borderWidth: 1,
          titleColor: "#dde1e7", bodyColor: "#dde1e7", padding: 10,
          callbacks: {
            title: (items) => {
              const pt = items[0];
              return `Run #${pt.raw.runId}`;
            },
            label: (ctx) => {
              const pt = ctx.raw;
              const r = btRuns.find(r => r.run_id === pt.runId);
              const lines = [
                `Duration: ${pt.x.toFixed(1)}yr`,
                `Annualized: ${(pt.y>=0?"+":"")+pt.y.toFixed(1)}%/yr`,
              ];
              if (r) {
                lines.push(`Total: ${(r.total_return_pct>=0?"+":"")+r.total_return_pct.toFixed(1)}%`);
                lines.push(`${r.start_date} → ${r.end_date}`);
              }
              return lines;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Window length (years)", color: "#50565f", font: { size: 10 } },
          ticks: { color: "#8b929d", callback: v => v.toFixed(0)+"yr" },
          grid: { color: "#1f2126" },
        },
        y: {
          title: { display: true, text: "Annualized return (%/yr)", color: "#50565f", font: { size: 10 } },
          ticks: { color: "#dde1e7", callback: v => (v>=0?"+":"") + v.toFixed(0) + "%" },
          grid: { color: "#1f2126" },
        },
      },
    },
  });
}

// ───────── Era × Duration heatmap table ─────────
const DUR_BUCKETS = [
  { label: "1yr",   min: 0,   max: 1.5 },
  { label: "2yr",   min: 1.5, max: 2.5 },
  { label: "3yr",   min: 2.5, max: 3.5 },
  { label: "4–5yr", min: 3.5, max: 5.5 },
  { label: "6–10yr",min: 5.5, max: 99  },
];

function renderEraHeatmap() {
  const el = document.getElementById("bt-heatmap");
  if (!el) return;
  const completed = btRuns.filter(r => r.status === "complete" && r.annualized_return_pct != null && r.duration_days);

  // Build cell data: era × dur bucket → [annualized values]
  const cells = {};
  ERA_DEFS.forEach(e => { cells[e.key] = {}; DUR_BUCKETS.forEach(b => { cells[e.key][b.label] = []; }); });

  completed.forEach(r => {
    const era = runEra(r);
    if (!era) return;
    const yrs = r.duration_days / 365.25;
    const bkt = DUR_BUCKETS.find(b => yrs >= b.min && yrs < b.max);
    if (!bkt) return;
    if (!cells[era.key]) cells[era.key] = {};
    if (!cells[era.key][bkt.label]) cells[era.key][bkt.label] = [];
    cells[era.key][bkt.label].push(r.annualized_return_pct);
  });

  // Find global min/max for color scaling
  let gmin = Infinity, gmax = -Infinity;
  ERA_DEFS.forEach(e => DUR_BUCKETS.forEach(b => {
    const vals = cells[e.key]?.[b.label] || [];
    if (vals.length) {
      const avg = vals.reduce((a,v)=>a+v,0)/vals.length;
      if (avg < gmin) gmin = avg;
      if (avg > gmax) gmax = avg;
    }
  }));

  // Color: negative → red, 0 → neutral, positive → green
  function heatColor(avg, n) {
    if (!n) return "rgba(255,255,255,0.03)";
    const norm = avg / Math.max(Math.abs(gmin), Math.abs(gmax), 1);
    if (avg >= 0) return `rgba(0,200,150,${Math.min(0.8, norm * 0.7 + 0.1)})`;
    return `rgba(239,83,80,${Math.min(0.8, -norm * 0.7 + 0.1)})`;
  }

  // Filter out eras with zero data
  const activeEras = ERA_DEFS.filter(e => DUR_BUCKETS.some(b => (cells[e.key]?.[b.label]||[]).length > 0));

  let html = `<table style="border-collapse:collapse;width:100%;font-size:12px;min-width:500px;">
    <thead><tr>
      <th style="text-align:left;padding:6px 10px;color:var(--text-muted);font-weight:500;border-bottom:1px solid var(--border);">Era (midpoint)</th>`;
  DUR_BUCKETS.forEach(b => {
    html += `<th style="text-align:center;padding:6px 10px;color:var(--text-muted);font-weight:500;border-bottom:1px solid var(--border);">${b.label}</th>`;
  });
  html += `</tr></thead><tbody>`;

  activeEras.forEach(e => {
    html += `<tr><td style="padding:6px 10px;color:${e.color};font-weight:500;white-space:nowrap;">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${e.color};margin-right:5px;"></span>${e.key}
    </td>`;
    DUR_BUCKETS.forEach(b => {
      const vals = cells[e.key]?.[b.label] || [];
      const n = vals.length;
      const avg = n ? vals.reduce((a,v)=>a+v,0)/n : null;
      const bg = heatColor(avg, n);
      const txt = avg != null ? `${avg>=0?"+":""}${avg.toFixed(1)}%` : "";
      const sub = n ? `<div style="font-size:10px;opacity:0.6;">n=${n}</div>` : `<div style="color:var(--text-muted);font-size:11px;">—</div>`;
      html += `<td style="text-align:center;padding:5px 8px;background:${bg};border:1px solid rgba(255,255,255,0.04);">
        ${txt ? `<div style="font-weight:600;color:${avg>=0?"#b6f0d8":"#ffaaaa"};">${txt}</div>` : ""}
        ${sub}
      </td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  el.innerHTML = html;
}

function selectRun(runId) {
  btSelectedRunId = (btSelectedRunId === runId) ? null : runId;
  renderLegend();
  renderTable();
  // In aggregate mode, selecting a run switches to individual view for the specific run
  if (btSelectedRunId != null && btChartMode === "aggregate") {
    setChartMode("individual");
  } else {
    redrawChart();
  }
  if (btSelectedRunId != null) loadRunDetail(btSelectedRunId);
  else closeDetail();
}

function toggleRun(runId) {
  if (btHiddenRuns.has(runId)) btHiddenRuns.delete(runId);
  else btHiddenRuns.add(runId);
  renderLegend();
  redrawChart();
}

function btToggleAll(show) {
  const vis = filteredRuns();
  btHiddenRuns = show ? new Set() : new Set(vis.map(r => r.run_id));
  renderLegend();
  redrawChart();
}

function closeDetail() {
  document.getElementById("bt-detail").style.display = "none";
  btSelectedRunId = null;
  renderLegend(); renderTable(); redrawChart();
}

function showBtSubtab(name) {
  btDetailSubtab = name;
  document.querySelectorAll(".bt-subpane").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".bt-tabs a").forEach(el => el.classList.remove("active"));
  document.getElementById("bt-tab-" + name).classList.add("active");
  document.getElementById("bt-tab-" + name + "-link").classList.add("active");
}

async function loadRunDetail(runId) {
  const wrap = document.getElementById("bt-detail");
  document.getElementById("bt-detail-id").textContent = "#" + runId;
  wrap.style.display = "block";
  const r = await fetch(API_PREFIX + `/api/backtests/${runId}`).then(r => r.json());
  const meta = [];
  if (r.seed != null) meta.push(`seed ${r.seed}`);
  const winStr = formatWindow(r.start_date, r.end_date);
  if (winStr) meta.push(winStr);
  if (r.status) meta.push(r.status);
  if (r.n_trades != null) meta.push(`${r.n_trades} trades`);
  if (r.n_decisions != null) meta.push(`${r.n_decisions} decisions`);
  if (r.notes) meta.push(r.notes);
  const metaEl = document.getElementById("bt-detail-meta");
  metaEl.innerHTML = "";
  metaEl.appendChild(document.createTextNode(meta.join(" · ")));
  // Era pill + data-source pills
  const era = classifyEra(r.start_date, r.end_date);
  if (era) {
    metaEl.insertAdjacentHTML(
      "beforeend",
      ` <span class="pill" style="background:${era.bg};color:${era.fg};margin-left:6px;">${era.tag}</span>`,
    );
  }
  const srcs = dataSourcesForWindow(r.start_date, r.end_date);
  if (srcs.length) {
    const pillRow = srcs.map(s => {
      const mark = s.ok ? "✓" : "✗";
      const color = s.ok ? "#00c896" : "#ff4455";
      const bg = s.ok ? "rgba(0,200,150,0.10)" : "rgba(255,68,85,0.10)";
      return `<span class="pill" title="${s.hint}" style="background:${bg};color:${color};font-size:11px;">${mark} ${s.label}</span>`;
    }).join(" ");
    metaEl.insertAdjacentHTML(
      "beforeend",
      `<div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:6px;">${pillRow}</div>`,
    );
  }

  const tBody = document.querySelector("#bt-trades-tbl tbody");
  tBody.innerHTML = (r.trades || []).map(t => {
    const cls = (t.action||"").startsWith("SELL") ? "sell" : "buy";
    return `<tr><td>${t.sim_date || ''}</td>
      <td><span class="pill ${cls}">${t.action || ''}</span></td>
      <td>${t.ticker || ''}</td>
      <td class="num">${fmt(t.qty,4)}</td>
      <td class="num">${fmt(t.price)}</td>
      <td class="num">${fmt(t.value)}</td>
      <td class="muted">${(t.reason||"").slice(0,140)}</td></tr>`;
  }).join("") || `<tr><td colspan="7" class="muted">no trades</td></tr>`;

  const dBody = document.querySelector("#bt-decisions-tbl tbody");
  dBody.innerHTML = (r.decisions || []).map(d => {
    return `<tr><td>${d.sim_date || ''}</td>
      <td>${d.action || ''}</td>
      <td>${d.ticker || ''}</td>
      <td><span class="pill">${d.status || ''}</span></td>
      <td class="muted">${(d.detail||"").slice(0,140)}</td>
      <td class="num">${fmt(d.total_value)}</td></tr>`;
  }).join("") || `<tr><td colspan="6" class="muted">no decisions</td></tr>`;
}

function tickLastUpdated() {
  const el = document.getElementById("bt-last-updated");
  if (!el) return;
  if (btLastUpdated == null) { el.textContent = "last update: —"; return; }
  const s = Math.floor((Date.now() - btLastUpdated)/1000);
  el.textContent = `last updated ${s}s ago`;
}
setInterval(tickLastUpdated, 1000);

// ───────── Signal feed (from Digital Intern) ─────────
async function refreshSignals() {
  const ul = document.getElementById("signal-feed");
  try {
    const r = await fetch("/intern/api/articles?limit=3");
    if (!r.ok) {
      ul.innerHTML = `<li class="muted">signal feed unavailable (HTTP ${r.status})</li>`;
      return;
    }
    const arts = await r.json();
    if (!Array.isArray(arts) || !arts.length) {
      ul.innerHTML = `<li class="muted">no signals yet</li>`;
      return;
    }
    ul.innerHTML = arts.map(a => {
      const score = (a.score != null ? a.score : 0).toFixed(1);
      const url = a.url || "#";
      const title = (a.title || "(no title)").replace(/</g,"&lt;");
      const src = (a.source || "").replace(/</g,"&lt;");
      return `<li style="padding:6px 0;border-bottom:1px solid #1f2126;">
        <span class="pill" style="background:#1f3a4d;color:#4d9eff;margin-right:8px;">${score}</span>
        <a href="${url}" target="_blank" rel="noopener" style="color:#dde1e7;text-decoration:none">${title}</a>
        <span class="muted" style="margin-left:6px;">· ${src}</span>
      </li>`;
    }).join("");
  } catch (e) {
    ul.innerHTML = `<li class="muted">digital intern unreachable</li>`;
  }
}

// ───────── Live Data Feed widget (collector pulse from Digital Intern) ─────────
async function refreshDataFeed() {
  try {
    const r = await fetch(API_PREFIX + "/api/data-feed");
    if (!r.ok) return;
    const d = await r.json();
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set("df-1h",  d.articles_1h  != null ? d.articles_1h  + " articles" : "—");
    set("df-24h", d.articles_24h != null ? d.articles_24h + " articles" : "—");
    const srcEl = document.getElementById("df-sources");
    if (srcEl) {
      const top = (d.top_sources || []).slice(0, 3);
      srcEl.innerHTML = top.length
        ? "top: " + top.map(s => `<b style="color:#dde1e7;">${(s.name||"?").replace(/</g,'&lt;')}</b> <span style="color:#8b929d;">${s.count}</span>`).join(" · ")
        : '<span class="muted">no sources active</span>';
    }
    const asof = document.getElementById("df-asof");
    if (asof) asof.textContent = new Date().toLocaleTimeString();
  } catch (e) { /* silent */ }
}

// ───────── Portfolio Analytics ─────────
const SECTOR_COLORS = {
  semis: "#0acdff", semis_lev: "#1e88e5",
  optical: "#ab47bc",
  broad: "#00c896", broad_lev: "#43a047",
  tech: "#ffb74d", tech_lev: "#fb8c00",
  crypto_lev: "#ffd54f",
  bio_lev: "#ec407a", health_lev: "#e91e63",
  fin_lev: "#26a69a", defense_lev: "#7e57c2",
  housing_lev: "#8d6e63", util_lev: "#90a4ae",
  cash: "#455a64", other: "#8b929d",
};

function _sectorColor(name) { return SECTOR_COLORS[name] || "#8b929d"; }

async function refreshAnalytics() {
  let a;
  try { a = await fetch(API_PREFIX + "/api/analytics").then(r => r.json()); }
  catch (e) { return; }
  if (!a || a.error) return;

  const setStat = (id, txt, cls) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = txt;
    el.className = "v" + (cls ? " " + cls : "");
  };
  const sign = v => v == null ? "" : (v >= 0 ? "+" : "");
  const fmtPct = (v, d=2) => v == null ? "—" : sign(v) + fmt(v, d) + "%";
  const fmtUsd = (v, d=2) => v == null ? "—" : sign(v) + "$" + fmt(Math.abs(v), d);

  setStat("an-daily", a.daily_pl_usd == null ? "—" :
          `${fmtUsd(a.daily_pl_usd)} (${fmtPct(a.daily_pl_pct, 2)})`,
          a.daily_pl_usd == null ? null : (a.daily_pl_usd >= 0 ? "pos" : "neg"));
  setStat("an-dd", a.max_drawdown_usd == null ? "—" :
          `-${fmt(a.max_drawdown_usd)} (${fmt(a.max_drawdown_pct)}%)`,
          a.max_drawdown_usd > 0 ? "neg" : null);
  setStat("an-sharpe", a.sharpe_annualized == null ? "—" : fmt(a.sharpe_annualized, 2),
          a.sharpe_annualized != null ? (a.sharpe_annualized >= 0 ? "pos" : "neg") : null);
  if (a.win_rate_pct == null) setStat("an-winrate", `— (0 trips)`);
  else setStat("an-winrate", `${fmt(a.win_rate_pct, 1)}% (${a.n_round_trips})`,
                a.win_rate_pct >= 50 ? "pos" : "neg");
  setStat("an-avgw", a.avg_winner_usd == null ? "—" : "$" + fmt(a.avg_winner_usd), a.avg_winner_usd != null ? "pos" : null);
  setStat("an-avgl", a.avg_loser_usd == null ? "—" : fmtUsd(a.avg_loser_usd), a.avg_loser_usd != null ? "neg" : null);
  setStat("an-realized", fmtUsd(a.realized_pl_usd, 2), a.realized_pl_usd >= 0 ? "pos" : "neg");

  setStat("an-pf", a.profit_factor == null ? "—" : fmt(a.profit_factor, 2),
          a.profit_factor != null ? (a.profit_factor >= 1 ? "pos" : "neg") : null);
  setStat("an-sortino", a.sortino_annualized == null ? "—" : fmt(a.sortino_annualized, 2),
          a.sortino_annualized != null ? (a.sortino_annualized >= 0 ? "pos" : "neg") : null);
  setStat("an-calmar", a.calmar_ratio == null ? "—" : fmt(a.calmar_ratio, 2),
          a.calmar_ratio != null ? (a.calmar_ratio >= 0 ? "pos" : "neg") : null);
  setStat("an-beta", a.sp500_beta == null ? "—" : fmt(a.sp500_beta, 2));
  setStat("an-corr", a.sp500_correlation == null ? "—" : fmt(a.sp500_correlation, 2));
  setStat("an-hold", a.avg_holding_days == null ? "—" :
          fmt(a.avg_holding_days, 1) + "d");

  // Sector stacked bar
  const sectors = a.sector_exposure_pct || {};
  const cashPct = a.cash_pct || 0;
  const segs = [];
  for (const [name, pct] of Object.entries(sectors)) {
    if (pct > 0) segs.push({ name, pct, color: _sectorColor(name) });
  }
  if (cashPct > 0) segs.push({ name: "cash", pct: cashPct, color: _sectorColor("cash") });
  segs.sort((a, b) => b.pct - a.pct);

  const barEl = document.getElementById("an-sector-bar");
  if (barEl) {
    barEl.innerHTML = segs.map(s =>
      `<div title="${s.name} ${fmt(s.pct,1)}%" style="flex:${s.pct};background:${s.color};border-right:1px solid #0d1117;"></div>`
    ).join("") || `<div class="muted" style="padding:3px 8px;font-size:12px;">no allocations</div>`;
  }
  const legEl = document.getElementById("an-sector-legend");
  if (legEl) {
    legEl.innerHTML = segs.map(s =>
      `<span><span style="display:inline-block;width:10px;height:10px;background:${s.color};border-radius:2px;margin-right:5px;vertical-align:middle;"></span>${s.name}: ${fmt(s.pct,1)}%</span>`
    ).join("") || `<span class="muted">no allocations</span>`;
  }
}

// ───────── Sector Pulse ─────────
async function refreshSectorPulse() {
  let r;
  try { r = await fetch(API_PREFIX + "/api/sector-pulse").then(r => r.json()); }
  catch (e) { return; }
  if (!r || !r.tickers) return;
  const grid = document.getElementById("sp-grid");
  if (!grid) return;
  document.getElementById("sp-asof").textContent = r.as_of ? "as of " + r.as_of.replace("T"," ").slice(0,16) + " UTC" : "";
  grid.innerHTML = r.tickers.map(t => {
    const rsi = t.rsi;
    const rsiCls = rsi == null ? "muted" :
                   rsi >= 70 ? "neg" :
                   rsi <= 30 ? "pos" : "";
    const mom5 = t.mom_5d;
    const mom5Cls = mom5 == null ? "muted" : (mom5 >= 0 ? "pos" : "neg");
    const px = t.price;
    const news = t.news_count_24h || 0;
    const urgent = t.news_urgent_24h || 0;
    const newsBadge = urgent > 0
      ? `<span style="background:#3a1b1b;color:#ff4455;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:600;">${urgent}!</span>`
      : news > 0
        ? `<span style="background:#1f3a4d;color:#4d9eff;padding:1px 6px;border-radius:8px;font-size:10px;">${news}</span>`
        : `<span class="muted" style="font-size:10px;">0</span>`;
    const headline = t.top_headline
      ? `<div style="margin-top:6px;font-size:11px;line-height:1.4;color:#dde1e7;">
           ${t.top_url ? `<a href="${t.top_url}" target="_blank" rel="noopener" style="color:#dde1e7;text-decoration:none;">${(t.top_headline||'').slice(0,100)}</a>` : (t.top_headline||'').slice(0,100)}
         </div>`
      : `<div class="muted" style="margin-top:6px;font-size:11px;">no news</div>`;
    return `<div style="background:#0d1117;border:1px solid #1f2126;border-radius:6px;padding:10px;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;">
        <span style="font-weight:600;color:#eceff1;font-size:14px;">${t.ticker}</span>
        <span style="font-size:13px;color:#dde1e7;font-variant-numeric:tabular-nums;">${px == null ? '—' : '$'+fmt(px)}</span>
      </div>
      <div style="display:flex;gap:8px;font-size:11px;margin-top:5px;color:#8b929d;">
        <span>RSI <span class="${rsiCls}">${rsi == null ? '—' : fmt(rsi,1)}</span></span>
        <span>5d <span class="${mom5Cls}">${mom5 == null ? '—' : (mom5>=0?'+':'')+fmt(mom5,1)+'%'}</span></span>
        <span style="margin-left:auto;">${newsBadge}</span>
      </div>
      ${headline}
    </div>`;
  }).join("");
}

// ───────── Daily briefing card ─────────
async function refreshBriefing() {
  try {
    const r = await fetch(API_PREFIX + "/api/briefing").then(r => r.json());
    if (r.error) return;
    const dot = document.getElementById("briefing-dot");
    if (dot) dot.style.background = r.market_open ? "#00c896" : "#ff4455";
    document.getElementById("briefing-status").textContent = r.status_line || "";
    document.getElementById("briefing-asof").textContent = (r.as_of || "").replace("T"," ").slice(0,19);
    // Futures row
    const futWrap = document.getElementById("briefing-futures");
    const futNames = {"ES=F":"S&P fut","NQ=F":"NQ fut","CL=F":"WTI","GC=F":"Gold"};
    futWrap.innerHTML = Object.entries(r.futures || {}).map(([sym,px]) => {
      const label = futNames[sym] || sym;
      const value = (px == null) ? "—" : Number(px).toLocaleString(undefined,{maximumFractionDigits:2});
      return `<div><span class="muted" style="font-size:11px;">${label}</span><div style="font-variant-numeric:tabular-nums;font-size:15px;color:#dde1e7;">${value}</div></div>`;
    }).join("");
    // Urgent news (top 5)
    const urgEl = document.getElementById("briefing-urgent");
    const urgent = r.urgent_news || [];
    if (!urgent.length) {
      urgEl.innerHTML = `<li class="muted" style="padding:4px 0;">no urgent news in the last 8h</li>`;
    } else {
      urgEl.innerHTML = urgent.map(u => {
        const sc = (u.ai_score != null) ? Number(u.ai_score).toFixed(1) : "—";
        const tk = (u.tickers || []).slice(0,3).join(" ");
        return `<li style="padding:4px 0;border-bottom:1px solid #1f2126;">
          <span style="display:inline-block;min-width:34px;color:#ff4455;font-variant-numeric:tabular-nums;font-weight:600;">${sc}</span>
          <span style="color:#dde1e7;">${(u.title || "").replace(/[<>]/g, '')}</span>
          ${tk ? `<span class="muted" style="font-size:11px;margin-left:6px;">[${tk}]</span>` : ""}
        </li>`;
      }).join("");
    }
  } catch (e) { console.error("briefing:", e); }
}

// ───────── Trade suggestions card ─────────
async function refreshSuggestions() {
  try {
    const r = await fetch(API_PREFIX + "/api/suggestions").then(r => r.json());
    if (r.error) {
      document.getElementById("sug-summary").textContent = "error: " + r.error;
      return;
    }
    const counts = r.action_counts || {};
    const summary = Object.entries(counts).map(([a,n]) => `${n} ${a}`).join(" · ") || "no actionable candidates";
    document.getElementById("sug-summary").textContent = `${r.n_candidates} candidates from ${r.n_signals_used} signals — ${summary}`;
    document.getElementById("sug-meta").textContent = (r.as_of || "").replace("T"," ").slice(0,19);
    const tbody = document.querySelector("#sug-tbl tbody");
    const items = r.suggestions || [];
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="9" class="muted">no suggestions — no actionable news in the last 6h</td></tr>`;
      return;
    }
    const actionStyle = {
      "BUY":   "background:#1b3a2a;color:#00c896;",
      "ADD":   "background:#1b3a2a;color:#00c896;",
      "TRIM":  "background:#3a2f1b;color:#ffb74d;",
      "EXIT":  "background:#3a1b1b;color:#ff4455;",
      "WATCH": "background:#1f3a4d;color:#4d9eff;",
      "HOLD":  "background:#1f2933;color:#dde1e7;",
    };
    tbody.innerHTML = items.map(s => {
      const styleA = actionStyle[s.action] || actionStyle["HOLD"];
      const px = (s.price == null) ? "—" : "$" + Number(s.price).toFixed(2);
      const qty = s.held_qty ? Number(s.held_qty).toFixed(2) : "—";
      const rsi = (s.rsi == null) ? "—" : Number(s.rsi).toFixed(0);
      const rsiCls = (s.rsi != null && s.rsi >= 70) ? "neg" : (s.rsi != null && s.rsi <= 35) ? "pos" : "";
      const urgent = s.news_urgent ? `<span style="color:#ff4455;font-weight:600;">!</span>` : "";
      const newsCell = s.news_count > 0
        ? `<span style="color:#4d9eff;">${s.news_count}</span> <span class="muted">@</span> ${Number(s.news_max_score).toFixed(1)} ${urgent}`
        : `<span class="muted">0</span>`;
      const reasons = (s.reasons || []).slice(0,3).join(" · ");
      const head = s.top_headline ? (s.top_url
        ? `<a href="${s.top_url}" target="_blank" rel="noopener" style="color:#dde1e7;">${s.top_headline.replace(/[<>]/g,'')}</a>`
        : `<span class="muted">${s.top_headline.replace(/[<>]/g,'')}</span>`) : `<span class="muted">—</span>`;
      return `<tr>
        <td><span class="pill" style="${styleA}padding:3px 8px;font-size:11px;font-weight:600;">${s.action}</span></td>
        <td style="font-weight:600;">${s.ticker}</td>
        <td class="num">${Number(s.conviction).toFixed(2)}</td>
        <td class="num">${px}</td>
        <td class="num muted">${qty}</td>
        <td class="num">${newsCell}</td>
        <td class="num ${rsiCls}">${rsi}</td>
        <td class="muted" style="font-size:11px;">${reasons}</td>
        <td style="font-size:12px;">${head}</td>
      </tr>`;
    }).join("");
  } catch (e) { console.error("suggestions:", e); }
}

// ───────── Risk panel card ─────────
async function refreshRisk() {
  try {
    const r = await fetch(API_PREFIX + "/api/risk").then(r => r.json());
    if (r.error) return;
    const top1Txt = r.concentration_top1_ticker
      ? `${Number(r.concentration_top1_pct).toFixed(1)}% <span class="muted" style="font-size:13px;">${r.concentration_top1_ticker}</span>`
      : "—";
    const top1El = document.getElementById("risk-top1");
    top1El.innerHTML = top1Txt;
    top1El.className = "v " + (r.concentration_top1_pct >= 40 ? "neg" : "");
    document.getElementById("risk-top3").textContent = (r.concentration_top3_pct != null) ? Number(r.concentration_top3_pct).toFixed(1) + "%" : "—";
    const levEl = document.getElementById("risk-lev");
    levEl.textContent = (r.leveraged_pct != null) ? Number(r.leveraged_pct).toFixed(1) + "%" : "—";
    levEl.className = "v " + (r.leveraged_pct >= 30 ? "neg" : "");
    const shockEl = document.getElementById("risk-shock");
    if (r.spy_shock_3pct_usd != null) {
      const v = Number(r.spy_shock_3pct_usd);
      const pct = Number(r.spy_shock_3pct_pct || 0);
      shockEl.innerHTML = `${v >= 0 ? "+" : ""}$${v.toFixed(2)} <span class="muted" style="font-size:12px;">(${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%)</span>`;
      shockEl.className = "v " + (v < 0 ? "neg" : "pos");
    }
    document.getElementById("risk-age").textContent = (r.median_age_days != null) ? r.median_age_days : "—";
    const staleEl = document.getElementById("risk-stale-n");
    const stale = r.stale_positions || [];
    staleEl.textContent = stale.length;
    staleEl.className = "v " + (stale.length > 0 ? "neg" : "");
    const staleList = document.getElementById("risk-stale-list");
    if (!stale.length) {
      staleList.innerHTML = `<span class="muted">no stale positions — all holds are either fresh or moving</span>`;
    } else {
      staleList.innerHTML = "Stale: " + stale.map(s =>
        `<span style="display:inline-block;background:#1f2126;border:1px solid #3a2f1b;border-radius:4px;padding:3px 8px;margin-right:6px;margin-bottom:4px;">${s.ticker} ${s.age_days}d ${s.pl_pct >= 0 ? "+" : ""}${s.pl_pct}%</span>`
      ).join("");
    }
  } catch (e) { console.error("risk:", e); }
}

// ───────── Earnings radar ─────────
async function refreshEarningsRisk() {
  let r;
  try { r = await fetch(API_PREFIX + "/api/earnings-risk").then(r => r.json()); }
  catch (e) { return; }
  const list = document.getElementById("er-list");
  const meta = document.getElementById("er-meta");
  const asof = document.getElementById("er-asof");
  if (!list) return;
  if (!r || r.error) { list.innerHTML = `<li class="muted">unavailable</li>`; return; }
  if (asof && r.as_of) asof.textContent = r.as_of.slice(11, 16) + " UTC";
  const evs = r.events || [];
  if (!r.source_ok) {
    meta.textContent = "earnings calendar (:8080) unreachable";
  } else {
    meta.innerHTML = `${r.n_held_reporting} holding(s) reporting · ` +
      `<span class="${r.n_imminent > 0 ? 'neg' : 'muted'}">${r.n_imminent} imminent (≤3d)</span> · ` +
      `$${Number(r.held_exposure_at_risk_usd || 0).toFixed(0)} exposure at risk`;
  }
  if (!evs.length) {
    list.innerHTML = `<li class="muted">no earnings within horizon for holdings or watchlist</li>`;
    return;
  }
  const tierStyle = {
    HELD_IMMINENT: "background:#3a1b1b;border:1px solid #7a2f2f;",
    HELD_SOON:     "background:#3a2f1b;border:1px solid #7a5f2f;",
    WATCH:         "background:#1f2126;border:1px solid #2f3540;",
  };
  const tierLabel = { HELD_IMMINENT: "⚠ HELD", HELD_SOON: "HELD", WATCH: "watch" };
  list.innerHTML = evs.slice(0, 14).map(e => {
    const d = e.days_away == null ? "?" : Number(e.days_away).toFixed(1) + "d";
    const exp = e.held ? ` · $${Number(e.exposure_usd).toFixed(0)}` : "";
    return `<li style="padding:6px 8px;margin-bottom:4px;border-radius:5px;${tierStyle[e.tier] || ''}">` +
      `<b>${e.ticker}</b> <span class="muted" style="font-size:11px;">${tierLabel[e.tier] || ''}</span>` +
      `<span style="float:right;">in ${d}${exp}</span></li>`;
  }).join("");
}

// ───────── Greeks card (options exposure) ─────────
async function refreshGreeks() {
  try {
    const r = await fetch(API_PREFIX + "/api/greeks").then(r => r.json());
    if (r.error) { return; }
    const positions = (r.positions || []).filter(p => p.type === "call" || p.type === "put");
    const card = document.getElementById("greeks-card");
    if (!card) return;
    // Hide card entirely when there are no option positions — keeps dashboard clean.
    if (positions.length === 0) { card.style.display = "none"; return; }
    card.style.display = "block";
    const t = r.totals || {};
    document.getElementById("gk-asof").textContent = r.as_of ? r.as_of.replace("T"," ").slice(0,16) : "—";
    const dElem = document.getElementById("gk-delta");
    dElem.textContent = fmt(t.delta, 2);
    dElem.className = "v " + ((t.delta || 0) >= 0 ? "pos" : "neg");
    document.getElementById("gk-gamma").textContent = fmt(t.gamma, 5);
    const thElem = document.getElementById("gk-theta");
    thElem.textContent = "$" + fmt(t.theta, 2);
    thElem.className = "v " + ((t.theta || 0) >= 0 ? "pos" : "neg");
    document.getElementById("gk-vega").textContent = "$" + fmt(t.vega, 2);
    document.getElementById("gk-notional").textContent = dollar(t.gross_notional);
    document.getElementById("gk-deltapct").textContent = (t.delta_pct_port != null) ? (fmt(t.delta_pct_port,1) + "%") : "—";
    const tbody = document.querySelector("#gk-tbl tbody");
    tbody.innerHTML = positions.map(p => {
      const cls = (p.delta || 0) >= 0 ? "pos" : "neg";
      const ivStr = p.iv != null ? (fmt(p.iv * 100, 1) + "%") : "—";
      const dteStr = p.days_to_expiry != null ? (p.days_to_expiry + "d") : "";
      return `<tr>
        <td>${p.ticker}</td>
        <td>${p.type.toUpperCase()}</td>
        <td class="num">${fmt(p.qty, 0)}</td>
        <td class="num">${p.strike || "—"} / ${p.expiry || "—"} ${dteStr ? `<span class="muted">(${dteStr})</span>` : ""}</td>
        <td class="num">${ivStr}</td>
        <td class="num ${cls}">${fmt(p.delta, 2)}</td>
        <td class="num">${fmt(p.gamma, 5)}</td>
        <td class="num">${fmt(p.theta, 2)}</td>
        <td class="num">${fmt(p.vega, 2)}</td>
      </tr>`;
    }).join("");
  } catch (e) { console.error("greeks:", e); }
}

// ───────── DRAM/Semis heatmap ─────────
function hmColorFor(pct) {
  if (pct == null) return "#1f2126";
  // Map [-5%..+5%] to red..green via HSL.
  const clamped = Math.max(-5, Math.min(5, pct));
  // -5 → hue 0 (red), +5 → hue 130 (green)
  const hue = 65 + clamped * 13;
  const sat = 55;
  const lit = 24 + Math.abs(clamped) * 1.5;
  return `hsl(${hue}, ${sat}%, ${lit}%)`;
}
async function refreshHeatmap() {
  try {
    const r = await fetch(API_PREFIX + "/api/sector-heatmap").then(r => r.json());
    if (r.error) {
      document.getElementById("hm-grid").innerHTML =
        `<div class="muted">heatmap error: ${r.error}</div>`;
      return;
    }
    document.getElementById("hm-asof").textContent = r.as_of ? r.as_of.replace("T"," ").slice(0,16) : "—";
    const bench = r.reference_mom_5d;
    const benchStr = bench != null ? `${r.reference} 5d ${bench >= 0 ? "+" : ""}${fmt(bench, 2)}%` : `${r.reference} —`;
    document.getElementById("hm-bench").textContent = "Benchmark: " + benchStr;

    const grid = document.getElementById("hm-grid");
    const buckets = r.buckets || [];
    grid.innerHTML = buckets.map(b => {
      const cells = (b.tickers || []).map(t => {
        const m5 = t.mom_5d;
        const rs = t.vs_sox_5d;
        const news = t.n || 0;
        const urg = t.urgent || 0;
        const bg = hmColorFor(m5);
        const rsStr = rs == null ? "" : `<span style="color:${rs >= 0 ? '#7fff00' : '#ff7b7b'};font-size:10px;margin-left:4px;">vs SOX ${rs >= 0 ? '+' : ''}${fmt(rs,1)}</span>`;
        const newsStr = news > 0
          ? `<span style="color:#dde1e7;font-size:10px;margin-left:6px;">📰 ${news}${urg ? `<span style="color:#ff4455">!</span>` : ""}</span>`
          : "";
        const rsi = t.rsi;
        const rsiStr = rsi == null ? "" : `<span style="color:${rsi > 70 ? '#ff7b7b' : (rsi < 30 ? '#80deea' : '#8b929d')};font-size:10px;margin-left:6px;">RSI ${fmt(rsi,0)}</span>`;
        const px = t.price == null ? "—" : "$" + fmt(t.price, 2);
        return `<div style="background:${bg};border:1px solid #1f2126;border-radius:4px;padding:6px 8px;min-width:130px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
            <span style="font-weight:bold;color:#fff;">${t.ticker}</span>
            <span style="font-size:11px;color:#dde1e7;">${px}</span>
          </div>
          <div style="font-size:13px;color:${(m5 || 0) >= 0 ? '#7fff00' : '#ff7b7b'};font-weight:bold;">${m5 == null ? "—" : (m5 >= 0 ? "+" : "") + fmt(m5, 2) + "%"}</div>
          <div style="margin-top:2px;">${rsStr}${rsiStr}${newsStr}</div>
        </div>`;
      }).join("");
      const bm = b.avg_mom_5d;
      const bmStr = bm == null ? "—" : (bm >= 0 ? "+" : "") + fmt(bm, 2) + "%";
      const bmCls = (bm || 0) >= 0 ? "pos" : "neg";
      return `<div style="margin-bottom:14px;">
        <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
          <span style="text-transform:uppercase;font-size:11px;letter-spacing:0.5px;color:#8b929d;">${b.name.replace(/_/g, " ")}</span>
          <span class="${bmCls}" style="font-size:11px;">avg 5d ${bmStr}</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;">${cells}</div>
      </div>`;
    }).join("");
  } catch (e) { console.error("heatmap:", e); }
}

// ───────── DecisionScorer per-position predictions ─────────
function scorerColor(v) {
  if (v == null) return "#dde1e7";
  if (v >= 2) return "#7fff00";
  if (v >= 0.5) return "#a5d6a7";
  if (v >= -0.5) return "#dde1e7";
  if (v >= -2) return "#ff9100";
  return "#ff4455";
}
function verdictBadge(v) {
  const colors = {
    STRONG_HOLD: ["#1b5e20", "#a5d6a7"],
    HOLD:        ["#2e7d32", "#c5e1a5"],
    NEUTRAL:     ["#37474f", "#dde1e7"],
    TRIM:        ["#ef6c00", "#ffe0b2"],
    EXIT:        ["#b71c1c", "#ffcdd2"],
  };
  const [bg, fg] = colors[v] || ["#1f2126", "#8b929d"];
  return `<span style="background:${bg};color:${fg};padding:1px 6px;border-radius:3px;font-size:11px;letter-spacing:0.5px;">${v || "—"}</span>`;
}
async function refreshScorer() {
  try {
    const r = await fetch(API_PREFIX + "/api/scorer-predictions").then(r => r.json());
    if (r.error) {
      document.getElementById("sc-meta").textContent = "scorer error: " + r.error;
      return;
    }
    document.getElementById("sc-asof").textContent = r.as_of ? r.as_of.replace("T"," ").slice(0,16) : "—";
    const meta = r.is_trained
      ? `trained (n=${r.n_train}) · regime mult ${fmt(r.regime_mult, 2)} · gate ≥ ${r.gate_threshold}`
      : `not trained yet (n=${r.n_train}/${r.gate_threshold}) — predictions will be 0.00 until threshold reached`;
    document.getElementById("sc-meta").textContent = meta;
    const tbody = document.querySelector("#sc-tbl tbody");
    const rows = r.predictions || [];
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="muted">no open stock positions</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.map(p => {
      const v = p.pred_5d_return_pct;
      const sign = v >= 0 ? "+" : "";
      const newsCell = (p.news_count || 0) > 0
        ? `${p.news_count}${(p.news_urgent || 0) > 0 ? ` <span style="color:#ff4455">!</span>` : ""}`
        : "—";
      return `<tr>
        <td><strong>${p.ticker}</strong></td>
        <td class="num" style="color:${scorerColor(v)};font-weight:bold;">${v == null ? "—" : sign + fmt(v, 2) + "%"}</td>
        <td>${verdictBadge(p.verdict)}</td>
        <td class="num">${p.rsi == null ? "—" : fmt(p.rsi, 0)}</td>
        <td class="num">${p.macd == null ? "—" : fmt(p.macd, 3)}</td>
        <td class="num">${p.mom_5d == null ? "—" : (p.mom_5d >= 0 ? "+" : "") + fmt(p.mom_5d, 2) + "%"}</td>
        <td class="num">${p.mom_20d == null ? "—" : (p.mom_20d >= 0 ? "+" : "") + fmt(p.mom_20d, 2) + "%"}</td>
        <td class="num">${newsCell}</td>
      </tr>`;
    }).join("");
  } catch (e) { console.error("scorer:", e); }
}

// ───────── Deduped signals feed ─────────
async function refreshDedupedNews() {
  try {
    const r = await fetch(API_PREFIX + "/api/news-deduped?hours=6&min_score=4").then(r => r.json());
    if (r.error) {
      document.getElementById("nd-list").innerHTML = `<li class="muted">${r.error}</li>`;
      return;
    }
    document.getElementById("nd-asof").textContent = r.as_of ? r.as_of.replace("T"," ").slice(0,16) : "—";
    const meta = `${r.n_after_dedup} unique signals from ${r.n_raw} raw articles (compression ${fmt(r.compression_ratio, 1)}x) · halflife ${r.halflife_hours}h`;
    document.getElementById("nd-meta").textContent = meta;
    const items = (r.articles || []).slice(0, 15);
    const list = document.getElementById("nd-list");
    if (!items.length) {
      list.innerHTML = `<li class="muted">no signals in window</li>`;
      return;
    }
    list.innerHTML = items.map(a => {
      const score = a.ai_score != null ? fmt(a.ai_score, 1) : "—";
      const urgD = a.urgency_decayed != null ? fmt(a.urgency_decayed, 2) : "—";
      const dups = a.dup_count && a.dup_count > 1
        ? `<span class="muted" style="font-size:11px;margin-left:6px;">×${a.dup_count}</span>` : "";
      const urgBadge = (a.urgency_decayed || 0) >= 0.7
        ? `<span style="background:#ff1744;color:#fff;border-radius:3px;padding:1px 5px;font-size:10px;margin-right:6px;">URG ${urgD}</span>`
        : ((a.urgency_decayed || 0) > 0
            ? `<span style="background:#ff9100;color:#000;border-radius:3px;padding:1px 5px;font-size:10px;margin-right:6px;">u ${urgD}</span>`
            : "");
      const tickers = (a.tickers || []).slice(0, 4).map(t =>
        `<span style="background:#1f2126;color:#0acdff;font-size:10px;padding:1px 5px;border-radius:3px;margin-left:4px;">${t}</span>`
      ).join("");
      const title = (a.title || "").replace(/</g, "&lt;");
      const ts = a.first_seen ? a.first_seen.replace("T", " ").slice(5, 16) : "";
      return `<li style="padding:6px 0;border-bottom:1px solid #1f2126;">
        ${urgBadge}<span style="color:#dde1e7;">${title}</span>${dups}
        <div class="muted" style="font-size:11px;margin-top:3px;">
          [${score}] ${a.source || "?"} · ${ts}${tickers}
        </div>
      </li>`;
    }).join("");
  } catch (e) { console.error("deduped:", e); }
}

// ───────── Position thesis (new 2026-05-15) ─────────
function verdictPill(v) {
  const colors = {
    STRONG_HOLD: ["#1b5e20", "#a5d6a7"],
    HOLD:        ["#33691e", "#c5e1a5"],
    WATCH:       ["#37474f", "#dde1e7"],
    TRIM:        ["#bf360c", "#ffccbc"],
    EXIT:        ["#b71c1c", "#ffcdd2"],
  };
  const [bg, fg] = colors[v] || ["#37474f", "#dde1e7"];
  return `<span style="background:${bg};color:${fg};border-radius:3px;padding:2px 8px;font-size:11px;font-weight:bold;letter-spacing:0.5px;">${v}</span>`;
}

async function refreshThesis() {
  try {
    const r = await fetch(API_PREFIX + "/api/position-thesis").then(r => r.json());
    document.getElementById("th-asof").textContent =
      r.as_of ? r.as_of.replace("T"," ").slice(0,16) : "—";
    if (r.error) {
      document.getElementById("th-meta").textContent = "error: " + r.error;
      document.getElementById("th-grid").innerHTML = "";
      return;
    }
    const cards = r.cards || [];
    const meta = `${r.n_positions || 0} open positions · scorer ` +
      (r.scorer_trained ? `trained (n=${r.scorer_n_train})` : `untrained (n=${r.scorer_n_train})`);
    document.getElementById("th-meta").textContent = meta;
    const grid = document.getElementById("th-grid");
    if (!cards.length) {
      grid.innerHTML = `<div class="muted">no open positions</div>`;
      return;
    }
    grid.innerHTML = cards.map(c => {
      const pl = c.unrealized_pl || 0;
      const plPct = c.pl_pct || 0;
      const plColor = pl >= 0 ? "#00c896" : "#ff4455";
      const plSign = pl >= 0 ? "+" : "";
      const news = c.news || {};
      const head = (news.headlines || [])[0];
      const headHtml = head
        ? `<div class="muted" style="font-size:11px;margin-top:4px;">📰 [${fmt(head.score,1)}] ${(head.title||"").replace(/</g,"&lt;").slice(0,120)}</div>`
        : `<div class="muted" style="font-size:11px;margin-top:4px;">no recent news</div>`;
      const ld = c.last_decision;
      const ldHtml = ld
        ? `<div style="font-size:11px;color:#90a4ae;margin-top:4px;">last: <strong>${ld.action.replace(/→.*/,'').trim()}</strong> conf=${ld.confidence!=null?fmt(ld.confidence,2):"?"} · ${(ld.reasoning||"").replace(/</g,"&lt;").slice(0,140)}</div>`
        : "";
      const rsi = c.rsi != null ? fmt(c.rsi, 0) : "—";
      const m5 = c.mom_5d != null ? (c.mom_5d >= 0 ? "+" : "") + fmt(c.mom_5d, 1) + "%" : "—";
      const m20 = c.mom_20d != null ? (c.mom_20d >= 0 ? "+" : "") + fmt(c.mom_20d, 1) + "%" : "—";
      const pred = c.scorer_pred_5d;
      const predHtml = pred != null
        ? `<span style="color:${scorerColor(pred)};">${pred>=0?"+":""}${fmt(pred,2)}%</span>`
        : "—";
      const newsPulse = news.n
        ? `${news.n}·<span style="color:#00c896">${news.bull||0}↑</span>/<span style="color:#ff4455">${news.bear||0}↓</span> avg ${fmt(news.avg_score,1)}`
        : "<span class='muted'>—</span>";
      return `<div style="background:#0d1117;border:1px solid #1f2126;border-radius:6px;padding:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <div><strong style="font-size:15px;color:#e0e0e0;">${c.ticker}</strong>
            <span class="muted" style="font-size:11px;margin-left:6px;">qty ${fmt(c.qty,4)} @ $${fmt(c.avg_cost,2)} · ${fmt(c.days_held,1)}d</span>
          </div>
          <div>${verdictPill(c.verdict)}</div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:12px;color:#dde1e7;margin-bottom:6px;">
          <span>P/L <span style="color:${plColor};font-weight:bold;">${plSign}$${fmt(pl,2)} (${plSign}${fmt(plPct,2)}%)</span></span>
          <span>scorer ${predHtml}</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:#90a4ae;margin-bottom:4px;">
          <span>RSI ${rsi}</span><span>mom5 ${m5}</span><span>mom20 ${m20}</span>
          <span>news ${newsPulse}</span>
        </div>
        <div style="font-size:11px;color:#dde1e7;font-style:italic;margin-top:6px;">→ ${c.thesis||"—"}</div>
        ${headHtml}
        ${ldHtml}
      </div>`;
    }).join("");
  } catch (e) { console.error("thesis:", e); }
}

// ───────── Drawdown anatomy (new 2026-05-15) ─────────
async function refreshDrawdown() {
  try {
    const r = await fetch(API_PREFIX + "/api/drawdown").then(r => r.json());
    document.getElementById("dd-asof").textContent =
      r.as_of ? r.as_of.replace("T"," ").slice(0,16) : "—";
    if (r.error) {
      document.getElementById("dd-pct").textContent = "err";
      return;
    }
    document.getElementById("dd-current").textContent = "$" + fmt(r.current_value, 2);
    document.getElementById("dd-peak").textContent = "$" + fmt(r.peak_value, 2);
    const ddPct = r.drawdown_pct || 0;
    const ddEl = document.getElementById("dd-pct");
    if (r.at_high_water) {
      ddEl.innerHTML = `<span style="color:#00c896;font-weight:bold;">◆ at high-water</span>`;
    } else {
      const col = ddPct <= -5 ? "#ff4455" : (ddPct <= -2 ? "#ff9100" : "#ffd54f");
      ddEl.innerHTML = `<span style="color:${col};">${fmt(ddPct,2)}% ($${fmt(r.drawdown_abs,2)})</span>`;
    }
    document.getElementById("dd-trough").textContent =
      r.trough_value != null ? `$${fmt(r.trough_value,2)} (${fmt(r.trough_pct,2)}%)` : "—";
    document.getElementById("dd-hours").textContent =
      r.hours_in_dd != null ? fmt(r.hours_in_dd, 1) + "h" : "—";
    document.getElementById("dd-rec").textContent =
      (r.at_high_water ? "100" : fmt(r.recovery_pct, 0)) + "%";
    const tbody = document.querySelector("#dd-tbl tbody");
    const rows = r.contributors || [];
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="muted">no open positions</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.map(p => {
      const pl = p.unrealized_pl || 0;
      const plPct = p.pl_pct || 0;
      const color = pl >= 0 ? "#00c896" : "#ff4455";
      const dragBadge = p.drag
        ? `<span style="background:#b71c1c;color:#ffcdd2;border-radius:3px;padding:1px 5px;font-size:10px;">DRAG</span>`
        : `<span class="muted">—</span>`;
      return `<tr>
        <td><strong>${p.ticker}</strong> <span class="muted" style="font-size:10px;">${p.type||""}</span></td>
        <td class="num">${p.qty}</td>
        <td class="num">$${fmt(p.avg_cost,2)}</td>
        <td class="num">$${fmt(p.current_price,2)}</td>
        <td class="num" style="color:${color};font-weight:bold;">${pl>=0?"+":""}$${fmt(pl,2)}</td>
        <td class="num" style="color:${color};">${plPct>=0?"+":""}${fmt(plPct,2)}%</td>
        <td>${dragBadge}</td>
      </tr>`;
    }).join("");
  } catch (e) { console.error("drawdown:", e); }
}

// ───────── Confidence calibration + signal attribution (new 2026-05-15) ─────────
async function refreshCalibration() {
  try {
    const r = await fetch(API_PREFIX + "/api/calibration").then(r => r.json());
    document.getElementById("cal-asof").textContent =
      r.as_of ? r.as_of.replace("T"," ").slice(0,16) : "—";
    if (r.error) {
      document.getElementById("cal-meta").textContent = "error: " + r.error;
      return;
    }
    document.getElementById("cal-meta").textContent =
      `${r.n_decisions_parsed||0} decisions parsed · ${r.n_realized_trades||0} realized round-trips matched`;
    const confTbody = document.querySelector("#cal-conf-tbl tbody");
    const confRows = r.confidence_buckets || [];
    if (!confRows.some(b => b.n)) {
      confTbody.innerHTML = `<tr><td colspan="5" class="muted">no closed trades yet — calibration builds over time</td></tr>`;
    } else {
      confTbody.innerHTML = confRows.map(b => {
        const wrColor = b.win_rate >= 60 ? "#00c896" : (b.win_rate >= 40 ? "#ffd54f" : "#ff4455");
        const retColor = b.avg_return > 0 ? "#00c896" : "#ff4455";
        return `<tr>
          <td>${b.bucket}</td>
          <td class="num">${b.n}</td>
          <td class="num" style="color:${b.n?wrColor:'#8b929d'};">${b.n?fmt(b.win_rate,1)+"%":"—"}</td>
          <td class="num" style="color:${b.n?retColor:'#8b929d'};">${b.n?(b.avg_return>=0?"+":"")+fmt(b.avg_return,2)+"%":"—"}</td>
          <td class="num">${b.n?fmt(b.avg_conf,2):"—"}</td>
        </tr>`;
      }).join("");
    }
    const srcTbody = document.querySelector("#cal-src-tbl tbody");
    const srcRows = (r.signal_sources || []).sort((a,b) => (b.n||0)-(a.n||0));
    if (!srcRows.some(s => s.n)) {
      srcTbody.innerHTML = `<tr><td colspan="5" class="muted">no realized trades yet</td></tr>`;
    } else {
      srcTbody.innerHTML = srcRows.map(s => {
        const wrColor = s.win_rate >= 60 ? "#00c896" : (s.win_rate >= 40 ? "#ffd54f" : "#ff4455");
        const retColor = s.avg_return > 0 ? "#00c896" : "#ff4455";
        const bw = s.n
          ? `<span style="color:#00c896;">+${fmt(s.best,1)}%</span> / <span style="color:#ff4455;">${fmt(s.worst,1)}%</span>`
          : "—";
        return `<tr>
          <td><strong>${s.source}</strong></td>
          <td class="num">${s.n}</td>
          <td class="num" style="color:${s.n?wrColor:'#8b929d'};">${s.n?fmt(s.win_rate,1)+"%":"—"}</td>
          <td class="num" style="color:${s.n?retColor:'#8b929d'};">${s.n?(s.avg_return>=0?"+":"")+fmt(s.avg_return,2)+"%":"—"}</td>
          <td class="num" style="font-size:11px;">${bw}</td>
        </tr>`;
      }).join("");
    }
    const rTbody = document.querySelector("#cal-recent-tbl tbody");
    const recent = (r.recent_realized || []).slice().reverse();  // most recent first
    if (!recent.length) {
      rTbody.innerHTML = `<tr><td colspan="6" class="muted">no realized round-trips yet</td></tr>`;
    } else {
      rTbody.innerHTML = recent.slice(0, 12).map(t => {
        const ret = t.return_pct;
        const color = ret >= 0 ? "#00c896" : "#ff4455";
        const sign = ret >= 0 ? "+" : "";
        const buyTs = (t.buy_ts || "").replace("T", " ").slice(5, 16);
        const sellTs = (t.sell_ts || "").replace("T", " ").slice(5, 16);
        const conf = t.confidence != null ? fmt(t.confidence, 2) : "—";
        const reason = (t.reasoning_excerpt || "").replace(/</g, "&lt;");
        return `<tr>
          <td class="muted" style="font-size:11px;">${buyTs} → ${sellTs}</td>
          <td><strong>${t.ticker}</strong></td>
          <td class="num" style="color:${color};font-weight:bold;">${sign}${fmt(ret,2)}%</td>
          <td class="num">${conf}</td>
          <td>${t.source||"—"}</td>
          <td style="font-size:11px;color:#dde1e7;">${reason}</td>
        </tr>`;
      }).join("");
    }
  } catch (e) { console.error("calibration:", e); }
}

// ───────── Decision pipeline health (new 2026-05-15, agent 4) ─────────
async function refreshDecisionHealth() {
  try {
    const r = await fetch(API_PREFIX + "/api/decision-health").then(r => r.json());
    if (r.error) {
      document.getElementById("dh-reason").textContent = "error: " + r.error;
      return;
    }
    const vmap = {
      HEALTHY:  ["#1b5e20", "#a5d6a7"],
      DEGRADED: ["#b8860b", "#000000"],
      CRITICAL: ["#b71c1c", "#ffffff"],
      NO_DATA:  ["#1f2126", "#8b929d"],
    };
    const [bg, fg] = vmap[r.verdict] || vmap.NO_DATA;
    const vEl = document.getElementById("dh-verdict");
    vEl.textContent = r.verdict + (r.verdict_window ? ` (${r.verdict_window})` : "");
    vEl.style.background = bg;
    vEl.style.color = fg;
    document.getElementById("dh-reason").textContent = r.verdict_reason || "";

    const w = (r.windows && r.windows["24h"]) || {};
    document.getElementById("dh-total").textContent = w.total != null ? w.total : "—";
    const failEl = document.getElementById("dh-fail");
    failEl.textContent = w.parse_fail_pct != null ? fmt(w.parse_fail_pct, 0) + "%" : "—";
    failEl.style.color = (w.parse_fail_pct || 0) >= 50 ? "#ff4455"
                       : (w.parse_fail_pct || 0) >= 25 ? "#ffa726" : "#4caf50";
    document.getElementById("dh-fills").textContent =
      (w.filled != null ? w.filled : "—") + (w.fill_pct != null ? ` (${fmt(w.fill_pct,1)}%)` : "");
    const c = r.confidence || {};
    const trendArrow = {rising:" ↑", falling:" ↓", flat:""}[c.trend] || "";
    document.getElementById("dh-conf").textContent =
      c.avg != null ? fmt(c.avg, 2) + trendArrow : "—";
    const cad = r.cadence || {};
    document.getElementById("dh-lastfill").textContent =
      cad.hours_since_fill != null ? fmt(cad.hours_since_fill, 1) + "h" : "never";
    const sc = r.signal_count || {};
    document.getElementById("dh-sigs").textContent =
      sc.avg != null ? fmt(sc.avg, 1) : "—";

    // action mix bars
    const mixColors = {FILLED:"#4caf50", HOLD:"#5c6bc0", BLOCKED:"#ffa726",
                       NO_DECISION:"#ff4455", OTHER:"#8b929d"};
    const mix = r.action_mix || [];
    const mixEl = document.getElementById("dh-mix");
    if (!mix.length) {
      mixEl.innerHTML = '<div class="muted">no decisions yet</div>';
    } else {
      mixEl.innerHTML = mix.map(m => `
        <div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px;">
          <span style="width:96px;color:#dde1e7;">${m.category}</span>
          <div style="flex:1;background:#1f2126;border-radius:3px;height:14px;overflow:hidden;">
            <div style="width:${m.pct}%;height:100%;background:${mixColors[m.category]||"#8b929d"};"></div>
          </div>
          <span class="muted" style="width:96px;text-align:right;">${m.n} · ${fmt(m.pct,1)}%</span>
        </div>`).join("");
    }

    // recent decision tape
    const tape = r.recent || [];
    const tb = document.querySelector("#dh-tape tbody");
    if (!tape.length) {
      tb.innerHTML = `<tr><td colspan="5" class="muted">no cycles</td></tr>`;
    } else {
      tb.innerHTML = tape.map(d => {
        const col = mixColors[d.category] || "#8b929d";
        const t = d.timestamp ? d.timestamp.replace("T", " ").slice(5, 16) : "—";
        return `<tr>
          <td class="muted">${t}</td>
          <td><span style="color:${col};font-weight:bold;">${d.category}</span></td>
          <td>${(d.action || "—").replace(/</g,"&lt;")}</td>
          <td class="num">${d.confidence != null ? fmt(d.confidence,2) : "—"}</td>
          <td class="num">${d.signal_count != null ? d.signal_count : "—"}</td>
        </tr>`;
      }).join("");
    }
  } catch (e) { console.error("decision-health:", e); }
}

// ───────── Capital deployment & liquidity (new 2026-05-15, agent 4) ─────────
async function refreshLiquidity() {
  try {
    const r = await fetch(API_PREFIX + "/api/liquidity").then(r => r.json());
    if (r.error) { document.getElementById("liq-headline").textContent = "error: " + r.error; return; }
    const smap = {
      NO_DRY_POWDER: ["#b71c1c", "#ffffff"],
      DRY_POWDER_LOW:["#b8860b", "#000000"],
      CASH_HEAVY:    ["#1565c0", "#ffffff"],
      BALANCED:      ["#1b5e20", "#a5d6a7"],
      NO_DATA:       ["#1f2126", "#8b929d"],
    };
    const [bg, fg] = smap[r.status] || smap.NO_DATA;
    const sEl = document.getElementById("liq-status");
    sEl.textContent = (r.status || "—").replace(/_/g, " ");
    sEl.style.background = bg; sEl.style.color = fg;
    document.getElementById("liq-headline").textContent = r.headline || "";
    document.getElementById("liq-cash").textContent =
      dollar(r.cash) + (r.cash_pct != null ? ` (${fmt(r.cash_pct,1)}%)` : "");
    const dEl = document.getElementById("liq-deployed");
    dEl.textContent = r.deployed_pct != null ? fmt(r.deployed_pct,1) + "%" : "—";
    dEl.style.color = (r.deployed_pct || 0) >= 98 ? "#ff4455"
                    : (r.deployed_pct || 0) >= 90 ? "#ffa726" : "#dde1e7";
    document.getElementById("liq-npos").textContent =
      (r.n_positions != null ? r.n_positions : "—") +
      (r.n_losers != null ? ` · ${r.n_losers}↓` : "");
    document.getElementById("liq-top").textContent =
      r.top_weight_pct != null ? fmt(r.top_weight_pct,1) + "%" +
        (r.largest_position ? ` ${r.largest_position}` : "") : "—";
    const uEl = document.getElementById("liq-upl");
    uEl.textContent = r.unrealized_pl != null
      ? dollar(r.unrealized_pl) + (r.unrealized_pl_pct != null ? ` (${fmt(r.unrealized_pl_pct,1)}%)` : "")
      : "—";
    uEl.style.color = (r.unrealized_pl || 0) < 0 ? "#ff4455"
                    : (r.unrealized_pl || 0) > 0 ? "#4caf50" : "#dde1e7";
    document.getElementById("liq-entry").textContent =
      r.days_since_last_entry != null ? fmt(r.days_since_last_entry,1) + "d ago" : "—";
    const dep = Math.max(0, Math.min(100, r.deployed_pct || 0));
    document.getElementById("liq-bar").innerHTML =
      `<div style="width:${dep}%;background:#5c6bc0;height:100%;"></div>` +
      `<div style="width:${100-dep}%;background:#2e7d32;height:100%;"></div>`;
    document.getElementById("liq-bar-legend").textContent =
      `deployed ${fmt(dep,1)}%  ·  cash ${fmt(100-dep,1)}%` +
      (r.can_act_on_signal === false ? "  ·  ⚠ cannot act on a new BUY" : "");
    const fl = r.flags || [];
    document.getElementById("liq-flags").innerHTML = fl.length
      ? fl.map(f => `<div style="margin:3px 0;">• ${f.replace(/</g,"&lt;")}</div>`).join("")
      : '<span class="muted">no liquidity flags</span>';
  } catch (e) { console.error("liquidity:", e); }
}

// ───────── Decision failure forensics (new 2026-05-15, agent 4) ─────────
async function refreshDecisionForensics() {
  try {
    const r = await fetch(API_PREFIX + "/api/decision-forensics").then(r => r.json());
    if (r.error) { document.getElementById("df-reason").textContent = "error: " + r.error; return; }
    const vmap = {
      HEALTHY:  ["#1b5e20", "#a5d6a7"],
      DEGRADED: ["#b8860b", "#000000"],
      CRITICAL: ["#b71c1c", "#ffffff"],
      NO_DATA:  ["#1f2126", "#8b929d"],
    };
    const [bg, fg] = vmap[r.verdict] || vmap.NO_DATA;
    const vEl = document.getElementById("df-verdict");
    vEl.textContent = r.verdict + (r.verdict_window ? ` (${r.verdict_window})` : "");
    vEl.style.background = bg; vEl.style.color = fg;
    document.getElementById("df-reason").textContent = r.verdict_reason || "";
    document.getElementById("df-hint").textContent = r.hint || "";
    document.getElementById("df-nfail").textContent =
      (r.n_failures != null ? r.n_failures : "—") +
      (r.failure_rate_pct != null ? ` (${fmt(r.failure_rate_pct,0)}% all)` : "");
    const rEl = document.getElementById("df-rate");
    rEl.textContent = r.failure_rate_24h_pct != null ? fmt(r.failure_rate_24h_pct,0) + "%" : "—";
    rEl.style.color = (r.failure_rate_24h_pct || 0) >= 50 ? "#ff4455"
                    : (r.failure_rate_24h_pct || 0) >= 25 ? "#ffa726" : "#4caf50";
    document.getElementById("df-retry").textContent =
      r.retry_exhausted != null ? r.retry_exhausted : "—";
    document.getElementById("df-dom").textContent =
      (r.dominant_mode || "—").replace(/_/g, " ");
    const bm = r.by_market || {};
    document.getElementById("df-open").textContent =
      bm.open ? fmt(bm.open.fail_pct,0) + "%" : "—";
    document.getElementById("df-closed").textContent =
      bm.closed ? fmt(bm.closed.fail_pct,0) + "%" : "—";

    const modeColors = {
      TIMEOUT_EMPTY:"#ff4455", TRUNCATED:"#ff7043", NO_JSON:"#ab47bc",
      FENCED:"#ffa726", PROSE_WRAPPED:"#ffca28", MALFORMED_JSON:"#ef5350",
      EMPTY:"#8b929d", LEGACY_UNKNOWN:"#5c6bc0", OTHER:"#8b929d",
    };
    const mix = r.mode_mix || [];
    const mixEl = document.getElementById("df-mix");
    mixEl.innerHTML = mix.length ? mix.map(m => `
      <div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px;">
        <span style="width:128px;color:#dde1e7;">${m.mode.replace(/_/g," ")}</span>
        <div style="flex:1;background:#1f2126;border-radius:3px;height:14px;overflow:hidden;">
          <div style="width:${m.pct}%;height:100%;background:${modeColors[m.mode]||"#8b929d"};"></div>
        </div>
        <span class="muted" style="width:84px;text-align:right;">${m.n} · ${fmt(m.pct,0)}%</span>
      </div>`).join("") : '<div class="muted">no NO_DECISION cycles 🎉</div>';

    const hrs = r.hourly || [];
    const hEl = document.getElementById("df-hourly");
    if (!hrs.length) {
      hEl.innerHTML = '<div class="muted">no cycles in last 24h</div>';
    } else {
      hEl.innerHTML = hrs.map(h => {
        const ph = Math.max(4, Math.round((h.fail_pct||0) * 0.42));
        const col = (h.fail_pct||0) >= 50 ? "#ff4455" : (h.fail_pct||0) >= 25 ? "#ffa726" : "#4caf50";
        const lbl = (h.hour||"").slice(11,16);
        return `<div title="${lbl}  ${h.failures}/${h.total} failed (${fmt(h.fail_pct,0)}%)"
          style="flex:1;min-width:5px;height:${ph}px;background:${col};border-radius:2px 2px 0 0;"></div>`;
      }).join("");
    }

    const tape = r.recent_failures || [];
    const tb = document.querySelector("#df-tape tbody");
    tb.innerHTML = tape.length ? tape.map(d => {
      const t = d.timestamp ? d.timestamp.replace("T"," ").slice(5,16) : "—";
      const col = modeColors[d.mode] || "#8b929d";
      const ex = (d.excerpt || "—").replace(/</g,"&lt;").slice(0,200);
      return `<tr>
        <td class="muted">${t}</td>
        <td><span style="color:${col};font-weight:bold;">${d.mode.replace(/_/g," ")}</span></td>
        <td class="muted">${d.market_open ? "open" : "—"}</td>
        <td style="font-family:monospace;color:#aab;max-width:380px;word-break:break-all;">${ex}</td>
      </tr>`;
    }).join("") : `<tr><td colspan="4" class="muted">no failures recorded</td></tr>`;
  } catch (e) { console.error("decision-forensics:", e); }
}

// ───────── Decision drought drift (new 2026-05-16, agent 4) ─────────
async function refreshDecisionDrought() {
  try {
    const r = await fetch(API_PREFIX + "/api/decision-drought").then(r => r.json());
    if (r.error) { document.getElementById("dd-reason").textContent = "error: " + r.error; return; }
    const vmap = {
      OK:          ["#1b5e20", "#a5d6a7"],
      NEVER_TRADED:["#b8860b", "#000000"],
      STUCK:       ["#b8860b", "#000000"],
      BLEEDING:    ["#b71c1c", "#ffffff"],
      NO_DATA:     ["#1f2126", "#8b929d"],
    };
    const [bg, fg] = vmap[r.verdict] || vmap.NO_DATA;
    const vEl = document.getElementById("dd-verdict");
    vEl.textContent = r.verdict || "—";
    vEl.style.background = bg; vEl.style.color = fg;
    document.getElementById("dd-reason").textContent = r.verdict_reason || "";
    document.getElementById("dd-fills").textContent =
      (r.n_fills != null ? r.n_fills : "—") + " / " + (r.n_cycles != null ? r.n_cycles : "—");
    document.getElementById("dd-n").textContent = r.n_droughts != null ? r.n_droughts : "—";
    document.getElementById("dd-npar").textContent =
      r.n_paralysis_droughts != null ? r.n_paralysis_droughts : "—";
    const bEl = document.getElementById("dd-bleed");
    const bleed = r.involuntary_alpha_bleed_pct;
    bEl.textContent = bleed != null ? fmt(bleed, 2) + "%" : "—";
    bEl.style.color = (bleed || 0) <= -1.0 ? "#ff4455" : (bleed || 0) < 0 ? "#ffa726" : "#4caf50";

    const cur = r.current_drought;
    const cEl = document.getElementById("drought-current");
    if (!cur) {
      cEl.textContent = "no ongoing drought — last cycle was a fill";
      cEl.style.borderColor = "#1f2126";
    } else {
      const para = cur.kind === "PARALYSIS";
      cEl.style.borderColor = para ? "#ff4455" : "#2b3038";
      const a = cur.alpha_pct;
      cEl.innerHTML = `<b style="color:${para?'#ff6b6b':'#dde1e7'};">ONGOING ${cur.kind}</b> — `
        + `${fmt(cur.duration_hours,1)}h, ${cur.n_cycles} cycles `
        + `(${cur.n_no_decision} NO_DECISION / ${cur.n_hold} HOLD). `
        + `portfolio ${cur.portfolio_pct!=null?fmt(cur.portfolio_pct,2)+'%':'—'}, `
        + `S&P ${cur.spy_pct!=null?fmt(cur.spy_pct,2)+'%':'—'}, `
        + `<b style="color:${a!=null&&a<0?'#ff4455':'#4caf50'};">alpha ${a!=null?fmt(a,2)+'%':'—'}</b>`;
    }

    const kindCol = { PARALYSIS:"#ff4455", DELIBERATE_HOLD:"#4caf50", MIXED:"#ffa726" };
    const tape = r.droughts || [];
    const tb = document.querySelector("#dd-tape tbody");
    tb.innerHTML = tape.length ? tape.map(d => {
      const t = d.start ? d.start.replace("T"," ").slice(5,16) : "—";
      const a = d.alpha_pct;
      const acol = a == null ? "#8b929d" : a < 0 ? "#ff4455" : "#4caf50";
      return `<tr>
        <td class="muted">${t}${d.ongoing?' <span style="color:#ffd479;">●live</span>':''}</td>
        <td class="num">${d.duration_hours!=null?fmt(d.duration_hours,1):'—'}</td>
        <td class="num">${d.n_cycles}</td>
        <td><span style="color:${kindCol[d.kind]||'#8b929d'};font-weight:bold;">${d.kind.replace(/_/g," ")}</span></td>
        <td class="num">${fmt(d.no_decision_pct,0)}</td>
        <td class="num">${d.portfolio_pct!=null?fmt(d.portfolio_pct,2):'—'}</td>
        <td class="num">${d.spy_pct!=null?fmt(d.spy_pct,2):'—'}</td>
        <td class="num" style="color:${acol};font-weight:bold;">${a!=null?fmt(a,2):'—'}</td>
      </tr>`;
    }).join("") : `<tr><td colspan="8" class="muted">no multi-cycle droughts</td></tr>`;
  } catch (e) { console.error("decision-drought:", e); }
}

// ───────── News edge (new 2026-05-16, agent 4) ─────────
async function refreshNewsEdge() {
  try {
    const r = await fetch(API_PREFIX + "/api/news-edge").then(r => r.json());
    if (r.error) { document.getElementById("ne-reason").textContent = "error: " + r.error; return; }
    const vmap = {
      EDGE_CONFIRMED:    ["#1b5e20", "#a5d6a7"],
      WEAK_EDGE:         ["#b8860b", "#000000"],
      NO_EDGE:           ["#b71c1c", "#ffffff"],
      INSUFFICIENT_DATA: ["#1f2126", "#8b929d"],
      NO_DATA:           ["#1f2126", "#8b929d"],
      ERROR:             ["#b71c1c", "#ffffff"],
    };
    const [bg, fg] = vmap[r.verdict] || vmap.NO_DATA;
    const vEl = document.getElementById("ne-verdict");
    vEl.textContent = (r.verdict || "—").replace(/_/g," ");
    vEl.style.background = bg; vEl.style.color = fg;
    document.getElementById("ne-reason").textContent = r.verdict_reason || "";
    document.getElementById("ne-days").textContent =
      r.lookback_days != null ? r.lookback_days + "d" : "—";
    document.getElementById("ne-narts").textContent = r.n_articles != null ? r.n_articles : "—";
    document.getElementById("ne-nres").textContent = r.n_resolved != null ? r.n_resolved : "—";
    document.getElementById("ne-ntk").textContent =
      r.n_tickers_priced != null ? r.n_tickers_priced : "—";
    document.getElementById("ne-ref").textContent =
      r.reference_horizon != null ? r.reference_horizon + "d" : "—";

    const cell = (h) => {
      if (!h || h.mean_abnormal_pct == null) return '<span class="muted">—</span>';
      const v = h.mean_abnormal_pct;
      const col = v > 0 ? "#4caf50" : v < 0 ? "#ff4455" : "#8b929d";
      const raw = h.mean_raw_pct != null ? ` <span class="muted">(${fmt(h.mean_raw_pct,1)})</span>` : "";
      return `<span style="color:${col};font-weight:bold;">${fmt(v,2)}</span>${raw}`;
    };
    const refH = String(r.reference_horizon || 3);
    const bands = r.bands || [];
    const tb = document.querySelector("#ne-bands tbody");
    tb.innerHTML = bands.length ? bands.map(b => {
      const h = b.horizons || {};
      // n + hit% track the adaptive reference horizon (the one the verdict is
      // judged on) — not a hardcoded 3d, which would read 0 while the 1d cell
      // shows real numbers in early/low-history data.
      const hr = h[refH] || {};
      const nRef = hr.n || 0;
      const hit = hr.abnormal_hit_rate;
      return `<tr>
        <td><b>${b.band}</b></td>
        <td class="num">${nRef}</td>
        <td class="num">${cell(h["1"])}</td>
        <td class="num">${cell(h["3"])}</td>
        <td class="num">${cell(h["5"])}</td>
        <td class="num">${hit!=null?fmt(hit,0)+'%':'—'}</td>
      </tr>`;
    }).join("") : `<tr><td colspan="6" class="muted">no priced articles in window</td></tr>`;

    const u = r.by_urgency || {};
    const ur = (u.urgent||{})["3"] || {}, no = (u.normal||{})["3"] || {};
    const fmtAbn = (x) => x.mean_abnormal_pct != null
      ? `${fmt(x.mean_abnormal_pct,2)}% (n=${x.n||0})` : "—";
    document.getElementById("ne-urg").innerHTML =
      `urgent: <b style="color:#ffd479;">${fmtAbn(ur)}</b> &nbsp;·&nbsp; normal: <b>${fmtAbn(no)}</b>`;
  } catch (e) { console.error("news-edge:", e); }
}

// ───────── Scorer reliability + confidence intervals (new 2026-05-15, agent 4) ─────────
async function refreshScorerConfidence() {
  try {
    const r = await fetch(API_PREFIX + "/api/scorer-confidence").then(r => r.json());
    if (r.error) {
      document.getElementById("scrl-meta").textContent = "error: " + r.error;
      return;
    }
    document.getElementById("scrl-asof").textContent =
      r.as_of ? r.as_of.replace("T", " ").slice(0, 16) : "—";
    const o = r.overall;
    if (!o) {
      document.getElementById("scrl-meta").textContent =
        `scorer not ready — ${r.n_samples || 0} replay samples (need more outcomes)`;
      return;
    }
    document.getElementById("scrl-meta").textContent =
      `trained on n=${r.n_train} · replayed over ${r.n_samples} historical outcomes · ` +
      `residual = predicted − realized return`;
    const dirEl = document.getElementById("scrl-dir");
    dirEl.textContent = fmt(o.directional_accuracy_pct, 1) + "%";
    dirEl.style.color = o.directional_accuracy_pct >= 65 ? "#4caf50"
                      : o.directional_accuracy_pct >= 55 ? "#ffa726" : "#ff4455";
    document.getElementById("scrl-mae").textContent = "±" + fmt(o.mae, 2) + "%";
    document.getElementById("scrl-band").textContent =
      fmt(o.resid_p10, 1) + " … +" + fmt(o.resid_p90, 1);
    document.getElementById("scrl-n").textContent = r.n_samples;

    // held positions with empirical band
    const pos = r.positions || [];
    const pb = document.querySelector("#scrl-pos tbody");
    if (!pos.length) {
      pb.innerHTML = `<tr><td colspan="6" class="muted">no open stock positions</td></tr>`;
    } else {
      const trustColor = {high:"#4caf50", medium:"#ffa726", low:"#ff4455", none:"#8b929d"};
      pb.innerHTML = pos.map(p => {
        const v = p.pred_5d_return_pct;
        const iv = p.interval || {};
        const range = (iv.low != null && iv.high != null)
          ? `${iv.low >= 0 ? "+" : ""}${fmt(iv.low,1)}% … ${iv.high >= 0 ? "+" : ""}${fmt(iv.high,1)}%`
          : "—";
        return `<tr>
          <td><strong>${p.ticker}</strong></td>
          <td class="num" style="color:${scorerColor(v)};font-weight:bold;">${v == null ? "—" : (v>=0?"+":"") + fmt(v,2) + "%"}</td>
          <td class="num" style="color:#dde1e7;">${range}</td>
          <td>${verdictBadge(p.verdict)}</td>
          <td class="num">${iv.directional_accuracy_pct != null ? fmt(iv.directional_accuracy_pct,0) + "%" : "—"}</td>
          <td><span style="color:${trustColor[iv.reliability]||"#8b929d"};">${iv.reliability || "—"}</span></td>
        </tr>`;
      }).join("");
    }

    // calibration table
    const cb = document.querySelector("#scrl-cal tbody");
    const buckets = r.buckets || [];
    if (!buckets.length) {
      cb.innerHTML = `<tr><td colspan="6" class="muted">not enough samples</td></tr>`;
    } else {
      cb.innerHTML = buckets.map(b => `<tr>
        <td>${(b.pred_lo>=0?"+":"") + fmt(b.pred_lo,1)}% … ${(b.pred_hi>=0?"+":"") + fmt(b.pred_hi,1)}%</td>
        <td class="num">${b.n}</td>
        <td class="num" style="color:${scorerColor(b.mean_actual)};">${(b.mean_actual>=0?"+":"") + fmt(b.mean_actual,2)}%</td>
        <td class="num muted">${fmt(b.resid_p10,1)} / +${fmt(b.resid_p90,1)}</td>
        <td class="num">±${fmt(b.mae,1)}</td>
        <td class="num" style="color:${b.directional_accuracy_pct>=65?"#4caf50":b.directional_accuracy_pct>=55?"#ffa726":"#ff4455"};">${fmt(b.directional_accuracy_pct,0)}%</td>
      </tr>`).join("");
    }
  } catch (e) { console.error("scorer-confidence:", e); }
}

// ───────── Signal Integrity validation ─────────
async function refreshValidation() {
  try {
    const r = await fetch(API_PREFIX + "/api/validation").then(r => r.json());
    const results = (r && r.results) || [];
    const latest = results[results.length - 1];
    if (!latest) return;

    const pv = latest.permutation_test || {};
    const verdictColor = {
      SIGNIFICANT: "#00c896",
      INCONCLUSIVE: "#fbbf24",
      WORSE_THAN_RANDOM: "#ff4455",
      UNKNOWN: "#8b929d",
    }[pv.verdict] || "#8b929d";
    const verdictEl = document.getElementById("val-perm-verdict");
    if (verdictEl) {
      verdictEl.textContent = pv.verdict || "—";
      verdictEl.style.color = verdictColor;
    }
    const setText = (id, t) => { const el = document.getElementById(id); if (el) el.textContent = t; };
    setText("val-perm-pvalue", pv.p_value != null ? `p=${Number(pv.p_value).toFixed(3)}` : "p=—");
    setText("val-perm-zscore", pv.z_score != null ? `z=${Number(pv.z_score).toFixed(2)}` : "z=—");
    setText("val-perm-original",
      pv.original_return != null ? `Strategy: ${Number(pv.original_return).toFixed(1)}%` : "");
    setText("val-perm-shuffled",
      pv.permuted_mean != null
        ? `Shuffled mean: ${Number(pv.permuted_mean).toFixed(1)}%  (n=${pv.n_permutations || 0})`
        : "");

    const audit = latest.label_audit || {};
    const rate = audit.contamination_rate;
    const contamColor = rate == null ? "#8b929d"
      : rate > 0.5 ? "#ff4455"
      : rate > 0.2 ? "#fbbf24"
      : "#00c896";
    const contamEl = document.getElementById("val-contam-rate");
    if (contamEl) {
      contamEl.textContent = rate == null ? "—" : `${(rate * 100).toFixed(0)}%`;
      contamEl.style.color = contamColor;
    }
    setText("val-contam-detail",
      audit.total_articles != null
        ? `${audit.contaminated_count}/${audit.total_articles} articles · verdict: ${audit.verdict || "—"}`
        : "");

    setText("val-last-cycle", latest.cycle != null ? `cycle ${latest.cycle}` : "—");
    setText("val-last-window", latest.window || "");
    setText("val-last-when", latest.timestamp || "");
  } catch (e) {
    console.error("validation:", e);
  }
}

async function refreshDisagreement() {
  try {
    const r = await fetch(API_PREFIX + "/api/disagreement").then(r => r.json());
    if (r.error) {
      document.getElementById("dis-meta").textContent = "error: " + r.error;
      return;
    }
    document.getElementById("dis-asof").textContent =
      r.as_of ? r.as_of.replace("T", " ").slice(0, 16) : "—";
    const c = r.counts || {};
    const setC = (id, v, col) => {
      const e = document.getElementById(id);
      e.textContent = (v == null ? "—" : v);
      if (col) e.style.color = col;
    };
    setC("dis-high", c.HIGH, (c.HIGH > 0) ? "#ff4455" : "#8b929d");
    setC("dis-med", c.MEDIUM, (c.MEDIUM > 0) ? "#ffa726" : "#8b929d");
    setC("dis-aln", c.ALIGNED, "#4caf50");
    setC("dis-n", r.n_positions);
    if (!r.scorer_trained) {
      document.getElementById("dis-meta").textContent =
        "scorer not trained yet — needs ≥500 decision outcomes before it can disagree with Opus";
    } else if (!r.n_positions) {
      document.getElementById("dis-meta").textContent =
        "no open stock positions to compare";
    } else {
      const h = c.HIGH || 0;
      document.getElementById("dis-meta").innerHTML = h > 0
        ? `<span style="color:#ff4455;font-weight:bold;">${h} position(s) where Opus is overriding the ML safety net</span> — scorer says exit/trim while Opus is still long. Canonical "why is the book losing money?" check.`
        : "scorer and Opus are aligned on every held position";
    }
    const tb = document.querySelector("#dis-tbl tbody");
    const rows = r.rows || [];
    if (!rows.length) {
      tb.innerHTML = `<tr><td colspan="6" class="muted">—</td></tr>`;
    } else {
      const sevColor = { HIGH: "#ff4455", MEDIUM: "#ffa726", ALIGNED: "#4caf50" };
      const actCls = a => !a ? "hold"
        : a.startsWith("SELL") ? "sell"
        : a === "HOLD" ? "hold" : "buy";
      let html = rows.map(x => {
        const p = x.scorer_pred_5d_pct;
        const od = x.off_distribution;
        const predTxt = p == null ? "—"
          : ((p >= 0 ? "+" : "") + fmt(p, 1) + "%" + (od ? " *" : ""));
        return `<tr>
          <td><strong>${x.ticker}</strong></td>
          <td>${verdictBadge(x.scorer_verdict)}</td>
          <td class="num" style="color:${od ? '#8b929d' : scorerColor(p)};">${predTxt}</td>
          <td><span class="pill ${actCls(x.last_action)}">${x.last_action || '—'}</span></td>
          <td><span style="color:${sevColor[x.severity] || '#8b929d'};font-weight:bold;">${x.severity}</span></td>
          <td class="muted" style="font-size:12px;">${x.label || ''}</td>
        </tr>`;
      }).join("");
      if (rows.some(x => x.off_distribution)) {
        html += `<tr><td colspan="6" class="muted" style="font-size:11px;">* off-distribution — scorer extrapolated past its label support; this conflict is de-weighted, not a real fight</td></tr>`;
      }
      tb.innerHTML = html;
    }
  } catch (e) {
    console.error("disagreement:", e);
  }
}

// ───────── Behavioural edge + orphaned-endpoint panels (2026-05-16, agent 4) ─────────
// All three endpoints are absent on a paper-trader process that booted before
// their commit (trade-asymmetry is brand new; capital-paralysis &
// open-attribution shipped in c994cba). Degrade to an explicit "restart to
// apply" message instead of a silent console error — mirrors the /api/build-info
// stale-banner contract rather than looking broken.
async function fetchMaybeStale(path) {
  try {
    const resp = await fetch(API_PREFIX + path);
    if (!resp.ok) return { __unavailable: true, __code: resp.status };
    const ct = resp.headers.get("content-type") || "";
    if (!ct.includes("json")) return { __unavailable: true };
    return await resp.json();
  } catch (e) { return { __unavailable: true }; }
}
function markStale(badgeId, headlineId, what) {
  const b = document.getElementById(badgeId);
  if (b) { b.textContent = "UNAVAILABLE"; b.style.background = "#3a2a00"; b.style.color = "#ffd479"; }
  const h = document.getElementById(headlineId);
  if (h) h.textContent = what + " not on the running process — restart paper-trader to apply (see /api/build-info `stale`).";
}
const _sgn = v => (v == null ? "" : v >= 0 ? "+" : "");
const _plColor = v => (v == null ? "#8b929d" : v > 0 ? "#4caf50" : v < 0 ? "#ff4455" : "#dde1e7");

async function refreshTradeAsymmetry() {
  const r = await fetchMaybeStale("/api/trade-asymmetry");
  if (r.__unavailable) { markStale("ta-verdict", "ta-headline", "Behavioural-edge endpoint"); return; }
  if (r.error) { document.getElementById("ta-headline").textContent = "error: " + r.error; return; }
  const vmap = {
    PAYOFF_TRAP:       ["#b71c1c", "#ffffff"],
    DISPOSITION_BLEED: ["#b8860b", "#000000"],
    EDGE_POSITIVE:     ["#1b5e20", "#a5d6a7"],
    FLAT:              ["#1f2126", "#8b929d"],
  };
  const stateBadge = { STABLE: null, EMERGING: ["#3a2a00", "#ffd479"], NO_DATA: ["#1f2126", "#8b929d"] };
  const vEl = document.getElementById("ta-verdict");
  if (r.state === "STABLE" && r.verdict) {
    const [bg, fg] = vmap[r.verdict] || vmap.FLAT;
    vEl.textContent = r.verdict.replace(/_/g, " ");
    vEl.style.background = bg; vEl.style.color = fg;
  } else {
    const [bg, fg] = stateBadge[r.state] || stateBadge.NO_DATA;
    vEl.textContent = r.state;
    vEl.style.background = bg; vEl.style.color = fg;
  }
  document.getElementById("ta-headline").textContent = r.headline || "";
  const exp = document.getElementById("ta-exp");
  exp.textContent = r.expectancy_usd != null ? _sgn(r.expectancy_usd) + "$" + fmt(Math.abs(r.expectancy_usd)) : "—";
  exp.style.color = _plColor(r.expectancy_usd);
  document.getElementById("ta-payoff").textContent = r.payoff_ratio != null ? fmt(r.payoff_ratio) : "—";
  const wr = document.getElementById("ta-wr");
  wr.textContent = r.actual_win_rate_pct != null ? fmt(r.actual_win_rate_pct, 1) + "%" : "—";
  // Red when the actual win-rate cannot carry the payoff ratio (the trap).
  wr.style.color = (r.actual_win_rate_pct != null && r.breakeven_win_rate_pct != null
                    && r.actual_win_rate_pct < r.breakeven_win_rate_pct) ? "#ff4455" : "#dde1e7";
  document.getElementById("ta-be").textContent = r.breakeven_win_rate_pct != null ? fmt(r.breakeven_win_rate_pct, 1) + "%" : "—";
  const real = document.getElementById("ta-real");
  real.textContent = r.realized_pl_usd != null ? _sgn(r.realized_pl_usd) + "$" + fmt(Math.abs(r.realized_pl_usd)) : "—";
  real.style.color = _plColor(r.realized_pl_usd);
  document.getElementById("ta-n").textContent =
    r.n_round_trips + " (" + r.n_wins + "W/" + r.n_losses + "L" + (r.n_washes ? "/" + r.n_washes + "≈" : "") + ")";
  const aw = document.getElementById("ta-avgw");
  aw.textContent = r.avg_winner_usd != null ? "+$" + fmt(r.avg_winner_usd) : "—"; aw.style.color = "#4caf50";
  const al = document.getElementById("ta-avgl");
  al.textContent = r.avg_loser_usd != null ? "-$" + fmt(Math.abs(r.avg_loser_usd)) : "—"; al.style.color = "#ff4455";
  document.getElementById("ta-hold").textContent =
    (r.avg_winner_hold_days != null ? fmt(r.avg_winner_hold_days, 2) + "d" : "—") + " / " +
    (r.avg_loser_hold_days != null ? fmt(r.avg_loser_hold_days, 2) + "d" : "—");
  const dg = document.getElementById("ta-disp");
  dg.textContent = r.disposition_gap_days != null ? _sgn(r.disposition_gap_days) + fmt(r.disposition_gap_days, 2) + "d" : "—";
  // Negative gap = winners cut faster than losers = the disposition effect.
  dg.style.color = (r.disposition_gap_days != null && r.disposition_gap_days < 0) ? "#ff4455" : _plColor(r.disposition_gap_days);
}

// ───────── Loser autopsy + concentration honesty (new, agent 4) ─────────
// Same /api/build-info `stale` degrade contract as the behavioural cluster:
// a process that booted before these endpoints' commit 404s them → explicit
// "restart to apply" instead of a silent console error. Table bodies are
// built with DOM nodes + textContent (never innerHTML) so a verbatim
// entry-reason string can't inject markup.
const _LA_MODE_COLOR = {
  KNIFE_CATCH: ["#b71c1c", "#ffffff"],
  SLOW_BLEED:  ["#b8860b", "#000000"],
  STOPPED_OUT: ["#1f2126", "#dde1e7"],
  WHIPSAW:     ["#1f3a5f", "#9ec5ff"],
};
function _cell(text, cls) {
  const td = document.createElement("td");
  if (cls) td.className = cls;
  td.textContent = (text == null ? "—" : String(text));
  return td;
}
async function refreshLoserAutopsy() {
  const r = await fetchMaybeStale("/api/loser-autopsy");
  if (r.__unavailable) { markStale("lautopsy-state", "lautopsy-headline", "Loser-autopsy endpoint"); return; }
  if (r.error) { document.getElementById("lautopsy-headline").textContent = "error: " + r.error; return; }
  const sEl = document.getElementById("lautopsy-state");
  if (r.state === "STABLE" && r.verdict) {
    const [bg, fg] = _LA_MODE_COLOR[r.verdict] || _LA_MODE_COLOR.STOPPED_OUT;
    sEl.textContent = r.verdict.replace(/_/g, " ");
    sEl.style.background = bg; sEl.style.color = fg;
  } else {
    const sb = { EMERGING: ["#3a2a00", "#ffd479"], NO_DATA: ["#1f2126", "#8b929d"], NO_LOSSES: ["#1b5e20", "#a5d6a7"] };
    const [bg, fg] = sb[r.state] || sb.NO_DATA;
    sEl.textContent = r.state || "—";
    sEl.style.background = bg; sEl.style.color = fg;
  }
  document.getElementById("lautopsy-headline").textContent = r.headline || "";
  const tot = document.getElementById("lautopsy-total");
  tot.textContent = r.total_loss_usd != null ? _sgn(r.total_loss_usd) + "$" + fmt(Math.abs(r.total_loss_usd)) : "—";
  tot.style.color = _plColor(r.total_loss_usd);
  document.getElementById("lautopsy-n").textContent = r.n_losers != null ? (r.n_losers + " / " + r.n_round_trips + " RT") : "—";
  const avg = document.getElementById("lautopsy-avg");
  avg.textContent = r.avg_loss_usd != null ? _sgn(r.avg_loss_usd) + "$" + fmt(Math.abs(r.avg_loss_usd)) : "—";
  avg.style.color = _plColor(r.avg_loss_usd);
  document.getElementById("lautopsy-hold").textContent = r.median_loser_hold_days != null ? fmt(r.median_loser_hold_days, 2) + "d" : "—";
  document.getElementById("lautopsy-mode").textContent = r.dominant_failure_mode ? r.dominant_failure_mode.replace(/_/g, " ") : "—";
  const tb = document.querySelector("#lautopsy-tbl tbody");
  tb.replaceChildren();
  const cards = r.worst_losers || [];
  if (!cards.length) {
    const tr = document.createElement("tr");
    const td = _cell(r.state === "NO_LOSSES" ? "no losing round-trips" : "no data", "muted");
    td.colSpan = 6; tr.appendChild(td); tb.appendChild(tr);
  } else {
    for (const c of cards) {
      const tr = document.createElement("tr");
      tr.appendChild(_cell(c.ticker));
      const pl = _cell((c.pnl_usd >= 0 ? "+" : "") + "$" + fmt(Math.abs(c.pnl_usd)), "num");
      pl.style.color = _plColor(c.pnl_usd);
      tr.appendChild(pl);
      tr.appendChild(_cell(c.pnl_pct != null ? (c.pnl_pct >= 0 ? "+" : "") + fmt(c.pnl_pct, 1) + "%" : "—", "num"));
      tr.appendChild(_cell(c.hold_days != null ? fmt(c.hold_days, 2) : "—", "num"));
      tr.appendChild(_cell(c.failure_mode ? c.failure_mode.replace(/_/g, " ") : "—"));
      tr.appendChild(_cell(c.entry_reason || "—"));
      tb.appendChild(tr);
    }
  }
}

async function refreshCorrelation() {
  const r = await fetchMaybeStale("/api/correlation");
  if (r.__unavailable) { markStale("pcorr-state", "pcorr-headline", "Concentration-honesty endpoint"); return; }
  if (r.error) { document.getElementById("pcorr-headline").textContent = "error: " + r.error; return; }
  const vmap = {
    SINGLE_NAME_RISK: ["#b71c1c", "#ffffff"],
    CONCENTRATED:     ["#b8860b", "#000000"],
    MODERATE:         ["#1f3a5f", "#9ec5ff"],
    DIVERSIFIED:      ["#1b5e20", "#a5d6a7"],
  };
  const sEl = document.getElementById("pcorr-state");
  if (r.state === "OK" && r.verdict) {
    const [bg, fg] = vmap[r.verdict] || vmap.MODERATE;
    sEl.textContent = r.verdict.replace(/_/g, " ");
    sEl.style.background = bg; sEl.style.color = fg;
  } else {
    const sb = { INSUFFICIENT: ["#3a2a00", "#ffd479"], NO_DATA: ["#1f2126", "#8b929d"] };
    const [bg, fg] = sb[r.state] || sb.NO_DATA;
    sEl.textContent = r.state || "—";
    sEl.style.background = bg; sEl.style.color = fg;
  }
  document.getElementById("pcorr-headline").textContent = r.headline || "";
  const mr = document.getElementById("pcorr-meanrho");
  mr.textContent = r.mean_pairwise_corr != null ? (r.mean_pairwise_corr >= 0 ? "+" : "") + fmt(r.mean_pairwise_corr, 2) : "—";
  // High co-movement is the risk → red as ρ climbs.
  mr.style.color = (r.mean_pairwise_corr != null && r.mean_pairwise_corr >= 0.7) ? "#ff4455"
                 : (r.mean_pairwise_corr != null && r.mean_pairwise_corr >= 0.4) ? "#ffd479" : "#dde1e7";
  document.getElementById("pcorr-effbets").textContent = r.effective_independent_bets != null ? fmt(r.effective_independent_bets, 2) : "—";
  document.getElementById("pcorr-effnaive").textContent = r.effective_positions_naive != null ? fmt(r.effective_positions_naive, 2) : "—";
  document.getElementById("pcorr-topw").textContent = r.top_weight_pct != null ? fmt(r.top_weight_pct, 1) + "% " + (r.top_weight_ticker || "") : "—";
  document.getElementById("pcorr-maxpair").textContent = (r.max_pair && r.max_pair.tickers)
    ? r.max_pair.tickers.join("/") + " " + (r.max_pair.corr >= 0 ? "+" : "") + fmt(r.max_pair.corr, 2) : "—";
  const tb = document.querySelector("#pcorr-tbl tbody");
  tb.replaceChildren();
  const pairs = (r.pairs || []).slice().sort((a, b) => (b.corr ?? -2) - (a.corr ?? -2));
  if (!pairs.length) {
    const tr = document.createElement("tr");
    const td = _cell(r.state === "NO_DATA" ? "no stock positions" : "not enough overlapping history", "muted");
    td.colSpan = 2; tr.appendChild(td); tb.appendChild(tr);
  } else {
    for (const p of pairs) {
      const tr = document.createElement("tr");
      tr.appendChild(_cell(p.a + " / " + p.b));
      const c = _cell(p.corr != null ? (p.corr >= 0 ? "+" : "") + fmt(p.corr, 2) : "n/a", "num");
      c.style.color = (p.corr != null && p.corr >= 0.7) ? "#ff4455" : (p.corr != null && p.corr >= 0.4) ? "#ffd479" : "#dde1e7";
      tr.appendChild(c);
      tb.appendChild(tr);
    }
  }
}

// ───────── Overtrading/churn + thesis-drift (new 2026-05-16, agent 4) ─────────
// Same stale-degrade contract as the behavioural cluster above: a process
// that booted before these endpoints' commit 404s them → explicit
// "restart to apply" instead of a silent failure.
async function refreshChurn() {
  const r = await fetchMaybeStale("/api/churn");
  if (r.__unavailable) { markStale("churn-state", "churn-headline", "Overtrading/churn endpoint"); return; }
  if (r.error) { document.getElementById("churn-headline").textContent = "error: " + r.error; return; }
  const vmap = {
    CHURNING:        ["#b71c1c", "#ffffff"],
    ACTIVE_TURNOVER: ["#b8860b", "#000000"],
    BUY_AND_HOLD:    ["#1b5e20", "#a5d6a7"],
  };
  const stateBadge = { STABLE: null, EMERGING: ["#3a2a00", "#ffd479"], NO_DATA: ["#1f2126", "#8b929d"] };
  const sEl = document.getElementById("churn-state");
  if (r.state === "STABLE" && r.verdict) {
    const [bg, fg] = vmap[r.verdict] || stateBadge.NO_DATA;
    sEl.textContent = r.verdict.replace(/_/g, " ");
    sEl.style.background = bg; sEl.style.color = fg;
  } else {
    const [bg, fg] = stateBadge[r.state] || stateBadge.NO_DATA;
    sEl.textContent = r.state; sEl.style.background = bg; sEl.style.color = fg;
  }
  document.getElementById("churn-headline").textContent = r.headline || "";
  const re = document.getElementById("churn-reentry");
  re.textContent = r.reentry_rate_pct != null ? r.n_reentries + " (" + fmt(r.reentry_rate_pct, 1) + "%)" : "—";
  re.style.color = (r.reentry_rate_pct != null && r.reentry_rate_pct >= 25) ? "#ff4455" : "#dde1e7";
  document.getElementById("churn-rtpd").textContent = r.round_trips_per_day != null ? fmt(r.round_trips_per_day, 2) : "—";
  document.getElementById("churn-hold").textContent = r.median_hold_days != null ? fmt(r.median_hold_days, 2) + "d" : "—";
  document.getElementById("churn-subday").textContent = r.sub_day_trip_pct != null ? fmt(r.sub_day_trip_pct, 1) + "%" : "—";
  const lc = document.getElementById("churn-lossconc");
  lc.textContent = r.churn_loss_concentration_pct != null ? fmt(r.churn_loss_concentration_pct, 1) + "%" : "—";
  lc.style.color = (r.churn_loss_concentration_pct != null && r.churn_loss_concentration_pct >= 50) ? "#ff4455" : "#dde1e7";
  const tb = document.querySelector("#churn-events tbody");
  const evs = r.reentry_events || [];
  if (!evs.length) {
    tb.innerHTML = '<tr><td colspan="4" class="muted">no fast same-name re-entries — clean turnover</td></tr>';
  } else {
    tb.innerHTML = evs.map(e => {
      const p = e.prior_pnl_usd;
      const pc = p == null ? "#8b929d" : (p > 0 ? "#4caf50" : p < 0 ? "#ff4455" : "#dde1e7");
      return '<tr><td>' + e.ticker + '</td><td class="num">' + fmt(e.gap_days, 2) +
        '</td><td class="num" style="color:' + pc + '">' +
        (p == null ? "—" : _sgn(p) + "$" + fmt(Math.abs(p))) +
        '</td><td>' + (e.prior_exit_ts || "").slice(0, 10) + ' → ' +
        (e.next_entry_ts || "").slice(0, 10) + '</td></tr>';
    }).join("");
  }
}

let _sessWindow = 360;
function setSessWindow(m) {
  _sessWindow = m;
  [60, 360, 1440].forEach(k => {
    const el = document.getElementById("sess-w-" + k);
    if (el) el.classList.toggle("active", k === m);
  });
  refreshSessionDelta();
}
async function refreshSessionDelta() {
  const r = await fetchMaybeStale("/api/session-delta?minutes=" + _sessWindow);
  if (r.__unavailable) { markStale("sess-state", "sess-headline", "Session-delta endpoint"); return; }
  if (r.error) { document.getElementById("sess-headline").textContent = "error: " + r.error; return; }
  const smap = {
    ACTIVE:  ["#0d3b4f", "#7fdbff"],
    QUIET:   ["#1b5e20", "#a5d6a7"],
    NO_DATA: ["#1f2126", "#8b929d"],
  };
  const sEl = document.getElementById("sess-state");
  const [bg, fg] = smap[r.state] || smap.NO_DATA;
  sEl.textContent = r.state + (r.n_events ? " · " + r.n_events : "");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("sess-headline").textContent = r.headline || "";
  const tb = document.querySelector("#sess-events tbody");
  const evs = r.events || [];
  if (!evs.length) {
    tb.innerHTML = '<tr><td colspan="3" class="muted">nothing material in this window</td></tr>';
    return;
  }
  const sevC = { HIGH: "#ff4455", MED: "#ffb74d", LOW: "#8b929d" };
  const kindLabel = {
    TRADE: "⇄ FILL", POSITION_CLOSED: "✓ CLOSED", EQUITY_MOVE: "$ EQUITY",
    DRAWDOWN_LOW: "▼ DRAWDOWN", INACTION: "… IDLE",
  };
  tb.innerHTML = evs.map(e => {
    const c = sevC[e.severity] || "#8b929d";
    const when = (e.ts || "").slice(11, 16);
    const lbl = kindLabel[e.kind] || e.kind;
    const txt = (e.summary || "").replace(/</g, "&lt;");
    return '<tr><td class="muted">' + when + '</td><td style="color:' + c +
      ';white-space:nowrap;">' + lbl + '</td><td>' + txt + '</td></tr>';
  }).join("");
}

async function refreshThesisDrift() {
  const r = await fetchMaybeStale("/api/thesis-drift");
  if (r.__unavailable) { markStale("tdrift-state", "tdrift-headline", "Thesis-drift endpoint"); return; }
  if (r.error) { document.getElementById("tdrift-headline").textContent = "error: " + r.error; return; }
  const sEl = document.getElementById("tdrift-state");
  const c = r.counts || {};
  if (r.state === "NO_DATA") {
    sEl.textContent = "NO DATA"; sEl.style.background = "#1f2126"; sEl.style.color = "#8b929d";
  } else if ((c.BROKEN || 0) > 0) {
    sEl.textContent = c.BROKEN + " BROKEN"; sEl.style.background = "#b71c1c"; sEl.style.color = "#fff";
  } else if ((c.WEAKENING || 0) > 0) {
    sEl.textContent = c.WEAKENING + " WEAKENING"; sEl.style.background = "#b8860b"; sEl.style.color = "#000";
  } else {
    sEl.textContent = "ALL INTACT"; sEl.style.background = "#1b5e20"; sEl.style.color = "#a5d6a7";
  }
  document.getElementById("tdrift-headline").textContent = r.headline || "";
  const tb = document.querySelector("#tdrift-rows tbody");
  const ps = r.positions || [];
  if (!ps.length) {
    tb.innerHTML = '<tr><td colspan="5" class="muted">no open positions</td></tr>';
    return;
  }
  const hmap = { BROKEN: ["#b71c1c", "#fff"], WEAKENING: ["#b8860b", "#000"], INTACT: ["#1b5e20", "#a5d6a7"] };
  tb.innerHTML = ps.map(p => {
    const [hb, hf] = hmap[p.health] || ["#1f2126", "#8b929d"];
    const reason = p.entry_reason || "—";
    const reasonShort = reason.length > 90 ? reason.slice(0, 90) + "…" : reason;
    const drift = (p.drift_reasons || []).join("; ");
    const plc = p.pl_pct == null ? "#8b929d" : (p.pl_pct > 0 ? "#4caf50" : p.pl_pct < 0 ? "#ff4455" : "#dde1e7");
    return '<tr><td>' + p.ticker + '</td>' +
      '<td><span style="padding:2px 7px;border-radius:3px;font-size:11px;background:' + hb + ';color:' + hf + '">' + p.health + '</span></td>' +
      '<td class="num" style="color:' + plc + '">' + (p.pl_pct == null ? "—" : _sgn(p.pl_pct) + fmt(p.pl_pct, 2) + "%") + '</td>' +
      '<td class="num">' + (p.days_held == null ? "—" : fmt(p.days_held, 1)) + '</td>' +
      '<td title="' + reason.replace(/"/g, "&quot;") + '"><span class="muted">' + reasonShort + '</span><br><span style="color:#dde1e7;">↳ ' + (drift || "—") + '</span></td></tr>';
  }).join("");
}

async function refreshGlobalStale() {
  try {
    const r = await fetch(API_PREFIX + "/api/build-info").then(r => r.json());
    const el = document.getElementById("global-stale-banner");
    const tx = document.getElementById("global-stale-text");
    if (r && (r.stale || (r.behind && r.behind > 0))) {
      tx.textContent = "⚠ Paper-trader is running stale code — booted " +
        (r.boot_sha || "?") + ", HEAD is " + (r.head_sha || "?") +
        (r.behind ? " (" + r.behind + " commit" + (r.behind === 1 ? "" : "s") + " behind)" : "") +
        ". Committed fixes (incl. the self-review mirror & newest endpoints) are NOT applied until paper-trader is restarted.";
      el.style.display = "block";
    } else {
      el.style.display = "none";
    }
  } catch (e) { /* build-info unreachable — leave banner hidden */ }
}

async function refreshCapitalParalysis() {
  const r = await fetchMaybeStale("/api/capital-paralysis");
  if (r.__unavailable) { markStale("cp-state", "cp-headline", "Capital-paralysis endpoint"); return; }
  if (r.error) { document.getElementById("cp-headline").textContent = "error: " + r.error; return; }
  const smap = {
    PINNED:  ["#b71c1c", "#ffffff"],
    EMPTY:   ["#b71c1c", "#ffffff"],
    FREE:    ["#1b5e20", "#a5d6a7"],
    NO_DATA: ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.state] || smap.NO_DATA;
  const sEl = document.getElementById("cp-state");
  sEl.textContent = r.state || "—"; sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("cp-headline").textContent = r.headline || "";
  document.getElementById("cp-cash").textContent =
    r.cash != null ? "$" + fmt(r.cash) + " (" + fmt(r.cash_pct, 1) + "%)" : "—";
  document.getElementById("cp-dep").textContent = r.deployed_pct != null ? fmt(r.deployed_pct, 1) + "%" : "—";
  const ca = document.getElementById("cp-canact");
  ca.textContent = r.can_act_on_signal ? "yes" : "no";
  ca.style.color = r.can_act_on_signal ? "#4caf50" : "#ff4455";
  document.getElementById("cp-stuck").textContent = r.cycles_since_last_fill != null ? r.cycles_since_last_fill : "—";
  const bleed = r.paralysis ? r.paralysis.involuntary_alpha_bleed_pct : null;
  const bEl = document.getElementById("cp-bleed");
  bEl.textContent = bleed != null ? fmt(bleed, 2) + "%" : "—";
  bEl.style.color = (bleed || 0) <= -1.0 ? "#ff4455" : (bleed || 0) < 0 ? "#ffa726" : "#4caf50";
  const lad = r.unlock_ladder || [];
  const recT = r.recommended_unlock ? r.recommended_unlock.ticker : null;
  const tb = document.querySelector("#cp-ladder tbody");
  tb.innerHTML = lad.length ? lad.map(p => {
    const rec = p.ticker === recT;
    const plc = _plColor(p.pl_pct);
    return `<tr${rec ? ' style="background:#15240f;"' : ''}>
      <td>${rec ? '★ ' : ''}${p.ticker}<span class="muted">${p.type && p.type !== 'stock' ? ' ' + p.type : ''}</span></td>
      <td class="num">${fmt(p.weight_pct, 1)}</td>
      <td class="num" style="color:${plc};">${_sgn(p.pl_pct)}${fmt(p.pl_pct, 1)}</td>
      <td class="num">$${fmt(p.frees_usd)}</td>
      <td class="num">$${fmt(p.cash_if_sold_alone)}</td>
      <td style="color:${p.restores_action_alone ? '#4caf50' : '#8b929d'};font-weight:${p.restores_action_alone ? 'bold' : 'normal'};">${p.restores_action_alone ? 'unlocks ✓' : '—'}</td>
    </tr>`;
  }).join("") : `<tr><td colspan="6" class="muted">no open positions to ladder</td></tr>`;
}

async function refreshOpenAttribution() {
  const r = await fetchMaybeStale("/api/open-attribution");
  if (r.__unavailable) { markStale("oa-status", "oa-headline", "Open-attribution endpoint"); return; }
  if (r.error) { document.getElementById("oa-headline").textContent = "error: " + r.error; return; }
  const smap = {
    SELECTION_ADDING: ["#1b5e20", "#a5d6a7"],
    SELECTION_DRAG:   ["#b71c1c", "#ffffff"],
    FLAT_VS_SPY:      ["#1f2126", "#8b929d"],
    NO_BENCHMARK:     ["#3a2a00", "#ffd479"],
    NO_DATA:          ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.status] || smap.NO_DATA;
  const sEl = document.getElementById("oa-status");
  sEl.textContent = (r.status || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("oa-headline").textContent = r.headline || "";
  const al = document.getElementById("oa-alpha");
  al.textContent = r.book_open_alpha_pct != null ? _sgn(r.book_open_alpha_pct) + fmt(r.book_open_alpha_pct, 2) + "%" : "—";
  al.style.color = _plColor(r.book_open_alpha_pct);
  const ex = document.getElementById("oa-excess");
  ex.textContent = r.net_excess_usd != null ? _sgn(r.net_excess_usd) + "$" + fmt(Math.abs(r.net_excess_usd)) : "—";
  ex.style.color = _plColor(r.net_excess_usd);
  const ur = document.getElementById("oa-unreal");
  ur.textContent = r.total_unrealized_usd != null ? _sgn(r.total_unrealized_usd) + "$" + fmt(Math.abs(r.total_unrealized_usd)) : "—";
  ur.style.color = _plColor(r.total_unrealized_usd);
  document.getElementById("oa-spyeq").textContent =
    r.total_spy_equivalent_usd != null ? _sgn(r.total_spy_equivalent_usd) + "$" + fmt(Math.abs(r.total_spy_equivalent_usd)) : "—";
  document.getElementById("oa-n").textContent =
    (r.n_anchored != null ? r.n_anchored : "—") + (r.n_positions != null ? " / " + r.n_positions : "");
  const rows = r.positions || [];
  const tb = document.querySelector("#oa-rows tbody");
  tb.innerHTML = rows.length ? rows.map(p => {
    if (!p.anchored) {
      return `<tr><td>${p.ticker}</td><td class="num">${fmt(p.position_return_pct, 2)}</td>
        <td class="num muted" colspan="3">unanchored — no SPY level at/after entry</td></tr>`;
    }
    return `<tr>
      <td>${p.ticker}</td>
      <td class="num" style="color:${_plColor(p.position_return_pct)};">${_sgn(p.position_return_pct)}${fmt(p.position_return_pct, 2)}</td>
      <td class="num" style="color:${_plColor(p.spy_return_pct)};">${_sgn(p.spy_return_pct)}${fmt(p.spy_return_pct, 2)}</td>
      <td class="num" style="color:${_plColor(p.alpha_pct)};font-weight:bold;">${_sgn(p.alpha_pct)}${fmt(p.alpha_pct, 2)}</td>
      <td class="num" style="color:${_plColor(p.excess_usd)};">${_sgn(p.excess_usd)}$${fmt(Math.abs(p.excess_usd))}</td>
    </tr>`;
  }).join("") : `<tr><td colspan="5" class="muted">no anchorable open stock positions</td></tr>`;
}

async function refreshFeedHealth() {
  const r = await fetchMaybeStale("/api/feed-health");
  if (r.__unavailable) { markStale("fh-state", "fh-headline", "Signal-feed-health endpoint"); return; }
  if (r.error) { document.getElementById("fh-headline").textContent = "error: " + r.error; return; }
  const smap = {
    BLIND:      ["#b71c1c", "#ffffff"],
    STALE_FEED: ["#b8860b", "#000000"],
    HEALTHY:    ["#1b5e20", "#a5d6a7"],
    NO_DATA:    ["#1f2126", "#8b929d"],
    ERROR:      ["#3a2a00", "#ffd479"],
  };
  const [bg, fg] = smap[r.verdict] || smap.NO_DATA;
  const sEl = document.getElementById("fh-state");
  sEl.textContent = (r.verdict || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("fh-headline").textContent =
    (r.restart_recommended ? "⚠ RESTART RECOMMENDED — " : "") + (r.headline || "");
  const st = document.getElementById("fh-streak");
  st.textContent = r.blind_streak != null
    ? r.blind_streak + " / " + (r.n_decisions != null ? r.n_decisions : "—")
    : "—";
  st.style.color = (r.blind_streak || 0) >= (r.blind_streak_min || 3) ? "#ff4455"
                  : (r.blind_streak || 0) > 0 ? "#ffa726" : "#4caf50";
  const ag = document.getElementById("fh-age");
  ag.textContent = r.resolved_newest_age_h != null
    ? fmt(r.resolved_newest_age_h, 1) + "h" : "never";
  ag.style.color = (r.resolved_newest_age_h == null
                    || r.resolved_newest_age_h >= (r.stale_hours || 6))
                   ? "#ff4455" : "#4caf50";
  document.getElementById("fh-live").textContent =
    (r.resolved_live_2h != null ? r.resolved_live_2h : "—") + " / "
    + (r.resolved_live_24h != null ? r.resolved_live_24h : "—");
  const sp = document.getElementById("fh-split");
  sp.textContent = r.split_brain ? "YES" : "no";
  sp.style.color = r.split_brain ? "#ff4455" : "#8b929d";
  document.getElementById("fh-path").textContent =
    r.resolved_path
      ? ("trader reads " + r.resolved_path
         + (r.split_brain && r.fresher_path
            ? "  ·  fresher copy: " + r.fresher_path
              + " (" + fmt(r.fresher_age_h, 1) + "h)"
            : ""))
      : "no resolved article DB";
}

async function refreshDecisionReliability() {
  const r = await fetchMaybeStale("/api/decision-reliability");
  if (r.__unavailable) { markStale("dr-state", "dr-headline", "Decision-reliability endpoint"); return; }
  if (r.error) { document.getElementById("dr-headline").textContent = "error: " + r.error; return; }
  const smap = {
    CRITICAL:               ["#b71c1c", "#ffffff"],
    DEGRADED:               ["#b8860b", "#000000"],
    HEALTHY:                ["#1b5e20", "#a5d6a7"],
    STALE_LEGACY_DOMINATED: ["#3a2a00", "#ffd479"],
    INSUFFICIENT:           ["#1f2126", "#8b929d"],
    NO_DATA:                ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.state] || smap.NO_DATA;
  const sEl = document.getElementById("dr-state");
  sEl.textContent = (r.state || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("dr-headline").textContent =
    (r.restart_recommended ? "⚠ RESTART RECOMMENDED — " : "") + (r.headline || "");
  const cur = document.getElementById("dr-cur");
  cur.textContent = r.current_failure_rate_pct != null
    ? fmt(r.current_failure_rate_pct, 1) + "% (" + (r.current_failures || 0) + "/" + (r.current_total || 0) + ")"
    : "—";
  cur.style.color = (r.current_failure_rate_pct >= 50) ? "#ff4455"
                    : (r.current_failure_rate_pct >= 25) ? "#ffa726" : "#4caf50";
  document.getElementById("dr-head").textContent =
    r.headline_failure_rate_pct != null ? fmt(r.headline_failure_rate_pct, 1) + "%" : "—";
  document.getElementById("dr-n").textContent =
    (r.current_total != null ? r.current_total : "—") + " / " + (r.n_decisions != null ? r.n_decisions : "—");
  const lg = document.getElementById("dr-legacy");
  lg.textContent = r.legacy_failures != null
    ? r.legacy_failures + " (" + fmt(r.legacy_share_pct, 1) + "%)" : "—";
  lg.style.color = (r.legacy_share_pct || 0) >= 50 ? "#ffa726" : "#8b929d";
  const dd = document.getElementById("dr-dead");
  dd.textContent = r.dead_cycles_per_day != null ? fmt(r.dead_cycles_per_day, 2) : "—";
  dd.style.color = (r.dead_cycles_per_day || 0) > 0 ? "#ff4455" : "#8b929d";
  const mm = (r.current_mode_mix || []).slice(0, 3)
    .map(m => m.mode.replace(/_/g, " ") + " " + m.n + " (" + fmt(m.pct, 0) + "%)").join(" · ");
  document.getElementById("dr-mode").textContent =
    mm ? "current failure modes: " + mm
       : (r.regime_boundary ? "regime boundary: " + r.regime_boundary
                            : "no current-regime failures recorded");
}

async function refreshFundedSuggestions() {
  const r = await fetchMaybeStale("/api/funded-suggestions");
  if (r.__unavailable) { markStale("fund-state", "fund-headline", "Funded-suggestions endpoint"); return; }
  if (r.error) { document.getElementById("fund-headline").textContent = "error: " + r.error; return; }
  const smap = {
    FREE:    ["#1b5e20", "#a5d6a7"],
    PINNED:  ["#b71c1c", "#ffffff"],
    EMPTY:   ["#b71c1c", "#ffffff"],
    NO_DATA: ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.state] || smap.NO_DATA;
  const sEl = document.getElementById("fund-state");
  sEl.textContent = r.state || "—"; sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("fund-headline").textContent = r.headline || "";
  document.getElementById("fund-n").textContent = r.n_actionable != null ? r.n_actionable : "—";
  const fF = document.getElementById("fund-funded");
  fF.textContent = r.n_funded != null ? r.n_funded : "—";
  fF.style.color = (r.n_funded || 0) > 0 ? "#4caf50" : "#8b929d";
  const fU = document.getElementById("fund-unlock");
  fU.textContent = r.n_unlockable != null ? r.n_unlockable : "—";
  fU.style.color = (r.n_unlockable || 0) > 0 ? "#ffa726" : "#8b929d";
  const fX = document.getElementById("fund-unfund");
  fX.textContent = r.n_unfundable != null ? r.n_unfundable : "—";
  fX.style.color = (r.n_unfundable || 0) > 0 ? "#ff4455" : "#8b929d";
  const pr = r.recommended_pairing;
  document.getElementById("fund-pair").textContent =
    pr ? ("sell " + pr.sell + " → buy " + pr.buy) : "—";
  const fmap = { FUNDED: "#4caf50", UNLOCKABLE: "#ffa726", UNFUNDABLE: "#ff4455" };
  const rows = r.ideas || [];
  const tb = document.querySelector("#fund-rows tbody");
  tb.innerHTML = rows.length ? rows.map(i => `<tr>
      <td>${i.action} ${i.ticker}</td>
      <td class="num">${fmt(i.conviction, 2)}</td>
      <td class="num">$${fmt(i.suggested_notional_usd)}</td>
      <td style="color:${fmap[i.fundability] || '#8b929d'};font-weight:bold;">${i.fundability}${i.enough === false && i.fundability === 'UNFUNDABLE' ? '' : ''}</td>
      <td>${(i.funded_by && i.funded_by.length) ? i.funded_by.join(' + ') : '<span class="muted">—</span>'}</td>
      <td class="num">${i.frees_usd ? '$' + fmt(i.frees_usd) : '—'}</td>
    </tr>`).join("") : `<tr><td colspan="6" class="muted">no actionable BUY/ADD ideas</td></tr>`;
}

// ───────── Signal follow-through — does it use its own news edge? (new 2026-05-16, agent 4) ─────────
async function refreshSignalFollowThrough() {
  const r = await fetchMaybeStale("/api/signal-followthrough");
  if (r.__unavailable) { markStale("sft-state", "sft-headline", "Signal-follow-through endpoint"); return; }
  if (r.error) { document.getElementById("sft-headline").textContent = "error: " + r.error; return; }
  const smap = {
    EXPLOITING_SIGNALS: ["#1b5e20", "#a5d6a7"],
    NEUTRAL_USE:        ["#b8860b", "#000000"],
    LOW_ACTIVITY:       ["#3a2a00", "#ffd479"],
    IGNORING_FEED:      ["#b71c1c", "#ffffff"],
    MISUSING_SIGNALS:   ["#b71c1c", "#ffffff"],
    INSUFFICIENT:       ["#1f2126", "#8b929d"],
    NO_DATA:            ["#1f2126", "#8b929d"],
    ERROR:              ["#b71c1c", "#ffffff"],
  };
  const [bg, fg] = smap[r.verdict] || smap.NO_DATA;
  const sEl = document.getElementById("sft-state");
  sEl.textContent = (r.verdict || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("sft-headline").textContent = r.verdict_reason || "";
  const ft = document.getElementById("sft-ft");
  ft.textContent = r.follow_through_rate_pct != null ? fmt(r.follow_through_rate_pct, 1) + "%" : "—";
  ft.style.color = r.follow_through_rate_pct == null ? "#8b929d"
                 : r.follow_through_rate_pct < 5 ? "#ff4455"
                 : r.follow_through_rate_pct < 25 ? "#ffa726" : "#4caf50";
  document.getElementById("sft-ai").textContent =
    (r.n_acted != null ? r.n_acted : "—") + " / " + (r.n_ignored != null ? r.n_ignored : "—");
  const ed = document.getElementById("sft-edge");
  ed.textContent = r.selection_edge_pct != null ? _sgn(r.selection_edge_pct) + fmt(r.selection_edge_pct, 2) + " pp" : "—";
  ed.style.color = _plColor(r.selection_edge_pct);
  const ref = String(r.reference_horizon || 3);
  const acted = (r.acted || {})[ref] || {}, ign = (r.ignored || {})[ref] || {};
  const aEl = document.getElementById("sft-acted");
  aEl.textContent = acted.mean_abnormal_pct != null ? _sgn(acted.mean_abnormal_pct) + fmt(acted.mean_abnormal_pct, 2) + "%" : "—";
  aEl.style.color = _plColor(acted.mean_abnormal_pct);
  const iEl = document.getElementById("sft-ign");
  iEl.textContent = ign.mean_abnormal_pct != null ? _sgn(ign.mean_abnormal_pct) + fmt(ign.mean_abnormal_pct, 2) + "%" : "—";
  iEl.style.color = _plColor(ign.mean_abnormal_pct);
  document.getElementById("sft-n").textContent =
    (r.n_resolved != null ? r.n_resolved : "—") + " / " + (r.n_signals != null ? r.n_signals : "—");
  document.getElementById("sft-meta").textContent =
    "ref " + (r.reference_horizon != null ? r.reference_horizon + "d" : "—")
    + " · " + (r.n_decisions != null ? r.n_decisions : "—") + " decisions"
    + " · " + (r.n_tickers_priced != null ? r.n_tickers_priced : "—") + " tickers priced"
    + (r.spy_adjusted ? " · SPY-adjusted" : " · raw only")
    + (r.lookback_days != null ? " · " + r.lookback_days + "d lookback" : "");
}

// ───────── News source edge — which collector is worth trusting? (new 2026-05-16, agent 4) ─────────
async function refreshSourceEdge() {
  const r = await fetchMaybeStale("/api/source-edge");
  if (r.__unavailable) { markStale("se-state", "se-headline", "Source-edge endpoint"); return; }
  if (r.error) { document.getElementById("se-headline").textContent = "error: " + r.error; return; }
  const smap = {
    EDGE_FOUND:        ["#1b5e20", "#a5d6a7"],
    NO_EDGE:           ["#b71c1c", "#ffffff"],
    INSUFFICIENT_DATA: ["#3a2a00", "#ffd479"],
    NO_DATA:           ["#1f2126", "#8b929d"],
    ERROR:             ["#b71c1c", "#ffffff"],
  };
  const [bg, fg] = smap[r.verdict] || smap.NO_DATA;
  const sEl = document.getElementById("se-state");
  sEl.textContent = (r.verdict || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("se-headline").textContent = r.verdict_reason || "";
  const ref = String(r.reference_horizon || 3);
  const vmap = {
    EXPLOITABLE:  "#4caf50", WEAK: "#ffa726",
    NEGATIVE:     "#ff4455", INSUFFICIENT: "#8b929d",
  };
  const rows = (r.sources || []).slice(0, 10).map(s => {
    const h = (s.horizons || {})[ref] || {};
    const abn = h.mean_abnormal_pct;
    const hit = h.abnormal_hit_rate;
    return "<tr style='border-top:1px solid #1f2126;'>"
      + "<td style='padding:4px 6px;'>" + s.source + "</td>"
      + "<td style='padding:4px 6px;color:" + _plColor(abn) + ";'>"
        + (abn != null ? _sgn(abn) + fmt(abn, 2) + "%" : "—") + "</td>"
      + "<td style='padding:4px 6px;'>" + (hit != null ? fmt(hit, 0) + "%" : "—") + "</td>"
      + "<td style='padding:4px 6px;'>" + (s.n_resolved != null ? s.n_resolved : "—") + "</td>"
      + "<td style='padding:4px 6px;color:" + (vmap[s.verdict] || "#8b929d") + ";'>"
        + (s.verdict || "—") + "</td></tr>";
  });
  document.getElementById("se-rows").innerHTML =
    rows.length ? rows.join("") : "<tr><td colspan='5' class='muted' style='padding:6px;'>no collector resolved a watchlist move yet</td></tr>";
  document.getElementById("se-meta").textContent =
    "ref " + (r.reference_horizon != null ? r.reference_horizon + "d" : "—")
    + " · " + (r.n_resolved != null ? r.n_resolved : "—") + " resolved / "
    + (r.n_scored != null ? r.n_scored : "—") + " scored"
    + " · " + (r.n_tickers_priced != null ? r.n_tickers_priced : "—") + " tickers priced"
    + (r.spy_adjusted ? " · SPY-adjusted" : " · raw only")
    + (r.lookback_days != null ? " · " + r.lookback_days + "d lookback" : "");
}

// ───────── Behavioural scorecard — verdict-alignment router (new 2026-05-16, agent 4) ─────────
// Same stale-degrade contract as the behavioural cluster: a process that
// booted before this endpoint's commit 404s it → explicit "restart to apply".
async function refreshScorecard() {
  const r = await fetchMaybeStale("/api/scorecard");
  if (r.__unavailable) { markStale("score-state", "score-headline", "Behavioural scorecard endpoint"); return; }
  if (r.error) { document.getElementById("score-headline").textContent = "error: " + r.error; return; }
  const smap = {
    FLAGS_PRESENT:   ["#b71c1c", "#ffffff"],
    ALIGNED_HEALTHY: ["#1b5e20", "#a5d6a7"],
    NO_DATA:         ["#1f2126", "#8b929d"],
  };
  const [bg, fg] = smap[r.state] || smap.NO_DATA;
  const sEl = document.getElementById("score-state");
  sEl.textContent = (r.state || "—").replace(/_/g, " ");
  sEl.style.background = bg; sEl.style.color = fg;
  document.getElementById("score-headline").textContent = r.headline || "";

  const fEl = document.getElementById("score-focus");
  if (r.focus) {
    fEl.innerHTML = "<span style='color:#ffd479;'>Look first:</span> "
      + "<b>" + r.focus.name.replace(/_/g, " ") + "</b> — "
      + (r.focus.headline || "");
  } else { fEl.textContent = ""; }

  const cEl = document.getElementById("score-concordance");
  const conc = (r.concordance || []);
  if (conc.length) {
    cEl.innerHTML = conc.map(n =>
      "<span style='color:#ff8a80;'>" + n.count
      + " independent checks concur on " + n.theme.replace(/_/g, " ")
      + ":</span> " + (n.labels || []).join(", ")).join("<br>");
  } else { cEl.textContent = ""; }

  const kcolor = { FLAG: "#ff4455", OK: "#4caf50", IMMATURE: "#8b929d", ERROR: "#ffd479" };
  const rows = (r.checks || []).map(c =>
    "<tr><td style='padding:4px 6px;'>" + c.name.replace(/_/g, " ") + "</td>"
    + "<td style='padding:4px 6px;color:" + (kcolor[c.klass] || "#8b929d") + ";'>"
      + (c.label || "—") + " <span class='muted' style='font-size:10px;'>("
      + c.klass + ")</span></td>"
    + "<td style='padding:4px 6px;color:#8b929d;'>" + (c.headline || "—") + "</td></tr>");
  document.getElementById("score-rows").innerHTML =
    rows.length ? rows.join("") : "<tr><td colspan='3' class='muted' style='padding:6px;'>—</td></tr>";
}

// ───────── boot ─────────
refresh();
refreshSignals();
refreshAnalytics();
refreshSectorPulse();
refreshBriefing();
refreshSuggestions();
refreshRisk();
refreshEarningsRisk();
refreshGreeks();
refreshHeatmap();
refreshDedupedNews();
refreshScorer();
refreshThesis();
refreshDrawdown();
refreshCalibration();
refreshDecisionHealth();
refreshLiquidity();
refreshDecisionForensics();
refreshDecisionDrought();
refreshNewsEdge();
refreshScorerConfidence();
refreshDisagreement();
refreshDataFeed();
refreshValidation();
refreshTradeAsymmetry();
refreshCapitalParalysis();
refreshOpenAttribution();
refreshFeedHealth();
refreshDecisionReliability();
refreshFundedSuggestions();
refreshSignalFollowThrough();
refreshChurn();
refreshThesisDrift();
refreshLoserAutopsy();
refreshCorrelation();
refreshSourceEdge();
refreshScorecard();
refreshSessionDelta();
refreshGlobalStale();
setInterval(refresh, 15_000);
setInterval(refreshSignals, 30_000);
setInterval(refreshAnalytics, 30_000);
setInterval(refreshSectorPulse, 60_000);
setInterval(refreshBriefing, 60_000);
setInterval(refreshSuggestions, 45_000);
setInterval(refreshRisk, 30_000);
setInterval(refreshEarningsRisk, 300_000);
setInterval(refreshGreeks, 60_000);
setInterval(refreshHeatmap, 60_000);
setInterval(refreshDedupedNews, 45_000);
setInterval(refreshScorer, 60_000);
setInterval(refreshThesis, 60_000);
setInterval(refreshDrawdown, 30_000);
setInterval(refreshCalibration, 120_000);
setInterval(refreshDecisionHealth, 60_000);
setInterval(refreshLiquidity, 30_000);
setInterval(refreshDecisionForensics, 60_000);
setInterval(refreshDecisionDrought, 60_000);
setInterval(refreshNewsEdge, 300_000);
setInterval(refreshScorerConfidence, 120_000);
setInterval(refreshDisagreement, 60_000);
setInterval(refreshDataFeed, 60_000);
setInterval(refreshValidation, 120_000);
setInterval(refreshTradeAsymmetry, 60_000);
setInterval(refreshCapitalParalysis, 45_000);
setInterval(refreshOpenAttribution, 60_000);
setInterval(refreshFeedHealth, 60_000);
setInterval(refreshDecisionReliability, 60_000);
setInterval(refreshFundedSuggestions, 45_000);
setInterval(refreshSignalFollowThrough, 300_000);
setInterval(refreshChurn, 60_000);
setInterval(refreshThesisDrift, 60_000);
setInterval(refreshLoserAutopsy, 60_000);
setInterval(refreshCorrelation, 120_000);
setInterval(refreshSourceEdge, 300_000);
setInterval(refreshScorecard, 60_000);
setInterval(refreshSessionDelta, 60_000);
setInterval(refreshGlobalStale, 60_000);
showTab(INITIAL_TAB || "trader");
</script>
</div><!-- /.page-content -->

<nav class="bottom-nav" id="bottomNav">
  <a href="/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1V9.5z"/></svg>
    <span>Home</span>
  </a>
  <a href="/intern/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/></svg>
    <span>Intern</span>
  </a>
  <a href="/trader/" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M17 7h4v4"/></svg>
    <span>Trader</span>
  </a>
  <a href="/intern/chat" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.5 8.5 0 0 1-3.8-.9L3 21l1.9-5.7A8.38 8.38 0 0 1 4 11.5 8.5 8.5 0 0 1 12.5 3 8.38 8.38 0 0 1 21 11.5z"/></svg>
    <span>Chat</span>
  </a>
  <a href="/trader/backtests" class="bottom-tab">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/></svg>
    <span>Backtests</span>
  </a>
</nav>
<script>
(function(){
  const navToggle = document.getElementById('navToggle');
  const navDrawer = document.getElementById('navDrawer');
  const navOverlay = document.getElementById('navOverlay');
  if (navToggle) {
    navToggle.addEventListener('click', () => {
      navDrawer.classList.toggle('open');
      navOverlay.classList.toggle('open');
    });
    navOverlay.addEventListener('click', () => {
      navDrawer.classList.remove('open');
      navOverlay.classList.remove('open');
    });
  }
  document.querySelectorAll('.bottom-tab').forEach(tab => {
    if (tab.getAttribute('href') === window.location.pathname) {
      tab.classList.add('active');
    }
  });
})();
</script>
</body>
</html>
"""


def _api_prefix() -> str:
    return request.headers.get("X-Forwarded-Prefix", "").rstrip("/")


@app.route("/")
def index():
    return render_template_string(TEMPLATE, initial_tab="trader", api_prefix=_api_prefix())


@app.route("/backtests")
def backtests_page():
    return render_template_string(TEMPLATE, initial_tab="backtests", api_prefix=_api_prefix())


@app.route("/api/state")
def state():
    store = get_store()
    pf = store.get_portfolio()
    positions = store.open_positions()
    trades = store.recent_trades(40)
    decisions = store.recent_decisions(20)
    eq = store.equity_curve(5000)  # full history for accurate chart
    sp = eq[-1]["sp500_price"] if eq else None
    # Include all trades for chart markers (not just recent 40)
    all_trades = store.recent_trades(500)
    return jsonify({
        "now": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "portfolio": pf,
        "positions": positions,
        "trades": trades,
        "decisions": decisions,
        "equity": eq,
        "sp500": sp,
        "all_trades": all_trades,
    })


@app.route("/api/portfolio")
def portfolio_api():
    """Compact public read of the portfolio — consumed by Digital Intern's dashboard."""
    store = get_store()
    pf = store.get_portfolio()
    return jsonify({
        "total_value": pf.get("total_value"),
        "cash": pf.get("cash"),
        "starting_value": INITIAL_CASH,
    })


@app.route("/api/data-feed")
def data_feed_api():
    """Live news collector pulse — proxies digital-intern's articles.db.

    Returns articles-per-hour, per-24h, and top active sources, all filtered to
    exclude backtest synthetic rows (per the live-only invariant — see CLAUDE.md
    §5 in digital-intern). Returns zeros if the article DB isn't reachable so
    the widget can render gracefully on the live trader page.
    """
    # Prefer the LOCAL DB (the live daemon writes here), fall back to the
    # USB-mounted copy.
    candidates = [
        Path("/home/zeph/digital-intern/data/articles.db"),      # LOCAL first (live daemon writes here)
        Path("/media/zeph/projects/digital-intern/db/articles.db"),  # USB fallback
    ]
    db_path = next((p for p in candidates if p.exists()), None)
    if db_path is None:
        return jsonify({"articles_1h": 0, "articles_24h": 0, "top_sources": [],
                        "error": "articles.db not found"})
    try:
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        cut_1h  = (now - timedelta(hours=1)).isoformat()
        cut_24h = (now - timedelta(hours=24)).isoformat()
        live_clause = (
            "url NOT LIKE 'backtest://%' "
            "AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%'"
        )
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=3.0)
        try:
            n1 = conn.execute(
                f"SELECT COUNT(*) FROM articles WHERE first_seen >= ? AND {live_clause}",
                (cut_1h,)
            ).fetchone()[0]
            n24 = conn.execute(
                f"SELECT COUNT(*) FROM articles WHERE first_seen >= ? AND {live_clause}",
                (cut_24h,)
            ).fetchone()[0]
            top = conn.execute(
                f"SELECT source, COUNT(*) FROM articles "
                f"WHERE first_seen >= ? AND {live_clause} "
                f"GROUP BY source ORDER BY 2 DESC LIMIT 5",
                (cut_1h,)
            ).fetchall()
        finally:
            conn.close()
        return jsonify({
            "articles_1h": int(n1 or 0),
            "articles_24h": int(n24 or 0),
            "top_sources": [{"name": r[0] or "?", "count": int(r[1] or 0)} for r in top],
        })
    except Exception as e:
        return jsonify({"articles_1h": 0, "articles_24h": 0, "top_sources": [],
                        "error": str(e)})


@app.route("/api/backtests")
def backtests_api():
    from datetime import datetime, timezone
    try:
        from .backtest import BacktestStore
        store = BacktestStore()
        # Strip equity curves from the list — clients fetch curves lazily via
        # /api/backtests/curves when needed. This cuts payload from ~5MB to ~50KB.
        runs = store.all_runs(include_curves=False)
        completed = [r for r in runs if r.get("status") == "complete"]
        spy_baseline = completed[0].get("spy_return_pct") if completed else None

        return jsonify({
            "runs": runs,
            "total_runs": len(runs),
            "spy_baseline": spy_baseline,
            "qqq_baseline": None,
            "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    except Exception as e:
        return jsonify({"runs": [], "error": str(e)})


@app.route("/api/backtests/curves")
def backtest_curves_api():
    """Return normalized equity curves for requested run_ids.

    Query: ?run_ids=1,2,3
    Returns: {run_id: [{date, day_index, value, value_pct}, ...], ...}
    value_pct is % gain from start_value — comparable across different windows.
    """
    try:
        raw_ids = request.args.get("run_ids", "").strip()
        ids = [int(x.strip()) for x in raw_ids.split(",") if x.strip().isdigit()]
        if not ids:
            return jsonify({"error": "missing run_ids"}), 400
        if len(ids) > 100:
            return jsonify({"error": "max 100 run_ids per request"}), 400
        from .backtest import BacktestStore
        store = BacktestStore()
        curves = store.run_curves(ids)
        # keyed by string for JSON compatibility
        return jsonify({str(k): v for k, v in curves.items()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtests/<int:run_id>")
def backtest_detail(run_id: int):
    try:
        from .backtest import BacktestStore
        store = BacktestStore()
        detail = store.run_detail(run_id)
        if not detail:
            return jsonify({"error": "not found"}), 404
        return jsonify(detail)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtests/compare")
def backtest_compare():
    """Side-by-side comparison of 2-4 backtest runs.

    Query: ``/api/backtests/compare?ids=1,2,3`` (comma-separated run_ids).

    Returns equity_curve points re-shaped for overlay rendering:
      - ``day_index`` = days since run's start_date, so runs with different
        windows can be drawn on the same x-axis.
      - ``value_pct`` = (value / start_value - 1) * 100, so returns compare
        on a normalized y-axis regardless of initial cash differences.

    Per-run summary fields (return %, vs_spy %, max drawdown, trade count,
    decision count, win rate) are computed from the same equity_curve + trades
    that the existing /api/backtests/<id> route already returns, so this is a
    pure aggregation — no new state.
    """
    raw_ids = request.args.get("ids", "").strip()
    if not raw_ids:
        return jsonify({"error": "missing ids — e.g. ?ids=1,2,3"}), 400
    try:
        ids = []
        for tok in raw_ids.split(","):
            tok = tok.strip()
            if not tok:
                continue
            ids.append(int(tok))
        if not ids:
            return jsonify({"error": "no valid ids"}), 400
        if len(ids) > 4:
            return jsonify({"error": "max 4 runs per comparison"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "ids must be comma-separated integers"}), 400

    try:
        from .backtest import BacktestStore
        from datetime import date
        store = BacktestStore()
        out_runs = []
        for rid in ids:
            detail = store.run_detail(rid)
            if not detail:
                out_runs.append({"run_id": rid, "error": "not found"})
                continue
            eq = detail.get("equity_curve") or []
            trades = detail.get("trades") or []
            # Normalize the equity curve for overlay.
            start_val = float(eq[0]["value"]) if eq else 1000.0
            start_date_str = detail.get("start_date") or (eq[0]["date"] if eq else None)
            try:
                start_d = date.fromisoformat(start_date_str) if start_date_str else None
            except (TypeError, ValueError):
                start_d = None

            curve = []
            peak = start_val
            max_dd = 0.0
            for p in eq:
                v = float(p.get("value") or 0.0)
                if v > peak:
                    peak = v
                if peak > 0:
                    dd = (peak - v) / peak * 100.0
                    if dd > max_dd:
                        max_dd = dd
                d_str = p.get("date")
                day_idx = None
                if start_d and d_str:
                    try:
                        day_idx = (date.fromisoformat(d_str) - start_d).days
                    except (TypeError, ValueError):
                        day_idx = None
                curve.append({
                    "date": d_str,
                    "day_index": day_idx,
                    "value": v,
                    "value_pct": round((v / start_val - 1.0) * 100.0, 3) if start_val else 0.0,
                })

            # Win rate from trades that we can pair: BUYs followed by a SELL on the
            # same ticker close at a higher price. Best-effort — backtest trades use
            # ``action`` ∈ {BUY, SELL, BUY_CALL, SELL_CALL, ...}; we score stocks only
            # so the metric stays interpretable.
            wins = 0
            losses = 0
            held: dict[str, list[tuple[float, float]]] = {}  # ticker -> [(qty, price)]
            for t in trades:
                act = (t.get("action") or "").upper()
                tk = t.get("ticker") or ""
                qty = float(t.get("qty") or 0)
                px = float(t.get("price") or 0)
                if not tk or qty <= 0 or px <= 0:
                    continue
                if act == "BUY":
                    held.setdefault(tk, []).append((qty, px))
                elif act == "SELL":
                    lots = held.get(tk) or []
                    remaining = qty
                    while remaining > 0 and lots:
                        lot_qty, lot_px = lots[0]
                        use = min(lot_qty, remaining)
                        if px > lot_px:
                            wins += 1
                        elif px < lot_px:
                            losses += 1
                        if use >= lot_qty:
                            lots.pop(0)
                        else:
                            lots[0] = (lot_qty - use, lot_px)
                        remaining -= use
                    held[tk] = lots
            total_rt = wins + losses
            win_rate = (wins / total_rt) if total_rt else None

            out_runs.append({
                "run_id": rid,
                "start_date": detail.get("start_date"),
                "end_date": detail.get("end_date"),
                "status": detail.get("status"),
                "total_return_pct": detail.get("total_return_pct"),
                "spy_return_pct": detail.get("spy_return_pct"),
                "vs_spy_pct": detail.get("vs_spy_pct"),
                "max_drawdown_pct": round(max_dd, 2),
                "n_trades": detail.get("n_trades"),
                "n_decisions": detail.get("n_decisions"),
                "n_round_trips": total_rt,
                "win_rate": round(win_rate, 4) if win_rate is not None else None,
                "final_value": detail.get("final_value"),
                "start_value": start_val,
                "n_points": len(curve),
                "equity_curve": curve,
            })
        return jsonify({
            "ids": ids,
            "n_runs": len(out_runs),
            "runs": out_runs,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtests/<int:run_id>/trades")
def backtest_trades(run_id: int):
    try:
        from .backtest import BacktestStore
        store = BacktestStore()
        detail = store.run_detail(run_id)
        if not detail:
            return jsonify({"error": "not found"}), 404
        return jsonify({"run_id": run_id, "trades": detail.get("trades", [])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtests/<int:run_id>/decisions")
def backtest_decisions(run_id: int):
    try:
        from .backtest import BacktestStore
        store = BacktestStore()
        detail = store.run_detail(run_id)
        if not detail:
            return jsonify({"error": "not found"}), 404
        return jsonify({"run_id": run_id, "decisions": detail.get("decisions", [])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-progress")
def model_progress():
    """Per-cycle aggregated returns for the Model Progress chart.

    Groups completed runs into cycles of RUNS_PER_CYCLE=5 by run_id order.
    Labels use actual run_id ranges so trimming old runs does not renumber cycles.
    """
    try:
        from .backtest import BacktestStore
        store = BacktestStore()
        rows = store.conn.execute(
            "SELECT run_id, total_return_pct, completed_at FROM backtest_runs "
            "WHERE status='complete' ORDER BY run_id"
        ).fetchall()
        if not rows:
            return jsonify({"cycles": []})

        cycle_size = 5  # RUNS_PER_CYCLE
        cycles = []
        for i in range(0, len(rows), cycle_size):
            chunk = rows[i:i + cycle_size]
            returns = [r["total_return_pct"] for r in chunk]
            run_ids = [r["run_id"] for r in chunk]
            # Use actual run_id range as label so chart is stable across trims
            label = f"#{run_ids[0]}" if len(run_ids) == 1 else f"#{run_ids[0]}-{run_ids[-1]}"
            cycles.append({
                "cycle": label,
                "run_start": run_ids[0],
                "best": round(max(returns), 2),
                "avg": round(sum(returns) / len(returns), 2),
                "worst": round(min(returns), 2),
                "n": len(returns),
                "completed_at": chunk[-1]["completed_at"],
            })
        return jsonify({"cycles": cycles, "total_runs": len(rows)})
    except Exception as e:
        return jsonify({"cycles": [], "error": str(e)})


@app.route("/api/analytics")
def analytics_api():
    """Derived portfolio analytics — sector exposure, drawdown, Sharpe, win rate, daily P/L."""
    try:
        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        # Pull a generous trades sample for round-trip accounting.
        trades = list(reversed(store.recent_trades(2000)))  # oldest → newest
        eq = store.equity_curve(5000)  # most recent 5000, ascending after the bugfix

        total_value = pf.get("total_value") or 0.0

        # ─── 1. Sector exposure ───
        sector_usd: dict[str, float] = {}
        for p in positions:
            mult = 100 if p["type"] in ("call", "put") else 1
            price = p.get("current_price") or p["avg_cost"]
            val = price * p["qty"] * mult
            sec = _classify(p["ticker"])
            sector_usd[sec] = sector_usd.get(sec, 0.0) + val

        sector_pct = {
            s: round((v / total_value * 100) if total_value else 0.0, 2)
            for s, v in sector_usd.items()
        }
        cash_pct = round((pf.get("cash", 0) / total_value * 100) if total_value else 0.0, 2)

        # ─── 2. Max drawdown (peak-to-trough on equity curve) ───
        # Return None (not 0.0) when there's no equity history so the frontend's
        # `== null` branch fires and renders "—" instead of "-0.00 (0.00%)".
        max_dd_usd: float | None = None
        max_dd_pct: float | None = None
        if eq:
            max_dd_usd = 0.0
            max_dd_pct = 0.0
            peak = eq[0]["total_value"]
            for p in eq:
                v = p["total_value"]
                if v > peak:
                    peak = v
                dd_usd = peak - v
                dd_pct = (dd_usd / peak * 100) if peak else 0.0
                if dd_usd > max_dd_usd:
                    max_dd_usd = dd_usd
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct

        # ─── 3. Sharpe estimate from daily-bucketed returns ───
        # Bucket equity_curve by date, take last value per date, compute log returns,
        # annualize as mean/std * sqrt(252).
        sharpe = None
        daily_returns: list[float] = []
        by_day: dict[str, float] = {}
        for p in eq:
            day = (p["timestamp"] or "")[:10]
            if day:
                by_day[day] = p["total_value"]  # last write wins, leaves us with EOD close
        day_keys = sorted(by_day.keys())
        for i in range(1, len(day_keys)):
            prev = by_day[day_keys[i - 1]]
            cur = by_day[day_keys[i]]
            if prev and prev > 0:
                daily_returns.append((cur / prev) - 1.0)
        if len(daily_returns) >= 5:
            mean = sum(daily_returns) / len(daily_returns)
            var = sum((r - mean) ** 2 for r in daily_returns) / len(daily_returns)
            std = var ** 0.5
            sharpe = round((mean / std) * (252 ** 0.5), 2) if std > 0 else None

        # ─── 4. Win rate (round-trips per distinct position) ───
        # A round-trip closes when held qty returns to ≈ 0. P/L = proceeds - cost.
        # Round-trip grouping is delegated to analytics.round_trips so this
        # endpoint and any future trade-attribution caller share one
        # implementation instead of drifting hand-maintained copies.
        # build_round_trips keys by (ticker, type, strike, expiry) — stock and
        # option legs of the same ticker stay distinct. pnl_usd is rounded to
        # 4dp there; the win/loss split below uses strict `> 0`, so a sub-cent
        # rounding artefact reads as a non-win (pinned by test_round_trips).
        from .analytics.round_trips import build_round_trips
        _rts = build_round_trips(trades)
        round_trips: list[float] = [rt["pnl_usd"] for rt in _rts]
        holding_days: list[float] = [
            rt["hold_days"] for rt in _rts if rt["hold_days"] is not None
        ]  # one entry per closed round-trip with a parseable entry/exit ts

        wins = [p for p in round_trips if p > 0]
        losses = [p for p in round_trips if p <= 0]
        win_rate = round(len(wins) / len(round_trips) * 100, 2) if round_trips else None
        avg_winner = round(sum(wins) / len(wins), 2) if wins else None
        avg_loser = round(sum(losses) / len(losses), 2) if losses else None
        total_realized = round(sum(round_trips), 2) if round_trips else 0.0

        # ─── 4b. Profit factor + avg holding period ───
        # Profit factor = gross wins / gross losses. >1 means the edge survives
        # losers; a 50% win rate with PF 2.0 is a real edge, PF 0.8 is bleeding.
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 1e-9 else None
        avg_hold_days = (round(sum(holding_days) / len(holding_days), 2)
                         if holding_days else None)

        # ─── 4c. Sortino — like Sharpe but only downside vol is "risk" ───
        sortino = None
        if len(daily_returns) >= 5:
            dmean = sum(daily_returns) / len(daily_returns)
            downside = [r for r in daily_returns if r < 0]
            if downside:
                dvar = sum(r * r for r in downside) / len(daily_returns)
                dstd = dvar ** 0.5
                if dstd > 0:
                    sortino = round((dmean / dstd) * (252 ** 0.5), 2)

        # ─── 4d. S&P 500 beta + correlation (paired daily returns) ───
        sp_by_day: dict[str, float] = {}
        for p in eq:
            day = (p["timestamp"] or "")[:10]
            spx = p.get("sp500_price")
            if day and spx:
                sp_by_day[day] = spx
        port_ret: list[float] = []
        spx_ret: list[float] = []
        for i in range(1, len(day_keys)):
            d0, d1 = day_keys[i - 1], day_keys[i]
            if d0 in sp_by_day and d1 in sp_by_day:
                pv0, sv0 = by_day[d0], sp_by_day[d0]
                pv1, sv1 = by_day[d1], sp_by_day[d1]
                if pv0 > 0 and sv0 > 0:
                    port_ret.append(pv1 / pv0 - 1.0)
                    spx_ret.append(sv1 / sv0 - 1.0)
        sp500_beta = None
        sp500_corr = None
        if len(port_ret) >= 5:
            n = len(port_ret)
            mp = sum(port_ret) / n
            ms = sum(spx_ret) / n
            cov = sum((port_ret[i] - mp) * (spx_ret[i] - ms) for i in range(n)) / n
            var_s = sum((s - ms) ** 2 for s in spx_ret) / n
            var_p = sum((p - mp) ** 2 for p in port_ret) / n
            if var_s > 0:
                sp500_beta = round(cov / var_s, 2)
                if var_p > 0:
                    sp500_corr = round(cov / ((var_s ** 0.5) * (var_p ** 0.5)), 3)

        # ─── 4e. Calmar — annualized return ÷ max drawdown ───
        # Meaningless on <20 trading days of history, so gate it hard.
        calmar = None
        if len(daily_returns) >= 20 and max_dd_pct and max_dd_pct > 0:
            # Baseline must come from the store constant, not a hardcoded
            # 1000.0 — a literal here silently desyncs Calmar if INITIAL_CASH
            # ever moves (same desync class fixed in reporter.py, commit 2a154df).
            total_return_pct = (total_value / INITIAL_CASH - 1.0) * 100.0
            years = len(day_keys) / 252.0
            if years > 0:
                calmar = round((total_return_pct / years) / max_dd_pct, 2)

        # ─── 5. Daily P/L (today only, UTC bucket) ───
        today = datetime.now(timezone.utc).date().isoformat()
        today_eq = [p for p in eq if (p["timestamp"] or "").startswith(today)]
        daily_pl = None
        daily_pl_pct = None
        if today_eq:
            open_val = today_eq[0]["total_value"]
            cur_val = total_value
            if open_val:
                daily_pl = round(cur_val - open_val, 2)
                daily_pl_pct = round(daily_pl / open_val * 100, 2)

        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_value": round(total_value, 2),
            "cash_pct": cash_pct,
            "sector_exposure_pct": sector_pct,
            "sector_exposure_usd": {s: round(v, 2) for s, v in sector_usd.items()},
            "max_drawdown_usd": round(max_dd_usd, 2) if max_dd_usd is not None else None,
            "max_drawdown_pct": round(max_dd_pct, 2) if max_dd_pct is not None else None,
            "sharpe_annualized": sharpe,
            "n_trading_days": len(daily_returns),
            "n_round_trips": len(round_trips),
            "win_rate_pct": win_rate,
            "avg_winner_usd": avg_winner,
            "avg_loser_usd": avg_loser,
            "realized_pl_usd": total_realized,
            "profit_factor": profit_factor,
            "avg_holding_days": avg_hold_days,
            "sortino_annualized": sortino,
            "calmar_ratio": calmar,
            "sp500_beta": sp500_beta,
            "sp500_correlation": sp500_corr,
            "daily_pl_usd": daily_pl,
            "daily_pl_pct": daily_pl_pct,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _articles_db_path() -> Path | None:
    """Match how paper_trader.signals discovers the digital-intern articles.db."""
    import os
    usb = Path(os.environ.get("DIGITAL_INTERN_USB",
                              "/media/zeph/projects/digital-intern/db")) / "articles.db"
    if usb.exists():
        return usb
    local = Path("/home/zeph/digital-intern/data/articles.db")
    if local.exists():
        return local
    return None


def _ticker_news_pulse(tickers: list[str], hours: int = 24) -> dict[str, dict]:
    """For each ticker, count + top headline of articles mentioning it.

    Reads the articles DB in read-only mode. Live-only filter is applied so
    backtest/opus_annotation synthetic rows are excluded.
    """
    out: dict[str, dict] = {t.upper(): {
        "n": 0, "urgent": 0, "top_title": None, "top_url": None, "top_score": 0.0,
    } for t in tickers}
    path = _articles_db_path()
    if path is None:
        return out
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = conn.execute(
            "SELECT title, url, full_text, ai_score, urgency FROM articles "
            "WHERE first_seen >= ? AND ai_score > 0 "
            "AND url NOT LIKE 'backtest://%' "
            "AND source NOT LIKE 'backtest_%' "
            "AND source NOT LIKE 'opus_annotation%' "
            "ORDER BY ai_score DESC LIMIT 2000",
            (since,),
        ).fetchall()
    except Exception:
        return out
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    patterns = {t.upper(): re.compile(rf"(?:\$|\b){re.escape(t.upper())}\b") for t in tickers}
    for r in rows:
        body = r["title"] or ""
        if r["full_text"]:
            try:
                body = body + " " + zlib.decompress(r["full_text"]).decode("utf-8", "replace")
            except Exception:
                pass
        body_up = body.upper()
        for t, pat in patterns.items():
            if pat.search(body_up):
                rec = out[t]
                rec["n"] += 1
                if (r["urgency"] or 0) >= 1:
                    rec["urgent"] += 1
                if (r["ai_score"] or 0) > rec["top_score"]:
                    rec["top_score"] = r["ai_score"]
                    rec["top_title"] = r["title"]
                    rec["top_url"] = r["url"]
    return out


@app.route("/api/sector-pulse")
def sector_pulse_api():
    """Compact semis-sector card: price, day %, RSI, news count, top headline per ticker."""
    try:
        from . import market
        from .strategy import _QUANT_CACHE, get_quant_signals_live

        tickers = SECTOR_PULSE_TICKERS
        # Warm the quant cache only for tickers we don't already have fresh data for.
        # get_quant_signals_live respects its own 5-min TTL.
        try:
            get_quant_signals_live(tickers)
        except Exception:
            pass

        prices = market.get_prices(tickers)
        news = _ticker_news_pulse(tickers, hours=24)

        out = []
        for t in tickers:
            cached = _QUANT_CACHE.get(t)
            quant = cached[0] if cached else {}
            # Compute today's % change from quant signals' 1y history if we cached it.
            rsi = quant.get("RSI")
            mom_5d = quant.get("mom_5d")
            mom_20d = quant.get("mom_20d")
            macd = quant.get("macd_signal")
            vol_ratio = quant.get("vol_ratio")
            pct_from_52h = quant.get("pct_from_52h")
            nrec = news.get(t.upper(), {})
            out.append({
                "ticker": t,
                "price": prices.get(t),
                "rsi": rsi,
                "macd": macd,
                "mom_5d": mom_5d,
                "mom_20d": mom_20d,
                "vol_ratio": vol_ratio,
                "pct_from_52h": pct_from_52h,
                "news_count_24h": nrec.get("n", 0),
                "news_urgent_24h": nrec.get("urgent", 0),
                "top_headline": nrec.get("top_title"),
                "top_url": nrec.get("top_url"),
                "top_score": nrec.get("top_score") or 0.0,
            })
        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tickers": out,
        })
    except Exception as e:
        return jsonify({"tickers": [], "error": str(e)}), 500


# ───────────────────────── Feature-dev additions (2026-05-14) ─────────────────────────
# Three additive endpoints + supporting helpers:
#   /api/suggestions  — co-pilot trade ideas from news × positions × quant signals
#   /api/risk         — concentration / leveraged-exposure / position-age / shock estimate
#   /api/briefing     — futures + market-open countdown + top urgent news
# All routes degrade gracefully — yfinance / signals / strategy imports are lazy and
# wrapped so a missing dependency returns a structured error instead of 500.

# Leverage factors for the SPY-shock dollar-at-risk estimate. Conservative single
# beta numbers chosen to be obviously approximate — this is decision support, not VaR.
_LEVERAGE_BETA = {
    "broad": 1.0,
    "broad_lev": 3.0,       # Most broad-leveraged are 3x; QLD/SSO are 2x but in the same bucket here
    "tech": 1.2,
    "tech_lev": 3.0,
    "crypto_lev": 2.5,
    "semis": 1.5,
    "semis_lev": 3.0,
    "optical": 1.4,
    "bio_lev": 3.0,
    "health_lev": 3.0,
    "fin_lev": 3.0,
    "housing_lev": 3.0,
    "util_lev": 3.0,
    "defense_lev": 3.0,
    "other": 1.0,
}

_LEVERAGED_SECTORS = {s for s in _LEVERAGE_BETA if s.endswith("_lev")}


def _position_ages_from_trades(open_positions: list[dict], trades_oldest_first: list[dict]) -> dict[str, int]:
    """For each currently-open ticker, return days since the earliest BUY in the
    most recent open lot. Walks trades chronologically and resets the open-lot
    timestamp every time the running quantity returns to ≈0."""
    open_tickers = {p["ticker"] for p in open_positions if p.get("type") == "stock"}
    earliest: dict[str, str] = {}
    held: dict[str, float] = {}
    for t in trades_oldest_first:
        tk = t.get("ticker")
        if tk not in open_tickers:
            continue
        act = (t.get("action") or "").upper()
        # Only stock trades affect stock-position age. BUY_CALL / SELL_PUT etc.
        # would otherwise corrupt the running stock quantity for this ticker.
        if act not in ("BUY", "SELL"):
            continue
        qty = float(t.get("qty") or 0)
        ts = t.get("timestamp") or ""
        if act == "BUY":
            if held.get(tk, 0.0) < 1e-6 or tk not in earliest:
                earliest[tk] = ts
            held[tk] = held.get(tk, 0.0) + qty
        else:  # SELL
            held[tk] = held.get(tk, 0.0) - qty
            if abs(held.get(tk, 0.0)) < 1e-6:
                earliest.pop(tk, None)
    now = datetime.now(timezone.utc)
    ages: dict[str, int] = {}
    for tk, ts in earliest.items():
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ages[tk] = max(0, (now - dt).days)
        except Exception:
            continue
    return ages


def _concentration_severity(top1_pct: float, top3_pct: float) -> tuple[str, bool]:
    """Bucket a portfolio's concentration into a severity label + boolean flag.

    HIGH triggers a UI alert and indicates dangerous over-concentration.
    Thresholds are deliberately strict — this is a $1000 paper book whose
    edge is breadth + speed, not single-name conviction."""
    if top1_pct >= 60 or top3_pct >= 90:
        return "HIGH", True
    if top1_pct >= 40 or top3_pct >= 75:
        return "MEDIUM", True
    return "LOW", False


@app.route("/api/risk")
def risk_api():
    """Risk-focused portfolio panel. Fields are intentionally disjoint from
    /api/analytics: concentration, leveraged exposure, position age, stale flags,
    SPY-shock dollar-at-risk estimate. Pair with /api/analytics for full picture."""
    try:
        store = get_store()
        pf = store.get_portfolio()
        positions = store.open_positions()
        total_value = float(pf.get("total_value") or 0.0)
        cash = float(pf.get("cash") or 0.0)

        # ── Per-position market values + sector classification ──
        rows = []
        leveraged_usd = 0.0
        shock_usd = 0.0  # estimated $ change if SPY drops 3%
        for p in positions:
            mult = 100 if p["type"] in ("call", "put") else 1
            price = p.get("current_price") or p.get("avg_cost") or 0.0
            qty = float(p.get("qty") or 0)
            val = price * qty * mult
            sec = _classify(p["ticker"])
            beta = _LEVERAGE_BETA.get(sec, 1.0)
            # Options inherit underlying sector beta but with a rough 3x payoff
            # multiplier for at-the-money ITM exposure; cap at 4.
            if p["type"] in ("call", "put"):
                beta = min(beta * 3.0, 4.0)
                if p["type"] == "put":
                    beta = -beta  # puts profit on a drop
            shock_usd += -0.03 * beta * val  # negative = loss on -3% SPY
            if sec in _LEVERAGED_SECTORS:
                leveraged_usd += val
            rows.append({
                "ticker": p["ticker"],
                "type": p["type"],
                "sector": sec,
                "market_value": round(val, 2),
                "pct_port": round((val / total_value * 100) if total_value else 0.0, 2),
                "beta_est": round(beta, 2),
            })

        rows.sort(key=lambda r: -r["market_value"])
        largest = rows[0] if rows else None
        top3_pct = round(sum(r["pct_port"] for r in rows[:3]), 2)
        top1_pct = round(largest["pct_port"], 2) if largest else 0.0
        conc_severity, conc_warning = _concentration_severity(top1_pct, top3_pct)

        # ── Position ages from trade history ──
        trades_oldest_first = list(reversed(store.recent_trades(2000)))
        ages = _position_ages_from_trades(positions, trades_oldest_first)

        # ── Stale flag: held > 7d, |P/L| < 2% — likely sitting on dead money ──
        # store.open_positions() rows have current_price/avg_cost but no pl_pct,
        # so derive it here rather than reading a key that's always missing.
        stale = []
        for p in positions:
            tk = p["ticker"]
            avg = float(p.get("avg_cost") or 0.0)
            cur = float(p.get("current_price") or 0.0) or avg
            pl_pct_signed = ((cur - avg) / avg * 100) if avg else 0.0
            age = ages.get(tk)
            if age is not None and age >= 7 and abs(pl_pct_signed) < 2.0:
                stale.append({
                    "ticker": tk,
                    "age_days": age,
                    "pl_pct": round(pl_pct_signed, 2),
                    "market_value": round(
                        cur * float(p.get("qty") or 0)
                        * (100 if p["type"] in ("call", "put") else 1),
                        2,
                    ),
                })

        ages_list = sorted(ages.values()) if ages else []
        if ages_list:
            mid = len(ages_list) // 2
            if len(ages_list) % 2:
                median_age = ages_list[mid]
            else:
                median_age = round((ages_list[mid - 1] + ages_list[mid]) / 2)
        else:
            median_age = None

        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_value": round(total_value, 2),
            "cash_usd": round(cash, 2),
            "cash_pct": round((cash / total_value * 100) if total_value else 0.0, 2),
            "n_positions": len(positions),
            "concentration_top1_pct": top1_pct,
            "concentration_top1_ticker": largest["ticker"] if largest else None,
            "concentration_top3_pct": top3_pct,
            "concentration_warning": conc_warning,
            "concentration_severity": conc_severity,
            "leveraged_usd": round(leveraged_usd, 2),
            "leveraged_pct": round((leveraged_usd / total_value * 100) if total_value else 0.0, 2),
            "spy_shock_3pct_usd": round(shock_usd, 2),  # negative = loss
            "spy_shock_3pct_pct": round((shock_usd / total_value * 100) if total_value else 0.0, 2),
            "median_age_days": median_age,
            "max_age_days": max(ages.values()) if ages else None,
            "position_ages": ages,
            "stale_positions": stale,
            "positions_by_value": rows,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _next_market_open() -> tuple[datetime | None, int | None]:
    """Return (next_open_dt_utc, seconds_until). If market is open right now,
    returns the next close instead with a sign convention noted by the caller.
    Uses paper_trader.market constants — keeps the NYSE holiday calendar in one place."""
    try:
        from . import market as _mkt
    except Exception:
        return None, None
    now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(_mkt.NY)
    open_min = 9 * 60 + 30
    cur_min = now_ny.hour * 60 + now_ny.minute
    # If currently open, return next close.
    if _mkt.is_market_open(now_utc):
        close_dt = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        return close_dt.astimezone(timezone.utc), int((close_dt - now_ny).total_seconds())
    # Walk forward day-by-day to find the next open day. The outer guard
    # `(not is_today or cur_min < open_min)` already excludes "today, past
    # market open" — by the time we'd consider returning today, we must be
    # before 9:30 AM NY, so no past-close edge case to handle.
    from datetime import timedelta as _td
    candidate = now_ny
    for _ in range(10):
        is_weekday = candidate.weekday() < 5
        is_holiday = candidate.date() in _mkt.NYSE_HOLIDAYS_2026
        is_today = candidate.date() == now_ny.date()
        if is_weekday and not is_holiday and (not is_today or cur_min < open_min):
            open_dt = candidate.replace(hour=9, minute=30, second=0, microsecond=0)
            return open_dt.astimezone(timezone.utc), int((open_dt - now_ny).total_seconds())
        candidate = candidate + _td(days=1)
        candidate = candidate.replace(hour=0, minute=0, second=0, microsecond=0)
    return None, None


@app.route("/api/briefing")
def briefing_api():
    """Pre-market / live briefing card. Combines market-open status, futures,
    top urgent overnight news, and a one-line summary string. Designed to be the
    first thing the user sees on the trader pane each morning."""
    try:
        from . import market as _mkt
        from . import signals as _sig

        now_utc = datetime.now(timezone.utc)
        is_open = _mkt.is_market_open(now_utc)
        next_dt, secs = _next_market_open()

        # ── Futures (cached 30s in market.get_futures_price) ──
        futures: dict[str, float | None] = {}
        for sym in ("ES=F", "NQ=F", "CL=F", "GC=F"):
            try:
                futures[sym] = _mkt.get_futures_price(sym)
            except Exception:
                futures[sym] = None

        # ── Urgent news from the last 8h (Reddit/Bloomberg-style overnight) ──
        urgent: list[dict] = []
        try:
            urgent = _sig.get_urgent_articles(minutes=8 * 60)[:5]
        except Exception:
            urgent = []
        urgent_compact = [{
            "title": (u.get("title") or "")[:140],
            "source": u.get("source"),
            "ai_score": u.get("ai_score"),
            "urgency": u.get("urgency"),
            "first_seen": u.get("first_seen"),
            "tickers": u.get("tickers", [])[:5],
        } for u in urgent]

        # ── High-score overnight signals as a secondary list ──
        top: list[dict] = []
        try:
            top = _sig.get_top_signals(n=5, hours=8, min_score=5.0)
        except Exception:
            top = []
        top_compact = [{
            "title": (s.get("title") or "")[:140],
            "source": s.get("source"),
            "ai_score": s.get("ai_score"),
            "tickers": s.get("tickers", [])[:5],
            "first_seen": s.get("first_seen"),
        } for s in top]

        # ── One-line summary ──
        if is_open:
            if secs is not None:
                hrs = secs // 3600
                mins = (secs % 3600) // 60
                status_line = f"Market OPEN — closes in {hrs}h{mins:02d}m"
            else:
                status_line = "Market OPEN"
        else:
            if secs is not None and next_dt is not None:
                hrs = secs // 3600
                mins = (secs % 3600) // 60
                status_line = f"Market CLOSED — opens in {hrs}h{mins:02d}m ({next_dt.astimezone(_mkt.NY).strftime('%a %H:%M %Z')})"
            else:
                status_line = "Market CLOSED"

        return jsonify({
            "as_of": now_utc.isoformat(timespec="seconds"),
            "market_open": is_open,
            "next_event_utc": next_dt.isoformat(timespec="seconds") if next_dt else None,
            "next_event_seconds": secs,
            "status_line": status_line,
            "futures": futures,
            "urgent_news": urgent_compact,
            "top_signals": top_compact,
            "urgent_count": len(urgent_compact),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _classify_action(ticker: str, held_qty: float, quant: dict, news_score: float, news_urgent: bool) -> tuple[str, float, list[str]]:
    """Co-pilot rules. Returns (action, conviction 0..1, reason_bullets).
    Conservative — never says BUY without at least one technical confirm."""
    notes: list[str] = []
    rsi = quant.get("RSI") if quant else None
    macd = quant.get("MACD") if quant else None
    mom5 = quant.get("mom_5d") if quant else None
    mom20 = quant.get("mom_20d") if quant else None

    # ── Technical scoring (-1..+1 bullish bias) ──
    bias = 0.0
    if rsi is not None:
        if rsi < 30:
            bias += 0.4; notes.append(f"RSI {rsi:.0f} oversold")
        elif rsi < 45:
            bias += 0.1; notes.append(f"RSI {rsi:.0f} cool")
        elif rsi > 70:
            bias -= 0.4; notes.append(f"RSI {rsi:.0f} overbought")
        elif rsi > 60:
            bias -= 0.1; notes.append(f"RSI {rsi:.0f} hot")
    if macd:
        if macd == "bullish":
            bias += 0.25; notes.append("MACD bullish")
        elif macd == "bearish":
            bias -= 0.25; notes.append("MACD bearish")
    if mom5 is not None:
        if mom5 > 3:
            bias += 0.15; notes.append(f"5d +{mom5:.1f}%")
        elif mom5 < -3:
            bias -= 0.15; notes.append(f"5d {mom5:.1f}%")
    if mom20 is not None:
        if mom20 > 8:
            bias += 0.1; notes.append(f"20d +{mom20:.1f}%")
        elif mom20 < -8:
            bias -= 0.1; notes.append(f"20d {mom20:.1f}%")

    bias = max(-1.0, min(1.0, bias))

    # ── News weight ──
    news_weight = min(news_score / 10.0, 1.0)
    if news_urgent:
        news_weight = min(news_weight + 0.2, 1.0)
        notes.insert(0, "URGENT news")

    # ── Action selection ──
    if held_qty > 0:
        # EXIT must be checked before TRIM: a strong bearish bias (< -0.5) also
        # satisfies the TRIM guard (bias < -0.3) when news is quiet, so testing
        # TRIM first swallowed the EXIT case and downgraded severity exactly
        # when the technical breakdown was strongest.
        if bias < -0.5:
            return "EXIT", min(0.65 + abs(bias) * 0.3, 0.95), notes
        if bias < -0.3 and news_weight < 0.4:
            return "TRIM", min(0.6 + abs(bias) * 0.3, 0.95), notes
        if bias > 0.25 and news_weight > 0.5:
            return "ADD", min(0.5 + bias * 0.3 + news_weight * 0.2, 0.95), notes
        return "HOLD", 0.4 + max(0.0, bias) * 0.2, notes
    else:
        # not held
        if news_weight > 0.65 and bias > 0.1:
            return "BUY", min(0.5 + news_weight * 0.3 + max(0.0, bias) * 0.2, 0.95), notes
        if news_weight > 0.5 or abs(bias) > 0.35:
            return "WATCH", min(0.3 + news_weight * 0.3 + abs(bias) * 0.2, 0.8), notes
        return "WATCH", 0.2 + news_weight * 0.2, notes


@app.route("/api/suggestions")
def suggestions_api():
    """Trade-idea co-pilot. Ranked list of BUY / ADD / TRIM / EXIT / WATCH cards.

    Inputs: top-scored articles from last 6h (digital-intern), live quant signals,
    current open positions. Output is *decision support*, not auto-execution —
    the live trader is still Opus 4.7 in strategy.py."""
    try:
        from . import signals as _sig

        # Pull top signals (broader window than the trader uses, for visibility).
        try:
            top_signals = _sig.get_top_signals(n=30, hours=6, min_score=5.0)
        except Exception as e:
            return jsonify({"error": f"signals unavailable: {e}", "suggestions": []})

        store = get_store()
        positions = store.open_positions()
        held: dict[str, float] = {}
        position_pl: dict[str, float] = {}
        for p in positions:
            if p.get("type") == "stock":
                held[p["ticker"]] = held.get(p["ticker"], 0.0) + float(p.get("qty") or 0)
                # store.open_positions() doesn't include pl_pct — derive from avg/current.
                avg = float(p.get("avg_cost") or 0.0)
                cur = float(p.get("current_price") or 0.0) or avg
                position_pl[p["ticker"]] = ((cur - avg) / avg * 100) if avg else 0.0

        # Build the candidate ticker set: (news-mentioned ∩ watchlist) ∪ currently held.
        # Constraining to the watchlist filters out the ticker-extractor's noise
        # (acronyms like GSPC / IXIC / DJI that yfinance can't price anyway).
        try:
            from .strategy import WATCHLIST as _WATCHLIST
            universe = {t.upper() for t in _WATCHLIST}
        except Exception:
            universe = set()
        universe |= {t.upper() for t in held}

        candidates: dict[str, dict] = {}
        for art in top_signals:
            for tk in art.get("tickers") or []:
                if not tk or len(tk) > 6:
                    continue
                if tk.upper() not in universe:
                    continue
                rec = candidates.setdefault(tk, {
                    "ticker": tk,
                    "news_count": 0,
                    "news_max_score": 0.0,
                    "news_urgent": False,
                    "top_headline": None,
                    "top_url": None,
                })
                rec["news_count"] += 1
                if (art.get("ai_score") or 0) > rec["news_max_score"]:
                    rec["news_max_score"] = float(art.get("ai_score") or 0)
                    rec["top_headline"] = (art.get("title") or "")[:140]
                    rec["top_url"] = art.get("url")
                if (art.get("urgency") or 0) >= 1:
                    rec["news_urgent"] = True
        for tk in held:
            candidates.setdefault(tk, {
                "ticker": tk,
                "news_count": 0,
                "news_max_score": 0.0,
                "news_urgent": False,
                "top_headline": None,
                "top_url": None,
            })

        # Pull quant signals in bulk (cached 5min).
        from . import market as _mkt
        try:
            from .strategy import get_quant_signals_live
            tickers = list(candidates.keys())
            quant = get_quant_signals_live(tickers) if tickers else {}
        except Exception:
            quant = {}

        # Live prices (bulk fetch from market.get_prices, cached 30s).
        try:
            prices = _mkt.get_prices(list(candidates.keys())) if candidates else {}
        except Exception:
            prices = {}

        out = []
        for tk, c in candidates.items():
            q = quant.get(tk, {})
            action, conviction, notes = _classify_action(
                tk,
                held.get(tk, 0.0),
                q,
                c["news_max_score"],
                c["news_urgent"],
            )
            out.append({
                "ticker": tk,
                "action": action,
                "conviction": round(conviction, 2),
                "price": prices.get(tk),
                "held_qty": held.get(tk, 0.0),
                "position_pl_pct": position_pl.get(tk),
                "news_count": c["news_count"],
                "news_max_score": round(c["news_max_score"], 1),
                "news_urgent": c["news_urgent"],
                "top_headline": c["top_headline"],
                "top_url": c["top_url"],
                "rsi": q.get("RSI"),
                "macd": q.get("MACD"),
                "mom_5d": q.get("mom_5d"),
                "mom_20d": q.get("mom_20d"),
                "reasons": notes,
            })

        # Rank: action priority then conviction.
        priority = {"EXIT": 0, "TRIM": 1, "BUY": 2, "ADD": 3, "WATCH": 4, "HOLD": 5}
        out.sort(key=lambda r: (priority.get(r["action"], 9), -r["conviction"]))
        out = out[:20]

        action_counts: dict[str, int] = {}
        for r in out:
            action_counts[r["action"]] = action_counts.get(r["action"], 0) + 1

        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_candidates": len(candidates),
            "n_signals_used": len(top_signals),
            "action_counts": action_counts,
            "suggestions": out,
        })
    except Exception as e:
        return jsonify({"error": str(e), "suggestions": []}), 500


# ───────── Feature-dev additions (2026-05-14 part 2) ─────────
# /api/greeks         — portfolio-wide option Greeks (delta/gamma/theta/vega)
# /api/sector-heatmap — DRAM/semis bucket momentum + relative strength + news
# /api/news-deduped   — top signals after dedup + urgency decay (kills syndication noise)


@app.route("/api/greeks")
def greeks_api():
    """Per-leg and portfolio-wide Black-Scholes Greeks for open option positions.

    Stocks contribute pure delta. Options use implied vol from the live yfinance
    chain (DEFAULT_IV fallback when the chain has nothing useful)."""
    try:
        from .analytics.greeks import compute_position_greeks
        store = get_store()
        positions = store.open_positions()
        result = compute_position_greeks(positions)
        # Quick portfolio-level summary so callers don't have to recompute.
        total_value = float(store.get_portfolio().get("total_value") or 0.0)
        totals = result.get("totals", {})
        if total_value > 0:
            result["totals"]["delta_pct_port"] = round(
                totals.get("gross_notional", 0) / total_value * 100, 2
            )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _scorer_verdict(pred: float) -> str:
    """Bucket a predicted 5-day return into a coarse verdict label."""
    if pred >= 3.0:
        return "STRONG_HOLD"
    if pred >= 1.0:
        return "HOLD"
    if pred >= -1.0:
        return "NEUTRAL"
    if pred >= -3.0:
        return "TRIM"
    return "EXIT"


@app.route("/api/scorer-predictions")
def scorer_predictions_api():
    """DecisionScorer prediction per currently-held stock position.

    Builds a feature vector from live RSI/MACD/momentum + news sentiment for
    each held ticker, runs the trained scorer, and returns predicted 5-day
    forward return %. When the scorer isn't trained yet (<500 outcomes), the
    response still lists positions but ``is_trained`` is False so the UI can
    grey them out."""
    try:
        from .ml.decision_scorer import DecisionScorer
        from .strategy import get_quant_signals_live
        from . import signals as _sig
        from . import market as _mkt

        scorer = DecisionScorer()

        store = get_store()
        positions = store.open_positions()
        held_tickers = sorted({
            p["ticker"] for p in positions
            if p.get("type") == "stock" and (p.get("qty") or 0) > 0
        })

        result = {
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "is_trained": scorer.is_trained,
            "n_train": scorer.n_train,
            "gate_threshold": 500,
            "predictions": [],
        }
        if not held_tickers:
            return jsonify(result)

        # Live RSI / MACD / momentum — same source the live trader uses.
        quant = get_quant_signals_live(held_tickers) or {}
        # News-based "ml_score" proxy — average ai_score across mentions in the
        # last 4 hours. Matches the feature the model was trained on, since
        # backtest decisions used ml_score from articles in the same window.
        sent_list = _sig.ticker_sentiments(held_tickers, hours=4) or []
        sent_by_tk = {s["ticker"]: s for s in sent_list}

        # Crude regime proxy — SPY 5d momentum as the multiplier seed. Falls
        # back to 1.0 when unavailable so prediction still returns sensible.
        regime_mult = 1.0
        try:
            spy_q = get_quant_signals_live(["SPY"]).get("SPY") or {}
            spy_mom = spy_q.get("mom_5d")
            if isinstance(spy_mom, (int, float)):
                # Map roughly: +2% = bull (1.15), -2% = bear (0.85)
                regime_mult = max(0.7, min(1.3, 1.0 + spy_mom * 0.075))
        except Exception:
            pass

        preds = []
        for tk in held_tickers:
            q = quant.get(tk) or {}
            sent = sent_by_tk.get(tk) or {}
            # Use max_score for ml_score proxy — captures the strongest signal
            # in the window rather than diluting by averaging across mentions.
            ml_score = float(sent.get("max_score") or 0.0)
            # predict_with_meta (not predict) so the response can flag when
            # the point estimate is a clamped ±50 floor/ceiling from an
            # off-distribution extrapolation. A bare clamped -50 reads as a
            # confident EXIT otherwise — and the unified conviction board
            # pins its ML axis to it. AGENTS.md documents this contract.
            meta = scorer.predict_with_meta(
                ml_score=ml_score,
                rsi=q.get("rsi"),
                macd=q.get("macd_signal"),
                mom5=q.get("mom_5d"),
                mom20=q.get("mom_20d"),
                regime_mult=regime_mult,
                ticker=tk,
                vol_ratio=q.get("vol_ratio"),
                bb_pos=q.get("bb_position"),
            )
            pred = meta["pred"]
            row = {
                "ticker": tk,
                "pred_5d_return_pct": round(float(pred), 3),
                "verdict": _scorer_verdict(float(pred)),
                "rsi": q.get("RSI"),
                "macd": q.get("MACD"),
                "mom_5d": q.get("mom_5d"),
                "mom_20d": q.get("mom_20d"),
                "ml_news_score": round(ml_score, 2),
                "news_count": sent.get("n", 0),
                "news_urgent": sent.get("urgent", 0),
                "off_distribution": bool(meta["off_distribution"]),
            }
            if meta["off_distribution"]:
                row["raw_pred_5d_return_pct"] = round(float(meta["raw"]), 3)
            preds.append(row)
        # Highest predicted return first so the trader sees winners at the top.
        preds.sort(key=lambda r: -(r["pred_5d_return_pct"] or 0))
        result["n_positions"] = len(preds)
        result["regime_mult"] = round(regime_mult, 3)
        result["predictions"] = preds
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "predictions": []}), 500


@app.route("/api/sector-heatmap")
def sector_heatmap_api():
    """DRAM / semis sector heatmap. Buckets: memory_core, semis_equipment, foundry,
    design, memory_leveraged, optical, etf. Each ticker carries mom_5d, mom_20d,
    RSI, vs_sox_5d, and the 24h news pulse from digital-intern."""
    try:
        from .analytics.sector_heatmap import compute_heatmap
        return jsonify(compute_heatmap())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/news-deduped")
def news_deduped_api():
    """Top signals after dedup + exponential urgency decay.

    Default window: last 6 hours, min_score 4.0. Halflife 4h means urgency=1 at
    t=0 becomes 0.5 at t=4h, 0.25 at t=8h, and falls out at 0.125 (5h+) when the
    default cutoff is 0.5. ?hours= and ?min_score= and ?halflife= are tunable."""
    try:
        from . import signals as _sig
        from .analytics.news_dedup import dedupe_and_decay
        hours = int(request.args.get("hours", 6))
        min_score = float(request.args.get("min_score", 4.0))
        halflife = float(request.args.get("halflife", 4.0))
        # Pull a fat candidate list — dedup will compress it heavily.
        raw = _sig.get_top_signals(n=80, hours=hours, min_score=min_score)
        cleaned = dedupe_and_decay(raw, halflife_hours=halflife, min_effective=0.0)
        # Compute the "compression ratio" for the UI so the user can see how
        # much noise was suppressed.
        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_raw": len(raw),
            "n_after_dedup": len(cleaned),
            "compression_ratio": round(len(raw) / max(len(cleaned), 1), 2),
            "halflife_hours": halflife,
            "articles": cleaned[:30],
        })
    except Exception as e:
        return jsonify({"error": str(e), "articles": []}), 500


# ───────── Feature-dev additions (2026-05-15) ─────────
# /api/position-thesis  — per-position integrated card (news, scorer, technicals, last decision, verdict)
# /api/calibration       — confidence calibration + signal-source attribution from realized trades
# /api/drawdown          — current DD anatomy: peak/trough, time-in-DD, per-position contribution


@app.route("/api/position-thesis")
def position_thesis_api():
    """Per-open-position thesis cards.

    Combines DecisionScorer prediction, live quant signals, news pulse from
    digital-intern, and the most recent Opus decision that touched the ticker.
    Each card carries a coarse verdict and a one-line thesis."""
    try:
        from .analytics.position_thesis import build_thesis_cards
        from .ml.decision_scorer import DecisionScorer
        from .strategy import get_quant_signals_live
        from . import signals as _sig

        store = get_store()
        positions = store.open_positions()
        held = sorted({p["ticker"] for p in positions
                       if p.get("type") == "stock" and (p.get("qty") or 0) > 0})

        # Reuse the same scorer prediction shape as /api/scorer-predictions
        # without duplicating its logic — call into the live trader helpers.
        quant = get_quant_signals_live(held) if held else {}
        sent_list = _sig.ticker_sentiments(held, hours=4) if held else []
        sent_by_tk = {s["ticker"]: s for s in sent_list}

        regime_mult = 1.0
        try:
            spy_q = (get_quant_signals_live(["SPY"]) or {}).get("SPY") or {}
            mm = spy_q.get("mom_5d")
            if isinstance(mm, (int, float)):
                regime_mult = max(0.7, min(1.3, 1.0 + mm * 0.075))
        except Exception:
            pass

        scorer = DecisionScorer()
        scorer_preds = []
        for tk in held:
            q = quant.get(tk) or {}
            sent = sent_by_tk.get(tk) or {}
            # Mirror /api/scorer-predictions exactly so both endpoints agree:
            # the scorer wants numeric macd_signal, not the "bullish"/"bearish"
            # MACD label (which _to_float silently zeroes).
            meta = scorer.predict_with_meta(
                ml_score=float(sent.get("max_score") or 0.0),
                rsi=q.get("rsi"), macd=q.get("macd_signal"),
                mom5=q.get("mom_5d"), mom20=q.get("mom_20d"),
                regime_mult=regime_mult, ticker=tk,
                vol_ratio=q.get("vol_ratio"), bb_pos=q.get("bb_position"),
            )
            row = {
                "ticker": tk,
                "pred_5d_return_pct": round(float(meta["pred"]), 3),
                "verdict": _scorer_verdict(float(meta["pred"])),
                "off_distribution": bool(meta["off_distribution"]),
            }
            if meta["off_distribution"]:
                row["raw_pred_5d_return_pct"] = round(float(meta["raw"]), 3)
            scorer_preds.append(row)

        decisions = store.recent_decisions(limit=80)
        out = build_thesis_cards(positions, decisions, scorer_preds, quant)
        out["scorer_trained"] = scorer.is_trained
        out["scorer_n_train"] = scorer.n_train
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e), "cards": []}), 500


@app.route("/api/calibration")
def calibration_api():
    """Confidence calibration + signal-source attribution.

    Buckets matched-and-closed BUY decisions by Opus's stated confidence
    (0.0-0.5, 0.5-0.65, 0.65-0.8, 0.8-1.0) and computes win rate + avg return
    per bucket. Also classifies decisions by reasoning keywords into
    news/technical/mixed/other and computes the same stats per source."""
    try:
        from .analytics.calibration import build_calibration
        store = get_store()
        decisions = store.recent_decisions(limit=500)
        trades = store.recent_trades(limit=500)
        return jsonify(build_calibration(decisions, trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/drawdown")
def drawdown_api():
    """Drawdown anatomy: peak/trough, time-in-DD, per-position contribution.

    Returns a structured 0% when the portfolio is at a fresh high so the UI
    can show a green high-water badge. ``recovery_pct`` measures how much of
    the trough has been clawed back."""
    try:
        from .analytics.drawdown import compute_drawdown
        store = get_store()
        eq = store.equity_curve(limit=2000)
        positions = store.open_positions()
        return jsonify(compute_drawdown(eq, positions))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/earnings-risk")
def earnings_risk_api():
    """Upcoming earnings cross-referenced against held positions + watchlist.

    Earnings are the #1 scheduled risk event — a position into a print can gap
    10%+ overnight. This pulls digital-intern's earnings calendar (:8080) and
    flags which holdings and watchlist names report soon, with a risk tier:
      HELD_IMMINENT  — you hold it and it reports within 3 days
      HELD_SOON      — you hold it and it reports within the horizon
      WATCH          — on the watchlist, not held
    """
    import json as _json
    import urllib.request as _urllib

    try:
        store = get_store()
        positions = store.open_positions()
        held: dict[str, float] = {}
        for p in positions:
            t = (p.get("ticker") or "").upper()
            if not t:
                continue
            mult = 100 if p.get("type") in ("call", "put") else 1
            price = p.get("current_price") or p.get("avg_cost") or 0.0
            held[t] = held.get(t, 0.0) + price * (p.get("qty") or 0.0) * mult

        try:
            from .strategy import WATCHLIST as _WATCHLIST
            watch = {t.upper() for t in _WATCHLIST}
        except Exception:
            watch = set()

        events = []
        source_ok = True
        try:
            with _urllib.urlopen(
                "http://127.0.0.1:8080/api/earnings", timeout=4) as resp:
                snap = _json.loads(resp.read().decode("utf-8"))
            events = snap.get("events") or []
        except Exception:
            source_ok = False

        out = []
        for ev in events:
            tk = (ev.get("ticker") or "").upper()
            if not tk:
                continue
            days = ev.get("days_away")
            in_port = tk in held
            on_watch = tk in watch
            if not in_port and not on_watch:
                continue
            if in_port and days is not None and days <= 3:
                tier = "HELD_IMMINENT"
            elif in_port:
                tier = "HELD_SOON"
            else:
                tier = "WATCH"
            out.append({
                "ticker": tk,
                "earnings_date": ev.get("earnings_date"),
                "days_away": days,
                "tier": tier,
                "held": in_port,
                "exposure_usd": round(held.get(tk, 0.0), 2) if in_port else 0.0,
            })
        # Held + soonest first; tier rank keeps imminent risk at the top.
        tier_rank = {"HELD_IMMINENT": 0, "HELD_SOON": 1, "WATCH": 2}
        out.sort(key=lambda e: (tier_rank.get(e["tier"], 9),
                                e["days_away"] if e["days_away"] is not None else 1e9))
        held_at_risk = round(sum(e["exposure_usd"] for e in out if e["held"]), 2)
        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_ok": source_ok,
            "events": out,
            "n_held_reporting": sum(1 for e in out if e["held"]),
            "n_imminent": sum(1 for e in out if e["tier"] == "HELD_IMMINENT"),
            "held_exposure_at_risk_usd": held_at_risk,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ───────── Feature-dev additions (2026-05-15, agent 4) ─────────
# /api/scorer-confidence — empirical ± bands + directional hit-rate for the
#                          DecisionScorer, so its point predictions can be
#                          trusted (or distrusted) with a real error bar.
# /api/decision-health   — is the live Opus trader actually deciding? Surfaces
#                          the NO_DECISION (parse-failure) rate the dashboard
#                          otherwise hides entirely.


def _live_scorer_predictions(scorer) -> list[dict]:
    """Predicted 5d return for each held stock position (live feature vector).

    Same feature construction as ``/api/scorer-predictions`` — kept as a shared
    helper so the confidence endpoint stays in lockstep with the original."""
    from .strategy import get_quant_signals_live
    from . import signals as _sig

    store = get_store()
    held = sorted({
        p["ticker"] for p in store.open_positions()
        if p.get("type") == "stock" and (p.get("qty") or 0) > 0
    })
    if not held:
        return []
    quant = get_quant_signals_live(held) or {}
    sent_by_tk = {s["ticker"]: s for s in (_sig.ticker_sentiments(held, hours=4) or [])}
    regime_mult = 1.0
    try:
        spy_mom = (get_quant_signals_live(["SPY"]).get("SPY") or {}).get("mom_5d")
        if isinstance(spy_mom, (int, float)):
            regime_mult = max(0.7, min(1.3, 1.0 + spy_mom * 0.075))
    except Exception:
        pass
    preds = []
    for tk in held:
        q = quant.get(tk) or {}
        sent = sent_by_tk.get(tk) or {}
        ml_score = float(sent.get("max_score") or 0.0)
        meta = scorer.predict_with_meta(
            ml_score=ml_score, rsi=q.get("rsi"), macd=q.get("macd_signal"),
            mom5=q.get("mom_5d"), mom20=q.get("mom_20d"), regime_mult=regime_mult,
            ticker=tk, vol_ratio=q.get("vol_ratio"), bb_pos=q.get("bb_position"),
        )
        pred = meta["pred"]
        row = {
            "ticker": tk,
            "pred_5d_return_pct": round(float(pred), 3),
            "verdict": _scorer_verdict(float(pred)),
            "rsi": q.get("RSI"), "mom_5d": q.get("mom_5d"), "mom_20d": q.get("mom_20d"),
            # Honesty flag: True ⇒ the model extrapolated past the empirical
            # label support, pred is a clamped floor/ceiling, and the verdict
            # should be read as "weak/low-trust", not a confident -50%.
            "off_distribution": bool(meta["off_distribution"]),
        }
        if meta["off_distribution"]:
            row["raw_pred_5d_return_pct"] = round(float(meta["raw"]), 3)
        preds.append(row)
    return preds


def _load_decision_outcomes(max_rows: int = 4000) -> list[dict]:
    """Tail of data/decision_outcomes.jsonl — the scorer's own training history."""
    import json as _json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "decision_outcomes.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(_json.loads(ln))
        except Exception:
            continue
    return rows[-max_rows:]


@app.route("/api/scorer-confidence")
def scorer_confidence_api():
    """Empirical prediction intervals + reliability for the DecisionScorer.

    Replays the trained scorer over its own outcome history to measure how far
    its predictions actually land from realized returns. Returns a calibration
    table (residual P10/P50/P90 + directional hit-rate per prediction band) and,
    for each held stock position, the live prediction wrapped in an empirical
    [low, high] band drawn from the matching band's residual quantiles."""
    try:
        from .ml.decision_scorer import DecisionScorer
        from .analytics.scorer_confidence import build_scorer_confidence, interval_for

        scorer = DecisionScorer()
        outcomes = _load_decision_outcomes()
        conf = build_scorer_confidence(outcomes, scorer)

        positions = []
        if conf.get("overall"):
            for p in _live_scorer_predictions(scorer):
                iv = interval_for(p["pred_5d_return_pct"], conf)
                positions.append({**p, "interval": iv})
            positions.sort(key=lambda r: -(r["pred_5d_return_pct"] or 0))
        conf["positions"] = positions
        return jsonify(conf)
    except Exception as e:
        return jsonify({"error": str(e), "buckets": [], "positions": []}), 500


def _parse_action_ticker(action_taken: str) -> tuple[str, str | None]:
    """Pull the (verb, ticker) out of a decisions.action_taken string.

    The column is free-text in the form 'BUY NVDA → FILLED' / 'HOLD MU → HOLD'
    / 'NO_DECISION'. Returns ('NO_DECISION', None) for malformed / sentinel
    rows so callers don't have to special-case them."""
    if not action_taken or action_taken in ("NO_DECISION", "BLOCKED"):
        return action_taken or "", None
    head = action_taken.split("→")[0].strip()
    parts = head.split()
    if not parts:
        return "", None
    verb = parts[0].upper()
    ticker = parts[1].upper() if len(parts) >= 2 else None
    if ticker in ("CASH", "NONE", ""):
        ticker = None
    return verb, ticker


_BUY_VERBS = {"BUY", "BUY_CALL", "BUY_PUT", "REBALANCE"}
_SELL_VERBS = {"SELL", "SELL_CALL", "SELL_PUT"}


def _classify_disagreement(verdict: str, last_verb: str | None) -> tuple[str, str]:
    """Map (scorer verdict, last Opus action verb on the same ticker) → (severity, label).

    HIGH = scorer says EXIT/TRIM while Opus is still adding or holding —
           the trader is fighting its own ML safety net.
    MEDIUM = scorer says NEUTRAL but Opus is leaning bullish, or scorer says
             STRONG_HOLD but Opus just sold.
    ALIGNED = the two agree (either both bullish or both bearish)."""
    verdict = (verdict or "").upper()
    verb = (last_verb or "").upper()
    bearish_scorer = verdict in ("EXIT", "TRIM")
    bullish_scorer = verdict in ("STRONG_HOLD", "HOLD")
    bullish_action = verb in _BUY_VERBS or verb == "HOLD"
    bearish_action = verb in _SELL_VERBS
    if bearish_scorer and bullish_action:
        return "HIGH", "scorer says exit, Opus still long"
    if bullish_scorer and bearish_action:
        return "MEDIUM", "scorer says hold, Opus exited"
    if verdict == "NEUTRAL" and verb in _BUY_VERBS:
        return "MEDIUM", "scorer neutral, Opus added"
    return "ALIGNED", "scorer and Opus agree"


@app.route("/api/disagreement")
def disagreement_api():
    """Where the scorer and Opus diverge on currently-held positions.

    For every open stock position, compare the scorer's verdict (drawn from
    /api/scorer-confidence's empirical-band logic) against the most recent
    parsed action that Opus took on the same ticker. A HIGH-severity row is
    a red flag: the trader is overriding the ML safety net. Used by the
    command-center and intended as a 'why is the portfolio losing money?'
    diagnostic when scorer/Opus drift apart silently."""
    try:
        from .ml.decision_scorer import DecisionScorer
        from .analytics.scorer_confidence import build_scorer_confidence, interval_for

        scorer = DecisionScorer()
        outcomes = _load_decision_outcomes()
        conf = build_scorer_confidence(outcomes, scorer)
        scorer_rows = _live_scorer_predictions(scorer) if conf.get("overall") else []

        # Last action verb per ticker — walk recent decisions newest first so we
        # capture the most recent Opus stance on each holding. Skip NO_DECISION
        # rows so a parse-failure storm doesn't blank the panel.
        store = get_store()
        last_verb: dict[str, str] = {}
        last_ts: dict[str, str] = {}
        for d in store.recent_decisions(limit=500):
            verb, tk = _parse_action_ticker(d.get("action_taken") or "")
            if not tk or verb == "NO_DECISION":
                continue
            if tk in last_verb:
                continue
            last_verb[tk] = verb
            last_ts[tk] = d.get("timestamp") or ""

        rows = []
        for p in scorer_rows:
            tk = p["ticker"]
            verb = last_verb.get(tk)
            severity, label = _classify_disagreement(p.get("verdict", ""), verb)
            iv = interval_for(p["pred_5d_return_pct"], conf) if conf.get("overall") else None
            drow = {
                "ticker": tk,
                "scorer_verdict": p.get("verdict"),
                "scorer_pred_5d_pct": p.get("pred_5d_return_pct"),
                "last_action": verb,
                "last_action_ts": last_ts.get(tk),
                "severity": severity,
                "label": label,
                "interval": iv,
                # Carry the honesty flag through so a HIGH-severity row
                # driven by a clamped extrapolation can be visually
                # de-weighted rather than read as a real scorer/Opus fight.
                "off_distribution": bool(p.get("off_distribution", False)),
            }
            if p.get("off_distribution"):
                drow["raw_pred_5d_return_pct"] = p.get("raw_pred_5d_return_pct")
            rows.append(drow)
        severity_order = {"HIGH": 0, "MEDIUM": 1, "ALIGNED": 2}
        rows.sort(key=lambda r: (severity_order.get(r["severity"], 9),
                                 r["scorer_pred_5d_pct"] or 0))
        counts = {"HIGH": 0, "MEDIUM": 0, "ALIGNED": 0}
        for r in rows:
            counts[r["severity"]] = counts.get(r["severity"], 0) + 1
        return jsonify({
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scorer_trained": bool(conf.get("overall")),
            "n_positions": len(rows),
            "counts": counts,
            "rows": rows,
        })
    except Exception as e:
        return jsonify({"error": str(e), "rows": []}), 500


@app.route("/api/validation")
def validation_api():
    """Signal Integrity validation results — permutation tests + label audits.

    Backed by data/validation_results.json which is appended to by the
    continuous loop's background validation runner. Returns the full history
    (capped at 50 entries on the writer side); the dashboard renders the
    most recent entry."""
    p = Path(__file__).resolve().parent.parent / "data" / "validation_results.json"
    if not p.exists():
        return jsonify({"results": []})
    try:
        return jsonify({"results": json.loads(p.read_text())})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)}), 500


@app.route("/api/decision-health")
def decision_health_api():
    """Health of the live decision pipeline — action mix, parse-failure rate,
    confidence trend, cadence. Surfaces NO_DECISION ('claude returned no
    parseable JSON') cycles that no other dashboard panel exposes."""
    try:
        from .analytics.decision_health import build_decision_health
        decisions = get_store().recent_decisions(limit=2000)
        rep = build_decision_health(decisions)
        # Surface a rolling-24h NO_DECISION rate as a top-level convenience
        # field so monitoring/alerting uses the *current* parse-failure rate,
        # not the all-time `action_mix` % that legacy failures permanently
        # inflate. The 24h window is already computed in build_decision_health
        # (windows.24h); we just hoist it. `_enough` mirrors the verdict
        # logic's >=10-sample gate so a fresh restart (few/no 24h decisions)
        # doesn't silently clear the alert via a tiny, noisy sample.
        try:
            w24 = (rep.get("windows") or {}).get("24h") or {}
            total24 = int(w24.get("total") or 0)
            nd24 = int(w24.get("no_decision") or 0)
            rep["no_decision_rate_24h"] = float(w24.get("parse_fail_pct") or 0.0)
            rep["no_decision_n_24h"] = nd24
            rep["n_decisions_24h"] = total24
            # True only when there is enough recent signal to trust the rate.
            rep["no_decision_24h_significant"] = total24 >= 10
        except Exception:
            rep["no_decision_rate_24h"] = None
            rep["no_decision_n_24h"] = None
            rep["n_decisions_24h"] = None
            rep["no_decision_24h_significant"] = False
        return jsonify(rep)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decision-forensics")
def decision_forensics_api():
    """*Why* the live trader produces no decision — failure-mode taxonomy.

    decision-health says HOW OFTEN parsing fails; this says WHY: timeout vs
    truncation vs prose-wrapping vs fenced vs malformed, the open/closed-market
    split, an hourly trend, retry-exhausted count, an actionable hint, and the
    raw model excerpts strategy.py captured but nothing else surfaces."""
    try:
        from .analytics.decision_forensics import build_decision_forensics
        decisions = get_store().recent_decisions(limit=2000)
        return jsonify(build_decision_forensics(decisions))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/liquidity")
def liquidity_api():
    """Capital deployment & liquidity — is the book pinned with no dry powder?

    Cash vs deployed %, position weights, unrealized P/L, days since the last
    opening trade, and a status (NO_DRY_POWDER / DRY_POWDER_LOW / BALANCED /
    CASH_HEAVY) with human flags. Complements /api/risk (concentration) with
    the 'can the trader still act on a signal?' view."""
    try:
        from .analytics.liquidity import build_liquidity
        store = get_store()
        return jsonify(build_liquidity(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(200),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decision-drought")
def decision_drought_api():
    """What the live trader's *inaction* cost — drift during decision droughts.

    decision-health gives the NO_DECISION rate; decision-forensics gives the
    WHY. This gives the COST: between FILLED trades, segment cycles into
    droughts, price each one's portfolio drift vs the S&P from the equity
    curve, and split involuntary (NO_DECISION/parse-failure) PARALYSIS from
    DELIBERATE_HOLD. The negative alpha of the paralysis droughts is
    'involuntary alpha bleed' — the parse-failure problem in P&L terms."""
    try:
        from .analytics.decision_drought import build_decision_drought
        store = get_store()
        return jsonify(build_decision_drought(
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/capital-paralysis")
def capital_paralysis_api():
    """Trap + cost + unlock — why the book is stuck and the way out.

    /api/liquidity sees the trap (no dry powder), /api/decision-drought sees
    the cost (alpha bled while pinned), /api/suggestions lists ideas it can't
    fund — none connect them. This composes liquidity + decision-drought
    (single source of truth, no re-derived metrics) and adds the unlock
    ladder: positions ranked in desk cut-priority (biggest loser first), each
    rung showing the cash a sale frees, deployed-% after, and whether that
    single sale restores the ability to act on a fresh signal. Advisory only —
    never gates Opus, adds no caps (AGENTS.md invariant #2)."""
    try:
        from .analytics.capital_paralysis import build_capital_paralysis
        store = get_store()
        return jsonify(build_capital_paralysis(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(200),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/open-attribution")
def open_attribution_api():
    """Selection-vs-market on the *open* book — the bot's dominant return.

    /api/analytics & /api/performance-attribution cover *closed* round-trips,
    but the live trader mostly HOLDs, so its return is dominated by open
    drift vs SPY — invisible until now. Per open stock position: return since
    opened_at, SPY return over the same window (anchored to the equity curve's
    sp500_price at-or-after entry), and alpha in % and $. Options are flagged
    and skipped (alpha-vs-SPY doesn't fit Greeks — /api/backtests/compare
    precedent)."""
    try:
        from .analytics.open_attribution import build_open_attribution
        store = get_store()
        return jsonify(build_open_attribution(
            store.open_positions(),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trade-asymmetry")
def trade_asymmetry_api():
    """Behavioural-edge pathology — the exit/sizing failure behind the P&L.

    /api/analytics gives the raw aggregates (win_rate, profit_factor, $ avgs);
    /api/calibration asks whether the confidence axis is accurate. Neither
    answers the desk question: given my payoff ratio, what win-rate do I need
    to break even, am I above or below it, and am I cutting winners faster
    than losers (the disposition effect)? This composes the single source of
    truth (build_round_trips, AGENTS.md #10) into payoff ratio, per-trade
    expectancy, breakeven-vs-actual win-rate, and the winner/loser hold-time
    disposition gap. The verdict label is withheld until n≥20 round-trips
    (news-edge INSUFFICIENT_DATA idiom) so a five-trade read can't mislead.
    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.trade_asymmetry import build_trade_asymmetry
        store = get_store()
        # Same trades convention as /api/analytics: oldest → newest.
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_trade_asymmetry(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/loser-autopsy")
def loser_autopsy_api():
    """Per-closed-losing-round-trip post-mortem — the desk question no panel
    answers. /api/thesis-drift re-tests *open* positions against their
    opening rationale; /api/trade-asymmetry gives *aggregate* payoff math;
    /api/churn counts re-entry *cadence*. None tell the story of *why each
    closed trade lost*. This composes the single source of truth
    (build_round_trips, AGENTS.md #10) and joins the verbatim entry/exit
    reason back from the contributing trade rows, classifies each loss into
    an objective failure mode (KNIFE_CATCH / WHIPSAW / SLOW_BLEED /
    STOPPED_OUT), and rolls up which name is the bleed + which mode
    dominates. The pattern verdict is withheld until n≥8 losers
    (trade_asymmetry STABLE idiom). Advisory only — never gates Opus, adds
    no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.loser_autopsy import build_loser_autopsy
        store = get_store()
        # Same trades convention as /api/analytics & /api/trade-asymmetry:
        # oldest → newest (build_round_trips reads in sequence).
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_loser_autopsy(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/correlation")
def correlation_api():
    """Concentration honesty — do the held names actually move together?
    /api/risk reports name-level concentration + a single SPY-shock; it
    cannot see whether the book is really one *factor* bet. This computes
    pairwise return correlation among held stock positions, the
    weight-Herfindahl effective-position count, and the
    correlation-adjusted effective number of *independent* bets (collapses
    toward 1 as the names co-move however many tickers are on the book).
    Options are flagged & skipped (the open_attribution "stocks only"
    carve-out). The builder is pure; the yfinance fetch lives here (the
    thesis_drift split) and degrades to INSUFFICIENT, never an error.
    Advisory only — never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.correlation import build_correlation
        store = get_store()
        positions = store.open_positions()
        poss, price_history = [], {}
        for p in positions:
            ptype = p.get("type")
            kind = ptype if ptype in ("call", "put") else "stock"
            mult = 100 if kind in ("call", "put") else 1
            price = p.get("current_price") or p.get("avg_cost") or 0.0
            qty = float(p.get("qty") or 0.0)
            poss.append({
                "ticker": p.get("ticker"),
                "market_value": round(float(price) * qty * mult, 2),
                "type": kind,
            })
            if kind == "stock" and p.get("ticker") not in price_history:
                try:
                    bars = _daily_history_cached(p["ticker"], "3mo")
                    price_history[p["ticker"]] = [c for _, c in bars]
                except Exception:
                    price_history[p["ticker"]] = []
        return jsonify(build_correlation(poss, price_history))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/churn")
def churn_api():
    """Overtrading & same-name re-entry churn — the turnover question.

    /api/analytics shows raw aggregates; /api/trade-asymmetry grades the
    *payoff* pathology (DISPOSITION_BLEED, breakeven-vs-actual win-rate).
    Neither measures how often the book re-buys a name it just fully
    closed (the live NVDA→LITE→NVDA shape on 2026-05-16) nor the
    round-trips-per-active-day cadence. This composes the single source of
    truth (build_round_trips, AGENTS.md #10 — no re-derived P&L) into the
    fast-re-entry count/rate, cadence, sub-day-loss concentration, and a
    CHURNING / ACTIVE_TURNOVER / BUY_AND_HOLD verdict withheld until
    n≥20 round-trips (trade_asymmetry STABLE idiom). Advisory only — never
    gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.churn import build_churn
        store = get_store()
        # Same trades convention as /api/analytics & /api/trade-asymmetry:
        # oldest → newest (build_round_trips reads in sequence).
        trades = list(reversed(store.recent_trades(2000)))
        return jsonify(build_churn(trades))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/thesis-drift")
def thesis_drift_api():
    """Entry-thesis vs current-reality, per open position.

    /api/position-thesis fuses *current* signals; /api/suggestions
    re-derives an action from scratch. Neither re-tests a holding against
    *the reason it was opened for* — which is sitting verbatim in the
    opening fill's trades.reason. This anchors each open position on its
    own opening BUY rationale (invariant #8: the BUY nearest opened_at is
    this lot's opener even on a re-entered name) and grades INTACT /
    WEAKENING / BROKEN off objective, deterministic inputs (P/L since
    entry, hold time, and optional live quant/news). Advisory only —
    never gates Opus, adds no caps (AGENTS.md #2/#12)."""
    try:
        from .analytics.thesis_drift import build_thesis_drift
        store = get_store()
        positions = store.open_positions()
        trades = store.recent_trades(2000)
        signals = None
        try:
            tickers = sorted({p["ticker"] for p in positions
                              if p.get("ticker")})
            if tickers:
                from .strategy import get_quant_signals_live
                quant = get_quant_signals_live(tickers) or {}
                news = _ticker_news_pulse(tickers, hours=24)
                signals = {}
                for tk in tickers:
                    q = quant.get(tk, {}) or {}
                    nrec = news.get(tk.upper(), {}) or {}
                    signals[tk] = {
                        "rsi": q.get("RSI"),
                        "macd": q.get("MACD"),
                        "mom_5d": q.get("mom_5d"),
                        "mom_20d": q.get("mom_20d"),
                        "news_count": nrec.get("n", 0),
                        "news_urgent": bool(nrec.get("urgent", 0)),
                    }
        except Exception:
            signals = None  # builder degrades to price-only health
        return jsonify(build_thesis_drift(positions, trades, signals))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/self-review")
def self_review_api():
    """The behavioural mirror the live trader now sees in its own prompt.

    Canonical single source (AGENTS.md invariant #10): composes
    build_trade_asymmetry + build_capital_paralysis + build_open_attribution
    verbatim — no re-derived P&L — into one report plus the exact
    `prompt_block` string injected into strategy._build_payload every live
    decision cycle. Observational only: it states facts and the builders' own
    calibrated verdicts, issues no directives, imposes no caps, and reaffirms
    full autonomy in its own preamble — it does not violate the "no hard risk
    limits / Opus has full autonomy" invariant (#2/#12), exactly as
    /api/capital-paralysis and /api/liquidity are advisory-only. Exposing it
    here keeps the dashboard, a future chat single-source and the in-prompt
    block from ever drifting apart (the inline-copy hazard #10 warns of)."""
    try:
        from .analytics.self_review import build_self_review
        store = get_store()
        # trades store-native newest-first — build_self_review reverses
        # internally for the asymmetry consumer, exactly as the two endpoints
        # above do (build_liquidity wants newest-first, build_round_trips
        # wants oldest→newest).
        return jsonify(build_self_review(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(2000),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scorecard")
def scorecard_api():
    """Behavioural-verdict alignment router across the five pure
    behavioural builders (trade_asymmetry, churn, capital_paralysis,
    decision_reliability, open_attribution).

    Synthesis without a new opinion: it classifies each builder's *own*
    verdict into FLAG/OK/IMMATURE, counts where independent checks concur on
    a theme, and forwards the builders' own headlines verbatim (single source
    of truth, AGENTS.md invariant #10). No grade, no directive, no cap —
    descriptive only, exactly the /api/self-review observational precedent
    (invariants #2/#12). Unlike self-review it is NOT injected into the live
    decision prompt; it is dashboard/chat only. Same store reads as
    /api/self-review so the two can't drift."""
    try:
        from .analytics.trader_scorecard import build_trader_scorecard
        store = get_store()
        return jsonify(build_trader_scorecard(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(2000),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Daily-bar history cache for the news-edge endpoint. Keyed by ticker; daily
# bars don't change intraday in a way that matters for forward-return analysis
# of *past* articles, so a generous TTL keeps the endpoint snappy after the
# first build without re-hammering yfinance every refresh.
_NEWS_EDGE_PX_CACHE: dict[str, tuple[list[tuple[str, float]], float]] = {}
_NEWS_EDGE_PX_TTL = 1800.0  # 30 min


def _daily_history_cached(ticker: str, period: str = "3mo") -> list[tuple[str, float]]:
    import time as _t
    hit = _NEWS_EDGE_PX_CACHE.get(ticker)
    if hit and _t.time() - hit[1] < _NEWS_EDGE_PX_TTL:
        return hit[0]
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        bars = [(idx.strftime("%Y-%m-%d"), float(c))
                for idx, c in zip(h.index, h["Close"]) if c == c]
    except Exception:
        bars = []
    _NEWS_EDGE_PX_CACHE[ticker] = (bars, _t.time())
    return bars


@app.route("/api/news-edge")
def news_edge_api():
    """Does digital-intern's scored news actually predict moves?

    For every live (non-backtest) scored article that names a watchlist
    ticker, look at that ticker's 1/3/5-trading-day forward return, both raw
    and SPY-abnormal, banded by ai_score. The verdict is judged on abnormal
    return only — a flat or inverted score→return curve means the score is
    noise. ``?days=`` (lookback, default 30) and ``?min_score=`` (default 2.0)
    are tunable. Validates the core premise of the whole stack."""
    try:
        from .analytics.news_edge import build_news_edge
        from .strategy import WATCHLIST

        days = max(7, min(120, int(request.args.get("days", 30))))
        min_score = float(request.args.get("min_score", 2.0))

        path = _articles_db_path()
        if path is None:
            return jsonify({"error": "articles.db not found", "bands": [],
                            "verdict": "NO_DATA"}), 200

        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=8)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT title, full_text, ai_score, urgency, first_seen "
                "FROM articles WHERE ai_score >= ? AND first_seen >= ? "
                "AND url NOT LIKE 'backtest://%' "
                "AND source NOT LIKE 'backtest_%' "
                "AND source NOT LIKE 'opus_annotation%' "
                "ORDER BY first_seen DESC LIMIT 4000",
                (min_score, since),
            ).fetchall()
        finally:
            conn.close()

        arts = []
        for r in rows:
            body = ""
            try:
                if r["full_text"]:
                    body = zlib.decompress(r["full_text"]).decode(
                        "utf-8", errors="replace")
            except Exception:
                body = ""
            arts.append({
                "text": f"{r['title'] or ''} {body}",
                "ai_score": r["ai_score"],
                "urgency": r["urgency"],
                "published": r["first_seen"],
            })

        # Only fetch prices for watchlist tickers that actually appear, most
        # frequent first, capped so a cold request can't stall on dozens of
        # yfinance round-trips.
        freq: dict[str, int] = {}
        pats = {tk: re.compile(rf"(?:\$|\b){re.escape(tk)}\b") for tk in WATCHLIST}
        for a in arts:
            up = a["text"].upper()
            for tk, pat in pats.items():
                if pat.search(up):
                    freq[tk] = freq.get(tk, 0) + 1
        wanted = [tk for tk, _ in sorted(
            freq.items(), key=lambda kv: -kv[1])][:30]

        price_history = {tk: _daily_history_cached(tk) for tk in wanted}
        spy_history = _daily_history_cached("SPY")

        result = build_news_edge(arts, price_history, spy_history, WATCHLIST)
        result["lookback_days"] = days
        result["min_score"] = min_score
        result["n_tickers_priced"] = len([tk for tk in wanted
                                          if price_history.get(tk)])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "bands": [],
                        "verdict": "ERROR"}), 500


# ───────── Decision-reliability + funded-suggestions (2026-05-16, agent 4) ─────────
# Appended at the tail of the route section (not interleaved) so a concurrent
# core-review pass editing existing endpoints doesn't collide on merge.


@app.route("/api/decision-reliability")
def decision_reliability_api():
    """The *true current-regime* NO_DECISION rate, not the inflated headline.

    decision-health/forensics/drought measure the rate / why / cost, but the
    headline % is dominated by legacy pre-diagnostics rows that stop accruing
    once the runner restarts onto diagnostic code. This partitions the log at
    the newest legacy failure and reports the post-restart rate with explicit
    sample-size honesty + a restart-recommended signal. Pure composition of
    build_decision_forensics + build_decision_drought (single source of truth,
    no re-derived metrics). Advisory only — never gates Opus (invariants
    #2/#12)."""
    try:
        from .analytics.decision_reliability import build_decision_reliability
        store = get_store()
        return jsonify(build_decision_reliability(
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/funded-suggestions")
def funded_suggestions_api():
    """Pair each unfundable BUY/ADD idea with the sale that funds it.

    Composes the existing /api/suggestions list with build_capital_paralysis'
    unlock ladder (single source of truth, no re-derived metrics — neither is
    refactored). When PINNED, attaches the minimum prefix of desk-cut-priority
    sales whose cumulative freed cash covers an advisory suggested notional.
    Advisory only — never gates Opus, sizes nothing, adds no caps
    (invariants #2/#12)."""
    try:
        from .analytics.capital_paralysis import build_capital_paralysis
        from .analytics.funded_suggestions import build_funded_suggestions

        # Reuse the existing suggestions view verbatim (no refactor).
        resp = suggestions_api()
        if isinstance(resp, tuple):
            resp = resp[0]
        sug_payload = resp.get_json(silent=True) or {}
        suggestions = sug_payload.get("suggestions", [])

        store = get_store()
        paralysis = build_capital_paralysis(
            store.get_portfolio(),
            store.open_positions(),
            store.recent_trades(200),
            store.recent_decisions(limit=3000),
            store.equity_curve(limit=5000),
        )
        out = build_funded_suggestions(suggestions, paralysis)
        # Surface a suggestions-side error rather than masking it as "no ideas".
        if sug_payload.get("error"):
            out["suggestions_error"] = sug_payload["error"]
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signal-followthrough")
def signal_followthrough_api():
    """Is the trader actually *using* its own news edge?

    news-edge grades the signal alone (ignoring the bot); decision-drought
    grades inaction vs SPY (not vs the signals present). This grades the
    *join*: of the high-score live signals visible at decision time (the
    exact ``get_top_signals(hours=2, min_score=4.0)`` window strategy.decide
    feeds Opus), did the trader transact that ticker, and did the signals it
    ACTED on beat — forward, SPY-abnormal — the ones it IGNORED? A near-zero
    follow-through ⇒ IGNORING_FEED; negative selection edge ⇒ MISUSING_SIGNALS.
    ``?days=`` (lookback, default 30) / ``?min_score=`` (default 4.0). Pure
    composition of build_signal_followthrough + news_edge resolution helpers
    (single source of truth). Advisory only — never gates Opus
    (invariants #2/#12)."""
    try:
        from .analytics.signal_followthrough import (
            _fetch_live_articles,
            build_signal_followthrough,
        )
        from .strategy import WATCHLIST

        days = max(7, min(120, int(request.args.get("days", 30))))
        min_score = float(request.args.get("min_score", 4.0))

        path = _articles_db_path()
        if path is None:
            return jsonify({"error": "articles.db not found",
                            "verdict": "NO_DATA", "acted": {}, "ignored": {}}), 200

        store = get_store()
        decs = store.recent_decisions(limit=3000)
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        arts = _fetch_live_articles(str(path), since, min_score=min_score)

        # Price only the watchlist tickers that actually appear in the feed,
        # most-frequent first, capped — same cold-start guard as news-edge.
        freq: dict[str, int] = {}
        pats = {tk: re.compile(rf"(?:\$|\b){re.escape(tk)}\b") for tk in WATCHLIST}
        for a in arts:
            up = a["text"].upper()
            for tk, pat in pats.items():
                if pat.search(up):
                    freq[tk] = freq.get(tk, 0) + 1
        wanted = [tk for tk, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:30]

        price_history = {tk: _daily_history_cached(tk) for tk in wanted}
        spy_history = _daily_history_cached("SPY")

        result = build_signal_followthrough(
            decs, arts, price_history, spy_history, WATCHLIST,
            lookback_hours=2.0, min_score=min_score)
        result["lookback_days"] = days
        result["min_score"] = min_score
        result["n_tickers_priced"] = len([tk for tk in wanted
                                          if price_history.get(tk)])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR",
                        "acted": {}, "ignored": {}}), 500


@app.route("/api/source-edge")
def source_edge_api():
    """Which of digital-intern's ~17 collectors is worth trusting?

    news-edge grades the *score* (8.0 vs 3.0 headline); signal-followthrough
    grades whether the bot *acted*. Neither answers the operator's question:
    of the *sources* feeding the pipeline, whose scored headlines actually
    precede abnormal moves and which are noise to cut/down-weight? This bins
    every scored live article by collector family (the dirty `source` column
    normalised once, see source_edge._source_family) and reports the 1/3/5d
    forward return, raw + SPY-abnormal, pooled across score bands per family.
    Pooled (not per-band) because digital-intern's live news is only days-deep
    — the pooled view is both the actionable one (cut a collector) and the one
    that reaches a usable sample first. ``?days=`` (lookback, default 30) /
    ``?min_score=`` (default 2.0). Verdict matures with history exactly like
    news-edge (NO_DATA → INSUFFICIENT_DATA → EDGE_FOUND/NO_EDGE). Pure
    composition reusing news_edge resolution helpers (single source of truth).
    Advisory only — never gates Opus, adds no caps (invariants #2/#12)."""
    try:
        from .analytics.source_edge import (
            _fetch_source_articles,
            build_source_edge,
        )
        from .strategy import WATCHLIST

        days = max(7, min(120, int(request.args.get("days", 30))))
        min_score = float(request.args.get("min_score", 2.0))

        path = _articles_db_path()
        if path is None:
            return jsonify({"error": "articles.db not found",
                            "sources": [], "verdict": "NO_DATA"}), 200

        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        arts = _fetch_source_articles(str(path), since, min_score=min_score)

        # Price only the watchlist tickers that actually appear, most-frequent
        # first, capped — same cold-start guard as news-edge.
        freq: dict[str, int] = {}
        pats = {tk: re.compile(rf"(?:\$|\b){re.escape(tk)}\b") for tk in WATCHLIST}
        for a in arts:
            up = a["text"].upper()
            for tk, pat in pats.items():
                if pat.search(up):
                    freq[tk] = freq.get(tk, 0) + 1
        wanted = [tk for tk, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:30]

        price_history = {tk: _daily_history_cached(tk) for tk in wanted}
        spy_history = _daily_history_cached("SPY")

        result = build_source_edge(arts, price_history, spy_history, WATCHLIST)
        result["lookback_days"] = days
        result["n_tickers_priced"] = len([tk for tk in wanted
                                          if price_history.get(tk)])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "sources": [],
                        "verdict": "ERROR"}), 500


def _feed_db_probe(db_path: str, want_counts: bool = False) -> dict:
    """Read newest *live* first_seen (and optionally 2h/24h live counts) from
    one candidate articles.db. The live-only clause is inlined verbatim (the
    canonical AGENTS.md invariant #1/#3 fragment, mirroring signals.py and
    data_feed_api) — a planted backtest:// row must never read as the freshest
    article or the split-brain detector would be defeated by training data.
    Returns ``{exists, newest, live_2h, live_24h}``; never raises."""
    out = {"exists": False, "newest": None, "live_2h": 0, "live_24h": 0}
    try:
        from pathlib import Path as _P
        if not _P(db_path).exists():
            return out
        out["exists"] = True
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3.0)
        try:
            live_clause = (
                "url NOT LIKE 'backtest://%' "
                "AND source NOT LIKE 'backtest_%' "
                "AND source NOT LIKE 'opus_annotation%'"
            )
            row = conn.execute(
                f"SELECT MAX(first_seen) FROM articles WHERE {live_clause}"
            ).fetchone()
            out["newest"] = row[0] if row else None
            if want_counts:
                # Cut-offs computed as ISO strings in Python, mirroring
                # signals.get_top_signals exactly — NOT sqlite's
                # datetime('now',...) (space-separated), which would
                # lexically mis-compare against the 'T'-separated ISO
                # first_seen the way data_feed_api's count subtly does.
                now = datetime.now(timezone.utc)
                s2 = (now - timedelta(hours=2)).isoformat()
                s24 = (now - timedelta(hours=24)).isoformat()
                out["live_2h"] = int(conn.execute(
                    f"SELECT COUNT(*) FROM articles WHERE "
                    f"first_seen >= ? AND {live_clause}", (s2,)
                ).fetchone()[0] or 0)
                out["live_24h"] = int(conn.execute(
                    f"SELECT COUNT(*) FROM articles WHERE "
                    f"first_seen >= ? AND {live_clause}", (s24,)
                ).fetchone()[0] or 0)
        finally:
            conn.close()
    except Exception:
        return out
    return out


@app.route("/api/feed-health")
def feed_health_api():
    """Is the live trader actually *seeing* any news, or flying blind?

    Every other panel measures behaviour *after* a decision and assumes the
    trader received signals. None answer the prior question when the book just
    HOLDs for hours. strategy.decide() builds Opus's prompt from
    signals.get_top_signals(hours=2) against signals._db_path(); if that DB is
    stale the prompt's signal block is empty, signal_count is recorded 0, and a
    0-signal HOLD is indistinguishable from a deliberate one. /api/data-feed
    shows raw counts with no verdict, path, or link to the decision log — a
    stale `articles_24h: 3801` reads as healthy. This adds the consecutive
    0-signal *decision streak*, the resolved DB path + its newest-live age, and
    split-brain detection (signals._db_path() prefers the USB mount while the
    daemon / unified_dashboard prefer the local copy — opposite precedence, so
    a stale USB mirror silently blinds the trader while every other surface
    reads the fresh one). Pure core: analytics/feed_health.build_feed_health
    (this endpoint does all the SQLite/filesystem IO; the builder stays pure).
    Advisory only — never gates Opus, adds no caps (invariants #2/#12)."""
    try:
        from . import signals as _sig
        from .analytics.feed_health import build_feed_health

        resolved = _sig._db_path()
        resolved_str = str(resolved)

        # The two candidates signals._db_path() chooses between, de-duped and
        # order-preserving. (Listing order here is presentational only — the
        # live trader resolves by *freshness* via signals._choose(), LOCAL-first
        # on a tie since 6227cd5; legacy_path below models the old USB-first
        # existence resolver, which is what split-brain detection compares to.)
        seen: set[str] = set()
        cand_paths: list[str] = []
        for p in (_sig.USB_DB, _sig.LOCAL_DB):
            ps = str(p)
            if ps not in seen:
                seen.add(ps)
                cand_paths.append(ps)

        # What a process still running the pre-freshness-aware resolver would
        # read (existence-first). When it differs from the freshly-resolved DB
        # and is materially staler, a stale runner/dashboard process
        # (/api/build-info `stale`) is blind and needs a RESTART — the
        # canonical split-brain shape now that _db_path() is freshness-aware.
        legacy_str = str(_sig._legacy_choice())

        candidates = []
        probe_by_path: dict[str, dict] = {}
        resolved_probe = {"exists": False, "newest": None,
                          "live_2h": 0, "live_24h": 0}
        for ps in cand_paths:
            probe = _feed_db_probe(ps, want_counts=(ps == resolved_str))
            probe_by_path[ps] = probe
            candidates.append({"path": ps, "exists": probe["exists"],
                               "newest": probe["newest"]})
            if ps == resolved_str:
                resolved_probe = probe

        legacy_probe = probe_by_path.get(legacy_str)
        feed = {
            "resolved_path": resolved_str if resolved_probe["exists"] else None,
            "resolved_newest": resolved_probe["newest"],
            "resolved_live_2h": resolved_probe["live_2h"],
            "resolved_live_24h": resolved_probe["live_24h"],
            "legacy_path": (legacy_str if legacy_probe
                            and legacy_probe["exists"] else None),
            "legacy_newest": legacy_probe["newest"] if legacy_probe else None,
            "candidates": candidates,
        }
        store = get_store()
        return jsonify(build_feed_health(
            store.recent_decisions(limit=3000), feed))
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR"}), 500


@app.route("/api/session-delta")
def session_delta_api():
    """What materially changed since you last looked.

    Every other panel is a current-state snapshot; /api/daily-recap is a
    calendar-"today" aggregate; /api/command-center is a one-shot current
    aggregate. None answers the operator's first question on reopening the
    dashboard after being away — "what happened while I was gone?" — which
    today means scanning ~19 panels. This is a ranked material-event timeline
    over a parameterised look-back window (fills, round-trip closes with
    realised P&L consumed verbatim from build_round_trips, equity move +
    SPY-relative alpha, intra-window drawdown, and an idle-cycle fact),
    reading only paper_trader.db (full history — no articles.db dependency).
    ``?minutes=`` (look-back, default 360 = 6h, clamped [5, 10080]) or an
    explicit ``?since=`` ISO-8601 instant. Advisory only — dashboard/chat
    surface, never injected into the decision prompt, never gates Opus, adds
    no caps (invariants #2/#12). Pure core:
    analytics/session_delta.build_session_delta."""
    try:
        from .analytics.session_delta import build_session_delta

        now = datetime.now(timezone.utc)
        since_arg = request.args.get("since")
        since_dt = None
        if since_arg:
            try:
                since_dt = datetime.fromisoformat(
                    since_arg.replace("Z", "+00:00"))
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=timezone.utc)
            except Exception:
                since_dt = None  # fall through to the minutes default
        if since_dt is None:
            minutes = max(5, min(10080, int(request.args.get("minutes", 360))))
            since_dt = now - timedelta(minutes=minutes)

        store = get_store()
        return jsonify(build_session_delta(
            list(reversed(store.recent_trades(2000))),
            store.recent_decisions(500),
            store.equity_curve(1000),
            since_dt,
            now,
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run(host: str = "0.0.0.0", port: int = 8090):
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    run()
