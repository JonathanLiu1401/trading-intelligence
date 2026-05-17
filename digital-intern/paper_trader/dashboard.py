"""Flask dashboard at :8090 — portfolio chart, trade log, positions, decisions, backtests."""
from __future__ import annotations

import json
from flask import Flask, jsonify, render_template_string

from .store import get_store

app = Flask(__name__)


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
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: -apple-system, "SF Mono", Menlo, monospace;
      background: #0b0f14; color: #cfd8dc; padding: 24px;
    }
    h1 { margin: 0 0 4px; font-size: 22px; letter-spacing: .3px; }
    .sub { color: #78909c; font-size: 12px; margin-bottom: 18px; }
    nav.tabs {
      display: flex; gap: 4px; margin-bottom: 18px;
      border-bottom: 1px solid #1b2229;
    }
    nav.tabs a {
      padding: 8px 16px; color: #78909c; text-decoration: none;
      border-bottom: 2px solid transparent; font-size: 13px;
      letter-spacing: .5px; text-transform: uppercase; cursor: pointer;
    }
    nav.tabs a.active { color: #42a5f5; border-bottom-color: #42a5f5; }
    nav.tabs a:hover { color: #cfd8dc; }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }
    .grid {
      display: grid; gap: 18px;
      grid-template-columns: 1fr 1fr;
    }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: #11161d; border: 1px solid #1b2229; border-radius: 12px;
      padding: 18px;
    }
    .card h2 {
      margin: 0 0 12px; font-size: 13px; letter-spacing: 1px;
      color: #b0bec5; text-transform: uppercase;
    }
    .stat-row { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 10px; }
    .stat { flex: 1 1 110px; }
    .stat .v { font-size: 22px; color: #eceff1; }
    .stat .l { color: #78909c; font-size: 11px; text-transform: uppercase; letter-spacing: .8px; }
    .pos, .pl { color: #4caf50; }
    .neg { color: #ef5350; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td {
      text-align: left; padding: 6px 8px;
      border-bottom: 1px solid #1b2229;
    }
    th { color: #78909c; font-weight: 500; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .muted { color: #78909c; }
    canvas { max-height: 280px; }
    .pill {
      display: inline-block; padding: 2px 8px; border-radius: 100px;
      background: #1f2933; color: #b0bec5; font-size: 10px; letter-spacing: .5px;
    }
    .pill.buy { background: #1b3a2a; color: #66bb6a; }
    .pill.sell { background: #3a1b1b; color: #ef5350; }
    .pill.run { background: #20303f; color: #82b1ff; }
    tr.bt-row { cursor: pointer; }
    tr.bt-row:hover td { background: #161d26; }
    tr.bt-row.best td { background: #143124; }
    tr.bt-row.beat td:first-child { border-left: 2px solid #4caf50; }
    tr.bt-row.miss td:first-child { border-left: 2px solid #ef5350; }
    #bt-trades { margin-top: 14px; display: none; }
    #bt-trades.show { display: block; }
    .bt-headline {
      display: flex; gap: 28px; flex-wrap: wrap; margin-bottom: 12px;
    }
    .bt-headline .stat .v { font-size: 26px; }
  </style>
</head>
<body>
  <nav style="background:#1a1a2e;padding:10px 20px;display:flex;gap:20px;align-items:center;font-family:monospace;border-bottom:1px solid #333;margin:-24px -24px 18px -24px">
    <span style="color:#e94560;font-weight:bold;font-size:1.1em">◈ TRADING STACK</span>
    <a href="http://10.19.203.44:8080" style="color:#00b4d8;text-decoration:none">Digital Intern</a>
    <a href="http://10.19.203.44:8090" style="color:#fff;border-bottom:2px solid #e94560;text-decoration:none">Paper Trader</a>
    <a href="http://10.19.203.44:8090/backtests" style="color:#00b4d8;text-decoration:none">Backtests</a>
    <span style="margin-left:auto;color:#666;font-size:0.8em">10.19.203.44</span>
  </nav>

  <h1>Paper Trader</h1>
  <div class="sub" id="hb">loading…</div>

  <div class="card" style="margin-bottom:18px;">
    <h2 style="display:flex;justify-content:space-between;align-items:center;">
      <span>Signal Feed — Digital Intern</span>
      <a href="http://10.19.203.44:8080" style="font-size:11px;color:#42a5f5;text-decoration:none;text-transform:none;letter-spacing:normal">View All Signals →</a>
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
    <div class="card" style="margin-bottom:18px;">
      <h2>Equity curve</h2>
      <div class="stat-row">
        <div class="stat"><div class="l">total value</div><div class="v" id="tv">—</div></div>
        <div class="stat"><div class="l">cash</div><div class="v" id="cash">—</div></div>
        <div class="stat"><div class="l">P/L vs $1000</div><div class="v" id="pl">—</div></div>
        <div class="stat"><div class="l">S&amp;P 500</div><div class="v" id="sp">—</div></div>
      </div>
      <canvas id="eq"></canvas>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Open positions</h2>
        <table id="pos-tbl">
          <thead><tr>
            <th>ticker</th><th>type</th><th class="num">qty</th>
            <th class="num">avg</th><th class="num">now</th><th class="num">P/L</th>
          </tr></thead><tbody></tbody>
        </table>
      </div>
      <div class="card">
        <h2>Recent trades</h2>
        <table id="trades-tbl">
          <thead><tr>
            <th>time</th><th>action</th><th>ticker</th>
            <th class="num">qty</th><th class="num">price</th><th>reason</th>
          </tr></thead><tbody></tbody>
        </table>
      </div>
    </div>

    <div class="card" style="margin-top:18px;">
      <h2>Decision log</h2>
      <table id="dec-tbl">
        <thead><tr>
          <th>time</th><th>open?</th><th class="num">signals</th>
          <th>action</th><th class="num">equity</th><th>reasoning</th>
        </tr></thead><tbody></tbody>
      </table>
    </div>
  </div>

  <!-- ────── Backtests pane ────── -->
  <div id="tab-backtests" class="tab-pane">
    <div class="card" style="margin-bottom:18px;">
      <h2>Backtest summary — 10 independent year-long runs ($1000 start)</h2>
      <div class="bt-headline">
        <div class="stat"><div class="l">average return</div><div class="v" id="bt-avg">—</div></div>
        <div class="stat"><div class="l">average final $</div><div class="v" id="bt-avg-final">—</div></div>
        <div class="stat"><div class="l">best run</div><div class="v" id="bt-best">—</div></div>
        <div class="stat"><div class="l">worst run</div><div class="v" id="bt-worst">—</div></div>
        <div class="stat"><div class="l">SPY baseline</div><div class="v" id="bt-spy">—</div></div>
        <div class="stat"><div class="l">runs vs SPY</div><div class="v" id="bt-beat">—</div></div>
      </div>
      <canvas id="bt-chart"></canvas>
    </div>

    <div class="card">
      <h2>Runs</h2>
      <table id="bt-tbl">
        <thead><tr>
          <th>#</th><th class="num">seed</th><th class="num">final $</th>
          <th class="num">return %</th><th class="num">vs SPY</th>
          <th class="num">trades</th><th>status</th>
        </tr></thead><tbody></tbody>
      </table>
      <div id="bt-trades">
        <h2 style="margin-top:18px;">Trade log — Run <span id="bt-trades-run">—</span></h2>
        <table id="bt-trades-tbl">
          <thead><tr>
            <th>date</th><th>action</th><th>ticker</th>
            <th class="num">qty</th><th class="num">price</th>
            <th class="num">value</th><th>reason</th>
          </tr></thead><tbody></tbody>
        </table>
      </div>
    </div>
  </div>

<script>
const fmt = (n, d=2) => (n == null ? "—" : Number(n).toLocaleString(undefined, {minimumFractionDigits:d, maximumFractionDigits:d}));
const dollar = n => (n == null ? "—" : "$" + fmt(n));
const dt = s => s ? s.replace("T", " ").slice(0,16) : "";

const INITIAL_TAB = "{{ initial_tab }}";
const RUN_COLORS = [
  "#42a5f5","#66bb6a","#ffb74d","#ba68c8","#26a69a",
  "#ef5350","#7986cb","#ffd54f","#4dd0e1","#ec407a"
];

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
async function refresh() {
  const r = await fetch("/api/state").then(r => r.json());
  document.getElementById("hb").textContent = "updated " + (r.now || "");
  document.getElementById("tv").textContent = dollar(r.portfolio.total_value);
  document.getElementById("cash").textContent = dollar(r.portfolio.cash);
  const pl = r.portfolio.total_value - 1000;
  const plEl = document.getElementById("pl");
  plEl.textContent = (pl >= 0 ? "+" : "") + dollar(pl);
  plEl.className = "v " + (pl >= 0 ? "pos" : "neg");
  document.getElementById("sp").textContent = r.sp500 ? fmt(r.sp500) : "—";

  const posBody = document.querySelector("#pos-tbl tbody");
  posBody.innerHTML = r.positions.map(p => {
    const cls = (p.unrealized_pl || 0) >= 0 ? "pos" : "neg";
    const label = p.type === "stock" ? p.type :
                  `${p.type.toUpperCase()} ${p.strike}/${p.expiry}`;
    return `<tr><td>${p.ticker}</td><td>${label}</td>
      <td class="num">${fmt(p.qty,4)}</td>
      <td class="num">${fmt(p.avg_cost)}</td>
      <td class="num">${fmt(p.current_price)}</td>
      <td class="num ${cls}">${fmt(p.unrealized_pl)}</td></tr>`;
  }).join("") || `<tr><td colspan="6" class="muted">no positions</td></tr>`;

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

  const labels = r.equity.map(p => dt(p.timestamp));
  const values = r.equity.map(p => p.total_value);
  const sp     = r.equity.map(p => p.sp500_price);
  if (!chart) {
    chart = new Chart(document.getElementById("eq"), {
      type: "line",
      data: { labels, datasets: [
        { label: "Equity", data: values, borderColor: "#42a5f5",
          backgroundColor: "rgba(66,165,245,0.08)", fill: true, tension: 0.18, borderWidth: 2, pointRadius: 0 },
        { label: "S&P 500 (raw)", data: sp, borderColor: "#ffb74d",
          backgroundColor: "rgba(255,183,77,0)", borderDash: [4,4], borderWidth: 1, pointRadius: 0, yAxisID: "y2" },
      ]},
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#cfd8dc" }}},
        scales: {
          x: { ticks: { color: "#78909c", maxTicksLimit: 8 }, grid: { color: "#1b2229" }},
          y: { ticks: { color: "#cfd8dc" }, grid: { color: "#1b2229" }},
          y2:{ position: "right", ticks: { color: "#78909c" }, grid: { display: false }}
        }
      }
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.data.datasets[1].data = sp;
    chart.update("none");
  }
}

// ───────── Backtests pane ─────────
let btLoaded = false;
let btChart;
let btRuns = [];

async function loadBacktests() {
  const r = await fetch("/api/backtests").then(r => r.json());
  btRuns = r.runs || [];
  btLoaded = true;

  // headline stats
  const completed = btRuns.filter(x => x.status === "complete");
  if (completed.length) {
    const avg = completed.reduce((a,b) => a + b.total_return_pct, 0) / completed.length;
    const avgF = completed.reduce((a,b) => a + b.final_value, 0) / completed.length;
    const best = completed.reduce((a,b) => a.final_value > b.final_value ? a : b);
    const worst = completed.reduce((a,b) => a.final_value < b.final_value ? a : b);
    const spy = completed[0].spy_return_pct;
    const beat = completed.filter(x => x.total_return_pct > spy).length;
    document.getElementById("bt-avg").textContent = (avg >= 0 ? "+" : "") + fmt(avg) + "%";
    document.getElementById("bt-avg").className = "v " + (avg >= 0 ? "pos" : "neg");
    document.getElementById("bt-avg-final").textContent = dollar(avgF);
    document.getElementById("bt-best").innerHTML =
      `<span class="pos">${dollar(best.final_value)}</span> <span class="muted" style="font-size:13px;">#${best.run_id}</span>`;
    document.getElementById("bt-worst").innerHTML =
      `<span class="${worst.total_return_pct >= 0 ? 'pos' : 'neg'}">${dollar(worst.final_value)}</span> <span class="muted" style="font-size:13px;">#${worst.run_id}</span>`;
    const spyEl = document.getElementById("bt-spy");
    spyEl.textContent = (spy >= 0 ? "+" : "") + fmt(spy) + "%";
    spyEl.className = "v " + (spy >= 0 ? "pos" : "neg");
    document.getElementById("bt-beat").textContent = `${beat} / ${completed.length}`;
  } else {
    ["bt-avg","bt-avg-final","bt-best","bt-worst","bt-spy","bt-beat"].forEach(id =>
      document.getElementById(id).textContent = "—");
  }

  // table
  const bestId = completed.length ?
    completed.reduce((a,b) => a.final_value > b.final_value ? a : b).run_id : -1;
  const tbody = document.querySelector("#bt-tbl tbody");
  tbody.innerHTML = btRuns.map(r => {
    const beat = r.status === "complete" && r.total_return_pct > r.spy_return_pct;
    const cls = ["bt-row",
                 r.run_id === bestId ? "best" : "",
                 r.status !== "complete" ? "" : (beat ? "beat" : "miss")].join(" ");
    const retCls = r.total_return_pct >= 0 ? "pos" : "neg";
    const vsCls  = r.vs_spy_pct >= 0 ? "pos" : "neg";
    return `<tr class="${cls}" onclick="showRunTrades(${r.run_id})">
      <td><span class="pill run">#${r.run_id}</span></td>
      <td class="num">${r.seed}</td>
      <td class="num">${dollar(r.final_value)}</td>
      <td class="num ${retCls}">${(r.total_return_pct >= 0 ? "+" : "") + fmt(r.total_return_pct)}%</td>
      <td class="num ${vsCls}">${(r.vs_spy_pct >= 0 ? "+" : "") + fmt(r.vs_spy_pct)}%</td>
      <td class="num">${r.n_trades}</td>
      <td>${r.status}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="7" class="muted">no backtest runs yet — run paper_trader.backtest</td></tr>`;

  // chart
  drawBacktestChart(btRuns);
}

function drawBacktestChart(runs) {
  // collect all unique dates across runs to use as labels
  const dateSet = new Set();
  runs.forEach(r => (r.equity_curve||[]).forEach(p => dateSet.add(p.date)));
  const labels = Array.from(dateSet).sort();

  const datasets = runs.map((r, i) => {
    const lookup = {};
    (r.equity_curve||[]).forEach(p => lookup[p.date] = p.value);
    let last = 1000;
    const data = labels.map(d => {
      if (lookup[d] != null) { last = lookup[d]; return lookup[d]; }
      return last;
    });
    return {
      label: `Run #${r.run_id}`,
      data,
      borderColor: RUN_COLORS[i % RUN_COLORS.length],
      borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: false,
    };
  });

  // SPY benchmark line: $1000 * (1 + spy_return * fraction_of_period)
  // Use the first completed run's spy_return_pct for the baseline.
  const completed = runs.filter(x => x.status === "complete");
  if (completed.length && labels.length > 1) {
    const spy = completed[0].spy_return_pct / 100;
    const spyData = labels.map((d, i) => 1000 * (1 + spy * i / (labels.length - 1)));
    datasets.push({
      label: `SPY (${(spy*100).toFixed(2)}%)`,
      data: spyData,
      borderColor: "#cfd8dc", borderDash: [6,4],
      borderWidth: 2, pointRadius: 0, tension: 0, fill: false,
    });
  }

  if (btChart) btChart.destroy();
  btChart = new Chart(document.getElementById("bt-chart"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#cfd8dc", boxWidth: 12, font: { size: 10 } }}},
      scales: {
        x: { ticks: { color: "#78909c", maxTicksLimit: 12 }, grid: { color: "#1b2229" }},
        y: { ticks: { color: "#cfd8dc", callback: v => "$"+v }, grid: { color: "#1b2229" }},
      }
    }
  });
}

async function showRunTrades(runId) {
  const r = await fetch(`/api/backtests/${runId}`).then(r => r.json());
  const wrap = document.getElementById("bt-trades");
  document.getElementById("bt-trades-run").textContent = runId;
  const tbody = document.querySelector("#bt-trades-tbl tbody");
  tbody.innerHTML = (r.trades || []).map(t => {
    const cls = t.action.startsWith("SELL") ? "sell" : "buy";
    return `<tr><td>${t.sim_date}</td>
      <td><span class="pill ${cls}">${t.action}</span></td>
      <td>${t.ticker}</td>
      <td class="num">${fmt(t.qty,4)}</td>
      <td class="num">${fmt(t.price)}</td>
      <td class="num">${fmt(t.value)}</td>
      <td class="muted">${(t.reason||"").slice(0,120)}</td></tr>`;
  }).join("") || `<tr><td colspan="7" class="muted">no trades</td></tr>`;
  wrap.classList.add("show");
  wrap.scrollIntoView({behavior:"smooth", block:"start"});
}

// ───────── Signal feed (from Digital Intern) ─────────
async function refreshSignals() {
  const ul = document.getElementById("signal-feed");
  try {
    const r = await fetch("http://10.19.203.44:8080/api/articles?limit=3");
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
      return `<li style="padding:6px 0;border-bottom:1px solid #1b2229;">
        <span class="pill" style="background:#1f3a4d;color:#82b1ff;margin-right:8px;">${score}</span>
        <a href="${url}" target="_blank" rel="noopener" style="color:#cfd8dc;text-decoration:none">${title}</a>
        <span class="muted" style="margin-left:6px;">· ${src}</span>
      </li>`;
    }).join("");
  } catch (e) {
    ul.innerHTML = `<li class="muted">digital intern unreachable</li>`;
  }
}

// ───────── boot ─────────
refresh();
refreshSignals();
setInterval(refresh, 15_000);
setInterval(refreshSignals, 30_000);
showTab(INITIAL_TAB || "trader");
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(TEMPLATE, initial_tab="trader")


@app.route("/backtests")
def backtests_page():
    return render_template_string(TEMPLATE, initial_tab="backtests")


@app.route("/api/state")
def state():
    store = get_store()
    pf = store.get_portfolio()
    positions = store.open_positions()
    trades = store.recent_trades(40)
    decisions = store.recent_decisions(20)
    eq = store.equity_curve(500)
    sp = eq[-1]["sp500_price"] if eq else None
    from datetime import datetime, timezone
    return jsonify({
        "now": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "portfolio": pf,
        "positions": positions,
        "trades": trades,
        "decisions": decisions,
        "equity": eq,
        "sp500": sp,
    })


@app.route("/api/portfolio")
def portfolio_api():
    """Compact public read of the portfolio — consumed by Digital Intern's dashboard."""
    store = get_store()
    pf = store.get_portfolio()
    return jsonify({
        "total_value": pf.get("total_value"),
        "cash": pf.get("cash"),
        "starting_value": 1000.0,
    })


@app.route("/api/backtests")
def backtests_api():
    try:
        from .backtest import BacktestStore
        store = BacktestStore()
        return jsonify({"runs": store.all_runs()})
    except Exception as e:
        return jsonify({"runs": [], "error": str(e)})


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


def run(host: str = "0.0.0.0", port: int = 8090):
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    run()
