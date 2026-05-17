# Trade-Asymmetry / Behavioral-Edge Diagnostic — design

*Date: 2026-05-16 — Agent 4 (feature-dev)*

## Problem

Live portfolio (observed 2026-05-16): `$972.69` (−2.7%), `win_rate 20%`,
`profit_factor 0.04`, `avg_winner_usd +0.57` vs `avg_loser_usd −3.75`
(losers 6.6× winners), `avg_holding_days 0.26`, 99.4% deployed, 80% in one
name. The single biggest pathology is **behavioral**: the bot cuts winners
tiny and rides losers large. The raw numbers exist in `/api/analytics`, but
nothing synthesizes them into the desk question: *"given my payoff ratio,
what win-rate do I need to break even, and am I cutting winners faster than
losers (disposition effect)?"* Two newer endpoints
(`/api/capital-paralysis`, `/api/open-attribution`) that diagnose adjacent
facets of this exact state have **no dashboard UI** at all.

## Scope (3 features, in priority order)

1. **`analytics/trade_asymmetry.py` + `/api/trade-asymmetry` + panel** —
   the new diagnostic. Real work.
2. **Wire `/api/capital-paralysis` + `/api/open-attribution` into the
   paper-trader dashboard** — template-pattern panels for two tested,
   orphaned endpoints. Graceful 404 (stale-process) handling.
3. **Enrich the unified-dashboard chat context** with the behavioral-edge
   verdict + open-attribution (chat already injects capital-paralysis;
   confirm before adding). Drop first if scope slips.

## Feature 1 — `build_trade_asymmetry`

### Distinction from existing analytics (the duplication test)

- `/api/analytics` = **raw aggregates** (win_rate, profit_factor, avg
  win/loss $).
- `/api/calibration` = **is the confidence axis accurate** (bucket win-rate
  vs stated confidence).
- `/api/trade-asymmetry` = **exit/sizing behavior pathology**: payoff ratio,
  per-trade expectancy, the *breakeven win-rate implied by the current
  payoff ratio* vs the *actual* win-rate (the gap is the verdict), and the
  **disposition gap** = mean winner hold-days − mean loser hold-days
  (negative ⇒ cutting winners faster than losers — the classic
  disposition effect that produces exactly this P&L shape).

### Single source of truth

Consumes `analytics/round_trips.py::build_round_trips(trades)` only — no
re-derived P&L (AGENTS.md invariant #10). Uses its `pnl_usd`, `pnl_pct`,
`hold_days` fields directly. A test asserts the module fails if it
recomputes pnl independently (we feed a trade list and assert the metrics
match `build_round_trips` output exactly).

### Tiered states (mirrors `news_edge.py` INSUFFICIENT_DATA idiom)

Keyed on closed round-trip count `n`:

| State | Condition | Emits |
|-------|-----------|-------|
| `NO_DATA` | `n == 0` | metrics all null, neutral headline |
| `EMERGING` | `1 ≤ n < 20` | numeric metrics + raw asymmetry, **no verdict label**, headline says "emerging — N of 20 round-trips for a stable read" |
| `STABLE` | `n ≥ 20` | metrics + verdict label + headline |

Numeric metrics (`payoff_ratio`, `expectancy_usd`, `breakeven_win_rate_pct`,
`actual_win_rate_pct`, `win_rate_gap_pct`, `disposition_gap_days`,
`avg_winner_usd`, `avg_loser_usd`, `n_round_trips`, `n_wins`, `n_losses`)
are emitted at **EMERGING and STABLE**. Only the `verdict` /
`verdict_reason` string is gated to `STABLE` (a 5-trade verdict would be
noise and embarrass the panel — advisor guidance).

### Verdict labels (STABLE only)

Concrete thresholds (named module constants, so tests pin them):

- `DISPOSITION_EPS_DAYS = 0.01` — a disposition gap inside ±0.01d (~15 min;
  meaningful at the observed 0.26d avg hold) is treated as "no skew".
- `FLAT_EPS_USD = 0.01` — expectancy within ±$0.01/trade is "flat".

Note: `sign(expectancy_usd)` is mathematically equivalent to
`actual_win_rate vs breakeven_win_rate` (washes contribute 0 to the sum), so
the verdicts are designed to be *independently reachable* rather than
overlapping restatements of the same inequality:

- `PAYOFF_TRAP` — `payoff_ratio is not None` and
  `actual_win_rate_pct < breakeven_win_rate_pct` (≡ expectancy < 0): the
  win-rate cannot carry the payoff ratio. Most actionable → highest
  precedence. When `disposition_gap_days < −DISPOSITION_EPS_DAYS` the
  headline appends the plain-English disposition clause ("cutting winners
  at +$X after Yd, riding losers to −$Z over Wd") — the *why* behind the
  trap, but the verdict label stays `PAYOFF_TRAP`.
- `DISPOSITION_BLEED` — `expectancy_usd > FLAT_EPS_USD` (book is net
  positive — would otherwise be `EDGE_POSITIVE`) **but**
  `disposition_gap_days < −DISPOSITION_EPS_DAYS`: profitable yet leaving
  money on the table by cutting winners faster than losers. Distinct from
  `PAYOFF_TRAP` (positive expectancy) and from `EDGE_POSITIVE` (the skew).
- `EDGE_POSITIVE` — `expectancy_usd > FLAT_EPS_USD` and the disposition
  skew is absent (`disposition_gap_days ≥ −DISPOSITION_EPS_DAYS` or
  `None`): a genuine, well-managed edge.
- `FLAT` — fallback when none fire (includes `|expectancy_usd| ≤
  FLAT_EPS_USD` and the no-decided-trips case).

Verdict precedence: check `PAYOFF_TRAP` first; else if expectancy is
positive, `DISPOSITION_BLEED` when the negative skew is present otherwise
`EDGE_POSITIVE`; else `FLAT`.

### Definitions

- winner = round-trip with `pnl_usd > 0` (strict; mirrors round_trips
  win/loss `> 0` convention, AGENTS.md #10); loser = `pnl_usd < 0`;
  `pnl_usd == 0` is excluded from both (a wash, like round_trips).
- `payoff_ratio = mean(winner pnl_usd) / mean(|loser pnl_usd|)`; `None` if
  no losers (can't form the ratio) — surfaced honestly, not as ∞.
- `breakeven_win_rate_pct = 1/(1+payoff_ratio) × 100` (Kelly breakeven). If
  `payoff_ratio is None` → `None`.
- `expectancy_usd = mean(all round-trip pnl_usd)` (this is just the per-trade
  realized mean; equals `realized_pl / n`).
- `disposition_gap_days = mean(winner hold_days) − mean(loser hold_days)`;
  round-trips with `hold_days is None` excluded from the hold-time means
  only (still counted in win/loss). `None` if either side has no
  hold-day-bearing trips.
- All means guard empty inputs → `None`, never division by zero.

### Purity / testability

Pure function `build_trade_asymmetry(trades, now=None) -> dict`. `now` only
for `as_of` stamp. No network, no DB. Smoke `__main__` against live store
like the sibling modules.

## Feature 1 — endpoint

`/api/trade-asymmetry` in `dashboard.py`: read `store.recent_trades(limit)`
(same limit convention as `analytics_api`, which uses a large cap), call
`build_trade_asymmetry`, return JSON. CORS already global.

## Feature 1 — UI panel

New card in the Trader pane near Portfolio Analytics / Calibration. Shows
state badge, headline, payoff ratio, expectancy, actual vs breakeven
win-rate (with the gap highlighted red when actual < breakeven),
disposition gap with plain-English label, win/loss counts. `refresh*` on a
60s timer, registered in the boot block.

## Feature 2 — orphaned panels

Two new cards: **Capital Paralysis & Unlock Ladder** (`/api/capital-paralysis`)
and **Open-Book Alpha (Selection vs Market)** (`/api/open-attribution`).
Render `state`/`status`, `headline`, the unlock ladder table (paralysis),
and the per-position alpha table (attribution). JS must treat a 404 / non-2xx
as "endpoint unavailable — restart paper-trader to apply (process is N
commits behind HEAD)", reusing the existing build-info stale pattern, so the
panels read as informative rather than broken on the currently-stale `:8090`.

## Feature 3 — chat context

In `unified_dashboard.py::_build_chat_context_block`, add a short
behavioral-edge line (from `/trader/api/trade-asymmetry`) and an open-book
alpha line (from `/trader/api/open-attribution`) to the `<live_context>`
block, alongside the existing capital-paralysis injection. Confirm
`open_attribution` is not already injected before adding (it is not, per
exploration). Keep within the existing 4s parallel-fetch deadline; degrade
silently on timeout like the other context fetches.

## Testing

- `tests/test_trade_asymmetry.py` — TDD first. Cases: empty→NO_DATA;
  1..19 trips→EMERGING with metrics but no verdict; ≥20→STABLE; exact
  payoff_ratio / breakeven / expectancy / disposition_gap for a fixed
  ledger; no-losers→payoff_ratio None & breakeven None (not ∞);
  PAYOFF_TRAP vs DISPOSITION_BLEED vs EDGE_POSITIVE classification with
  hand-computed fixtures; metrics equal `build_round_trips` output (no
  reimplementation); `hold_days None` excluded from disposition mean only.
- Full suite `python3 -m pytest tests/ -v` must pass before commit; no
  weakening existing tests.

## Constraints honored

- AGENTS.md #10 (round-trip single source of truth) — consume, don't copy.
- AGENTS.md #2/#12 (no hard risk limits) — diagnostic/advisory only, never
  gates Opus, adds no caps.
- Multi-agent git: `git add <paths>` only, never `-A` (memory).
- Stale `:8090` is an ops concern, not fixed here; panels degrade gracefully.
- Will NOT restart services (paper-trader dual-service restart footgun,
  memory).
