# Feature-dev session — 2026-05-18 (Agent 4)

Autonomous feature-dev pass. No interactive user (Discord-completion agent), so
the brainstorming gate is skipped but its *principle* applied: enumerate many,
pick by leverage, write the decision down before coding (the 2026-05-17
event_calendar precedent).

## Brainstorm — candidates, scored by leverage

The stack is exceptionally mature (~60 `/api/*` endpoints, ~44 analytics
builders, 15+ prior hybrid passes, last ~25 commits all features). The
discriminating filter, established by every prior pass: a grep must return
**no prior implementation in the decision path**, and the change should
ideally alter *what gets traded* (a prompt-facing fact), not add the 45th
read-only dashboard panel. Hot-path safety is load-bearing — no extra store
read, no network on the live cycle (the `risk_mirror`/`buying_power`
discipline).

Current prompt-facing advisory stack to Opus: `self_review` (payoff /
disposition / paralysis / open-book alpha), `track_record` (per-name closed
trades), `risk_mirror` (name concentration + churn), `event_calendar`
(upcoming earnings), `buying_power` (deployable cash), `ML advisor` (when
qualified).

1. **Live-book SECTOR concentration + marginal-trade sector impact in the
   decision prompt.** `risk_mirror` closed *name*-level HHI; the book's
   documented #3 pathology is literally *sector* clustering (`risk_mirror.py`
   docstring: "the book 60.9% in one name's **sector** … The dashboard
   already exposes both … but the decision engine itself never saw them").
   `/api/analytics` computes `sector_exposure_pct` and `/api/risk` per-position
   sector, but **the decision path has zero sector awareness** (grep:
   `sector` in strategy.py prompt path → 0 hits). The marginal question — "the
   trade you're about to make piles onto your single most concentrated
   sector" — is invisible at decision time. *Highest leverage: changes what
   gets traded; the literal "one dimension over" the risk_mirror docstring
   names.* ✅ no prior impl. ✅ hot-path safe (pure arithmetic over the
   already-marked snapshot + static SECTOR_MAP + the signals already in
   `merged`). **PICK.**
2. Earnings reaction history / implied move — extends `event_calendar` but
   needs price history around past earnings → network on the hot path.
   Out of hot-path budget.
3. Sector-rotation momentum in the prompt — overlaps `sector_heatmap`
   (dashboard-only, network-bound) and `_sector_rotation` (backtest); the
   *book exposure* gap (#1) is sharper and pure.
4. Catalyst-staleness ("is the top signal already priced in?") — overlaps
   `news_edge` / `signal_followthrough`; harder to make deterministic.
5. Session-phase awareness (open auction / midday / power hour) — cheap but
   low leverage; Opus can infer from the timestamp already in the payload.
6. Drawdown / underwater-duration as a prompt fact — `self_review` already
   carries open-book alpha; `drawdown.py` is dashboard-only but lower
   leverage than #1.
7. Thematic (not sector) clustering — novel but needs a theme model;
   non-deterministic, fails the exact-value-test bar.
8. Per-position stop suggestion — conflicts with invariants #2/#12 (no caps).
9. Options pin/max-pain — book rarely holds options; `greeks` covers it.
10. Equal-weight-watchlist counterfactual — read-only analytics surface,
    distinct from `/api/benchmark` but lower leverage than #1.
11. Chat portfolio-history context — `/api/chat` already enriched.

## Decision

Ship **#1: live-book sector concentration + marginal-trade sector impact,
wired into the live decision prompt**, following the `risk_mirror` /
`buying_power` precedent exactly: `_safe`-wrapped pure builder,
observational-only (invariants #2/#12 — preamble disclaims directive/cap,
never gates), single-source-of-truth, prompt-block + `/api/*` parity, locked
by a dedicated exact-value test suite.

### Single-source-of-truth design (the load-bearing decision)

The canonical book-sector taxonomy is `dashboard.py::SECTOR_MAP` + `_classify`
(what `/api/analytics sector_exposure_pct` and `/api/risk` use, what
`game_plan` consumes). A new/different taxonomy would make the dashboard show
two contradictory sector breakdowns — the "prompt sections can't disagree"
anti-pattern.

But importing `dashboard.py` onto the **live decision hot path** executes a
~9k-line Flask module and makes the live cycle fragile to any sibling edit
that breaks dashboard import (a catastrophic "no decision" risk, even if
`_safe` catches it). The codebase's own established resolution for
"SSoT-via-import is structurally bad" is the `_ml_live_opinion` precedent:
**duplicate the table in the new pure module, document why, and pin it
identical to the canonical source with a test** so drift fails CI. That is
strictly safer than a Flask import on the trading hot path and touches **no
contested file**.

### Deliverables

- `paper_trader/analytics/sector_exposure.py` — `build_sector_exposure(...)`,
  with a test-pinned verbatim copy of `dashboard.SECTOR_MAP`/`_classify`.
- `decide()` + `_build_payload(..., sector_exposure_block=)` wiring (a new
  slot in the advisory stack, after `risk_mirror`, sibling to the others).
- `/api/sector-exposure` (prompt↔endpoint parity; `/api/analytics` and
  `/api/risk` left untouched — different concerns, already tested).
- `tests/test_sector_exposure.py` — bug-catching exact-value assertions
  (SECTOR_MAP drift-lock vs dashboard, exposure %, top-sector pick, marginal
  candidate-sector impact, options ×100, missing/garbage → no-raise,
  observational voice, `_build_payload` wiring + None-renders-nothing).
- AGENTS.md / CLAUDE.md updates.

### Prompt shape & cardinality (locked pre-implementation)

Two parts, one block:

- **State** (the whole book): top sector + its %, sector-HHI + a calibrated
  label, full breakdown (top ≤6 sectors), cash %.
- **Marginal** (the lean `_names_in_play(positions, merged, WATCHLIST)` set —
  same universe as `buying_power`/`track_record`, NOT the full watchlist):
  each in-play name → its sector → that sector's *current* book weight, with a
  "already heaviest / already heavy" flag. **No fabricated fill size** — Opus
  chooses size, so the honest deterministic fact is "MU is SEMIS, SEMIS is
  already 61% of your book", not an invented "would take 61%→73%".

States: `NO_DATA` (no priced book — the `buying_power` fallback pattern),
`CONCENTRATED` (top sector ≥ heavy threshold), `DIVERSIFIED` (below it).

### Parity SSoT (which dashboard number this must equal)

`/api/sector-exposure` must equal `/api/analytics` `sector_exposure_pct` for
the same snapshot — that is the breakdown a trader cross-checks. So the
builder mirrors **`analytics_api`'s exact formula verbatim**: `price =
current_price or avg_cost; val = price*qty*(100 if option else 1); pct =
val/total_value*100`, classified by the SECTOR_MAP copy. (Distinct from
`buying_power`, which matches `/api/capital-paralysis` and prefers enriched
`market_value` — a different SSoT.) The heavy-threshold constant is
drift-locked to `game_plan._SECTOR_HEAVY_PCT = 60.0` so the prompt and the
dashboard game-plan card can't disagree.

### Test targets (exact-value bar)

SECTOR_MAP drift-lock vs `dashboard.SECTOR_MAP`; heavy-threshold drift-lock vs
`game_plan._SECTOR_HEAVY_PCT`; WATCHLIST coverage floor (a future watchlist
add missing a SECTOR_MAP entry must fail, not silently become "% other");
options ×100 sector-USD path; signal `ticker=None` filtered not crashed;
fresh/all-cash → `NO_DATA`; parity lock (builder `sector_pct` == an
independent `analytics_api`-formula recompute); hand-computed sector-HHI;
deterministic top-sector tie-break (−pct, then name); observational voice (no
directive verb in preamble, autonomy disclaimer present); `_build_payload`
wiring + `None`-renders-nothing.

### Concurrency

≥3 sibling agents commit in parallel. New module + new test file are
exclusively mine (`git add` whole). `strategy.py` / `dashboard.py` /
`AGENTS.md` are contested → stage only my own hunk (extract → `git apply
--cached`), verify `git diff --cached -- <file> | grep -c <sibling-token>`
== 0 before commit. Never `git add -A`.
