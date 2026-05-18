# Design — `mark_trust`: fold mark-integrity into the equity-risk surface

**Date:** 2026-05-18
**Author:** Agent 4 (feature-dev), autonomous run
**Repo:** `trading-intelligence/paper-trader`
**Status:** approved-by-PO (autonomous; no interactive user — agent acts as product owner)

## Context

The live trader marks every open position to market each cycle via
`strategy._mark_to_market`. When yfinance returns nothing for a held name it
falls back to `avg_cost`, sets `unrealized_pl = 0.0`, and flags the row
`stale_mark = True` (the chronic live pathology: `MU 0.5 @ 724.12`,
`current_price == avg_cost`, `P/L $0.00` — indistinguishable from a genuinely
flat position). `record_equity_point` then writes a **cost-frozen, flat**
equity point for that cycle.

`build_mark_integrity` (`analytics/mark_integrity.py`, 2026-05-17) quantifies
the **current** snapshot's stale share and is surfaced at `/api/mark-integrity`
and embedded in `/api/decision-context`. Its own docstring names the harm it
does **not** fix:

> "A book that is 60% stale-marked makes `/api/analytics` Sharpe,
> `/api/drawdown`, the equity curve and the headline P&L all quietly
> fictional, **with nothing saying so**."

## Audit finding (verified, not theorised)

`grep` for `stale_mark` across `paper_trader/`: referenced **only** in
`mark_integrity.py`, `strategy.py`, `dashboard.py`, `reporter.py`. **No
equity-curve-derived risk builder is stale-aware.** Concretely:

- `/api/tail-risk` = `build_tail_risk(store.equity_curve(5000))` — VaR/CVaR,
  annualised vol, downside deviation, skew, Ulcer index over a series whose
  stale-cycle points are cost-frozen flats. Flat artefacts **deflate vol &
  drawdown, inflate Sharpe/Sortino/Calmar, truncate the VaR tail**. No caveat.
- `/api/drawdown` = `compute_drawdown(eq, positions, …)` — same series; a real
  drawdown is **masked** while marks are frozen at cost. No caveat.
- `hold_discipline`: `upl = float(p.get("unrealized_pl") or 0.0); is_losing =
  upl < 0.0` — a stale loser (`unrealized_pl == 0.0`) silently drops out of
  `disposition_drag_usd` and reads DISCIPLINED. **But** its endpoint feeds
  `store.open_positions()`, which AGENTS.md states *lacks* the `stale_mark`
  key, so a builder-side fix is inert without re-sourcing the endpoint —
  which would risk its existing discriminating `TestEndpoint`. Out of scope
  (documented as future work).
- `thesis_drift`: stale-flat `pl_pct == 0.0` reads INTACT — same class; same
  endpoint-sourcing caveat. Out of scope.

## Brainstorm (≥10 considered)

1. Historical per-equity-point stale tagging — **rejected**: no persisted
   per-cycle stale signal; reconstruction is fuzzy; needs a schema change
   (invariant #13). The codebase rejects fuzzy honesty.
2. New "risk-trust" diagnostic builder — **rejected**: surface is saturated
   (~45 builders); advisor explicitly warned against another builder.
3. Re-source `/api/hold-discipline` to the readonly snapshot + stale-aware
   builder — **deferred**: real bug, but higher blast radius (risks the
   existing exact-value `TestEndpoint`); concurrent agents.
4. Same for `/api/thesis-drift` — **deferred**, same reason.
5. Fix `_parse_decision` for NO_DECISION — **rejected**: explicit documented
   trap, another agent's lane, cause is most likely quota not parser.
6. Sector heatmap / Sharpe / VaR / correlation panels — **already built**.
7. Auto-suggest trades — **already built** (`/api/suggestions`,
   `/api/funded-suggestions`, `/api/game-plan`).
8. Chat-context enrichment — **already built**, heavily.
9. Browser-discovered dashboard bug — **investigated, none found**: the
   curl-sweep "hangs" were git-watcher restart noise (process bounced for
   `b6a1934` mid-sweep); all endpoints fast on a clean process; no browser
   MCP for visual panel testing.
10. **Fold `build_mark_integrity` into `/api/tail-risk` + `/api/drawdown` as
    an additive `mark_trust` honesty key — CHOSEN.**
11. Alert dedup/urgency decay (digital-intern) — out of repo scope; concurrent
    agents own those files.
12. Equity-curve "stale-cycle count" via `decisions` join — fuzzy, rejected
    (see #1).

## Chosen design

Wire the **existing** single source of truth (`build_mark_integrity`) into the
**three** equity-risk endpoints whose numbers its docstring names as silently
fictional — `/api/tail-risk`, `/api/drawdown`, and `/api/analytics` (the
docstring's first-named victim: "/api/analytics Sharpe"). This is the
*opposite* of adding a builder — it connects an existing truth-source to the
surfaces that are lying.

**Scope-decision note (supersedes the earlier "out of scope" line below):**
`/api/analytics` was extended in the same pass after the tail-risk/drawdown
fold proved clean. Rationale: it is the *first* surface mark_integrity's
docstring names, it already establishes the additive-key + keyed-assertion
contract (it folds `build_tail_risk` the same way), and the `_mark_trust_block`
helper was already built — so the marginal blast radius is one keyed-safe key,
not a new pattern. All three now self-flag a stale book.

### Component

`dashboard._mark_trust_block(store) -> dict | None`

- Calls `strategy.portfolio_snapshot_readonly(store)` — the **exact**
  write-free path `/api/mark-integrity` uses (shares `_mark_to_market` with
  the live path → cannot drift, invariant #10; never writes from the
  dashboard thread).
- Runs `build_mark_integrity(snap["positions"])` (verbatim — no re-derived
  staleness).
- Returns a compact additive dict:
  `{verdict, n_stale, n_positions, stale_value_pct, stale_tickers, headline,
  note}` where `note` precisely scopes the claim: the risk metrics are from
  the equity curve and the **current** book is N% cost-marked, so the recent
  equity points / live tail of these metrics are at cost-basis. Does **not**
  overclaim the whole history is contaminated (honesty discipline).
- `_safe`: any exception → returns `None`; the caller omits the key. The
  endpoint's pre-existing payload/behaviour is byte-identical on fault and
  never 500s for this reason.

### Data flow

```
tail_risk_api / drawdown_api
  └─ result = build_tail_risk(...) / compute_drawdown(...)   # UNCHANGED
  └─ mt = _mark_trust_block(store)                            # NEW, _safe
  └─ if mt is not None: result["mark_trust"] = mt             # additive key
  └─ jsonify(result)
```

Additive top-level key only — keyed-assertion-safe. Existing exact-value
suites (`test_tail_risk.py`, `test_core_analytics.py::TestTailRiskIntegration`,
drawdown tests) assert specific keys, not whole-dict equality, and the
`tail_risk`→`/api/analytics` fold already established this additive pattern.

### Invariants honoured

- #2/#12 observational only — never gates Opus, no caps, **not** injected into
  the decision prompt.
- #10 single source of truth — composes `build_mark_integrity` verbatim;
  reuses `portfolio_snapshot_readonly` (no drift vs the live mark).
- #13 — **no schema change**.
- Write-free — `portfolio_snapshot_readonly` never mutates live state from the
  dashboard thread.
- Files touched: **only** `paper_trader/dashboard.py` + new
  `tests/test_mark_trust.py`. No digital-intern files (concurrent agents).
  Stage by path, never `git add -A`.

### Tests (`tests/test_mark_trust.py`) — discriminating

Flask test client on a fresh temp `Store`:

1. **Stale book surfaces trust loss** — snapshot with a `stale_mark=True`
   position → `/api/tail-risk` body has `mark_trust.verdict` ∈
   {`DEGRADED`,`UNTRUSTWORTHY`}, exact `n_stale`, `stale_value_pct`, note
   present. RED before the wiring (no key).
2. **Clean book** — all marks live → `mark_trust.verdict == "CLEAN"`.
3. **No-drift / additive contract** — the risk keys in the response equal
   `build_tail_risk(store.equity_curve(5000))` computed directly (the fold
   must not mutate any risk number); `mark_trust` is the *only* added key.
4. **`_safe` contract** — monkeypatch `portfolio_snapshot_readonly` (or
   `build_mark_integrity`) to raise → endpoint still 200, original risk
   payload intact, **no** `mark_trust` key. A naive try-less impl 500s here.
5. Same stale/clean/-safe trio for `/api/drawdown`.
6. Single-source-of-truth lock — `mark_trust` sub-dict equals the relevant
   fields of `build_mark_integrity(snap["positions"])` (no re-derivation).

Run: `cd paper-trader && python3 -m pytest tests/test_mark_trust.py -v`
plus the touched-surface regression:
`python3 -m pytest tests/test_tail_risk.py tests/test_core_analytics.py -v`.

### Deploy note

The running `:8090` is current now (`b6a1934`) but a new commit will make it
stale until the in-process git-watcher restarts it (~3 min) or a manual
`systemctl --user restart paper-trader`. The completion message must state the
apply-on-restart caveat and not claim "live" without `/api/build-info`
`behind:0` on the new SHA.

### Deliberately out of scope (future work)

- `hold_discipline` / `thesis_drift` stale-aware re-sourcing (needs endpoint
  data-source change; risks existing discriminating endpoint tests).
- (`/api/analytics` was *promoted into scope* — see the scope-decision note
  above. It is no longer deferred.)
