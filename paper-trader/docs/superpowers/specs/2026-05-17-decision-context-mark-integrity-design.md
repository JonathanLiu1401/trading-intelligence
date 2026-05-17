# 2026-05-17 — Decision-context inspector + mark-integrity

## Problem (from live observation, not speculation)

`:8090` build-info: `stale:false behind:0` (sha `728bd09`) — fresh.
Live state: portfolio $972.69 (−2.7%), cash **$18.49** (capital-pinned),
2 positions. `MU 0.5 @ 724.12` shows `current_price == avg_cost`,
`P/L $0.00` — a **live stale price mark** (yfinance returned nothing,
snapshot fell back to cost). Recent decisions are dominated by
`NO_DECISION (timeout/empty)` and `HOLD`.

The system has ~45 endpoints and ~38 analytics builders covering nearly
every *observational* angle. Two real gaps remain:

1. **No decision-input transparency.** `decisions` stores only
   `action_taken` + `reasoning`; the only raw capture is 1000 chars of the
   *response* on parse failure. An operator cannot see *what the trader was
   shown* when it timed out / held. Every diagnostic presupposes the inputs.
2. **No mark-trust meta-metric.** `stale_mark` is surfaced per-position to
   Opus and Discord, but nothing answers "what fraction of displayed book
   value is fictional right now?" — a stale book makes every P&L panel
   partially false.

## Approach (advisor-reviewed)

Reconstruct-on-demand, not capture-on-write — avoids the `decisions`
schema change (CLAUDE.md #13), hot-path edits, and storage growth.

### strategy.py — behaviour-preserving extraction
Extract the mark-to-market enrich loop from `_portfolio_snapshot` into a
pure `_mark_to_market(positions, stock_prices) -> (enriched, open_value,
marks)`. `_portfolio_snapshot` keeps identical behaviour (still writes
marks + portfolio) — locked green by existing
`tests/test_core_strategy.py::TestPortfolioSnapshot*`. New
`portfolio_snapshot_readonly(store)` calls the same helper but **skips
both store writes** → identical mark logic (incl. expired-option intrinsic
+ `stale_mark`), zero side effects, single source of truth (#10).

### Feature 1 — `analytics/decision_context.py` + `GET /api/decision-context`
Pure builder; the "network in the endpoint, builder takes the dicts"
split (the `thesis_drift`/`correlation` precedent). Renders the prompt via
the **pure `strategy._build_payload`** + the exact `decide()` SYSTEM_PROMPT
/ ML-ADVISOR framing → byte-identical to the live prompt given identical
inputs. **`_claude_call` is never invoked.** Returns: rendered `prompt`
(bounded to 40 000 chars + honesty keys), `input_summary` (signal counts,
watchlist resolved/missing — surfaces yfinance starvation, the NO_DECISION
root cause), `advisory_blocks` presence flags, embedded `mark_integrity`,
and a `feed_state` ∈ `BLIND` (0 merged signals) / `DEGRADED` (≥half
watchlist prices missing) / `OK`. SWR-cached 30s. `__main__` CLI prints
the summary (and `--full` the prompt) — works when `:8090` is wedged
(the `desk_pulse` / `signals --check-freshness` precedent).

### Feature 2 — `analytics/mark_integrity.py` + `GET /api/mark-integrity`
Pure builder over the read-only enriched positions: `n_stale`,
`stale_value_usd`, `stale_value_pct` of gross, per-name rows, verdict
`NO_DATA`/`CLEAN`/`DEGRADED` (0<pct<50)/`UNTRUSTWORTHY` (≥50). Embedded
in decision-context too. Never raises.

## Constraints honoured
Advisory only, read-only, never injected into `decide()`, no caps
(CLAUDE.md #2/#12). `:8090`-only — **no `unified_dashboard.py` edit**
(a concurrent agent's CSS WIP is uncommitted there; avoid the footgun).
No UI panel this pass — endpoint + CLI only, exactly the `desk_pulse`
precedent (scope discipline: two well-tested features beat five partial).

## Tests (exact-value, Flask test client — the codebase idiom)
- `test_mark_integrity.py`: empty→NO_DATA; all-fresh→CLEAN; the live
  MU-stale shape exact; ≥50%→UNTRUSTWORTHY; gross 0→pct None (no div0);
  never-raises-on-garbage.
- `test_decision_context.py`: prompt contains `PORTFOLIO:` /
  `WATCHLIST PRICES:` / `TOP SCORED SIGNALS` / `NO RISK LIMITS` /
  `Return JSON only.`; input_summary counts exact; advisory flags exact;
  truncation honesty; `feed_state` boundaries; `portfolio_snapshot_readonly`
  shape-parity with `_portfolio_snapshot` **and** asserts it does NOT
  write; endpoint 200 with `strategy._claude_call` monkeypatched to raise
  (proves it is never called); SWR honesty-key shape.
- Full suite stays green (clear `__pycache__` first if phantom failures).
