# Design — Decision Reliability + Unlock-Funded Suggestions + Chat Truth

Date: 2026-05-16 · Author: feature-dev agent · Repo: paper-trader (+ unified_dashboard)

## Problem (evidence)

Live system probe (2026-05-16):

- `decisions`: 671 rows over ~2.4 days. **412 (61.4%) are NO_DECISION.**
- Of those, 410 are the *legacy* string `"claude returned no parseable JSON"`
  (newest 2026-05-15T17:42); only 2 are the *current* code's
  `"claude returned no response (timeout/empty)"`; **0 `parse_failed:` /
  `retry_failed:`**. The runner restarted onto diagnostic code ~05-15 evening
  (`/api/build-info` → `stale:true, behind:1`).
- `/api/capital-paralysis`: book PINNED — $6.23 cash (0.6%) of $972, two
  underwater names, `-2.21%` realized involuntary alpha bleed.
- `/api/suggestions`: returns only HOLD — every BUY idea is unfundable while
  pinned, with no acknowledgement of that.
- Unified `/api/chat` context injects macro/ideas/divergence/attribution/
  behavioural-edge but **nothing about the trader being pinned or 62%
  NO_DECISION** — a user asking "why isn't it trading?" gets no truthful answer.

The mature analytics layer *measures* the NO_DECISION pathology three ways
(`decision-health` = rate, `decision-forensics` = why, `decision-drought` =
cost) but: (a) the headline 61.4% is dominated by legacy rows that stop
accruing on restart — nobody computes the **true current-regime rate**; (b)
nobody pairs an unfundable idea with the specific sale that funds it; (c) the
chat can't explain the idleness.

## Non-goals / constraints

- **No change to the decision loop** (`strategy.py`). The timeout-retry
  hardening idea is deferred — current-regime evidence is n=2; Feature A will
  accumulate the data that would justify it.
- Advisory only. No risk caps, no gating. Respects paper-trader AGENTS.md
  invariants #2 / #12 exactly as `liquidity`/`capital_paralysis` do.
- Pure-core builders with injectable `now`; compose existing
  single-source-of-truth builders — no re-derived metrics (capital_paralysis
  precedent).
- Surgical. New files + endpoint/panel additions; no refactor of existing
  builders. `retry-rescue rate` is intentionally **excluded** — the `retried`
  flag is not persisted to the DB, so it is not derivable; inventing it would
  be dishonest.

## Feature A — `/api/decision-reliability`

New pure core `paper_trader/analytics/decision_reliability.py::build_decision_reliability(decisions, equity_curve, now=None)`.

Composes `build_decision_forensics` + `build_decision_drought` (single source
of truth) and adds the genuinely new synthesis:

- **Regime partition.** Boundary = timestamp of the newest `legacy`-tagged
  failure (via `classify_failure`). `current` rows = decisions strictly after
  the boundary (all rows if no legacy failure exists). Report
  `legacy_share_pct` = legacy failures / total decisions.
- **True current failure rate** = current NO_DECISION / current total, with a
  current-mode breakdown (reusing forensics `classify_failure`, restricted to
  current rows).
- **Cost linkage** (descriptive, not fabricated): pass through
  decision-drought's realized `involuntary_alpha_bleed_pct`; report
  `dead_cycles_per_day` = current_failure_rate × decisions/day (cadence from
  the decision timestamps).
- **Sample-size honesty** (news_edge/trade_asymmetry precedent): verdict
  withheld until `current_total ≥ MIN_CURRENT` (=12). States:
  `NO_DATA` → `STALE_LEGACY_DOMINATED` (legacy ≫ current, restart recommended)
  → `INSUFFICIENT` → `HEALTHY` / `DEGRADED` / `CRITICAL` on the **current**
  rate (thresholds mirror forensics: ≥50 CRITICAL, ≥25 DEGRADED).
- `restart_recommended` bool + headline string.

Endpoint `/api/decision-reliability` in `dashboard.py` (store reads only).
UI panel on the `:8090` trader page (`dr-rel-card`), JS degrades to the
`/api/build-info` `stale` message like the other recent panels.

## Feature B — `/api/funded-suggestions`

New pure core
`paper_trader/analytics/funded_suggestions.py::build_funded_suggestions(suggestions, paralysis, now=None)`.

Inputs already computed by the endpoint: the `/api/suggestions` list and the
`build_capital_paralysis` dict (no refactor of either; the endpoint calls the
existing inline suggestions logic + the paralysis builder, then composes).

For each actionable BUY/ADD idea (conviction-ranked), classify fundability:

- `can_act` → `FUNDED` (cash available now).
- PINNED → walk the paralysis `unlock_ladder` (already in desk cut-priority)
  and attach the **minimum prefix of sales** whose `cumulative_freed_usd`
  ≥ a suggested notional (`round(conviction × total_value, 2)`, advisory,
  clearly labelled). Emit `funded_by` = list of tickers to sell, `frees_usd`,
  `enough` bool.
- EMPTY/NO_DATA → `UNFUNDABLE`.

`top_actionable` = highest-conviction BUY/ADD; `recommended_pairing` =
`{sell: recommended_unlock.ticker, buy: top_actionable}` when PINNED. Headline:
"PINNED: best unfunded idea BUY NVDA (conv 0.72); sell LITE → free $786 → can act."

Endpoint + UI panel (`fund-card`), same stale-degrade contract.

## Feature C — Chat truth (unified_dashboard.py)

Add `_fetch_decision_reliability()` sub-block to `_build_chat_context_block`
(same parallel-deadline pattern as the existing sub-fetches). One compact line
from `/api/decision-reliability` + `/api/capital-paralysis`:
`TRADER STATE: PINNED ($6 cash) · current-regime parse-fail 0% (n=2,
INSUFFICIENT) · realized involuntary alpha bleed -2.21%`. Pure string
assembly with try/except → "" on any failure (existing block convention);
no new endpoint.

## Testing

- `tests/test_decision_reliability.py` — hand-computed: regime boundary
  detection (legacy newest ts), current-only rate + mode breakdown, the
  `STALE_LEGACY_DOMINATED` vs `INSUFFICIENT` vs `CRITICAL` state machine, the
  no-legacy (all-current) path, `dead_cycles_per_day` arithmetic, drought
  bleed pass-through. Assert exact values.
- `tests/test_funded_suggestions.py` — hand-computed: FUNDED when can_act;
  minimum-prefix unlock selection (one sale enough vs needing two vs
  UNFUNDABLE); conviction ranking of `top_actionable`; recommended pairing;
  empty/no-data. Assert exact values.
- Full suite green before commit (`python3 -m pytest tests/ -v`).
- digital-intern suite unaffected (no digital-intern change); run it to
  confirm no cross-repo breakage from the vendored snapshot.

## Files

| Action | Path |
|--------|------|
| add | `paper_trader/analytics/decision_reliability.py` |
| add | `paper_trader/analytics/funded_suggestions.py` |
| add | `tests/test_decision_reliability.py` |
| add | `tests/test_funded_suggestions.py` |
| edit | `paper_trader/dashboard.py` (2 endpoints + 2 panels + JS) |
| edit | `unified_dashboard.py` (chat context sub-block) |
| edit | `AGENTS.md` (endpoint table + invariant notes) |
| add | this spec |

## Status — IMPLEMENTED (2026-05-16)

All three features shipped as specified, TDD, on branch
`feature/decision-reliability-funded-suggestions` from an isolated git
worktree (three concurrent review agents were doing `git add -A` on the main
checkout — worktree isolation prevented the cross-agent sweep).

- **Feature A** — `paper_trader/analytics/decision_reliability.py` +
  `/api/decision-reliability` + `dr-card` panel. 9 exact-value tests. Validated
  read-only on the live DB: the misleading **61.5% headline** correctly
  resolves to a **14.3% current-regime rate** over 28 post-restart cycles
  (410 dead legacy rows / 60.9% inflate the headline; regime boundary
  `2026-05-15T17:42` matches the evidence above). State had *matured*
  STALE→HEALTHY as post-restart cycles accumulated since this spec was written
  — the sample-size honesty works as designed. The 4 real recent failures are
  all `TIMEOUT_EMPTY` (actionable).
- **Feature B** — `paper_trader/analytics/funded_suggestions.py` +
  `/api/funded-suggestions` + `fund-card` panel. 9 exact-value tests.
  Validated on the live PINNED book: BUY NVDA (conv 0.30, $291.81) →
  `UNLOCKABLE` via "sell LITE → free $786.28"; HOLD LITE correctly bypassed.
- **Feature C** — `_fetch_decision_reliability` sub-block in
  `unified_dashboard.py::_build_chat_context_block`; one compact
  `TRADER STATE:` line, degrades to the pinned/bleed half alone until the
  trader restarts onto `/api/decision-reliability`. Four cases verified.

Full paper-trader suite: **544 passed, 0 failures** (526 baseline + 18 new).
No decision-loop change, no risk caps, no gating — invariants #2/#12 honoured.
The timeout-retry hardening stays deferred as planned; Feature A now
accumulates the current-regime evidence that would justify it (live current
failures are all `TIMEOUT_EMPTY`).
