# Feature-Dev Agent 4 — Brainstorm & Selection (2026-05-15)

Senior-product-engineer pass over the trading stack. The system is **mature**:
~30 paper-trader API endpoints, a unified command center, conviction board,
persona leaderboard, news pulse, ops panel, dedup+urgency-decay, scorer
confidence bands, calibration, drawdown anatomy, cross-system chat context.
Most "ideas to consider" in the brief are already shipped. So this pass
hunts for **evidence-backed gaps observed in the live system**, not
re-implementations.

## Live-system observations (curled, 2026-05-15 ~23:30 UTC)

- Portfolio: total **$972.69** of $1000 start (-2.7%); cash **$6.23**
  (99.4% deployed) across 2 positions (NVDA, LITE) — both red.
- `/api/decision-health`: lifetime NO_DECISION **62.6%** (411/657);
  last-24h **31.4%** (16/51). FILLED 2.6% lifetime.
- `decisions.reasoning` for failures is captured by `strategy.py`
  (`parse_failed:` / `retry_failed:` / "no response (timeout/empty)",
  capped at `RAW_CAPTURE_CHARS=1000`) — but **no dashboard surface shows
  it**. `/api/decision-health` `recent[]` has no `reasoning` field.
  Operator sees "31% NO_DECISION" and can do nothing.
- Current runner (PID 367696) started 23:23 UTC; nearly all history is
  from a prior runner on older code (legacy "no parseable JSON" string,
  absent from current source). New-format excerpts will accumulate going
  forward — the panel must work on legacy + new rows.
- `decision_outcomes.jsonl` = 9894 rows ≫ 500 gate → DecisionScorer long
  active. (Kills "scorer-progress-to-gate" idea.)
- 0 BLOCKED decisions ever. (Kills "blocked-no-cash missed-opportunity"
  idea — no data.)
- Chat context (`_build_chat_context_block` + DI `/api/chat`) already
  pools macro + watchlist + ideas + divergence + equity trend. Brief's
  "improved chat context" largely done.

## Ideas (10+)

1. **Decision Failure Forensics** — classify NO_DECISION/parse failures
   into modes (empty/timeout, fenced, prose-wrapped, truncated, oversized,
   legacy-unknown), trend the rate in hourly buckets, segment open vs
   closed market, compute retry-rescue rate, expose recent capped
   excerpts. *Gap: the core engine silently fails ~1/3 of cycles with zero
   diagnostic surface.* **HIGH.**
2. NO_DECISION segmented by `market_open` — folds into (1).
3. **Capital Deployment & Liquidity** panel — cash %, deployed %, position
   count, top-weight, days since last new entry, "fully-invested-and-idle"
   warning. *Gap: $6 cash / 2 red names / no rotation is invisible as a
   first-class signal; `/api/risk` covers concentration but not
   liquidity/deployment + idle-entry framing.* **HIGH.**
4. Retry-rescue effectiveness — sub-metric of (1).
5. Prompt data-quality indicator (watchlist N/A price count per cycle) —
   genuine gap but needs runner instrumentation (not just dashboard);
   higher risk, defer.
6. Signal→action latency — largely covered by unified `/api/missed-signals`.
7. Per-decision confidence vs realized scatter — covered by `/api/calibration`.
8. Persona edge stability — covered by persona leaderboard.
9. Drawdown alerting — covered by `/api/drawdown`.
10. NO_DECISION self-heal fallback — changes the live loop; out of scope
    for additive feature work (risk).
11. Decision-timeline sparkline — frontend enhancement, folds into (1).
12. Forensics-driven prompt hint (most common failure mode call-out) —
    analytical output of (1).

## Selected for implementation

**A. Decision Failure Forensics** (ideas 1, 2, 4, 11, 12) —
`GET /api/decision-forensics` + a dashboard card in the live-trader pane.
Pure-function classifier (highly testable), time-bucketed trend,
open/closed split, retry-rescue rate, capped/scrubbed recent excerpts.

**B. Capital Deployment & Liquidity** (idea 3) —
`GET /api/liquidity` + a dashboard card. Deterministic metrics from
portfolio + positions + trade history (testable to specific numbers).

Two features done well with bug-catching tests beat three rushed. Both are
purely additive, follow `dashboard.py` patterns, touch no load-bearing
invariant (live-only filter, WAL/`mode=ro`, no risk caps, scorer gate),
and cross-fetchable by the unified dashboard via existing CORS.

**Honesty constraint:** the forensics panel reflects whatever is in
`paper_trader.db`. Legacy rows lack excerpts (counted as `legacy-unknown`);
rich new-format rows accumulate as the current runner produces them.
