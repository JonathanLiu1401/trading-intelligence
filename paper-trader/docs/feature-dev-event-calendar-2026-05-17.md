# Feature-dev session — 2026-05-17 (Agent 4)

Autonomous feature-dev pass. No interactive user (Discord-completion agent), so
the brainstorming gate is skipped but its *principle* applied: enumerate many,
pick by leverage, write the decision down before coding.

## Brainstorm — 10+ candidates, scored by leverage

The stack is extremely mature (~60 `/api/*` endpoints, many concurrent agents,
last 15 commits all features). The discriminating filter: a grep must return
**no prior implementation**, and the change should ideally alter *what gets
traded*, not just add a 30th read-only panel.

1. **Earnings/event-calendar awareness in the decision prompt** — Opus is
   completely blind to scheduled binary catalysts. `/api/earnings-risk` exists
   but is dashboard-only; `strategy.py` has zero earnings/FOMC/OPEX awareness.
   *Highest leverage: changes what gets traded.* ✅ no prior impl.
2. Decision→fill slippage tracker — overlaps `decision_context` / `mark_integrity`.
3. Equal-weight-watchlist counterfactual benchmark — distinct from `/api/benchmark`
   (SPY) but a whole new analytics surface, read-only, lower leverage.
4. Options max-pain / pin risk for held options — book rarely holds options;
   `/api/greeks` already covers Greeks; complex, low live relevance.
5. Intraday VIX / vol-regime in the prompt — real gap, but VIX data path +
   regime calibration is a larger, fuzzier change than earnings.
6. Sector-rotation momentum panel — overlaps `sector_heatmap` / `sector_pulse`.
7. Alert urgency-decay — digital-intern side; `news_dedup` already exists.
8. Per-position stop-suggestion — conflicts with invariant #2/#12 (no caps).
9. Chat: portfolio-history context — `/api/chat` already enriched (commit 26caf94).
10. Backtest persona-vs-regime attribution — `persona_skill.py` already exists.
11. Web-UI stale-mark flag parity with the Discord ⚠ STALE line — small, real,
    but low leverage vs. #1.

## Decision

Ship **#1: earnings/event-calendar awareness wired into the live decision
prompt**, following the `risk_mirror.py` precedent exactly (the closest pattern:
`_safe`-wrapped, observational-only per invariants #2/#12, single-source-of-truth,
prompt-block + `/api/*` parity, locked by a dedicated test suite).

### Why this is the right pick
- It is the **#1 scheduled risk event** a discretionary desk tracks and the one
  thing the engine currently cannot see. A "max profit, no caps, size by
  conviction" mandate makes earnings *more* decision-relevant, not less: Opus
  may want to size **up** into a conviction earnings play or **avoid adding**
  right before an unpredictable print — either way, trading blind is strictly
  worse. Live right now: **NVDA reports in ~1.9 days, MRVL in ~8.9** — the
  trader sees neither.
- Grep proved no prior implementation in the decision path.

### Hot-path safety (load-bearing constraint)
The decision loop must stay fast and must never be sunk by a diagnostics fault.
digital-intern's `/api/earnings` derives from a local file
`digital-intern/data/earnings_calendar.json`. The new builder reads **that file
directly from disk** (the `signals.py` filesystem pattern) — **no network hop to
:8080** (the documented hang/latency hazard on the live cycle). `_safe`-wrapped:
missing/corrupt/stale file → honest fallback line, never an exception.

### Single source of truth
- Tier rule reused verbatim from `/api/earnings-risk` semantics: held & ≤3d →
  `HELD_IMMINENT`, held & within-horizon → `HELD_SOON`, in-play watch → `WATCH`.
- `days_away` recomputed from `earnings_date` vs `now` (mirrors digital-intern's
  `api_earnings` — a stale snapshot still yields accurate counters). This is the
  single most bug-prone line → it gets the most discriminating test.
- "Names in play" reuses `strategy._names_in_play` (same set the quant /
  track-record blocks use) so prompt sections can't disagree.

### Deliverables
- `paper_trader/analytics/event_calendar.py` — `build_event_calendar(...)`.
- `decide()` + `_build_payload(..., event_calendar_block=)` wiring (the
  `risk_section` rendering slot).
- `/api/event-calendar` (prompt↔endpoint parity; existing `/api/earnings-risk`
  left untouched — different concern, already tested).
- `tests/test_event_calendar.py` — bug-catching assertions (days_away recompute,
  tier boundaries, past-event drop, sort, missing/corrupt → no-raise, freshness
  pick, preamble voice, `_build_payload` wiring + None-renders-nothing).
- AGENTS.md / CLAUDE.md updates.
