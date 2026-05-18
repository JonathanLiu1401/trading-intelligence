# Macro (FOMC rate-decision) calendar — design

**Date:** 2026-05-18 · **Author:** feature-dev agent (Agent 4)

## Problem

The live Opus decision prompt has exactly **one** forward-looking awareness
block: `event_calendar` (single-name **earnings**). Across 47 analytics
modules there is **zero** macro-event awareness. Yet this is a
leveraged-ETF/semis-heavy book (SOXL, TQQQ, NVDL, SOXS …) and the system's own
5h Opus briefings repeatedly *lead* with macro (bond rout, 10Y, FOMC). A
leveraged book entering the FOMC rate-decision instant blind is the macro
analog of the exact "added the day before an earnings print, blind" mistake
`event_calendar` was built to close — the same gap, one dimension over.

## Decision: FOMC-only, verified-data scope

FOMC 2026 dates are **fully verifiable** from federalreserve.gov (fetched +
confirmed: 8 meetings). BLS CPI/NFP forward dates are **not** reliably
verifiable here (bls.gov hard-blocks all fetches with HTTP 403; archive-URL
dates conflict with search summaries by ±2d; Jul–Dec unreleased). Encoding
unverified dates on the live decision path is dishonest. FOMC is also the
single highest-impact macro series for a leveraged-rate-sensitive book.
CPI/NFP extensibility is documented in the module as a deliberate
scope decision pending a verifiable source — **not** an oversight.

## Architecture (mirrors `event_calendar` precedent exactly)

- **New `analytics/macro_calendar.py`** — `build_macro_calendar(now=None,
  horizon_days=14.0)`. Pure, deterministic, **no file/network I/O** (even
  safer than `event_calendar`, which does disk reads). Static `_FOMC_2026`
  table of exact **UTC instants** of each rate-decision statement (14:00 ET,
  ET→UTC resolved per the 2026 DST boundary so no tz dependency). Returns
  `{as_of, summary, prompt_block, events, source_ok,
  schedule_valid_through}`. Never raises (`_safe` contract).
- **Market-wide, not per-ticker** — the structural difference from
  `event_calendar`. FOMC moves the whole book; the block always applies
  regardless of holdings (no `positions`/`names_in_play` args).
- **Honesty bound** — `SCHEDULE_VALID_THROUGH` == the last encoded FOMC
  instant. `now` past it ⇒ `source_ok=False`, one honest degrade line, no
  fabricated events. A regression that extends the table but not the bound
  (or vice-versa) fails a dedicated RED test (written first).
- **Tiers by time-to-event:** `IMMINENT_HOURS` (<24h, rendered "in Xh") >
  `IMMINENT` (<3d) > `UPCOMING` (≤ horizon). Past dropped; beyond-horizon
  dropped. Time precision (14:00 ET) is the material differentiator vs
  `event_calendar`'s date-only granularity — Opus deciding at 13:55 ET on
  FOMC day ≠ 09:00 ET same day.
- **Observational only** (invariants #2/#12) — autonomy-preserving preamble,
  no directive verb, never gates, no caps. Same contract as `event_calendar`.

## Wiring (all in ONE commit — the standing checklist item)

1. `strategy._build_payload`: `macro_calendar_block` kwarg; render between
   `{event_section}` and `{bp_section}`. Order becomes
   `risk<sector<event<macro<bp<WATCHLIST` (forward blocks stay adjacent;
   minimal diff).
2. `strategy.decide()`: build it `_safe` after `event_calendar_block`.
3. `decision_context.build_decision_context` + `assemble_inputs` + the
   `advisory_blocks` dict + `__main__` CLI line — the AGENTS.md pass-#17
   "3rd instance of this bug class" standing checklist: any new
   `_build_payload` advisory block MUST update `decision_context.py` in the
   same commit.
4. `dashboard.py`: `@app.route("/api/macro-calendar")` — prompt↔endpoint
   parity (the `event_calendar`/`risk_mirror` discipline).

## Testing (TDD; bug-catching asserts, not "no crash")

`tests/test_macro_calendar.py` (honesty-bound test written FIRST):
schedule-bound degrade; **table↔bound no-drift**; the 8 encoded dates equal
the federalreserve.gov-verified 2026 set; ET→UTC = 19:00 (Jan/Dec EST) /
18:00 (rest EDT); IMMINENT_HOURS/IMMINENT/UPCOMING tier + exact boundaries;
past/beyond-horizon dropped; soonest-first sort; observational/no-directive;
never raises; `_build_payload` render position + None-renders-nothing;
`/api/macro-calendar` Flask parity. Plus `test_decision_context.py`:
new `macro_calendar` reach-prompt + flag test, updated exact
`advisory_blocks` dict + the `risk<sector<event<macro<bp` ordering lock.

Ships on next paper-trader restart (live `:8090` is `stale`, `behind:2`).
