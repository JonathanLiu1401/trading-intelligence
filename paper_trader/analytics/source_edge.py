"""News Source Edge — which of digital-intern's ~17 collectors is worth trusting?

The stack runs ~17 collectors (rss, gdelt, reddit, scraped, google_news,
finnhub, sec_edgar, …) into one ``ai_score``. ``news_edge`` grades the *score*
(does an 8.0 headline beat a 3.0 one?); ``signal_followthrough`` grades whether
the bot *acted*. Neither asks the operator's actual question: **of the sources
feeding the pipeline, whose scored headlines actually precede abnormal moves,
and which are noise that should be cut or down-weighted?**

This bins every scored live article by its *collector family* and reports the
1/3/5-trading-day forward return — raw and SPY-abnormal — pooled across score
bands per family. Pooling (not per-band) is deliberate: digital-intern only
retains a few days of live news (memory: ``project_articles_db_shallow_history``),
so a per-source × per-band × per-horizon split is starved on day 1. The pooled
per-source view is both the actionable one (cut a collector) and the one that
reaches a usable sample first.

**The ``source`` column is dirty — normalisation is a load-bearing design
choice, not an afterthought.** Live values look like ``scraped/finance.yahoo.com``,
``reddit/r/wallstreetbets``, ``GoogleNews/Yahoo Finance``, ``GDELT/finance.yahoo.com``,
plus the schema-doc'd rolling form ``gdelt_2025-09`` and bare RSS feed names
(``Investing.com``). ``_source_family`` collapses these to one key per collector:
substring before the first ``/``, trailing ``_YYYY-MM[-DD]`` stripped, lower-cased
— so the live ``GDELT/…`` and historical ``gdelt_2025-09`` rows pool, while
distinct collectors stay distinct. Without it the leaderboard fragments into
dozens of n<3 buckets and every verdict reads NOISE.

Two honesty controls (identical to ``news_edge``): SPY-abnormal return (the
verdict is judged on abnormal only) and a per-source sample-size gate
(``_MIN_SOURCE_N``). Below the gate a source is reported but not graded, and the
overall verdict is the honest ``INSUFFICIENT_DATA`` — never a fabricated edge.

``build_source_edge`` is pure and deterministic (ticker resolution / day
parsing / bar lookup imported from ``news_edge`` so the two panels can never
disagree — single source of truth, AGENTS.md invariant #10 spirit). The
endpoint does the I/O; ``_fetch_source_articles`` inlines the canonical
live-only clause verbatim (invariant #1 / the ``signals.py`` mirror).
"""
from __future__ import annotations

import re
import sqlite3
import zlib
from datetime import datetime, timezone

from .news_edge import _index_at_or_after, _parse_date, _resolve_ticker

DEFAULT_HORIZONS = (1, 3, 5)
# A collector family needs at least this many SPY-adjusted forward samples at
# the reference horizon before its mean is allowed to drive a verdict —
# mirrors news_edge._MIN_BAND_N so the two panels' sample-size honesty agree.
_MIN_SOURCE_N = 8
# Abnormal-return gap (pp) a graded source must clear to read as real edge
# rather than noise — mirrors signal_followthrough._EDGE_EPS.
_EDGE_EPS = 0.25

_LIVE_ONLY_SQL = (
    "url NOT LIKE 'backtest://%' "
    "AND source NOT LIKE 'backtest_%' "
    "AND source NOT LIKE 'opus_annotation%'"
)

_DATE_SUFFIX = re.compile(r"^(.*?)_\d{4}-\d{2}(?:-\d{2})?$")


def _source_family(source: str | None) -> str:
    """Collapse a dirty ``source`` value to one stable collector-family key.

    Rule (load-bearing — see module docstring): take the substring before the
    first ``/``, strip a trailing ``_YYYY-MM[-DD]`` rolling suffix, lower-case.
    Empty / whitespace-only ⇒ ``"unknown"``."""
    s = (source or "").strip()
    if not s:
        return "unknown"
    head = s.split("/", 1)[0].strip()
    m = _DATE_SUFFIX.match(head)
    if m:
        head = m.group(1)
    head = head.strip().lower()
    return head or "unknown"


def _fetch_source_articles(db_path: str, since_iso: str,
                           min_score: float = 2.0) -> list[dict]:
    """Live (non-backtest) scored articles since ``since_iso``, **with their
    ``source`` column** (the axis this module bins on).

    The canonical live-only clause is inlined verbatim (invariant #1 / the
    ``signals.py`` mirror): a ``backtest://`` URL, a ``backtest_*`` source or an
    ``opus_annotation*`` source must never be graded as a real collector's
    signal. Returns the ``build_source_edge`` article shape."""
    out: list[dict] = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=8)
    except Exception:
        return out
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT title, full_text, source, ai_score, urgency, first_seen "
            "FROM articles WHERE ai_score >= ? AND first_seen >= ? "
            f"AND {_LIVE_ONLY_SQL} "
            "ORDER BY first_seen DESC LIMIT 6000",
            (min_score, since_iso),
        ).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    for r in rows:
        body = ""
        try:
            if r["full_text"]:
                body = zlib.decompress(r["full_text"]).decode(
                    "utf-8", errors="replace")
        except Exception:
            body = ""
        out.append({
            "text": f"{r['title'] or ''} {body}".strip(),
            "source": r["source"],
            "ai_score": r["ai_score"],
            "urgency": r["urgency"],
            "published": r["first_seen"],
        })
    return out


def build_source_edge(
    articles: list[dict],
    price_history: dict[str, list[tuple[str, float]]],
    spy_history: list[tuple[str, float]],
    tickers: list[str],
    now: datetime | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    min_score: float = 2.0,
) -> dict:
    """Forward-return edge of scored news, pooled by collector family.

    Args:
      articles: ``[{text, source, ai_score, urgency, published}]`` — caller
        must have applied the live-only filter (use ``_fetch_source_articles``).
      price_history: ``{TICKER: [(YYYY-MM-DD, close), ...]}`` ascending bars.
      spy_history: ``[(YYYY-MM-DD, close), ...]`` ascending — pass ``[]`` for
        raw-only. The verdict is judged on abnormal return only.
      tickers: resolution universe (the live WATCHLIST), tried in order.
      min_score: pool every article with ``ai_score >= min_score`` (the
        "worth scoring" floor); below it never enters a bucket.
    """
    now = now or datetime.now(timezone.utc)
    horizons = tuple(sorted(set(int(h) for h in horizons if h > 0)))

    ph: dict[str, tuple[list[str], list[float]]] = {}
    for tk, bars in (price_history or {}).items():
        s = sorted(bars, key=lambda b: b[0])
        ph[tk.upper()] = ([d for d, _ in s], [float(c) for _, c in s])
    spy_by_date = {d: float(c) for d, c in (spy_history or [])}

    def _blank() -> dict:
        return {h: {"n": 0, "raw_sum": 0.0, "raw_up": 0,
                    "abn_n": 0, "abn_sum": 0.0, "abn_up": 0} for h in horizons}

    acc: dict[str, dict] = {}
    resolved_per_src: dict[str, int] = {}
    rx_cache: dict[str, re.Pattern] = {}
    n_total = len(articles or [])
    n_scored = 0
    n_resolved = 0

    for art in articles or []:
        try:
            score = float(art.get("ai_score") or 0.0)
        except (TypeError, ValueError):
            continue
        if score < min_score:
            continue
        n_scored += 1
        fam = _source_family(art.get("source"))
        text = str(art.get("text") or art.get("title") or "")
        tk = _resolve_ticker(text, tickers, rx_cache)
        if not tk or tk.upper() not in ph:
            continue
        day = _parse_date(art.get("published") or art.get("first_seen"))
        if day is None:
            continue
        dates, closes = ph[tk.upper()]
        i0 = _index_at_or_after(dates, day)
        if i0 is None:
            continue
        entry = closes[i0]
        if entry <= 0:
            continue
        entry_date = dates[i0]
        spy_entry = spy_by_date.get(entry_date)
        cells = acc.setdefault(fam, _blank())
        resolved_any = False
        for h in horizons:
            j = i0 + h
            if j >= len(dates):
                continue
            fwd = closes[j]
            if fwd <= 0:
                continue
            raw = (fwd / entry - 1.0) * 100.0
            c = cells[h]
            c["n"] += 1
            c["raw_sum"] += raw
            if raw > 0:
                c["raw_up"] += 1
            spy_fwd = spy_by_date.get(dates[j])
            if spy_entry and spy_fwd and spy_entry > 0:
                abn = raw - (spy_fwd / spy_entry - 1.0) * 100.0
                c["abn_n"] += 1
                c["abn_sum"] += abn
                if abn > 0:
                    c["abn_up"] += 1
            resolved_any = True
        if resolved_any:
            n_resolved += 1
            resolved_per_src[fam] = resolved_per_src.get(fam, 0) + 1

    def _finalize(blank: dict) -> dict:
        rows = {}
        for h in horizons:
            c = blank[h]
            n, an = c["n"], c["abn_n"]
            rows[str(h)] = {
                "n": n,
                "mean_raw_pct": round(c["raw_sum"] / n, 3) if n else None,
                "raw_up_rate": round(c["raw_up"] / n * 100, 1) if n else None,
                "n_abnormal": an,
                "mean_abnormal_pct": round(c["abn_sum"] / an, 3) if an else None,
                "abnormal_hit_rate": round(c["abn_up"] / an * 100, 1) if an else None,
            }
        return rows

    # Adaptive reference horizon: the longest horizon at which *some* family is
    # well-sampled (a 5d edge is the strongest claim); fall back to the longest
    # with any abnormal data, then 3, then the middle. Matures with history
    # exactly like news_edge — early on only 1d has a forward window.
    ref = None
    if horizons:
        def _max_abn_n(h: int) -> int:
            return max((cells[h]["abn_n"] for cells in acc.values()),
                       default=0)
        well = [h for h in horizons if _max_abn_n(h) >= _MIN_SOURCE_N]
        some = [h for h in horizons if _max_abn_n(h) > 0]
        if well:
            ref = max(well)
        elif some:
            ref = max(some)
        else:
            ref = 3 if 3 in horizons else horizons[len(horizons) // 2]

    src_rows = []
    for fam, cells in acc.items():
        fin = _finalize(cells)
        ref_cell = fin.get(str(ref), {}) if ref is not None else {}
        an = ref_cell.get("n_abnormal") or 0
        abn = ref_cell.get("mean_abnormal_pct")
        hit = ref_cell.get("abnormal_hit_rate")
        graded = an >= _MIN_SOURCE_N and abn is not None
        if not graded:
            sv = "INSUFFICIENT"
        elif abn <= 0:
            sv = "NEGATIVE"
        elif abn > _EDGE_EPS and (hit or 0) >= 50.0:
            sv = "EXPLOITABLE"
        else:
            sv = "WEAK"
        src_rows.append({
            "source": fam,
            "n_resolved": resolved_per_src.get(fam, 0),
            "graded": graded,
            "ref_abnormal_pct": abn if graded else None,
            "verdict": sv,
            "horizons": fin,
        })

    # Graded best-first by abnormal return; ungraded after, by sample size.
    src_rows.sort(key=lambda s: (
        0 if s["graded"] else 1,
        -(s["ref_abnormal_pct"] if s["ref_abnormal_pct"] is not None else 0.0),
        -s["n_resolved"],
    ))

    graded = [s for s in src_rows if s["graded"]]
    best = graded[0] if graded else None
    worst = graded[-1] if graded else None

    if n_resolved == 0:
        verdict = "NO_DATA"
        reason = ("no scored live article resolved to a watchlist ticker with "
                  "a forward window — accumulate news/price history")
        headline = "NO_DATA: no scored live news resolved to a watchlist move"
    elif not graded:
        verdict = "INSUFFICIENT_DATA"
        reason = (f"{n_resolved} resolved across {len(src_rows)} source(s) but "
                  f"none has ≥{_MIN_SOURCE_N} SPY-adjusted {ref}d samples yet "
                  "— digital-intern's live news is only days-deep; matures with "
                  "history")
        headline = (f"INSUFFICIENT_DATA: {n_resolved} resolved, "
                    f"{len(src_rows)} sources, none gradable at {ref}d yet")
    else:
        ba = best["ref_abnormal_pct"]
        wa = worst["ref_abnormal_pct"]
        bn = (best["horizons"].get(str(ref), {}) or {}).get("n_abnormal") or 0
        if ba > _EDGE_EPS:
            verdict = "EDGE_FOUND"
            reason = (f"{best['source']} headlines lead {ba:+.2f}pp abnormal at "
                      f"{ref}d (n={bn}); {worst['source']} the weakest at "
                      f"{wa:+.2f}pp — weight/cut by source")
        else:
            verdict = "NO_EDGE"
            reason = (f"even the best collector ({best['source']}) is only "
                      f"{ba:+.2f}pp abnormal at {ref}d — source selection "
                      "shows no edge over the market")
        headline = (f"{verdict}: best {best['source']} {ba:+.2f}pp/{ref}d "
                    f"(n={bn}); worst {worst['source']} {wa:+.2f}pp "
                    f"[{len(graded)}/{len(src_rows)} graded]")

    return {
        "as_of": now.isoformat(timespec="seconds"),
        # n_articles = everything the caller fed in (incl. below-min_score
        # rows); n_scored = the subset at/above min_score that entered a
        # bucket. The gap is intentional honest reporting ("we saw N, only M
        # were worth scoring") — do NOT "fix" n_articles to mean scored-only.
        "n_articles": n_total,
        "n_scored": n_scored,
        "n_resolved": n_resolved,
        "min_score": min_score,
        "horizons": list(horizons),
        "reference_horizon": ref,
        "spy_adjusted": bool(spy_by_date),
        "sources": src_rows,
        "best_source": best["source"] if best else None,
        "worst_source": worst["source"] if worst else None,
        "verdict": verdict,
        "verdict_reason": reason,
        "headline": headline,
    }


if __name__ == "__main__":  # smoke test against the live DB + yfinance
    import json
    from pathlib import Path

    import yfinance as yf

    from paper_trader.strategy import WATCHLIST

    di = Path("/media/zeph/projects/digital-intern/db/articles.db")
    if not di.exists():
        di = Path("/home/zeph/digital-intern/data/articles.db")
    since = (datetime.now(timezone.utc).replace(microsecond=0)
             ).isoformat().replace("+00:00", "") + "-30 days"
    arts = _fetch_source_articles(
        str(di),
        (datetime.now(timezone.utc)).isoformat()[:10] + "T00:00:00",
        min_score=2.0)
    universe = sorted({tk for a in arts for tk in WATCHLIST
                       if re.search(rf"(?:\$|\b){tk}\b", a["text"].upper())})
    ph = {}
    for tk in universe + ["SPY"]:
        h = yf.Ticker(tk).history(period="2mo", auto_adjust=False)
        ph[tk] = [(d.strftime("%Y-%m-%d"), float(c))
                  for d, c in zip(h.index, h["Close"]) if c == c]
    spy = ph.pop("SPY", [])
    print(json.dumps(build_source_edge(arts, ph, spy, WATCHLIST), indent=2,
                      default=str))
