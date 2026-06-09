"""Portfolio-construction target for the live paper book.

This is the deterministic implementation of the "Portfolio Construction"
prompt pattern: given candidate assets, a risk tolerance, and a time horizon,
return allocation %, expected-return proxy, risk proxy, and the reason each
asset is included.

It is advisory only. It never gates the live trader, never places orders, and
never enforces a rebalance. The value is that Opus, the dashboard, and tests
can all see the same concrete target book instead of hand-waving about
"diversification" while the live portfolio drifts into one sector.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone


_RISK_PROFILES = {
    "low": {
        "cash_buffer_pct": 20.0,
        "max_single_name_pct": 20.0,
        "max_sector_pct": 35.0,
        "max_leveraged_pct": 5.0,
        "max_assets": 8,
    },
    "medium": {
        "cash_buffer_pct": 10.0,
        "max_single_name_pct": 35.0,
        "max_sector_pct": 45.0,
        "max_leveraged_pct": 10.0,
        "max_assets": 8,
    },
    "high": {
        "cash_buffer_pct": 5.0,
        "max_single_name_pct": 45.0,
        "max_sector_pct": 60.0,
        "max_leveraged_pct": 15.0,
        "max_assets": 10,
    },
}

_BROAD_DIVERSIFIERS = {"SPY", "QQQ", "VOO", "VTI"}


def _f(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _z(value: float | None, ndigits: int = 2) -> float | None:
    if value is None:
        return None
    out = round(float(value), ndigits)
    return 0.0 if out == 0 else out


def _risk_key(value: str | None) -> str:
    key = (value or "medium").strip().lower()
    return key if key in _RISK_PROFILES else "medium"


def _horizon_factor(value: str | None) -> tuple[str, float, float]:
    """Return (normalized label, return_scale, risk_cap_adjustment_pct)."""
    raw = (value or "1-3 years").strip()
    low = raw.lower()
    if any(tok in low for tok in ("day", "week", "month", "6m", "3m",
                                  "<1", "0-1")):
        return raw, 0.45, -5.0
    if any(tok in low for tok in ("5+", "5 years", "long", "multi-year")):
        return raw, 1.25, 5.0
    return raw, 1.0, 0.0


def _position_value(pos: dict) -> float:
    if pos.get("market_value") is not None:
        return _f(pos.get("market_value"))
    mult = 100 if pos.get("type") in ("call", "put") else 1
    price = _f(pos.get("current_price")) or _f(pos.get("avg_cost"))
    return price * _f(pos.get("qty")) * mult


def _signal_stats(signal_rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in signal_rows or []:
        score = _f(row.get("ai_score"))
        urgency = _f(row.get("urgency"))
        title = str(row.get("title") or "")[:140]
        for raw in row.get("tickers") or []:
            tk = str(raw or "").upper().strip()
            if not tk or len(tk) > 6:
                continue
            rec = out.setdefault(tk, {
                "news_count": 0,
                "news_max_score": 0.0,
                "news_urgent": False,
                "top_headline": None,
            })
            rec["news_count"] += 1
            if score > rec["news_max_score"]:
                rec["news_max_score"] = score
                rec["top_headline"] = title or None
            if urgency >= 1:
                rec["news_urgent"] = True
    return out


def _quant_return_proxy(ticker: str, quant: dict, news: dict,
                        sector: str, beta: float,
                        horizon_scale: float) -> tuple[float, list[str]]:
    """Expected-return proxy in percent over the requested horizon.

    This is deliberately simple and transparent. It is not a fitted forecast;
    it turns the same live evidence the desk already shows into a bounded
    allocation input.
    """
    reasons: list[str] = []
    score = _f(news.get("news_max_score"))
    news_count = int(_f(news.get("news_count")))
    edge = 0.0

    if news_count:
        edge += max(-2.0, min(8.0, (score - 5.0) * 1.6))
        reasons.append(f"news score {score:.1f} across {news_count} catalyst(s)")
        if news.get("news_urgent"):
            edge += 1.0
            reasons.append("urgent catalyst")

    mom_20d = quant.get("mom_20d")
    mom_5d = quant.get("mom_5d")
    if mom_20d is not None:
        m20 = max(-12.0, min(12.0, _f(mom_20d)))
        edge += 0.35 * m20
        reasons.append(f"20d momentum {m20:+.1f}%")
    if mom_5d is not None:
        m5 = max(-8.0, min(8.0, _f(mom_5d)))
        edge += 0.20 * m5

    macd = str(quant.get("MACD") or quant.get("macd") or "").lower()
    if "bull" in macd:
        edge += 2.0
        reasons.append("MACD bullish")
    elif "bear" in macd:
        edge -= 2.0
        reasons.append("MACD bearish")

    ma_cross = str(quant.get("MA_cross") or quant.get("ma_cross") or "").lower()
    if "bull" in ma_cross:
        edge += 1.2
        reasons.append("MA trend bullish")
    elif "bear" in ma_cross:
        edge -= 1.2
        reasons.append("MA trend bearish")

    rsi = quant.get("rsi", quant.get("RSI"))
    if rsi is not None:
        rv = _f(rsi)
        if rv >= 70.0:
            edge -= 2.0
            reasons.append(f"RSI {rv:.1f} overbought")
        elif rv <= 35.0:
            edge += 1.0
            reasons.append(f"RSI {rv:.1f} reset/oversold")

    if ticker in _BROAD_DIVERSIFIERS:
        edge += 1.0
        reasons.append("broad-market diversifier")
    elif sector != "other":
        reasons.append(f"{sector} exposure")

    # Higher-beta instruments can earn more, but the raw score should not
    # let 3x products dominate a construction target by construction.
    if sector.endswith("_lev") or abs(beta) >= 2.5:
        edge -= 1.0
        reasons.append("leveraged-instrument decay risk")

    if not reasons:
        reasons.append("candidate asset in current watch universe")

    expected = max(-10.0, min(20.0, edge * horizon_scale))
    return _z(expected), reasons[:4]


def _risk_proxy(sector: str, beta: float, quant: dict,
                current_weight_pct: float) -> tuple[float, float]:
    base_vol = 9.0 + abs(beta) * 7.0
    if sector.endswith("_lev") or abs(beta) >= 2.5:
        base_vol += 8.0
    bb = quant.get("bb_position")
    if bb is not None:
        base_vol += min(4.0, abs(_f(bb)) * 1.5)
    if current_weight_pct >= 40.0:
        base_vol += 2.0
    vol = max(8.0, min(45.0, base_vol))
    drawdown = -min(75.0, vol * 1.6)
    return _z(vol), _z(drawdown)


def _candidate_score(asset: dict, risk_key: str) -> float:
    exp = _f(asset.get("expected_return_pct"))
    vol = max(1.0, _f(asset.get("volatility_pct"), 20.0))
    news = _f(asset.get("news_max_score"))
    current = _f(asset.get("current_weight_pct"))
    score = max(0.05, (exp + 10.0) / vol)
    score += min(0.45, news / 25.0)
    if current > 0:
        score += min(0.20, current / 250.0)
    if asset["ticker"] in _BROAD_DIVERSIFIERS:
        score += 0.25 if risk_key == "low" else 0.15
    if asset.get("leveraged"):
        score *= 0.55 if risk_key == "low" else 0.75
    return max(0.01, score)


def _allocate(candidates: list[dict], profile: dict) -> tuple[dict[str, float], float]:
    deployable = max(0.0, 100.0 - profile["cash_buffer_pct"])
    targets = {c["ticker"]: 0.0 for c in candidates}
    sector_used: dict[str, float] = {}

    for _ in range(12):
        remaining = deployable - sum(targets.values())
        if remaining <= 0.01:
            break
        open_rows = []
        for c in candidates:
            single_cap = profile["max_single_name_pct"]
            if c.get("leveraged"):
                single_cap = min(single_cap, profile["max_leveraged_pct"])
            single_room = single_cap - targets[c["ticker"]]
            sector_room = (
                profile["max_sector_pct"]
                - sector_used.get(c["sector"], 0.0)
            )
            if single_room > 0.01 and sector_room > 0.01:
                open_rows.append(c)
        if not open_rows:
            break

        score_sum = sum(c["allocation_score"] for c in open_rows)
        if score_sum <= 0:
            break
        progressed = False
        for c in open_rows:
            single_cap = profile["max_single_name_pct"]
            if c.get("leveraged"):
                single_cap = min(single_cap, profile["max_leveraged_pct"])
            single_room = single_cap - targets[c["ticker"]]
            sector_room = (
                profile["max_sector_pct"]
                - sector_used.get(c["sector"], 0.0)
            )
            room = min(single_room, sector_room)
            if room <= 0.01:
                continue
            want = remaining * c["allocation_score"] / score_sum
            add = min(want, room)
            if add <= 0.001:
                continue
            tk = c["ticker"]
            targets[tk] += add
            sector_used[c["sector"]] = sector_used.get(c["sector"], 0.0) + add
            progressed = True
        if not progressed:
            break

    leftover_cash = max(0.0, 100.0 - profile["cash_buffer_pct"] - sum(targets.values()))
    return ({tk: _z(v) for tk, v in targets.items() if v >= 0.01},
            _z(profile["cash_buffer_pct"] + leftover_cash))


def build_portfolio_construction(
    snapshot: dict,
    signal_rows: list[dict],
    quant_signals: dict[str, dict],
    prices: dict[str, float | None],
    classify,
    beta_map: dict,
    risk_tolerance: str = "medium",
    time_horizon: str = "1-3 years",
    fallback_tickers: list[str] | tuple[str, ...] | None = None,
    now: datetime | None = None,
) -> dict:
    """Build a target allocation over held names plus live signal candidates."""
    now = now or datetime.now(timezone.utc)
    risk = _risk_key(risk_tolerance)
    horizon_label, horizon_scale, cap_adjust = _horizon_factor(time_horizon)
    profile = dict(_RISK_PROFILES[risk])
    profile["max_single_name_pct"] = max(
        5.0, profile["max_single_name_pct"] + cap_adjust)
    profile["max_sector_pct"] = max(20.0, profile["max_sector_pct"] + cap_adjust)
    if cap_adjust < 0:
        profile["cash_buffer_pct"] += abs(cap_adjust)

    snap = snapshot or {}
    positions = list(snap.get("positions") or [])
    total_value = _f(snap.get("total_value"))
    cash = _f(snap.get("cash"))
    if total_value <= 0:
        total_value = cash + sum(max(0.0, _position_value(p)) for p in positions)

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "risk_tolerance": risk,
        "time_horizon": horizon_label,
        "constraints": {k: _z(v) for k, v in profile.items()
                        if k != "max_assets"},
        "total_value": _z(total_value),
        "cash_current_pct": _z(cash / total_value * 100.0) if total_value > 0 else None,
        "cash_target_pct": None,
        "allocations": [],
        "portfolio_expected_return_pct": None,
        "portfolio_volatility_pct": None,
        "portfolio_drawdown_pct": None,
        "diversification": {},
        "prompt_block": None,
    }

    if total_value <= 0:
        base["state"] = "NO_DATA"
        base["headline"] = "Portfolio construction: no portfolio value to allocate."
        return base

    current_value: dict[str, float] = {}
    for p in positions:
        tk = str(p.get("ticker") or "").upper()
        if not tk:
            continue
        current_value[tk] = current_value.get(tk, 0.0) + max(0.0, _position_value(p))

    sig_stats = _signal_stats(signal_rows)
    universe = set(current_value) | set(sig_stats) | {
        str(t).upper() for t in (quant_signals or {}) if t
    }
    for tk in ("SPY", "QQQ"):
        if tk in (prices or {}) or tk in (quant_signals or {}):
            universe.add(tk)
    if len(universe) < 4:
        for tk in fallback_tickers or ():
            tk = str(tk or "").upper()
            if tk:
                universe.add(tk)
            if len(universe) >= 6:
                break

    candidates: list[dict] = []
    for tk in sorted(universe):
        q = (quant_signals or {}).get(tk) or {}
        news = sig_stats.get(tk) or {
            "news_count": 0,
            "news_max_score": 0.0,
            "news_urgent": False,
            "top_headline": None,
        }
        sector = classify(tk)
        beta = _f(beta_map.get(sector), 1.0)
        current_weight = current_value.get(tk, 0.0) / total_value * 100.0
        exp, reasons = _quant_return_proxy(tk, q, news, sector, beta,
                                           horizon_scale)
        vol, dd = _risk_proxy(sector, beta, q, current_weight)
        asset = {
            "ticker": tk,
            "sector": sector,
            "price": prices.get(tk) if prices else None,
            "current_weight_pct": _z(current_weight),
            "news_count": int(news.get("news_count") or 0),
            "news_max_score": _z(news.get("news_max_score") or 0.0),
            "top_headline": news.get("top_headline"),
            "expected_return_pct": exp,
            "volatility_pct": vol,
            "max_drawdown_pct": dd,
            "beta": _z(beta, 3),
            "leveraged": sector.endswith("_lev") or abs(beta) >= 2.5,
            "why_included": (["held position"] if current_weight > 0 else [])
            + reasons,
        }
        asset["allocation_score"] = _candidate_score(asset, risk)
        candidates.append(asset)

    if not candidates:
        base["state"] = "NO_DATA"
        base["headline"] = "Portfolio construction: no held or candidate assets."
        return base

    # Keep all held names, then the highest-scoring candidates.
    held = {tk for tk, value in current_value.items() if value > 0}
    candidates.sort(key=lambda c: (
        c["ticker"] not in held,
        -c["allocation_score"],
        c["ticker"],
    ))
    max_assets = int(profile["max_assets"])
    chosen = [c for c in candidates if c["ticker"] in held]
    for c in candidates:
        if c["ticker"] in held:
            continue
        if len(chosen) >= max_assets:
            break
        chosen.append(c)
    chosen.sort(key=lambda c: (-c["allocation_score"], c["ticker"]))

    targets, cash_target = _allocate(chosen, profile)
    base["cash_target_pct"] = cash_target

    by_sector: dict[str, float] = {}
    allocations = []
    for c in chosen:
        target = targets.get(c["ticker"], 0.0)
        by_sector[c["sector"]] = by_sector.get(c["sector"], 0.0) + target
        delta = target - _f(c.get("current_weight_pct"))
        c_out = {k: v for k, v in c.items() if k != "allocation_score"}
        c_out.update({
            "target_allocation_pct": _z(target),
            "delta_pct": _z(delta),
            "delta_usd": _z(delta / 100.0 * total_value),
        })
        allocations.append(c_out)

    allocations.sort(key=lambda r: (-_f(r.get("target_allocation_pct")),
                                    r["ticker"]))
    base["allocations"] = allocations
    base["n_assets"] = len(allocations)

    target_sum = sum(_f(a.get("target_allocation_pct")) for a in allocations)
    if target_sum > 0:
        base["portfolio_expected_return_pct"] = _z(sum(
            _f(a.get("target_allocation_pct")) * _f(a.get("expected_return_pct"))
            for a in allocations
        ) / 100.0)
        vol = sum(
            _f(a.get("target_allocation_pct")) * _f(a.get("volatility_pct"))
            for a in allocations
        ) / 100.0
        max_sector = max(by_sector.values()) if by_sector else 0.0
        base["portfolio_volatility_pct"] = _z(vol)
        base["portfolio_drawdown_pct"] = _z(-(vol * 1.45 + max_sector * 0.12))

    invested = sum(by_sector.values())
    hhi = (sum((v / invested) ** 2 for v in by_sector.values())
           if invested > 0 else None)
    max_sector = max(by_sector.values()) if by_sector else 0.0
    current_by_sector: dict[str, float] = {}
    for tk, value in current_value.items():
        sec = classify(tk)
        current_by_sector[sec] = current_by_sector.get(sec, 0.0) + value
    current_max_sector = (
        max(current_by_sector.values()) / total_value * 100.0
        if current_by_sector else 0.0
    )
    base["diversification"] = {
        "target_sector_pct": {k: _z(v) for k, v in sorted(by_sector.items())},
        "target_max_sector_pct": _z(max_sector),
        "target_sector_hhi": _z(hhi, 4) if hhi is not None else None,
        "current_max_sector_pct": _z(current_max_sector),
    }

    constrained = cash_target and cash_target > profile["cash_buffer_pct"] + 0.5
    over_current = current_max_sector > profile["max_sector_pct"]
    if constrained:
        state = "CONSTRAINED"
    elif over_current:
        state = "REBALANCE_AWARE"
    else:
        state = "OK"
    base["state"] = state
    base["headline"] = (
        f"Portfolio construction [{risk}, {horizon_label}]: "
        f"{len(allocations)} asset target, cash {cash_target:.1f}%, "
        f"target max sector {max_sector:.1f}%."
    )

    prompt_lines = [
        "PORTFOLIO CONSTRUCTION (target allocation, expected return, and "
        "risk proxy; advisory only, NOT a directive or limit):",
        f"  constraints: risk={risk} horizon={horizon_label} "
        f"max_single={profile['max_single_name_pct']:.0f}% "
        f"max_sector={profile['max_sector_pct']:.0f}% "
        f"cash_target={cash_target:.1f}%",
    ]
    for a in allocations[:6]:
        why = "; ".join(a.get("why_included") or [])[:110]
        prompt_lines.append(
            f"  {a['ticker']}: target={_f(a.get('target_allocation_pct')):.1f}% "
            f"now={_f(a.get('current_weight_pct')):.1f}% "
            f"delta={_f(a.get('delta_pct')):+.1f}% "
            f"exp={_f(a.get('expected_return_pct')):+.1f}% "
            f"vol={_f(a.get('volatility_pct')):.1f}% "
            f"dd={_f(a.get('max_drawdown_pct')):+.1f}% "
            f"why={why}"
        )
    prompt_lines.append(
        f"  portfolio proxy: exp="
        f"{_f(base.get('portfolio_expected_return_pct')):+.1f}% "
        f"vol={_f(base.get('portfolio_volatility_pct')):.1f}% "
        f"drawdown={_f(base.get('portfolio_drawdown_pct')):+.1f}%"
    )
    base["prompt_block"] = "\n".join(prompt_lines)
    return base
