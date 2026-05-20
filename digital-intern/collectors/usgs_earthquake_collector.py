"""USGS earthquake collector.

Pulls significant seismic events from the USGS public GeoJSON feed. M≥5.0
quakes near populated/industrial regions are a recurring, market-moving
catalyst no other collector covers:

  * Property & casualty insurers (ALL, TRV, AIG) and reinsurance (RNR, RE)
    repricing on catastrophe-loss expectations.
  * Semiconductor supply-chain disruption (TSMC fabs around Hsinchu /
    Tainan, Toyota / Murata plants in Japan, SK Hynix in Korea).
  * Energy infrastructure (LNG terminals, refineries, offshore platforms)
    and shipping lane closures (Suez/Bosphorus historical precedents).
  * Sovereign / muni bond credit spreads in affected regions.

Feed: ``4.5_day.geojson`` — every M≥4.5 in the past 24h, refreshed by USGS
every minute. We poll every 30min and filter to M≥5.0 (USGS's own
"significant" weighting also pulls in low-mag-but-high-impact events; we
prefer a hard magnitude floor so the threshold is stable and auditable —
``tsunami=1`` lower-magnitude events are also kept since the cross-Pacific
warning itself is the market signal regardless of recorded magnitude).

Dedup pattern matches ``cisa_kev_collector`` / ``treasury_auctions``:
  1. shared ``data/seen_articles.db`` (WAL, busy_timeout=30000) keyed by
     the USGS event id (sha256 of ``"usgs:" + event_id`` — stable across
     magnitude revisions of the same quake).
  2. ``articles.db`` PRIMARY KEY = sha256(url||title) inside ``insert_batch``.

Returned dicts use the standard collector contract (``title``, ``link``,
``summary``, ``source``, ``published``); ``_ingest()`` runs them through
the heuristic scorer and ``insert_batch`` verbatim.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"

# 4.5_day is high enough cadence (1-min refresh on USGS side) to surface
# emerging events within our 30min poll, low-volume enough (~15/day) to
# never stress insert_batch. The "_week" variant goes 1000+ rows and the
# 30-day variants are not relevant on a financial timescale.
ENDPOINT = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"

# Minimum magnitude to surface as an article. M5.0 is the empirical lower
# bound where damage / industrial-shutdown headlines reliably appear. M<5
# events still arrive in the feed but are filtered out here; raise this
# only with care — historical Tōhoku-aftershock M4.7s did move TM/SNE.
MIN_MAGNITUDE = 5.0
SOURCE = "usgs_earthquake"


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS seen_articles (
            id TEXT PRIMARY KEY, link TEXT, title TEXT,
            source TEXT, first_seen TEXT
        )"""
    )
    conn.commit()
    return conn


def _article_id(event_id: str) -> str:
    # USGS event_id (e.g. "us6000sy84") is stable across magnitude revisions
    # — the quake itself doesn't change, only the estimate does. Use that
    # directly so a M6.0→M6.2 revision doesn't double-emit.
    return hashlib.sha256(f"usgs:{event_id}".encode()).hexdigest()


def _fmt_title(props: dict, mag: float) -> str:
    place = (props.get("place") or "unknown location").strip()
    tsunami_tag = " — TSUNAMI WARNING" if props.get("tsunami") == 1 else ""
    return f"M{mag:.1f} earthquake: {place}{tsunami_tag}"


def _fmt_summary(props: dict, geometry: dict, mag: float) -> str:
    place = (props.get("place") or "").strip()
    coords = (geometry or {}).get("coordinates") or []
    # USGS GeoJSON coordinates are [lon, lat, depth_km].
    lon = coords[0] if len(coords) > 0 else None
    lat = coords[1] if len(coords) > 1 else None
    depth = coords[2] if len(coords) > 2 else None
    parts = [f"Magnitude {mag:.1f} earthquake"]
    if place:
        parts.append(f"at {place}")
    if depth is not None:
        try:
            parts.append(f"depth {float(depth):.0f}km")
        except (TypeError, ValueError):
            pass
    if lat is not None and lon is not None:
        try:
            parts.append(f"({float(lat):.2f}, {float(lon):.2f})")
        except (TypeError, ValueError):
            pass
    if props.get("tsunami") == 1:
        parts.append("Tsunami warning issued.")
    felt = props.get("felt")
    if felt:
        parts.append(f"{felt} felt reports.")
    alert = props.get("alert")
    if alert:
        # USGS PAGER alert level: green/yellow/orange/red — orange/red are
        # the casualty/damage tiers that usually drive market reaction.
        parts.append(f"PAGER alert: {alert}.")
    return ". ".join(parts) + "."


def collect_usgs_earthquakes(limit: int = 50) -> list[dict]:
    try:
        r = requests.get(
            ENDPOINT,
            headers={"User-Agent": "Digital-Intern/1.0 (+seismic-collector)"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[usgs_earthquake] fetch error: {e}")
        return []

    features = data.get("features") if isinstance(data, dict) else None
    if not isinstance(features, list):
        return []

    conn = _ensure_db()
    out: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for feat in features[:limit]:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        event_id = feat.get("id") or ""
        if not event_id:
            continue
        mag = props.get("mag")
        try:
            mag_f = float(mag) if mag is not None else None
        except (TypeError, ValueError):
            mag_f = None
        # M<5.0 is filtered unless USGS has issued a tsunami warning — the
        # warning itself is the market-relevant signal even for smaller quakes.
        if mag_f is None:
            continue
        if mag_f < MIN_MAGNITUDE and props.get("tsunami") != 1:
            continue

        aid = _article_id(event_id)
        if conn.execute("SELECT 1 FROM seen_articles WHERE id=?", (aid,)).fetchone():
            continue

        title = _fmt_title(props, mag_f)
        summary = _fmt_summary(props, geom, mag_f)
        # USGS hosts an event page per id; this is the canonical permalink.
        link = props.get("url") or f"https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}"
        # event_time_ms is unix-millis; convert to ISO so the rest of the
        # pipeline (published-recency boosts, time-decay ranking) treats it
        # like any other timestamped article.
        event_time_ms = props.get("time")
        if isinstance(event_time_ms, (int, float)) and event_time_ms > 0:
            published = datetime.fromtimestamp(event_time_ms / 1000.0,
                                               tz=timezone.utc).isoformat()
        else:
            published = now_iso

        conn.execute(
            "INSERT OR IGNORE INTO seen_articles (id, link, title, source, first_seen) "
            "VALUES (?, ?, ?, ?, ?)",
            (aid, link, title, SOURCE, now_iso),
        )
        out.append({
            "id": aid,
            "title": title,
            "link": link,
            "summary": summary,
            "source": SOURCE,
            "published": published,
            "first_seen": now_iso,
            "magnitude": mag_f,
            "tsunami": bool(props.get("tsunami") == 1),
            "place": props.get("place"),
            "event_id": event_id,
        })

    conn.commit()
    conn.close()
    return out


if __name__ == "__main__":
    items = collect_usgs_earthquakes()
    print(f"[usgs_earthquake] new items: {len(items)}")
    for it in items[:10]:
        print(f"  M{it['magnitude']:.1f}  {it['title']}")
        print(f"    -> {it['link']}")
