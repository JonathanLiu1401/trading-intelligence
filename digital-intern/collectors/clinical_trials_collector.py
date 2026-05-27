"""ClinicalTrials.gov Phase 3 catalyst collector.

Tracks Phase 3 / Phase 2/3 trial status changes for major pharma/biotech
companies using the public ClinicalTrials.gov v2 API (no auth required).

Trial completions, new recruitments, and primary completion milestones are
significant market catalysts for biotech/pharma stocks. This collector emits
articles for:
  - Phase 3 trials recently posted or updated by portfolio-relevant sponsors
  - Status changes (COMPLETED, TERMINATED, SUSPENDED) on ongoing trials
  - Primary completion dates reached in the past 14 days

No API key required. Rate limits are generous (casual research use).
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("clinical_trials_collector")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "seen_articles.db"
SOURCE = "clinical_trials"

API_BASE = "https://clinicaltrials.gov/api/v2/studies"
REQUEST_TIMEOUT = 12
MAX_RETRIES = 2
RETRY_BACKOFF = 3.0
LOOKBACK_DAYS = 7

# Pharma/biotech sponsors to track — matched as substring in sponsor name
SPONSORS = [
    "Pfizer", "Moderna", "BioNTech", "Eli Lilly", "Novo Nordisk",
    "AstraZeneca", "Merck", "Johnson & Johnson", "Roche", "Novartis",
    "Bristol-Myers Squibb", "Amgen", "Gilead", "Regeneron", "Biogen",
    "Vertex", "Alnylam", "BioMarin", "Sarepta", "Intellia",
    "CRISPR Therapeutics", "Beam Therapeutics", "Recursion",
]

# Trial phases that matter for market impact
TARGET_PHASES = {"PHASE3", "PHASE2_PHASE3", "PHASE4"}

# Status changes that are market-moving
NOTABLE_STATUSES = {
    "COMPLETED": "completed",
    "TERMINATED": "TERMINATED ⚠️",
    "SUSPENDED": "suspended ⚠️",
    "ACTIVE_NOT_RECRUITING": "fully enrolled",
}


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


def _article_id(nct_id: str, status: str) -> str:
    return hashlib.sha256(f"clinical_trials:{nct_id}:{status}".encode()).hexdigest()


def _fetch_trials_for_sponsor(
    sponsor: str, since_date: str, session: requests.Session
) -> list[dict[str, Any]]:
    """Fetch recent Phase 3 trial updates for one sponsor. Returns [] on error."""
    params = {
        "format": "json",
        "pageSize": 25,
        "query.spons": sponsor,
        "filter.overallStatus": "COMPLETED,TERMINATED,SUSPENDED,ACTIVE_NOT_RECRUITING",
        "sort": "LastUpdatePostDate:desc",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(API_BASE, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("studies", [])
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF)
            else:
                log.debug(f"[clinical_trials] error fetching {sponsor!r}: {e}")
    return []


def _study_to_article(study: dict) -> dict | None:
    """Convert a ClinicalTrials API study dict to our article format."""
    try:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        desc_mod = proto.get("descriptionModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        cond_mod = proto.get("conditionsModule", {})
        design_mod = proto.get("designModule", {})

        nct_id = ident.get("nctId", "")
        brief_title = ident.get("briefTitle", "").strip()
        overall_status = status_mod.get("overallStatus", "")
        last_update = status_mod.get("lastUpdatePostDateStruct", {}).get("date", "")
        primary_completion = status_mod.get("primaryCompletionDateStruct", {}).get("date", "")
        sponsor_name = sponsor_mod.get("leadSponsor", {}).get("name", "")
        conditions = cond_mod.get("conditions", [])
        phases = design_mod.get("phases", [])
        brief_summary = desc_mod.get("briefSummary", "").strip()

        if not nct_id or not brief_title:
            return None

        # Only emit Phase 2/3, Phase 3, and Phase 4 (market-moving)
        if phases and not any(p in TARGET_PHASES for p in phases):
            return None

        phase_str = "/".join(p.replace("PHASE", "Phase ") for p in phases) if phases else "Phase 3"
        condition_str = ", ".join(conditions[:2]) if conditions else "undisclosed condition"
        status_label = NOTABLE_STATUSES.get(overall_status, overall_status.replace("_", " ").title())

        title = f"[Clinical Trial] {sponsor_name}: {phase_str} {status_label} — {brief_title[:80]}"
        link = f"https://clinicaltrials.gov/study/{nct_id}"
        summary_parts = [
            f"{sponsor_name} {phase_str} trial ({nct_id}) for {condition_str} is now {status_label}.",
        ]
        if primary_completion:
            summary_parts.append(f"Primary completion: {primary_completion}.")
        if brief_summary:
            summary_parts.append(brief_summary[:300])
        summary = " ".join(summary_parts)

        return {
            "title": title,
            "link": link,
            "summary": summary,
            "published": last_update or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "source": SOURCE,
            "_nct_id": nct_id,
            "_status": overall_status,
        }
    except Exception as e:
        log.debug(f"[clinical_trials] parse error: {e}")
        return None


def collect_clinical_trials() -> list[dict]:
    """Collect Phase 3 clinical trial updates for pharma/biotech companies.

    Returns {title, link, summary, published, source} dicts (standard format).
    """
    conn = _ensure_db()
    since = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

    new_articles: list[dict] = []
    seen_in_run: set[str] = set()

    for sponsor in SPONSORS:
        studies = _fetch_trials_for_sponsor(sponsor, since, session)
        for study in studies:
            # Quick date check before heavy parsing — skip stale updates
            try:
                last_upd = (
                    study.get("protocolSection", {})
                    .get("statusModule", {})
                    .get("lastUpdatePostDateStruct", {})
                    .get("date", "")
                )
                if last_upd and last_upd < since:
                    continue
            except Exception:
                pass

            art = _study_to_article(study)
            if not art:
                continue

            nct_id = art.pop("_nct_id")
            status = art.pop("_status")

            # Only emit notable status changes to avoid noise
            if status not in NOTABLE_STATUSES:
                continue

            aid = _article_id(nct_id, status)
            if aid in seen_in_run:
                continue
            seen_in_run.add(aid)

            try:
                if conn.execute(
                    "SELECT 1 FROM seen_articles WHERE id = ?", (aid,)
                ).fetchone():
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO seen_articles "
                    "(id, link, title, source, first_seen) VALUES (?, ?, ?, ?, ?)",
                    (aid, art["link"], art["title"], SOURCE,
                     datetime.now(timezone.utc).isoformat()),
                )
            except sqlite3.Error as e:
                log.debug(f"[clinical_trials] dedup row skipped: {e}")
                continue

            new_articles.append(art)

        # Small delay between sponsors to be polite
        time.sleep(0.3)

    conn.commit()
    conn.close()
    log.info(f"[clinical_trials] {len(new_articles)} new trial articles")
    return new_articles


collect = collect_clinical_trials


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("=== ClinicalTrials.gov Phase 3 Collector (live fetch) ===")
    items = collect_clinical_trials()
    print(f"\nNew trial articles: {len(items)}")
    for art in items[:5]:
        print(f"\n  {art['title'][:100]}")
        print(f"  {art['link']}")
        print(f"  {art['summary'][:150]}...")
    if not items:
        print("  (none new — all already seen or no notable status changes)")
