from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "project" / "notebooks" / "L3_9_2_homepage_crawling.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        r"""
# Homepage Crawling for Archive-Proxy Features

This notebook builds the homepage archive-proxy input needed by the exit prediction notebook. It queries the Internet Archive CDX API for company homepage domains at annual year-end cutoff dates and writes files named `homepage_domain_archive_proxy.csv` under `data/live_root/Crawl`.

The resulting files can be picked up by `L3_9_2_VC_and_PE_Exit_Prediction.ipynb` to construct:

- `cc_presence_streak`
- `cc_first_seen_web_age`
- `cc_decay_transition_flag`
- `cc_coverage_missing_flag`

The crawler is resumable. By default it runs a small smoke test; set `RUN_FULL_CRAWL = True` only when ready for a long run.

To keep runtime manageable, each domain is queried once for a CDX snapshot list covering 2010-2023. The notebook then maps that single response to all annual cutoff dates locally.
"""
    ),
    md(
        r"""
## Design

For each normalized company website domain and each annual cutoff date from 2010 through 2023, the notebook asks whether the Wayback Machine has a usable homepage snapshot at or before that cutoff. A positive archive hit is treated as evidence that the homepage was present before the model observation date.

The downstream model applies an additional leakage control: a cutoff observed in quarter `q` first becomes visible in quarter `q + 1`.

This notebook does not use exit labels, funding outcomes, or model predictions.
"""
    ),
    code(
        r"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

try:
    import requests
except Exception as exc:
    raise RuntimeError("This notebook needs the requests package for HTTP crawling.") from exc

PROJECT_ROOT = Path(r"D:\Columbia\SummerProject\L3_9_2_VC_and_PE_Exit_Prediction_distribution")
LIVE_ROOT = PROJECT_ROOT / "data" / "live_root"
COMPANIES_PATH = LIVE_ROOT / "CrunchBase" / "companies.csv"

OUTPUT_ROOT = LIVE_ROOT / "Crawl"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CUTOFF_DATES = [f"{year}-12-31" for year in range(2010, 2024)]

RUN_FULL_CRAWL = False
SMOKE_TEST_DOMAINS = 250
SLEEP_SECONDS = 0.35
REQUEST_TIMEOUT_SECONDS = 25
USER_AGENT = "academic-homepage-archive-proxy/1.0"
CDX_FROM_DATE = "2010-01-01"
CDX_TO_DATE = "2023-12-31"
CDX_MAX_ROWS_PER_DOMAIN = 5000
ACTIVE_LOOKBACK_DAYS = 730

print("Companies:", COMPANIES_PATH)
print("Output root:", OUTPUT_ROOT)
"""
    ),
    md("## Domain Queue"),
    code(
        r"""
def normalize_domain(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.split("/")[0].split("?")[0].strip()
    s = s.split(":")[0].strip()
    if not s or "." not in s:
        return ""
    return s

companies = pd.read_csv(
    COMPANIES_PATH,
    usecols=["company_uuid", "name", "website", "founded_on"],
    low_memory=False,
)
companies["website_domain"] = companies["website"].map(normalize_domain)
domain_company_mapping = (
    companies[companies["website_domain"].ne("")]
    [["company_uuid", "website_domain", "website", "founded_on"]]
    .drop_duplicates()
    .copy()
)
domains = (
    domain_company_mapping[["website_domain"]]
    .drop_duplicates()
    .sort_values("website_domain")
    .reset_index(drop=True)
)

if not RUN_FULL_CRAWL:
    domains = domains.head(SMOKE_TEST_DOMAINS).copy()

print("Company-domain rows:", len(domain_company_mapping))
print("Unique domains queued:", len(domains))
display(domains.head())
"""
    ),
    md("## Wayback CDX Helpers"),
    code(
        r"""
def cutoff_to_timestamp(cutoff_date: str) -> str:
    return pd.Timestamp(cutoff_date).strftime("%Y%m%d") + "235959"

def date_to_timestamp(date_value: str, end_of_day: bool = False) -> str:
    suffix = "235959" if end_of_day else "000000"
    return pd.Timestamp(date_value).strftime("%Y%m%d") + suffix

def cdx_list_url(domain: str) -> str:
    params = {
        "url": f"{domain}/",
        "matchType": "prefix",
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype,digest",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",
        "from": date_to_timestamp(CDX_FROM_DATE),
        "to": date_to_timestamp(CDX_TO_DATE, end_of_day=True),
        "limit": str(CDX_MAX_ROWS_PER_DOMAIN),
    }
    return "https://web.archive.org/cdx?" + urlencode(params)

def empty_cutoff_row(domain: str, cutoff_date: str, status: str, failure_reason: str, failure_class: str, cdx_http_status: int = 0) -> dict:
    base = {
        "website_domain": domain,
        "cutoff_date": cutoff_date,
        "source_url": cdx_list_url(domain),
        "homepage_archive_status": status,
        "snapshot_timestamp": "",
        "http_status": pd.NA,
        "final_url": "",
        "failure_reason": failure_reason,
        "archive_method": "wayback_cdx_list",
        "evidence_strength": "none",
        "failure_class": failure_class,
        "evidence_scope": "homepage_or_domain_prefix",
        "needs_homepage_validation": True,
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "cdx_http_status": cdx_http_status,
    }
    return base

def snapshot_cutoff_row(domain: str, cutoff_date: str, row: dict, evidence_strength: str, cdx_http_status: int) -> dict:
    timestamp = row.get("timestamp", "")
    original = row.get("original", "")
    http_status = row.get("statuscode", "")
    mimetype = row.get("mimetype", "")
    digest = row.get("digest", "")
    return {
        "website_domain": domain,
        "cutoff_date": cutoff_date,
        "source_url": cdx_list_url(domain),
        "homepage_archive_status": "archived_200",
        "snapshot_timestamp": timestamp,
        "http_status": int(http_status) if str(http_status).isdigit() else pd.NA,
        "final_url": original,
        "failure_reason": "",
        "archive_method": "wayback_cdx_list",
        "evidence_strength": evidence_strength,
        "failure_class": "",
        "evidence_scope": "homepage_or_domain_prefix",
        "needs_homepage_validation": evidence_strength != "strong",
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "cdx_http_status": cdx_http_status,
        "mimetype": mimetype,
        "digest": digest,
    }

def parse_cdx_payload(payload) -> list[dict]:
    if not isinstance(payload, list) or len(payload) <= 1:
        return []
    header = payload[0]
    rows = []
    for raw in payload[1:]:
        if not isinstance(raw, list):
            continue
        item = dict(zip(header, raw))
        ts = pd.to_datetime(item.get("timestamp", ""), format="%Y%m%d%H%M%S", errors="coerce")
        if pd.notna(ts):
            item["snapshot_dt"] = ts
            rows.append(item)
    return rows

def rows_for_cutoffs(domain: str, snapshots: list[dict], cdx_http_status: int, error: str = "") -> list[dict]:
    if error:
        return [
            empty_cutoff_row(domain, cutoff, "request_error", error, "request_error", cdx_http_status)
            for cutoff in CUTOFF_DATES
        ]
    if not snapshots:
        return [
            empty_cutoff_row(domain, cutoff, "missing", "no_cdx_snapshot_in_query_window", "no_archive_hit", cdx_http_status)
            for cutoff in CUTOFF_DATES
        ]

    snapshots = sorted(snapshots, key=lambda x: x["snapshot_dt"])
    out = []
    for cutoff in CUTOFF_DATES:
        cutoff_dt = pd.Timestamp(cutoff)
        lookback_start = cutoff_dt - pd.Timedelta(days=ACTIVE_LOOKBACK_DAYS)
        eligible = [s for s in snapshots if lookback_start <= s["snapshot_dt"] <= cutoff_dt]
        if eligible:
            chosen = eligible[-1]
            mimetype = str(chosen.get("mimetype", "")).lower()
            strength = "strong" if mimetype.startswith("text/html") else "medium"
            out.append(snapshot_cutoff_row(domain, cutoff, chosen, strength, cdx_http_status))
            continue
        any_prior = any(s["snapshot_dt"] <= cutoff_dt for s in snapshots)
        if any_prior:
            out.append(empty_cutoff_row(
                domain, cutoff, "stale_snapshot", f"no_snapshot_within_{ACTIVE_LOOKBACK_DAYS}_days_before_cutoff",
                "stale_archive", cdx_http_status,
            ))
        else:
            out.append(empty_cutoff_row(
                domain, cutoff, "missing", "no_cdx_snapshot_before_cutoff", "no_archive_hit", cdx_http_status,
            ))
    return out

def query_wayback_domain(domain: str, session: requests.Session) -> tuple[list[dict], dict]:
    url = cdx_list_url(domain)
    log = {
        "website_domain": domain,
        "source_url": url,
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "cdx_http_status": 0,
        "snapshot_rows": 0,
        "error": "",
    }
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        log["cdx_http_status"] = int(response.status_code)
        response.raise_for_status()
        snapshots = parse_cdx_payload(response.json())
        log["snapshot_rows"] = len(snapshots)
        return rows_for_cutoffs(domain, snapshots, log["cdx_http_status"]), log
    except Exception as exc:
        log["error"] = str(exc)
        return rows_for_cutoffs(domain, [], log["cdx_http_status"], str(exc)), log
"""
    ),
    md("## Run Resumable Crawl"),
    code(
        r"""
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

summary_rows = []
rows_by_cutoff = {}
done_by_cutoff = {}
output_paths = {}

for cutoff_date in CUTOFF_DATES:
    run_dir = OUTPUT_ROOT / f"cutoff={cutoff_date}_crawl=wayback_cdx"
    run_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = run_dir / "homepage_domain_archive_proxy.csv"
    mapping_path = run_dir / "domain_company_mapping.csv"
    domains_path = run_dir / "domains.csv"

    domain_company_mapping.to_csv(mapping_path, index=False)
    domains.to_csv(domains_path, index=False)

    if proxy_path.exists():
        existing = pd.read_csv(proxy_path, low_memory=False)
        done_by_cutoff[cutoff_date] = set(existing["website_domain"].dropna().astype(str))
        rows_by_cutoff[cutoff_date] = existing.to_dict("records")
    else:
        done_by_cutoff[cutoff_date] = set()
        rows_by_cutoff[cutoff_date] = []
    output_paths[cutoff_date] = proxy_path

done_all_cutoffs = set.intersection(*done_by_cutoff.values()) if done_by_cutoff else set()
todo = [d for d in domains["website_domain"].tolist() if d not in done_all_cutoffs]
print(f"CDX list crawl: existing_all_cutoffs={len(done_all_cutoffs):,}, todo={len(todo):,}")

query_log_path = OUTPUT_ROOT / "domain_cdx_query_log.csv"
query_logs = pd.read_csv(query_log_path, low_memory=False).to_dict("records") if query_log_path.exists() else []

for i, domain in enumerate(todo, start=1):
    cutoff_rows, log_row = query_wayback_domain(domain, session)
    query_logs.append(log_row)
    for row in cutoff_rows:
        cutoff = row["cutoff_date"]
        if domain not in done_by_cutoff[cutoff]:
            rows_by_cutoff[cutoff].append(row)
            done_by_cutoff[cutoff].add(domain)
    if i % 100 == 0 or i == len(todo):
        for cutoff_date, rows in rows_by_cutoff.items():
            pd.DataFrame(rows).to_csv(output_paths[cutoff_date], index=False)
        pd.DataFrame(query_logs).to_csv(query_log_path, index=False)
        print(f"  saved after {i:,}/{len(todo):,} domain-level CDX queries")
    time.sleep(SLEEP_SECONDS)

for cutoff_date, rows in rows_by_cutoff.items():
    proxy_path = output_paths[cutoff_date]
    frame = pd.DataFrame(rows)
    frame.to_csv(proxy_path, index=False)
    summary_rows.append({
        "cutoff_date": cutoff_date,
        "rows": len(frame),
        "archived_200_rows": int(frame["homepage_archive_status"].eq("archived_200").sum()) if len(frame) else 0,
        "missing_rows": int(frame["homepage_archive_status"].ne("archived_200").sum()) if len(frame) else 0,
        "output_path": str(proxy_path),
    })

crawl_summary = pd.DataFrame(summary_rows)
crawl_summary.to_csv(OUTPUT_ROOT / "homepage_crawl_summary.csv", index=False)
display(crawl_summary)
"""
    ),
    md("## Validate Output for Modeling Notebook"),
    code(
        r"""
required_cols = [
    "website_domain", "cutoff_date", "source_url", "homepage_archive_status",
    "snapshot_timestamp", "http_status", "final_url", "failure_reason",
    "archive_method", "evidence_strength", "failure_class", "evidence_scope",
    "needs_homepage_validation",
]

validation_rows = []
for path in sorted(OUTPUT_ROOT.rglob("homepage_domain_archive_proxy.csv")):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    frame = pd.read_csv(path, usecols=[c for c in required_cols if c in header], low_memory=False)
    validation_rows.append({
        "path": str(path),
        "rows": len(frame),
        "missing_required_columns": ", ".join([c for c in required_cols if c not in header]),
        "unique_domains": frame["website_domain"].nunique() if "website_domain" in frame.columns else 0,
        "archived_200_share": frame["homepage_archive_status"].eq("archived_200").mean() if "homepage_archive_status" in frame.columns and len(frame) else pd.NA,
    })

validation = pd.DataFrame(validation_rows)
validation.to_csv(OUTPUT_ROOT / "homepage_crawl_validation.csv", index=False)
display(validation)

print("Main modeling notebook will discover these files automatically from data/live_root/Crawl.")
"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(NOTEBOOK_PATH)
