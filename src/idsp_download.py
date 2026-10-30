"""Discover and download IDSP weekly outbreak report PDFs, 2021 onward.

idsp.mohfw.gov.in (the originally-targeted index + WriteReadData host) is
unreachable from this environment on every path tried (DNS resolves, but
every TCP connect on :80/:443 times out or is refused, from two independent
networks) — confirmed before writing this script, not assumed. The same
reports are mirrored on ncdc.mohfw.gov.in, which IS reachable: its
"Weekly Outbreaks" page (includes/WeeklyOutbreaks.php) links
uploads/weekly_outbreaks/<year>/week<N>[_<timestamp>].pdf for every week
back to 2009. Spot-checked against the exact example in this project's task
instructions: week23.pdf (2021) prints "(7th June 2021 to 13th June 2021)"
on its own page 1 — same underlying document set, different (working) host.

The listing page can be stale (2023 week 15 has a real, correctly-dated PDF
at its predictable URL that simply isn't linked from the page), so discovery
tries the direct URL for every expected week in addition to the page's own
links, before concluding a week is genuinely missing.

Resumable: a PDF already present on disk (non-zero size) is not
re-downloaded; its inventory row is rebuilt by re-reading the local file,
not by re-fetching it.
"""
import csv
import hashlib
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup

LISTING_URL = "https://ncdc.mohfw.gov.in/includes/WeeklyOutbreaks.php"
BASE_URL = "https://ncdc.mohfw.gov.in"
PDF_HREF_RE = re.compile(r"/uploads/weekly_outbreaks/(\d{4})/week(\d+)(?:_\d+)?\.pdf", re.I)

OUT_DIR = Path("data/external/idsp_pdfs")
INVENTORY_PATH = Path("results/metrics/idsp_download_inventory.csv")

YEARS = range(2021, 2027)
DELAY_SECONDS = 1.5
RETRIES = 3
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}

DATE_RANGE_RE = re.compile(r"\(\s*(\d{1,2}(?:st|nd|rd|th)\s+\w+\s+\d{4})\s+to\s+(\d{1,2}(?:st|nd|rd|th)\s+\w+\s+\d{4})\s*\)")
SUBMITTED_RE = re.compile(r"Submitted outbreak report\s*\(\s*including NIL report\s*\)\s*(\d+)", re.I)
NIL_RE = re.compile(r"Submitted\s*.NIL.\s*outbreak report\s*(\d+)", re.I)
ORDINAL_RE = re.compile(r"(\d+)(st|nd|rd|th)")


def iso_weeks_in_year(year: int) -> int:
    return date(year, 12, 28).isocalendar()[1]


def get_with_retries(url: str, retries: int = RETRIES, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return requests.get(url, headers=HEADERS, timeout=30, **kwargs)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(DELAY_SECONDS)
    raise last_exc


def discover_urls() -> dict:
    """Returns {(year, week): url}, preferring the listing page's own link
    (which may carry a cache-busting timestamp suffix) and falling back to
    the plain week<N>.pdf URL for any expected week the page didn't link.
    """
    print(f"Fetching listing page: {LISTING_URL}")
    resp = get_with_retries(LISTING_URL)
    soup = BeautifulSoup(resp.text, "html.parser")

    linked = {}
    for a in soup.find_all("a", href=True):
        m = PDF_HREF_RE.search(a["href"])
        if m:
            year, week = int(m.group(1)), int(m.group(2))
            linked[(year, week)] = BASE_URL + a["href"]

    urls = {}
    for year in YEARS:
        max_week = max((w for (y, w) in linked if y == year), default=0)
        expected_weeks = range(1, iso_weeks_in_year(year) + 1) if max_week >= 50 else range(1, max_week + 1)
        for week in expected_weeks:
            key = (year, week)
            if key in linked:
                urls[key] = linked[key]
            else:
                # Listing page can be stale (e.g. 2023 week 15) — probe the
                # plain URL directly before accepting the week as missing.
                guess = f"{BASE_URL}/uploads/weekly_outbreaks/{year}/week{week}.pdf"
                urls[key] = guess  # verified for real/missing at download time
    return urls


def parse_page1(pdf_bytes: bytes) -> dict:
    with pdfplumber.open(__import__("io").BytesIO(pdf_bytes)) as pdf:
        text = pdf.pages[0].extract_text() or ""

    out = {
        "report_start_date": None, "report_end_date": None,
        "iso_year": None, "iso_week": None,
        "n_states_submitted": None, "n_states_nil": None,
    }
    m = DATE_RANGE_RE.search(text)
    if m:
        start_raw, end_raw = m.group(1), m.group(2)
        try:
            start = datetime.strptime(ORDINAL_RE.sub(r"\1", start_raw), "%d %B %Y").date()
            end = datetime.strptime(ORDINAL_RE.sub(r"\1", end_raw), "%d %B %Y").date()
            out["report_start_date"] = start.isoformat()
            out["report_end_date"] = end.isoformat()
            iso_year, iso_week, _ = start.isocalendar()
            out["iso_year"], out["iso_week"] = iso_year, iso_week
        except ValueError:
            pass  # left as None/TODO — never guessed

    sm = SUBMITTED_RE.search(text)
    if sm:
        out["n_states_submitted"] = int(sm.group(1))
    nm = NIL_RE.search(text)
    if nm:
        out["n_states_nil"] = int(nm.group(1))
    return out


def download_one(year: int, week: int, url: str) -> dict:
    year_dir = OUT_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    local_path = year_dir / filename

    if local_path.exists() and local_path.stat().st_size > 0:
        pdf_bytes = local_path.read_bytes()
        http_status = 200
    else:
        try:
            resp = get_with_retries(url)
        except requests.exceptions.RequestException as exc:
            return {"url": url, "local_path": "", "bytes": 0, "sha256": "", "http_status": f"ERROR:{exc}",
                    "report_start_date": None, "report_end_date": None, "iso_year": year, "iso_week": week,
                    "n_states_submitted": None, "n_states_nil": None}
        time.sleep(DELAY_SECONDS)
        if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
            return {"url": url, "local_path": "", "bytes": 0, "sha256": "", "http_status": resp.status_code,
                    "report_start_date": None, "report_end_date": None, "iso_year": year, "iso_week": week,
                    "n_states_submitted": None, "n_states_nil": None}
        pdf_bytes = resp.content
        local_path.write_bytes(pdf_bytes)
        http_status = resp.status_code

    row = {
        "url": url, "local_path": str(local_path), "bytes": len(pdf_bytes),
        "sha256": hashlib.sha256(pdf_bytes).hexdigest(), "http_status": http_status,
    }
    try:
        row.update(parse_page1(pdf_bytes))
    except Exception as exc:
        row.update({"report_start_date": None, "report_end_date": None, "iso_year": year, "iso_week": week,
                     "n_states_submitted": None, "n_states_nil": None})
        print(f"  WARNING: could not parse page 1 of {local_path}: {exc}")
    # Fall back to the expected (year, week) for rows whose date range didn't parse,
    # so the inventory still has a best-effort key — never silently drop the row.
    if row.get("iso_year") is None:
        row["iso_year"], row["iso_week"] = year, week
    return row


def main():
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    urls = discover_urls()
    print(f"Discovered {len(urls)} expected (year, week) slots across {list(YEARS)}.\n")

    fieldnames = ["url", "local_path", "bytes", "sha256", "http_status",
                  "report_start_date", "report_end_date", "iso_year", "iso_week",
                  "n_states_submitted", "n_states_nil"]
    rows = []
    for year in YEARS:
        print(f"=== {year} ===")
        year_rows = []
        for week in range(1, iso_weeks_in_year(year) + 1):
            key = (year, week)
            if key not in urls:
                continue
            row = download_one(year, week, urls[key])
            year_rows.append(row)
            status = "ok" if row["http_status"] == 200 else f"MISSING ({row['http_status']})"
            print(f"  week {week}: {status}")
        rows.extend(year_rows)

    with open(INVENTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved inventory to {INVENTORY_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
