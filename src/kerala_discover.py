"""Audit which Kerala DHS IDSP daily-report PDF URLs actually exist.

Three stages, all HEAD-only for PDFs — no PDF is ever downloaded:

1. Crawl https://dhs.kerala.gov.in/en/data-on-communicable-diseases/ and any
   same-domain linked pages that look like an archive/listing/pagination
   page (bounded to MAX_CRAWL_PAGES so a live site can't turn this into an
   unbounded crawl), extracting every .pdf href found via GET (HTML only).
   (In practice this found 0 IDSP-Daily-Report PDFs when checked manually —
   they appear to be upload-only, never linked from static HTML — but the
   crawl still runs so that finding is recorded, not assumed.)
2. Coarse probe: HEAD the 1st and 15th of every month from PROBE_START_YEAR
   through END_DATE, 4-digit filename variant only (~250 requests, ~4
   minutes). This exists because an initial ad-hoc sample of single dates
   found hits and misses that did not follow a clean "coverage starts in
   year X" pattern (e.g. 2024-07-01 hit, 2024-01-15 and 2025-07-01 both
   missed) — a full ~1h45m brute force over a guessed range was not
   justified before checking a cheap probe first.
3. Full daily brute force, both filename variants (4-digit and 2-digit
   year), HEAD only, restricted to [earliest year the probe found any 200,
   END_DATE] — not the full PROBE_START_YEAR..END_DATE span. If the probe
   finds no 200 at all, the full sweep is skipped rather than guessing a
   range.

All HTTP calls share one rate limiter at RATE_LIMIT_SECONDS, since all
three stages put load on the same small government site. Timeouts and
connection errors are caught per-request and recorded as a status string
(e.g. "ERROR:Timeout") rather than crashing the run.

Rows from every stage are written to results/metrics/kerala_url_audit.csv
incrementally (flushed after every row), distinguished by the `source`
column (listing_crawl / coarse_probe / brute_force).

A fourth mode, --sweep-start/--sweep-end, runs only the full daily brute
force for an explicit date range (used once the coarse probe has already
bounded the range worth sweeping), in caller-controlled chunks. Every
attempted URL still goes to results/metrics/kerala_url_audit.csv (the full
audit trail, including 404s), but confirmed (HTTP 200) URLs are also
written to data/processed/kerala_daily_urls_confirmed.csv — deliberately
NOT data/raw/, which CLAUDE.md marks non-negotiably read-only; this file
is derived by us hitting a live server, not externally-supplied raw data,
so data/processed/ is where the project's own rules put it.
"""
import csv
import re
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE_LISTING_URL = "https://dhs.kerala.gov.in/en/data-on-communicable-diseases/"
UPLOAD_BASE = "https://dhs.kerala.gov.in/wp-content/uploads"
OUT_PATH = Path("results/metrics/kerala_url_audit.csv")
CONFIRMED_OUT_PATH = Path("data/processed/kerala_daily_urls_confirmed.csv")
CONFIRMED_FIELDNAMES = ["date", "url", "variant", "content_length"]

PROBE_START_YEAR = 2016
END_DATE = date(2026, 6, 30)

REQUEST_TIMEOUT = 15
RATE_LIMIT_SECONDS = 1.0
MAX_CRAWL_PAGES = 40

HEADERS = {
    "User-Agent": "disease-outbreak-ai-research/1.0 (academic project; HEAD-only URL audit)"
}

FIELDNAMES = ["date", "url", "variant", "source", "status_code", "content_length"]

PDF_HREF_RE = re.compile(r'href=["\']([^"\']+\.pdf)["\']', re.IGNORECASE)
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
# Matches DD.MM.YYYY or DD.MM.YY inside a filename.
DATE_IN_FILENAME_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{2}|\d{4})")


def rate_limited_get(session: requests.Session, url: str):
    time.sleep(RATE_LIMIT_SECONDS)
    try:
        return session.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
    except requests.exceptions.RequestException as exc:
        print(f"  GET failed for {url}: {type(exc).__name__}: {exc}")
        return None


def rate_limited_head(session: requests.Session, url: str):
    time.sleep(RATE_LIMIT_SECONDS)
    try:
        resp = session.head(url, timeout=REQUEST_TIMEOUT, headers=HEADERS, allow_redirects=True)
        return resp.status_code, resp.headers.get("Content-Length")
    except requests.exceptions.RequestException as exc:
        return f"ERROR:{type(exc).__name__}", None


def parse_date_from_filename(filename: str):
    m = DATE_IN_FILENAME_RE.search(filename)
    if not m:
        return None
    dd, mm, yy = m.groups()
    try:
        year = 2000 + int(yy) if len(yy) == 2 else int(yy)
        return date(year, int(mm), int(dd))
    except ValueError:
        return None


def crawl_listing_pages(session: requests.Session) -> set:
    to_visit = [BASE_LISTING_URL]
    visited = set()
    pdf_urls = set()

    while to_visit and len(visited) < MAX_CRAWL_PAGES:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"Crawling listing page: {url}")
        resp = rate_limited_get(session, url)
        if resp is None or not resp.ok:
            print(f"  could not fetch (status={'N/A' if resp is None else resp.status_code})")
            continue

        html = resp.text
        for href in PDF_HREF_RE.findall(html):
            pdf_urls.add(urljoin(url, href))

        for href in HREF_RE.findall(html):
            full = urljoin(url, href)
            if "dhs.kerala.gov.in" not in full:
                continue
            if full in visited or full in to_visit:
                continue
            if re.search(r"(archive|/page/\d+|communicable-disease|idsp)", full, re.IGNORECASE):
                to_visit.append(full)

    print(f"Listing crawl visited {len(visited)} page(s), found {len(pdf_urls)} PDF href(s).\n")
    return pdf_urls


def month_range(start_year: int, end_date: date):
    """Yield (year, month) pairs from Jan of start_year through end_date's month."""
    year, month = start_year, 1
    while (year, month) <= (end_date.year, end_date.month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1


def build_probe_dates() -> list:
    """The full, fixed list of (1st, 15th)-of-every-month probe dates.

    Fixed order/content regardless of chunking, so a chunk selected by
    index range always means the same dates run standalone or as part of
    a full run.
    """
    dates = []
    for year, month in month_range(PROBE_START_YEAR, END_DATE):
        for day in (1, 15):
            try:
                d = date(year, month, day)
            except ValueError:
                continue
            if d <= END_DATE:
                dates.append(d)
    return dates


def run_coarse_probe(session: requests.Session, write_row, probe_dates: list = None) -> list:
    """HEAD the given probe dates (default: all of them), 4-digit variant only."""
    if probe_dates is None:
        probe_dates = build_probe_dates()
    probe_rows = []
    print(f"Coarse probe: {len(probe_dates)} date(s), 4-digit variant only "
          f"(~{len(probe_dates) * RATE_LIMIT_SECONDS / 60:.1f} min at the rate limit alone)...")

    for d in probe_dates:
        url = f"{UPLOAD_BASE}/{d.year}/{d.month:02d}/IDSP-Daily-Report-{d.day:02d}.{d.month:02d}.{d.year}.pdf"
        status, content_length = rate_limited_head(session, url)
        row = {
            "date": d.isoformat(),
            "url": url,
            "variant": "4digit",
            "source": "coarse_probe",
            "status_code": status,
            "content_length": content_length,
        }
        write_row(row)
        probe_rows.append(row)
        print(f"  {d.isoformat()} -> {status}")

    return probe_rows


def print_probe_summary(probe_rows: list) -> tuple:
    """Print per-year hit counts and the earliest month with a 200.

    Returns (earliest_hit_year, latest_probed_year) — the range to hand to
    the full sweep — or (None, None) if the probe found nothing.
    """
    by_year = {}
    for r in probe_rows:
        year = date.fromisoformat(r["date"]).year
        by_year.setdefault(year, {"hits": 0, "total": 0})
        by_year[year]["total"] += 1
        if r["status_code"] == 200:
            by_year[year]["hits"] += 1

    print("\nCoarse probe results per year (200s out of 2 probes/month):")
    for year in sorted(by_year):
        stats = by_year[year]
        print(f"  {year}: {stats['hits']}/{stats['total']}")

    hits_sorted = sorted(
        (date.fromisoformat(r["date"]) for r in probe_rows if r["status_code"] == 200)
    )
    if not hits_sorted:
        print("\nEarliest month with any 200: none found in the coarse probe.")
        return None, None

    earliest_hit = hits_sorted[0]
    print(f"\nEarliest month with any 200: {earliest_hit.strftime('%Y-%m')}")
    latest_probed_year = max(by_year)
    return earliest_hit.year, latest_probed_year


def run_sweep_chunk(session: requests.Session, write_audit_row, write_confirmed_row,
                     start_date: date, end_date: date) -> list:
    """Full daily brute force (both filename variants) for [start_date, end_date]."""
    chunk_rows = []
    d = start_date
    while d <= end_date:
        yy = d.year % 100
        variants = [
            ("4digit", f"{UPLOAD_BASE}/{d.year}/{d.month:02d}/"
                       f"IDSP-Daily-Report-{d.day:02d}.{d.month:02d}.{d.year}.pdf"),
            ("2digit", f"{UPLOAD_BASE}/{d.year}/{d.month:02d}/"
                       f"IDSP-Daily-Report-{d.day:02d}.{d.month:02d}.{yy:02d}.pdf"),
        ]
        for variant, url in variants:
            status, content_length = rate_limited_head(session, url)
            row = {
                "date": d.isoformat(),
                "url": url,
                "variant": variant,
                "source": "brute_force",
                "status_code": status,
                "content_length": content_length,
            }
            write_audit_row(row)
            chunk_rows.append(row)
            if status == 200:
                write_confirmed_row({
                    "date": d.isoformat(), "url": url,
                    "variant": variant, "content_length": content_length,
                })
        d += timedelta(days=1)
    return chunk_rows


def print_running_monthly_counts(confirmed_path: Path) -> None:
    """Re-read the confirmed-URL file and print 200-count per year-month."""
    if not confirmed_path.exists():
        print("  (no confirmed URLs yet)")
        return
    per_month = Counter()
    with open(confirmed_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_month[row["date"][:7]] += 1
    print("  Running confirmed (200) count per month so far:")
    for ym in sorted(per_month):
        print(f"    {ym}: {per_month[ym]}")


def print_summary(rows: list) -> None:
    ok_rows = [r for r in rows if r["status_code"] == 200 and r["date"]]
    if not ok_rows:
        print("No 200 responses found for any dated URL.")
        return

    dates_ok = sorted({date.fromisoformat(r["date"]) for r in ok_rows})
    print(f"Earliest 200: {dates_ok[0].isoformat()}")
    print(f"Latest 200: {dates_ok[-1].isoformat()}")

    per_year = Counter(d.year for d in dates_ok)
    print("\nAvailable days per year:")
    for year in sorted(per_year):
        print(f"  {year}: {per_year[year]}")

    longest_gap = timedelta(0)
    gap_start = gap_end = None
    for prev, curr in zip(dates_ok, dates_ok[1:]):
        gap = curr - prev
        if gap > longest_gap:
            longest_gap = gap
            gap_start, gap_end = prev, curr
    if gap_start is not None:
        print(f"\nLongest gap in coverage: {longest_gap.days} day(s), from "
              f"{gap_start.isoformat()} to {gap_end.isoformat()}")
    else:
        print("\nLongest gap in coverage: only one covered date, no gap to measure.")

    variant_by_year = {}
    for r in ok_rows:
        if r["variant"] not in ("4digit", "2digit"):
            continue
        year = date.fromisoformat(r["date"]).year
        variant_by_year.setdefault(year, Counter())[r["variant"]] += 1
    print("\nDominant filename variant per year (brute-force 200s only):")
    for year in sorted(variant_by_year):
        counts = variant_by_year[year]
        dominant, dominant_count = counts.most_common(1)[0]
        print(f"  {year}: {dominant} ({dict(counts)})")


def main(
    probe_only: bool = False,
    skip_crawl: bool = False,
    probe_start_index: int = None,
    probe_end_index: int = None,
    append: bool = False,
):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    rows = []

    file_mode = "a" if append else "w"
    write_header = not (append and OUT_PATH.exists() and OUT_PATH.stat().st_size > 0)

    with open(OUT_PATH, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        def write_row(row):
            rows.append(row)
            writer.writerow(row)
            f.flush()

        # --- 1. Listing/archive crawl ---
        if skip_crawl:
            print("Skipping listing/archive crawl (--skip-crawl).")
        else:
            pdf_urls = crawl_listing_pages(session)
            for i, url in enumerate(sorted(pdf_urls), start=1):
                filename = url.rsplit("/", 1)[-1]
                parsed_date = parse_date_from_filename(filename)
                status, content_length = rate_limited_head(session, url)
                write_row({
                    "date": parsed_date.isoformat() if parsed_date else "",
                    "url": url,
                    "variant": "discovered",
                    "source": "listing_crawl",
                    "status_code": status,
                    "content_length": content_length,
                })
                print(f"  [{i}/{len(pdf_urls)}] HEAD {url} -> {status}")

        # --- 2. Coarse probe: 1st and 15th of every month, 4-digit only ---
        all_probe_dates = build_probe_dates()
        if probe_start_index is not None or probe_end_index is not None:
            start_i = probe_start_index or 0
            end_i = probe_end_index if probe_end_index is not None else len(all_probe_dates)
            probe_dates = all_probe_dates[start_i:end_i]
            print(f"Probe chunk: indices [{start_i}:{end_i}] of {len(all_probe_dates)} total dates.")
        else:
            probe_dates = all_probe_dates

        probe_rows = run_coarse_probe(session, write_row, probe_dates)
        earliest_hit_year, latest_probed_year = print_probe_summary(probe_rows)

        if probe_only:
            print(f"\n--probe-only: stopping here. Saved {len(rows)} row(s) to {OUT_PATH}.")
            return

        # --- 3. Full daily brute force, both variants, restricted to the ---
        #        range the probe actually found coverage in.
        if earliest_hit_year is None:
            print("\nSkipping full daily sweep: the coarse probe found no 200 responses "
                  "anywhere, so there is no evidenced range to sweep. Not guessing one.")
        else:
            sweep_start = date(earliest_hit_year, 1, 1)
            sweep_end = END_DATE
            n_days = (sweep_end - sweep_start).days + 1
            print(f"\nFull daily sweep restricted to {sweep_start.isoformat()}..{sweep_end.isoformat()} "
                  f"({earliest_hit_year}..{latest_probed_year}, the probe-evidenced range): "
                  f"{n_days} dates x 2 filename variants "
                  f"({RATE_LIMIT_SECONDS}s/request, ~{n_days * 2 * RATE_LIMIT_SECONDS / 3600:.1f}h estimated)...")
            d = sweep_start
            day_i = 0
            while d <= sweep_end:
                day_i += 1
                yy = d.year % 100
                variants = [
                    ("4digit", f"{UPLOAD_BASE}/{d.year}/{d.month:02d}/"
                               f"IDSP-Daily-Report-{d.day:02d}.{d.month:02d}.{d.year}.pdf"),
                    ("2digit", f"{UPLOAD_BASE}/{d.year}/{d.month:02d}/"
                               f"IDSP-Daily-Report-{d.day:02d}.{d.month:02d}.{yy:02d}.pdf"),
                ]
                for variant, url in variants:
                    status, content_length = rate_limited_head(session, url)
                    write_row({
                        "date": d.isoformat(),
                        "url": url,
                        "variant": variant,
                        "source": "brute_force",
                        "status_code": status,
                        "content_length": content_length,
                    })
                if day_i % 50 == 0 or d == sweep_end:
                    print(f"  ... {day_i}/{n_days} days checked ({d.isoformat()})")
                d += timedelta(days=1)

    print(f"\nSaved {len(rows)} row(s) to {OUT_PATH}\n")
    print_summary(rows)


def run_sweep_mode(start_date: date, end_date: date) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIRMED_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    audit_write_header = not (OUT_PATH.exists() and OUT_PATH.stat().st_size > 0)
    confirmed_write_header = not (CONFIRMED_OUT_PATH.exists() and CONFIRMED_OUT_PATH.stat().st_size > 0)

    with open(OUT_PATH, "a", newline="", encoding="utf-8") as audit_f, \
         open(CONFIRMED_OUT_PATH, "a", newline="", encoding="utf-8") as confirmed_f:
        audit_writer = csv.DictWriter(audit_f, fieldnames=FIELDNAMES)
        if audit_write_header:
            audit_writer.writeheader()
        confirmed_writer = csv.DictWriter(confirmed_f, fieldnames=CONFIRMED_FIELDNAMES)
        if confirmed_write_header:
            confirmed_writer.writeheader()

        def write_audit_row(row):
            audit_writer.writerow(row)
            audit_f.flush()

        def write_confirmed_row(row):
            confirmed_writer.writerow(row)
            confirmed_f.flush()

        n_days = (end_date - start_date).days + 1
        print(f"Sweep chunk: {start_date.isoformat()}..{end_date.isoformat()} "
              f"({n_days} day(s) x 2 variants = {n_days * 2} requests)")
        chunk_rows = run_sweep_chunk(session, write_audit_row, write_confirmed_row, start_date, end_date)
        n_200 = sum(1 for r in chunk_rows if r["status_code"] == 200)
        print(f"  chunk done: {n_200}/{len(chunk_rows)} requests returned 200")

    print_running_monthly_counts(CONFIRMED_OUT_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-only", action="store_true",
        help="Run the listing crawl + coarse probe, print the probe summary, and stop "
             "(skip the full daily sweep even if the probe finds coverage).",
    )
    parser.add_argument(
        "--skip-crawl", action="store_true",
        help="Skip the listing/archive crawl stage (use once it has already run).",
    )
    parser.add_argument(
        "--probe-start-index", type=int, default=None,
        help="Start index (0-based, inclusive) into the fixed probe-date list, for chunking.",
    )
    parser.add_argument(
        "--probe-end-index", type=int, default=None,
        help="End index (0-based, exclusive) into the fixed probe-date list, for chunking.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to the existing CSV instead of overwriting it (for resuming/chunking).",
    )
    parser.add_argument(
        "--sweep-start", type=str, default=None,
        help="Run ONLY the full daily brute-force sweep for this ISO date (YYYY-MM-DD) "
             "through --sweep-end, appending to results/metrics/kerala_url_audit.csv and "
             "data/processed/kerala_daily_urls_confirmed.csv. Skips crawl/probe entirely.",
    )
    parser.add_argument(
        "--sweep-end", type=str, default=None,
        help="End ISO date (inclusive) for --sweep-start.",
    )
    args = parser.parse_args()

    if args.sweep_start or args.sweep_end:
        if not (args.sweep_start and args.sweep_end):
            parser.error("--sweep-start and --sweep-end must be given together.")
        run_sweep_mode(date.fromisoformat(args.sweep_start), date.fromisoformat(args.sweep_end))
    else:
        main(
            probe_only=args.probe_only,
            skip_crawl=args.skip_crawl,
            probe_start_index=args.probe_start_index,
            probe_end_index=args.probe_end_index,
            append=args.append,
        )
