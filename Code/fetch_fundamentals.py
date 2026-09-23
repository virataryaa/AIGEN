"""
Fetch full available XBRL fundamentals history (quarterly + annual,
P&L + balance sheet + cash flow) for the top 50 (NIFTY 50) companies from
NSE's public, unauthenticated filing APIs.

Two endpoints are needed to cover the full history:
  - Legacy: /api/corporates-financial-results (pre-2025, frozen Dec-2024)
  - Integrated: /api/integrated-filing-results (2025 onward, SEBI's new regime)

Stores one long-format parquet per ticker in a SEPARATE subfolder
(Database/fundamentals/), not mixed with the OHLCV prices folder.

Dates for each XBRL fact are resolved from the filing's own <xbrli:context>
definitions (not guessed from tag-name prefixes like One*/Four*, which is
a legacy-taxonomy-specific convention that doesn't generalize cleanly
across all filing eras) -- each fact's duration is classified by actual
day-count (quarter vs 9-month YTD vs annual vs balance-sheet instant).
"""

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "Database"
FUNDAMENTALS_DIR = DATABASE_DIR / "fundamentals"
FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)

# Single source of truth for every derived/summary file living alongside the
# per-ticker parquets in FUNDAMENTALS_DIR -- any script that globs
# FUNDAMENTALS_DIR/*.parquet must exclude these (see README mistake #16:
# this exact class of bug -- a new derived file added here without also
# updating every glob's exclusion set -- has happened three times).
NON_TICKER_FILES = {"ratios_wide.parquet", "valuation_snapshot.parquet"}

SCRATCH = Path(
    r"C:/Users/VIRAT~1.ARY/AppData/Local/Temp/claude/C--Users-virat-arya/"
    r"dde5ced3-1028-4c8f-91f7-137de3f81533/scratchpad"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
}
REQUEST_DELAY = 0.2


def load_top50_tickers(csv_name: str = "nifty50.csv") -> list[str]:
    df = pd.read_csv(SCRATCH / csv_name, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return sorted(df["Symbol"].str.strip().tolist())


def fetch_json(url: str, session: requests.Session):
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def get_filing_list(symbol: str, session: requests.Session) -> pd.DataFrame:
    rows = []

    for period in ("Quarterly", "Annual"):
        url = (f"https://www.nseindia.com/api/corporates-financial-results"
               f"?index=equities&symbol={symbol}&period={period}")
        data = fetch_json(url, session)
        if data:
            for r in data:
                rows.append({
                    "period_type": period,
                    "consolidated": r.get("consolidated"),
                    "from_date": r.get("fromDate"),
                    "to_date": r.get("toDate"),
                    "xbrl": r.get("xbrl"),
                    "source": "legacy",
                    "filing_date": r.get("broadCastDate") or r.get("filingDate"),
                })
        time.sleep(REQUEST_DELAY)

        url = (f"https://www.nseindia.com/api/integrated-filing-results"
               f"?index=equities&symbol={symbol}&period={period}")
        data = fetch_json(url, session)
        if data:
            for r in data.get("data", []):
                if "INDAS" not in (r.get("xbrl") or ""):
                    continue
                rows.append({
                    "period_type": period,
                    "consolidated": r.get("consolidated"),
                    "from_date": None,
                    "to_date": r.get("qe_Date"),
                    "xbrl": r.get("xbrl"),
                    "source": "integrated",
                    "filing_date": r.get("broadcast_Date") or r.get("creation_Date"),
                })
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["xbrl"].fillna("").str.endswith(".xml")]

    # Prefer Consolidated over Non-Consolidated, then the most recently
    # filed/broadcast version for the same (period_type, to_date) -- guards
    # against silently keeping a superseded/revised filing.
    df["is_consolidated"] = (df["consolidated"] == "Consolidated").astype(int)
    df["filing_date_parsed"] = pd.to_datetime(df["filing_date"], errors="coerce", dayfirst=True)
    df = df.sort_values(["is_consolidated", "filing_date_parsed"], ascending=[False, False])
    df = df.drop_duplicates(subset=["period_type", "to_date"], keep="first")
    return df.drop(columns=["is_consolidated", "filing_date_parsed"])


def classify_duration(start, end, ctx_ref: str = "") -> str:
    if start is not None and end is not None:
        days = (end - start).days
        if days <= 0:
            return "Instant"  # context resolved fine -- a true point-in-time balance sheet date
        if days <= 100:
            return "Quarter"
        if days <= 290:
            return "YTD_9M"
        return "Annual"

    # Context not declared/resolved (a real, documented NSE XBRL quirk --
    # some filings reference e.g. "OneD" without ever declaring that exact
    # context, only its suffixed siblings). Fall back to the legacy
    # taxonomy's ID-prefix convention rather than silently mislabeling the
    # fact as "Instant" and dropping it from flow-statement analysis:
    # One* = single quarter (duration), Four* = cumulative YTD (duration).
    if ctx_ref.startswith("One"):
        return "Quarter"
    if ctx_ref.startswith("Four"):
        return "YTD_9M"
    return "Instant"


def parse_xbrl(xml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)

    contexts = {}
    for ctx in root.iter():
        if not ctx.tag.endswith("}context") and ctx.tag != "context":
            continue
        ctx_id = ctx.attrib.get("id")
        start = end = None
        for period in ctx.iter():
            tag = period.tag.split("}")[-1]
            if tag == "startDate":
                start = pd.to_datetime(period.text, errors="coerce")
            elif tag == "endDate":
                end = pd.to_datetime(period.text, errors="coerce")
            elif tag == "instant":
                start = end = pd.to_datetime(period.text, errors="coerce")
        contexts[ctx_id] = (start, end)

    rows = []
    for el in root.iter():
        ctx_ref = el.attrib.get("contextRef")
        if not ctx_ref or el.text is None:
            continue
        tag = el.tag.split("}")[-1]
        start, end = contexts.get(ctx_ref, (None, None))
        rows.append({
            "FieldName": tag,
            "Value": el.text,
            "ContextRef": ctx_ref,
            "ContextStart": start,
            "ContextEnd": end,
            "DurationClass": classify_duration(start, end, ctx_ref),
        })
    return pd.DataFrame(rows)


def fetch_ticker_fundamentals(symbol: str, session: requests.Session) -> tuple[pd.DataFrame | None, int]:
    filings = get_filing_list(symbol, session)
    if filings.empty:
        return None, 0

    all_facts = []
    for _, filing in filings.iterrows():
        try:
            resp = session.get(filing["xbrl"], headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                continue
            facts = parse_xbrl(resp.content)
            if facts.empty:
                continue
            facts.insert(0, "Ticker", symbol)
            facts.insert(1, "PeriodType", filing["period_type"])
            facts.insert(2, "Consolidated", filing["consolidated"])
            facts.insert(3, "FilingToDate", filing["to_date"])
            facts.insert(4, "Source", filing["source"])
            facts.insert(5, "SourceFile", filing["xbrl"])
            all_facts.append(facts)
        except Exception:
            pass
        time.sleep(REQUEST_DELAY)

    if not all_facts:
        return None, len(filings)

    return pd.concat(all_facts, ignore_index=True), len(filings)


CATEGORICAL_COLS = ["Ticker", "PeriodType", "Consolidated", "Source",
                     "FieldName", "ContextRef", "DurationClass",
                     "Industry", "CanonicalName", "Statement"]


def load_industry_map() -> dict:
    universe = pd.read_parquet(DATABASE_DIR / "universe.parquet")
    return dict(zip(universe["Symbol"].str.strip(), universe["Industry"]))


def load_field_map() -> pd.DataFrame:
    return pd.read_csv(FUNDAMENTALS_DIR / "field_mapping.csv")


def enrich_and_optimize(df: pd.DataFrame, industry_map: dict, field_map: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Industry"] = df["Ticker"].map(industry_map).fillna("Unknown")

    merged = df.merge(
        field_map[["raw_tag", "canonical_name", "statement"]],
        left_on="FieldName", right_on="raw_tag", how="left",
    )
    df["CanonicalName"] = merged["canonical_name"]
    df["Statement"] = merged["statement"]

    df["ValueNumeric"] = pd.to_numeric(df["Value"], errors="coerce")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def main(csv_name: str = "nifty50.csv", label: str = "NIFTY 50"):
    tickers = load_top50_tickers(csv_name)
    print(f"Fetching full fundamentals history for {len(tickers)} ({label}) tickers...\n")

    industry_map = load_industry_map()
    field_map = load_field_map()

    session = requests.Session()
    t0 = time.time()

    stats = {"ok": 0, "empty": 0, "total_filings": 0, "total_facts": 0}
    for i, ticker in enumerate(tickers, 1):
        result, n_filings = fetch_ticker_fundamentals(ticker, session)
        stats["total_filings"] += n_filings

        if result is None or result.empty:
            stats["empty"] += 1
            print(f"[{i}/{len(tickers)}] {ticker}: NO DATA ({n_filings} filings found, 0 parsed)")
            continue

        result = enrich_and_optimize(result, industry_map, field_map)
        result.to_parquet(FUNDAMENTALS_DIR / f"{ticker}.parquet", index=False, compression="brotli")
        stats["ok"] += 1
        stats["total_facts"] += len(result)
        n_filings_used = result[["PeriodType", "FilingToDate"]].drop_duplicates().shape[0]
        print(f"[{i}/{len(tickers)}] {ticker}: {n_filings_used} filings, {len(result):,} facts "
              f"({result['FilingToDate'].min()} -> {result['FilingToDate'].max()})")

    elapsed = time.time() - t0
    print(f"\n=== SUMMARY ===")
    print(f"Tickers OK: {stats['ok']}  Empty/Failed: {stats['empty']}")
    print(f"Total filings found: {stats['total_filings']}")
    print(f"Total facts stored: {stats['total_facts']:,}")
    print(f"Total time: {elapsed:.1f} sec ({elapsed/60:.2f} min)")
    print(f"Avg per ticker: {elapsed/len(tickers):.1f} sec")
    print(f"Saved to: {FUNDAMENTALS_DIR}")


if __name__ == "__main__":
    main()
