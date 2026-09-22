"""
Add the "next 500" equities to the universe and ingest their OHLCV history.

Composition of the next 500:
  - 254 tickers: NSE Nifty Microcap 250 index (NSE-curated, ranked, has Industry tag)
  - 246 tickers: top-by-market-cap from the remaining EQ universe not covered
    by any NSE broad-market index (Industry fetched via yfinance)

Appends to Database/universe.parquet and writes new OHLCV parquet files
into Database/prices/, from 2000-01-01.
"""

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "Database"
PRICES_DIR = DATABASE_DIR / "prices"
SCRATCH = Path(
    r"C:/Users/VIRAT~1.ARY/AppData/Local/Temp/claude/C--Users-virat-arya/"
    r"dde5ced3-1028-4c8f-91f7-137de3f81533/scratchpad"
)

START_DATE = "2000-01-01"


def load_microcap250() -> pd.DataFrame:
    df = pd.read_csv(SCRATCH / "ind_niftymicrocap250_list.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["Symbol"] = df["Symbol"].str.strip()
    return df[["Company Name", "Industry", "Symbol", "Series", "ISIN Code"]]


def load_top246_with_industry() -> pd.DataFrame:
    symbols = (SCRATCH / "top246_by_marketcap.txt").read_text().splitlines()
    symbols = [s.strip() for s in symbols if s.strip()]

    rows = []
    for i, sym in enumerate(symbols, 1):
        try:
            info = yf.Ticker(sym + ".NS").get_info()
            rows.append({
                "Company Name": info.get("longName") or info.get("shortName") or sym,
                "Industry": info.get("industryDisp") or info.get("industry") or "Unknown",
                "Symbol": sym,
                "Series": "EQ",
                "ISIN Code": "",
            })
        except Exception:
            rows.append({
                "Company Name": sym,
                "Industry": "Unknown",
                "Symbol": sym,
                "Series": "EQ",
                "ISIN Code": "",
            })
        if i % 50 == 0:
            print(f"  industry lookup {i}/{len(symbols)}...")
    return pd.DataFrame(rows)


def update_universe(new_rows: pd.DataFrame):
    new_rows = new_rows.copy()
    new_rows["Ticker"] = new_rows["Symbol"] + ".NS"

    universe_path = DATABASE_DIR / "universe.parquet"
    existing = pd.read_parquet(universe_path)

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset="Ticker", keep="first")
    combined.to_parquet(universe_path, index=False)
    print(f"Universe updated: {len(existing)} -> {len(combined)} tickers")
    return combined


def ingest_prices(tickers: list[str]) -> dict:
    stats = {"ok": 0, "empty": 0, "error": 0}
    for i, ticker in enumerate(tickers, 1):
        try:
            hist = yf.Ticker(ticker).history(
                start=START_DATE, interval="1d", auto_adjust=False
            )
            if hist.empty:
                stats["empty"] += 1
                print(f"[{i}/{len(tickers)}] {ticker}: EMPTY")
                continue

            hist = hist.reset_index()
            hist.columns = [c.replace(" ", "_") for c in hist.columns]
            hist.insert(0, "Ticker", ticker)
            hist.to_parquet(PRICES_DIR / f"{ticker}.parquet", index=False)

            stats["ok"] += 1
            print(f"[{i}/{len(tickers)}] {ticker}: {len(hist)} rows "
                  f"({hist['Date'].min().date()} -> {hist['Date'].max().date()})")
        except Exception as exc:
            stats["error"] += 1
            print(f"[{i}/{len(tickers)}] {ticker}: ERROR {exc}")
    return stats


def main():
    t_start = time.time()

    print("Loading Microcap250 (254 tickers)...")
    microcap = load_microcap250()

    print("Fetching industry for top-246-by-marketcap tickers...")
    top246 = load_top246_with_industry()

    new_rows = pd.concat([microcap, top246], ignore_index=True)
    print(f"Total new tickers to add: {len(new_rows)}")

    combined_universe = update_universe(new_rows)

    tickers = (new_rows["Symbol"] + ".NS").tolist()
    print(f"\nIngesting OHLCV for {len(tickers)} new tickers from {START_DATE}...\n")
    stats = ingest_prices(tickers)

    elapsed = time.time() - t_start
    print("\n=== SUMMARY ===")
    print(f"New tickers added to universe: {len(new_rows)}")
    print(f"Total universe size now: {len(combined_universe)}")
    print(f"OHLCV OK: {stats['ok']}  EMPTY: {stats['empty']}  ERROR: {stats['error']}")
    print(f"Total time: {elapsed:.1f} sec ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()
