"""
Ingest daily OHLCV history (from 2000-01-01) for the NIFTY 500 universe
and store one parquet file per ticker in Database/prices/.

Also saves the NIFTY 500 constituent list (with industry tags) as
Database/universe.parquet.
"""

import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "Database"
PRICES_DIR = DATABASE_DIR / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)

NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
START_DATE = "2000-01-01"


def fetch_universe() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(NIFTY500_URL, headers=headers, timeout=15)
    resp.raise_for_status()

    raw_path = DATABASE_DIR / "_nifty500_raw.csv"
    raw_path.write_bytes(resp.content)

    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["Symbol"] = df["Symbol"].str.strip()
    df["Ticker"] = df["Symbol"] + ".NS"

    df.to_parquet(DATABASE_DIR / "universe.parquet", index=False)
    raw_path.unlink()
    return df


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

    universe = fetch_universe()
    tickers = universe["Ticker"].tolist()
    print(f"Universe: {len(tickers)} tickers\n")

    stats = ingest_prices(tickers)

    elapsed = time.time() - t_start
    print("\n=== SUMMARY ===")
    print(f"OK: {stats['ok']}  EMPTY: {stats['empty']}  ERROR: {stats['error']}")
    print(f"Total time: {elapsed:.1f} sec ({elapsed/60:.2f} min)")
    print(f"Avg per ticker: {elapsed/len(tickers):.2f} sec")


if __name__ == "__main__":
    main()
