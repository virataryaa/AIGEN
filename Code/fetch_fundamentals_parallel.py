"""
Threaded version of fetch_fundamentals.py -- same logic, but tickers are
processed concurrently (ThreadPoolExecutor) since this is I/O-bound
(network requests to NSE), not CPU-bound. No rate-limiting observed on
the sequential run, so this trades a bit of politeness for speed -- if
NSE starts returning errors/429s under concurrent load, drop MAX_WORKERS
or add a delay (same fix as the yfinance rate-limit mistake in README).
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from fetch_fundamentals import (
    FUNDAMENTALS_DIR,
    fetch_ticker_fundamentals,
    load_top50_tickers,
)

MAX_WORKERS = 8


def process_ticker(ticker: str) -> dict:
    session = requests.Session()  # one session per thread, not shared
    result, n_filings = fetch_ticker_fundamentals(ticker, session)
    if result is None or result.empty:
        return {"ticker": ticker, "ok": False, "n_filings": n_filings, "n_facts": 0}

    result.to_parquet(FUNDAMENTALS_DIR / f"{ticker}.parquet", index=False)
    n_filings_used = result[["PeriodType", "FilingToDate"]].drop_duplicates().shape[0]
    return {"ticker": ticker, "ok": True, "n_filings": n_filings_used, "n_facts": len(result)}


def main():
    tickers = load_top50_tickers()
    print(f"Fetching fundamentals for {len(tickers)} tickers with {MAX_WORKERS} parallel workers...\n")

    t0 = time.time()
    stats = {"ok": 0, "empty": 0, "total_facts": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_ticker, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r["ok"]:
                stats["ok"] += 1
                stats["total_facts"] += r["n_facts"]
                print(f"[{i}/{len(tickers)}] {r['ticker']}: {r['n_filings']} filings, {r['n_facts']:,} facts")
            else:
                stats["empty"] += 1
                print(f"[{i}/{len(tickers)}] {r['ticker']}: NO DATA")

    elapsed = time.time() - t0
    print(f"\n=== SUMMARY ===")
    print(f"Tickers OK: {stats['ok']}  Empty/Failed: {stats['empty']}")
    print(f"Total facts stored: {stats['total_facts']:,}")
    print(f"Total time: {elapsed:.1f} sec ({elapsed/60:.2f} min)")
    print(f"Avg per ticker (wall-clock, parallel): {elapsed/len(tickers):.1f} sec")


if __name__ == "__main__":
    main()
