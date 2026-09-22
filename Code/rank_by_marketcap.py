"""
Rank the NSE EQ tickers that fall outside NSE's Total Market Index (top 755)
by market cap, using yfinance fast_info, threaded for speed.

Writes a CSV of symbol,market_cap sorted descending.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yfinance as yf

SCRATCH = Path(
    r"C:/Users/VIRAT~1.ARY/AppData/Local/Temp/claude/C--Users-virat-arya/"
    r"dde5ced3-1028-4c8f-91f7-137de3f81533/scratchpad"
)
INPUT_FILE = SCRATCH / "remaining_eq.txt"
OUTPUT_FILE = SCRATCH / "remaining_eq_marketcap.csv"

MAX_WORKERS = 16


def get_market_cap(symbol: str):
    ticker = symbol + ".NS"
    try:
        fi = yf.Ticker(ticker).fast_info
        mcap = fi.get("marketCap")
        return symbol, mcap
    except Exception:
        return symbol, None


def main():
    symbols = [l.strip() for l in INPUT_FILE.read_text().splitlines() if l.strip()]
    print(f"Fetching market cap for {len(symbols)} tickers...")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_market_cap, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            symbol, mcap = fut.result()
            results.append((symbol, mcap))
            if i % 100 == 0:
                print(f"  {i}/{len(symbols)} done...")

    elapsed = time.time() - t0
    print(f"Fetched in {elapsed:.1f} sec")

    valid = [(s, m) for s, m in results if m is not None]
    invalid = [s for s, m in results if m is None]
    print(f"Valid market caps: {len(valid)}, missing: {len(invalid)}")

    valid.sort(key=lambda x: x[1], reverse=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write("Symbol,MarketCap\n")
        for s, m in valid:
            f.write(f"{s},{m}\n")

    print(f"Saved ranked list to {OUTPUT_FILE}")
    print("\nTop 10:")
    for s, m in valid[:10]:
        print(f"  {s}: {m:,.0f}")


if __name__ == "__main__":
    main()
