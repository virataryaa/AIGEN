"""
Retry OHLCV ingest for the 500 newly-added tickers (universe already updated
by add_next500.py) using a small delay + exponential backoff to avoid
Yahoo's rate limiting that hit the previous run.
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
DELAY_SEC = 1.0
MAX_RETRIES = 4


def load_target_tickers() -> list[str]:
    microcap = pd.read_csv(SCRATCH / "ind_niftymicrocap250_list.csv", encoding="utf-8-sig")
    microcap.columns = [c.strip() for c in microcap.columns]
    microcap_syms = microcap["Symbol"].str.strip().tolist()

    top246 = (SCRATCH / "top246_by_marketcap.txt").read_text().splitlines()
    top246 = [s.strip() for s in top246 if s.strip()]

    symbols = microcap_syms + top246
    return [s + ".NS" for s in symbols]


def fetch_with_retry(ticker: str):
    delay = DELAY_SEC
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            hist = yf.Ticker(ticker).history(
                start=START_DATE, interval="1d", auto_adjust=False
            )
            return hist
        except Exception as exc:
            if "Too Many Requests" in str(exc) or "Rate limited" in str(exc):
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return pd.DataFrame()


def main():
    tickers = load_target_tickers()
    print(f"Retrying OHLCV ingest for {len(tickers)} tickers...\n")

    t0 = time.time()
    stats = {"ok": 0, "empty": 0, "error": 0}
    for i, ticker in enumerate(tickers, 1):
        out_path = PRICES_DIR / f"{ticker}.parquet"
        if out_path.exists():
            stats["ok"] += 1
            continue

        hist = fetch_with_retry(ticker)
        if hist.empty:
            stats["empty"] += 1
            print(f"[{i}/{len(tickers)}] {ticker}: EMPTY/FAILED")
        else:
            hist = hist.reset_index()
            hist.columns = [c.replace(" ", "_") for c in hist.columns]
            hist.insert(0, "Ticker", ticker)
            hist.to_parquet(out_path, index=False)
            stats["ok"] += 1
            print(f"[{i}/{len(tickers)}] {ticker}: {len(hist)} rows "
                  f"({hist['Date'].min().date()} -> {hist['Date'].max().date()})")

        time.sleep(DELAY_SEC)

    elapsed = time.time() - t0
    print("\n=== SUMMARY ===")
    print(f"OK: {stats['ok']}  EMPTY/FAILED: {stats['empty']}")
    print(f"Total time: {elapsed:.1f} sec ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()
