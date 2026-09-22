"""
Run the composite signal engine (compute_signals.py) across the full
ticker universe and save the LATEST signal snapshot per ticker to
Database/signals.parquet.

This is the only heavy-compute step — runs locally. The Streamlit
dashboard just reads Database/signals.parquet (small: one row per ticker).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from compute_signals import compute_composite

BASE_DIR = Path(__file__).resolve().parent.parent
PRICES_DIR = BASE_DIR / "Database" / "prices"
OUTPUT_FILE = BASE_DIR / "Database" / "signals.parquet"

MIN_ROWS_REQUIRED = 60  # need at least ~60 trading days for momentum/RSI to be meaningful


def main():
    tickers = sorted(p.stem for p in PRICES_DIR.glob("*.parquet"))
    print(f"Computing composite signals for {len(tickers)} tickers...\n")

    t0 = time.time()
    rows = []
    skipped = 0
    for i, ticker in enumerate(tickers, 1):
        df = pd.read_parquet(PRICES_DIR / f"{ticker}.parquet")
        df = df.sort_values("Date").reset_index(drop=True)

        if len(df) < MIN_ROWS_REQUIRED:
            skipped += 1
            continue

        sig_df = compute_composite(df)
        latest = sig_df.iloc[-1]
        rows.append({
            "Ticker": ticker,
            "Date": latest["Date"],
            "Close": latest["Close"],
            "Momentum_20": latest["Momentum_20"],
            "Momentum_100": latest["Momentum_100"],
            "MA_cross_50_200": latest["MA_cross_50_200"],
            "Donchian_20": latest["Donchian_20"],
            "RSI_14": latest["RSI_14"] if not np.isnan(latest["RSI_14"]) else None,
            "Composite": latest["Composite"],
        })

        if i % 200 == 0:
            print(f"  {i}/{len(tickers)} done...")

    elapsed = time.time() - t0

    result = pd.DataFrame(rows).sort_values("Composite", ascending=False)
    result.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n=== SUMMARY ===")
    print(f"Signals computed: {len(result)}  Skipped (insufficient history): {skipped}")
    print(f"Saved -> {OUTPUT_FILE}")
    print(f"Total time: {elapsed:.1f} sec ({elapsed/60:.2f} min)")
    print(f"\nTop 10 by composite score:")
    print(result.head(10)[["Ticker", "Close", "Composite"]].to_string(index=False))


if __name__ == "__main__":
    main()
