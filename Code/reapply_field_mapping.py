"""
Re-apply the current field_mapping.csv (CanonicalName/Statement) to all
already-fetched fundamentals parquet files, LOCALLY -- no network calls.

Needed because enrich_and_optimize() bakes CanonicalName into each
ticker's file at fetch time; a later field_mapping.csv fix (e.g. the bank
Capital/ReservesAndSurplus/ProfitLossForThePeriod additions) does not
retroactively apply to tickers fetched before the fix. Re-fetching them
from NSE would work but wastes ~15-20 min of network calls for a change
that's actually a pure local join on FieldName -- same lesson as mistake
#10 (don't delete+refetch when a local reprocess will do).
"""

import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FUNDAMENTALS_DIR = BASE_DIR / "Database" / "fundamentals"
NON_TICKER_FILES = {"field_mapping.csv", "ratios_wide.parquet"}

CATEGORICAL_COLS = ["Ticker", "PeriodType", "Consolidated", "Source",
                     "FieldName", "ContextRef", "DurationClass",
                     "Industry", "CanonicalName", "Statement"]


def main():
    field_map = pd.read_csv(FUNDAMENTALS_DIR / "field_mapping.csv")
    files = sorted(f for f in FUNDAMENTALS_DIR.glob("*.parquet") if f.name not in NON_TICKER_FILES)
    print(f"Re-applying field mapping to {len(files)} tickers (local only, no network)...\n")

    t0 = time.time()
    updated = 0
    for f in files:
        df = pd.read_parquet(f)
        df["FieldName"] = df["FieldName"].astype(str)

        merged = df.drop(columns=["CanonicalName", "Statement"]).merge(
            field_map[["raw_tag", "canonical_name", "statement"]],
            left_on="FieldName", right_on="raw_tag", how="left",
        )
        df["CanonicalName"] = merged["canonical_name"].values
        df["Statement"] = merged["statement"].values

        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")

        df.to_parquet(f, index=False, compression="brotli")
        updated += 1

    elapsed = time.time() - t0
    print(f"Done: {updated} files updated in {elapsed:.1f}s ({elapsed/max(updated,1):.2f}s/file)")


if __name__ == "__main__":
    main()
