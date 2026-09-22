"""
Build combined "mother" analysis files from the per-ticker parquet files
in Database/prices/, using DuckDB to query across all files at once.

Outputs:
  - Database/all_prices.parquet   : long format, all tickers combined
  - Database/returns_matrix.parquet : wide format, daily returns
                                       (Date x Ticker), for correlation work
"""

import time
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "Database"
PRICES_GLOB = str(DATABASE_DIR / "prices" / "*.parquet")


def main():
    t0 = time.time()
    con = duckdb.connect()

    print("Reading all per-ticker parquet files with DuckDB...")
    con.execute(f"""
        CREATE TABLE all_prices AS
        SELECT Ticker, Date, Open, High, Low, Close, Adj_Close, Volume,
               Dividends, Stock_Splits
        FROM read_parquet('{PRICES_GLOB}')
        ORDER BY Ticker, Date
    """)
    n_rows = con.execute("SELECT COUNT(*) FROM all_prices").fetchone()[0]
    n_tickers = con.execute("SELECT COUNT(DISTINCT Ticker) FROM all_prices").fetchone()[0]
    print(f"Combined: {n_rows:,} rows across {n_tickers} tickers")

    all_prices_path = DATABASE_DIR / "all_prices.parquet"
    con.execute(f"COPY all_prices TO '{all_prices_path}' (FORMAT PARQUET)")
    print(f"Saved -> {all_prices_path}")

    print("\nComputing daily returns and pivoting to wide matrix...")
    returns_df = con.execute("""
        SELECT Ticker, Date,
               Adj_Close / LAG(Adj_Close) OVER (PARTITION BY Ticker ORDER BY Date) - 1 AS ret
        FROM all_prices
    """).fetchdf()

    wide = returns_df.pivot(index="Date", columns="Ticker", values="ret")
    wide = wide.sort_index()

    returns_path = DATABASE_DIR / "returns_matrix.parquet"
    wide.reset_index().to_parquet(returns_path, index=False)
    print(f"Saved -> {returns_path}  shape={wide.shape}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} sec")


if __name__ == "__main__":
    main()
