"""Fetch fundamentals for the final remaining tickers to complete the full
NIFTY 500 fundamentals universe."""

from fetch_fundamentals import main

if __name__ == "__main__":
    main(csv_name="remaining158.csv", label="Final remaining (NIFTY 500 completion)")
