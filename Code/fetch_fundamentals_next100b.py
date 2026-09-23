"""Fetch fundamentals for the next fresh 100 tickers (NIFTY 500 remainder,
after the first 258 companies) -- same pipeline, different universe."""

from fetch_fundamentals import main

if __name__ == "__main__":
    main(csv_name="next100b.csv", label="Next 100 batch 2 (NIFTY 500 remainder)")
