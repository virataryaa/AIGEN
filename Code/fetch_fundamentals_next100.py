"""Fetch fundamentals for the next 100 tickers (NIFTY 500 remainder, after
the first 163 companies) -- same pipeline, different universe."""

from fetch_fundamentals import main

if __name__ == "__main__":
    main(csv_name="next100.csv", label="Next 100 (NIFTY 500 remainder)")
