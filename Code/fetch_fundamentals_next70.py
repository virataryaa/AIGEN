"""Fetch fundamentals for the next 70 tickers (NIFTY 200 minus NIFTY 100
already fetched) -- same pipeline, different universe."""

from fetch_fundamentals import main

if __name__ == "__main__":
    main(csv_name="next70.csv", label="Next 70 (NIFTY 200 remainder)")
