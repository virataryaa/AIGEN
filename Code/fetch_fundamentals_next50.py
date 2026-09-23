"""Fetch fundamentals for the NIFTY Next 50 tickers (ranks 51-100 by index
tier) -- same pipeline as fetch_fundamentals.py, different universe."""

from fetch_fundamentals import main

if __name__ == "__main__":
    main(csv_name="niftynext50.csv", label="NIFTY Next 50")
