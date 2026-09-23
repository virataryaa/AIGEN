"""
Combine fundamentals (ratios_wide.parquet) with the latest price
(Database/prices/<TICKER>.NS.parquet) to compute valuation multiples that
need BOTH: PE and PB. Neither exists in the XBRL data alone -- price is a
market fact, not a filed fundamental.

  PE = LatestPrice / TTM EPS        (TTM = sum of last 4 quarterly EPS_Basic)
  PB = LatestPrice / Book Value Per Share
       BVPS = Equity_Final / SharesOutstanding
       SharesOutstanding = PaidUpCapital / FaceValue  (both filed by the company)

Output: Database/fundamentals/valuation_snapshot.parquet -- one row per
ticker, latest available figures, ready for the dashboard.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FUNDAMENTALS_DIR = BASE_DIR / "Database" / "fundamentals"
PRICES_DIR = BASE_DIR / "Database" / "prices"


def load_latest_price(ticker: str) -> tuple[float | None, str | None]:
    path = PRICES_DIR / f"{ticker}.NS.parquet"
    if not path.exists():
        return None, None
    df = pd.read_parquet(path, columns=["Date", "Adj_Close"])
    if df.empty:
        return None, None
    df = df.sort_values("Date")
    return float(df["Adj_Close"].iloc[-1]), str(pd.to_datetime(df["Date"].iloc[-1]).date())


def compute_ttm_eps(group: pd.DataFrame) -> float | None:
    q = group[group["PeriodType"] == "Quarterly"].copy()
    q["_date"] = pd.to_datetime(q["FilingToDate"], format="mixed", dayfirst=True, errors="coerce")
    q = q.dropna(subset=["_date", "EPS_Basic"]).sort_values("_date")
    if len(q) < 4:
        return None
    return q["EPS_Basic"].tail(4).sum()


LATEST_VALUE_COLS = ["Equity_Final", "SharesOutstanding", "ROE", "DebtEquity_Calc",
                      "NetMargin", "InterestCoverage", "Revenue_YoY", "NetProfit_YoY",
                      "Revenue", "NetProfit", "Industry"]


def latest_nonnull(group: pd.DataFrame, col: str):
    """Most recent non-null value for this column specifically -- a ratio
    can be missing in the single latest filing (e.g. balance-sheet items
    aren't tagged in every quarter) without meaning the company has no
    recent value for it at all."""
    if col not in group.columns:
        return np.nan
    sub = group.dropna(subset=[col]).sort_values("_date")
    return sub[col].iloc[-1] if not sub.empty else np.nan


def main():
    ratios = pd.read_parquet(FUNDAMENTALS_DIR / "ratios_wide.parquet")
    universe = pd.read_parquet(BASE_DIR / "Database" / "universe.parquet")
    company_map = dict(zip(universe["Symbol"].str.strip(), universe["Company Name"]))

    rows = []
    for ticker, group in ratios.groupby("Ticker"):
        group = group.copy()
        group["_date"] = pd.to_datetime(group["FilingToDate"], format="mixed", dayfirst=True, errors="coerce")
        latest_q = group[group["PeriodType"] == "Quarterly"].dropna(subset=["_date"]).sort_values("_date")
        if latest_q.empty:
            continue
        latest_filing_date = latest_q["FilingToDate"].iloc[-1]

        latest_vals = {col: latest_nonnull(group, col) for col in LATEST_VALUE_COLS}
        equity = latest_vals["Equity_Final"]
        shares = latest_vals["SharesOutstanding"]

        price, price_date = load_latest_price(ticker)
        ttm_eps = compute_ttm_eps(group)

        bvps = equity / shares if shares and shares > 0 else np.nan
        pe = price / ttm_eps if price and ttm_eps and ttm_eps > 0 else np.nan
        pb = price / bvps if price and bvps and bvps > 0 else np.nan

        rows.append({
            "Ticker": ticker,
            "Company": company_map.get(ticker, ticker),
            "Industry": latest_vals["Industry"],
            "LatestFilingDate": latest_filing_date,
            "LatestPrice": price,
            "PriceDate": price_date,
            "TTM_EPS": ttm_eps,
            "PE": pe,
            "BVPS": bvps,
            "PB": pb,
            "SharesOutstanding": shares,
            "ROE": latest_vals["ROE"],
            "DebtEquity_Calc": latest_vals["DebtEquity_Calc"],
            "NetMargin": latest_vals["NetMargin"],
            "InterestCoverage": latest_vals["InterestCoverage"],
            "Revenue_YoY": latest_vals["Revenue_YoY"],
            "NetProfit_YoY": latest_vals["NetProfit_YoY"],
            "Revenue": latest_vals["Revenue"],
            "NetProfit": latest_vals["NetProfit"],
        })

    out = pd.DataFrame(rows)
    out_path = FUNDAMENTALS_DIR / "valuation_snapshot.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Built valuation snapshot: {len(out)} tickers")
    print(f"Saved -> {out_path}")
    print(f"\nCoverage:")
    for col in ["LatestPrice", "PE", "PB", "ROE", "DebtEquity_Calc"]:
        print(f"  {col}: {out[col].notna().sum()}/{len(out)}")

    print(f"\nSample (sorted by PE, valid PE only):")
    sample = out.dropna(subset=["PE"]).sort_values("PE")
    print(sample[["Ticker", "PE", "PB", "ROE", "DebtEquity_Calc"]].head(5).to_string(index=False))
    print(sample[["Ticker", "PE", "PB", "ROE", "DebtEquity_Calc"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
