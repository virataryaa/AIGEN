"""
Pivot the long-format fundamentals facts (Database/fundamentals/<TICKER>.parquet)
into a wide per-filing ratio table, using the canonical field names from
field_mapping.csv so it works consistently across companies/sectors.

Duration handling:
  - P&L items (Revenue, NetProfit, InterestExpense, TaxExpense, EPS) use the
    'Quarter' DurationClass for Quarterly filings, 'Annual' for Annual filings.
  - Balance sheet items (TotalEquity, Debt, TotalAssets) use 'Instant'.
  - Cash flow items (OperatingCashFlow, CapEx) only exist in Annual filings
    (confirmed earlier -- quarterly results filings don't carry a cash flow
    statement), so FCF is only computable at annual frequency.

Output: Database/fundamentals/ratios_wide.parquet
  One row per (Ticker, PeriodType, FilingToDate) with ratio columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FUNDAMENTALS_DIR = BASE_DIR / "Database" / "fundamentals"

FLOW_FIELDS = ["Revenue", "NetProfit", "NetProfitToOwners", "InterestExpense",
               "TaxExpense", "ProfitBeforeTax", "EPS_Basic", "EPS_Diluted",
               "OperatingCashFlow", "InvestingCashFlow", "FinancingCashFlow",
               "CapEx", "DividendsPaid",
               # Conceptually point-in-time (share capital), but NSE tags
               # them under the filing's duration context, not Instant --
               # confirmed empirically (0 Instant rows, all Quarter/Annual).
               "PaidUpCapital", "FaceValue"]
STOCK_FIELDS = ["TotalEquity", "EquityToOwners", "DebtCurrent", "DebtNonCurrent",
                "TotalAssets", "TotalLiabilities", "Cash", "Inventory",
                "Receivables", "Payables", "DebtEquityRatio_Reported",
                "BankCapital", "BankReserves", "BankBorrowings"]


NON_TICKER_FILES = {"ratios_wide.parquet", "valuation_snapshot.parquet"}


def load_all_facts() -> pd.DataFrame:
    files = sorted(f for f in FUNDAMENTALS_DIR.glob("*.parquet") if f.name not in NON_TICKER_FILES)
    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=[
            "Ticker", "PeriodType", "FilingToDate", "Industry",
            "CanonicalName", "DurationClass", "ValueNumeric",
        ])
        dfs.append(df.dropna(subset=["CanonicalName"]))
    return pd.concat(dfs, ignore_index=True)


def pivot_wide(facts: pd.DataFrame) -> pd.DataFrame:
    # Flow items: Quarter-duration for Quarterly filings, Annual-duration for Annual filings
    flow = facts[
        (facts["CanonicalName"].isin(FLOW_FIELDS)) &
        (
            ((facts["PeriodType"] == "Quarterly") & (facts["DurationClass"] == "Quarter")) |
            ((facts["PeriodType"] == "Annual") & (facts["DurationClass"] == "Annual"))
        )
    ]
    flow_wide = flow.pivot_table(
        index=["Ticker", "PeriodType", "FilingToDate", "Industry"],
        columns="CanonicalName", values="ValueNumeric", aggfunc="first",
    )

    # Stock (balance sheet) items: Instant, same filing
    stock = facts[
        (facts["CanonicalName"].isin(STOCK_FIELDS)) &
        (facts["DurationClass"] == "Instant")
    ]
    stock_wide = stock.pivot_table(
        index=["Ticker", "PeriodType", "FilingToDate", "Industry"],
        columns="CanonicalName", values="ValueNumeric", aggfunc="first",
    )

    wide = flow_wide.join(stock_wide, how="outer").reset_index()
    return wide


def compute_ratios(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()

    zeros = pd.Series(0.0, index=df.index)
    nans = pd.Series(np.nan, index=df.index)
    # Banks report Capital + Reserves&Surplus instead of a single Equity
    # tag (different Ind-AS schedule for financial companies) -- fall back
    # to their sum when the general-taxonomy Equity tags are absent.
    bank_equity = df.get("BankCapital", zeros).fillna(0) + df.get("BankReserves", zeros).fillna(0)
    equity = df.get("TotalEquity", df.get("EquityToOwners", nans))
    equity = equity.where(equity.notna(), bank_equity.where(bank_equity > 0, np.nan))
    net_profit = df.get("NetProfit", df.get("NetProfitToOwners", nans))
    debt = df.get("DebtCurrent", zeros).fillna(0) + df.get("DebtNonCurrent", zeros).fillna(0)
    debt = debt.where(debt > 0, df.get("BankBorrowings", nans))

    df["Equity_Final"] = equity
    df["SharesOutstanding"] = df.get("PaidUpCapital", nans) / df.get("FaceValue", nans)
    df["NetProfit_Final"] = net_profit

    df = df.sort_values(["Ticker", "PeriodType", "FilingToDate"])

    # ROE divides a per-QUARTER profit flow by a full (annual-scale) equity
    # stock -- using the raw single-quarter NetProfit directly understates
    # ROE by ~4x for Quarterly rows (caught by cross-checking against
    # yfinance: TCS showed 12.7% here vs 47.7% there, ~4x apart, same
    # pattern on every ticker checked). Fix: use trailing-twelve-month
    # (last 4 quarters, rolling) net profit for Quarterly-period rows.
    # Annual rows already report a full-year profit, so they're untouched.
    ttm_profit = (
        df.groupby(["Ticker", "PeriodType"])["NetProfit_Final"]
        .transform(lambda s: s.rolling(4, min_periods=4).sum())
    )
    roe_numerator = df["NetProfit_Final"].where(df["PeriodType"] == "Annual", ttm_profit)

    df["ROE"] = roe_numerator / equity
    df["DebtEquity_Calc"] = debt / equity
    df["NetMargin"] = net_profit / df.get("Revenue", nans)
    df["InterestCoverage"] = df.get("ProfitBeforeTax", nans) / df.get("InterestExpense", nans)

    if "OperatingCashFlow" in df.columns and "CapEx" in df.columns:
        df["FCF"] = df["OperatingCashFlow"] - df["CapEx"].abs()

    df["Revenue_YoY"] = df.groupby(["Ticker", "PeriodType"])["Revenue"].pct_change(4)
    df["NetProfit_YoY"] = df.groupby(["Ticker", "PeriodType"])[
        "NetProfit" if "NetProfit" in df.columns else "NetProfitToOwners"
    ].pct_change(4)

    return df


def main():
    print("Loading all fundamentals facts...")
    facts = load_all_facts()
    print(f"Loaded {len(facts):,} canonical facts across {facts['Ticker'].nunique()} tickers")

    wide = pivot_wide(facts)
    print(f"Pivoted to {len(wide)} (Ticker, PeriodType, FilingToDate) rows")

    ratios = compute_ratios(wide)
    out_path = FUNDAMENTALS_DIR / "ratios_wide.parquet"
    ratios.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")

    print("\nSample (RELIANCE, Quarterly):")
    sample = ratios[(ratios["Ticker"] == "RELIANCE") & (ratios["PeriodType"] == "Quarterly")]
    cols = ["FilingToDate", "Revenue", "NetProfit", "ROE", "DebtEquity_Calc", "NetMargin"]
    print(sample[cols].tail(6).to_string(index=False))

    print("\nCoverage:")
    for col in ["ROE", "DebtEquity_Calc", "NetMargin", "FCF", "InterestCoverage"]:
        if col in ratios.columns:
            n = ratios[col].notna().sum()
            print(f"  {col}: {n}/{len(ratios)} rows ({ratios[ratios[col].notna()]['Ticker'].nunique()} tickers)")


if __name__ == "__main__":
    main()
