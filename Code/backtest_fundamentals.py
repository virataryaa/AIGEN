"""
Backtest the fundamental ratios (ROE, Debt/Equity, Net Margin, FCF, YoY
growth) the same way the technical signals were tested: cross-sectional
quintile/IC analysis with a train/test split.

Key difference from the technical backtest: fundamentals update quarterly/
annually, not daily, and there's a REAL reporting lag between a quarter
ending and the results actually being public. Using the quarter-end date
as if the market already knew the numbers that day would be lookahead
bias. NSE mandates results within 45 days (quarterly) / 60 days (annual)
of period end -- used here as a conservative "earliest the market could
have known" date, since we don't have each filing's exact broadcast
timestamp in the ratio table.

Small-sample caveat: only 42-47 companies, ~6-8 years of history -> a few
hundred observations at most, far fewer than the technical backtest's
tens of thousands. Results here are preliminary/lower-confidence and
should be read that way, not with the same statistical weight.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
FUNDAMENTALS_DIR = BASE_DIR / "Database" / "fundamentals"
PRICES_DIR = BASE_DIR / "Database" / "prices"

REPORTING_LAG_DAYS = {"Quarterly": 45, "Annual": 60}
FWD_HORIZONS = [63, 126]  # ~3mo, ~6mo trading days
SPLIT_DATE = pd.Timestamp("2022-01-01")  # smaller sample -> less room than the 2018 split used for technical signals

RATIO_COLS = ["ROE", "DebtEquity_Calc", "NetMargin", "FCF", "InterestCoverage",
              "Revenue_YoY", "NetProfit_YoY"]
# Sign convention: for the backtest, "higher = better" -- DebtEquity is
# inverted (lower debt is better) so its IC/quintile direction matches the
# others (a genuinely good signal should show positive IC after this flip).
INVERT = {"DebtEquity_Calc"}


def load_price_series(ticker: str) -> pd.DataFrame | None:
    path = PRICES_DIR / f"{ticker}.NS.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["Date", "Adj_Close"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df.sort_values("Date").reset_index(drop=True)


def forward_return(price_df: pd.DataFrame, known_date: pd.Timestamp, horizon: int) -> float:
    idx = price_df["Date"].searchsorted(known_date)
    if idx >= len(price_df) or idx + horizon >= len(price_df):
        return np.nan
    return price_df["Adj_Close"].iloc[idx + horizon] / price_df["Adj_Close"].iloc[idx] - 1


def build_panel() -> pd.DataFrame:
    ratios = pd.read_parquet(FUNDAMENTALS_DIR / "ratios_wide.parquet")
    ratios["FilingToDate"] = pd.to_datetime(ratios["FilingToDate"], format="mixed", dayfirst=True, errors="coerce")
    ratios = ratios.dropna(subset=["FilingToDate"])
    ratios["KnownDate"] = ratios.apply(
        lambda r: r["FilingToDate"] + pd.Timedelta(days=REPORTING_LAG_DAYS.get(r["PeriodType"], 45)),
        axis=1,
    )

    price_cache = {}
    rows = []
    for ticker, grp in ratios.groupby("Ticker"):
        if ticker not in price_cache:
            price_cache[ticker] = load_price_series(ticker)
        price_df = price_cache[ticker]
        if price_df is None:
            continue

        for _, r in grp.iterrows():
            row = {"Ticker": ticker, "KnownDate": r["KnownDate"], "PeriodType": r["PeriodType"],
                   "Industry": r.get("Industry")}
            for col in RATIO_COLS:
                val = r.get(col)
                row[col] = -val if col in INVERT and pd.notna(val) else val
            for h in FWD_HORIZONS:
                row[f"FwdRet_{h}"] = forward_return(price_df, r["KnownDate"], h)
            rows.append(row)

    return pd.DataFrame(rows)


def analyze(panel: pd.DataFrame, col: str, horizon: int, rebalance_freq: str = "Q"):
    ret_col = f"FwdRet_{horizon}"
    # Ratios like NetMargin/InterestCoverage divide by a denominator that's
    # occasionally exactly 0 (e.g. a quarter with 0 reported Revenue) ->
    # +/-inf. Left in, these silently break qcut for whichever bucket they
    # land in (caught by the except below), which shrinks that ratio's
    # spread sample to only the "lucky" unaffected buckets -- producing a
    # misleadingly clean-looking hit rate. Treat as missing, not a value.
    sub = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=[col, ret_col]).copy()
    if sub.empty:
        return None
    sub["Bucket"] = sub["KnownDate"].dt.to_period(rebalance_freq)

    ics, spreads = [], []
    for _, grp in sub.groupby("Bucket"):
        if len(grp) < 8:  # need a reasonable number of names to rank meaningfully
            continue
        ic, _ = stats.spearmanr(grp[col], grp[ret_col])
        if not np.isnan(ic):
            ics.append(ic)
        try:
            g = grp.copy()
            g["T"] = pd.qcut(g[col], 3, duplicates="drop")  # tercile, not quintile -- small N per bucket
            means = g.groupby("T", observed=True)[ret_col].mean()
            if len(means) >= 2:
                sorted_t = means.index.sort_values()
                spreads.append(means[sorted_t[-1]] - means[sorted_t[0]])
        except (ValueError, IndexError):
            continue

    if not ics:
        return None
    ics = pd.Series(ics)
    spreads = pd.Series(spreads, dtype=float)
    return {
        "n_buckets": len(ics), "mean_ic": ics.mean(),
        "hit_rate": (spreads > 0).mean() if len(spreads) else np.nan,
        "mean_spread": spreads.mean() if len(spreads) else np.nan,
        "n_obs": len(sub),
    }


def main():
    print("Building fundamentals + forward-returns panel...")
    panel = build_panel()
    print(f"Panel: {len(panel)} ticker-period observations, {panel['Ticker'].nunique()} tickers")
    print(f"Date range: {panel['KnownDate'].min().date()} -> {panel['KnownDate'].max().date()}\n")

    train = panel[panel["KnownDate"] < SPLIT_DATE]
    test = panel[panel["KnownDate"] >= SPLIT_DATE]
    print(f"Train: {len(train)} obs before {SPLIT_DATE.date()} | Test: {len(test)} obs after\n")

    for horizon in FWD_HORIZONS:
        print(f"{'='*80}\nHORIZON: {horizon} trading days\n{'='*80}")
        results = []
        for col in RATIO_COLS:
            tr = analyze(train, col, horizon)
            te = analyze(test, col, horizon)
            results.append({
                "ratio": col,
                "train_ic": tr["mean_ic"] if tr else np.nan,
                "train_n": tr["n_obs"] if tr else 0,
                "test_ic": te["mean_ic"] if te else np.nan,
                "test_hit": te["hit_rate"] if te else np.nan,
                "test_n": te["n_obs"] if te else 0,
            })
        df = pd.DataFrame(results)
        df["train_ic"] = df["train_ic"].round(4)
        df["test_ic"] = df["test_ic"].round(4)
        df["test_hit"] = (df["test_hit"] * 100).round(1)
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
