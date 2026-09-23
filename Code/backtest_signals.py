"""
Backtest the composite trend-following signal (compute_signals.py) —
standard cross-sectional factor test:

For each monthly rebalance date, rank all tickers by their composite score
(computed using only information up to that date — no lookahead), bucket
into quintiles, and measure forward returns (21 and 63 trading days ahead).
A useful signal should show top quintile (Q1) consistently outperforming
bottom quintile (Q5), with a positive average Information Coefficient (IC)
= cross-sectional rank correlation between score and forward return.

Runs entirely locally. Prints a stats summary — no dashboard changes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from compute_signals import compute_composite

BASE_DIR = Path(__file__).resolve().parent.parent
PRICES_DIR = BASE_DIR / "Database" / "prices"

MIN_ROWS_REQUIRED = 300  # need ~250+ for MA_200/Donchian warmup before trusting the score
FWD_HORIZONS = [21, 63]  # trading days: ~1 month, ~3 months


def build_ticker_panel(ticker: str) -> pd.DataFrame | None:
    df = pd.read_parquet(PRICES_DIR / f"{ticker}.parquet")
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < MIN_ROWS_REQUIRED:
        return None

    sig = compute_composite(df)
    sig["Date"] = pd.to_datetime(sig["Date"]).dt.tz_localize(None)
    sig["YearMonth"] = sig["Date"].dt.to_period("M")

    # first available trading day of each month = rebalance point
    rebal_idx = sig.groupby("YearMonth").head(1).index
    # drop warmup period (before MIN_ROWS_REQUIRED valid rows)
    rebal_idx = rebal_idx[rebal_idx >= MIN_ROWS_REQUIRED]

    rows = []
    close = sig["Close"].to_numpy()
    n = len(sig)
    for i in rebal_idx:
        if pd.isna(sig["Composite"].iloc[i]):
            continue
        row = {
            "Ticker": ticker,
            "Date": sig["Date"].iloc[i],
            "Composite": sig["Composite"].iloc[i],
        }
        ok = True
        for h in FWD_HORIZONS:
            if i + h >= n:
                ok = False
                break
            row[f"FwdRet_{h}"] = close[i + h] / close[i] - 1
        if ok:
            rows.append(row)

    return pd.DataFrame(rows) if rows else None


def quintile_analysis(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    ret_col = f"FwdRet_{horizon}"
    results = []
    for date, grp in panel.groupby("Date"):
        grp = grp.dropna(subset=["Composite", ret_col])
        if len(grp) < 20:  # need enough names to form quintiles meaningfully
            continue
        grp = grp.copy()
        grp["Quintile"] = pd.qcut(grp["Composite"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        q_means = grp.groupby("Quintile", observed=True)[ret_col].mean()
        ic, _ = stats.spearmanr(grp["Composite"], grp[ret_col])
        results.append({"Date": date, "IC": ic, "N": len(grp), **{f"Q{q}": q_means.get(q, np.nan) for q in [1, 2, 3, 4, 5]}})
    return pd.DataFrame(results)


def main():
    tickers = sorted(p.stem for p in PRICES_DIR.glob("*.parquet"))
    print(f"Building signal/forward-return panel for {len(tickers)} tickers...\n")

    panels = []
    for i, ticker in enumerate(tickers, 1):
        result = build_ticker_panel(ticker)
        if result is not None:
            panels.append(result)
        if i % 200 == 0:
            print(f"  {i}/{len(tickers)} processed...")

    panel = pd.concat(panels, ignore_index=True)
    print(f"\nPanel built: {len(panel)} ticker-month observations, "
          f"{panel['Ticker'].nunique()} tickers, "
          f"{panel['Date'].min().date()} -> {panel['Date'].max().date()}\n")

    for horizon in FWD_HORIZONS:
        print(f"\n{'='*70}\nHORIZON: {horizon} trading days (~{horizon//21} month(s))\n{'='*70}")
        qa = quintile_analysis(panel, horizon)

        print(f"Rebalance months analyzed: {len(qa)}")
        print(f"\nMean forward return by quintile (Q1=highest composite score, Q5=lowest):")
        q_avg = qa[["Q1", "Q2", "Q3", "Q4", "Q5"]].mean()
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            print(f"  {q}: {q_avg[q]*100:6.2f}%")

        spread = qa["Q1"] - qa["Q5"]
        t_stat, p_val = stats.ttest_1samp(spread.dropna(), 0)
        hit_rate = (spread > 0).mean()

        print(f"\nQ1 - Q5 spread (long top / short bottom quintile):")
        print(f"  Mean spread per period: {spread.mean()*100:.2f}%")
        print(f"  Annualized (x{12//(horizon//21)} periods/year approx): {spread.mean()*100*(12/(horizon/21)):.2f}%")
        print(f"  t-stat: {t_stat:.2f}  (p-value: {p_val:.4f})")
        print(f"  Hit rate (Q1 > Q5): {hit_rate*100:.1f}% of {spread.notna().sum()} months")

        print(f"\nInformation Coefficient (Spearman rank corr, score vs fwd return):")
        print(f"  Mean IC: {qa['IC'].mean():.4f}")
        print(f"  IC t-stat: {stats.ttest_1samp(qa['IC'].dropna(), 0)[0]:.2f}")
        print(f"  % months IC > 0: {(qa['IC'] > 0).mean()*100:.1f}%")

        monotonic = (q_avg["Q1"] > q_avg["Q2"] > q_avg["Q3"] > q_avg["Q4"] > q_avg["Q5"])
        print(f"\nMonotonic Q1>Q2>Q3>Q4>Q5: {monotonic}")


if __name__ == "__main__":
    main()
