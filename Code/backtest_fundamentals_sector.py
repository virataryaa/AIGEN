"""
Sector-neutral fundamentals backtest: within each (Industry, quarter)
group, rank companies by each ratio and go long the top half / short the
bottom half -- same-sector comparison only, so a "good ROE for a bank"
isn't compared against a "good ROE for an IT company" (different natural
baselines per sector). Tests whether sector-relative ranking recovers
signal that the universe-wide ranking (backtest_fundamentals.py) missed.

Small-sample reality: 97 companies across ~20 industries averages ~5
companies/industry -- many sector-quarter groups won't have enough names
for a meaningful split. MIN_GROUP_SIZE filters those out.
"""

import numpy as np
import pandas as pd
from scipy import stats

from backtest_fundamentals import RATIO_COLS, SPLIT_DATE, build_panel

FWD_HORIZONS = [63, 126]
MIN_GROUP_SIZE = 4  # need at least 4 same-sector, same-quarter companies to split top/bottom half


def analyze_sector_neutral(panel: pd.DataFrame, col: str, horizon: int):
    ret_col = f"FwdRet_{horizon}"
    sub = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=[col, ret_col, "Industry"]).copy()
    if sub.empty:
        return None

    sub["Bucket"] = sub["KnownDate"].dt.to_period("Q")

    ics, spreads = [], []
    for (bucket, industry), grp in sub.groupby(["Bucket", "Industry"]):
        if len(grp) < MIN_GROUP_SIZE:
            continue
        ic, _ = stats.spearmanr(grp[col], grp[ret_col])
        if not np.isnan(ic):
            ics.append(ic)

        median = grp[col].median()
        top = grp[grp[col] >= median][ret_col].mean()
        bottom = grp[grp[col] < median][ret_col].mean()
        if pd.notna(top) and pd.notna(bottom):
            spreads.append(top - bottom)

    if not ics:
        return None
    ics = pd.Series(ics)
    spreads = pd.Series(spreads, dtype=float)
    return {
        "n_groups": len(ics),
        "mean_ic": ics.mean(),
        "ic_tstat": stats.ttest_1samp(ics, 0)[0] if len(ics) > 1 else np.nan,
        "hit_rate": (spreads > 0).mean() if len(spreads) else np.nan,
        "mean_spread": spreads.mean() if len(spreads) else np.nan,
        "n_obs": len(sub),
    }


def main():
    print("Building fundamentals + forward-returns panel...")
    panel = build_panel()
    print(f"Panel: {len(panel)} obs, {panel['Ticker'].nunique()} tickers, "
          f"{panel['Industry'].nunique()} industries\n")

    industry_counts = panel.groupby("Industry")["Ticker"].nunique().sort_values(ascending=False)
    print("Companies per industry (top 10):")
    print(industry_counts.head(10).to_string())
    print()

    for horizon in FWD_HORIZONS:
        print(f"{'='*90}\nHORIZON: {horizon} trading days -- SECTOR-NEUTRAL (long top half / short bottom half, same industry+quarter)\n{'='*90}")
        results = []
        for col in RATIO_COLS:
            train = panel[panel["KnownDate"] < SPLIT_DATE]
            test = panel[panel["KnownDate"] >= SPLIT_DATE]
            tr = analyze_sector_neutral(train, col, horizon)
            te = analyze_sector_neutral(test, col, horizon)
            all_r = analyze_sector_neutral(panel, col, horizon)
            results.append({
                "ratio": col,
                "train_ic": tr["mean_ic"] if tr else np.nan,
                "test_ic": te["mean_ic"] if te else np.nan,
                "test_hit": te["hit_rate"] * 100 if te else np.nan,
                "test_spread": te["mean_spread"] * 100 if te else np.nan,
                "full_ic": all_r["mean_ic"] if all_r else np.nan,
                "full_n_groups": all_r["n_groups"] if all_r else 0,
            })
        df = pd.DataFrame(results).round(4)
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
