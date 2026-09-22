"""
Grid-search all signal families/parameters in signal_library.py against
forward returns, with an in-sample (train) / out-of-sample (test) split
to guard against overfitting/curve-fitting to noise (34 signal instances x
~900 tickers x 25 years — some will look good by chance on the full
period alone; only trust what holds up in BOTH windows).

Train: rebalance dates < 2018-01-01
Test:  rebalance dates >= 2018-01-01

Runs entirely locally. Prints a ranked leaderboard, sorted by TEST-period
hit rate (out-of-sample is what matters — train performance alone proves
nothing about a signal that was cherry-picked from many candidates).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from signal_library import compute_all_signals, is_discrete

BASE_DIR = Path(__file__).resolve().parent.parent
PRICES_DIR = BASE_DIR / "Database" / "prices"

MIN_ROWS_REQUIRED = 300
FWD_HORIZONS = [21, 63]
SPLIT_DATE = pd.Timestamp("2018-01-01")


def build_ticker_panel(ticker: str) -> pd.DataFrame | None:
    df = pd.read_parquet(PRICES_DIR / f"{ticker}.parquet")
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < MIN_ROWS_REQUIRED:
        return None

    close = df["Adj_Close"].astype("float64")
    signals = compute_all_signals(close)
    signal_names = list(signals.keys())

    dates = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    year_month = dates.dt.to_period("M")
    rebal_idx = pd.Series(range(len(df))).groupby(year_month.values).head(1).to_numpy()
    rebal_idx = rebal_idx[rebal_idx >= MIN_ROWS_REQUIRED]

    close_arr = close.to_numpy()
    n = len(df)
    rows = []
    for i in rebal_idx:
        ok = True
        fwd = {}
        for h in FWD_HORIZONS:
            if i + h >= n:
                ok = False
                break
            fwd[f"FwdRet_{h}"] = close_arr[i + h] / close_arr[i] - 1
        if not ok:
            continue
        row = {"Ticker": ticker, "Date": dates.iloc[i], **fwd}
        for name in signal_names:
            row[name] = signals[name].iloc[i]
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def analyze_signal(panel: pd.DataFrame, signal: str, horizon: int) -> dict:
    ret_col = f"FwdRet_{horizon}"
    discrete = is_discrete(signal)
    ics, spreads = [], []

    for date, grp in panel.groupby("Date"):
        grp = grp.dropna(subset=[signal, ret_col])
        if len(grp) < 20:
            continue
        ic, _ = stats.spearmanr(grp[signal], grp[ret_col])
        if not np.isnan(ic):
            ics.append(ic)

        if discrete:
            means = grp.groupby(signal)[ret_col].mean()
            if len(means) < 2:
                continue
            # directional: highest STATE VALUE (e.g. +1) minus lowest STATE
            # VALUE (e.g. -1) -- NOT max(means)-min(means), which is always
            # >= 0 by construction and silently produces a fake 100% hit
            # rate (caught in review, see README mistake log).
            top_val, bot_val = means.index.max(), means.index.min()
            spreads.append(means[top_val] - means[bot_val])
        else:
            try:
                grp = grp.copy()
                grp["Q"] = pd.qcut(grp[signal], 5, duplicates="drop")
                means = grp.groupby("Q", observed=True)[ret_col].mean()
                if len(means) < 2:
                    continue
                # top bin = bin with the highest underlying signal values
                sorted_bins = means.index.sort_values()
                spreads.append(means[sorted_bins[-1]] - means[sorted_bins[0]])
            except (ValueError, IndexError):
                continue

    ics = pd.Series(ics, dtype=float)
    spreads = pd.Series(spreads, dtype=float)
    hit_rate = (spreads > 0).mean() if len(spreads) else np.nan
    return {
        "n_periods": len(spreads),
        "mean_ic": ics.mean() if len(ics) else np.nan,
        "mean_spread": spreads.mean() if len(spreads) else np.nan,
        "hit_rate": hit_rate,
    }


def main():
    tickers = sorted(p.stem for p in PRICES_DIR.glob("*.parquet"))
    print(f"Building expanded signal panel for {len(tickers)} tickers "
          f"(34 signal instances each)...\n")

    t0 = time.time()
    panels = []
    for i, ticker in enumerate(tickers, 1):
        result = build_ticker_panel(ticker)
        if result is not None:
            panels.append(result)
        if i % 200 == 0:
            print(f"  {i}/{len(tickers)} processed... ({time.time()-t0:.0f}s elapsed)")

    panel = pd.concat(panels, ignore_index=True)
    elapsed = time.time() - t0
    print(f"\nPanel built in {elapsed:.1f}s: {len(panel)} obs, {panel['Ticker'].nunique()} tickers\n")

    signal_cols = [c for c in panel.columns if c not in ("Ticker", "Date", "FwdRet_21", "FwdRet_63")]
    print(f"Testing {len(signal_cols)} signal instances across train/test split "
          f"(split date: {SPLIT_DATE.date()})...\n")

    train_panel = panel[panel["Date"] < SPLIT_DATE]
    test_panel = panel[panel["Date"] >= SPLIT_DATE]
    print(f"Train: {train_panel['Date'].min().date()} -> {train_panel['Date'].max().date()} "
          f"({train_panel['Date'].nunique()} months)")
    print(f"Test:  {test_panel['Date'].min().date()} -> {test_panel['Date'].max().date()} "
          f"({test_panel['Date'].nunique()} months)\n")

    for horizon in FWD_HORIZONS:
        print(f"\n{'='*100}\nHORIZON: {horizon} trading days\n{'='*100}")
        results = []
        for signal in signal_cols:
            tr = analyze_signal(train_panel, signal, horizon)
            te = analyze_signal(test_panel, signal, horizon)
            results.append({
                "signal": signal,
                "train_hit": tr["hit_rate"], "train_ic": tr["mean_ic"],
                "test_hit": te["hit_rate"], "test_ic": te["mean_ic"],
                "robust": (tr["hit_rate"] > 0.5) and (te["hit_rate"] > 0.5) and (tr["mean_ic"] > 0) and (te["mean_ic"] > 0),
            })

        df = pd.DataFrame(results).sort_values("test_hit", ascending=False)
        df["train_hit"] = (df["train_hit"] * 100).round(1)
        df["test_hit"] = (df["test_hit"] * 100).round(1)
        df["train_ic"] = df["train_ic"].round(4)
        df["test_ic"] = df["test_ic"].round(4)

        print(f"\nFull leaderboard (sorted by TEST hit rate — out-of-sample):")
        print(df.to_string(index=False))

        print(f"\n--- ROBUST signals (hit rate > 50% AND IC > 0 in BOTH train and test) ---")
        robust_df = df[df["robust"]].sort_values("test_hit", ascending=False)
        if robust_df.empty:
            print("  None found at this horizon.")
        else:
            print(robust_df.drop(columns="robust").to_string(index=False))


if __name__ == "__main__":
    main()
