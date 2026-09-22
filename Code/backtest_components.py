"""
Decompose backtest_signals.py's negative Composite result: test each of the
6 underlying signals standalone (same monthly-rebalance, forward-return,
IC/quintile methodology), plus two alternative composite re-weightings, to
find which component(s) are dragging performance down.

Continuous signals (Momentum_20, Momentum_100, RSI_14, Composite variants):
quintile analysis. Discrete signals (MA_cross_50_200 in {-1,1}, Donchian_20
in {-1,0,1}): grouped by actual value instead of artificial quintiles.

Runs entirely locally. Prints stats only.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from compute_signals import compute_composite

BASE_DIR = Path(__file__).resolve().parent.parent
PRICES_DIR = BASE_DIR / "Database" / "prices"

MIN_ROWS_REQUIRED = 300
FWD_HORIZONS = [21, 63]

CONTINUOUS_SIGNALS = ["Momentum_20", "Momentum_100", "RSI_14", "Composite",
                      "Composite_NoDonchian", "Composite_MomentumOnly"]
DISCRETE_SIGNALS = ["MA_cross_50_200", "Donchian_20"]
ALL_SIGNALS = CONTINUOUS_SIGNALS + DISCRETE_SIGNALS


def build_ticker_panel(ticker: str) -> pd.DataFrame | None:
    df = pd.read_parquet(PRICES_DIR / f"{ticker}.parquet")
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < MIN_ROWS_REQUIRED:
        return None

    sig = compute_composite(df)
    sig["Composite_NoDonchian"] = (
        0.55 * sig[["Momentum_20", "Momentum_100"]].mean(axis=1)
        + 0.45 * sig["MA_cross_50_200"]
    )
    sig["Composite_MomentumOnly"] = sig[["Momentum_20", "Momentum_100"]].mean(axis=1)

    sig["Date"] = pd.to_datetime(sig["Date"]).dt.tz_localize(None)
    sig["YearMonth"] = sig["Date"].dt.to_period("M")

    rebal_idx = sig.groupby("YearMonth").head(1).index
    rebal_idx = rebal_idx[rebal_idx >= MIN_ROWS_REQUIRED]

    rows = []
    close = sig["Close"].to_numpy()
    n = len(sig)
    for i in rebal_idx:
        row = {"Ticker": ticker, "Date": sig["Date"].iloc[i]}
        for s in ALL_SIGNALS:
            row[s] = sig[s].iloc[i]
        ok = True
        for h in FWD_HORIZONS:
            if i + h >= n:
                ok = False
                break
            row[f"FwdRet_{h}"] = close[i + h] / close[i] - 1
        if ok:
            rows.append(row)

    return pd.DataFrame(rows) if rows else None


def analyze_continuous(panel: pd.DataFrame, signal: str, horizon: int) -> dict:
    ret_col = f"FwdRet_{horizon}"
    ics, q_means_list = [], []
    for date, grp in panel.groupby("Date"):
        grp = grp.dropna(subset=[signal, ret_col])
        if len(grp) < 20:
            continue
        try:
            grp = grp.copy()
            grp["Q"] = pd.qcut(grp[signal], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
            q_means_list.append(grp.groupby("Q", observed=True)[ret_col].mean())
        except ValueError:
            continue
        ic, _ = stats.spearmanr(grp[signal], grp[ret_col])
        ics.append(ic)

    ics = pd.Series(ics).dropna()
    q_df = pd.DataFrame(q_means_list)
    q_avg = q_df.mean()
    top_label = q_df.columns.max() if len(q_df.columns) else None
    bot_label = q_df.columns.min() if len(q_df.columns) else None
    spread = (q_df[top_label] - q_df[bot_label]).dropna() if top_label is not None else pd.Series(dtype=float)
    t_stat, p_val = (stats.ttest_1samp(spread, 0) if len(spread) > 1 else (np.nan, np.nan))

    return {
        "signal": signal, "horizon": horizon, "n_periods": len(ics),
        "mean_ic": ics.mean(), "ic_tstat": stats.ttest_1samp(ics, 0)[0] if len(ics) > 1 else np.nan,
        "pct_ic_pos": (ics > 0).mean() if len(ics) else np.nan,
        "top_q_ret": q_avg.get(top_label, np.nan), "bot_q_ret": q_avg.get(bot_label, np.nan),
        "spread": spread.mean(), "spread_tstat": t_stat, "spread_pval": p_val,
        "hit_rate": (spread > 0).mean() if len(spread) else np.nan,
    }


def analyze_discrete(panel: pd.DataFrame, signal: str, horizon: int) -> dict:
    ret_col = f"FwdRet_{horizon}"
    ics, group_means_list = [], []
    for date, grp in panel.groupby("Date"):
        grp = grp.dropna(subset=[signal, ret_col])
        if len(grp) < 20:
            continue
        group_means_list.append(grp.groupby(signal)[ret_col].mean())
        ic, _ = stats.spearmanr(grp[signal], grp[ret_col])
        ics.append(ic)

    ics = pd.Series(ics).dropna()
    g_df = pd.DataFrame(group_means_list)
    g_avg = g_df.mean()
    vals = sorted(g_avg.index)
    top_val, bot_val = max(vals), min(vals)
    spread = (g_df[top_val] - g_df[bot_val]).dropna() if top_val in g_df.columns and bot_val in g_df.columns else pd.Series(dtype=float)
    t_stat, p_val = (stats.ttest_1samp(spread, 0) if len(spread) > 1 else (np.nan, np.nan))

    return {
        "signal": signal, "horizon": horizon, "n_periods": len(ics),
        "mean_ic": ics.mean(), "ic_tstat": stats.ttest_1samp(ics, 0)[0] if len(ics) > 1 else np.nan,
        "pct_ic_pos": (ics > 0).mean() if len(ics) else np.nan,
        "group_means": g_avg.to_dict(),
        "spread": spread.mean(), "spread_tstat": t_stat, "spread_pval": p_val,
        "hit_rate": (spread > 0).mean() if len(spread) else np.nan,
    }


def main():
    tickers = sorted(p.stem for p in PRICES_DIR.glob("*.parquet"))
    print(f"Building full signal panel for {len(tickers)} tickers...\n")

    t0 = time.time()
    panels = []
    for i, ticker in enumerate(tickers, 1):
        result = build_ticker_panel(ticker)
        if result is not None:
            panels.append(result)
        if i % 200 == 0:
            print(f"  {i}/{len(tickers)} processed...")

    panel = pd.concat(panels, ignore_index=True)
    elapsed = time.time() - t0
    print(f"\nPanel built in {elapsed:.1f}s: {len(panel)} obs, {panel['Ticker'].nunique()} tickers\n")

    all_results = []
    for horizon in FWD_HORIZONS:
        print(f"\n{'='*90}\nHORIZON: {horizon} trading days\n{'='*90}")
        for signal in CONTINUOUS_SIGNALS:
            r = analyze_continuous(panel, signal, horizon)
            all_results.append(r)
            print(f"\n--- {signal} ---")
            print(f"  Top-Bottom quintile: {r['top_q_ret']*100:.2f}% vs {r['bot_q_ret']*100:.2f}%  "
                  f"(spread {r['spread']*100:+.2f}%, t={r['spread_tstat']:.2f}, p={r['spread_pval']:.4f}, hit={r['hit_rate']*100:.1f}%)")
            print(f"  Mean IC: {r['mean_ic']:+.4f}  (t={r['ic_tstat']:.2f}, %IC>0={r['pct_ic_pos']*100:.1f}%)  n_periods={r['n_periods']}")

        for signal in DISCRETE_SIGNALS:
            r = analyze_discrete(panel, signal, horizon)
            all_results.append(r)
            print(f"\n--- {signal} (discrete) ---")
            gm = {k: f"{v*100:.2f}%" for k, v in sorted(r["group_means"].items())}
            print(f"  Mean fwd return by state: {gm}")
            print(f"  High-Low spread: {r['spread']*100:+.2f}%  (t={r['spread_tstat']:.2f}, p={r['spread_pval']:.4f}, hit={r['hit_rate']*100:.1f}%)")
            print(f"  Mean IC: {r['mean_ic']:+.4f}  (t={r['ic_tstat']:.2f}, %IC>0={r['pct_ic_pos']*100:.1f}%)  n_periods={r['n_periods']}")

    print(f"\n\n{'='*90}\nSUMMARY TABLE (spread t-stat, negative = signal is INVERTED/harmful)\n{'='*90}")
    summary = pd.DataFrame(all_results)[["signal", "horizon", "mean_ic", "spread", "spread_tstat", "hit_rate"]]
    summary["spread"] = (summary["spread"] * 100).round(2)
    summary["mean_ic"] = summary["mean_ic"].round(4)
    summary["hit_rate"] = (summary["hit_rate"] * 100).round(1)
    summary["spread_tstat"] = summary["spread_tstat"].round(2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
