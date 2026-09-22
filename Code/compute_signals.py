"""
Simplified trend-following composite signal engine for Indian equities —
adapted from the CTA (commodities) project's methodology, cut down from
144 indicators to ~7, and 3 time-buckets to 2.

All computation runs LOCALLY (this script). The Streamlit dashboard only
ever reads the resulting parquet output — no recompute in the dashboard.

Hand-rolled (no pandas_ta) — pandas_ta pulls in numba, which has broken
Streamlit Cloud deploys before (confirmed in the CTA project). Same lesson
applies here even though this script never runs on Cloud itself, to keep
the codebase consistent if any of this logic is ever reused dashboard-side.

Signals (all normalized to roughly [-1, 1], +1 = bullish):
  Momentum_20, Momentum_100  — vol-normalized momentum (tanh)
  MA_cross_50_200            — golden/death cross
  Donchian_20                — stateful breakout signal
  RSI_14                     — (RSI-50)/50, continuous

Composite = 0.40 * mean(momentum signals)
          + 0.30 * MA_cross_50_200
          + 0.30 * Donchian_20
(RSI_14 reported separately, as an overbought/oversold sanity check —
not part of the composite weight, since it's more mean-reversion than trend)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PRICES_DIR = BASE_DIR / "Database" / "prices"

N_TEST_TICKERS = 15


def vol_norm_momentum(close: pd.Series, n: int) -> pd.Series:
    ret = close.pct_change(n)
    daily_vol = close.pct_change().rolling(n).std()
    return np.tanh(ret / (daily_vol * np.sqrt(n))).fillna(0)


def ma_cross_signal(close: pd.Series, short: int, long: int) -> pd.Series:
    sma_s = close.rolling(short).mean()
    sma_l = close.rolling(long).mean()
    sig = pd.Series(np.where(sma_s > sma_l, 1.0, -1.0), index=close.index)
    return sig.where(sma_s.notna() & sma_l.notna(), 0.0)


def donchian_signal(close: pd.Series, n: int) -> pd.Series:
    """Stateful breakout signal: +1 on new n-day high, -1 on new n-day low, hold otherwise."""
    rolling_high = close.rolling(n).max()
    rolling_low = close.rolling(n).min()
    signals = np.zeros(len(close))
    state = 0
    c, hi, lo = close.to_numpy(), rolling_high.to_numpy(), rolling_low.to_numpy()
    for i in range(len(close)):
        if np.isnan(hi[i]) or np.isnan(lo[i]):
            signals[i] = 0
            continue
        if c[i] >= hi[i]:
            state = 1
        elif c[i] <= lo[i]:
            state = -1
        signals[i] = state
    return pd.Series(signals, index=close.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Adj_Close"].astype("float64")

    mom20 = vol_norm_momentum(close, 20)
    mom100 = vol_norm_momentum(close, 100)
    ma_cross = ma_cross_signal(close, 50, 200)
    donchian = donchian_signal(close, 20)
    rsi14 = rsi(close, 14)

    out = pd.DataFrame({
        "Date": df["Date"],
        "Close": close,
        "Momentum_20": mom20,
        "Momentum_100": mom100,
        "MA_cross_50_200": ma_cross,
        "Donchian_20": donchian,
        "RSI_14": rsi14,
    })
    out["Composite"] = (
        0.40 * out[["Momentum_20", "Momentum_100"]].mean(axis=1)
        + 0.30 * out["MA_cross_50_200"]
        + 0.30 * out["Donchian_20"]
    )
    return out


def main():
    tickers = sorted(p.stem for p in PRICES_DIR.glob("*.parquet"))[:N_TEST_TICKERS]
    print(f"Testing composite signal engine on {len(tickers)} tickers:\n{tickers}\n")

    t0 = time.time()
    latest_rows = []
    for ticker in tickers:
        df = pd.read_parquet(PRICES_DIR / f"{ticker}.parquet")
        df = df.sort_values("Date").reset_index(drop=True)
        sig_df = compute_composite(df)
        latest = sig_df.iloc[-1]
        latest_rows.append({
            "Ticker": ticker,
            "Date": latest["Date"],
            "Close": round(latest["Close"], 2),
            "Momentum_20": round(latest["Momentum_20"], 3),
            "Momentum_100": round(latest["Momentum_100"], 3),
            "MA_cross_50_200": latest["MA_cross_50_200"],
            "Donchian_20": latest["Donchian_20"],
            "RSI_14": round(latest["RSI_14"], 1) if not np.isnan(latest["RSI_14"]) else None,
            "Composite": round(latest["Composite"], 3),
        })

    elapsed = time.time() - t0

    result = pd.DataFrame(latest_rows).sort_values("Composite", ascending=False)
    print(result.to_string(index=False))

    print(f"\n=== TIMING ===")
    print(f"Total: {elapsed:.3f} sec for {len(tickers)} tickers")
    print(f"Avg per ticker: {elapsed/len(tickers)*1000:.1f} ms")
    print(f"Extrapolated for 982 tickers: {elapsed/len(tickers)*982:.1f} sec ({elapsed/len(tickers)*982/60:.2f} min)")


if __name__ == "__main__":
    main()
