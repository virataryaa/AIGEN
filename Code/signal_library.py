"""
Expanded technical signal library — hand-rolled (no pandas_ta, see README
mistake log #3: pandas_ta's numba dependency breaks Streamlit Cloud).
BBands/TRIX/KAMA formulas adapted from the sibling LSEG-CTA project
(validated there against pandas_ta's own output, max abs diff 0.0).

Each function returns {signal_name: pd.Series} for a grid of parameters,
so optimize_signals.py can just merge all the dicts together.
"""

import numpy as np
import pandas as pd

MOMENTUM_WINDOWS = [10, 20, 50, 100, 150, 200, 300, 400, 500]
MA_PAIRS = [(10, 50), (20, 50), (20, 100), (50, 150), (50, 200), (100, 300), (150, 400), (200, 500)]
EMA_PAIRS = [(10, 50), (20, 50), (20, 100), (50, 150), (50, 200), (100, 300), (150, 400), (200, 500)]
RSI_PERIODS = [7, 14, 21]
TRIX_PERIODS = [10, 20, 50, 100]
KAMA_PERIODS = [10, 20, 50, 100]
DONCHIAN_WINDOWS = [10, 20, 55, 100]
BOLLINGER_WINDOWS = [20, 50, 100]

# Trend-following additions
LRS_WINDOWS = [50, 100, 150, 200]          # linear regression slope (sign)
HIGH52W_WINDOW = 252                        # proximity to 52-week high

# Mean-reversion candidates (opposite hypothesis to everything above --
# short-term overbought/oversold reversal instead of trend continuation)
ZSCORE_WINDOWS = [10, 20, 50]               # (close - SMA) / std, raw value
SHORT_REVERSAL_WINDOWS = [5, 10]            # raw n-day return (let IC sign show if it reverses or continues)

# Raw (non-vol-normalized) N-day returns -- base for cross-sectional
# relative-strength-vs-market, computed at the panel level in
# optimize_signals_nifty500.py (needs all tickers on the same date, which
# a single-ticker signal function can't see).
RAW_RET_WINDOWS = [100, 150, 200]

# Volume-based (untested data dimension until now -- everything above is
# price-only). OBV = On Balance Volume: cumulative sum of volume, signed by
# the day's price direction. Its own trend (LRS sign) is the signal.
OBV_LRS_WINDOWS = [50, 100, 200]


def _vol_norm_momentum(close: pd.Series, n: int) -> pd.Series:
    ret = close.pct_change(n)
    daily_vol = close.pct_change().rolling(n).std()
    return np.tanh(ret / (daily_vol * np.sqrt(n))).fillna(0)


def _ma_cross(close: pd.Series, s: int, l: int) -> pd.Series:
    sma_s, sma_l = close.rolling(s).mean(), close.rolling(l).mean()
    sig = pd.Series(np.where(sma_s > sma_l, 1.0, -1.0), index=close.index)
    return sig.where(sma_s.notna() & sma_l.notna(), 0.0)


def _ema_cross(close: pd.Series, s: int, l: int) -> pd.Series:
    ema_s = close.ewm(span=s, adjust=False).mean()
    ema_l = close.ewm(span=l, adjust=False).mean()
    return pd.Series(np.where(ema_s > ema_l, 1.0, -1.0), index=close.index)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _ema_presma(close: pd.Series, n: int) -> pd.Series:
    s2 = close.copy()
    if len(s2) >= n:
        sma_seed = s2.iloc[:n].mean()
        s2.iloc[:n - 1] = np.nan
        s2.iloc[n - 1] = sma_seed
    return s2.ewm(span=n, adjust=False).mean()


def _trix_sign(close: pd.Series, n: int) -> pd.Series:
    eff_n = max(n, 9)
    ema1 = _ema_presma(close, eff_n)
    ema2 = _ema_presma(ema1, eff_n)
    ema3 = _ema_presma(ema2, eff_n)
    return np.sign(ema3.pct_change(1)).fillna(0)


def _kama(close: pd.Series, n: int, fast: int = 2, slow: int = 30) -> pd.Series:
    fr, sr = 2 / (fast + 1), 2 / (slow + 1)
    abs_diff = (close - close.shift(n)).abs()
    peer_diff_sum = (close - close.shift(1)).abs().rolling(n).sum()
    sc = ((abs_diff / peer_diff_sum) * (fr - sr) + sr) ** 2
    sc_arr = sc.to_numpy()
    arr = close.to_numpy()
    m = len(arr)
    result = np.full(m, np.nan)
    if m >= n:
        result[n - 1] = arr[:n].mean()
        for i in range(n, m):
            result[i] = sc_arr[i] * arr[i] + (1 - sc_arr[i]) * result[i - 1]
    return pd.Series(result, index=close.index)


def _kama_signal(close: pd.Series, n: int) -> pd.Series:
    k = _kama(close, n)
    sig = pd.Series(np.where(close > k, 1.0, -1.0), index=close.index)
    return sig.where(k.notna(), 0.0)


def _donchian_signal(close: pd.Series, n: int) -> pd.Series:
    rolling_high, rolling_low = close.rolling(n).max(), close.rolling(n).min()
    signals = np.zeros(len(close))
    state = 0
    c, hi, lo = close.to_numpy(), rolling_high.to_numpy(), rolling_low.to_numpy()
    for i in range(len(close)):
        if np.isnan(hi[i]) or np.isnan(lo[i]):
            continue
        if c[i] >= hi[i]:
            state = 1
        elif c[i] <= lo[i]:
            state = -1
        signals[i] = state
    return pd.Series(signals, index=close.index)


def _bbands(close: pd.Series, n: int):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=1)
    return mid + 2.0 * std, mid, mid - 2.0 * std


def _bb_signal(close: pd.Series, n: int) -> pd.Series:
    upper, mid, lower = _bbands(close, n)
    signals = np.zeros(len(close))
    state = 0
    c, u, m, lo = close.to_numpy(), upper.to_numpy(), mid.to_numpy(), lower.to_numpy()
    for i in range(len(close)):
        if np.isnan(c[i]) or np.isnan(u[i]):
            continue
        if c[i] >= u[i]:
            state = 1
        elif c[i] <= lo[i]:
            state = -1
        elif state == 1 and c[i] <= m[i]:
            state = 0
        elif state == -1 and c[i] >= m[i]:
            state = 0
        signals[i] = state
    return pd.Series(signals, index=close.index)


def _linreg_slope(close: pd.Series, n: int) -> pd.Series:
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.arange(1, n + 1, dtype=float)
    x_sum = 0.5 * n * (n + 1)
    divisor = n * (x_sum * (2 * n + 1) / 3) - x_sum * x_sum
    arr = close.to_numpy()
    out = np.full(len(arr), np.nan)
    if len(arr) >= n:
        windows = sliding_window_view(arr, n)
        out[n - 1:] = (n * (windows @ x) - x_sum * windows.sum(axis=1)) / divisor
    return pd.Series(out, index=close.index)


def _lrs_sign(close: pd.Series, n: int) -> pd.Series:
    return np.sign(_linreg_slope(close, n)).fillna(0)


def _high52w_proximity(close: pd.Series, n: int = 252) -> pd.Series:
    """1.0 = at the rolling n-day high, lower = further below it. Continuous, 0-1."""
    rolling_high = close.rolling(n, min_periods=n // 2).max()
    return (close / rolling_high).fillna(0)


def _zscore(close: pd.Series, n: int) -> pd.Series:
    """(close - SMA) / std -- raw value. Positive IC = trend-following
    (overbought keeps going up), negative IC = mean-reversion (overbought
    reverts). Let the backtest decide which regime this universe is in."""
    sma = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=1)
    return (close - sma) / std


def _short_reversal(close: pd.Series, n: int) -> pd.Series:
    """Raw n-day return. Same sign-agnostic logic as _zscore -- positive IC
    here would mean short-term continuation, negative would mean reversal."""
    return close.pct_change(n)


def _raw_return(close: pd.Series, n: int) -> pd.Series:
    """Raw (non-vol-normalized) n-day return -- base series for
    cross-sectional relative-strength-vs-market, computed at the panel
    level since a single ticker can't see the market average by itself."""
    return close.pct_change(n)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _obv_lrs_sign(close: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    obv = _obv(close, volume)
    return np.sign(_linreg_slope(obv, n)).fillna(0)


def compute_all_signals(close: pd.Series, volume: pd.Series | None = None) -> dict:
    """Returns {signal_name: pd.Series} for every parameter instance of
    every family. Keys are used as-is in the optimization leaderboard.
    volume is optional (backward-compatible with callers that only have
    price) -- OBV-based signals are skipped if not provided."""
    out = {}

    for n in MOMENTUM_WINDOWS:
        out[f"Momentum_{n}"] = _vol_norm_momentum(close, n)

    for s, l in MA_PAIRS:
        out[f"MA_{s}_{l}"] = _ma_cross(close, s, l)

    for s, l in EMA_PAIRS:
        out[f"EMA_{s}_{l}"] = _ema_cross(close, s, l)

    for n in RSI_PERIODS:
        out[f"RSI_{n}"] = _rsi(close, n)

    for n in TRIX_PERIODS:
        out[f"TRIX_{n}"] = _trix_sign(close, n)

    for n in KAMA_PERIODS:
        out[f"KAMA_{n}"] = _kama_signal(close, n)

    for n in DONCHIAN_WINDOWS:
        out[f"Donchian_{n}"] = _donchian_signal(close, n)

    for n in BOLLINGER_WINDOWS:
        out[f"BB_{n}"] = _bb_signal(close, n)

    for n in LRS_WINDOWS:
        out[f"LRS_{n}"] = _lrs_sign(close, n)

    out["High52W"] = _high52w_proximity(close, HIGH52W_WINDOW)

    for n in ZSCORE_WINDOWS:
        out[f"ZScore_{n}"] = _zscore(close, n)

    for n in SHORT_REVERSAL_WINDOWS:
        out[f"ShortRet_{n}"] = _short_reversal(close, n)

    for n in RAW_RET_WINDOWS:
        out[f"RawRet_{n}"] = _raw_return(close, n)

    if volume is not None:
        for n in OBV_LRS_WINDOWS:
            out[f"OBV_LRS_{n}"] = _obv_lrs_sign(close, volume, n)

    return out


DISCRETE_FAMILIES = ("MA_", "EMA_", "TRIX_", "KAMA_", "Donchian_", "BB_", "LRS_", "OBV_LRS_")


def is_discrete(signal_name: str) -> bool:
    return signal_name.startswith(DISCRETE_FAMILIES)
