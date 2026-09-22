# Aigen Vector

Indian equities database + signal screener. Daily OHLCV for ~982 NSE
tickers, a simplified trend-following composite score, and a Streamlit
dashboard. This README is a running log — updated as we build, break, and
fix things — so future work doesn't repeat past mistakes.

GitHub: https://github.com/virataryaa/AIGEN

## Structure
- `Code/` — all ingest, signal, and backtest scripts (heavy compute lives here, never in the dashboard)
- `Database/` — parquet data (per-ticker prices, universe, signals)
- `Dashboard/` — Streamlit app (`app.py`) + standalone static HTML report (`signal_screener.html`)
- `Automator/` — (not yet set up — scheduled refresh TBD)

## Data pipeline

### Universe construction
No single NSE source ranks the full market by size, so the universe was
built in two tiers:
1. **NIFTY 500** (`ind_nifty500list.csv`, NSE archives) — 500 tickers, comes with an `Industry` tag.
2. **Next 500**: NSE's own broad-market coverage stops at **NIFTY Total Market (top 750)** = NIFTY 500 + **NIFTY Microcap 250**. Beyond that NSE publishes no ranking at all. So:
   - 254 tickers from NIFTY Microcap 250 (NSE-ranked, has Industry tag)
   - 246 tickers ranked by market cap (via yfinance `fast_info`) from the remaining ~1,573 EQ tickers not in any NSE broad index — industry fetched via `yfinance.get_info()` per ticker (slower, only needed once)

Result: **982 tickers actually ingested** (out of 1,001 attempted — some failed/empty, mostly very recent listings with <300 days of history).

### OHLCV ingest (`Code/ingest_ohlcv.py`, `Code/add_next500.py`, `Code/retry_next500_ohlcv.py`)
- One parquet per ticker in `Database/prices/<TICKER>.NS.parquet` — chosen over one giant file because incremental updates only need to touch the one changed ticker's file, not rewrite everything.
- Columns: `Ticker, Date, Open, High, Low, Close, Adj_Close, Volume, Dividends, Stock_Splits` — raw (unadjusted) OHLC kept alongside `Adj_Close`, so adjustments stay correct even as new splits happen later.
- `period='max'` / `start='2000-01-01'` — most tickers' real history starts wherever Yahoo's own NSE coverage begins (~2002-07-01 for many, regardless of actual listing date) or their IPO date, whichever is later.
- Full 500-ticker NIFTY 500 batch: **114 sec**. Next-500 batch: **795 sec** (13 sec/ticker slower — see rate-limiting mistake below).

### Universe reference (`Database/universe.parquet`)
`Company Name, Industry, Symbol, Series, ISIN Code, Ticker` — kept **separate** from price data on purpose, so industry/classification updates don't touch price files and vice versa. Joined on `Ticker` whenever needed.

## Signal engine (`Code/compute_signals.py`, `Code/build_signals.py`)

Adapted from a sibling project (`LSEG-CTA`, a commodities trend-following
dashboard) that runs a 144-indicator composite score. Simplified for
equities:
- 144 indicators → **6**: vol-normalized momentum (20d, 100d, `tanh(ret / (vol·√n))` formula), 50/200 MA cross, 20-day Donchian breakout (stateful), RSI-14 (reported separately, not weighted into the composite — it's mean-reversion, not trend)
- 3 time-buckets (ST/MT/LT) → **2**, since most tickers don't have 400+ days of history
- **Same fixed parameters for every ticker** — no per-stock tuning (this part *is* carried over unchanged from CTA, since it already worked that way there)
- Hand-rolled, no `pandas_ta` — see mistake log below
- All computation runs locally (`Code/build_signals.py`): **19 sec for 962 scored tickers** (20 skipped for insufficient history), output is a 58 KB `Database/signals.parquet`. Dashboard never recomputes.

This is a **technical/momentum signal, not a value-investing signal** —
with only OHLCV data there's no way to judge intrinsic value (no PE, ROE,
debt data yet). Worth remembering before treating screener output as
"cheap stocks" rather than "trending stocks."

## Backtest (`Code/backtest_signals.py`)
Standard cross-sectional factor test: monthly rebalance, rank all tickers
by composite score (computed causally — only data up to that date), split
into quintiles, measure forward returns (21d / 63d), check whether Q1
(highest score) beats Q5 (lowest) and compute the Information Coefficient
(Spearman rank correlation, score vs forward return).

**CORRECTED (see mistake #6 below) — the first version of this backtest
reported the opposite of the true result due to a labeling bug. The
composite signal actually works: positive, statistically significant.**

Result (982 tickers, 305 monthly rebalances, 2001-03 to 2026-06):

| Metric | 1-month horizon | 3-month horizon |
|---|---|---|
| Top quintile avg fwd return | 3.48% | 12.69% |
| Bottom quintile avg fwd return | 2.23% | 7.27% |
| Top − Bottom spread | +1.25% (t=3.06, p=0.0024) | **+5.41%** (t=3.38, p=0.0008) |
| Hit rate (top > bottom) | 60.0% | 68.2% |
| Mean IC (Spearman) | +0.016 | +0.039 |

### Component decomposition (`Code/backtest_components.py`)
Tested each of the 6 signals standalone to see which drive the composite's
performance:

| Signal | IC (63d) | Spread t-stat (63d) | Note |
|---|---|---|---|
| **MA_cross_50_200** | **+0.048** | **7.43** | Strongest, most consistent single signal — simple golden-cross beats everything else standalone |
| Momentum_100 | +0.044 | 5.65 | Second strongest |
| Full Composite | +0.039 | 5.04 | |
| Donchian_20 | +0.015 | 3.42 | Contributes positively — dropping it *weakens* the composite (tested, see below) |
| Momentum_20 | +0.014 | 2.02 | Weak at 1-month, only useful at 3-month |
| RSI_14 | +0.010 | 2.12 | Weakest, near-zero/insignificant at 1-month |

`Composite_NoDonchian` variant (momentum + MA-cross only, reweighted)
scored *lower* than the full composite (63d spread 4.99% vs 5.41%) — the
original hypothesis that Donchian breakout was hurting the signal
(momentum-crash theory) was **wrong**. Signal works better at the 3-month
horizon than 1-month across the board — makes sense for a trend-following
approach, which needs time to play out.

**Takeaway:** the composite is usable as a long-screen input, with
`MA_cross_50_200` as the standout individual driver — worth considering a
revised weighting that leans more on MA-cross and less on RSI/short
momentum, next time this is revisited.

**Known caveat: survivorship bias.** The universe is today's active NSE
list — any company that delisted/went bankrupt between 2000-2026 is
invisible to this backtest, which will make historical performance look
better than a real-time strategy would have achieved.

## Mistakes made + fixes (read before repeating them)

1. **Committed derived combined files to git — hit GitHub's size limit.**
   `all_prices.parquet` (100 MB) and `returns_matrix.parquet` (31 MB) were
   built once and committed. GitHub warned `all_prices.parquet` was already
   at the 100 MB hard limit. Fix: stopped committing them — they're 100%
   reproducible from the per-ticker files, so the dashboard now builds both
   in-memory via DuckDB (`read_parquet('Database/prices/*.parquet')`),
   cached with `@st.cache_data`. **Lesson: never commit a file whose only
   purpose is "convenient to have precomputed" — regenerate it instead.**

2. **Yahoo rate-limited a rapid-fire ingest run.** Running `.get_info()`
   (heavy per-ticker call) for 246 tickers immediately followed by 500
   `.history()` calls with no delay triggered "Too Many Requests" on
   every single one (0 OK / 500 errors). Fix: added a 1-second delay +
   exponential backoff between requests on retry. **Lesson: back-to-back
   heavy yfinance calls across hundreds of tickers need throttling, even
   though the API has no documented rate limit.**

3. **`pandas_ta` breaks Streamlit Cloud deploys.** Learned this from the
   sibling `LSEG-CTA` project before it bit us here: `pandas_ta`
   unconditionally imports `numba`, which has no published wheel for some
   Python versions Streamlit Cloud runs — the deploy crashes the moment
   `pandas_ta` is imported. Fix: `Code/compute_signals.py` hand-rolls every
   indicator (momentum, MA cross, Donchian, RSI) in plain pandas/numpy,
   no `pandas_ta` anywhere. **Lesson: don't add `pandas_ta` to
   `requirements.txt` for anything that touches the deployed dashboard.**

4. **Correlation heatmap crashed on `reset_index()`.** `returns_matrix`'s
   columns are named `Ticker`, so `.corr()` produces a matrix whose index
   *and* columns are both named `Ticker`. Stacking that and calling
   `.reset_index()` makes pandas try to create two columns with the same
   name → `ValueError: cannot insert 0, already exists` (surfaced on
   Streamlit Cloud's newer pandas/Python 3.14, not caught locally at the
   time). Fix: explicitly rename the axes to `Ticker A`/`Ticker B` before
   stacking. **Lesson: when both axes of a pivoted DataFrame share a name,
   `stack()` + `reset_index()` will collide — rename first.**
   (Since fixed — this whole tab was later removed from the dashboard
   anyway, keeping only the Signal Screener.)

6. **Misread `pd.qcut` label ordering — reported an inverted backtest
   result that was actually correct.** `pd.qcut(x, 5, labels=[1,2,3,4,5])`
   assigns label **1 to the lowest-value bin and label 5 to the
   highest-value bin** (ascending, always — verified directly). The first
   backtest script (`backtest_signals.py`) assumed "Q1 = highest composite
   score" without checking this, so `Q1 - Q5` was actually computing
   *(lowest score) - (highest score)* — a negative number that looked like
   "the signal is inverted and harmful," when the real relationship was
   positive and significant the whole time. Caught by writing a second,
   independent script (`backtest_components.py`) that took the top/bottom
   groups explicitly by `q_df.columns.max()/min()` instead of trusting the
   label numbers, and getting the opposite sign. **Lesson: never trust
   `qcut`/`cut` label numbers to mean "highest"/"lowest" — always derive
   top/bottom from the actual bin values, and when a backtest result looks
   dramatically wrong (a working signal being *actively* harmful, not just
   noisy), suspect the harness before the signal.**

7. **Discrete-signal spread calc silently produced a fake 100% hit rate.**
   `optimize_signals.py`'s first version computed the discrete-signal
   spread as `means.max() - means.min()` (max and min of the group-mean
   *returns*), which is mathematically always ≥ 0 for any signal with 2+
   states — so every discrete signal (MA-cross, EMA-cross, Donchian, BB,
   TRIX, KAMA — 24 of the 34 tested) showed a suspicious, impossible 100%
   hit rate in both train and test. Fix: spread must be directional —
   mean return of the **highest signal state value** (e.g. +1) minus mean
   return of the **lowest state value** (e.g. -1), using `means.index.max()
   /min()` (the state values), not `means.max()/min()` (the return
   values). **Lesson: any backtest metric that comes back at a suspiciously
   perfect number (100%, 0%, exactly 1.0) is almost always a harness bug,
   not a real result — check the metric's math before trusting the
   number, same as mistake #6.**

5. **Streamlit Cloud served a stale build after a push.** After fixing
   mistake #1, the deployed app still threw `FileNotFoundError` for
   `signals.parquet` even though it was confirmed present on GitHub via
   the API — the app just hadn't picked up the latest commit yet. Fix:
   manually reboot the app from "Manage app". **Lesson: confirm the file
   is actually in the latest commit via the GitHub API before debugging
   the app code — the deploy lag is a separate, simpler problem.**

## Current dashboard scope
Deliberately cut down to **Signal Screener only** (Universe Overview,
Price Explorer, and Correlation Explorer were removed on request — they
still exist in git history if needed again). A standalone static
`Dashboard/signal_screener.html` (Plotly, no server needed) is generated
by `Code/generate_static_report.py` for quick local viewing without
running Streamlit at all.

## Open items / next steps
- Composite signal validated (positive, significant) — consider
  reweighting to lean more on `MA_cross_50_200` (strongest standalone
  driver) and less on `RSI_14`/`Momentum_20` (weakest), per the component
  backtest above.
- Decide whether to add fundamentals (PE/PB/ROE/debt) — current screener
  is technical-only, not true value investing
- `Automator/` not yet set up for scheduled daily refresh
- No `.NS` ticker de-duplication check yet if a symbol changes over time
