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
*(Results to be filled in once the run completes — see conversation for latest numbers.)*

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
- Fill in backtest results once the run completes
- Decide whether to add fundamentals (PE/PB/ROE/debt) — current screener
  is technical-only, not true value investing
- `Automator/` not yet set up for scheduled daily refresh
- No `.NS` ticker de-duplication check yet if a symbol changes over time
