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

**Takeaway (superseded below):** the composite is usable as a long-screen
input, with `MA_cross_50_200` as the standout individual driver — worth
considering a revised weighting that leans more on MA-cross and less on
RSI/short momentum, next time this is revisited.

### Full grid search — 34 signals, train/test split (`Code/optimize_signals.py`)
Expanded to 8 families (Momentum, MA-cross, EMA-cross, RSI, TRIX, KAMA,
Donchian, Bollinger) x multiple parameters = 34 instances, hand-rolled
(TRIX/KAMA/BB formulas from the sibling CTA project). Split: train
2001-03 to 2017-12 (202 months), test 2018-01 to 2026-06 (107 months) —
only trust what holds up out-of-sample.

Top 10 by out-of-sample (test) hit rate:

| Signal | 1mo Test Hit | 1mo Test IC | 3mo Test Hit | 3mo Test IC |
|---|---|---|---|---|
| Momentum_200 | 73.8% | 0.031 | 75.7% | **0.068** (highest IC) |
| **Momentum_150** | 63.1% | 0.024 | **77.7%** (best hit rate) | 0.057 |
| TRIX_100 | 68.9% | 0.035 | 76.7% | 0.055 |
| TRIX_50 | 68.0% | 0.029 | 75.7% | 0.045 |
| Donchian_100 | 71.8% | 0.020 | 71.8% | 0.036 |
| MA_50_200 | 69.9% | 0.030 | 74.8% | 0.052 |
| EMA_50_150 | 69.9% | 0.030 | 74.8% | 0.051 |
| EMA_50_200 | 68.0% | 0.031 | 74.8% | 0.052 |
| MA_50_150 | 69.9% | 0.028 | 71.8% | 0.040 |
| EMA_20_100 | 66.0% | 0.020 | 68.9% | 0.040 |

**Findings:**
- **Long lookbacks win everywhere.** 150-200 day momentum, 50-200 day
  MA/EMA cross, 50-100 period TRIX, 100-day Donchian all land in the top
  10. Short lookbacks (10-20 day momentum/Donchian/KAMA) are weak-to-failed
  (test hit rate 44-59%, sometimes negative IC) — noise, not signal, at
  this frequency.
- **RSI is useless at every period tested** (7/14/21) — 45-57% test hit
  rate, essentially random.
- **`Momentum_150`/`Momentum_200` are the standout winners**, not
  `MA_cross_50_200` as the earlier (smaller) test suggested — and their
  test-period performance is *higher* than train, which is a good
  robustness sign (not curve-fit to the training window).
- **TRIX (long period) was previously untested and turned out strong** —
  worth keeping now that the zoo was expanded.

**Revised composite candidate** (not yet implemented in
`compute_signals.py`/`build_signals.py` — still using the original 6-signal
version as of this writing):
`0.35*Momentum_150 + 0.35*Momentum_200 + 0.15*TRIX_100 + 0.15*MA_50_200`
— all four robust on both train and test, no short-term/RSI noise.

### NIFTY-500-only re-test + mean-reversion candidates (`Code/optimize_signals_nifty500.py`)
Repeated the grid search restricted to the original 500 (liquid) tickers
only — addresses the concern that the full-982-universe edge might just be
microcap noise/mispricing rather than a real, tradeable effect. Also added
4 new signal families: Linear Regression Slope (trend), 52-week high
proximity (trend), Bollinger z-score and short-term N-day return
(mean-reversion candidates) — 44 signal instances total.

**Result: the effect survives on liquid names, just at slightly smaller
magnitude — this is a good sign, not a weakness.** Best 3-month hit rate
drops from 77.7% (full 982) to 70.9% (NIFTY 500 only), which is expected
(momentum/trend effects are well documented to be stronger in smaller/less
liquid names) but the *pattern holds*: same signal families win, same ones
lose.

| Signal (NIFTY 500 only) | 3mo Test Hit | 3mo Test IC |
|---|---|---|
| TRIX_100 (new #1) | 70.9% | 0.047 |
| **LRS_100** (new addition — strong) | 69.9% | 0.030 |
| Momentum_200 | 67.0% | 0.066 (highest IC) |
| MA_50_200 | 66.0% | 0.046 |
| TRIX_50 | 66.0% | 0.038 |
| LRS_200 | 65.0% | 0.049 |
| High52W (new, moderate) | 62.1% | 0.044 |

**Mean-reversion candidates definitively rejected.** `ZScore_10/20/50`
(Bollinger-style overbought/oversold) and `ShortRet_5/10` (short-term
reversal) all showed negative-to-near-zero IC and failed the robustness
check in both train and test. **This universe/period is trend-following,
not mean-reverting, at the horizons tested (1-3 months)** — don't spend
more time on short-term reversal strategies without a different rationale
(e.g. event-driven, not pure price-based).

**Linear Regression Slope (LRS) is the standout new addition** — robust
across both universes, both horizons, multiple windows (50/100/150/200 all
positive). Worth folding into the production composite alongside
Momentum/TRIX/MA-cross.

### Adding a 1-week horizon — signals are useless short-term
Re-ran the NIFTY-500 grid with a 3rd horizon (5 trading days, ~1 week)
alongside the existing 1-month/3-month. (Caught mistake #8 while doing
this — see below.)

| Signal | 1-Week | 1-Month | 3-Month | Avg |
|---|---|---|---|---|
| **TRIX_100** | 55.3% | 68.0% | 70.9% | **64.7% (best avg)** |
| Momentum_200 | 57.3% | 65.0% | 67.0% | 63.1% |
| EMA_50_200 | 56.3% | 67.0% | 64.1% | 62.5% |
| LRS_100 | 54.4% | 62.1% | 69.9% | 62.1% |
| RSI (7/14/21) | 43-49% | 45-52% | 53-55% | 48-51% |
| ZScore (mean-reversion) | 46-51% | 51-54% | 50-58% | 50-53% |
| Donchian_10 (worst) | 48.5% | 47.6% | 47.6% | 47.9% |

**No signal exceeds 58% hit rate at the 1-week horizon** — every single
one of the 44 signals is weak-to-random short-term. Performance climbs
steadily from 1-week to 1-month to 3-month across the board. **Conclusion:
this signal engine should not be used for short-term/weekly trading
decisions — it only becomes usable at a 1-3 month holding horizon.**
`TRIX_100` remains the most consistent performer across all three
horizons.

### Volume, relative strength, and longer windows (`signal_library.py` v3)
Added 3 more candidate directions per the "what else could we check"
brainstorm: raw (non-vol-normalized) N-day returns as a base for
cross-sectional relative-strength-vs-market, an OBV-based volume signal
(`OBV_LRS_n` — linear regression slope sign of On-Balance-Volume), and
much longer Momentum/MA/EMA windows (300/400/500 days, extending the
"longer lookback wins" pattern). 62 signal instances total, NIFTY 500,
all 3 horizons.

Top result (avg hit rate across 1wk/1mo/3mo):

| Signal | Avg Hit Rate | Note |
|---|---|---|
| RawRet_100 / RelStrength_100 | 64.8% | Tied — see finding #1 below |
| TRIX_100 | 64.7% | Still the most consistent overall performer |
| MA_100_300 (new) | 63.7% | Good new entrant |
| Momentum_200 | 63.1% | |
| Momentum_300 (new) | 62.1% | |
| **OBV_LRS_200 (new, volume-based)** | 61.8% | **First non-price signal to actually work** |
| Momentum_400 / Momentum_500 (new) | 57.6% / 57.9% | **Worse** than 200-300 day versions |

**Finding 1 — relative strength vs market has zero effect on rank-based
metrics, and this is expected, not a bug.** `RelStrength_100` (raw return
minus that date's cross-sectional mean return) produced numbers
*identical* to `RawRet_100` alone. Reason: IC and quintile-spread are both
computed from the **rank order** of the signal within each date —
subtracting the same constant (the market average) from every ticker on
that date shifts all values equally and **cannot change their relative
ranking**. Relative-strength adjustment only matters for something that
cares about the *absolute* return level (e.g. a market-neutral long-short
portfolio's actual P&L), not for a rank-based stock-picking screener like
this one. **Lesson: know what a metric is invariant to before spending
compute testing a variant that can't possibly move it** — this
demeaning transformation was mathematically guaranteed to be a no-op
under IC/quintile-spread, and that could have been reasoned out before
running the backtest, not just after.

**Finding 2 — volume-based signals work.** `OBV_LRS_200` (the slope
direction of On-Balance-Volume over 200 days) reaches 61.8% average hit
rate, competitive with the best price-only signals, and is notably the
best performer at the 3-month horizon among all new additions (69.9%).
First confirmation that volume, not just price, carries usable signal
here — `OBV_LRS_50` (short window) is weak (49.8%), consistent with the
broader "long lookback wins" pattern extending to volume too.

**Finding 3 — the "longer is always better" pattern has a ceiling.**
Momentum/MA/EMA windows beyond ~200-300 days actually *underperform*
the 100-200 day versions (e.g. Momentum_500 at 57.9% vs Momentum_200 at
63.1%). There's a sweet spot, not a monotonic relationship — very long
lookbacks become stale/irrelevant to current price action rather than
more robust.

8. **Forgot to exclude the new horizon's own forward-return column from
   the signal list.** When adding the 5-day horizon, `signal_cols` was
   still hardcoded to exclude only `FwdRet_21`/`FwdRet_63` by name — the
   new `FwdRet_5` column slipped through and got tested as if it were a
   "signal," producing a ~100% hit rate (it was being correlated against
   itself/near-itself). Fix: exclude by prefix
   (`not c.startswith("FwdRet_")`) instead of hardcoding each horizon's
   column name. **Lesson: when a new parameterized column is added to a
   loop-generated panel, don't hardcode exclusion lists by exact name —
   any all-metrics-look-impossible result (see mistake #7) is a prompt to
   check whether a new column silently joined the wrong side of the
   analysis.**

9. **Parallelizing the NSE fundamentals fetch triggered blocking after the
   first burst.** `fetch_fundamentals_parallel.py` (8 threads) got the
   first 8 tickers through cleanly, then every ticker after that came back
   "NO DATA" — 42/50 failed. The sequential version (`fetch_fundamentals.py`,
   one ticker at a time with a 0.2s delay between requests) had zero
   failures. **Lesson: this is the same pattern as the earlier yfinance
   rate-limit mistake — NSE's public API tolerates a slow, steady request
   rate but not a concurrent burst, even though no rate limit is
   documented. Don't parallelize scrape-style fetches against
   undocumented public APIs without testing failure behavior on a small
   batch first — the failure mode is silent (empty response, not an
   HTTP error), so a naive parallel run can look like it's "working
   fast" while actually returning nothing for most tickers.**

10. **Deleted already-fetched raw fundamentals before reprocessing them,
    forcing an unnecessary full network re-fetch.** When adding the
    enrichment step (Industry flag, canonical field mapping, ValueNumeric,
    compression), the 23 tickers that had already downloaded successfully
    were wiped (`rm *.parquet`) before rerunning the fetch script, instead
    of loading their existing raw facts and passing them through
    `enrich_and_optimize()` locally (zero network calls needed). Only the
    27 that had failed actually needed a fresh network fetch. **Lesson:
    when changing a downstream transform/enrichment step, re-run the
    transform on already-fetched raw data first — don't delete and
    re-fetch from the network unless the raw data itself is what changed.
    Network calls are the expensive, rate-limit-risking part; local
    reprocessing is nearly free.**

11. **Older (2018-2021) legacy filings' quarterly facts were silently
    misclassified as "Instant" and dropped from the ratio pivot,** because
    their `contextRef` (e.g. `"OneD"`) wasn't declared as its own
    `<xbrli:context>` element in the file — only its suffixed siblings
    (e.g. `"OneOperatingExpenses01D"`) were declared. This is the exact
    gotcha already flagged in `reference-nse-filings-api` memory before
    this project even started fetching data, and it still got missed on
    first implementation. `classify_duration()` fell back to "Instant"
    whenever context lookup failed, silently dropping ~4 years of history
    for most non-bank companies from the ratio table (discovered because
    the fundamentals backtest's train set had only 150/1116 observations
    before 2022, despite the raw fetched files clearly containing 2018+
    data). Fix: fall back to the ID-prefix convention (`One*` = quarter,
    `Four*` = YTD) when the context can't be resolved, instead of
    defaulting to "Instant". **Lesson: a documented gotcha in a reference
    memory doesn't apply itself — implement the stated fallback the first
    time, and when a downstream analysis shows a suspicious date-range
    skew (most data crammed into a recent window despite fetching "full
    history"), check the parsing logic before trusting the sample.**

12. **The fix for mistake #11 broke genuine instant (balance-sheet) facts
    the same way.** NSE also uses instant-type context IDs starting with
    "One" (e.g. `"OneI"`, resolved fine, start==end==same date). The new
    prefix-fallback branch was reached whenever `days <= 0` fell through
    from the main duration check, not only when the context failed to
    resolve — so a resolved instant context (`OneI`, `days == 0`) matched
    `ctx_ref.startswith("One")` and got wrongly classified as "Quarter"
    instead of "Instant", deleting `TotalEquity`/`Assets`/`Debt` etc.
    entirely from the ratio table (ROE and Debt/Equity coverage silently
    went from 42 tickers to 0). Fix: return "Instant" immediately when
    `days <= 0` on a *resolved* context, before ever reaching the
    prefix-fallback branch (which now only fires when start/end are
    actually `None`, i.e. resolution genuinely failed). **Lesson: a
    "One*"/"Four*" prefix is not unique to duration contexts — instant
    contexts use the same prefix family, so the fallback must be gated on
    "context resolution failed" specifically, not on any zero/short
    duration reaching the fallback code path by accident.**

13. **`NetMargin` had ~28 `+/-inf` values (Revenue=0 in some quarters)
    silently breaking `qcut` in `optimize_signals_nifty500.py`'s
    quintile analysis, producing a fake 100% hit rate** for whichever
    buckets happened to avoid an inf value — the same "impossible-number
    is a bug, not a result" pattern as mistake #7. Fix: `.replace([inf,
    -inf], nan)` before dropping NaNs, in every backtest script that
    consumes a ratio built from a division.

14. **ROE was computed from a single quarter's net profit divided by full
    (annual-scale) equity — understating it by ~4x for every
    `PeriodType == "Quarterly"` row.** Caught by cross-checking against
    yfinance on request (external verification, not something the
    internal backtest would ever catch on its own): TCS showed 12.7% here
    vs yfinance's 47.7%, INFY 9.1% vs 32.0%, HDFCBANK 3.7% vs 13.8% — a
    consistent ~4x gap on every ticker checked, because a quarter's profit
    is ~1/4 of a year's. **This means every fundamentals backtest result
    documented above (the 97-company and NIFTY-100 sector-neutral runs)
    used an ROE that was systematically wrong, not just noisy — those
    results should be re-run, not just re-read, before trusting them.**
    Fix: `build_fundamental_ratios.py` now uses trailing-twelve-month
    (rolling 4-quarter sum) net profit for `PeriodType == "Quarterly"`
    rows; `PeriodType == "Annual"` rows were already correct (a full
    year's profit) and are untouched. Re-verified post-fix: TCS 45.7% vs
    47.7%, INFY 30.5% vs 32.0%, HDFCBANk 11.6% vs 13.8% — all within a few
    points now. **Lesson: a ratio that divides a period-flow number
    (profit, revenue) by a point-in-time stock number (equity, assets)
    needs the flow side annualized/TTM'd before comparing across
    different reporting frequencies — this class of bug won't show up as
    a crash or a NaN, just a quietly-wrong number that still looks
    plausible in isolation. External cross-checking (a different data
    provider, not just internal consistency checks) is what caught this,
    and is worth doing again before trusting other computed ratios
    (Debt/Equity, margins) at face value.**

**Known caveat: survivorship bias.** The universe is today's active NSE
list — any company that delisted/went bankrupt between 2000-2026 is
invisible to this backtest, which will make historical performance look
better than a real-time strategy would have achieved.

## Fundamentals pipeline & backtest

Separate from the OHLCV/technical pipeline above — this is the actual
"value investing" side of the project (PE/PB/ROE/debt), started because
the technical signal engine has no way to judge whether a stock is
*cheap*, only whether it's *trending*.

### Data source: NSE XBRL (free, no subscription needed)
Considered paying for screener.in (bulk CSV export) but tested NSE's own
public XBRL filings first — same free, unauthenticated pattern as the
price data. **Verdict: don't pay for screener.in** — NSE XBRL covers every
listed company, is the authoritative source (not a scraped/re-derived
one), and matches this project's own principle of never letting an LLM
touch a number that Python can extract directly from a primary source.
- Two endpoints needed for full history: legacy `corporates-financial-results`
  (pre-2025, frozen Dec-2024) + `integrated-filing-results` (2025+, SEBI's
  new regime) — same filing, same numbers, different plumbing depending on
  era.
- **Quarterly filings**: P&L only (Revenue, Profit, EPS, segment data — 95
  distinct tagged fields in a sample filing).
- **Annual (March) filings**: full P&L + Balance Sheet + Cash Flow (257
  fields) — Balance Sheet/Cash Flow items don't exist in the quarterly
  filings at all.
- **History depth: 2018 onward only** (XBRL mandate start), not the full
  2000+ price history. ~32 quarters per company at most.
- **Sector taxonomy differs**: banks/NBFCs use different tags entirely
  (`InterestEarned`/`InterestExpended` instead of `RevenueFromOperations`/
  `FinanceCosts`) — confirmed only 102/261 fields in common between
  RELIANCE and KOTAKBANK. Handled via `Database/fundamentals/field_mapping.csv`,
  a canonical-name lookup (`raw_tag -> canonical_name`) so cross-company
  ratio computation doesn't silently break on sector-specific tag names.

### Pipeline (`Code/fetch_fundamentals.py` + universe-specific wrappers)
One long-format parquet per ticker in **`Database/fundamentals/`** (kept
separate from `Database/prices/` on purpose — different refresh cadence,
different schema). Each row: `Ticker, PeriodType, Consolidated,
FilingToDate, Source, SourceFile, FieldName, Value, ContextRef,
ContextStart, ContextEnd, DurationClass, Industry, CanonicalName,
Statement, ValueNumeric`. Compressed with `brotli` + categorical dtypes
(~40% smaller than default snappy+string storage, tested and confirmed).

Universe built incrementally, one exchange-defined index tier at a time
(mirrors how the technical signal universe was built up from NIFTY 500):
| Tier | Source | Count | Script |
|---|---|---|---|
| Ranks 1-50 | NIFTY 50 | 47/50 fetched | `fetch_fundamentals.py` |
| Ranks 51-100 | NIFTY Next 50 | 50/50 fetched | `fetch_fundamentals_next50.py` |
| Ranks 101-170 | NIFTY 200 remainder | 66/70 fetched | `fetch_fundamentals_next70.py` |

| Ranks 171-270ish | NIFTY 500 remainder (batch 3) | 95/100 fetched | `fetch_fundamentals_next100.py` |
| Ranks ~271-370 | NIFTY 500 remainder (batch 4) | 84/100 fetched | `fetch_fundamentals_next100b.py` |
| Final remainder | NIFTY 500 completion | 138/159 fetched | `fetch_fundamentals_remaining.py` |

**COMPLETE as of this writing: 480/500 NIFTY 500 companies**, 30 MB
storage, 1,067,909+ facts fetched in the final batch alone. The ~20 total
failures across all batches return "0 filings found" — a symbol-lookup
issue on NSE's side, not a rate-limit or parsing problem. Two recurring
names worth a closer look eventually: **`M&M` and `M&MFIN` failed in
every batch attempted** — likely the `&` character isn't being
URL-encoded correctly in the API request (not yet fixed, low priority — 2
tickers out of 500). **Ratios and backtest not yet re-run on the full
480** — data build and validation are being kept as separate steps on
request; this is the natural next step now that the universe is
essentially complete.

### Spot-check validation (before continuing to scale further)
Randomly sampled companies across all 4 batches to check for silent data
issues before trusting the growing dataset:
- **TITAN Q4 FY24: Revenue ₹12,494 cr, Net Profit ₹771 cr** — matches
  real-world public figures. Raw fetch/parse pipeline confirmed correct,
  no new bugs found in this check.
- **Found a genuine `field_mapping.csv` gap (not a fetch bug): banks use
  a different Ind-AS schedule for both Equity and Net Profit.** Equity is
  reported as `Capital` + `ReservesAndSurplus` (not a single `Equity`
  tag), and net profit as `ProfitLossForThePeriod` (note "ForThe", not
  "For") instead of `ProfitLossForPeriod`. Neither was in the mapping
  table, so bank ROE/Debt-Equity were silently more incomplete than they
  needed to be. Fixed: added both variants to `field_mapping.csv`, plus a
  fallback in `build_fundamental_ratios.py` that sums `BankCapital +
  BankReserves` when the general-taxonomy equity tags are absent. **Not
  yet re-verified end-to-end** (ratios haven't been rebuilt since this
  fix) — do that before trusting bank-sector ratios specifically.

Fetch speed: **sequential only, ~14-18 sec/ticker** (~15 min per 50-70
company batch). Never parallelize this — see mistake #9.

### Ratio computation (`Code/build_fundamental_ratios.py`)
Pivots the long-format facts to one row per (Ticker, PeriodType,
FilingToDate) using `CanonicalName`, then computes: `ROE`, `DebtEquity_Calc`,
`NetMargin`, `InterestCoverage`, `FCF` (annual only, since cash flow is
annual-only), `Revenue_YoY`, `NetProfit_YoY`. Output: `Database/fundamentals/ratios_wide.parquet`.

### Backtest (`Code/backtest_fundamentals.py`, `Code/backtest_fundamentals_sector.py`)
Same philosophy as the technical backtest (cross-sectional rank vs forward
return, train/test split) but adapted for quarterly-not-daily data:
- **Reporting lag applied** (45 days quarterly / 60 days annual added to
  the period-end date) to avoid lookahead bias — the market can't react to
  a quarter's numbers before the company actually files them.
- Rebalanced quarterly (not monthly, since fundamentals don't update
  faster than that), horizons tested: 63 and 126 trading days (~3mo, ~6mo).

**⚠ STALE — computed before mistake #14 (ROE understated ~4x) was fixed.**
DebtEquity_Calc/NetMargin/etc. below are unaffected by that specific bug,
but ROE's row is not to be trusted, and the whole backtest hasn't been
re-run since. Treat this table as history of what was tried, not a
current finding — re-run `backtest_fundamentals.py` before drawing any
conclusion from it.

**Result on 97 companies (NIFTY 100): inconclusive, mostly weak/negative.**

| Ratio | Test IC (3mo) | Test IC (6mo) |
|---|---|---|
| ROE | +0.02 | +0.03 (mildly promising, only positive one both horizons) |
| DebtEquity_Calc | +0.02 | -0.003 (inconsistent) |
| NetMargin | -0.02 | -0.004 |
| InterestCoverage | -0.02 | -0.03 |
| Revenue_YoY | -0.01 | -0.04 |
| NetProfit_YoY | -0.03 | -0.03 |
| FCF | -0.004 | -0.07 (thin sample, only ~130-140 obs) |

For comparison, the technical signals' best IC was 0.05-0.07 — fundamentals
are meaningfully weaker on every ratio tested so far. **Sector-neutral
ranking (long top-half/short bottom-half within the same industry+quarter)
did not improve this** — mostly similar or slightly worse (e.g.
DebtEquity_Calc IC went from +0.02 to -0.10), likely because with ~97
companies across 17 industries, most sector-quarter groups only have
4-6 names, making a median-split extremely noisy (confirmed by absurd
spread magnitudes in the raw output, a small-N artifact, not a real
effect).

**Working hypothesis, not yet proven: this is a sample-size problem, not
a "fundamentals don't work" problem.** NIFTY 100 is already "the best of
the best" — there's much less quality/value dispersion to detect a signal
from than in a 500+ stock universe. Scaling the universe (in progress,
ranks 101-170 being fetched) is the test of this hypothesis, not a random
next step.

### Time-series / other ideas discussed, not yet built
- **Fundamentals *trend* (YoY change, margin expansion, earnings surprise
  vs own history) instead of static levels** — theoretically stronger for
  large-caps where the current quality *level* is already priced in, but
  a *change* might not be. Highest-priority idea to try once the universe
  is large enough to test anything reliably.
- Combining technical + fundamental signals (a stock trending AND
  improving fundamentally) — classic quality+momentum combo, likely more
  robust than either alone.
- TTM (trailing 4-quarter average) ratios instead of single-quarter
  snapshots, to smooth one-off noise.

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
- **Fundamentals: scale to 170+ companies (in progress) and re-run
  `backtest_fundamentals.py`** to test whether the current weak/negative
  results are a sample-size artifact or a real finding. Don't draw
  conclusions from the 97-company result yet.
- Once fundamentals show a real signal (or don't, at scale) — try the
  time-series/trend versions (YoY change, margin expansion) before giving
  up on the fundamentals angle entirely.
- `Automator/` not yet set up for scheduled daily refresh
- No `.NS` ticker de-duplication check yet if a symbol changes over time
