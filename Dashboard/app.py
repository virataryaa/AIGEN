from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "Database"
PRICES_GLOB = str(DATABASE_DIR / "prices" / "*.parquet")

st.set_page_config(page_title="Aigen Vector", layout="wide")


@st.cache_data
def load_universe() -> pd.DataFrame:
    return pd.read_parquet(DATABASE_DIR / "universe.parquet")


@st.cache_data
def load_all_prices() -> pd.DataFrame:
    """Built in-memory from the per-ticker parquet files (not stored on disk,
    to avoid committing large derived files to git)."""
    con = duckdb.connect()
    return con.execute(f"""
        SELECT Ticker, Date, Open, High, Low, Close, Adj_Close, Volume,
               Dividends, Stock_Splits
        FROM read_parquet('{PRICES_GLOB}')
        ORDER BY Ticker, Date
    """).fetchdf()


@st.cache_data
def load_returns_matrix() -> pd.DataFrame:
    """Daily returns pivoted wide (Date x Ticker), built in-memory via DuckDB."""
    con = duckdb.connect()
    returns_df = con.execute(f"""
        SELECT Ticker, Date,
               Adj_Close / LAG(Adj_Close) OVER (PARTITION BY Ticker ORDER BY Date) - 1 AS ret
        FROM read_parquet('{PRICES_GLOB}')
    """).fetchdf()
    wide = returns_df.pivot(index="Date", columns="Ticker", values="ret")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


universe = load_universe()
all_prices = load_all_prices()
returns_matrix = load_returns_matrix()

st.title("Aigen Vector")
st.caption("Indian equities database — daily OHLCV, NSE universe, correlation explorer")

tab_overview, tab_price, tab_corr = st.tabs(
    ["Universe Overview", "Price Explorer", "Correlation Explorer"]
)

with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Tickers tracked", f"{universe['Ticker'].nunique():,}")
    col2.metric("Industries", f"{universe['Industry'].nunique():,}")
    col3.metric("Total price rows", f"{len(all_prices):,}")

    industry_counts = (
        universe.groupby("Industry")["Ticker"].nunique().sort_values(ascending=False)
    )
    fig = px.bar(
        industry_counts,
        orientation="h",
        labels={"value": "Number of tickers", "Industry": ""},
        title="Tickers by Industry",
    )
    fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Universe table")
    industry_filter = st.multiselect(
        "Filter by industry", sorted(universe["Industry"].dropna().unique())
    )
    display_df = universe
    if industry_filter:
        display_df = universe[universe["Industry"].isin(industry_filter)]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab_price:
    tickers = sorted(universe["Ticker"].unique())
    default_idx = tickers.index("RELIANCE.NS") if "RELIANCE.NS" in tickers else 0
    selected = st.selectbox("Select a ticker", tickers, index=default_idx)

    ticker_prices = all_prices[all_prices["Ticker"] == selected].sort_values("Date")

    if ticker_prices.empty:
        st.warning("No price data for this ticker.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=ticker_prices["Date"],
                open=ticker_prices["Open"],
                high=ticker_prices["High"],
                low=ticker_prices["Low"],
                close=ticker_prices["Close"],
                name=selected,
            )
        )
        fig.update_layout(
            title=f"{selected} — Daily Price History",
            xaxis_rangeslider_visible=False,
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        vol_fig = px.bar(ticker_prices, x="Date", y="Volume", title="Volume")
        st.plotly_chart(vol_fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Close", f"{ticker_prices['Close'].iloc[-1]:,.2f}")
        col2.metric("52W High", f"{ticker_prices['Close'].tail(252).max():,.2f}")
        col3.metric("52W Low", f"{ticker_prices['Close'].tail(252).min():,.2f}")
        col4.metric("Data since", str(ticker_prices["Date"].min().date()))

with tab_corr:
    st.write(
        "Pick an industry (correlation among peers) or hand-pick tickers "
        "for a custom correlation heatmap, computed from daily returns."
    )

    mode = st.radio("Mode", ["By Industry", "Custom Selection"], horizontal=True)

    if mode == "By Industry":
        industry = st.selectbox(
            "Industry", sorted(universe["Industry"].dropna().unique())
        )
        peer_tickers = universe.loc[
            universe["Industry"] == industry, "Ticker"
        ].tolist()
    else:
        peer_tickers = st.multiselect(
            "Select tickers (recommended: under 40 for a readable heatmap)",
            sorted(universe["Ticker"].unique()),
            default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"],
        )

    available = [t for t in peer_tickers if t in returns_matrix.columns]

    if len(available) < 2:
        st.info("Select at least 2 tickers with available data.")
    else:
        corr = returns_matrix[available].corr()
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title=f"Correlation Heatmap ({len(available)} tickers, daily returns)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Most correlated pairs")
        corr_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
        )
        st.dataframe(
            corr_pairs.head(15).reset_index().rename(
                columns={"level_0": "Ticker A", "level_1": "Ticker B", 0: "Correlation"}
            ),
            hide_index=True,
        )
