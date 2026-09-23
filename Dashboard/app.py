from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "Database"

st.set_page_config(page_title="Aigen Vector", layout="wide")


@st.cache_data
def load_universe() -> pd.DataFrame:
    return pd.read_parquet(DATABASE_DIR / "universe.parquet")


@st.cache_data
def load_signals() -> pd.DataFrame:
    signals = pd.read_parquet(DATABASE_DIR / "signals.parquet")
    universe = load_universe()
    return signals.merge(
        universe[["Ticker", "Company Name", "Industry"]], on="Ticker", how="left"
    )


@st.cache_data
def load_valuation() -> pd.DataFrame:
    return pd.read_parquet(DATABASE_DIR / "fundamentals" / "valuation_snapshot.parquet")


st.title("Aigen Vector")

tab_fundamentals, tab_signals = st.tabs(["Fundamentals", "Signal Screener"])

with tab_fundamentals:
    val = load_valuation()

    st.caption(
        "Latest quarterly fundamentals combined with current price — PE/PB "
        "computed from TTM EPS and reported book value. NIFTY 500 universe, "
        "480 companies. History from 2018 (XBRL mandate start)."
    )

    col1, col2, col3 = st.columns(3)
    industry_filter = col1.multiselect(
        "Industry", sorted(val["Industry"].dropna().unique())
    )
    pe_range = col2.slider("PE range", 0.0, 100.0, (0.0, 100.0), step=1.0)
    roe_range = col3.slider("ROE range (%)", -50.0, 50.0, (-50.0, 50.0), step=1.0)

    f = val.copy()
    if industry_filter:
        f = f[f["Industry"].isin(industry_filter)]
    f = f[
        (f["PE"].between(*pe_range) | f["PE"].isna())
        & (f["ROE"] * 100 >= roe_range[0]) & (f["ROE"] * 100 <= roe_range[1])
    ]

    display_cols = ["Ticker", "Company", "Industry", "LatestPrice", "PE", "PB",
                     "ROE", "DebtEquity_Calc", "NetMargin", "Revenue_YoY",
                     "NetProfit_YoY", "LatestFilingDate"]
    display_df = f[display_cols].copy()
    for pct_col in ["ROE", "NetMargin", "Revenue_YoY", "NetProfit_YoY"]:
        display_df[pct_col] = (display_df[pct_col] * 100).round(1)
    display_df[["PE", "PB", "DebtEquity_Calc", "LatestPrice"]] = display_df[
        ["PE", "PB", "DebtEquity_Calc", "LatestPrice"]
    ].round(2)

    st.dataframe(
        display_df.sort_values("PE"),
        use_container_width=True,
        hide_index=True,
        column_config={
            "ROE": st.column_config.NumberColumn("ROE %"),
            "NetMargin": st.column_config.NumberColumn("Net Margin %"),
            "Revenue_YoY": st.column_config.NumberColumn("Revenue YoY %"),
            "NetProfit_YoY": st.column_config.NumberColumn("Net Profit YoY %"),
            "DebtEquity_Calc": st.column_config.NumberColumn("Debt/Equity"),
        },
    )

    st.subheader("Value vs Quality")
    st.caption("Bottom-left = cheap AND profitable. Bubble size = Debt/Equity (bigger = more leveraged).")
    scatter_df = f.dropna(subset=["PE", "ROE"]).copy()
    scatter_df = scatter_df[(scatter_df["PE"] > 0) & (scatter_df["PE"] < 100)]
    scatter_df["ROE_pct"] = scatter_df["ROE"] * 100
    scatter_df["DebtEquity_abs"] = scatter_df["DebtEquity_Calc"].abs().fillna(0).clip(upper=5)
    fig = px.scatter(
        scatter_df, x="PE", y="ROE_pct", color="Industry",
        size="DebtEquity_abs", size_max=20,
        hover_data=["Ticker", "Company"],
        labels={"ROE_pct": "ROE %", "PE": "PE ratio"},
    )
    fig.update_layout(height=550, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab_signals:
    signals = load_signals()

    st.caption(
        "Simplified trend-following composite score, computed locally "
        "(momentum + MA-cross + Donchian breakout). +1 = strongly bullish, "
        "-1 = strongly bearish. Not a fundamental/value signal — purely "
        "price/volume based."
    )
    st.caption(f"Signals as of {pd.to_datetime(signals['Date']).max().date()} "
               f"— {len(signals)} tickers scored")

    col1, col2 = st.columns(2)
    industry_filter2 = col1.multiselect(
        "Filter by industry", sorted(signals["Industry"].dropna().unique()), key="sig_industry"
    )
    min_composite, max_composite = col2.slider(
        "Composite score range", -1.0, 1.0, (-1.0, 1.0), step=0.05
    )

    filtered = signals[
        (signals["Composite"] >= min_composite) & (signals["Composite"] <= max_composite)
    ]
    if industry_filter2:
        filtered = filtered[filtered["Industry"].isin(industry_filter2)]

    display_cols2 = [
        "Ticker", "Company Name", "Industry", "Close", "Composite",
        "Momentum_20", "Momentum_100", "MA_cross_50_200", "Donchian_20", "RSI_14",
    ]
    st.dataframe(
        filtered[display_cols2].sort_values("Composite", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 15 — most bullish")
        top15 = filtered.nlargest(15, "Composite")
        fig = px.bar(top15, x="Composite", y="Ticker", orientation="h",
                     color="Composite", color_continuous_scale="RdYlGn")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Top 15 — most bearish")
        bottom15 = filtered.nsmallest(15, "Composite")
        fig = px.bar(bottom15, x="Composite", y="Ticker", orientation="h",
                     color="Composite", color_continuous_scale="RdYlGn")
        fig.update_layout(yaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig, use_container_width=True)
