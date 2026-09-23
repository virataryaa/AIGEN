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


@st.cache_data
def load_ratios_history() -> pd.DataFrame:
    df = pd.read_parquet(DATABASE_DIR / "fundamentals" / "ratios_wide.parquet")
    df["_date"] = pd.to_datetime(df["FilingToDate"], format="mixed", dayfirst=True, errors="coerce")
    return df


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
    st.caption("Bottom-left of the dashed lines = cheaper AND more profitable than the median stock here. Bubble size = Debt/Equity (bigger = more leveraged).")
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
    if len(scatter_df) > 1:
        fig.add_vline(x=scatter_df["PE"].median(), line_dash="dash", line_color="gray")
        fig.add_hline(y=scatter_df["ROE_pct"].median(), line_dash="dash", line_color="gray")
    fig.update_layout(height=550, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 20 — Value + Quality combined rank")
    st.caption("Average of PE rank (cheaper = better) and ROE rank (higher = better), among stocks with 0 < PE < 100.")
    rank_df = f.dropna(subset=["PE", "ROE"]).copy()
    rank_df = rank_df[(rank_df["PE"] > 0) & (rank_df["PE"] < 100)]
    if len(rank_df) > 0:
        rank_df["PE_rank"] = rank_df["PE"].rank(ascending=True)
        rank_df["ROE_rank"] = rank_df["ROE"].rank(ascending=False)
        rank_df["CombinedRank"] = (rank_df["PE_rank"] + rank_df["ROE_rank"]) / 2
        rank_df["ROE_pct"] = (rank_df["ROE"] * 100).round(1)
        rank_df["PE"] = rank_df["PE"].round(1)
        top20 = rank_df.nsmallest(20, "CombinedRank")
        st.dataframe(
            top20[["Ticker", "Company", "Industry", "PE", "ROE_pct", "DebtEquity_Calc"]].rename(
                columns={"ROE_pct": "ROE %"}
            ).round(2),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Company Trend")
    st.caption("How a company's fundamentals have moved over its reported quarters — not just today's snapshot.")
    trend_tickers = sorted(f["Ticker"].unique())
    if trend_tickers:
        selected_ticker = st.selectbox("Select a company", trend_tickers)
        history = load_ratios_history()
        th = history[
            (history["Ticker"] == selected_ticker) & (history["PeriodType"] == "Quarterly")
        ].dropna(subset=["_date"]).sort_values("_date")

        if th.empty:
            st.info("No quarterly history available for this ticker.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig_rev = px.line(th, x="_date", y="Revenue", markers=True,
                                   title="Revenue (quarterly)", labels={"_date": ""})
                st.plotly_chart(fig_rev, use_container_width=True)
            with col2:
                roe_hist = th.dropna(subset=["ROE"]).copy()
                roe_hist["ROE_pct"] = roe_hist["ROE"] * 100
                fig_roe = px.line(roe_hist, x="_date", y="ROE_pct", markers=True,
                                   title="ROE % (quarterly)", labels={"_date": "", "ROE_pct": "ROE %"})
                st.plotly_chart(fig_roe, use_container_width=True)

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
