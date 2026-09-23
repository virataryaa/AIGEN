"""
Generate a standalone, double-clickable Plotly HTML report of the signal
screener — no Streamlit server needed, just open the file in a browser.

Reads Database/signals.parquet + Database/universe.parquet (both built
locally beforehand by build_signals.py / the ingest scripts) and writes
Dashboard/signal_screener.html.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "Database"
OUTPUT_FILE = BASE_DIR / "Dashboard" / "signal_screener.html"


def load_signals() -> pd.DataFrame:
    signals = pd.read_parquet(DATABASE_DIR / "signals.parquet")
    universe = pd.read_parquet(DATABASE_DIR / "universe.parquet")
    return signals.merge(
        universe[["Ticker", "Company Name", "Industry"]], on="Ticker", how="left"
    ).sort_values("Composite", ascending=False)


def build_table_figure(df: pd.DataFrame) -> go.Figure:
    cols = ["Ticker", "Company Name", "Industry", "Close", "Composite",
            "Momentum_20", "Momentum_100", "MA_cross_50_200", "Donchian_20", "RSI_14"]
    display_df = df[cols].round(3)

    fill_colors = [
        ["#e8f5e9" if v > 0.3 else "#ffebee" if v < -0.3 else "white" for v in display_df["Composite"]]
        for _ in cols
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(values=cols, fill_color="#37474f", font=dict(color="white"), align="left"),
        cells=dict(values=[display_df[c] for c in cols], fill_color=fill_colors, align="left"),
    )])
    fig.update_layout(title="All Tickers — Composite Signal Score", height=min(400 + len(df) * 3, 1400))
    return fig


def build_bar_figure(df: pd.DataFrame, title: str, ascending_order: bool) -> go.Figure:
    fig = px.bar(df, x="Composite", y="Ticker", orientation="h",
                 color="Composite", color_continuous_scale="RdYlGn", title=title)
    fig.update_layout(yaxis={"categoryorder": "total ascending" if ascending_order else "total descending"})
    return fig


def main():
    signals = load_signals()

    table_fig = build_table_figure(signals)
    top15_fig = build_bar_figure(signals.nlargest(15, "Composite"), "Top 15 — Most Bullish", True)
    bottom15_fig = build_bar_figure(signals.nsmallest(15, "Composite"), "Top 15 — Most Bearish", False)

    as_of = pd.to_datetime(signals["Date"]).max().date()

    html_parts = [
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Aigen Vector - Signal Screener</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 24px; background: #fafafa; }}
  h1 {{ margin-bottom: 4px; }}
  .caption {{ color: #555; margin-bottom: 24px; }}
  .chart-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .chart-row > div {{ flex: 1; min-width: 400px; }}
</style>
</head><body>
<h1>Aigen Vector &mdash; Signal Screener</h1>
<p class="caption">
  Simplified trend-following composite score, computed locally (momentum + MA-cross
  + Donchian breakout). +1 = strongly bullish, -1 = strongly bearish. Not a
  fundamental/value signal &mdash; purely price/volume based.<br>
  Signals as of {as_of} &mdash; {len(signals)} tickers scored.
</p>
""",
        top15_fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="top15"),
        bottom15_fig.to_html(full_html=False, include_plotlyjs=False, div_id="bottom15"),
        table_fig.to_html(full_html=False, include_plotlyjs=False, div_id="table"),
        "</body></html>",
    ]

    # wrap the two bar charts side by side
    full_html = html_parts[0] + '<div class="chart-row">' + html_parts[1] + html_parts[2] + '</div>' + html_parts[3] + html_parts[4]

    OUTPUT_FILE.write_text(full_html, encoding="utf-8")
    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
