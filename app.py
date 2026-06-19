"""
Stock Signal Dashboard
A Streamlit app that gives a rule-based, explainable Buy/Hold/Sell signal
for a stock, combining technical indicators with seasonality analysis,
plus a backtest of the strategy's historical performance.

NOT FINANCIAL ADVICE. Educational / decision-support tool only.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import (
    fetch_data, add_indicators, seasonal_decompose,
    generate_signal, backtest_strategy,
)

st.set_page_config(page_title="Stock Signal Dashboard", layout="wide", page_icon="📈")

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
st.sidebar.title("📈 Settings")
ticker = st.sidebar.text_input(
    "Ticker symbol",
    value="RELIANCE.NS",
    help="NSE stocks need '.NS' (e.g. TCS.NS, INFY.NS). "
         "US stocks just need the symbol (e.g. AAPL, MSFT).",
)
period = st.sidebar.selectbox(
    "History length", ["1y", "2y", "5y", "10y", "max"], index=2
)
run_btn = st.sidebar.button("Analyze", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Disclaimer**: This tool generates signals from a fixed set of "
    "technical and seasonal rules applied to historical price data. It is "
    "for educational purposes only and is **not financial advice**. Past "
    "performance does not guarantee future results. Always do your own "
    "research or consult a licensed advisor before trading."
)

st.title("📈 Stock Buy/Sell Signal & Seasonality Dashboard")
st.caption("Rule-based, fully explainable signals — no black-box predictions.")

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

if run_btn:
    st.session_state.analyzed = True
    st.session_state.ticker = ticker
    st.session_state.period = period

if not st.session_state.analyzed:
    st.info("👈 Enter a ticker and click **Analyze** to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Fetch + compute
# ---------------------------------------------------------------------------
with st.spinner(f"Fetching data for {st.session_state.ticker}..."):
    try:
        raw = fetch_data(st.session_state.ticker, st.session_state.period)
    except Exception as e:
        st.error(str(e))
        st.stop()

with st.spinner("Computing indicators and seasonality..."):
    df = add_indicators(raw)
    decomp = seasonal_decompose(raw)
    signal = generate_signal(df, decomp["monthly_avg_return_pct"])

# ---------------------------------------------------------------------------
# Top: signal card
# ---------------------------------------------------------------------------
call_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[signal["call"]]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Signal", f"{call_color} {signal['call']}")
col2.metric("Score (-100 to +100)", f"{signal['score']}")
col3.metric("Latest Close", f"{signal['close']:.2f}")
col4.metric("RSI (14)", f"{signal['rsi']:.1f}")

with st.expander("Why this signal? (full reasoning)", expanded=True):
    for r in signal["reasons"]:
        st.markdown(f"- {r}")
    st.caption(f"As of {signal['date'].strftime('%Y-%m-%d')}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Price chart with indicators
# ---------------------------------------------------------------------------
st.subheader("Price Chart with Indicators")

plot_df = df.tail(300)
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    row_heights=[0.55, 0.2, 0.25],
    subplot_titles=("Price + Moving Averages + Bollinger Bands", "RSI (14)", "MACD"),
)

fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
    low=plot_df["Low"], close=plot_df["Close"], name="Price"
), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_50"], name="SMA 50",
                          line=dict(width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_200"], name="SMA 200",
                          line=dict(width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_upper"], name="BB Upper",
                          line=dict(width=1, dash="dot"), opacity=0.5), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_lower"], name="BB Lower",
                          line=dict(width=1, dash="dot"), opacity=0.5), row=1, col=1)

fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI_14"], name="RSI",
                          line=dict(color="purple")), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["MACD_hist"], name="MACD Hist"),
              row=3, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"], name="MACD",
                          line=dict(color="blue", width=1)), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD_signal"], name="Signal",
                          line=dict(color="orange", width=1)), row=3, col=1)

fig.update_layout(height=750, xaxis_rangeslider_visible=False,
                   legend=dict(orientation="h", y=1.05))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------
st.subheader("Seasonality Analysis")

sc1, sc2 = st.columns([1, 1])

with sc1:
    st.markdown("**Average daily return by month (historical)**")
    monthly = decomp["monthly_avg_return_pct"]
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = monthly.reindex([m for m in month_order if m in monthly.index])
    colors = ["#16a34a" if v > 0 else "#dc2626" for v in monthly.values]
    bar_fig = go.Figure(go.Bar(x=monthly.index, y=monthly.values, marker_color=colors))
    bar_fig.update_layout(height=350, yaxis_title="Avg daily return (%)")
    st.plotly_chart(bar_fig, use_container_width=True)

with sc2:
    st.markdown("**STL Trend & Seasonal Decomposition**")
    decomp_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("Trend", "Seasonal Component"))
    decomp_fig.add_trace(go.Scatter(x=decomp["trend"].index, y=decomp["trend"],
                                     name="Trend"), row=1, col=1)
    decomp_fig.add_trace(go.Scatter(x=decomp["seasonal"].index, y=decomp["seasonal"],
                                     name="Seasonal"), row=2, col=1)
    decomp_fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(decomp_fig, use_container_width=True)

best_month = monthly.idxmax()
worst_month = monthly.idxmin()
st.info(f"📅 Historically strongest month: **{best_month}** "
        f"({monthly.max():.2f}% avg daily return) · "
        f"Weakest: **{worst_month}** ({monthly.min():.2f}%)")

st.markdown("---")

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
st.subheader("Strategy Backtest (vs Buy & Hold)")
st.caption("Simulates following this exact rule-based signal historically. "
           "Shows whether the strategy would have beaten simply holding the stock.")

with st.spinner("Running backtest..."):
    bt = backtest_strategy(df, decomp["monthly_avg_return_pct"])

bc1, bc2, bc3, bc4 = st.columns(4)
bc1.metric("Strategy Return", f"{bt['strategy_return_pct']:.1f}%")
bc2.metric("Buy & Hold Return", f"{bt['buyhold_return_pct']:.1f}%")
bc3.metric("Sharpe Ratio", f"{bt['sharpe']:.2f}")
bc4.metric("Buy / Sell Signals Fired", f"{bt['num_buy_signals']} / {bt['num_sell_signals']}")

bt_fig = go.Figure()
bt_fig.add_trace(go.Scatter(x=bt["curve"].index, y=bt["curve"]["Strategy"],
                             name="Strategy", line=dict(color="#2563eb", width=2)))
bt_fig.add_trace(go.Scatter(x=bt["curve"].index, y=bt["curve"]["BuyHold"],
                             name="Buy & Hold", line=dict(color="gray", width=2, dash="dash")))
bt_fig.update_layout(height=400, yaxis_title="Growth of ₹1 invested",
                      legend=dict(orientation="h", y=1.05))
st.plotly_chart(bt_fig, use_container_width=True)

if bt["strategy_return_pct"] < bt["buyhold_return_pct"]:
    st.warning("This rule-based strategy underperformed simple buy-and-hold over "
               "this period — common for short timeframes or strongly trending stocks. "
               "Treat signals as one input, not a guarantee.")
else:
    st.success("This strategy outperformed buy-and-hold over this backtest period.")

st.markdown("---")
st.caption("Built with Streamlit · Data via Yahoo Finance · "
           "Indicators: RSI, MACD, Bollinger Bands, SMA · Seasonality: STL decomposition")
