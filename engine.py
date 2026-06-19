"""
Core analysis engine for the stock signal app.
Handles data fetch, technical indicators, seasonality decomposition,
rule-based buy/sell signal generation, and backtesting.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.seasonal import STL
import ta


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download historical OHLCV data for a ticker."""
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol "
                          f"(NSE tickers need '.NS' suffix, e.g. RELIANCE.NS).")
    # yfinance sometimes returns MultiIndex columns for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, moving averages, Bollinger Bands to the dataframe."""
    out = df.copy()
    close = out["Close"]

    out["SMA_20"] = close.rolling(20).mean()
    out["SMA_50"] = close.rolling(50).mean()
    out["SMA_200"] = close.rolling(200).mean()

    out["RSI_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close)
    out["MACD"] = macd.macd()
    out["MACD_signal"] = macd.macd_signal()
    out["MACD_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["BB_upper"] = bb.bollinger_hband()
    out["BB_lower"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()

    out["Volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    return out


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------

def seasonal_decompose(df: pd.DataFrame, period: int = 252) -> dict:
    """
    STL decomposition of the close price into trend/seasonal/residual.
    period=252 ~ one trading year, captures annual seasonality.
    Returns dict of pd.Series plus monthly seasonal averages.
    """
    close = df["Close"].dropna()
    if len(close) < period * 2:
        period = max(21, len(close) // 3)  # fallback for shorter histories

    stl = STL(close, period=period, robust=True)
    res = stl.fit()

    # Monthly seasonality from raw returns (more interpretable than STL seasonal term)
    monthly_returns = df["Close"].pct_change().groupby(df.index.month).mean() * 100
    monthly_returns.index = monthly_returns.index.map(
        lambda m: pd.Timestamp(2000, m, 1).strftime("%b")
    )

    return {
        "trend": res.trend,
        "seasonal": res.seasonal,
        "resid": res.resid,
        "monthly_avg_return_pct": monthly_returns,
    }


# ---------------------------------------------------------------------------
# Signal generation (rule-based, explainable)
# ---------------------------------------------------------------------------

def generate_signal(df: pd.DataFrame, monthly_seasonality: pd.Series) -> dict:
    """
    Combine technical indicators + seasonality into a single Buy/Hold/Sell
    call for the most recent row, with a transparent reason list and a
    -100 to +100 score.
    """
    latest = df.iloc[-1]
    reasons = []
    score = 0

    # --- Trend: price vs moving averages ---
    if latest["Close"] > latest["SMA_50"] > latest["SMA_200"]:
        score += 20
        reasons.append("Price above both 50-day and 200-day moving averages (uptrend)")
    elif latest["Close"] < latest["SMA_50"] < latest["SMA_200"]:
        score -= 20
        reasons.append("Price below both 50-day and 200-day moving averages (downtrend)")

    # Golden/death cross proximity
    if latest["SMA_50"] > latest["SMA_200"]:
        score += 5
        reasons.append("50-day MA above 200-day MA (golden cross regime)")
    else:
        score -= 5
        reasons.append("50-day MA below 200-day MA (death cross regime)")

    # --- RSI ---
    rsi = latest["RSI_14"]
    if rsi < 30:
        score += 20
        reasons.append(f"RSI at {rsi:.1f} — oversold, potential bounce")
    elif rsi > 70:
        score -= 20
        reasons.append(f"RSI at {rsi:.1f} — overbought, potential pullback")
    else:
        reasons.append(f"RSI at {rsi:.1f} — neutral zone")

    # --- MACD ---
    if latest["MACD"] > latest["MACD_signal"]:
        score += 15
        reasons.append("MACD above signal line (bullish momentum)")
    else:
        score -= 15
        reasons.append("MACD below signal line (bearish momentum)")

    # --- Bollinger Bands ---
    if latest["Close"] <= latest["BB_lower"]:
        score += 15
        reasons.append("Price at/below lower Bollinger Band (stretched downward)")
    elif latest["Close"] >= latest["BB_upper"]:
        score -= 15
        reasons.append("Price at/above upper Bollinger Band (stretched upward)")

    # --- Seasonality ---
    current_month = pd.Timestamp.now().strftime("%b")
    seasonal_avg = monthly_seasonality.get(current_month, 0)
    if seasonal_avg > 0.3:
        score += 10
        reasons.append(f"{current_month} has historically been a strong month "
                        f"(avg daily return {seasonal_avg:.2f}%)")
    elif seasonal_avg < -0.3:
        score -= 10
        reasons.append(f"{current_month} has historically been a weak month "
                        f"(avg daily return {seasonal_avg:.2f}%)")
    else:
        reasons.append(f"{current_month} shows no strong seasonal bias historically")

    score = max(-100, min(100, score))

    if score >= 25:
        call = "BUY"
    elif score <= -25:
        call = "SELL"
    else:
        call = "HOLD"

    return {
        "call": call,
        "score": score,
        "reasons": reasons,
        "rsi": rsi,
        "close": latest["Close"],
        "date": df.index[-1],
    }


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def backtest_strategy(df: pd.DataFrame, monthly_seasonality: pd.Series,
                       lookback_window: int = 252) -> dict:
    """
    Walk forward day by day, generate the same rule-based signal using only
    data available up to that point, and simulate a simple long/flat strategy:
    go long when call == BUY, exit to cash when call == SELL, hold position on HOLD.
    Compares to buy-and-hold over the same period.
    """
    dates, equity_curve, buyhold_curve, calls = [], [], [], []
    position = 0  # 0 = cash, 1 = long
    equity = 1.0
    buyhold_equity = 1.0
    entry_price = None
    bh_entry_price = None

    start_idx = max(200, len(df) - lookback_window)  # need 200d for SMA_200

    for i in range(start_idx, len(df)):
        window = df.iloc[: i + 1]
        if len(window) < 200:
            continue

        sig = generate_signal(window, monthly_seasonality)
        price = window["Close"].iloc[-1]
        date = window.index[-1]

        if bh_entry_price is None:
            bh_entry_price = price
        buyhold_equity = price / bh_entry_price

        if sig["call"] == "BUY" and position == 0:
            position = 1
            entry_price = price
        elif sig["call"] == "SELL" and position == 1:
            equity *= price / entry_price
            position = 0
            entry_price = None

        # mark-to-market equity if currently long
        current_equity = equity * (price / entry_price) if position == 1 else equity

        dates.append(date)
        equity_curve.append(current_equity)
        buyhold_curve.append(buyhold_equity)
        calls.append(sig["call"])

    result_df = pd.DataFrame({
        "Date": dates,
        "Strategy": equity_curve,
        "BuyHold": buyhold_curve,
        "Call": calls,
    }).set_index("Date")

    strat_return = (result_df["Strategy"].iloc[-1] - 1) * 100 if len(result_df) else 0
    bh_return = (result_df["BuyHold"].iloc[-1] - 1) * 100 if len(result_df) else 0

    strat_daily_ret = result_df["Strategy"].pct_change().dropna()
    sharpe = (strat_daily_ret.mean() / strat_daily_ret.std() * np.sqrt(252)
              if strat_daily_ret.std() > 0 else 0)

    return {
        "curve": result_df,
        "strategy_return_pct": strat_return,
        "buyhold_return_pct": bh_return,
        "sharpe": sharpe,
        "num_buy_signals": sum(1 for c in calls if c == "BUY"),
        "num_sell_signals": sum(1 for c in calls if c == "SELL"),
    }
