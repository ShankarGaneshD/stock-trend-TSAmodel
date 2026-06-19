# Stock Signal Dashboard

A rule-based, fully explainable Buy/Hold/Sell signal tool for stocks — combines
technical indicators (RSI, MACD, Bollinger Bands, moving averages) with
seasonality analysis (STL decomposition + monthly return patterns), and
backtests the strategy against simple buy-and-hold.

**This is a decision-support / educational tool, not financial advice.**

## Files
- `app.py` — Streamlit UI
- `engine.py` — data fetching, indicators, seasonality, signal logic, backtest
- `requirements.txt` — dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy to Streamlit Cloud (free, shareable public link)

1. Create a GitHub repo and push these 3 files (`app.py`, `engine.py`, `requirements.txt`) to it.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app** → pick your repo → set main file to `app.py` → **Deploy**.
4. You'll get a public URL like `https://your-app-name.streamlit.app` you can share with anyone.

Redeploys automatically whenever you push changes to the repo.

## Usage

- Enter a ticker (NSE stocks need `.NS`, e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`;
  US stocks just need the symbol, e.g. `AAPL`)
- Pick a history length
- Click **Analyze**

You'll get: a Buy/Sell/Hold call with full reasoning, an indicator chart,
seasonality breakdown by month, and a backtest comparing the strategy to
buy-and-hold.

## How the signal works

The score (-100 to +100) is built from:
- Price vs 50-day/200-day moving averages (trend)
- Golden cross / death cross regime
- RSI (overbought/oversold)
- MACD vs signal line (momentum)
- Bollinger Band position (mean reversion)
- Historical seasonality for the current month

Score ≥ 25 → BUY · Score ≤ -25 → SELL · otherwise → HOLD.

No machine learning black box — every signal is traceable to a specific rule,
so you can see exactly why it fired.

## Customizing

Want to add more tickers, change thresholds, or add new indicators? Edit
`generate_signal()` in `engine.py` — each rule is a clearly separated block
that adds/subtracts from the score with a logged reason.
