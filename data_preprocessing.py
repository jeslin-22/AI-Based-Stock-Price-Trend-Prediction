import yfinance as yf
import pandas as pd
import numpy as np


def download_stock_data(ticker, period="5y"):
    """
    Download historical stock data from Yahoo Finance.
    """

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No stock data found. Check the ticker symbol.")

    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()

    return data


def calculate_rsi(series, period=14):
    """
    Calculate Relative Strength Index.
    """

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def create_features(data):
    """
    Create technical indicators and target variable.
    """

    df = data.copy()

    # Daily return
    df["Daily_Return"] = df["Close"].pct_change()

    # Moving averages
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()

    # Volatility
    df["Volatility"] = df["Daily_Return"].rolling(window=10).std()

    # RSI
    df["RSI"] = calculate_rsi(df["Close"])

    # Target:
    # 1 = price goes UP tomorrow
    # 0 = price goes DOWN tomorrow
    df["Tomorrow_Close"] = df["Close"].shift(-1)

    df["Target"] = (
        df["Tomorrow_Close"] > df["Close"]
    ).astype(int)

    # Remove missing values
    df.dropna(inplace=True)

    return df