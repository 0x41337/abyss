"""
    Example of daily forecast using monthly data.
"""

import numpy as np
import pickle as pk
import pandas as pd
import yfinance as yf


# EMA (Exponential Moving Average)
def EMA(span=10):
    return df["Close"].ewm(span, adjust=False).mean()


# MACD (Moving Average Convergence Divergence)
def MACD():
    return EMA(12) - EMA(26)


# RSI (Relative Strength Index)
def RSI(window=14):
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# Logarithmic returns
def LogReturn():
    return np.log(df["Close"] / df["Close"].shift(1))

# Volatility
def Volatility(window=10):
    return df["Close"].rolling(window).std()


# Bollinger Bands
def BollingerBands(window=20):
    rolling_mean = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band


# Load model
with open("dist/abyss.pkl", "rb") as file:
    model = pk.load(file)

# Get data for the entire month
df = yf.download("BTC-USD", period="1mo", progress=False)

# Generate the indicators
df["EMA"] = EMA()
df["MACD"] = MACD()
df["RSI"] = RSI()
df["LogReturn"] = LogReturn()
df["Volatility"] = Volatility()
df["Upper Band"], df["Lower Band"] = BollingerBands()

# Define independent (X) and dependent (y) variables
X = df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "EMA",
        "MACD",
        "RSI",
        "LogReturn",
        "Volatility",
        "Upper Band",
        "Lower Band",
    ]
]
y = df["Close"]

# Forecast
forecast = model.predict(X)

# Show Forecast
forecast_df = pd.DataFrame(
    {
        "Date": df.index,
        "Predicted Close": forecast,
        "Real Close": df["Close"].values.ravel(),
    }
)
print(forecast_df)
