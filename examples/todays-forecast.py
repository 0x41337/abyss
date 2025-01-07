"""
    Example of daily forecast using monthly data.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import onnxruntime as ort


# EMA (Exponential Moving Average)
def EMA(span=14):
    return df["Close"].ewm(span, adjust=False).mean()


# MACD (Moving Average Convergence Divergence)
def MACD():
    return EMA(6) - EMA(18)


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
def Volatility(window=7):
    return df["Close"].rolling(window).std()


# Bollinger Bands
def BollingerBands(window=14):
    rolling_mean = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band


# Momentum
def Momentum(window=10):
    return df["Close"] - df["Close"].shift(window)


# Load ONNX model using onnxruntime
onnx_model_path = "dist/abyss.onnx"
session = ort.InferenceSession(onnx_model_path)

# Get data for the entire month
df = yf.download("BTC-USD", period="1mo", progress=False)

# Generate the indicators
df["EMA"] = EMA()
df["MACD"] = MACD()
df["Momentum"] = Momentum()
df["RSI"] = RSI()
df["LogReturn"] = LogReturn()
df["Volatility"] = Volatility()
df["Upper Band"], df["Lower Band"] = BollingerBands()

# Define independent (X) variables for prediction
X = df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Momentum",
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

# Prepare the input data in the right format (NumPy array)
X_input = X.values.astype(np.float32)  # Convert to float32 for compatibility with ONNX

# Get the model input name (usually "input" or something similar)
input_name = session.get_inputs()[0].name

# Perform the inference (prediction)
forecast = session.run(None, {input_name: X_input})[0]

# Show Forecast
forecast_df = pd.DataFrame(
    {
        "Date": df.index,
        "Predicted Close": forecast.flatten(),
        "Real Close": df["Close"].values.ravel(),
    }
)
print(forecast_df)
