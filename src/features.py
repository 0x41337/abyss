import numpy as np
from data import df


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
