from data import df

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

# Bollinger Bands
def BollingerBands(window=20):
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band