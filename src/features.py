from data import df

# EMA (Exponential Moving Average)
def f1(span=10):
    return df["Close"].ewm(span, adjust=False).mean()

# MACD (Moving Average Convergence Divergence)
def f2():
    return f1(12) - f1(26)

# RSI (Relative Strength Index)
def f3(window=14):
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
