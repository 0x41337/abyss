import yfinance as yf
from settings import load_config

general, _ = load_config()
df = yf.download("BTC-USD", start=general["data"]["start"], end=general["data"]["end"])
