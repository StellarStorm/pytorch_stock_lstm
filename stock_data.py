import yfinance as yf


def fetch_stock_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    df.dropna(inplace=True)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df
