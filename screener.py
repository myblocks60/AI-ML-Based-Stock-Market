import pandas as pd

def screen_stocks(stock_data, min_price=30.0, max_price=500.0):
    """
    Screens the dictionary of stock data.
    Filters stocks where Last Traded Price (LTP) is between min_price and max_price.
    Returns a pandas DataFrame.
    """
    records = []
    for ticker, info in stock_data.items():
        ltp = info.get("ltp")
        if ltp is not None and min_price <= ltp <= max_price:
            records.append({
                "Ticker": ticker,
                "Symbol": info.get("symbol"),
                "LTP (₹)": round(ltp, 2),
                "Change (%)": round(info.get("change_pct", 0.0), 2),
                "Volume": info.get("volume", 0)
            })
            
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="Volume", ascending=False).reset_index(drop=True)
    return df

def get_detailed_stock_history(ticker, period="1mo", interval="1d"):
    """
    Fetches detailed historical data for a specific stock ticker.
    """
    try:
        stock = yf_history_direct(ticker, period, interval)
        return stock
    except Exception:
        return pd.DataFrame()

def yf_history_direct(ticker, period, interval):
    import yfinance as yf
    return yf.Ticker(ticker).history(period=period, interval=interval)
