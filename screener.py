import pandas as pd

def screen_stocks(stock_data, min_price=30.0, max_price=500.0, min_bid_qty=0, min_ask_qty=0):
    """
    Screens the dictionary of stock data.
    Filters stocks where Last Traded Price (LTP) is between min_price and max_price,
    and optionally checks if Bid & Ask quantities exceed the specified thresholds.
    Returns a pandas DataFrame.
    """
    records = []
    for ticker, info in stock_data.items():
        ltp = info.get("ltp")
        bid_qty = info.get("bid_qty", 0)
        ask_qty = info.get("ask_qty", 0)
        
        if ltp is not None and min_price <= ltp <= max_price:
            if bid_qty >= min_bid_qty and ask_qty >= min_ask_qty:
                records.append({
                    "Ticker": ticker,
                    "Symbol": info.get("symbol"),
                    "LTP (₹)": round(ltp, 2),
                    "Change (%)": round(info.get("change_pct", 0.0), 2),
                    "Volume": info.get("volume", 0),
                    "Bid Price": info.get("bid_price", 0.0),
                    "Bid Qty": bid_qty,
                    "Ask Price": info.get("ask_price", 0.0),
                    "Ask Qty": ask_qty
                })


            
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="Volume", ascending=False).reset_index(drop=True)
    return df

def calculate_smma(series, period):
    """
    Calculates Smoothed Moving Average (SMMA) for a given pandas Series and period.
    SMMA(i) = (Prev_SMMA * (N - 1) + Price(i)) / N
    First value is simple SMA.
    """
    if len(series) < period:
        return pd.Series([None] * len(series), index=series.index)
    
    smma = [None] * len(series)
    sma = series.iloc[:period].mean()
    smma[period - 1] = sma
    
    for i in range(period, len(series)):
        smma[i] = (smma[i - 1] * (period - 1) + series.iloc[i]) / period
        
    return pd.Series(smma, index=series.index)

def get_detailed_stock_history(ticker, period="1y", interval="1d"):
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

def calculate_etq(ticker):
    """
    Calculates Exchange Traded Quantity (ETQ) for the last 5, 20, and 60 minutes.
    """
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period="5d", interval="1m")
        if df.empty:
            return 0, 0, 0
        
        last_5 = int(df['Volume'].iloc[-5:].sum()) if len(df) >= 5 else int(df['Volume'].sum())
        last_20 = int(df['Volume'].iloc[-20:].sum()) if len(df) >= 20 else int(df['Volume'].sum())
        last_60 = int(df['Volume'].iloc[-60:].sum()) if len(df) >= 60 else int(df['Volume'].sum())
        return last_5, last_20, last_60
    except Exception:
        return 0, 0, 0

def calculate_avg_price(ticker):
    """
    Calculates the average Close price (LTP) for the last 20 and 60 minutes.
    """
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period="5d", interval="1m")
        if df.empty:
            return 0.0, 0.0
        
        avg_20 = float(df['Close'].iloc[-20:].mean()) if len(df) >= 20 else float(df['Close'].mean())
        avg_60 = float(df['Close'].iloc[-60:].mean()) if len(df) >= 60 else float(df['Close'].mean())
        return avg_20, avg_60
    except Exception:
        return 0.0, 0.0


