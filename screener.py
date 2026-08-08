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

def calculate_batch_metrics(tickers):
    """
    Fetches daily and intraday data in batch, calculates SMMA, ETQ, and Average Price,
    and returns a dictionary of metrics per ticker.
    """
    if not tickers:
        return {}
    
    import yfinance as yf
    import numpy as np
    
    metrics = {}
    try:
        daily_data = yf.download(tickers, period="1y", interval="1d", group_by="ticker", progress=False, threads=True)
    except Exception:
        daily_data = pd.DataFrame()
        
    try:
        intra_data = yf.download(tickers, period="5d", interval="1m", group_by="ticker", progress=False, threads=True)
    except Exception:
        intra_data = pd.DataFrame()
        
    for ticker in tickers:
        m = {
            "smma20": None, "smma120": None,
            "etq5": 0, "etq20": 0, "etq60": 0,
            "avg20": 0.0, "avg60": 0.0
        }
        try:
            if ticker in daily_data.columns.levels[0]:
                closes = daily_data[ticker]['Close'].dropna()
                if len(closes) >= 120:
                    s20 = calculate_smma(closes, 20).iloc[-1]
                    m["smma20"] = round(float(s20), 2) if s20 is not None and not np.isnan(s20) else None
                    s120 = calculate_smma(closes, 120).iloc[-1]
                    m["smma120"] = round(float(s120), 2) if s120 is not None and not np.isnan(s120) else None
        except Exception:
            pass
            
        try:
            if ticker in intra_data.columns.levels[0]:
                t_intra = intra_data[ticker].dropna(subset=['Close'])
                if not t_intra.empty:
                    m["etq5"] = int(t_intra['Volume'].iloc[-5:].sum()) if len(t_intra) >= 5 else int(t_intra['Volume'].sum())
                    m["etq20"] = int(t_intra['Volume'].iloc[-20:].sum()) if len(t_intra) >= 20 else int(t_intra['Volume'].sum())
                    m["etq60"] = int(t_intra['Volume'].iloc[-60:].sum()) if len(t_intra) >= 60 else int(t_intra['Volume'].sum())
                    m["avg20"] = round(float(t_intra['Close'].iloc[-20:].mean()), 2) if len(t_intra) >= 20 else round(float(t_intra['Close'].mean()), 2)
                    m["avg60"] = round(float(t_intra['Close'].iloc[-60:].mean()), 2) if len(t_intra) >= 60 else round(float(t_intra['Close'].mean()), 2)
        except Exception:
            pass
        metrics[ticker] = m
    return metrics



