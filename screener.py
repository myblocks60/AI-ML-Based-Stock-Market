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
                    "Bid Qty": bid_qty,
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
