import pandas as pd
import requests
import io

NSE_CSV_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

def get_nse_symbols():
    """
    Downloads the active equity list from NSE.
    Returns a list of symbols with '.NS' suffix.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(NSE_CSV_URL, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            # Filter EQ series (ordinary equities)
            df = df[df[' SERIES'].str.strip() == 'EQ']
            symbols = [f"{sym.strip()}.NS" for sym in df['SYMBOL'].tolist()]
            return symbols
    except Exception as e:
        print(f"Error fetching NSE symbols list: {e}")
    
    # Fallback to some common NSE stocks if download fails
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "BHARTIARTL.NS", "SBI.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS",
        "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "ADANIENT.NS", "KOTAKBANK.NS", "TATAMOTORS.NS", "AXISBANK.NS", "NTPC.NS",
        "ONGC.NS", "ADANIPORTS.NS", "POWERGRID.NS", "COALINDIA.NS", "TATASTEEL.NS"
    ]

def fetch_all_nse_prices(symbols, fyers_token=None, fyers_client_id=None):
    """
    Fetches prices for all symbols using Fyers API quotes.
    """
    if not fyers_token or not fyers_client_id:
        return {}

    fyers_syms = [f"NSE:{sym.replace('.NS', '')}-EQ" for sym in symbols]
    chunk_size_fyers = 50
    chunks = [fyers_syms[i:i + chunk_size_fyers] for i in range(0, len(fyers_syms), chunk_size_fyers)]
    from fyers_client import fetch_fyers_quotes
    all_results = {}
    for chunk in chunks:
        res = fetch_fyers_quotes(fyers_token, fyers_client_id, chunk)
        all_results.update(res)
    return all_results

