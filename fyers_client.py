from fyers_apiv3 import fyersModel

def get_session_model(client_id, secret_key, redirect_uri):
    """
    Creates and returns a Fyers SessionModel helper for authentication.
    """
    return fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code"
    )

def generate_auth_link(session):
    """
    Generates Fyers login link.
    """
    return session.generate_authcode()

def get_access_token(session, auth_code):
    """
    Generates and returns an access token from the redirect authorization code.
    """
    session.set_token(auth_code)
    response = session.generate_token()
    return response.get("access_token")

def fetch_fyers_quotes(access_token, client_id, symbols):
    """
    Fetches real-time stock details from Fyers quotes API.
    Fyers symbols format: ['NSE:ADANIPOWER-EQ', 'NSE:RELIANCE-EQ']
    """
    if not symbols:
        return {}
        
    try:
        # FyersModel constructor requires client_id (which is app_id) and access_token
        fyers = fyersModel.FyersModel(
            client_id=client_id,
            token=access_token,
            is_async=False,
            log_path=""
        )
        
        # Fyers batch quotes accepts comma-separated symbol string
        symbol_str = ",".join(symbols)
        response = fyers.quotes(data={"symbols": symbol_str})
        
        results = {}
        if response.get("s") == "ok" and "d" in response:
            for quote in response["d"]:
                # Parse fields
                sym = quote.get("n", "") # Symbol name e.g. NSE:RELIANCE-EQ
                val = quote.get("v", {}) # Value dictionary
                
                # Extract bid/ask from market depth in the quote
                bid_price = val.get("bid", 0.0)
                ask_price = val.get("ask", 0.0)
                # Fallback to depth list if bid/ask direct properties are missing
                depth = val.get("depth", {})
                bids = depth.get("bids", [])
                asks = depth.get("asks", [])
                
                bid_qty = bids[0].get("q", 0) if bids else 0
                ask_qty = asks[0].get("q", 0) if asks else 0
                if bid_price == 0.0 and bids:
                    bid_price = bids[0].get("p", 0.0)
                if ask_price == 0.0 and asks:
                    ask_price = asks[0].get("p", 0.0)
                
                ticker_key = sym.replace("NSE:", "").replace("-EQ", "") + ".NS"
                results[ticker_key] = {
                    "symbol": sym.replace("NSE:", "").replace("-EQ", ""),
                    "ltp": float(val.get("lp", 0.0)),
                    "change_pct": float(val.get("chp", 0.0)),
                    "volume": int(val.get("vol", 0)),
                    "bid_qty": int(bid_qty),
                    "ask_qty": int(ask_qty),
                    "bid_price": float(bid_price),
                    "ask_price": float(ask_price),
                    "ltq": int(val.get("ltq", 0))
                }
        return results
    except Exception as e:
        print(f"Error fetching Fyers quotes: {e}")
        return {}
