import pytest
import pandas as pd
from unittest.mock import MagicMock
from data_fetcher import get_nse_symbols, fetch_all_nse_prices

def test_get_nse_symbols_success(mocker):
    # Mock requests.get and ensure it extracts symbols correctly
    csv_data = "SYMBOL, SERIES\nRELIANCE , EQ\nTCS , EQ\nSBIN , EQ\nNIFTY , FUT\n"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = csv_data
    
    mocker.patch("requests.get", return_value=mock_resp)
    
    symbols = get_nse_symbols()
    assert len(symbols) == 3
    assert "RELIANCE.NS" in symbols
    assert "TCS.NS" in symbols
    assert "SBIN.NS" in symbols
    assert "NIFTY.NS" not in symbols  # Non-EQ filtered out

def test_get_nse_symbols_failure_fallback(mocker):
    # Mock requests.get to throw exception
    mocker.patch("requests.get", side_effect=Exception("Connection Error"))
    
    symbols = get_nse_symbols()
    assert len(symbols) > 0
    assert "RELIANCE.NS" in symbols
    assert "TCS.NS" in symbols

def test_fetch_all_nse_prices_no_auth():
    # Empty token scenario
    res = fetch_all_nse_prices(["RELIANCE.NS"], fyers_token=None, fyers_client_id="app_id")
    assert res == {}

def test_fetch_all_nse_prices_success(mocker):
    # Mock fetch_fyers_quotes
    mock_fetch = mocker.patch("fyers_client.fetch_fyers_quotes", return_value={
        "RELIANCE.NS": {"symbol": "RELIANCE", "ltp": 2500.0}
    })
    
    res = fetch_all_nse_prices(["RELIANCE.NS"], fyers_token="token", fyers_client_id="client_id")
    assert "RELIANCE.NS" in res
    assert res["RELIANCE.NS"]["ltp"] == 2500.0
    mock_fetch.assert_called_once()
