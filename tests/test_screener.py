import pytest
import pandas as pd
from unittest.mock import MagicMock
from screener import (
    screen_stocks,
    calculate_smma,
    calculate_etq,
    calculate_avg_price,
    calculate_batch_metrics
)

def test_screen_stocks(sample_stock_data):
    # RELIANCE ltp is 2500, TCS is 3400 (both above max_price=500.0)
    # PENNY ltp is 15 (below min_price=30.0)
    # Filter with wider range to catch RELIANCE & TCS
    df = screen_stocks(sample_stock_data, min_price=10.0, max_price=4000.0, min_bid_qty=0, min_ask_qty=0)
    assert not df.empty
    assert "PENNY.NS" in df['Ticker'].tolist()
    assert "RELIANCE.NS" in df['Ticker'].tolist()
    
    # Filter with narrow range where everything is filtered out
    df_empty = screen_stocks(sample_stock_data, min_price=100.0, max_price=200.0)
    assert df_empty.empty
    
    # Check liquidity limits
    df_liq = screen_stocks(sample_stock_data, min_price=10.0, max_price=4000.0, min_bid_qty=1000000, min_ask_qty=1000000)

    # RELIANCE has 1.5M bid, 1.2M ask. TCS has 800k bid, 950k ask.
    # So only RELIANCE should match.
    tickers = df_liq['Ticker'].tolist()
    assert "RELIANCE.NS" in tickers
    assert "TCS.NS" not in tickers

def test_calculate_smma():
    prices = pd.Series([10.0, 12.0, 11.0, 13.0, 12.0])
    # Case when history is too short for period
    res = calculate_smma(prices, period=10)
    assert res.isna().all()
    
    # Case with period 3
    res_3 = calculate_smma(prices, period=3)
    # SMA for first 3: (10 + 12 + 11) / 3 = 11.0
    assert res_3.iloc[2] == 11.0
    # SMMA(4) = (11.0 * 2 + 13.0) / 3 = 11.666...
    assert round(res_3.iloc[3], 2) == 11.67

def test_calculate_etq(mocker):
    mock_ticker = MagicMock()
    mock_df = pd.DataFrame({
        "Volume": [100, 200, 300, 400, 500, 600]
    })
    mock_ticker.history.return_value = mock_df
    mocker.patch("yfinance.Ticker", return_value=mock_ticker)
    
    l5, l20, l60 = calculate_etq("RELIANCE.NS")
    # Last 5 elements sum: 200+300+400+500+600 = 2000
    assert l5 == 2000
    # All 6 sum: 2100
    assert l20 == 2100
    assert l60 == 2100

def test_calculate_avg_price(mocker):
    mock_ticker = MagicMock()
    mock_df = pd.DataFrame({
        "Close": [10.0, 20.0, 30.0, 40.0]
    })
    mock_ticker.history.return_value = mock_df
    mocker.patch("yfinance.Ticker", return_value=mock_ticker)
    
    avg20, avg60 = calculate_avg_price("RELIANCE.NS")
    # Average of all: 25.0
    assert avg20 == 25.0
    assert avg60 == 25.0

def test_calculate_batch_metrics(mocker):
    # Mock yfinance.download
    # Needs to return a MultiIndex columns dataframe for yf.download(group_by="ticker")
    arrays = [['RELIANCE.NS', 'RELIANCE.NS'], ['Close', 'Volume']]
    columns = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=['Ticker', 'Field'])
    
    # 130 periods to satisfy SMMA120
    close_prices = [100.0 + i for i in range(130)]
    volumes = [1000] * 130
    
    mock_df_daily = pd.DataFrame(
        zip(close_prices, volumes),
        columns=columns,
        index=pd.date_range("2026-01-01", periods=130)
    )
    
    mocker.patch("yfinance.download", side_effect=[mock_df_daily, pd.DataFrame()])
    
    metrics = calculate_batch_metrics(["RELIANCE.NS"])
    assert "RELIANCE.NS" in metrics
    assert metrics["RELIANCE.NS"]["smma20"] is not None
    assert metrics["RELIANCE.NS"]["smma120"] is not None
