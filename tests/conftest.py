import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_stock_data():
    return {
        "RELIANCE.NS": {
            "symbol": "RELIANCE",
            "ltp": 2500.0,
            "change_pct": 1.5,
            "volume": 5000000,
            "ltq": 10,
            "bid_price": 2499.0,
            "bid_qty": 1500000,
            "ask_price": 2501.0,
            "ask_qty": 1200000
        },
        "TCS.NS": {
            "symbol": "TCS",
            "ltp": 3400.0,
            "change_pct": -0.5,
            "volume": 2000000,
            "ltq": 5,
            "bid_price": 3398.0,
            "bid_qty": 800000,
            "ask_price": 3402.0,
            "ask_qty": 950000
        },
        "PENNY.NS": {
            "symbol": "PENNY",
            "ltp": 15.0,
            "change_pct": 4.0,
            "volume": 100000,
            "ltq": 20,
            "bid_price": 14.9,
            "bid_qty": 5000,
            "ask_price": 15.1,
            "ask_qty": 4000
        }
    }

@pytest.fixture
def sample_history_df():
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=150, freq="D")
    
    # We want a series that goes up and down to trigger actual crossovers
    close_prices = []
    current_price = 100.0
    for i in range(150):
        # Sine wave + trend to force multiple crossovers
        current_price = 150.0 + 30.0 * np.sin(i / 10.0) + np.random.normal(0, 1.0)
        close_prices.append(current_price)
        
    df = pd.DataFrame({
        "Open": [p - 0.5 for p in close_prices],
        "High": [p + 1.0 for p in close_prices],
        "Low": [p - 1.0 for p in close_prices],
        "Close": close_prices,
        "Volume": [100000 + int(10000 * np.random.rand()) for _ in range(150)]
    }, index=dates)
    return df

