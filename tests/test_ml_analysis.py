import pytest
import pandas as pd
from ml_analysis import (
    extract_crossover_features,
    analyze_crossover_signal,
    get_all_historical_crossovers
)

def test_extract_crossover_features_bounds(sample_history_df):
    # Testing index boundary conditions (idx too small or too close to end)
    res_low = extract_crossover_features(sample_history_df, 10)
    assert res_low is None
    
    res_high = extract_crossover_features(sample_history_df, len(sample_history_df) - 2)
    assert res_high is None

def test_analyze_crossover_signal_insufficient_data():
    empty_df = pd.DataFrame()
    signal, conf, desc = analyze_crossover_signal(empty_df)
    assert signal == "INSUFFICIENT DATA"
    assert conf == 50.0

def test_analyze_crossover_signal_heuristics(sample_history_df):
    # Use a slice with only 129 entries
    short_df = sample_history_df.iloc[:129]
    signal, conf, desc = analyze_crossover_signal(short_df)
    assert signal == "INSUFFICIENT DATA"

    # With full history, but forcing fewer crossover events
    # We can pass standard sample_history_df.
    # It has length 150 which is > 130. Since data is mostly upward trending after index 100,
    # it may have few crossovers. Let's see if it triggers heuristics or KNN.
    signal, conf, desc = analyze_crossover_signal(sample_history_df)
    assert signal is not None
    assert conf >= 50.0

def test_get_all_historical_crossovers(sample_history_df):
    crossovers = get_all_historical_crossovers(sample_history_df)
    # Check return type is a DataFrame
    assert isinstance(crossovers, pd.DataFrame)
    if not crossovers.empty:
        assert "Signal" in crossovers.columns
        assert "Entry LTP (₹)" in crossovers.columns
        assert "ML Recommendation" in crossovers.columns
