import pandas as pd
import numpy as np
from screener import calculate_smma

def extract_crossover_features(df, idx, period_after=5):
    """
    Extracts features for a given index where a crossover occurred.
    Labels it based on forward price return (positive return -> Accept, negative -> Avoid).
    """
    if idx < 120 or idx >= len(df) - period_after:
        return None
    close, volume = df['Close'], df['Volume']
    smma20, smma120 = df['SMMA20'], df['SMMA120']
    
    vol_sma20 = volume.iloc[idx-20:idx].mean()
    vol_ratio = volume.iloc[idx] / vol_sma20 if vol_sma20 > 0 else 1.0
    
    # Feature 2: LTQ 2-period vs 5-period average volume ratio
    vol_2 = volume.iloc[idx-2:idx].mean()
    vol_5 = volume.iloc[idx-5:idx].mean()
    ltq_ratio = vol_2 / vol_5 if vol_5 > 0 else 1.0
    
    smma120_slope = (smma120.iloc[idx] - smma120.iloc[idx-5]) / smma120.iloc[idx-5] if smma120.iloc[idx-5] else 0.0
    hist_vol = close.iloc[idx-5:idx].std() / close.iloc[idx-5:idx].mean() if close.iloc[idx-5:idx].mean() > 0 else 0.0
    
    forward_return = (close.iloc[idx + period_after] - close.iloc[idx]) / close.iloc[idx]
    label = 1 if forward_return > 0 else 0
    return [vol_ratio, ltq_ratio, smma120_slope, hist_vol, label]

def analyze_crossover_signal(history, bid_qty=0, ask_qty=0):
    """
    Identifies crossover signals and trains a KNN model on historical crossovers
    using LTQ acceleration ratios, volume, trend slope, and volatility.
    """
    if len(history) < 130:
        return "INSUFFICIENT DATA", 50.0, "Need at least 130 periods of historical data."
        
    if 'SMMA20' not in history.columns:
        history['SMMA20'] = calculate_smma(history['Close'], 20)
    if 'SMMA120' not in history.columns:
        history['SMMA120'] = calculate_smma(history['Close'], 120)
        
    history = history.copy()
    history['Diff'] = history['SMMA20'] - history['SMMA120']
    history['Prev_Diff'] = history['Diff'].shift(1)
    crossovers = history[(history['Diff'] * history['Prev_Diff'] < 0) & (history['Diff'].notna()) & (history['Prev_Diff'].notna())]
    
    if len(crossovers) < 3:
        latest_row = history.iloc[-1]
        is_bullish = latest_row['Diff'] > 0
        volume_ok = latest_row['Volume'] > history['Volume'].mean()
        action = "ACCEPT" if volume_ok else "AVOID"
        signal_type = "BULLISH (Golden Cross)" if is_bullish else "BEARISH (Death Cross)"
        return f"{action} - {signal_type}", 75.0 if volume_ok else 55.0, "Analyzed using heuristic rules (insufficient crossover events for ML training)."

    dataset = []
    for idx in crossovers.index:
        pos = history.index.get_loc(idx)
        feat = extract_crossover_features(history, pos)
        if feat:
            dataset.append(feat)
            
    if len(dataset) < 3:
        return "AVOID - NO PATTERN", 50.0, "Lack of stable historical trends."
        
    X = np.array([f[:-1] for f in dataset])
    y = np.array([f[-1] for f in dataset])
    
    vol_sma20 = history['Volume'].iloc[-20:].mean()
    vol_ratio = history['Volume'].iloc[-1] / vol_sma20 if vol_sma20 > 0 else 1.0
    vol_2 = history['Volume'].iloc[-2:].mean()
    vol_5 = history['Volume'].iloc[-5:].mean()
    ltq_ratio = vol_2 / vol_5 if vol_5 > 0 else 1.0
    smma120_slope = (history['SMMA120'].iloc[-1] - history['SMMA120'].iloc[-5]) / history['SMMA120'].iloc[-5] if history['SMMA120'].iloc[-5] else 0.0
    hist_vol = history['Close'].iloc[-5:].std() / history['Close'].iloc[-5:].mean() if history['Close'].iloc[-5:].mean() > 0 else 0.0
    current_x = np.array([vol_ratio, ltq_ratio, smma120_slope, hist_vol])
    
    distances = np.linalg.norm(X - current_x, axis=1)
    nearest_indices = np.argsort(distances)[:3]
    nearest_labels = y[nearest_indices]
    
    accept_prob = np.mean(nearest_labels)
    confidence = float(max(accept_prob, 1 - accept_prob) * 100)
    
    latest_diff = history['Diff'].iloc[-1]
    is_bullish = latest_diff > 0
    signal_type = "BULLISH (Golden Cross)" if is_bullish else "BEARISH (Death Cross)"
    action = "ACCEPT" if accept_prob >= 0.5 else "AVOID"
    return f"{action} - {signal_type}", confidence, f"Trained on {len(dataset)} crossover signals using LTQ ratio and trend statistics."

def get_all_historical_crossovers(history):
    """
    Identifies all historical crossovers, runs ML prediction, and performs trade backtesting.
    """
    if len(history) < 130:
        return pd.DataFrame()
        
    history = history.copy()
    if 'SMMA20' not in history.columns:
        history['SMMA20'] = calculate_smma(history['Close'], 20)
    if 'SMMA120' not in history.columns:
        history['SMMA120'] = calculate_smma(history['Close'], 120)
        
    history['Diff'] = history['SMMA20'] - history['SMMA120']
    history['Prev_Diff'] = history['Diff'].shift(1)
    crossover_idx = history[(history['Diff'] * history['Prev_Diff'] < 0) & (history['Diff'].notna()) & (history['Prev_Diff'].notna())].index
    
    records = []
    for idx in crossover_idx:
        pos = history.index.get_loc(idx)
        current_price = history['Close'].iloc[pos]
        is_bullish = history['Diff'].iloc[pos] > 0
        
        exit_price, exit_date, pl_text = 0.0, "N/A", "N/A"
        next_crosses = [c for c in crossover_idx if c > idx]
        
        if is_bullish:
            next_sell = next((c for c in next_crosses if history['Diff'].loc[c] < 0), None)
            exit_price = history['Close'].loc[next_sell] if next_sell else history['Close'].iloc[-1]
            exit_date = next_sell.strftime("%Y-%m-%d") if next_sell else "Open Position"
            pl = exit_price - current_price
            pl_pct = (pl / current_price) * 100
            pl_text = f"₹{pl:+.2f} ({pl_pct:+.1f}%)"
        else:
            next_buy = next((c for c in next_crosses if history['Diff'].loc[c] > 0), None)
            exit_price = history['Close'].loc[next_buy] if next_buy else history['Close'].iloc[-1]
            exit_date = next_buy.strftime("%Y-%m-%d") if next_buy else "Open Position"
            pl = current_price - exit_price
            pl_pct = (pl / current_price) * 100
            pl_text = f"₹{pl:+.2f} ({pl_pct:+.1f}%)"
            
        vol_sma20 = history['Volume'].iloc[max(0, pos-20):pos].mean()
        vol_ratio = history['Volume'].iloc[pos] / vol_sma20 if vol_sma20 > 0 else 1.0
        
        # Calculate LTQ (2-period avg vs 5-period avg volume ratio)
        vol_2 = history['Volume'].iloc[max(0, pos-2):pos].mean()
        vol_5 = history['Volume'].iloc[max(0, pos-5):pos].mean()
        ltq_ratio = vol_2 / vol_5 if vol_5 > 0 else 1.0
        
        smma120_slope = (history['SMMA120'].iloc[pos] - history['SMMA120'].iloc[max(0, pos-5)]) / history['SMMA120'].iloc[max(0, pos-5)] if history['SMMA120'].iloc[max(0, pos-5)] else 0.0
        
        reasons = []
        is_profitable = True
        
        if abs(smma120_slope) < 0.001:
            reasons.append("Flat SMMA(120) / ranging market risk")
            is_profitable = False
        if vol_ratio < 1.1:
            reasons.append("Low volume breakout")
            is_profitable = False
        if ltq_ratio < 1.0:
            reasons.append("LTQ decelerating (2-period avg < 5-period avg)")
            is_profitable = False
        else:
            reasons.append(f"LTQ accelerating ({ltq_ratio:.1f}x)")
            
        action = "ACCEPT" if is_profitable else "AVOID"
        explanation = " | ".join(reasons) if reasons else "Strong trend breakout."
        
        records.append({
            "Date": idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx),
            "Signal": "Buy (Golden)" if is_bullish else "Sell (Death)",
            "Entry LTP (₹)": round(current_price, 2),
            "Exit Date": exit_date,
            "Exit LTP (₹)": round(exit_price, 2) if exit_price else None,
            "Trade P/L": pl_text,
            "ML Recommendation": action,
            "Market Observations": explanation
        })
    return pd.DataFrame(records).sort_values(by="Date", ascending=False).reset_index(drop=True)
