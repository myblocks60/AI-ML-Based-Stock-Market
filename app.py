import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_fetcher import get_nse_symbols, fetch_all_nse_prices
from screener import screen_stocks, get_detailed_stock_history

# Set page configuration with premium dark layout styling
st.set_page_config(
    page_title="AeroScreen - Real-Time NSE Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium look with dark-mode aesthetic
st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 AeroScreen")
st.subheader("Real-Time NSE Stock Screener & Technical Analyzer")

# Sidebar Configuration
st.sidebar.header("Filter Criteria")
min_price, max_price = st.sidebar.slider(
    "LTP Range (₹)",
    min_value=10.0,
    max_value=1000.0,
    value=(30.0, 500.0),
    step=5.0
)

# Limit the number of symbols scanned to speed up Streamlit load time, or scan full list
scan_limit = st.sidebar.number_input("Maximum Stocks to Scan", min_value=10, max_value=2000, value=200)

@st.cache_data(ttl=300)
def load_and_fetch_stock_data(limit):
    symbols = get_nse_symbols()[:limit]
    return fetch_all_nse_prices(symbols)

with st.spinner("Fetching real-time stock prices..."):
    raw_data = load_and_fetch_stock_data(scan_limit)

# Screen the stocks
df_screened = screen_stocks(raw_data, min_price=min_price, max_price=max_price)

# Top KPI metrics
if not df_screened.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    top_gainer = df_screened.loc[df_screened["Change (%)"].idxmax()]
    top_loser = df_screened.loc[df_screened["Change (%)"].idxmin()]
    
    col1.metric("Total Scanned", f"{len(raw_data)}")
    col2.metric("Screened (₹30-₹500)", f"{len(df_screened)}")
    col3.metric("Top Gainer", f"{top_gainer['Symbol']}", f"+{top_gainer['Change (%)']}%")
    col4.metric("Top Loser", f"{top_loser['Symbol']}", f"{top_loser['Change (%)']}%")

# Search and table display
st.write("### Screened Stocks")
search_term = st.text_input("🔍 Search symbol...", "")
if search_term:
    df_display = df_screened[df_screened["Symbol"].str.contains(search_term.upper(), na=False)]
else:
    df_display = df_screened

st.dataframe(df_display, use_container_width=True, hide_index=True)

# Detailed analysis section
if not df_display.empty:
    st.write("---")
    st.write("### 🔍 Technical Chart & Analysis")
    
    selected_symbol = st.selectbox(
        "Select a stock to analyze:",
        options=df_display["Symbol"].tolist()
    )
    
    if selected_symbol:
        ticker = f"{selected_symbol}.NS"
        with st.spinner(f"Loading history for {selected_symbol}..."):
            history = get_detailed_stock_history(ticker, period="3mo", interval="1d")
            
        if not history.empty:
            # Interactive Candlestick Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=history.index,
                open=history['Open'],
                high=history['High'],
                low=history['Low'],
                close=history['Close'],
                name="Price"
            ))
            
            # Simple Moving Average lines
            history['SMA20'] = history['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=history.index,
                y=history['SMA20'],
                line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5),
                name="20 SMA"
            ))
            
            fig.update_layout(
                title=f"{selected_symbol} - 3 Month Price Trend",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=40, b=20),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No historical data available for the selected symbol.")
else:
    st.info("No stocks match the screening criteria.")
