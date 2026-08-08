import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from data_fetcher import get_nse_symbols, fetch_all_nse_prices
from screener import screen_stocks, get_detailed_stock_history, calculate_smma, calculate_batch_metrics
from ml_analysis import analyze_crossover_signal, get_all_historical_crossovers

# Set page configuration with premium dark layout styling
st.set_page_config(
    page_title="AeroScreen - Live NSE Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium look with dark-mode aesthetic
st.markdown("<style>.reportview-container { background: #0d1117; }</style>", unsafe_allow_html=True)

st.title("📈 AeroScreen Live Dashboard")
st.subheader("Real-Time Tabular Stock Market Screening & Analysis")

# Sidebar Configuration
st.sidebar.header("Filter Criteria")
min_price, max_price = st.sidebar.slider(
    "LTP Range (₹)", min_value=10.0, max_value=1000.0, value=(30.0, 500.0), step=5.0
)

apply_liquidity = st.sidebar.checkbox("Filter by Liquidity (Bid/Ask)", value=True)
if apply_liquidity:
    min_bid = st.sidebar.number_input("Min Bid Quantity", min_value=0, value=1000000, step=100000)
    min_ask = st.sidebar.number_input("Min Ask Quantity", min_value=0, value=1000000, step=100000)
else:
    min_bid, min_ask = 0, 0

scan_limit = st.sidebar.number_input("Maximum Stocks to Scan", min_value=10, max_value=500, value=50)
refresh_interval = st.sidebar.slider("Auto Refresh Interval (sec)", min_value=5, max_value=60, value=10, step=5)

# Fyers Broker Integration
st.sidebar.markdown("---")
st.sidebar.header("🔌 Fyers Broker API")

if "fyers_token" not in st.session_state:
    st.session_state["fyers_token"] = None
if "fyers_client_id" not in st.session_state:
    st.session_state["fyers_client_id"] = ""

if st.session_state["fyers_token"]:
    st.sidebar.success("🟢 Authenticated with Fyers")
    if st.sidebar.button("Logout"):
        st.session_state["fyers_token"] = None
        st.rerun()
else:
    st.sidebar.info("🔴 Running Demo Mode (YFinance)")
    fyers_client_id = st.sidebar.text_input("App ID (Client ID)", value=st.session_state["fyers_client_id"], type="password")
    fyers_secret = st.sidebar.text_input("Secret ID", type="password")

    fyers_redirect = st.sidebar.text_input("Redirect URI", value="http://localhost:8501/")
    
    if fyers_client_id and fyers_secret and fyers_redirect:
        from fyers_client import get_session_model, generate_auth_link, get_access_token
        try:
            session = get_session_model(fyers_client_id, fyers_secret, fyers_redirect)
            auth_url = generate_auth_link(session)
            st.sidebar.markdown(f"[🔗 Generate Auth Code]({auth_url})", unsafe_allow_html=True)
            auth_code = st.sidebar.text_input("Enter Redirect URL or Auth Code:")
            if auth_code:
                if "auth_code=" in auth_code:
                    auth_code = auth_code.split("auth_code=")[1].split("&")[0]
                token = get_access_token(session, auth_code)
                if token:
                    st.session_state["fyers_token"] = token
                    st.session_state["fyers_client_id"] = fyers_client_id
                    st.rerun()
        except Exception as e:
            st.sidebar.error(f"Config Error: {e}")

@st.cache_data(ttl=10)
def load_and_fetch_stock_data(limit, fyers_token, fyers_client_id):
    symbols = get_nse_symbols()[:limit]
    return fetch_all_nse_prices(
        symbols, fyers_token=fyers_token, fyers_client_id=fyers_client_id
    )


with st.spinner("Fetching real-time stock data..."):
    raw_data = load_and_fetch_stock_data(
        scan_limit, st.session_state["fyers_token"], st.session_state["fyers_client_id"]
    )


# Screen the stocks based on Price and Liquidity
df_screened = screen_stocks(
    raw_data, min_price=min_price, max_price=max_price, min_bid_qty=min_bid, min_ask_qty=min_ask
)

# Batch calculate technical & volume metrics for screened stocks
if not df_screened.empty:
    with st.spinner("Calculating technical indicators & ETQs in batch..."):
        batch_metrics = calculate_batch_metrics(df_screened['Ticker'].tolist())
        
    df_screened['SMMA(20)'] = df_screened['Ticker'].map(lambda x: batch_metrics.get(x, {}).get('smma20'))
    df_screened['SMMA(120)'] = df_screened['Ticker'].map(lambda x: batch_metrics.get(x, {}).get('smma120'))
    df_screened['ETQ 5m'] = df_screened['Ticker'].map(lambda x: batch_metrics.get(x, {}).get('etq5', 0))
    df_screened['ETQ 20m'] = df_screened['Ticker'].map(lambda x: batch_metrics.get(x, {}).get('etq20', 0))
    df_screened['ETQ 60m'] = df_screened['Ticker'].map(lambda x: batch_metrics.get(x, {}).get('etq60', 0))
    df_screened['Avg Price 20m'] = df_screened['Ticker'].map(lambda x: batch_metrics.get(x, {}).get('avg20', 0.0))
    df_screened['Avg Price 60m'] = df_screened['Ticker'].map(lambda x: batch_metrics.get(x, {}).get('avg60', 0.0))

# Top KPI metrics
if not df_screened.empty:
    col1, col2, col3, col4 = st.columns(4)
    top_gainer = df_screened.loc[df_screened["Change (%)"].idxmax()]
    top_loser = df_screened.loc[df_screened["Change (%)"].idxmin()]
    col1.metric("Total Scanned", f"{len(raw_data)}")
    col2.metric("Screened", f"{len(df_screened)}")
    col3.metric("Top Gainer", f"{top_gainer['Symbol']}", f"+{top_gainer['Change (%)']}%")
    col4.metric("Top Loser", f"{top_loser['Symbol']}", f"{top_loser['Change (%)']}%")

# Search and tabular live dashboard display
st.write("### 📊 Live Tabular Dashboard")
search_term = st.text_input("🔍 Search symbol...", "")
df_display = df_screened[df_screened["Symbol"].str.contains(search_term.upper(), na=False)] if search_term else df_screened

# Display all real-time parameters in one live table
st.dataframe(df_display, use_container_width=True, hide_index=True)

# Technical analysis and detailed visualization section
if not df_display.empty:
    st.write("---")
    st.write("### 🔍 Technical Chart & Analysis")
    selected_symbol = st.selectbox("Select a stock to analyze:", options=df_display["Symbol"].tolist())
    
    if selected_symbol:
        ticker = f"{selected_symbol}.NS"
        row = df_display[df_display["Symbol"] == selected_symbol].iloc[0]
        
        # Display all metrics side by side
        c1, c2, c3 = st.columns(3)
        c1.write("**📖 Market Depth & Last Trade**")
        c1.success(f"Best Bid: ₹{row.get('Bid Price', 0.0):,.2f} ({int(row.get('Bid Qty', 0)):,})")
        c1.error(f"Best Ask: ₹{row.get('Ask Price', 0.0):,.2f} ({int(row.get('Ask Qty', 0)):,})")
        c1.info(f"**Last Traded Qty (LTQ)**: {int(row.get('LTQ', 0)):,}")
        
        c2.write("**📊 Exchange Traded Quantity (ETQ)**")
        c2.info(f"Last 5m: {int(row.get('ETQ 5m', 0)):,}\n\nLast 20m: {int(row.get('ETQ 20m', 0)):,}\n\nLast 60m: {int(row.get('ETQ 60m', 0)):,}")
        
        c3.write("**💵 Average Price & Indicators**")
        c3.info(f"Avg LTP 20m: ₹{row.get('Avg Price 20m', 0.0):,.2f}\n\nAvg LTP 60m: ₹{row.get('Avg Price 60m', 0.0):,.2f}\n\nSMMA(20): ₹{row.get('SMMA(20)'):,.2f}\n\nSMMA(120): ₹{row.get('SMMA(120)'):,.2f}")

        
        with st.spinner("Loading candlestick history & running AI/ML analysis..."):
            history = get_detailed_stock_history(ticker, period="1y", interval="1d")
            
        if not history.empty:
            history['SMMA20'] = calculate_smma(history['Close'], 20)
            history['SMMA120'] = calculate_smma(history['Close'], 120)
            
            # Run AI/ML Signal Analysis
            signal, confidence, ml_desc = analyze_crossover_signal(
                history, bid_qty=row.get('Bid Qty', 0), ask_qty=row.get('Ask Qty', 0)
            )
            
            st.write("---")
            st.write("### 🤖 AI/ML Crossover Signal Analysis")
            if "ACCEPT" in signal:
                st.success(f"**Current Recommendation: {signal}**\n\n**Confidence**: {confidence:.1f}%\n\n*Details: {ml_desc}*")
            else:
                st.warning(f"**Current Recommendation: {signal}**\n\n**Confidence**: {confidence:.1f}%\n\n*Details: {ml_desc}*")
            
            # Display all detected crossovers
            st.write("**📜 Detected Historical Crossover Signals & ML Evaluations**")
            df_crossovers = get_all_historical_crossovers(history)
            if not df_crossovers.empty:
                st.dataframe(df_crossovers, use_container_width=True, hide_index=True)
            else:
                st.info("No historical SMMA crossover signals detected for this stock in the 1-year period.")
            st.write("---")


            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name="Price"
            ))
            fig.add_trace(go.Scatter(x=history.index, y=history['SMMA20'], line=dict(color='#ff9f43', width=1.5), name="SMMA 20"))
            fig.add_trace(go.Scatter(x=history.index, y=history['SMMA120'], line=dict(color='#00d2d3', width=1.5), name="SMMA 120"))
            fig.update_layout(
                title=f"{selected_symbol} - 1 Year Price & SMMA Trends", template="plotly_dark", xaxis_rangeslider_visible=False, height=450
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No stocks match the screening criteria.")

# Auto-refresh loop
time.sleep(refresh_interval)
st.rerun()
