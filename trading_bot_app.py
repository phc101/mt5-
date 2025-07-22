#!/usr/bin/env python3
"""
MT5 Signal Generator with Price Verification
Shows exactly how prices are fetched from Yahoo Finance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import time
import requests

st.set_page_config(
    page_title="MT5 Signals with Price Verification",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .price-verification {
        background: #e8f4f8;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .data-source {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .real-time-indicator {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

FOREX_SYMBOLS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X', 
    'USDCHF': 'USDCHF=X',
    'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X'
}

class VerifiablePriceBot:
    def __init__(self):
        self.lookback_days = 7
        self.data_source = "Yahoo Finance"
        self.last_fetch_time = None
        self.fetch_details = {}
        
    def get_forex_data_with_verification(self, symbol, days=30):
        """Fetch forex data with detailed verification info"""
        fetch_start_time = datetime.now()
        
        try:
            yf_symbol = FOREX_SYMBOLS.get(symbol, f"{symbol}=X")
            
            # Show what we're fetching
            st.info(f"🔄 Fetching {symbol} data from Yahoo Finance...")
            st.code(f"yfinance.Ticker('{yf_symbol}').history()")
            
            ticker = yf.Ticker(yf_symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch the data
            data = ticker.history(
                start=start_date, 
                end=end_date,
                interval="1d"
            )
            
            fetch_end_time = datetime.now()
            fetch_duration = (fetch_end_time - fetch_start_time).total_seconds()
            
            if data.empty:
                st.error(f"❌ No data returned for {symbol}")
                return None, None
                
            # Process data
            df = pd.DataFrame({
                'Date': data.index,
                'Open': data['Open'],
                'High': data['High'],
                'Low': data['Low'],
                'Close': data['Close'],
                'Volume': data['Volume']
            }).reset_index(drop=True)
            
            # Verification details
            verification_info = {
                'symbol': symbol,
                'yahoo_symbol': yf_symbol,
                'fetch_time': fetch_end_time,
                'fetch_duration': fetch_duration,
                'data_points': len(df),
                'date_range': f"{df['Date'].min().date()} to {df['Date'].max().date()}",
                'latest_price': df['Close'].iloc[-1],
                'latest_date': df['Date'].iloc[-1],
                'data_age_hours': (datetime.now() - df['Date'].iloc[-1].to_pydatetime()).total_seconds() / 3600
            }
            
            return df, verification_info
            
        except Exception as e:
            st.error(f"❌ Error fetching data for {symbol}: {str(e)}")
            return None, None
    
    def verify_yahoo_finance_connection(self):
        """Verify connection to Yahoo Finance"""
        st.markdown("### 🔍 Yahoo Finance Connection Test")
        
        try:
            # Test direct connection
            test_url = "https://finance.yahoo.com"
            response = requests.get(test_url, timeout=5)
            
            if response.status_code == 200:
                st.success(f"✅ Yahoo Finance is accessible (Status: {response.status_code})")
            else:
                st.warning(f"⚠️ Yahoo Finance returned status: {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Cannot reach Yahoo Finance: {str(e)}")
    
    def calculate_pivot_points(self, df):
        """Calculate pivot points with verification"""
        if len(df) < self.lookback_days:
            return df
            
        pivot_data = []
        
        for i in range(self.lookback_days, len(df)):
            window = df.iloc[i-self.lookback_days:i]
            avg_high = window['High'].mean()
            avg_low = window['Low'].mean()
            avg_close = window['Close'].mean()
            
            pivot = (avg_high + avg_low + avg_close) / 3
            r2 = pivot + (avg_high - avg_low)
            s2 = pivot - (avg_high - avg_low)
            
            pivot_data.append({
                'Date': df.loc[i, 'Date'],
                'Pivot': pivot,
                'R2': r2,
                'S2': s2,
            })
        
        pivot_df = pd.DataFrame(pivot_data)
        df = df.merge(pivot_df, on='Date', how='left')
        return df

# Initialize bot
bot = VerifiablePriceBot()

# Header
st.title("🔍 MT5 Signals with Price Verification")
st.markdown("**Transparent view of how prices are fetched from Yahoo Finance**")

# Verification section
st.markdown("## 🌐 Data Source Verification")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="data-source">', unsafe_allow_html=True)
    st.markdown("### 📊 Data Source Details")
    st.markdown(f"""
    - **Provider:** Yahoo Finance  
    - **Library:** yfinance (Python)
    - **Update Frequency:** Real-time (15-20 min delay)
    - **Data Type:** OHLCV (Open, High, Low, Close, Volume)
    - **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.button("🔄 Test Yahoo Finance Connection"):
        bot.verify_yahoo_finance_connection()

        # Symbol selection with dynamic chart
st.sidebar.header("⚙️ Configuration")
selected_symbol = st.sidebar.selectbox(
    "Select Currency Pair:",
    list(FOREX_SYMBOLS.keys()),
    index=0,
    key="symbol_selector"
)

# Chart configuration
st.sidebar.subheader("📊 Chart Settings")
chart_days = st.sidebar.slider("Days to Display", 7, 30, 7)
show_volume = st.sidebar.checkbox("Show Volume", value=False)
show_pivot_history = st.sidebar.checkbox("Show Historical Pivots", value=True)

# Data options
st.sidebar.subheader("📋 Data Options")
show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
show_verification = st.sidebar.checkbox("Show Verification Details", value=True)
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)

# Dynamic chart that updates when symbol changes
st.markdown(f"## 📊 Live {selected_symbol} Chart - Last {chart_days} Days")

# Container for dynamic chart
chart_container = st.empty()
data_container = st.empty()

# Function to update chart
def update_chart_and_data():
    with st.spinner(f"Fetching {selected_symbol} data..."):
        df, verification = bot.get_forex_data_with_verification(selected_symbol, max(chart_days + 10, 30))
    
    if df is not None and verification:
        # Calculate pivot points
        df_with_pivots = bot.calculate_pivot_points(df)
        
        # Get chart data
        chart_data = df_with_pivots.tail(chart_days)
        latest_row = df_with_pivots.iloc[-1]
        
        # Create advanced chart
        fig = create_advanced_chart(chart_data, latest_row, selected_symbol, show_volume, show_pivot_history)
        
        with chart_container.container():
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_symbol}")
        
        # Show current levels and signal analysis
        with data_container.container():
            display_current_analysis(latest_row, verification, selected_symbol)
            
            # Show verification details if enabled
            if show_verification:
                display_verification_details(verification)
            
            # Show raw data if enabled
            if show_raw_data:
                display_raw_data(chart_data)
        
        return df_with_pivots, verification
    else:
        with chart_container.container():
            st.error(f"❌ Could not fetch data for {selected_symbol}")
        return None, None

def create_advanced_chart(df, latest_row, symbol, show_volume, show_pivot_history):
    """Create advanced OHLC chart with pivot points"""
    
    # Create subplots
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
    else:
        fig = go.Figure()
    
    # OHLC Candlestick
    candlestick = go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=f'{symbol} OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Current pivot points
    if pd.notna(latest_row.get('Pivot')):
        current_pivots = {
            'R2': (latest_row['R2'], 'red', 'dash'),
            'R1': (latest_row.get('R1', latest_row['Pivot']), 'orange', 'dot'), 
            'Pivot': (latest_row['Pivot'], 'purple', 'solid'),
            'S1': (latest_row.get('S1', latest_row['Pivot']), 'blue', 'dot'),
            'S2': (latest_row['S2'], 'darkblue', 'dash')
        }
        
        for level_name, (level_value, color, line_style) in current_pivots.items():
            if pd.notna(level_value):
                line_params = dict(
                    y=level_value,
                    line_dash=line_style,
                    line_color=color,
                    line_width=2,
                    annotation_text=f"{level_name}: {level_value:.5f}",
                    annotation_position="bottom right",
                    annotation_font_size=10
                )
                
                if show_volume:
                    fig.add_hline(row=1, col=1, **line_params)
                else:
                    fig.add_hline(**line_params)
    
    # Historical pivot points
    if show_pivot_history and 'Pivot' in df.columns:
        pivot_data = df.dropna(subset=['Pivot']).tail(chart_days)
        
        if len(pivot_data) > 1:
            # Pivot line
            pivot_line = go.Scatter(
                x=pivot_data['Date'],
                y=pivot_data['Pivot'],
                mode='lines',
                name='Historical Pivot',
                line=dict(color='purple', width=1, dash='dot'),
                opacity=0.7
            )
            
            if show_volume:
                fig.add_trace(pivot_line, row=1, col=1)
            else:
                fig.add_trace(pivot_line)
    
    # Volume chart
    if show_volume and 'Volume' in df.columns:
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(df['Close'], df['Open'])]
        
        volume_bars = go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        )
        fig.add_trace(volume_bars, row=2, col=1)
    
    # Current price marker
    current_price = latest_row['Close']
    current_date = latest_row['Date']
    
    price_marker = go.Scatter(
        x=[current_date],
        y=[current_price],
        mode='markers+text',
        marker=dict(
            size=12,
            color='orange',
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        text=[f'{current_price:.5f}'],
        textposition='top center',
        name='Current Price',
        showlegend=True
    )
    
    if show_volume:
        fig.add_trace(price_marker, row=1, col=1)
    else:
        fig.add_trace(price_marker)
    
    # Update layout
    title_text = f"{symbol} - Last {chart_days} Days | Current: {current_price:.5f}"
    
    fig.update_layout(
        title=title_text,
        title_font_size=16,
        height=600 if show_volume else 500,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    if show_volume:
        fig.update_yaxes(title_text="Price", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=True, gridwidth=1, gridcolor='lightgray')
    else:
        fig.update_yaxes(title_text="Price", showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def display_current_analysis(latest_row, verification, symbol):
    """Display current price analysis and signals"""
    current_price = latest_row['Close']
    
    st.markdown("### 🎯 Current Market Analysis")
    
    # Price metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"{current_price:.5f}")
        
    with col2:
        if pd.notna(latest_row.get('Pivot')):
            pivot_diff = current_price - latest_row['Pivot']
            st.metric("vs Pivot", f"{pivot_diff:+.5f}", delta=f"{(pivot_diff/latest_row['Pivot']*100):+.2f}%")
        else:
            st.metric("vs Pivot", "N/A")
    
    with col3:
        # Daily change
        if len(verification) > 0 and 'data_points' in verification and verification['data_points'] > 1:
            prev_close = latest_row.get('Open', current_price)  # Approximation
            daily_change = current_price - prev_close
            st.metric("Daily Change", f"{daily_change:+.5f}", delta=f"{(daily_change/prev_close*100):+.2f}%")
        else:
            st.metric("Daily Change", "N/A")
    
    with col4:
        st.metric("Data Age", f"{verification.get('data_age_hours', 0):.1f}h")
    
    # Signal analysis
    if pd.notna(latest_row.get('S2')) and pd.notna(latest_row.get('R2')):
        signal = None
        signal_strength = "NEUTRAL"
        
        if current_price < latest_row['S2']:
            signal = "🟢 BUY SIGNAL"
            signal_strength = "STRONG"
            signal_reason = f"Price {current_price:.5f} broke below S2 level {latest_row['S2']:.5f}"
        elif current_price > latest_row['R2']:
            signal = "🔴 SELL SIGNAL"
            signal_strength = "STRONG"
            signal_reason = f"Price {current_price:.5f} broke above R2 level {latest_row['R2']:.5f}"
        elif current_price < latest_row['S1']:
            signal = "🟡 WEAK BUY"
            signal_strength = "WEAK"
            signal_reason = f"Price {current_price:.5f} below S1 level {latest_row['S1']:.5f}"
        elif current_price > latest_row['R1']:
            signal = "🟡 WEAK SELL"
            signal_strength = "WEAK" 
            signal_reason = f"Price {current_price:.5f} above R1 level {latest_row['R1']:.5f}"
        
        if signal:
            if signal_strength == "STRONG":
                st.success(f"**{signal}** - {signal_reason}")
            else:
                st.warning(f"**{signal}** - {signal_reason}")
        else:
            st.info(f"**No Signal** - Price {current_price:.5f} between pivot levels")
        
        # Pivot levels table
        st.markdown("#### 📊 Current Pivot Levels")
        pivot_df = pd.DataFrame({
            'Level': ['R2', 'R1', 'Pivot', 'S1', 'S2'],
            'Value': [
                latest_row['R2'],
                latest_row.get('R1', 'N/A'),
                latest_row['Pivot'], 
                latest_row.get('S1', 'N/A'),
                latest_row['S2']
            ],
            'Distance': [
                f"{((latest_row['R2'] - current_price) / current_price * 100):+.2f}%" if pd.notna(latest_row['R2']) else 'N/A',
                f"{((latest_row.get('R1', current_price) - current_price) / current_price * 100):+.2f}%" if pd.notna(latest_row.get('R1')) else 'N/A',
                f"{((latest_row['Pivot'] - current_price) / current_price * 100):+.2f}%" if pd.notna(latest_row['Pivot']) else 'N/A',
                f"{((latest_row.get('S1', current_price) - current_price) / current_price * 100):+.2f}%" if pd.notna(latest_row.get('S1')) else 'N/A',
                f"{((latest_row['S2'] - current_price) / current_price * 100):+.2f}%" if pd.notna(latest_row['S2']) else 'N/A'
            ]
        })
        st.dataframe(pivot_df, use_container_width=True, hide_index=True)

def display_verification_details(verification):
    """Display data verification details"""
    st.markdown('<div class="price-verification">', unsafe_allow_html=True)
    st.markdown("### ✅ Data Verification")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fetch Duration", f"{verification['fetch_duration']:.2f}s")
        st.metric("Data Points", verification['data_points'])
    
    with col2:
        st.metric("Yahoo Symbol", verification['yahoo_symbol'])
        st.metric("Date Range", verification['date_range'])
    
    with col3:
        st.metric("Fetch Time", verification['fetch_time'].strftime('%H:%M:%S'))
        st.markdown(f"**Latest Date:** {verification['latest_date'].strftime('%Y-%m-%d')}")
    
    st.markdown(f"🔗 **Verify at:** https://finance.yahoo.com/quote/{verification['yahoo_symbol']}")
    st.markdown('</div>', unsafe_allow_html=True)

def display_raw_data(df):
    """Display raw OHLC data"""
    st.markdown("### 📋 Raw OHLC Data")
    display_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Format numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close']
    for col in numeric_cols:
        display_df[col] = display_df[col].round(5)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# Update chart when symbol changes or on refresh
update_chart_and_data()

# Manual refresh button
if st.button("🔄 Refresh Data", type="primary"):
    update_chart_and_data()

# Auto refresh logic  
if auto_refresh:
    time.sleep(2)
    st.rerun()

# Manual verification section
st.markdown("---")
st.markdown("## 🔍 Manual Verification Steps")

with st.expander("How to verify prices yourself"):
    st.markdown(f"""
    ### 1. Compare with Yahoo Finance Website:
    - Go to: https://finance.yahoo.com/quote/{FOREX_SYMBOLS.get(selected_symbol, 'EURUSD=X')}
    - Compare the current price with what's shown above
    
    ### 2. Check yfinance directly in Python:
    ```python
    import yfinance as yf
    ticker = yf.Ticker('{FOREX_SYMBOLS.get(selected_symbol, 'EURUSD=X')}')
    data = ticker.history(period="1d")
    print("Latest price:", data['Close'][-1])
    ```
    
    ### 3. Verify data freshness:
    - Market hours: Forex trades 24/5 (Sunday 5 PM - Friday 5 PM EST)
    - Data delay: ~15-20 minutes during market hours
    - Weekend: Shows Friday's closing price
    
    ### 4. Alternative verification:
    - Compare with MT5 Android app prices
    - Check other sources: Investing.com, TradingView
    - Verify timestamps match recent market activity
    """)

# Real-time status
st.markdown("---")
current_time = datetime.now()
market_status = "🟢 OPEN" if current_time.weekday() < 5 else "🔴 CLOSED (Weekend)"

st.markdown(f"""
**🕐 Current Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC  
**📈 Forex Market:** {market_status}  
**🔄 Last Data Fetch:** {bot.last_fetch_time or 'Not yet fetched'}  
**🌐 Data Source:** Yahoo Finance API via yfinance library
""")

# Auto-refresh option
if st.sidebar.checkbox("Auto-refresh every 60s"):
    time.sleep(2)
    st.rerun()
