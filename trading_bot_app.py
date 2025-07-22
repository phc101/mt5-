#!/usr/bin/env python3
"""
MT5 Signal Generator with Dynamic Charts
Fixed version with proper error handling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="MT5 Trading Signals with Dynamic Charts",
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
    .signal-strong {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .signal-weak {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Forex symbols mapping
FOREX_SYMBOLS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X', 
    'AUDUSD': 'AUDUSD=X',
    'NZDUSD': 'NZDUSD=X',
    'USDCAD': 'USDCAD=X',
    'USDCHF': 'USDCHF=X',
    'USDJPY': 'USDJPY=X',
    'EURJPY': 'EURJPY=X',
    'GBPJPY': 'GBPJPY=X',
    'EURGBP': 'EURGBP=X'
}

class ForexTradingBot:
    def __init__(self):
        self.lookback_days = 7
        
    def get_forex_data(self, symbol, days=30):
        """Fetch forex data with robust error handling"""
        try:
            yf_symbol = FOREX_SYMBOLS.get(symbol, f"{symbol}=X")
            ticker = yf.Ticker(yf_symbol)
            
            # Try different methods to fetch data
            data = None
            
            # Method 1: Using period
            try:
                data = ticker.history(period=f"{days}d", interval="1d")
                if not data.empty:
                    st.success(f"✅ Fetched {len(data)} days of data for {symbol}")
            except Exception as e1:
                st.warning(f"Method 1 failed: {str(e1)[:100]}...")
            
            # Method 2: Using date range if Method 1 failed
            if data is None or data.empty:
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days + 5)
                    data = ticker.history(start=start_date, end=end_date, interval="1d")
                    if not data.empty:
                        st.success(f"✅ Fetched {len(data)} days using date range")
                except Exception as e2:
                    st.warning(f"Method 2 failed: {str(e2)[:100]}...")
            
            if data is None or data.empty:
                st.error(f"❌ Could not fetch data for {symbol}")
                return None, None
            
            # Clean and process data
            data = data.dropna()
            
            # Handle timezone issues
            if hasattr(data.index, 'tz_localize'):
                try:
                    if data.index.tz is not None:
                        data.index = data.index.tz_convert(None)
                    else:
                        data.index = data.index.tz_localize(None)
                except:
                    pass
            
            # Create clean DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime(data.index),
                'Open': data['Open'].astype(float),
                'High': data['High'].astype(float), 
                'Low': data['Low'].astype(float),
                'Close': data['Close'].astype(float),
                'Volume': data['Volume'].astype(float) if 'Volume' in data.columns else 0
            }).reset_index(drop=True)
            
            # Remove any NaN values
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            if len(df) < 5:
                st.error(f"❌ Insufficient data for {symbol}")
                return None, None
            
            # Verification info
            verification = {
                'symbol': symbol,
                'yahoo_symbol': yf_symbol,
                'data_points': len(df),
                'latest_price': float(df['Close'].iloc[-1]),
                'latest_date': df['Date'].iloc[-1],
                'fetch_time': datetime.now()
            }
            
            return df, verification
            
        except Exception as e:
            st.error(f"❌ Error fetching {symbol}: {str(e)}")
            return None, None
    
    def calculate_pivot_points(self, df):
        """Calculate pivot points"""
        if len(df) < self.lookback_days:
            return df
            
        pivot_data = []
        
        for i in range(self.lookback_days, len(df)):
            window = df.iloc[i-self.lookback_days:i]
            avg_high = window['High'].mean()
            avg_low = window['Low'].mean()
            avg_close = window['Close'].mean()
            
            pivot = (avg_high + avg_low + avg_close) / 3
            r1 = 2 * pivot - avg_low
            r2 = pivot + (avg_high - avg_low)
            s1 = 2 * pivot - avg_high
            s2 = pivot - (avg_high - avg_low)
            
            pivot_data.append({
                'Date': df.loc[i, 'Date'],
                'Pivot': pivot,
                'R1': r1,
                'R2': r2,
                'S1': s1,
                'S2': s2,
            })
        
        if pivot_data:
            pivot_df = pd.DataFrame(pivot_data)
            df = df.merge(pivot_df, on='Date', how='left')
        
        return df

# Initialize bot
@st.cache_resource
def get_bot():
    return ForexTradingBot()

bot = get_bot()

# Header
st.title("📈 MT5 Trading Signals - Dynamic Charts")
st.markdown("**Live Forex data with Pivot Points analysis**")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
selected_symbol = st.sidebar.selectbox(
    "Select Currency Pair:",
    list(FOREX_SYMBOLS.keys()),
    index=0
)

chart_days = st.sidebar.slider("Chart Days", 7, 30, 14)
show_volume = st.sidebar.checkbox("Show Volume", False)
show_pivots = st.sidebar.checkbox("Show Pivot Levels", True)
auto_refresh = st.sidebar.checkbox("Auto Refresh (60s)", False)

# Main chart section
st.markdown(f"## 📊 Live {selected_symbol} Chart - Last {chart_days} Days")

def create_chart(df, symbol, days, show_vol, show_piv):
    """Create OHLC chart with pivot points"""
    try:
        if df is None or len(df) == 0:
            st.error("No data available for chart")
            return None
        
        # Get recent data
        chart_data = df.tail(days)
        
        if show_vol:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[f'{symbol} Price', 'Volume'],
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=chart_data['Date'],
            open=chart_data['Open'],
            high=chart_data['High'],
            low=chart_data['Low'],
            close=chart_data['Close'],
            name=f'{symbol} OHLC'
        )
        
        if show_vol:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Add pivot levels
        if show_piv:
            latest_row = chart_data.iloc[-1]
            if pd.notna(latest_row.get('Pivot')):
                
                levels = {
                    'R2': (latest_row.get('R2'), 'red'),
                    'R1': (latest_row.get('R1'), 'orange'),
                    'Pivot': (latest_row['Pivot'], 'purple'),
                    'S1': (latest_row.get('S1'), 'blue'),
                    'S2': (latest_row.get('S2'), 'darkblue')
                }
                
                for name, (value, color) in levels.items():
                    if pd.notna(value):
                        hline_args = {
                            'y': value,
                            'line_dash': 'dash',
                            'line_color': color,
                            'annotation_text': f'{name}: {value:.5f}'
                        }
                        if show_vol:
                            fig.add_hline(row=1, col=1, **hline_args)
                        else:
                            fig.add_hline(**hline_args)
        
        # Add volume
        if show_vol and 'Volume' in chart_data.columns:
            colors = ['green' if c >= o else 'red' for c, o in zip(chart_data['Close'], chart_data['Open'])]
            fig.add_trace(
                go.Bar(x=chart_data['Date'], y=chart_data['Volume'], 
                      name='Volume', marker_color=colors),
                row=2, col=1
            )
        
        # Current price marker
        current_price = chart_data['Close'].iloc[-1]
        current_date = chart_data['Date'].iloc[-1]
        
        price_marker = go.Scatter(
            x=[current_date],
            y=[current_price],
            mode='markers+text',
            marker=dict(size=12, color='orange'),
            text=[f'{current_price:.5f}'],
            textposition='top center',
            name='Current Price'
        )
        
        if show_vol:
            fig.add_trace(price_marker, row=1, col=1)
        else:
            fig.add_trace(price_marker)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Current: {current_price:.5f}",
            height=600 if show_vol else 400,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

# Fetch data and create chart
with st.spinner(f"Loading {selected_symbol} data..."):
    df, verification = bot.get_forex_data(selected_symbol, max(chart_days + 10, 30))

if df is not None and verification:
    # Calculate pivots
    df_with_pivots = bot.calculate_pivot_points(df)
    
    # Create and display chart
    fig = create_chart(df_with_pivots, selected_symbol, chart_days, show_volume, show_pivots)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Current analysis
    st.markdown("### 🎯 Current Analysis")
    
    latest_row = df_with_pivots.iloc[-1]
    current_price = latest_row['Close']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"{current_price:.5f}")
    
    with col2:
        if pd.notna(latest_row.get('Pivot')):
            pivot_diff = current_price - latest_row['Pivot']
            st.metric("vs Pivot", f"{pivot_diff:+.5f}")
        else:
            st.metric("vs Pivot", "N/A")
    
    with col3:
        st.metric("Data Points", verification['data_points'])
    
    with col4:
        st.metric("Latest Date", verification['latest_date'].strftime('%Y-%m-%d'))
    
    # Signal analysis
    if pd.notna(latest_row.get('S2')) and pd.notna(latest_row.get('R2')):
        
        # Determine signal
        signal_type = None
        signal_strength = "NEUTRAL"
        
        if current_price < latest_row['S2']:
            signal_type = "🟢 BUY SIGNAL"
            signal_strength = "STRONG"
            signal_reason = f"Price {current_price:.5f} below S2 level {latest_row['S2']:.5f}"
            css_class = "signal-strong"
        elif current_price > latest_row['R2']:
            signal_type = "🔴 SELL SIGNAL"
            signal_strength = "STRONG"
            signal_reason = f"Price {current_price:.5f} above R2 level {latest_row['R2']:.5f}"
            css_class = "signal-strong"
        elif current_price < latest_row.get('S1', latest_row['Pivot']):
            signal_type = "🟡 Weak Buy"
            signal_strength = "WEAK"
            signal_reason = f"Price {current_price:.5f} below S1"
            css_class = "signal-weak"
        elif current_price > latest_row.get('R1', latest_row['Pivot']):
            signal_type = "🟡 Weak Sell"
            signal_strength = "WEAK"
            signal_reason = f"Price {current_price:.5f} above R1"
            css_class = "signal-weak"
        
        if signal_type:
            st.markdown(f'<div class="{css_class}"><h4>{signal_type}</h4><p>{signal_reason}</p></div>', 
                       unsafe_allow_html=True)
        else:
            st.info(f"**No Signal** - Price {current_price:.5f} between pivot levels")
        
        # Pivot levels table
        st.markdown("#### 📊 Current Pivot Levels")
        
        levels_data = []
        for level_name in ['R2', 'R1', 'Pivot', 'S1', 'S2']:
            value = latest_row.get(level_name)
            if pd.notna(value):
                distance = ((value - current_price) / current_price * 100)
                levels_data.append({
                    'Level': level_name,
                    'Value': f"{value:.5f}",
                    'Distance': f"{distance:+.2f}%"
                })
        
        if levels_data:
            levels_df = pd.DataFrame(levels_data)
            st.dataframe(levels_df, use_container_width=True, hide_index=True)

    # Verification details
    with st.expander("🔍 Data Verification"):
        st.markdown(f"""
        **Symbol:** {verification['symbol']} ({verification['yahoo_symbol']})  
        **Data Points:** {verification['data_points']}  
        **Latest Price:** {verification['latest_price']:.5f}  
        **Fetch Time:** {verification['fetch_time'].strftime('%H:%M:%S')}  
        **Verify at:** https://finance.yahoo.com/quote/{verification['yahoo_symbol']}
        """)
        
        # Show recent data
        recent_data = df_with_pivots.tail(5)[['Date', 'Open', 'High', 'Low', 'Close']].copy()
        recent_data['Date'] = recent_data['Date'].dt.strftime('%Y-%m-%d')
        for col in ['Open', 'High', 'Low', 'Close']:
            recent_data[col] = recent_data[col].round(5)
        st.dataframe(recent_data, use_container_width=True, hide_index=True)

else:
    st.error("Could not load data. Please try a different currency pair or refresh.")

# Manual refresh
if st.button("🔄 Refresh Data", type="primary"):
    st.cache_resource.clear()
    st.rerun()

# Auto refresh
if auto_refresh:
    time.sleep(3)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
**🕐 Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**⚠️ For Demo Trading Only** - Always test on demo accounts first  
**📊 Data Source:** Yahoo Finance with ~15min delay
""")
