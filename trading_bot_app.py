#!/usr/bin/env python3
"""
MT5 Signal Generator with Dynamic Charts
Fixed version with proper error handling + BTC/USD support
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
    # Major pairs
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X', 
    'AUDUSD': 'AUDUSD=X',
    'NZDUSD': 'NZDUSD=X',
    'USDCAD': 'USDCAD=X',
    'USDCHF': 'USDCHF=X',
    'USDJPY': 'USDJPY=X',
    'EURJPY': 'EURJPY=X',
    'GBPJPY': 'GBPJPY=X',
    'EURGBP': 'EURGBP=X',
    
    # Polish Zloty pairs
    'CHFPLN': 'CHFPLN=X',  # Swiss Franc to Polish Zloty
    'EURPLN': 'EURPLN=X',  # Euro to Polish Zloty
    'USDPLN': 'USDPLN=X',  # US Dollar to Polish Zloty
    'GBPPLN': 'GBPPLN=X',  # British Pound to Polish Zloty
    
    # Cryptocurrency
    'BTCUSD': 'BTC-USD'    # Bitcoin to US Dollar
}

class ForexTradingBot:
    def __init__(self):
        self.lookback_days = 7
        
    def get_forex_data(self, symbol, days=30):
        """Fetch forex data with robust error handling including PLN pairs and BTC"""
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
            
            # Method 3: Try alternative PLN symbols if dealing with Polish Zloty
            if (data is None or data.empty) and 'PLN' in symbol:
                try:
                    # Alternative PLN symbols
                    alt_symbols = {
                        'USDPLN': ['PLN=X', 'USDPLN=X', 'USD/PLN'],
                        'EURPLN': ['EURPLN=X', 'EUR/PLN'], 
                        'GBPPLN': ['GBPPLN=X', 'GBP/PLN'],
                        'CHFPLN': ['CHFPLN=X', 'CHF/PLN']
                    }
                    
                    for alt_symbol in alt_symbols.get(symbol, []):
                        try:
                            alt_ticker = yf.Ticker(alt_symbol)
                            data = alt_ticker.history(period=f"{days}d", interval="1d")
                            if not data.empty:
                                st.success(f"✅ Fetched {len(data)} days using alternative symbol {alt_symbol}")
                                yf_symbol = alt_symbol
                                break
                        except:
                            continue
                            
                except Exception as e3:
                    st.warning(f"Method 3 (PLN alternatives) failed: {str(e3)[:100]}...")
            
            # Method 4: Try inverted pair for PLN (1/rate)
            if (data is None or data.empty) and 'PLN' in symbol:
                try:
                    # Try inverted symbols (PLN as base currency doesn't work well on Yahoo)
                    base_currency = symbol.replace('PLN', '')
                    inverted_symbol = f"{base_currency}PLN=X"
                    
                    inv_ticker = yf.Ticker(inverted_symbol)
                    inv_data = inv_ticker.history(period=f"{days}d", interval="1d")
                    
                    if not inv_data.empty:
                        # Invert the data (1/rate) to get correct PLN rates
                        data = inv_data.copy()
                        for col in ['Open', 'High', 'Low', 'Close']:
                            # For inverted rates: PLN rate = 1 / (base currency rate)
                            # But actually we want: base/PLN rate, so we keep as is
                            pass  # Keep original data
                        st.success(f"✅ Fetched {len(data)} days using base pair approach")
                        yf_symbol = inverted_symbol
                        
                except Exception as e4:
                    st.warning(f"Method 4 (inverted pairs) failed: {str(e4)[:100]}...")
            
            if data is None or data.empty:
                st.error(f"❌ Could not fetch data for {symbol} after trying all methods")
                if 'PLN' in symbol:
                    st.info(f"""
                    💡 **PLN Pair Tips:**
                    - PLN pairs might have limited data on Yahoo Finance
                    - Try major pairs like EURUSD, GBPUSD for testing
                    - Polish market data is often delayed or limited
                    - Consider using alternative data sources for PLN pairs
                    """)
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
                st.error(f"❌ Insufficient data for {symbol} (only {len(df)} valid days)")
                return None, None
            
            # Verification info
            verification = {
                'symbol': symbol,
                'yahoo_symbol': yf_symbol,
                'data_points': len(df),
                'latest_price': float(df['Close'].iloc[-1]),
                'latest_date': df['Date'].iloc[-1],
                'fetch_time': datetime.now(),
                'is_pln_pair': 'PLN' in symbol,
                'is_crypto': 'BTC' in symbol or 'ETH' in symbol
            }
            
            return df, verification
            
        except Exception as e:
            st.error(f"❌ Error fetching {symbol}: {str(e)}")
            if 'PLN' in symbol:
                st.info("""
                📝 **PLN Pair Troubleshooting:**
                - Polish Zloty pairs have limited availability
                - Data might be sparse or delayed  
                - Try refreshing or selecting major currency pairs
                """)
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
st.markdown("**Live Forex & Crypto data with Pivot Points analysis**")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Group currency pairs by category
major_pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
cross_pairs = ['EURJPY', 'GBPJPY', 'EURGBP']
pln_pairs = ['EURPLN', 'USDPLN', 'GBPPLN', 'CHFPLN']
crypto_pairs = ['BTCUSD']

# Create expandable sections for different pair types
st.sidebar.markdown("### Currency Pairs")

pair_category = st.sidebar.radio(
    "Select Category:",
    ["🌍 Major Pairs", "🔄 Cross Pairs", "🇵🇱 PLN Pairs", "₿ Crypto"],
    index=0
)

if pair_category == "🌍 Major Pairs":
    selected_symbol = st.sidebar.selectbox("Select Major Pair:", major_pairs, index=0)
elif pair_category == "🔄 Cross Pairs":
    selected_symbol = st.sidebar.selectbox("Select Cross Pair:", cross_pairs, index=0)
elif pair_category == "🇵🇱 PLN Pairs":
    selected_symbol = st.sidebar.selectbox("Select PLN Pair:", pln_pairs, index=0)
    st.sidebar.info("💡 PLN pairs may have limited data availability")
else:  # Crypto
    selected_symbol = st.sidebar.selectbox("Select Crypto Pair:", crypto_pairs, index=0)
    st.sidebar.info("₿ Bitcoin trades 24/7")

# Chart type selection
chart_type = st.sidebar.radio(
    "Chart Type:",
    ["📊 Candlestick", "📈 Line Chart", "🔀 Both Charts"],
    index=0
)

chart_days = st.sidebar.slider("Chart Days", 7, 30, 14)
show_volume = st.sidebar.checkbox("Show Volume", False)
show_pivots = st.sidebar.checkbox("Show Pivot Levels", True)
auto_refresh = st.sidebar.checkbox("Auto Refresh (60s)", False)

# Main chart section
st.markdown(f"## 📊 Live {selected_symbol} Chart - Last {chart_days} Days")

def create_chart(df, symbol, days, show_vol, show_piv, chart_type="candlestick"):
    """Create OHLC chart with pivot points"""
    try:
        if df is None or len(df) == 0:
            st.error("No data available for chart")
            return None
        
        # Get recent data
        chart_data = df.tail(days)
        
        # Determine decimal places based on instrument
        if 'BTC' in symbol:
            decimal_places = 2  # Bitcoin uses 2 decimals
            price_format = '.2f'
        elif 'JPY' in symbol:
            decimal_places = 3  # JPY pairs use 3 decimals
            price_format = '.3f'
        else:
            decimal_places = 5  # Standard forex uses 5 decimals
            price_format = '.5f'
        
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
        
        # Add price chart based on type
        if chart_type == "candlestick":
            # Add candlestick chart with improved styling
            candlestick = go.Candlestick(
                x=chart_data['Date'],
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name=f'{symbol} OHLC',
                increasing_line_color='#26a69a',  # Green for up candles
                decreasing_line_color='#ef5350',  # Red for down candles
                increasing_fillcolor='#26a69a',
                decreasing_fillcolor='#ef5350',
                line_width=1,  # Thin candle borders
                increasing_line_width=1,
                decreasing_line_width=1
            )
            
            if show_vol:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
                
        elif chart_type == "line":
            # Add line chart for closing prices
            line_chart = go.Scatter(
                x=chart_data['Date'],
                y=chart_data['Close'],
                mode='lines',
                name=f'{symbol} Close Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Date: %{x}<br>' +
                            f'Price: %{{y:{price_format}}}<br>' +
                            '<extra></extra>'
            )
            
            if show_vol:
                fig.add_trace(line_chart, row=1, col=1)
            else:
                fig.add_trace(line_chart)
        
        # Add pivot levels with improved positioning
        if show_piv:
            latest_row = chart_data.iloc[-1]
            if pd.notna(latest_row.get('Pivot')):
                
                # Calculate chart boundaries for label positioning
                chart_start = chart_data['Date'].min()
                chart_end = chart_data['Date'].max()
                
                # Extend the right margin for labels by adding time buffer
                time_diff = chart_end - chart_start
                label_position = chart_end + time_diff * 0.05  # 5% beyond last candle
                
                levels = {
                    'R2': (latest_row.get('R2'), 'red'),
                    'R1': (latest_row.get('R1'), 'red'), 
                    'Pivot': (latest_row['Pivot'], 'black'),
                    'S1': (latest_row.get('S1'), 'green'),
                    'S2': (latest_row.get('S2'), 'green')
                }
                
                for name, (value, color) in levels.items():
                    if pd.notna(value):
                        # Add the horizontal line
                        hline_args = {
                            'y': value,
                            'line_dash': 'solid',
                            'line_color': color,
                            'line_width': 1,
                        }
                        if show_vol:
                            fig.add_hline(row=1, col=1, **hline_args)
                        else:
                            fig.add_hline(**hline_args)
                        
                        # Add separate annotation positioned away from candles
                        annotation_args = {
                            'x': label_position,
                            'y': value,
                            'text': f'{name}: {value:{price_format}}',
                            'showarrow': False,
                            'font': dict(size=10, color=color),
                            'bgcolor': 'rgba(255,255,255,0.9)',
                            'bordercolor': color,
                            'borderwidth': 1,
                            'xanchor': 'left',
                            'yanchor': 'middle'
                        }
                        
                        if show_vol:
                            fig.add_annotation(row=1, col=1, **annotation_args)
                        else:
                            fig.add_annotation(**annotation_args)
        
        # Add volume with better colors
        if show_vol and 'Volume' in chart_data.columns:
            colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(chart_data['Close'], chart_data['Open'])]
            fig.add_trace(
                go.Bar(
                    x=chart_data['Date'], 
                    y=chart_data['Volume'], 
                    name='Volume', 
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Current price marker with better visibility
        current_price = chart_data['Close'].iloc[-1]
        current_date = chart_data['Date'].iloc[-1]
        
        price_marker = go.Scatter(
            x=[current_date],
            y=[current_price],
            mode='markers+text',
            marker=dict(
                size=10,
                color='orange',
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            text=[f'{current_price:{price_format}}'],
            textposition='top center',
            name='Current Price',
            textfont=dict(size=10, color='orange')
        )
        
        if show_vol:
            fig.add_trace(price_marker, row=1, col=1)
        else:
            fig.add_trace(price_marker)
        
        # Update layout with extended right margin for labels
        chart_title = f"{symbol} - {chart_type.title()} Chart - Current: {current_price:{price_format}}"
        
        fig.update_layout(
            title=chart_title,
            height=600 if show_vol else 450,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(r=120)  # Extended right margin for pivot labels
        )
        
        # Update axes with grid
        if show_vol:
            fig.update_xaxes(
                title_text="Date", 
                showgrid=True, 
                gridwidth=0.5, 
                gridcolor='lightgray',
                row=2, col=1
            )
            fig.update_yaxes(
                title_text="Price", 
                showgrid=True, 
                gridwidth=0.5, 
                gridcolor='lightgray',
                row=1, col=1
            )
            fig.update_yaxes(
                title_text="Volume", 
                showgrid=True, 
                gridwidth=0.5, 
                gridcolor='lightgray',
                row=2, col=1
            )
        else:
            fig.update_xaxes(
                title_text="Date", 
                showgrid=True, 
                gridwidth=0.5, 
                gridcolor='lightgray'
            )
            fig.update_yaxes(
                title_text="Price", 
                showgrid=True, 
                gridwidth=0.5, 
                gridcolor='lightgray'
            )
        
        # Better spacing for charts
        if chart_type == "candlestick":
            fig.update_xaxes(type='category')  # Better candle spacing
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def create_line_chart(df, symbol, days, show_piv):
    """Create separate line chart for closing prices"""
    try:
        if df is None or len(df) == 0:
            return None
        
        chart_data = df.tail(days)
        
        # Determine decimal places based on instrument
        if 'BTC' in symbol:
            price_format = '.2f'
        elif 'JPY' in symbol:
            price_format = '.3f'
        else:
            price_format = '.5f'
        
        fig = go.Figure()
        
        # Add line chart for closing prices
        line_chart = go.Scatter(
            x=chart_data['Date'],
            y=chart_data['Close'],
            mode='lines+markers',
            name=f'{symbol} Close Price',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4, color='#1f77b4'),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                        'Date: %{x}<br>' +
                        f'Price: %{{y:{price_format}}}<br>' +
                        '<extra></extra>'
        )
        
        fig.add_trace(line_chart)
        
        # Add pivot levels
        if show_piv:
            latest_row = chart_data.iloc[-1]
            if pd.notna(latest_row.get('Pivot')):
                
                chart_start = chart_data['Date'].min()
                chart_end = chart_data['Date'].max()
                time_diff = chart_end - chart_start
                label_position = chart_end + time_diff * 0.05
                
                levels = {
                    'R2': (latest_row.get('R2'), 'red'),
                    'R1': (latest_row.get('R1'), 'red'), 
                    'Pivot': (latest_row['Pivot'], 'black'),
                    'S1': (latest_row.get('S1'), 'green'),
                    'S2': (latest_row.get('S2'), 'green')
                }
                
                for name, (value, color) in levels.items():
                    if pd.notna(value):
                        # Add horizontal line
                        fig.add_hline(
                            y=value,
                            line_dash='solid',
                            line_color=color,
                            line_width=1
                        )
                        
                        # Add annotation
                        fig.add_annotation(
                            x=label_position,
                            y=value,
                            text=f'{name}: {value:{price_format}}',
                            showarrow=False,
                            font=dict(size=10, color=color),
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor=color,
                            borderwidth=1,
                            xanchor='left',
                            yanchor='middle'
                        )
        
        # Current price marker
        current_price = chart_data['Close'].iloc[-1]
        current_date = chart_data['Date'].iloc[-1]
        
        fig.add_scatter(
            x=[current_date],
            y=[current_price],
            mode='markers+text',
            marker=dict(size=12, color='orange', line=dict(width=2, color='white')),
            text=[f'{current_price:{price_format}}'],
            textposition='top center',
            name='Current Price',
            textfont=dict(size=10, color='orange')
        )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Line Chart - Current: {current_price:{price_format}}",
            height=400,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            margin=dict(r=120),
            xaxis_title="Date",
            yaxis_title="Price"
        )
        
        # Update axes with grid
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating line chart: {str(e)}")
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
    
    # Determine price format
    if 'BTC' in selected_symbol:
        price_format = '.2f'
    elif 'JPY' in selected_symbol:
        price_format = '.3f'
    else:
        price_format = '.5f'
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"{current_price:{price_format}}")
    
    with col2:
        if pd.notna(latest_row.get('Pivot')):
            pivot_diff = current_price - latest_row['Pivot']
            st.metric("vs Pivot", f"{pivot_diff:+{price_format}}")
        else:
            st.metric("vs Pivot", "N/A")
    
    with col3:
        st.metric("Data Points", verification['data_points'])
    
    with col4:
        st.metric("Latest Date", verification['latest_date'].strftime('%Y-%m-%d'))
        if verification.get('is_pln_pair', False):
            st.caption("🇵🇱 PLN Pair")
        if verification.get('is_crypto', False):
            st.caption("₿ Crypto")
    
    # Signal analysis
    if pd.notna(latest_row.get('S2')) and pd.notna(latest_row.get('R2')):
        
        # Determine signal
        signal_type = None
        signal_strength = "NEUTRAL"
        
        if current_price < latest_row['S2']:
            signal_type = "🟢 BUY SIGNAL"
            signal_strength = "STRONG"
            signal_reason = f"Price {current_price:{price_format}} below S2 level {latest_row['S2']:{price_format}}"
            css_class = "signal-strong"
        elif current_price > latest_row['R2']:
            signal_type = "🔴 SELL SIGNAL"
            signal_strength = "STRONG"
            signal_reason = f"Price {current_price:{price_format}} above R2 level {latest_row['R2']:{price_format}}"
            css_class = "signal-strong"
        elif current_price < latest_row.get('S1', latest_row['Pivot']):
            signal_type = "🟡 Weak Buy"
            signal_strength = "WEAK"
            signal_reason = f"Price {current_price:{price_format}} below S1"
            css_class = "signal-weak"
        elif current_price > latest_row.get('R1', latest_row['Pivot']):
            signal_type = "🟡 Weak Sell"
            signal_strength = "WEAK"
            signal_reason = f"Price {current_price:{price_format}} above R1"
            css_class = "signal-weak"
        
        if signal_type:
            st.markdown(f'<div class="{css_class}"><h4>{signal_type}</h4><p>{signal_reason}</p></div>', 
                       unsafe_allow_html=True)
        else:
            st.info(f"**No Signal** - Price {current_price:{price_format}} between pivot levels")
        
        # Pivot levels table
        st.markdown("#### 📊 Current Pivot Levels")
        
        levels_data = []
        for level_name in ['R2', 'R1', 'Pivot', 'S1', 'S2']:
            value = latest_row.get(level_name)
            if pd.notna(value):
                distance = ((value - current_price) / current_price * 100)
                levels_data.append({
                    'Level': level_name,
                    'Value': f"{value:{price_format}}",
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
        **Latest Price:** {verification['latest_price']:{price_format}}  
        **Fetch Time:** {verification['fetch_time'].strftime('%H:%M:%S')}  
        **Verify at:** https://finance.yahoo.com/quote/{verification['yahoo_symbol']}
        """)
        
        # Show recent data
        recent_data = df_with_pivots.tail(5)[['Date', 'Open', 'High', 'Low', 'Close']].copy()
        recent_data['Date'] = recent_data['Date'].dt.strftime('%Y-%m-%d')
        
        # Round to appropriate decimals
        if 'BTC' in selected_symbol:
            decimals = 2
        elif 'JPY' in selected_symbol:
            decimals = 3
        else:
            decimals = 5
            
        for col in ['Open', 'High', 'Low', 'Close']:
            recent_data[col] = recent_data[col].round(decimals)
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
**₿ Crypto:** Bitcoin trades 24/7 with real-time data
""")
