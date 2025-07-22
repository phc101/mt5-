#!/usr/bin/env python3
"""
MT5 Signal Generator - Streamlit Cloud Ready
Momentum Pivot Points Trading Strategy
Deploy: streamlit.app
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

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="MT5 Trading Signals ⚡",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "MT5 Momentum Trading Signals Generator"
    }
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .signal-card {
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .buy-signal {
        border-left: 5px solid #28a745;
        background: linear-gradient(90deg, #d4edda, #f8f9fa);
    }
    .sell-signal {
        border-left: 5px solid #dc3545;
        background: linear-gradient(90deg, #f8d7da, #f8f9fa);
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-active { background-color: #28a745; }
    .status-inactive { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Forex symbols mapping for yfinance
FOREX_SYMBOLS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X', 
    'USDCHF': 'USDCHF=X',
    'USDJPY': 'USDJPY=X',
    'EURJPY': 'EURJPY=X',
    'GBPJPY': 'GBPJPY=X',
    'AUDUSD': 'AUDUSD=X',
    'NZDUSD': 'NZDUSD=X',
    'USDCAD': 'USDCAD=X',
    'EURAUD': 'EURAUD=X',
    'EURGBP': 'EURGBP=X',
    'EURPLN': 'EURPLN=X',
    'USDPLN': 'USDPLN=X',
    'GBPPLN': 'GBPPLN=X',
    'CHFPLN': 'CHFPLN=X'
}

class ForexMomentumBot:
    def __init__(self):
        self.lookback_days = 7
        self.holding_days = 3
        self.stop_loss_percent = 2.0
        
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_forex_data(_self, symbol, days=30):
        """Fetch forex data with caching"""
        try:
            yf_symbol = FOREX_SYMBOLS.get(symbol, f"{symbol}=X")
            ticker = yf.Ticker(yf_symbol)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = ticker.history(
                start=start_date, 
                end=end_date,
                interval="1d",
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                return None
                
            df = pd.DataFrame({
                'Date': data.index,
                'Open': data['Open'],
                'High': data['High'],
                'Low': data['Low'],
                'Close': data['Close'],
                'Volume': data['Volume']
            }).reset_index(drop=True)
            
            # Remove any NaN values
            df = df.dropna()
            
            return df if len(df) > 10 else None
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_pivot_points(self, df):
        """Calculate momentum pivot points"""
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
        
        if not pivot_data:
            return df
            
        pivot_df = pd.DataFrame(pivot_data)
        df = df.merge(pivot_df, on='Date', how='left')
        return df
    
    def calculate_momentum_strength(self, df, latest_row, signal_type):
        """Calculate signal strength based on momentum"""
        try:
            if len(df) < 5:
                return 50.0
                
            # Price momentum
            price_changes = df['Close'].pct_change(periods=3).dropna()
            avg_momentum = abs(price_changes.mean()) * 100
            
            # Volume momentum (if available)
            volume_momentum = 0
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                vol_changes = df['Volume'].pct_change().dropna()
                volume_momentum = abs(vol_changes.mean()) * 10
            
            # Distance from pivot level
            current_price = latest_row['Close']
            if signal_type == 'BUY':
                level_distance = abs(current_price - latest_row['S2']) / current_price * 1000
            else:
                level_distance = abs(current_price - latest_row['R2']) / current_price * 1000
            
            # Volatility factor
            volatility = df['Close'].pct_change().std() * 100
            vol_factor = max(0, 50 - volatility * 5)
            
            # Combine factors
            strength = min(100, max(30, 
                avg_momentum * 20 + 
                volume_momentum + 
                level_distance * 10 + 
                vol_factor
            ))
            
            return round(strength, 1)
            
        except Exception:
            return 50.0
    
    def generate_signal(self, symbol):
        """Generate trading signal"""
        df = self.get_forex_data(symbol, 35)
        if df is None or len(df) < self.lookback_days + 5:
            return None, None
            
        df = self.calculate_pivot_points(df)
        latest_row = df.iloc[-1]
        
        # Check if we have pivot data
        if pd.isna(latest_row.get('S2')) or pd.isna(latest_row.get('R2')):
            return None, df
            
        current_price = latest_row['Close']
        signal = None
        
        # BUY Signal: Price below S2 (strong downward momentum break)
        if current_price < latest_row['S2']:
            signal = {
                'symbol': symbol,
                'type': 'BUY',
                'current_price': current_price,
                'entry_zone': latest_row['S2'],
                'stop_loss': current_price * (1 - self.stop_loss_percent / 100),
                'take_profit_1': latest_row['S1'],
                'take_profit_2': latest_row['Pivot'],
                'pivot_levels': {
                    'S2': latest_row['S2'],
                    'S1': latest_row['S1'],
                    'Pivot': latest_row['Pivot'],
                    'R1': latest_row['R1'],
                    'R2': latest_row['R2']
                },
                'timestamp': datetime.now(),
                'holding_days': self.holding_days,
                'strength': self.calculate_momentum_strength(df, latest_row, 'BUY'),
                'risk_reward': abs(latest_row['Pivot'] - current_price) / abs(current_price - (current_price * (1 - self.stop_loss_percent / 100)))
            }
        
        # SELL Signal: Price above R2 (strong upward momentum break)
        elif current_price > latest_row['R2']:
            signal = {
                'symbol': symbol,
                'type': 'SELL', 
                'current_price': current_price,
                'entry_zone': latest_row['R2'],
                'stop_loss': current_price * (1 + self.stop_loss_percent / 100),
                'take_profit_1': latest_row['R1'],
                'take_profit_2': latest_row['Pivot'],
                'pivot_levels': {
                    'S2': latest_row['S2'],
                    'S1': latest_row['S1'],
                    'Pivot': latest_row['Pivot'],
                    'R1': latest_row['R1'],
                    'R2': latest_row['R2']
                },
                'timestamp': datetime.now(),
                'holding_days': self.holding_days,
                'strength': self.calculate_momentum_strength(df, latest_row, 'SELL'),
                'risk_reward': abs(latest_row['Pivot'] - current_price) / abs((current_price * (1 + self.stop_loss_percent / 100)) - current_price)
            }
            
        return signal, df

# Initialize the bot
@st.cache_resource
def get_bot():
    return ForexMomentumBot()

bot = get_bot()

# Header
st.markdown('<p class="main-header">🎯 MT5 Momentum Trading Signals</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.image("https://via.placeholder.com/200x80/1f77b4/white?text=MT5+SIGNALS", width=200)
st.sidebar.markdown("### ⚙️ Configuration")

# Symbol selection
selected_symbols = st.sidebar.multiselect(
    "Select Currency Pairs:",
    list(FOREX_SYMBOLS.keys()),
    default=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
)

# Parameters
bot.holding_days = st.sidebar.slider("Holding Period (days)", 1, 10, 3)
bot.stop_loss_percent = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 2.0, 0.1)

# Auto refresh
auto_refresh = st.sidebar.checkbox("Auto Refresh (60s)", value=False)
manual_refresh = st.sidebar.button("🔄 Refresh Now", type="primary")

# Risk management settings
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Risk Management")
min_strength = st.sidebar.slider("Minimum Signal Strength", 30, 80, 50)
max_risk_reward = st.sidebar.slider("Min Risk:Reward Ratio", 1.0, 5.0, 2.0, 0.1)

# Display current time
st.sidebar.markdown("---")
st.sidebar.markdown(f"**🕐 Last Update:** {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.markdown(f"**📅 Date:** {datetime.now().strftime('%Y-%m-%d')}")

# Main content
if not selected_symbols:
    st.warning("⚠️ Please select at least one currency pair from the sidebar to begin analysis.")
    st.stop()

# Generate signals
with st.spinner("🔍 Analyzing market conditions..."):
    all_signals = []
    charts_data = {}
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(selected_symbols):
        progress_bar.progress((i + 1) / len(selected_symbols))
        
        signal, df = bot.generate_signal(symbol)
        if signal and signal['strength'] >= min_strength and signal['risk_reward'] >= max_risk_reward:
            all_signals.append(signal)
        
        if df is not None:
            charts_data[symbol] = df
    
    progress_bar.empty()

# Display active signals
st.markdown("## 🚨 Active Trading Signals")

if all_signals:
    # Sort by strength
    all_signals = sorted(all_signals, key=lambda x: x['strength'], reverse=True)
    
    for i, signal in enumerate(all_signals):
        signal_class = "buy-signal" if signal['type'] == 'BUY' else "sell-signal"
        signal_emoji = "🟢" if signal['type'] == 'BUY' else "🔴"
        
        # Strength indicator
        if signal['strength'] >= 70:
            strength_color = "🟢"
            strength_label = "STRONG"
        elif signal['strength'] >= 50:
            strength_color = "🟡"
            strength_label = "MODERATE"
        else:
            strength_color = "🟠"
            strength_label = "WEAK"
        
        # Risk-reward indicator
        rr_color = "🟢" if signal['risk_reward'] >= 2.5 else "🟡" if signal['risk_reward'] >= 2.0 else "🟠"
        
        st.markdown(f"""
        <div class="signal-card {signal_class}">
            <h3>{signal_emoji} {signal['symbol']} - {signal['type']} SIGNAL #{i+1}</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                <div>
                    <h4>📊 Entry Details</h4>
                    <strong>Current Price:</strong> {signal['current_price']:.5f}<br>
                    <strong>Entry Zone:</strong> {signal['entry_zone']:.5f}<br>
                    <strong>Stop Loss:</strong> {signal['stop_loss']:.5f}<br>
                    <strong>Holding Period:</strong> {signal['holding_days']} days
                </div>
                <div>
                    <h4>🎯 Profit Targets</h4>
                    <strong>Target 1:</strong> {signal['take_profit_1']:.5f}<br>
                    <strong>Target 2:</strong> {signal['take_profit_2']:.5f}<br>
                    <strong>Risk:Reward:</strong> {rr_color} 1:{signal['risk_reward']:.2f}<br>
                </div>
                <div>
                    <h4>📈 Signal Quality</h4>
                    <strong>Strength:</strong> {strength_color} {signal['strength']:.1f}% ({strength_label})<br>
                    <strong>Generated:</strong> {signal['timestamp'].strftime('%H:%M:%S')}<br>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chart for each signal
        with st.expander(f"📊 Chart Analysis - {signal['symbol']}", expanded=False):
            if signal['symbol'] in charts_data:
                df = charts_data[signal['symbol']]
                
                # Create chart
                fig = go.Figure()
                
                # Candlestick chart
                recent_data = df.tail(30)  # Last 30 days
                fig.add_trace(go.Candlestick(
                    x=recent_data['Date'],
                    open=recent_data['Open'],
                    high=recent_data['High'], 
                    low=recent_data['Low'],
                    close=recent_data['Close'],
                    name=signal['symbol']
                ))
                
                # Add pivot levels
                pivot_levels = signal['pivot_levels']
                level_colors = {
                    'R2': 'red', 'R1': 'orange', 
                    'Pivot': 'purple', 
                    'S1': 'blue', 'S2': 'darkblue'
                }
                
                for level_name, level_value in pivot_levels.items():
                    fig.add_hline(
                        y=level_value,
                        line_dash="dash",
                        line_color=level_colors.get(level_name, 'gray'),
                        annotation_text=f"{level_name}: {level_value:.5f}",
                        annotation_position="bottom right"
                    )
                
                # Add signal point
                fig.add_scatter(
                    x=[recent_data['Date'].iloc[-1]],
                    y=[signal['current_price']],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color='green' if signal['type'] == 'BUY' else 'red',
                        symbol='triangle-up' if signal['type'] == 'BUY' else 'triangle-down'
                    ),
                    text=[f"{signal['type']}<br>{signal['strength']:.0f}%"],
                    textposition='top center',
                    name=f"{signal['type']} Signal",
                    showlegend=True
                )
                
                fig.update_layout(
                    title=f"{signal['symbol']} - Momentum Pivot Analysis",
                    height=500,
                    xaxis_rangeslider_visible=False,
                    showlegend=True,
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("🔍 No signals meeting your criteria at the moment. Try adjusting the minimum strength or risk-reward ratio.")

# Market overview
st.markdown("## 📊 Market Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Pairs Monitored", len(selected_symbols))

with col2:
    st.metric("Active Signals", len(all_signals))

with col3:
    if all_signals:
        avg_strength = sum(s['strength'] for s in all_signals) / len(all_signals)
        st.metric("Avg Strength", f"{avg_strength:.1f}%")
    else:
        st.metric("Avg Strength", "N/A")

with col4:
    buy_signals = sum(1 for s in all_signals if s['type'] == 'BUY')
    sell_signals = len(all_signals) - buy_signals
    st.metric("BUY/SELL", f"{buy_signals}/{sell_signals}")

# Instructions panel
st.markdown("## 📱 MT5 Android Instructions")

if all_signals:
    st.success("### 🎯 Execute these trades on your MT5 Android app:")
    
    for i, signal in enumerate(all_signals, 1):
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**Trade #{i}**")
                st.markdown(f"{'🟢 BUY' if signal['type'] == 'BUY' else '🔴 SELL'}")
                
            with col2:
                st.markdown(f"""
                **{signal['symbol']}** - {signal['type']}
                - **Volume:** 0.01-0.10 lots (adjust to your risk)
                - **Stop Loss:** {signal['stop_loss']:.5f}
                - **Take Profit 1:** {signal['take_profit_1']:.5f}
                - **Take Profit 2:** {signal['take_profit_2']:.5f}
                - **Max Hold:** {signal['holding_days']} days
                """)

# Strategy explanation
with st.expander("📋 Strategy Rules & Risk Management"):
    st.markdown("""
    ### 🎯 Momentum Pivot Strategy Rules:
    
    **Signal Generation:**
    - **BUY Signal:** Price breaks below S2 (strong support break = potential reversal)
    - **SELL Signal:** Price breaks above R2 (strong resistance break = potential reversal)
    
    **Pivot Calculation:**
    - Based on 7-day average: Pivot = (AvgHigh + AvgLow + AvgClose) / 3
    - R2 = Pivot + (AvgHigh - AvgLow)
    - S2 = Pivot - (AvgHigh - AvgLow)
    
    **Risk Management:**
    - Fixed Stop Loss: 2% (adjustable)
    - Position Size: 1-2% of account per trade
    - Max Holding: 3 days (adjustable)
    - Signal Strength: Minimum 50% (adjustable)
    
    ### ⚠️ Important Disclaimers:
    - **DEMO ACCOUNT ONLY** - Test thoroughly before live trading
    - Past performance does not guarantee future results
    - Never risk more than you can afford to lose
    - Always use proper risk management
    - Consider market conditions and news events
    """)

# Auto refresh logic
if auto_refresh:
    time.sleep(2)
    st.rerun()

if manual_refresh:
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em; padding: 20px;">
    <p><strong>⚠️ EDUCATIONAL PURPOSE ONLY - USE DEMO ACCOUNT FOR TESTING</strong></p>
    <p>📊 Data: Yahoo Finance | 🔄 Auto-refresh available | 🚀 Built with Streamlit</p>
    <p>💡 Always practice proper risk management and never invest more than you can afford to lose</p>
</div>
""", unsafe_allow_html=True)
