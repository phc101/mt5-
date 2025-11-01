import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import io

# Konfiguracja strony
st.set_page_config(
    page_title="Pivot Points Backtest", 
    page_icon="📊",
    layout="wide"
)

# Tytuł aplikacji
st.title("📊 Pivot Points Backtester")
st.markdown("### Testuj strategię pivot points na różnych parach walutowych i lewarach")

# Sidebar - parametry
st.sidebar.header("⚙️ Parametry strategii")

# Upload pliku
uploaded_file = st.sidebar.file_uploader(
    "Wgraj plik CSV z danymi", 
    type=['csv'],
    help="Plik musi zawierać kolumny: Date, Price, Open, High, Low"
)

# Parametry strategii
lookback_days = st.sidebar.slider(
    "Liczba dni do obliczenia pivot points",
    min_value=5,
    max_value=50,
    value=20,
    step=5
)

threshold = st.sidebar.slider(
    "Próg ±% od Pivot Point",
    min_value=0.1,
    max_value=3.0,
    value=0.5,
    step=0.1
)

leverages = st.sidebar.multiselect(
    "Wybierz lewary do przetestowania",
    options=[1, 2, 3, 5, 10, 15, 20],
    default=[1, 5, 10, 20]
)

initial_capital = st.sidebar.number_input(
    "Kapitał początkowy",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000
)

# Funkcje
@st.cache_data
def load_and_prepare_data(uploaded_file):
    """Wczytaj i przygotuj dane"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Usuń BOM jeśli istnieje
        df.columns = df.columns.str.replace('ï»¿', '').str.strip()
        
        # Parsuj datę
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        if df['Date'].isna().any():
            df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Konwersja do float
        for col in ['Price', 'Open', 'High', 'Low']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Dodaj dzień tygodnia
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        return df, None
    except Exception as e:
        return None, str(e)

def calculate_pivot_points(high, low, close):
    """Oblicz pivot points"""
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    r2 = pp + (high - low)
    r3 = high + 2 * (pp - low)
    s1 = 2 * pp - high
    s2 = pp - (high - low)
    s3 = low - 2 * (high - pp)
    return pp, r1, r2, r3, s1, s2, s3

def run_backtest(df, threshold_pct, lookback, leverage, initial_capital):
    """Uruchom backtest"""
    mondays = df[df['DayOfWeek'] == 0].copy()
    trades = []
    
    for idx, monday_row in mondays.iterrows():
        monday_date = monday_row['Date']
        monday_price = monday_row['Open']
        
        prev_days = df[df['Date'] < monday_date].tail(lookback)
        
        if len(prev_days) >= lookback:
            high_period = prev_days['High'].max()
            low_period = prev_days['Low'].min()
            close_period = prev_days['Price'].iloc[-1]
            
            pp, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(high_period, low_period, close_period)
            
            buy_threshold = pp * (1 - threshold_pct/100)
            sell_threshold = pp * (1 + threshold_pct/100)
            
            future_dates = df[df['Date'] > monday_date]
            friday = future_dates[future_dates['DayOfWeek'] == 4].head(1)
            
            if not friday.empty:
                friday_date = friday.iloc[0]['Date']
                friday_price = friday.iloc[0]['Price']
                
                signal = None
                pnl_pct = 0
                
                if monday_price <= buy_threshold:
                    signal = 'BUY'
                    pnl_pct = ((friday_price - monday_price) / monday_price) * 100
                elif monday_price >= sell_threshold:
                    signal = 'SELL'
                    pnl_pct = ((monday_price - friday_price) / monday_price) * 100
                
                if signal:
                    pnl_leveraged = pnl_pct * leverage
                    trades.append({
                        'Monday_Date': monday_date,
                        'Friday_Date': friday_date,
                        'Signal': signal,
                        'Entry_Price': monday_price,
                        'Exit_Price': friday_price,
                        'PnL_%': pnl_pct,
                        'PnL_Leveraged_%': pnl_leveraged,
                        'Year': monday_date.year,
                        'Month': monday_date.month,
                    })
    
    if not trades:
        return None
    
    trades_df = pd.DataFrame(trades)
    
    # Oblicz kapitał
    trades_df['Capital'] = initial_capital
    for i in range(len(trades_df)):
        if i > 0:
            trades_df.loc[i, 'Capital'] = trades_df.loc[i-1, 'Capital'] * (1 + trades_df.loc[i, 'PnL_Leveraged_%'] / 100)
        else:
            trades_df.loc[i, 'Capital'] = initial_capital * (1 + trades_df.loc[i, 'PnL_Leveraged_%'] / 100)
    
    return trades_df

def calculate_metrics(trades_df, initial_capital):
    """Oblicz metryki"""
    if trades_df is None or len(trades_df) == 0:
        return None
    
    total_return = trades_df['PnL_Leveraged_%'].sum()
    final_capital = trades_df['Capital'].iloc[-1]
    roi = ((final_capital / initial_capital) - 1) * 100
    
    winning = len(trades_df[trades_df['PnL_Leveraged_%'] > 0])
    losing = len(trades_df[trades_df['PnL_Leveraged_%'] < 0])
    win_rate = (winning / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    
    trades_df['Cumulative_Return'] = trades_df['PnL_Leveraged_%'].cumsum()
    max_dd = (trades_df['Cumulative_Return'].cummax() - trades_df['Cumulative_Return']).max()
    
    return {
        'num_trades': len(trades_df),
        'total_return': total_return,
        'roi': roi,
        'final_capital': final_capital,
        'profit_loss': final_capital - initial_capital,
        'win_rate': win_rate,
        'winning_trades': winning,
        'losing_trades': losing,
        'avg_return': trades_df['PnL_Leveraged_%'].mean(),
        'best_trade': trades_df['PnL_Leveraged_%'].max(),
        'worst_trade': trades_df['PnL_Leveraged_%'].min(),
        'max_drawdown': max_dd
    }

# Główna logika
if uploaded_file is not None:
    # Wczytaj dane
    df, error = load_and_prepare_data(uploaded_file)
    
    if error:
        st.error(f"❌ Błąd wczytywania pliku: {error}")
        st.info("💡 Plik musi zawierać kolumny: Date, Price, Open, High, Low")
    else:
        st.success(f"✅ Wczytano {len(df)} rekordów z okresu {df['Date'].min().date()} do {df['Date'].max().date()}")
        
        # Pokaż przykładowe dane
        with st.expander("📋 Podgląd danych"):
            st.dataframe(df.head(10))
        
        # Uruchom backtesty dla wybranych lewarów
        if st.sidebar.button("🚀 Uruchom backtest", type="primary"):
            results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, lev in enumerate(leverages):
                status_text.text(f"Testowanie lewaru {lev}x...")
                trades = run_backtest(df, threshold, lookback_days, lev, initial_capital)
                if trades is not None:
                    metrics = calculate_metrics(trades, initial_capital)
                    results[lev] = {'trades': trades, 'metrics': metrics}
                else:
                    results[lev] = None
                progress_bar.progress((i + 1) / len(leverages))
            
            status_text.empty()
            progress_bar.empty()
            
            if not results or all(v is None for v in results.values()):
                st.warning("⚠️ Brak transakcji spełniających kryteria strategii. Spróbuj zmienić parametry.")
            else:
                # Tabela porównawcza
                st.header("📊 Wyniki Backtestów")
                
                comparison_data = []
                for lev in leverages:
                    if results[lev] is not None:
                        m = results[lev]['metrics']
                        comparison_data.append({
                            'Lewar': f"{lev}x",
                            'Transakcje': m['num_trades'],
                            'ROI (%)': round(m['roi'], 2),
                            'Kapitał końcowy': f"{m['final_capital']:,.2f}",
                            'Zysk/Strata': f"{m['profit_loss']:+,.2f}",
                            'Win Rate (%)': round(m['win_rate'], 1),
                            'Max Drawdown (%)': round(m['max_drawdown'], 2),
                            'Najlepsza (%)': round(m['best_trade'], 2),
                            'Najgorsza (%)': round(m['worst_trade'], 2)
                        })
                    else:
                        comparison_data.append({
                            'Lewar': f"{lev}x",
                            'Transakcje': 0,
                            'ROI (%)': 0,
                            'Kapitał końcowy': f"{initial_capital:,.2f}",
                            'Zysk/Strata': "0.00",
                            'Win Rate (%)': 0,
                            'Max Drawdown (%)': 0,
                            'Najlepsza (%)': 0,
                            'Najgorsza (%)': 0
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Podświetl najlepszy ROI
                def highlight_best_roi(row):
                    if row['ROI (%)'] == comparison_df['ROI (%)'].max() and row['ROI (%)'] > 0:
                        return ['background-color: #90EE90'] * len(row)
                    elif row['ROI (%)'] < 0:
                        return ['background-color: #FFB6C1'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    comparison_df.style.apply(highlight_best_roi, axis=1),
                    use_container_width=True
                )
                
                # Wykresy
                st.header("📈 Wizualizacje")
                
                # Wykres 1: Kapitał w czasie
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                for lev in leverages:
                    if results[lev] is not None:
                        trades = results[lev]['trades']
                        ax1.plot(trades['Monday_Date'], trades['Capital'], 
                                label=f'Lewar {lev}x', linewidth=2, marker='o', markersize=3)
                
                ax1.axhline(y=initial_capital, color='black', linestyle='--', 
                           linewidth=1, alpha=0.5, label='Kapitał początkowy')
                ax1.set_title('Wartość Portfela w Czasie', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Data')
                ax1.set_ylabel('Kapitał (PLN)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.xticks(rotation=45)
                st.pyplot(fig1)
                
                # Wykres 2: ROI porównanie
                col1, col2 = st.columns(2)
                
                with col1:
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    valid_results = [(lev, results[lev]['metrics']['roi']) 
                                    for lev in leverages if results[lev] is not None]
                    if valid_results:
                        levs, rois = zip(*valid_results)
                        colors = ['green' if r > 0 else 'red' for r in rois]
                        bars = ax2.bar([f"{l}x" for l in levs], rois, color=colors, alpha=0.7, edgecolor='black')
                        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
                        ax2.set_title('ROI - Porównanie', fontsize=12, fontweight='bold')
                        ax2.set_ylabel('ROI (%)')
                        ax2.grid(True, alpha=0.3, axis='y')
                        
                        for bar in bars:
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.1f}%', ha='center', 
                                    va='bottom' if height > 0 else 'top', fontweight='bold')
                    st.pyplot(fig2)
                
                with col2:
                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    valid_results = [(lev, results[lev]['metrics']['max_drawdown']) 
                                    for lev in leverages if results[lev] is not None]
                    if valid_results:
                        levs, dds = zip(*valid_results)
                        ax3.bar([f"{l}x" for l in levs], dds, color='orange', alpha=0.7, edgecolor='black')
                        ax3.set_title('Max Drawdown - Porównanie', fontsize=12, fontweight='bold')
                        ax3.set_ylabel('Max Drawdown (%)')
                        ax3.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig3)
                
                # Szczegóły dla wybranego lewaru
                st.header("🔍 Szczegóły")
                selected_lev = st.selectbox("Wybierz lewar do szczegółowej analizy:", leverages)
                
                if results[selected_lev] is not None:
                    trades = results[selected_lev]['trades']
                    metrics = results[selected_lev]['metrics']
                    
                    # Metryki w kolumnach
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Liczba transakcji", metrics['num_trades'])
                        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                    with col2:
                        st.metric("ROI", f"{metrics['roi']:.2f}%", 
                                 delta=f"{metrics['profit_loss']:+,.2f} PLN")
                        st.metric("Kapitał końcowy", f"{metrics['final_capital']:,.2f} PLN")
                    with col3:
                        st.metric("Najlepsza transakcja", f"{metrics['best_trade']:.2f}%")
                        st.metric("Najgorsza transakcja", f"{metrics['worst_trade']:.2f}%")
                    with col4:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                        st.metric("Średni zwrot", f"{metrics['avg_return']:.2f}%")
                    
                    # Zwroty roczne
                    st.subheader("📅 Zwroty roczne")
                    yearly = trades.groupby('Year')['PnL_Leveraged_%'].agg(['sum', 'count']).round(2)
                    yearly.columns = ['Suma (%)', 'Liczba transakcji']
                    st.dataframe(yearly, use_container_width=True)
                    
                    # Tabela transakcji
                    st.subheader("📋 Wszystkie transakcje")
                    trades_display = trades[['Monday_Date', 'Friday_Date', 'Signal', 
                                            'Entry_Price', 'Exit_Price', 'PnL_Leveraged_%']].copy()
                    trades_display['Monday_Date'] = trades_display['Monday_Date'].dt.strftime('%Y-%m-%d')
                    trades_display['Friday_Date'] = trades_display['Friday_Date'].dt.strftime('%Y-%m-%d')
                    trades_display = trades_display.round(4)
                    st.dataframe(trades_display, use_container_width=True)
                    
                    # Download CSV
                    csv = trades.to_csv(index=False)
                    st.download_button(
                        label="💾 Pobierz szczegóły transakcji (CSV)",
                        data=csv,
                        file_name=f"backtest_leverage_{selected_lev}x.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning(f"⚠️ Brak transakcji dla lewaru {selected_lev}x")

else:
    # Instrukcja użycia
    st.info("""
    ### 👋 Witaj w Pivot Points Backtest Tool!
    
    **Jak używać:**
    1. 📁 Wgraj plik CSV z danymi historycznymi w lewym panelu
    2. ⚙️ Ustaw parametry strategii (lookback, próg, lewary)
    3. 🚀 Kliknij "Uruchom backtest"
    4. 📊 Analizuj wyniki i porównaj różne lewary!
    
    **Format pliku CSV:**
    - Kolumny: `Date`, `Price`, `Open`, `High`, `Low`
    - Format daty: MM/DD/YYYY
    - Separator: przecinek
    
    **Strategia:**
    - Oblicza pivot points z ostatnich N dni przed każdym poniedziałkiem
    - Kupuje gdy cena ≤ X% poniżej Pivot Point
    - Sprzedaje gdy cena ≥ X% powyżej Pivot Point
    - Zamyka pozycje w najbliższy piątek
    """)
    
    # Przykładowy format danych
    st.subheader("📋 Przykładowy format pliku CSV:")
    example_data = pd.DataFrame({
        'Date': ['10/31/2025', '10/30/2025', '10/29/2025'],
        'Price': [4.2537, 4.2456, 4.2418],
        'Open': [4.2455, 4.2418, 4.2310],
        'High': [4.2615, 4.2493, 4.2481],
        'Low': [4.2394, 4.2378, 4.2284]
    })
    st.dataframe(example_data, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 O strategii")
st.sidebar.info("""
**Pivot Points** to poziomy wsparcia i oporu obliczone na podstawie 
historycznych cen (High, Low, Close). Strategia wykorzystuje odchylenia 
od tych poziomów do generowania sygnałów kupna i sprzedaży.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")
