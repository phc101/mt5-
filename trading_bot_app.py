import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import io

# Konfiguracja strony
st.set_page_config(
    page_title="Pivot Points Backtest", 
    page_icon="📊",
    layout="wide"
)

# Tytuł aplikacji
st.title("📊 Pivot Points Backtester - Multi-Currency Portfolio")
st.markdown("### Testuj strategię pivot points na koszyku par walutowych i kryptowalut")

# Sidebar - parametry
st.sidebar.header("⚙️ Parametry strategii")

# Upload plików
st.sidebar.subheader("📁 Wgraj dane par walutowych")
uploaded_files = []
pair_names = []

for i in range(5):
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        file = st.sidebar.file_uploader(
            f"Para #{i+1}", 
            type=['csv'],
            key=f"file_{i}",
            help="Plik CSV z danymi historycznymi"
        )
    with col2:
        name = st.sidebar.text_input(
            "Nazwa",
            value=f"Para{i+1}",
            key=f"name_{i}",
            label_visibility="collapsed"
        )
    
    if file is not None:
        uploaded_files.append(file)
        pair_names.append(name)

# Parametry strategii
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parametry")

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

holding_days = st.sidebar.slider(
    "Liczba dni trzymania pozycji",
    min_value=1,
    max_value=10,
    value=5,
    step=1
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

capital_per_pair = st.sidebar.radio(
    "Alokacja kapitału:",
    options=["Równomiernie na wszystkie pary", "Pełny kapitał na każdą parę"],
    help="Równomiernie = kapitał podzielony przez liczbę par\nPełny = pełny kapitał na każdą parę (wyższe ryzyko)"
)

# Funkcje
@st.cache_data
def load_and_prepare_data(uploaded_file):
    """Wczytaj i przygotuj dane - obsługuje format forex i BTC"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Usuń BOM jeśli istnieje
        df.columns = df.columns.str.replace('ï»¿', '').str.strip()
        
        # Wykryj format pliku
        has_vol = 'Vol.' in df.columns
        has_change = 'Change %' in df.columns
        
        # Parsuj datę
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        if df['Date'].isna().any():
            df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Konwersja do float - usuń przecinki z liczb (format BTC)
        for col in ['Price', 'Open', 'High', 'Low']:
            if col in df.columns:
                if df[col].dtype == 'object':
                    # Usuń przecinki i konwertuj
                    df[col] = df[col].str.replace(',', '').astype(float)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Dodaj dzień tygodnia
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        # Info o formacie
        file_type = "BTC/Crypto format" if has_vol else "Forex format"
        
        return df, None, file_type
    except Exception as e:
        return None, str(e), None

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

def run_backtest(df, threshold_pct, lookback, leverage, initial_capital, holding_days, pair_name=""):
    """Uruchom backtest z duration N dni"""
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
            
            # Znajdź datę wyjścia (N dni roboczych później)
            future_dates = df[df['Date'] > monday_date].head(holding_days)
            
            if len(future_dates) >= holding_days:
                exit_date = future_dates.iloc[holding_days - 1]['Date']
                exit_price = future_dates.iloc[holding_days - 1]['Price']
                
                signal = None
                pnl_pct = 0
                
                if monday_price <= buy_threshold:
                    signal = 'BUY'
                    pnl_pct = ((exit_price - monday_price) / monday_price) * 100
                elif monday_price >= sell_threshold:
                    signal = 'SELL'
                    pnl_pct = ((monday_price - exit_price) / monday_price) * 100
                
                if signal:
                    pnl_leveraged = pnl_pct * leverage
                    trades.append({
                        'Entry_Date': monday_date,
                        'Exit_Date': exit_date,
                        'Signal': signal,
                        'Entry_Price': monday_price,
                        'Exit_Price': exit_price,
                        'PnL_%': pnl_pct,
                        'PnL_Leveraged_%': pnl_leveraged,
                        'Year': monday_date.year,
                        'Month': monday_date.month,
                        'Pair': pair_name
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

def combine_portfolio_trades(all_trades, initial_capital, num_pairs, allocation_method):
    """Połącz transakcje z różnych par w jeden portfel"""
    if not all_trades:
        return None
    
    # Wszystkie transakcje z wszystkich par
    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values('Entry_Date').reset_index(drop=True)
    
    # Oblicz kapitał w zależności od metody alokacji
    if allocation_method == "Równomiernie na wszystkie pary":
        # Kapitał podzielony na pary
        capital_multiplier = 1.0 / num_pairs
    else:
        # Pełny kapitał na każdą parę
        capital_multiplier = 1.0
    
    combined['Capital'] = initial_capital
    for i in range(len(combined)):
        if i > 0:
            pnl_effect = combined.loc[i, 'PnL_Leveraged_%'] * capital_multiplier / 100
            combined.loc[i, 'Capital'] = combined.loc[i-1, 'Capital'] * (1 + pnl_effect)
        else:
            pnl_effect = combined.loc[i, 'PnL_Leveraged_%'] * capital_multiplier / 100
            combined.loc[i, 'Capital'] = initial_capital * (1 + pnl_effect)
    
    return combined

def calculate_metrics(trades_df, initial_capital):
    """Oblicz metryki"""
    if trades_df is None or len(trades_df) == 0:
        return None
    
    final_capital = trades_df['Capital'].iloc[-1]
    roi = ((final_capital / initial_capital) - 1) * 100
    
    winning = len(trades_df[trades_df['PnL_Leveraged_%'] > 0])
    losing = len(trades_df[trades_df['PnL_Leveraged_%'] < 0])
    win_rate = (winning / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    
    trades_df['Cumulative_Return'] = (trades_df['Capital'] / initial_capital - 1) * 100
    max_dd = (trades_df['Cumulative_Return'].cummax() - trades_df['Cumulative_Return']).max()
    
    return {
        'num_trades': len(trades_df),
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
if len(uploaded_files) > 0:
    st.success(f"✅ Wczytano {len(uploaded_files)} par: {', '.join(pair_names)}")
    
    # Wczytaj wszystkie dane
    all_data = {}
    file_types = {}
    
    for file, name in zip(uploaded_files, pair_names):
        df, error, file_type = load_and_prepare_data(file)
        if error:
            st.error(f"❌ Błąd w pliku {name}: {error}")
        else:
            all_data[name] = df
            file_types[name] = file_type
            st.info(f"📊 {name}: {len(df)} rekordów ({df['Date'].min().date()} - {df['Date'].max().date()}) | {file_type}")
    
    if len(all_data) == 0:
        st.error("❌ Nie udało się wczytać żadnych danych")
    else:
        # Pokaż przykładowe dane
        with st.expander("📋 Podgląd danych wszystkich par"):
            for name, df in all_data.items():
                st.subheader(f"{name} ({file_types[name]})")
                st.dataframe(df.head(5))
        
        # Uruchom backtesty
        if st.sidebar.button("🚀 Uruchom backtest", type="primary"):
            results_by_pair = {}
            portfolio_results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_steps = len(leverages) * (len(all_data) + 1)
            current_step = 0
            
            for lev in leverages:
                pair_trades = []
                pair_results = {}
                
                # Backtest dla każdej pary osobno
                for name, df in all_data.items():
                    status_text.text(f"Testowanie {name} z lewarem {lev}x...")
                    
                    if capital_per_pair == "Równomiernie na wszystkie pary":
                        capital_for_pair = initial_capital / len(all_data)
                    else:
                        capital_for_pair = initial_capital
                    
                    trades = run_backtest(df, threshold, lookback_days, lev, capital_for_pair, holding_days, name)
                    
                    if trades is not None:
                        pair_trades.append(trades)
                        metrics = calculate_metrics(trades, capital_for_pair)
                        pair_results[name] = {'trades': trades, 'metrics': metrics}
                    else:
                        pair_results[name] = None
                    
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                
                results_by_pair[lev] = pair_results
                
                # Połącz wszystkie transakcje w portfel
                status_text.text(f"Tworzenie portfela dla lewaru {lev}x...")
                if pair_trades:
                    portfolio_trades = combine_portfolio_trades(pair_trades, initial_capital, len(all_data), capital_per_pair)
                    if portfolio_trades is not None:
                        portfolio_metrics = calculate_metrics(portfolio_trades, initial_capital)
                        portfolio_results[lev] = {'trades': portfolio_trades, 'metrics': portfolio_metrics}
                    else:
                        portfolio_results[lev] = None
                else:
                    portfolio_results[lev] = None
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            status_text.empty()
            progress_bar.empty()
            
            # WYNIKI - PORTFEL
            st.header("🎯 Wyniki Portfela (Wszystkie pary razem)")
            
            if not portfolio_results or all(v is None for v in portfolio_results.values()):
                st.warning("⚠️ Brak transakcji w portfelu")
            else:
                # Tabela porównawcza portfela
                portfolio_comparison = []
                for lev in leverages:
                    if portfolio_results[lev] is not None:
                        m = portfolio_results[lev]['metrics']
                        portfolio_comparison.append({
                            'Lewar': f"{lev}x",
                            'Transakcje': m['num_trades'],
                            'ROI (%)': round(m['roi'], 2),
                            'Kapitał końcowy': f"{m['final_capital']:,.2f}",
                            'Zysk/Strata': f"{m['profit_loss']:+,.2f}",
                            'Win Rate (%)': round(m['win_rate'], 1),
                            'Max Drawdown (%)': round(m['max_drawdown'], 2),
                        })
                
                portfolio_df = pd.DataFrame(portfolio_comparison)
                
                def highlight_portfolio(row):
                    if row['Lewar'] == '5x':
                        return ['background-color: #FFD700; font-weight: bold'] * len(row)
                    elif row['ROI (%)'] == portfolio_df['ROI (%)'].max() and row['ROI (%)'] > 0:
                        return ['background-color: #90EE90'] * len(row)
                    elif row['ROI (%)'] < 0:
                        return ['background-color: #FFB6C1'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    portfolio_df.style.apply(highlight_portfolio, axis=1),
                    use_container_width=True
                )
                
                st.caption(f"🟨 Złoty = Lewar x5 | 🟩 Zielony = Najlepszy ROI | 🟥 Czerwony = ROI ujemny | "
                          f"Alokacja: {capital_per_pair}")
                
                # Wykres kapitału portfela
                st.subheader("📈 Kapitał Portfela w czasie")
                fig_portfolio, ax_portfolio = plt.subplots(figsize=(14, 7))
                
                for lev in leverages:
                    if portfolio_results[lev] is not None:
                        trades = portfolio_results[lev]['trades']
                        linewidth = 3 if lev == 5 else 2
                        alpha = 1.0 if lev == 5 else 0.7
                        ax_portfolio.plot(trades['Entry_Date'], trades['Capital'], 
                                label=f'Lewar {lev}x' + (' ⭐' if lev == 5 else ''), 
                                linewidth=linewidth, marker='o', markersize=4 if lev == 5 else 3,
                                alpha=alpha)
                
                ax_portfolio.axhline(y=initial_capital, color='black', linestyle='--', 
                           linewidth=1, alpha=0.5, label='Kapitał początkowy')
                ax_portfolio.set_title('Wartość Portfela Multi-Currency', fontsize=16, fontweight='bold')
                ax_portfolio.set_xlabel('Data', fontsize=12)
                ax_portfolio.set_ylabel('Kapitał (PLN)', fontsize=12)
                ax_portfolio.legend(fontsize=10)
                ax_portfolio.grid(True, alpha=0.3)
                ax_portfolio.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_portfolio)
                
                # Wykresy ROI i Drawdown
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_roi, ax_roi = plt.subplots(figsize=(8, 6))
                    valid_results = [(lev, portfolio_results[lev]['metrics']['roi']) 
                                    for lev in leverages if portfolio_results[lev] is not None]
                    if valid_results:
                        levs, rois = zip(*valid_results)
                        colors = ['gold' if l == 5 else ('green' if r > 0 else 'red') 
                                 for l, r in zip(levs, rois)]
                        bars = ax_roi.bar([f"{l}x" for l in levs], rois, color=colors, 
                                      alpha=0.8, edgecolor='black', linewidth=2)
                        ax_roi.axhline(y=0, color='black', linestyle='-', linewidth=1)
                        ax_roi.set_title('ROI Portfela - Porównanie', fontsize=12, fontweight='bold')
                        ax_roi.set_ylabel('ROI (%)')
                        ax_roi.grid(True, alpha=0.3, axis='y')
                        
                        for bar, lev in zip(bars, levs):
                            height = bar.get_height()
                            weight = 'bold' if lev == 5 else 'normal'
                            ax_roi.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.1f}%', ha='center', 
                                    va='bottom' if height > 0 else 'top', 
                                    fontweight=weight, fontsize=10 if lev == 5 else 9)
                        plt.tight_layout()
                    st.pyplot(fig_roi)
                
                with col2:
                    fig_dd, ax_dd = plt.subplots(figsize=(8, 6))
                    valid_results = [(lev, portfolio_results[lev]['metrics']['max_drawdown']) 
                                    for lev in leverages if portfolio_results[lev] is not None]
                    if valid_results:
                        levs, dds = zip(*valid_results)
                        colors = ['gold' if l == 5 else 'orange' for l in levs]
                        bars = ax_dd.bar([f"{l}x" for l in levs], dds, color=colors, 
                                      alpha=0.8, edgecolor='black', linewidth=2)
                        ax_dd.set_title('Max Drawdown Portfela', fontsize=12, fontweight='bold')
                        ax_dd.set_ylabel('Max Drawdown (%)')
                        ax_dd.grid(True, alpha=0.3, axis='y')
                        
                        for bar, lev in zip(bars, levs):
                            height = bar.get_height()
                            weight = 'bold' if lev == 5 else 'normal'
                            ax_dd.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.1f}%', ha='center', va='bottom',
                                    fontweight=weight, fontsize=10 if lev == 5 else 9)
                        plt.tight_layout()
                    st.pyplot(fig_dd)
            
            # WYNIKI - POSZCZEGÓLNE PARY
            st.header("📊 Wyniki poszczególnych par")
            
            selected_lev_comparison = st.selectbox(
                "Wybierz lewar do porównania par:",
                leverages,
                index=leverages.index(5) if 5 in leverages else 0
            )
            
            if results_by_pair[selected_lev_comparison]:
                comparison_by_pair = []
                
                for name in pair_names:
                    if name in results_by_pair[selected_lev_comparison] and results_by_pair[selected_lev_comparison][name] is not None:
                        m = results_by_pair[selected_lev_comparison][name]['metrics']
                        comparison_by_pair.append({
                            'Para': name,
                            'Transakcje': m['num_trades'],
                            'ROI (%)': round(m['roi'], 2),
                            'Win Rate (%)': round(m['win_rate'], 1),
                            'Max Drawdown (%)': round(m['max_drawdown'], 2),
                            'Najlepsza (%)': round(m['best_trade'], 2),
                            'Najgorsza (%)': round(m['worst_trade'], 2)
                        })
                
                if comparison_by_pair:
                    pairs_df = pd.DataFrame(comparison_by_pair)
                    
                    def highlight_best_pair(row):
                        if row['ROI (%)'] == pairs_df['ROI (%)'].max() and row['ROI (%)'] > 0:
                            return ['background-color: #90EE90'] * len(row)
                        elif row['ROI (%)'] < 0:
                            return ['background-color: #FFB6C1'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(
                        pairs_df.style.apply(highlight_best_pair, axis=1),
                        use_container_width=True
                    )
                    
                    # Wykres porównania ROI par
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pairs_roi, ax_pairs_roi = plt.subplots(figsize=(10, 6))
                        colors = ['green' if r > 0 else 'red' for r in pairs_df['ROI (%)']]
                        bars = ax_pairs_roi.barh(pairs_df['Para'], pairs_df['ROI (%)'], 
                                                 color=colors, alpha=0.7, edgecolor='black')
                        ax_pairs_roi.axvline(x=0, color='black', linestyle='-', linewidth=1)
                        ax_pairs_roi.set_title(f'ROI par przy lewarze {selected_lev_comparison}x', 
                                              fontsize=12, fontweight='bold')
                        ax_pairs_roi.set_xlabel('ROI (%)')
                        ax_pairs_roi.grid(True, alpha=0.3, axis='x')
                        
                        for bar in bars:
                            width = bar.get_width()
                            ax_pairs_roi.text(width, bar.get_y() + bar.get_height()/2.,
                                            f'{width:.1f}%', ha='left' if width > 0 else 'right',
                                            va='center', fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig_pairs_roi)
                    
                    with col2:
                        fig_pairs_trades, ax_pairs_trades = plt.subplots(figsize=(10, 6))
                        ax_pairs_trades.barh(pairs_df['Para'], pairs_df['Transakcje'], 
                                            color='steelblue', alpha=0.7, edgecolor='black')
                        ax_pairs_trades.set_title(f'Liczba transakcji przy lewarze {selected_lev_comparison}x', 
                                                 fontsize=12, fontweight='bold')
                        ax_pairs_trades.set_xlabel('Liczba transakcji')
                        ax_pairs_trades.grid(True, alpha=0.3, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig_pairs_trades)
            
            # Szczegóły wybranej pary
            st.header("🔍 Szczegółowa analiza")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_pair = st.selectbox("Wybierz parę:", pair_names)
            with col2:
                selected_lev = st.selectbox("Wybierz lewar:", leverages,
                                           index=leverages.index(5) if 5 in leverages else 0)
            
            if (selected_pair in results_by_pair[selected_lev] and 
                results_by_pair[selected_lev][selected_pair] is not None):
                
                trades = results_by_pair[selected_lev][selected_pair]['trades']
                metrics = results_by_pair[selected_lev][selected_pair]['metrics']
                
                # Metryki
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Transakcje", metrics['num_trades'])
                    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                with col2:
                    st.metric("ROI", f"{metrics['roi']:.2f}%", 
                             delta=f"{metrics['profit_loss']:+,.2f} PLN")
                    st.metric("Kapitał końcowy", f"{metrics['final_capital']:,.2f} PLN")
                with col3:
                    st.metric("Najlepsza", f"{metrics['best_trade']:.2f}%")
                    st.metric("Najgorsza", f"{metrics['worst_trade']:.2f}%")
                with col4:
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                    st.metric("Średni zwrot", f"{metrics['avg_return']:.2f}%")
                
                # Tabela transakcji
                st.subheader("📋 Historia transakcji")
                trades_display = trades[['Entry_Date', 'Exit_Date', 'Signal', 
                                        'Entry_Price', 'Exit_Price', 'PnL_Leveraged_%']].copy()
                trades_display['Entry_Date'] = trades_display['Entry_Date'].dt.strftime('%Y-%m-%d')
                trades_display['Exit_Date'] = trades_display['Exit_Date'].dt.strftime('%Y-%m-%d')
                trades_display = trades_display.round(4)
                st.dataframe(trades_display, use_container_width=True, height=300)
                
                # Download
                csv = trades.to_csv(index=False)
                st.download_button(
                    label=f"💾 Pobierz transakcje {selected_pair} (CSV)",
                    data=csv,
                    file_name=f"backtest_{selected_pair}_lev{selected_lev}x.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"⚠️ Brak transakcji dla {selected_pair} z lewarem {selected_lev}x")

else:
    # Instrukcja użycia
    st.info(f"""
    ### 👋 Witaj w Multi-Currency Pivot Points Backtest Tool!
    
    **Jak używać:**
    1. 📁 Wgraj do 5 plików CSV z różnymi parami walutowymi lub kryptowalutami
    2. 🏷️ Nadaj nazwy parom (np. USDPLN, EURPLN, BTC)
    3. ⚙️ Ustaw parametry strategii
    4. 💰 Wybierz metodę alokacji kapitału
    5. 🚀 Kliknij "Uruchom backtest"
    6. 📊 Analizuj wyniki portfela i poszczególnych par!
    
    **Obsługiwane formaty:**
    - **Forex format**: Date, Price, Open, High, Low (format MM/DD/YYYY)
    - **BTC/Crypto format**: Date, Price, Open, High, Low, Vol., Change % (z przecinkami w liczbach)
    
    **Metody alokacji:**
    - **Równomiernie**: Kapitał 10,000 PLN / 3 pary = 3,333 PLN na parę
    - **Pełny kapitał**: 10,000 PLN na każdą parę (wyższe ryzyko/zwrot)
    """)
    
    # Przykładowy format
    st.subheader("📋 Przykładowe formaty plików CSV:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Format Forex:**")
        example_forex = pd.DataFrame({
            'Date': ['10/31/2025', '10/30/2025', '10/29/2025'],
            'Price': [4.2537, 4.2456, 4.2418],
            'Open': [4.2455, 4.2418, 4.2310],
            'High': [4.2615, 4.2493, 4.2481],
            'Low': [4.2394, 4.2378, 4.2284]
        })
        st.dataframe(example_forex, use_container_width=True)
    
    with col2:
        st.markdown("**Format BTC/Crypto:**")
        example_btc = pd.DataFrame({
            'Date': ['11/01/2025', '10/31/2025', '10/30/2025'],
            'Price': ['110,510.0', '109,820.0', '108,500.0'],
            'Open': ['109,820.0', '108,500.0', '110,240.0'],
            'High': ['110,750.0', '111,270.0', '111,720.0'],
            'Low': ['109,600.0', '108,490.0', '106,510.0']
        })
        st.dataframe(example_btc, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 O strategii")
st.sidebar.info(f"""
**Multi-Currency Portfolio** łączy sygnały z różnych par walutowych 
i kryptowalut w jeden zdywersyfikowany portfel.

**Duration:** {holding_days} dni
**Pivot Points:** Obliczone z {lookback_days} dni wstecz
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")
