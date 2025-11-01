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
st.markdown("### Testuj strategię pivot points na koszyku par walutowych")

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
            f"Para walutowa #{i+1}", 
            type=['csv'],
            key=f"file_{i}",
            help="Plik CSV z kolumnami: Date, Price, Open, High, Low"
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
    st.success(f"✅ Wczytano {len(uploaded_files)} par walutowych: {', '.join(pair_names)}")
    
    # Wczytaj wszystkie dane
    all_data = {}
    for file, name in zip(uploaded_files, pair_names):
        df, error = load_and_prepare_data(file)
        if error:
            st.error(f"❌ Błąd w pliku {name}: {error}")
        else:
            all_data[name] = df
            st.info(f"📊 {name}: {len(df)} rekordów ({df['Date'].min().date()} - {df['Date'].max().date()})")
    
    if len(all_data) == 0:
        st.error("❌ Nie udało się wczytać żadnych danych")
    else:
        # Pokaż przykładowe dane
        with st.expander("📋 Podgląd danych wszystkich par"):
            for name, df in all_data.items():
                st.subheader(name)
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
            
            # ============================================
            # SEKCJA 1: WYNIKI KAŻDEJ PARY OSOBNO
            # ============================================
            st.header("📈 WYNIKI POSZCZEGÓLNYCH PAR")
            st.markdown("---")
            
            # Wybór lewaru dla porównania par
            selected_lev_pairs = st.selectbox(
                "🎚️ Wybierz lewar do analizy par:",
                leverages,
                index=leverages.index(5) if 5 in leverages else 0,
                key="lev_pairs"
            )
            
            if results_by_pair[selected_lev_pairs]:
                # Dla każdej pary osobny panel
                for name in pair_names:
                    if name in results_by_pair[selected_lev_pairs] and results_by_pair[selected_lev_pairs][name] is not None:
                        
                        st.subheader(f"💱 {name}")
                        
                        trades = results_by_pair[selected_lev_pairs][name]['trades']
                        m = results_by_pair[selected_lev_pairs][name]['metrics']
                        
                        # Metryki w 5 kolumnach
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Transakcje", m['num_trades'])
                        with col2:
                            st.metric("ROI", f"{m['roi']:.2f}%", 
                                     delta=f"{m['profit_loss']:+,.0f} PLN")
                        with col3:
                            st.metric("Win Rate", f"{m['win_rate']:.1f}%")
                        with col4:
                            st.metric("Kapitał końcowy", f"{m['final_capital']:,.0f} PLN")
                        with col5:
                            st.metric("Max Drawdown", f"{m['max_drawdown']:.2f}%")
                        
                        # Mini wykres kapitału
                        fig_mini, ax_mini = plt.subplots(figsize=(12, 3))
                        color = 'green' if m['roi'] > 0 else 'red'
                        ax_mini.plot(trades['Entry_Date'], trades['Capital'], 
                                    color=color, linewidth=2, marker='o', markersize=3)
                        
                        if capital_per_pair == "Równomiernie na wszystkie pary":
                            cap_for_pair = initial_capital / len(all_data)
                        else:
                            cap_for_pair = initial_capital
                        
                        ax_mini.axhline(y=cap_for_pair, color='gray', linestyle='--', 
                                       linewidth=1, alpha=0.5)
                        ax_mini.set_ylabel('Kapitał (PLN)', fontsize=9)
                        ax_mini.grid(True, alpha=0.3)
                        ax_mini.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                        plt.xticks(rotation=45, fontsize=8)
                        st.pyplot(fig_mini)
                        plt.close()
                        
                        # Tabela z ostatnimi transakcjami
                        with st.expander(f"📋 Ostatnie 10 transakcji - {name}"):
                            trades_display = trades[['Entry_Date', 'Exit_Date', 'Signal', 
                                                    'Entry_Price', 'Exit_Price', 'PnL_Leveraged_%']].tail(10).copy()
                            trades_display['Entry_Date'] = trades_display['Entry_Date'].dt.strftime('%Y-%m-%d')
                            trades_display['Exit_Date'] = trades_display['Exit_Date'].dt.strftime('%Y-%m-%d')
                            trades_display = trades_display.round(4)
                            st.dataframe(trades_display, use_container_width=True, height=300)
                        
                        st.markdown("---")
                    
                    else:
                        st.subheader(f"💱 {name}")
                        st.warning(f"⚠️ Brak transakcji dla tej pary przy lewarze {selected_lev_pairs}x")
                        st.markdown("---")
                
                # Tabela porównawcza wszystkich par
                st.subheader("📊 Porównanie wszystkich par")
                
                comparison_by_pair = []
                for name in pair_names:
                    if name in results_by_pair[selected_lev_pairs] and results_by_pair[selected_lev_pairs][name] is not None:
                        m = results_by_pair[selected_lev_pairs][name]['metrics']
                        comparison_by_pair.append({
                            'Para': name,
                            'Transakcje': m['num_trades'],
                            'ROI (%)': round(m['roi'], 2),
                            'Kapitał końcowy': f"{m['final_capital']:,.0f}",
                            'Zysk/Strata': f"{m['profit_loss']:+,.0f}",
                            'Win Rate (%)': round(m['win_rate'], 1),
                            'Max DD (%)': round(m['max_drawdown'], 2),
                            'Średni zwrot (%)': round(m['avg_return'], 2)
                        })
                    else:
                        comparison_by_pair.append({
                            'Para': name,
                            'Transakcje': 0,
                            'ROI (%)': 0,
                            'Kapitał końcowy': "0",
                            'Zysk/Strata': "0",
                            'Win Rate (%)': 0,
                            'Max DD (%)': 0,
                            'Średni zwrot (%)': 0
                        })
                
                if comparison_by_pair:
                    pairs_df = pd.DataFrame(comparison_by_pair)
                    
                    def highlight_best_pair(row):
                        if row['ROI (%)'] == pairs_df['ROI (%)'].max() and row['ROI (%)'] > 0:
                            return ['background-color: #90EE90; font-weight: bold'] * len(row)
                        elif row['ROI (%)'] < 0:
                            return ['background-color: #FFB6C1'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(
                        pairs_df.style.apply(highlight_best_pair, axis=1),
                        use_container_width=True
                    )
                    
                    st.caption("🟩 Zielony = Najlepsza para | 🟥 Czerwony = ROI ujemny")
                    
                    # Wykresy porównawcze
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_roi, ax_roi = plt.subplots(figsize=(8, 5))
                        colors = ['green' if r > 0 else 'red' for r in pairs_df['ROI (%)']]
                        bars = ax_roi.barh(pairs_df['Para'], pairs_df['ROI (%)'], 
                                          color=colors, alpha=0.7, edgecolor='black')
                        ax_roi.axvline(x=0, color='black', linestyle='-', linewidth=1)
                        ax_roi.set_title(f'ROI poszczególnych par (Lewar {selected_lev_pairs}x)', 
                                        fontsize=11, fontweight='bold')
                        ax_roi.set_xlabel('ROI (%)')
                        ax_roi.grid(True, alpha=0.3, axis='x')
                        
                        for bar in bars:
                            width = bar.get_width()
                            if width != 0:
                                ax_roi.text(width, bar.get_y() + bar.get_height()/2.,
                                          f'{width:.1f}%', ha='left' if width > 0 else 'right',
                                          va='center', fontweight='bold', fontsize=9)
                        st.pyplot(fig_roi)
                        plt.close()
                    
                    with col2:
                        fig_winrate, ax_winrate = plt.subplots(figsize=(8, 5))
                        ax_winrate.barh(pairs_df['Para'], pairs_df['Win Rate (%)'], 
                                       color='steelblue', alpha=0.7, edgecolor='black')
                        ax_winrate.set_title(f'Win Rate poszczególnych par (Lewar {selected_lev_pairs}x)', 
                                           fontsize=11, fontweight='bold')
                        ax_winrate.set_xlabel('Win Rate (%)')
                        ax_winrate.set_xlim(0, 100)
                        ax_winrate.grid(True, alpha=0.3, axis='x')
                        
                        for i, (para, wr) in enumerate(zip(pairs_df['Para'], pairs_df['Win Rate (%)'])):
                            if wr > 0:
                                ax_winrate.text(wr, i, f'{wr:.1f}%', 
                                              ha='left', va='center', fontweight='bold', fontsize=9)
                        st.pyplot(fig_winrate)
                        plt.close()
            
            # ============================================
            # SEKCJA 2: WYNIK KOSZYKA (PORTFELA)
            # ============================================
            st.markdown("---")
            st.header("🎯 WYNIK KOSZYKA (PORTFOLIO)")
            st.markdown("### Połączone wyniki wszystkich par razem")
            
            if not portfolio_results or all(v is None for v in portfolio_results.values()):
                st.warning("⚠️ Brak transakcji w portfelu")
            else:
                # Wybór lewaru dla portfela
                selected_lev_portfolio = st.selectbox(
                    "🎚️ Wybierz lewar do analizy portfela:",
                    leverages,
                    index=leverages.index(5) if 5 in leverages else 0,
                    key="lev_portfolio"
                )
                
                if portfolio_results[selected_lev_portfolio] is not None:
                    portfolio_trades = portfolio_results[selected_lev_portfolio]['trades']
                    portfolio_metrics = portfolio_results[selected_lev_portfolio]['metrics']
                    
                    # Duże metryki portfela
                    st.subheader(f"📊 Wyniki portfela przy lewarze {selected_lev_portfolio}x")
                    
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    with col1:
                        st.metric("Wszystkich transakcji", portfolio_metrics['num_trades'])
                    with col2:
                        st.metric("**ROI PORTFELA**", 
                                 f"{portfolio_metrics['roi']:.2f}%",
                                 delta=f"{portfolio_metrics['profit_loss']:+,.0f} PLN")
                    with col3:
                        st.metric("Win Rate", f"{portfolio_metrics['win_rate']:.1f}%")
                    with col4:
                        st.metric("Kapitał końcowy", f"{portfolio_metrics['final_capital']:,.0f} PLN")
                    with col5:
                        st.metric("Max Drawdown", f"{portfolio_metrics['max_drawdown']:.2f}%")
                    with col6:
                        st.metric("Średni zwrot", f"{portfolio_metrics['avg_return']:.2f}%")
                    
                    # Duży wykres kapitału portfela
                    st.subheader("📈 Kapitał portfela w czasie")
                    fig_port_main, ax_port_main = plt.subplots(figsize=(14, 6))
                    
                    color = 'green' if portfolio_metrics['roi'] > 0 else 'red'
                    ax_port_main.plot(portfolio_trades['Entry_Date'], portfolio_trades['Capital'], 
                                     color=color, linewidth=3, marker='o', markersize=4, 
                                     label=f"Portfolio (ROI: {portfolio_metrics['roi']:.2f}%)")
                    
                    ax_port_main.axhline(y=initial_capital, color='black', linestyle='--', 
                                        linewidth=2, alpha=0.7, label='Kapitał początkowy')
                    ax_port_main.set_title(f'Wartość Portfela Multi-Currency (Lewar {selected_lev_portfolio}x)', 
                                          fontsize=14, fontweight='bold')
                    ax_port_main.set_xlabel('Data', fontsize=11)
                    ax_port_main.set_ylabel('Kapitał (PLN)', fontsize=11)
                    ax_port_main.legend(fontsize=10)
                    ax_port_main.grid(True, alpha=0.3)
                    ax_port_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    plt.xticks(rotation=45)
                    st.pyplot(fig_port_main)
                    plt.close()
                    
                    # Informacja o alokacji
                    st.info(f"""
                    **Metoda alokacji:** {capital_per_pair}
                    
                    {'- Kapitał podzielony równomiernie na ' + str(len(all_data)) + ' par' if capital_per_pair == "Równomiernie na wszystkie pary" else '- Pełny kapitał zaangażowany w każdą parę'}
                    """)
                    
                    # Rozkład transakcji po parach w portfelu
                    st.subheader("📊 Udział par w portfelu")
                    pair_contribution = portfolio_trades.groupby('Pair').agg({
                        'PnL_Leveraged_%': ['count', 'sum', 'mean']
                    }).round(2)
                    pair_contribution.columns = ['Liczba transakcji', 'Suma ROI (%)', 'Średni ROI (%)']
                    pair_contribution = pair_contribution.sort_values('Suma ROI (%)', ascending=False)
                    st.dataframe(pair_contribution, use_container_width=True)
                    
                    # Historia transakcji portfela
                    with st.expander("📋 Historia wszystkich transakcji portfela"):
                        portfolio_display = portfolio_trades[['Entry_Date', 'Exit_Date', 'Pair', 'Signal', 
                                                              'Entry_Price', 'Exit_Price', 'PnL_Leveraged_%']].copy()
                        portfolio_display['Entry_Date'] = portfolio_display['Entry_Date'].dt.strftime('%Y-%m-%d')
                        portfolio_display['Exit_Date'] = portfolio_display['Exit_Date'].dt.strftime('%Y-%m-%d')
                        portfolio_display = portfolio_display.round(4)
                        st.dataframe(portfolio_display, use_container_width=True, height=400)
                    
                    # Download portfela
                    csv_portfolio = portfolio_trades.to_csv(index=False)
                    st.download_button(
                        label=f"💾 Pobierz wszystkie transakcje portfela (CSV)",
                        data=csv_portfolio,
                        file_name=f"portfolio_lev{selected_lev_portfolio}x.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.warning(f"⚠️ Brak danych portfela dla lewaru {selected_lev_portfolio}x")
                
                # Tabela porównawcza różnych lewarów dla portfela
                st.markdown("---")
                st.subheader("⚖️ Porównanie lewarów dla całego portfela")
                
                portfolio_comparison = []
                for lev in leverages:
                    if portfolio_results[lev] is not None:
                        m = portfolio_results[lev]['metrics']
                        portfolio_comparison.append({
                            'Lewar': f"{lev}x",
                            'Transakcje': m['num_trades'],
                            'ROI (%)': round(m['roi'], 2),
                            'Kapitał końcowy': f"{m['final_capital']:,.0f}",
                            'Zysk/Strata': f"{m['profit_loss']:+,.0f}",
                            'Win Rate (%)': round(m['win_rate'], 1),
                            'Max Drawdown (%)': round(m['max_drawdown'], 2),
                        })
                
                if portfolio_comparison:
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
                    
                    st.caption("🟨 Złoty = Lewar x5 | 🟩 Zielony = Najlepszy ROI | 🟥 Czerwony = ROI ujemny")
                    
                    # Wykres porównania lewarów
                    fig_lev_comp, ax_lev_comp = plt.subplots(figsize=(10, 5))
                    colors = ['gold' if lev == '5x' else ('green' if roi > 0 else 'red') 
                             for lev, roi in zip(portfolio_df['Lewar'], portfolio_df['ROI (%)'])]
                    bars = ax_lev_comp.bar(portfolio_df['Lewar'], portfolio_df['ROI (%)'], 
                                          color=colors, alpha=0.8, edgecolor='black', linewidth=2)
                    ax_lev_comp.axhline(y=0, color='black', linestyle='-', linewidth=1)
                    ax_lev_comp.set_title('ROI Portfela przy różnych lewarach', fontsize=13, fontweight='bold')
                    ax_lev_comp.set_xlabel('Lewar')
                    ax_lev_comp.set_ylabel('ROI (%)')
                    ax_lev_comp.grid(True, alpha=0.3, axis='y')
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax_lev_comp.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{height:.1f}%', ha='center',
