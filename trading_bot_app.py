import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from itertools import product
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

# Tryb pracy
mode = st.sidebar.radio(
    "Tryb pracy:",
    options=["Manual Backtest", "Strategy Optimization"],
    help="Manual = testuj wybrane parametry\nOptimization = znajdź top 5 kombinacji"
)

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
            help="Plik CSV z danymi historycznymi (Forex lub BTC/Crypto format)"
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

if mode == "Manual Backtest":
    # Parametry manualne
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
        max_value=20,
        value=5,
        step=1
    )

    strategy_mode = st.sidebar.radio(
        "Strategia transakcyjna:",
        options=["Both (Buy & Sell)", "Buy Only", "Sell Only"],
        help="Both = kupuj i sprzedawaj\nBuy Only = tylko kupuj\nSell Only = tylko sprzedawaj (short)"
    )

    leverages = st.sidebar.multiselect(
        "Wybierz lewary do przetestowania",
        options=[1, 2, 3, 5, 10, 15, 20],
        default=[1, 5, 10, 20]
    )
else:
    # Parametry optymalizacji
    st.sidebar.markdown("**Zakresy parametrów do optymalizacji:**")
    
    lookback_range = st.sidebar.multiselect(
        "Lookback days",
        options=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        default=[10, 20, 30]
    )
    
    threshold_range = st.sidebar.multiselect(
        "Próg % od PP",
        options=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0],
        default=[0.3, 0.5, 1.0]
    )
    
    holding_range = st.sidebar.multiselect(
        "Holding days",
        options=[1, 3, 5, 7, 10, 15, 20],
        default=[5, 10, 15]
    )
    
    strategy_range = st.sidebar.multiselect(
        "Strategie",
        options=["Both (Buy & Sell)", "Buy Only", "Sell Only"],
        default=["Both (Buy & Sell)", "Buy Only"]
    )
    
    leverage_opt = st.sidebar.selectbox(
        "Lewar do optymalizacji",
        options=[1, 2, 3, 5, 10, 15, 20],
        index=3
    )
    
    optimization_metric = st.sidebar.selectbox(
        "Optymalizuj według:",
        options=["ROI (%)", "Sharpe Ratio", "Win Rate (%)", "Profit Factor"],
        help="Metryka do rankingu strategii"
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

def run_backtest(df, threshold_pct, lookback, leverage, initial_capital, holding_days, pair_name="", strategy_mode="Both (Buy & Sell)"):
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
                
                # Określ sygnał na podstawie strategii
                if strategy_mode == "Both (Buy & Sell)":
                    if monday_price <= buy_threshold:
                        signal = 'BUY'
                        pnl_pct = ((exit_price - monday_price) / monday_price) * 100
                    elif monday_price >= sell_threshold:
                        signal = 'SELL'
                        pnl_pct = ((monday_price - exit_price) / monday_price) * 100
                elif strategy_mode == "Buy Only":
                    if monday_price <= buy_threshold:
                        signal = 'BUY'
                        pnl_pct = ((exit_price - monday_price) / monday_price) * 100
                elif strategy_mode == "Sell Only":
                    if monday_price >= sell_threshold:
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
    
    # Sharpe Ratio (uproszczony)
    returns = trades_df['PnL_Leveraged_%']
    sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0
    
    # Profit Factor
    gross_profit = trades_df[trades_df['PnL_Leveraged_%'] > 0]['PnL_Leveraged_%'].sum()
    gross_loss = abs(trades_df[trades_df['PnL_Leveraged_%'] < 0]['PnL_Leveraged_%'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    
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
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'profit_factor': profit_factor
    }

def optimize_strategy(all_data, lookback_range, threshold_range, holding_range, 
                     strategy_range, leverage, initial_capital, capital_per_pair, optimization_metric):
    """Optymalizuj strategię - znajdź top 5 kombinacji"""
    
    # Generuj wszystkie kombinacje
    combinations = list(product(lookback_range, threshold_range, holding_range, strategy_range))
    
    st.info(f"🔍 Testuję {len(combinations)} kombinacji parametrów...")
    
    results = []
    progress_bar = st.progress(0)
    
    for idx, (lookback, threshold, holding, strategy) in enumerate(combinations):
        # Test dla każdej kombinacji
        pair_trades = []
        
        for name, df in all_data.items():
            if capital_per_pair == "Równomiernie na wszystkie pary":
                capital_for_pair = initial_capital / len(all_data)
            else:
                capital_for_pair = initial_capital
            
            trades = run_backtest(df, threshold, lookback, leverage, capital_for_pair, holding, name, strategy)
            
            if trades is not None:
                pair_trades.append(trades)
        
        # Połącz w portfel
        if pair_trades:
            portfolio_trades = combine_portfolio_trades(pair_trades, initial_capital, len(all_data), capital_per_pair)
            if portfolio_trades is not None:
                metrics = calculate_metrics(portfolio_trades, initial_capital)
                
                if metrics:
                    results.append({
                        'Lookback': lookback,
                        'Threshold (%)': threshold,
                        'Holding Days': holding,
                        'Strategy': strategy,
                        'Leverage': f"{leverage}x",
                        'Trades': metrics['num_trades'],
                        'ROI (%)': round(metrics['roi'], 2),
                        'Win Rate (%)': round(metrics['win_rate'], 1),
                        'Sharpe Ratio': round(metrics['sharpe_ratio'], 2),
                        'Profit Factor': round(metrics['profit_factor'], 2),
                        'Max DD (%)': round(metrics['max_drawdown'], 2),
                        'Final Capital': metrics['final_capital'],
                        'portfolio_trades': portfolio_trades
                    })
        
        progress_bar.progress((idx + 1) / len(combinations))
    
    progress_bar.empty()
    
    if not results:
        return None
    
    # Sortuj według wybranej metryki
    results_df = pd.DataFrame(results)
    
    # Usuń portfolio_trades z DataFrame do sortowania
    portfolio_trades_dict = {i: r['portfolio_trades'] for i, r in enumerate(results)}
    results_df_display = results_df.drop('portfolio_trades', axis=1)
    
    # Sortuj
    if optimization_metric == "ROI (%)":
        results_df_display = results_df_display.sort_values('ROI (%)', ascending=False)
    elif optimization_metric == "Sharpe Ratio":
        results_df_display = results_df_display.sort_values('Sharpe Ratio', ascending=False)
    elif optimization_metric == "Win Rate (%)":
        results_df_display = results_df_display.sort_values('Win Rate (%)', ascending=False)
    elif optimization_metric == "Profit Factor":
        results_df_display = results_df_display.sort_values('Profit Factor', ascending=False)
    
    # Weź top 5
    top_5 = results_df_display.head(5).reset_index(drop=True)
    
    # Dodaj portfolio_trades do top 5
    top_5_with_trades = []
    for idx in top_5.index:
        original_idx = results_df_display.index[idx]
        row_dict = top_5.loc[idx].to_dict()
        row_dict['portfolio_trades'] = portfolio_trades_dict[original_idx]
        top_5_with_trades.append(row_dict)
    
    return top_5_with_trades

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
        
        # TRYB OPTYMALIZACJI
        if mode == "Strategy Optimization":
            if st.sidebar.button("🔍 Znajdź Top 5 Strategii", type="primary"):
                if not lookback_range or not threshold_range or not holding_range or not strategy_range:
                    st.error("❌ Wybierz przynajmniej jedną wartość dla każdego parametru!")
                else:
                    top_5 = optimize_strategy(
                        all_data, lookback_range, threshold_range, holding_range,
                        strategy_range, leverage_opt, initial_capital, capital_per_pair, optimization_metric
                    )
                    
                    if top_5:
                        st.header("🏆 Top 5 Najlepszych Strategii")
                        st.caption(f"Ranking według: {optimization_metric}")
                        
                        # Tabela top 5
                        display_df = pd.DataFrame([{k: v for k, v in s.items() if k != 'portfolio_trades'} for s in top_5])
                        display_df.index = ['🥇 #1', '🥈 #2', '🥉 #3', '4️⃣ #4', '5️⃣ #5']
                        
                        def highlight_top(row):
                            if row.name == '🥇 #1':
                                return ['background-color: #FFD700; font-weight: bold'] * len(row)
                            elif row.name == '🥈 #2':
                                return ['background-color: #C0C0C0; font-weight: bold'] * len(row)
                            elif row.name == '🥉 #3':
                                return ['background-color: #CD7F32; font-weight: bold'] * len(row)
                            return [''] * len(row)
                        
                        st.dataframe(
                            display_df.style.apply(highlight_top, axis=1),
                            use_container_width=True
                        )
                        
                        # Wykres porównawczy Top 5
                        st.subheader("📊 Porównanie Top 5 Strategii")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_roi, ax_roi = plt.subplots(figsize=(10, 6))
                            colors = ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen']
                            bars = ax_roi.barh(
                                ['#1', '#2', '#3', '#4', '#5'],
                                [s['ROI (%)'] for s in top_5],
                                color=colors,
                                edgecolor='black',
                                linewidth=2
                            )
                            ax_roi.set_xlabel('ROI (%)', fontsize=12)
                            ax_roi.set_title('ROI - Top 5', fontsize=14, fontweight='bold')
                            ax_roi.grid(True, alpha=0.3, axis='x')
                            
                            for bar, roi in zip(bars, [s['ROI (%)'] for s in top_5]):
                                width = bar.get_width()
                                ax_roi.text(width, bar.get_y() + bar.get_height()/2.,
                                          f'{width:.1f}%', ha='left', va='center',
                                          fontweight='bold', fontsize=10)
                            plt.tight_layout()
                            st.pyplot(fig_roi)
                        
                        with col2:
                            fig_sharpe, ax_sharpe = plt.subplots(figsize=(10, 6))
                            bars = ax_sharpe.barh(
                                ['#1', '#2', '#3', '#4', '#5'],
                                [s['Sharpe Ratio'] for s in top_5],
                                color=colors,
                                edgecolor='black',
                                linewidth=2
                            )
                            ax_sharpe.set_xlabel('Sharpe Ratio', fontsize=12)
                            ax_sharpe.set_title('Sharpe Ratio - Top 5', fontsize=14, fontweight='bold')
                            ax_sharpe.grid(True, alpha=0.3, axis='x')
                            
                            for bar, sharpe in zip(bars, [s['Sharpe Ratio'] for s in top_5]):
                                width = bar.get_width()
                                ax_sharpe.text(width, bar.get_y() + bar.get_height()/2.,
                                             f'{width:.2f}', ha='left', va='center',
                                             fontweight='bold', fontsize=10)
                            plt.tight_layout()
                            st.pyplot(fig_sharpe)
                        
                        # Equity curves dla top 5
                        st.subheader("📈 Krzywe Kapitału - Top 5 Strategii")
                        fig_equity, ax_equity = plt.subplots(figsize=(14, 7))
                        
                        for idx, strategy in enumerate(top_5):
                            trades = strategy['portfolio_trades']
                            label = f"#{idx+1}: L{strategy['Lookback']} T{strategy['Threshold (%)']}% H{strategy['Holding Days']} {strategy['Strategy'][:4]}"
                            ax_equity.plot(trades['Entry_Date'], trades['Capital'],
                                         label=label, linewidth=2, marker='o', markersize=3,
                                         alpha=0.8)
                        
                        ax_equity.axhline(y=initial_capital, color='black', linestyle='--',
                                        linewidth=1, alpha=0.5, label='Start Capital')
                        ax_equity.set_title('Porównanie Kapitału - Top 5 Strategii', fontsize=16, fontweight='bold')
                        ax_equity.set_xlabel('Data', fontsize=12)
                        ax_equity.set_ylabel('Kapitał', fontsize=12)
                        ax_equity.legend(fontsize=9, loc='best')
                        ax_equity.grid(True, alpha=0.3)
                        ax_equity.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_equity)
                        
                        # Szczegóły wybranej strategii z top 5
                        st.header("🔍 Szczegóły Wybranej Strategii")
                        selected_rank = st.selectbox(
                            "Wybierz strategię:",
                            options=['#1', '#2', '#3', '#4', '#5']
                        )
                        
                        rank_idx = int(selected_rank[1]) - 1
                        selected_strategy = top_5[rank_idx]
                        
                        st.markdown(f"""
                        **Parametry:**
                        - Lookback: {selected_strategy['Lookback']} dni
                        - Threshold: {selected_strategy['Threshold (%)']}%
                        - Holding: {selected_strategy['Holding Days']} dni
                        - Strategy: {selected_strategy['Strategy']}
                        - Leverage: {selected_strategy['Leverage']}
                        """)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("ROI", f"{selected_strategy['ROI (%)']:.2f}%")
                            st.metric("Transakcje", selected_strategy['Trades'])
                        with col2:
                            st.metric("Win Rate", f"{selected_strategy['Win Rate (%)']:.1f}%")
                            st.metric("Sharpe Ratio", f"{selected_strategy['Sharpe Ratio']:.2f}")
                        with col3:
                            st.metric("Profit Factor", f"{selected_strategy['Profit Factor']:.2f}")
                            st.metric("Max DD", f"{selected_strategy['Max DD (%)']:.2f}%")
                        with col4:
                            st.metric("Final Capital", f"{selected_strategy['Final Capital']:,.2f}")
                        
                        # Tabela transakcji
                        st.subheader("📋 Historia Transakcji")
                        trades_display = selected_strategy['portfolio_trades'][['Entry_Date', 'Exit_Date', 'Signal', 
                                                                                 'Entry_Price', 'Exit_Price', 'PnL_Leveraged_%', 'Pair']].copy()
                        trades_display['Entry_Date'] = trades_display['Entry_Date'].dt.strftime('%Y-%m-%d')
                        trades_display['Exit_Date'] = trades_display['Exit_Date'].dt.strftime('%Y-%m-%d')
                        trades_display = trades_display.round(4)
                        st.dataframe(trades_display, use_container_width=True, height=300)
                        
                        # Download
                        csv = selected_strategy['portfolio_trades'].to_csv(index=False)
                        st.download_button(
                            label=f"💾 Pobierz transakcje strategii {selected_rank}",
                            data=csv,
                            file_name=f"top_strategy_{selected_rank}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ Nie znaleziono żadnych strategii spełniających kryteria")
        
        # TRYB MANUAL BACKTEST (poprzedni kod)
        else:
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
                        
                        trades = run_backtest(df, threshold, lookback_days, lev, capital_for_pair, holding_days, name, strategy_mode)
                        
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
                
                # [Reszta poprzedniego kodu Manual Backtest - wyświetlanie wyników]
                # WYNIKI - PORTFEL
                st.header("🎯 Wyniki Portfela (Wszystkie pary razem)")
                
                if not portfolio_results or all(v is None for v in portfolio_results.values()):
                    st.warning("⚠️ Brak transakcji w portfelu")
                else:
                    # ... (cały poprzedni kod wyświetlania wyników)
                    pass

else:
    # Instrukcja użycia
    st.info(f"""
    ### 👋 Witaj w Multi-Currency Pivot Points Backtest Tool!
    
    **Dwa tryby pracy:**
    
    **1️⃣ Manual Backtest:**
    - Testuj wybrane przez siebie parametry
    - Porównaj różne lewary
    - Analizuj szczegółowe wyniki
    
    **2️⃣ Strategy Optimization:** ⭐
    - Automatycznie znajdź top 5 najlepszych kombinacji parametrów
    - Testuje setki kombinacji: lookback, threshold, holding, strategy
    - Ranking według wybranej metryki (ROI, Sharpe Ratio, Win Rate, Profit Factor)
    - Porównanie equity curves dla top 5
    
    **Jak używać:**
    1. 📁 Wgraj do 5 plików CSV
    2. 🏷️ Nadaj nazwy parom
    3. 🎯 Wybierz tryb (Manual / Optimization)
    4. ⚙️ Ustaw parametry
    5. 🚀 Uruchom!
    
    **Obsługiwane formaty:**
    - **Forex**: Date, Price, Open, High, Low
    - **BTC/Crypto**: Date, Price, Open, High, Low, Vol., Change %
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 O aplikacji")
if mode == "Manual Backtest":
    st.sidebar.info(f"""
    **Manual Backtest Mode**
    
    Testuj wybraną strategię z określonymi parametrami.
    """)
else:
    st.sidebar.info(f"""
    **Optimization Mode** 🔍
    
    Automatycznie znajduje najlepsze kombinacje parametrów metodą brute-force.
    
    Liczba kombinacji: {len(lookback_range) * len(threshold_range) * len(holding_range) * len(strategy_range) if 'lookback_range' in locals() else 0}
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")
