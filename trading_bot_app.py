import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Wczytaj dane
df = pd.read_csv('/mnt/user-data/uploads/EUR_PLN_Historical_Data__1_.csv')

# Usuń BOM i przetwórz dane
df.columns = df.columns.str.replace('ï»¿', '')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

# Konwersja kolumn na float
for col in ['Price', 'Open', 'High', 'Low']:
    df[col] = df[col].astype(float)

# Dodaj dzień tygodnia
df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Poniedziałek, 4=Piątek

print("Zakres danych:", df['Date'].min(), "do", df['Date'].max())
print("Liczba rekordów:", len(df))

# Funkcja do obliczania pivot points
def calculate_pivot_points(high, low, close):
    """
    Oblicza klasyczne pivot points
    """
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    r2 = pp + (high - low)
    r3 = high + 2 * (pp - low)
    s1 = 2 * pp - high
    s2 = pp - (high - low)
    s3 = low - 2 * (high - pp)
    
    return pp, r1, r2, r3, s1, s2, s3

print("\n" + "="*100)
print("NOWA STRATEGIA:")
print("="*100)
print("1. W piątek: Oblicz pivot points z ostatnich 20 dni")
print("2. W piątek: Sprawdź czy kurs zamknięcia jest <= S3 lub >= R3")
print("3. W poniedziałek: Otwórz pozycję po cenie otwarcia")
print("4. Po 5 dniach: Zamknij pozycję")
print("="*100)

# Znajdź wszystkie piątki
fridays = df[df['DayOfWeek'] == 4].copy()
print(f"\nZnaleziono {len(fridays)} piątków")

# Dla każdego piątku sprawdź sygnał
trades = []
LOOKBACK_DAYS = 20

for idx, friday_row in fridays.iterrows():
    friday_date = friday_row['Date']
    friday_close = friday_row['Price']  # Cena zamknięcia w piątek
    
    # Znajdź ostatnie 20 dni PRZED piątkiem (włącznie z piątkiem)
    prev_days = df[df['Date'] <= friday_date].tail(20)
    
    if len(prev_days) >= 20:
        # Oblicz pivot points z ostatnich 20 dni
        high_20d = prev_days['High'].max()
        low_20d = prev_days['Low'].min()
        close_20d = prev_days['Price'].iloc[-2] if len(prev_days) > 1 else prev_days['Price'].iloc[-1]  # Przedostatnie zamknięcie
        
        pp, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(high_20d, low_20d, close_20d)
        
        # Sprawdź sygnał w piątek (zmienione na S1/R1 bo S3/R3 zbyt ekstremalne)
        signal = None
        if friday_close <= s1:
            signal = 'BUY'
        elif friday_close >= r1:
            signal = 'SELL'
        
        if signal:
            # Znajdź następny poniedziałek
            future_dates = df[df['Date'] > friday_date]
            monday = future_dates[future_dates['DayOfWeek'] == 0].head(1)
            
            if not monday.empty:
                monday_date = monday.iloc[0]['Date']
                monday_open = monday.iloc[0]['Open']  # Otwieramy po cenie otwarcia w poniedziałek
                
                # Znajdź dzień zamknięcia (5 dni roboczych po poniedziałku)
                days_after_monday = df[df['Date'] > monday_date].head(5)
                
                if len(days_after_monday) >= 5:
                    close_date = days_after_monday.iloc[4]['Date']  # 5-ty dzień
                    close_price = days_after_monday.iloc[4]['Price']  # Cena zamknięcia
                    
                    # Oblicz PnL
                    if signal == 'BUY':
                        pnl_pct = ((close_price - monday_open) / monday_open) * 100
                    else:  # SELL
                        pnl_pct = ((monday_open - close_price) / monday_open) * 100
                    
                    trades.append({
                        'Friday_Signal_Date': friday_date,
                        'Friday_Close': friday_close,
                        'Monday_Entry_Date': monday_date,
                        'Monday_Entry_Price': monday_open,
                        'Close_Date': close_date,
                        'Close_Price': close_price,
                        'Signal': signal,
                        'PnL_%': pnl_pct,
                        'PP': pp,
                        'S3': s3,
                        'R3': r3,
                        'High_20d': high_20d,
                        'Low_20d': low_20d,
                        'Year': monday_date.year,
                        'Month': monday_date.month,
                        'Year_Month': monday_date.to_period('M')
                    })

# Utwórz DataFrame z wynikami
if not trades:
    print("\n" + "="*100)
    print("❌ BRAK TRANSAKCJI!")
    print("="*100)
    print("Strategia nie wygenerowała żadnych sygnałów spełniających kryteria (piątek <= S3 lub >= R3)")
    print("\nSprawdźmy jak blisko były ceny do poziomów S3/R3...")
    
    # Analiza
    analysis_data = []
    for idx, friday_row in fridays.iterrows():
        friday_date = friday_row['Date']
        friday_close = friday_row['Price']
        prev_days = df[df['Date'] <= friday_date].tail(20)
        
        if len(prev_days) >= 20:
            high_20d = prev_days['High'].max()
            low_20d = prev_days['Low'].min()
            close_20d = prev_days['Price'].iloc[-2] if len(prev_days) > 1 else prev_days['Price'].iloc[-1]
            pp, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(high_20d, low_20d, close_20d)
            
            dist_s3 = ((friday_close - s3) / s3) * 100
            dist_r3 = ((friday_close - r3) / r3) * 100
            
            analysis_data.append({
                'Date': friday_date,
                'Close': friday_close,
                'S3': s3,
                'R3': r3,
                'Dist_S3_%': dist_s3,
                'Dist_R3_%': dist_r3
            })
    
    analysis_df = pd.DataFrame(analysis_data)
    print(f"\nNajbliższe S3: {analysis_df['Dist_S3_%'].min():.2f}% (powyżej)")
    print(f"Najbliższe R3: {analysis_df['Dist_R3_%'].max():.2f}% (poniżej)")
    
    print("\n5 piątków najbliższych S3:")
    closest_s3 = analysis_df.nsmallest(5, 'Dist_S3_%')[['Date', 'Close', 'S3', 'Dist_S3_%']]
    print(closest_s3.to_string(index=False))
    
    print("\n5 piątków najbliższych R3:")
    closest_r3 = analysis_df.nlargest(5, 'Dist_R3_%')[['Date', 'Close', 'R3', 'Dist_R3_%']]
    print(closest_r3.to_string(index=False))
    
else:
    results_df = pd.DataFrame(trades)
    
    print("\n" + "="*100)
    print("WYNIKI NOWEJ STRATEGII")
    print("="*100)
    print(f"Liczba sygnałów z piątków: {len(results_df)}")
    print(f"Transakcje BUY: {len(results_df[results_df['Signal'] == 'BUY'])}")
    print(f"Transakcje SELL: {len(results_df[results_df['Signal'] == 'SELL'])}")
    
    # Statystyki
    total_return = results_df['PnL_%'].sum()
    avg_return = results_df['PnL_%'].mean()
    winning = len(results_df[results_df['PnL_%'] > 0])
    losing = len(results_df[results_df['PnL_%'] < 0])
    win_rate = (winning / len(results_df)) * 100 if len(results_df) > 0 else 0
    
    print("\n" + "="*100)
    print("STATYSTYKI BEZ LEWARU")
    print("="*100)
    print(f"Całkowita stopa zwrotu: {total_return:.2f}%")
    print(f"Średnia stopa zwrotu: {avg_return:.3f}%")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Transakcje zyskowne: {winning}")
    print(f"Transakcje stratne: {losing}")
    print(f"Najlepsza transakcja: {results_df['PnL_%'].max():.3f}%")
    print(f"Najgorsza transakcja: {results_df['PnL_%'].min():.3f}%")
    
    # Zwroty roczne
    print("\n" + "="*100)
    print("ZWROTY ROCZNE")
    print("="*100)
    yearly = results_df.groupby('Year').agg({
        'PnL_%': ['sum', 'mean', 'count']
    }).round(3)
    yearly.columns = ['Suma_%', 'Średnia_%', 'Liczba']
    print(yearly)
    
    # Zwroty miesięczne
    print("\n" + "="*100)
    print("ZWROTY MIESIĘCZNE (Top 5 najlepszych i najgorszych)")
    print("="*100)
    monthly = results_df.groupby('Year_Month').agg({
        'PnL_%': ['sum', 'count']
    }).round(3)
    monthly.columns = ['Suma_%', 'Liczba']
    monthly_sorted = monthly.sort_values('Suma_%', ascending=False)
    
    print("\n✅ Najlepsze:")
    print(monthly_sorted.head(5).to_string())
    print("\n❌ Najgorsze:")
    print(monthly_sorted.tail(5).to_string())
    
    # Szczegóły transakcji
    print("\n" + "="*100)
    print("WSZYSTKIE TRANSAKCJE")
    print("="*100)
    display_df = results_df[['Friday_Signal_Date', 'Monday_Entry_Date', 'Close_Date', 
                              'Signal', 'Friday_Close', 'Monday_Entry_Price', 
                              'Close_Price', 'S3', 'R3', 'PnL_%']].copy()
    display_df['Friday_Signal_Date'] = display_df['Friday_Signal_Date'].dt.strftime('%Y-%m-%d')
    display_df['Monday_Entry_Date'] = display_df['Monday_Entry_Date'].dt.strftime('%Y-%m-%d')
    display_df['Close_Date'] = display_df['Close_Date'].dt.strftime('%Y-%m-%d')
    print(display_df.round(4).to_string(index=False))
    
    # Zapisz
    results_df.to_csv('/mnt/user-data/outputs/new_strategy_results.csv', index=False)
    print("\n✅ Zapisano: new_strategy_results.csv")
    
    # Teraz przetestuj z lewarami
    print("\n" + "="*100)
    print("TESTOWANIE Z RÓŻNYMI LEWARAMI")
    print("="*100)
    
    leverages = [1, 5, 10, 20]
    INITIAL_CAPITAL = 10000
    
    leverage_results = []
    
    for lev in leverages:
        results_df[f'PnL_Lev_{lev}x_%'] = results_df['PnL_%'] * lev
        
        # Oblicz kapitał
        capital = [INITIAL_CAPITAL]
        for i in range(len(results_df)):
            new_capital = capital[-1] * (1 + results_df.iloc[i][f'PnL_Lev_{lev}x_%'] / 100)
            capital.append(new_capital)
        
        results_df[f'Capital_{lev}x'] = capital[1:]
        
        final_capital = capital[-1]
        roi = ((final_capital / INITIAL_CAPITAL) - 1) * 100
        total_ret_lev = results_df[f'PnL_Lev_{lev}x_%'].sum()
        
        winning_lev = len(results_df[results_df[f'PnL_Lev_{lev}x_%'] > 0])
        losing_lev = len(results_df[results_df[f'PnL_Lev_{lev}x_%'] < 0])
        win_rate_lev = (winning_lev / len(results_df)) * 100
        
        # Drawdown
        results_df[f'Cum_Return_{lev}x'] = results_df[f'PnL_Lev_{lev}x_%'].cumsum()
        max_dd = (results_df[f'Cum_Return_{lev}x'].cummax() - results_df[f'Cum_Return_{lev}x']).max()
        
        leverage_results.append({
            'Lewar': f'{lev}x',
            'Transakcje': len(results_df),
            'Zwrot_całk_%': round(total_ret_lev, 2),
            'ROI_%': round(roi, 2),
            'Kapitał_końcowy': round(final_capital, 2),
            'Zysk_Strata': round(final_capital - INITIAL_CAPITAL, 2),
            'Win_Rate_%': round(win_rate_lev, 1),
            'Max_Drawdown_%': round(max_dd, 2),
            'Najlepsza_%': round(results_df[f'PnL_Lev_{lev}x_%'].max(), 2),
            'Najgorsza_%': round(results_df[f'PnL_Lev_{lev}x_%'].min(), 2)
        })
    
    lev_comparison = pd.DataFrame(leverage_results)
    print("\n" + "="*100)
    print("PORÓWNANIE LEWARÓW")
    print("="*100)
    print(lev_comparison.to_string(index=False))
    
    # Zapisz z lewarami
    results_df.to_csv('/mnt/user-data/outputs/new_strategy_with_leverage.csv', index=False)
    print("\n✅ Zapisano: new_strategy_with_leverage.csv")
    
    # Wykresy
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Nowa Strategia: Sygnał z Piątku + Trzymanie 5 Dni', fontsize=16, fontweight='bold')
    
    # 1. Kapitał w czasie
    ax1 = axes[0, 0]
    for lev in leverages:
        ax1.plot(results_df['Monday_Entry_Date'], results_df[f'Capital_{lev}x'], 
                label=f'Lewar {lev}x', linewidth=2, marker='o', markersize=3)
    ax1.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_title('Wartość Portfela w Czasie', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Kapitał (PLN)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. ROI porównanie
    ax2 = axes[0, 1]
    rois = [r['ROI_%'] for r in leverage_results]
    colors = ['green' if r > 0 else 'red' for r in rois]
    bars = ax2.bar([r['Lewar'] for r in leverage_results], rois, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('ROI - Porównanie Lewarów', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ROI (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rois):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    # 3. Zwroty roczne
    ax3 = axes[1, 0]
    yearly_data = results_df.groupby('Year')['PnL_%'].sum()
    colors_yr = ['green' if y > 0 else 'red' for y in yearly_data]
    ax3.bar(yearly_data.index.astype(str), yearly_data.values, color=colors_yr, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Zwroty Roczne (Bez Lewaru)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Zwrot (%)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Kapitał końcowy
    ax4 = axes[1, 1]
    final_caps = [r['Kapitał_końcowy'] for r in leverage_results]
    colors_cap = ['green' if c > INITIAL_CAPITAL else 'red' for c in final_caps]
    bars_cap = ax4.bar([r['Lewar'] for r in leverage_results], final_caps, 
                       color=colors_cap, alpha=0.7, edgecolor='black')
    ax4.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', linewidth=2, label='Start')
    ax4.set_title('Kapitał Końcowy (Start: 10,000 PLN)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Kapitał (PLN)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val, roi in zip(bars_cap, final_caps, rois):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 200,
                f'{val:,.0f}\n({roi:+.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/new_strategy_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Wykres zapisany: new_strategy_analysis.png")

print("\n" + "="*100)
print("ANALIZA ZAKOŃCZONA")
print("="*100)
