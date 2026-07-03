#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punkty pivot – sygnały i backtest strategii
Dane: Yahoo Finance, interwał dzienny (opóźnienie ok. 15 min)
Uruchomienie: streamlit run pivot_backtest_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ============================================================================
# Konfiguracja strony i stałe
# ============================================================================
PHC_BLUE  = "#2e68a5"
COL_UP    = "#26a69a"
COL_DOWN  = "#ef5350"
COL_PIVOT = "#37474f"
COL_R     = "#c62828"
COL_S     = "#2e7d32"

st.set_page_config(
    page_title="Punkty pivot – sygnały i backtest",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.2rem;}
    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e3e9f2;
        border-radius: 10px;
        padding: 10px 14px;
    }
    .sygnal-kupno    {background:#e8f5e9;border-left:6px solid #2e7d32;border-radius:8px;padding:14px 18px;margin:8px 0;}
    .sygnal-sprzedaz {background:#ffebee;border-left:6px solid #c62828;border-radius:8px;padding:14px 18px;margin:8px 0;}
    .sygnal-slaby    {background:#fff8e1;border-left:6px solid #f9a825;border-radius:8px;padding:14px 18px;margin:8px 0;}
    .sygnal-brak     {background:#eceff1;border-left:6px solid #607d8b;border-radius:8px;padding:14px 18px;margin:8px 0;}
    .sygnal-kupno h4, .sygnal-sprzedaz h4, .sygnal-slaby h4, .sygnal-brak h4 {margin:0 0 4px 0;}
    .sygnal-kupno p, .sygnal-sprzedaz p, .sygnal-slaby p, .sygnal-brak p {margin:0;}
</style>
""", unsafe_allow_html=True)

FOREX_SYMBOLS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
    "USDJPY": "USDJPY=X", "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X",
    "EURGBP": "EURGBP=X",
    "EURPLN": "EURPLN=X", "USDPLN": "USDPLN=X",
    "GBPPLN": "GBPPLN=X", "CHFPLN": "CHFPLN=X",
}

PARY_GLOWNE  = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
PARY_KRZYZOWE = ["EURJPY", "GBPJPY", "EURGBP"]
PARY_PLN     = ["EURPLN", "USDPLN", "GBPPLN", "CHFPLN"]


def pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001


def price_dec(symbol: str) -> int:
    return 3 if "JPY" in symbol else 4


def fmt(v: float, symbol: str) -> str:
    return f"{v:.{price_dec(symbol)}f}"


# ============================================================================
# Pobieranie danych
# ============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(symbol: str, days: int):
    """Pobiera dzienne notowania z Yahoo Finance. Zwraca (df, symbol_yahoo) lub (None, symbol_yahoo)."""
    yf_symbol = FOREX_SYMBOLS.get(symbol, f"{symbol}=X")
    end = datetime.now()
    start = end - timedelta(days=days)

    data = pd.DataFrame()
    for proba in (1, 2):
        try:
            if proba == 1:
                data = yf.Ticker(yf_symbol).history(start=start, end=end, interval="1d")
            else:
                data = yf.download(yf_symbol, start=start, end=end, interval="1d", progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
        except Exception:
            data = pd.DataFrame()
        if data is not None and not data.empty:
            break

    if data is None or data.empty:
        return None, yf_symbol

    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    idx = pd.to_datetime(data.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)

    df = pd.DataFrame({
        "Date":  idx,
        "Open":  data["Open"].astype(float).values,
        "High":  data["High"].astype(float).values,
        "Low":   data["Low"].astype(float).values,
        "Close": data["Close"].astype(float).values,
        "Volume": data["Volume"].astype(float).values if "Volume" in data.columns else 0.0,
    }).reset_index(drop=True)

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    if len(df) < 5:
        return None, yf_symbol
    return df, yf_symbol


# ============================================================================
# Punkty pivot (rolowane, bez zaglądania w przyszłość)
# ============================================================================
def calculate_pivot_points(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Pivot dnia D liczony wyłącznie z dni D-lookback ... D-1 (shift(1))."""
    df = df.copy()
    ah = df["High"].rolling(lookback).mean().shift(1)
    al = df["Low"].rolling(lookback).mean().shift(1)
    ac = df["Close"].rolling(lookback).mean().shift(1)
    p = (ah + al + ac) / 3.0
    df["Pivot"] = p
    df["R1"] = 2.0 * p - al
    df["R2"] = p + (ah - al)
    df["S1"] = 2.0 * p - ah
    df["S2"] = p - (ah - al)
    return df


# ============================================================================
# Wykres cenowy
# ============================================================================
def create_price_chart(df, symbol, n_days, chart_type="Świece", show_volume=False,
                       show_pivots=True, trades=None, height=520, title=None):
    d = df.tail(n_days).reset_index(drop=True)
    if len(d) == 0:
        return None

    dec = price_dec(symbol)
    cur = d["Close"].iloc[-1]

    if show_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.04, row_heights=[0.76, 0.24])
        r_kw = dict(row=1, col=1)
        v_kw = dict(row=2, col=1)
    else:
        fig = go.Figure()
        r_kw, v_kw = {}, {}

    # --- cena ---
    if chart_type == "Świece":
        fig.add_trace(go.Candlestick(
            x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
            name=symbol,
            increasing_line_color=COL_UP, decreasing_line_color=COL_DOWN,
            increasing_fillcolor="rgba(38,166,154,0.85)",
            decreasing_fillcolor="rgba(239,83,80,0.85)",
            line=dict(width=1), whiskerwidth=0.6,
        ), **r_kw)
    else:
        fig.add_trace(go.Scatter(
            x=d["Date"], y=d["Close"], mode="lines", name=symbol,
            line=dict(color=PHC_BLUE, width=2.2),
            fill="tozeroy", fillcolor="rgba(46,104,165,0.06)",
            hovertemplate="Data: %{x|%Y-%m-%d}<br>Kurs: %{y:." + str(dec) + "f}<extra></extra>",
        ), **r_kw)
        fig.update_yaxes(range=[d["Low"].min() * 0.998, d["High"].max() * 1.002], **r_kw)

    # --- poziomy pivot (ostatnie wartości) ---
    if show_pivots and pd.notna(d["Pivot"].iloc[-1]):
        last = d.iloc[-1]
        t0, t1 = d["Date"].min(), d["Date"].max()
        label_x = t1 + (t1 - t0) * 0.04

        # strefy wsparcia i oporu
        if pd.notna(last.get("S1")) and pd.notna(last.get("S2")):
            fig.add_hrect(y0=last["S2"], y1=last["S1"], fillcolor="rgba(46,125,50,0.05)",
                          line_width=0, **r_kw)
        if pd.notna(last.get("R1")) and pd.notna(last.get("R2")):
            fig.add_hrect(y0=last["R1"], y1=last["R2"], fillcolor="rgba(198,40,40,0.05)",
                          line_width=0, **r_kw)

        poziomy = [
            ("R2", last.get("R2"), COL_R, "dash"),
            ("R1", last.get("R1"), COL_R, "dot"),
            ("Pivot", last.get("Pivot"), COL_PIVOT, "solid"),
            ("S1", last.get("S1"), COL_S, "dot"),
            ("S2", last.get("S2"), COL_S, "dash"),
        ]
        for nazwa, val, kolor, styl in poziomy:
            if pd.isna(val):
                continue
            fig.add_hline(y=val, line_color=kolor, line_width=1.2, line_dash=styl,
                          opacity=0.85, **r_kw)
            fig.add_annotation(x=label_x, y=val, text=f"<b>{nazwa}</b> {val:.{dec}f}",
                               showarrow=False, font=dict(size=10, color=kolor),
                               bgcolor="rgba(255,255,255,0.92)", bordercolor=kolor,
                               borderwidth=1, xanchor="left", yanchor="middle", **r_kw)

    # --- transakcje z backtestu ---
    if trades is not None and len(trades) > 0:
        t = trades[(trades["entry_date"] >= d["Date"].min()) & (trades["entry_date"] <= d["Date"].max())]
        dl = t[t["dir"] == 1]
        kr = t[t["dir"] == -1]
        if len(dl):
            fig.add_trace(go.Scatter(
                x=dl["entry_date"], y=dl["entry_price"], mode="markers", name="Wejście – kupno",
                marker=dict(symbol="triangle-up", size=11, color=COL_S,
                            line=dict(width=1, color="white")),
                hovertemplate="Kupno: %{y:." + str(dec) + "f}<br>%{x|%Y-%m-%d}<extra></extra>",
            ), **r_kw)
        if len(kr):
            fig.add_trace(go.Scatter(
                x=kr["entry_date"], y=kr["entry_price"], mode="markers", name="Wejście – sprzedaż",
                marker=dict(symbol="triangle-down", size=11, color=COL_R,
                            line=dict(width=1, color="white")),
                hovertemplate="Sprzedaż: %{y:." + str(dec) + "f}<br>%{x|%Y-%m-%d}<extra></extra>",
            ), **r_kw)

        kolory_wyjsc = np.where(t["pnl_pct"] > 0, COL_UP, COL_DOWN)
        fig.add_trace(go.Scatter(
            x=t["exit_date"], y=t["exit_price"], mode="markers", name="Wyjście",
            marker=dict(symbol="circle", size=8, color=kolory_wyjsc,
                        line=dict(width=1, color="white")),
            hovertemplate="Wyjście: %{y:." + str(dec) + "f}<br>%{x|%Y-%m-%d}<extra></extra>",
        ), **r_kw)

        # łączniki wejście-wyjście
        xs, ys = [], []
        for _, tr in t.iterrows():
            xs += [tr["entry_date"], tr["exit_date"], None]
            ys += [tr["entry_price"], tr["exit_price"], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Przebieg pozycji",
                                 line=dict(color="rgba(96,125,139,0.5)", width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False), **r_kw)

    # --- znacznik bieżącej ceny ---
    fig.add_trace(go.Scatter(
        x=[d["Date"].iloc[-1]], y=[cur], mode="markers+text",
        marker=dict(size=10, color="#ff9800", line=dict(width=2, color="white")),
        text=[f"{cur:.{dec}f}"], textposition="top center",
        textfont=dict(size=11, color="#e65100"), name="Kurs bieżący",
    ), **r_kw)

    # --- wolumen ---
    if show_volume and "Volume" in d.columns:
        kolory = [COL_UP if c >= o else COL_DOWN for c, o in zip(d["Close"], d["Open"])]
        fig.add_trace(go.Bar(x=d["Date"], y=d["Volume"], name="Wolumen",
                             marker_color=kolory, opacity=0.6), **v_kw)

    # --- układ ---
    fig.update_layout(
        title=title or f"{symbol} · kurs bieżący {cur:.{dec}f}",
        template="plotly_white",
        height=height + (140 if show_volume else 0),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(r=110, t=70, l=10, b=10),
        font=dict(family="Segoe UI, sans-serif", size=12),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])],
                     showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                     tickformat=f".{dec}f")
    return fig


# ============================================================================
# Silnik backtestu
# ============================================================================
def run_backtest(df: pd.DataFrame, symbol: str, cfg: dict):
    """
    Zwraca (trades_df, curve_df).
    Zasady konserwatywne:
      - pivot dnia D liczony wyłącznie z danych do dnia D-1,
      - jeśli w jednym dniu w zasięgu jest SL i TP, przyjmowany jest SL (gorszy wariant),
      - w dniu wejścia TP realizowany tylko na cenie zamknięcia (kolejność ruchów w świecy nieznana),
      - spread odejmowany od wyniku każdej transakcji,
      - jedna pozycja w danym momencie, bez dźwigni (pełne zaangażowanie kapitału).
    """
    pip = pip_size(symbol)
    lvl_l = cfg["entry_long"]    # 'S1' lub 'S2'
    lvl_s = cfg["entry_short"]   # 'R1' lub 'R2'
    opp_l = {"S1": "R1", "S2": "R2"}[lvl_l]
    opp_s = {"R1": "S1", "R2": "S2"}[lvl_s]

    d = df.reset_index(drop=True)
    trades = []
    pos = None
    pending = 0
    eq = float(cfg["capital"])
    curve_dates, curve_eq = [], []

    def zamknij(pos, exit_price, exit_date, exit_idx, powod):
        nonlocal eq
        brutto = (exit_price - pos["entry_price"]) * pos["dir"]
        netto = brutto - cfg["spread_pips"] * pip
        pnl_pct = netto / pos["entry_price"] * 100.0
        eq *= (1.0 + pnl_pct / 100.0)
        trades.append({
            "dir": pos["dir"],
            "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
            "exit_date": exit_date, "exit_price": exit_price,
            "pnl_pips": netto / pip, "pnl_pct": pnl_pct,
            "reason": powod, "days": exit_idx - pos["entry_idx"],
        })

    def sprawdz_wyjscie(pos, row, i):
        dni = i - pos["entry_idx"]
        if pos["dir"] == 1:
            if row["Low"] <= pos["sl"]:
                return pos["sl"], "SL"
            if dni > 0 and row["High"] >= pos["tp"]:
                return pos["tp"], "TP"
            if dni == 0 and row["Close"] >= pos["tp"]:
                return pos["tp"], "TP"
        else:
            if row["High"] >= pos["sl"]:
                return pos["sl"], "SL"
            if dni > 0 and row["Low"] <= pos["tp"]:
                return pos["tp"], "TP"
            if dni == 0 and row["Close"] <= pos["tp"]:
                return pos["tp"], "TP"
        if dni >= cfg["max_hold"]:
            return row["Close"], "Czas"
        return None, None

    def tp_dla_mr(kier, entry, row):
        if cfg["tp_mode"] == "pivot":
            return row["Pivot"]
        if cfg["tp_mode"] == "opposite":
            return row[opp_l] if kier == 1 else row[opp_s]
        return entry + kier * cfg["tp_pips"] * pip

    def otworz_mr(kier, row, i):
        lvl = row[lvl_l] if kier == 1 else row[lvl_s]
        entry = min(row["Open"], lvl) if kier == 1 else max(row["Open"], lvl)
        tp = tp_dla_mr(kier, entry, row)
        if pd.isna(tp) or (tp - entry) * kier < pip:
            return None
        sl = entry - kier * cfg["sl_pips"] * pip
        return {"dir": kier, "entry_price": entry, "entry_date": row["Date"],
                "entry_idx": i, "tp": tp, "sl": sl}

    for i in range(len(d)):
        row = d.loc[i]
        if pd.isna(row["Pivot"]):
            curve_dates.append(row["Date"]); curve_eq.append(eq)
            continue

        # 1) realizacja oczekującego wejścia z wybicia (po cenie otwarcia)
        if pos is None and pending != 0:
            entry = row["Open"]
            tp = entry + pending * cfg["tp_pips"] * pip
            sl = entry - pending * cfg["sl_pips"] * pip
            pos = {"dir": pending, "entry_price": entry, "entry_date": row["Date"],
                   "entry_idx": i, "tp": tp, "sl": sl}
            pending = 0

        # 2) obsługa otwartej pozycji
        if pos is not None:
            xp, powod = sprawdz_wyjscie(pos, row, i)
            if xp is not None:
                zamknij(pos, xp, row["Date"], i, powod)
                pos = None

        # 3) nowe sygnały
        if pos is None and pending == 0:
            if cfg["strategy"] == "mr":
                dotk_l = cfg["allow_long"] and pd.notna(row[lvl_l]) and row["Low"] <= row[lvl_l]
                dotk_s = cfg["allow_short"] and pd.notna(row[lvl_s]) and row["High"] >= row[lvl_s]
                if dotk_l and dotk_s:
                    # oba poziomy w zasięgu – wybór bliższego ceny otwarcia
                    if abs(row["Open"] - row[lvl_l]) > abs(row["Open"] - row[lvl_s]):
                        dotk_l = False
                    else:
                        dotk_s = False
                kier = 1 if dotk_l else (-1 if dotk_s else 0)
                if kier != 0:
                    pos = otworz_mr(kier, row, i)
                    if pos is not None:
                        xp, powod = sprawdz_wyjscie(pos, row, i)
                        if xp is not None:
                            zamknij(pos, xp, row["Date"], i, powod)
                            pos = None
            else:  # wybicie – sygnał na zamknięciu, wejście następnego dnia na otwarciu
                if i >= 1:
                    prev = d.loc[i - 1]
                    if (cfg["allow_long"] and pd.notna(row[lvl_s]) and pd.notna(prev[lvl_s])
                            and row["Close"] > row[lvl_s] and prev["Close"] <= prev[lvl_s]):
                        pending = 1
                    elif (cfg["allow_short"] and pd.notna(row[lvl_l]) and pd.notna(prev[lvl_l])
                            and row["Close"] < row[lvl_l] and prev["Close"] >= prev[lvl_l]):
                        pending = -1

        # 4) wycena bieżąca (otwarta pozycja po cenie zamknięcia)
        if pos is not None:
            mtm = eq * (1.0 + (row["Close"] - pos["entry_price"]) * pos["dir"] / pos["entry_price"])
        else:
            mtm = eq
        curve_dates.append(row["Date"]); curve_eq.append(mtm)

    # zamknięcie pozycji otwartej na końcu danych
    if pos is not None:
        ostatni = d.iloc[-1]
        zamknij(pos, ostatni["Close"], ostatni["Date"], len(d) - 1, "Koniec danych")
        curve_eq[-1] = eq

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame({"Date": curve_dates, "equity": curve_eq})
    return trades_df, curve_df


def compute_metrics(trades_df, curve_df, capital):
    m = {}
    m["liczba"] = len(trades_df)
    m["zysk_calk"] = curve_df["equity"].iloc[-1] / capital * 100.0 - 100.0
    if len(trades_df):
        wygrane = trades_df[trades_df["pnl_pct"] > 0]
        przegrane = trades_df[trades_df["pnl_pct"] <= 0]
        m["skutecznosc"] = len(wygrane) / len(trades_df) * 100.0
        suma_strat = abs(przegrane["pnl_pct"].sum())
        m["profit_factor"] = (wygrane["pnl_pct"].sum() / suma_strat) if suma_strat > 0 else float("inf")
        m["sr_pips"] = trades_df["pnl_pips"].mean()
        m["sr_dni"] = trades_df["days"].mean()
        m["sr_zysk"] = wygrane["pnl_pct"].mean() if len(wygrane) else 0.0
        m["sr_strata"] = przegrane["pnl_pct"].mean() if len(przegrane) else 0.0
    dd = (curve_df["equity"] / curve_df["equity"].cummax() - 1.0) * 100.0
    m["max_dd"] = dd.min()
    m["dd_series"] = dd
    return m


def equity_chart(curve_df, dd_series, bh_df=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.72, 0.28],
                        subplot_titles=["Krzywa kapitału", "Obsunięcie kapitału (%)"])
    fig.add_trace(go.Scatter(
        x=curve_df["Date"], y=curve_df["equity"], mode="lines", name="Strategia",
        line=dict(color=PHC_BLUE, width=2.4),
        hovertemplate="%{x|%Y-%m-%d}<br>Kapitał: %{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    if bh_df is not None:
        fig.add_trace(go.Scatter(
            x=bh_df["Date"], y=bh_df["equity"], mode="lines", name="Kup i trzymaj",
            line=dict(color="#90a4ae", width=1.6, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>Kapitał: %{y:,.0f}<extra></extra>",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=curve_df["Date"], y=dd_series, mode="lines", name="Obsunięcie",
        line=dict(color=COL_DOWN, width=1.4), fill="tozeroy",
        fillcolor="rgba(239,83,80,0.15)", showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}<br>Obsunięcie: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)
    fig.update_layout(
        template="plotly_white", height=520, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        margin=dict(r=20, t=50, l=10, b=10),
        font=dict(family="Segoe UI, sans-serif", size=12),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])],
                     showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    return fig


# ============================================================================
# Panel boczny
# ============================================================================
st.sidebar.header("⚙️ Ustawienia")

kategoria = st.sidebar.radio("Kategoria par:", ["🌍 Główne", "🔄 Krzyżowe", "🇵🇱 Pary PLN"], index=2)
if kategoria == "🌍 Główne":
    symbol = st.sidebar.selectbox("Para walutowa:", PARY_GLOWNE, index=0)
elif kategoria == "🔄 Krzyżowe":
    symbol = st.sidebar.selectbox("Para walutowa:", PARY_KRZYZOWE, index=0)
else:
    symbol = st.sidebar.selectbox("Para walutowa:", PARY_PLN, index=0)

st.sidebar.markdown("### Wykres")
typ_wykresu = st.sidebar.radio("Rodzaj wykresu:", ["Świece", "Liniowy"], index=0, horizontal=True)
dni_wykresu = st.sidebar.slider("Zakres wykresu (dni sesyjne)", 10, 120, 40)
lookback_wykres = st.sidebar.slider("Okno punktów pivot (dni)", 3, 20, 7)
pokaz_wolumen = st.sidebar.checkbox("Pokaż wolumen", False)
pokaz_pivoty = st.sidebar.checkbox("Pokaż poziomy pivot", True)
auto_odswiez = st.sidebar.checkbox("Automatyczne odświeżanie (60 s)", False)

if st.sidebar.button("🔄 Odśwież dane", use_container_width=True):
    fetch_data.clear()
    st.rerun()

# ============================================================================
# Zakładki
# ============================================================================
st.title("📈 Punkty pivot – sygnały i backtest")
tab_wykres, tab_backtest = st.tabs(["📊 Wykres i sygnały", "🧪 Backtest strategii"])

# ----------------------------------------------------------------------------
# Zakładka 1: wykres i bieżąca analiza
# ----------------------------------------------------------------------------
with tab_wykres:
    with st.spinner(f"Pobieranie danych {symbol}..."):
        dni_kalendarzowe = int((dni_wykresu + lookback_wykres + 10) * 1.55)
        df, yf_sym = fetch_data(symbol, dni_kalendarzowe)

    if df is None:
        st.error(f"Nie udało się pobrać danych dla {symbol}. Spróbuj ponownie lub wybierz inną parę.")
        if "PLN" in symbol:
            st.info("Pary z PLN bywają rzadziej aktualizowane w Yahoo Finance – warto odświeżyć po chwili.")
    else:
        dfp = calculate_pivot_points(df, lookback_wykres)
        fig = create_price_chart(dfp, symbol, dni_wykresu, typ_wykresu,
                                 pokaz_wolumen, pokaz_pivoty)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        last = dfp.iloc[-1]
        cur = last["Close"]
        dec = price_dec(symbol)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kurs bieżący", fmt(cur, symbol))
        if pd.notna(last.get("Pivot")):
            roznica = cur - last["Pivot"]
            c2.metric("Odchylenie od pivota", f"{roznica:+.{dec}f}",
                      f"{roznica / last['Pivot'] * 100:+.2f}%")
        else:
            c2.metric("Odchylenie od pivota", "—")
        c3.metric("Liczba sesji", len(dfp))
        c4.metric("Ostatnia sesja", last["Date"].strftime("%Y-%m-%d"))

        # --- sygnał ---
        if pd.notna(last.get("S2")) and pd.notna(last.get("R2")):
            if cur < last["S2"]:
                st.markdown(f'<div class="sygnal-kupno"><h4>🟢 Sygnał kupna (silny)</h4>'
                            f'<p>Kurs {fmt(cur, symbol)} poniżej poziomu S2 {fmt(last["S2"], symbol)} – '
                            f'strefa wyprzedania względem rolowanych pivotów.</p></div>',
                            unsafe_allow_html=True)
            elif cur > last["R2"]:
                st.markdown(f'<div class="sygnal-sprzedaz"><h4>🔴 Sygnał sprzedaży (silny)</h4>'
                            f'<p>Kurs {fmt(cur, symbol)} powyżej poziomu R2 {fmt(last["R2"], symbol)} – '
                            f'strefa wykupienia względem rolowanych pivotów.</p></div>',
                            unsafe_allow_html=True)
            elif pd.notna(last.get("S1")) and cur < last["S1"]:
                st.markdown(f'<div class="sygnal-slaby"><h4>🟡 Słaby sygnał kupna</h4>'
                            f'<p>Kurs {fmt(cur, symbol)} poniżej S1 {fmt(last["S1"], symbol)}.</p></div>',
                            unsafe_allow_html=True)
            elif pd.notna(last.get("R1")) and cur > last["R1"]:
                st.markdown(f'<div class="sygnal-slaby"><h4>🟡 Słaby sygnał sprzedaży</h4>'
                            f'<p>Kurs {fmt(cur, symbol)} powyżej R1 {fmt(last["R1"], symbol)}.</p></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="sygnal-brak"><h4>⚪ Brak sygnału</h4>'
                            f'<p>Kurs {fmt(cur, symbol)} pomiędzy poziomami S1 i R1.</p></div>',
                            unsafe_allow_html=True)

            # --- tabela poziomów ---
            st.markdown("#### Bieżące poziomy pivot")
            wiersze = []
            for nazwa in ["R2", "R1", "Pivot", "S1", "S2"]:
                val = last.get(nazwa)
                if pd.notna(val):
                    wiersze.append({
                        "Poziom": nazwa,
                        "Wartość": fmt(val, symbol),
                        "Odległość": f"{(val - cur) / cur * 100:+.2f}%",
                        "Pipsy": f"{(val - cur) / pip_size(symbol):+.0f}",
                    })
            st.dataframe(pd.DataFrame(wiersze), use_container_width=True, hide_index=True)

        with st.expander("🔍 Weryfikacja danych"):
            st.markdown(
                f"**Symbol:** {symbol} ({yf_sym})  \n"
                f"**Liczba sesji:** {len(dfp)}  \n"
                f"**Ostatni kurs:** {fmt(cur, symbol)}  \n"
                f"**Czas pobrania:** {datetime.now().strftime('%H:%M:%S')}  \n"
                f"**Weryfikacja:** https://finance.yahoo.com/quote/{yf_sym}"
            )
            ost = dfp.tail(5)[["Date", "Open", "High", "Low", "Close"]].copy()
            ost["Date"] = ost["Date"].dt.strftime("%Y-%m-%d")
            for c in ["Open", "High", "Low", "Close"]:
                ost[c] = ost[c].round(price_dec(symbol))
            st.dataframe(ost, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# Zakładka 2: backtest
# ----------------------------------------------------------------------------
with tab_backtest:
    st.markdown(f"### Backtest strategii pivot – {symbol}")

    strategia_opis = st.radio(
        "Typ strategii:",
        ["Powrót do średniej (odbicie od S/R)", "Wybicie (podążanie za ruchem)"],
        horizontal=True,
        help="Powrót do średniej: kupno przy dotknięciu wsparcia, sprzedaż przy dotknięciu oporu. "
             "Wybicie: wejście zgodne z kierunkiem przełamania poziomu (na otwarciu kolejnej sesji).",
    )
    strategia = "mr" if strategia_opis.startswith("Powrót") else "bo"

    with st.form("parametry_backtestu"):
        k1, k2, k3 = st.columns(3)
        with k1:
            okres_opis = st.selectbox("Okres testu:", ["3 miesiące", "6 miesięcy", "12 miesięcy", "24 miesiące"], index=2)
            lookback_bt = st.slider("Okno punktów pivot (dni)", 3, 20, lookback_wykres)
            kierunek = st.selectbox("Kierunek transakcji:", ["Oba kierunki", "Tylko kupno", "Tylko sprzedaż"])
        with k2:
            poziom_opis = st.selectbox("Poziom wejścia:", ["S2 / R2 (skrajne)", "S1 / R1 (bliższe)"])
            if strategia == "mr":
                tp_opis = st.selectbox("Poziom docelowy (TP):", ["Pivot", "Poziom przeciwny (R/S)", "Stała liczba pipsów"])
            else:
                tp_opis = "Stała liczba pipsów"
                st.selectbox("Poziom docelowy (TP):", ["Stała liczba pipsów"], disabled=True)
            tp_pips = st.number_input("TP (pipsy) – dla trybu stałego", 10, 2000, 150, step=10)
        with k3:
            sl_pips = st.number_input("SL (pipsy)", 10, 2000, 200, step=10)
            max_hold = st.number_input("Maks. czas w pozycji (sesje)", 1, 60, 10)
            spread_pips = st.number_input("Spread + koszty (pipsy na transakcję)", 0.0, 100.0, 2.0, step=0.5)

        kapital = st.number_input("Kapitał początkowy", 1000, 100_000_000, 100_000, step=1000)
        uruchom = st.form_submit_button("▶️ Uruchom backtest", type="primary", use_container_width=True)

    if uruchom:
        miesiace = {"3 miesiące": 3, "6 miesięcy": 6, "12 miesięcy": 12, "24 miesiące": 24}[okres_opis]
        dni_potrzebne = int(miesiace * 31 + (lookback_bt + 10) * 1.6)

        with st.spinner("Pobieranie danych i obliczanie wyników..."):
            df_bt, _ = fetch_data(symbol, dni_potrzebne)
            if df_bt is None:
                st.error("Nie udało się pobrać danych do backtestu.")
            else:
                dfp_bt = calculate_pivot_points(df_bt, lookback_bt)
                cfg = {
                    "strategy": strategia,
                    "allow_long": kierunek in ("Oba kierunki", "Tylko kupno"),
                    "allow_short": kierunek in ("Oba kierunki", "Tylko sprzedaż"),
                    "entry_long": "S2" if "S2" in poziom_opis else "S1",
                    "entry_short": "R2" if "R2" in poziom_opis else "R1",
                    "tp_mode": {"Pivot": "pivot", "Poziom przeciwny (R/S)": "opposite",
                                "Stała liczba pipsów": "pips"}[tp_opis],
                    "tp_pips": float(tp_pips),
                    "sl_pips": float(sl_pips),
                    "max_hold": int(max_hold),
                    "spread_pips": float(spread_pips),
                    "capital": float(kapital),
                }
                trades_df, curve_df = run_backtest(dfp_bt, symbol, cfg)
                st.session_state["bt"] = {
                    "symbol": symbol, "trades": trades_df, "curve": curve_df,
                    "df": dfp_bt, "cfg": cfg, "okres": okres_opis,
                }

    # --- prezentacja wyników ---
    bt = st.session_state.get("bt")
    if bt is not None:
        if bt["symbol"] != symbol:
            st.info(f"Wyświetlane wyniki dotyczą pary **{bt['symbol']}** – uruchom backtest ponownie dla {symbol}.")

        trades_df, curve_df = bt["trades"], bt["curve"]
        cfg, dfp_bt = bt["cfg"], bt["df"]

        if len(trades_df) == 0:
            st.warning("Strategia nie wygenerowała żadnej transakcji w wybranym okresie. "
                       "Zmień poziom wejścia (np. S1/R1) albo wydłuż okres testu.")
        else:
            met = compute_metrics(trades_df, curve_df, cfg["capital"])

            st.markdown("#### Wyniki")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Wynik całkowity", f"{met['zysk_calk']:+.2f}%")
            m2.metric("Liczba transakcji", met["liczba"])
            m3.metric("Skuteczność", f"{met['skutecznosc']:.0f}%")
            pf = met["profit_factor"]
            m4.metric("Współczynnik zysku", "∞" if np.isinf(pf) else f"{pf:.2f}")
            m5.metric("Maks. obsunięcie", f"{met['max_dd']:.2f}%")
            m6.metric("Śr. wynik / transakcję", f"{met['sr_pips']:+.0f} pipsów")

            m7, m8, m9 = st.columns(3)
            m7.metric("Śr. zysk (wygrane)", f"{met['sr_zysk']:+.2f}%")
            m8.metric("Śr. strata (przegrane)", f"{met['sr_strata']:+.2f}%")
            m9.metric("Śr. czas w pozycji", f"{met['sr_dni']:.1f} sesji")

            # krzywa kapitału + porównanie z pasywnym utrzymaniem pozycji
            valid = dfp_bt["Pivot"].notna()
            base = dfp_bt.loc[valid].iloc[0]["Close"] if valid.any() else dfp_bt.iloc[0]["Close"]
            bh_df = pd.DataFrame({
                "Date": dfp_bt.loc[valid, "Date"],
                "equity": cfg["capital"] * dfp_bt.loc[valid, "Close"] / base,
            })
            st.plotly_chart(equity_chart(curve_df, met["dd_series"], bh_df),
                            use_container_width=True)

            # transakcje na wykresie ceny
            st.markdown("#### Transakcje na wykresie")
            n_sesji = int(dfp_bt["Pivot"].notna().sum())
            typ_bt = "Świece" if n_sesji <= 190 else "Liniowy"
            fig_t = create_price_chart(dfp_bt, bt["symbol"], n_sesji, typ_bt,
                                       False, False, trades=trades_df, height=480,
                                       title=f"{bt['symbol']} · okres testu: {bt['okres']}")
            if fig_t:
                st.plotly_chart(fig_t, use_container_width=True)

            # tabela transakcji
            st.markdown("#### Rejestr transakcji")
            dec = price_dec(bt["symbol"])
            tab = trades_df.copy()
            tab.insert(0, "Nr", range(1, len(tab) + 1))
            tab["Kierunek"] = np.where(tab["dir"] == 1, "Kupno", "Sprzedaż")
            tab["Data wejścia"] = tab["entry_date"].dt.strftime("%Y-%m-%d")
            tab["Kurs wejścia"] = tab["entry_price"].round(dec)
            tab["Data wyjścia"] = tab["exit_date"].dt.strftime("%Y-%m-%d")
            tab["Kurs wyjścia"] = tab["exit_price"].round(dec)
            tab["Wynik (pipsy)"] = tab["pnl_pips"].round(0)
            tab["Wynik (%)"] = tab["pnl_pct"].round(2)
            tab["Powód wyjścia"] = tab["reason"]
            tab["Sesje"] = tab["days"]
            widok = tab[["Nr", "Kierunek", "Data wejścia", "Kurs wejścia", "Data wyjścia",
                         "Kurs wyjścia", "Wynik (pipsy)", "Wynik (%)", "Powód wyjścia", "Sesje"]]
            st.dataframe(widok, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Pobierz rejestr transakcji (CSV)",
                widok.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"backtest_{bt['symbol']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

            st.caption(
                "Założenia: pivot dnia D liczony wyłącznie z danych do dnia D-1; przy jednoczesnym zasięgu "
                "SL i TP w jednej sesji przyjmowany jest SL (wariant konserwatywny); w dniu wejścia TP "
                "realizowany tylko na zamknięciu; spread odejmowany od każdej transakcji; jedna pozycja "
                "naraz, pełne zaangażowanie kapitału, bez dźwigni."
            )

# ============================================================================
# Stopka i automatyczne odświeżanie
# ============================================================================
st.markdown("---")
st.markdown(
    f"**🕐 Czas:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
    f"**⚠️ Materiał wyłącznie testowy** – wyniki historyczne nie gwarantują wyników przyszłych · "
    f"**📊 Źródło danych:** Yahoo Finance (opóźnienie ok. 15 min)"
)

if auto_odswiez:
    time.sleep(60)
    fetch_data.clear()
    st.rerun()
