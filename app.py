# =============================================================================
#  HMM Market Regime Detector — Interactive Streamlit Dashboard
# =============================================================================
"""
Usage:
    streamlit run app.py
"""

import sys
import warnings
import datetime
import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


from sr_engine import (
    full_sr_analysis, walk_forward, build_sr_chart,
    detect_pivots, cluster_levels, analyse_zone_interactions
)
from options_engine import rank_and_select_strategy
from tesouro_direto import (
    buscar_tesouro_direto, calcular_duration_portfolio, recomendar_estrategia, ESTRATEGIAS
)
from rf_engine import (

    prepare_rf_data, train_rf_model, get_rf_prediction,
    generate_trader_advice, build_importance_chart,
    optimize_rf
)
from scraper_opcoes import buscar_dados_opcoes
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    st.error("yfinance not installed. Run: `pip install yfinance`")
    st.stop()

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    st.error("hmmlearn not installed. Run: `pip install hmmlearn`")
    st.stop()

from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
INTERVAL_MAX_DAYS = {
    "1m": 7, "5m": 60, "15m": 60, "30m": 60,
    "1h": 730, "1d": 10000, "1wk": 10000, "1mo": 10000,
}
INTERVAL_BARS_PER_DAY = {
    "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
    "1h": 24, "1d": 1, "1wk": 1/7, "1mo": 1/30,
}

REGIME_TAGS = {
    3: [("BEAR", "#ef5350"), ("NEUTRAL", "#ffb74d"), ("BULL", "#00e676")],
    4: [("BEAR", "#ef5350"), ("WEAK BEAR", "#ffb74d"),
        ("WEAK BULL", "#29b6f6"), ("BULL", "#00e676")],
    5: [("BEAR", "#d50000"), ("WEAK BEAR", "#ef5350"), ("NEUTRAL", "#fff176"),
        ("WEAK BULL", "#29b6f6"), ("BULL", "#00e676")],
}

REGIME_ACTIONS = {
    "BEAR":      "Strong downtrend — reduce exposure, hedge, protect capital.",
    "WEAK BEAR": "Drift bearish — tighten stops, be cautious with new longs.",
    "NEUTRAL":   "No clear direction — reduce size, wait for breakout.",
    "WEAK BULL": "Drift bullish — hold positions, add selectively on dips.",
    "BULL":      "Strong uptrend — ride momentum, accumulate on pullbacks.",
}

FEATURES = [
    "Smoothed_Return", "Smoothed_Range", "Smoothed_VolChg",
    "EMA_Cross", "BB_Pos", "RSI_norm", "MACD_Hist", "Fib_Level",
    "Cacas_Pos",
]



# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: adapt bars to interval
# ═══════════════════════════════════════════════════════════════════════════
def _bars(hours, interval):
    bpd = INTERVAL_BARS_PER_DAY.get(interval, 24)
    return max(1, int(hours * bpd / 24))


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner="Downloading data...")
def download_data(ticker, interval, total_days, chunk_days=59):
    max_days = INTERVAL_MAX_DAYS.get(interval, 730)
    actual_days = min(total_days, max_days)
    end   = datetime.datetime.now()
    start = end - datetime.timedelta(days=actual_days)

    use_chunks = interval in ("1m", "5m", "15m", "30m", "1h")
    cd = min(chunk_days, max_days) if use_chunks else actual_days

    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + datetime.timedelta(days=cd), end)
        part = yf.download(ticker, start=cursor, end=chunk_end,
                           interval=interval, auto_adjust=True, progress=False)
        if part is not None and not part.empty:
            chunks.append(part)
        cursor = chunk_end

    if not chunks:
        return pd.DataFrame()

    raw = pd.concat(chunks)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    raw.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    return raw


# ═══════════════════════════════════════════════════════════════════════════
#  2. TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
def compute_indicators(df, ema_fast=9, ema_slow=21, bb_period=20, bb_std=2,
                       rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9,
                       fib_lookback=100,
                       cacas_upper=20, cacas_under=30, cacas_ema=9):
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    ef = close.ewm(span=ema_fast, adjust=False).mean()
    es = close.ewm(span=ema_slow, adjust=False).mean()
    df["EMA_Cross"] = (ef - es) / close

    bb_mid = close.rolling(bb_period).mean()
    bb_s   = close.rolling(bb_period).std()
    bb_u   = bb_mid + bb_std * bb_s
    bb_l   = bb_mid - bb_std * bb_s
    bw     = bb_u - bb_l
    df["BB_Pos"] = np.where(bw > 0, (close - bb_l) / bw, 0.5)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(rsi_period).mean()
    loss  = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI_norm"] = (100 - 100 / (1 + rs)) / 100

    ml = close.ewm(span=macd_fast, adjust=False).mean() - close.ewm(span=macd_slow, adjust=False).mean()
    ms = ml.ewm(span=macd_signal, adjust=False).mean()
    df["MACD_Hist"] = (ml - ms) / close

    rh = high.rolling(fib_lookback, min_periods=10).max()
    rl = low.rolling(fib_lookback, min_periods=10).min()
    fr = rh - rl
    df["Fib_Level"] = np.where(fr > 0, (close - rl) / fr, 0.5)

    # --- Cacas Channel (Pine Script Port) ---
    c_high = high.rolling(cacas_upper).max()
    c_low  = low.rolling(cacas_under).min()
    c_mid  = (c_high + c_low) / 2
    c_ema  = c_mid.ewm(span=cacas_ema, adjust=False).mean()
    
    # Feature for RF: position relative to mid-EMA
    # (close - c_ema) / (price range)
    df["Cacas_Pos"] = (close - c_ema) / (c_high - c_low).replace(0, np.nan)
    df["Cacas_Mid"] = c_mid # For plotting
    df["Cacas_EMA"] = c_ema # For plotting
    df["Cacas_Upper"] = c_high
    df["Cacas_Lower"] = c_low

    return df



# ═══════════════════════════════════════════════════════════════════════════
#  3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
def engineer_features(raw, interval, smoothing_hrs, **indicator_kwargs):
    df = raw.copy()
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Range"]  = (df["High"] - df["Low"]) / df["Close"]
    df["VolChg"] = df["Volume"].pct_change()

    w = _bars(smoothing_hrs, interval)
    df["Smoothed_Return"] = df["Return"].rolling(w, min_periods=1).mean()
    df["Smoothed_Range"]  = df["Range"].rolling(w, min_periods=1).mean()
    df["Smoothed_VolChg"] = df["VolChg"].rolling(w, min_periods=1).mean()

    df = compute_indicators(df, **indicator_kwargs)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  4. HMM FIT
# ═══════════════════════════════════════════════════════════════════════════
def fit_hmm(df, n_states, interval, min_regime_hrs):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].values)

    model = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=1000, random_state=42, verbose=False)
    model.fit(X)

    # Fix degenerate states: if any transmat row sums to 0, normalize it
    transmat = model.transmat_.copy()
    row_sums = transmat.sum(axis=1, keepdims=True)
    bad_rows = (row_sums == 0).flatten()
    if bad_rows.any():
        # Give uniform transition to degenerate states
        transmat[bad_rows] = 1.0 / n_states
        row_sums[bad_rows] = 1.0
    model.transmat_ = transmat / row_sums

    try:
        raw_states = model.predict(X)
    except ValueError:
        # Fallback: use K-means clustering if HMM fails
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_states, random_state=42, n_init=10)
        raw_states = km.fit_predict(X)

    # Reorder by mean return
    means = {}
    for s in range(n_states):
        mask = raw_states == s
        means[s] = df.loc[mask, "Return"].mean() if mask.any() else 0.0
    remap = {old: new for new, old in enumerate(sorted(means, key=means.get))}
    df["State"] = np.array([remap[s] for s in raw_states])

    # Smooth short flips
    min_b = _bars(min_regime_hrs, interval)
    states = df["State"].values.copy()
    i = 0
    while i < len(states):
        j = i
        while j < len(states) and states[j] == states[i]:
            j += 1
        if (j - i) < min_b and i > 0:
            states[i:j] = states[i - 1]
        else:
            i = j
            continue
        i = j
    df["State"] = states

    converged = model.monitor_.converged
    ll = model.score(X)
    return df, converged, ll


# ═══════════════════════════════════════════════════════════════════════════
#  5. BUILD SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
def build_summary(df, n_states):
    HOURS_PER_YEAR = 365.25 * 24
    summary = (
        df.groupby("State")["Return"]
        .agg(Count="count", Mean_Return="mean", Std_Return="std")
        .sort_index()
    )
    summary["Ann_Return"] = summary["Mean_Return"] * HOURS_PER_YEAR
    summary["Ann_Vol"]    = summary["Std_Return"]  * np.sqrt(HOURS_PER_YEAR)
    summary["Pct_Time"]   = (summary["Count"] / summary["Count"].sum() * 100).round(1)

    tags = REGIME_TAGS.get(n_states, REGIME_TAGS[4])
    for i, (tag, colour) in enumerate(tags):
        if i in summary.index:
            summary.loc[i, "Tag"]    = tag
            summary.loc[i, "Colour"] = colour
            summary.loc[i, "Action"] = REGIME_ACTIONS.get(tag, "")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  6. CONFIRMED SIGNALS
# ═══════════════════════════════════════════════════════════════════════════
def find_transitions(df, n_states):
    bull_thresh = n_states // 2
    is_bull = df["State"] >= bull_thresh

    rsi   = df["RSI_norm"].values
    macd  = df["MACD_Hist"].values
    ema_x = df["EMA_Cross"].values

    buys, sells = [], []
    prev = is_bull.iloc[0]
    for i in range(1, len(df)):
        curr = is_bull.iloc[i]
        if curr and not prev:
            bv = int(rsi[i] > 0.40) + int(macd[i] > 0) + int(ema_x[i] > 0)
            if bv >= 2:
                buys.append((df.index[i], df["Close"].iloc[i]))
        elif not curr and prev:
            sv = int(rsi[i] < 0.60) + int(macd[i] < 0) + int(ema_x[i] < 0)
            if sv >= 2:
                sells.append((df.index[i], df["Close"].iloc[i]))
        prev = curr
    return buys, sells


# ═══════════════════════════════════════════════════════════════════════════
#  7. BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════
def backtest(buys, sells):
    signals = [(t, p, "BUY") for t, p in buys] + [(t, p, "SELL") for t, p in sells]
    signals.sort(key=lambda x: x[0])

    trades = []
    entry_price = None
    entry_time  = None
    for t, price, sig in signals:
        if sig == "BUY" and entry_price is None:
            entry_price, entry_time = price, t
        elif sig == "SELL" and entry_price is not None:
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({"entry": entry_time, "exit": t,
                           "entry_price": entry_price, "exit_price": price,
                           "pnl_pct": pnl_pct})
            entry_price = None

    if not trades:
        return {"trades": 0, "sharpe": -999}

    tdf = pd.DataFrame(trades)
    wins   = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]

    total    = len(tdf)
    win_rate = len(wins) / total
    avg_win  = wins["pnl_pct"].mean() if len(wins) else 0
    avg_loss = abs(losses["pnl_pct"].mean()) if len(losses) else 0.001
    rr       = avg_win / avg_loss if avg_loss > 0 else 0
    expect   = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    kelly    = max(0, win_rate - (1 - win_rate) / rr) * 100 if rr > 0 else 0

    rets     = tdf["pnl_pct"].values / 100
    sharpe   = rets.mean() / rets.std() * np.sqrt(len(rets)) if rets.std() > 0 else 0
    cum      = (1 + rets).cumprod()
    run_max  = np.maximum.accumulate(cum)
    max_dd   = ((cum - run_max) / run_max * 100).min()
    total_ret = (cum[-1] - 1) * 100

    return {
        "trades": total, "win_rate": win_rate * 100, "avg_win": avg_win,
        "avg_loss": avg_loss, "rr_ratio": rr, "expectation": expect,
        "kelly": kelly, "sharpe": sharpe, "max_dd": max_dd,
        "total_return": total_ret, "trade_log": tdf,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  8. PLOTLY CHART
# ═══════════════════════════════════════════════════════════════════════════
def build_plotly_chart(df, summary, buys, sells, ticker, interval):
    colour_map = {s: summary.loc[s, "Colour"] for s in summary.index}
    tag_map    = {s: summary.loc[s, "Tag"]    for s in summary.index}

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.20, 0.15],
        vertical_spacing=0.03,
        subplot_titles=[
            f"{ticker} ({interval}) — HMM Regime Detection — BUY / SELL",
            "RSI (14)",
            "Regime Timeline",
        ],
    )

    # ── Panel 1: Price + regime bands + signals ─────────────────────────────
    # Regime bands
    prev_s = df["State"].iloc[0]
    bstart = df.index[0]
    shapes = []
    for i in range(1, len(df)):
        c = df["State"].iloc[i]
        if c != prev_s or i == len(df) - 1:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=bstart, x1=df.index[i], y0=0, y1=1,
                fillcolor=colour_map.get(prev_s, "#444"),
                opacity=0.12, layer="below", line_width=0,
            ))
            bstart = df.index[i]
            prev_s = c

    # Price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines",
        line=dict(color="white", width=1.2),
        name="Price", showlegend=False,
    ), row=1, col=1)

    # BUY markers
    if buys:
        bt, bp = zip(*buys)
        fig.add_trace(go.Scatter(
            x=list(bt), y=list(bp), mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#00e676",
                        line=dict(width=1, color="white")),
            name=f"BUY ({len(buys)})",
        ), row=1, col=1)

    # SELL markers
    if sells:
        st, sp = zip(*sells)
        fig.add_trace(go.Scatter(
            x=list(st), y=list(sp), mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#ef5350",
                        line=dict(width=1, color="white")),
            name=f"SELL ({len(sells)})",
        ), row=1, col=1)

    # Legend entries for regimes
    for state in sorted(summary.index):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=colour_map[state], symbol="square"),
            name=tag_map[state], showlegend=True,
        ), row=1, col=1)

    # ── Panel 2: RSI ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI_norm"] * 100, mode="lines",
        line=dict(color="#ab47bc", width=1),
        name="RSI", showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350",
                  line_width=0.8, opacity=0.6, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00e676",
                  line_width=0.8, opacity=0.6, row=2, col=1)

    fig.add_hrect(y0=70, y1=100, fillcolor="#ef5350", opacity=0.07,
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="#00e676", opacity=0.07,
                  line_width=0, row=2, col=1)

    # ── Panel 3: Regime timeline ─────────────────────────────────────────
    prev_s = df["State"].iloc[0]
    bstart = df.index[0]
    for i in range(1, len(df)):
        c = df["State"].iloc[i]
        if c != prev_s or i == len(df) - 1:
            fig.add_vrect(
                x0=bstart, x1=df.index[i],
                fillcolor=colour_map.get(prev_s, "#444"),
                opacity=0.85, line_width=0, row=3, col=1,
            )
            bstart = df.index[i]
            prev_s = c

    # Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121212",
        plot_bgcolor="#1a1a1a",
        height=800,
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=11),
            bgcolor="rgba(30,30,30,0.8)",
        ),
        shapes=shapes,
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1, gridcolor="#333")
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100], gridcolor="#333")
    fig.update_yaxes(showticklabels=False, row=3, col=1)
    fig.update_xaxes(gridcolor="#333")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  9. OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════
def run_optimization(raw, interval):
    grid = {
        "n_states":       [3, 4, 5],
        "smoothing_hrs":  [6, 12, 24],
        "min_regime_hrs": [4, 8, 12],
        "rsi_period":     [10, 14, 21],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    results = []

    progress = st.progress(0, text="Optimising parameters...")
    for idx, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        try:
            df = engineer_features(raw, interval, p["smoothing_hrs"],
                                   rsi_period=p["rsi_period"])
            if len(df) < p["n_states"]:
                continue
            df, _, _ = fit_hmm(df, p["n_states"], interval, p["min_regime_hrs"])
            buys, sells = find_transitions(df, p["n_states"])
            m = backtest(buys, sells)
            results.append({**p, **{k: v for k, v in m.items() if k != "trade_log"}})
        except Exception:
            continue
        progress.progress((idx + 1) / len(combos),
                          text=f"Testing {idx+1}/{len(combos)} combos...")

    progress.empty()
    if not results:
        return None, None
    rdf = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    best = rdf.iloc[0]
    return rdf, best


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="HMM Regime Detector", layout="wide",
                   page_icon="📈", initial_sidebar_state="expanded")

# ── Session State Initialization ──────────────────────────────────────────
# Define defaults and sync with session state keys
defaults = {
    "ticker": "BTC-USD",
    "interval": "1h",
    "days": 365,
    "n_states": 4,
    "smoothing_hrs": 24,
    "min_regime_hrs": 8,
    "rsi_period": 14,
    "sr_pivot_window": 10,
    "sr_cluster_pct": 1.5,
    "sr_train_pct": 70,
    "rf_trees": 100,
    "rf_depth": 10,
    "rf_horizon": 1,
    "hmm_opt_results": None,
    "sr_wf_results": None,
    "rf_opt_results": None
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Callbacks ────────────────────────────────────────────────────────────
def apply_hmm_params():
    if st.session_state.hmm_opt_results:
        best = st.session_state.hmm_opt_results["best"]
        st.session_state.n_states = int(best['n_states'])
        st.session_state.smoothing_hrs = int(best['smoothing_hrs'])
        st.session_state.min_regime_hrs = int(best['min_regime_hrs'])
        st.session_state.rsi_period = int(best['rsi_period'])

def apply_sr_params():
    if st.session_state.sr_wf_results:
        best = st.session_state.sr_wf_results["best"]
        st.session_state.sr_pivot_window = int(best['pivot_window'])
        st.session_state.sr_cluster_pct = float(best['cluster_pct'])

def apply_rf_params():
    if "rf_opt_results" in st.session_state and st.session_state["rf_opt_results"] is not None:
        if not st.session_state["rf_opt_results"].empty:
            best = st.session_state["rf_opt_results"].iloc[0]
            st.session_state.rf_trees   = int(best['n_estimators'])
            st.session_state.rf_depth   = int(best['max_depth'])
            st.session_state.rf_horizon = int(best['horizon'])



# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30475e;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card h3 { color: #a0a0a0; font-size: 0.85em; margin: 0; }
    .metric-card p { color: #ffffff; font-size: 1.5em; font-weight: 700; margin: 4px 0 0 0; }
    .regime-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        border: 2px solid #30475e;
        margin-bottom: 16px;
    }
    .sr-assess {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
        border-radius: 16px;
        padding: 22px 28px;
        border: 2px solid #30475e;
        margin-bottom: 16px;
    }
    div[data-testid="stSidebar"] { background: #0a0e14; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    ticker = st.text_input("Ticker", key="ticker",
                           help="e.g. BTC-USD, ETH-USD, AAPL, TSLA, MSFT")

    interval = st.selectbox("Interval", ["1h", "15m", "30m", "1d", "1wk"],
                            key="interval")

    max_d = INTERVAL_MAX_DAYS.get(interval, 730)
    # Visual cap for daily/weekly (10 years), mechanical cap for others (yfinance limits)
    visual_cap = 3650 if interval in ["1d", "1wk", "1mo"] else 730
    max_slider = min(max_d, visual_cap)
    
    # Clamp session state days to new max if interval changed
    st.session_state.days = min(st.session_state.days, max_slider)
    
    days = st.slider("History (days)", min_value=7, max_value=max_slider,
                     key="days", step=1)

    st.markdown("---")
    st.markdown("### 🔧 HMM Parameters")
    n_states       = st.selectbox("Number of states", [3, 4, 5], key="n_states")
    smoothing_hrs  = st.slider("Smoothing (hours)", 4, 48, key="smoothing_hrs", step=2)
    min_regime_hrs = st.slider("Min regime duration (hours)", 2, 24, key="min_regime_hrs", step=2)
    rsi_period     = st.slider("RSI period", 5, 30, key="rsi_period")

    st.markdown("---")
    st.markdown("### 🎯 S/R Parameters")
    sr_pivot_window = st.slider("Pivot window", 3, 30, key="sr_pivot_window", step=1,
                                help="Bars left/right to identify swing highs/lows")
    sr_cluster_pct  = st.slider("Cluster tolerance (%)", 0.3, 5.0, key="sr_cluster_pct", step=0.1,
                                help="Merge pivots within this % of each other")
    sr_train_pct    = st.slider("Walk-forward train %", 50, 90, key="sr_train_pct", step=5,
                                help="% of data used for training (rest = test)")

    st.markdown("---")
    st.markdown("### 🌲 RF Parameters")
    rf_trees = st.slider("Number of Trees", 50, 500, key="rf_trees", step=50)
    rf_depth = st.slider("Max Depth", 5, 20, key="rf_depth")
    rf_horizon = st.selectbox("Prediction Horizon", [1, 3, 5], key="rf_horizon", index=0, help="Bars ahead to predict")


    st.markdown("---")
    run_btn      = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    optimize_btn = st.button("🔍 Optimize Parameters", use_container_width=True)

# ── Main Area ────────────────────────────────────────────────────────────
st.markdown(f"# 📈 HMM Market Regime Detector")

# ── TABS ────────────────────────────────────────────────────────────────
tab_hmm, tab_sr, tab_rf, tab_opcoes, tab_tesouro = st.tabs([
    "📈 HMM Regimes",
    "🎯 Support / Resistance",
    "🌲 Random Forest Prediction",
    "📜 Opções de Mercado",
    "🏦 Tesouro Direto"
])


# ═════════════════════════════════════════════════════════════════════════
#  TAB 1 — HMM REGIMES (existing)
# ═════════════════════════════════════════════════════════════════════════
with tab_hmm:

    if optimize_btn:
        st.info(f"Downloading {ticker} ({interval}) data...")
        raw = download_data(ticker, interval, days)
        if raw.empty:
            st.error(f"No data returned for {ticker}. Check the ticker symbol.")
            st.stop()

        st.success(f"Downloaded {len(raw):,} rows. Running optimization...")
        rdf, best = run_optimization(raw, interval)

        if rdf is None:
            st.error("Optimization failed — no valid parameter combos.")
        else:
            st.session_state.hmm_opt_results = {"rdf": rdf, "best": best}

    if st.session_state.hmm_opt_results:
        res = st.session_state.hmm_opt_results
        rdf = res["rdf"]
        best = res["best"]

        st.markdown("### 🏆 Top 10 Parameter Combinations")
        display_cols = ["n_states", "smoothing_hrs", "min_regime_hrs", "rsi_period",
                        "trades", "win_rate", "sharpe", "total_return", "max_dd", "kelly"]
        st.dataframe(rdf.head(10)[display_cols].style.format({
            "win_rate": "{:.1f}%", "sharpe": "{:.2f}", "total_return": "{:+.1f}%",
            "max_dd": "{:.1f}%", "kelly": "{:.1f}%",
        }), use_container_width=True)

        st.markdown(f"""
        **Best config:** States={int(best['n_states'])} | Smooth={int(best['smoothing_hrs'])}h
        | MinRegime={int(best['min_regime_hrs'])}h | RSI={int(best['rsi_period'])}
        | **Sharpe={best['sharpe']:.2f}** | Return={best['total_return']:+.1f}%
        """)

        c1, c2 = st.columns(2)
        if c1.button("✅ Apply Best HMM Parameters", use_container_width=True, type="primary", on_click=apply_hmm_params):
            st.success("Parameters applied! Re-running analysis...")
            # No need for manual state update or rerun here, callback handles it
        if c2.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.hmm_opt_results = None
            st.rerun()

    elif run_btn:
        # ── Run full pipeline ────────────────────────────────────────────
        with st.spinner(f"Downloading {ticker} ({interval})..."):
            raw = download_data(ticker, interval, days)

        if raw.empty:
            st.error(f"No data returned for {ticker}. Check the ticker symbol.")
            st.stop()

        with st.spinner("Computing indicators & fitting HMM..."):
            df = engineer_features(raw, interval, smoothing_hrs, rsi_period=rsi_period)

            if df.empty or len(df) < n_states:
                st.error("Not enough data after feature engineering. Try more days or a different interval.")
                st.stop()

            df, converged, ll = fit_hmm(df, n_states, interval, min_regime_hrs)
            summary = build_summary(df, n_states)
            buys, sells = find_transitions(df, n_states)
            metrics = backtest(buys, sells)

        # ── Current Regime Diagnosis ──────────────────────────────────────
        current_state = df["State"].iloc[-1]
        current_price = df["Close"].iloc[-1]
        row = summary.loc[current_state]

        avg_24h   = df["Return"].iloc[-24:].mean()
        trend_dir = "↑ UP" if avg_24h > 0 else "↓ DOWN"
        rsi_now   = df["RSI_norm"].iloc[-1] * 100
        macd_now  = df["MACD_Hist"].iloc[-1]

        regime_colour = row["Colour"]

        st.markdown(f"""
        <div class="regime-box" style="border-color: {regime_colour};">
            <h2 style="color: {regime_colour}; margin:0; text-align:center;">
                {row['Tag']}
            </h2>
            <p style="color: #ccc; text-align:center; font-size:1.1em; margin:8px 0;">
                {ticker} — ${current_price:,.2f} — Trend: {trend_dir}
            </p>
            <p style="color: #888; text-align:center; margin:0;">
                {row['Action']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric Cards ──────────────────────────────────────────────────
        if metrics["trades"] > 0:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            card = lambda col, title, val: col.markdown(
                f'<div class="metric-card"><h3>{title}</h3><p>{val}</p></div>',
                unsafe_allow_html=True)
            card(c1, "Trades", str(metrics["trades"]))
            card(c2, "Win Rate", f"{metrics['win_rate']:.1f}%")
            card(c3, "Sharpe", f"{metrics['sharpe']:.2f}")
            card(c4, "Total Return", f"{metrics['total_return']:+.1f}%")
            card(c5, "Max Drawdown", f"{metrics['max_dd']:.1f}%")
            card(c6, "Kelly %", f"{metrics['kelly']:.1f}%")

        st.markdown("")

        # ── Plotly Chart ──────────────────────────────────────────────────
        fig = build_plotly_chart(df, summary, buys, sells, ticker, interval)
        st.plotly_chart(fig, use_container_width=True)

        # ── Extra Details ─────────────────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 📊 Regime Summary")
            disp = summary[["Tag", "Count", "Pct_Time", "Ann_Return", "Ann_Vol"]].copy()
            disp.columns = ["Regime", "Count", "% Time", "Ann Return", "Ann Vol"]
            st.dataframe(disp.style.format({
                "% Time": "{:.1f}%", "Ann Return": "{:+.2f}",
                "Ann Vol": "{:.2f}",
            }), use_container_width=True)

        with col_right:
            st.markdown("### 📡 Indicator Snapshot")
            snap_data = {
                "Indicator": ["RSI", "MACD Hist", "Bollinger Pos", "EMA Cross", "Fib Level"],
                "Value": [
                    f"{rsi_now:.1f}",
                    f"{macd_now:+.6f}",
                    f"{df['BB_Pos'].iloc[-1]:.2f}",
                    f"{df['EMA_Cross'].iloc[-1]:+.6f}",
                    f"{df['Fib_Level'].iloc[-1]:.2f}",
                ],
                "Signal": [
                    "Overbought" if rsi_now > 70 else "Oversold" if rsi_now < 30 else "Neutral",
                    "Bullish" if macd_now > 0 else "Bearish",
                    "Near Upper" if df["BB_Pos"].iloc[-1] > 0.8 else "Near Lower" if df["BB_Pos"].iloc[-1] < 0.2 else "Mid Band",
                    "Bullish" if df["EMA_Cross"].iloc[-1] > 0 else "Bearish",
                    "Near High" if df["Fib_Level"].iloc[-1] > 0.786 else "Near Low" if df["Fib_Level"].iloc[-1] < 0.382 else "Mid Range",
                ],
            }
            st.dataframe(pd.DataFrame(snap_data), use_container_width=True, hide_index=True)

        # Recent signals
        st.markdown("### 🔔 Recent Signals")
        all_sigs = [(t, p, "🟢 BUY") for t, p in buys] + [(t, p, "🔴 SELL") for t, p in sells]
        all_sigs.sort(key=lambda x: x[0])
        recent = all_sigs[-10:]
        if recent:
            sig_df = pd.DataFrame(recent, columns=["Time", "Price", "Signal"])
            sig_df["Price"] = sig_df["Price"].apply(lambda x: f"${x:,.2f}")
            sig_df["Time"]  = sig_df["Time"].astype(str).str[:19]
            st.dataframe(sig_df.iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("No signals generated with current parameters.")

        # HMM info
        st.caption(f"HMM: {n_states} states | Converged: {converged} | LL: {ll:,.2f} | "
                   f"Smoothing: {smoothing_hrs}h ({_bars(smoothing_hrs, interval)} bars) | "
                   f"Min regime: {min_regime_hrs}h ({_bars(min_regime_hrs, interval)} bars)")

    else:
        # Landing state
        st.markdown("""
        <div style="text-align: center; padding: 80px 20px;">
            <h2 style="color: #666;">Configure parameters in the sidebar and click <b>Run Analysis</b></h2>
            <p style="color: #555; font-size: 1.1em;">
                Supports any asset available on Yahoo Finance: stocks, crypto, ETFs, indices.<br>
                Examples: <code>BTC-USD</code>, <code>ETH-USD</code>, <code>AAPL</code>,
                <code>TSLA</code>, <code>SPY</code>, <code>MSFT</code>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
#  TAB 2 — SUPPORT / RESISTANCE
# ═════════════════════════════════════════════════════════════════════════
with tab_sr:

    sr_run_btn = st.button("🎯 Detect S/R Levels", key="sr_run",
                           use_container_width=True, type="primary")
    sr_wf_btn  = st.button("🔬 Walk-Forward Validation", key="sr_wf",
                           use_container_width=True)

    if sr_wf_btn:
        # ── Walk-forward validation ──────────────────────────────────────
        with st.spinner(f"Downloading {ticker} ({interval}) data..."):
            raw = download_data(ticker, interval, days)
        if raw.empty:
            st.error(f"No data returned for {ticker}. Check the ticker symbol.")
            st.stop()

        st.info(f"Running walk-forward validation ({sr_train_pct}% train / "
                f"{100 - sr_train_pct}% test) on {len(raw):,} bars...")

        prog = st.progress(0, text="Optimising S/R parameters...")
        rdf, best, split_idx = walk_forward(
            raw, train_pct=sr_train_pct / 100.0,
            progress_cb=lambda p, t: prog.progress(p, text=t),
        )
        prog.empty()

        if rdf is None or best is None:
            st.error("Walk-forward failed — not enough pivot points detected.")
        else:
            st.session_state.sr_wf_results = {
                "rdf": rdf, "best": best, "split_idx": split_idx
            }

    if st.session_state.sr_wf_results:
        res = st.session_state.sr_wf_results
        rdf = res["rdf"]
        best = res["best"]
        split_idx = res["split_idx"]

        # ── Best parameters ───────────────────────────────────────────
        st.markdown(f"""
        <div class="sr-assess">
            <h3 style="color:#ffeb3b; margin:0 0 12px 0; text-align:center;">
                🏆 Best Walk-Forward Parameters
            </h3>
            <p style="color:#ccc; text-align:center; font-size:1.1em; margin:0;">
                Pivot Window = <b>{best['pivot_window']}</b> |
                Cluster = <b>{best['cluster_pct']}%</b> |
                In-Sample Hold = <b>{best['in_sample_hold']:.1f}%</b> |
                Out-Sample Hold = <b>{best['out_sample_hold']:.1f}%</b>
            </p>
            <p style="color:#888; text-align:center; font-size:0.95em; margin:6px 0 0 0;">
                Out-of-sample: {best['out_trades']} trades | Win Rate: {best['out_win_rate']:.1f}%
                | Return: {best['out_total_ret']:+.2f}% | Sharpe: {best['out_sharpe']:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        if c1.button("✅ Apply Best S/R Parameters", use_container_width=True, type="primary", on_click=apply_sr_params):
            st.success("S/R Parameters applied! Re-running analysis...")
        if c2.button("🗑️ Clear S/R Results", use_container_width=True):
            st.session_state.sr_wf_results = None
            st.rerun()

        # ── Top combos table ──────────────────────────────────────────
        st.markdown("### 📊 Top 10 Parameter Combinations")
        show_cols = ["pivot_window", "cluster_pct", "in_sample_hold",
                     "out_sample_hold", "out_trades", "out_win_rate",
                     "out_total_ret", "out_sharpe"]
        st.dataframe(rdf.head(10)[show_cols].style.format({
            "cluster_pct": "{:.1f}%",
            "in_sample_hold": "{:.1f}%",
            "out_sample_hold": "{:.1f}%",
            "out_win_rate": "{:.1f}%",
            "out_total_ret": "{:+.2f}%",
            "out_sharpe": "{:.2f}",
        }), use_container_width=True)

        # ── Chart with best params ────────────────────────────────────
        # Re-fetch raw if not available in this rerun scope
        if 'raw' not in locals() or raw.empty:
            raw = download_data(ticker, interval, days)

        best_zones = best.get("zones_test", best.get("zones_train", []))
        fig_wf = build_sr_chart(raw, best_zones, ticker, interval,
                                cluster_pct=best['cluster_pct'],
                                split_idx=split_idx)
        st.plotly_chart(fig_wf, use_container_width=True)

        # ── Level details ─────────────────────────────────────────────
        if best_zones:
            st.markdown("### 🎯 Detected Levels (trained on first "
                        f"{sr_train_pct}% of data)")
            zt = pd.DataFrame(best_zones)
            zt["level"] = zt["level"].apply(lambda x: f"${x:,.2f}")
            show = ["level", "type", "touch_count", "hold_rate",
                    "break_rate", "avg_bounce_pct", "trade_score"]
            show = [c for c in show if c in zt.columns]
            st.dataframe(zt[show].style.format({
                "hold_rate": "{:.1f}%",
                "break_rate": "{:.1f}%",
                "avg_bounce_pct": "{:.2f}%",
                "trade_score": "{:.1f}",
            }, na_rep="-"), use_container_width=True, hide_index=True)

        st.caption(f"Train: {split_idx:,} bars | Test: {len(raw)-split_idx:,} bars | "
                   f"Best pivot_window={best['pivot_window']} cluster={best['cluster_pct']}%")

    elif sr_run_btn:
        # ── Standard S/R run ──────────────────────────────────────────────
        with st.spinner(f"Downloading {ticker} ({interval})..."):
            raw = download_data(ticker, interval, days)

        if raw.empty:
            st.error(f"No data returned for {ticker}. Check the ticker symbol.")
            st.stop()

        with st.spinner("Detecting support/resistance zones..."):
            zones, assessment = full_sr_analysis(
                raw, pivot_window=sr_pivot_window,
                cluster_pct=sr_cluster_pct)

        if not zones:
            st.warning("No S/R zones found. Try lowering the pivot window or cluster tolerance.")
            st.stop()

        current_price = raw["Close"].iloc[-1]

        # ── Position Assessment Card ──────────────────────────────────────
        ns = assessment["nearest_support"]
        nr = assessment["nearest_resistance"]
        rr = assessment["rr_ratio"]
        tp = assessment["trade_probability"]

        sup_str = f"${ns['level']:,.2f} (Hold: {ns['hold_rate']:.0f}%)" if ns else "—"
        res_str = f"${nr['level']:,.2f} (Hold: {nr['hold_rate']:.0f}%)" if nr else "—"
        rr_str  = f"1:{rr:.1f}" if rr else "—"
        tp_clr  = "#00e676" if tp >= 60 else "#ffb74d" if tp >= 45 else "#ef5350"

        st.markdown(f"""
        <div class="sr-assess">
            <h3 style="color:#29b6f6; margin:0 0 12px 0; text-align:center;">
                🎯 Current Position Assessment — ${current_price:,.2f}
            </h3>
            <div style="display:flex; justify-content:space-around; flex-wrap:wrap;">
                <div style="text-align:center; min-width:160px;">
                    <span style="color:#aaa; font-size:0.85em;">Nearest Support</span><br>
                    <span style="color:#00e676; font-size:1.2em; font-weight:700;">{sup_str}</span>
                </div>
                <div style="text-align:center; min-width:160px;">
                    <span style="color:#aaa; font-size:0.85em;">Nearest Resistance</span><br>
                    <span style="color:#ef5350; font-size:1.2em; font-weight:700;">{res_str}</span>
                </div>
                <div style="text-align:center; min-width:120px;">
                    <span style="color:#aaa; font-size:0.85em;">Risk / Reward</span><br>
                    <span style="color:#fff; font-size:1.2em; font-weight:700;">{rr_str}</span>
                </div>
                <div style="text-align:center; min-width:120px;">
                    <span style="color:#aaa; font-size:0.85em;">Trade Probability</span><br>
                    <span style="color:{tp_clr}; font-size:1.2em; font-weight:700;">{tp:.0f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric Cards ──────────────────────────────────────────────────
        n_sup = sum(1 for z in zones if z["type"].upper() == "SUPPORT")
        n_res = sum(1 for z in zones if z["type"].upper() == "RESISTANCE")
        avg_hold = np.mean([z["hold_rate"] for z in zones]) if zones else 0
        strongest = max(zones, key=lambda z: z["trade_score"])

        mc1, mc2, mc3, mc4 = st.columns(4)
        card = lambda col, title, val: col.markdown(
            f'<div class="metric-card"><h3>{title}</h3><p>{val}</p></div>',
            unsafe_allow_html=True)
        card(mc1, "Support Zones", str(n_sup))
        card(mc2, "Resistance Zones", str(n_res))
        card(mc3, "Avg Hold Rate", f"{avg_hold:.0f}%")
        card(mc4, "Strongest Level", f"${strongest['level']:,.2f}")

        st.markdown("")

        # ── Plotly Chart ──────────────────────────────────────────────────
        fig_sr = build_sr_chart(raw, zones, ticker, interval, cluster_pct=sr_cluster_pct)
        st.plotly_chart(fig_sr, use_container_width=True)

        # ── Levels Table ──────────────────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 🟢 Support Levels")
            sup_zones = [z for z in zones if z["type"].upper() == "SUPPORT"]
            if sup_zones:
                sdf = pd.DataFrame(sup_zones)
                sdf = sdf.sort_values("level", ascending=False)
                sdf["level"] = sdf["level"].apply(lambda x: f"${x:,.2f}")
                show = ["level", "touch_count", "hold_rate", "break_rate",
                        "avg_bounce_pct", "trade_score"]
                show = [c for c in show if c in sdf.columns]
                st.dataframe(sdf[show].style.format({
                    "hold_rate": "{:.1f}%", "break_rate": "{:.1f}%",
                    "avg_bounce_pct": "{:.2f}%", "trade_score": "{:.1f}",
                }, na_rep="-"), use_container_width=True, hide_index=True)
            else:
                st.info("No support zones detected.")

        with col_b:
            st.markdown("### 🔴 Resistance Levels")
            res_zones = [z for z in zones if z["type"].upper() == "RESISTANCE"]
            if res_zones:
                rdf2 = pd.DataFrame(res_zones)
                rdf2 = rdf2.sort_values("level", ascending=True)
                rdf2["level"] = rdf2["level"].apply(lambda x: f"${x:,.2f}")
                show = ["level", "touch_count", "hold_rate", "break_rate",
                        "avg_bounce_pct", "trade_score"]
                show = [c for c in show if c in rdf2.columns]
                st.dataframe(rdf2[show].style.format({
                    "hold_rate": "{:.1f}%", "break_rate": "{:.1f}%",
                    "avg_bounce_pct": "{:.2f}%", "trade_score": "{:.1f}",
                }, na_rep="-"), use_container_width=True, hide_index=True)
            else:
                st.info("No resistance zones detected.")

        st.caption(f"Pivot window: {sr_pivot_window} | Cluster: {sr_cluster_pct}% | "
                   f"Zones: {len(zones)} ({n_sup} support, {n_res} resistance)")

        # ── Market Intelligence Section ───────────────────────────────────
        st.markdown("---")
        st.markdown("### 🧠 Inteligência de Mercado (S/R Summary)")

        total_sup = len(sup_zones)
        total_res = len(res_zones)
        total_zones_all = total_sup + total_res or 1
        bias_label = "COMPRADOR" if total_sup > total_res else "VENDEDOR" if total_res > total_sup else "NEUTRO"
        bias_color = "#00e676" if total_sup > total_res else "#ef5350" if total_res > total_sup else "#b0bec5"
        bias_emoji = "🟢" if total_sup > total_res else "🔴" if total_res > total_sup else "⚪"
        sup_bar_pct = int(total_sup / total_zones_all * 100)
        res_bar_pct = 100 - sup_bar_pct

        # Proximity
        dist_s = ((current_price - ns['level']) / ns['level'] * 100) if ns else 999
        dist_r = ((nr['level'] - current_price) / current_price * 100) if nr else 999
        ns_hold  = f"{ns['hold_rate']:.0f}%" if ns else "—"
        nr_hold  = f"{nr['hold_rate']:.0f}%" if nr else "—"
        ns_touch = ns['touch_count'] if ns else 0
        nr_touch = nr['touch_count'] if nr else 0

        # Format price strings (no backslash needed inside f-string)
        sup_val_str = f"${ns['level']:,.2f}" if ns else "N/A"
        res_val_str = f"${nr['level']:,.2f}" if nr else "N/A"
        dist_s_str  = f"{dist_s:.2f}%" if ns else "N/A"
        dist_r_str  = f"{dist_r:.2f}%" if nr else "N/A"

        if dist_s < sr_cluster_pct:
            prox_title = f"⚡ Testando Suporte em {sup_val_str}"
            prox_color = "#00e676"
            prox_detail = f"Hold rate histórico: <b>{ns_hold}</b> · {ns_touch} toques detectados"
        elif dist_r < sr_cluster_pct:
            prox_title = f"⚡ Testando Resistência em {res_val_str}"
            prox_color = "#ef5350"
            prox_detail = f"Rejeição histórica: <b>{nr_hold}</b> · {nr_touch} toques detectados"
        else:
            prox_title = "Entre Níveis Estruturais"
            prox_color = "#29b6f6"
            prox_detail = f"Suporte: <b>{sup_val_str}</b> ({dist_s_str} abaixo) · Resistência: <b>{res_val_str}</b> ({dist_r_str} acima)"

        # Verdict
        if tp >= 60 and rr >= 2:
            rec_label  = "OPORTUNIDADE DE ALTA PROBABILIDADE"
            rec_emoji  = "✅"
            rec_color  = "#00e676"
            rec_bg     = "rgba(0, 230, 118, 0.12)"
            rec_border = "#00e676"
        elif tp >= 45:
            rec_label  = "OBSERVAR REAÇÃO NOS NÍVEIS"
            rec_emoji  = "👀"
            rec_color  = "#ffb74d"
            rec_bg     = "rgba(255, 183, 77, 0.10)"
            rec_border = "#ffb74d"
        else:
            rec_label  = "AGUARDAR CONFIRMAÇÃO"
            rec_emoji  = "✋"
            rec_color  = "#ef5350"
            rec_bg     = "rgba(239, 83, 80, 0.10)"
            rec_border = "#ef5350"

        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px;">

          <!-- BIAS CARD -->
          <div style="background:linear-gradient(135deg,#0d1b2a,#1b2838); border:1px solid {bias_color};
                      border-radius:14px; padding:20px 24px;">
            <div style="color:#aaa; font-size:0.78em; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;">
              Viés Estrutural
            </div>
            <div style="color:{bias_color}; font-size:1.35em; font-weight:800; margin-bottom:14px;">
              {bias_emoji} {bias_label}
              <span style="color:#777; font-size:0.65em; font-weight:400; margin-left:8px;">
                {total_sup} sup · {total_res} res
              </span>
            </div>
            <!-- ratio bar -->
            <div style="background:#1e2a37; border-radius:6px; height:10px; overflow:hidden; margin-bottom:10px;">
              <div style="background:linear-gradient(90deg,#00e676,#00c853); width:{sup_bar_pct}%;
                          height:100%; border-radius:6px 0 0 6px; display:inline-block;"></div>
              <div style="background:linear-gradient(90deg,#e53935,#ef5350); width:{res_bar_pct}%;
                          height:100%; border-radius:0 6px 6px 0; display:inline-block;"></div>
            </div>
            <div style="color:#888; font-size:0.82em;">
              Eficácia média das zonas: <b style="color:#e0e0e0;">{avg_hold:.0f}%</b>
            </div>
          </div>

          <!-- PROXIMITY CARD -->
          <div style="background:linear-gradient(135deg,#0d1b2a,#1b2838); border:1px solid {prox_color};
                      border-radius:14px; padding:20px 24px;">
            <div style="color:#aaa; font-size:0.78em; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;">
              Análise de Proximidade
            </div>
            <div style="color:{prox_color}; font-size:1.1em; font-weight:700; margin-bottom:12px;">
              {prox_title}
            </div>
            <div style="color:#ccc; font-size:0.88em; margin-bottom:12px;">{prox_detail}</div>
            <div style="display:flex; gap:20px;">
              <div>
                <div style="color:#aaa; font-size:0.75em; margin-bottom:2px;">↓ Dist. Suporte</div>
                <div style="color:#00e676; font-weight:700; font-size:1.0em;">{dist_s_str}</div>
              </div>
              <div>
                <div style="color:#aaa; font-size:0.75em; margin-bottom:2px;">↑ Dist. Resistência</div>
                <div style="color:#ef5350; font-weight:700; font-size:1.0em;">{dist_r_str}</div>
              </div>
              <div>
                <div style="color:#aaa; font-size:0.75em; margin-bottom:2px;">Assimetria R/R</div>
                <div style="color:#fff; font-weight:700; font-size:1.0em;">{rr_str}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- VERDICT BANNER -->
        <div style="background:{rec_bg}; border:1.5px solid {rec_border};
                    border-radius:14px; padding:18px 28px; display:flex;
                    align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
          <div>
            <div style="color:#aaa; font-size:0.78em; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px;">
              Veredito Técnico
            </div>
            <div style="color:{rec_color}; font-size:1.3em; font-weight:900; letter-spacing:0.03em;">
              {rec_emoji} {rec_label}
            </div>
          </div>
          <div style="display:flex; gap:24px; flex-wrap:wrap;">
            <div style="text-align:center;">
              <div style="color:#aaa; font-size:0.75em; margin-bottom:2px;">Prob. de Trade</div>
              <div style="color:{rec_color}; font-size:1.4em; font-weight:800;">{tp:.0f}%</div>
            </div>
            <div style="text-align:center;">
              <div style="color:#aaa; font-size:0.75em; margin-bottom:2px;">Assimetria R/R</div>
              <div style="color:#fff; font-size:1.4em; font-weight:800;">{rr_str}</div>
            </div>
            <div style="text-align:center;">
              <div style="color:#aaa; font-size:0.75em; margin-bottom:2px;">Zonas Ativas</div>
              <div style="color:#fff; font-size:1.4em; font-weight:800;">{len(zones)}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Landing state
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <h2 style="color: #666;">Click <b>Detect S/R Levels</b> to find support &amp; resistance zones</h2>
            <p style="color: #555; font-size: 1.05em;">
                Or use <b>Walk-Forward Validation</b> to auto-tune parameters —<br>
                trains on the first portion of data, tests on the rest, and finds the best settings.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
#  TAB 4 — OPÇÕES DE MERCADO
# ═════════════════════════════════════════════════════════════════════════
with tab_opcoes:
    st.markdown("### 📜 Análise de Opções (B3)")
    st.markdown("Dados buscados via API REST do **opcoes.net.br** — greeks reais (Delta, Gamma, Theta, Vega, IV).")

    # Strip .SA suffix — the scraper works with tickers like PETR4, VALE3
    search_ticker = ticker.replace(".SA", "").upper().strip()
    
    max_exp_opt = st.slider("Máx. vencimentos a buscar", 1, 12, 4, key="opcoes_max_exp",
                            help="Quantos vencimentos futuros serão consultados na API")

    if st.button("Buscar Grade de Opções", type="primary", use_container_width=True):
        with st.spinner(f"Buscando opções para {search_ticker} via opcoes.net.br..."):
            df_opcoes = buscar_dados_opcoes(search_ticker, max_expirations=max_exp_opt)
            if df_opcoes is not None and not df_opcoes.empty:
                # Persist in session_state so reruns (widget changes) don't lose the data
                st.session_state["df_opcoes"] = df_opcoes
                st.session_state["opcoes_ticker"] = search_ticker

    # Restore from session_state if already fetched
    df_opcoes = st.session_state.get("df_opcoes", None)
    opcoes_ticker_cached = st.session_state.get("opcoes_ticker", "")

    if df_opcoes is not None and not df_opcoes.empty:
        if opcoes_ticker_cached != search_ticker:
            st.info(f"Mostrando dados de **{opcoes_ticker_cached}**. Clique em 'Buscar' para atualizar para **{search_ticker}**.")
        st.success(f"✅ {len(df_opcoes)} contratos válidos para **{opcoes_ticker_cached}**!")

        # Vencimento filter
        venc_disponiveis = sorted(df_opcoes["Vencimento"].unique())
        venc_sel = st.selectbox("Filtrar por Vencimento", ["Todos"] + venc_disponiveis)
        df_show = df_opcoes if venc_sel == "Todos" else df_opcoes[df_opcoes["Vencimento"] == venc_sel]

        # CALL/PUT filter
        tipo_sel = st.radio("Tipo", ["Todos", "CALL", "PUT"], horizontal=True)
        if tipo_sel != "Todos":
            df_show = df_show[df_show["Tipo"] == tipo_sel]

        # Format display dataframe
        df_display = df_show.copy()
        for col in ["Strike", "Ultimo"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(2)
        for col in ["Delta", "Gama", "Theta", "Vega"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(4)
        if "Vol_Impl" in df_display.columns:
            df_display["Vol_Impl"] = df_display["Vol_Impl"].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "--"
            )

        st.dataframe(df_display, use_container_width=True, height=500)
        colunas = df_opcoes.columns.tolist()
        st.caption(f"Colunas: {', '.join(colunas)}")

        # --- OPTIONS STRATEGY RECOMMENDER ENGINE (IA) ---
        st.markdown("---")
        st.markdown("### 🤖 Sugestão Estratégica (IA + Gregas)")

        if "rf_results" in st.session_state:
            res = st.session_state["rf_results"]
            pred, prob_up = res["pred"], res["prob_up"]
            current_price = res["df_feat"]["Close"].iloc[-1]

            dir_pt = "DE ALTA 🟢" if pred == 1 else "DE BAIXA 🔴"
            conf_pt = (prob_up if pred == 1 else 1 - prob_up) * 100
            horizon_text = str(rf_horizon) if 'rf_horizon' in globals() else "futuros"

            st.info(f"**Detecção da Inteligência Artificial:** O modelo prevê que os próximos {horizon_text} períodos serão **{dir_pt}** com **{conf_pt:.1f}% de Confiança**.")

            from options_engine import rank_and_select_strategy
            strategy_data = rank_and_select_strategy(df_opcoes, pred, prob_up, current_price)

            if strategy_data:
                st.success(f"### Estratégia Recomendada pela IA: {strategy_data['strategy_name']}")
                st.markdown(f"**Por quê?** {strategy_data['rationale']}")
                for step in strategy_data['action_steps']:
                    st.write(f"• {step}")
                st.write(f"📊 **Nota de Adequação:** {strategy_data['engine_score']:.1f}/100")

                st.subheader("Simulador de Perfil da Estratégia (Payoff no Vencimento)")
                price_range = np.linspace(current_price * 0.85, current_price * 1.15, 100)
                total_payoff = np.zeros_like(price_range)
                for leg in strategy_data['plot_legs']:
                    s = leg['strike']
                    if leg['type'] == 'BUY_CALL':         total_payoff += np.maximum(price_range - s, 0)
                    elif leg['type'] == 'SELL_CALL':      total_payoff -= np.maximum(price_range - s, 0)
                    elif leg['type'] == 'BUY_PUT':        total_payoff += np.maximum(s - price_range, 0)
                    elif leg['type'] == 'SELL_PUT':       total_payoff -= np.maximum(s - price_range, 0)
                    elif leg['type'] == 'COVERED_CALL':   total_payoff += (price_range - current_price) - np.maximum(price_range - s, 0)
                    elif leg['type'] == 'SELL_CALL_2X':   total_payoff -= 2 * np.maximum(price_range - s, 0)

                fig_payoff = go.Figure()
                fig_payoff.add_trace(go.Scatter(
                    x=price_range, y=total_payoff,
                    mode='lines', name='Payoff da Estratégia',
                    line=dict(color='#00ffcc', width=3), fill='tozeroy'
                ))
                fig_payoff.update_layout(
                    template="plotly_dark", height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title="Preço no Vencimento", yaxis_title="Lucro/Prejuízo Teórico",
                    showlegend=False
                )
                st.plotly_chart(fig_payoff, use_container_width=True)
            else:
                st.warning("⚠️ O Motor não encontrou combinações de opções com liquidez suficiente para esta estratégia hoje.")

        else:
            st.warning("⚠️ Rode o módulo de **Random Forest ML** primeiro (na aba de Previsão) para que a IA destrave o Motor Sugestor de Opções.")

        # ── SELETOR MANUAL DE ESTRATÉGIA ──
        st.markdown("---")
        st.markdown("### 🎯 Escolha sua Estratégia")
        st.markdown(
            "Selecione uma estratégia e o motor buscará as melhores opções "
            "disponíveis para montá-la, com explicação detalhada de cada escolha."
        )

        from options_engine import AVAILABLE_STRATEGIES, build_specific_strategy as _build_strat

        col_strat1, col_strat2 = st.columns([3, 1])
        with col_strat1:
            strat_choice = st.selectbox(
                "Estratégia",
                ["— selecione —"] + AVAILABLE_STRATEGIES,
                key="manual_strat_select",
            )
        with col_strat2:
            venc_strat = st.selectbox(
                "Vencimento",
                ["Mais próximo"] + sorted(df_opcoes["Vencimento"].unique().tolist()),
                key="manual_strat_venc",
            )

        if st.button("🔍 Montar Estratégia", key="btn_montar_strat", use_container_width=True, type="primary"):
            if strat_choice == "— selecione —":
                st.warning("Por favor, selecione uma estratégia acima.")
            else:
                venc_ref = (sorted(df_opcoes["Vencimento"].unique())[0]
                            if venc_strat == "Mais próximo" else venc_strat)
                df_strat = df_opcoes[df_opcoes["Vencimento"] == venc_ref].copy()
                try:
                    spot_strat = float(df_strat["Strike"].dropna().median())
                except Exception:
                    spot_strat = float(df_opcoes["Strike"].dropna().median())

                with st.spinner(f"Montando {strat_choice}..."):
                    result_strat = _build_strat(df_strat, strat_choice, spot_strat)

                if result_strat:
                    st.success(f"✅ **{result_strat['strategy_name']}**")
                    st.caption(f"Vencimento: `{venc_ref}` | Preço ref.: R$ {spot_strat:.2f}")
                    st.markdown("---")

                    st.markdown("#### 📝 Por que essas opções foram escolhidas?")
                    st.info(result_strat["rationale"])

                    st.markdown("#### 📋 Pernas da Operação")
                    legs_data = pd.DataFrame([
                        {"Ação": lg["role"], "Opção Selecionada": lg["detalhe"]}
                        for lg in result_strat.get("legs", [])
                    ])
                    st.dataframe(legs_data, use_container_width=True, hide_index=True)

                    st.markdown("#### 💡 Perfil de Risco / Retorno")
                    st.markdown(result_strat["explanation"])

                    st.markdown("#### 📈 Payoff no Vencimento")
                    pr = np.linspace(spot_strat * 0.80, spot_strat * 1.20, 200)
                    py = np.zeros_like(pr)
                    for lg in result_strat.get("plot_legs", []):
                        sk = lg["strike"]
                        if lg["type"] == "BUY_CALL":   py += np.maximum(pr - sk, 0)
                        elif lg["type"] == "SELL_CALL": py -= np.maximum(pr - sk, 0)
                        elif lg["type"] == "BUY_PUT":   py += np.maximum(sk - pr, 0)
                        elif lg["type"] == "SELL_PUT":  py -= np.maximum(sk - pr, 0)

                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(
                        x=pr, y=py, mode="lines", fill="tozeroy",
                        line=dict(color="#FFD700", width=3), name="Payoff"
                    ))
                    fig_s.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_dash="dot")
                    fig_s.add_vline(
                        x=spot_strat, line_dash="dash", line_color="#00FFCC",
                        annotation_text=f"Preço ref R${spot_strat:.2f}",
                        annotation_position="top right"
                    )
                    fig_s.update_layout(
                        template="plotly_dark", height=360,
                        margin=dict(l=10, r=10, t=30, b=20),
                        xaxis_title="Preço do Ativo no Vencimento (R$)",
                        yaxis_title="Lucro / Prejuízo (R$)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    st.error(
                        f"⚠️ Motor não encontrou opções suficientes para **{strat_choice}** "
                        f"no vencimento `{venc_ref}`. Tente outro vencimento ou estratégia."
                    )

    else:
        st.error(f"Não foi possível encontrar opções para **{search_ticker}**. Verifique se o ticker é válido na B3.")

with tab_rf:
    rf_run_btn = st.button("🚀 Train & Predict (RF)", key="rf_run_main", 
                           use_container_width=True, type="primary")


    if rf_run_btn:
        with st.spinner("Preparing data and HMM states..."):
            raw = download_data(ticker, interval, days)
            if raw.empty:
                st.error("No data found.")
                st.stop()

            # Use engineer_features to get HMM states and indicators
            df_feat = engineer_features(raw, interval, smoothing_hrs, rsi_period=rsi_period)
            df_feat, _, _ = fit_hmm(df_feat, n_states, interval, min_regime_hrs)
            summary = build_summary(df_feat, n_states)

            X, y, feature_cols, returns_horizon = prepare_rf_data(df_feat, horizon=rf_horizon)



        with st.spinner("Training Random Forest model..."):
            model, acc, importance, X_test, y_test = train_rf_model(
                X, y, n_estimators=rf_trees, max_depth=rf_depth)

            # --- Get Prediction ---
            pred, prob_up = get_rf_prediction(model, df_feat.iloc[-1], feature_cols)

            # --- ATR for stops/targets ---
            high_low = df_feat["High"] - df_feat["Low"]
            high_cp  = np.abs(df_feat["High"] - df_feat["Close"].shift(1))
            low_cp   = np.abs(df_feat["Low"] - df_feat["Close"].shift(1))
            tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]

            last_price = df_feat["Close"].iloc[-1]
            hmm_tag = summary.loc[df_feat["State"].iloc[-1], "Tag"]

            advice = generate_trader_advice(
                pred, prob_up, last_price, atr, hmm_tag, df_feat["Cacas_Pos"].iloc[-1]
            )

            # --- Persist Results ---
            st.session_state["rf_results"] = {
                "model": model, "acc": acc, "importance": importance,
                "X": X, "y": y, "feature_cols": feature_cols,
                "returns_horizon": returns_horizon,
                "pred": pred, "prob_up": prob_up, "advice": advice,

                "df_feat": df_feat, "X_test": X_test
            }
        st.rerun() # Rerun to display results from session state

    if "rf_results" in st.session_state:
        res = st.session_state["rf_results"]
        model, acc, importance = res["model"], res["acc"], res["importance"]
        X, y, feature_cols = res["X"], res["y"], res["feature_cols"]
        returns_horizon = res.get("returns_horizon")
        pred, prob_up, advice = res["pred"], res["prob_up"], res["advice"]

        df_feat = res["df_feat"]
        X_test = res["X_test"]

        # ── Prediction Card ───────────────────────────────────────────────

        pred_label = "UP 🟢" if pred == 1 else "DOWN 🔴"

        # Color follows predicted direction, not raw prob_up value
        conf_color = "#00e676" if pred == 1 else "#ef5350"
        # Confidence = probability of the predicted direction
        conf_value = prob_up if pred == 1 else (1 - prob_up)
        conf_label = "Confidence (UP)" if pred == 1 else "Confidence (DOWN)"

        st.markdown(f"""
        <div class="sr-assess">
            <h3 style="color:#29b6f6; margin:0 0 12px 0; text-align:center;">
                🌲 RF Prediction (Next {rf_horizon} Bars)
            </h3>
            <div style="display:flex; justify-content:space-around; align-items:center;">
                <div style="text-align:center;">
                    <span style="color:#aaa; font-size:0.9em;">Model Accuracy (Test)</span><br>
                    <span style="color:#fff; font-size:1.8em; font-weight:700;">{acc*100:.1f}%</span>
                </div>
                <div style="text-align:center;">
                    <span style="color:#aaa; font-size:0.9em;">Predicted Direction</span><br>
                    <span style="color:{conf_color}; font-size:2.2em; font-weight:900;">{pred_label}</span>
                </div>
                <div style="text-align:center;">
                    <span style="color:#aaa; font-size:0.9em;">{conf_label}</span><br>
                    <span style="color:{conf_color}; font-size:1.8em; font-weight:700;">{conf_value*100:.1f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_la, col_lb = st.columns([1, 1])
        
        with col_la:
            st.markdown("### 📊 Importance Analysis")
            fig_imp = build_importance_chart(importance)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with col_lb:
            st.markdown("### 📈 Model Info")
            st.info(f"""
            - **Horizon:** {rf_horizon} bars
            - **Trees:** {rf_trees}
            - **Max Depth:** {rf_depth}
            - **Test Size:** {len(X_test)} bars
            - **Total Rows:** {len(X)}
            """)
            
        # ── Trader Advice Section ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎙️ Trader's Perspective")
        
        adv_col1, adv_col2 = st.columns([1, 2])
        
        with adv_col1:
            st.metric("Recomendação", advice["action"])
            st.write(f"**Entrada:** `{advice['entry']}`")
            st.write(f"**Stop Loss:** `{advice['stop']}`")
            st.write(f"**Alvo:** `{advice['target']}`")
            
        with adv_col2:
            st.markdown(f"**Por que este trade?**")
            st.info(advice["rationale"])
            
        # ── Optimization Section ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🛠️ Optimization (Hyper-parameters)")
        opt_col1, opt_col2 = st.columns([1, 2])
        
        with opt_col1:
            if st.button("🚀 Optimize RF Model", use_container_width=True):
                if df_feat is not None:
                    _prog_bar  = st.progress(0, text="Starting optimization...")
                    _prog_text = st.empty()

                    def _rf_progress_cb(count, total, msg):
                        _prog_bar.progress(count / total,
                                           text=f"Combo {count}/{total} — {msg}")
                        _prog_text.caption(f"✅ Last: {msg}")

                    opt_res = optimize_rf(df_feat, interval=interval,
                                         progress_cb=_rf_progress_cb)
                    _prog_bar.empty()
                    _prog_text.empty()

                    if opt_res.empty:
                        st.warning("Not enough data for meaningful optimization (need >60 bars).")
                    else:
                        st.session_state["rf_opt_results"] = opt_res
                else:
                    st.error("Run Analysis first to prepare data.")
                    
        if "rf_opt_results" in st.session_state and st.session_state["rf_opt_results"] is not None:
            df_opt = st.session_state["rf_opt_results"]
            # Guard: stale cache from old optimizer (no 'horizon' column) — clear and prompt re-run
            if "horizon" not in df_opt.columns:
                st.session_state["rf_opt_results"] = None
                st.info("⚠️ Previous optimization results are outdated. Please click **Optimize RF Model** again to include horizon search.")
                df_opt = None
            if df_opt is not None and not df_opt.empty:
                best = df_opt.iloc[0]

                with opt_col2:
                    st.success(
                        f"Best: Horizon=**{int(best['horizon'])}** bars · "
                        f"Trees={int(best['n_estimators'])} · Depth={int(best['max_depth'])}"
                    )
                    st.write(f"Acc: {best['accuracy']*100:.1f}% | Sharpe: {best['sharpe']:.2f} | "
                             f"Total Ret: {best['total_ret']*100:.2f}%")
                    if st.button("✅ Apply Best RF Parameters", on_click=apply_rf_params):
                        st.success("RF Parameters applied (Trees, Depth & Horizon)! Re-run analysis to use them.")

                st.markdown("#### 📊 Optimization Landscape")
                st.caption(
                    f"Showing heatmap for best horizon = {int(best['horizon'])} bar(s). "
                    "Other horizons available in the full table below."
                )
                # Filter to the best horizon for a clean 2-D pivot
                df_best_h = df_opt[df_opt["horizon"] == int(best["horizon"])]
                pivot = df_best_h.pivot(index="max_depth", columns="n_estimators", values="score")

                fig_grid = px.imshow(
                    pivot,
                    labels=dict(x="Number of Trees", y="Max Depth", color="Blended Score"),
                    title=f"Optimization Landscape — Horizon {int(best['horizon'])} bar(s) "
                          "(Acc 40% / Sharpe 60%)",
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="Viridis"
                )
                fig_grid.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig_grid, use_container_width=True)

                with st.expander("View Full Optimization Table (all horizons)"):
                    st.dataframe(
                        df_opt.style.format({
                            "accuracy":  "{:.1%}",
                            "sharpe":    "{:.2f}",
                            "total_ret": "{:.3f}",
                            "score":     "{:.3f}",
                        }),
                        use_container_width=True
                    )


            
    else:


        st.markdown(f"""
        <div style="text-align: center; padding: 60px 20px;">
            <h2 style="color: #666;">Click <b>Train &amp; Predict</b> to run the Random Forest ML module</h2>
            <p style="color: #555; font-size: 1.05em;">
                This model learns patterns from RSI, MACD, and <b>HMM Regime States</b><br>
                to predict the price direction of the next bars.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
#  TAB 5 — TESOURO DIRETO
# ═════════════════════════════════════════════════════════════════════════
with tab_tesouro:
    st.markdown("### 🏦 Tesouro Direto — Análise de Duration e Estratégia")
    st.markdown(
        "Carregue os títulos públicos disponíveis, veja a **duration** de cada um "
        "e monte sua estratégia de acordo com o cenário de taxa de juros."
    )

    # ── Botão de carregamento ─────────────────────────────────────────────
    col_btn_td, col_info_td = st.columns([1, 3])
    with col_btn_td:
        load_td_btn = st.button("📥 Carregar Títulos", type="primary",
                                use_container_width=True, key="btn_load_tesouro")
    with col_info_td:
        st.caption(
            "Fonte: Tesouro Transparente (gov.br). "
            f"Se indisponível, utiliza snapshot de referência de {datetime.date.today().year}."
        )

    if load_td_btn:
        with st.spinner("Buscando dados do Tesouro Direto..."):
            df_td = buscar_tesouro_direto()
        st.session_state["df_tesouro"] = df_td
        src = df_td.attrs.get("source", "fallback")
        if src == "selenium":
            st.success(f"✅ {len(df_td)} títulos carregados ao vivo via **site oficial** do Tesouro Direto (Selenium)")
        elif src == "tesouro_transparente":
            st.success(f"✅ {len(df_td)} títulos carregados via **Tesouro Transparente** (gov.br)")
        else:
            st.warning(
                f"⚠️ Site indisponível — exibindo **{len(df_td)} títulos** do snapshot de referência (07/03/2026). "
                "Os valores de taxa/preço são indicativos."
            )

    df_td = st.session_state.get("df_tesouro", None)

    if df_td is not None and not df_td.empty:

        # ── Seção 1: Tabela de Títulos ─────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📋 Tabela de Letras do Tesouro")

        tipos_disponiveis_td = sorted(df_td["Tipo"].unique())
        tipos_sel_td = st.multiselect(
            "Filtrar por tipo", tipos_disponiveis_td, default=tipos_disponiveis_td,
            key="td_tipos_sel"
        )
        df_show_td = df_td[df_td["Tipo"].isin(tipos_sel_td)] if tipos_sel_td else df_td

        def _style_duration(val):
            if isinstance(val, (int, float)):
                if val < 1:
                    return "color: #4fc3f7; font-weight: 700"
                elif val < 3:
                    return "color: #81c784; font-weight: 700"
                elif val < 7:
                    return "color: #ffb74d; font-weight: 700"
                else:
                    return "color: #ef9a9a; font-weight: 700"
            return ""

        st.dataframe(
            df_show_td.style
                .applymap(_style_duration, subset=["Duration Macaulay", "Duration Modificada"])
                .format({
                    "Taxa_Compra (% a.a.)": "{:.2f}%",
                    "Preco_Compra (R$)":    "R$ {:,.2f}",
                    "Anos_Venc":            "{:.1f} anos",
                    "Duration Macaulay":    "{:.2f}",
                    "Duration Modificada":  "{:.2f}",
                    "DV01 (R$/bp)":         "{:.4f}",
                }),
            use_container_width=True, height=400,
        )

        with st.expander("ℹ️ Guia de Leitura das Métricas"):
            st.markdown("""
| Métrica | O que mede |
|---|---|
| **Duration Macaulay** | Prazo médio ponderado dos fluxos de caixa (em anos). Quanto maior, mais sensível a juros. |
| **Duration Modificada** | % de variação no preço para cada 1% de variação na taxa. Ex: MD = 5 → preço cai ~5% se taxa sobe 1%. |
| **DV01 (R$/bp)** | Variação monetária no preço para 1 ponto-base (0,01%) de variação na taxa. |
| **LFT (Selic)** | Duration ≈ 0 — preço praticamente não oscila com mudanças de taxa. |
| **LTN / NTN-F (Prefixado)** | Taxa travada — maior risco de preço se Selic subir. |
| **NTN-B / NTN-B P (IPCA+)** | Protege da inflação; duration depende do prazo e dos cupons. |
""")

        # ── Seção 2: Curva de Yield ────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📈 Curva de Yield × Duration")
        st.caption("Eixo X = Duration Modificada | Eixo Y = Taxa de Compra | Tamanho = Anos até vencimento")

        _TD_COLOR_MAP = {
            "LFT":     "#4fc3f7",
            "LTN":     "#81c784",
            "NTN-F":   "#a5d6a7",
            "NTN-B P": "#ffb74d",
            "NTN-B":   "#ff8a65",
            "NTN-C":   "#ce93d8",
            "Renda+":  "#f06292",  # rosa coral — renda mensal / aposentadoria
            "Educa+":  "#ba68c8",  # roxo médio — educação
        }

        fig_yc = go.Figure()
        for tipo_yc in df_show_td["Tipo"].unique():
            df_tipo_yc = df_show_td[df_show_td["Tipo"] == tipo_yc]
            cor_yc = _TD_COLOR_MAP.get(tipo_yc, "#ffffff")
            fig_yc.add_trace(go.Scatter(
                x=df_tipo_yc["Duration Modificada"],
                y=df_tipo_yc["Taxa_Compra (% a.a.)"],
                mode="markers+text",
                name=tipo_yc,
                marker=dict(
                    size=df_tipo_yc["Anos_Venc"].clip(lower=1) * 4 + 8,
                    color=cor_yc, opacity=0.85,
                    line=dict(width=1, color="rgba(255,255,255,0.4)"),
                ),
                text=df_tipo_yc["Vencimento"].astype(str).str[:7],
                textposition="top center",
                textfont=dict(size=9, color=cor_yc),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Duration Mod.: %{x:.2f} anos<br>"
                    "Taxa: %{y:.2f}% a.a.<br>"
                    "Preço: R$ %{customdata[1]:,.2f}<extra></extra>"
                ),
                customdata=list(zip(df_tipo_yc["Titulo"], df_tipo_yc["Preco_Compra (R$)"])),
            ))

        fig_yc.update_layout(
            template="plotly_dark",
            paper_bgcolor="#121212", plot_bgcolor="#1a1a1a",
            height=500, margin=dict(l=60, r=30, t=40, b=50),
            xaxis_title="Duration Modificada (anos)",
            yaxis_title="Taxa de Compra (% a.a.)",
            legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(30,30,30,0.8)"),
            hovermode="closest",
        )
        fig_yc.update_xaxes(gridcolor="#333", zeroline=False)
        fig_yc.update_yaxes(gridcolor="#333", zeroline=False)
        st.plotly_chart(fig_yc, use_container_width=True)

        # ── Seção 3: Simulador de Portfólio ───────────────────────────────
        st.markdown("---")
        st.markdown("#### ⚖️ Simulador de Duration do Portfólio")
        st.caption("Ajuste os pesos de cada título. A duration ponderada do portfólio é recalculada.")

        titulos_td = df_td["Titulo"].tolist()
        
        titulos_selecionados = st.multiselect(
            "Selecione os títulos para incluir no portfólio:",
            options=titulos_td,
            # Seleciona os 3 primeiros por padrão se houver, pra não vir vazio
            default=titulos_td[:3] if len(titulos_td) >= 3 else titulos_td,
            key="td_portfolio_select"
        )
        
        n_td = len(titulos_selecionados)
        peso_default_td = round(100.0 / n_td, 1) if n_td > 0 else 0.0

        if n_td > 0:
            with st.form("form_portfolio_td"):
                st.markdown("**Pesos por título (%):**")
                cols_peso = st.columns(min(3, n_td))
                pesos_input = {}
                for idx_td, titulo_td in enumerate(titulos_selecionados):
                    col_idx_td = idx_td % 3
                    with cols_peso[col_idx_td]:
                        short_td = titulo_td.replace("Tesouro ", "")
                        pesos_input[titulo_td] = st.number_input(
                            short_td, min_value=0.0, max_value=100.0,
                            value=peso_default_td, step=1.0,
                            key=f"td_peso_{titulo_td}"
                        )
                calcular_port_btn = st.form_submit_button(
                    "🔢 Calcular Duration do Portfólio", use_container_width=True
                )
        else:
            st.info("Selecione pelo menos um título acima para montar o portfólio.")
            calcular_port_btn = False

        if calcular_port_btn:
            total_pesos_td = sum(pesos_input.values())
            if total_pesos_td <= 0:
                st.error("A soma dos pesos deve ser maior que zero.")
            else:
                result_port = calcular_duration_portfolio(df_td, pesos_input)
                mac_p  = result_port["macaulay_portfolio"]
                mod_p  = result_port["modified_portfolio"]
                dv01_p = result_port["dv01_portfolio"]

                if mod_p < 1:
                    bucket_td = ("CURTÍSSIMA", "#4fc3f7")
                elif mod_p < 3:
                    bucket_td = ("CURTA", "#81c784")
                elif mod_p < 7:
                    bucket_td = ("MÉDIA", "#ffb74d")
                else:
                    bucket_td = ("LONGA", "#ef9a9a")

                st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
     border:2px solid {bucket_td[1]}; border-radius:14px;
     padding:20px 28px; margin-bottom:12px;">
  <h4 style="color:{bucket_td[1]}; margin:0 0 12px 0; text-align:center;">
    Perfil de Duration: {bucket_td[0]}
  </h4>
  <div style="display:flex; justify-content:space-around; flex-wrap:wrap; gap:16px;">
    <div style="text-align:center;">
      <div style="color:#aaa; font-size:0.8em;">Duration Macaulay</div>
      <div style="color:#fff; font-size:2em; font-weight:800;">{mac_p:.2f} <span style="font-size:0.5em">anos</span></div>
    </div>
    <div style="text-align:center;">
      <div style="color:#aaa; font-size:0.8em;">Duration Modificada</div>
      <div style="color:{bucket_td[1]}; font-size:2em; font-weight:800;">{mod_p:.2f} <span style="font-size:0.5em">anos</span></div>
    </div>
    <div style="text-align:center;">
      <div style="color:#aaa; font-size:0.8em;">DV01 Ponderado</div>
      <div style="color:#fff; font-size:1.6em; font-weight:700;">R$ {dv01_p:.4f} <span style="font-size:0.5em">/bp</span></div>
    </div>
    <div style="text-align:center;">
      <div style="color:#aaa; font-size:0.8em;">Sensibilidade a +1% Selic</div>
      <div style="color:#ef5350; font-size:1.6em; font-weight:700;">-{mod_p:.1f}% <span style="font-size:0.5em">no preço</span></div>
    </div>
  </div>
</div>
                """, unsafe_allow_html=True)

                # Gráfico de pizza da alocação
                pesos_nz = {k: v for k, v in pesos_input.items() if v > 0}
                if pesos_nz:
                    short_labels_pie = [t.replace("Tesouro ", "") for t in pesos_nz.keys()]
                    pie_colors = []
                    for t_pie in pesos_nz.keys():
                        rows_pie = df_td[df_td["Titulo"] == t_pie]
                        tipo_pie = rows_pie["Tipo"].values[0] if len(rows_pie) > 0 else "LTN"
                        pie_colors.append(_TD_COLOR_MAP.get(tipo_pie, "#888"))
                    fig_pie = go.Figure(go.Pie(
                        labels=short_labels_pie, values=list(pesos_nz.values()),
                        hole=0.45, marker=dict(colors=pie_colors),
                        textinfo="percent+label", textfont=dict(size=11),
                    ))
                    fig_pie.update_layout(
                        template="plotly_dark", paper_bgcolor="#121212",
                        height=340, margin=dict(l=10, r=10, t=20, b=10),
                        showlegend=False,
                        annotations=[dict(
                            text=f"MD<br>{mod_p:.1f}a", x=0.5, y=0.5,
                            font_size=18, font_color=bucket_td[1], showarrow=False
                        )],
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

        # ── Seção 4: Estratégia Sugerida ──────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🎯 Motor de Estratégia por Cenário de Juros")

        cenarios_td = list(ESTRATEGIAS.keys())
        cenario_sel_td = st.selectbox(
            "Selecione o cenário macro de juros:", cenarios_td, key="td_cenario_sel"
        )
        strat_td = recomendar_estrategia(cenario_sel_td)

        dur_color_td = {
            "LONGA (> 5 anos)":   "#ef9a9a",
            "MEDIA (2–5 anos)":   "#ffb74d",
            "CURTA (< 2 anos)":   "#81c784",
        }.get(strat_td["duration_alvo"], "#fff")

        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1b2a,#1b2838);
     border:2px solid {strat_td['color']}; border-radius:16px;
     padding:24px 28px; margin-bottom:12px;">
  <h3 style="color:{strat_td['color']}; margin:0 0 10px 0;">
    {strat_td['emoji']} {cenario_sel_td}
  </h3>
  <p style="color:#ccc; margin:0 0 16px 0;">{strat_td['logica']}</p>
  <div style="display:flex; gap:28px; flex-wrap:wrap; align-items:center;">
    <div>
      <span style="color:#aaa; font-size:0.8em;">Duration Alvo</span><br>
      <span style="color:{dur_color_td}; font-size:1.3em; font-weight:800;">{strat_td['duration_alvo']}</span>
    </div>
    <div>
      <span style="color:#aaa; font-size:0.8em;">Tipos Preferidos</span><br>
      <span style="color:{strat_td['color']}; font-weight:700;">{", ".join(strat_td['tipos_preferidos']) or "–"}</span>
    </div>
    <div>
      <span style="color:#aaa; font-size:0.8em;">Evitar</span><br>
      <span style="color:#ef5350; font-weight:700;">{", ".join(strat_td['tipos_evitar']) or "Nenhum"}</span>
    </div>
  </div>
</div>
        """, unsafe_allow_html=True)

        # Gráfico de barras da alocação sugerida
        alocacao_td = {k: v for k, v in strat_td["alocacao"].items() if v > 0}
        if alocacao_td:
            st.markdown("**Alocação sugerida para este cenário:**")
            fig_alloc = go.Figure(go.Bar(
                x=list(alocacao_td.values()),
                y=list(alocacao_td.keys()),
                orientation="h",
                marker=dict(
                    color=[strat_td["color"]] * len(alocacao_td),
                    opacity=[0.4 + 0.6 * (v / max(alocacao_td.values()))
                             for v in alocacao_td.values()],
                    line=dict(color=strat_td["color"], width=1),
                ),
                text=[f"{v}%" for v in alocacao_td.values()],
                textposition="outside",
                textfont=dict(color="#ffffff", size=12),
            ))
            fig_alloc.update_layout(
                template="plotly_dark",
                paper_bgcolor="#121212", plot_bgcolor="#1a1a1a",
                height=240 + len(alocacao_td) * 24,
                margin=dict(l=20, r=60, t=20, b=20),
                xaxis=dict(title="Peso sugerido (%)", range=[0, 80], gridcolor="#333"),
                yaxis=dict(automargin=True, tickfont=dict(size=11)),
                showlegend=False,
            )
            st.plotly_chart(fig_alloc, use_container_width=True)

        # Títulos alinhados ao cenário
        if strat_td["tipos_preferidos"]:
            df_match_td = df_td[df_td["Tipo"].isin(strat_td["tipos_preferidos"])].copy()
            if not df_match_td.empty:
                st.markdown("**Títulos disponíveis alinhados ao cenário:**")
                st.dataframe(
                    df_match_td[
                        ["Titulo", "Tipo", "Vencimento",
                         "Taxa_Compra (% a.a.)", "Duration Macaulay", "Duration Modificada"]
                    ].style.format({
                        "Taxa_Compra (% a.a.)": "{:.2f}%",
                        "Duration Macaulay":    "{:.2f}",
                        "Duration Modificada":  "{:.2f}",
                    }),
                    use_container_width=True, hide_index=True,
                )

        # ── Seção 5: Simulador de Marcação a Mercado ───────────────────────────────
        st.markdown("---")
        st.markdown("#### 💸 Simulador de Marcação a Mercado")
        st.caption("Escolha os fundos/títulos, a quantidade aplicada e simule como uma alteração na taxa Selic ou juros futuros impactará o valor da sua carteira na marcação a mercado.")
        
        sim_titulos_selecionados = st.multiselect(
            "Selecione os títulos para simular:",
            options=titulos_td,
            default=titulos_td[:1] if len(titulos_td) > 0 else [],
            key="td_mtm_select"
        )
        
        if len(sim_titulos_selecionados) > 0:
            with st.form("form_mtm_td"):
                st.markdown("**Valor Aplicado (R$):**")
                cols_mtm = st.columns(min(3, len(sim_titulos_selecionados)))
                valores_aplicados = {}
                for idx_sim, titulo_sim in enumerate(sim_titulos_selecionados):
                    col_idx_sim = idx_sim % 3
                    with cols_mtm[col_idx_sim]:
                        short_sim = titulo_sim.replace("Tesouro ", "")
                        valores_aplicados[titulo_sim] = st.number_input(
                            short_sim, min_value=0.0,
                            value=1000.0, step=100.0, format="%.2f",
                            key=f"td_mtm_{titulo_sim}"
                        )
                
                # Valores Atuais (Base)
                st.markdown("**Taxas Atuais (Base para o Cálculo):**")
                col_base1, col_base2 = st.columns(2)
                with col_base1:
                    selic_atual = st.number_input(
                        "Selic Atual (% a.a.)",
                        min_value=2.0, max_value=20.0, value=10.50, step=0.25,
                        format="%.2f",
                        help="A taxa básica de juros atual (Meta Selic definida pelo COPOM)."
                    )
                with col_base2:
                    ipca_atual = st.number_input(
                        "Inflação Implícita / Meta (% a.a.)",
                        min_value=0.0, max_value=15.0, value=4.50, step=0.25,
                        format="%.2f",
                        help="A inflação de longo prazo atualmente precificada no mercado."
                    )
                
                st.markdown("---")
                
                # Slider para simular choque nos juros
                st.markdown("**Projeções de Mercado (Simulação):**")
                
                col_slider1, col_slider2 = st.columns(2)
                with col_slider1:
                    st.caption("Expectativa de Juros Base")
                    selic_projetada = st.slider(
                        "Selic Projetada (% a.a.)",
                        min_value=2.0, max_value=20.0, value=float(selic_atual), step=0.25,
                        format="%.2f%%",
                        help="Simulação da expectativa para a taxa básica de juros. Uma queda na Selic geralmente valoriza títulos de renda fixa."
                    )
                    
                    # Usa a Selic atual escolhida pelo usuário para calcular um "choque" paralelo na curva
                    choque_juros = selic_projetada - selic_atual
                with col_slider2:
                    st.caption("Expectativa de Inflação (IPCA)")
                    inflacao_projetada = st.slider(
                        "Inflação Projetada (% a.a.)",
                        min_value=0.0, max_value=15.0, value=float(ipca_atual), step=0.1,
                        format="%.1f%%",
                        help="Impacta EXCLUSIVAMENTE títulos atrelados à inflação (IPCA+, Renda+, Educa+). Uma queda na inflação projetada desvaloriza esses títulos."
                    )
                    
                    # Usa a Inflação atual escolhida pelo usuário para calcular o "choque" de mercado
                    choque_inflacao = inflacao_projetada - ipca_atual
                
                simular_mtm_btn = st.form_submit_button("Calculadora de Marcação a Mercado 🧮", use_container_width=True)
                
            if simular_mtm_btn:
                resultados_mtm = []
                lucro_total = 0.0
                valor_final_total = 0.0
                valor_aplicado_total = 0.0
                
                for titulo_sim in sim_titulos_selecionados:
                    vl_aplicado = valores_aplicados[titulo_sim]
                    linha_titulo = df_td[df_td["Titulo"] == titulo_sim].iloc[0]
                    dur_mod = linha_titulo["Duration Modificada"]
                    conv = linha_titulo["Convexidade"] if "Convexidade" in linha_titulo else 0.0
                    taxa_atual = linha_titulo["Taxa_Compra (% a.a.)"]
                    tipo = linha_titulo["Tipo"]
                    
                    # Se o título é atrelado à inflação, o choque total na YTM nominal 
                    # é a soma do choque na taxa real com o choque na inflação implícita.
                    is_ipca = tipo in ["NTN-B P", "NTN-B", "Renda+", "Educa+"]
                    choque_total_aplicado = choque_juros
                    if is_ipca:
                        choque_total_aplicado += choque_inflacao
                    
                    # Nova taxa estimada
                    taxa_simulada = taxa_atual + choque_total_aplicado
                    
                    # Aproximação MTM com Convexidade: 
                    # Variação % Preço = -Duration_Modificada * dY + 0.5 * Convexidade * (dY)^2
                    choque_decimal = choque_total_aplicado / 100.0
                    variacao_preco_pct = (-dur_mod * choque_decimal) + (0.5 * conv * (choque_decimal ** 2))
                    
                    vl_final = vl_aplicado * (1 + variacao_preco_pct)
                    lucro_titulo = vl_final - vl_aplicado
                    
                    resultados_mtm.append({
                        "Título": titulo_sim.replace("Tesouro ", ""),
                        "Valor Aplicado (R$)": vl_aplicado,
                        "Taxa Simulada (% a.a.)": taxa_simulada,
                        "Variação Estimada (%)": variacao_preco_pct * 100,
                        "Novo Valor Estimado (R$)": vl_final,
                        "Lucro/Prejuízo (R$)": lucro_titulo
                    })
                    
                    valor_aplicado_total += vl_aplicado
                    valor_final_total += vl_final
                    lucro_total += lucro_titulo
                    
                df_mtm = pd.DataFrame(resultados_mtm)
                
                # Explicando o cálculo matemático
                with st.expander("📚 Entenda o Cálculo e a Inflação"):
                    st.markdown(f"""
                    O sistema utiliza a **Duration Modificada** e a **Convexidade** de cada título para estimar a variação no preço com precisão para movimentos de taxa:
                    `Variação no Preço ≈ [- (Duration) × (Variação Total)] + [½ × Convexidade × (Variação Total)²]`
                    
                    **E a Inflação? Onde ela entra?**
                    Títulos Prefixados têm sua rentabilidade já cravada. Mas títulos como o IPCA+ são compostos de duas partes: uma **Taxa Real** fixa (ex: 6% a.a.) + a **Inflação** (IPCA). O mercado projeta uma inflação futura (a *inflação implícita*). 
                    Se a expectativa de inflação subir, os investidores exigirão prêmios maiores para compensar, o que significa que a *taxa de mercado nominal* aumenta, derrubando o preço do título hoje.
                    
                    Por isso dividimos a simulação em duas partes absolutas:
                    1. **Alteração na Expectativa de Juros Base:** O simulador usa a sua Selic Atual de **{selic_atual:.2f}%**. Ao projetar a nova Selic para **{selic_projetada:.2f}%**, o sistema aplica um choque real de mercado de **{choque_juros:+.2f}%** na rentabilidade de todos os títulos.
                    2. **Alteração na Expectativa de Inflação:** Impacta *apenas* títulos atrelados ao IPCA. O simulador considera a inflação base de **{ipca_atual:.2f}%**. Ao projetar a inflação para **{inflacao_projetada:.1f}%**, o sistema soma à curva destes títulos um novo choque puramente inflacionário de **{choque_inflacao:+.2f}%**.
                    """)

                # Definir cor de destaque do resultado total
                cor_resultado = "#81c784" if lucro_total >= 0 else "#ef5350"
                sinal_resultado = "+" if lucro_total >= 0 else ""
                
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1b2838,#16213e);
                     border:2px solid {cor_resultado}; border-radius:12px;
                     padding:16px 20px; margin-top:10px; margin-bottom:20px; text-align:center;">
                  <h4 style="margin:0; color:#eee;">Resumo da Simulação de MTM</h4>
                  <p style="margin:5px 0 0 0; color:#aaa; font-size:1.05em;">Impacto estimado considerando uma variação de <strong style="color:{cor_resultado};">{choque_juros:+.2f}%</strong> nas taxas de juros</p>
                  <div style="display:flex; justify-content:center; gap:40px; margin-top:16px;">
                    <div>
                        <span style="color:#aaa; font-size:0.9em;">Total Aplicado</span><br>
                        <span style="color:#fff; font-size:1.5em; font-weight:bold;">R$ {valor_aplicado_total:,.2f}</span>
                    </div>
                    <div>
                        <span style="color:#aaa; font-size:0.9em;">Novo Saldo (MTM)</span><br>
                        <span style="color:{cor_resultado}; font-size:1.5em; font-weight:bold;">R$ {valor_final_total:,.2f}</span>
                    </div>
                    <div>
                        <span style="color:#aaa; font-size:0.9em;">Resultado Nominal</span><br>
                        <span style="color:{cor_resultado}; font-size:1.5em; font-weight:bold;">{sinal_resultado} R$ {lucro_total:,.2f}</span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                
                def colorir_resultado(val):
                    if isinstance(val, (int, float)) and val < 0:
                        return "color: #ef5350; font-weight: bold"
                    elif isinstance(val, (int, float)) and val > 0:
                        return "color: #81c784; font-weight: bold"
                    return ""
                
                st.dataframe(
                    df_mtm.style
                        .applymap(colorir_resultado, subset=["Variação Estimada (%)", "Lucro/Prejuízo (R$)"])
                        .format({
                            "Valor Aplicado (R$)": "R$ {:,.2f}",
                            "Taxa Simulada (% a.a.)": "{:.2f}%",
                            "Variação Estimada (%)": "{:+.2f}%",
                            "Novo Valor Estimado (R$)": "R$ {:,.2f}",
                            "Lucro/Prejuízo (R$)": "R$ {:+,.2f}"
                        }),
                    use_container_width=True, hide_index=True
                )
        else:
            st.info("Selecione os títulos que deseja simular e clique em 'Calculadora de Marcação a Mercado'.")

    else:
        st.markdown("""
<div style="text-align:center; padding:60px 20px;">
    <h2 style="color:#666;">Clique em <b>📥 Carregar Títulos</b> para começar</h2>
    <p style="color:#555; font-size:1.05em;">
        O sistema buscará os títulos disponíveis do Tesouro Direto e calculará<br>
        a <b>Duration Macaulay</b>, <b>Duration Modificada</b> e <b>DV01</b> de cada uno.
    </p>
</div>
        """, unsafe_allow_html=True)
