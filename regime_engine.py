import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# LAYER 1: TREND REGIME
# ==============================================================================
def calculate_trend_regime(df, ema_short=20, ema_mid=50, sma_long=200, adx_period=14):
    """
    Computes directional trend using Moving Average configurations and ADX.
    Assumes df has 'High', 'Low', 'Close'.
    """
    df = df.copy()
    
    # EMAs / SMA
    df['EMA_S'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA_M'] = df['Close'].ewm(span=ema_mid, adjust=False).mean()
    df['SMA_L'] = df['Close'].rolling(window=sma_long).mean()
    
    # ADX Calculation (Simplified Wilder's)
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DoMove'] = df['Low'].shift(1) - df['Low']
    
    df['+DM'] = np.where((df['UpMove'] > df['DoMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DoMove'] > df['UpMove']) & (df['DoMove'] > 0), df['DoMove'], 0)
    
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    df['ATR_14'] = df['TR'].rolling(window=adx_period).mean()
    df['+DI'] = 100 * (df['+DM'].rolling(window=adx_period).mean() / df['ATR_14'])
    df['-DI'] = 100 * (df['-DM'].rolling(window=adx_period).mean() / df['ATR_14'])
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']).replace(0, np.nan))
    df['ADX'] = df['DX'].rolling(window=adx_period).mean()
    
    # Trend Classification
    def classify_trend(row):
        if pd.isna(row['SMA_L']):
            return 'NEUTRAL'
        
        # Bullish conditions
        if row['EMA_S'] > row['EMA_M'] > row['SMA_L']:
            if row['ADX'] > 25 and row['+DI'] > row['-DI']:
                return 'ALTA FORTE (STRONG BULL)'
            return 'ALTA FRACA (WEAK BULL)'
            
        # Bearish conditions
        if row['EMA_S'] < row['EMA_M'] < row['SMA_L']:
            if row['ADX'] > 25 and row['-DI'] > row['+DI']:
                return 'BAIXA FORTE (STRONG BEAR)'
            return 'BAIXA FRACA (WEAK BEAR)'
            
        return 'LATERALIZADO / NEUTRO'
        
    df['Trend_Regime'] = df.apply(classify_trend, axis=1)
    return df

# ==============================================================================
# LAYER 2: VOLATILITY REGIME
# ==============================================================================
def calculate_volatility_regime(df, atr_period=14, lookback=100):
    """
    Computes volatility regime using ATR percentiles.
    """
    df = df.copy()
    
    # Ensure ATR exists if not computed by Trend layer
    if 'ATR_14' not in df.columns:
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        df['ATR_14'] = tr.rolling(window=atr_period).mean()
        
    # ATR as a % of price to normalize
    df['ATR_Pct'] = (df['ATR_14'] / df['Close']) * 100
    
    # Rolling percentiles
    def classify_volatility(row):
        val = row['ATR_Pct']
        p75 = row['ATR_75']
        p25 = row['ATR_25']
        
        if pd.isna(val) or pd.isna(p75):
            return 'VOLATILIDADE NORMAL'
            
        if val > p75:
            return 'ALTA VOLATILIDADE (RISCO)'
        elif val < p25:
            return 'BAIXA VOLATILIDADE (CONTRAÇÃO)'
        else:
            return 'VOLATILIDADE NORMAL'
            
    df['ATR_75'] = df['ATR_Pct'].rolling(window=lookback).quantile(0.75)
    df['ATR_25'] = df['ATR_Pct'].rolling(window=lookback).quantile(0.25)
    
    df['Vol_Regime'] = df.apply(classify_volatility, axis=1)
    return df

# ==============================================================================
# LAYER 3: STATISTICAL HMM REGIME
# ==============================================================================
def fit_advanced_hmm(df, n_states=4):
    """
    Fits an HMM focused strictly on returns and variance (the true statistical regime).
    """
    df = df.copy()
    
    # Features focused strictly on microstructure dynamics
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Range_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['Ret_Roll_Std'] = df['Log_Ret'].rolling(10).std()
    
    df_clean = df.dropna(subset=['Log_Ret', 'Range_Pct', 'Ret_Roll_Std'])
    if df_clean.empty:
        df['HMM_State_Raw'] = -1
        df['HMM_Regime'] = "UNKNOWN"
        return df, False, 0.0
        
    features = ['Log_Ret', 'Range_Pct', 'Ret_Roll_Std']
    X = StandardScaler().fit_transform(df_clean[features])
    
    try:
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X)
        states = model.predict(X)
        converged = model.monitor_.converged
        score = model.score(X)
    except:
        return df, False, 0.0
        
    # Map back to subset
    df_clean['HMM_State_Raw'] = states
    
    # Characterize States based on Mean Return and Volatility(Variance)
    summary = df_clean.groupby('HMM_State_Raw')[['Log_Ret', 'Range_Pct']].mean()
    
    # Smart Mapping:
    # High Ret, Low Var -> Mark-up
    # High Ret, High Var -> Mark-up (Volatile)
    # Low/Neg Ret, Low Var -> Markdown (Bleed) / Accumulation
    # Low/Neg Ret, High Var -> Markdown (Panic) / Distribution
    
    # Sort by return to establish a baseline 0,1,2,3...
    remap = {s: i for i, s in enumerate(summary['Log_Ret'].sort_values().index)}
    df_clean['HMM_Mapped'] = df_clean['HMM_State_Raw'].map(remap)
    
    def assign_hmm_name(val):
        if val == n_states - 1: return "MARK-UP / EXPANSÃO" # Highest return
        if val == 0: return "MARK-DOWN / QUEDA"          # Lowest return
        
        # Intermediate states
        if val == n_states - 2: return "ACUMULAÇÃO / LENTA ALTA" 
        return "DISTRIBUIÇÃO / RUIDOSO"
        
    df_clean['HMM_Regime'] = df_clean['HMM_Mapped'].apply(assign_hmm_name)
    
    # Merge back to original DataFrame to maintain index
    df['HMM_Regime'] = df_clean['HMM_Regime']
    df['HMM_State_Mapped'] = df_clean['HMM_Mapped']
    df['HMM_Regime'].fillna("CALCULATING...", inplace=True)
    
    return df, converged, score

# ==============================================================================
# COMPOSITE ENGINE
# ==============================================================================
def build_composite_regime(df):
    """
    Executes the 3-Layer architecture and returns the unified DataFrame and a strategic summary.
    """
    df = calculate_trend_regime(df)
    df = calculate_volatility_regime(df)
    df, hmm_ok, hmm_score = fit_advanced_hmm(df, n_states=4)
    
    # Determine actionable strategy based on the current (last) row
    last_row = df.iloc[-1]
    
    tr = last_row.get('Trend_Regime', 'UNKNOWN')
    vr = last_row.get('Vol_Regime', 'UNKNOWN')
    hm = last_row.get('HMM_Regime', 'UNKNOWN')
    
    # ── Dynamic Entry / Exit Calculation ───────────
    current_price = last_row['Close']
    atr = last_row.get('ATR_14', current_price * 0.02) # Fallback to 2% if missing
    ema_s = last_row.get('EMA_S', current_price)
    sma_l = last_row.get('SMA_L', current_price - atr)
    
    entry_zone = "Aguardar"
    stop_loss = "N/A"
    take_profit = "N/A"
    
    if "ALTA FORTE" in tr:
        if "BAIXA" in vr:
            strategy = ("Aproveite breakouts para compras. Acumule com confiança. "
                        "A baixa volatilidade indica menor risco de interrupções bruscas de tendência (stop hunts).")
            conviction = 90
        elif "ALTA" in vr:
            strategy = ("Risco de Topo Exaustivo ou mercado de euforia extrema. "
                        "Siga a tendência, mas limite o tamanho da sua posição devido à forte oscilação de preço.")
            conviction = 60
        else:
            strategy = "Seguidor de tendência. Compre nas retrações em suporte dinâmico de preços."
            conviction = 80
            
        entry_zone = f"R$ {ema_s:,.2f} (Retração na EMA Curta) ou Rompimento de R$ {last_row['High']:,.2f}"
        stop_loss = f"R$ {(current_price - 1.5 * atr):,.2f} (Perda de Momento)"
        take_profit = f"R$ {(current_price + 2.5 * atr):,.2f} (Alvo de Expansão)"
            
    elif "BAIXA FORTE" in tr:
        if "ALTA" in vr:
            strategy = ("Fase de Pânico (Mark-Down). NÃO tente segurar a queda. "
                        "Venda nas retrações ou utilize posições de proteção/hedge defensivo fortes.")
            conviction = 90
        else:
            strategy = "Derretimento lento do preço. Venda em falhas de resistências curtas."
            conviction = 70
            
        entry_zone = f"Venda (Short) próximo a R$ {ema_s:,.2f} ou perda de R$ {last_row['Low']:,.2f}"
        stop_loss = f"R$ {(current_price + 1.5 * atr):,.2f} (Recuperação de Tendência)"
        take_profit = f"R$ {(current_price - 2.5 * atr):,.2f} (Alvo de Despejo)"
            
    elif "LATERALIZADO" in tr:
        if "BAIXA" in vr:
            strategy = ("Provavelmente forjando uma base / acumulação em andamento. "
                        "Opere canais ou venda volatilidade com opções (e.g. Iron Condors). Aguarde explosão de volume para posicionar direcionalmente.")
            conviction = 50
        else:
            strategy = ("Situação de forte ruído / Whipsaw. Alto risco direcional. "
                        "Utilize apenas estratégias de reversão à média nas extremidades do range.")  
            conviction = 40
            
        entry_zone = f"Compra em R$ {(current_price - atr):,.2f} / Venda em R$ {(current_price + atr):,.2f} (Bandas)"
        stop_loss = f"Fechamento externo a ±1.5x ATR"
        take_profit = f"Retorno à Média (R$ {ema_s:,.2f})"
            
    return df, {
        "Trend": tr,
        "Volatility": vr,
        "HMM_Stat": hm,
        "Strategy_Advice": strategy,
        "Conviction": conviction,
        "Entry_Zone": entry_zone,
        "Stop_Loss": stop_loss,
        "Take_Profit": take_profit,
        "HMM_Converged": hmm_ok
    }
