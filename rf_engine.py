# =============================================================================
#  Random Forest Engine — Predictive ML for Price Direction
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go


def prepare_rf_data(df, horizon=1):
    """
    Prepare features and target for Random Forest.
    Target: 1 if Close[t+horizon] > Close[t], else 0.
    Features: RSI, MACD, Bollinger Bands, EMA Cross, HMM State (if available).
    """
    data = df.copy()
    
    # Target: Next direction
    data["Target"] = (data["Close"].shift(-horizon) > data["Close"]).astype(int)
    
    # Features already existing in df from engineer_features:
    # ['RSI_norm', 'MACD_Hist', 'BB_Pos', 'EMA_Cross', 'Fib_Level', 'State']
    
    feature_cols = ["RSI_norm", "MACD_Hist", "BB_Pos", "EMA_Cross", "Fib_Level", "Cacas_Pos"]
    if "State" in data.columns:
        feature_cols.append("State")
    
    # Drop NaNs from shift and pointers
    data = data.dropna(subset=feature_cols + ["Target"])
    
    X = data[feature_cols]
    y = data["Target"]
    
    # Optional: Actual pct returns for the horizon (for Sharjah/Strategy calc)
    # We use log returns shift to get the 'future' return
    returns_horizon = np.log(data["Close"].shift(-horizon) / data["Close"])
    
    return X, y, feature_cols, returns_horizon


def train_rf_model(X, y, n_estimators=100, max_depth=10):
    """Train Random Forest and return model + metrics."""
    # Split chronologically to avoid lookahead bias
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Eval
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Importance
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return model, acc, importance, X_test, y_test

def get_rf_prediction(model, last_row, feature_cols):
    """Predict direction for the VERY NEXT bar."""
    X_now = last_row[feature_cols].values.reshape(1, -1)
    prob = model.predict_proba(X_now)[0] # [prob_down, prob_up]
    pred = model.predict(X_now)[0]
    
    return pred, prob[1] # direction, probability of UP

def generate_trader_advice(pred, prob_up, current_price, atr, hmm_tag, cacas_val):
    """Generate a human-like trader advice based on ML + Context."""
    is_bull_signal = prob_up > 0.55
    is_bear_signal = prob_up < 0.45
    confidence = abs(prob_up - 0.5) * 200 # 0-100% scale
    
    # HMM Sentiment
    hmm_sentiment = "BULL" if "BULL" in hmm_tag.upper() else "BEAR" if "BEAR" in hmm_tag.upper() else "NEUTRAL"
    
    advice = {
        "action": "AGUARDAR ⚖️",
        "entry": "-",
        "stop": "-",
        "target": "-",
        "rationale": "O modelo não encontrou uma vantagem clara (edge) nas condições atuais."
    }
    
    if is_bull_signal and confidence > 15:
        advice["action"] = "COMPRA / LONG 🟢"
        advice["entry"] = f"${current_price:,.2f}"
        advice["stop"] = f"${current_price - (atr * 1.5):,.2f}"
        advice["target"] = f"${current_price + (atr * 3.0):,.2f}"
        
        if hmm_sentiment == "BULL":
            logic = f"Confluência de alta detectada: RF ({prob_up*100:.1f}%) alinhado com o Regime HMM {hmm_tag}."
        elif hmm_sentiment == "BEAR":
            logic = f"Sinal de contra-tendência: RF indica força compradora ({prob_up*100:.1f}%) apesar do Regime HMM {hmm_tag}."
        else:
            logic = f"RF ({prob_up*100:.1f}%) sugere rompimento de alta em regime {hmm_tag}."
            
        advice["rationale"] = f"{logic} O Cacas Channel confirma posicionamento favorável para continuidade."

    elif is_bear_signal and confidence > 15:
        advice["action"] = "VENDA / SHORT 🔴"
        advice["entry"] = f"${current_price:,.2f}"
        advice["stop"] = f"${current_price + (atr * 1.5):,.2f}"
        advice["target"] = f"${current_price - (atr * 3.0):,.2f}"
        
        if hmm_sentiment == "BEAR":
            logic = f"ALTA PROBABILIDADE DE QUEDA ({(1-prob_up)*100:.1f}%): O modelo prevê continuidade da queda, em total confluência com o Regime HMM {hmm_tag}."
        elif hmm_sentiment == "BULL":
            logic = f"ALERTA DE CORREÇÃO ({(1-prob_up)*100:.1f}% de chance de queda): Apesar do mercado estar em regime {hmm_tag}, o modelo estatístico detectou um forte sinal de reversão/topo."
        else:
            logic = f"ALTA PROBABILIDADE DE QUEDA ({(1-prob_up)*100:.1f}%): O Random Forest sugere fraqueza iminente saindo de um regime {hmm_tag}."
            
        advice["rationale"] = f"{logic} Venda na perda da mínima para surfar o movimento de baixa."

        
    return advice


# Bars per year per interval — used to annualize Sharpe correctly
_BARS_PER_YEAR = {
    "1m":  252 * 390,
    "5m":  252 * 78,
    "15m": 252 * 26,
    "30m": 252 * 13,
    "1h":  252 * 6,
    "2h":  252 * 3,
    "4h":  252 * 1.5,
    "1d":  252,
    "1wk": 52,
    "1mo": 12,
}


def build_importance_chart(importance):
    """Plotly bar chart for feature importance — most important on top."""
    # Sort ascending so horizontal bars render top = most important
    imp_sorted = importance.sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=imp_sorted.values,
        y=imp_sorted.index,
        orientation="h",
        marker_color="#29b6f6"
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Predictor Importance (Random Forest)",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def optimize_rf(df, interval="1d", progress_cb=None):
    """
    Walk-forward grid search over n_estimators, max_depth, AND horizon.
    Accepts the feature-engineered df directly so horizon can be varied.
    Sharpe is annualized correctly for the given interval.
    progress_cb(count, total, msg) is called after each combo if provided.
    """
    ann_factor = np.sqrt(_BARS_PER_YEAR.get(interval, 252))

    estimators_list = [50, 100, 200, 300]
    depth_list      = [5, 8, 12, 16, 20]
    horizon_list    = [1, 3, 5]

    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    total_combos = len(horizon_list) * len(estimators_list) * len(depth_list)
    combo_count  = 0

    for h in horizon_list:
        # Build a fresh X, y, returns for this horizon
        X, y, feature_cols, returns_h = prepare_rf_data(df, horizon=h)
        if len(X) < 60:
            continue  # Not enough data for this horizon

        for n in estimators_list:
            for d in depth_list:
                fold_accs    = []
                fold_sharpes = []
                fold_returns = []

                for train_idx, test_idx in tscv.split(X):
                    X_train = X.iloc[train_idx]
                    X_test  = X.iloc[test_idx]
                    y_train = y.iloc[train_idx]

                    # Drop tail NaNs from horizon shift
                    r_test  = returns_h.iloc[test_idx].dropna()
                    if r_test.empty:
                        continue
                    X_test_r = X_test.loc[r_test.index]

                    model = RandomForestClassifier(
                        n_estimators=n, max_depth=d,
                        random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)

                    fold_accs.append(model.score(X_test, y.iloc[test_idx]))

                    # Strategy: long (+1) or short (-1)
                    y_pred     = model.predict(X_test_r)
                    strat_rets = (np.where(y_pred == 1, 1, -1)) * r_test
                    fold_returns.append(strat_rets.sum())

                    if strat_rets.std() > 0:
                        sharpe = (strat_rets.mean() / strat_rets.std()) * ann_factor
                        fold_sharpes.append(sharpe)
                    else:
                        fold_sharpes.append(0)

                if not fold_accs:
                    continue

                results.append({
                    "horizon":      h,
                    "n_estimators": n,
                    "max_depth":    d,
                    "accuracy":     np.mean(fold_accs),
                    "sharpe":       np.mean(fold_sharpes),
                    "total_ret":    np.mean(fold_returns),
                })

                combo_count += 1
                if progress_cb:
                    progress_cb(
                        combo_count, total_combos,
                        f"Horizon={h}bar | Trees={n} | Depth={d} "
                        f"→ Acc={np.mean(fold_accs)*100:.1f}% "
                        f"Sharpe={np.mean(fold_sharpes):.2f} "
                        f"({combo_count}/{total_combos})"
                    )

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    # Blended score: 40% accuracy + 60% Sharpe
    df_res["score"] = (df_res["accuracy"] * 0.4) + (df_res["sharpe"] * 0.6)
    return df_res.sort_values("score", ascending=False)


