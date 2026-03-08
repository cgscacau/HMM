# =============================================================================
#  Bitcoin Market Regime Detection — HMM + Technical Indicators + Backtest
# =============================================================================
"""
Usage:
    pip install yfinance hmmlearn scikit-learn matplotlib pandas numpy
    python btc_hmm_regime_detection.py                          # BTC-USD 1h (default)
    python btc_hmm_regime_detection.py --ticker ETH-USD         # Ethereum
    python btc_hmm_regime_detection.py --ticker AAPL --interval 1d  # Apple daily
    python btc_hmm_regime_detection.py --optimize               # grid-search best params

    Supported intervals: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
"""

import os
import sys
import warnings
import datetime
import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not installed.  Run:  pip install yfinance")

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    sys.exit("hmmlearn not installed.  Run:  pip install hmmlearn")


# ═══════════════════════════════════════════════════════════════════════════
#  DEFAULT CONFIGURATION  (overridden by --optimize)
# ═══════════════════════════════════════════════════════════════════════════
# Max download days per interval (yfinance limits)
INTERVAL_MAX_DAYS = {
    "1m": 7, "5m": 60, "15m": 60, "30m": 60,
    "1h": 730, "1d": 10000, "1wk": 10000, "1mo": 10000,
}
# How many bars = 1 "period" for smoothing (adapts to interval)
INTERVAL_BARS_PER_DAY = {
    "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
    "1h": 24, "1d": 1, "1wk": 1/7, "1mo": 1/30,
}

class Config:
    TICKER         = "BTC-USD"
    INTERVAL       = "1h"
    N_STATES       = 4
    SMOOTHING_HRS  = 24
    MIN_REGIME_HRS = 8
    RSI_PERIOD     = 14
    EMA_FAST       = 9
    EMA_SLOW       = 21
    BB_PERIOD      = 20
    BB_STD         = 2
    MACD_FAST      = 12
    MACD_SLOW      = 26
    MACD_SIGNAL    = 9
    FIB_LOOKBACK   = 100
    TOTAL_DAYS     = 730
    CHUNK_DAYS     = 59

    @property
    def smoothing_bars(self):
        """Adapt smoothing window to interval (e.g. 12h = 12 bars for 1h, 288 for 1m)."""
        bpd = INTERVAL_BARS_PER_DAY.get(self.INTERVAL, 24)
        return max(1, int(self.SMOOTHING_HRS * bpd / 24))

    @property
    def min_regime_bars(self):
        """Adapt min regime duration to interval."""
        bpd = INTERVAL_BARS_PER_DAY.get(self.INTERVAL, 24)
        return max(1, int(self.MIN_REGIME_HRS * bpd / 24))

CFG = Config()

REGIME_TAGS = {
    3: [("BEAR", "#ef5350"), ("WEAK BEAR", "#ffb74d"), ("BULL", "#00e676")],
    4: [("BEAR", "#ef5350"), ("WEAK BEAR", "#ffb74d"),
        ("WEAK BULL", "#29b6f6"), ("BULL", "#00e676")],
    5: [("BEAR", "#d50000"), ("WEAK BEAR", "#ef5350"), ("NEUTRAL", "#fff176"),
        ("WEAK BULL", "#29b6f6"), ("BULL", "#00e676")],
}

REGIME_ACTIONS = {
    "BEAR":      "Strong downtrend -- reduce exposure, hedge, protect capital.",
    "WEAK BEAR": "Drift bearish -- tighten stops, be cautious with new longs.",
    "NEUTRAL":   "No clear direction -- reduce size, wait for breakout.",
    "WEAK BULL": "Drift bullish -- hold positions, add selectively on dips.",
    "BULL":      "Strong uptrend -- ride momentum, accumulate on pullbacks.",
}


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════
def download_data(verbose=True) -> pd.DataFrame:
    # Adapt total_days to interval limits
    max_days = INTERVAL_MAX_DAYS.get(CFG.INTERVAL, 730)
    actual_days = min(CFG.TOTAL_DAYS, max_days)

    end   = datetime.datetime.now()
    start = end - datetime.timedelta(days=actual_days)
    if verbose:
        print(f"[1/6] Downloading {CFG.TICKER} ({CFG.INTERVAL})  {start.date()} -> {end.date()} ...")

    # For daily+ intervals, single download works fine
    use_chunks = CFG.INTERVAL in ("1m", "5m", "15m", "30m", "1h")
    chunk_days = min(CFG.CHUNK_DAYS, max_days) if use_chunks else actual_days

    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + datetime.timedelta(days=chunk_days), end)
        part = yf.download(CFG.TICKER, start=cursor, end=chunk_end,
                           interval=CFG.INTERVAL, auto_adjust=True, progress=False)
        if part is not None and not part.empty:
            chunks.append(part)
        cursor = chunk_end

    if not chunks:
        sys.exit("ERROR: yfinance returned no data.")

    raw = pd.concat(chunks)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    raw.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    if verbose:
        print(f"      {len(raw):,} rows")
    return raw


# ═══════════════════════════════════════════════════════════════════════════
#  2. TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators from raw OHLCV."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # ── EMA Crossover ───────────────────────────────────────────────────────
    ema_fast = close.ewm(span=CFG.EMA_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=CFG.EMA_SLOW, adjust=False).mean()
    df["EMA_Cross"] = (ema_fast - ema_slow) / close   # normalised

    # ── Bollinger Bands ─────────────────────────────────────────────────────
    bb_mid   = close.rolling(CFG.BB_PERIOD).mean()
    bb_std   = close.rolling(CFG.BB_PERIOD).std()
    bb_upper = bb_mid + CFG.BB_STD * bb_std
    bb_lower = bb_mid - CFG.BB_STD * bb_std
    bb_width = bb_upper - bb_lower
    df["BB_Pos"] = np.where(bb_width > 0, (close - bb_lower) / bb_width, 0.5)

    # ── RSI ─────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(CFG.RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(CFG.RSI_PERIOD).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI_norm"] = (100 - 100 / (1 + rs)) / 100   # 0-1

    # ── MACD ────────────────────────────────────────────────────────────────
    macd_line   = close.ewm(span=CFG.MACD_FAST, adjust=False).mean() - \
                  close.ewm(span=CFG.MACD_SLOW, adjust=False).mean()
    macd_signal = macd_line.ewm(span=CFG.MACD_SIGNAL, adjust=False).mean()
    df["MACD_Hist"] = (macd_line - macd_signal) / close  # normalised

    # ── Fibonacci Level ─────────────────────────────────────────────────────
    roll_high = high.rolling(CFG.FIB_LOOKBACK, min_periods=10).max()
    roll_low  = low.rolling(CFG.FIB_LOOKBACK, min_periods=10).min()
    fib_range = roll_high - roll_low
    df["Fib_Level"] = np.where(fib_range > 0, (close - roll_low) / fib_range, 0.5)

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
FEATURES = [
    "Smoothed_Return", "Smoothed_Range", "Smoothed_VolChg",
    "EMA_Cross", "BB_Pos", "RSI_norm", "MACD_Hist", "Fib_Level",
]


def engineer_features(raw: pd.DataFrame, verbose=True) -> pd.DataFrame:
    if verbose:
        print(f"[2/6] Computing indicators + smoothed features ...")
    if raw.empty:
        sys.exit("ERROR: No data.")

    df = raw.copy()

    # Raw price features
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Range"]  = (df["High"] - df["Low"]) / df["Close"]
    df["VolChg"] = df["Volume"].pct_change()

    # Smooth core features
    w = CFG.smoothing_bars
    df["Smoothed_Return"] = df["Return"].rolling(w, min_periods=1).mean()
    df["Smoothed_Range"]  = df["Range"].rolling(w, min_periods=1).mean()
    df["Smoothed_VolChg"] = df["VolChg"].rolling(w, min_periods=1).mean()

    # Technical indicators
    df = compute_indicators(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES, inplace=True)

    if df.empty:
        sys.exit("ERROR: No valid data after feature engineering.")
    if verbose:
        print(f"      Feature matrix: {df[FEATURES].shape}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  4. HMM FIT + STATE REORDERING + SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════
def fit_hmm(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    if verbose:
        print(f"[3/6] Fitting HMM ({CFG.N_STATES} states) ...")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].values)

    model = GaussianHMM(
        n_components=CFG.N_STATES, covariance_type="full",
        n_iter=1000, random_state=42, verbose=False,
    )
    model.fit(X)
    if verbose:
        print(f"      Converged: {model.monitor_.converged}  |  LL: {model.score(X):,.2f}")

    raw_states = model.predict(X)

    # Reorder states by mean return (bearish=0 → bullish=N-1)
    means = {}
    for s in range(CFG.N_STATES):
        mask = raw_states == s
        means[s] = df.loc[mask, "Return"].mean() if mask.any() else 0.0
    remap = {old: new for new, old in enumerate(sorted(means, key=means.get))}
    df["State"] = np.array([remap[s] for s in raw_states])

    # Smooth regime flips shorter than min_regime_bars
    min_bars = CFG.min_regime_bars
    states = df["State"].values.copy()
    i = 0
    while i < len(states):
        j = i
        while j < len(states) and states[j] == states[i]:
            j += 1
        if (j - i) < min_bars and i > 0:
            states[i:j] = states[i - 1]
        else:
            i = j
            continue
        i = j
    df["State"] = states

    if verbose:
        print(f"      Regimes smoothed (min {CFG.MIN_REGIME_HRS}h)")
    return df


def build_summary(df):
    HOURS_PER_YEAR = 365.25 * 24
    summary = (
        df.groupby("State")["Return"]
        .agg(Count="count", Mean_Return="mean", Std_Return="std")
        .sort_index()
    )
    summary["Ann_Return"] = summary["Mean_Return"] * HOURS_PER_YEAR
    summary["Ann_Vol"]    = summary["Std_Return"]  * np.sqrt(HOURS_PER_YEAR)
    summary["Pct_Time"]   = (summary["Count"] / summary["Count"].sum() * 100).round(1)

    tags = REGIME_TAGS.get(CFG.N_STATES, REGIME_TAGS[4])
    for i, (tag, colour) in enumerate(tags):
        if i in summary.index:
            summary.loc[i, "Tag"]    = tag
            summary.loc[i, "Colour"] = colour
            summary.loc[i, "Action"] = REGIME_ACTIONS.get(tag, "")

    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  5. TRANSITIONS — CONFIRMED SIGNALS (BUY / SELL)
# ═══════════════════════════════════════════════════════════════════════════
def find_transitions(df):
    """Generate BUY/SELL signals using regime change + indicator confirmation.
    A signal only fires when the regime flips AND at least 2 of 3 indicators agree."""
    bull_thresh = CFG.N_STATES // 2
    is_bull = df["State"] >= bull_thresh

    rsi   = df["RSI_norm"].values    # 0-1 scale
    macd  = df["MACD_Hist"].values
    ema_x = df["EMA_Cross"].values

    buys, sells = [], []
    prev = is_bull.iloc[0]

    for i in range(1, len(df)):
        curr = is_bull.iloc[i]

        if curr and not prev:
            # Regime flipped bullish — confirm with indicators
            bull_votes = int(rsi[i] > 0.40) + int(macd[i] > 0) + int(ema_x[i] > 0)
            if bull_votes >= 2:
                buys.append((df.index[i], df["Close"].iloc[i]))

        elif not curr and prev:
            # Regime flipped bearish — confirm with indicators
            bear_votes = int(rsi[i] < 0.60) + int(macd[i] < 0) + int(ema_x[i] < 0)
            if bear_votes >= 2:
                sells.append((df.index[i], df["Close"].iloc[i]))

        prev = curr

    return buys, sells


# ═══════════════════════════════════════════════════════════════════════════
#  6. BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def backtest(df, buys, sells, verbose=True):
    """Long-only backtest: enter at BUY, exit at SELL."""
    if verbose:
        print("[4/6] Running backtest ...")

    # Merge into chronological signal list
    signals = ([(t, p, "BUY") for t, p in buys] +
               [(t, p, "SELL") for t, p in sells])
    signals.sort(key=lambda x: x[0])

    trades = []
    entry_price = None
    entry_time  = None

    for t, price, sig in signals:
        if sig == "BUY" and entry_price is None:
            entry_price = price
            entry_time  = t
        elif sig == "SELL" and entry_price is not None:
            pnl_pct = (price - entry_price) / entry_price * 100
            trades.append({
                "entry_time":  entry_time,
                "exit_time":   t,
                "entry_price": entry_price,
                "exit_price":  price,
                "pnl_pct":     pnl_pct,
                "duration_h":  (t - entry_time).total_seconds() / 3600,
            })
            entry_price = None
            entry_time  = None

    if not trades:
        if verbose:
            print("      No completed trades.")
        return {"sharpe": -999, "trades": 0}

    tdf = pd.DataFrame(trades)
    wins  = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]

    total      = len(tdf)
    win_rate   = len(wins) / total if total else 0
    avg_win    = wins["pnl_pct"].mean() if len(wins) else 0
    avg_loss   = abs(losses["pnl_pct"].mean()) if len(losses) else 0.001
    rr_ratio   = avg_win / avg_loss if avg_loss > 0 else 0
    expectation = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    kelly      = win_rate - ((1 - win_rate) / rr_ratio) if rr_ratio > 0 else 0
    kelly      = max(0, kelly)

    # Cumulative returns for Sharpe
    returns = tdf["pnl_pct"].values / 100
    sharpe  = (returns.mean() / returns.std() * np.sqrt(len(returns))
               if returns.std() > 0 else 0)

    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns   = (cum_returns - running_max) / running_max * 100
    max_dd      = drawdowns.min()

    total_return = (cum_returns[-1] - 1) * 100

    metrics = {
        "trades":       total,
        "win_rate":     win_rate,
        "avg_win":      avg_win,
        "avg_loss":     avg_loss,
        "rr_ratio":     rr_ratio,
        "expectation":  expectation,
        "kelly":        kelly * 100,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "total_return": total_return,
    }

    if verbose:
        print()
        print("=" * 65)
        print("  BACKTEST RESULTS")
        print("=" * 65)
        print(f"  Total trades:       {total}")
        print(f"  Win Rate:           {win_rate*100:.1f}%")
        print(f"  Avg Win:            +{avg_win:.2f}%")
        print(f"  Avg Loss:           -{avg_loss:.2f}%")
        print(f"  Risk/Reward:        1:{rr_ratio:.2f}")
        print(f"  Expectation:        {expectation:+.3f}% per trade")
        print(f"  Kelly Criterion:    {kelly*100:.1f}% of capital")
        print(f"  Sharpe Ratio:       {sharpe:.2f}")
        print(f"  Max Drawdown:       {max_dd:.1f}%")
        print(f"  Total Return:       {total_return:+.1f}%")
        print("=" * 65)
        print()

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  7. PARAMETER OPTIMISATION (Grid Search)
# ═══════════════════════════════════════════════════════════════════════════
def optimize(raw):
    """Grid search over key HMM + indicator params, scored by Sharpe."""
    print()
    print("=" * 65)
    print("  PARAMETER OPTIMISATION  (Grid Search)")
    print("=" * 65)

    param_grid = {
        "N_STATES":       [3, 4, 5],
        "SMOOTHING_HRS":  [6, 12, 24],
        "MIN_REGIME_HRS": [4, 6, 12],
        "RSI_PERIOD":     [10, 14, 21],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)
    print(f"  Testing {total} parameter combinations ...\n")

    results = []

    for idx, values in enumerate(combos):
        params = dict(zip(keys, values))

        # Apply params to global config
        CFG.N_STATES       = params["N_STATES"]
        CFG.SMOOTHING_HRS  = params["SMOOTHING_HRS"]
        CFG.MIN_REGIME_HRS = params["MIN_REGIME_HRS"]
        CFG.RSI_PERIOD     = params["RSI_PERIOD"]

        try:
            df = engineer_features(raw, verbose=False)
            df = fit_hmm(df, verbose=False)
            buys, sells = find_transitions(df)
            metrics = backtest(df, buys, sells, verbose=False)
            results.append({**params, **metrics})
        except Exception:
            continue

        # Progress
        if (idx + 1) % 20 == 0 or idx == total - 1:
            print(f"  [{idx+1}/{total}] completed")

    if not results:
        print("  ERROR: No valid results from grid search.")
        return

    rdf = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    # Top 5
    print()
    print("=" * 65)
    print("  TOP 5 PARAMETER COMBINATIONS  (by Sharpe Ratio)")
    print("=" * 65)
    top = rdf.head(5)
    display_cols = ["N_STATES", "SMOOTHING_HRS", "MIN_REGIME_HRS", "RSI_PERIOD",
                    "trades", "win_rate", "sharpe", "total_return", "max_dd", "kelly"]
    print(top[display_cols].to_string(index=False, float_format="{:.2f}".format))
    print("=" * 65)

    # Apply best
    best = rdf.iloc[0]
    print(f"\n  >>> BEST: States={int(best['N_STATES'])}  "
          f"Smooth={int(best['SMOOTHING_HRS'])}h  "
          f"MinRegime={int(best['MIN_REGIME_HRS'])}h  "
          f"RSI={int(best['RSI_PERIOD'])}  "
          f"Sharpe={best['sharpe']:.2f}  "
          f"Return={best['total_return']:+.1f}%\n")

    CFG.N_STATES       = int(best["N_STATES"])
    CFG.SMOOTHING_HRS  = int(best["SMOOTHING_HRS"])
    CFG.MIN_REGIME_HRS = int(best["MIN_REGIME_HRS"])
    CFG.RSI_PERIOD     = int(best["RSI_PERIOD"])

    return rdf


# ═══════════════════════════════════════════════════════════════════════════
#  8. DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════
def diagnose(df, summary, buys, sells):
    print()
    print("[5/6] -- CURRENT REGIME DIAGNOSIS ---------------------------------")

    current_state = df["State"].iloc[-1]
    current_price = df["Close"].iloc[-1]
    current_time  = df.index[-1]
    row = summary.loc[current_state]

    avg_24h   = df["Return"].iloc[-24:].mean()
    trend_dir = "UP" if avg_24h > 0 else "DOWN"

    # Streak
    states_rev = df["State"].iloc[::-1].values
    streak = sum(1 for s in states_rev if s == current_state)

    # Current indicator values
    rsi_now   = df["RSI_norm"].iloc[-1] * 100
    macd_now  = df["MACD_Hist"].iloc[-1]
    bb_now    = df["BB_Pos"].iloc[-1]
    ema_cross = df["EMA_Cross"].iloc[-1]
    fib_now   = df["Fib_Level"].iloc[-1]

    print()
    print(f"  +--------------------------------------------------------------+")
    print(f"  |  REGIME:  {row['Tag']:<50s}|")
    print(f"  +--------------------------------------------------------------+")
    print(f"  |  Price:        ${current_price:>10,.2f}                         |")
    print(f"  |  Time:         {str(current_time)[:19]:<41s}|")
    print(f"  |  24h trend:    {trend_dir:<6s} (avg: {avg_24h:+.5f})                  |")
    print(f"  |  Streak:       {streak:>4d}h ({streak/24:.1f} days)                      |")
    print(f"  +--------------------------------------------------------------+")
    print(f"  |  RSI:          {rsi_now:>6.1f}  {'(overbought)' if rsi_now>70 else '(oversold)' if rsi_now<30 else '(neutral)':<28s}|")
    print(f"  |  MACD Hist:    {macd_now:>+10.6f}  {'(bullish)' if macd_now>0 else '(bearish)':<20s}|")
    print(f"  |  Bollinger:    {bb_now:>6.2f}  {'(near upper)' if bb_now>0.8 else '(near lower)' if bb_now<0.2 else '(mid-band)':<28s}|")
    print(f"  |  EMA Cross:    {ema_cross:>+10.6f}  {'(bullish)' if ema_cross>0 else '(bearish)':<20s}|")
    print(f"  |  Fib Level:    {fib_now:>6.2f}  {'(near high)' if fib_now>0.786 else '(near 61.8%)' if fib_now>0.618 else '(mid)' if fib_now>0.382 else '(near low)':<28s}|")
    print(f"  +--------------------------------------------------------------+")
    print()
    print(f"  ACTION: {row['Action']}")
    print()

    # Last 8 signals
    all_sigs = [(t, p, "BUY") for t, p in buys] + [(t, p, "SELL") for t, p in sells]
    all_sigs.sort(key=lambda x: x[0])
    recent = all_sigs[-8:]
    if recent:
        print("  RECENT SIGNALS:")
        for t, p, sig in recent:
            arrow = ">>>" if sig == "BUY" else "<<<"
            print(f"     {str(t)[:16]}  {arrow} {sig:4s}  @ ${p:>10,.2f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  9. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════
def plot_dashboard(df, summary, buys, sells):
    print("[6/6] Generating chart ...")

    colour_map = {s: summary.loc[s, "Colour"] for s in summary.index}
    tag_map    = {s: summary.loc[s, "Tag"]    for s in summary.index}

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(24, 13))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[5, 1.5, 1], hspace=0.06)

    # ── Panel 1: Price + regime bands + BUY/SELL ────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    # Background bands
    prev_s = df["State"].iloc[0]
    bstart = df.index[0]
    for i in range(1, len(df)):
        c = df["State"].iloc[i]
        if c != prev_s or i == len(df) - 1:
            ax1.axvspan(bstart, df.index[i],
                        color=colour_map.get(prev_s, "#444"), alpha=0.15, lw=0)
            bstart = df.index[i]
            prev_s = c

    # Price
    ax1.plot(df.index, df["Close"], color="white", lw=1.0, alpha=0.92)

    # BUY / SELL arrows
    if buys:
        bt, bp = zip(*buys)
        ax1.scatter(bt, bp, marker="^", c="#00e676", s=80, zorder=5,
                    edgecolors="white", lw=0.5, label=f"BUY ({len(buys)})")
    if sells:
        st, sp = zip(*sells)
        ax1.scatter(st, sp, marker="v", c="#ef5350", s=80, zorder=5,
                    edgecolors="white", lw=0.5, label=f"SELL ({len(sells)})")

    # Legend colours
    for state in sorted(summary.index):
        ax1.scatter([], [], c=colour_map[state], s=100, marker="s",
                    label=tag_map[state], edgecolors="none")

    # NOW marker
    ax1.axhline(df["Close"].iloc[-1], color="#666", ls="--", lw=0.5, alpha=0.4)
    ax1.annotate(f"  NOW  ${df['Close'].iloc[-1]:,.0f}",
                 xy=(df.index[-1], df["Close"].iloc[-1]),
                 fontsize=11, fontweight="bold", color="#00e676",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#1e1e1e",
                           ec="#00e676", lw=1.2))

    ax1.set_title(f"{CFG.TICKER} ({CFG.INTERVAL})  --  HMM + Technical Indicators  --  BUY / SELL",
                  fontsize=17, fontweight="bold", color="white", pad=14)
    ax1.set_ylabel("Price (USD)", fontsize=12, color="#ccc")
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.8,
              facecolor="#1e1e1e", edgecolor="#555", ncol=1, labelcolor="white")
    ax1.grid(color="#333", alpha=0.3, lw=0.5)
    ax1.tick_params(colors="#aaa")
    ax1.set_xticklabels([])

    # ── Panel 2: RSI + overbought/oversold bands ───────────────────────────
    ax_rsi = fig.add_subplot(gs[1], sharex=ax1)
    ax_rsi.plot(df.index, df["RSI_norm"] * 100, color="#ab47bc", lw=0.8, alpha=0.9)
    ax_rsi.axhline(70, color="#ef5350", ls="--", lw=0.6, alpha=0.6)
    ax_rsi.axhline(30, color="#00e676", ls="--", lw=0.6, alpha=0.6)
    ax_rsi.fill_between(df.index, 70, 100, color="#ef5350", alpha=0.08)
    ax_rsi.fill_between(df.index, 0, 30, color="#00e676", alpha=0.08)
    ax_rsi.set_ylabel("RSI", fontsize=10, color="#ccc")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(color="#333", alpha=0.2, lw=0.5)
    ax_rsi.tick_params(colors="#aaa")
    ax_rsi.set_xticklabels([])

    # ── Panel 3: Regime timeline ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    prev_s = df["State"].iloc[0]
    bstart = df.index[0]
    for i in range(1, len(df)):
        c = df["State"].iloc[i]
        if c != prev_s or i == len(df) - 1:
            ax2.axvspan(bstart, df.index[i],
                        color=colour_map.get(prev_s, "#444"), alpha=0.85, lw=0)
            bstart = df.index[i]
            prev_s = c

    ax2.set_ylabel("Regime", fontsize=10, color="#ccc")
    ax2.set_yticks([])
    ax2.set_xlabel("Date", fontsize=12, color="#ccc")
    ax2.tick_params(colors="#aaa")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.grid(False)

    fig.patch.set_facecolor("#121212")
    for a in [ax1, ax_rsi, ax2]:
        a.set_facecolor("#1a1a1a")
    fig.tight_layout()

    ticker_clean = CFG.TICKER.replace("-", "_").lower()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"{ticker_clean}_regimes.png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"      Saved: {out}")

    plt.show()
    plt.style.use("default")


# ═══════════════════════════════════════════════════════════════════════════
#  10. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
def print_summary(summary):
    cols = ["Tag", "Count", "Pct_Time", "Ann_Return", "Ann_Vol"]
    print()
    print("=" * 65)
    print("  DETECTED REGIMES  (bearish -> bullish)")
    print("=" * 65)
    print(summary[cols].to_string(float_format="{:.2f}".format))
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="HMM Market Regime Detector")
    parser.add_argument("--ticker", default="BTC-USD",
                        help="Asset ticker symbol (default: BTC-USD). Examples: ETH-USD, AAPL, TSLA, MSFT")
    parser.add_argument("--interval", default="1h",
                        choices=["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"],
                        help="Timeframe interval (default: 1h)")
    parser.add_argument("--days", type=int, default=None,
                        help="How many days of history to download (auto-limited per interval)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run grid-search parameter optimisation first")
    args = parser.parse_args()

    # Apply CLI args to config
    CFG.TICKER   = args.ticker.upper()
    CFG.INTERVAL = args.interval
    if args.days:
        CFG.TOTAL_DAYS = args.days

    print(f"  Asset: {CFG.TICKER}  |  Interval: {CFG.INTERVAL}  |  History: {CFG.TOTAL_DAYS}d")
    print()

    # Download once (reused for optimisation)
    raw = download_data()

    if args.optimize:
        optimize(raw)
        print("\n  Re-running with optimal parameters ...\n")

    # Full pipeline with current CFG
    df = engineer_features(raw)
    df = fit_hmm(df)
    summary = build_summary(df)
    print_summary(summary)

    buys, sells = find_transitions(df)
    backtest(df, buys, sells)
    diagnose(df, summary, buys, sells)
    plot_dashboard(df, summary, buys, sells)


if __name__ == "__main__":
    main()
