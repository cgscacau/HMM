import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

################################################################################
# SUPPORT / RESISTANCE (SR) ENGINE v1.1
# Detects pivot points, clusters them into zones, and validates via walk-forward.
################################################################################

def detect_pivots(df, window=10):
    """
    Identifies swing highs and swing lows using a rolling window.
    """
    pivots = []
    for i in range(window, len(df) - window):
        # High Check
        is_high = True
        for j in range(i - window, i + window + 1):
            if df['High'].iloc[j] > df['High'].iloc[i]:
                is_high = False
                break
        if is_high:
            pivots.append({'time': df.index[i], 'price': df['High'].iloc[i], 'type': 'RESISTANCE'})

        # Low Check
        is_low = True
        for j in range(i - window, i + window + 1):
            if df['Low'].iloc[j] < df['Low'].iloc[i]:
                is_low = False
                break
        if is_low:
            pivots.append({'time': df.index[i], 'price': df['Low'].iloc[i], 'type': 'SUPPORT'})
            
    return pivots

def cluster_levels(pivots, cluster_pct=1.5):
    """
    Clusters individual pivot points into merged price zones.
    """
    if not pivots: return []
    
    # Sort by price
    pdf = pd.DataFrame(pivots).sort_values('price')
    zones = []
    
    if pdf.empty: return []
    
    current_zone = [pdf.iloc[0].to_dict()]
    
    for i in range(1, len(pdf)):
        prev_p = pdf.iloc[i-1]['price']
        curr_p = pdf.iloc[i]['price']
        
        # If within % threshold, add to current zone
        if (curr_p - prev_p) / prev_p * 100 <= cluster_pct:
            current_zone.append(pdf.iloc[i].to_dict())
        else:
            # Finalize current zone, start new one
            avg_price = sum(z['price'] for z in current_zone) / len(current_zone)
            zones.append({
                'level': avg_price,
                'type': current_zone[0]['type'], # Inherit from first pivot
                'touch_count': len(current_zone),
                'pivots': current_zone
            })
            current_zone = [pdf.iloc[i].to_dict()]
            
    # Add last zone
    if current_zone:
        avg_price = sum(z['price'] for z in current_zone) / len(current_zone)
        zones.append({
            'level': avg_price,
            'type': current_zone[0]['type'],
            'touch_count': len(current_zone),
            'pivots': current_zone
        })
        
    return zones

def analyse_zone_interactions(df, zones, cluster_pct=1.0):
    """
    Analyzes how price interacted with zones: bounces vs breaks.
    """
    if not zones: return []
    
    final_zones = []
    for z in zones:
        lvl = z['level']
        tol = lvl * (cluster_pct / 100.0)
        
        touches = 0
        bounces = 0
        breaks = 0
        total_volume = 0.0
        
        # Very simplified interaction check
        for i in range(1, len(df)):
            low = df['Low'].iloc[i]
            high = df['High'].iloc[i]
            close = df['Close'].iloc[i]
            prev_close = df['Close'].iloc[i-1]
            vol = df['Volume'].iloc[i] if 'Volume' in df.columns else 0.0
            
            # Did price enter the zone?
            if low <= lvl + tol and high >= lvl - tol:
                touches += 1
                total_volume += vol
                # Did it reverse (bounce)? 
                # (Simple logic: if previous close was outside and current close is still outside)
                if (prev_close > lvl + tol and close > lvl + tol) or \
                   (prev_close < lvl - tol and close < lvl - tol):
                    bounces += 1
                else:
                    breaks += 1
                    
        hold_rate = (bounces / touches * 100) if touches > 0 else 0
        
        # Append updated stats
        nz = z.copy()
        nz['hold_rate'] = hold_rate
        nz['break_rate'] = 100 - hold_rate
        nz['touch_count'] = touches
        nz['trade_score'] = (hold_rate * z['touch_count']) / 10.0 # Heuristic
        nz['volume'] = total_volume
        nz['is_reliable'] = touches >= 3 and hold_rate >= 60.0
        final_zones.append(nz)
        
    return final_zones

def full_sr_analysis(df, pivot_window=10, cluster_pct=1.5):
    """
    Complete pipeline for a single timeframe.
    """
    pivots = detect_pivots(df, window=pivot_window)
    zones = cluster_levels(pivots, cluster_pct=cluster_pct)
    zones = analyse_zone_interactions(df, zones, cluster_pct=cluster_pct)
    
    # Simple assessment
    curr_p = df['Close'].iloc[-1]
    sorted_zones = sorted(zones, key=lambda x: x['level'])
    
    support = [z for z in sorted_zones if z['level'] < curr_p]
    resistance = [z for z in sorted_zones if z['level'] > curr_p]
    
    # Position scoring
    ns = support[-1] if support else None
    nr = resistance[0] if resistance else None
    
    rr = 1.0
    if ns and nr:
        dist_sup = curr_p - ns['level']
        dist_res = nr['level'] - curr_p
        rr = dist_res / dist_sup if dist_sup > 0 else 1.0
        
    # Trade Probability: Weighted average of hold rates
    tp = 50.0
    if ns and nr:
        tp = (ns['hold_rate'] + nr['hold_rate']) / 2.0
    elif ns:
        tp = ns['hold_rate']
    elif nr:
        tp = nr['hold_rate']

    assessment = {
        'nearest_support': ns,
        'nearest_resistance': nr,
        'rr_ratio': rr,
        'trade_probability': tp
    }
    
    return zones, assessment

def walk_forward(df, train_pct=0.7, progress_cb=None):
    """
    Optimizes pivot_window and cluster_pct using Out-of-Sample metrics.
    """
    split_idx = int(len(df) * train_pct)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    results = []
    
    # Param Grid
    windows = [5, 10, 15, 20]
    clusters = [0.5, 1.0, 1.5, 2.0]
    
    total = len(windows) * len(clusters)
    count = 0
    
    for w in windows:
        for c in clusters:
            count += 1
            if progress_cb:
                progress_cb(count/total, f"Testing window={w}, cluster={c}%...")
            
            # 1. Detect on Train
            p_train = detect_pivots(train_df, window=w)
            z_train = cluster_levels(p_train, cluster_pct=c)
            z_train = analyse_zone_interactions(train_df, z_train, cluster_pct=c)
            
            # 2. Validate on Test (how many "touches" in test set actually held?)
            z_test = analyse_zone_interactions(test_df, z_train, cluster_pct=c)
            
            avg_hold_in = sum(z['hold_rate'] for z in z_train) / len(z_train) if z_train else 0
            avg_hold_out = sum(z['hold_rate'] for z in z_test) / len(z_test) if z_test else 0
            
            results.append({
                'pivot_window': w,
                'cluster_pct': c,
                'in_sample_hold': avg_hold_in,
                'out_sample_hold': avg_hold_out,
                'out_trades': sum(z['touch_count'] for z in z_test),
                'out_win_rate': avg_hold_out,
                'out_total_ret': (avg_hold_out - 50) * 0.1, # Dummy metric
                'out_sharpe': avg_hold_out / 50.0, # Dummy metric
                'zones_train': z_train,
                'zones_test': z_test
            })
            
    rdf = pd.DataFrame(results).sort_values('out_sample_hold', ascending=False)
    best = rdf.iloc[0].to_dict() if not rdf.empty else None
    
    return rdf, best, split_idx

def build_sr_chart(df, zones, ticker, interval, cluster_pct=1.0, split_idx=None):
    """
    Builds a Plotly chart with horizontal S/R bands.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Support/Resistance Analysis', 'Volume'), 
                        row_width=[0.2, 0.7])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)

    # Zones (Horizontal Rectangles)
    x_start = df.index[0]
    x_end = df.index[-1]
    
    for z in zones:
        lvl = z['level']
        # Match cluster_pct for height (tol = lvl * cluster_pct / 100)
        h = lvl * (cluster_pct / 200.0) # half on each side
        color = "rgba(0, 230, 118, 0.25)" if z['type'] == 'SUPPORT' else "rgba(239, 83, 80, 0.25)"
        
        fig.add_shape(type="rect",
                      x0=x_start, y0=lvl-h, x1=x_end, y1=lvl+h,
                      fillcolor=color, line_width=0, layer="below", row=1, col=1)
        
        # Add Label (Probability/Hold Rate + Volume + Touches)
        if z['touch_count'] > 0:
            vol = z.get('volume', 0)
            if vol >= 1_000_000_000:
                vol_str = f"{vol/1_000_000_000:.1f}B"
            elif vol >= 1_000_000:
                vol_str = f"{vol/1_000_000:.1f}M"
            elif vol >= 1_000:
                vol_str = f"{vol/1_000:.1f}k"
            else:
                vol_str = f"{vol:.0f}"
                
            star = "⭐ " if z.get('is_reliable', False) else ""
            label_text = f"<b>{star}{lvl:.2f} | {z['hold_rate']:.0f}%</b><br><span style='font-size:10px;'>{z['touch_count']} Toques | Vol: {vol_str}</span>"

            fig.add_annotation(
                x=x_end, y=lvl,
                text=label_text,
                showarrow=False,
                xanchor="left",
                font=dict(size=12, color="#00e676" if z['type']=='SUPPORT' else "#ef5350"),
                bgcolor="rgba(0,0,0,0.6)",
                row=1, col=1
            )
        
    # Split line
    if split_idx is not None and 0 <= split_idx < len(df):
        split_val = df.index[split_idx]
        # Use Unix timestamp (ms) to avoid Plotly's internal sum(Timestamps) error
        x_coord = split_val.timestamp() * 1000 if hasattr(split_val, 'timestamp') else split_val
        fig.add_vline(x=x_coord, line_dash="dash", line_color="orange", 
                      annotation_text="Walk-Forward Split", row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='gray'), row=2, col=1)

    fig.update_layout(height=800, template='plotly_dark', showlegend=False, xaxis_rangeslider_visible=False)
    return fig
