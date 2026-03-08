import pandas as pd
import numpy as np

################################################################################
# STRATEGY RECOMMENDER (SR) ENGINE v2.0
# Contains the mathematical logic for 15 quantitative options strategies.
# Driven by Random Forest Direction/Confidence and Implied Volatility (Vega) edge.
################################################################################

def format_leg(opt_row, fallback_text):
    if opt_row is None: return fallback_text
    try:
        strike = float(opt_row['Strike'])
        delta = float(opt_row['Delta'])
        return f"**{opt_row['Ativo']}** (Strike R${strike:.2f} | Delta {delta:.2f})"
    except:
        return f"**{opt_row['Ativo']}**"

def find_best_option(df, tipo, target_delta, exclude_tickers=None):
    """
    Finds the closest exact Delta option ticker available in the dataframe.
    """
    if exclude_tickers is None: exclude_tickers = []
    try:
        # Support both numeric (float/NaN) and legacy string ('--') Delta columns
        valid = df[df['Tipo'] == tipo].copy()
        valid['Delta'] = pd.to_numeric(valid['Delta'], errors='coerce')
        valid = valid.dropna(subset=['Delta'])
        if len(exclude_tickers) > 0:
            valid = valid[~valid['Ativo'].isin(exclude_tickers)]
        if valid.empty: return None
        valid['Delta_Diff'] = abs(valid['Delta'] - target_delta)
        closest = valid.sort_values('Delta_Diff').iloc[0]
        return closest
    except Exception:
        return None

def find_closest_strike(df, tipo, target_strike, exclude_tickers=None):
    """
    Finds the option closest to a specific Strike (for calendars/diagonals).
    """
    if exclude_tickers is None: exclude_tickers = []
    try:
        # Support both numeric (float/NaN) and legacy string ('--') Strike columns
        valid = df[df['Tipo'] == tipo].copy()
        valid['Strike'] = pd.to_numeric(valid['Strike'], errors='coerce')
        valid = valid.dropna(subset=['Strike'])
        if len(exclude_tickers) > 0:
            valid = valid[~valid['Ativo'].isin(exclude_tickers)]
        if valid.empty: return None
        valid['Strike_Diff'] = abs(valid['Strike'] - target_strike)
        closest = valid.sort_values('Strike_Diff').iloc[0]
        return closest
    except Exception:
        return None

def calculate_strategy_score(strat, rf_dir, rf_conf, iv_proxy):
    """
    Scoring algorithm to rank strategies out of 100 max.
    """
    score = 50 # Base score
    
    # 1. Directional Alignment
    if strat['direction'] == 'BULLISH' and rf_dir == 1:
        score += (rf_conf * 30)
    elif strat['direction'] == 'BEARISH' and rf_dir == -1:
        score += (rf_conf * 30)
    elif strat['direction'] == 'NEUTRAL':
        # Neutral strategies prefer low confidences (consolidation)
        if rf_conf < 0.55:
            score += 30
        else:
            score -= 20
    else:
        score -= 20 # Fighting the AI trend
        
    # 2. Volatility (Vega) Alignment
    # IV proxy: greater than 0.35 is considered HIGH volatility
    high_iv = iv_proxy > 0.35 
    
    if strat['vega'] == 'NEGATIVE' and high_iv:
        score += 15 # Sell premium when IV is high
    elif strat['vega'] == 'POSITIVE' and not high_iv:
        score += 15 # Buy premium when IV is low
    else:
        score -= 5
        
    return score


def rank_and_select_strategy(df_opcoes, pred, prob_up, current_price):
    """
    Master function: Parses Dataframe, attempts to build distinct archetypes,
    scores them, and returns the highest scoring valid Strategy Object.
    """
    if df_opcoes is None or df_opcoes.empty:
        return None
        
    ai_confidence = prob_up if pred == 1 else 1 - prob_up
    rf_dir = 1 if pred == 1 else -1
    
    try:
        valid_ivs = df_opcoes[df_opcoes['Vol_Impl'] != '--']['Vol_Impl']
        valid_ivs = pd.to_numeric(valid_ivs.str.replace('%',''), errors='coerce') / 100
        median_iv = valid_ivs.median() if not valid_ivs.empty else 0.30
    except:
        median_iv = 0.30

    strategies = []

    # =========================================================================
    # DIRECIONAIS BÁSICOS
    # =========================================================================
    # 1. Long Call
    l1 = find_best_option(df_opcoes, 'CALL', 0.50)
    if l1 is not None:
        strategies.append({
            'name': 'Compra Seca de Call (Long Call)',
            'desc': 'Estratégia direcional bruta. Aposta alta e ilimitada na subida do papel limitando o prejuízo ao capital gasto.',
            'direction': 'BULLISH', 'vega': 'POSITIVE', 'legs': 1,
            'steps': [f"1. **COMPRAR** {format_leg(l1, 'Call')} — Aposta Direta"],
            'plot_legs': [{'type': 'BUY_CALL', 'strike': float(l1['Strike'])}]
        })
        
    # 2. Long Put
    l1 = find_best_option(df_opcoes, 'PUT', -0.50)
    if l1 is not None:
        strategies.append({
            'name': 'Compra Seca de Put (Long Put)',
            'desc': 'Aposta direcional pura na queda livre. Retorno excelente quando o terror reina no mercado.',
            'direction': 'BEARISH', 'vega': 'POSITIVE', 'legs': 1,
            'steps': [f"1. **COMPRAR** {format_leg(l1, 'Put')} — Aposta de Queda"],
            'plot_legs': [{'type': 'BUY_PUT', 'strike': float(l1['Strike'])}]
        })

    # =========================================================================
    # TRAVAS (SPREADS)
    # =========================================================================
    # 3. Bull Call Spread
    l1 = find_best_option(df_opcoes, 'CALL', 0.60)
    t1 = l1['Ativo'] if l1 is not None else None
    l2 = find_best_option(df_opcoes, 'CALL', 0.30, exclude_tickers=[t1])
    if l1 is not None and l2 is not None:
        strategies.append({
            'name': 'Trava de Alta com Calls (Debit Spread)',
            'desc': 'Ganha na alta da ação, mas reduz drasticamente o custo de entrada vendendo o teto da subida.',
            'direction': 'BULLISH', 'vega': 'POSITIVE', 'legs': 2,
            'steps': [
                f"1. **COMPRAR** {format_leg(l1, 'Call')}",
                f"2. **VENDER** {format_leg(l2, 'Call')}"
            ],
            'plot_legs': [{'type': 'BUY_CALL', 'strike': float(l1['Strike'])}, {'type': 'SELL_CALL', 'strike': float(l2['Strike'])}]
        })
        
    # 4. Bear Put Spread
    l1 = find_best_option(df_opcoes, 'PUT', -0.60)
    t1 = l1['Ativo'] if l1 is not None else None
    l2 = find_best_option(df_opcoes, 'PUT', -0.30, exclude_tickers=[t1])
    if l1 is not None and l2 is not None:
        strategies.append({
            'name': 'Trava de Baixa com Puts (Debit Spread)',
            'desc': 'Aposta na queda mas abaixa o custo do seu seguro de put vendendo outra put de strike inferior.',
            'direction': 'BEARISH', 'vega': 'POSITIVE', 'legs': 2,
            'steps': [
                f"1. **COMPRAR** {format_leg(l1, 'Put')}",
                f"2. **VENDER** {format_leg(l2, 'Put')}"
            ],
            'plot_legs': [{'type': 'BUY_PUT', 'strike': float(l1['Strike'])}, {'type': 'SELL_PUT', 'strike': float(l2['Strike'])}]
        })
        
    # 5. Bull Put Spread 
    l1 = find_best_option(df_opcoes, 'PUT', -0.30)
    t1 = l1['Ativo'] if l1 is not None else None
    l2 = find_best_option(df_opcoes, 'PUT', -0.15, exclude_tickers=[t1])
    if l1 is not None and l2 is not None:
        strategies.append({
            'name': 'Trava de Alta com Puts (Credit Spread)',
            'desc': 'Estratégia de renda. Vender opções recolhendo prêmio na conta hoje confiando que a ação vai subir ou pelo menos lateralizar.',
            'direction': 'BULLISH', 'vega': 'NEGATIVE', 'legs': 2,
            'steps': [
                f"1. **VENDER** {format_leg(l1, 'Put')}",
                f"2. **COMPRAR** {format_leg(l2, 'Put OTM')}"
            ],
            'plot_legs': [{'type': 'SELL_PUT', 'strike': float(l1['Strike'])}, {'type': 'BUY_PUT', 'strike': float(l2['Strike'])}]
        })
        
    # 6. Bear Call Spread
    l1 = find_best_option(df_opcoes, 'CALL', 0.30)
    t1 = l1['Ativo'] if l1 is not None else None
    l2 = find_best_option(df_opcoes, 'CALL', 0.15, exclude_tickers=[t1])
    if l1 is not None and l2 is not None:
        strategies.append({
            'name': 'Trava de Baixa com Calls (Credit Spread)',
            'desc': 'Vender teto: Recolhe prêmio acreditando que a ação NÃO vai subir além de certo patamar.',
            'direction': 'BEARISH', 'vega': 'NEGATIVE', 'legs': 2,
            'steps': [
                f"1. **VENDER** {format_leg(l1, 'Call')}",
                f"2. **COMPRAR** {format_leg(l2, 'Call OTM')}"
            ],
            'plot_legs': [{'type': 'SELL_CALL', 'strike': float(l1['Strike'])}, {'type': 'BUY_CALL', 'strike': float(l2['Strike'])}]
        })

    # =========================================================================
    # VOLATILIDADE EXTREMA E ESTRATÉGIAS NEUTRAS EXÓTICAS
    # =========================================================================
    # 7. Straddle
    l1 = find_best_option(df_opcoes, 'CALL', 0.50)
    l2 = find_closest_strike(df_opcoes, 'PUT', float(l1['Strike']) if l1 is not None else current_price)
    if l1 is not None and l2 is not None:
        strategies.append({
            'name': 'Straddle In-The-Money',
            'desc': 'Aposta que o papel vai EXPLODIR, para cima ou para baixo. Estratégia cara focada em volatilidade bruta.',
            'direction': 'NEUTRAL', 'vega': 'POSITIVE', 'legs': 2,
            'steps': [
                f"1. **COMPRAR** {format_leg(l1, 'Call')} (Perna Alta)",
                f"2. **COMPRAR** {format_leg(l2, 'Put')} (Perna Baixa)"
            ],
            'plot_legs': [{'type': 'BUY_CALL', 'strike': float(l1['Strike'])}, {'type': 'BUY_PUT', 'strike': float(l2['Strike'])}]
        })
        
    # 8. Strangle
    l1 = find_best_option(df_opcoes, 'CALL', 0.25)
    l2 = find_best_option(df_opcoes, 'PUT', -0.25)
    if l1 is not None and l2 is not None:
        strategies.append({
            'name': 'Strangle Direcional OTM',
            'desc': 'Versão distanciada do Straddle. Compra proteção violenta de caudas em cenários extremos (ex: véspera de balanço).',
            'direction': 'NEUTRAL', 'vega': 'POSITIVE', 'legs': 2,
            'steps': [
                f"1. **COMPRAR** {format_leg(l1, 'Call OTM')}",
                f"2. **COMPRAR** {format_leg(l2, 'Put OTM')}"
            ],
            'plot_legs': [{'type': 'BUY_CALL', 'strike': float(l1['Strike'])}, {'type': 'BUY_PUT', 'strike': float(l2['Strike'])}]
        })
        
    # 9. Iron Condor
    lc_sell = find_best_option(df_opcoes, 'CALL', 0.20)
    t_cs = lc_sell['Ativo'] if lc_sell is not None else None
    lc_buy = find_best_option(df_opcoes, 'CALL', 0.05, exclude_tickers=[t_cs])
    
    lp_sell = find_best_option(df_opcoes, 'PUT', -0.20)
    t_ps = lp_sell['Ativo'] if lp_sell is not None else None
    lp_buy = find_best_option(df_opcoes, 'PUT', -0.05, exclude_tickers=[t_ps])
    
    if all(x is not None for x in [lc_sell, lc_buy, lp_sell, lp_buy]):
         strategies.append({
            'name': 'Condor de Ferro (Iron Condor)',
            'desc': 'Apostas na LETARGIA do papel. Lucra quando tudo fica chato de lado consumindo prêmios das pontas.',
            'direction': 'NEUTRAL', 'vega': 'NEGATIVE', 'legs': 4,
            'steps': [
                f"1. **VENDER** {format_leg(lc_sell, 'Call')} e **COMPRAR** {format_leg(lc_buy, 'Call Proteção')}",
                f"2. **VENDER** {format_leg(lp_sell, 'Put')} e **COMPRAR** {format_leg(lp_buy, 'Put Proteção')}"
            ],
            'plot_legs': [
                {'type': 'SELL_CALL', 'strike': float(lc_sell['Strike'])}, {'type': 'BUY_CALL', 'strike': float(lc_buy['Strike'])},
                {'type': 'SELL_PUT', 'strike': float(lp_sell['Strike'])}, {'type': 'BUY_PUT', 'strike': float(lp_buy['Strike'])}
            ]
         })

    # 10. Iron Butterfly
    l_mid = find_best_option(df_opcoes, 'CALL', 0.50)
    if l_mid is not None:
        t_mid = l_mid['Ativo']
        l_c_buy = find_best_option(df_opcoes, 'CALL', 0.15, exclude_tickers=[t_mid])
        l_p_buy = find_best_option(df_opcoes, 'PUT', -0.15)
        l_p_sell = find_closest_strike(df_opcoes, 'PUT', float(l_mid['Strike']), exclude_tickers=[l_p_buy['Ativo'] if l_p_buy is not None else None])
        
        if all(x is not None for x in [l_mid, l_c_buy, l_p_buy, l_p_sell]):
            strategies.append({
                'name': 'Borboleta de Ferro (Iron Butterfly)',
                'desc': 'Versão agressiva do Condor. Vende o straddle no miolo e compra as asas. Lucro máximo se o papel fechar exatamente no strike central.',
                'direction': 'NEUTRAL', 'vega': 'NEGATIVE', 'legs': 4,
                'steps': [
                    f"1. **VENDER** Call @{l_mid['Strike']} e Put @{l_p_sell['Strike']}",
                    f"2. **COMPRAR** Call @{l_c_buy['Strike']} e Put @{l_p_buy['Strike']} (Proteção)"
                ],
                'plot_legs': [
                    {'type': 'SELL_CALL', 'strike': float(l_mid['Strike'])}, {'type': 'BUY_CALL', 'strike': float(l_c_buy['Strike'])},
                    {'type': 'SELL_PUT', 'strike': float(l_p_sell['Strike'])}, {'type': 'BUY_PUT', 'strike': float(l_p_buy['Strike'])}
                ]
            })

    # =========================================================================
    # ASSIMÉTRICAS E ALICERÇADAS
    # =========================================================================
    # 10. Call Ratio Spread
    l1 = find_best_option(df_opcoes, 'CALL', 0.50)
    t1 = l1['Ativo'] if l1 is not None else None
    l2 = find_best_option(df_opcoes, 'CALL', 0.25, exclude_tickers=[t1])
    if l1 is not None and l2 is not None:
        strategies.append({
            'name': 'Ratio Spread de Call',
            'desc': 'Usa alavancagem de 2 para 1 para entrar comprado em Alta a um custo quase nulo, gerando risco ilimitado se o papel subir DEMAIS.',
            'direction': 'BULLISH', 'vega': 'NEGATIVE', 'legs': 2,
            'steps': [
                f"1. **COMPRAR 1X** {format_leg(l1, 'Call')}",
                f"2. **VENDER 2X** {format_leg(l2, 'Call')}"
            ],
            'plot_legs': [{'type': 'BUY_CALL', 'strike': float(l1['Strike'])}, {'type': 'SELL_CALL', 'strike': float(l2['Strike'])}, {'type': 'SELL_CALL', 'strike': float(l2['Strike'])}]
        })
        
    # 11. Put Ratio Spread
    l1 = find_best_option(df_opcoes, 'PUT', -0.50)
    t1 = l1['Ativo'] if l1 is not None else None
    l2 = find_best_option(df_opcoes, 'PUT', -0.25, exclude_tickers=[t1])
    if l1 is not None and l2 is not None:
        strategies.append({
            'name': 'Ratio Spread de Put',
            'desc': 'Aposta direcional na queda barata, mas muito cuidado pois o risco passa a ser ser acionado na parte inferior comprando as ações.',
            'direction': 'BEARISH', 'vega': 'NEGATIVE', 'legs': 2,
            'steps': [
                f"1. **COMPRAR 1X** {format_leg(l1, 'Put')}",
                f"2. **VENDER 2X** {format_leg(l2, 'Put')}"
            ],
            'plot_legs': [{'type': 'BUY_PUT', 'strike': float(l1['Strike'])}, {'type': 'SELL_PUT', 'strike': float(l2['Strike'])}, {'type': 'SELL_PUT', 'strike': float(l2['Strike'])}]
        })

    # 12. Broken Wing Butterfly (Call)
    l1 = find_best_option(df_opcoes, 'CALL', 0.60)
    l2 = find_best_option(df_opcoes, 'CALL', 0.40, exclude_tickers=[l1['Ativo'] if l1 is not None else None])
    l3 = find_best_option(df_opcoes, 'CALL', 0.10, exclude_tickers=[l1['Ativo'] if l1 is not None else None, l2['Ativo'] if l2 is not None else None])
    if all(x is not None for x in [l1, l2, l3]):
        strategies.append({
            'name': 'Borboleta de Asa Quebrada (Broken Wing)',
            'desc': 'Estratégia de crédito que lucra se o papel subir, mas protege contra quedas fortes. Uma asa é mais longa que a outra criando assimetria favorável.',
            'direction': 'BULLISH', 'vega': 'NEGATIVE', 'legs': 3,
            'steps': [
                f"1. **COMPRAR 1X** {format_leg(l1, 'Call ITM')}",
                f"2. **VENDER 2X** {format_leg(l2, 'Call ATM')}",
                f"3. **COMPRAR 1X** {format_leg(l3, 'Call OTM Distante')}"
            ],
            'plot_legs': [
                {'type': 'BUY_CALL', 'strike': float(l1['Strike'])},
                {'type': 'SELL_CALL', 'strike': float(l2['Strike'])}, {'type': 'SELL_CALL', 'strike': float(l2['Strike'])},
                {'type': 'BUY_CALL', 'strike': float(l3['Strike'])}
            ]
        })

    # 12. Covered Call
    l1 = find_best_option(df_opcoes, 'CALL', 0.25)
    if l1 is not None:
        strategies.append({
            'name': 'Venda Coberta (Covered Call)',
            'desc': 'Venda Coberta pura de longo prazo contra a ação-base. O famoso "Dividendo Sintético".',
            'direction': 'BULLISH', 'vega': 'NEGATIVE', 'legs': 1,
            'steps': [
                f"Pré-Requisito: Ter as 100 ações em custódia",
                f"1. **VENDER** {format_leg(l1, 'Call')}"
            ],
            'plot_legs': [{'type': 'COVERED_CALL', 'strike': float(l1['Strike'])}]
        })
        
    # 13. Cash Secured Put
    l1 = find_best_option(df_opcoes, 'PUT', -0.25)
    if l1 is not None:
         strategies.append({
            'name': 'Venda de Put Coberta ao Dinheiro (CSP)',
            'desc': 'Estratégia Warren Buffet: Receber prêmio imediato jurando comprar a ação por um valor menor amanhã caso ela caia.',
            'direction': 'BULLISH', 'vega': 'NEGATIVE', 'legs': 1, # Bullish because you want to keep the premium
            'steps': [
                f"1. Imobilize financeiro na corretora => 100 * (Strike)",
                f"2. **VENDER** {format_leg(l1, 'Put')}" 
            ],
            'plot_legs': [{'type': 'SELL_PUT', 'strike': float(l1['Strike'])}]
        })
         
    # 14. Collar
    l_call = find_best_option(df_opcoes, 'CALL', 0.20)
    l_put = find_best_option(df_opcoes, 'PUT', -0.30)
    if l_call is not None and l_put is not None:
         strategies.append({
            'name': 'Cerca ou Colar com Ações (Collar)',
            'desc': 'Para alargar sua segurança. Venda teto de Call para financiar compra de seguro de Put e proteger suas Ações sem pagar nada!',
            'direction': 'NEUTRAL', 'vega': 'NEGATIVE', 'legs': 2,
            'steps': [
                f"1. **VENDER** {format_leg(l_call, 'Call')} (Teto de Alta)",
                f"2. **COMPRAR** {format_leg(l_put, 'Put')} (Seguro Contra Quedas)"
            ],
             # A collar natively assumes stock ownership starting at current price
            'plot_legs': [{'type': 'BUY_PUT', 'strike': float(l_put['Strike'])}, {'type': 'SELL_CALL_COLLAR', 'strike': float(l_call['Strike'])}]
        })
        

    # -------------------------------------------------------------
    # Ranking Engine execution
    # -------------------------------------------------------------
    for strat in strategies:
        score = calculate_strategy_score(strat, rf_dir, ai_confidence, median_iv)
        strat['final_score'] = score
        
    strategies.sort(key=lambda x: x['final_score'], reverse=True)
    
    if strategies:
        best_strat = strategies[0]
        return {
            'strategy_name': best_strat['name'],
            'rationale': best_strat['desc'],
            'action_steps': best_strat['steps'],
            'plot_legs': best_strat['plot_legs'],
            'engine_score': best_strat['final_score']
        }
    
    return None


# ─── User-selectable strategies ────────────────────────────────────────────────
AVAILABLE_STRATEGIES = [
    "Long Call — Compra de Call (Alta Direcional)",
    "Long Put — Compra de Put (Baixa Direcional)",
    "Bull Call Spread — Trava de Alta com Calls",
    "Bear Put Spread — Trava de Baixa com Puts",
    "Bull Put Spread — Trava de Crédito na Alta",
    "Bear Call Spread — Trava de Crédito na Baixa",
    "Long Straddle — Compra de Volatilidade ATM",
    "Long Strangle — Compra de Volatilidade OTM",
    "Covered Call — Venda Coberta",
    "Protective Put — Proteção de Carteira",
    "Iron Condor — Acumulação em Range",
    "Butterfly — Ganho no Ponto ATM",
]


def build_specific_strategy(df, strategy_name, spot):
    """
    Build a user-selected options strategy.
    Finds the best available legs from `df` for the given strategy.
    Returns dict with: strategy_name, rationale, legs, plot_legs, explanation
    """
    if df is None or df.empty:
        return None

    df = df.copy()
    for col in ["Strike", "Delta", "Gama", "Vega", "Theta", "Vol_Impl", "Ultimo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def _fmt(opt):
        if opt is None:
            return "nao encontrada"
        try:
            ativo = opt["Ativo"]
            k = float(opt["Strike"])
            d = float(opt["Delta"])
            iv = float(opt["Vol_Impl"])
            ul = float(opt["Ultimo"])
            return f"{ativo} | K R${k:.2f} | Delta {d:.3f} | IV {iv:.1f}% | Ultimo R${ul:.2f}"
        except Exception:
            return str(opt.get("Ativo", ""))

    def ac(e=None):
        return find_best_option(df, "CALL",  0.50, e)

    def ap(e=None):
        return find_best_option(df, "PUT",  -0.50, e)

    def oc(d=0.30, e=None):
        return find_best_option(df, "CALL",  d, e)

    def op(d=-0.30, e=None):
        return find_best_option(df, "PUT",   d, e)

    def _e(opt):
        return [opt["Ativo"]] if opt is not None else []

    key = strategy_name.split("—")[0].strip()

    # ── Long Call ─────────────────────────────────────────────────────────────
    if key == "Long Call":
        g = ac()
        if g is None:
            return None
        k, p = float(g["Strike"]), float(g["Ultimo"])
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"Call ATM **{g['Ativo']}** (Strike R${k:.2f}, Delta {float(g['Delta']):.3f}) "
                f"selecionada por ter a maior sensibilidade direcional de alta disponível. "
                f"Premio pago: R${p:.2f}. IV: {float(g['Vol_Impl']):.1f}%."
            ),
            "legs": [{"role": "COMPRAR 1 CALL ATM", "detalhe": _fmt(g)}],
            "plot_legs": [{"type": "BUY_CALL", "strike": k}],
            "explanation": (
                f"Ganho ilimitado acima de R${k:.2f}. "
                f"Perda maxima: R${p:.2f} (premio pago). "
                f"Breakeven: R${k+p:.2f}."
            ),
        }

    # ── Long Put ──────────────────────────────────────────────────────────────
    elif key == "Long Put":
        g = ap()
        if g is None:
            return None
        k, p = float(g["Strike"]), float(g["Ultimo"])
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"Put ATM **{g['Ativo']}** (Strike R${k:.2f}, Delta {float(g['Delta']):.3f}) "
                f"oferece maxima sensibilidade na baixa. Premio: R${p:.2f}."
            ),
            "legs": [{"role": "COMPRAR 1 PUT ATM", "detalhe": _fmt(g)}],
            "plot_legs": [{"type": "BUY_PUT", "strike": k}],
            "explanation": (
                f"Ganho ilimitado abaixo de R${k:.2f}. "
                f"Perda maxima: R${p:.2f}. "
                f"Breakeven: R${k-p:.2f}."
            ),
        }

    # ── Bull Call Spread ───────────────────────────────────────────────────────
    elif key == "Bull Call Spread":
        b = ac()
        s = oc(0.30, _e(b))
        if b is None or s is None:
            return None
        kb, ks = float(b["Strike"]), float(s["Strike"])
        cost = round(float(b["Ultimo"]) - float(s["Ultimo"]), 2)
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"Compra call ATM **{b['Ativo']}** (K=R${kb:.2f}) e vende call OTM "
                f"**{s['Ativo']}** (K=R${ks:.2f}) para reduzir custo da operacao. "
                f"Custo liquido: R${cost:.2f}. Lucro maximo: R${ks-kb-cost:.2f}."
            ),
            "legs": [
                {"role": "COMPRAR 1 CALL ATM", "detalhe": _fmt(b)},
                {"role": "VENDER 1 CALL OTM",  "detalhe": _fmt(s)},
            ],
            "plot_legs": [{"type": "BUY_CALL", "strike": kb}, {"type": "SELL_CALL", "strike": ks}],
            "explanation": (
                f"Ganho max: R${ks-kb-cost:.2f} | Perda max: R${cost:.2f} | "
                f"Breakeven: R${kb+cost:.2f} | Zona lucro: R${kb:.2f}–R${ks:.2f}."
            ),
        }

    # ── Bear Put Spread ────────────────────────────────────────────────────────
    elif key == "Bear Put Spread":
        b = ap()
        s = op(-0.25, _e(b))
        if b is None or s is None:
            return None
        kb, ks = float(b["Strike"]), float(s["Strike"])
        cost = round(float(b["Ultimo"]) - float(s["Ultimo"]), 2)
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"Compra put ATM **{b['Ativo']}** (K=R${kb:.2f}) e vende put OTM "
                f"**{s['Ativo']}** (K=R${ks:.2f}) para reduzir custo. Custo: R${cost:.2f}."
            ),
            "legs": [
                {"role": "COMPRAR 1 PUT ATM", "detalhe": _fmt(b)},
                {"role": "VENDER 1 PUT OTM",  "detalhe": _fmt(s)},
            ],
            "plot_legs": [{"type": "BUY_PUT", "strike": kb}, {"type": "SELL_PUT", "strike": ks}],
            "explanation": (
                f"Ganho max: R${kb-ks-cost:.2f} | Perda max: R${cost:.2f} | "
                f"Breakeven: R${kb-cost:.2f}."
            ),
        }

    # ── Bull Put Spread ────────────────────────────────────────────────────────
    elif key == "Bull Put Spread":
        sv = ap()
        b = op(-0.15, _e(sv))
        if sv is None or b is None:
            return None
        ks, kb = float(sv["Strike"]), float(b["Strike"])
        cred = round(float(sv["Ultimo"]) - float(b["Ultimo"]), 2)
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"Vende put ATM **{sv['Ativo']}** (K=R${ks:.2f}) e compra put OTM "
                f"**{b['Ativo']}** (K=R${kb:.2f}) como protecao. "
                f"Credito recebido: R${cred:.2f}. Operacao de credito — lucra se ativo subir."
            ),
            "legs": [
                {"role": "VENDER 1 PUT ATM",  "detalhe": _fmt(sv)},
                {"role": "COMPRAR 1 PUT OTM", "detalhe": _fmt(b)},
            ],
            "plot_legs": [{"type": "SELL_PUT", "strike": ks}, {"type": "BUY_PUT", "strike": kb}],
            "explanation": (
                f"Ganho max (credito): R${cred:.2f} | Perda max: R${ks-kb-cred:.2f} | "
                f"Breakeven: R${ks-cred:.2f}."
            ),
        }

    # ── Bear Call Spread ───────────────────────────────────────────────────────
    elif key == "Bear Call Spread":
        sv = ac()
        b = oc(0.20, _e(sv))
        if sv is None or b is None:
            return None
        ks, kb = float(sv["Strike"]), float(b["Strike"])
        cred = round(float(sv["Ultimo"]) - float(b["Ultimo"]), 2)
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"Vende call ATM **{sv['Ativo']}** (K=R${ks:.2f}) e compra call OTM "
                f"**{b['Ativo']}** (K=R${kb:.2f}). Credito: R${cred:.2f}. "
                f"Lucra se ativo cair ou ficar estavel."
            ),
            "legs": [
                {"role": "VENDER 1 CALL ATM",  "detalhe": _fmt(sv)},
                {"role": "COMPRAR 1 CALL OTM", "detalhe": _fmt(b)},
            ],
            "plot_legs": [{"type": "SELL_CALL", "strike": ks}, {"type": "BUY_CALL", "strike": kb}],
            "explanation": (
                f"Ganho max (credito): R${cred:.2f} | Perda max: R${kb-ks-cred:.2f} | "
                f"Breakeven: R${ks+cred:.2f}."
            ),
        }

    # ── Long Straddle ─────────────────────────────────────────────────────────
    elif key == "Long Straddle":
        c = ac()
        p2 = ap()
        if c is None or p2 is None:
            return None
        k = float(c["Strike"])
        cost = round(float(c["Ultimo"]) + float(p2["Ultimo"]), 2)
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"Compra call ATM **{c['Ativo']}** e put ATM **{p2['Ativo']}** "
                f"simultaneamente. Lucra com qualquer grande movimento. Custo: R${cost:.2f}."
            ),
            "legs": [
                {"role": "COMPRAR 1 CALL ATM", "detalhe": _fmt(c)},
                {"role": "COMPRAR 1 PUT ATM",  "detalhe": _fmt(p2)},
            ],
            "plot_legs": [
                {"type": "BUY_CALL", "strike": k},
                {"type": "BUY_PUT",  "strike": float(p2["Strike"])},
            ],
            "explanation": (
                f"Breakeven alto: R${k+cost:.2f} | Breakeven baixo: R${k-cost:.2f}. "
                f"Necessario movimento > R${cost:.2f} para qualquer lado."
            ),
        }

    # ── Long Strangle ─────────────────────────────────────────────────────────
    elif key == "Long Strangle":
        c = oc(0.30)
        p2 = op(-0.30)
        if c is None or p2 is None:
            return None
        kc, kp = float(c["Strike"]), float(p2["Strike"])
        cost = round(float(c["Ultimo"]) + float(p2["Ultimo"]), 2)
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"Compra call OTM **{c['Ativo']}** (K=R${kc:.2f}) e put OTM "
                f"**{p2['Ativo']}** (K=R${kp:.2f}). Mais barato que Straddle, "
                f"requer movimento maior. Custo: R${cost:.2f}."
            ),
            "legs": [
                {"role": "COMPRAR 1 CALL OTM", "detalhe": _fmt(c)},
                {"role": "COMPRAR 1 PUT OTM",  "detalhe": _fmt(p2)},
            ],
            "plot_legs": [
                {"type": "BUY_CALL", "strike": kc},
                {"type": "BUY_PUT",  "strike": kp},
            ],
            "explanation": (
                f"Breakeven alto: R${kc+cost:.2f} | Breakeven baixo: R${kp-cost:.2f}."
            ),
        }

    # ── Covered Call ──────────────────────────────────────────────────────────
    elif key == "Covered Call":
        sv = oc(0.30)
        if sv is None:
            return None
        k, p = float(sv["Strike"]), float(sv["Ultimo"])
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"Vende call OTM **{sv['Ativo']}** (K=R${k:.2f}, Delta~0,30) "
                f"contra acoes em carteira. Gera renda de R${p:.2f}/contrato. "
                f"Delta proximo a 0,30 equilibra premio recebido e probabilidade de exercicio."
            ),
            "legs": [{"role": "VENDER 1 CALL OTM", "detalhe": _fmt(sv)}],
            "plot_legs": [{"type": "SELL_CALL", "strike": k}],
            "explanation": (
                f"Renda: R${p:.2f} | Preco efetivo de venda se exercida: R${k+p:.2f} | "
                f"Risco: upside limitado acima de R${k:.2f}."
            ),
        }

    # ── Protective Put ────────────────────────────────────────────────────────
    elif key == "Protective Put":
        b = op(-0.30)
        if b is None:
            return None
        k, p = float(b["Strike"]), float(b["Ultimo"])
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"Compra put OTM **{b['Ativo']}** (K=R${k:.2f}) como seguro da carteira. "
                f"Custo do hedge: R${p:.2f}. Delta {float(b['Delta']):.3f} — "
                f"ativa protecao se ativo cair abaixo de R${k:.2f}."
            ),
            "legs": [{"role": "COMPRAR 1 PUT OTM", "detalhe": _fmt(b)}],
            "plot_legs": [{"type": "BUY_PUT", "strike": k}],
            "explanation": (
                f"Protecao completa abaixo de R${k:.2f}. "
                f"Custo do seguro: R${p:.2f}/contrato."
            ),
        }

    # ── Iron Condor ───────────────────────────────────────────────────────────
    elif key == "Iron Condor":
        sp2 = op(-0.25)
        bp2 = op(-0.15, _e(sp2))
        sc2 = oc(0.25)
        bc2 = oc(0.15, _e(sc2))
        if any(x is None for x in [sp2, bp2, sc2, bc2]):
            return None
        cred = round(
            float(sp2["Ultimo"]) + float(sc2["Ultimo"])
            - float(bp2["Ultimo"]) - float(bc2["Ultimo"]), 2
        )
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"4 pernas formando corredor de lucro entre R${float(sp2['Strike']):.2f} e "
                f"R${float(sc2['Strike']):.2f}. Credito total: R${cred:.2f}. "
                f"Ideal quando ativo consolida sem tendencia definida — lucra com o tempo."
            ),
            "legs": [
                {"role": "COMPRAR PUT protecao baixa", "detalhe": _fmt(bp2)},
                {"role": "VENDER PUT receita",         "detalhe": _fmt(sp2)},
                {"role": "VENDER CALL receita",        "detalhe": _fmt(sc2)},
                {"role": "COMPRAR CALL protecao alta", "detalhe": _fmt(bc2)},
            ],
            "plot_legs": [
                {"type": "BUY_PUT",   "strike": float(bp2["Strike"])},
                {"type": "SELL_PUT",  "strike": float(sp2["Strike"])},
                {"type": "SELL_CALL", "strike": float(sc2["Strike"])},
                {"type": "BUY_CALL",  "strike": float(bc2["Strike"])},
            ],
            "explanation": (
                f"Ganho max: R${cred:.2f} | "
                f"Zona de lucro: R${float(sp2['Strike']):.2f}–R${float(sc2['Strike']):.2f}. "
                f"Perda max se ativo sair do corredor."
            ),
        }

    # ── Butterfly ─────────────────────────────────────────────────────────────
    elif key == "Butterfly":
        lo = oc(0.30)
        mi = ac(_e(lo))
        hi = oc(0.70, (_e(lo) or []) + (_e(mi) or []))
        if any(x is None for x in [lo, mi, hi]):
            return None
        kl, km, kh = float(lo["Strike"]), float(mi["Strike"]), float(hi["Strike"])
        cost = round(float(lo["Ultimo"]) - 2*float(mi["Ultimo"]) + float(hi["Ultimo"]), 2)
        return {
            "strategy_name": strategy_name,
            "rationale": (
                f"Compra **{lo['Ativo']}** (K=R${kl:.2f}), vende 2x **{mi['Ativo']}** "
                f"(K=R${km:.2f}), compra **{hi['Ativo']}** (K=R${kh:.2f}). "
                f"Lucro maximo se acao fechar exatamente em R${km:.2f} no vencimento."
            ),
            "legs": [
                {"role": "COMPRAR 1 CALL baixo",  "detalhe": _fmt(lo)},
                {"role": "VENDER 2 CALL ATM",     "detalhe": _fmt(mi)},
                {"role": "COMPRAR 1 CALL alto",   "detalhe": _fmt(hi)},
            ],
            "plot_legs": [
                {"type": "BUY_CALL",  "strike": kl},
                {"type": "SELL_CALL", "strike": km},
                {"type": "SELL_CALL", "strike": km},
                {"type": "BUY_CALL",  "strike": kh},
            ],
            "explanation": (
                f"Custo: R${cost:.2f} | Lucro max em K=R${km:.2f} | "
                f"Breakevens: R${kl+cost:.2f} e R${kh-cost:.2f}."
            ),
        }

    return None
