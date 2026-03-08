"""Script to append build_specific_strategy to options_engine.py"""
import os

addition = '''

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
'''

target = r'c:\Users\Usuario\Desktop\Projects\HMM\options_engine.py'
with open(target, 'a', encoding='utf-8') as f:
    f.write(addition)
print("Appended successfully. Lines:", len(open(target, encoding='utf-8').readlines()))
