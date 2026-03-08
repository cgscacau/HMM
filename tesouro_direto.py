# =============================================================================
#  Tesouro Direto — Data Fetcher + Duration Engine
# =============================================================================
"""
Busca os títulos disponíveis do Tesouro Direto e calcula métricas de duration
(Macaulay, Modificada e DV01) para cada título.

Hierarquia de fontes:
  1. Selenium (headless Chrome) — scraper do site oficial tesourodireto.com.br, dados de hoje
  2. Tesouro Transparente CSV (gov.br) — gratuita, sem chave, dados históricos
  3. Snapshot hardcoded (fallback offline) — dados reais de 07/03/2026
"""

import datetime
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup as _BS4
    _SELENIUM_OK = True
except ImportError:
    _SELENIUM_OK = False

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
# URL do site oficial do Tesouro Direto (renderizada por JS)
_TD_PAGE_URL = (
    "https://www.tesourodireto.com.br/produtos/"
    "dados-sobre-titulos/rendimento-dos-titulos"
)

# URL dos preços e taxas ofertados hoje — dataset aberto do Tesouro Nacional
_TESOURO_TRANSPARENTE_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-ca2e93d4d042/download/"
    "PrecoTaxaTesouroDireto.csv"
)

# Coupon rates padrão por tipo de título (% a.a. nominal)
COUPON_RATES: Dict[str, float] = {
    "LTN":     0.0,    # zero coupon
    "NTN-F":   0.10,   # 10% a.a. nominal, pago semestralmente
    "NTN-B":   0.06,   # IPCA + 6% a.a., pago semestralmente
    "NTN-B P": 0.0,    # IPCA + principal (zero coupon)
    "NTN-C":   0.06,   # IGP-M + 6%, semestralmente
    "LFT":     0.0,    # floating (Selic), duration ≈ 0
    "Renda+":  0.0,    # IPCA+, pagamentos mensais na fase de beneício (trat. zero-coupon para duration)
    "Educa+":  0.0,    # IPCA+, pagamentos mensais na fase educacional (trat. zero-coupon)
}

# Periodicidade de cupons por tipo
COUPON_FREQ: Dict[str, int] = {
    "LTN":    1, "NTN-F": 2, "NTN-B": 2,
    "NTN-B P": 1, "NTN-C": 2, "LFT": 1,
    "Renda+": 1, "Educa+": 1,
}

# Mapeamento: termos livres → tipo canônico
# IMPORTANTE: a ordem importa — os mais específicos devem vir primeiro
_TYPE_MAP = {
    "RENDA+":                         "Renda+",
    "EDUCA+":                         "Educa+",
    "PREFIXADO COM JUROS SEMESTRAIS": "NTN-F",
    "PREFIXADO":                      "LTN",
    "IPCA+ COM JUROS SEMESTRAIS":     "NTN-B",
    "IPCA+":                          "NTN-B P",
    "SELIC":                          "LFT",
    "IGPM+ COM JUROS SEMESTRAIS":     "NTN-C",
    "NTN-F":  "NTN-F",
    "NTN-B":  "NTN-B",
    "NTN-B P":"NTN-B P",
    "LTN":    "LTN",
    "LFT":    "LFT",
    "NTN-C":  "NTN-C",
}

# Snapshot — dados reais do site Tesouro Direto em 07/03/2026
_FALLBACK_DATA = [
    # ── Tesouro Selic ──────────────────────────────────────────────────────────
    {"Titulo": "Tesouro Selic 2031",
     "Tipo": "LFT",      "Vencimento": "2031-03-01", "Taxa_Compra": 13.0989, "Preco_Compra": 18_449.66},

    # ── Tesouro Prefixado ──────────────────────────────────────────────────────
    {"Titulo": "Tesouro Prefixado 2029",
     "Tipo": "LTN",     "Vencimento": "2029-01-01", "Taxa_Compra": 13.33,   "Preco_Compra": 704.98},
    {"Titulo": "Tesouro Prefixado 2032",
     "Tipo": "LTN",     "Vencimento": "2032-01-01", "Taxa_Compra": 13.94,   "Preco_Compra": 470.23},
    {"Titulo": "Tesouro Prefixado com Juros Semestrais 2037",
     "Tipo": "NTN-F",   "Vencimento": "2037-01-01", "Taxa_Compra": 14.11,   "Preco_Compra": 805.46},

    # ── Tesouro IPCA+ (sem cupons) ─────────────────────────────────────────────
    {"Titulo": "Tesouro IPCA+ 2032",
     "Tipo": "NTN-B P", "Vencimento": "2032-08-15", "Taxa_Compra": 7.78,    "Preco_Compra": 2_862.14},
    {"Titulo": "Tesouro IPCA+ 2040",
     "Tipo": "NTN-B P", "Vencimento": "2040-08-15", "Taxa_Compra": 7.29,    "Preco_Compra": 1_683.52},
    {"Titulo": "Tesouro IPCA+ 2050",
     "Tipo": "NTN-B P", "Vencimento": "2050-08-15", "Taxa_Compra": 6.95,    "Preco_Compra": 904.24},

    # ── Tesouro IPCA+ com Juros Semestrais ────────────────────────────────────
    {"Titulo": "Tesouro IPCA+ com Juros Semestrais 2037",
     "Tipo": "NTN-B",   "Vencimento": "2037-05-15", "Taxa_Compra": 7.58,    "Preco_Compra": 4_194.35},
    {"Titulo": "Tesouro IPCA+ com Juros Semestrais 2045",
     "Tipo": "NTN-B",   "Vencimento": "2045-05-15", "Taxa_Compra": 7.27,    "Preco_Compra": 4_139.32},
    {"Titulo": "Tesouro IPCA+ com Juros Semestrais 2060",
     "Tipo": "NTN-B",   "Vencimento": "2060-08-15", "Taxa_Compra": 7.18,    "Preco_Compra": 3_982.69},

    # ── Tesouro Renda+ (IPCA+, pagamentos mensais na fase de beneício) ─────────
    {"Titulo": "Tesouro Renda+ Aposentadoria Extra 2030",
     "Tipo": "Renda+",  "Vencimento": "2049-12-15", "Taxa_Compra": 7.35,    "Preco_Compra": 1_894.67},
    {"Titulo": "Tesouro Renda+ Aposentadoria Extra 2035",
     "Tipo": "Renda+",  "Vencimento": "2054-12-15", "Taxa_Compra": 7.17,    "Preco_Compra": 1_369.15},
    {"Titulo": "Tesouro Renda+ Aposentadoria Extra 2040",
     "Tipo": "Renda+",  "Vencimento": "2059-12-15", "Taxa_Compra": 7.04,    "Preco_Compra": 996.07},
    {"Titulo": "Tesouro Renda+ Aposentadoria Extra 2045",
     "Tipo": "Renda+",  "Vencimento": "2064-12-15", "Taxa_Compra": 6.98,    "Preco_Compra": 721.04},
    {"Titulo": "Tesouro Renda+ Aposentadoria Extra 2050",
     "Tipo": "Renda+",  "Vencimento": "2069-12-15", "Taxa_Compra": 6.97,    "Preco_Compra": 517.14},
    {"Titulo": "Tesouro Renda+ Aposentadoria Extra 2055",
     "Tipo": "Renda+",  "Vencimento": "2074-12-15", "Taxa_Compra": 6.98,    "Preco_Compra": 368.72},
    {"Titulo": "Tesouro Renda+ Aposentadoria Extra 2060",
     "Tipo": "Renda+",  "Vencimento": "2079-12-15", "Taxa_Compra": 6.98,    "Preco_Compra": 263.65},
    {"Titulo": "Tesouro Renda+ Aposentadoria Extra 2065",
     "Tipo": "Renda+",  "Vencimento": "2084-12-15", "Taxa_Compra": 6.98,    "Preco_Compra": 188.54},

    # ── Tesouro Educa+ (IPCA+, pagamentos mensais na fase educacional) ─────────
    {"Titulo": "Tesouro Educa+ 2027",
     "Tipo": "Educa+",  "Vencimento": "2031-12-15", "Taxa_Compra": 7.84,    "Preco_Compra": 3_628.64},
    {"Titulo": "Tesouro Educa+ 2028",
     "Tipo": "Educa+",  "Vencimento": "2032-12-15", "Taxa_Compra": 7.81,    "Preco_Compra": 3_370.18},
    {"Titulo": "Tesouro Educa+ 2029",
     "Tipo": "Educa+",  "Vencimento": "2033-12-15", "Taxa_Compra": 7.78,    "Preco_Compra": 3_131.91},
    {"Titulo": "Tesouro Educa+ 2030",
     "Tipo": "Educa+",  "Vencimento": "2034-12-15", "Taxa_Compra": 7.76,    "Preco_Compra": 2_909.91},
    {"Titulo": "Tesouro Educa+ 2031",
     "Tipo": "Educa+",  "Vencimento": "2035-12-15", "Taxa_Compra": 7.72,    "Preco_Compra": 2_708.35},
    {"Titulo": "Tesouro Educa+ 2032",
     "Tipo": "Educa+",  "Vencimento": "2036-12-15", "Taxa_Compra": 7.68,    "Preco_Compra": 2_522.80},
    {"Titulo": "Tesouro Educa+ 2033",
     "Tipo": "Educa+",  "Vencimento": "2037-12-15", "Taxa_Compra": 7.62,    "Preco_Compra": 2_355.86},
    {"Titulo": "Tesouro Educa+ 2034",
     "Tipo": "Educa+",  "Vencimento": "2038-12-15", "Taxa_Compra": 7.56,    "Preco_Compra": 2_202.81},
    {"Titulo": "Tesouro Educa+ 2035",
     "Tipo": "Educa+",  "Vencimento": "2039-12-15", "Taxa_Compra": 7.49,    "Preco_Compra": 2_064.08},
    {"Titulo": "Tesouro Educa+ 2036",
     "Tipo": "Educa+",  "Vencimento": "2040-12-15", "Taxa_Compra": 7.43,    "Preco_Compra": 1_933.97},
    {"Titulo": "Tesouro Educa+ 2037",
     "Tipo": "Educa+",  "Vencimento": "2041-12-15", "Taxa_Compra": 7.36,    "Preco_Compra": 1_816.24},
    {"Titulo": "Tesouro Educa+ 2038",
     "Tipo": "Educa+",  "Vencimento": "2042-12-15", "Taxa_Compra": 7.30,    "Preco_Compra": 1_705.67},
    {"Titulo": "Tesouro Educa+ 2039",
     "Tipo": "Educa+",  "Vencimento": "2043-12-15", "Taxa_Compra": 7.25,    "Preco_Compra": 1_601.24},
    {"Titulo": "Tesouro Educa+ 2040",
     "Tipo": "Educa+",  "Vencimento": "2044-12-15", "Taxa_Compra": 7.20,    "Preco_Compra": 1_504.71},
    {"Titulo": "Tesouro Educa+ 2041",
     "Tipo": "Educa+",  "Vencimento": "2045-12-15", "Taxa_Compra": 7.15,    "Preco_Compra": 1_415.43},
    {"Titulo": "Tesouro Educa+ 2042",
     "Tipo": "Educa+",  "Vencimento": "2046-12-15", "Taxa_Compra": 7.11,    "Preco_Compra": 1_330.53},
    {"Titulo": "Tesouro Educa+ 2043",
     "Tipo": "Educa+",  "Vencimento": "2047-12-15", "Taxa_Compra": 7.08,    "Preco_Compra": 1_249.53},
    {"Titulo": "Tesouro Educa+ 2044",
     "Tipo": "Educa+",  "Vencimento": "2048-12-15", "Taxa_Compra": 7.05,    "Preco_Compra": 1_174.12},
]

# ─────────────────────────────────────────────────────────────────────────────
#  INFERÊNCIA DE TIPO
# ─────────────────────────────────────────────────────────────────────────────
def _infer_type(titulo: str) -> str:
    """Infere o código de tipo (LTN, NTN-B, etc.) a partir do nome do título."""
    t = titulo.upper()
    for key, val in _TYPE_MAP.items():
        if key in t:
            return val
    if "SELIC" in t:
        return "LFT"
    if "IPCA" in t and "JUROS" in t:
        return "NTN-B"
    if "IPCA" in t:
        return "NTN-B P"
    if "PREFIXADO" in t and "JUROS" in t:
        return "NTN-F"
    if "PREFIXADO" in t:
        return "LTN"
    return "LTN"  # fallback


# ─────────────────────────────────────────────────────────────────────────────
#  CÁLCULO DE DURATION
# ─────────────────────────────────────────────────────────────────────────────
def calcular_duration(
    ytm: float,
    maturity: datetime.date,
    tipo: str,
    face_value: float = 1000.0,
    price: Optional[float] = None,
    reference_date: Optional[datetime.date] = None,
) -> Dict[str, float]:
    """
    Calcula Macaulay Duration, Modified Duration e DV01 para um título.

    Args:
        ytm:            Yield to maturity anual (decimal, ex: 0.1435 para 14,35%)
        maturity:       Data de vencimento
        tipo:           Código do tipo ("LTN", "NTN-B", "NTN-F", "LFT", etc.)
        face_value:     Valor nominal (default R$1.000)
        price:          Preço de compra atual; se None, estimado pelo fluxo
        reference_date: Data-base; default = hoje

    Returns:
        dict com chaves: macaulay, modified, dv01, ytm_pct
    """
    if reference_date is None:
        reference_date = datetime.date.today()

    # LFT: taxa flutuante — duration quase zero (próximo reset = 1 dia)
    if tipo == "LFT":
        pu = price if price else face_value
        return {"macaulay": 0.01, "modified": 0.01, "dv01": pu * 0.01 / 10_000,
                "ytm_pct": ytm * 100}

    years_to_mat = (maturity - reference_date).days / 365.25
    if years_to_mat <= 0:
        return {"macaulay": 0.0, "modified": 0.0, "dv01": 0.0, "ytm_pct": ytm * 100}

    freq = COUPON_FREQ.get(tipo, 1)
    coupon_rate = COUPON_RATES.get(tipo, 0.0)

    # Gera datas de cupom
    if coupon_rate == 0.0:
        # Zero-coupon: único fluxo no vencimento
        cash_flows = [(years_to_mat, face_value)]
    else:
        # Determina datas de pagamento de cupom (retroativamente a partir do vencimento)
        coupon_amount = coupon_rate / freq * face_value
        payment_dates = []
        payment_date = maturity
        while True:
            months_back = 12 // freq
            prev = payment_date.replace(
                year=payment_date.year - (1 if payment_date.month <= months_back else 0),
                month=(payment_date.month - months_back - 1) % 12 + 1,
            )
            t_years = (payment_date - reference_date).days / 365.25
            if t_years > 0:
                payment_dates.append((t_years, coupon_amount))
            payment_date = prev
            if payment_date <= reference_date:
                break
        payment_dates.sort(key=lambda x: x[0])
        # Adiciona principal no vencimento
        cash_flows = payment_dates
        if cash_flows:
            cash_flows[-1] = (cash_flows[-1][0], cash_flows[-1][1] + face_value)
        else:
            cash_flows = [(years_to_mat, face_value)]

    # PV de cada fluxo
    pv_total = 0.0
    weighted_time = 0.0
    periodic_ytm = ytm / freq

    for t_years, cf in cash_flows:
        periods = t_years * freq
        pv = cf / ((1 + periodic_ytm) ** periods)
        pv_total += pv
        weighted_time += t_years * pv

    if pv_total <= 0:
        macaulay = years_to_mat
    else:
        macaulay = weighted_time / pv_total

    modified = macaulay / (1 + ytm / freq)

    pu = price if price and price > 0 else pv_total
    dv01 = modified * pu / 10_000  # R$ por R$10.000 face, por bp

    return {
        "macaulay":  round(macaulay, 4),
        "modified":  round(modified, 4),
        "dv01":      round(dv01, 6),
        "ytm_pct":   round(ytm * 100, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  BUSCA DE DADOS
# ─────────────────────────────────────────────────────────────────────────────
def _buscar_selenium() -> Optional[pd.DataFrame]:
    """
    Abre o site oficial do Tesouro Direto em modo headless com Selenium,
    aguarda os dados carregarem via JS e extrai a tabela de rendimentos.

    Retorna DataFrame com colunas: Titulo, Tipo, Vencimento, Taxa_Compra, Preco_Compra
    ou None em caso de erro.
    """
    if not _SELENIUM_OK:
        return None
    driver = None
    try:
        opts = ChromeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
        # Silencia logs do webdriver-manager
        import logging
        logging.getLogger("WDM").setLevel(logging.ERROR)

        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=opts)
        driver.set_page_load_timeout(30)

        driver.get(_TD_PAGE_URL)

        # Aguarda a tabela de títulos aparecer (selector do card de cada título)
        # O site renderiza cards com class que contém 'td-title-name' ou similar
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "table, [class*='title'], [class*='bond'], [class*='produto']")
            )
        )
        # Espera adicional para JS terminar de popular os preços
        import time
        time.sleep(3)

        soup = _BS4(driver.page_source, "html.parser")
        rows = _parse_td_html(soup)
        if rows:
            return pd.DataFrame(rows)
        return None

    except Exception as e:
        warnings.warn(f"Selenium scraper falhou: {e}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def _parse_td_html(soup) -> list:
    """
    Extrai linhas de títulos do HTML renderizado pelo Selenium.
    Suporta a estrutura de tabela e a estrutura de cards do site.

    Retorna lista de dicts: Titulo, Tipo, Vencimento, Taxa_Compra, Preco_Compra.
    """
    import re
    rows = []

    # ── Tenta estrutura de tabela HTML ────────────────────────────────────
    tables = soup.find_all("table")
    for tbl in tables:
        thead = tbl.find("thead")
        tbody = tbl.find("tbody")
        if not tbody:
            continue
        # Determina índices das colunas a partir do cabeçalho
        headers = [th.get_text(strip=True).lower() for th in (thead or tbl).find_all("th")]
        idx_nome     = next((i for i, h in enumerate(headers) if "título" in h or "titulo" in h or "nome" in h), 0)
        idx_taxa     = next((i for i, h in enumerate(headers) if "rendimento" in h or "taxa" in h or "rentabilidade" in h), 1)
        idx_preco    = next((i for i, h in enumerate(headers) if "preço" in h or "preco" in h or "unit" in h), 3)
        idx_venc     = next((i for i, h in enumerate(headers) if "vencimento" in h or "data" in h), 4)

        for tr in tbody.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            def txt(i):
                return cells[i].get_text(" ", strip=True) if i < len(cells) else ""

            nome  = txt(idx_nome)
            taxa_raw  = txt(idx_taxa)
            preco_raw = txt(idx_preco)
            venc_raw  = txt(idx_venc)

            if not nome or not taxa_raw:
                continue

            # Limpa taxa: extrai número de "IPCA + 7,78%" ou "13,33%" ou "SELIC + 0,0989%"
            taxa_nums = re.findall(r"[\d]+[,.]?[\d]*", taxa_raw.replace(".", "").replace(",", "."))
            if not taxa_nums:
                continue
            # Pega o número da taxa real (o último número significativo)
            taxa_val = float(taxa_nums[-1])

            # Limpa preço: "R$\xa018.449,66" -> float
            preco_nums = re.findall(r"[\d]+[,.]?[\d]*", preco_raw.replace(".", "").replace(",", "."))
            preco_val  = float(preco_nums[0]) if preco_nums else 0.0

            # Limpa vencimento: "01/03/2031" -> "2031-03-01"
            venc_match = re.search(r"(\d{2})/(\d{2})/(\d{4})", venc_raw)
            if not venc_match:
                continue
            venc_iso = f"{venc_match.group(3)}-{venc_match.group(2)}-{venc_match.group(1)}"

            rows.append({
                "Titulo":      nome,
                "Tipo":        _infer_type(nome),
                "Vencimento":  datetime.datetime.strptime(venc_iso, "%Y-%m-%d").date(),
                "Taxa_Compra": taxa_val,
                "Preco_Compra": preco_val,
            })

    if rows:
        return rows

    # ── Tenta estrutura de cards / divs (fallback do HTML do site) ────────────
    # Busca qualquer elemento que contenha o nome de um título conhecido
    _KNOWN_PREFIXES = [
        "Tesouro Selic", "Tesouro Prefixado", "Tesouro IPCA+",
        "Tesouro Renda+", "Tesouro Educa+"
    ]
    all_text_blocks = soup.find_all(string=re.compile(
        r"Tesouro (Selic|Prefixado|IPCA\+|Renda\+|Educa\+)"
    ))
    seen = set()
    for text_node in all_text_blocks:
        nome = text_node.strip()
        if nome in seen:
            continue
        seen.add(nome)

        # Sobe na árvore para encontrar o container do card
        container = text_node.parent
        for _ in range(6):
            if container is None:
                break
            full = container.get_text(" ", strip=True)
            # Procura taxa
            taxa_match = re.search(
                r"(IPCA\+|SELIC)\s*\+?\s*([\d]+[,.]?[\d]*)%"
                r"|([\d]+[,.]?[\d]*)%",
                full
            )
            venc_match = re.search(r"(\d{2})/(\d{2})/(\d{4})", full)
            preco_match = re.search(
                r"R\$\s*([\d.]+[,]?[\d]*)", full.replace("\xa0", " ")
            )
            if taxa_match and venc_match:
                if taxa_match.group(2):
                    taxa_val = float(taxa_match.group(2).replace(",", "."))
                else:
                    taxa_val = float(taxa_match.group(3).replace(",", "."))
                preco_val = 0.0
                if preco_match:
                    p = preco_match.group(1).replace(".", "").replace(",", ".")
                    try:
                        preco_val = float(p)
                    except ValueError:
                        pass
                venc_iso = (
                    f"{venc_match.group(3)}-"
                    f"{venc_match.group(2)}-"
                    f"{venc_match.group(1)}"
                )
                rows.append({
                    "Titulo":       nome,
                    "Tipo":         _infer_type(nome),
                    "Vencimento":   datetime.datetime.strptime(venc_iso, "%Y-%m-%d").date(),
                    "Taxa_Compra":  taxa_val,
                    "Preco_Compra": preco_val,
                })
                break
            container = container.parent

    return rows


def _buscar_tesouro_transparente() -> Optional[pd.DataFrame]:
    """Tenta baixar o CSV de preços/taxas do Tesouro Transparente."""
    if not _REQUESTS_OK:
        return None
    try:
        resp = requests.get(_TESOURO_TRANSPARENTE_URL, timeout=15)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), sep=";", decimal=",", encoding="latin-1")
        return df
    except Exception:
        return None


def _parse_tesouro_transparente(raw: pd.DataFrame) -> pd.DataFrame:
    """Parseia o CSV do Tesouro Transparente e retorna df padronizado."""
    # Renomear colunas variáveis
    col_map = {}
    for c in raw.columns:
        cu = c.strip().upper()
        if "TIPO" in cu and "TITULO" in cu:
            col_map[c] = "Titulo"
        elif "DATA" in cu and "VENCIMENTO" in cu:
            col_map[c] = "Vencimento_str"
        elif "TAXA" in cu and "COMPRA" in cu:
            col_map[c] = "Taxa_Compra"
        elif "PU" in cu and "COMPRA" in cu:
            col_map[c] = "Preco_Compra"
        elif "DATA" in cu and "BASE" in cu:
            col_map[c] = "Data_Base"
    raw = raw.rename(columns=col_map)

    needed = ["Titulo", "Vencimento_str", "Taxa_Compra", "Preco_Compra"]
    for col in needed:
        if col not in raw.columns:
            return pd.DataFrame()

    # Filtrar última data disponível
    if "Data_Base" in raw.columns:
        raw["Data_Base_dt"] = pd.to_datetime(raw["Data_Base"], dayfirst=True, errors="coerce")
        latest = raw["Data_Base_dt"].max()
        raw = raw[raw["Data_Base_dt"] == latest]

    df = raw[needed].copy()
    df["Taxa_Compra"] = pd.to_numeric(df["Taxa_Compra"], errors="coerce")
    df["Preco_Compra"] = pd.to_numeric(df["Preco_Compra"], errors="coerce")
    df = df.dropna(subset=["Taxa_Compra", "Preco_Compra"])
    df = df[df["Taxa_Compra"] > 0]

    df["Vencimento"] = pd.to_datetime(
        df["Vencimento_str"], dayfirst=True, errors="coerce"
    ).dt.date
    df = df.dropna(subset=["Vencimento"])
    df["Tipo"] = df["Titulo"].apply(_infer_type)
    return df[["Titulo", "Tipo", "Vencimento", "Taxa_Compra", "Preco_Compra"]]


def buscar_tesouro_direto() -> pd.DataFrame:
    """
    Retorna DataFrame com os títulos do Tesouro Direto disponíveis
    enriquecidos com métricas de duration.

    Hierarquia:
        1. Selenium (site oficial) -> dados ao vivo
        2. Tesouro Transparente CSV (gov.br) -> dados históricos
        3. Snapshot hardcoded -> fallback offline

    Colunas de saída:
        Titulo, Tipo, Vencimento, Taxa_Compra, Preco_Compra,
        Anos_Venc, Duration Macaulay, Duration Modificada, DV01
    """
    df = None
    source = "fallback"

    # Fonte 1: Selenium (site oficial, dados do dia)
    sel_df = _buscar_selenium()
    if sel_df is not None and not sel_df.empty:
        df = sel_df
        source = "selenium"

    # Fonte 2: Tesouro Transparente CSV
    if df is None or df.empty:
        raw_csv = _buscar_tesouro_transparente()
        if raw_csv is not None and not raw_csv.empty:
            parsed = _parse_tesouro_transparente(raw_csv)
            if not parsed.empty:
                df = parsed
                source = "tesouro_transparente"

    # Fallback: snapshot hardcoded
    if df is None or df.empty:
        rows = []
        today = datetime.date.today()
        for b in _FALLBACK_DATA:
            venc = datetime.datetime.strptime(b["Vencimento"], "%Y-%m-%d").date()
            if venc > today:
                rows.append({
                    "Titulo":      b["Titulo"],
                    "Tipo":        b["Tipo"],
                    "Vencimento":  venc,
                    "Taxa_Compra": b["Taxa_Compra"],
                    "Preco_Compra": b["Preco_Compra"],
                })
        df = pd.DataFrame(rows)
        source = "fallback"

    # Enriquece com duration
    today = datetime.date.today()
    records = []
    for _, row in df.iterrows():
        ytm = float(row["Taxa_Compra"]) / 100.0
        venc = row["Vencimento"]
        tipo = row["Tipo"]
        price = float(row["Preco_Compra"])
        anos = max((venc - today).days / 365.25, 0)

        dur = calcular_duration(ytm, venc, tipo, face_value=1000.0,
                                price=price, reference_date=today)
        records.append({
            "Titulo":             row["Titulo"],
            "Tipo":               tipo,
            "Vencimento":         str(venc),
            "Anos_Venc":          round(anos, 2),
            "Taxa_Compra (% a.a.)": round(ytm * 100, 2),
            "Preco_Compra (R$)":  price,
            "Duration Macaulay":  dur["macaulay"],
            "Duration Modificada": dur["modified"],
            "DV01 (R$/bp)":       dur["dv01"],
        })

    result = pd.DataFrame(records).sort_values("Anos_Venc")
    result.attrs["source"] = source
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  PORTFOLIO DURATION
# ─────────────────────────────────────────────────────────────────────────────
def calcular_duration_portfolio(df_bonds: pd.DataFrame, pesos: Dict[str, float]) -> Dict[str, float]:
    """
    Calcula a duration ponderada do portfólio.

    Args:
        df_bonds: DataFrame retornado por buscar_tesouro_direto()
        pesos:    {Titulo: peso_percentual}  (somam 100)

    Returns:
        dict com macaulay_portfolio, modified_portfolio, dv01_portfolio
    """
    total_peso = sum(pesos.values())
    if total_peso <= 0:
        return {"macaulay_portfolio": 0.0, "modified_portfolio": 0.0, "dv01_portfolio": 0.0}

    mac_pond = 0.0
    mod_pond = 0.0
    dv01_pond = 0.0

    for titulo, peso in pesos.items():
        row = df_bonds[df_bonds["Titulo"] == titulo]
        if row.empty:
            continue
        w = peso / total_peso
        mac_pond  += w * float(row["Duration Macaulay"].iloc[0])
        mod_pond  += w * float(row["Duration Modificada"].iloc[0])
        dv01_pond += w * float(row["DV01 (R$/bp)"].iloc[0])

    return {
        "macaulay_portfolio":  round(mac_pond, 3),
        "modified_portfolio":  round(mod_pond, 3),
        "dv01_portfolio":      round(dv01_pond, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MOTOR DE ESTRATÉGIA
# ─────────────────────────────────────────────────────────────────────────────
ESTRATEGIAS = {
    # ── CENÁRIO 1: SELIC EM QUEDA → preços sobem → prefira duration longa ────
    "Selic em Queda (Alta de Preços)": {
        "emoji": "📉",
        "color": "#00e676",
        "duration_alvo": "LONGA (> 5 anos)",
        "tipos_preferidos": ["NTN-B", "NTN-F", "Renda+"],
        "tipos_evitar": ["LFT"],
        "logica": (
            "Quando a Selic cai, títulos de renda fixa sobem de preço — quanto maior a duration, "
            "maior o ganho de capital. Prefira:\n"
            "• **NTN-B** (Tesouro IPCA+ com Juros Semestrais) longo prazo: ganho de marcação a mercado + proteção ao IPCA.\n"
            "• **NTN-F** (Tesouro Prefixado com Juros Semestrais 2037): trava taxa alta e ainda ganha com queda dos juros.\n"
            "• **Renda+** (Aposentadoria Extra): duration ultra-longa (20–60 anos), máxima sensibilidade à queda da Selic — "
            "ideal para quem tem horizonte de longo prazo e quer maximizar a marcação a mercado.\n"
            "Evite **LFT** (Tesouro Selic): duration ≈ 0, rentabilidade cai junto com a Selic."
        ),
        "alocacao": {
            "NTN-B — IPCA+ com Juros Semestrais longo (≥ 2045)": 35,
            "Renda+ — Aposentadoria Extra (duration ultra-longa)": 25,
            "NTN-F — Prefixado com Juros Semestrais 2037":         25,
            "NTN-B P — IPCA+ sem cupom médio (2032–2040)":         10,
            "LFT — Tesouro Selic (liquidez mínima)":                5,
        },
    },

    # ── CENÁRIO 2: SELIC ESTÁVEL → carrego domina → duration média ──────────
    "Selic Estável / Neutro": {
        "emoji": "⚖️",
        "color": "#ffb74d",
        "duration_alvo": "MEDIA (2–5 anos)",
        "tipos_preferidos": ["NTN-B P", "LTN", "Educa+"],
        "tipos_evitar": [],
        "logica": (
            "Com Selic estável, o carrego (acúmulo de juros) é a principal fonte de retorno. "
            "Diversifique:\n"
            "• **NTN-B P** (Tesouro IPCA+): protege contra inflação e reinveste automaticamente — "
            "sem risco de reinvestimento de cupons.\n"
            "• **LTN** (Tesouro Prefixado 2029/2032): trava a taxa atual sem volatilidade de cupons.\n"
            "• **Educa+**: opção IPCA+ com prazo médio (5–15 anos) e pagamentos futuros definidos — "
            "excelente para objetivos com data específica.\n"
            "Mantenha parte em **LFT** (Tesouro Selic) para aproveitar oportunidades sem risco de preço."
        ),
        "alocacao": {
            "NTN-B P — IPCA+ sem cupom (2032–2050)":    30,
            "LTN — Prefixado 2029 ou 2032":              25,
            "Educa+ — prazo médio (2029–2036)":          20,
            "LFT — Tesouro Selic 2031 (liquidez)":       15,
            "NTN-B — IPCA+ com Juros Semestrais 2037":   10,
        },
    },

    # ── CENÁRIO 3: SELIC EM ALTA → preços caem → minimize duration ──────────
    "Selic em Alta (Queda de Preços)": {
        "emoji": "📈",
        "color": "#ef5350",
        "duration_alvo": "CURTA (< 2 anos)",
        "tipos_preferidos": ["LFT"],
        "tipos_evitar": ["NTN-F", "LTN", "Renda+", "NTN-B"],
        "logica": (
            "Quando a Selic sobe, preços de renda fixa caem — quanto maior a duration, maior a perda. "
            "Estratégia defensiva:\n"
            "• **LFT** (Tesouro Selic 2031): duration ≈ 0, rentabilidade sobe automaticamente com a Selic. "
            "Único tipo que se beneficia diretamente da alta.\n"
            "• Evite **Renda+**: duration de 20–60 anos — catástrofe de preço em ambiente de Selic alta.\n"
            "• Evite **NTN-B** longo e **NTN-F**: alta duration e taxa travada abaixo da nova Selic.\n"
            "Se já tiver posição em IPCA+, prefira o prazo mais curto disponível "
            "(**NTN-B P 2032**) para minimizar a perda de marcação."
        ),
        "alocacao": {
            "LFT — Tesouro Selic 2031 (proteção total)":  70,
            "NTN-B P — IPCA+ curto 2032 (se necessário)": 20,
            "LTN — Prefixado 2029 (prazo curto)":          10,
            "NTN-B / NTN-F / Renda+ / Educa+":             0,
        },
    },
}


def recomendar_estrategia(cenario: str) -> Dict:
    """Retorna o dicionário de estratégia para o cenário selecionado."""
    return ESTRATEGIAS.get(cenario, list(ESTRATEGIAS.values())[0])

