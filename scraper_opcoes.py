"""
scraper_opcoes.py — B3 options data from opcoes.net.br
=======================================================
Strategy (confirmed via DOM inspection):
  - Navigate the HTML page with headless Selenium
  - The page renders the DataTable (#tblListaOpc) with FULL data from
    "Último pregão" by default — including real Último, Vol.Impl, Delta,
    Gamma, Theta and Vega — even on weekends
  - Read ALL rows via the DOM (paginating through DataTable pages)
  - Parse BR-format numbers: "38,40" → 38.40, "33,6" → 33.6

DOM confirmed column order (18 cols, 0-indexed):
  0  Ticker       e.g. PETRC392
  1  Tipo         CALL / PUT
  2  F.M.         formador de mercado (√ or blank)
  3  Mod.         A / E
  4  Strike       e.g. 38,40
  5  A/I/OTM      ITM / ATM / OTM
  6  Dist(%)
  7  Último       last price
  8  Var(%)
  9  Data/Hora
  10 Núm.Neg
  11 Vol.Financeiro
  12 Vol.Impl(%)  implied volatility %
  13 Delta
  14 Gamma
  15 Theta($)
  16 Theta(%)     ← EXTRA column — shifts Vega to index 17
  17 Vega
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# ─── Constants ─────────────────────────────────────────────────────────────────
_PAGE = "https://opcoes.net.br/opcoes/bovespa/{ticker}"


# ─── Chrome (headless) ─────────────────────────────────────────────────────────
def _make_driver() -> webdriver.Chrome:
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--log-level=3")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    opts.add_experimental_option("useAutomationExtension", False)
    if os.name == 'posix':
        # Em Linux (Streamlit Cloud), precisamos usar o Chromium do SO
        opts.binary_location = "/usr/bin/chromium"
        # O ChromeDriverManager às vezes falha ao achar o driver certo para o Chromium de repositório,
        # tentar forçar o uso do /usr/bin/chromedriver que vem no packages.txt do Streamlit Cloud
        try:
            service = Service("/usr/bin/chromedriver")
        except:
            service = Service(ChromeDriverManager().install())
    else:
        service = Service(ChromeDriverManager().install())
        
    drv = webdriver.Chrome(service=service, options=opts)
    drv.execute_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    return drv


# ─── Black-Scholes fallback ─────────────────────────────────────────────────────
def _bs_price(S: float, K: float, T: float, r: float, sigma: float, ot: str = "c") -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if ot == "c" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return (S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            if ot == "c" else
            K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _bs_greeks(S: float, K: float, T: float, sigma: float, ot: str):
    """Returns (delta, gamma, theta$, vega) given sigma (decimal, e.g. 0.30)."""
    r = 0.135
    if T <= 0 or sigma <= 0:
        return 0.5, 0.0, 0.0, 0.0
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    gamma = norm.pdf(d1) / (S * sigma * sqT)
    vega  = S * norm.pdf(d1) * sqT / 100.0
    if ot == "c":
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqT)
                 - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * sqT)
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0
    return delta, gamma, theta, vega


# ─── Helpers ───────────────────────────────────────────────────────────────────
def _to_float(txt: str) -> Optional[float]:
    """Convert BR-format number '38,40' or '33,6' or '2,11' to float."""
    if not txt:
        return None
    txt = txt.strip().replace("\xa0", "").replace(" ", "")
    if txt in ("", "-", "--", "N/A"):
        return None
    # BR format: thousands = '.', decimal = ','
    # Remove thousand separators then swap comma for dot
    txt = re.sub(r"\.(?=\d{3})", "", txt)   # remove '.' used as thousands sep
    txt = txt.replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return None


def _get_expirations(drv: webdriver.Chrome) -> list[str]:
    """Extract expiration dates from checkbox inputs (id='v2026-03-20' format)."""
    inputs = drv.find_elements(By.CSS_SELECTOR, "input[type='checkbox'][id^='v']")
    exps: list[str] = []
    seen: set[str] = set()
    for inp in inputs:
        vid = inp.get_attribute("id") or ""
        # IDs like 'v2026-03-20'
        m = re.match(r"^v(\d{4}-\d{2}-\d{2})$", vid)
        if m:
            v = m.group(1)
            if v not in seen:
                seen.add(v)
                exps.append(v)
    if not exps:
        # Fallback: value attribute
        for inp in drv.find_elements(By.CSS_SELECTOR, "input[type='checkbox'][value]"):
            v = inp.get_attribute("value") or ""
            if re.match(r"^\d{4}-\d{2}-\d{2}$", v) and v not in seen:
                seen.add(v)
                exps.append(v)
    print(f"[opcoes] Vencimentos: {exps}")
    return exps


def _select_only(drv: webdriver.Chrome, exp: str) -> bool:
    """Select only the checkbox for `exp`, deselect others. Returns True on success."""
    try:
        # Deselect all currently checked boxes
        for cb in drv.find_elements(By.CSS_SELECTOR, "input[type='checkbox']"):
            vid = cb.get_attribute("id") or ""
            val = cb.get_attribute("value") or ""
            is_exp_cb = re.match(r"^v\d{4}-\d{2}-\d{2}$", vid) or re.match(r"^\d{4}-\d{2}-\d{2}$", val)
            if is_exp_cb and cb.is_selected():
                drv.execute_script("arguments[0].click();", cb)
                time.sleep(0.3)

        # Select target
        target = drv.find_element(By.CSS_SELECTOR, f"input[id='v{exp}'], input[value='{exp}']")
        if not target.is_selected():
            drv.execute_script("arguments[0].click();", target)

        time.sleep(4)   # wait for DataTable AJAX reload
        return True
    except Exception as e:
        print(f"  [select] {exp}: {e}")
        return False


def _read_all_rows(drv: webdriver.Chrome, exp: str) -> list[dict]:
    """
    Read ALL rows from #tblListaOpc, paginating through DataTable pages.

    Column mapping (0-indexed, DOM confirmed):
      0=ticker 1=tipo 2=fm 3=mod 4=strike 5=moneyness
      6=dist% 7=ultimo 8=var% 9=data 10=negocios 11=volume
      12=vol_impl 13=delta 14=gamma 15=theta_s 16=theta_pct 17=vega
    """
    rows_data: list[dict] = []
    try:
        WebDriverWait(drv, 12).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#tblListaOpc tbody tr"))
        )
    except Exception:
        print(f"  [table] Timeout waiting for #tblListaOpc for {exp}")
        return []

    page_num = 0
    while True:
        page_num += 1
        rows = drv.find_elements(By.CSS_SELECTOR, "#tblListaOpc tbody tr")
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) < 12:
                continue
            def c(i):
                return cells[i].text.strip() if i < len(cells) else ""

            ticker    = c(0)
            if not ticker or ticker.startswith("Nenhum") or ticker.startswith("No data"):
                continue
            rows_data.append({
                "ticker"    : ticker,
                "tipo"      : c(1).upper(),
                "fm"        : c(2),
                "mod"       : c(3),
                "strike"    : _to_float(c(4)),
                "moneyness" : c(5),
                "dist_pct"  : _to_float(c(6)),
                "ultimo"    : _to_float(c(7)),
                "var_pct"   : _to_float(c(8)),
                "data_hora" : c(9),
                "negocios"  : _to_float(c(10)) or 0,
                "volume"    : _to_float(c(11)) or 0.0,
                "vol_impl"  : _to_float(c(12)),       # e.g. 33.6
                "delta"     : _to_float(c(13)),
                "gamma"     : _to_float(c(14)),
                "theta"     : _to_float(c(15)),        # Theta($)
                # c(16) = Theta(%) — skip
                "vega"      : _to_float(c(17)) if len(cells) > 17 else None,
            })

        # Paginate
        try:
            nxt = drv.find_element(By.CSS_SELECTOR, "#tblListaOpc_next")
            if "disabled" in (nxt.get_attribute("class") or ""):
                break
            drv.execute_script("arguments[0].click();", nxt)
            time.sleep(1.5)
        except Exception:
            break

    print(f"  [table] {exp}: {len(rows_data)} rows (page {page_num})")
    return rows_data


def _spot_from_page(drv: webdriver.Chrome, ticker: str) -> Optional[float]:
    """Get current/last price from page or yfinance fallback."""
    try:
        spans = drv.find_elements(By.CSS_SELECTOR, ".cotacao-valor, [data-mkt-prop='p']")
        for sp in spans:
            v = _to_float(sp.text)
            if v and v > 0:
                return v
    except Exception:
        pass
    # Try the "mínima/máxima" area: look for R$ pattern in page source
    try:
        import re as r2
        src = drv.page_source
        # Find something like "R$ 39,33" in the price area
        m = r2.search(r"R\$\s*([\d\.]+,\d+)", src)
        if m:
            v = _to_float(m.group(1))
            if v and v > 0:
                return v
    except Exception:
        pass
    try:
        import yfinance as yf
        h = yf.Ticker(f"{ticker}.SA").history(period="5d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None


# ─── Public API ────────────────────────────────────────────────────────────────
def buscar_dados_opcoes(ticker: str, max_expirations: int = 4) -> pd.DataFrame:
    """
    Fetch B3 options for `ticker` from opcoes.net.br's rendered DataTable.

    The page loads full last-session data (Último, Vol.Impl, Delta, Gamma,
    Theta, Vega) via JavaScript by default — no login required.
    Uses headless Chrome so no window appears.
    """
    ticker_clean = ticker.replace(".SA", "").upper().strip()
    print(f"\n[opcoes] === {ticker_clean} ===")

    drv   = _make_driver()
    today = date.today()
    all_rows: list[dict] = []

    try:
        # ── 1. Open page & extract expirations ───────────────────────────────
        print(f"[opcoes] Abrindo {_PAGE.format(ticker=ticker_clean)}")
        drv.get(_PAGE.format(ticker=ticker_clean))
        time.sleep(5)

        exps   = _get_expirations(drv)
        future = [e for e in exps
                  if datetime.strptime(e, "%Y-%m-%d").date() > today][:max_expirations]
        if not future:
            print("[opcoes] Nenhum vencimento futuro.")
            return pd.DataFrame()

        # ── 2. Spot price ─────────────────────────────────────────────────────
        spot = _spot_from_page(drv, ticker_clean)
        print(f"[opcoes] Preço spot: {spot}")

        # ── 3. For each expiration: select checkbox → read table ──────────────
        for exp in future:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            days     = (exp_date - today).days
            if days <= 0:
                continue
            T  = days / 365.0
            ot_default = "c"

            if not _select_only(drv, exp):
                continue

            raw_rows = _read_all_rows(drv, exp)
            exp_ok   = 0

            for r in raw_rows:
                try:
                    opt_type = r["tipo"]
                    if opt_type not in ("CALL", "PUT"):
                        continue
                    strike = r["strike"]
                    if not strike or strike <= 0:
                        continue

                    ot = "c" if opt_type == "CALL" else "p"

                    # ── Greeks from DOM (real last-session data) ──────────────
                    vol_impl = r["vol_impl"]   # e.g. 33.6 (already decimal percent)
                    delta    = r["delta"]
                    gamma    = r["gamma"]
                    theta    = r["theta"]
                    vega     = r["vega"]
                    ultimo   = r["ultimo"]

                    # ── Fallback: compute via Black-Scholes ───────────────────
                    if vol_impl is None or delta is None:
                        if ultimo and ultimo > 0 and spot and spot > 0:
                            # Estimate IV via Newton-Raphson
                            r_ = 0.135
                            sigma = 0.35
                            for _ in range(50):
                                price_bs = _bs_price(spot, strike, T, r_, sigma, ot)
                                vega_bs  = spot * norm.pdf(
                                    (math.log(spot / strike) + (r_ + .5 * sigma**2) * T)
                                    / (sigma * math.sqrt(T))
                                ) * math.sqrt(T)
                                diff = price_bs - ultimo
                                if abs(diff) < 1e-5:
                                    break
                                sigma -= diff / max(vega_bs, 1e-8)
                                sigma = max(0.01, min(sigma, 5.0))
                            vol_impl = sigma * 100.0
                            delta, gamma, theta, vega = _bs_greeks(spot, strike, T, sigma, ot)
                        elif spot and spot > 0:
                            # No price either — use 30% IV estimate
                            sigma    = 0.30
                            vol_impl = 30.0
                            delta, gamma, theta, vega = _bs_greeks(spot, strike, T, sigma, ot)
                            ultimo   = _bs_price(spot, strike, T, 0.135, sigma, ot)

                    # Moneyness fallback
                    moneyness = r["moneyness"] or ""
                    if not moneyness and spot:
                        eps = 0.005
                        if opt_type == "CALL":
                            moneyness = ("OTM" if strike > spot * (1 + eps)
                                         else "ITM" if strike < spot * (1 - eps) else "ATM")
                        else:
                            moneyness = ("OTM" if strike < spot * (1 - eps)
                                         else "ITM" if strike > spot * (1 + eps) else "ATM")

                    all_rows.append({
                        "Ativo"    : r["ticker"],
                        "Tipo"     : opt_type,
                        "Mod"      : r["mod"],
                        "Moneyness": moneyness or "ATM",
                        "Strike"   : round(float(strike), 2),
                        "Ultimo"   : round(float(ultimo), 4) if ultimo else 0.0,
                        "Bid/Ask"  : "--",
                        "Negocios" : int(r["negocios"] or 0),
                        "Volume"   : float(r["volume"] or 0.0),
                        "Vencimento": exp,
                        "Dias"     : days,
                        "Vol_Impl" : round(float(vol_impl), 2) if vol_impl is not None else float("nan"),
                        "Delta"    : round(float(delta), 4) if delta is not None else float("nan"),
                        "Gama"     : round(float(gamma), 4) if gamma is not None else float("nan"),
                        "Theta"    : round(float(theta), 4) if theta is not None else float("nan"),
                        "Vega"     : round(float(vega), 4) if vega is not None else float("nan"),
                    })
                    exp_ok += 1
                except Exception:
                    continue

            print(f"  [ok] {exp}: {exp_ok} contratos processados")

    finally:
        drv.quit()
        print("[opcoes] Browser fechado.")

    if not all_rows:
        print(f"[opcoes] Nenhum dado para {ticker_clean}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["Vencimento", "Tipo", "Strike"]).reset_index(drop=True)
    for col in ["Strike", "Ultimo", "Volume", "Vol_Impl", "Delta", "Gama", "Theta", "Vega"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[opcoes] Total: {len(df)} contratos para {ticker_clean}.\n")
    return df
