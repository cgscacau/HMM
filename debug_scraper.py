from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def debug_scraper():
    print("Iniciando scraper debug...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        url = "https://www.opcoes.net.br/opcoes/bovespa/PETR4"
        print(f"Acessando {url}...")
        driver.get(url)
        
        # Tirar um print da tela para ver o que o navegador está renderizando (captcha? erro 404? site diferente?)
        driver.save_screenshot("debug_opcoes.png")
        print("Screenshot salvo em debug_opcoes.png")
        
        # Imprimir o título da página
        print(f"Título da página: {driver.title}")
        
        # Procurar pela tabela com timeout de 15 segundos
        print("Aguardando tabela 'grid-opcoes'...")
        wait = WebDriverWait(driver, 15)
        
        # Vamos tentar ver se tem IFRAMES
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        print(f"Iframes encontrados na página: {len(iframes)}")
        
        tabela = wait.until(EC.presence_of_element_located((By.ID, "grid-opcoes")))
        print("Tabela encontrada com sucesso!")
        
        html = tabela.get_attribute('outerHTML')
        print(f"Tamanho do HTML da tabela: {len(html)} caracteres")
        
    except Exception as e:
        print(f"Erro durante a extração: {e}")
        # Se deu erro, salvar o HTML da pagina inteira pra investigar
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("HTML da página salvo em debug_page.html para análise.")
        
    finally:
        driver.quit()
        print("Driver finalizado.")

if __name__ == "__main__":
    debug_scraper()
