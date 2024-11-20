import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



def scrape_table_livebench(driver, url, save_path):
    """
    Scrape un tableau de données depuis l'URL spécifiée et sauvegarde dans un fichier CSV.

    :param driver: Instance Selenium WebDriver.
    :param url: URL de la page à scraper.
    :param save_path: Répertoire où sauvegarder les fichiers CSV.
    """
    print(f"Chargement de l'URL : {url}")
    driver.get(url)

    try:
        # Attendre que le tableau soit chargé
        table_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.table-container'))
        )
        print("Tableau trouvé.")

        # Extraire la date à partir du div spécifique
        date_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div[style*="background-color: rgb(62, 62, 62);"]'))
        )
        date_text = date_element.text.strip()
        print(f"Date récupérée : {date_text}")

        # Localiser le tableau
        table = table_container.find_element(By.CSS_SELECTOR, 'table.main-tabl')
        rows = table.find_elements(By.TAG_NAME, 'tr')

        # Capturer les en-têtes spécifiés
        headers = [
            "Model", "Global Average", "Reasoning Average", "Coding Average",
            "Mathematics Average", "Data Analysis Average", "Language Average", "IF Average"
        ]
        table_data = [headers]

        # Extraire les lignes du tableau
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, 'td')
            cols = [col.text for col in cols]  # Extraire le texte de chaque cellule
            if cols:  # Ignorer les lignes vides
                table_data.append(cols)

        # Convertir en DataFrame
        df = pd.DataFrame(table_data[1:], columns=table_data[0])  # Ignorer la première ligne pour les données

        # Sauvegarder dans un fichier CSV
        file_name = f'Livebench_text_{date_text}.csv'
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, file_name)
        df.to_csv(file_path, index=False)
        print(f"Tableau sauvegardé : {file_path}")

    except Exception as e:
        print(f"Erreur lors du scraping : {e}")


def scrape_livebench(sliders, base_url, save_path):
    """
    Scrape les tableaux pour les différentes dates spécifiées via les sliders.

    :param sliders: Liste des indices de sliders à scraper.
    :param base_url: URL de base de la page Livebench.
    :param save_path: Répertoire où sauvegarder les fichiers CSV.
    """
    driver =  webdriver.Chrome()
    try:
        for slider in sliders:
            url = f"{base_url}{slider}"
            scrape_table_livebench(driver, url, save_path)
    finally:
        driver.quit()
        print("Scraping terminé.")
