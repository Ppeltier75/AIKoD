import os
import requests
import pandas as pd
import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# Fonction pour récupérer les données de performance depuis l'API
def fetch_performance_data(APIkey, method="data/v1/llm/model-provider-endpoints", parallel_queries=1, prompt_length=1000, start_date="2024-04-01"):
    root_url = "https://artificialanalysis.ai/api/"
    myurl = f"{root_url}{method}?parallel_queries={parallel_queries}&prompt_length={prompt_length}&start_date={start_date}"
    
    headers = {
        "X-API-Key": APIkey
    }
    response = requests.get(myurl, headers=headers)
    content = response.json()
    return content

# Fonction pour récupérer les informations depuis l'API des évaluations
def AA_API_model_info(APIkey, method="data/v1/llm/models", parallel_queries=1, prompt_length=1000):
    root_url = "https://artificialanalysis.ai/api/"
    myurl = f"{root_url}{method}?parallel_queries={parallel_queries}&prompt_length={prompt_length}"
    
    headers = {
        "X-API-Key": APIkey
    }
    response = requests.get(myurl, headers=headers)
    content = response.json()
    
    return content

# Fonction pour traiter et sauvegarder les données de performance dans des fichiers CSV
def process_performance_data(api_response, base_path):
    provider_data = {}

    # Parcourir les données des modèles
    for model in api_response['data']:
        host_slug = model['host_slug']
        model_name = model['model_name']

        # Si le provider n'est pas encore dans provider_data, initialiser des listes vides
        if host_slug not in provider_data:
            provider_data[host_slug] = {
                'model_details': [],
                'evaluations': [],
                'pricing': [],
                'speed_performance': []
            }

        # Ajouter les informations pour ce modèle dans les listes correspondantes
        provider_data[host_slug]['model_details'].append({
            'model_name': model_name,
            **model['model_details']  # Ajout de toutes les informations de 'model_details'
        })

        provider_data[host_slug]['evaluations'].append({
            'model_name': model_name,
            **model['evaluations']  # Ajout de toutes les informations de 'evaluations'
        })

        provider_data[host_slug]['pricing'].append({
            'model_name': model_name,
            **model['pricing']  # Ajout de toutes les informations de 'pricing'
        })

        provider_data[host_slug]['speed_performance'].append({
            'model_name': model_name,
            'median_output_tokens_per_second': model['median_output_tokens_per_second'],
            'percentile_05_output_tokens_per_second': model['percentile_05_output_tokens_per_second'],
            'percentile_95_output_tokens_per_second': model['percentile_95_output_tokens_per_second'],
            'quartile_25_output_tokens_per_second': model['quartile_25_output_tokens_per_second'],
            'quartile_75_output_tokens_per_second': model['quartile_75_output_tokens_per_second'],
            'median_throughput_tokens_per_second': model['median_throughput_tokens_per_second'],
            'percentile_05_throughput_tokens_per_second': model['percentile_05_throughput_tokens_per_second'],
            'percentile_95_throughput_tokens_per_second': model['percentile_95_throughput_tokens_per_second'],
            'quartile_25_throughput_tokens_per_second': model['quartile_25_throughput_tokens_per_second'],
            'quartile_75_throughput_tokens_per_second': model['quartile_75_throughput_tokens_per_second'],
            'median_time_to_first_token_seconds': model['median_time_to_first_token_seconds'],
            'percentile_05_time_to_first_token_seconds': model['percentile_05_time_to_first_token_seconds'],
            'percentile_95_time_to_first_token_seconds': model['percentile_95_time_to_first_token_seconds'],
            'quartile_25_time_to_first_token_seconds': model['quartile_25_time_to_first_token_seconds'],
            'quartile_75_time_to_first_token_seconds': model['quartile_75_time_to_first_token_seconds'],
            'median_time_to_first_chunk_seconds': model['median_time_to_first_chunk_seconds'],
            'percentile_05_time_to_first_chunk_seconds': model['percentile_05_time_to_first_chunk_seconds'],
            'percentile_95_time_to_first_chunk_seconds': model['percentile_95_time_to_first_chunk_seconds'],
            'quartile_25_time_to_first_chunk_seconds': model['quartile_25_time_to_first_chunk_seconds'],
            'quartile_75_time_to_first_chunk_seconds': model['quartile_75_time_to_first_chunk_seconds'],
            'median_estimated_total_seconds_for_100_output_tokens': model['median_estimated_total_seconds_for_100_output_tokens'],
            'percentile_05_estimated_total_seconds_for_100_output_tokens': model['percentile_05_estimated_total_seconds_for_100_output_tokens'],
            'percentile_95_estimated_total_seconds_for_100_output_tokens': model['percentile_95_estimated_total_seconds_for_100_output_tokens'],
            'quartile_25_estimated_total_seconds_for_100_output_tokens': model['quartile_25_estimated_total_seconds_for_100_output_tokens'],
            'quartile_75_estimated_total_seconds_for_100_output_tokens': model['quartile_75_estimated_total_seconds_for_100_output_tokens']
        })

    # Sauvegarder les données dans les fichiers CSV pour chaque provider
    for host_slug, data in provider_data.items():
        provider_folder = os.path.join(base_path, host_slug)
        os.makedirs(provider_folder, exist_ok=True)

        # Sauvegarder model_details.csv
        pd.DataFrame(data['model_details']).to_csv(os.path.join(provider_folder, 'model_details.csv'), index=False)

        # Sauvegarder evaluations.csv
        pd.DataFrame(data['evaluations']).to_csv(os.path.join(provider_folder, 'evaluations.csv'), index=False)

        # Sauvegarder pricing.csv
        pd.DataFrame(data['pricing']).to_csv(os.path.join(provider_folder, 'pricing.csv'), index=False)

        # Sauvegarder speed_performance.csv
        pd.DataFrame(data['speed_performance']).to_csv(os.path.join(provider_folder, 'speed_performance.csv'), index=False)

        print(f"Fichiers CSV sauvegardés pour le provider {host_slug}")

# Fonction pour extraire les évaluations et créer un CSV
def extract_evaluations(api_response, base_path):
    # Liste pour stocker les informations extraites
    evaluations_data = []

    # Parcourir les données de l'API
    for model in api_response['data']:
        model_name = model['model_name']
        evaluations = model['evaluations']
        median_output_tokens_per_second = model['median_output_tokens_per_second']
        
        # Ajouter les informations dans une liste
        evaluations_data.append({
            'model_name': model_name,
            'chatbot_arena_elo': evaluations.get('chatbot_arena_elo'),
            'quality_index': evaluations.get('quality_index'),
            'mmlu': evaluations.get('mmlu'),
            'gpqa': evaluations.get('gpqa'),
            'humaneval': evaluations.get('humaneval'),
            'math': evaluations.get('math'),
            'mgsm': evaluations.get('mgsm'),
            'median_output_tokens_per_second': median_output_tokens_per_second
        })

    # Convertir en DataFrame
    df = pd.DataFrame(evaluations_data)

    # Sauvegarder dans un fichier CSV nommé 'AA_quality_date.csv'
    file_name = f'AA_quality_{datetime.today().strftime("%Y-%m-%d")}.csv'
    df.to_csv(os.path.join(base_path, file_name), index=False)
    print(f"Fichier CSV d'évaluations sauvegardé : {file_name}")



def scrappe_table_texttoimageAA(output_dir):
    """
    Scrape a text-to-image metrics table and save the data to a CSV file.

    Args:
        output_dir (str): The output directory where the CSV file will be saved.

    Returns:
        str: Path to the created CSV file.
    """
    # Define the URL to scrape
    url = "https://artificialanalysis.ai/text-to-image"
    
    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        # Open the URL
        driver.get(url)
        
        # Locate the table
        table_xpath = '/html/body/div/main/div[3]/div[2]/div[4]/div[2]/div/div/div/table'
        table = driver.find_element(By.XPATH, table_xpath)
        
        # Locate all rows within the table body
        rows = table.find_elements(By.XPATH, './tbody/tr')
        
        # Initialize an empty list to hold the table data
        table_data = []

        # Loop through each row and extract cell data
        for row in rows:
            cells = row.find_elements(By.XPATH, './td')
            # Extract data from each cell
            if len(cells) >= 6:  # Ensure row has the expected number of cells
                data = {
                    "Provider": cells[0].text.strip(),
                    "Model": cells[1].text.strip(),
                    "Footnotes": cells[2].text.strip(),
                    "Model Quality ELO": cells[3].text.strip(),
                    "Median Generation Time (s)": cells[4].text.strip(),
                    "Price per 1k Images (USD)": cells[5].text.strip()
                }
                table_data.append(data)

        # Get the current date to name the CSV file
        current_date = datetime.now().strftime("%Y-%m-%d")
        csv_filename = f"texttoimage_{current_date}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save data to CSV
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=table_data[0].keys())
            writer.writeheader()
            writer.writerows(table_data)

        print(f"Data successfully saved to {csv_path}")
        return csv_path

    except Exception as e:
        print(f"An error occurred while scraping the table: {e}")
        return None

    finally:
        # Close the driver
        driver.quit()


def scrappe_table_speechtotextAA(output_dir):
    """
    Scrape a speech-to-text metrics table and save the data to a CSV file.

    Args:
        base_path (str): The base directory where the 'AA/speechtotext' folder will be created.

    Returns:
        str: Path to the created CSV file.
    """
    # Define the URL to scrape
    url = "https://artificialanalysis.ai/speech-to-text"
    
    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        # Open the URL
        driver.get(url)
        
        # Locate the table
        table_xpath = '/html/body/div/main/div[3]/div[2]/div[4]/div[2]/div/div/div/table'
        table = driver.find_element(By.XPATH, table_xpath)
        
        # Locate all rows within the table body
        rows = table.find_elements(By.XPATH, './tbody/tr')
        
        # Initialize an empty list to hold the table data
        table_data = []

        # Loop through each row and extract cell data
        for row in rows:
            cells = row.find_elements(By.XPATH, './td')
            # Extract data from each cell
            if len(cells) >= 7:  # Ensure row has the expected number of cells
                data = {
                    "Provider": cells[0].text.strip(),
                    "Model": cells[1].text.strip(),
                    "Whisper Version": cells[2].text.strip(),
                    "Footnotes": cells[3].text.strip(),
                    "Word Error Rate (%)": cells[4].text.strip(),
                    "Median Speed Factor": cells[5].text.strip(),
                    "Price (USD per 1000 Minutes)": cells[6].text.strip(),
                }
                table_data.append(data)

        # Get the current date to name the CSV file
        current_date = datetime.now().strftime("%Y-%m-%d")
        csv_filename = f"speechtotext_{current_date}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save data to CSV
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=table_data[0].keys())
            writer.writeheader()
            writer.writerows(table_data)

        print(f"Data successfully saved to {csv_path}")
        return csv_path

    except Exception as e:
        print(f"An error occurred while scraping the table: {e}")
        return None

    finally:
        # Close the driver
        driver.quit()

def scrappe_table_texttospeechAA(output_dir):
    """
    Scrape a text-to-speech metrics table and save the data to a CSV file.

    Args:
        base_path (str): The base directory where the 'AA/texttospeech' folder will be created.

    Returns:
        str: Path to the created CSV file.
    """
    # Define the URL to scrape
    url = "https://artificialanalysis.ai/text-to-speech"
    
    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        # Open the URL
        driver.get(url)
        
        # Locate the table
        table_xpath = '/html/body/div/main/div[3]/div[2]/div[4]/div[2]/div/div/div/table'
        table = driver.find_element(By.XPATH, table_xpath)
        
        # Locate all rows within the table body
        rows = table.find_elements(By.XPATH, './tbody/tr')
        
        # Initialize an empty list to hold the table data
        table_data = []

        # Loop through each row and extract cell data
        for row in rows:
            cells = row.find_elements(By.XPATH, './td')
            # Extract data from each cell
            if len(cells) >= 7:  # Ensure row has the expected number of cells
                data = {
                    "Provider": cells[0].text.strip(),
                    "Model": cells[1].text.strip(),
                    "Streaming Support": cells[2].text.strip(),
                    "Footnotes": cells[3].text.strip(),
                    "Model Arena ELO": cells[4].text.strip(),
                    "Characters Per Second": cells[5].text.strip(),
                    "Price Per 1M Characters (USD)": cells[6].text.strip(),
                }
                table_data.append(data)

        # Get the current date to name the CSV file
        current_date = datetime.now().strftime("%Y-%m-%d")
        csv_filename = f"texttospeech_{current_date}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save data to CSV
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=table_data[0].keys())
            writer.writeheader()
            writer.writerows(table_data)

        print(f"Data successfully saved to {csv_path}")
        return csv_path

    except Exception as e:
        print(f"An error occurred while scraping the table: {e}")
        return None

    finally:
        # Close the driver
        driver.quit()


def scrappe_table_textAA(output_dir):
    """
    Scrape a leaderboards models metrics table and save the data to a CSV file.

    Args:
        base_path (str): The base directory where the 'AA/text' folder will be created.

    Returns:
        str: Path to the created CSV file.
    """
    # Define the URL to scrape
    url = "https://artificialanalysis.ai/leaderboards/models"
    
    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        # Open the URL
        driver.get(url)

        # Wait for the page to load
        time.sleep(3)  # Adjust as necessary based on loading time
        
        # Locate the table
        table_xpath = '/html/body/div/main/div[3]/div[2]/div[1]/div/div[2]/div/table'
        table = driver.find_element(By.XPATH, table_xpath)
        
        # Locate all rows within the table body
        rows = table.find_elements(By.XPATH, './tbody/tr')
        
        # Initialize an empty list to hold the table data
        table_data = []

        # Loop through each row and extract cell data
        for row in rows:
            cells = row.find_elements(By.XPATH, './td')
            # Extract data from each cell using nested div for text
            if len(cells) >= 7:  # Ensure row has the expected number of cells
                data = {
                    "Model": cells[0].text.strip(),
                    "Creator": cells[1].text.strip(),
                    "Context Window": cells[2].find_element(By.XPATH, './div').text.strip(),
                    "Quality Index (Normalized avg)": cells[3].find_element(By.XPATH, './div').text.strip(),
                    "Blended Price (USD/1M Tokens)": cells[4].find_element(By.XPATH, './div').text.strip(),
                    "Output Tokens/S Median": cells[5].find_element(By.XPATH, './div').text.strip(),
                    "Latency Median (First Chunk)": cells[6].find_element(By.XPATH, './div').text.strip(),
                }
                table_data.append(data)

        # Get the current date to name the CSV file
        current_date = datetime.now().strftime("%Y-%m-%d")
        csv_filename = f"text_{current_date}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save data to CSV
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=table_data[0].keys())
            writer.writeheader()
            writer.writerows(table_data)

        print(f"Data successfully saved to {csv_path}")
        return csv_path

    except Exception as e:
        print(f"An error occurred while scraping the table: {e}")
        return None

    finally:
        # Close the driver
        driver.quit()