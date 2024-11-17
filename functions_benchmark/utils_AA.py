import os
import requests
import pandas as pd
from datetime import datetime

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
