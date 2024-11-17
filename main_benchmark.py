# %%
# 
# import os
from datetime import datetime
from dotenv import load_dotenv
from functions_benchmark.utils_AA import (
    fetch_performance_data,
    process_performance_data,
    AA_API_model_info,
    extract_evaluations,
)

def main():
    # Charger les variables d'environnement depuis .env
    base_path = os.path.abspath(os.path.dirname(__file__))
    env_path = os.path.join(base_path, ".env")
    load_dotenv(env_path)

    # Récupérer la clé API depuis .env
    APIkey = os.getenv("AA_key")
    if not APIkey:
        raise ValueError("Clé API 'AA_key' introuvable. Vérifiez votre fichier .env.")

    # Récupérer la date actuelle pour nommer le dossier
    today_date = datetime.today().strftime('%Y-%m-%d')
    output_dir = os.path.join(base_path, f'data/benchmark/AA/{today_date}')

    # Vérifier si le dossier existe déjà
    if os.path.exists(output_dir):
        print(f"Le dossier pour la date {today_date} existe déjà, arrêt de la génération.")
        return

    # Créer le répertoire pour la date actuelle
    os.makedirs(output_dir, exist_ok=True)

    # Étape 1 : Fetch performance data et traitement
    print("Récupération des données de performance...")
    performance_data = fetch_performance_data(APIkey)
    process_performance_data(performance_data, output_dir)

    # Étape 2 : Fetch et extraire les évaluations
    print("Extraction des évaluations...")
    api_response = AA_API_model_info(APIkey)
    extract_evaluations(api_response, output_dir)

    print(f"Traitement terminé. Données sauvegardées dans le répertoire : {output_dir}")

# Lancer le script
if __name__ == "__main__":
    main()

# %%
import os
from functions_benchmark.utils_Livebench import scrape_livebench

def main():
    # Définir le chemin de base du projet
    base_path = os.path.abspath(os.path.dirname(__file__))

    # Chemin pour sauvegarder les fichiers
    save_path = os.path.join(base_path, "data/benchmarks/Livebench")

    # URL de base pour Livebench
    base_url = 'https://livebench.ai/#/?slider='

    # Liste des sliders à scraper
    sliders = [0, 1, 2]

    # Lancer le scraping
    print("Démarrage du scraping Livebench...")
    scrape_livebench(sliders, base_url, save_path)

# Lancer le script
if __name__ == "__main__":
    main()

# %%
# %% Imports nécessaires
import os  # Pour gérer les chemins et répertoires
from functions_benchmark.utils_HF import process_pickle_file, collect_category_data  # Pour traitement des pickles et des catégories
from functions_benchmark.utils_HF import update_csv_from_leaderboards_Absolubench # Pour générer les CSV Hugging_Face

# Fonction principale pour traiter les pickles
def main():
    """
    Fonction principale pour traiter tous les fichiers pickle dans le dossier Arena_Elo et générer les CSV de catégorie.
    """
    # Obtenir le chemin absolu du répertoire du script actuel
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construire le chemin vers le répertoire Hugging_Face/Arena_Elo
    hugging_face_dir = os.path.join(base_dir, 'data', 'benchmark', 'Hugging_Face')
    arena_elo_dir = os.path.join(hugging_face_dir, 'Arena_Elo')
    # Vérifier si le dossier Arena_Elo existe
    if not os.path.exists(arena_elo_dir):
        print(f"Le dossier {arena_elo_dir} n'existe pas.")
        return
    # Obtenir tous les fichiers pickle commençant par 'elo_results_'
    pickle_files = [os.path.join(arena_elo_dir, f) for f in os.listdir(arena_elo_dir)
                    if f.startswith('elo_results_') and f.endswith('.pkl')]
    if not pickle_files:
        print("Aucun fichier 'elo_results_*.pkl' trouvé dans le dossier Arena_Elo.")
        return
    # Traiter chaque fichier pickle
    for file_path in pickle_files:
        print(f"Traitement du fichier : {file_path}")
        process_pickle_file(file_path, arena_elo_dir)
    # Après avoir traité tous les pickles, collecter les données de catégorie
    collect_category_data(arena_elo_dir)

# %% Imports nécessaires
import os
from functions_benchmark.utils_HF import complete_ae  # Importer la fonction complète AE
from functions_benchmark.utils_HF import update_csv_from_leaderboards_Absolubench

if __name__ == "__main__":
    # Définir les chemins pour les fichiers et répertoires
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.join(base_path, "data", "benchmark", "Hugging_Face", "Absolu_bench")
    output_dir = os.path.join(base_path, "data", "benchmark", "Hugging_Face")
    hf_ae_path = os.path.join(output_dir, "HF_AE.csv")
    text_full_hf_path = os.path.join(output_dir, "Arena_Elo", "category", "text_full_HF.csv")

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Lancer le traitement des fichiers leaderboard
    update_csv_from_leaderboards_Absolubench(input_dir, output_dir)

    # Compléter HF_AE.csv avec text_full_HF.csv
    complete_ae(hf_ae_path, text_full_hf_path)

# %%
