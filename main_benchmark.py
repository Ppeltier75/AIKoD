# %%
# 
# import os
from datetime import datetime
from dotenv import load_dotenv
from function_benchmark.utils_AA import (
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
from function_benchmark.utils_Livebench import scrape_livebench

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
from function_benchmark.utils_HF import process_pickle_file, collect_category_data  # Pour traitement des pickles et des catégories
from function_benchmark.utils_HF import update_csv_from_leaderboards_Absolubench # Pour générer les CSV Hugging_Face

# Fonction principale pour traiter les pickles
def main():
    """
    Fonction principale pour traiter tous les fichiers pickle dans le dossier Arena_Elo et générer les CSV de catégorie.
    """
    # Obtenir le chemin absolu du répertoire du script actuel
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construire le chemin vers le répertoire Hugging_Face/Arena_Elo
    hugging_face_dir = os.path.join(base_dir, 'data', 'benchmark', 'HF')
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
from function_benchmark.utils_HF import complete_ae, update_csv_from_leaderboards_Absolubench, add_ratings_HF  # Importer la fonction complète AE

if __name__ == "__main__":
    # Définir les chemins pour les fichiers et répertoires
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.join(base_path, "data", "benchmark", "HF", "Absolu_bench")
    output_dir = os.path.join(base_path, "data", "benchmark", "HF")
    hf_ae_path = os.path.join(output_dir, "HF_text_AE.csv")
    text_full_hf_path = os.path.join(output_dir, "Arena_Elo", "category", "text_full_HF.csv")

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # # Lancer le traitement des fichiers leaderboard
    # update_csv_from_leaderboards_Absolubench(input_dir, output_dir)

    # # Compléter HF_AE.csv avec text_full_HF.csv
    # complete_ae(hf_ae_path, text_full_hf_path)

    # # Ajouter les ratings aux fichiers HF
    hf_directory = os.path.join(base_path, "data", "benchmark", "HF")
    add_ratings_HF(hf_directory)

# %%
from function_benchmark.utils_AA import scrappe_table_texttoimageAA
import os

base_path = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(base_path, "data/benchmark/AA/texttoimage")
csv_path = scrappe_table_texttoimageAA(output_dir)

if csv_path:
    print(f"CSV créé : {csv_path}")
else:
    print("Erreur lors de la création du CSV.")
# %%

from function_benchmark.utils_AA import scrappe_table_audiototextAA
import os

base_path = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(base_path, "data/benchmark/AA/audiototext")
csv_path = scrappe_table_audiototextAA(output_dir)

if csv_path:
    print(f"CSV créé : {csv_path}")
else:
    print("Erreur lors de la création du CSV.")
# %%
from function_benchmark.utils_AA import scrappe_table_texttoaudioAA
import os

base_path = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(base_path, "data/benchmark/AA/texttoaudio")
csv_path = scrappe_table_texttoaudioAA(output_dir)

if csv_path:
    print(f"CSV créé : {csv_path}")
else:
    print("Erreur lors de la création du CSV.")
# %%
from function_benchmark.utils_AA import scrappe_table_textAA
import os

base_path = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(base_path, "data/benchmark/AA/text")
csv_path = scrappe_table_textAA(output_dir)

if csv_path:
    print(f"CSV créé : {csv_path}")
else:
    print("Erreur lors de la création du CSV.")
# %%

import os 
from function_benchmark.utils_benchmark_id_name import update_model_names_AA

if __name__ == "__main__":
    # Définir les chemins d'entrée et de sortie
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.join(base_path, "data", "benchmark", "AA")
    output_dir = os.path.join(base_path, "data", "id_name", "benchmark", "AA")

    # Appeler la fonction pour analyser les modèles et générer les fichiers *_idname.csv
    update_model_names_AA(input_dir, output_dir)
# %%
import os 
from function_benchmark.utils_benchmark_id_name import update_model_names_HF_Livebench_EpochAI

if __name__ == "__main__":
    # Définir les chemins d'entrée et de sortie
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.join(base_path, "data", "benchmark", "HF")
    output_dir = os.path.join(base_path, "data", "id_name", "benchmark", "HF")

    # Appeler la fonction pour analyser les modèles et générer le fichier HF_text_idname.csv
    update_model_names_HF_Livebench_EpochAI(input_dir, output_dir)
# %%
import os 
from function_benchmark.utils_benchmark_id_name import update_model_names_HF_Livebench_EpochAI

if __name__ == "__main__":
    # Définir les chemins d'entrée et de sortie
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.join(base_path, "data", "benchmark", "Livebench")
    output_dir = os.path.join(base_path, "data", "id_name", "benchmark", "Livebench")

    # Appeler la fonction pour analyser les modèles et générer le fichier HF_text_idname.csv
    update_model_names_HF_Livebench_EpochAI(input_dir, output_dir)
# %%
import os 
from function_benchmark.utils_benchmark_id_name import update_model_names_HF_Livebench_EpochAI

if __name__ == "__main__":
    # Définir les chemins d'entrée et de sortie
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.join(base_path, "data", "models_infos", "EpochAI")
    output_dir = os.path.join(base_path, "data", "id_name", "models_infos", "EpochAI")

    # Appeler la fonction pour analyser les modèles et générer le fichier HF_text_idname.csv
    update_model_names_HF_Livebench_EpochAI(input_dir, output_dir)

# %%
import os
from dotenv import load_dotenv
from function_benchmark.utils_benchmark_id_name import Benchmark_update_id_names

def main():
    # Définir le chemin de base
    base_path = os.path.abspath(os.path.dirname(__file__))

    # Charger les variables d'environnement depuis .env
    env_path = os.path.join(base_path, ".env")
    load_dotenv(env_path)

    # Récupérer la clé API OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Clé API OpenAI introuvable. Vérifiez votre fichier .env.")

    # Répertoires pour les CSV et les fichiers exemples
    root_directory = os.path.join(base_path, "data", "id_name","benchmark", "AA")
    examples_directory = os.path.join(base_path, "data", "id_name", "exemple")

    # Appeler la fonction pour générer les id_name
    Benchmark_update_id_names(root_directory, examples_directory, openai_api_key, reset=True)


if __name__ == "__main__":
    main()
# %%
from function_benchmark.utils_benchmark_id_name import add_idname_benchmark

if __name__ == "__main__":
    base_path = os.path.abspath(os.path.dirname(__file__))
    benchmark_dir = os.path.join(base_path, "data", "benchmark")
    id_benchmark_dir = os.path.join(base_path, "data", "id_name", "benchmark")

    print("Ajout des id_name aux fichiers benchmark...")
    add_idname_benchmark(benchmark_dir, id_benchmark_dir, reset=True)   

# %%
from function_benchmark.utils_benchmark_id_name import add_id_name_benchmark_bis
import os 

if __name__ == "__main__":
    base_path = os.path.abspath(os.path.dirname(__file__))
    
    # Répertoire contenant les fichiers benchmark
    benchmark_dir = os.path.join(base_path, "data", "benchmark","AA")
    
    # Fichier CSV contenant les id_name
    id_name_csv = os.path.join(base_path, "data", "id_name", "benchmark", "AA", "AA_text_idname.csv")
    
    # Colonnes à rechercher pour le merge
    column_names = ["model", "model_name", "Model"]
    # Appeler la fonction
    add_id_name_benchmark_bis(benchmark_dir, id_name_csv, column_names, reset=True)

# %%
from function_utils.utils_id_name import generate_and_update_id_names
import os
from dotenv import load_dotenv

if __name__ == "__main__":

    base_path = os.path.abspath(os.path.dirname(__file__))
    # Charger les variables d'environnement depuis .env
    env_path = os.path.join(base_path, ".env")
    load_dotenv(env_path)

    # Récupérer la clé API OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    csv_path = os.path.join(base_path, "data", "models_infos", "AIOS", "AIOS_2024-10-08.csv")
    examples_csv_path = os.path.join(base_path, "data", "id_name", "exemple", "text_exemple.csv")
    model_type = "text"
    column_name ="name"
    generate_and_update_id_names(csv_path, examples_csv_path, model_type, openai_api_key, column_name)
# %%
import os 
from function_benchmark.utils_AA import correct_AA_benchmark
if __name__ == "__main__":
    correct_AA_benchmark()

# %%
import os 
from function_utils.utils_cleaning import clean_name_AA, convert_dirs_to_lowercase
# Chemin de base de votre projet
base_path = os.path.abspath(os.path.dirname(__file__))
# Chemin vers le fichier JSON généré par 'init_API'
directory_AA = os.path.join(base_path, "data", 'benchmark', 'AA', "2024-11-16")
# Exemple d'appel de la fonction
convert_dirs_to_lowercase(directory_AA)

from function_utils.utils_cleaning import  harmonize_company_name
from function_utils.utils_models_infos import column_name_modelsinfos, add_country_to_csv

# Chemin vers le fichier CSV
base_path = os.path.abspath(os.path.dirname(__file__))

texttoimage_path = os.path.join(base_path, "data", "benchmark", "AA", "texttoimage", "AA_texttoimage_2024-11-19.csv")  
audiototext_path = os.path.join(base_path, "data", "benchmark", "AA", "audiototext", "AA_audiototext_2024-11-19.csv")  



# Nom de la colonne à harmoniser
column_name = 'Provider'


# Appel de la fonction
harmonize_company_name(texttoimage_path , column_name)
harmonize_company_name(audiototext_path, column_name)