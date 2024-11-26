# %%
# process pour extraire model_names par type
import os
import sys

from function_utils.utils_add_infos import add_model_type
from function_utils.utils_id_name import extract_names_by_type, update_model_names_in_csv

# Définir le chemin de base du projet
base_path = os.path.abspath(os.path.dirname(__file__))

# Construire les chemins absolus
json_path = os.path.join(base_path, "data/raw/AIKoD_brut_API_v1.json")
updated_json_path = os.path.join(base_path, "data/raw/AIKoD_brut_API_v1.json")
output_dir = os.path.join(base_path, "data/id_name/AIKoD")

# Ajouter les types aux modèles
add_model_type(json_path, updated_json_path)

# Extraire les noms des modèles par type et les enregistrer dans des CSV
update_model_names_in_csv(updated_json_path, output_dir)

# %%
import json 
import os 
from function_utils.utils_add_infos import add_date_release
if __name__ == "__main__":
    base_path = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v2.json")
    models_infos_path = os.path.join(base_path, "data", "models_infos", "Perplexity", "Models_infos_Pplx.json")
    output_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v2.json")

    # Appeler la fonction pour ajouter les dates de publication
    add_date_release(json_path, models_infos_path, output_path)


# %%
#processe pour add idname
import os
from dotenv import load_dotenv
from function_utils.utils_id_name import AIKoD_update_id_names

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
    csv_directory = os.path.join(base_path, "data", "id_name", "AIKoD")
    examples_directory = os.path.join(base_path, "data", "id_name", "exemple")

    # Appeler la fonction pour générer les id_name
    AIKoD_update_id_names(csv_directory, examples_directory, openai_api_key)

if __name__ == "__main__":
    main()


# %% Imports nécessaires
import os
from function_utils.utils_add_infos import add_id_name_to_json_with_type

if __name__ == "__main__":
    # Définir les chemins
    base_path = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v2.json")
    csv_dir = os.path.join(base_path, "data", "id_name", "AIKoD")
    
    # Si vous souhaitez fournir un chemin de sortie :
    output_json_path = None  # Changez ceci pour un chemin spécifique si nécessaire

    # Ajouter les id_name au JSON
    add_id_name_to_json_with_type(json_path, csv_dir, output_json_path)

# %%
import os
from dotenv import load_dotenv
from function_utils.utils_models_infos import generate_csv_with_infos

# Définir le chemin de base du projet
base_path = os.path.abspath(os.path.dirname(__file__))

# Charger les variables d'environnement depuis .env
env_path = os.path.join(base_path, ".env")
load_dotenv(env_path)

# Construire les chemins absolus pour les répertoires d'entrée et de sortie
input_dir = os.path.join(base_path, "data/id_name/AIKoD")  # Répertoire contenant les fichiers _idname.csv
output_dir = os.path.join(base_path, "data/models_infos")  # Répertoire où les fichiers _infos.csv seront générés

# Appeler la fonction pour générer les fichiers _infos.csv
generate_csv_with_infos(input_dir, output_dir)


# %%

import os
from function_prep.utils_prep_texttoimage import AIKoD_texttoimage_infos

if __name__ == "__main__":
    # Définir les chemins
    base_path = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v2.json")
    output_file = os.path.join(base_path, "data", "models_infos", "AIKoD_texttoimage_infos.csv")
    merge_file = os.path.join(base_path, "data", "models_infos", "AA", "AA_texttoimage_infos.csv")

    # Vérification des fichiers
    for file_path in [json_path, output_file, merge_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

    # Appeler la fonction pour mettre à jour et fusionner les informations
    try:
        print("Mise à jour et fusion des fichiers en cours...")
        AIKoD_texttoimage_infos(
            json_path=json_path,
            output_file=output_file,
            merge_file=merge_file,
        )
        print("Mise à jour et fusion terminées avec succès.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

# %%
import os
from function_prep.utils_prep_text import add_rating_text

if __name__ == "__main__":
    # Définir les chemins
    base_path = os.path.abspath(os.path.dirname(__file__))
    text_file = os.path.join(base_path, "data", "models_infos", "AIKoD_text_infos.csv")
    rating_file_1 = os.path.join(base_path, "data", "benchmark", "AA", "2024-11-16", "AA_quality_2024-11-16.csv")
    rating_file_2 = os.path.join(base_path, "data", "benchmark", "Livebench", "Livebench_text_2024-08-31.csv")
    output_file = os.path.join(base_path, "data", "models_infos", "AIKoD_text_infos_with_ratings.csv")

    # Appeler la fonction pour ajouter les notations
    add_rating_text(
        text_file=text_file,
        rating_file_1=rating_file_1,
        rating_file_2=rating_file_2,
        output_file=output_file
    )


# %% Imports nécessaires
import os
from function_prep.utils_prep_text import AIKoD_text_infos

if __name__ == "__main__":
    # Définir les chemins
    base_path = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v2.json")
    output_dir = os.path.join(base_path, "data", "models_infos", "AIKoD_text_infos.csv")  # Répertoire contenant les fichiers _infos.csv

    # Appeler la fonction pour mettre à jour AIKoD_text_infos.csv
    AIKoD_text_infos(json_path, output_dir)




# %%

import os
from function_utils.utils_extract_pricing import extract_pricing_text
from function_utils.utils_extract_pricing import add_id_name_to_pricing_files

if __name__ == "__main__":
    # Définir le chemin de base et les répertoires
    base_path = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v2.json")
    pricing_dir = os.path.join(base_path, "data", "pricing")

    # Appeler la fonction pour extraire les prix
    print("Extraction des prix en cours...")
    extract_pricing_text(json_path, pricing_dir)
    print("Extraction terminée.")



# %%
import os 
from function_utils.utils_extract_pricing import extract_pricing_audiototext

if __name__ == "__main__":
    # Définir le chemin de base et les répertoires
    base_path = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v2.json")
    pricing_dir = os.path.join(base_path, "data", "pricing")

    # Extraire les prix pour les modèles audio to text
    print("Extraction des prix pour les modèles audio to text en cours...")
    extract_pricing_audiototext(json_path, pricing_dir)
    print("Extraction terminée.")

# %%

import os 
from function_utils.utils_extract_pricing import extract_pricing_texttoimage

if __name__ == "__main__":
    # Définir le chemin de base et les répertoires
    base_path = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v2.json")
    pricing_dir = os.path.join(base_path, "data", "pricing")

    # Extraire les prix pour les modèles text to image
    print("Extraction des prix pour les modèles text to image en cours...")
    extract_pricing_texttoimage(json_path, pricing_dir)
    print("Extraction terminée.")

# %%
