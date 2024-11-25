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
    json_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v1.json")
    models_infos_path = os.path.join(base_path, "data", "models_infos", "Perplexity", "Models_infos_Pplx.json")
    output_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v1.json")

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
# %% Imports nécessaires
import os
from function_prep.utils_prep_text import AIKoD_text_infos

if __name__ == "__main__":
    # Définir les chemins
    base_path = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v2.json")
    output_dir = os.path.join(base_path, "data", "models_infos")  # Répertoire contenant les fichiers _infos.csv

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
