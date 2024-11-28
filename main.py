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

# %%
#processe pour add idname
import os
from dotenv import load_dotenv
from function_utils.utils_cleaning import remove_id_names_with_wrong_segments

base_path = os.path.abspath(os.path.dirname(__file__))
csv_path = os.path.join(base_path, "data", "id_name", "AIKoD", "AIKoD_text_idname.csv")
expected_segments = 9  # Vous pouvez modifier ce nombre selon vos besoins

remove_id_names_with_wrong_segments(csv_path, expected_segments)

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

# %%

import os
from function_prep.utils_prep_texttoimage import AIKoD_texttoimage_infos

if __name__ == "__main__":
    # Définir les chemins
    base_path = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v2.json")
    output_file = os.path.join(base_path, "data", "models_infos", "AIKoD_texttoimage_infos.csv")


    AIKoD_texttoimage_infos(json_path=json_path,output_file=output_file)


# %%
import os
from function_prep.utils_prep_audiototext import AIKoD_audiototext_infos

if __name__ == "__main__":
    # Définir les chemins
    base_path = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(base_path, "data", "raw", "AIKoD_brut_API_v2.json")
    base_csv_path= os.path.join(base_path, "data", "models_infos", "AIKoD_audiototext_infos.csv")


    AIKoD_audiototext_infos(json_path=json_path,base_csv_path=base_csv_path)

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

from function_utils.utils_cleaning import  harmonize_company_name
from function_utils.utils_models_infos import column_name_modelsinfos, add_country_to_csv

# Chemin vers le fichier CSV
base_path = os.path.abspath(os.path.dirname(__file__))

path_modelsinfos = os.path.join(base_path, "data", "models_infos")

text_path = os.path.join(base_path, "data", "models_infos", "AIKoD_text_infos.csv")  
texttoimage_path = os.path.join(base_path, "data", "models_infos", "AIKoD_texttoimage_infos.csv")  
audiototext_path = os.path.join(base_path, "data", "models_infos", "AIKoD_audiototext_infos.csv")  


# Nom de la colonne à harmoniser
column_name = 'company'


# Appel de la fonction
harmonize_company_name(text_path , column_name)
harmonize_company_name(texttoimage_path , column_name)
harmonize_company_name(audiototext_path, column_name)

column_name_modelsinfos(path_modelsinfos)


add_country_to_csv(text_path, column_name)
add_country_to_csv(texttoimage_path, column_name)
add_country_to_csv(audiototext_path, column_name)

# %% Imports nécessaires
from function_prep.utils_prep_texttoimage import create_adjusted_price_text_to_image

base_path = os.path.abspath(os.path.dirname(__file__))

# Chemin vers le répertoire contenant les fichiers 'texttoimage_priceoutput.csv'
directory_csv = os.path.join(base_path, "data", "pricing")

# Chemin vers le fichier AIKoD_texttoimage.csv
csv_path = os.path.join(base_path, "data", "models_infos", "AIKoD_texttoimage_pricing_infos.csv")

# Appel de la fonction
create_adjusted_price_text_to_image(directory_csv, csv_path)

# %% Imports nécessaires
from function_prep.utils_prep_texttoimage import reorganize_prices_by_resolution_and_steps

base_path = os.path.abspath(os.path.dirname(__file__))

# Chemin vers le répertoire contenant les fichiers 'texttoimage_priceoutput.csv'
input_csv = os.path.join(base_path, "data", "models_infos", "AIKoD_texttoimage_pricing_infos.csv")


# Appel de la fonction
reorganize_prices_by_resolution_and_steps(input_csv)

# %%

import os
from function_utils.utils_extract_pricing import extract_pricing_text

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
# Chemin vers le répertoire 'pricing'

import os 
from function_utils.utils_api import init_API

if __name__ == "__main__":
    # Définir le chemin de base et les répertoires
    base_path = os.path.abspath(os.path.dirname(__file__))
    output_json_path = os.path.join(base_path, "data", "API", "AIKoD_API_base_v0.json")
    pricing_dir = os.path.join(base_path, "data", "pricing")


# Appel de la fonction
init_API(pricing_dir, output_json_path)

# %%
import os 
from function_utils.utils_api import add_infos_to_API

base_path = os.path.abspath(os.path.dirname(__file__))

# Chemin vers le répertoire contenant les fichiers CSV 'AIKoD_{type}_infos.csv'
models_infos_directory = os.path.join(base_path, "data", "models_infos")

# Chemin vers le fichier JSON généré par 'init_API'
output_json_path= os.path.join(base_path, "data", "API", "AIKoD_API_base_v0.json")

# Appel de la fonction
add_infos_to_API(models_infos_directory, output_json_path)

# %%
# Chargement des données depuis le fichier JSON existant
import os
from function_utils.utils_api import generate_API_date

# Chemin de base de votre projet
base_path = os.path.abspath(os.path.dirname(__file__))

# Chemin vers le fichier JSON généré par 'init_API'
input_json_path = os.path.join(base_path, "data", "API", "AIKoD_API_base_v0.json")

# Chemin où vous souhaitez enregistrer le nouveau fichier JSON
output_json_path = os.path.join(base_path, "data", "API", "API_date_v3.json")

# Appel de la fonction
generate_API_date(input_json_path, output_json_path)

# %%
