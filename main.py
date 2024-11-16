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
output_dir = os.path.join(base_path, "data/id_name")

# Ajouter les types aux modèles
add_model_type(json_path, updated_json_path)

# Extraire les noms des modèles par type et les enregistrer dans des CSV
update_model_names_in_csv(updated_json_path, output_dir)



# %%
#processe pour add idname
import os
from dotenv import load_dotenv
from function_utils.utils_id_name import generate_and_update_id_names

# Charger les variables d'environnement depuis .env
load_dotenv()

# Récupérer la clé API OpenAI depuis .env
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("Clé API OpenAI introuvable. Vérifiez votre fichier .env.")

# Chemins des fichiers
csv_path = "AIKoD/data/id_name/AIKoD_multimodal_idname.csv"
examples_csv_path = "AIKoD/data/id_name/example/text_exemple.csv"
output_csv_path = "AIKoD/data/id_name/AIKoD_multimodal_idname_updated.csv"

# Type de modèle
model_type = "text"

# Appliquer la fonction pour générer et mettre à jour les id_name
print("Début de la mise à jour des `id_name`...")
added_models = generate_and_update_id_names(
    csv_path, examples_csv_path, model_type, openai_api_key, output_csv_path
)

# Afficher les modèles ajoutés
if added_models:
    print(f"Modèles ajoutés : {added_models}")
else:
    print("Aucun modèle ajouté.")
