import json
from function_utils.utils_cleaning import clean_model_name
import os
import re
import pandas as pd

def add_model_type(json_path, output_path=None):
    """
    Ajoute un type à chaque modèle en fonction de ses modalités et unités, avec des catégories étendues.

    :param json_path: Chemin du fichier JSON d'entrée
    :param output_path: Chemin du fichier JSON de sortie avec les types de modèles ajoutés
    """
    # Charger les données JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Parcourir les données pour chaque provider et chaque modèle
    for provider, date_dict in data.items():
        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])

                for model in models:
                    modality_input = model.get("modality_input", [])
                    modality_output = model.get("modality_output", [])
                    unit_input = model.get("unit_input", [])
                    unit_output = model.get("unit_output", [])

                    # Convertir les modalités et unités en minuscule pour éviter les erreurs de casse
                    modality_input = [mod.lower() for mod in modality_input]
                    modality_output = [mod.lower() for mod in modality_output]
                    unit_input = [unit.lower() for unit in unit_input]
                    unit_output = [unit.lower() for unit in unit_output]

                    # Vérification spéciale pour `text` en input/output avec d'autres modalités
                    if (
                        "text" in modality_input and
                        "text" in modality_output and
                        any(mod not in ["text", "image", "audio"] for mod in modality_input + modality_output)
                    ):
                        model_type = "task"
                    elif any('api request' in unit for unit in unit_input + unit_output):
                        model_type = "task"
                    elif "embedding" in modality_output:
                        model_type = "embeddings"
                    elif ("text" in modality_input or "code" in modality_input) and \
                         ("text" in modality_output or "code" in modality_output) and \
                         ("audio" in modality_input or "audio" in modality_output or
                          "image" in modality_input or "image" in modality_output):
                        model_type = "multimodal"
                    elif ("text" in modality_input and "text" in modality_output):
                        model_type = "text"
                    elif ("text" in modality_input and "image" in modality_output):
                        model_type = "text to image"
                    elif ("image" in modality_input and "image" in modality_output):
                        model_type = "image to image"
                    elif ("image" in modality_input and "text" in modality_output):
                        model_type = "image to text"
                    elif ("audio" in modality_input and "audio" in modality_output):
                        model_type = "audio to audio"
                    elif ("audio" in modality_input and "text" in modality_output):
                        model_type = "audio to text"
                    elif ("text" in modality_input and "audio" in modality_output):
                        model_type = "text to audio"
                    else:
                        model_type = "unknown"

                    # Ajouter le type au modèle
                    model["type"] = model_type

    # Sauvegarder les données avec les types ajoutés dans un nouveau fichier JSON
    final_output_path = output_path if output_path else json_path
    with open(final_output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    print(f"Les types de modèles ont été ajoutés et sauvegardés dans {final_output_path}.")
    

def add_id_name_to_json_with_type(json_path, csv_dir, output_path=None):
    """
    Supprime tous les `id_name` existants, ajoute des `id_name` et réattribue les `type` pour les modèles
    en fonction des correspondances dans les fichiers CSV et des contraintes spécifiées.

    :param json_path: Chemin du fichier JSON à traiter.
    :param csv_dir: Chemin du dossier contenant les fichiers CSV.
    :param output_path: (Optionnel) Chemin du fichier JSON de sortie. Si non fourni, modifie le fichier d'entrée.
    """
    # Charger les données JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Charger les fichiers CSV et construire les correspondances
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith("_idname.csv")]
    name_to_id_by_type = {}

    for csv_file in csv_files:
        file_type = csv_file.replace("AIKoD_", "").replace("_idname.csv", "").lower()
        csv_path = os.path.join(csv_dir, csv_file)
        csv_data = pd.read_csv(csv_path)

        # Vérifier si le CSV contient les colonnes nécessaires
        if "name" not in csv_data.columns or "id_name" not in csv_data.columns:
            print(f"Colonne 'name' ou 'id_name' manquante dans {csv_file}. Ignoré.")
            continue

        # Construire le dictionnaire pour ce type en ignorant les lignes avec un id_name qui commence par 'unknown'
        name_to_id_by_type[file_type] = {
            clean_model_name(row["name"]): row["id_name"]
            for _, row in csv_data.iterrows()
            if pd.notna(row["id_name"]) and not row["id_name"].startswith("unknown")
        }

    # Parcourir les données JSON
    for provider, date_dict in data.items():
        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])

                for model in models:
                    # Supprimer tous les id_name existants
                    model["id_name"] = None

                    # Nettoyer le nom du modèle
                    cleaned_name = clean_model_name(model.get("name", ""))
                    model_type = model.get("type", "").replace(" ", "").lower()

                    # Vérifier la correspondance par type et ajouter un id_name
                    if model_type in name_to_id_by_type and cleaned_name in name_to_id_by_type[model_type]:
                        model["id_name"] = name_to_id_by_type[model_type][cleaned_name]

    # Sauvegarder les données avec les id_name mis à jour
    final_output_path = output_path if output_path else json_path

    with open(final_output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    print(f"Les id_name ont été mis à jour et enregistrés dans le fichier '{final_output_path}'.")

def add_date_release(json_path, models_infos_path, output_path=None):
    """
    Ajoute une date de publication (`date_release`) aux modèles dans le JSON brut
    en se basant sur les correspondances dans le fichier `models_infos_PPlx`.

    :param json_path: Chemin du fichier JSON brut contenant les modèles.
    :param models_infos_path: Chemin du fichier JSON `models_infos_PPlx` contenant les informations de date_release.
    :param output_path: Chemin du fichier JSON de sortie avec les dates ajoutées. Si non fourni, modifie le fichier d'entrée.
    """
    # Charger le fichier JSON brut
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Charger les informations des modèles avec date_release
    with open(models_infos_path, 'r', encoding='utf-8') as file:
        models_infos = json.load(file)

    # Construire un dictionnaire de correspondances {model_name: date_release}
    model_name_to_date = {
        info["model_name"].strip().lower(): info["date_release"]
        for info in models_infos
        if info.get("date_release") and info["date_release"] != "null"
    }

    # Ajouter la date_release aux modèles dans le JSON brut
    for provider, date_dict in data.items():
        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    # Vérifier si la date_release existe déjà
                    if "date_release" in model and model["date_release"]:
                        continue  # Ne pas écraser les dates existantes

                    # Chercher la date_release correspondante au model_name
                    model_name = model.get("name", "").strip().lower()
                    if model_name in model_name_to_date:
                        model["date_release"] = model_name_to_date[model_name]

    # Sauvegarder les modifications dans le fichier JSON
    final_output_path = output_path if output_path else json_path
    with open(final_output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    print(f"Les dates de publication ont été ajoutées et enregistrées dans le fichier '{final_output_path}'.")
