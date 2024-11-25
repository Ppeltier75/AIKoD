import os
import pandas as pd
import json
from fuzzywuzzy import fuzz

from function_utils.utils_merge_id import merge_with_flexibility  # Importez votre fonction de fusion flexible

def AIKoD_texttoimage_infos(json_path, output_file, merge_files=None, merge_columns='all', merge_strategy='exact', 
                            segments_order=None, segments_no_order=None, fuzzy_match=False, fuzzy_threshold=85):
    """
    Met à jour un fichier CSV avec des informations provenant d'un JSON et fusionne les fichiers spécifiés.

    :param json_path: Chemin vers le fichier JSON d'entrée.
    :param output_file: Chemin vers le fichier CSV de sortie.
    :param merge_files: Liste de fichiers à fusionner avec le CSV de sortie.
    :param merge_columns: Colonnes à garder lors du merge ('all' pour tout sauf les colonnes contenant 'name').
    :param merge_strategy: Stratégie de merge ('exact', 'partial', 'no_order').
    :param segments_order: Segments à utiliser pour la correspondance partielle ordonnée.
    :param segments_no_order: Segments à utiliser pour la correspondance sans ordre.
    :param fuzzy_match: Active ou désactive la correspondance floue.
    :param fuzzy_threshold: Score minimal pour une correspondance floue.
    """
    # Charger le JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extraire les informations des modèles de type 'image'
    image_data = []
    for provider, date_dict in data.items():
        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    if model.get("type") == "image":
                        model_name = model.get("name", "").strip()
                        id_name = model.get("id_name", "").strip()
                        date_release = model.get("date_release", date_str)
                        image_data.append({"name": model_name, "id_name": id_name, "date_release": date_release})

    # Charger ou créer le fichier CSV de sortie
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
    else:
        existing_df = pd.DataFrame(columns=["name", "id_name", "date_release"])

    # Ajouter les nouvelles informations au DataFrame existant
    image_df = pd.DataFrame(image_data)
    combined_df = pd.concat([existing_df, image_df]).drop_duplicates(subset=["id_name"], keep="first")

    # Sauvegarder le fichier mis à jour
    combined_df.to_csv(output_file, index=False)
    print(f"Fichier mis à jour : {output_file}")

    # Si des fichiers à fusionner sont spécifiés
    if merge_files:
        for merge_file in merge_files:
            if not os.path.exists(merge_file):
                print(f"Fichier à fusionner introuvable : {merge_file}")
                continue

            merge_df = pd.read_csv(merge_file)

            # Filtrer les colonnes si 'all' n'est pas spécifié
            if merge_columns != 'all':
                merge_df = merge_df[["id_name"] + [col for col in merge_df.columns if col in merge_columns]]

            # Appliquer la stratégie de fusion flexible
            combined_df = merge_with_flexibility(
                base_df=combined_df,
                merge_df=merge_df,
                strategy=merge_strategy,
                segments_order=segments_order,
                segments_no_order=segments_no_order,
                fuzzy_match=fuzzy_match,
                fuzzy_threshold=fuzzy_threshold
            )

        # Sauvegarder le fichier fusionné
        combined_df.to_csv(output_file, index=False)
        print(f"Fichier fusionné sauvegardé : {output_file}")
