import pandas as pd
import os
import json
from collections import Counter
# Importation des fonctions depuis merge_utils.py
from function_utils.utils_merge_id import select_specific_segments, select_segments_no_order, merge_csv_id_name

def add_csv_audio_to_text(base_csv_path):
    """
    Adds audio-to-text related columns to the base CSV file by merging data from the specified CSV file.
    The function reads the base CSV, merges additional columns, and saves the updated DataFrame.

    :param base_csv_path: Path to the base CSV file.
    """
    # Définition des stratégies de correspondance
    strategies = [
        lambda x: x,  # Correspondance exacte
        lambda x: select_specific_segments(x, [1, 2, 4]),
        lambda x: select_segments_no_order(x, [1, 2, 4]),
        lambda x: select_specific_segments(x, [1, 2, 3]),
        lambda x: select_segments_no_order(x, [1, 2, 3]),
        lambda x: select_specific_segments(x, [1, 4]),
        lambda x: select_segments_no_order(x, [1, 4]),
        # Vous pouvez ajouter d'autres stratégies si nécessaire
    ]

    # Lecture du fichier de base
    df_base = pd.read_csv(base_csv_path)

    # Création d'une copie du DataFrame de base pour les fusions successives
    df_merged = df_base.copy()

    # Chemin vers le fichier à fusionner
    audio_to_text_csv = r'C:\Users\piwip\OneDrive\Documents\OCDE\AIKoD\data\benchmark\AA\audiototext\AA_audiototext_2024-11-19.csv'

    # Lecture du fichier à fusionner
    df_merge = pd.read_csv(audio_to_text_csv)

    # Colonnes à conserver
    keep_columns = ['Word Error Rate (%)', 'Median Speed Factor']

    # Fusion
    df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)

    # Enregistrement du DataFrame fusionné
    # Définition du chemin de sortie (vous pouvez le modifier si nécessaire)
    output_csv_path = os.path.splitext(base_csv_path)[0] + '.csv'

    df_merged.to_csv(output_csv_path, index=False)

    print(f"Le fichier fusionné a été enregistré sous {output_csv_path}")

    return df_merged

def AIKoD_audiototext_infos(json_path, base_csv_path):
    """
    Met à jour un fichier CSV avec des informations extraites d'un JSON et fusionne avec d'autres fichiers CSV
    en utilisant add_csv_audio_to_text.

    :param json_path: Chemin vers le fichier JSON contenant les données des modèles.
    :param base_csv_path: Chemin vers le fichier CSV à mettre à jour.
    """
    # Charger les données JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Charger le fichier CSV existant
    base_df = pd.read_csv(base_csv_path)

    # Parcourir les modèles dans le JSON
    id_name_to_info = {}
    for provider, date_dict in data.items():
        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    # Vérifier que le type est 'audio_to_text'
                    if model.get("type") == "audio to text" and "id_name" in model and model["id_name"]:
                        id_name = model["id_name"]
                        company = model.get("company", None)
                        date_release = model.get("date_release", None)

                        id_name_to_info.setdefault(id_name, {
                            "company": [],
                            "date_release": []
                        })

                        if company:
                            id_name_to_info[id_name]["company"].append(company)
                        if date_release:
                            id_name_to_info[id_name]["date_release"].append(date_release)

    # Analyser les informations majoritaires et compléter les id_name
    rows_to_update = []
    for id_name, info in id_name_to_info.items():
        # Calculer les valeurs majoritaires
        company = (
            Counter(info["company"]).most_common(1)[0][0]
            if info["company"]
            else None
        )
        date_release = (
            Counter([str(d) for d in info["date_release"] if isinstance(d, (str, int, float))]).most_common(1)[0][0]
            if info.get("date_release") and info["date_release"]
            else None
        )

        # Ajouter les informations pour cet id_name
        rows_to_update.append({
            "id_name": id_name,
            "company": company,
            "date_release": date_release
        })

    # Créer un DataFrame avec les mises à jour
    updates_df = pd.DataFrame(rows_to_update)

    # Fusionner les mises à jour avec le DataFrame existant
    base_df = pd.merge(
        base_df,
        updates_df,
        on="id_name",
        how="left",
        suffixes=("", "_new"),
    )

    # Mettre à jour les colonnes avec les nouvelles valeurs
    for col in ["company", "date_release"]:
        if f"{col}_new" in base_df.columns:
            base_df[col] = base_df[f"{col}_new"].combine_first(base_df[col])
            base_df.drop(columns=[f"{col}_new"], inplace=True)

    # Sauvegarder le DataFrame mis à jour temporairement
    temp_csv_path = base_csv_path.replace('.csv', '_temp.csv')
    base_df.to_csv(temp_csv_path, index=False)

    # Utiliser add_csv_audio_to_text pour effectuer les fusions supplémentaires
    df_final = add_csv_audio_to_text(temp_csv_path)

    # Enregistrer le DataFrame final à l'emplacement d'origine
    df_final.to_csv(base_csv_path, index=False)
    print(f"Le fichier {base_csv_path} a été mis à jour avec les informations audio-to-text.")

    # Supprimer le fichier temporaire
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

    return df_final
