import pandas as pd
import json
import os
from collections import Counter

# Importation des fonctions depuis merge_utils.py
from function_utils.utils_merge_id import select_specific_segments, select_segments_no_order, merge_csv_id_name

def extract_average_resolution(resolution_segment):
    """
    Extrait la résolution moyenne d'un segment de type '1024x1024'.
    """
    try:
        resolutions = resolution_segment.split('x')
        resolutions = [int(res) for res in resolutions if res.isdigit()]
        if resolutions:
            return sum(resolutions) / len(resolutions)
        else:
            return None
    except:
        return None


def extract_steps(steps_segment):
    """
    Extrait les steps d'un segment.
    """
    try:
        if steps_segment.isdigit():
            return int(steps_segment)
        else:
            return None
    except:
        return None


def extract_earliest_date(json_data, id_name):
    """
    Trouve la date la plus ancienne pour un id_name donné dans un fichier JSON.
    """
    earliest_date = None
    for provider_data in json_data.values():
        if not isinstance(provider_data, dict):  # Vérifie que provider_data est un dictionnaire
            continue
        for date_str, content in provider_data.items():
            if not isinstance(content, dict):  # Vérifie que content est un dictionnaire
                continue
            models = content.get("models_extract_GPT4o", {}).get("models", [])
            for model in models:
                if model.get("id_name") == id_name:
                    if earliest_date is None or date_str < earliest_date:
                        earliest_date = date_str
    return earliest_date

def extract_earliest_date(json_data, id_name):
    """
    Trouve la date la plus ancienne pour un id_name donné dans un fichier JSON.
    """
    earliest_date = None
    for provider_data in json_data.values():
        if not isinstance(provider_data, dict):
            continue
        for date_str, content in provider_data.items():
            if not isinstance(content, dict):
                continue
            models = content.get("models_extract_GPT4o", {}).get("models", [])
            for model in models:
                if model.get("id_name") == id_name:
                    if earliest_date is None or date_str < earliest_date:
                        earliest_date = date_str
    return earliest_date


def extract_average_resolution(resolution_segment):
    """
    Extrait la résolution moyenne d'un segment de type '1024x1024'.
    """
    try:
        resolutions = resolution_segment.split('x')
        resolutions = [int(res) for res in resolutions if res.isdigit()]
        if resolutions:
            return sum(resolutions) / len(resolutions)
        else:
            return None
    except:
        return None


def extract_steps(steps_segment):
    """
    Extrait les steps d'un segment.
    """
    try:
        if steps_segment.isdigit():
            return int(steps_segment)
        else:
            return None
    except:
        return None


import pandas as pd
import os



def add_csv_texttoimage(base_csv_path):
    """
    Adds image-related columns to the base CSV file by merging data from specified CSV files.
    The function reads the base CSV, merges additional columns, and saves the updated DataFrame.

    :param base_csv_path: Path to the base CSV file.
    """
    # Définition des stratégies de correspondance
    strategies = [
        lambda x: x,  # Correspondance exacte
        lambda x: select_specific_segments(x, [1, 2, 3, 4, 7]),
        lambda x: select_segments_no_order(x, [1, 2, 3, 4, 7]),
        lambda x: select_specific_segments(x, [1, 2, 3, 4]),
        lambda x: select_segments_no_order(x, [1, 2, 3, 4]),
        lambda x: select_specific_segments(x, [1, 2, 4]),
        lambda x: select_segments_no_order(x, [1, 2, 4]),
        lambda x: select_specific_segments(x, [1, 4]),
        lambda x: select_segments_no_order(x, [1, 4]),
        lambda x: select_specific_segments(x, [1, 2]),
        lambda x: select_segments_no_order(x, [1, 2]),
        # Vous pouvez ajouter d'autres stratégies si nécessaire
    ]

    # Lecture du fichier de base
    df_base = pd.read_csv(base_csv_path)

    # Création d'une copie du DataFrame de base pour les fusions successives
    df_merged = df_base.copy()

    # Chemins vers les fichiers à fusionner
    paths = {
        'AA_texttoimage': r'C:\Users\piwip\OneDrive\Documents\OCDE\AIKoD\data\benchmark\AA\texttoimage\AA_texttoimage_2024-11-19.csv',
        'AA_texttoimage_infos': r'C:\Users\piwip\OneDrive\Documents\OCDE\AIKoD\data\models_infos\AA\AA_texttoimage_infos.csv',
    }

    # Fusion avec AA_texttoimage_2024-11-19.csv
    df_merge = pd.read_csv(paths['AA_texttoimage'])

    # Colonnes à conserver
    keep_columns = ['Model Quality ELO', 'Median Generation Time (s)']

    # Fusion
    df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)

    # Fusion avec AA_texttoimage_infos.csv
    df_merge = pd.read_csv(paths['AA_texttoimage_infos'])

    keep_columns = ['Default Steps', 'Default Resolution']

    df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)

    # Enregistrement du DataFrame fusionné
    # Définition du chemin de sortie (vous pouvez le modifier si nécessaire)
    output_csv_path = os.path.splitext(base_csv_path)[0] + '.csv'

    df_merged.to_csv(output_csv_path, index=False)

    print(f"Le fichier fusionné a été enregistré sous {output_csv_path}")

    return df_merged

def AIKoD_texttoimage_infos(json_path, output_file):
    """
    Met à jour un fichier CSV avec des informations extraites d'un JSON et fusionne avec d'autres fichiers CSV
    en utilisant add_csv_image.

    :param json_path: Chemin vers le fichier JSON contenant les données des modèles.
    :param output_file: Chemin vers le fichier CSV à mettre à jour.
    """
    # Charger les données JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Charger le fichier CSV existant
    base_df = pd.read_csv(output_file)

    # Parcourir les modèles dans le JSON
    id_name_to_info = {}
    for provider, date_dict in data.items():
        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    # Vérifier que le type est 'text_to_image'
                    if model.get("type") == "text to image" and "id_name" in model and model["id_name"]:
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

    # Analyser chaque ligne pour mettre à jour base_df avec resolution et Steps
    for idx, row in base_df.iterrows():
        id_name = row['id_name']
        segments = id_name.split('-')

        # Supposons que vous avez des fonctions pour extraire les informations
        resolution = extract_average_resolution(segments[4]) if len(segments) > 4 else None
        steps = extract_steps(segments[5]) if len(segments) > 5 else None

        if pd.isnull(row.get('resolution')) and resolution is not None:
            base_df.at[idx, 'resolution'] = resolution
        if pd.isnull(row.get('Steps')) and steps is not None:
            base_df.at[idx, 'Steps'] = steps

    # Sauvegarder le DataFrame mis à jour temporairement
    temp_csv_path = output_file.replace('.csv', '_temp.csv')
    base_df.to_csv(temp_csv_path, index=False)

    # Utiliser add_csv_image pour effectuer les fusions supplémentaires
    df_final = add_csv_texttoimage(temp_csv_path)

    # Enregistrer le DataFrame final à l'emplacement d'origine
    df_final.to_csv(output_file, index=False)
    print(f"Le fichier {output_file} a été mis à jour avec succès en utilisant add_csv_image.")

    # Supprimer le fichier temporaire
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

    return df_final