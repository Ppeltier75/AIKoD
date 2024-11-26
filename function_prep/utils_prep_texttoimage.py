import pandas as pd
import json
from function_utils.utils_merge_id import   strategy_merge
import os

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



def AIKoD_texttoimage_infos(json_path, output_file, merge_file, segments_order=[1, 2, 3, 4]):
    """
    Met à jour un fichier CSV avec des informations extraites d'un JSON et fusionne avec un autre fichier CSV
    en utilisant plusieurs stratégies de merge.

    :param json_path: Chemin vers le fichier JSON contenant les données initiales.
    :param output_file: Chemin vers le fichier CSV à mettre à jour.
    :param merge_file: Chemin vers le fichier CSV à fusionner.
    :param segments_order: Liste des indices de segments à utiliser pour la stratégie segments_order.
    """
    # Charger le JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Charger le fichier CSV existant et le fichier à fusionner
    base_df = pd.read_csv(output_file)
    merge_df = pd.read_csv(merge_file)

    # Analyser chaque ligne pour mettre à jour base_df avec date_release, resolution, et Steps
    updated_data = []
    for _, row in base_df.iterrows():
        id_name = row['id_name']
        segments = id_name.split('-')

        resolution = extract_average_resolution(segments[4]) if len(segments) > 4 else None
        steps = extract_steps(segments[5]) if len(segments) > 5 else None
        date_release = extract_earliest_date(json_data, id_name)

        updated_data.append({
            "id_name": id_name,
            "date_release": date_release,
            "resolution": resolution,
            "Steps": steps
        })

    updated_df = pd.DataFrame(updated_data)

    # Ajouter le merge
    merged_df = strategy_merge(
        base_df=updated_df,
        merge_df=merge_df,
        strategies=["exact", "segments_order"],
        segments_order=segments_order
    )

    # Sauvegarder le fichier final
    merged_df.to_csv(output_file, index=False)
    print(f"Fichier mis à jour et fusionné : {output_file}")
