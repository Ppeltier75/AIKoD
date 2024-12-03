import pandas as pd
import json
import os
from collections import Counter
from function_prep.utils_prep_text import normalize_elo_rating

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
    keep_columns = ['Model Quality ELO', 'speed_index']

    # Fusion
    df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)

    # Fusion avec AA_texttoimage_infos.csv
    df_merge = pd.read_csv(paths['AA_texttoimage_infos'])

    keep_columns = ['default_steps', 'default_resolution']

    df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)
    df_merged.rename(columns={'Model Quality ELO': 'quality_index'}, inplace=True)

    # Enregistrement du DataFrame fusionné
    # Définition du chemin de sortie (vous pouvez le modifier si nécessaire)
    output_csv_path = os.path.splitext(base_csv_path)[0] + '.csv'

    df_merged.to_csv(output_csv_path, index=False)

    print(f"Le fichier fusionné a été enregistré sous {output_csv_path}")

    return df_merged


def AIKoD_texttoimage_infos(json_path, output_file):
    """
    Met à jour un fichier CSV avec des informations extraites d'un JSON et fusionne avec d'autres fichiers CSV
    en utilisant add_csv_texttoimage. Normalise également la colonne quality_index.

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

    # Utiliser add_csv_texttoimage pour effectuer les fusions supplémentaires
    df_final = add_csv_texttoimage(temp_csv_path)

    # Normaliser la colonne quality_index
    if "quality_index" in df_final.columns:
        df_final["quality_index"] = df_final["quality_index"].apply(lambda x: normalize_elo_rating(x, elo_min=600,elo_max=1300))
        print("La colonne 'quality_index' a été normalisée avec succès.")

    # Enregistrer le DataFrame final à l'emplacement d'origine
    df_final.to_csv(output_file, index=False)
    print(f"Le fichier {output_file} a été mis à jour avec succès en utilisant add_csv_texttoimage.")

    # Supprimer le fichier temporaire
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

    return df_final


def create_adjusted_price_text_to_image(directory_csv, csv_path):
    """
    Parcourt tous les fichiers 'texttoimage_priceoutput.csv' dans le répertoire donné,
    calcule la moyenne des valeurs numériques pour chaque 'id_name',
    et stocke le résultat dans un fichier CSV avec les colonnes 'id_name' et 'mean_price'.

    :param directory_csv: Répertoire où chercher les fichiers 'texttoimage_priceoutput.csv'.
    :param csv_path: Chemin du fichier CSV où les résultats seront stockés.
    """
    # Initialiser un dictionnaire pour stocker les listes de prix par id_name
    id_name_prices = {}

    # Parcourir le répertoire pour trouver tous les fichiers 'texttoimage_priceoutput.csv'
    for root, dirs, files in os.walk(directory_csv):
        for file in files:
            if file == 'texttoimage_priceoutput.csv':
                file_path = os.path.join(root, file)
                # Lire le fichier CSV
                df = pd.read_csv(file_path)
                # Vérifier que les colonnes nécessaires sont présentes
                if 'id_name' not in df.columns:
                    print(f"Le fichier {file_path} ne contient pas la colonne 'id_name'.")
                    continue
                # Itérer sur chaque ligne du DataFrame
                for index, row in df.iterrows():
                    id_name = row['id_name']
                    # Obtenir les colonnes de dates (exclure 'name' et 'id_name')
                    date_columns = [col for col in df.columns if col not in ['name', 'id_name']]
                    # Obtenir les valeurs numériques non nulles pour cette ligne
                    prices = row[date_columns].dropna().tolist()
                    # Convertir les prix en float (au cas où ils sont sous forme de chaînes)
                    prices = [float(price) for price in prices if price != '']
                    if prices:
                        # Initialiser la liste pour cet id_name si nécessaire
                        if id_name not in id_name_prices:
                            id_name_prices[id_name] = []
                        # Ajouter les nouveaux prix à la liste
                        id_name_prices[id_name].extend(prices)

    # Calculer le prix moyen pour chaque id_name
    id_name_avg_price = []
    for id_name, prices in id_name_prices.items():
        avg_price = sum(prices) / len(prices)
        id_name_avg_price.append({'id_name': id_name, 'mean_price': avg_price})

    # Créer un DataFrame avec les résultats
    df_results = pd.DataFrame(id_name_avg_price)

    # Sauvegarder le DataFrame dans le fichier CSV
    df_results.to_csv(csv_path, index=False)

    print(f"Le fichier '{csv_path}' a été créé avec les colonnes 'id_name' et 'mean_price'.")




def reorganize_prices_by_resolution_and_steps(input_csv_path):
    """
    Lit le fichier CSV à input_csv_path, traite les id_names pour regrouper par id_name de base
    (en supprimant uniquement le segment de résolution ou de steps selon le cas),
    et génère deux nouveaux CSV dans le même répertoire :
    - Un pour les résolutions (en conservant les steps).
    - Un pour les steps (en conservant les résolutions).
    
    :param input_csv_path: Chemin vers le fichier CSV d'entrée.
    """
    # Lire le fichier CSV
    df = pd.read_csv(input_csv_path)

    # Vérifier que les colonnes requises sont présentes
    if 'id_name' not in df.columns or 'mean_price' not in df.columns:
        print("Le fichier CSV doit contenir les colonnes 'id_name' et 'mean_price'.")
        return

    # Fonction pour supprimer uniquement le segment de résolution (segment 5)
    def remove_resolution(id_name):
        segments = id_name.split('-')
        if len(segments) >= 6:
            resolution = segments[4]
            # Retirer le segment de résolution, conserver les steps
            base_id_name = segments[:4] + segments[5:]
            base_id_name = '-'.join(base_id_name)
            return base_id_name, resolution
        else:
            return id_name, ''

    # Fonction pour supprimer uniquement le segment des steps (segment 6)
    def remove_steps(id_name):
        segments = id_name.split('-')
        if len(segments) >= 7:
            steps = segments[5]
            # Retirer le segment des steps, conserver la résolution
            base_id_name = segments[:5] + segments[6:]
            base_id_name = '-'.join(base_id_name)
            return base_id_name, steps
        else:
            return id_name, ''

    # Initialiser des dictionnaires pour collecter les données
    data_resolution = {}
    data_steps = {}

    # Traitement pour les résolutions (en conservant les steps)
    for idx, row in df.iterrows():
        id_name = row['id_name']
        mean_price = row['mean_price']
        base_id_name, resolution = remove_resolution(id_name)

        # Collecter les données pour les résolutions
        if base_id_name not in data_resolution:
            data_resolution[base_id_name] = {}
        data_resolution[base_id_name][resolution] = mean_price

    # Traitement pour les steps (en conservant les résolutions)
    for idx, row in df.iterrows():
        id_name = row['id_name']
        mean_price = row['mean_price']
        base_id_name, steps = remove_steps(id_name)

        # Collecter les données pour les steps
        if base_id_name not in data_steps:
            data_steps[base_id_name] = {}
        data_steps[base_id_name][steps] = mean_price

    # Obtenir toutes les résolutions uniques pour définir les colonnes
    resolutions = set()
    for resolutions_dict in data_resolution.values():
        resolutions.update(resolutions_dict.keys())
    resolutions.discard('')  # Supprimer la résolution vide si présente
    resolutions = sorted(resolutions)

    # Obtenir tous les steps uniques pour définir les colonnes
    steps_set = set()
    for steps_dict in data_steps.values():
        steps_set.update(steps_dict.keys())
    steps_set.discard('')  # Supprimer les steps vides si présents

    # Tri corrigé des steps
    steps_set = sorted(steps_set, key=lambda x: (0, float(x)) if x.replace('.', '', 1).isdigit() else (1, x))

    # Construire le nouveau DataFrame pour les résolutions
    rows_resolution = []
    for base_id_name, resolutions_dict in data_resolution.items():
        row = {'id_name': base_id_name}
        for res in resolutions:
            row[res] = resolutions_dict.get(res, '')
        rows_resolution.append(row)
    df_resolutions = pd.DataFrame(rows_resolution)

    # Construire le nouveau DataFrame pour les steps
    rows_steps = []
    for base_id_name, steps_dict in data_steps.items():
        row = {'id_name': base_id_name}
        for step in steps_set:
            row[step] = steps_dict.get(step, '')
        rows_steps.append(row)
    df_steps = pd.DataFrame(rows_steps)

    # Déterminer les chemins de sortie (même répertoire que le fichier d'entrée)
    dir_name = os.path.dirname(input_csv_path)
    output_csv_path_resolutions = os.path.join(dir_name, 'resolution_prices.csv')
    output_csv_path_steps = os.path.join(dir_name, 'steps_prices.csv')

    # Sauvegarder les DataFrames dans les fichiers CSV
    df_resolutions.to_csv(output_csv_path_resolutions, index=False)
    df_steps.to_csv(output_csv_path_steps, index=False)

    print(f"Les nouveaux fichiers CSV ont été créés :")
    print(f"- Résolutions : {output_csv_path_resolutions}")
    print(f"- Steps : {output_csv_path_steps}")