import os
import pandas as pd
import json
import re
from collections import Counter
import numpy as np

# Importation des fonctions depuis merge_utils.py
from function_utils.utils_merge_id import select_specific_segments, select_segments_no_order, merge_csv_id_name

# Fonction pour analyser un id_name et extraire les informations
def analyze_id_name(id_name):
    """
    Extrait le nombre de paramètres, la taille de la fenêtre de contexte et le statut finetuned
    à partir du id_name.
    """
    id_parts = id_name.split('-')
    number_of_parameters = None
    context_window = None
    finetuned = None

    # Vérifier si les positions 6 et 7 contiennent des informations valides
    if len(id_parts) >= 7:
        try:
            number_of_parameters = float(id_parts[5]) if re.match(r"^\d+(\.\d+)?$", id_parts[5]) else None
            context_window = int(id_parts[6]) * 1000 if re.match(r"^\d+$", id_parts[6]) else None
        except ValueError:
            pass

    # Vérifier si le dernier élément indique finetuned (false/true)
    if len(id_parts) >= 8:
        finetuned = id_parts[7].strip().lower() == "true"

    return number_of_parameters, context_window, finetuned


def add_csv_text(base_csv_path):
    # Définition des stratégies de correspondance
    strategies = [
        lambda x: x,  # Correspondance exacte
        lambda x: select_specific_segments(x, [1, 2, 3, 4, 5, 6, 7, 8]),
        lambda x: select_segments_no_order(x, [1, 2, 3, 4, 5, 6, 7, 8]),
        lambda x: select_specific_segments(x, [1, 2, 3, 4, 5, 6, 7]),
        lambda x: select_segments_no_order(x, [1, 2, 3, 4, 5, 6, 7]),
        lambda x: select_specific_segments(x, [1, 2, 3, 4, 5, 6]),
        lambda x: select_segments_no_order(x, [1, 2, 3, 4, 5, 6]),
        lambda x: select_specific_segments(x, [1, 2, 3, 4, 6]),
        lambda x: select_segments_no_order(x, [1, 2, 3, 4, 6]),
        lambda x: select_specific_segments(x, [1, 2, 4, 6]),
        lambda x: select_segments_no_order(x, [1, 2, 3, 4]),
        lambda x: select_specific_segments(x, [1, 2, 4]),
        lambda x: select_specific_segments(x, [1, 4, 6]),
        # Vous pouvez ajouter d'autres stratégies si nécessaire
    ]

    # Lecture du fichier de base
    df_base = pd.read_csv(base_csv_path)

    # Création d'une copie du DataFrame de base pour les fusions successives
    df_merged = df_base.copy()

    # Chemins vers les fichiers à fusionner
    paths = {
        'AA_quality': r'C:\Users\piwip\OneDrive\Documents\OCDE\AIKoD\data\benchmark\AA\2024-11-16\AA_quality_2024-11-16.csv',
        'Livebench_text': r'C:\Users\piwip\OneDrive\Documents\OCDE\AIKoD\data\benchmark\Livebench\Livebench_text_2024-08-31.csv',
        'HF_text_AE': r'C:\Users\piwip\OneDrive\Documents\OCDE\AIKoD\data\benchmark\HF\HF_text_AE.csv',
        'HF_text_MMLU': r'C:\Users\piwip\OneDrive\Documents\OCDE\AIKoD\data\benchmark\HF\HF_text_MMLU.csv',
        'HF_text_MT': r'C:\Users\piwip\OneDrive\Documents\OCDE\AIKoD\data\benchmark\HF\HF_text_MT.csv',
        'AA_text': r'C:\Users\piwip\OneDrive\Documents\OCDE\AIKoD\data\benchmark\AA\text\AA_text_2024-11-19.csv',
    }

    # Fusion avec AA_quality_2024-11-16
    df_merge = pd.read_csv(paths['AA_quality'])

    # Renommer les colonnes selon vos spécifications
    df_merge.rename(columns={
        'chatbot_arena_elo': 'AA_arenaelo',
        'mmlu': 'aa_mmlu',
        'gpqa': 'aa_gpqa',
        'humaneval': 'aa_humaneval',
        'math': 'aa_math',
        'mgsm': 'aa_mgsm'
    }, inplace=True)

    # Colonnes à conserver
    keep_columns = ['AA_arenaelo', 'quality_index', 'aa_mmlu', 'aa_gpqa', 'aa_humaneval', 'aa_math', 'aa_mgsm']

    # Fusion
    df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)

    # Fusion avec Livebench_text_2024-08-31
    df_merge = pd.read_csv(paths['Livebench_text'])

    # Renommer la colonne 'Global Average' en 'Livebench_rating'
    df_merge.rename(columns={'Global Average': 'Livebench_rating'}, inplace=True)

    keep_columns = ['Livebench_rating']

    df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)

    # Fusion avec HF_text_AE
    df_merge = pd.read_csv(paths['HF_text_AE'])

    keep_columns = ['AE']

    df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)

    # Fusion avec HF_text_MMLU
    df_merge = pd.read_csv(paths['HF_text_MMLU'])

    keep_columns = ['MMLU']

    df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)

    # Fusion avec HF_text_MT
    df_merge = pd.read_csv(paths['HF_text_MT'])

    keep_columns = ['MT']

    df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)

    # Fusion avec AA_text_2024-11-19
    df_merge = pd.read_csv(paths['AA_text'])

    keep_columns = ['Output Tokens/S Median', 'Latency Median (First Chunk)']

    df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)

    df_merged.to_csv(base_csv_path, index=False)

    print(f"Le fichier fusionné a été enregistré sous {base_csv_path}")

    return df_merged


def normalize_elo_rating(elo, elo_min=1000, elo_max=2000):
    """Normalize an ELO rating to a 0-1 scale."""
    if pd.isnull(elo):
        return np.nan
    return (elo - elo_min) / (elo_max - elo_min)

def add_quality_index(csv_path):
    """
    Calcule l'indice de qualité pour les modèles en fonction des colonnes spécifiées et des coefficients.
    La fonction lit le fichier CSV, calcule l'indice de qualité, supprime les colonnes intermédiaires,
    et enregistre le DataFrame mis à jour dans le même fichier CSV.

    :param csv_path: Chemin vers le fichier CSV.
    """
    # Lire le fichier CSV dans un DataFrame
    df = pd.read_csv(csv_path)

    # Définir les colonnes à utiliser
    columns = {
        'Livebench_rating': 'Livebench_rating',
        'aa_mmlu': 'aa_mmlu',
        'MMLU': 'MMLU',
        'AA_arenaelo': 'AA_arenaelo',
        'AE': 'AE',
        'aa_gpqa': 'aa_gpqa'
    }

    # S'assurer que toutes les colonnes requises sont présentes
    missing_cols = [col for col in columns.values() if col not in df.columns]
    if missing_cols:
        print(f"Les colonnes suivantes sont manquantes dans le CSV : {missing_cols}")
        return df  # Retourner le DataFrame tel quel si des colonnes manquent

    # Combiner MMLU et aa_mmlu
    def compute_mmlu(row):
        values = []
        if pd.notnull(row['MMLU']):
            values.append(row['MMLU'])
        if pd.notnull(row['aa_mmlu']):
            values.append(row['aa_mmlu'])
        if values:
            return sum(values) / len(values)
        else:
            return np.nan

    df['MMLU_value'] = df.apply(compute_mmlu, axis=1)

    # Combiner AE et AA_arenaelo
    def compute_arenaelo(row):
        values = []
        if pd.notnull(row['AE']):
            values.append(row['AE'])
        if pd.notnull(row['AA_arenaelo']):
            values.append(row['AA_arenaelo'])
        if values:
            return sum(values) / len(values)
        else:
            return np.nan

    df['Arenaelo_value_raw'] = df.apply(compute_arenaelo, axis=1)

    # Normaliser Arenaelo_value en utilisant la fonction normalize_elo_rating
    df['Arenaelo_value'] = df['Arenaelo_value_raw'].apply(lambda x: normalize_elo_rating(x, elo_min=1000, elo_max=2000))

    # Valeur GPQA
    df['GPQA_value'] = df['aa_gpqa']

    # Livebench rating normalisé en le divisant par 100
    df['Livebench_value_raw'] = df['Livebench_rating']
    df['Livebench_value'] = df['Livebench_value_raw'] / 100

    # Coefficients pour l'indice de qualité
    coefficients = {
        'MMLU_value': 0.35,
        'Arenaelo_value': 0.35,
        'GPQA_value': 0.15,
        'Livebench_value': 0.15
    }

    # Fonction pour calculer l'indice de qualité pour une ligne
    def calculate_quality_index(row):
        # Vérifier si l'une des sources MMLU ou Arenaelo a des valeurs
        has_mmlu = pd.notnull(row['MMLU']) or pd.notnull(row['aa_mmlu'])
        has_arenaelo = pd.notnull(row['AE']) or pd.notnull(row['AA_arenaelo'])

        # Procéder uniquement si au moins l'une des valeurs MMLU ou Arenaelo est présente
        if has_mmlu or has_arenaelo:
            # Obtenir les valeurs pour les métriques
            mmlu = row['MMLU_value']
            arenaelo = row['Arenaelo_value']
            gpqa = row['GPQA_value']
            livebench = row['Livebench_value']

            # Préparer un dictionnaire de valeurs
            values = {
                'MMLU_value': mmlu,
                'Arenaelo_value': arenaelo,
                'GPQA_value': gpqa,
                'Livebench_value': livebench
            }

            # Imputer les valeurs manquantes en utilisant l'imputation par déciles
            for key in values:
                if pd.isnull(values[key]):
                    values[key] = impute_value(df, key, row)

            # Calculer l'indice de qualité
            quality_index = sum(values[k] * coefficients[k] for k in coefficients)
            return round(quality_index, 3)
        else:
            return np.nan

    # Fonction pour imputer les valeurs manquantes basées sur les déciles en utilisant des valeurs normalisées
    def impute_value(df, column, row):
        # Obtenir les déciles pour la colonne
        deciles = df[column].quantile([i/10 for i in range(1, 10)]).values

        # Obtenir les positions des déciles des métriques disponibles (en excluant la colonne actuelle)
        available_metrics = ['MMLU_value', 'Arenaelo_value', 'GPQA_value', 'Livebench_value']
        available_metrics = [m for m in available_metrics if pd.notnull(row[m]) and m != column]

        if not available_metrics:
            return df[column].mean()  # Repli sur la moyenne si aucune métrique n'est disponible

        # Calculer la position moyenne des déciles
        decile_positions = []
        for metric in available_metrics:
            metric_value = row[metric]
            metric_deciles = df[metric].quantile([i/10 for i in range(1, 10)]).values
            position = np.searchsorted(metric_deciles, metric_value, side='right')
            decile_positions.append(position)

        avg_decile = int(np.floor(np.mean(decile_positions)))
        if avg_decile >= len(deciles):
            avg_decile = len(deciles) - 1
        elif avg_decile < 0:
            avg_decile = 0

        # Imputer la valeur basée sur le décile moyen
        return deciles[avg_decile]

    # Calculer l'indice de qualité pour chaque ligne
    df['quality_index'] = df.apply(calculate_quality_index, axis=1)

    # Supprimer les colonnes intermédiaires
    columns_to_drop = [
        'MMLU_value',
        'Arenaelo_value_raw',
        'Arenaelo_value',
        'GPQA_value',
        'Livebench_value_raw',
        'Livebench_value'
    ]
    df.drop(columns=columns_to_drop, inplace=True)

    # Sauvegarder le DataFrame mis à jour dans le fichier CSV
    df.to_csv(csv_path, index=False)
    print(f"La colonne 'quality_index' a été ajoutée au fichier CSV '{csv_path}'.")

    # Retourner le DataFrame mis à jour
    return df

def AIKoD_text_infos(json_path, text_infos_csv_path):
    """
    Analyse un fichier JSON pour les modèles avec type 'text' et ajoute des informations
    aux fichiers CSV existants.

    :param json_path: Chemin du fichier JSON contenant les données des modèles.
    :param text_infos_csv_path: Chemin vers le fichier CSV à mettre à jour.
    """

    # Charger les données JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Charger le fichier AIKoD_text_infos.csv
    text_infos_df = pd.read_csv(text_infos_csv_path)

    # Parcourir les modèles dans le JSON
    id_name_to_info = {}
    for provider, date_dict in data.items():
        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    # Vérifier que le type est 'text'
                    if model.get("type") == "text" and "id_name" in model and model["id_name"]:
                        id_name = model["id_name"]
                        company = model.get("company", None)
                        date_release = model.get("date_release", None)
                        number_of_parameters = model.get("number_of_parameters")
                        context_window = model.get("context_window")
                        finetuned = not model.get("id_name", "").endswith("false")

                        id_name_to_info.setdefault(id_name, {
                            "number_of_parameters": [],
                            "context_window": [],
                            "finetuned": finetuned,
                            "company": [],
                            "date_release": []
                        })

                        if number_of_parameters is not None:
                            id_name_to_info[id_name]["number_of_parameters"].append(number_of_parameters)
                        if context_window is not None:
                            id_name_to_info[id_name]["context_window"].append(context_window)
                        if company:
                            id_name_to_info[id_name]["company"].append(company)
                        if date_release:
                            id_name_to_info[id_name]["date_release"].append(date_release)

    # Analyser les informations majoritaires et compléter les id_name
    rows_to_update = []
    for id_name, info in id_name_to_info.items():
        # Calculer les valeurs majoritaires
        number_of_parameters = (
            Counter(info["number_of_parameters"]).most_common(1)[0][0]
            if info["number_of_parameters"]
            else None
        )
        context_window = (
            Counter(info["context_window"]).most_common(1)[0][0]
            if info["context_window"]
            else None
        )
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
        finetuned = info["finetuned"]

        # Ajouter les informations pour cet id_name
        rows_to_update.append({
            "id_name": id_name,
            "number_of_parameters": number_of_parameters,
            "context_window": context_window,
            "finetuned": finetuned,
            "company": company,
            "date_release": date_release
        })

    # Créer un DataFrame avec les mises à jour
    updates_df = pd.DataFrame(rows_to_update)

    # Fusionner les mises à jour avec le DataFrame existant
    text_infos_df = pd.merge(
        text_infos_df,
        updates_df,
        on="id_name",
        how="outer",
        suffixes=("", "_new"),
    )

    # Remplacer les colonnes avec les nouvelles valeurs (si elles existent)
    for col in ["number_of_parameters", "context_window", "finetuned", "company", "date_release"]:
        if f"{col}_new" in text_infos_df.columns:
            text_infos_df[col] = text_infos_df[f"{col}_new"].combine_first(text_infos_df[col])

    # Supprimer les colonnes temporaires
    text_infos_df.drop(
        columns=[col for col in ["number_of_parameters_new", "context_window_new", "finetuned_new", "company_new", "date_release_new"]
                 if col in text_infos_df.columns],
        inplace=True,
    )

    # Sauvegarder le fichier mis à jour temporairement
    temp_csv_path = text_infos_csv_path.replace('.csv', '_temp.csv')
    text_infos_df.to_csv(temp_csv_path, index=False)

    # Utiliser add_csv_text pour effectuer les fusions supplémentaires
    df_final = add_csv_text(temp_csv_path)
    df_final = add_quality_index(temp_csv_path)

    # Enregistrer le DataFrame final à l'emplacement d'origine
    df_final.to_csv(text_infos_csv_path, index=False)
    print(f"Le fichier {text_infos_csv_path} a été mis à jour avec succès en utilisant add_csv_text.")

    # Supprimer le fichier temporaire
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)


