import os
import pandas as pd
import json
import numpy as np

from datetime import datetime, timedelta
from collections import defaultdict

def init_API(pricing_directory, output_json_path):
    """
    Parcourt les fichiers CSV de pricing pour chaque fournisseur et génère un fichier JSON avec les informations des modèles.

    :param pricing_directory: Chemin vers le répertoire 'pricing' contenant les sous-dossiers des providers.
    :param output_json_path: Chemin vers le fichier JSON de sortie.
    """
    models = []
    
    # Parcours de chaque fournisseur (sous-dossier) dans le répertoire pricing
    for provider_name in os.listdir(pricing_directory):
        provider_path = os.path.join(pricing_directory, provider_name)
        if os.path.isdir(provider_path):
            # Dictionnaire pour regrouper les fichiers CSV par type
            type_files = {}
            
            # Liste de tous les fichiers dans le répertoire du fournisseur
            for filename in os.listdir(provider_path):
                if filename.endswith('.csv'):
                    # Extraction du type à partir du nom du fichier
                    file_parts = filename.split('_')
                    if len(file_parts) >= 2:
                        type_name = file_parts[0]  # Partie avant le premier '_'
                        # Assurez-vous que le type est dans le dictionnaire
                        if type_name not in type_files:
                            type_files[type_name] = {}
                        # Déterminer le type de prix à partir du nom du fichier
                        price_type = file_parts[1].replace('.csv', '')
                        # Ajouter le chemin du fichier au dictionnaire
                        file_path = os.path.join(provider_path, filename)
                        type_files[type_name][price_type] = file_path
            # Maintenant, pour chaque type, traiter les fichiers CSV
            for type_name, price_files in type_files.items():
                # Dictionnaire pour stocker les DataFrames pour chaque type de prix
                price_dfs = {}
                # Traitement de chaque type de prix
                for price_type, file_path in price_files.items():
                    # Lire le fichier CSV
                    df = pd.read_csv(file_path)
                    # Vérifier si les colonnes 'name' et 'id_name' sont présentes
                    if 'name' not in df.columns or 'id_name' not in df.columns:
                        print(f"Le fichier {file_path} est ignoré car les colonnes 'name' ou 'id_name' sont manquantes.")
                        continue
                    # Ajouter le DataFrame au dictionnaire
                    price_dfs[price_type] = df
                if not price_dfs:
                    continue  # Aucun fichier CSV valide pour ce type
                # Maintenant, traiter les DataFrames
                # Convertir chaque DataFrame au format long
                melted_dfs = []
                for price_type, df in price_dfs.items():
                    # Obtenir les colonnes de dates
                    date_columns = [col for col in df.columns if col not in ['name', 'id_name']]
                    # Convertir le DataFrame au format long
                    df_melted = df.melt(id_vars=['name', 'id_name'], value_vars=date_columns, var_name='date', value_name='price')
                    df_melted['price_type'] = price_type
                    # Ne conserver que les lignes où le prix n'est pas nul
                    df_melted = df_melted[df_melted['price'].notnull()]
                    # Ajouter à la liste
                    melted_dfs.append(df_melted)
                if not melted_dfs:
                    continue  # Aucune donnée à traiter pour ce type
                # Concaténer tous les DataFrames convertis
                df_all = pd.concat(melted_dfs, ignore_index=True)
                # Pivot pour obtenir une ligne par 'name', 'id_name', 'date'
                df_pivot = df_all.pivot_table(index=['name', 'id_name', 'date'], columns='price_type', values='price', aggfunc='first').reset_index()
                # Pour chaque ligne, créer une entrée de modèle
                for idx, row in df_pivot.iterrows():
                    # Vérifier si au moins un prix est présent
                    price_call = row.get('pricecall', None)
                    price_input = row.get('priceinput', None)
                    price_output = row.get('priceoutput', None)
                    if pd.isnull(price_call) and pd.isnull(price_input) and pd.isnull(price_output):
                        continue  # Ignorer les entrées sans aucun prix
                    model_entry = {
                        'provider': provider_name,
                        'name': row['name'],
                        'id_name': row['id_name'],
                        'type': type_name,
                        'date': row['date'],
                        'price_call': float(price_call) if not pd.isnull(price_call) else None,
                        'price_input': float(price_input) if not pd.isnull(price_input) else None,
                        'price_output': float(price_output) if not pd.isnull(price_output) else None,
                    }
                    models.append(model_entry)
    # Écrire la liste des modèles dans le fichier JSON de sortie
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(models, f, ensure_ascii=False, indent=4)
    print(f"Le fichier JSON a été créé à {output_json_path} avec {len(models)} entrées de modèles.")





def add_infos_to_API(models_infos_directory, api_json_path):
    """
    Complète le fichier JSON des modèles avec des informations supplémentaires provenant des fichiers CSV appropriés.
    Remplace les valeurs NaN par null dans le fichier JSON final.

    :param models_infos_directory: Chemin vers le répertoire contenant les fichiers CSV (par exemple, 'AIKoD_text_infos.csv').
    :param api_json_path: Chemin vers le fichier JSON généré par 'init_API'.
    """
    # Fonction pour remplacer les NaN par None dans les structures de données
    def replace_nan_with_none(data):
        if isinstance(data, dict):
            for k in data:
                data[k] = replace_nan_with_none(data[k])
            return data
        elif isinstance(data, list):
            for i in range(len(data)):
                data[i] = replace_nan_with_none(data[i])
            return data
        elif isinstance(data, float):
            if np.isnan(data):
                return None
            else:
                return data
        else:
            return data

    # Lire le fichier JSON existant
    with open(api_json_path, 'r', encoding='utf-8') as f:
        models = json.load(f)

    # Créer un dictionnaire pour regrouper les modèles par type
    models_by_type = {}
    for model in models:
        type_name = model['type']
        if type_name not in models_by_type:
            models_by_type[type_name] = []
        models_by_type[type_name].append(model)

    # Parcourir chaque type et compléter les informations
    for type_name, models_list in models_by_type.items():
        # Déterminer le nom du fichier CSV correspondant
        csv_filename = f'AIKoD_{type_name}_infos.csv'
        csv_path = os.path.join(models_infos_directory, csv_filename)

        # Vérifier si le fichier CSV existe
        if not os.path.exists(csv_path):
            print(f"Le fichier CSV pour le type '{type_name}' n'existe pas : {csv_path}")
            continue

        # Lire le fichier CSV dans un DataFrame
        df = pd.read_csv(csv_path)

        # Vérifier que la colonne 'id_name' est présente
        if 'id_name' not in df.columns:
            print(f"La colonne 'id_name' est manquante dans le fichier CSV : {csv_path}")
            continue

        # Remplacer les NaN par None dans le DataFrame
        df = df.where(pd.notnull(df), None)

        # Créer un dictionnaire des informations du CSV indexé par 'id_name'
        df.set_index('id_name', inplace=True)
        csv_info_dict = df.to_dict('index')

        # Pour chaque modèle du type courant, ajouter les informations du CSV
        for model in models_list:
            id_name = model['id_name']
            if id_name in csv_info_dict:
                # Obtenir les informations du CSV pour cet 'id_name'
                csv_info = csv_info_dict[id_name]
                # Ajouter les informations au modèle, en excluant 'id_name'
                for key, value in csv_info.items():
                    model[key] = value
            else:
                print(f"L'id_name '{id_name}' n'a pas été trouvé dans le fichier CSV : {csv_path}")

    # Remplacer les NaN par None dans l'ensemble de la liste des modèles
    replace_nan_with_none(models)

    # Enregistrer le fichier JSON mis à jour
    with open(api_json_path, 'w', encoding='utf-8') as f:
        json.dump(models, f, ensure_ascii=False, indent=4, allow_nan=False)
    print(f"Le fichier JSON a été mis à jour avec les informations supplémentaires et enregistré à {api_json_path}.")



def pareto_frontier(models, price_field, quality_field, maximize_quality):
    """
    Calcule le front de Pareto pour une liste de modèles en fonction du prix et de la qualité.

    Paramètres:
    - models (list): Liste de modèles avec les champs '_parsed_price' et '_parsed_quality'.
    - price_field (str): Nom du champ de prix dans les modèles.
    - quality_field (str): Nom du champ de qualité dans les modèles.
    - maximize_quality (bool): True si la qualité doit être maximisée, False si elle doit être minimisée.

    Retourne:
    - pareto_models (list): Liste des modèles sur le front de Pareto.
    """
    # Trier les modèles par prix croissant
    models_sorted = sorted(models, key=lambda x: x['_parsed_price'])
    pareto_models = []
    current_best_quality = None

    for model in models_sorted:
        quality = model['_parsed_quality']
        if current_best_quality is None:
            pareto_models.append(model)
            current_best_quality = quality
        else:
            if maximize_quality:
                if quality > current_best_quality:
                    pareto_models.append(model)
                    current_best_quality = quality
            else:
                if quality < current_best_quality:
                    pareto_models.append(model)
                    current_best_quality = quality

    return pareto_models


def generate_API_date(input_json_path, output_json_path):
    """
    Génère un fichier JSON contenant les modèles disponibles pour chaque mois de 2023-01 à 2024-09,
    catégorisés par 'type', avec les 'models_star' par 'type' représentant le Pareto optimal
    en termes de qualité et de prix selon les critères spécifiés.

    Paramètres:
    - input_json_path (str): Chemin vers le fichier JSON existant (généré par 'init_API').
    - output_json_path (str): Chemin où le fichier JSON sera enregistré.

    Le fichier JSON généré aura la structure suivante:
    {
        "YYYY_MM": {
            "models_list": {
                "text": [...],
                "audiototext": [...],
                "texttoimage": [...],
                ...
            },
            "models_star": {
                "text": [...],
                "audiototext": [...],
                "texttoimage": [...],
                ...
            }
        },
        ...
    }
    """
    # Lire les données depuis le fichier JSON existant
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Date de début et de fin
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 9, 30)

    # Initialiser un dictionnaire pour stocker les données
    api_data = {}

    # Préparer les dates cibles (dernier jour de chaque mois)
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        # Obtenir le dernier jour du mois
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day_of_month = next_month - timedelta(days=1)
        date_list.append(last_day_of_month)
        if month == 12:
            current_date = datetime(year + 1, 1, 1)
        else:
            current_date = datetime(year, month + 1, 1)

    # Pré-traiter les données pour obtenir les dates disponibles par fournisseur
    provider_dates = defaultdict(set)
    for entry in data:
        provider = entry.get('provider')
        date_str = entry.get('date')
        if not provider or not date_str:
            continue
        try:
            entry_date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            continue
        provider_dates[provider].add(entry_date)

    # Maintenant, pour chaque date cible
    for target_date in date_list:
        # Formater la clé de date au format 'YYYY_MM'
        date_key = target_date.strftime('%Y_%m')

        # Pour chaque fournisseur, trouver la dernière date pertinente jusqu'à target_date
        provider_latest_date = {}
        for provider, dates in provider_dates.items():
            dates_before_target = [d for d in dates if d <= target_date]
            if dates_before_target:
                latest_date = max(dates_before_target)
                provider_latest_date[provider] = latest_date

        # Collecter les modèles pour chaque fournisseur à la dernière date pertinente
        models_for_date = []
        for entry in data:
            provider = entry.get('provider')
            date_str = entry.get('date')
            if not provider or not date_str:
                continue
            if provider in provider_latest_date:
                latest_date = provider_latest_date[provider]
                if date_str == latest_date.strftime('%Y-%m-%d'):
                    models_for_date.append(entry)

        # Catégoriser les modèles par 'type' dans 'models_list'
        models_list = defaultdict(list)
        for model in models_for_date:
            type_ = model.get('type')
            if type_:
                models_list[type_].append(model)

        # Trouver les 'models_star' par 'type' en utilisant le filtrage Pareto
        models_star = {}
        for type_, models in models_list.items():
            # Fonction utilitaire pour convertir en float
            def parse_float(value):
                if value in [None, '', 'null']:
                    return None
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None

            # Définir les champs de prix et de qualité selon le type
            if type_ == 'text':
                price_field = 'price_output'
                quality_field = 'quality_index'
                maximize_quality = True
            elif type_ == 'audiototext':
                price_field = 'price_input'
                quality_field = 'Word Error Rate (%)'
                maximize_quality = False  # On veut minimiser l'erreur
            elif type_ == 'texttoimage':
                price_field = 'price_output'
                quality_field = 'Model Quality ELO'
                maximize_quality = True
            else:
                # Pour les autres types, vous pouvez ajouter des critères spécifiques ou les ignorer
                continue

            # Préparer les modèles avec les champs nécessaires
            models_filtered = []
            for m in models:
                price = parse_float(m.get(price_field))
                quality = parse_float(m.get(quality_field))
                if price is not None and quality is not None:
                    m['_parsed_price'] = price
                    m['_parsed_quality'] = quality
                    models_filtered.append(m)

            if models_filtered:
                # Appliquer le filtrage Pareto
                pareto_models = pareto_frontier(models_filtered, '_parsed_price', '_parsed_quality', maximize_quality)
                # Enlever les champs temporaires
                for m in pareto_models:
                    del m['_parsed_price']
                    del m['_parsed_quality']
                models_star[type_] = pareto_models

        # Convertir models_list en dictionnaire standard
        models_list = dict(models_list)

        # Ajouter les données au dictionnaire principal
        api_data[date_key] = {
            'models_list': models_list,
            'models_star': models_star
        }

    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = os.path.dirname(output_json_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Écrire les données dans un fichier JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(api_data, f, ensure_ascii=False, indent=4)

    print(f"Le fichier JSON a été généré et enregistré à {output_json_path}.")

