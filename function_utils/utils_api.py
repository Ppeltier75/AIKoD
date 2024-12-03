import os
import pandas as pd
import json
import numpy as np

from datetime import datetime, timedelta
from collections import defaultdict
from datetime import datetime

from function_utils.utils_merge_id import select_specific_segments, select_segments_no_order

def init_API(pricing_directory, output_json_path):
    """
    Parcourt les fichiers CSV de pricing pour chaque fournisseur et génère un fichier JSON avec les informations des modèles,
    en ajoutant une information 'country_provider' basée sur le fournisseur.

    :param pricing_directory: Chemin vers le répertoire 'pricing' contenant les sous-dossiers des providers.
    :param output_json_path: Chemin vers le fichier JSON de sortie.
    :param country_mapping_path: Chemin vers le fichier JSON de mapping des pays par fournisseur.
    """
    models = []

    base_path = os.path.abspath(os.path.dirname(__file__))
    country_mapping_path = os.path.join(base_path, '..', 'data', 'models_infos', 'mapping', 'country_mapping.json')
    
    # Charger le mapping des pays
    if not os.path.exists(country_mapping_path):
        print(f"Le fichier de mapping des pays n'existe pas : {country_mapping_path}")
        country_mapping = {}
    else:
        with open(country_mapping_path, 'r', encoding='utf-8') as f:
            country_mapping = json.load(f)
        # Normaliser les clés pour une recherche insensible à la casse
        country_mapping_normalized = {key.lower(): value for key, value in country_mapping.items()}
    
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
                        type_name = file_parts[0].lower()  # Partie avant le premier '_', en minuscule
                        # Assurez-vous que le type est dans le dictionnaire
                        if type_name not in type_files:
                            type_files[type_name] = {}
                        # Déterminer le type de prix à partir du nom du fichier
                        price_type = file_parts[1].replace('.csv', '').lower()
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
                    try:
                        df = pd.read_csv(file_path)
                    except Exception as e:
                        print(f"  Erreur lors de la lecture du fichier {file_path} : {e}")
                        continue
                    # Vérifier si les colonnes 'name' et 'id_name' sont présentes
                    if 'name' not in df.columns or 'id_name' not in df.columns:
                        print(f"  Le fichier {file_path} est ignoré car les colonnes 'name' ou 'id_name' sont manquantes.")
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
                df_pivot = df_all.pivot_table(index=['name', 'id_name', 'date'], 
                                             columns='price_type', 
                                             values='price', 
                                             aggfunc='first').reset_index()
                # Pour chaque ligne, créer une entrée de modèle
                for idx, row in df_pivot.iterrows():
                    # Vérifier si au moins un prix est présent
                    price_call = row.get('pricecall', None)
                    price_input = row.get('priceinput', None)
                    price_output = row.get('priceoutput', None)
                    if pd.isnull(price_call) and pd.isnull(price_input) and pd.isnull(price_output):
                        continue  # Ignorer les entrées sans aucun prix
                    # Déterminer le country_provider
                    provider_key = provider_name.lower()
                    country_provider = country_mapping_normalized.get(provider_key, None)
                    model_entry = {
                        'provider': provider_name,
                        'model_name': row['name'],
                        'id_name': row['id_name'],
                        'type': type_name,
                        'date': row['date'],
                        'price_call': float(price_call) if not pd.isnull(price_call) else None,
                        'price_input': float(price_input) if not pd.isnull(price_input) else None,
                        'price_output': float(price_output) if not pd.isnull(price_output) else None,
                        'country_provider': country_provider
                    }
                    models.append(model_entry)
    # Écrire la liste des modèles dans le fichier JSON de sortie
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(models, f, ensure_ascii=False, indent=4)
    print(f"Le fichier JSON a été créé à {output_json_path} avec {len(models)} entrées de modèles.")



def add_infos_to_API(models_infos_directory, api_json_path):
    """
    Complète le fichier JSON des modèles avec des informations supplémentaires provenant des fichiers CSV appropriés.
    Ajoute le blended_price pour les modèles de type 'text', 'audiototext', et 'texttoimage'.

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
    try:
        with open(api_json_path, 'r', encoding='utf-8') as f:
            models = json.load(f)
        print(f"Le fichier JSON a été chargé depuis {api_json_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON {api_json_path} : {e}")
        return

    # Créer un dictionnaire pour regrouper les modèles par type
    try:
        models_by_type = {}
        for model in models:
            type_name = model.get('type')
            if type_name not in models_by_type:
                models_by_type[type_name] = []
            models_by_type[type_name].append(model)
        print("Les modèles ont été regroupés par type.")
    except Exception as e:
        print(f"Erreur lors du regroupement des modèles par type : {e}")
        return

    # Parcourir chaque type et compléter les informations
    for type_name, models_list in models_by_type.items():
        try:
            # Déterminer le nom du fichier CSV correspondant
            csv_filename = f'AIKoD_{type_name}_infos.csv'
            csv_path = os.path.join(models_infos_directory, csv_filename)

            # Vérifier si le fichier CSV existe
            if not os.path.exists(csv_path):
                print(f"Le fichier CSV pour le type '{type_name}' n'existe pas : {csv_path}")
                continue

            # Lire le fichier CSV dans un DataFrame
            try:
                df = pd.read_csv(csv_path)
                print(f"Fichier CSV chargé : {csv_path}")
            except Exception as e:
                print(f"Erreur lors du chargement du fichier CSV '{csv_path}' : {e}")
                continue

            # Vérifier que la colonne 'id_name' est présente
            if 'id_name' not in df.columns:
                print(f"La colonne 'id_name' est manquante dans le fichier CSV : {csv_path}")
                continue

            # Remplacer les NaN par None dans le DataFrame
            df = df.where(pd.notnull(df), None)
            print(f"Les valeurs NaN ont été remplacées par None dans le DataFrame '{csv_path}'.")

            # Créer un dictionnaire des informations du CSV indexé par 'id_name'
            df.set_index('id_name', inplace=True)
            csv_info_dict = df.to_dict('index')
            print(f"Dictionnaire des informations CSV créé pour le type '{type_name}'.")

        except Exception as e:
            print(f"Erreur lors du traitement du type '{type_name}' : {e}")
            continue

        # Pour chaque modèle du type courant, ajouter les informations du CSV
        for model in models_list:
            try:
                id_name = model.get('id_name')
                if id_name in csv_info_dict:
                    # Obtenir les informations du CSV pour cet 'id_name'
                    csv_info = csv_info_dict[id_name]
                    # Ajouter les informations au modèle, en excluant 'id_name'
                    for key, value in csv_info.items():
                        if key != 'id_name':
                            model[key] = value
                    print(f"Informations du CSV ajoutées pour le modèle '{id_name}'.")
                else:
                    print(f"L'id_name '{id_name}' n'a pas été trouvé dans le fichier CSV : {csv_path}")

                # Ajouter blended_price selon le type
                blended_price = None
                if type_name == 'text':
                    # Pour les modèles de type 'text', blended_price = (3/4)*price_input + (1/4)*price_output + 1000*price_call
                    price_input = model.get('price_input')
                    price_output = model.get('price_output')
                    price_call = model.get('price_call', 0.0)  # Default to 0.0 if not present

                    # Convertir les prix en float si possible
                    def parse_price(value):
                        try:
                            return float(value) if value not in [None, '', 'null'] else None
                        except (ValueError, TypeError):
                            return None

                    price_input = parse_price(price_input)
                    price_output = parse_price(price_output)
                    price_call = parse_price(price_call) or 0.0

                    # Calculer blended_price si possible
                    if price_input is not None and price_output is not None:
                        blended_price = (3/4) * price_input + (1/4) * price_output + 1000 * price_call
                    else:
                        blended_price = None

                elif type_name == 'audiototext':
                    # Pour les modèles de type 'audiototext', blended_price = 1 * price_input
                    price_input = model.get('price_input')

                    # Convertir le prix en float si possible
                    def parse_price(value):
                        try:
                            return float(value) if value not in [None, '', 'null'] else None
                        except (ValueError, TypeError):
                            return None

                    price_input = parse_price(price_input)

                    if price_input is not None:
                        blended_price = 1 * price_input
                    else:
                        blended_price = None

                elif type_name == 'texttoimage':
                    # Pour les modèles de type 'texttoimage', blended_price = 1 * price_output
                    price_output = model.get('price_output')

                    # Convertir le prix en float si possible
                    def parse_price(value):
                        try:
                            return float(value) if value not in [None, '', 'null'] else None
                        except (ValueError, TypeError):
                            return None

                    price_output = parse_price(price_output)

                    if price_output is not None:
                        blended_price = 1 * price_output
                    else:
                        blended_price = None

                # Assigner blended_price au modèle
                model['blended_price'] = blended_price
                print(f"blended_price calculé pour le modèle '{id_name}' : {blended_price}")

            except Exception as e:
                print(f"Erreur lors de l'ajout de blended_price pour le modèle '{id_name}' : {e}")
                continue

    # Remplacer les NaN par None dans l'ensemble de la liste des modèles
    try:
        models = replace_nan_with_none(models)
        print("Valeurs NaN remplacées par None dans le JSON.")
    except Exception as e:
        print(f"Erreur lors de la conversion des NaN en None : {e}")
        return

    # Enregistrer le fichier JSON mis à jour
    try:
        with open(api_json_path, 'w', encoding='utf-8') as f:
            json.dump(models, f, ensure_ascii=False, indent=4, allow_nan=False)
        print(f"Le fichier JSON a été mis à jour et enregistré à {api_json_path}.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier JSON mis à jour : {e}")
        return

    print("La fonction add_infos_to_API s'est exécutée avec succès.")


def pareto_frontier(data, price_field, quality_field, maximize_quality=True):
    """
    Identifie les modèles qui sont sur le front de Pareto en fonction des champs de prix et de qualité.

    :param data: Liste de dictionnaires contenant les modèles filtrés avec '_parsed_price' et '_parsed_quality'.
    :param price_field: Nom du champ de prix utilisé pour le tri.
    :param quality_field: Nom du champ de qualité utilisé pour le tri.
    :param maximize_quality: Booléen indiquant si la qualité doit être maximisée ou minimisée.
    :return: Liste des modèles sur le front de Pareto.
    """
    # Trier les données en fonction du prix (ascendant) et de la qualité (descendant ou ascendant)
    sorted_data = sorted(
        data,
        key=lambda x: (x[price_field], -x[quality_field] if maximize_quality else x[quality_field])
    )
    
    pareto = []
    current_best_quality = -np.inf if maximize_quality else np.inf
    
    for model in sorted_data:
        quality = model[quality_field]
        if (maximize_quality and quality > current_best_quality) or (not maximize_quality and quality < current_best_quality):
            pareto.append(model)
            current_best_quality = quality
    
    return pareto

def generate_API_date(input_json_path, output_json_path, exclude_provider=None, exclude_company=None):
    """
    Génère un fichier JSON contenant les modèles disponibles pour chaque mois de 2023-01 à 2024-11,
    catégorisés par 'type', avec les 'models_star' par 'type' représentant le Pareto optimal
    en termes de qualité et de prix selon les critères spécifiés.

    Paramètres:
    - input_json_path (str): Chemin vers le fichier JSON existant (généré par 'init_API').
    - output_json_path (str): Chemin où le fichier JSON sera enregistré.
    - exclude_provider (List[str], optionnel): Liste des fournisseurs à exclure de la génération des frontières.
      Exemple : exclude_provider=["OpenAI", "AI21"]
    - exclude_company (List[str], optionnel): Liste des sociétés à exclure de la génération des frontières.
      Exemple : exclude_company=["OpenAI Inc", "Anthropic LLC"]

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
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Données JSON chargées depuis {input_json_path}")
    except FileNotFoundError:
        print(f"Erreur : Le fichier JSON spécifié n'existe pas : {input_json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON dans le fichier {input_json_path} : {e}")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON {input_json_path} : {e}")
        return

    # Date de début et de fin
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 11, 30)

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

    # Normaliser les listes exclude_provider et exclude_company pour une comparaison insensible à la casse
    if exclude_provider:
        exclude_provider_normalized = [provider.lower() for provider in exclude_provider]
        print(f"Fournisseurs exclus (après normalisation) : {exclude_provider_normalized}")
    else:
        exclude_provider_normalized = []
        print("Aucun fournisseur n'est exclu.")

    if exclude_company:
        exclude_company_normalized = [company.lower() for company in exclude_company]
        print(f"Sociétés exclues (après normalisation) : {exclude_company_normalized}")
    else:
        exclude_company_normalized = []
        print("Aucune société n'est exclue.")

    # Pré-traiter les données pour obtenir les dates disponibles par fournisseur
    provider_dates = defaultdict(set)
    for entry in data:
        provider = entry.get('provider')
        # Assurer que 'company' est une chaîne de caractères avant d'appeler 'strip' et 'lower'
        company = (entry.get('company') or '').strip().lower()
        date_str = entry.get('date')
        if not provider or not date_str:
            continue
        try:
            entry_date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            continue
        provider_dates[provider].add(entry_date)
    print(f"Fournisseurs traités : {list(provider_dates.keys())}")

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

        # Collecter tous les modèles pour chaque fournisseur à la dernière date pertinente
        models_for_date_list = []
        models_for_date_star = []
        for entry in data:
            provider = entry.get('provider')
            # Assurer que 'company' est une chaîne de caractères avant d'appeler 'strip' et 'lower'
            company = (entry.get('company') or '').strip().lower()
            date_str = entry.get('date')
            if not provider or not date_str:
                continue
            # Vérifier si le modèle correspond à la dernière date pertinente pour son fournisseur
            if provider in provider_latest_date:
                latest_date = provider_latest_date[provider]
                if date_str == latest_date.strftime('%Y-%m-%d'):
                    models_for_date_list.append(entry)
                    # Vérifier si le modèle doit être exclu pour `models_star`
                    if (provider.lower() not in exclude_provider_normalized) and (company not in exclude_company_normalized):
                        models_for_date_star.append(entry)

        print(f"Date cible {date_key} : {len(models_for_date_list)} modèles dans models_list, {len(models_for_date_star)} modèles dans models_star après exclusion.")

        # Catégoriser les modèles par 'type' dans 'models_list'
        models_list = defaultdict(list)
        for model in models_for_date_list:
            type_ = model.get('type')
            if type_:
                models_list[type_].append(model)

        # Catégoriser les modèles par 'type' dans 'models_star'
        models_star_candidates = defaultdict(list)
        for model in models_for_date_star:
            type_ = model.get('type')
            if type_:
                models_star_candidates[type_].append(model)

        # Trouver les 'models_star' par 'type' en utilisant le filtrage Pareto
        models_star = {}
        for type_, models in models_star_candidates.items():
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
                price_field = 'price_input'  # Assurez-vous que cela correspond à vos données
                quality_field = 'quality_index'  # Assurez-vous que cela correspond à vos données
                maximize_quality = True
            elif type_ == 'audiototext':
                price_field = 'price_input'
                quality_field = 'quality_index'
                maximize_quality = False  # On veut minimiser l'erreur
            elif type_ == 'texttoimage':
                price_field = 'price_output'
                quality_field = 'quality_index'
                maximize_quality = True
            else:
                # Pour les autres types, vous pouvez ajouter des critères spécifiques ou les ignorer
                continue

            # Préparer les modèles avec les champs nécessaires
            models_filtered = []
            for m in models:
                price = parse_float(m.get(price_field))
                quality = parse_float(m.get(quality_field))
                if price is not None and price > 0 and quality is not None:
                    m['_parsed_price'] = price
                    m['_parsed_quality'] = quality
                    models_filtered.append(m)

            if models_filtered:
                # Appliquer le filtrage Pareto
                pareto_models = pareto_frontier(models_filtered, '_parsed_price', '_parsed_quality', maximize_quality)
                # Enlever les champs temporaires des modèles Pareto
                for m in pareto_models:
                    m.pop('_parsed_price', None)
                    m.pop('_parsed_quality', None)
                models_star[type_] = pareto_models
                print(f"Type '{type_}' : {len(pareto_models)} modèles sur le front de Pareto.")
            else:
                print(f"Type '{type_}' : Aucun modèle filtré pour le front de Pareto.")

        # Supprimer les champs temporaires `_parsed_` de tous les modèles dans `models_list`
        for type_, models in models_list.items():
            for model in models:
                keys_to_remove = [key for key in model if key.startswith('_parsed_')]
                for key in keys_to_remove:
                    model.pop(key, None)

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
        try:
            os.makedirs(output_dir)
            print(f"Répertoire de sortie créé : {output_dir}")
        except Exception as e:
            print(f"Erreur lors de la création du répertoire de sortie {output_dir} : {e}")
            return

    # Écrire les données dans un fichier JSON
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(api_data, f, ensure_ascii=False, indent=4)
        print(f"Le fichier JSON a été généré et enregistré à {output_json_path}.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier JSON mis à jour : {e}")
        return

    print("La fonction generate_API_date s'est exécutée avec succès.")


def add_speed_provider_text_AA(json_path, aa_directory):
    """
    Ajoute les colonnes 'median_time_to_first_token_seconds' et 'median_output_tokens_per_second'
    aux modèles de type 'text', 'audiototext' et 'texttoimage' dans le fichier JSON 'AIKoD_API_base_v0.json'
    en se basant sur les fichiers 'speed_performance.csv' situés dans les sous-dossiers du répertoire AA.
    Calcule également le 'blended_price' selon le type de modèle.
    Indique le nombre de modèles mis à jour et enregistre le JSON modifié.
    """
    # Définir les stratégies de transformation pour id_name
    strategies = [
        lambda x: x,  # Correspondance exacte
        lambda x: select_specific_segments(x, [1, 2, 3, 4, 5, 6, 7, 8]),
        lambda x: select_segments_no_order(x, [1, 2, 3, 4, 5, 6, 7, 8]),
        lambda x: select_specific_segments(x, [1, 2, 3, 4, 5, 6, 7]),
        lambda x: select_segments_no_order(x, [1, 2, 3, 4, 5, 6, 7]),
        lambda x: select_specific_segments(x, [1, 2, 3, 4, 5, 6]),
        lambda x: select_segments_no_order(x, [1, 2, 3, 4, 5, 6]),
        lambda x: select_specific_segments(x, [1, 2, 4, 6]),
        lambda x: select_segments_no_order(x, [1, 2, 4, 6]),
        lambda x: select_specific_segments(x, [1, 2, 4]),
        lambda x: select_specific_segments(x, [1, 4, 6]),
        # Ajoutez d'autres stratégies si nécessaire
    ]

    # Vérifier l'existence du répertoire AA
    if not os.path.exists(aa_directory):
        print(f"Le répertoire AA spécifié n'existe pas : {aa_directory}")
        return

    # Vérifier l'existence du fichier JSON
    if not os.path.exists(json_path):
        print(f"Le fichier JSON spécifié n'existe pas : {json_path}")
        return

    # Charger le JSON existant
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            api_data = json.load(f)
        print(f"Le fichier JSON a été chargé depuis {json_path}")
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON dans le fichier {json_path} : {e}")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON {json_path} : {e}")
        return

    updated_models_count = 0  # Compteur des modèles mis à jour

    # Parcourir les modèles dans le JSON
    for model in api_data:
        try:
            if model.get('type') not in ['text', 'audiototext', 'texttoimage']:
                continue  # Ne traiter que les types spécifiés

            id_name_original = model.get('id_name')
            if not id_name_original:
                print(f"Modèle sans 'id_name' : {model.get('model_name', 'Unknown')}")
                continue

            provider = model.get('provider', '').lower()
            model_date_str = model.get('date')
            if not model_date_str:
                print(f"Modèle sans 'date' : {id_name_original}")
                continue

            # Parse the model date
            try:
                model_date = datetime.strptime(model_date_str, '%Y-%m-%d')
            except ValueError:
                print(f"Format de date invalide pour le modèle {id_name_original} : {model_date_str}")
                continue

            # Find all date folders
            date_folders = []
            try:
                for entry in os.listdir(aa_directory):
                    entry_path = os.path.join(aa_directory, entry)
                    if os.path.isdir(entry_path):
                        try:
                            folder_date = datetime.strptime(entry, '%Y-%m-%d')
                            date_folders.append(folder_date)
                        except ValueError:
                            continue  # Ignore folders that don't match date format
                print(f"Date folders trouvés dans {aa_directory} : {len(date_folders)}")
            except Exception as e:
                print(f"Erreur lors de la liste des dossiers dans {aa_directory} : {e}")
                continue

            if not date_folders:
                print(f"Aucun dossier de date trouvé dans {aa_directory} pour le modèle {id_name_original}")
                continue

            # Find the closest date <= model_date
            closest_date = max([d for d in date_folders if d <= model_date], default=None)
            if not closest_date:
                print(f"Aucune date appropriée trouvée pour le modèle {id_name_original} avec la date {model_date_str}")
                continue

            closest_date_str = closest_date.strftime('%Y-%m-%d')

            # Build the path to the provider's folder
            date_folder_path = os.path.join(aa_directory, closest_date_str)
            provider_dir = None
            try:
                for entry in os.listdir(date_folder_path):
                    entry_path = os.path.join(date_folder_path, entry)
                    if os.path.isdir(entry_path) and entry.lower() == provider:
                        provider_dir = entry_path
                        break
                if not provider_dir:
                    print(f"Aucun dossier fournisseur trouvé pour '{provider}' à la date {closest_date_str} pour le modèle {id_name_original}")
                    continue
            except Exception as e:
                print(f"Erreur lors de la recherche du dossier fournisseur dans {date_folder_path} : {e}")
                continue

            # Path to speed_performance.csv
            speed_csv_path = os.path.join(provider_dir, 'speed_performance.csv')
            if not os.path.exists(speed_csv_path):
                print(f"Fichier 'speed_performance.csv' non trouvé dans {provider_dir} pour le modèle {id_name_original}")
                continue

            # Load the CSV
            try:
                speed_df = pd.read_csv(speed_csv_path)
                print(f"Fichier 'speed_performance.csv' chargé : {speed_csv_path}")
            except Exception as e:
                print(f"Erreur lors du chargement du CSV {speed_csv_path} pour le modèle {id_name_original} : {e}")
                continue

            # Apply matching strategies on id_name
            matched_row = None
            for strategy in strategies:
                transformed_id_name = strategy(id_name_original)
                # Look for the transformed_id_name in the 'id_name' column of CSV
                matched = speed_df[speed_df['id_name'] == transformed_id_name]
                if not matched.empty:
                    matched_row = matched.iloc[0]
                    print(f"Correspondance trouvée avec la stratégie : {transformed_id_name}")
                    break  # Found, exit the strategies loop

            if matched_row is None:
                print(f"Aucune correspondance trouvée dans le CSV pour le modèle {id_name_original} après application des stratégies")
                continue

            # Extract desired columns
            median_time_to_first_token_seconds = matched_row.get('median_time_to_first_token_seconds', None)
            median_output_tokens_per_second = matched_row.get('median_output_tokens_per_second', None)

            # Add to the model
            model['median_time_to_first_token_seconds'] = median_time_to_first_token_seconds
            model['median_output_tokens_per_second'] = median_output_tokens_per_second
            print(f"Ajout des colonnes de performance pour le modèle {id_name_original}")

            # Calculate blended_price
            blended_price = None
            type_name = model.get('type')
            if type_name == 'text':
                price_input = model.get('price_input')
                price_output = model.get('price_output')
                price_call = model.get('price_call', 0.0)  # Default to 0.0 if not present

                # Convert prices to float
                def parse_price(value):
                    try:
                        return float(value) if value not in [None, '', 'null'] else None
                    except (ValueError, TypeError):
                        return None

                price_input = parse_price(price_input)
                price_output = parse_price(price_output)
                price_call = parse_price(price_call) or 0.0

                if price_input is not None and price_output is not None:
                    blended_price = (3/4) * price_input + (1/4) * price_output + 1000 * price_call
                else:
                    blended_price = None

            elif type_name == 'audiototext':
                price_input = model.get('price_input')

                # Convert price to float
                def parse_price(value):
                    try:
                        return float(value) if value not in [None, '', 'null'] else None
                    except (ValueError, TypeError):
                        return None

                price_input = parse_price(price_input)

                if price_input is not None:
                    blended_price = 1 * price_input
                else:
                    blended_price = None

            elif type_name == 'texttoimage':
                price_output = model.get('price_output')

                # Convert price to float
                def parse_price(value):
                    try:
                        return float(value) if value not in [None, '', 'null'] else None
                    except (ValueError, TypeError):
                        return None

                price_output = parse_price(price_output)

                if price_output is not None:
                    blended_price = 1 * price_output
                else:
                    blended_price = None

            # Assign blended_price
            model['blended_price'] = blended_price
            print(f"blended_price calculé pour le modèle {id_name_original} : {blended_price}")

            updated_models_count += 1

        except Exception as e:
            print(f"Erreur lors du traitement du modèle {model.get('id_name', 'Unknown')} : {e}")
            continue

    # Remplacer les NaN par None dans l'ensemble de la liste des modèles
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

    try:
        api_data = replace_nan_with_none(api_data)
        print("Valeurs NaN remplacées par None dans le JSON.")
    except Exception as e:
        print(f"Erreur lors de la conversion des NaN en None : {e}")
        return

    # Enregistrer le fichier JSON mis à jour
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(api_data, f, ensure_ascii=False, indent=4, allow_nan=False)
        print(f"Le fichier JSON a été mis à jour avec les informations de performance et enregistré à {json_path}.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier JSON mis à jour : {e}")
        return

    print(f"La fonction add_speed_provider_text_AA s'est exécutée avec succès. {updated_models_count} modèles ont été mis à jour.")

