import os
import pandas as pd
import json
import numpy as np

from datetime import datetime


from function_utils.utils_merge_id import select_specific_segments, select_segments_no_order, replace_segment

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


def add_provider_infos_texttoimage(json_path):
    """
    Enrichit les modèles de type 'texttoimage' dans le fichier JSON avec l'information 'speed_provider_index'
    provenant du fichier CSV spécifié.
    
    :param json_path: Chemin vers le fichier JSON à modifier.
    """
    # Chemin vers le fichier CSV spécifique
    csv_path = os.path.join("data", "benchmark", "AA", "texttoimage", "AA_texttoimage_2024-11-19.csv")
    
    # Définition des stratégies de correspondance avec une nouvelle stratégie pour remplacer le 5ème segment
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
        lambda x: replace_segment(x, 4, 'unknown'),  # Nouvelle stratégie : Remplacer le 5ème segment par 'unknown'
    ]
    
    # Charger le fichier JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Données JSON chargées depuis {json_path}")
    except FileNotFoundError:
        print(f"Erreur : Le fichier JSON spécifié n'existe pas : {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON dans le fichier {json_path} : {e}")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON {json_path} : {e}")
        return
    
    # Charger le fichier CSV
    if not os.path.exists(csv_path):
        print(f"Erreur : Le fichier CSV spécifié n'existe pas : {csv_path}")
        return
    
    try:
        csv_df = pd.read_csv(csv_path)
        print(f"Fichier CSV chargé depuis {csv_path}")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV {csv_path} : {e}")
        return
    
    # Parcourir les modèles de type 'texttoimage'
    for model in data:
        if model.get('type') != 'texttoimage':
            continue
        
        json_provider = model.get('provider', '').strip().lower()
        json_id_name = model.get('id_name', '').strip()
        
        if not json_id_name or not json_provider:
            continue
        
        # Chercher une correspondance dans le CSV
        speed_provider_index = None
        for strategy in strategies:
            transformed_id = strategy(json_id_name)
            # Filtrer le DataFrame avec id_name transformé et provider égal
            matched_rows = csv_df[
                (csv_df['id_name'] == transformed_id) & 
                (csv_df['Provider'].str.strip().str.lower() == json_provider)
            ]
            if not matched_rows.empty:
                # Prendre la première correspondance trouvée
                speed_provider_index = matched_rows.iloc[0]['Median Generation Time (s)']
                break  # Arrêter après la première correspondance trouvée
        
        if speed_provider_index is not None:
            model['speed_provider_index'] = speed_provider_index
            print(f"Modèle '{model.get('model_name')}' enrichi avec speed_provider_index: {speed_provider_index}")
        else:
            model['speed_provider_index'] = None
            print(f"Aucune correspondance trouvée pour le modèle '{model.get('model_name')}' avec id_name '{json_id_name}' et provider '{json_provider}'")
    
    # Supprimer les champs temporaires commençant par '_parsed_'
    for model in data:
        keys_to_remove = [key for key in model if key.startswith('_parsed_')]
        for key in keys_to_remove:
            del model[key]
    
    # Écrire les données enrichies dans le même fichier JSON
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Données JSON mises à jour et enregistrées à {json_path}")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier JSON mis à jour : {e}")
        return
    
    print("La fonction add_provider_infos_texttoimage s'est exécutée avec succès.")


    
def add_provider_infos_audiototext(json_path):
    """
    Enrichit les modèles de type 'audiototext' dans le fichier JSON avec les informations
    'quality_provider_index' et 'speed_provider_index' provenant du fichier CSV spécifié.
    Pour 'quality_provider_index', la valeur est transformée par 1 - (valeur / 100).
    
    :param json_path: Chemin vers le fichier JSON à modifier.
    """
    import os
    import json
    import pandas as pd

    # Chemin vers le fichier CSV spécifique
    csv_path = os.path.join("data", "benchmark", "AA", "audiototext", "AA_audiototext_2024-11-19.csv")
    
    # Définition des fonctions de stratégie de correspondance
    def select_specific_segments(id_name, indices):
        """
        Sélectionne des segments spécifiques d'un id_name basé sur les indices fournis.
        """
        segments = id_name.split('-')
        selected = [segments[i - 1] for i in indices if 0 < i <= len(segments)]
        return '-'.join(selected)
    
    def select_segments_no_order(id_name, indices):
        """
        Sélectionne des segments spécifiques d'un id_name sans tenir compte de l'ordre des segments.
        """
        segments = id_name.split('-')
        selected = sorted([segments[i - 1] for i in indices if 0 < i <= len(segments)])
        return '-'.join(selected)
    
    def normalize_id_name(id_name):
        """
        Remplace certains segments spécifiques par des valeurs génériques pour faciliter la correspondance.
        """
        return id_name.replace('transcription', 'unknown')
    
    # Définition des stratégies de correspondance avec normalisation
    strategies = [
        lambda x: x,  # Correspondance exacte
        lambda x: normalize_id_name(x),
        lambda x: select_specific_segments(normalize_id_name(x), [1, 2, 4]),
        lambda x: select_segments_no_order(normalize_id_name(x), [1, 2, 4]),
        lambda x: select_specific_segments(normalize_id_name(x), [1, 2, 3]),
        lambda x: select_segments_no_order(normalize_id_name(x), [1, 2, 3]),
        lambda x: select_specific_segments(normalize_id_name(x), [1, 4]),
        lambda x: select_segments_no_order(normalize_id_name(x), [1, 4]),
        lambda x: select_specific_segments(normalize_id_name(x), [1, 2]),
        lambda x: select_segments_no_order(normalize_id_name(x), [1, 2]),
    ]
    
    # Charger le fichier JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Données JSON chargées depuis {json_path}")
    except FileNotFoundError:
        print(f"Erreur : Le fichier JSON spécifié n'existe pas : {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON dans le fichier {json_path} : {e}")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON {json_path} : {e}")
        return
    
    # Charger le fichier CSV
    if not os.path.exists(csv_path):
        print(f"Erreur : Le fichier CSV spécifié n'existe pas : {csv_path}")
        return
    
    try:
        csv_df = pd.read_csv(csv_path)
        print(f"Fichier CSV chargé depuis {csv_path}")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV {csv_path} : {e}")
        return
    
    # Parcourir les modèles de type 'audiototext'
    for model in data:
        if model.get('type') != 'audiototext':
            continue
        
        json_provider = model.get('provider', '').strip().lower()
        json_id_name = model.get('id_name', '').strip()
        
        if not json_id_name or not json_provider:
            continue
        
        # Chercher une correspondance dans le CSV
        quality_provider_index = None
        speed_provider_index = None
        for strategy in strategies:
            transformed_id = strategy(json_id_name)
            # Filtrer le DataFrame avec id_name transformé et provider égal
            matched_rows = csv_df[
                (csv_df['id_name'] == transformed_id) & 
                (csv_df['Provider'].str.strip().str.lower() == json_provider)
            ]
            if not matched_rows.empty:
                # Prendre la première correspondance trouvée
                raw_quality = matched_rows.iloc[0]['Word Error Rate (%)']
                raw_speed = matched_rows.iloc[0]['Median Speed Factor']
                
                # Transformation de quality_provider_index
                try:
                    quality_value = float(raw_quality)
                    quality_provider_index = 1 - (quality_value / 100)
                except (ValueError, TypeError):
                    quality_provider_index = None
                    print(f"Valeur invalide pour 'Word Error Rate (%)' : {raw_quality} dans le modèle '{model.get('model_name')}'")
                
                # Transformation de speed_provider_index (pas de transformation spécifiée, donc directement)
                try:
                    speed_provider_index = float(raw_speed)
                except (ValueError, TypeError):
                    speed_provider_index = None
                    print(f"Valeur invalide pour 'Median Speed Factor' : {raw_speed} dans le modèle '{model.get('model_name')}'")
                
                break  # Arrêter après la première correspondance trouvée
        
        if quality_provider_index is not None and speed_provider_index is not None:
            model['quality_provider_index'] = quality_provider_index
            model['speed_provider_index'] = speed_provider_index
            print(f"Modèle '{model.get('model_name')}' enrichi avec quality_provider_index: {quality_provider_index} et speed_provider_index: {speed_provider_index}")
        else:
            model['quality_provider_index'] = None
            model['speed_provider_index'] = None
            print(f"Aucune correspondance valide trouvée pour le modèle '{model.get('model_name')}' avec id_name '{json_id_name}' et provider '{json_provider}'")
    
    # Supprimer les champs temporaires commençant par '_parsed_'
    for model in data:
        keys_to_remove = [key for key in model if key.startswith('_parsed_')]
        for key in keys_to_remove:
            del model[key]
    
    # Écrire les données enrichies dans le même fichier JSON
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Données JSON mises à jour et enregistrées à {json_path}")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier JSON mis à jour : {e}")
        return
    
    print("La fonction add_provider_infos_audiototext s'est exécutée avec succès.")


import os
import json
from datetime import datetime
from collections import defaultdict

def Adapting_idname_aikod(json_path):
    """
    Met à jour les champs 'id_name' des modèles de types 'text', 'texttoimage', et 'audiototext'
    dans le fichier JSON spécifié. Si le 4ème segment de 'id_name' est 'unknown', la fonction
    recherche les modèles similaires avec des versions connues et met à jour le 4ème segment
    avec la version la plus élevée trouvée pour des dates similaires ou antérieures.

    Seuls les modèles dont le 'id_name' a été modifié sont affichés.

    :param json_path: Chemin vers le fichier JSON à modifier.
    """
    # Définir une fonction pour générer une clé basée sur 'id_name' selon le type
    def generate_key(id_name, model_type):
        segments = id_name.split('-')
        if model_type == 'text':
            # Remplacer les segments 4 et 5 par 'unknown'
            if len(segments) >= 5:
                segments_key = segments.copy()
                segments_key[3] = 'unknown'  # 4ème segment
                segments_key[4] = 'unknown'  # 5ème segment
            else:
                # Remplir avec 'unknown' jusqu'à 5 segments
                segments_key = segments.copy()
                while len(segments_key) < 5:
                    segments_key.append('unknown')
                segments_key[3] = 'unknown'
                segments_key[4] = 'unknown'
        else:  # 'texttoimage' ou 'audiototext'
            # Remplacer uniquement le 4ème segment par 'unknown'
            if len(segments) >= 4:
                segments_key = segments.copy()
                segments_key[3] = 'unknown'  # 4ème segment
            else:
                # Remplir avec 'unknown' jusqu'à 4 segments
                segments_key = segments.copy()
                while len(segments_key) < 4:
                    segments_key.append('unknown')
                segments_key[3] = 'unknown'
        key = '-'.join(segments_key)
        return key

    # Charger les données depuis le fichier JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Données JSON chargées depuis {json_path}")
    except FileNotFoundError:
        print(f"Erreur : Le fichier JSON spécifié n'existe pas : {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON dans le fichier {json_path} : {e}")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON {json_path} : {e}")
        return

    # Créer un mapping des clés aux versions disponibles avec leurs dates
    key_to_versions = defaultdict(list)
    for model in data:
        model_type = model.get('type', '').strip().lower()
        if model_type not in ['text', 'texttoimage', 'audiototext']:
            continue

        id_name = model.get('id_name', '').strip()
        if not id_name:
            continue

        key = generate_key(id_name, model_type)
        segments = id_name.split('-')

        if model_type == 'text':
            # Pour 'text', considérer les segments 4 et 5
            if len(segments) >= 5:
                version_segment = segments[3]  # 4ème segment
            else:
                version_segment = 'unknown'  # Si moins de 5 segments

            if version_segment != 'unknown':
                try:
                    version = float(version_segment)
                except ValueError:
                    # Ignorer les segments de version non numériques
                    continue

                # Parser la date du modèle
                date_str = model.get('date', '').strip()
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d') if date_str else None
                except ValueError:
                    # Ignorer les modèles avec un format de date invalide
                    date_obj = None

                key_to_versions[key].append((version, date_obj))
        else:  # 'texttoimage' ou 'audiototext'
            # Pour ces types, considérer uniquement le 4ème segment
            if len(segments) >= 4:
                version_segment = segments[3]  # 4ème segment
            else:
                version_segment = 'unknown'

            if version_segment != 'unknown':
                try:
                    version = float(version_segment)
                except ValueError:
                    # Ignorer les segments de version non numériques
                    continue

                # Parser la date du modèle
                date_str = model.get('date', '').strip()
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d') if date_str else None
                except ValueError:
                    # Ignorer les modèles avec un format de date invalide
                    date_obj = None

                key_to_versions[key].append((version, date_obj))

    # Déterminer la version maximale par clé en fonction des dates
    key_to_max_version = {}
    for key, versions_dates in key_to_versions.items():
        # Trier les versions par date croissante
        sorted_versions = sorted(
            versions_dates, 
            key=lambda x: (x[1] if x[1] else datetime.min, x[0])
        )
        max_version = None
        for version, date_obj in sorted_versions:
            if max_version is None or version > max_version:
                max_version = version
        key_to_max_version[key] = max_version

    # Mettre à jour les modèles avec segment 4 'unknown' en utilisant la version maximale trouvée
    for model in data:
        model_type = model.get('type', '').strip().lower()
        if model_type not in ['text', 'texttoimage', 'audiototext']:
            continue

        id_name = model.get('id_name', '').strip()
        if not id_name:
            continue

        segments = id_name.split('-')
        if len(segments) < 4:
            continue

        # Vérifier si le 4ème segment est 'unknown'
        if model_type == 'text':
            if len(segments) >= 4 and segments[3] != 'unknown':
                continue  # Ignorer les modèles dont le 4ème segment n'est pas 'unknown'
        else:  # 'texttoimage' ou 'audiototext'
            if segments[3] != 'unknown':
                continue  # Ignorer les modèles dont le 4ème segment n'est pas 'unknown'

        key = generate_key(id_name, model_type)
        max_version = key_to_max_version.get(key)

        if max_version is not None:
            new_id_name_segments = segments.copy()
            new_id_name_segments[3] = str(max_version)
            new_id_name = '-'.join(new_id_name_segments)

            if new_id_name != id_name:
                model['id_name'] = new_id_name
                print(f"Modèle '{model.get('model_name')}' id_name mis à jour de '{id_name}' à '{new_id_name}'")

    # Supprimer les champs temporaires commençant par '_parsed_'
    for model in data:
        keys_to_remove = [key for key in model if key.startswith('_parsed_')]
        for key in keys_to_remove:
            del model[key]

    # Écrire les données mises à jour dans le même fichier JSON
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Données JSON mises à jour et enregistrées à {json_path}")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier JSON mis à jour : {e}")
        return

    print("La fonction Adapting_idname_aikod s'est exécutée avec succès.")
