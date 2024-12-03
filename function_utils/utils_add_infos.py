import json
from function_utils.utils_cleaning import clean_model_name
import os
import re
import pandas as pd
from function_utils.utils_merge_id import merge_csv_id_name, select_specific_segments, select_segments_no_order
from collections import Counter
from datetime import datetime
from collections import defaultdict

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

                    # Traiter "code" comme "text" pour la détermination des types
                    temp_input = [mod.replace("code", "text") for mod in modality_input]
                    temp_output = [mod.replace("code", "text") for mod in modality_output]

                    # Si 'api request' est dans modality_input ou modality_output, type = 'task'
                    if "api request" in modality_input or "api request" in modality_output:
                        model_type = "task"
                    # Vérification spéciale pour `text` en input/output avec d'autres modalités
                    elif (
                        "text" in temp_input and
                        "text" in temp_output and
                        any(mod not in ["text", "image", "audio", "video", "code"] for mod in modality_input + modality_output)
                    ):
                        model_type = "task"
                    elif "embedding" in modality_output:
                        model_type = "embeddings"
                    elif ("text" in temp_input and "text" in temp_output) and \
                         ("audio" in modality_input or "audio" in modality_output or
                          "image" in modality_input or "image" in modality_output):
                        model_type = "multimodal"
                    elif ("text" in temp_input and "text" in temp_output):
                        model_type = "text"
                    elif ("text" in temp_input and "image" in temp_output):
                        model_type = "text to image"
                    elif ("image" in temp_input and "image" in temp_output):
                        model_type = "image to image"
                    elif ("image" in temp_input and "text" in temp_output):
                        model_type = "image to text"
                    elif ("audio" in temp_input and "audio" in temp_output):
                        model_type = "audio to audio"
                    elif ("audio" in temp_input and "text" in temp_output):
                        model_type = "audio to text"
                    elif ("text" in temp_input and "audio" in temp_output):
                        model_type = "text to audio"
                    elif ("video" in modality_output or "vidéo" in modality_output):
                        model_type = "video"
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
    Ajoute une date de publication (`date_release`) aux modèles dans le JSON brut.
    Si `date_release` est manquant, cherche la date la plus ancienne dans les clés secondaires où le `id_name` ou `name` apparaît.

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

    # Fonction pour trouver la première date à laquelle un `id_name` ou `name` apparaît
    def find_earliest_date(data, identifier, key="id_name"):
        earliest_date = None
        for provider_data in data.values():
            if not isinstance(provider_data, dict):
                continue
            for date_str, content in provider_data.items():
                if not isinstance(content, dict):
                    continue
                models = content.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    if model.get(key) == identifier:
                        if earliest_date is None or date_str < earliest_date:
                            earliest_date = date_str
        return earliest_date

    # Ajouter ou corriger les `date_release` dans le JSON brut
    for provider, date_dict in data.items():
        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    # Vérifier si la date_release existe déjà
                    if "date_release" in model and model["date_release"]:
                        continue  # Ne pas écraser les dates existantes

                    # Chercher la date_release correspondante au model_name ou au `id_name`
                    model_name = model.get("name", "").strip().lower()
                    id_name = model.get("id_name", "").strip() if model.get("id_name") else None

                    if model_name in model_name_to_date:
                        model["date_release"] = model_name_to_date[model_name]
                    elif id_name:  # Si `id_name` existe
                        model["date_release"] = find_earliest_date(data, id_name, key="id_name")
                    elif model_name:  # Si `id_name` est manquant, utiliser `name`
                        model["date_release"] = find_earliest_date(data, model_name, key="name")

    # Sauvegarder les modifications dans le fichier JSON
    final_output_path = output_path if output_path else json_path
    with open(final_output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    print(f"Les dates de publication ont été ajoutées et enregistrées dans le fichier '{final_output_path}'.")


def process_id_name(id_name):
    """Process id_name by removing 'unknown' and replacing '-' with spaces, and removing non-essential segments."""
    # Split the id_name by '-'
    parts = id_name.split('-')
    # Remove 'unknown' parts and non-essential segments like 'false', 'true'
    parts = [part for part in parts if part.lower() not in ['unknown', 'false', 'true']]
    # Replace '-' with spaces and join
    processed_name = ' '.join(parts)
    return processed_name

def add_os_multi_pplx():
    """
    Fusionne les données du fichier JSON 'pplx_os_multi.json' avec le fichier CSV 'AIKoD_text_infos.csv'.
    Ajoute ou met à jour les colonnes 'is_open_source' et 'is_multimodal' dans le CSV en fonction des données du JSON.
    """
    # Définir les chemins absolus
    try:
        script_dir = os.path.abspath(os.path.dirname(__file__))  # Répertoire du script actuel
        csv_path = os.path.join(script_dir, '..', 'data', 'models_infos', 'AIKoD_text_infos.csv')
        json_path = os.path.join(script_dir, '..', 'data', 'models_infos', 'Perplexity', 'pplx_os_multi.json')
        output_csv_path = os.path.join(script_dir, '..', 'data', 'models_infos', 'AIKoD_text_infos.csv')  # Optionnel : sauvegarder sous un nouveau nom

        print(f"Chemin du fichier CSV : {csv_path}")
        print(f"Chemin du fichier JSON : {json_path}")
    except Exception as e:
        print(f"Erreur lors de la définition des chemins : {e}")
        return

    # Vérifier l'existence du fichier CSV
    if not os.path.exists(csv_path):
        print(f"Erreur : Le fichier CSV '{csv_path}' n'existe pas.")
        return

    # Lire le fichier CSV
    try:
        df_csv = pd.read_csv(csv_path)
        print(f"Fichier CSV chargé : '{csv_path}'")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV '{csv_path}' : {e}")
        return

    # Vérifier la présence de 'id_name' dans le CSV
    if 'id_name' not in df_csv.columns:
        print("Erreur : La colonne 'id_name' est absente du fichier CSV.")
        return

    # Créer 'model_name' à partir de 'id_name'
    try:
        df_csv['model_name'] = df_csv['id_name'].apply(process_id_name).str.strip().str.lower()
        print("Colonne 'model_name' créée à partir de 'id_name'.")
    except Exception as e:
        print(f"Erreur lors de la création de la colonne 'model_name' : {e}")
        return

    # Vérifier l'existence du fichier JSON
    if not os.path.exists(json_path):
        print(f"Erreur : Le fichier JSON '{json_path}' n'existe pas.")
        return

    # Lire le fichier JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            pplx_data = json.load(f)
        print(f"Fichier JSON chargé : '{json_path}'")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON '{json_path}' : {e}")
        return

    # Convertir le JSON en DataFrame
    try:
        df_pplx = pd.DataFrame(pplx_data)
        print("JSON converti en DataFrame.")
    except Exception as e:
        print(f"Erreur lors de la conversion du JSON en DataFrame : {e}")
        return

    # Vérifier les colonnes nécessaires dans le JSON
    required_columns = ['model_name', 'Licence_open_source', 'multimodal']
    missing_columns = [col for col in required_columns if col not in df_pplx.columns]
    if missing_columns:
        print(f"Erreur : Les colonnes suivantes sont absentes du fichier JSON : {missing_columns}")
        return

    # Nettoyer les colonnes 'Licence_open_source' et 'multimodal'
    try:
        def convert_to_bool(val):
            """Convertit une valeur en booléen ou None."""
            if isinstance(val, bool):
                return val
            elif isinstance(val, str):
                val_lower = val.strip().lower()
                if val_lower == 'true':
                    return True
                elif val_lower == 'false':
                    return False
            return None

        df_pplx['is_open_source'] = df_pplx['Licence_open_source'].apply(convert_to_bool)
        df_pplx['is_multimodal'] = df_pplx['multimodal'].apply(convert_to_bool)
        print("Colonnes 'Licence_open_source' et 'multimodal' converties en 'is_open_source' et 'is_multimodal'.")
    except Exception as e:
        print(f"Erreur lors du nettoyage des colonnes booléennes : {e}")
        return

    # Préparer le DataFrame JSON pour la fusion
    try:
        df_pplx_clean = df_pplx[['model_name', 'is_open_source', 'is_multimodal']].copy()
        df_pplx_clean['model_name'] = df_pplx_clean['model_name'].str.lower()  # Assurer la casse
        print("DataFrame JSON préparé pour la fusion.")
    except Exception as e:
        print(f"Erreur lors de la préparation du DataFrame JSON pour la fusion : {e}")
        return

    # Vérifier que 'model_name' est unique dans le JSON
    try:
        if df_pplx_clean['model_name'].duplicated().any():
            print("Attention : Des 'model_name' dupliqués ont été trouvés dans le JSON. Les doublons seront supprimés.")
            # Agréger les valeurs en prenant la première occurrence ou une autre stratégie
            df_pplx_clean = df_pplx_clean.drop_duplicates(subset=['model_name'], keep='first')
            print("Doublons supprimés du DataFrame JSON.")
    except Exception as e:
        print(f"Erreur lors de la vérification des doublons dans le DataFrame JSON : {e}")
        return

    # Supprimer les colonnes 'is_open_source' et 'is_multimodal' existantes dans le CSV pour éviter les conflits
    try:
        for col in ['is_open_source', 'is_multimodal']:
            if col in df_csv.columns:
                df_csv.drop(columns=[col], inplace=True)
                print(f"Colonne existante '{col}' supprimée du CSV pour éviter les conflits.")
            else:
                print(f"Aucune colonne '{col}' trouvée dans le CSV.")
    except Exception as e:
        print(f"Erreur lors de la suppression des colonnes existantes dans le CSV : {e}")
        return

    # Fusionner le CSV et le JSON sur 'model_name'
    try:
        df_merged = pd.merge(df_csv, df_pplx_clean, on='model_name', how='left')
        print("Fusion des DataFrames réussie.")
    except Exception as e:
        print(f"Erreur lors de la fusion des DataFrames : {e}")
        return

    # Remplacer les NaN par None (optionnel)
    try:
        df_merged['is_open_source'] = df_merged['is_open_source'].where(pd.notnull(df_merged['is_open_source']), None)
        df_merged['is_multimodal'] = df_merged['is_multimodal'].where(pd.notnull(df_merged['is_multimodal']), None)
        print("Valeurs NaN remplacées par None dans les colonnes 'is_open_source' et 'is_multimodal'.")
    except Exception as e:
        print(f"Erreur lors du remplacement des valeurs NaN : {e}")
        return

    # Sauvegarder le DataFrame mis à jour
    try:
        df_merged.to_csv(output_csv_path, index=False)
        print(f"Fichier CSV mis à jour enregistré sous '{output_csv_path}'.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier CSV mis à jour : {e}")
        return

    print("Fusion terminée avec succès.")


def add_speed_provider_text_AA():
    """
    Ajoute les colonnes 'median_time_to_first_token_seconds' et 'median_output_tokens_per_second'
    aux modèles de type 'text' dans le fichier JSON 'API_date_v4.8.json' en se basant sur les fichiers
    'speed_performance.csv' situés dans les sous-dossiers du répertoire AA.
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

    # Définir le chemin de base relatif au script actuel
    base_path = os.path.abspath(os.path.dirname(__file__))
    
    # Définir le chemin relatif vers le fichier JSON API_date_v4.8.json
    api_json_path = os.path.join(base_path, '..', 'data', 'API', 'API_date_v4.8.json')
    
    # Définir le chemin relatif vers le répertoire AA
    aa_directory = os.path.join(base_path, '..', 'data', 'benchmark', 'AA')
    
    # Vérifier l'existence du répertoire AA
    if not os.path.exists(aa_directory):
        print(f"Le répertoire AA spécifié n'existe pas : {aa_directory}")
        return
    
    # Vérifier l'existence du fichier JSON
    if not os.path.exists(api_json_path):
        print(f"Le fichier JSON spécifié n'existe pas : {api_json_path}")
        return
    
    # Charger le JSON existant
    try:
        with open(api_json_path, 'r', encoding='utf-8') as f:
            api_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON dans le fichier {api_json_path} : {e}")
        return
    
    # Parcourir les modèles dans le JSON
    for date_key, content in api_data.items():
        models_list = content.get('models_list', {}).get('text', [])
        for model in models_list:
            if model.get('type') != 'text':
                continue  # Just in case, though models_list is already 'text'
            
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
            for entry in os.listdir(aa_directory):
                entry_path = os.path.join(aa_directory, entry)
                if os.path.isdir(entry_path):
                    try:
                        folder_date = datetime.strptime(entry, '%Y-%m-%d')
                        date_folders.append(folder_date)
                    except ValueError:
                        continue  # Ignore folders that don't match date format

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
            for entry in os.listdir(date_folder_path):
                entry_path = os.path.join(date_folder_path, entry)
                if os.path.isdir(entry_path) and entry.lower() == provider:
                    provider_dir = entry_path
                    break

            if not provider_dir:
                print(f"Aucun dossier fournisseur trouvé pour '{provider}' à la date {closest_date_str} pour le modèle {id_name_original}")
                continue

            # Path to speed_performance.csv
            speed_csv_path = os.path.join(provider_dir, 'speed_performance.csv')
            if not os.path.exists(speed_csv_path):
                print(f"Fichier 'speed_performance.csv' non trouvé dans {provider_dir} pour le modèle {id_name_original}")
                continue

            # Load the CSV
            try:
                speed_df = pd.read_csv(speed_csv_path)
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
                    break  # Found, exit the strategies loop

            # Correction de la condition de vérification
            if matched_row is None:
                print(f"Aucune correspondance trouvée dans le CSV pour le modèle {id_name_original} après application des stratégies")
                continue

            # Extract desired columns
            median_time_to_first_token_seconds = matched_row.get('median_time_to_first_token_seconds', None)
            median_output_tokens_per_second = matched_row.get('median_output_tokens_per_second', None)

            # Add to the model
            model['median_time_to_first_token_seconds'] = median_time_to_first_token_seconds
            model['median_output_tokens_per_second'] = median_output_tokens_per_second

    # Sauvegarder le JSON mis à jour
    try:
        with open(api_json_path, 'w', encoding='utf-8') as f:
            json.dump(api_data, f, ensure_ascii=False, indent=4)
        print(f"Le fichier JSON a été mis à jour avec les informations de performance et enregistré à {api_json_path}.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier JSON mis à jour : {e}")

