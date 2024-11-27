import os
import pandas as pd
import json
import re
from collections import Counter
from function_utils.utils_merge_id import generate_partial_key

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


def AIKoD_text_infos(json_path, text_infos_csv_path):
    """
    Analyse un fichier JSON pour les modèles avec type 'text' et ajoute des informations
    aux fichiers CSV existants dans le répertoire `output_dir`.

    :param json_path: Chemin du fichier JSON contenant les données des modèles.
    :param output_dir: Répertoire contenant les fichiers _infos.csv à mettre à jour.
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

        # Analyser id_name pour combler les informations manquantes
        analyzed_number_of_parameters, analyzed_context_window, analyzed_finetuned = analyze_id_name(id_name)
        number_of_parameters = number_of_parameters or analyzed_number_of_parameters
        context_window = context_window or analyzed_context_window
        finetuned = finetuned or analyzed_finetuned

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

    # Fusionner avec le fichier existant (toujours remplacer pour garantir la mise à jour)
    text_infos_df = pd.merge(
        text_infos_df,
        updates_df,
        on="id_name",
        how="left",
        suffixes=("", "_new"),
    )

    # Toujours remplacer les colonnes avec les nouvelles valeurs majoritaires
    for col in ["number_of_parameters", "context_window", "finetuned", "company", "date_release"]:
        if f"{col}_new" in text_infos_df.columns:
            text_infos_df[col] = text_infos_df[f"{col}_new"]

    # Supprimer les colonnes temporaires si elles existent
    text_infos_df.drop(
        columns=[col for col in ["number_of_parameters_new", "context_window_new", "finetuned_new", "company_new", "date_release_new"]
                 if col in text_infos_df.columns],
        inplace=True,
    )

    # Sauvegarder le fichier mis à jour
    text_infos_df.to_csv(text_infos_csv_path, index=False)
    print(f"Le fichier {text_infos_csv_path} a été mis à jour avec succès.")


def determine_suffix(filename):
    """
    Détermine le suffixe basé sur le nom du fichier.
    
    Args:
        filename (str): Nom du fichier CSV.
    
    Returns:
        str or None: Suffixe déterminé ou None si non déterminé.
    """
    if 'HF_text_AE' in filename:
        return 'AE'
    elif 'HF_text_MMLU' in filename:
        return 'MMLU'
    elif 'HF_text_MT' in filename:
        return 'MT'
    elif 'AA_quality' in filename:
        return 'AA'
    elif 'Livebench_text' in filename:
        return 'Livebench_rating'
    else:
        return None

def add_ratings_text(directory_merge, base_file, segments_to_keep=[1, 4, 6], separator='-'):
    """
    Ajoute des colonnes de ratings moyennes au fichier de base en fusionnant avec les fichiers de fusion présents dans directory_merge.
    
    Args:
        directory_merge (str): Chemin vers le répertoire contenant les fichiers CSV de fusion.
        base_file (str): Chemin vers le fichier CSV de base à mettre à jour.
        segments_to_keep (list, optional): Liste des indices de segments à garder pour la clé partielle. Defaults to [1, 4, 6].
        separator (str, optional): Séparateur utilisé dans 'id_name'. Defaults to '-'.
    
    Returns:
        None
    """
    # Vérifier l'existence du fichier de base
    if not os.path.exists(base_file):
        print(f"Le fichier de base {base_file} n'existe pas.")
        return
    
    # Charger le fichier de base
    base_df = pd.read_csv(base_file)
    
    # Générer la clé partielle pour base_df
    base_df = base_df.copy()
    base_df['partial_key'] = generate_partial_key(base_df['id_name'], segments_to_keep, separator)
    
    # Parcourir tous les fichiers de fusion dans directory_merge et ses sous-répertoires
    for root, dirs, files in os.walk(directory_merge):
        for file in files:
            if file.endswith('.csv'):
                merge_file_path = os.path.join(root, file)
                
                # Déterminer le suffixe basé sur le nom du fichier
                suffix = determine_suffix(file)
                if not suffix:
                    print(f"Suffixe non déterminé pour le fichier {file}. Skipping...\n")
                    continue
                
                try:
                    # Charger le fichier de fusion
                    merge_df = pd.read_csv(merge_file_path)
                    
                    # Traitement spécifique selon le suffixe
                    if suffix in ['AE', 'MMLU', 'MT']:
                        # Identifier les colonnes de date (noms entièrement numériques)
                        date_columns = [col for col in merge_df.columns if col.isdigit()]
                        print(f"Traitement du fichier {file} avec suffixe '{suffix}'. Colonnes de date trouvées : {date_columns}")
                        
                        if not date_columns:
                            print(f"Aucune colonne de date trouvée dans {file}. Skipping...\n")
                            continue
                        
                        # Convertir les colonnes de date en numérique, forçant les erreurs à NaN
                        merge_df[date_columns] = merge_df[date_columns].apply(pd.to_numeric, errors='coerce')
                        
                        # Calculer la moyenne des colonnes de date pour chaque ligne, en ignorant les valeurs NaN
                        merge_df[suffix] = merge_df[date_columns].mean(axis=1, skipna=True)
                        
                        # Générer la clé partielle pour merge_df
                        merge_df['partial_key'] = generate_partial_key(merge_df['id_name'], segments_to_keep, separator)
                        
                        # Supprimer les doublons dans merge_df basé sur partial_key, garder la première occurrence
                        ratings_df_unique = merge_df.drop_duplicates(subset='partial_key', keep='first')
                        
                        # Créer un mapping de partial_key vers le rating moyen
                        ratings_mapping = ratings_df_unique.set_index('partial_key')[suffix]
                        
                        # Assigner le rating moyen au base_df en utilisant le mapping
                        base_df[suffix] = base_df['partial_key'].map(ratings_mapping)
                        
                        # Afficher le nombre de lignes mises à jour
                        updated_count = base_df[suffix].notna().sum()
                        print(f"Ajout de la colonne '{suffix}' au fichier de base depuis {file}. Nombre de lignes mises à jour : {updated_count}\n")
                    
                    elif suffix == 'AA':
                        # Colonnes spécifiques à fusionner
                        columns_to_merge = ['chatbot_arena_elo', 'mmlu', 'gpqa', 'humaneval', 'math', 'mgsm']
                        print(f"Traitement du fichier {file} avec suffixe '{suffix}'. Colonnes à fusionner : {columns_to_merge}")
                        
                        # Vérifier si les colonnes existent
                        if not all(col in merge_df.columns for col in columns_to_merge):
                            print(f"Certaines colonnes requises ne sont pas présentes dans {file}. Skipping...\n")
                            continue
                        
                        # Convertir les colonnes en numérique, forçant les erreurs à NaN
                        merge_df[columns_to_merge] = merge_df[columns_to_merge].apply(pd.to_numeric, errors='coerce')
                        
                        # Calculer la moyenne des colonnes spécifiques pour chaque ligne
                        merge_df[suffix] = merge_df[columns_to_merge].mean(axis=1, skipna=True)
                        
                        # Générer la clé partielle pour merge_df
                        merge_df['partial_key'] = generate_partial_key(merge_df['id_name'], segments_to_keep, separator)
                        
                        # Supprimer les doublons dans merge_df basé sur partial_key, garder la première occurrence
                        ratings_df_unique = merge_df.drop_duplicates(subset='partial_key', keep='first')
                        
                        # Créer un mapping de partial_key vers le rating moyen
                        ratings_mapping = ratings_df_unique.set_index('partial_key')[suffix]
                        
                        # Assigner le rating moyen au base_df en utilisant le mapping
                        base_df[suffix] = base_df['partial_key'].map(ratings_mapping)
                        
                        # Afficher le nombre de lignes mises à jour
                        updated_count = base_df[suffix].notna().sum()
                        print(f"Ajout de la colonne '{suffix}' au fichier de base depuis {file}. Nombre de lignes mises à jour : {updated_count}\n")
                    
                    elif suffix == 'Livebench_rating':
                        # Colonne spécifique à fusionner
                        column_to_merge = 'Global Average'
                        print(f"Traitement du fichier {file} avec suffixe '{suffix}'. Colonne à fusionner : {column_to_merge}")
                        
                        if column_to_merge not in merge_df.columns:
                            print(f"La colonne '{column_to_merge}' n'est pas présente dans {file}. Skipping...\n")
                            continue
                        
                        # Convertir la colonne en numérique, forçant les erreurs à NaN
                        merge_df[column_to_merge] = pd.to_numeric(merge_df[column_to_merge], errors='coerce')
                        
                        # Renommer la colonne pour correspondre au suffixe
                        merge_df[suffix] = merge_df[column_to_merge]
                        
                        # Générer la clé partielle pour merge_df
                        merge_df['partial_key'] = generate_partial_key(merge_df['id_name'], segments_to_keep, separator)
                        
                        # Supprimer les doublons dans merge_df basé sur partial_key, garder la première occurrence
                        ratings_df_unique = merge_df.drop_duplicates(subset='partial_key', keep='first')
                        
                        # Créer un mapping de partial_key vers le rating
                        ratings_mapping = ratings_df_unique.set_index('partial_key')[suffix]
                        
                        # Assigner le rating au base_df en utilisant le mapping
                        base_df[suffix] = base_df['partial_key'].map(ratings_mapping)
                        
                        # Afficher le nombre de lignes mises à jour
                        updated_count = base_df[suffix].notna().sum()
                        print(f"Ajout de la colonne '{suffix}' au fichier de base depuis {file}. Nombre de lignes mises à jour : {updated_count}\n")
                    
                except Exception as e:
                    print(f"Erreur lors du traitement du fichier {file}: {e}\n")
    
    # Sauvegarder le fichier de base mis à jour
    base_df.drop(columns=['partial_key'], inplace=True)
    base_df.to_csv(base_file, index=False)
    print(f"Fichier de base mis à jour sauvegardé : {base_file}")