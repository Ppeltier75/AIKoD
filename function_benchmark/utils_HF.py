import os
import pickle
import re
import pandas as pd
import shutil

def extract_date_from_pickle(filename):
    """
    Extrait la date du nom de fichier au format 'elo_results_YYYYMMDD.pkl'.
    """
    match = re.search(r'elo_results_(\d{8})\.pkl', filename)
    if match:
        date_str = match.group(1)
        return date_str
    else:
        return None

def process_pickle_file(file_path, arena_elo_dir):
    """
    Traite un fichier pickle :
    - Extrait la date du nom du fichier.
    - Supprime le dossier de la date s'il existe déjà.
    - Crée un dossier pour la date.
    - Parcourt récursivement les clés du pickle pour extraire les DataFrames et Series.
    - Crée des dossiers pour les clés uniquement s'il y a des CSV à enregistrer.
    """
    filename = os.path.basename(file_path)
    date_str = extract_date_from_pickle(filename)
    if date_str is None:
        print(f"Impossible d'extraire la date du nom de fichier : {filename}")
        return
    # Chemin du dossier de la date
    date_dir = os.path.join(arena_elo_dir, date_str)
    # Supprimer le dossier de la date s'il existe déjà
    if os.path.exists(date_dir):
        shutil.rmtree(date_dir)
        print(f"Dossier existant '{date_dir}' supprimé.")
    # Créer le dossier pour la date
    os.makedirs(date_dir, exist_ok=True)
    # Charger le fichier pickle
    with open(file_path, 'rb') as f:
        elo_results = pickle.load(f)
    # Parcourir récursivement le dictionnaire
    def save_data_recursive(data, current_dir):
        data_saved = False
        if isinstance(data, dict):
            data_to_save = []
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    data_to_save.append((key, value))
                elif isinstance(value, pd.Series):
                    data_to_save.append((key, value.to_frame()))
                elif isinstance(value, dict):
                    # Traiter le sous-dictionnaire
                    sub_dir = os.path.join(current_dir, key)
                    saved_in_subdir = save_data_recursive(value, sub_dir)
                    data_saved = data_saved or saved_in_subdir
                else:
                    # Ne pas traiter les autres types
                    continue
            if data_to_save:
                if not os.path.exists(current_dir):
                    os.makedirs(current_dir, exist_ok=True)
                for sub_key, df in data_to_save:
                    csv_filename = f"{sub_key}.csv"
                    csv_path = os.path.join(current_dir, csv_filename)
                    # S'assurer que la colonne 'Model' a un en-tête lors de l'enregistrement
                    if df.index.name == 'Model' or df.index.name is None:
                        df = df.reset_index()
                        df.rename(columns={'index': 'Model'}, inplace=True)
                    df.to_csv(csv_path, index=False)
                    print(f"Enregistré CSV : {csv_path}")
                data_saved = True
        return data_saved
    # Démarrer le parcours récursif à partir de la racine du pickle
    save_data_recursive(elo_results, date_dir)

    # Après le traitement, ajuster la structure des répertoires selon vos exigences
    adjust_directory_structure(date_dir)

def adjust_directory_structure(date_dir):
    """
    Ajuste la structure du répertoire :
    - Si le dossier de la date ne contient pas de dossier 'text', crée un dossier 'text' et y déplace tous les sous-dossiers et fichiers.
    - Ensuite, s'il n'y a pas de sous-répertoires dans 'text', crée un répertoire 'full' sous 'text' et y déplace tous les CSV.
    """
    # Vérifier si le répertoire 'text' existe sous date_dir
    text_dir = os.path.join(date_dir, 'text')
    if not os.path.exists(text_dir):
        # Créer le répertoire 'text'
        os.makedirs(text_dir, exist_ok=True)
        # Déplacer tous les sous-dossiers et fichiers dans 'text'
        for item in os.listdir(date_dir):
            item_path = os.path.join(date_dir, item)
            if item != 'text':
                dst_path = os.path.join(text_dir, item)
                shutil.move(item_path, dst_path)
        print(f"Déplacé tous les éléments dans le répertoire 'text' sous '{date_dir}'")
    # Maintenant, vérifier si 'text' n'a pas de sous-répertoires
    subdirs_in_text = [d for d in os.listdir(text_dir) if os.path.isdir(os.path.join(text_dir, d))]
    if not subdirs_in_text:
        # Créer 'full' sous 'text' et y déplacer tous les CSV
        full_dir = os.path.join(text_dir, 'full')
        os.makedirs(full_dir, exist_ok=True)
        for file in os.listdir(text_dir):
            if file.endswith('.csv'):
                src = os.path.join(text_dir, file)
                dst = os.path.join(full_dir, file)
                shutil.move(src, dst)
        print(f"Déplacé les CSV dans le répertoire 'full' sous '{text_dir}'")

def collect_category_data(arena_elo_dir):
    """
    Collecte les données sur toutes les dates et clés pour créer des CSV de catégorie.
    Ne traite que les fichiers 'leaderboard_table_df.csv'.
    """
    # Chemin vers le répertoire 'category'
    category_dir = os.path.join(arena_elo_dir, 'category')
    os.makedirs(category_dir, exist_ok=True)
    # Initialiser un dictionnaire pour stocker les données pour chaque catégorie
    category_data = {}
    # Parcourir les répertoires de date
    date_dirs = [d for d in os.listdir(arena_elo_dir) if os.path.isdir(os.path.join(arena_elo_dir, d)) and d.isdigit()]
    for date_dir in date_dirs:
        date_path = os.path.join(arena_elo_dir, date_dir)
        # Parcourir la structure des répertoires
        for root, dirs, files in os.walk(date_path):
            for file in files:
                if file == 'leaderboard_table_df.csv':
                    # Obtenir le chemin relatif à partir du répertoire de la date
                    rel_path = os.path.relpath(root, date_path)
                    if rel_path == '.':
                        # S'il n'y a pas de sous-répertoire, utiliser 'full' comme clé
                        key = 'full'
                    else:
                        # Construire la clé à partir du chemin relatif
                        key_parts = rel_path.split(os.sep)
                        key = '_'.join(key_parts)
                    # Lire le fichier CSV
                    csv_path = os.path.join(root, file)
                    df = pd.read_csv(csv_path, header=0)
                    # Gérer le cas où la première colonne 'Model' n'a pas d'en-tête
                    if df.columns[0] == 'Unnamed: 0':
                        df.rename(columns={'Unnamed: 0': 'Model'}, inplace=True)
                    # S'assurer que la colonne 'Model' existe
                    if 'Model' not in df.columns:
                        # Essayer de réinitialiser l'index et assigner 'Model' comme nom de colonne
                        df.reset_index(inplace=True)
                        df.rename(columns={'index': 'Model'}, inplace=True)
                    if 'Model' not in df.columns:
                        print(f"Aucune colonne 'Model' trouvée dans {csv_path}")
                        continue
                    # S'assurer que la colonne 'Rating' existe
                    rating_column = None
                    for col in df.columns:
                        if col.lower() in ['rating', 'elo_rating']:
                            rating_column = col
                            break
                    if rating_column is None:
                        print(f"Aucune colonne 'Rating' trouvée dans {csv_path}")
                        continue
                    # Préparer les données
                    models = df['Model']
                    ratings = df[rating_column]
                    # Créer un DataFrame pour cette clé s'il n'existe pas
                    if key not in category_data:
                        category_data[key] = pd.DataFrame({'Model': models})
                    # S'assurer que les modèles sont alignés
                    existing_models = category_data[key]['Model']
                    if not existing_models.equals(models):
                        # Fusionner sur la colonne 'Model'
                        category_data[key] = pd.merge(
                            category_data[key],
                            pd.DataFrame({'Model': models, date_dir: ratings}),
                            on='Model',
                            how='outer'
                        )
                    else:
                        # Ajouter les ratings pour cette date
                        category_data[key][date_dir] = ratings.values
    # Enregistrer les CSV de catégorie
    for key, df in category_data.items():
        # Supprimer les doublons
        df = df.drop_duplicates(subset='Model')
        df = df.set_index('Model')
        # Trier les modèles
        df = df.sort_index()
        # Trier les colonnes (dates)
        df = df.reindex(sorted(df.columns), axis=1)
        # Enregistrer le CSV
        csv_filename = f"{key}_HF.csv"
        csv_path = os.path.join(category_dir, csv_filename)
        # S'assurer que la colonne 'Model' a un en-tête lors de l'enregistrement
        df.to_csv(csv_path)
        print(f"Enregistré CSV de catégorie : {csv_path}")



def extract_date_from_csv(filename):
    """
    Extrait la date du fichier au format YYYYMMDD à partir du nom du fichier leaderboard.
    """
    match = re.search(r'leaderboard_table_(\d{8})', filename)
    if match:
        return match.group(1)
    return None

def update_csv_from_leaderboards_Absolubench(input_dir, output_dir):
    """
    Met à jour ou crée les fichiers HF_AE.csv, HF_MMLU.csv, et HF_MT.csv
    à partir des fichiers leaderboard dans le dossier spécifié.

    :param input_dir: Dossier contenant les fichiers leaderboard.
    :param output_dir: Dossier où les fichiers CSV seront sauvegardés.
    """
    # Chemins des fichiers CSV finaux
    files_to_generate = {
        "Arena Elo rating": os.path.join(output_dir, "HF_AE.csv"),
        "MMLU": os.path.join(output_dir, "HF_MMLU.csv"),
        "MT-bench (score)": os.path.join(output_dir, "HF_MT.csv"),
    }

    # Créer un DataFrame vide pour chaque fichier si non existant
    for key, file_path in files_to_generate.items():
        if not os.path.exists(file_path):
            pd.DataFrame(columns=["Model"]).to_csv(file_path, index=False)

    # Parcourir tous les fichiers leaderboard dans le dossier d'entrée
    for filename in os.listdir(input_dir):
        if filename.startswith("leaderboard_table_") and filename.endswith(".csv"):
            # Extraire la date du fichier
            date = extract_date_from_csv(filename)
            if not date:
                print(f"Impossible d'extraire la date du fichier : {filename}")
                continue

            # Charger le fichier leaderboard
            file_path = os.path.join(input_dir, filename)
            leaderboard_df = pd.read_csv(file_path)

            # Vérifier la présence de la colonne `Model`
            if "Model" not in leaderboard_df.columns:
                print(f"Colonne 'Model' absente dans {filename}. Ignoré.")
                continue

            # Supprimer les doublons dans `Model` et réindexer
            leaderboard_df = leaderboard_df.drop_duplicates(subset=["Model"])
            leaderboard_df.set_index("Model", inplace=True, drop=False)

            # Parcourir les colonnes à traiter
            for column, output_file in files_to_generate.items():
                if column not in leaderboard_df.columns:
                    print(f"Colonne {column} absente dans {filename}. Ignoré.")
                    continue

                # Charger ou initialiser le DataFrame cible
                output_df = pd.read_csv(output_file)

                # Vérifier si la colonne `Model` est bien dans le DataFrame cible
                if "Model" not in output_df.columns:
                    print(f"Colonne 'Model' absente dans {output_file}. Création d'un DataFrame vide.")
                    output_df = pd.DataFrame(columns=["Model"])

                # Ajouter les modèles manquants
                new_models = leaderboard_df.index.difference(output_df["Model"])
                if not new_models.empty:
                    new_rows = pd.DataFrame({"Model": new_models})
                    output_df = pd.concat([output_df, new_rows], ignore_index=True)

                # Réindexer les deux DataFrames pour aligner les données
                output_df.set_index("Model", inplace=True, drop=False)

                # Ajouter ou mettre à jour la colonne pour la date
                try:
                    output_df[date] = leaderboard_df[column]
                except KeyError:
                    print(f"Erreur : colonne {column} absente dans le fichier {filename}.")
                    continue

                # Réinitialiser l'index avant sauvegarde
                output_df.reset_index(drop=True, inplace=True)

                # Sauvegarder le fichier mis à jour
                output_df.to_csv(output_file, index=False)
                print(f"Mise à jour de {output_file} avec les données de {filename} pour la colonne {column}.")


def complete_ae(hf_ae_path, text_full_hf_path):
    """
    Complète le fichier HF_AE.csv en fusionnant avec les données de text_full_HF.csv.
    Ajoute les nouveaux modèles comme nouvelles lignes et trie le fichier par ordre alphabétique.

    :param hf_ae_path: Chemin du fichier HF_AE.csv à compléter.
    :param text_full_hf_path: Chemin du fichier text_full_HF.csv à analyser.
    """
    # Vérifier l'existence des fichiers
    if not os.path.exists(hf_ae_path):
        print(f"Le fichier {hf_ae_path} n'existe pas. Aucune mise à jour possible.")
        return

    if not os.path.exists(text_full_hf_path):
        print(f"Le fichier {text_full_hf_path} n'existe pas. Aucune donnée à fusionner.")
        return

    # Charger les fichiers
    hf_ae_df = pd.read_csv(hf_ae_path)
    text_full_df = pd.read_csv(text_full_hf_path)

    # Vérifier que les deux DataFrames ont bien une colonne `Model`
    if "Model" not in hf_ae_df.columns:
        print(f"Colonne 'Model' absente dans {hf_ae_path}. Fusion impossible.")
        return

    if "Model" not in text_full_df.columns:
        print(f"Colonne 'Model' absente dans {text_full_hf_path}. Fusion impossible.")
        return

    # Ajouter les nouveaux modèles manquants
    new_models = text_full_df[~text_full_df["Model"].isin(hf_ae_df["Model"])]
    if not new_models.empty:
        print(f"{len(new_models)} nouveaux modèles trouvés dans {text_full_hf_path}. Ajout au fichier {hf_ae_path}.")
        hf_ae_df = pd.concat([hf_ae_df, new_models], ignore_index=True)

    # Fusionner les colonnes manquantes
    hf_ae_df.set_index("Model", inplace=True)
    text_full_df.set_index("Model", inplace=True)
    for column in text_full_df.columns:
        if column not in hf_ae_df.columns:
            hf_ae_df[column] = text_full_df[column]

    # Réinitialiser l'index et trier par ordre alphabétique
    hf_ae_df.reset_index(inplace=True)
    hf_ae_df.sort_values(by="Model", inplace=True)

    # Sauvegarder le fichier mis à jour
    hf_ae_df.to_csv(hf_ae_path, index=False)
    print(f"Le fichier {hf_ae_path} a été mis à jour et trié par ordre alphabétique.")