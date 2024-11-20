import os
import pandas as pd
import re

from function_utils.utils_cleaning import clean_model_name
from function_utils.utils_id_name import generate_and_update_id_names

# Fonction principale pour analyser les CSV et générer des fichiers *_idname.csv
def update_model_names_AA(input_dir, output_dir):
    """
    Met à jour les fichiers *_idname.csv pour les différents types de benchmarks dans le dossier AA,
    en ajoutant uniquement les nouveaux noms de modèles sans modifier les lignes existantes.

    :param input_dir: Chemin du répertoire contenant les fichiers AA.
    :param output_dir: Chemin du répertoire où les fichiers *_idname.csv seront générés/actualisés.
    """
    if not os.path.exists(input_dir):
        print(f"Le répertoire {input_dir} n'existe pas.")
        return

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Identifier les sous-dossiers dans AA
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"Sous-dossiers détectés dans {input_dir}: {subdirs}")

    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        csv_files = [f for f in os.listdir(subdir_path) if f.endswith(".csv")]
        print(f"Fichiers CSV détectés dans {subdir_path}: {csv_files}")

        # Déterminer le nom du fichier de sortie
        if re.match(r'^\d{4}-\d{2}-\d{2}$', subdir):  # Vérifie le format de date
            output_csv_name = "AA_text_idname.csv"
        else:
            output_csv_name = f"AA_{subdir.lower()}_idname.csv"

        output_csv_path = os.path.join(output_dir, output_csv_name)

        # Charger le fichier existant ou créer un DataFrame vide si le fichier n'existe pas
        if os.path.exists(output_csv_path):
            existing_df = pd.read_csv(output_csv_path)
        else:
            existing_df = pd.DataFrame(columns=["name"])

        # Charger les modèles existants dans un set pour vérification rapide
        existing_models = set(existing_df["name"].dropna().apply(clean_model_name))

        new_models = set()

        for csv_file in csv_files:
            csv_path = os.path.join(subdir_path, csv_file)
            try:
                # Lire le CSV
                df = pd.read_csv(csv_path)
                # Identifier les colonnes strictement nommées 'Model' ou 'model_name'
                model_columns = [col for col in df.columns if col.lower() in ["model", "model_name"]]
                print(f"Colonnes prises en compte dans {csv_path}: {model_columns}")

                for col in model_columns:
                    # Nettoyer les noms de modèles et les ajouter au set
                    cleaned_models = df[col].dropna().apply(clean_model_name).unique()
                    print(f"Noms extraits depuis la colonne '{col}' de {csv_path}: {cleaned_models}")
                    new_models.update(cleaned_models)

            except Exception as e:
                print(f"Erreur lors du traitement de {csv_path}: {e}")

        # Identifier les nouveaux modèles à ajouter
        models_to_add = new_models - existing_models

        # Ajouter uniquement les nouveaux modèles au DataFrame existant
        if models_to_add:
            new_rows = pd.DataFrame(models_to_add, columns=["name"])
            updated_df = pd.concat([existing_df, new_rows], ignore_index=True)
            updated_df.to_csv(output_csv_path, index=False)
            print(f"Fichier mis à jour : {output_csv_path}")
        else:
            print(f"Aucun nouveau modèle à ajouter dans {output_csv_path}.")


def update_model_names_HF_Livebench_EpochAI(input_dir, output_dir):
    """
    Met à jour un fichier unique [nom_du_dossier]_text_idname.csv pour les modèles
    trouvés dans les fichiers CSV du dossier Hugging_Face, sans modifier les lignes existantes.

    :param input_dir: Chemin du répertoire contenant les fichiers Hugging_Face.
    :param output_dir: Chemin du répertoire où le fichier [nom_du_dossier]_text_idname.csv sera généré/actualisé.
    """
    if not os.path.exists(input_dir):
        print(f"Le répertoire {input_dir} n'existe pas.")
        return

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Récupérer le nom du dossier pour construire le nom du fichier CSV
    base_folder_name = os.path.basename(input_dir.rstrip(os.sep))
    output_csv_name = f"{base_folder_name}_text_idname.csv"
    output_csv_path = os.path.join(output_dir, output_csv_name)

    # Charger le fichier existant ou créer un DataFrame vide si le fichier n'existe pas
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
    else:
        existing_df = pd.DataFrame(columns=["name"])

    # Charger les modèles existants dans un set pour vérification rapide
    existing_models = set(existing_df["name"].dropna().apply(clean_model_name))
    
    new_models = set()

    # Parcourir tous les fichiers dans le dossier Hugging_Face
    for root, _, files in os.walk(input_dir):
        csv_files = [f for f in files if f.endswith(".csv")]
        print(f"Fichiers CSV détectés dans {root}: {csv_files}")

        for csv_file in csv_files:
            csv_path = os.path.join(root, csv_file)
            try:
                # Lire le CSV
                df = pd.read_csv(csv_path)

                # Identifier les colonnes strictement nommées 'Model'
                model_columns = [col for col in df.columns if col.lower() == "model"]
                print(f"Colonnes prises en compte dans {csv_path}: {model_columns}")

                for col in model_columns:
                    # Nettoyer les noms de modèles et ignorer les non-textes
                    cleaned_models = (
                        df[col]
                        .dropna()
                        .apply(lambda x: clean_model_name(x) if isinstance(x, str) else None)
                        .dropna()
                        .unique()
                    )
                    print(f"Noms extraits depuis la colonne '{col}' de {csv_path}: {cleaned_models}")
                    new_models.update(cleaned_models)

            except Exception as e:
                print(f"Erreur lors du traitement de {csv_path}: {e}")

    # Identifier les nouveaux modèles à ajouter
    models_to_add = new_models - existing_models

    # Ajouter uniquement les nouveaux modèles au DataFrame existant
    if models_to_add:
        new_rows = pd.DataFrame(models_to_add, columns=["name"])
        updated_df = pd.concat([existing_df, new_rows], ignore_index=True)
        updated_df.to_csv(output_csv_path, index=False)
        print(f"Fichier mis à jour : {output_csv_path}")
    else:
        print(f"Aucun nouveau modèle à ajouter dans {output_csv_path}.")



def Benchmark_update_id_names(root_directory, examples_directory, openai_api_key):
    """
    Parcourt tous les CSV, y compris ceux des sous-dossiers, dans `root_directory` et met à jour les `id_name`.
    Utilise les fichiers d'exemples pour déterminer le type des modèles.

    :param root_directory: Répertoire racine contenant les fichiers CSV à analyser (inclut les sous-dossiers).
    :param examples_directory: Répertoire contenant les fichiers exemples utilisés pour chaque type.
    :param openai_api_key: Clé API OpenAI pour les appels à l'API.
    """
    # Vérifier que les répertoires existent
    if not os.path.exists(root_directory):
        raise FileNotFoundError(f"Le répertoire racine {root_directory} n'existe pas.")
    if not os.path.exists(examples_directory):
        raise FileNotFoundError(f"Le répertoire des fichiers exemples {examples_directory} n'existe pas.")

    # Liste des fichiers exemples disponibles
    example_files = {
        "audio": os.path.join(examples_directory, "audio_exemple.csv"),
        "image": os.path.join(examples_directory, "image_exemple.csv"),
        "text": os.path.join(examples_directory, "text_exemple.csv")
    }

    # Parcourir tous les fichiers CSV dans le répertoire racine et ses sous-dossiers
    for root, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".csv"):
                # Déterminer le chemin absolu du fichier CSV
                csv_path = os.path.join(root, file)

                # Identifier le type de modèle
                if "audio" in file.lower():
                    model_type = "audio"
                    examples_csv_path = example_files["audio"]
                elif "image" in file.lower():
                    model_type = "image"
                    examples_csv_path = example_files["image"]
                elif "text" in file.lower() or "multimodal" in file.lower():
                    model_type = "text"
                    examples_csv_path = example_files["text"]
                else:
                    print(f"Impossible de déterminer le type pour {file}. Ignoré.")
                    continue

                # Vérifier que le fichier exemple existe pour le type
                if not os.path.exists(examples_csv_path):
                    print(f"Fichier exemple manquant pour {model_type}: {examples_csv_path}. Ignoré.")
                    continue

                # Appliquer la fonction pour générer et mettre à jour les id_name
                print(f"Traitement du fichier : {csv_path} avec le type {model_type}...")
                added_models = generate_and_update_id_names(
                    csv_path=csv_path,
                    examples_csv_path=examples_csv_path,
                    model_type=model_type,
                    openai_api_key=openai_api_key,
                    column_name="name"
                )

                # Afficher les résultats
                if added_models:
                    print(f"Modèles ajoutés pour {file} : {added_models}")
                else:
                    print(f"Aucun modèle ajouté ou mise à jour non requise pour {file}.")


def add_idname_benchmark(benchmark_dir, id_benchmark_dir):
    """
    Ajoute une colonne `id_name` aux fichiers CSV dans benchmark_dir en se basant uniquement
    sur les fichiers correspondants (par préfixe) dans id_benchmark_dir.
    Supprime toutes les colonnes commençant par 'id_name' avant de faire le merge.

    :param benchmark_dir: Répertoire contenant les fichiers de benchmark (avec sous-dossiers).
    :param id_benchmark_dir: Répertoire contenant les fichiers avec les `id_name`.
    """
    if not os.path.exists(benchmark_dir) or not os.path.exists(id_benchmark_dir):
        print("Un des répertoires spécifiés n'existe pas.")
        return

    # Parcourir les fichiers dans id_benchmark_dir
    id_name_files = {}
    for root, _, files in os.walk(id_benchmark_dir):
        subdir_name = os.path.basename(root)
        for f in files:
            if f.endswith("_idname.csv"):
                key = os.path.basename(f).split("_idname")[0]
                id_name_files[(subdir_name, key)] = os.path.join(root, f)

    print(f"Fichiers d'ID_name détectés : {id_name_files}")

    # Parcourir tous les fichiers CSV dans benchmark_dir, y compris les sous-dossiers
    for root, _, files in os.walk(benchmark_dir):
        subdir_name = os.path.basename(root)
        for benchmark_file in files:
            if not benchmark_file.endswith(".csv"):
                continue

            benchmark_file_path = os.path.join(root, benchmark_file)
            print(f"Traitement du fichier benchmark : {benchmark_file_path}")

            try:
                # Lire le fichier benchmark
                benchmark_df = pd.read_csv(benchmark_file_path)

                # Supprimer toutes les colonnes commençant par 'id_name'
                id_name_columns = [col for col in benchmark_df.columns if col.startswith("id_name")]
                if id_name_columns:
                    benchmark_df.drop(columns=id_name_columns, inplace=True)
                    print(f"Colonnes supprimées : {id_name_columns}")

                # Identifier la colonne des modèles (Model ou model_name)
                model_column = None
                for col in benchmark_df.columns:
                    if col.lower() in ["model", "model_name"]:
                        model_column = col
                        break

                if not model_column:
                    print(f"Aucune colonne 'Model' ou 'model_name' trouvée dans {benchmark_file_path}. Ignoré.")
                    continue

                # Nettoyer les modèles pour le merge
                benchmark_df["cleaned_model"] = benchmark_df[model_column].apply(
                    lambda x: clean_model_name(x) if isinstance(x, str) else None
                )

                # Identifier le fichier id_name correspondant
                prefix = "_".join(benchmark_file.split("_")[:2])  # Extraire le préfixe jusqu'au deuxième '_'
                id_name_file = id_name_files.get((subdir_name, prefix))

                if not id_name_file:
                    print(f"Aucun fichier ID_name correspondant trouvé pour {benchmark_file_path}. Ignoré.")
                    continue

                # Charger le fichier id_name et nettoyer les noms
                id_name_df = pd.read_csv(id_name_file)
                if "name" not in id_name_df.columns or "id_name" not in id_name_df.columns:
                    print(f"Colonnes 'name' ou 'id_name' manquantes dans {id_name_file}. Ignoré.")
                    continue

                id_name_df["cleaned_name"] = id_name_df["name"].apply(clean_model_name)

                # Effectuer le merge
                merged_df = benchmark_df.merge(
                    id_name_df[["cleaned_name", "id_name"]],
                    left_on="cleaned_model",
                    right_on="cleaned_name",
                    how="left"
                )

                # Supprimer les colonnes temporaires
                merged_df.drop(columns=["cleaned_model", "cleaned_name"], inplace=True)

                # Sauvegarder le fichier mis à jour
                merged_df.to_csv(benchmark_file_path, index=False)
                print(f"Fichier mis à jour : {benchmark_file_path}")

            except Exception as e:
                print(f"Erreur lors du traitement de {benchmark_file_path}: {e}")


def add_id_name_benchmark_bis(input_dir, id_name_csv, column_names):
    """
    Analyse tous les fichiers CSV dans un répertoire donné et effectue un merge basé sur la colonne id_name.
    
    :param input_dir: Répertoire contenant les fichiers CSV à analyser.
    :param id_name_csv: Chemin du fichier CSV contenant les id_name.
    :param column_names: Liste des noms de colonnes à rechercher pour le merge.
    """
    if not os.path.exists(input_dir) or not os.path.exists(id_name_csv):
        print("Le répertoire d'entrée ou le fichier d'ID_name spécifié n'existe pas.")
        return

    try:
        # Charger le fichier id_name
        id_name_df = pd.read_csv(id_name_csv)
        if "name" not in id_name_df.columns or "id_name" not in id_name_df.columns:
            print(f"Les colonnes 'name' ou 'id_name' sont absentes dans {id_name_csv}.")
            return

        # Nettoyer les noms dans le fichier id_name
        id_name_df["cleaned_name"] = id_name_df["name"].apply(clean_model_name)

    except Exception as e:
        print(f"Erreur lors du chargement du fichier ID_name : {e}")
        return

    # Parcourir tous les fichiers dans le répertoire d'entrée
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(root, file)
            print(f"Traitement du fichier : {file_path}")

            try:
                # Charger le fichier CSV
                benchmark_df = pd.read_csv(file_path)

                # Identifier les colonnes correspondantes pour le merge
                target_columns = [col for col in benchmark_df.columns if col in column_names]
                if not target_columns:
                    print(f"Aucune des colonnes spécifiées {column_names} trouvée dans {file_path}. Ignoré.")
                    continue

                # Supprimer les colonnes id_name existantes
                id_name_columns = [col for col in benchmark_df.columns if col.startswith("id_name")]
                if id_name_columns:
                    benchmark_df.drop(columns=id_name_columns, inplace=True)
                    print(f"Colonnes supprimées : {id_name_columns}")

                # Nettoyer les colonnes cibles pour le merge
                for target_col in target_columns:
                    benchmark_df[f"cleaned_{target_col}"] = benchmark_df[target_col].apply(
                        lambda x: clean_model_name(x) if isinstance(x, str) else None
                    )

                # Effectuer le merge pour chaque colonne cible
                for target_col in target_columns:
                    merged_df = benchmark_df.merge(
                        id_name_df[["cleaned_name", "id_name"]],
                        left_on=f"cleaned_{target_col}",
                        right_on="cleaned_name",
                        how="left"
                    )

                    # Supprimer les colonnes temporaires
                    merged_df.drop(columns=[f"cleaned_{target_col}", "cleaned_name"], inplace=True)

                    # Sauvegarder le fichier mis à jour
                    merged_df.to_csv(file_path, index=False)
                    print(f"Fichier mis à jour : {file_path}")

            except Exception as e:
                print(f"Erreur lors du traitement du fichier {file_path} : {e}")