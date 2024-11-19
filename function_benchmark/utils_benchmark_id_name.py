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
