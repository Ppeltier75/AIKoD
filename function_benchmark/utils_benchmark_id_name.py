import os
import pandas as pd
import re

from function_utils.utils_cleaning import clean_model_name

# Fonction principale pour analyser les CSV et générer des fichiers *_idname.csv
def update_model_names_AA(input_dir, output_dir):
    """
    Crée des fichiers *_idname.csv pour les différents types de benchmarks dans le dossier AA.
    Ne prend en compte que les colonnes nommées 'Model' ou 'model_name'.

    :param input_dir: Chemin du répertoire contenant les fichiers AA.
    :param output_dir: Chemin du répertoire où les fichiers *_idname.csv seront générés.
    """
    if not os.path.exists(input_dir):
        print(f"Le répertoire {input_dir} n'existe pas.")
        return

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Dictionnaire pour stocker les noms par type
    type_to_models = {}

    # Identifier les sous-dossiers dans AA
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"Sous-dossiers détectés dans {input_dir}: {subdirs}")

    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        csv_files = [f for f in os.listdir(subdir_path) if f.endswith(".csv")]
        print(f"Fichiers CSV détectés dans {subdir_path}: {csv_files}")

        # Si le sous-dossier correspond à une date, ajouter à `AA_text_idname`
        if re.match(r'^\d{4}-\d{2}-\d{2}$', subdir):  # Vérifie le format de date
            type_key = "AA_text_idname"
        else:  # Sinon, utiliser le nom du sous-dossier pour définir le type
            type_key = f"AA_{subdir.lower()}_idname"

        # Initialiser le set pour ce type s'il n'existe pas
        if type_key not in type_to_models:
            type_to_models[type_key] = set()

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
                    type_to_models[type_key].update(cleaned_models)

            except Exception as e:
                print(f"Erreur lors du traitement de {csv_path}: {e}")

    # Enregistrer les résultats dans des fichiers CSV
    for type_key, models in type_to_models.items():
        output_csv_path = os.path.join(output_dir, f"{type_key}.csv")
        
        # Convertir toutes les valeurs en chaînes avant de trier
        cleaned_models = [str(model) for model in models]
        models_df = pd.DataFrame(sorted(cleaned_models), columns=["name"])
        
        # Sauvegarder dans un fichier CSV
        models_df.to_csv(output_csv_path, index=False)
        print(f"Fichier sauvegardé : {output_csv_path}")


def update_model_names_HF_Livebench_EpochAI(input_dir, output_dir):
    """
    Analyse tous les fichiers CSV dans le dossier Hugging_Face pour extraire
    les noms de modèles à partir des colonnes 'Model'. Génère un fichier unique [nom_du_dossier]_text_idname.csv.

    :param input_dir: Chemin du répertoire contenant les fichiers Hugging_Face.
    :param output_dir: Chemin du répertoire où le fichier [nom_du_dossier]_text_idname.csv sera généré.
    """
    if not os.path.exists(input_dir):
        print(f"Le répertoire {input_dir} n'existe pas.")
        return

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Récupérer le nom du dossier pour construire le nom du fichier CSV
    base_folder_name = os.path.basename(input_dir.rstrip(os.sep))
    output_csv_name = f"{base_folder_name}_text_idname.csv"

    # Set pour stocker tous les noms de modèles uniques
    all_models = set()

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
                    all_models.update(cleaned_models)

            except Exception as e:
                print(f"Erreur lors du traitement de {csv_path}: {e}")

    # Créer un DataFrame unique avec les noms de modèles
    models_df = pd.DataFrame(sorted(all_models), columns=["name"])

    # Sauvegarder le fichier avec le nom basé sur le dossier
    output_csv_path = os.path.join(output_dir, output_csv_name)
    models_df.to_csv(output_csv_path, index=False)
    print(f"Fichier sauvegardé : {output_csv_path}")