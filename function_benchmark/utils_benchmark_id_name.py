import os
import pandas as pd
import re

from function_utils.utils_cleaning import clean_model_name

# Fonction pour mettre à jour les fichiers benchmark idname dans AA
def update_model_names_in_AA(output_dir):
    """
    Crée des fichiers *_idname.csv pour les différents types de benchmarks dans le dossier AA.

    :param output_dir: Chemin du répertoire contenant les fichiers AA.
    """
    if not os.path.exists(output_dir):
        print(f"Le répertoire {output_dir} n'existe pas.")
        return

    # Identifier les sous-dossiers dans AA
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    # Dictionnaire pour stocker les données pour chaque type
    type_to_models = {}

    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        csv_files = [f for f in os.listdir(subdir_path) if f.endswith(".csv")]

        for csv_file in csv_files:
            csv_path = os.path.join(subdir_path, csv_file)
            try:
                # Lire le CSV
                df = pd.read_csv(csv_path)
                # Identifier les colonnes contenant "model" (insensible à la casse)
                model_columns = [col for col in df.columns if "model" in col.lower()]

                for col in model_columns:
                    # Nettoyer les noms de modèles
                    cleaned_models = df[col].dropna().apply(clean_model_name).unique()

                    # Ajouter au type correspondant
                    type_key = f"AA_{subdir.lower()}_idname"
                    if type_key not in type_to_models:
                        type_to_models[type_key] = set()
                    type_to_models[type_key].update(cleaned_models)

            except Exception as e:
                print(f"Erreur lors du traitement de {csv_path}: {e}")

    # Exception: Ajouter les données des sous-dossiers avec des noms de dates dans AA_text_idname.csv
    aa_text_key = "AA_text_idname"
    for subdir in subdirs:
        if subdir.isdigit():  # Vérifie si le nom du dossier est une date
            subdir_path = os.path.join(output_dir, subdir)
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith(".csv")]

            for csv_file in csv_files:
                csv_path = os.path.join(subdir_path, csv_file)
                try:
                    # Lire le CSV
                    df = pd.read_csv(csv_path)
                    # Identifier les colonnes contenant "model" (insensible à la casse)
                    model_columns = [col for col in df.columns if "model" in col.lower()]

                    for col in model_columns:
                        # Nettoyer les noms de modèles
                        cleaned_models = df[col].dropna().apply(clean_model_name).unique()

                        # Ajouter à AA_text_idname
                        if aa_text_key not in type_to_models:
                            type_to_models[aa_text_key] = set()
                        type_to_models[aa_text_key].update(cleaned_models)

                except Exception as e:
                    print(f"Erreur lors du traitement de {csv_path}: {e}")

    # Enregistrer les résultats dans des fichiers CSV
    for type_key, models in type_to_models.items():
        output_csv_path = os.path.join(output_dir, f"{type_key}.csv")
        models_df = pd.DataFrame(sorted(models), columns=["name"])
        models_df.to_csv(output_csv_path, index=False)
        print(f"Fichier sauvegardé : {output_csv_path}")