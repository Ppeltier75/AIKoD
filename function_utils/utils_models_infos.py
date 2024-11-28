import os
import pandas as pd
import numpy as np
import json 


def generate_csv_with_infos(input_dir, output_dir):
    """
    Génère des fichiers CSV basés sur les fichiers existants (finissant par `_idname`) avec une colonne `id_name`
    contenant les identifiants uniques.

    :param input_dir: Répertoire contenant les fichiers CSV d'entrée.
    :param output_dir: Répertoire où les fichiers de sortie seront enregistrés.
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir les fichiers d'entrée
    for file_name in os.listdir(input_dir):
        if file_name.endswith("_idname.csv"):
            # Charger le fichier CSV
            input_path = os.path.join(input_dir, file_name)
            try:
                df = pd.read_csv(input_path)
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier {input_path} : {e}")
                continue

            # Vérifier si la colonne `id_name` existe
            if "id_name" not in df.columns:
                print(f"Aucune colonne `id_name` trouvée dans le fichier {input_path}.")
                continue

            # Filtrer les identifiants uniques
            unique_id_names = df["id_name"].dropna().unique()

            # Créer un nouveau DataFrame pour la sortie
            df_infos = pd.DataFrame({"id_name": unique_id_names})

            # Sauvegarder le fichier avec le suffixe `_infos`
            output_file_name = file_name.replace("_idname.csv", "_infos.csv")
            output_path = os.path.join(output_dir, output_file_name)

            try:
                df_infos.to_csv(output_path, index=False)
                print(f"Fichier généré : {output_path}")
            except Exception as e:
                print(f"Erreur lors de l'écriture du fichier {output_path} : {e}")




def column_name_modelsinfos(directory):
    # Dictionnaire contenant les fichiers et leurs colonnes à renommer
    rename_mapping = {
        "AIKoD_audiototext_infos.csv": {
            "Word Error Rate (%)": "quality_index",
            "Median Speed Factor": "speed_index",
        },
        "AIKoD_text_infos.csv": {
            "AA_arenaelo": "aa_ae_rating",
            "quality_index": "quality_index",  # Pas de changement pour celui-ci
            "aa_mmlu": "aa_mmlu_rating",
            "aa_gpqa": "aa_gpqa_rating",
            "aa_humaneval": "aa_humaneval_rating",
            "aa_math": "aa_math_rating",
            "aa_mgsm": "aa_mgsm_rating",
            "Livebench_rating": "livebench_rating",
            "AE": "hf_ae_rating",
            "MMLU": "hf_mmlu_rating",
            "MT": "hf_mt_rating",
            "Output Tokens/S Median": "speed_index",
            "Latency Median (First Chunk)": "latence_first_chunk",
        },
        "AIKoD_texttoimage_infos.csv": {
            "Steps": "steps",
            "resolution": "resolution",
            "Model Quality ELO": "quality_index",
            "Median Generation Time (s)": "speed_index",
            "Default Steps": "default_steps",
            "Default Resolution": "default_resolution",
        },
    }
    
    # Parcourir les fichiers spécifiés dans le répertoire
    for filename, columns_to_rename in rename_mapping.items():
        file_path = os.path.join(directory, filename)
        
        if os.path.exists(file_path):
            # Chargement du fichier CSV
            df = pd.read_csv(file_path)
            
            # Renommer les colonnes
            df.rename(columns=columns_to_rename, inplace=True)
            
            # Sauvegarde des modifications dans le même fichier
            df.to_csv(file_path, index=False)
            print(f"Colonnes renommées pour le fichier : {filename}")
        else:
            print(f"Fichier non trouvé : {filename}")



def add_country_to_csv(csv_path, column_name):
    """
    Ajoute une colonne 'country' à un fichier CSV en utilisant un mapping de noms de compagnies vers des pays.

    :param csv_path: Chemin vers le fichier CSV.
    :param column_name: Nom de la colonne contenant les noms de compagnies.
    :param mapping_file_path: Chemin vers le fichier JSON contenant le mapping des compagnies vers les pays.
    :return: DataFrame avec la colonne 'country' ajoutée.
    """
    # Lire le fichier CSV dans un DataFrame
    df = pd.read_csv(csv_path)
    
    # Vérifier si la colonne existe
    if column_name not in df.columns:
        print(f"La colonne '{column_name}' n'a pas été trouvée dans le fichier CSV.")
        return df
    
    # Convertir les noms de compagnies en minuscules en gérant les valeurs NaN
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name].replace('nan', np.nan, inplace=True)
    
    mapping_file_path = os.path.join('C:\\Users\\piwip\\OneDrive\\Documents\\OCDE\\AIKoD', 'data', 'models_infos', 'mapping', 'country_mapping.json')
    # Lire le mapping depuis le fichier JSON
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        country_mapping = json.load(f)
    
    # Ajouter la colonne 'country' en mappant les noms de compagnies aux pays
    df['country'] = df[column_name].map(country_mapping)
    
    # Enregistrer le DataFrame modifié dans le fichier CSV (optionnel)
    df.to_csv(csv_path, index=False)
    
    # Retourner le DataFrame modifié
    return df