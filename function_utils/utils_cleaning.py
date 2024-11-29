import re
import pandas as pd
import numpy as np
import json
import os 

# Fonction pour nettoyer les noms des modèles
def clean_model_name(name):
    """
    Normalizes the model name by replacing commas and hyphens with spaces,
    converting to lowercase except for numbers, and removing extra spaces.
    """
    if isinstance(name, str):  # Check if the name is a string
        name = name.lower()  # Convert to lowercase
        name = re.sub(r'[,\-]', ' ', name)  # Replace commas and hyphens with spaces
        name = re.sub(r'\s+', ' ', name).strip()  # Remove extra spaces
    return name  # Return the normalized name




def remove_id_names_with_wrong_segments(csv_path, expected_segments=9):
    """
    Lit le fichier CSV à csv_path, supprime les lignes où 'id_name' n'a pas exactement
    le nombre de segments spécifié lorsqu'il est séparé par '-', et sauvegarde le DataFrame
    nettoyé dans le même fichier CSV.

    :param csv_path: Chemin vers le fichier CSV.
    :param expected_segments: Nombre de segments attendus dans 'id_name'.
    """
    # Lire le fichier CSV
    df = pd.read_csv(csv_path)

    # Vérifier si la colonne 'id_name' existe
    if 'id_name' not in df.columns:
        print("La colonne 'id_name' n'existe pas dans le fichier CSV.")
        return

    # Fonction pour vérifier si 'id_name' a le nombre de segments attendu
    def has_expected_segments(id_name):
        if pd.isnull(id_name):
            return False
        segments = str(id_name).split('-')
        return len(segments) == expected_segments

    # Identifier les lignes où 'id_name' n'a pas le nombre de segments attendu
    invalid_id_name_mask = ~df['id_name'].apply(has_expected_segments)

    # Vider le champ 'id_name' dans ces lignes
    df.loc[invalid_id_name_mask, 'id_name'] = ''

    # Sauvegarder le DataFrame modifié dans le même fichier CSV
    df.to_csv(csv_path, index=False)

    num_cleared = invalid_id_name_mask.sum()
    print(f"Le champ 'id_name' a été vidé dans {num_cleared} lignes où il n'avait pas {expected_segments} segments.")




def harmonize_company_name(csv_path, column_name):
    """
    Harmonise les noms de compagnies dans une colonne spécifique d'un fichier CSV en utilisant un mapping externe.

    :param csv_path: Chemin vers le fichier CSV.
    :param column_name: Nom de la colonne contenant les noms de compagnies à harmoniser.
    :param mapping_file_path: Chemin vers le fichier JSON contenant le mapping.
    :return: DataFrame avec les noms de compagnies harmonisés.
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
    
    base_path = os.path.abspath(os.path.dirname(__file__))
    mapping_file_path = os.path.join(base_path, '..', 'data', 'models_infos', 'mapping', 'company_mapping.json')

    # Lire le mapping depuis le fichier JSON
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        company_mapping = json.load(f)
    
    # Remplacer les noms de compagnies en utilisant le mapping
    df[column_name] = df[column_name].map(company_mapping).fillna(df[column_name])
    
    # Enregistrer le DataFrame modifié dans le fichier CSV (optionnel)
    df.to_csv(csv_path, index=False)
    
    # Retourner le DataFrame modifié
    return df
