import re
import pandas as pd

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


