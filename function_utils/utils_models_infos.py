import os
import pandas as pd


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
