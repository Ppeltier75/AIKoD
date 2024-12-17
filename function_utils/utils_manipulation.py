import pandas as pd
import os

def count_unique_id_names(csv_path, select_segments=None):
    """
    Compte le nombre d'id_name uniques dans un fichier CSV.
    
    :param csv_path: Chemin vers le fichier CSV.
    :param select_segments: Liste des indices de segments à sélectionner (par exemple, [1, 2, 4, 6]).
                            Si None, compte les id_name uniques complets.
    :return: Nombre d'id_name uniques après transformation (int).
    """
    # Vérifier si le fichier existe
    if not os.path.exists(csv_path):
        print(f"Le fichier CSV spécifié n'existe pas : {csv_path}")
        return None
    
    try:
        # Charger le fichier CSV
        df = pd.read_csv(csv_path)
        print(f"Fichier CSV chargé avec succès : {csv_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV {csv_path} : {e}")
        return None
    
    # Vérifier si la colonne 'id_name' existe
    if 'id_name' not in df.columns:
        print("La colonne 'id_name' est absente du fichier CSV.")
        return None
    
    # Si select_segments est fourni, transformer les id_name
    if select_segments:
        def transform_id_name(id_name, segments):
            """
            Transforme un id_name en sélectionnant uniquement les segments spécifiés.
            
            :param id_name: La chaîne id_name à transformer.
            :param segments: Liste des indices de segments à sélectionner.
            :return: id_name transformé.
            """
            try:
                # Séparer l'id_name en segments
                parts = id_name.split('-')
                # Sélectionner les segments désirés (1-based)
                selected_parts = [parts[i - 1] for i in segments if i - 1 < len(parts)]
                # Rejoindre les segments sélectionnés avec '-'
                return '-'.join(selected_parts)
            except Exception as e:
                print(f"Erreur lors de la transformation de l'id_name '{id_name}' : {e}")
                return id_name  # Retourner l'id_name original en cas d'erreur
        
        # Appliquer la transformation sur la colonne 'id_name'
        df['id_name_transformed'] = df['id_name'].apply(lambda x: transform_id_name(x, select_segments))
        
        # Compter les id_name uniques transformés
        unique_count = df['id_name_transformed'].nunique()
        print(f"Nombre d'id_name uniques après sélection des segments {select_segments} : {unique_count}")
    else:
        # Compter les id_name uniques complets
        unique_count = df['id_name'].nunique()
        print(f"Nombre d'id_name uniques : {unique_count}")
    
    return unique_count
