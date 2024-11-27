import os
import pandas as pd
import json

def init_API(pricing_directory, output_json_path):
    """
    Parcourt les fichiers CSV de pricing pour chaque fournisseur et génère un fichier JSON avec les informations des modèles.

    :param pricing_directory: Chemin vers le répertoire 'pricing' contenant les sous-dossiers des providers.
    :param output_json_path: Chemin vers le fichier JSON de sortie.
    """
    models = []
    
    # Parcours de chaque fournisseur (sous-dossier) dans le répertoire pricing
    for provider_name in os.listdir(pricing_directory):
        provider_path = os.path.join(pricing_directory, provider_name)
        if os.path.isdir(provider_path):
            # Dictionnaire pour regrouper les fichiers CSV par type
            type_files = {}
            
            # Liste de tous les fichiers dans le répertoire du fournisseur
            for filename in os.listdir(provider_path):
                if filename.endswith('.csv'):
                    # Extraction du type à partir du nom du fichier
                    file_parts = filename.split('_')
                    if len(file_parts) >= 2:
                        type_name = file_parts[0]  # Partie avant le premier '_'
                        # Assurez-vous que le type est dans le dictionnaire
                        if type_name not in type_files:
                            type_files[type_name] = {}
                        # Déterminer le type de prix à partir du nom du fichier
                        price_type = file_parts[1].replace('.csv', '')
                        # Ajouter le chemin du fichier au dictionnaire
                        file_path = os.path.join(provider_path, filename)
                        type_files[type_name][price_type] = file_path
            # Maintenant, pour chaque type, traiter les fichiers CSV
            for type_name, price_files in type_files.items():
                # Dictionnaire pour stocker les DataFrames pour chaque type de prix
                price_dfs = {}
                # Traitement de chaque type de prix
                for price_type, file_path in price_files.items():
                    # Lire le fichier CSV
                    df = pd.read_csv(file_path)
                    # Vérifier si les colonnes 'name' et 'id_name' sont présentes
                    if 'name' not in df.columns or 'id_name' not in df.columns:
                        print(f"Le fichier {file_path} est ignoré car les colonnes 'name' ou 'id_name' sont manquantes.")
                        continue
                    # Ajouter le DataFrame au dictionnaire
                    price_dfs[price_type] = df
                if not price_dfs:
                    continue  # Aucun fichier CSV valide pour ce type
                # Maintenant, traiter les DataFrames
                # Convertir chaque DataFrame au format long
                melted_dfs = []
                for price_type, df in price_dfs.items():
                    # Obtenir les colonnes de dates
                    date_columns = [col for col in df.columns if col not in ['name', 'id_name']]
                    # Convertir le DataFrame au format long
                    df_melted = df.melt(id_vars=['name', 'id_name'], value_vars=date_columns, var_name='date', value_name='price')
                    df_melted['price_type'] = price_type
                    # Ne conserver que les lignes où le prix n'est pas nul
                    df_melted = df_melted[df_melted['price'].notnull()]
                    # Ajouter à la liste
                    melted_dfs.append(df_melted)
                if not melted_dfs:
                    continue  # Aucune donnée à traiter pour ce type
                # Concaténer tous les DataFrames convertis
                df_all = pd.concat(melted_dfs, ignore_index=True)
                # Pivot pour obtenir une ligne par 'name', 'id_name', 'date'
                df_pivot = df_all.pivot_table(index=['name', 'id_name', 'date'], columns='price_type', values='price', aggfunc='first').reset_index()
                # Pour chaque ligne, créer une entrée de modèle
                for idx, row in df_pivot.iterrows():
                    # Vérifier si au moins un prix est présent
                    price_call = row.get('pricecall', None)
                    price_input = row.get('priceinput', None)
                    price_output = row.get('priceoutput', None)
                    if pd.isnull(price_call) and pd.isnull(price_input) and pd.isnull(price_output):
                        continue  # Ignorer les entrées sans aucun prix
                    model_entry = {
                        'provider': provider_name,
                        'name': row['name'],
                        'id_name': row['id_name'],
                        'type': type_name,
                        'date': row['date'],
                        'price_call': float(price_call) if not pd.isnull(price_call) else None,
                        'price_input': float(price_input) if not pd.isnull(price_input) else None,
                        'price_output': float(price_output) if not pd.isnull(price_output) else None,
                    }
                    models.append(model_entry)
    # Écrire la liste des modèles dans le fichier JSON de sortie
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(models, f, ensure_ascii=False, indent=4)
    print(f"Le fichier JSON a été créé à {output_json_path} avec {len(models)} entrées de modèles.")
