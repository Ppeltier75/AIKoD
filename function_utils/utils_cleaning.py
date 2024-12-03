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



def clean_name_AA(aa_directory):
    """
    Nettoie les noms des sous-dossiers dans le répertoire spécifié en les renommant
    selon un mapping prédéfini, puis convertit tous les noms de dossiers en minuscules.

    :param aa_directory: Chemin relatif ou absolu vers le répertoire AA contenant les sous-dossiers à renommer.
    """
    # Définir le chemin de base relatif au script actuel
    base_path = os.path.abspath(os.path.dirname(__file__))
    
    # Définir le chemin relatif vers le fichier de mapping
    mapping_file_path = os.path.join(base_path, '..', 'data', 'models_infos', 'mapping', 'company_mapping.json')
    
    # Vérifier l'existence du fichier de mapping
    if not os.path.exists(mapping_file_path):
        print(f"Fichier de mapping non trouvé : {mapping_file_path}")
        return
    
    # Charger le mapping depuis le fichier JSON
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON dans le fichier de mapping : {e}")
        return
    
    # Vérifier que le mapping est un dictionnaire
    if not isinstance(mapping, dict):
        print("Le fichier de mapping ne contient pas un dictionnaire valide.")
        return
    
    # Normaliser les clés et les valeurs du mapping pour une correspondance insensible à la casse
    normalized_mapping = {}
    for k, v in mapping.items():
        if isinstance(k, str) and isinstance(v, str):
            normalized_mapping[k.lower()] = v.lower()
        else:
            print(f"Entrée ignorée dans le mapping car clé ou valeur n'est pas une chaîne de caractères : clé={k}, valeur={v}")
    
    # Afficher le mapping normalisé pour débogage
    print("Mapping normalisé :", normalized_mapping)
    
    # Définir le chemin complet vers le répertoire AA
    aa_path = os.path.join(base_path, aa_directory) if not os.path.isabs(aa_directory) else aa_directory
    
    # Vérifier l'existence du répertoire AA
    if not os.path.exists(aa_path):
        print(f"Le répertoire AA spécifié n'existe pas : {aa_path}")
        return
    
    # Appliquer le mapping et convertir en minuscules
    for root, dirs, files in os.walk(aa_path, topdown=False):
        for dir_name in dirs:
            dir_name_lower = dir_name.lower()
            if dir_name_lower in normalized_mapping:
                # Obtenir le nouveau nom depuis le mapping et le convertir en minuscules
                new_name = normalized_mapping[dir_name_lower].lower()
                current_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, new_name)
                
                # Vérifier si le nouveau nom est déjà en minuscules pour éviter les conflits
                if current_path != new_path:
                    if not os.path.exists(new_path):
                        try:
                            os.rename(current_path, new_path)
                            print(f"Renommé : '{current_path}' -> '{new_path}'")
                        except Exception as e:
                            print(f"Erreur lors du renommage de '{current_path}' en '{new_path}' : {e}")
                    else:
                        print(f"Le dossier cible existe déjà : '{new_path}'. Impossible de renommer '{current_path}'.")
            else:
                # Si le dossier n'est pas dans le mapping, le renommer en minuscules directement
                new_name = dir_name_lower
                current_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, new_name)
                
                if current_path != new_path:
                    if not os.path.exists(new_path):
                        try:
                            os.rename(current_path, new_path)
                            print(f"Converti en minuscules : '{current_path}' -> '{new_path}'")
                        except Exception as e:
                            print(f"Erreur lors de la conversion de '{current_path}' en minuscules : {e}")
                    else:
                        print(f"Le dossier cible en minuscules existe déjà : '{new_path}'. Impossible de renommer '{current_path}'.")


def aikod_clean_company():
    """
    Parcourt le fichier JSON spécifié et met à jour le champ 'company' des modèles
    de type 'text', 'multimodal', 'text to image' et 'audio to text' en fonction
    des correspondances exactes trouvées dans les fichiers CSV fournis.
    Si aucune correspondance n'est trouvée, le champ 'company' est mis à jour avec une chaîne vide "".
    Utilise des chemins relatifs.
    """
    # Définir les chemins relatifs des fichiers
    base_path = os.path.abspath(os.path.dirname(__file__))
    
    # Chemin relatif vers le fichier JSON
    json_path = os.path.join(base_path, '..', 'data', 'raw', 'AIKoD_brut_API_v2.json')
    
    # Chemins relatifs vers les fichiers CSV
    csv_text_path = os.path.join(base_path, '..', 'data', 'models_infos', 'AIKoD_text_infos.csv')
    csv_texttoimage_path = os.path.join(base_path, '..', 'data', 'models_infos', 'AIKoD_texttoimage_infos.csv')
    csv_audiototext_path = os.path.join(base_path, '..', 'data', 'models_infos', 'AIKoD_audiototext_infos.csv')
    
    # Vérifier l'existence des fichiers
    missing_files = []
    for path, desc in [
        (json_path, "fichier JSON"),
        (csv_text_path, "CSV 'text'"),
        (csv_texttoimage_path, "CSV 'text to image'"),
        (csv_audiototext_path, "CSV 'audio to text'")
    ]:
        if not os.path.exists(path):
            missing_files.append(f"{desc} : {path}")
    
    if missing_files:
        for msg in missing_files:
            print(f"Erreur : {msg}")
        return
    
    # Charger les fichiers CSV dans des DataFrames
    try:
        df_text = pd.read_csv(csv_text_path)
        df_texttoimage = pd.read_csv(csv_texttoimage_path)
        df_audiototext = pd.read_csv(csv_audiototext_path)
    except Exception as e:
        print(f"Erreur lors du chargement des fichiers CSV : {e}")
        return
    
    # Remplacer les NaN dans les colonnes 'company' par ""
    df_text['company'] = df_text['company'].fillna('')
    df_texttoimage['company'] = df_texttoimage['company'].fillna('')
    df_audiototext['company'] = df_audiototext['company'].fillna('')
    
    # Créer des dictionnaires de mappage 'id_name' -> 'company' pour chaque CSV
    dict_text = pd.Series(df_text.company.values, index=df_text.id_name).to_dict()
    dict_texttoimage = pd.Series(df_texttoimage.company.values, index=df_texttoimage.id_name).to_dict()
    dict_audiototext = pd.Series(df_audiototext.company.values, index=df_audiototext.id_name).to_dict()
    
    # Charger le fichier JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON dans le fichier {json_path} : {e}")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON : {e}")
        return
    
    # Initialiser un compteur pour les mises à jour
    updates_count = 0
    
    # Types de modèles à traiter et leur CSV associé
    type_to_dict = {
        'text': dict_text,
        'multimodal': dict_text,
        'text to image': dict_texttoimage,
        'audio to text': dict_audiototext
    }
    
    # Parcourir les modèles dans le JSON
    for top_key, top_value in data.items():
        if not isinstance(top_value, dict):
            print(f"Avertissement : La clé principale '{top_key}' ne contient pas de dictionnaire. Ignoré.")
            continue
        for date_key, date_value in top_value.items():
            if not isinstance(date_value, dict):
                print(f"Avertissement : La clé date '{date_key}' sous '{top_key}' ne contient pas de dictionnaire. Ignoré.")
                continue
            # Identifier les sections contenant des modèles (ex: 'models_extract_GPT4o')
            for section_key, section_value in date_value.items():
                if not isinstance(section_value, dict):
                    # Ignorer les sections qui ne sont pas des dictionnaires (comme 'url')
                    continue
                models = section_value.get('models', [])
                if not isinstance(models, list):
                    print(f"Avertissement : La clé 'models' dans la section '{section_key}' sous '{date_key}' n'est pas une liste. Ignoré.")
                    continue
                for model in models:
                    if not isinstance(model, dict):
                        # Ignorer les éléments qui ne sont pas des dictionnaires
                        continue
                    model_type = model.get('type', '').lower()
                    if model_type not in type_to_dict:
                        continue  # Ne traiter que les types spécifiés
                    id_name_original = model.get('id_name')
                    if not id_name_original:
                        print(f"Modèle sans 'id_name' : {model.get('name', 'Unknown')}")
                        continue
                    # Sélectionner le dictionnaire de mappage approprié
                    mapping_dict = type_to_dict.get(model_type)
                    if not mapping_dict:
                        continue  # Aucun mappage disponible
                    # Rechercher la correspondance exacte dans le dictionnaire de mappage
                    company = mapping_dict.get(id_name_original)
                    if company:
                        # Mettre à jour le champ 'company' dans le modèle
                        original_company = model.get('company', 'Non spécifié')
                        model['company'] = company
                        updates_count += 1
                        print(f"Mise à jour du modèle '{model.get('name', 'Unknown')}' : '{original_company}' -> '{company}'")
                    else:
                        # Mettre à jour le champ 'company' avec une chaîne vide si aucune correspondance
                        original_company = model.get('company', 'Non spécifié')
                        model['company'] = ""
                        updates_count += 1
                        print(f"Mise à jour du modèle '{model.get('name', 'Unknown')}' : '{original_company}' -> '' (aucune correspondance trouvée)")
    
    # Sauvegarder le JSON mis à jour
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\nLe fichier JSON a été mis à jour avec {updates_count} modifications et enregistré à {json_path}.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier JSON mis à jour : {e}")