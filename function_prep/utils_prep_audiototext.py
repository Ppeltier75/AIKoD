import pandas as pd
import os
import json
from collections import Counter
# Importation des fonctions depuis merge_utils.py
from function_utils.utils_merge_id import select_specific_segments, select_segments_no_order, merge_csv_id_name


def add_csv_audio_to_text(base_csv_path):
    """
    Adds audio-to-text related columns to the base CSV file by merging data from the specified CSV file.
    The function reads the base CSV, merges additional columns, and saves the updated DataFrame.

    :param base_csv_path: Path to the base CSV file.
    """
    # Définition des stratégies de correspondance sous forme de tuples (fonction, nom)
    strategies = [
        (lambda x: x, 'exact_match'),  # Correspondance exacte
        (lambda x: select_specific_segments(x, [1, 2, 4]), 'strategy_1'),
        (lambda x: select_segments_no_order(x, [1, 2, 4]), 'strategy_2'),
        (lambda x: select_specific_segments(x, [1, 2, 3]), 'strategy_3'),
        (lambda x: select_segments_no_order(x, [1, 2, 3]), 'strategy_4'),
        (lambda x: select_specific_segments(x, [1, 4]), 'strategy_5'),
        (lambda x: select_segments_no_order(x, [1, 4]), 'strategy_6'),
        # Vous pouvez ajouter d'autres stratégies si nécessaire
    ]

    # Lecture du fichier de base
    try:
        df_base = pd.read_csv(base_csv_path)
        print(f"Fichier de base chargé : {base_csv_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier de base {base_csv_path} : {e}")
        return

    # Création d'une copie du DataFrame de base pour les fusions successives
    df_merged = df_base.copy()

    # Chemin vers le fichier à fusionner
    audio_to_text_csv = r'C:\Users\piwip\OneDrive\Documents\OCDE\AIKoD\data\benchmark\AA\audiototext\AA_audiototext_2024-11-19.csv'

    # Vérifier si le fichier à fusionner existe
    if not os.path.exists(audio_to_text_csv):
        print(f"Le fichier à fusionner {audio_to_text_csv} n'existe pas. Opération ignorée.")
        return

    # Lecture du fichier à fusionner
    try:
        df_merge = pd.read_csv(audio_to_text_csv)
        print(f"Fichier à fusionner chargé : {audio_to_text_csv}")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier à fusionner {audio_to_text_csv} : {e}")
        return

    # Colonnes à conserver après le merge
    keep_columns = ['speed_index', 'quality_index']

    # Nettoyer la colonne 'Word Error Rate (%)' dans df_merge
    if 'Word Error Rate (%)' in df_merge.columns:
        print("Nettoyage de la colonne 'Word Error Rate (%)'")
        # Vérifier si la colonne contient des chaînes de caractères avec '%'
        if df_merge['Word Error Rate (%)'].dtype == object:
            # Vérifier s'il y a des '%' dans les valeurs
            contains_percent = df_merge['Word Error Rate (%)'].str.contains('%', na=False).any()
            if contains_percent:
                # Remplacer '%' et convertir en float
                df_merge['Word Error Rate (%)'] = df_merge['Word Error Rate (%)'].str.replace('%', '', regex=False)
                print("Symboles '%' supprimés de 'Word Error Rate (%)'")
        # Convertir en numérique et diviser par 100
        df_merge['Word Error Rate (%)'] = pd.to_numeric(df_merge['Word Error Rate (%)'], errors='coerce') / 100
        print("Conversion de 'Word Error Rate (%)' en numérique et division par 100")
    else:
        print("La colonne 'Word Error Rate (%)' est absente dans le fichier à fusionner.")

    # Nettoyer la colonne 'Median Speed Factor' si nécessaire
    if 'Median Speed Factor' in df_merge.columns:
        print("Nettoyage de la colonne 'Median Speed Factor'")
        # Vérifier si la colonne contient des chaînes de caractères
        if df_merge['Median Speed Factor'].dtype == object:
            # Remplacer les éventuels caractères non numériques (par exemple, espaces)
            df_merge['Median Speed Factor'] = df_merge['Median Speed Factor'].str.replace(',', '.', regex=False)  # Remplacer la virgule par un point si nécessaire
        # Convertir en numérique
        df_merge['Median Speed Factor'] = pd.to_numeric(df_merge['Median Speed Factor'], errors='coerce')
        print("Conversion de 'Median Speed Factor' en numérique")
    else:
        print("La colonne 'Median Speed Factor' est absente dans le fichier à fusionner.")

    # Fusion
    try:
        df_merged = merge_csv_id_name(df_merged, df_merge, keep_columns, strategies)
        print("Fusion effectuée avec succès.")
    except Exception as e:
        print(f"Erreur lors de la fusion des DataFrames : {e}")
        return

    # Enregistrement du DataFrame fusionné
    try:
        output_csv_path = os.path.splitext(base_csv_path)[0] + '.csv'
        df_merged.to_csv(output_csv_path, index=False)
        print(f"Le fichier fusionné a été enregistré sous {output_csv_path}")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier fusionné : {e}")
        return

    return df_merged

def AIKoD_audiototext_infos(json_path, base_csv_path):
    """
    Met à jour un fichier CSV avec des informations extraites d'un JSON et fusionne avec d'autres fichiers CSV
    en utilisant add_csv_audio_to_text. Modifie la colonne quality_index selon les spécifications.

    :param json_path: Chemin vers le fichier JSON contenant les données des modèles.
    :param base_csv_path: Chemin vers le fichier CSV à mettre à jour.
    """
    # Charger les données JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"Données JSON chargées depuis {json_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON {json_path} : {e}")
        return

    # Charger le fichier CSV existant
    try:
        base_df = pd.read_csv(base_csv_path)
        print(f"Fichier CSV de base chargé : {base_csv_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV de base {base_csv_path} : {e}")
        return

    # Parcourir les modèles dans le JSON
    id_name_to_info = {}
    for provider, date_dict in data.items():
        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    # Vérifier que le type est 'audio_to_text'
                    if model.get("type") == "audio to text" and "id_name" in model and model["id_name"]:
                        id_name = model["id_name"]
                        company = model.get("company", None)
                        date_release = model.get("date_release", None)

                        id_name_to_info.setdefault(id_name, {
                            "company": [],
                            "date_release": []
                        })

                        if company:
                            id_name_to_info[id_name]["company"].append(company)
                        if date_release:
                            id_name_to_info[id_name]["date_release"].append(date_release)

    # Analyser les informations majoritaires et compléter les id_name
    rows_to_update = []
    for id_name, info in id_name_to_info.items():
        # Calculer les valeurs majoritaires
        company = (
            Counter(info["company"]).most_common(1)[0][0]
            if info["company"]
            else None
        )
        date_release = (
            Counter([str(d) for d in info["date_release"] if isinstance(d, (str, int, float))]).most_common(1)[0][0]
            if info.get("date_release") and info["date_release"]
            else None
        )

        # Ajouter les informations pour cet id_name
        rows_to_update.append({
            "id_name": id_name,
            "company": company,
            "date_release": date_release
        })

    # Créer un DataFrame avec les mises à jour
    try:
        updates_df = pd.DataFrame(rows_to_update)
        print("DataFrame des mises à jour créé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la création du DataFrame des mises à jour : {e}")
        return

    # Fusionner les mises à jour avec le DataFrame existant
    try:
        base_df = pd.merge(
            base_df,
            updates_df,
            on="id_name",
            how="left",
            suffixes=("", "_new"),
        )
        print("Mises à jour fusionnées avec succès.")
    except Exception as e:
        print(f"Erreur lors de la fusion des mises à jour avec le DataFrame existant : {e}")
        return

    # Mettre à jour les colonnes avec les nouvelles valeurs
    for col in ["company", "date_release"]:
        if f"{col}_new" in base_df.columns:
            base_df[col] = base_df[f"{col}_new"].combine_first(base_df[col])
            base_df.drop(columns=[f"{col}_new"], inplace=True)
            print(f"Colonne '{col}' mise à jour.")
        else:
            print(f"Colonne '{col}_new' absente. Aucun changement apporté.")

    # Sauvegarder le DataFrame mis à jour temporairement
    temp_csv_path = base_csv_path.replace('.csv', '_temp.csv')
    try:
        base_df.to_csv(temp_csv_path, index=False)
        print(f"Fichier temporaire sauvegardé sous : {temp_csv_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier temporaire {temp_csv_path} : {e}")
        return

    # Utiliser add_csv_audio_to_text pour effectuer les fusions supplémentaires
    try:
        df_final = add_csv_audio_to_text(temp_csv_path)
    except Exception as e:
        print(f"Erreur lors de l'exécution de add_csv_audio_to_text : {e}")
        return

    # Transformer la colonne quality_index
    if "quality_index" in df_final.columns:
        try:
            df_final["quality_index"] = 1 - (df_final["quality_index"] / 100)
            print("La colonne 'quality_index' a été transformée.")
        except Exception as e:
            print(f"Erreur lors de la transformation de 'quality_index' : {e}")

    # Enregistrer le DataFrame final à l'emplacement d'origine
    try:
        df_final.to_csv(base_csv_path, index=False)
        print(f"Le fichier {base_csv_path} a été mis à jour avec succès en utilisant add_csv_audio_to_text.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier final {base_csv_path} : {e}")
        return

    # Supprimer le fichier temporaire
    if os.path.exists(temp_csv_path):
        try:
            os.remove(temp_csv_path)
            print(f"Fichier temporaire {temp_csv_path} supprimé.")
        except Exception as e:
            print(f"Erreur lors de la suppression du fichier temporaire {temp_csv_path} : {e}")

    return df_final