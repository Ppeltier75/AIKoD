import os
import pandas as pd
import json
import re
from collections import Counter

# Fonction pour analyser un id_name et extraire les informations
def analyze_id_name(id_name):
    """
    Extrait le nombre de paramètres, la taille de la fenêtre de contexte et le statut finetuned
    à partir du id_name.
    """
    id_parts = id_name.split('-')
    number_of_parameters = None
    context_window = None
    finetuned = None

    # Vérifier si les positions 6 et 7 contiennent des informations valides
    if len(id_parts) >= 7:
        try:
            number_of_parameters = float(id_parts[5]) if re.match(r"^\d+(\.\d+)?$", id_parts[5]) else None
            context_window = int(id_parts[6]) * 1000 if re.match(r"^\d+$", id_parts[6]) else None
        except ValueError:
            pass

    # Vérifier si le dernier élément indique finetuned (false/true)
    if len(id_parts) >= 8:
        finetuned = id_parts[7].strip().lower() == "true"

    return number_of_parameters, context_window, finetuned


def AIKoD_text_infos(json_path, output_dir):
    """
    Analyse un fichier JSON pour les modèles avec type 'text' et ajoute des informations
    aux fichiers CSV existants dans le répertoire `output_dir`.

    :param json_path: Chemin du fichier JSON contenant les données des modèles.
    :param output_dir: Répertoire contenant les fichiers _infos.csv à mettre à jour.
    """
    # Charger les données JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Chemin du fichier AIKoD_text_infos.csv existant
    text_infos_csv_path = os.path.join(output_dir, "AIKoD_text_infos.csv")
    if not os.path.exists(text_infos_csv_path):
        print(f"Le fichier {text_infos_csv_path} n'existe pas. Aucun traitement effectué.")
        return

    # Charger le fichier AIKoD_text_infos.csv
    text_infos_df = pd.read_csv(text_infos_csv_path)

    # Parcourir les modèles dans le JSON
    id_name_to_info = {}
    for provider, date_dict in data.items():
        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    # Vérifier que le type est 'text'
                    if model.get("type") == "text" and "id_name" in model and model["id_name"]:
                        id_name = model["id_name"]
                        # Collecter les informations majoritaires
                        number_of_parameters = model.get("number_of_parameters")
                        context_window = model.get("context_window")
                        finetuned = not model.get("id_name", "").endswith("false")
                        id_name_to_info.setdefault(id_name, {"number_of_parameters": [], "context_window": [], "finetuned": finetuned})

                        if number_of_parameters is not None:
                            id_name_to_info[id_name]["number_of_parameters"].append(number_of_parameters)
                        if context_window is not None:
                            id_name_to_info[id_name]["context_window"].append(context_window)

    # Analyser les informations majoritaires et compléter les id_name manquants
    rows_to_update = []
    for id_name, info in id_name_to_info.items():
        if id_name in text_infos_df["id_name"].values:
            # Calculer les valeurs majoritaires
            number_of_parameters = (
                Counter(info["number_of_parameters"]).most_common(1)[0][0]
                if info["number_of_parameters"]
                else None
            )
            context_window = (
                Counter(info["context_window"]).most_common(1)[0][0]
                if info["context_window"]
                else None
            )
            # Analyser id_name pour combler les informations manquantes
            if number_of_parameters is None or context_window is None:
                analyzed_number_of_parameters, analyzed_context_window, analyzed_finetuned = analyze_id_name(id_name)
                number_of_parameters = number_of_parameters or analyzed_number_of_parameters
                context_window = context_window or analyzed_context_window
                info["finetuned"] = analyzed_finetuned  # Remplacer finetuned si nécessaire

            # Mettre à jour les lignes pour cet id_name
            rows_to_update.append(
                {
                    "id_name": id_name,
                    "number_of_parameters": number_of_parameters,
                    "context_window": context_window,
                    "finetuned": info["finetuned"],
                }
            )

    # Si aucune mise à jour, afficher un message et retourner
    if not rows_to_update:
        print("Aucune mise à jour trouvée pour les modèles avec type 'text'.")
        return

    # Créer un DataFrame avec les mises à jour
    updates_df = pd.DataFrame(rows_to_update)

    # Fusionner avec le fichier existant
    text_infos_df = pd.merge(
        text_infos_df,
        updates_df,
        on="id_name",
        how="left",
        suffixes=("", "_new"),
    )

    # Vérifier si les colonnes de mise à jour existent avant de les utiliser
    if "number_of_parameters_new" in text_infos_df.columns:
        text_infos_df["number_of_parameters"] = text_infos_df["number_of_parameters"].combine_first(
            text_infos_df["number_of_parameters_new"]
        )
    if "context_window_new" in text_infos_df.columns:
        text_infos_df["context_window"] = text_infos_df["context_window"].combine_first(
            text_infos_df["context_window_new"]
        )
    if "finetuned_new" in text_infos_df.columns:
        text_infos_df["finetuned"] = text_infos_df["finetuned"].combine_first(
            text_infos_df["finetuned_new"]
        )

    # Supprimer les colonnes temporaires si elles existent
    text_infos_df.drop(
        columns=[col for col in ["number_of_parameters_new", "context_window_new", "finetuned_new"] if col in text_infos_df.columns],
        inplace=True,
    )

    # Sauvegarder le fichier mis à jour
    text_infos_df.to_csv(text_infos_csv_path, index=False)
    print(f"Le fichier {text_infos_csv_path} a été mis à jour avec succès.")
