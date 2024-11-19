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
                        company = model.get("company", None)
                        date_release = model.get("date_release", None)
                        number_of_parameters = model.get("number_of_parameters")
                        context_window = model.get("context_window")
                        finetuned = not model.get("id_name", "").endswith("false")

                        id_name_to_info.setdefault(id_name, {
                            "number_of_parameters": [],
                            "context_window": [],
                            "finetuned": finetuned,
                            "company": [],
                            "date_release": []
                        })

                        if number_of_parameters is not None:
                            id_name_to_info[id_name]["number_of_parameters"].append(number_of_parameters)
                        if context_window is not None:
                            id_name_to_info[id_name]["context_window"].append(context_window)
                        if company:
                            id_name_to_info[id_name]["company"].append(company)
                        if date_release:
                            id_name_to_info[id_name]["date_release"].append(date_release)

    # Analyser les informations majoritaires et compléter les id_name
    rows_to_update = []
    for id_name, info in id_name_to_info.items():
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
        finetuned = info["finetuned"]

        # Analyser id_name pour combler les informations manquantes
        analyzed_number_of_parameters, analyzed_context_window, analyzed_finetuned = analyze_id_name(id_name)
        number_of_parameters = number_of_parameters or analyzed_number_of_parameters
        context_window = context_window or analyzed_context_window
        finetuned = finetuned or analyzed_finetuned

        # Ajouter les informations pour cet id_name
        rows_to_update.append({
            "id_name": id_name,
            "number_of_parameters": number_of_parameters,
            "context_window": context_window,
            "finetuned": finetuned,
            "company": company,
            "date_release": date_release
        })

    # Créer un DataFrame avec les mises à jour
    updates_df = pd.DataFrame(rows_to_update)

    # Fusionner avec le fichier existant (toujours remplacer pour garantir la mise à jour)
    text_infos_df = pd.merge(
        text_infos_df,
        updates_df,
        on="id_name",
        how="left",
        suffixes=("", "_new"),
    )

    # Toujours remplacer les colonnes avec les nouvelles valeurs majoritaires
    for col in ["number_of_parameters", "context_window", "finetuned", "company", "date_release"]:
        if f"{col}_new" in text_infos_df.columns:
            text_infos_df[col] = text_infos_df[f"{col}_new"]

    # Supprimer les colonnes temporaires si elles existent
    text_infos_df.drop(
        columns=[col for col in ["number_of_parameters_new", "context_window_new", "finetuned_new", "company_new", "date_release_new"]
                 if col in text_infos_df.columns],
        inplace=True,
    )

    # Sauvegarder le fichier mis à jour
    text_infos_df.to_csv(text_infos_csv_path, index=False)
    print(f"Le fichier {text_infos_csv_path} a été mis à jour avec succès.")
