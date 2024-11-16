import json


def add_model_type(json_path, output_path=None):
    """
    Ajoute un type à chaque modèle en fonction de ses modalités et unités, avec des catégories étendues.

    :param json_path: Chemin du fichier JSON d'entrée
    :param output_path: Chemin du fichier JSON de sortie avec les types de modèles ajoutés
    """
    # Charger les données JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Parcourir les données pour chaque provider et chaque modèle
    for provider, date_dict in data.items():
        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])

                for model in models:
                    modality_input = model.get("modality_input", [])
                    modality_output = model.get("modality_output", [])
                    unit_input = model.get("unit_input", [])
                    unit_output = model.get("unit_output", [])

                    # Convertir les modalités et unités en minuscule pour éviter les erreurs de casse
                    modality_input = [mod.lower() for mod in modality_input]
                    modality_output = [mod.lower() for mod in modality_output]
                    unit_input = [unit.lower() for unit in unit_input]
                    unit_output = [unit.lower() for unit in unit_output]

                    # Déterminer le type du modèle
                    if any('api request' in unit for unit in unit_input + unit_output):
                        model_type = "task"
                    elif "embedding" in modality_output:
                        model_type = "embeddings"
                    elif ("text" in modality_input or "code" in modality_input) and \
                         ("text" in modality_output or "code" in modality_output) and \
                         ("audio" in modality_input or "audio" in modality_output or
                          "image" in modality_input or "image" in modality_output):
                        model_type = "multimodal"
                    elif ("text" in modality_input and "text" in modality_output):
                        model_type = "text"
                    elif ("text" in modality_input and "image" in modality_output):
                        model_type = "text to image"
                    elif ("image" in modality_input and "image" in modality_output):
                        model_type = "image to image"
                    elif ("image" in modality_input and "text" in modality_output):
                        model_type = "image to text"
                    elif ("audio" in modality_input and "audio" in modality_output):
                        model_type = "audio to audio"
                    elif ("audio" in modality_input and "text" in modality_output):
                        model_type = "audio to text"
                    elif ("text" in modality_input and "audio" in modality_output):
                        model_type = "text to audio"
                    else:
                        model_type = "unknown"

                    # Ajouter le type au modèle
                    model["type"] = model_type

    # Sauvegarder les données avec les types ajoutés dans un nouveau fichier JSON
    final_output_path = output_path if output_path else json_path
    with open(final_output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    print(f"Les types de modèles ont été ajoutés et enregistrés dans le fichier '{final_output_path}'.")


