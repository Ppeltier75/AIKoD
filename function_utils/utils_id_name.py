import os
import json
import pandas as pd
import openai

from function_utils.utils_cleaning import clean_model_name

def extract_names_by_type(json_path, output_dir):
    """
    Extrait les noms des modèles par type et les enregistre dans des fichiers CSV.

    :param json_path: Chemin du fichier JSON contenant les données.
    :param output_dir: Répertoire où enregistrer les fichiers CSV.
    """
    # Charger le fichier JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Créer une structure pour regrouper les noms par type
    type_to_names = {}

    # Parcourir les données
    for provider, date_dict in data.items():
        # Vérifiez si `date_dict` est un dictionnaire
        if not isinstance(date_dict, dict):
            continue  # Passer à la prochaine itération si ce n'est pas le cas

        for date_str, models_extract in date_dict.items():
            # Vérifiez si `models_extract` est un dictionnaire
            if isinstance(models_extract, dict):
                models_data = models_extract.get("models_extract_GPT4o", {})
            elif isinstance(models_extract, list):
                # Si `models_extract` est une liste, on ignore les clés non pertinentes
                models_data = models_extract
            else:
                models_data = []

            # Si `models_data` est une liste, traitez-la directement
            if isinstance(models_data, list):
                models = models_data
            # Si `models_data` est un dictionnaire, accédez à la clé "models"
            elif isinstance(models_data, dict):
                models = models_data.get("models", [])
            else:
                models = []

            # Parcourez les modèles
            for model in models:
                # Assurez-vous que `model` est un dictionnaire avant d'accéder aux clés
                if not isinstance(model, dict):
                    continue

                model_type = model.get("type", "unknown")
                model_name = clean_model_name(model.get("name", ""))
                if model_name:
                    type_to_names.setdefault(model_type, set()).add(model_name)

    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Enregistrer chaque type dans un fichier CSV
    for model_type, names in type_to_names.items():
        # Créer un nom de fichier à partir du type
        file_name = f"AIKoD_{model_type.replace(' ', '').lower()}_idname.csv"
        file_path = os.path.join(output_dir, file_name)

        # Enregistrer dans un CSV
        df = pd.DataFrame(sorted(names), columns=["name"])
        df.to_csv(file_path, index=False)

        print(f"Fichier créé : {file_path}")

def update_model_names_in_csv(json_path, csv_dir):
    """
    Analyse les fichiers CSV et le JSON pour détecter de nouveaux noms de modèles
    et les ajouter aux fichiers CSV correspondants.

    :param json_path: Chemin du fichier JSON contenant les données.
    :param csv_dir: Répertoire contenant les fichiers CSV à mettre à jour.
    """
    # Utiliser extract_names_by_type pour récupérer les modèles par type
    temp_output_dir = os.path.join(csv_dir, "temp")  # Dossier temporaire pour comparer les fichiers
    os.makedirs(temp_output_dir, exist_ok=True)

    # Extraire les noms des modèles dans des fichiers temporaires
    extract_names_by_type(json_path, temp_output_dir)

    # Parcourir les fichiers temporaires pour comparer avec les CSV existants
    for temp_file in os.listdir(temp_output_dir):
        temp_csv_path = os.path.join(temp_output_dir, temp_file)
        final_csv_path = os.path.join(csv_dir, temp_file)

        # Charger les modèles extraits (nouveaux) depuis le fichier temporaire
        temp_df = pd.read_csv(temp_csv_path)
        new_names = set(temp_df["name"].tolist())

        # Charger les modèles existants depuis le fichier CSV final (s'il existe)
        if os.path.exists(final_csv_path):
            final_df = pd.read_csv(final_csv_path)
            existing_names = set(final_df["name"].tolist())
        else:
            existing_names = set()
            print(f"Création d'un nouveau fichier CSV : {final_csv_path}")

        # Trouver les modèles à ajouter
        added_names = new_names - existing_names

        if added_names:
            # Ajouter les nouveaux modèles au fichier final
            updated_names = sorted(existing_names | added_names)
            updated_df = pd.DataFrame(updated_names, columns=["name"])
            updated_df.to_csv(final_csv_path, index=False)

            # Afficher les modèles ajoutés
            for added_name in added_names:
                print(f"Ajouté : {added_name} au fichier : {final_csv_path}")
        else:
            print(f"Aucun nouveau modèle à ajouter pour le fichier : {final_csv_path}")

    # Nettoyer le dossier temporaire
    for temp_file in os.listdir(temp_output_dir):
        os.remove(os.path.join(temp_output_dir, temp_file))
    os.rmdir(temp_output_dir)


def create_prompt_id_name(model_names, model_type, examples_csv_path, column_name='name'):
    """
    Creates a prompt to generate IDs based on the provided model names and type.
    """
    # Load examples from the CSV file
    df_examples = pd.read_csv(examples_csv_path)
    original_names = df_examples[column_name].tolist()
    generated_ids = df_examples['id_name'].tolist()

    # Create few-shot examples
    few_shot_examples = "\n".join([f"{original_name} | {id_name}" for original_name, id_name in zip(original_names, generated_ids)])

    # Instructions and explanations based on model_type
    if model_type == 'text':
        instructions = (
            f"Create model identifiers based on the original model names in the '{column_name}' column I give you. "
            "Use this template: {foundation_model}-{model_variant}-{model_variant2}-{version_number}-{date}-{model_size}-{context}-{specialization}-{finetuned}. "
            "Return ONLY a two-column table with the columns 'name' and 'id_name', using '|' as the separator between columns, without any additional text or markdown formatting. "
            "Please keep the 'name' exactly as provided, without any modifications."
        )

        explanations = (
            "The lists provided below for foundation templates and versions are not exhaustive, but illustrate the typical structure used to name LLM templates. "
            "Models that need to be completed often correspond to new names or variants, and these examples serve as a guide to naming conventions.\n\n"
            "Foundation model refers to the generic name of the basic AI model, excluding details of size, version or context window.\n"
            "Model variant refers to the name of the model that is not part of its foundation model name, e.g., for 'claude haiku 3', the variant is 'haiku'.\n"
            "Model variant 2 usually takes no value except when the model variant is already filled in and needs to be completed again. It can take a value like 'Turbo'.\n"
            "The version is a number preceded by a “v” (e.g., v1 or v0.8) or a number like 3.5 or 6.1 or just 3, etc.\n"
            "Date corresponds to the number that corresponds to a date in the model name.\n"
            "Model size is the number of parameters expressed in billions (e.g., 8B or 70B) or labels such as small, medium, or large; simply write the number.\n"
            "Context is the number of tokens in the context window, usually expressed like 4k or 32k tokens. Simply write the number.\n"
            "Specialization refers to any specific task or domain the model is specialized in.\n"
            "It's essential to write 'unknown' when any of the information is missing.\n"
            "Example for the model name 'llama 3.1 instruct 8b 32k 0621': llama-unknown-unknown-3.1-0621-8-32-instruct.\n"
            "Finetuned is a booleen is the model is finetuned or lora or custom so it need to take true"
        )

    elif model_type == 'image':
        instructions = (
            f"Create image model identifiers based on the original model names in the '{column_name}' column I provided. "
            "Use this template: {foundation_model}-{model_variant}-{model_variant2}-{version_number}-{image_size_pixel}-{number_of_steps}-{hd}-{sampler}-{finetuned}. "
            "Return ONLY a two-column table with the columns 'name' and 'id_name', using '|' as the separator between columns, without any additional text or markdown formatting. "
            "Please keep the 'name' exactly as provided, without any modifications."
        )

        explanations = (
            "The lists provided below for foundation models and versions are not exhaustive but illustrate the typical structure used in naming image models. "
            "Models that need to be completed often correspond to new names or variants, and these examples serve as a guide to naming conventions.\n\n"
            "The foundation model refers to the generic name of the base image model, excluding details such as size, version, or additional context.\n"
            "Model variant refers to the name that is not part of the foundation model, e.g., Large or XL.\n"
            "Model variant 2 usually takes no value except when model variant is already filled and needs further detail. It may take values such as 'Turbo'.\n"
            "Version is a number preceded by 'v' or simply a version number.\n"
            "Image size pixel corresponds to the pixel dimensions of the image. If only a single number is provided, use format 256x256 to indicate both width and height.\n"
            "The number of steps corresponds to the number of steps; write just the number.\n"
            "The HD is a boolean that takes the value True if 'hd' is present in the provided name, and False otherwise.\n"
            "The sampler corresponds to the image sampler, which can take the form of a ddim, for example.\n"
            "Finetuned takes on a Boolean value if the model name contains custom, finetuned or Lora, in which case it should take on a True value.\n"
            "Example for the model name 'japanese stable diffusion xl 30 steps': stablediffusion-japanese-xl-v1-1024x1024-30-False.\n"
        )

    elif model_type == 'audio':
        instructions = (
            f"Create audio model identifiers based on the original model names in the '{column_name}' column I provided. "
            "Use this template: {foundation_model}-{model_variant}-{model_variant2}-{version_number}. "
            "Return ONLY a two-column table with the columns 'name' and 'id_name', using '|' as the separator between columns, without any additional text or markdown formatting. "
            "Please keep the 'name' exactly as provided, without any modifications."
        )

        explanations = (
            "The lists provided below for foundation models and versions are not exhaustive but illustrate the typical structure used in naming audio models. "
            "Models that need to be completed often correspond to new names or variants, and these examples serve as a guide to naming conventions.\n\n"
            "The foundation model refers to the generic name of the base audio model, excluding details such as specific versions or additional context.\n"
            "Model variant refers to the specific variant name, if any, that is not part of the foundation model, e.g., Large or Pro.\n"
            "Model variant 2 usually takes no value except when model variant is already filled and needs further detail. It may take values such as 'Turbo'.\n"
            "Version is a number preceded by 'v' or simply a version number.\n"
            "Example for the model name 'Whisper (large-v3), Deepinfra': whisper-large-deepinfra-3.\n"
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model_names_str = "\n".join(model_names)

    # Build the prompt
    prompt = f"""{instructions}

{explanations}
Here are a few examples of how to construct the IDs:
name | id_name
{few_shot_examples}

Model names to process:
{model_names_str}

Provide the table with generated IDs:
"""
    return prompt



def generate_and_update_id_names(
    csv_path, examples_csv_path, model_type, openai_api_key, output_csv_path=None
):
    """
    Complète les `id_name` manquants dans un CSV à l'aide de l'API OpenAI (GPT-4) et met à jour le fichier CSV.

    :param csv_path: Chemin du fichier CSV contenant les noms des modèles.
    :param examples_csv_path: Chemin du fichier CSV contenant les exemples pour le prompt.
    :param model_type: Type de modèle ('text', 'image', 'audio', etc.).
    :param openai_api_key: Clé API OpenAI pour effectuer les appels.
    :param output_csv_path: Chemin du fichier CSV mis à jour (optionnel, remplace l'original si non fourni).
    :return: Liste des modèles ajoutés.
    """
    # Charger le fichier CSV
    df = pd.read_csv(csv_path)

    # Identifier les lignes avec des `id_name` manquants
    missing_id_names = df[df["id_name"].isnull()]

    # Si aucune ligne à compléter, retourner
    if missing_id_names.empty:
        print("Aucun `id_name` manquant. Aucun changement effectué.")
        return []

    # Préparer les données pour le prompt
    model_names = missing_id_names["name"].tolist()

    # Construire le prompt
    prompt = create_prompt_id_name(
        model_names, model_type, examples_csv_path, column_name="name"
    )

    # Appeler l'API OpenAI
    openai.api_key = openai_api_key

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a highly skilled assistant specialized in generating structured identifiers for AI models. "
                        "You will follow a given template to create accurate and unique IDs based on the provided examples."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        generated_text = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API OpenAI : {e}")
        return []

    # Convertir la sortie API en DataFrame
    try:
        generated_data = [
            line.split(",") for line in generated_text.split("\n") if line
        ]
        df_generated = pd.DataFrame(generated_data, columns=["name", "id_name"])
    except Exception as e:
        print(f"Erreur lors du traitement de la réponse API : {e}")
        return []

    # Fusionner les nouveaux `id_name` avec le CSV original
    updated_df = pd.merge(
        df,
        df_generated,
        on="name",
        how="left",
        suffixes=("", "_new"),
    )
    updated_df["id_name"] = updated_df["id_name"].combine_first(
        updated_df["id_name_new"]
    )
    updated_df = updated_df.drop(columns=["id_name_new"])

    # Sauvegarder le fichier CSV mis à jour
    output_path = output_csv_path if output_csv_path else csv_path
    updated_df.to_csv(output_path, index=False)

    # Afficher les modèles ajoutés
    added_models = df_generated["name"].tolist()
    print(f"Modèles ajoutés : {added_models}")
    return added_models
