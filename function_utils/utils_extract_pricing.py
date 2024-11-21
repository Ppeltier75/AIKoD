import os
import json
import pandas as pd

def extract_pricing_text(json_path, pricing_dir):
    """
    Extrait les informations de prix des modèles de type 'text' depuis un fichier JSON
    et génère des fichiers CSV (text_priceinput.csv et text_priceoutput.csv) par fournisseur.

    :param json_path: Chemin vers le fichier JSON contenant les données des modèles.
    :param pricing_dir: Répertoire de sortie où les dossiers et CSV seront générés.
    """
    # Charger le fichier JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(pricing_dir, exist_ok=True)

    # Parcourir les données du JSON
    for provider, date_dict in data.items():
        provider_dir = os.path.join(pricing_dir, provider)
        os.makedirs(provider_dir, exist_ok=True)  # Créer le dossier pour chaque provider
        
        price_input_data = {}
        price_output_data = {}

        for date, model_data in date_dict.items():
            if not isinstance(model_data, dict):
                continue
            
            models = model_data.get("models_extract_GPT4o", {}).get("models", [])
            for model in models:
                # Filtrer uniquement les modèles de type 'text'
                if model.get("type") == "text":
                    name = model.get("name", "unknown").strip()
                    
                    # Extraire et convertir price_input
                    price_input = model.get("price_input", None)
                    if isinstance(price_input, list) and price_input:
                        price_input = float(price_input[0]) if price_input[0].replace('.', '', 1).isdigit() else None
                    elif isinstance(price_input, str) and price_input.replace('.', '', 1).isdigit():
                        price_input = float(price_input)
                    
                    # Extraire et convertir price_output
                    price_output = model.get("price_output", None)
                    if isinstance(price_output, list) and price_output:
                        price_output = float(price_output[0]) if price_output[0].replace('.', '', 1).isdigit() else None
                    elif isinstance(price_output, str) and price_output.replace('.', '', 1).isdigit():
                        price_output = float(price_output)
                    
                    if name not in price_input_data:
                        price_input_data[name] = {}
                    if name not in price_output_data:
                        price_output_data[name] = {}

                    # Ajouter les prix pour la date correspondante
                    price_input_data[name][date] = price_input
                    price_output_data[name][date] = price_output

        # Convertir les données en DataFrame pour chaque fichier
        if price_input_data:
            price_input_df = pd.DataFrame(price_input_data).T
            price_input_df.index.name = "name"
            price_input_df.sort_index(axis=1, inplace=True)  # Trier les colonnes par date
            price_input_csv_path = os.path.join(provider_dir, "text_priceinput.csv")
            price_input_df.to_csv(price_input_csv_path, index=True)
            print(f"Fichier généré : {price_input_csv_path}")

        if price_output_data:
            price_output_df = pd.DataFrame(price_output_data).T
            price_output_df.index.name = "name"
            price_output_df.sort_index(axis=1, inplace=True)  # Trier les colonnes par date
            price_output_csv_path = os.path.join(provider_dir, "text_priceoutput.csv")
            price_output_df.to_csv(price_output_csv_path, index=True)
            print(f"Fichier généré : {price_output_csv_path}")
