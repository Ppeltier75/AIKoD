import os
import json
import pandas as pd
from collections import defaultdict
import re


def extract_pricing_text(json_path, output_dir):
    """
    Extrait les prix des modèles de type `text` et `multimodal` et les enregistre en CSV.

    :param json_path: Chemin vers le fichier JSON d'entrée.
    :param output_dir: Répertoire de sortie pour les fichiers CSV.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Charger le JSON
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for provider, date_dict in data.items():
        provider_dir = os.path.join(output_dir, provider)
        if not os.path.exists(provider_dir):
            os.makedirs(provider_dir)

        price_input_data = []
        price_output_data = []

        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    # Filtrer les types pertinents
                    if model.get("type") not in ["text", "multimodal"]:
                        continue

                    model_name = model.get("name", "unknown")
                    company = model.get("company", "unknown")
                    currency = model.get("currency", [])
                    unit_input = model.get("unit_input", [])
                    price_input = model.get("price_input", [])
                    unit_output = model.get("unit_output", [])
                    price_output = model.get("price_output", [])

                    # Harmoniser les prix
                    true_price_input = harmonize_and_convert_prices(unit_input, price_input, currency, company)
                    true_price_output = harmonize_and_convert_prices(unit_output, price_output, currency, company)

                    # Calculer les prix moyens
                    avg_price_input = sum(true_price_input) / len(true_price_input) if true_price_input else None
                    avg_price_output = sum(true_price_output) / len(true_price_output) if true_price_output else None

                    # Ajouter aux données des CSV
                    price_input_data.append({"name": model_name, date_str: avg_price_input})
                    price_output_data.append({"name": model_name, date_str: avg_price_output})

        # Convertir en DataFrame et sauvegarder
        if price_input_data:
            input_df = pd.DataFrame(price_input_data).groupby("name").mean()
            input_csv_path = os.path.join(provider_dir, "text_priceinput.csv")
            input_df.to_csv(input_csv_path)
            print(f"Fichier sauvegardé : {input_csv_path}")

        if price_output_data:
            output_df = pd.DataFrame(price_output_data).groupby("name").mean()
            output_csv_path = os.path.join(provider_dir, "text_priceoutput.csv")
            output_df.to_csv(output_csv_path)
            print(f"Fichier sauvegardé : {output_csv_path}")



def harmonize_and_convert_prices(units, prices, currencies, company):
    """
    Harmonise les prix en fonction des unités textuelles et convertit en USD.

    :param units: Liste des unités associées.
    :param prices: Liste des prix associés.
    :param currencies: Liste des devises associées.
    :param company: Nom de la société (utilisé pour des conversions spécifiques).
    :return: Liste des prix harmonisés en USD.
    """
    conversion_rates = {'EUR': 1.1, 'CHF': 1.17, 'CNY': 0.15, 'CREDITS': 0.01, 'DBU': 0.070, 'USD': 1.0}
    harmonized_prices = []

    for i, (unit, price) in enumerate(zip(units, prices)):
        unit = unit.lower()
        # Extraire les valeurs numériques même si elles sont précédées par un symbole
        price = float(re.search(r'\d+(\.\d+)?', str(price)).group()) if re.search(r'\d+(\.\d+)?', str(price)) else 0.0

        # Détecter la devise, appliquer le taux de conversion
        currency = currencies[i % len(currencies)].upper() if currencies else 'USD'
        rate = conversion_rates.get(currency, 1.0)

        # Harmonisation des unités textuelles uniquement
        if '1k tokens' in unit:
            price_per_million = price * 1000
        elif '1k characters' in unit:
            price_per_million = price * 4000
        elif '10k tokens' in unit:
            price_per_million = price * 100
        elif '10k characters' in unit:
            price_per_million = price * 400
        elif '1m tokens' in unit:
            price_per_million = price
        elif '1m characters' in unit:
            price_per_million = price * 4
        else:
            # Ignorer les unités non pertinentes
            continue

        price_in_usd = price_per_million * rate
        harmonized_prices.append(price_in_usd)

    return harmonized_prices
