import os
import json
import pandas as pd
from collections import defaultdict
import re
import numpy as np
from datetime import datetime
from collections import defaultdict

def extract_pricing_text(json_path, output_dir):
    """
    Extrait les prix des modèles de type `text` et `multimodal`, ainsi que les prix d'appel `price_call`.
    Ajoute également la colonne `id_name` aux CSV générés.

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
        price_call_data = []

        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    # Filtrer les types pertinents
                    if model.get("type") not in ["text", "multimodal"]:
                        continue

                    model_name = model.get("name", "unknown").strip()
                    id_name = model.get("id_name", "").strip() if model.get("id_name") else None
                    company = model.get("company", "unknown")
                    currency = model.get("currency", [])
                    unit_input = model.get("unit_input", [])
                    price_input = model.get("price_input", [])
                    unit_output = model.get("unit_output", [])
                    price_output = model.get("price_output", [])
                    price_call = model.get("price_call", [])

                    # Harmoniser les prix pour price_input et price_output
                    true_price_input = harmonize_and_convert_prices_text(unit_input, price_input, currency, company)
                    true_price_output = harmonize_and_convert_prices_text(unit_output, price_output, currency, company)

                    # Calculer les prix moyens pour price_input et price_output
                    avg_price_input = sum(true_price_input) / len(true_price_input) if true_price_input else None
                    avg_price_output = sum(true_price_output) / len(true_price_output) if true_price_output else None

                    # Ajouter les données aux listes pour CSV
                    row_input = {"name": model_name, date_str: avg_price_input}
                    row_output = {"name": model_name, date_str: avg_price_output}
                    row_call = {"name": model_name, date_str: None}

                    if id_name:
                        row_input["id_name"] = id_name
                        row_output["id_name"] = id_name
                        row_call["id_name"] = id_name

                    # Ajouter les données brutes de price_call
                    price_call_values = []
                    for pc in price_call:
                        try:
                            pc_str = str(pc).strip()
                            if pc_str == '':
                                continue
                            # Supprimer tout symbole de devise ou caractère non numérique
                            pc_clean = re.sub(r'[^\d\.]', '', pc_str)
                            if pc_clean == '':
                                continue
                            price_call_value = float(pc_clean)
                            price_call_values.append(price_call_value)
                        except (ValueError, TypeError):
                            continue

                    # Pour price_call, au lieu de faire la moyenne, prendre la première valeur non nulle
                    if price_call_values:
                        # Vous pouvez choisir 'first', 'max' ou une autre agrégation selon vos besoins
                        avg_price_call = price_call_values[0]  # Prendre la première valeur non nulle
                        row_call[date_str] = avg_price_call
                    else:
                        row_call[date_str] = None

                    price_input_data.append(row_input)
                    price_output_data.append(row_output)
                    price_call_data.append(row_call)

        # Convertir en DataFrame et nettoyer les colonnes
        price_input_df = pd.DataFrame(price_input_data)
        price_output_df = pd.DataFrame(price_output_data)
        price_call_df = pd.DataFrame(price_call_data)

        # Sauvegarder les fichiers CSV
        if not price_input_df.empty:
            # Group by 'name' and 'id_name', puis faire la moyenne pour price_input
            input_df = price_input_df.groupby(["name", "id_name"], dropna=False).mean(numeric_only=True).reset_index()
            input_csv_path = os.path.join(provider_dir, "text_priceinput.csv")
            input_df.to_csv(input_csv_path, index=False)
            print(f"Fichier sauvegardé : {input_csv_path}")

        if not price_output_df.empty:
            # Group by 'name' and 'id_name', puis faire la moyenne pour price_output
            output_df = price_output_df.groupby(["name", "id_name"], dropna=False).mean(numeric_only=True).reset_index()
            output_csv_path = os.path.join(provider_dir, "text_priceoutput.csv")
            output_df.to_csv(output_csv_path, index=False)
            print(f"Fichier sauvegardé : {output_csv_path}")

        if not price_call_df.empty:
            # Group by 'name' and 'id_name', puis prendre la première valeur pour chaque date
            call_df = price_call_df.groupby(["name", "id_name"], dropna=False).first().reset_index()
            # Si nécessaire, effectuer un forward-fill sur les colonnes de dates
            date_columns = [col for col in call_df.columns if is_date_column(col)]
            if date_columns:
                call_df[date_columns] = call_df[date_columns].ffill(axis=1)
            call_csv_path = os.path.join(provider_dir, "text_pricecall.csv")
            call_df.to_csv(call_csv_path, index=False)
            print(f"Fichier sauvegardé : {call_csv_path}")

def is_date_column(column_name):
    """
    Vérifie si le nom de la colonne correspond au format de date 'YYYY-MM-DD'.

    :param column_name: Nom de la colonne.
    :return: Booléen indiquant si la colonne est une colonne de date.
    """
    try:
        datetime.strptime(column_name, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def harmonize_and_convert_prices_text(units, prices, currencies, company):
    """
    Harmonise les prix en fonction des unités textuelles et convertit en USD.

    :param units: Liste des unités associées.
    :param prices: Liste des prix associés.
    :param currencies: Liste des devises associées.
    :param company: Nom de la société (utilisé pour des conversions spécifiques).
    :return: Liste des prix harmonisés en USD.
    """
    conversion_rates = {'EUR': 1.1, 'CHF': 1.17, 'CNY': 0.15, 'CREDITS': 0.01,
                        'DBU': 0.070, 'USD': 1.0}
    harmonized_prices = []

    for i, (unit, price) in enumerate(zip(units, prices)):
        unit = unit.lower()
        price_str = str(price).strip()
        price_match = re.search(r'\d+(\.\d+)?', price_str)
        if price_match:
            price = float(price_match.group())
        else:
            # Ignorer les prix invalides ou manquants
            continue

        # Détecter la devise et appliquer le taux de conversion
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

def harmonize_and_convert_price_audiototext(units, prices, currencies, company):
    """
    Harmonise les prix en fonction des unités audio et les convertit en 'minute audio',
    en tenant compte des devises.

    :param units: Liste des unités associées.
    :param prices: Liste des prix associés.
    :param currencies: Liste des devises associées.
    :param company: Nom de la société (utilisé pour des conversions spécifiques).
    :return: Liste des prix harmonisés en 'minute audio' (convertis en USD si applicable).
    """
    conversion_rates = {'EUR': 1.1, 'CHF': 1.17, 'CNY': 0.15, 'CREDITS': 0.01,
                        'DBU': 0.070, 'USD': 1.0}
    harmonized_prices = []

    for i, (unit, price) in enumerate(zip(units, prices)):
        unit = unit.lower()
        price_str = str(price).strip()
        price_match = re.search(r'\d+(\.\d+)?', price_str)
        if price_match:
            price = float(price_match.group())
        else:
            # Ignorer les prix invalides ou manquants
            continue

        # Détecter la devise et appliquer le taux de conversion
        currency = currencies[i % len(currencies)].upper() if currencies else 'USD'
        rate = conversion_rates.get(currency, 1.0)

        if "second audio" in unit or "audio second" in unit:
            price_per_minute = price * 60
        elif "minute audio" in unit:
            price_per_minute = price
        elif "hour audio" in unit or "hour" in unit:
            price_per_minute = price / 60
        else:
            # Ignorer les unités non pertinentes
            continue

        # Convertir en USD
        price_in_usd = price_per_minute * rate
        harmonized_prices.append(price_in_usd)

    return harmonized_prices


def extract_pricing_audiototext(json_path, output_dir):
    """
    Extrait les prix des modèles de type `audio to text`, ajoute la colonne `id_name`,
    et les enregistre en CSV.

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

        # Initialiser une liste pour stocker les données
        price_input_data = []

        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    # Filtrer uniquement les modèles de type `audio to text`
                    if model.get("type") != "audio to text":
                        continue

                    model_name = model.get("name", "unknown")
                    id_name = model.get("id_name", None)
                    company = model.get("company", "unknown")
                    unit_input = model.get("unit_input", [])
                    price_input = model.get("price_input", [])
                    currencies = model.get("currency", [])

                    # Harmoniser et convertir les prix en 'minute audio'
                    harmonized_prices = harmonize_and_convert_price_audiototext(unit_input, price_input, currencies, company)

                    # Prendre le prix maximum harmonisé
                    max_price_input = max(harmonized_prices) if harmonized_prices else None

                    # Ajouter les données au CSV
                    existing_entry = next((entry for entry in price_input_data if entry["name"] == model_name and entry["id_name"] == id_name), None)
                    if existing_entry:
                        existing_entry[date_str] = max_price_input
                    else:
                        new_entry = {"name": model_name, "id_name": id_name, date_str: max_price_input}
                        price_input_data.append(new_entry)

        # Convertir en DataFrame
        price_input_df = pd.DataFrame(price_input_data)

        if not price_input_df.empty:
            # Réorganiser les colonnes pour s'assurer que `name` et `id_name` sont en premier
            columns_order = ["name", "id_name"] + [col for col in price_input_df.columns if col not in ["name", "id_name"]]
            price_input_df = price_input_df[columns_order]

            # Sauvegarder le fichier CSV
            input_csv_path = os.path.join(provider_dir, "audiototext_priceinput.csv")
            price_input_df.to_csv(input_csv_path, index=False)
            print(f"Fichier sauvegardé : {input_csv_path}")


def extract_pricing_texttoimage(json_path, output_dir):
    """
    Extrait les prix des modèles de type `text to image`, ajoute la colonne `id_name`,
    et les enregistre en CSV.

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

        # Initialiser une liste pour stocker les données
        price_output_data = []

        for date_str, models_extract in date_dict.items():
            if isinstance(models_extract, dict):
                models = models_extract.get("models_extract_GPT4o", {}).get("models", [])
                for model in models:
                    # Filtrer uniquement les modèles de type `text to image`
                    if model.get("type") != "text to image":
                        continue

                    model_name = model.get("name", "unknown")
                    id_name = model.get("id_name", None)
                    company = model.get("company", "unknown")
                    unit_output = model.get("unit_output", [])
                    price_output = model.get("price_output", [])
                    currencies = model.get("currency", [])

                    # Harmoniser et convertir les prix
                    harmonized_prices = harmonize_and_convert_price_texttoimage(unit_output, price_output, currencies, company)

                    # Prendre le prix maximum harmonisé
                    max_price_output = max(harmonized_prices) if harmonized_prices else None

                    # Ajouter les données au CSV
                    existing_entry = next((entry for entry in price_output_data if entry["name"] == model_name and entry["id_name"] == id_name), None)
                    if existing_entry:
                        existing_entry[date_str] = max_price_output
                    else:
                        new_entry = {"name": model_name, "id_name": id_name, date_str: max_price_output}
                        price_output_data.append(new_entry)

        # Convertir en DataFrame
        price_output_df = pd.DataFrame(price_output_data)

        if not price_output_df.empty:
            # Réorganiser les colonnes pour s'assurer que `name` et `id_name` sont en premier
            columns_order = ["name", "id_name"] + [col for col in price_output_df.columns if col not in ["name", "id_name"]]
            price_output_df = price_output_df[columns_order]

            # Sauvegarder le fichier CSV
            output_csv_path = os.path.join(provider_dir, "texttoimage_priceoutput.csv")
            price_output_df.to_csv(output_csv_path, index=False)
            print(f"Fichier sauvegardé : {output_csv_path}")

def harmonize_and_convert_price_texttoimage(units, prices, currencies, company):
    """
    Harmonise les prix en fonction des unités contenant 'image' et convertit en USD.

    :param units: Liste des unités associées.
    :param prices: Liste des prix associés.
    :param currencies: Liste des devises associées.
    :param company: Nom de la société (utilisé pour des conversions spécifiques).
    :return: Liste des prix harmonisés en USD.
    """
    conversion_rates = {'EUR': 1.1, 'CHF': 1.17, 'CNY': 0.15, 'CREDITS': 0.01,
                        'DBU': 0.070, 'USD': 1.0}
    harmonized_prices = []

    for i, (unit, price) in enumerate(zip(units, prices)):
        unit = unit.lower()
        price_str = str(price).strip()
        price_match = re.search(r'\d+(\.\d+)?', price_str)
        if price_match:
            price = float(price_match.group())
        else:
            # Ignorer les prix invalides ou manquants
            continue

        # Détecter la devise et appliquer le taux de conversion
        currency = currencies[i % len(currencies)].upper() if currencies else 'USD'
        rate = conversion_rates.get(currency, 1.0)

        if "image" in unit:
            # Diviser le prix par le nombre dans l'unité, si applicable
            match = re.search(r'(\d+)\s*image', unit)
            if match:
                divisor = int(match.group(1))
                price_per_image = (price / divisor) * rate
            else:
                price_per_image = price * rate
            harmonized_prices.append(price_per_image)

    return harmonized_prices


import os
import pandas as pd
from datetime import datetime

def correct_pricing_aikod(directory):
    """
    Corrige les fichiers de tarification dans le répertoire spécifié en :
    1. Organisant les colonnes de dates du plus ancien au plus récent.
    2. Fusionnant les lignes ayant le même 'id_name' sans conflits dans les colonnes de dates.
    3. Comblant les trous dans les colonnes de dates avec la dernière valeur connue (forward-fill).
    4. Imprimant les lignes conflictuelles où la fusion n'est pas possible.

    :param directory: Chemin absolu ou relatif vers le répertoire 'pricing' contenant les sous-dossiers par provider et les fichiers CSV.
    """

    def is_date_column(column_name):
        """
        Vérifie si le nom de la colonne correspond au format de date 'YYYY-MM-DD'.

        :param column_name: Nom de la colonne.
        :return: Booléen indiquant si la colonne est une colonne de date.
        """
        try:
            datetime.strptime(column_name, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    # Initialiser les compteurs
    total_csvs = 0
    total_models_merged = 0
    total_conflicts = 0

    # Parcourir tous les fichiers dans le répertoire spécifié et ses sous-répertoires
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                total_csvs += 1
                csv_path = os.path.join(root, file)
                print(f"\nTraitement du fichier CSV : {csv_path}")

                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"  Erreur lors de la lecture du fichier CSV : {e}")
                    continue

                # Identifier les colonnes de dates
                date_columns = [col for col in df.columns if is_date_column(col)]

                if not date_columns:
                    print("  Aucun colonne de date trouvée. Passage au fichier suivant.")
                    continue

                # Trier les colonnes de dates du plus ancien au plus récent
                try:
                    date_columns_sorted = sorted(date_columns, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
                except Exception as e:
                    print(f"  Erreur lors du tri des colonnes de dates : {e}")
                    continue

                # Réorganiser les colonnes : 'name', 'id_name', colonnes de dates triées, puis les autres colonnes
                non_date_columns = [col for col in df.columns if col not in date_columns_sorted]
                ordered_columns = []

                if 'name' in non_date_columns:
                    ordered_columns.append('name')
                    non_date_columns.remove('name')
                if 'id_name' in non_date_columns:
                    ordered_columns.append('id_name')
                    non_date_columns.remove('id_name')

                ordered_columns += date_columns_sorted
                ordered_columns += non_date_columns  # Ajouter les autres colonnes telles quelles

                df = df[ordered_columns]

                # Grouper par 'id_name'
                grouped = df.groupby('id_name')

                merged_rows = []
                conflicts = []

                for id_name, group in grouped:
                    if len(group) == 1:
                        merged_rows.append(group.iloc[0].to_dict())
                    else:
                        conflict_found = False
                        merged_dict = group.iloc[0].to_dict()
                        for idx, row in group.iloc[1:].iterrows():
                            # Vérifier les conflits dans les colonnes de dates
                            overlapping = False
                            for date_col in date_columns_sorted:
                                val1 = merged_dict.get(date_col)
                                val2 = row.get(date_col)
                                # Overlapping only if both have non-empty values and different
                                if (pd.notnull(val1) and val1 != '' and
                                    pd.notnull(val2) and val2 != '' and
                                    val1 != val2):
                                    overlapping = True
                                    break
                            if overlapping:
                                conflict_found = True
                                conflicts.append(group)
                                print(f"  Conflit détecté pour 'id_name' : '{id_name}'. Fusion annulée.")
                                print(group.to_string(index=False))
                                break  # Sortir de la boucle de fusion pour ce groupe
                            else:
                                # Fusionner les valeurs non vides
                                for date_col in date_columns_sorted:
                                    if (pd.isnull(merged_dict.get(date_col)) or merged_dict.get(date_col) == '') and pd.notnull(row.get(date_col)) and row.get(date_col) != '':
                                        merged_dict[date_col] = row.get(date_col)
                        if not conflict_found:
                            # Convert merged_dict to DataFrame to perform forward-fill on date columns
                            temp_df = pd.DataFrame([merged_dict])
                            # Apply forward-fill along the row for date columns
                            temp_df[date_columns_sorted] = temp_df[date_columns_sorted].ffill(axis=1)
                            merged_dict_filled = temp_df.iloc[0].to_dict()
                            merged_rows.append(merged_dict_filled)
                            total_models_merged += (len(group) - 1)

                # Compter les conflits
                total_conflicts += len(conflicts)

                # Créer un nouveau DataFrame avec les lignes fusionnées
                new_df = pd.DataFrame(merged_rows)

                # Maintenant, effectuer un forward-fill global sur les colonnes de dates
                new_df[date_columns_sorted] = new_df[date_columns_sorted].ffill(axis=1)

                # Sauvegarder le DataFrame mis à jour dans le fichier CSV original
                try:
                    new_df.to_csv(csv_path, index=False)
                    print(f"  Fichier CSV mis à jour : {csv_path}")
                except Exception as e:
                    print(f"  Erreur lors de l'écriture du fichier CSV mis à jour : {e}")
                    continue

                print(f"  Modèles fusionnés dans ce CSV : {total_models_merged}")
                print(f"  Conflits rencontrés dans ce CSV : {len(conflicts)}")

    print("\nTraitement terminé.")
    print(f"Total de fichiers CSV traités : {total_csvs}")
    print(f"Total de modèles fusionnés : {total_models_merged}")
    print(f"Total de conflits rencontrés : {total_conflicts}")
