import pandas as pd


def select_specific_segments(id_name, indices):
    """
    Sélectionne des segments spécifiques d'un id_name basé sur les indices fournis.
    """
    segments = id_name.split('-')
    selected = [segments[i - 1] for i in indices if 0 < i <= len(segments)]
    return '-'.join(selected)

def select_segments_no_order(id_name, indices):
    """
    Sélectionne des segments spécifiques d'un id_name sans tenir compte de l'ordre des segments.
    """
    segments = id_name.split('-')
    selected = sorted([segments[i - 1] for i in indices if 0 < i <= len(segments)])
    return '-'.join(selected)

def replace_segment(id_name, index, value):
    """
    Remplace le segment à la position donnée par une nouvelle valeur.
    
    :param id_name: La chaîne 'id_name' à modifier.
    :param index: L'indice du segment à remplacer (0-based).
    :param value: La nouvelle valeur du segment.
    :return: La chaîne 'id_name' modifiée.
    """
    segments = id_name.split('-')
    if len(segments) > index:
        segments[index] = value
    return '-'.join(segments)


def merge_csv_id_name(df_base, df_merge, keep_columns, strategies):
    """
    Fusionne les données de df_merge dans df_base en fonction des stratégies de transformation d'id_name.

    :param df_base: DataFrame de base.
    :param df_merge: DataFrame à fusionner.
    :param keep_columns: Liste des colonnes à conserver depuis df_merge.
    :param strategies: Liste de tuples (fonction_stratégie, nom_stratégie).
    :return: DataFrame fusionné.
    """
    df_merged = df_base.copy()

    # Pour chaque ligne du df_base
    for idx_base, row_base in df_base.iterrows():
        matched = False
        for strategy_func, strategy_name in strategies:
            # Transformer l'id_name de la ligne de base
            transformed_id_base = strategy_func(row_base['id_name'])
            # Transformer les id_name du df_merge
            df_merge['id_name_transformed'] = df_merge['id_name'].apply(strategy_func)
            # Rechercher les correspondances
            matches = df_merge[df_merge['id_name_transformed'] == transformed_id_base]

            if not matches.empty:
                if strategy_name == 'proxy_parameters':
                    # Extraire le 6ème segment du base id_name
                    base_segments = row_base['id_name'].split('-')
                    if len(base_segments) >= 6:
                        base_seg6 = base_segments[5]
                        try:
                            base_num = float(base_seg6)
                        except ValueError:
                            # Si le 6ème segment n'est pas numérique, ignorer cette correspondance
                            break

                        # Ajouter une colonne temporaire pour le 6ème segment numérique
                        matches['seg6_numeric'] = pd.to_numeric(matches['id_name'].str.split('-').str[5], errors='coerce')
                        # Filtrer les lignes où le 6ème segment est numérique
                        valid_matches = matches.dropna(subset=['seg6_numeric'])

                        if valid_matches.empty:
                            break  # Aucune correspondance valide, passer à la stratégie suivante

                        # Calculer la différence absolue entre les segments 6
                        valid_matches['diff'] = (valid_matches['seg6_numeric'] - base_num).abs()
                        # Trouver la ligne avec la différence minimale
                        closest_match = valid_matches.loc[valid_matches['diff'].idxmin()]

                        # Fusionner les colonnes spécifiées
                        for col in keep_columns:
                            df_merged.at[idx_base, col] = closest_match[col]
                        matched = True
                        break  # Passer à la ligne suivante du df_base
                else:
                    # Pour les autres stratégies, prendre la première correspondance
                    match_row = matches.iloc[0]
                    for col in keep_columns:
                        df_merged.at[idx_base, col] = match_row[col]
                    matched = True
                    break  # Passer à la ligne suivante du df_base

        # Optionnel : gérer les lignes non correspondantes
        # if not matched:
        #     print(f"Aucune correspondance pour la ligne {idx_base} avec id_name {row_base['id_name']}")

    # Supprimer la colonne temporaire
    if 'id_name_transformed' in df_merge.columns:
        df_merge.drop(columns=['id_name_transformed'], inplace=True)

    return df_merged
