import pandas as pd

def select_specific_segments(id_name, segments_to_select):
    segments = id_name.split('-')
    selected_segments = [segments[i - 1] for i in segments_to_select if i - 1 < len(segments)]
    return '-'.join(selected_segments)

def select_segments_no_order(id_name, segments_to_select):
    segments = id_name.split('-')
    selected_segments = {segments[i - 1] for i in segments_to_select if i - 1 < len(segments)}
    return '-'.join(sorted(selected_segments))

def merge_csv_id_name(df_base, df_merge, keep_columns, strategies):
    df_merged = df_base.copy()

    # Pour chaque ligne du df_base
    for idx_base, row_base in df_base.iterrows():
        matched = False
        # Parcourir les stratégies
        for strategy in strategies:
            # Transformer l'id_name de la ligne de base
            transformed_id_base = strategy(row_base['id_name'])
            # Transformer les id_name du df_merge
            df_merge['id_name_transformed'] = df_merge['id_name'].apply(strategy)
            # Rechercher les correspondances
            matches = df_merge[df_merge['id_name_transformed'] == transformed_id_base]
            if not matches.empty:
                # Si plusieurs correspondances, prendre la première
                match_row = matches.iloc[0]
                # Fusionner les colonnes spécifiées
                for col in keep_columns:
                    df_merged.at[idx_base, col] = match_row[col]
                matched = True
                break  # Passer à la ligne suivante du df_base
        # Optionnel : afficher si aucune correspondance n'a été trouvée
        # if not matched:
        #     print(f"Aucune correspondance pour la ligne {idx_base} avec id_name {row_base['id_name']}")
    # Supprimer la colonne temporaire
    if 'id_name_transformed' in df_merge.columns:
        df_merge.drop(columns=['id_name_transformed'], inplace=True)
    return df_merged
