import pandas as pd

def generate_partial_key(id_name, segments_to_keep, separator):
    """
    Génère une clé partielle en extrayant des segments spécifiques de 'id_name'.
    
    Args:
        id_name (str or pd.Series): La colonne 'id_name'.
        segments_to_keep (list): Indices des segments à garder (1-based).
        separator (str): Séparateur utilisé dans 'id_name'.
    
    Returns:
        str or pd.Series: Clé partielle.
    """
    if isinstance(id_name, pd.Series):
        return id_name.apply(lambda x: separator.join([x.split(separator)[i-1] for i in segments_to_keep if i-1 < len(x.split(separator))]))
    else:
        segments = id_name.split(separator)
        partial_segments = [segments[i-1] for i in segments_to_keep if i-1 < len(segments)]
        return separator.join(partial_segments)

def merge_dataframes(base_df, merge_file, mode='all', segments_to_keep=None, key_name='id_name', separator='-', keep_merged_columns=True):
    """
    Fusionne un DataFrame de base avec un fichier de fusion en fonction de 'id_name' avec une option de fusion exacte ou partielle.
    
    Args:
        base_df (pd.DataFrame): DataFrame de base.
        merge_file (str): Chemin vers le fichier CSV à fusionner.
        mode (str): 'all' pour une fusion exacte, 'partial' pour une fusion partielle.
        segments_to_keep (list, optional): Liste des indices de segments à garder pour la fusion partielle.
        key_name (str): Nom de la colonne clé pour la fusion.
        separator (str): Séparateur utilisé dans 'id_name'.
        keep_merged_columns (bool): Si True, conserve les colonnes fusionnées avec un suffixe.
    
    Returns:
        pd.DataFrame: DataFrame fusionné.
    """
    if mode not in ['all', 'partial']:
        raise ValueError("Le mode doit être 'all' ou 'partial'")
    
    # Charger merge_df
    merge_df = pd.read_csv(merge_file)
    
    if key_name not in base_df.columns or key_name not in merge_df.columns:
        raise ValueError(f"La colonne clé '{key_name}' doit exister dans les deux DataFrames.")
    
    if mode == 'all':
        # Fusion exacte sur 'id_name'
        merged_df = pd.merge(base_df, merge_df, on=key_name, how='left', suffixes=('', '_merge'))
        
        if keep_merged_columns:
            # Combiner les colonnes fusionnées
            for col in merge_df.columns:
                if col == key_name:
                    continue
                merge_col = f"{col}_merge"
                if merge_col in merged_df.columns:
                    merged_df[col] = merged_df[col].combine_first(merged_df[merge_col])
                    merged_df.drop(columns=[merge_col], inplace=True)
    elif mode == 'partial':
        if not segments_to_keep:
            raise ValueError("segments_to_keep doit être fourni pour le mode 'partial'")
        
        # Générer la clé partielle pour base_df
        base_df = base_df.copy()
        base_df['partial_key'] = generate_partial_key(base_df[key_name], segments_to_keep, separator)
        
        # Générer la clé partielle pour merge_df
        merge_df = merge_df.copy()
        merge_df['partial_key'] = generate_partial_key(merge_df[key_name], segments_to_keep, separator)
        
        # Supprimer les doublons dans merge_df basé sur partial_key, garder la première occurrence
        merge_df_unique = merge_df.drop_duplicates(subset='partial_key', keep='first')
        
        # Fusionner base_df avec merge_df_unique sur partial_key
        merged_df = pd.merge(base_df, merge_df_unique, on='partial_key', how='left', suffixes=('', '_merge'))
        
        if keep_merged_columns:
            # Combiner les colonnes fusionnées
            for col in merge_df.columns:
                if col in [key_name, 'partial_key']:
                    continue
                merge_col = f"{col}_merge"
                if merge_col in merged_df.columns:
                    merged_df[col] = merged_df[col].combine_first(merged_df[merge_col])
                    merged_df.drop(columns=[merge_col], inplace=True)
        
        # Supprimer la colonne temporaire de clé partielle
        merged_df.drop(columns=['partial_key'], inplace=True)
    
    # Supprimer les doublons en gardant la première occurrence
    merged_df = merged_df.drop_duplicates(subset=key_name, keep='first')
    
    return merged_df
