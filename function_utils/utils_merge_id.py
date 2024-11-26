import pandas as pd


def strategy_merge(
    base_df, 
    merge_df, 
    strategies=["exact"], 
    segments_order=None, 
    segments_no_order=None, 
    columns_to_keep=None
):
    """
    Fusionne deux DataFrames en utilisant plusieurs stratégies.

    :param base_df: DataFrame de base (premier fichier CSV).
    :param merge_df: DataFrame à fusionner avec base_df (deuxième fichier CSV).
    :param strategies: Liste des stratégies à appliquer. Exemples: ["exact", "segments_order"].
    :param segments_order: Indices des segments à utiliser pour la fusion avec ordre.
    :param segments_no_order: Indices des segments à utiliser pour la fusion sans ordre.
    :param columns_to_keep: Liste des colonnes à conserver dans le DataFrame final.
    :return: DataFrame fusionné.
    """
    result_df = base_df.copy()
    merge_df = merge_df.copy()

    for strategy in strategies:
        if strategy == "exact":
            # Fusion exacte sur id_name
            result_df = pd.merge(
                result_df, merge_df, on="id_name", how="left", suffixes=("", "_merge")
            )
        elif strategy == "segments_order" and segments_order:
            # Fusion basée sur les segments ordonnés
            def extract_ordered_segments(id_name, indices):
                parts = id_name.split("-")
                return "-".join(parts[i - 1] for i in indices if i - 1 < len(parts))

            result_df["merge_key"] = result_df["id_name"].apply(
                lambda x: extract_ordered_segments(x, segments_order)
            )
            merge_df["merge_key"] = merge_df["id_name"].apply(
                lambda x: extract_ordered_segments(x, segments_order)
            )
            result_df = pd.merge(
                result_df, merge_df, on="merge_key", how="left", suffixes=("", "_merge")
            ).drop(columns=["merge_key"])
        elif strategy == "segments_no_order" and segments_no_order:
            # Fusion basée sur les segments sans ordre
            def extract_unordered_segments(id_name, indices):
                parts = id_name.split("-")
                return frozenset(parts[i - 1] for i in indices if i - 1 < len(parts))

            result_df["merge_key"] = result_df["id_name"].apply(
                lambda x: extract_unordered_segments(x, segments_no_order)
            )
            merge_df["merge_key"] = merge_df["id_name"].apply(
                lambda x: extract_unordered_segments(x, segments_no_order)
            )
            result_df = pd.merge(
                result_df, merge_df, on="merge_key", how="left", suffixes=("", "_merge")
            ).drop(columns=["merge_key"])

    # Nettoyer les colonnes dupliquées
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    # Filtrer les colonnes à conserver si spécifié
    if columns_to_keep:
        result_df = result_df[columns_to_keep]

    return result_df
