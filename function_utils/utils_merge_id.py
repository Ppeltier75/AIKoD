import pandas as pd
import re
from fuzzywuzzy import fuzz

# Normalisation et utilitaires
def normalize_id_name(id_name):
    """
    Normalise l'ID en supprimant 'unknown' et les tirets inutiles.
    """
    segments = [seg for seg in id_name.split('-') if seg != 'unknown']
    return '-'.join(segments).strip('-')


def clean_for_fuzzy(id_name):
    """
    Prépare l'ID pour une correspondance floue : remplace les tirets par des espaces blancs.
    """
    return normalize_id_name(id_name).replace('-', ' ')


def compare_segments(id1, id2, segments):
    """
    Compare des segments spécifiques de deux IDs.
    """
    id1_segments = id1.split('-')
    id2_segments = id2.split('-')

    # Vérifie si les indices spécifiés existent dans les deux IDs
    if max(segments) > len(id1_segments) or max(segments) > len(id2_segments):
        return False

    # Compare les segments spécifiés
    return all(id1_segments[i - 1] == id2_segments[i - 1] for i in segments)


def compare_segments_no_order(id1, id2, segments):
    """
    Compare des segments spécifiques de deux IDs sans tenir compte de l'ordre.
    """
    id1_segments = id1.split('-')
    id2_segments = id2.split('-')

    # Vérifie si les indices spécifiés existent dans les deux IDs
    if max(segments) > len(id1_segments) or max(segments) > len(id2_segments):
        return False

    # Récupère les segments spécifiés pour chaque ID
    id1_subset = {id1_segments[i - 1] for i in segments}
    id2_subset = {id2_segments[i - 1] for i in segments}

    return id1_subset == id2_subset


def fuzzy_match(id1, id2, threshold=85):
    """
    Effectue une correspondance floue entre deux IDs.
    """
    return fuzz.ratio(clean_for_fuzzy(id1), clean_for_fuzzy(id2)) >= threshold


# Correspondance et fusion
def find_exact_match(df, id_name):
    """
    Trouve une correspondance exacte dans le DataFrame.
    """
    return df[df['id_name'] == id_name]


def find_partial_match(df, id_name, segments):
    """
    Trouve une correspondance basée sur des segments spécifiques.
    """
    return df[df['id_name'].apply(lambda x: compare_segments(id_name, x, segments))]


def find_match_no_order(df, id_name, segments):
    """
    Trouve une correspondance basée sur des segments spécifiques sans tenir compte de l'ordre.
    """
    return df[df['id_name'].apply(lambda x: compare_segments_no_order(id_name, x, segments))]


def find_fuzzy_match(df, id_name, threshold=85):
    """
    Trouve une correspondance floue avec nettoyage préalable des IDs.
    """
    return df[df['id_name'].apply(lambda x: fuzzy_match(id_name, x, threshold))]


# Grouper et fusionner les IDs similaires
def merge_with_flexibility(
    base_df,
    merge_df,
    strategy="exact",
    segments_order=None,
    segments_no_order=None,
    fuzzy_match=False,
    fuzzy_threshold=85,
):
    """
    Fusionne deux DataFrames sur des id_name en fonction de diverses stratégies de correspondance.

    :param base_df: DataFrame de base.
    :param merge_df: DataFrame à fusionner avec base_df.
    :param strategy: Stratégie de fusion ('exact', 'partial', 'no_order').
    :param segments_order: Liste des indices de segments (dans l'ordre) pour correspondance partielle.
    :param segments_no_order: Liste des indices de segments (sans ordre) pour correspondance partielle.
    :param fuzzy_match: Active ou désactive le fuzzy matching.
    :param fuzzy_threshold: Score minimal pour une correspondance fuzzy.
    :return: DataFrame fusionné.
    """
    def extract_segments(id_name, indices):
        """Extrait les segments spécifiés d'un id_name."""
        segments = id_name.split('-')
        return '-'.join(segments[i - 1] for i in indices if i - 1 < len(segments))

    if strategy == "exact":
        merged_df = base_df.merge(merge_df, on="id_name", how="left")
    elif strategy == "partial" and segments_order:
        base_df["merge_key"] = base_df["id_name"].apply(lambda x: extract_segments(x, segments_order))
        merge_df["merge_key"] = merge_df["id_name"].apply(lambda x: extract_segments(x, segments_order))
        merged_df = base_df.merge(merge_df, on="merge_key", how="left").drop(columns=["merge_key"])
    elif strategy == "no_order" and segments_no_order:
        base_df["merge_key"] = base_df["id_name"].apply(lambda x: frozenset(extract_segments(x, segments_no_order).split('-')))
        merge_df["merge_key"] = merge_df["id_name"].apply(lambda x: frozenset(extract_segments(x, segments_no_order).split('-')))
        merged_df = base_df.merge(merge_df, on="merge_key", how="left").drop(columns=["merge_key"])
    elif fuzzy_match:
        merge_df["fuzzy_key"] = merge_df["id_name"].apply(clean_for_fuzzy)
        base_df["fuzzy_key"] = base_df["id_name"].apply(clean_for_fuzzy)

        match_results = []
        for base_row in base_df.itertuples():
            best_match = None
            best_score = 0
            for merge_row in merge_df.itertuples():
                score = fuzz.ratio(base_row.fuzzy_key, merge_row.fuzzy_key)
                if score > best_score and score >= fuzzy_threshold:
                    best_score = score
                    best_match = merge_row

            if best_match:
                match_results.append(merge_df.loc[merge_row.Index].to_dict())
            else:
                match_results.append({col: None for col in merge_df.columns})

        fuzzy_merge = pd.DataFrame(match_results)
        merged_df = pd.concat([base_df.reset_index(drop=True), fuzzy_merge.reset_index(drop=True)], axis=1)
    else:
        raise ValueError("Stratégie de fusion non reconnue ou paramètres insuffisants.")
    return merged_df
