"""Calculate wARIs metric and best wARIs matching.

Jaccard can be also used instead of ARI
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from fisher import pvalue
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.multitest import fdrcorrection

from unpast.utils.logs import get_logger

logger = get_logger(__name__)


def calc_ari_matching(
    sample_clusters_: Optional[pd.DataFrame],  # data.Frame with "samples" column
    known_groups: dict[
        str, dict[str, set]
    ],  # dict={"classification1":{"group1":{"s1","s2",...},"group2":{...}, ...}}
    all_samples: set,  # set of all samples in input; needed for overlap p-value computations
    matching_measure: str = "ARI",  # must be "ARI" or "Jaccard"
    adjust_pvals: Union[
        str, bool
    ] = "B",  # ["B", "BH", False] # correction for multiple testing
    pval_cutoff: float = 0.05,  # cutoff for p-values to select significant matches
    min_SNR: float = 0,
    min_n_genes: Union[int, bool] = False,
    min_n_samples: int = 1,
    verbose: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    # select the sample set best matching the subtype based on overlap p-value
    # adj. overlap p-value should be:
    # below pval_cutoff, e.g. < 0.05
    # the lowest among overlap p-values computed for this sample set vs all subtypes
    # cluster with the stonges overlap (max. ARI or max. Jaccard) is chosen as the best match
    if sample_clusters_ is None or sample_clusters_.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()

    ### TBD: ensure sample_clusters_ use the same sample ids as in all_samples
    sample_clusters = _filter_sample_clusters(
        sample_clusters_,
        min_n_samples=min_n_samples,
        min_SNR=min_SNR,
        min_n_genes=min_n_genes,
    )

    if sample_clusters.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()

    best_matches = []
    performances = {}
    # for each sample classification in known_groups
    for cl in known_groups.keys():
        # find best matches of true/knonw clusters `known_groups[cl]` among found sample_clusters
        perf_val, best_match_stats = _process_classification(
            cl,
            sample_clusters,
            known_groups[cl],
            all_samples,
            matching_measure,
            adjust_pvals,
            pval_cutoff,
        )
        best_matches.append(best_match_stats)
        performances[cl] = perf_val

    performances = pd.Series(performances)
    best_matches = pd.concat(best_matches, axis=0)
    return performances, best_matches


def _process_classification(
    cl: str,
    sample_clusters: pd.DataFrame,
    known_groups_cl: dict[str, set],
    all_samples: set,
    performance_measure: str,
    adjust_pvals: Union[str, bool],
    pval_cutoff: float,
) -> tuple[float, pd.DataFrame]:
    """Process a single classification 'cl' and return (performance_value, best_match_stats_df).

    This extracts the logic previously inside the 'for cl in known_groups.keys()' loop.
    """
    # denominator for weights
    N = 0
    for subt in known_groups_cl.keys():
        N += len(known_groups_cl[subt])

    # calculates raw p-values, ARIs/Jaccards
    # for all predicted vs all true clusters   
    pvals, is_enriched, performance = _evaluate_overlaps(
        sample_clusters, known_groups_cl, all_samples, method=performance_measure
    )

    pvals = _adjust_pvals_df(pvals, adjust_pvals=adjust_pvals, pval_cutoff=pval_cutoff)
    
    best_match_stats = {}
    best_pval = pvals.min(axis=1) # lowest p-val per bic. across subtypes
    for subt in known_groups_cl.keys():
        w = len(known_groups_cl[subt]) / N  # weight
        subt_pval = pvals[subt]
        # keep only the matches that pass the adjusted p-value cutoff
        passed_pvals = subt_pval[subt_pval < pval_cutoff].sort_values(ascending=False)
        if passed_pvals.shape[0] == 0: # if no significant matches 
            best_match_stats[subt] = {
                "bm_id": np.nan,
                performance_measure: 0,
                "weight": w,
                "adj_pval": np.nan,
                "is_enriched": np.nan,
                "samples": set([]),
                "n_samples": 0,
            }
        else:
            # sort passed matches by their performance decreasing and p-vals increasing
            d = pd.concat([performance.loc[passed_pvals.index.values, subt],passed_pvals],axis=1)
            d.columns = [performance_measure,"adj_pval"]
            # choose significant match with highest ARI (or J) and lowest pval
            d = d.sort_values(by= [performance_measure,"adj_pval"], ascending=[False, True])
            bm_id = d.index.values[0] # best match index
            bm_pval, bm_performance = d.loc[bm_id,["adj_pval",performance_measure]].values
            #if performance_measure == "ARI": # lower-limit is 0, ARI can be negative
            #    bm_performance = max(0,bm_performance)
                
            bm_is_enrich = is_enriched.loc[bm_id, subt]
            bm_samples = sample_clusters.loc[bm_id, "samples"]
            best_match_stats[subt] = {
                "bm_id": bm_id,
                performance_measure: bm_performance, 
                "weight": w,
                "adj_pval": bm_pval,
                "is_enriched": bm_is_enrich,
                "samples": bm_samples,
                "n_samples": len(bm_samples),
            }

    best_match_stats = pd.DataFrame.from_dict(best_match_stats).T
    best_match_stats["classification"] = cl

    perf_value = sum(best_match_stats[performance_measure] * best_match_stats["weight"])

    return perf_value, best_match_stats


def _filter_sample_clusters(
    sample_clusters_: pd.DataFrame,
    min_n_samples: int,
    min_SNR: float,
    min_n_genes: Union[int, bool],
) -> pd.DataFrame:
    """Filter sample_clusters DataFrame according to provided thresholds.

    Args:
        sample_clusters_: DataFrame containing "samples" column with sample clusters to filter.
        min_n_samples: Minimum number of samples required in a cluster.
        min_SNR: Minimum signal-to-noise ratio threshold.
        min_n_genes: Minimum number of genes required (or False to skip).

    Returns:
        Filtered DataFrame with an added "n_samples" column.
        Returns empty DataFrame if input is empty.
    """
    if sample_clusters_.shape[0] == 0:
        return pd.DataFrame()

    sample_clusters = sample_clusters_[
        sample_clusters_["samples"].apply(len) >= min_n_samples
    ]
    sample_clusters["n_samples"] = sample_clusters["samples"].apply(len)
    if min_SNR:
        sample_clusters = sample_clusters[sample_clusters["SNR"] >= min_SNR]
    if min_n_genes:
        sample_clusters = sample_clusters[
            sample_clusters["genes"].apply(len) >= min_n_genes
        ]

    return sample_clusters


def _apply_bh(df_pval: pd.DataFrame, a: float = 0.05) -> pd.DataFrame:
    """Apply Benjamini-Hochberg procedure to each column of p-value table.

    Args:
        df_pval: DataFrame of p-values.
        a: Significance level (alpha) for FDR correction.

    Returns:
        DataFrame with adjusted p-values.
    """
    df_adj = {}
    for group in df_pval.columns.values:
        bh_res, adj_pval = fdrcorrection(df_pval[group].fillna(1).values, alpha=a)
        df_adj[group] = adj_pval
    df_adj = pd.DataFrame.from_dict(df_adj)
    df_adj.index = df_pval.index
    return df_adj


def _adjust_pvals_df(
    pvals: pd.DataFrame, adjust_pvals: Union[str, bool], pval_cutoff: float = 0.05
) -> pd.DataFrame:
    """Adjust p-values DataFrame according to requested method.

    Args:
        pvals: DataFrame of p-values to adjust.
        adjust_pvals: Method for adjustment ('B' for Bonferroni, 'BH' for Benjamini-Hochberg, False for none).
        pval_cutoff: Significance cutoff for BH correction.

    Returns:
        DataFrame with adjusted p-values.

    Note:
        - If adjust_pvals is 'B', apply Bonferroni (multiply by number of tests and cap at 1).
        - If adjust_pvals is 'BH', apply Benjamini-Hochberg using _apply_bh.
        - If adjust_pvals is False or None, return pvals unchanged.
    """
    if adjust_pvals == "B":
        # Bonferroni across each row/entry: multiply p-values by number of tests (columns)
        pvals = pvals * pvals.shape[0]
        pvals = pvals.applymap(lambda x: min(x, 1))
        return pvals
    elif adjust_pvals == "BH":
        return _apply_bh(pvals, a=pval_cutoff)
    elif adjust_pvals == False:
        return pvals
    else:
        raise ValueError(
            f"Unrecognized adjust_pvals value: {adjust_pvals}, expected 'B', 'BH', or False."
        )


def _evaluate_overlaps(
    biclusters: pd.DataFrame,
    known_groups: dict[str, set],
    all_elements: set,
    method: str,
    dimension: str = "samples",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute Fisher's p-values and Jaccard/ARI overlaps between biclusters and known groups.

    Args:
        biclusters: DataFrame containing biclusters to evaluate.
        known_groups: Dictionary mapping group names to sets of members.
        all_elements: Set of all elements in the dataset.
        method: Overlap measure to use ("Jaccard" or "ARI").
        dimension: Dimension to evaluate ("samples" or "genes").

    Returns:
        Tuple containing:
            - pvals: DataFrame of p-values from Fisher's exact test
            - is_enriched: DataFrame indicating enrichment direction
            - metric_vals: DataFrame of Jaccard or ARI values
    """

    assert method in ["Jaccard", "ARI"], (
        f"Unsupported method: {method}, expected 'Jaccard' or 'ARI'."
    )
    pvals = {}
    is_enriched = {}
    metric_vals = {}
    N = len(all_elements)

    if method == "ARI":
        all_elements_list = sorted(all_elements)

    # sanity check - biclusters
    for i in biclusters.index.values:
        bic_members = biclusters.loc[i, dimension]
        if not bic_members.intersection(all_elements) == bic_members:
            _diff_str = " ".join(bic_members.difference(all_elements))
            logger.warning(
                f"bicluster {i} elements {_diff_str} are not in 'all_elements'"
            )
            bic_members = bic_members.intersection(all_elements)

    # sanity check and sorting
    group_names = list(known_groups.keys())
    sorted_group_names = [group_names[0]]  # group names ordered by group size
    for group in group_names:
        group_members = known_groups[group]
        if not group_members.intersection(all_elements) == group_members:
            _diff_str = " ".join(group_members.difference(all_elements))
            logger.error(f"{group} elements {_diff_str} are not in 'all_elements'")
            raise RuntimeError(
                f"{group} elements {_diff_str} are not in 'all_elements'"
            )

        if group != group_names[0]:
            for gn in range(len(sorted_group_names)):
                if len(group_members) < len(known_groups[sorted_group_names[gn]]):
                    sorted_group_names = (
                        sorted_group_names[:gn] + [group] + sorted_group_names[gn:]
                    )
                    break
                elif gn == len(sorted_group_names) - 1:
                    sorted_group_names = [group] + sorted_group_names

    # print(sorted_group_names)
    for group in sorted_group_names:
        group_members = known_groups[group]
        pvals[group] = {}
        is_enriched[group] = {}
        metric_vals[group] = {}

        if method == "ARI":
            # binary vector for target cluster
            group_binary = np.zeros(len(all_elements_list))
            for j in range(len(all_elements_list)):
                if all_elements_list[j] in group_members:
                    group_binary[j] = 1

        for i in biclusters.index.values:
            bic = biclusters.loc[i, :]
            bic_members = bic[dimension]

            if method == "ARI":
                bic_binary = np.zeros(len(all_elements_list))
                for j in range(len(all_elements_list)):
                    if all_elements_list[j] in bic_members:
                        bic_binary[j] = 1
                # calculate ARI for 2 binary vectors:
                metric_vals[group][i] = adjusted_rand_score(group_binary, bic_binary)

            # Fisher's exact test
            shared = len(bic_members.intersection(group_members))
            bic_only = len(bic_members.difference(group_members))
            group_only = len(group_members.difference(bic_members))
            union = shared + bic_only + group_only
            pval = pvalue(shared, bic_only, group_only, N - union)

            pvals[group][i] = pval.two_tail
            is_enriched[group][i] = False
            if pval.right_tail < pval.left_tail:
                is_enriched[group][i] = True

            if method == "Jaccard":
                metric_vals[group][i] = shared / union

    pvals = pd.DataFrame.from_dict(pvals).loc[:, sorted_group_names]
    is_enriched = pd.DataFrame.from_dict(is_enriched).loc[:, sorted_group_names]
    metric_vals = pd.DataFrame.from_dict(metric_vals).loc[:, sorted_group_names]
    return pvals, is_enriched, metric_vals
