import sys
import numpy as np
import pandas as pd
from fisher import pvalue
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import fdrcorrection

from unpast.utils.method import zscore
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test


def generate_exprs(
    data_sizes,
    g_size=5,
    frac_samples=[0.05, 0.1, 0.25, 0.5],
    m=2.0,
    std=1,
    z=True,
    outdir="./",
    outfile_basename="",
    g_overlap=False,
    s_overlap=True,
    seed=42,
    add_coexpressed=[],
):
    n_genes, N = data_sizes
    biclusters = {}
    coexpressed_modules = []

    # generate background model
    np.random.seed(seed)
    exprs = pd.DataFrame(np.random.normal(loc=0, scale=1.0, size=(n_genes, N)))
    exprs.columns = ["s_" + str(x) for x in exprs.columns.values]
    exprs.index = ["g_" + str(x) for x in exprs.index.values]

    # implant bicluster
    N = exprs.shape[1]
    bic_g = []
    bic_s = []
    bg_g = set(exprs.index.values).difference(set(bic_g))
    bg_s = set(exprs.columns.values).difference(set(bic_s))
    bicluster_genes = []
    for s_frac in frac_samples:
        s_size = int(s_frac * N)
        # select random sets of samples and genes from the background
        bic_genes = list(np.random.choice(list(bg_g), size=g_size, replace=False))
        bic_samples = list(np.random.choice(list(bg_s), size=s_size, replace=False))
        bic_g += bic_genes
        bic_s += bic_samples
        # identify samples outside the bicluster
        if not g_overlap:
            bg_g = bg_g.difference(set(bic_g))
        if not s_overlap:
            bg_s = bg_s.difference(set(bic_s))
        # generate bicluster
        biclusters[s_frac] = {
            "genes": set(bic_genes),
            "samples": set(bic_samples),
            "frac": s_frac,
            "n_genes": len(bic_genes),
            "n_samples:": len(bic_samples),
        }
        bic_exprs = np.random.normal(loc=m, scale=std, size=(g_size, s_size))
        # implant biclusters
        exprs.loc[bic_genes, bic_samples] += bic_exprs
        bicluster_genes += bic_genes

    # add modules of co-expressed genes
    bg_g = set(exprs.index.values).difference(set(bicluster_genes))
    r = 0.5
    for module in add_coexpressed:
        module_genes = list(np.random.choice(list(bg_g), size=module, replace=False))
        n = exprs.loc[module_genes[0], :]
        for i in range(1, module):
            n_i = n * r + np.sqrt(1 - r ** 2) * exprs.loc[module_genes[i], :]
            exprs.loc[module_genes[i], :] = n_i
        print(
            "\tco-exprs. module ",
            module,
            "r=",
            (exprs.loc[module_genes, :].T.corr().sum().sum() - module)
            / (module ** 2 / 2 - module),
        )
        coexpressed_modules.append(module_genes)

    if z:
        # center to 0 and scale std to 1
        exprs = zscore(exprs)
    biclusters = pd.DataFrame.from_dict(biclusters).T
    # biclusters.set_index("frac",inplace = True,drop=True)
    biclusters_ = biclusters.copy()

    if outfile_basename:
        # overlap extension
        if s_overlap:
            if g_overlap:
                overlap_ext = ",overlap=yes"
            else:
                overlap_ext = ",overlap=s"
        elif g_overlap:
            overlap_ext = ",overlap=g"
        else:
            overlap_ext = ",overlap=no"
        # save expressions
        exprs_file = (
            outdir
            + "/"
            + outfile_basename
            + ".n_genes="
            + str(g_size)
            + ",m="
            + str(m)
            + ",std="
            + str(std)
        )
        exprs_file += overlap_ext + ".exprs_z.tsv"
        print("expressions:", exprs_file)
        exprs.to_csv(exprs_file, sep="\t")

        # save ground truth
        biclusters["n_genes"] = biclusters["genes"].apply(lambda x: len(x))
        biclusters["n_samples"] = biclusters["samples"].apply(lambda x: len(x))
        biclusters["genes"] = biclusters["genes"].apply(
            lambda x: " ".join((map(str, sorted(x))))
        )
        biclusters["samples"] = biclusters["samples"].apply(
            lambda x: " ".join((map(str, sorted(x))))
        )

        biclusters_file = (
            outdir
            + "/"
            + outfile_basename
            + ".n_genes="
            + str(g_size)
            + ",m="
            + str(m)
            + ",std="
            + str(std)
        )
        biclusters_file += overlap_ext + ".biclusters.tsv"
        biclusters.to_csv(biclusters_file, sep="\t")
        print("true bilusters:", biclusters_file)
        biclusters.to_csv(biclusters_file, sep="\t")

    return exprs, biclusters_, coexpressed_modules


def make_ref_groups(subtypes, annotation, exprs):
    import copy
    from collections import OrderedDict

    # prepared a dict of subtype classifications {"class1":{"subt1":[],"subt2":[]},"class2":{"subtA":[],"subtB":[]}}
    all_samples = set(exprs.columns.values).intersection(set(subtypes.index.values))
    s = sorted(all_samples)

    pam50 = make_known_groups(
        subtypes.loc[s, :], exprs.loc[:, s], target_col="PAM50", verbose=False
    )
    lum = {}
    lum["Luminal"] = pam50["LumA"].union(pam50["LumB"]).intersection(all_samples)
    scmod2 = make_known_groups(
        subtypes.loc[s, :], exprs.loc[:, s], target_col="SCMOD2", verbose=False
    )
    claudin = {}
    claudin["Claudin-low"] = set(
        subtypes.loc[subtypes["claudin_low"] == 1, :].index.values
    ).intersection(all_samples)

    ihc = {}
    for x in ["IHC_HER2", "IHC_ER", "IHC_PR"]:
        ihc[x] = set(
            annotation.loc[annotation[x] == "Positive", :].index.values
        )  # .intersection(all_samples)
    ihc["IHC_TNBC"] = set(
        annotation.loc[annotation["IHC_TNBC"] == 1, :].index.values
    )  # .intersection(all_samples)

    pam50_lum = copy.copy(pam50)
    del pam50_lum["LumA"]
    del pam50_lum["LumB"]
    pam50_lum["Luminal"] = lum["Luminal"]
    intrinsic = copy.copy(pam50_lum)
    intrinsic["Claudin-low"] = claudin["Claudin-low"]
    known_groups = OrderedDict(
        {
            "PAM50": pam50_lum,
            "Intrinsic": intrinsic,
            "PAM50_AB": pam50,
            "SCMOD2": scmod2,
            "IHC": ihc,
        }
    )
    known_groups["Luminal"] = {"Luminal": pam50_lum["Luminal"]}
    known_groups["Basal"] = {"Basal": pam50["Basal"]}
    known_groups["Her2"] = {"Her2": pam50["Her2"]}
    known_groups["LumA"] = {"LumA": pam50["LumA"]}
    known_groups["LumB"] = {"LumB": pam50["LumB"]}
    known_groups["Normal"] = {"Normal": pam50["Normal"]}
    known_groups["Claudin-low"] = {"Claudin-low": claudin["Claudin-low"]}
    known_groups["IHC_HER2"] = {"IHC_HER2": ihc["IHC_HER2"]}
    known_groups["IHC_ER"] = {"IHC_ER": ihc["IHC_ER"]}
    known_groups["IHC_PR"] = {"IHC_PR": ihc["IHC_PR"]}
    known_groups["IHC_TNBC"] = {"IHC_TNBC": ihc["IHC_TNBC"]}
    known_groups["NET_kmeans"] = {
        "NET": set(subtypes.loc[subtypes["NET_km"] == 1, :].index.values)
    }
    known_groups["NET_ward"] = {
        "NET": set(subtypes.loc[subtypes["NET_w"] == 1, :].index.values)
    }
    return known_groups, all_samples


def make_known_groups(annot, exprs, target_col="genefu_z", verbose=False):
    samples = set(exprs.columns.values).intersection(set(annot.index.values))
    if verbose:
        print("Total samples:", len(samples), file=sys.stdout)
    annot = annot.loc[list(samples), :]
    groups = set(annot.loc[:, target_col].values)

    known_groups = {}
    for group in groups:
        if group == group:
            group_samples = set(annot.loc[annot[target_col] == group, :].index.values)
            group_samples = group_samples.intersection(samples)
            if len(group_samples) > int(len(samples) / 2):
                print("take complement of ", group, file=sys.stderr)
                group_samples = samples.difference(group_samples)
            known_groups[
                group
            ] = group_samples  # {"set":group_samples,"complement": samples.difference(group_samples)}
            if verbose:
                print(
                    group,
                    round(len(group_samples) / len(samples), 2),
                    len(group_samples),
                    len(samples.difference(group_samples)),
                )
    return known_groups


def compare_gene_clusters(bics1, bics2, N):
    # N - total number of genes
    # finds best matched B1 -> B2 and B2 -> B1
    # calculates % of matched clusters, number of genes in matched clusters,
    # and the average J index for best matches
    bm = find_best_matching_biclusters(bics1, bics2, (N, 0), by="genes")
    bm = bm.dropna()
    bm2 = find_best_matching_biclusters(bics2, bics1, (N, 0), by="genes")
    bm2 = bm2.dropna()

    if "n_shared_genes" in bm.columns:
        bm = bm.loc[bm["n_shared_genes"] > 1, :].sort_values(
            by="n_shared_genes", ascending=False
        )
    else:
        # no match -> remove all rows
        bm = bm.head(0)
    if "n_shared_genes" in bm2.columns:
        bm2 = bm2.loc[bm2["n_shared_genes"] > 1, :].sort_values(
            by="n_shared_genes", ascending=False
        )
    else:
        bm2 = bm.head(0)

    clust_similarity = {}
    # number of biclusters
    clust_similarity["n_1"] = bics1.shape[0]
    clust_similarity["n_2"] = bics2.shape[0]
    # print("% matched biclusters:",bm.shape[0]/tcga_result.shape[0],bm2.shape[0]/metabric_result.shape[0])
    clust_similarity["percent_matched_1"] = bm.shape[0] / bics1.shape[0]
    clust_similarity["percent_matched_2"] = bm2.shape[0] / bics2.shape[0]

    # print("n matched genes:",bm.loc[:,"n_shared"].sum(),bm2.loc[:,"n_shared"].sum())
    if "n_shared_genes" in bm.columns:
        clust_similarity["n_shared_genes_1"] = bm.loc[:, "n_shared_genes"].sum()
        clust_similarity["avg_bm_J_1"] = bm.loc[:, "J"].mean()
    if "n_shared_genes" in bm2.columns:
        clust_similarity["n_shared_genes_2"] = bm2.loc[:, "n_shared_genes"].sum()
        # print("avg. J:",bm.loc[:,"J"].mean(),bm2.loc[:,"J"].mean())
        clust_similarity["avg_bm_J_2"] = bm2.loc[:, "J"].mean()

    return clust_similarity, bm, bm2


from sklearn.metrics import adjusted_rand_score


def calculate_perfromance(
    sample_clusters_,  # data.Frame with "samples" column
    known_groups,  # dict={"classification1":{"group1":{"s1","s2",...},"group2":{...}, ...}}
    all_samples,  # set of all samples in input; needed for overlap p-value computations
    performance_measure="Jaccard",  # must be "ARI" or "Jaccard"
    adjust_pvals="B",  # ["B", "BH", False] # correction for multiple testing
    pval_cutoff=0.05,  # cutoff for p-values to select significant matches
    min_SNR=0,
    min_n_genes=False,
    min_n_samples=1,
    verbose=False,
):
    # select the sample set best matching the subtype based on p-value
    # adj. overlap p-value should be:
    # below pval_cutoff, e.g. < 0.05
    # the lowest among overlap p-values computed for this sample set vs all subtypes
    # cluster with the highest J is chosen as the best match
    if sample_clusters_ is None or sample_clusters_.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()

    if adjust_pvals not in ["B", "BH", False]:
        print(
            adjust_pvals,
            "is not recognized, Bonferroni method will be used,",
            file=sys.stderr,
        )
    sample_clusters = sample_clusters_[
        sample_clusters_["samples"].apply(lambda x: len(x)) >= min_n_samples
    ]
    sample_clusters["n_samples"] = sample_clusters["samples"].apply(lambda x: len(x))
    if min_SNR:
        sample_clusters = sample_clusters[sample_clusters["SNR"] >= min_SNR]
    if min_n_genes:
        sample_clusters = sample_clusters[
            sample_clusters["genes"].apply(lambda x: len(x)) >= min_n_genes
        ]

    if sample_clusters.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()

    best_matches = []
    performances = {}
    for cl in known_groups.keys():
        # denominator for weights
        N = 0
        for subt in known_groups[cl].keys():
            N += len(known_groups[cl][subt])
        if performance_measure == "ARI":
            pvals, is_enriched, performance = evaluate_overlaps_ARI(
                sample_clusters, known_groups[cl], all_samples
            )
        if performance_measure == "Jaccard":
            pvals, is_enriched, performance = evaluate_overlaps(
                sample_clusters, known_groups[cl], all_samples
            )
        if adjust_pvals:
            if adjust_pvals == "B":
                pvals = pvals * pvals.shape[0]
                pvals = pvals.applymap(lambda x: min(x, 1))
            elif adjust_pvals == "BH":
                pvals = apply_bh(pvals, a=pval_cutoff)
        best_match_stats = {}
        best_pval = pvals.min(axis=1)
        for subt in known_groups[cl].keys():
            w = len(known_groups[cl][subt]) / N  # weight
            subt_pval = pvals[subt]
            passed_pvals = subt_pval[subt_pval == best_pval]
            passed_pvals = subt_pval[subt_pval < pval_cutoff].index.values
            d = performance.loc[passed_pvals, subt].sort_values(ascending=False)
            if d.shape[0] == 0:
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
                bm_j = d.values[0]
                d = d[d == bm_j]
                bm_id = sorted(
                    pvals.loc[d.index.values, :]
                    .sort_values(by=subt, ascending=True)
                    .index.values
                )[0]
                bm_pval = pvals.loc[bm_id, subt]
                bm_j = performance.loc[bm_id, subt]
                bm_is_enrich = is_enriched.loc[bm_id, subt]
                bm_samples = sample_clusters.loc[bm_id, "samples"]
                best_match_stats[subt] = {
                    "bm_id": bm_id,
                    performance_measure: bm_j,
                    "weight": w,
                    "adj_pval": bm_pval,
                    "is_enriched": bm_is_enrich,
                    "samples": bm_samples,
                    "n_samples": len(bm_samples),
                }

        best_match_stats = pd.DataFrame.from_dict(best_match_stats).T
        best_match_stats["classification"] = cl
        best_matches.append(best_match_stats)
        performances[cl] = sum(
            best_match_stats[performance_measure] * best_match_stats["weight"]
        )

    performances = pd.Series(performances)
    best_matches = pd.concat(best_matches, axis=0)
    return performances, best_matches


def evaluate_overlaps_ARI(biclusters, known_groups, all_elements):
    # compute exact Fisher's p-values and Jaccard overlaps for samples
    pvals = {}
    is_enriched = {}
    ARI = {}
    N = len(all_elements)
    all_elements_list = sorted(all_elements)
    # sanity check - biclusters
    for i in biclusters.index.values:
        bic_members = biclusters.loc[i, "samples"]
        if not bic_members.intersection(all_elements) == bic_members:
            print(
                "bicluster {} elements {} are not in 'all_elements'".format(
                    i, " ".join(bic_members.difference(all_elements))
                ),
                file=sys.stderr,
            )
            bic_members = bic_members.intersection(all_elements)
    # sanity check and sorting
    group_names = list(known_groups.keys())
    sorted_group_names = [group_names[0]]  # group names ordered by group size
    for group in group_names:
        group_members = known_groups[group]
        if not group_members.intersection(all_elements) == group_members:
            print(group, "elements are not in 'all_elements'", file=sys.stderr)
            return

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
        ARI[group] = {}
        # binary vector for target cluster
        group_binary = np.zeros(len(all_elements_list))
        for j in range(len(all_elements_list)):
            if all_elements_list[j] in group_members:
                group_binary[j] = 1
        for i in biclusters.index.values:
            bic = biclusters.loc[i, :]
            bic_members = bic["samples"]
            bic_binary = np.zeros(len(all_elements_list))
            for j in range(len(all_elements_list)):
                if all_elements_list[j] in bic_members:
                    bic_binary[j] = 1
            # calculate ARI for 2 binary vectors:
            ARI[group][i] = adjusted_rand_score(group_binary, bic_binary)

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

    pvals = pd.DataFrame.from_dict(pvals).loc[:, sorted_group_names]
    is_enriched = pd.DataFrame.from_dict(is_enriched).loc[:, sorted_group_names]
    ARI = pd.DataFrame.from_dict(ARI).loc[:, sorted_group_names]
    return pvals, is_enriched, ARI


def apply_bh(df_pval, a=0.05):
    # applies BH procedure to each column of p-value table
    df_adj = {}
    for group in df_pval.columns.values:
        bh_res, adj_pval = fdrcorrection(df_pval[group].fillna(1).values, alpha=a)
        df_adj[group] = adj_pval
    df_adj = pd.DataFrame.from_dict(df_adj)
    df_adj.index = df_pval.index
    return df_adj


def evaluate_overlaps(biclusters, known_groups, all_elements, dimension="samples"):
    # compute exact Fisher's p-values and Jaccard overlaps for samples
    pvals = {}
    is_enriched = {}
    jaccards = {}
    N = len(all_elements)
    # sanity check - biclusters
    for i in biclusters.index.values:
        bic_members = biclusters.loc[i, dimension]
        if not bic_members.intersection(all_elements) == bic_members:
            print(
                "bicluster {} elements {} are not in 'all_elements'".format(
                    i, " ".join(bic_members.difference(all_elements))
                ),
                file=sys.stderr,
            )
            bic_members = bic_members.intersection(all_elements)
    # sanity check and sorting
    group_names = list(known_groups.keys())
    sorted_group_names = [group_names[0]]  # group names ordered by group size
    for group in group_names:
        group_members = known_groups[group]
        if not group_members.intersection(all_elements) == group_members:
            print(group, "elements are not in 'all_elements'", file=sys.stderr)
            return

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
        jaccards[group] = {}
        for i in biclusters.index.values:
            bic = biclusters.loc[i, :]
            bic_members = bic[dimension]
            shared = len(bic_members.intersection(group_members))
            bic_only = len(bic_members.difference(group_members))
            group_only = len(group_members.difference(bic_members))
            union = shared + bic_only + group_only
            pval = pvalue(shared, bic_only, group_only, N - union)
            """pvals[group][i] = 1 
            if pval.right_tail < pval.left_tail:
                pvals[group][i] = pval.right_tail
                is_enriched[group][i] = True
            # if under-representation, flip query set
            else:
                pvals[group][i] = pval.left_tail # save left-tail p-value and record that this is not enrichment
                is_enriched[group][i] = False
                bic_members = all_elements.difference(bic_members)
                shared = len(bic_members.intersection(group_members))
                union = len(bic_members.union(group_members))"""
            pvals[group][i] = pval.two_tail
            is_enriched[group][i] = False
            if pval.right_tail < pval.left_tail:
                is_enriched[group][i] = True

            jaccards[group][i] = shared / union

        # print(group,jaccards[group])

    pvals = pd.DataFrame.from_dict(pvals).loc[:, sorted_group_names]
    is_enriched = pd.DataFrame.from_dict(is_enriched).loc[:, sorted_group_names]
    jaccards = pd.DataFrame.from_dict(jaccards).loc[:, sorted_group_names]
    return pvals, is_enriched, jaccards


def calc_overlap_pval(overlap, group1_only, group2_only, background, max_N=5000):
    # if sample size < max_N), use Fisher's exact
    # otherwise replacing exact Fisher's with chi2
    if overlap + group1_only + group2_only + background < max_N:
        pval = pvalue(overlap, group1_only, group2_only, background).right_tail
    else:
        chi2, pval, dof, expected = chi2_contingency(
            [[overlap, group1_only], [group2_only, background]]
        )
    return pval


def find_best_matching_biclusters(
    bics1, bics2, sizes, by="genes", adj_pval_thr=0.05, min_g=2
):
    # takes two biluster dafaframes from read_bic_table
    # by = "genes" or "samples" or "both"
    # sizes - dimensions of input matrix (n_genes,n_samples)
    # finds best matches of bics1 biclusters among bics2 biclusters

    N_g, N_s = sizes
    n_bics1 = bics1.shape[0]
    n_bics2 = bics2.shape[0]

    best_matches = {}  # OrderedDict({})
    for row1 in bics1.iterrows():
        bic1 = row1[1]
        i1 = row1[0]

        best_matches[i1] = {}
        bm_J = 0
        bm_o = 0
        bm_adj_pval = 1
        bm_id = None

        for row2 in bics2.iterrows():
            bic2 = row2[1]
            i2 = row2[0]

            g1 = bic1["genes"]
            s1 = bic1["samples"]
            g2 = bic2["genes"]
            s2 = bic2["samples"]
            o_g = len(g1.intersection(g2))
            o_s = len(s1.intersection(s2))
            s1 = len(s1)
            s2 = len(s2)
            g1 = len(g1)
            g2 = len(g2)
            J = 0
            adj_pval = 1
            # if not by="samples", ignore overlaps with gene < min_g
            if (by != "samples" and o_g >= min_g) or by == "samples":
                if by == "genes" or by == "both":
                    g2_ = g2 - o_g  # genes exclusively in bicluster 2
                    g1_ = g1 - o_g
                    u_g = g1_ + g2_ + o_g
                    bg_g = N_g - u_g
                    J_g = o_g * 1.0 / u_g
                    if not by == "both":
                        pval_g = calc_overlap_pval(o_g, g1_, g2_, bg_g)
                elif by == "samples" or by == "both":
                    s2_ = s2 - o_s  # samples exclusively in bicluster 2
                    s1_ = s1 - o_s
                    u_s = s1_ + s2_ + o_s
                    bg_s = N_s - u_s
                    pval_s = calc_overlap_pval(o_s, s1_, s2_, bg_s)
                    # if p-val is high but one of the biclusters is large,
                    # try flipping the largest bicluster if it is close to 50% of the cohort
                    if pval_s > adj_pval_thr and max(s1, s2) > 0.4 * N_s:
                        if s1 > s2:  # flip s1
                            s1 = N_s - s1
                            u_s = bg_s + s2
                            o_s = s2_
                            s2_ = s2 - o_s
                            bg_s = s1_
                            s1_ = s1 - o_s
                        else:  # flip s2
                            s2 = N_s - s2
                            u_s = bg_s + s1
                            o_s = s1_
                            s1_ = s1 - o_s
                            bg_s = s2_
                            s2_ = s2 - o_s
                        assert bg_s == N_s - u_s, (
                            "i1=%s; i2=%s: bg=%s, N_s=%s, u_s=%s"
                            % (i1, i2, bg_s, N_s, u_s)
                        )
                        assert u_s == o_s + s1_ + s2_, (
                            "i1=%s; i2=%s: u_s=%s, o_s=%s, s1_=%s, s2_=%s"
                            % (i1, i2, u_s, o_s, s1_, s2_)
                        )
                        if not by == "both":
                            # compute p-value again
                            pval_s = calc_overlap_pval(o_s, s1_, s2_, bg_s)
                    J_s = o_s * 1.0 / u_s

                if by == "genes":
                    J = J_g
                    pval = pval_g
                    o = o_g
                elif by == "samples":
                    J = J_s
                    pval = pval_s
                    o = o_s
                else:
                    o = o_s * o_g  # bicluster overlap
                    b1_ = s1 * g1 - o  # exclusive bicluster 1 area
                    b2_ = s2 * g2 - o
                    u = o + b1_ + b2_
                    bg = N_s * N_g - u
                    J = o * 1.0 / u
                    pval = calc_overlap_pval(o, b1_, b2_, bg)

                adj_pval = pval * n_bics2 * n_bics1
                if adj_pval < adj_pval_thr and J > 0:
                    if J > bm_J or (J == bm_J and adj_pval < bm_adj_pval):
                        bm_J = J
                        bm_adj_pval = adj_pval
                        bm_id = i2
                        bm_o = o
        best_matches[i1]["bm_id"] = bm_id
        best_matches[i1]["J"] = bm_J
        best_matches[i1]["adj_pval"] = bm_adj_pval
        if "genes" in bics1.columns and "genes" in bics2.columns:
            if bm_id:
                best_matches[i1]["shared_genes"] = bics2.loc[
                    bm_id, "genes"
                ].intersection(bics1.loc[i1, "genes"])
                best_matches[i1]["n_shared_genes"] = len(
                    best_matches[i1]["shared_genes"]
                )
                best_matches[i1]["bm_genes"] = bics2.loc[bm_id, "genes"]
                best_matches[i1]["bm_n_genes"] = bics2.loc[bm_id, "n_genes"]
            best_matches[i1]["genes"] = bics1.loc[i1, "genes"]
            best_matches[i1]["n_genes"] = bics1.loc[i1, "n_genes"]
        if "samples" in bics1.columns and "samples" in bics2.columns:
            best_matches[i1]["n_samples"] = bics1.loc[i1, "n_samples"]
            best_matches[i1]["samples"] = bics1.loc[i1, "samples"]
            if bm_id:
                best_matches[i1]["bm_n_samples"] = bics2.loc[bm_id, "n_samples"]
                best_matches[i1]["bm_samples"] = bics2.loc[bm_id, "samples"]
                best_matches[i1]["shared_samples"] = bics2.loc[
                    bm_id, "samples"
                ].intersection(bics1.loc[i1, "samples"])
                best_matches[i1]["n_shared_samples"] = len(
                    best_matches[i1]["shared_samples"]
                )

    best_matches = pd.DataFrame.from_dict(best_matches).T
    return best_matches


def bic_survival(surv_anno, samples, event="OS", surv_time="", lr=True, verbose=True):
    # surival annotation - annotation matrix with time,event, and covariates
    # samples - samples in a group, e.g. biclsuter samples
    # check  complete separation
    # if all events are either inside or outside sample group
    if not surv_time:
        surv_time = event + ".time"
    surv_data = surv_anno.copy()
    surv_data = surv_data.dropna(axis=0)

    # check zero variance columns:
    v = surv_data.var()
    for col in v.index:
        if v[col] == 0:
            if verbose:
                print(col, "with zero variance excluded", file=sys.stderr)
            surv_data = surv_data.drop(col, axis=1)

    surv_data.loc[:, "x"] = 0
    surv_data.loc[list(set(samples).intersection(set(surv_data.index.values))), "x"] = 1

    pval = np.nan
    hr, upper_95CI, lower_95CI = np.nan, np.nan, np.nan
    results = {}

    events = surv_data[event].astype(bool)

    v1 = surv_data.loc[events, "x"].var()
    v2 = surv_data.loc[~events, "x"].var()

    v3 = surv_data.loc[surv_data["x"] == 1, event].var()
    v4 = surv_data.loc[surv_data["x"] == 0, event].var()

    if v1 == 0 or v2 == 0:
        if verbose:
            in_bic = surv_data.loc[surv_data["x"] == 1, :].shape[0]
            in_bg = surv_data.loc[surv_data["x"] == 0, :].shape[0]
            print(
                "perfect separation for biclsuter of  %s/%s samples" % (in_bic, in_bg),
                "variances: {:.2f} {:.2f}".format(v1, v2),
                file=sys.stderr,
            )
    if v3 == 0:
        print(
            "zero variance for events in group; all events are ",
            set(surv_data.loc[surv_data["x"] == 1, event].values),
        )
    if v4 == 0:
        print(
            "zero variance for events in background; all events are ",
            set(surv_data.loc[surv_data["x"] == 0, event].values),
        )

    # check variance of covariates in event groups
    exclude_covars = []
    for c in [x for x in surv_data.columns.values if not x in ["x", event, surv_time]]:
        if surv_data.loc[events, c].var() == 0:
            exclude_covars.append(c)
            print("\t", c, "variance is 0 in event group", file=sys.stdout)
        if surv_data.loc[~events, c].var() == 0:
            exclude_covars.append(c)
            print("\t", c, "variance is 0 in no-event group", file=sys.stdout)
    # if len(exclude_covars)>0:
    #    cols = surv_data.columns.values
    #    cols = [x for x in cols if not x in exclude_covars]
    #    surv_data = surv_data.loc[:,cols]

    else:
        try:
            cph = CoxPHFitter()
            res = cph.fit(
                surv_data, duration_col=surv_time, event_col=event, show_progress=False
            )
            res_table = res.summary
            res_table = res_table  # .sort_values("p")
            pval = res_table.loc["x", "p"]
            hr = res_table.loc["x", "exp(coef)"]
            upper_95CI = res_table.loc["x", "exp(coef) upper 95%"]
            lower_95CI = res_table.loc["x", "exp(coef) lower 95%"]
        except:
            pass

    results = {
        "p_value": pval,
        "HR": hr,
        "upper_95CI": upper_95CI,
        "lower_95CI": lower_95CI,
    }
    # Log-rank test
    if lr:
        bic = surv_data.loc[surv_data["x"] == 1, :]
        bg = surv_data.loc[surv_data["x"] == 0, :]

        lr_result = logrank_test(
            bic.loc[:, surv_time],
            bg.loc[:, surv_time],
            event_observed_A=bic.loc[:, event],
            event_observed_B=bg.loc[:, event],
        )
        results["LogR_p_value"] = lr_result.p_value

    return results


def add_survival(
    biclusters,  # dataframes with biclustes
    sample_data,  # sample annotation
    event="OS",
    surv_time="",  # event and time column names
    covariates=[],
    min_n_events=5,
    verbose=True,
):
    # if too few events, add na columns
    if sample_data[event].sum() < min_n_events:
        df = biclusters.copy()
        for col in [
            ".p_value",
            ".p_value_BH",
            ".HR",
            ".upper_95CI",
            ".lower_95CI",
            ".LogR_p_value",
            ".LogR_p_value_BH",
        ]:
            df[event + col] = np.nan
        return df
    if not surv_time:
        surv_time = event + ".time"
    surv_results = {}
    for bic in biclusters.iterrows():
        sample_set = bic[1]["samples"]
        surv_data = sample_data.loc[:, covariates + [event, surv_time]]

        surv_results[bic[0]] = bic_survival(
            surv_data, sample_set, event=event, surv_time=surv_time, verbose=verbose
        )
        if "pval" in surv_results[bic[0]].keys():
            if np.isnan(surv_results[bic[0]]["pval"]):
                print(
                    "failed to fit CPH model for %s ~ bicluster %s" % (event, bic[0]),
                    file=sys.stderr,
                )

    surv_results = pd.DataFrame.from_dict(surv_results).T
    surv_results.columns = [event + "." + x for x in surv_results.columns]

    pvals = surv_results.loc[
        ~surv_results[event + ".p_value"].isna(), event + ".p_value"
    ].values
    bh_res, adj_pval = fdrcorrection(pvals, alpha=0.05)
    surv_results.loc[
        ~surv_results[event + ".p_value"].isna(), event + ".p_value_BH"
    ] = adj_pval

    pvals = surv_results.loc[
        ~surv_results[event + ".LogR_p_value"].isna(), event + ".LogR_p_value"
    ].values
    bh_res, adj_pval = fdrcorrection(pvals, alpha=0.05)
    surv_results.loc[
        ~surv_results[event + ".LogR_p_value"].isna(), event + ".LogR_p_value_BH"
    ] = adj_pval

    return pd.concat([biclusters, surv_results], axis=1)


def test_sample_overlap(row, sample_set, N):
    # usage:
    # biclusters_df.apply(lambda row: test_sample_overlap(row, sample_set, N),axis=1)
    # N - total number of samples in dataset
    bic_samples = row["samples"]
    o = len(sample_set.intersection(bic_samples))
    bic_only = len(bic_samples) - o
    sample_set_only = len(sample_set) - o
    bg = N - o - bic_only - sample_set_only
    p = pvalue(o, bic_only, sample_set_only, bg).right_tail
    # if p<0.001:
    #    print(p,(o,bic_only,sample_set_only,bg),row["genes"])
    return pd.Series({"pval": p, "counts": (o, bic_only, sample_set_only, bg)})


def add_sex(biclusters, males=[], females=[]):
    sample_sets = {}
    # if len(males)>0:
    sample_sets["male"] = set(males)
    # if len(females)>0:
    sample_sets["female"] = set(females)

    N = len(males) + len(females)
    dfs = []
    for sex in sample_sets.keys():
        sample_set = sample_sets[sex]
        df = biclusters.apply(
            lambda row: test_sample_overlap(row, sample_set, N), axis=1
        )
        df.columns = [sex + "." + x for x in df.columns]
        bh_res, adj_pval = fdrcorrection(df[sex + ".pval"].values, alpha=0.05)
        df[sex + ".pval_BH"] = adj_pval
        dfs.append(df)
    dfs = pd.concat(dfs, axis=1)
    dfs["sex.pval_BH"] = dfs.loc[:, ["male.pval_BH", "female.pval_BH"]].min(axis=1)
    dfs["sex"] = ""
    try:
        dfs.loc[dfs["male.pval_BH"] < 0.05, "sex"] = "male"
    except:
        pass
    try:
        dfs.loc[dfs["female.pval_BH"] < 0.05, "sex"] = "female"
    except:
        pass
    return pd.concat([biclusters, dfs], axis=1)
