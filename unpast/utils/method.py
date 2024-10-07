import sys
import os
import subprocess
import random
import warnings
import pandas as pd
import numpy as np
from time import time
import math

from scipy.interpolate import interp1d
from scipy.sparse.csr import csr_matrix
from scipy.stats import chi2_contingency

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from fisher import pvalue

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection

# optimizer
TRY_USE_NUMBA = True


def jit_if_available(func):
    # default "do nothing" decorator with the numba-like interface
    def decorated(*args, **kwargs):
        return func(*args, **kwargs)

    return decorated


if TRY_USE_NUMBA:
    try:
        from numba import jit  # as jit_if_available

        jit_if_available = jit()
    except:
        print("Numba is not available. Install numba for a bit faster calculations")


def zscore(df):
    m = df.mean(axis=1)
    df = df.T - m
    df = df.T
    s = df.std(axis=1)
    df = df.T / s
    df = df.T
    # set to 0 not variable genes
    zero_var_genes = s[s == 0].index.values
    if len(zero_var_genes) > 0:
        print(
            len(zero_var_genes),
            "zero variance rows detected, assign zero z-scores ",
            file=sys.stderr,
        )
    df.loc[zero_var_genes, :] = 0
    return df


def prepare_input_matrix(
    input_matrix: pd.DataFrame,
    min_n_samples: int =5,
    tol: float =0.01,
    standradize: bool = True,
    ceiling: float =0,  # if float>0, limit z-scores to [-x,x]
    verbose: bool =False,
):
    exprs = input_matrix.copy()
    exprs.index = [str(x) for x in exprs.index.values]
    exprs.columns = [str(x) for x in exprs.columns.values]
    m = exprs.mean(axis=1)
    std = exprs.std(axis=1)
    # find zero variance rows
    zero_var = list(std[std == 0].index.values)
    if len(zero_var) > 0:
        if verbose:
            print("\tZero variance rows will be dropped: %s"%len(zero_var),
                file=sys.stdout,
            )
        exprs = exprs.loc[std > 0]
        m = m[std > 0]
        std = std[std > 0]
        if exprs.shape[0]<=2:
            print("After excluding constant features (rows) , less than 3 features (rows) remain in the input matrix."% exprs.shape[0], file=sys.stderr)

    mean_passed = np.all(np.abs(m) < tol)
    std_passed = np.all(np.abs(std - 1) < tol)
    if not (mean_passed and std_passed):
        if verbose:
            print("\tInput is not standardized.", file=sys.stdout)
        if standradize:
            exprs = zscore(exprs)
            if not mean_passed:
                if verbose:
                    print("\tCentering mean to 0", file=sys.stdout)
            if not std_passed:
                if verbose:
                    print("\tScaling std to 1", file=sys.stdout)
    if len(set(exprs.index.values)) < exprs.shape[0]:
        print("\tRow names are not unique.", file=sys.stderr)
    missing_values = exprs.isna().sum(axis=1)
    n_na = missing_values[missing_values > 0].shape[0]
    if n_na > 0:
        if verbose:
            print(
                "\tMissing values detected in %s rows"%missing_values[missing_values > 0].shape[0],
                file=sys.stdout,
            )
        keep_features = missing_values[
            missing_values <= exprs.shape[1] - min_n_samples
        ].index.values
        if verbose:
            print(
                "\tFeatures with too few values (<%s) dropped: %s"
                % (min_n_samples, exprs.shape[0] - len(keep_features)),
                file=sys.stdout,
            )
        exprs = exprs.loc[keep_features, :]

    if standradize:
        if ceiling>0:
            if verbose:
                print(
                    "\tStandardized expressions will be limited to [-%s,%s]:"
                    % (ceiling, ceiling),
                    file=sys.stdout,
                )
            exprs[exprs > ceiling] = ceiling
            exprs[exprs < -ceiling] = -ceiling
            if n_na > 0:
                exprs.fillna(-ceiling, inplace=True)
                if verbose:
                    print(
                        "\tMissing values will be replaced with -%s."
                        % ceiling,
                        file=sys.stdout,
                    )
    return exprs


def calc_snr_per_row(s, N, exprs, exprs_sums, exprs_sq_sums):
    bic = exprs[:, :s]
    bic_sums = bic.sum(axis=1)
    bic_sq_sums = np.square(bic).sum(axis=1)

    bg_counts = N - s
    bg_sums = exprs_sums - bic_sums
    bg_sq_sums = exprs_sq_sums - bic_sq_sums

    bic_mean, bic_std = calc_mean_std_by_powers((s, bic_sums, bic_sq_sums))
    bg_mean, bg_std = calc_mean_std_by_powers((bg_counts, bg_sums, bg_sq_sums))

    snr_dist = (bic_mean - bg_mean) / (bic_std + bg_std)

    return snr_dist


def calc_mean_std_by_powers(powers):
    count, val_sum, sum_sq = powers

    mean = val_sum / count  # what if count == 0?
    std = np.sqrt((sum_sq / count) - mean * mean)
    return mean, std


def calc_SNR(ar1, ar2, pd_mode=False):
    """Calculate Signal-to-Noise Ratio (SNR) for two arrays.

    Args:
        ar1 (array): first array
        ar2 (array): second array
        pd_mode (bool): if True, use pandas-like mean/std methods
            i.e. n-1 for std, ignore nans

    Returns:
        float: SNR value
    """

    if pd_mode:
        std = lambda x: np.nanstd(x, ddof=1.0)
        mean = np.nanmean
    else:
        std = np.nanstd
        mean = np.mean

    mean_diff = mean(ar1) - mean(ar2)
    std_sum = std(ar1) + std(ar2)

    if std_sum == 0:
        return np.inf * mean_diff

    return mean_diff / std_sum


######### Binarization #########
def generate_null_dist(
    N, sizes, n_permutations=10000, pval=0.001, seed=42, verbose=True
):
    # samples 'N' values from standard normal distribution, and split them into bicluster and background groups
    # 'sizes' defines bicluster sizes to test
    # returns a dataframe with the distribution of SNR for each bicluster size (sizes x n_permutations )
    t0 = time()

    if verbose:
        print(
            "\tGenerate background distribuition of SNR depending on the bicluster size ...",
            file=sys.stdout,
        )
        print(
            "\t\ttotal samples: %s,\n\t\tnumber of samples in a bicluster: %s - %s,\n\t\tn_permutations: %s"
            % (N, min(sizes), max(sizes), n_permutations),
            file=sys.stdout,
        )
        print("\t\tsnr pval threshold:", pval, file=sys.stdout)

    exprs = np.zeros((n_permutations, N))  # generate random expressions from st.normal
    # values = exprs.values.reshape(-1) # random samples from expression matrix
    # exprs = np.random.choice(values,size=exprs.shape[1])
    np.random.seed(seed=seed)
    for i in range(n_permutations):
        exprs[i,] = sorted(np.random.normal(size=N))

    exprs_sums = exprs.sum(axis=1)
    exprs_sq_sums = np.square(exprs).sum(axis=1)

    null_distribution = pd.DataFrame(
        np.zeros((sizes.shape[0], n_permutations)),
        index=sizes,
        columns=range(n_permutations),
    )

    for s in sizes:
        null_distribution.loc[s, :] = -1 * calc_snr_per_row(
            s, N, exprs, exprs_sums, exprs_sq_sums
        )

    if verbose:
        print(
            "\tBackground ditribution generated in {:.2f} s".format(time() - t0),
            file=sys.stdout,
        )
    return null_distribution


def get_trend(sizes, thresholds, plot=True, verbose=True):
    """
    Smoothens the trend and retunrs a function min_SNR(size; p-val. cutoff)
        Given a set of points (x_i, y_i), 
        returns a function f(x) that interpolates the data with LOWESS+linear interpolation

    Args:
        sizes: values of x_i
        thresholds: values of y_i
        plot: if True, plots the trend
        verbose: if True, prints the LOWESS frac
    
    Returns:
        get_min_snr: a function that returns the minimal SNR for a given size
    """
    assert len(sizes) >= 0
    if len(sizes) == 1: 
        return lambda x: thresholds[0]
    
    lowess = sm.nonparametric.lowess
    frac = max(1, min(math.floor(int(0.1 * len(sizes))), 15) / len(sizes))
    # if verbose:
    #    print("\t\t\tLOWESS frac=",round(frac,2), file = sys.stdout)
    lowess_curve = lowess(
        thresholds, sizes, frac=frac, return_sorted=True, is_sorted=False
    )
    get_min_snr = interp1d(
        lowess_curve[:, 0], lowess_curve[:, 1]
    )  # ,kind="nearest-up",fill_value="extrapolate")
    if plot:
        plt.plot(sizes, thresholds, "b--", lw=2)
        plt.plot(sizes, get_min_snr(sizes), "r-", lw=2)
        plt.xlabel("n_samples")
        plt.ylabel("SNR")
        plt.ylim((0, 5))
        plt.show()
    return get_min_snr


def calc_e_pval(snr, size, null_distribution):
    e_dist = null_distribution.loc[int(size), :]
    return (len(e_dist[e_dist >= abs(snr)]) + 1.0) / (null_distribution.shape[1] + 1.0)


def plot_binarized_feature(feature_name, down_group, up_group, colors, hist_range, snr):
    down_color, up_color = colors
    n_bins = int(max(20, (len(down_group) + len(up_group)) / 10))
    n_bins = min(n_bins, 200)
    fig, ax = plt.subplots()
    tmp = ax.hist(
        down_group, bins=n_bins, alpha=0.5, color=down_color, range=hist_range
    )
    tmp = ax.hist(up_group, bins=n_bins, alpha=0.5, color=up_color, range=hist_range)
    # tmp = plt.title("{}:    SNR={:.2f},    neg={}, pos={}".format(feature_name,snr,len(down_group),len(up_group)))
    n_samples = min(len(down_group), len(up_group))
    # tmp = ax.set_title("SNR={:.2f},   n_samples={}".format(snr,n_samples))
    ax.text(
        0.05,
        0.95,
        feature_name,
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=24,
    )
    ax.text(
        0.95,
        0.95,
        "SNR=" + str(round(snr, 2)) + "\nn_samples=" + str(n_samples),
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=14,
    )
    # tmp = plt.savefig("figs_binarization/"+feature_name+".hist.svg", bbox_inches='tight', transparent=True)
    plt.show()


def select_pos_neg(row, min_n_samples, seed=42, prob_cutoff=0.5, method="GMM"):
    """ find 'higher' (positive), and 'lower' (negative) signal in vals. 
        vals are found with GM binarization
    """
    is_converged = None
    if method == "GMM":
        with warnings.catch_warnings():  # this is to ignore convergence warnings
            warnings.simplefilter("ignore")
            row2d = row[:, np.newaxis]  # adding mock axis for GM interface
            # np.random.seed(seed=seed)
            model = GaussianMixture(
                n_components=2,
                init_params="kmeans",
                max_iter=len(row),
                n_init=1,
                covariance_type="spherical",
                random_state=seed,
            ).fit(row2d)
            is_convergend = model.converged_
            p0 = model.predict_proba(row2d)[:, 0]
            labels = np.zeros(len(row), dtype=bool)

            # let labels == True be always a smaller sample set
            if len(labels[p0 >= prob_cutoff]) >= len(labels[p0 < 1 - prob_cutoff]):
                labels[p0 < 1 - prob_cutoff] = True
            else:
                labels[p0 >= prob_cutoff] = True

    elif method in ["kmeans", "ward"]:
        row2d = row[:, np.newaxis]  # adding mock axis
        if method == "kmeans":
            model = KMeans(n_clusters=2, max_iter=len(row), n_init=1, random_state=seed)
        elif method == "ward":
            model = AgglomerativeClustering(n_clusters=2, linkage="ward")
        # elif method == "HC_ward":
        #    model = Ward(n_clusters=2)
        labels = np.zeros(len(row), dtype=bool)
        pred_labels = model.fit_predict(row2d)
        # let labels == True be always a smaller sample set
        if len(pred_labels[pred_labels == 1]) >= len(pred_labels[pred_labels == 0]):
            labels[pred_labels == 0] = True
        else:
            labels[pred_labels == 1] = True
    else:
        print(
            "wrong method name",
            method,
            "must be ['GMM','kmeans','ward']",
            file=sys.stderr,
        )

    # special treatment for cases when bic distribution is too wide and overlaps bg distribution
    # remove from bicluster samples with the sign different from its median sign
    if len(row[labels]) > 0:
        if np.median(row[labels]) >= 0:
            labels[row < 0] = False
        else:
            labels[row > 0] = False

    n0 = len(labels[labels])
    n1 = len(labels[~labels])

    assert n0 + n1 == len(row)

    snr = np.nan
    e_pval = np.nan
    size = np.nan
    mask_pos = np.zeros(len(row), dtype=bool)
    mask_neg = np.zeros(len(row), dtype=bool)

    if n0 >= min_n_samples:
        snr = calc_SNR(row[labels], row[~labels])
        size = n0

        if snr > 0:
            mask_pos = labels
            mask_neg = ~labels
        else:
            mask_neg = labels
            mask_pos = ~labels

    return mask_pos, mask_neg, abs(snr), size, is_converged


def sklearn_binarization(
    exprs,
    min_n_samples,
    verbose=True,
    plot=True,
    plot_SNR_thr=2,
    show_fits=[],
    seed=1,
    prob_cutoff=0.5,
    method="GMM",
):
    t0 = time()

    binarized_expressions = {}

    stats = {}
    for i, (gene, row) in enumerate(exprs.iterrows()):
        e_pval = -1
        row = row.values
        pos_mask, neg_mask, snr, size, is_converged = select_pos_neg(
            row, min_n_samples, seed=seed, prob_cutoff=prob_cutoff, method=method
        )

        # logging
        if verbose:
            if i % 1000 == 0:
                print("\t\tgenes processed:", i)

        up_group = row[pos_mask]
        down_group = row[neg_mask]
        n_up = len(up_group)
        n_down = len(down_group)

        # if smaller sample group shows over- or under-expression
        if n_up <= n_down:  # up-regulated group is bicluster
            binarized_expressions[gene] = pos_mask.astype(int)
            direction = "UP"
        else:  # down-regulated group is bicluster
            binarized_expressions[gene] = neg_mask.astype(int)
            direction = "DOWN"

        stats[gene] = {
            "pval": 0,
            "SNR": snr,
            "size": size,
            "direction": direction,
            "convergence": is_converged,
        }

        if gene in show_fits or (abs(snr) > plot_SNR_thr and plot):
            hist_range = row.min(), row.max()

            # set colors to two sample groups
            # red - overexpression
            # blue - under-expression
            # grey - background (group size > 1/2 of all samples)
            colors = ["grey", "grey"]

            if n_down - n_up >= 0:  # up-regulated group is bicluster
                colors[1] = "red"

            if n_up - n_down > 0:  # down-regulated group is bicluster
                colors[0] = "blue"

            # in case of insignificant size difference
            # between up- and down-regulated groups
            # the bigger half is treated as signal too
            if abs(n_up - n_down) <= min_n_samples:
                colors = "blue", "red"

            # plotting
            plot_binarized_feature(gene, down_group, up_group, colors, hist_range, snr)

    stats = pd.DataFrame.from_dict(stats).T

    binarized_expressions = pd.DataFrame.from_dict(binarized_expressions)

    # logging
    if verbose:
        print(
            "\tBinarization for {} features completed in {:.2f} s".format(
                len(exprs), time() - t0
            )
        )

    return binarized_expressions, stats


def binarize(
    binarized_fname_prefix,
    exprs=None,
    method="GMM",
    save=True,
    load=False,
    min_n_samples=5,
    pval=0.001,
    plot_all=True,
    plot_SNR_thr=np.inf,
    show_fits=[],
    verbose=True,
    seed=random.randint(0, 100000),
    prob_cutoff=0.5,
    n_permutations=10000,
):
    """
       binarized_fname_prefix is a basename of binarized data file;
       exprs is a dataframe with normalized features to be binarized.
    """
    t0 = time()

    # a file with binarized gene expressions
    bin_exprs_fname = (
        binarized_fname_prefix
        + ".seed="
        + str(seed)
        + ".bin_method="
        + method
        + ".min_ns="
        + str(min_n_samples)
        + ".binarized.tsv"
    )
    # a file with statistics of binarization results
    bin_stats_fname = (
        binarized_fname_prefix
        + ".seed="
        + str(seed)
        + ".bin_method="
        + method
        + ".min_ns="
        + str(min_n_samples)
        + ".binarization_stats.tsv"
    )
    # a file with background SNR distributions for each biclsuter size
    n_permutations = max(n_permutations, int(1.0 / pval * 10))
    bin_bg_fname = (
        binarized_fname_prefix
        + ".seed="
        + str(seed)
        + ".n="
        + str(n_permutations)
        + ".min_ns="
        + str(min_n_samples)
        + ".background.tsv"
    )

    if load:
        load_failed = False
        try:
            if verbose:
                print(
                    "Load binarized features from",
                    bin_exprs_fname,
                    "\n",
                    file=sys.stdout,
                )
            # load binary expressions
            binarized_data = pd.read_csv(bin_exprs_fname, sep="\t", index_col=0)
        except:
            print(
                "file " + bin_exprs_fname + " is not found and will be created",
                file=sys.stderr,
            )
            load_failed = True
        try:
            # load stats
            stats = pd.read_csv(bin_stats_fname, sep="\t", index_col=0)
            if verbose:
                print("Load statistics from", bin_stats_fname, "\n", file=sys.stdout)
        except:
            print(
                "file " + bin_stats_fname + " is not found and will be created",
                file=sys.stderr,
            )
            load_failed = True

    if not load or load_failed:
        if exprs is None:
            print("Provide either raw or binarized data.", file=sys.stderr)
            return None

        # binarize features
        start_time = time()
        if verbose:
            print("\nBinarization started ....\n")

        t0 = time()

        if method in ["GMM", "kmeans", "ward"]:
            binarized_data, stats = sklearn_binarization(
                exprs,
                min_n_samples,
                plot=plot_all,
                plot_SNR_thr=plot_SNR_thr,
                prob_cutoff=prob_cutoff,
                show_fits=show_fits,
                verbose=verbose,
                seed=seed,
                method=method,
            )
        else:
            print("Method must be 'GMM','kmeans', or 'ward'.", file=sys.stderr)
            return

    # load or generate empirical distributions for all bicluster sizes
    N = exprs.shape[1]
    # sizes of binarized features
    sizes1 = set([x for x in stats["size"].values if not np.isnan(x)])
    # no more than 100 of bicluster sizes are computed
    # step = max(int((N - min_n_samples) / 100), 1) 
    step = max(int((int(N / 2) - min_n_samples) / 100), 1) 
    #sizes2 = set(map(int, np.arange(min_n_samples, int(N / 2), step)))
    sizes2 = set(map(int, np.arange(min_n_samples, int(N / 2)+1, step)))
    sizes = np.array(sorted(sizes1 | sizes2))

    load_failed = False
    if load:
        try:
            # load background distribution
            null_distribution = pd.read_csv(bin_bg_fname, sep="\t", index_col=0)
            null_distribution.columns = [
                int(x) for x in null_distribution.columns.values
            ]
            if verbose:
                print(
                    "Loaded background distribution from",
                    bin_bg_fname,
                    "\n",
                    file=sys.stdout,
                )
            # check if any new sizes need to be precomputed
            precomputed_sizes = null_distribution.index.values
            add_sizes = np.array(sorted(set(sizes).difference(set(precomputed_sizes))))
            if len(add_sizes) > 0:
                null_distribution2 = generate_null_dist(
                    N,
                    add_sizes,
                    pval=pval,
                    n_permutations=n_permutations,
                    seed=seed,
                    verbose=verbose,
                )
                null_distribution2.columns = [
                    int(x) for x in null_distribution2.columns.values
                ]
                null_distribution = pd.concat(
                    [
                        null_distribution,
                        null_distribution2.loc[:, null_distribution.columns.values],
                    ],
                    axis=0,
                )
                if save:
                    null_distribution.loc[
                        sorted(null_distribution.index.values), :
                    ].to_csv(bin_bg_fname, sep="\t")
                    if verbose:
                        print(
                            "Background ditribution in %s is updated" % bin_bg_fname,
                            file=sys.stdout,
                        )
                null_distribution = null_distribution.loc[sizes, :]
        except:
            print(
                "file " + bin_bg_fname + " is not found and will be created",
                file=sys.stderr,
            )
            load_failed = True
    if not load or load_failed:
        null_distribution = generate_null_dist(
            N,
            sizes,
            pval=pval,
            n_permutations=n_permutations,
            seed=seed,
            verbose=verbose,
        )

    # if not load or load_failed:
    # add SNR p-val depends on bicluster size
    stats = stats.dropna(subset=["size"])
    stats["pval"] = stats.apply(
        lambda row: calc_e_pval(row["SNR"], row["size"], null_distribution), axis=1
    )
    accepted, pval_adj = fdrcorrection(stats["pval"])
    stats["pval_BH"] = pval_adj

    # find SNR threshold
    thresholds = np.quantile(null_distribution.loc[sizes, :].values, q=1 - pval, axis=1)
    size_snr_trend = get_trend(sizes, thresholds, plot=False, verbose=verbose)
    stats["SNR_threshold"] = stats["size"].apply(lambda x: size_snr_trend(x))

    if save:
        # save binarized data
        fpath = "/".join(bin_exprs_fname.split("/")[:-1])
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        if not os.path.exists(bin_exprs_fname):
            binarized_data.to_csv(bin_exprs_fname, sep="\t")
            if verbose:
                print(
                    "Binarized gene expressions are saved to",
                    bin_exprs_fname,
                    file=sys.stdout,
                )

        # save binarization statistics
        if not os.path.exists(bin_stats_fname):
            stats.to_csv(bin_stats_fname, sep="\t")
            if verbose:
                print("Statistics is saved to", bin_stats_fname, file=sys.stdout)

        # save null distribution: null_distribution, size,threshold
        if not os.path.exists(bin_bg_fname):
            null_distribution.to_csv(bin_bg_fname, sep="\t")
            if verbose:
                print(
                    "Background sitribution is saved to", bin_bg_fname, file=sys.stdout
                )

    ### keep features passed binarization
    passed = stats.loc[stats["SNR"] > stats["SNR_threshold"], :]
    # passed = stats.loc[stats["pval_BH"]<pval,:]

    if verbose:
        print(
            "\t\tUP-regulated features:\t%s"
            % (passed.loc[passed["direction"] == "UP", :].shape[0]),
            file=sys.stdout,
        )
        print(
            "\t\tDOWN-regulated features:\t%s"
            % (passed.loc[passed["direction"] == "DOWN", :].shape[0]),
            file=sys.stdout,
        )
        # print("\t\tambiguous features:\t%s"%(passed.loc[passed["direction"]=="UP,DOWN",:].shape[0]),file = sys.stdout)

    # keep only binarized features
    binarized_data = binarized_data.loc[:, list(passed.index.values)]
    # add sample names
    binarized_data.index = exprs.columns.values

    if plot_all:
        fig, ax = plt.subplots(figsize=(10, 4.5))

        # plot binarization results for real genes
        passed = stats.loc[stats["SNR"] > stats["SNR_threshold"], :]
        failed = stats.loc[stats["SNR"] <= stats["SNR_threshold"], :]
        tmp = ax.scatter(
            failed["size"], failed["SNR"], alpha=1, color="black", label="not passed"
        )
        for i, txt in enumerate(failed["size"].index.values):
            ax.annotate(txt, (failed["size"][i], failed["SNR"][i] + 0.1), fontsize=18)
        tmp = ax.scatter(
            passed["size"], passed["SNR"], alpha=1, color="red", label="passed"
        )
        for i, txt in enumerate(passed["size"].index.values):
            ax.annotate(txt, (passed["size"][i], passed["SNR"][i] + 0.1), fontsize=18)

        # plot cutoff
        tmp = ax.plot(
            sizes,
            [size_snr_trend(x) for x in sizes],
            color="darkred",
            lw=2,
            ls="--",
            label="e.pval<" + str(pval),
        )
        plt.gca().legend(("not passed", "passed", "e.pval<" + str(pval)), fontsize=18)
        tmp = ax.set_xlabel("n_samples", fontsize=18)
        tmp = ax.yaxis.tick_right()
        tmp = ax.set_ylabel("SNR", fontsize=18)
        tmp = ax.set_ylim((0, 4))
        # tmp = plt.savefig("figs_binarization/dist.svg", bbox_inches='tight', transparent=True)
        tmp = plt.show()

    return binarized_data, stats, null_distribution


#### Cluster binarized genes #####


def run_WGCNA_iterative(
    binarized_expressions,
    tmp_prefix="",
    deepSplit=0,
    detectCutHeight=0.995,
    nt="signed_hybrid",  # see WGCNA documentation
    max_power=10,
    precluster=False,
    verbose=False,
    rscr_path=False,
    rpath="",
):

    t0 = time()

    not_clustered = binarized_expressions.columns.values
    binarized_expressions_ = binarized_expressions.loc[:, :].copy()
    stop_condition = False

    modules = []
    i = 0
    while len(not_clustered) >= 3 and not stop_condition:
        binarized_expressions_ = binarized_expressions_.loc[:, not_clustered]

        m, not_clustered = run_WGCNA(
            binarized_expressions_,
            tmp_prefix=tmp_prefix,
            deepSplit=deepSplit,
            detectCutHeight=detectCutHeight,
            nt=nt,
            max_power=max_power,
            precluster=precluster,
            verbose=verbose,
            rscr_path=rscr_path,
            rpath=rpath,
        )
        if verbose:
            print(
                "\t\t\tWGCNA iteration %s, modules:%s, not clustered:%s"
                % (i, len(m), len(not_clustered)),
                file=sys.stdout,
            )
        modules += m
        # stop when the number of not clustred samples does not change
        if len(m) == 0:
            stop_condition = True
            if verbose:
                print("\t\t\tWGCNA iterations terminated at step ", i, file=sys.stdout)

        i += 1
    return (modules, not_clustered)


def run_WGCNA(
    binarized_expressions,
    tmp_prefix="",
    deepSplit=0,
    detectCutHeight=0.995,
    nt="signed_hybrid",  # see WGCNA documentation
    max_power=10,
    precluster=False,
    verbose=False,
    rscr_path=False,
    rpath="",
):
    t0 = time()
    # create unique suffix for tmp files
    from datetime import datetime

    now = datetime.now()
    fname = "tmpWGCNA_" + now.strftime("%y.%m.%d_%H:%M:%S") + ".tsv"
    if len(tmp_prefix) > 0:
        fname = tmp_prefix + "." + fname

    if verbose:
        print("\t\tWGCNA pre-clustering:", precluster, file=sys.stdout)
    if precluster:
        precluster = "T"
    else:
        precluster = "F"

    deepSplit = int(deepSplit)
    if not deepSplit in [0, 1, 2, 3, 4]:
        print("deepSplit must be 1,2,3 or 4. See WGCNA documentation.", file=sys.stderr)
        return ([], [])
    if not 0 < detectCutHeight < 1:
        print(
            "detectCutHeight must be between 0 and 1. See WGCNA documentation.",
            file=sys.stderr,
        )
        return ([], [])
    if verbose:
        print("\tRunning WGCNA for", fname, "...", file=sys.stdout)
    if not rscr_path:
        # assume run_WGCNA.R is in the same folder
        rscr_path = (
            "/".join(os.path.realpath(__file__).split("/")[:-1]) + "/run_WGCNA.R"
        )

    binarized_expressions_ = binarized_expressions.loc[:, :].copy()

    # add suffixes to duplicated feature names
    feature_names = binarized_expressions.columns.values
    duplicated_feature_ndxs = np.arange(binarized_expressions.shape[1])[
        binarized_expressions.columns.duplicated()
    ]

    if len(duplicated_feature_ndxs) > 0:
        new_feature_names = []
        for i in range(len(feature_names)):
            fn = feature_names[i]
            if i in duplicated_feature_ndxs:
                fn = str(fn) + "*" + str(i)
            new_feature_names.append(fn)
        print(
            "\t\t%s duplicated feature names detected." % len(duplicated_feature_ndxs),
            file=sys.stdout,
        )
        dup_fn_mapping = dict(zip(new_feature_names, feature_names))
        binarized_expressions_.columns = new_feature_names

    # replace spaces in feature names
    # otherwise won't parse R output
    feature_names = binarized_expressions.columns.values
    feature_names_with_space = [x for x in feature_names if " " in x]
    if len(feature_names_with_space) > 0:
        if verbose:
            print(
                "\t\tfeature names containing spaces (will be replaced):%s"
                % len(feature_names_with_space),
                file=sys.stdout,
            )
        fn_mapping = {}
        fn_mapping_back = {}
        for fn in feature_names:
            if " " in fn:
                fn_ = fn.replace(" ", "_")
                fn_mapping[fn] = fn_
                fn_mapping_back[fn_] = fn
        binarized_expressions_ = binarized_expressions.rename(
            fn_mapping, axis="columns"
        )

    # save binarized expression to a file
    binarized_expressions_.to_csv(fname, sep="\t")

    # run Rscript
    if len(rpath) > 0:
        rpath = rpath + "/"

    if verbose:
        print("\tR command line:", file=sys.stdout)
        print(
            "\t"
            + " ".join(
                [
                    rpath + "Rscript",
                    rscr_path,
                    fname,
                    str(deepSplit),
                    str(detectCutHeight),
                    nt,
                    str(max_power),
                    precluster,
                ]
            ),
            file=sys.stdout,
        )

    process = subprocess.Popen(
        [
            rpath + "Rscript",
            rscr_path,
            fname,
            str(deepSplit),
            str(detectCutHeight),
            nt,
            str(max_power),
            precluster,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    # stdout = stdout.decode('utf-8')
    module_file = fname.replace(".tsv", ".modules.tsv")  # stdout.rstrip()
    try:
        modules_df = pd.read_csv(module_file, sep="\t", index_col=0)
    except:
        # print("WGCNA output:", stdout, file = sys.stdout)
        stderr = stderr.decode("utf-8")
        if verbose:
            print("WGCNA error:", stderr, file=sys.stdout)
        modules_df = pd.DataFrame.from_dict({})
    if verbose:
        print(
            "\tWGCNA runtime: modules detected in {:.2f} s.".format(time() - t0),
            file=sys.stdout,
        )

    # read WGCNA output
    modules = []
    not_clustered = []
    module_dict = modules_df.T.to_dict()
    for i in module_dict.keys():
        genes = module_dict[i]["genes"].strip().split()
        # change feature names if they were modified

        # return spaces in feature names back if necessary
        if len(feature_names_with_space) > 0:
            for j in range(len(genes)):
                if genes[j] in fn_mapping_back.keys():
                    genes[j] = fn_mapping_back[genes[j]]
        # remove suffixes from duplicated feature names
        if len(duplicated_feature_ndxs) > 0:
            for j in range(len(genes)):
                if genes[j] in dup_fn_mapping.keys():
                    genes[j] = dup_fn_mapping[genes[j]]

        if i == 0:
            not_clustered = genes
        else:
            modules.append(genes)

    # remove WGCNA input and output files
    try:
        os.remove(fname)
        os.remove(module_file)
    except:
        pass

    if verbose:
        print(
            "\tmodules: {}, not clustered features {} ".format(
                len(modules), len(not_clustered)
            ),
            file=sys.stdout,
        )
        # print(stdout,file = sys.stdout)
        if len(stderr) > 0:
            print(stderr, file=sys.stderr)

    return (modules, not_clustered)


def run_Louvain(
    similarity,
    similarity_cutoffs=np.arange(0.33, 0.95, 0.05),
    m=False,
    verbose=True,
    plot=False,
    modularity_measure="newman",
):
    t0 = time()
    if similarity.shape[0] == 0:
        print("no features to cluster", file=sys.stderr)
        return [], [], None

    if verbose:
        print("\tRunning Louvain ...", file=sys.stdout)
        print("\t\tmodularity:", modularity_measure, file=sys.stdout)

    from sknetwork.clustering import Louvain, modularity

    modularities = []
    feature_clusters = {}
    best_Q = np.nan
    for cutoff in similarity_cutoffs:
        # scan the whole range of similarity cutoffs
        # e.g. [1/4;9/10] with step 0.5
        sim_binary = similarity.copy()
        sim_binary[sim_binary < cutoff] = 0
        sim_binary[sim_binary != 0] = 1
        rsums = sim_binary.sum()
        non_zero_features = rsums[rsums > 0].index
        sim_binary = sim_binary.loc[non_zero_features, non_zero_features]
        gene_names = sim_binary.index.values
        sparse_matrix = csr_matrix(sim_binary)
        labels = Louvain(modularity=modularity_measure).fit_transform(sparse_matrix)
        Q = modularity(sparse_matrix, labels)
        modularities.append(Q)
        # if binary similarity matrix contains no zeroes
        # bugfix for Louvain()
        if sim_binary.min().min() == 1:
            labels = np.zeros(len(labels))
        feature_clusters[cutoff] = labels

    # if similarity_cutoffs contains only one value, choose it as best_cutoff
    if len(similarity_cutoffs) == 1:
        best_cutoff = similarity_cutoffs[0]
        best_Q = Q

    # find best_cutoff automatically
    else:
        # check if modularity(cutoff)=const
        if len(set(modularities)) == 1:
            best_cutoff = similarity_cutoffs[-1]
            best_Q = modularities[-1]
            labels = feature_clusters[best_cutoff]

        #  if modularity!= const, scan the whole range of similarity cutoffs
        #  e.g. [1/4;9/10] with step 0.05
        else:
            # find the knee point in the dependency modularity(similarity curoff)
            from kneed import KneeLocator

            # define the type of the curve
            curve_type = "increasing"
            if modularities[0] >= modularities[-1]:
                curve_type = "decreasing"
            if verbose:
                print("\tcurve type:", curve_type, file=sys.stdout)
            # detect knee and choose the one with the highest modularity
            try:
                kn = KneeLocator(
                    similarity_cutoffs,
                    modularities,
                    curve="concave",
                    direction=curve_type,
                    online=True,
                )
                best_cutoff = kn.knee
                best_Q = kn.knee_y
                labels = feature_clusters[best_cutoff]
            except:
                print("Failed to identify similarity cutoff", file=sys.stderr)
                print(
                    "Similarity cutoff: set to ", similarity_cutoffs[0], file=sys.stdout
                )
                best_cutoff = similarity_cutoffs[0]
                best_Q = np.nan
                print("Modularity:", modularities, file=sys.stdout)
                if plot:
                    plt.plot(similarity_cutoffs, modularities, "bx-")
                    plt.xlabel("similarity cutoff")
                    plt.ylabel("modularity")
                    plt.show()
                    # return [], [], None
            if m:
                # if upper threshold for modularity m is specified
                # chose the lowest similarity cutoff at which modularity reaches >= m
                best_cutoff_m = 1
                for i in range(len(modularities)):
                    if modularities[i] >= m:
                        best_cutoff_m = similarity_cutoffs[i]
                        best_Q_m = modularities[i]
                        labels_m = feature_clusters[best_cutoff]
                        break
                if best_cutoff_m < best_cutoff:
                    best_cutoff = best_cutoff_m
                    best_Q = best_Q_m
                    labels = labels_m

    if plot and len(similarity_cutoffs) > 1:
        plt.plot(similarity_cutoffs, modularities, "bx-")
        plt.vlines(
            best_cutoff, plt.ylim()[0], plt.ylim()[1], linestyles="dashed", color="red"
        )
        plt.xlabel("similarity cutoff")
        plt.ylabel("modularity")
        plt.show()

    modules = []
    not_clustered = []

    for label in set(labels):
        ndx = np.argwhere(labels == label).reshape(1, -1)[0]
        genes = gene_names[ndx]
        if len(genes) > 1:
            modules.append(genes)
        else:
            not_clustered.append(genes[0])
    if verbose:
        print(
            "\tLouvain runtime: modules detected in {:.2f} s.".format(time() - t0),
            file=sys.stdout,
        )
        print(
            "\tmodules: {}, not clustered features {} ".format(
                len(modules), len(not_clustered)
            ),
            file=sys.stdout,
        )
        print(
            "\t\tsimilarity cutoff: {:.2f} modularity: {:.3f}".format(
                best_cutoff, best_Q
            ),
            file=sys.stdout,
        )
    return modules, not_clustered, best_cutoff

def get_similarity_jaccard(binarized_data, verbose=True):  # ,J=0.5
    t0 = time()
    genes = binarized_data.columns.values
    n_samples = binarized_data.shape[0]
    size_threshold = int(min(0.45 * n_samples, (n_samples) / 2 - 10))
    # print("size threshold",size_threshold)
    n_genes = binarized_data.shape[1]
    df = np.array(binarized_data.T, dtype=bool)
    results = np.zeros((n_genes, n_genes))
    for i in range(0, n_genes):
        results[i, i] = 1
        g1 = df[i]

        for j in range(i + 1, n_genes):
            g2 = df[j]
            o = g1 * g2
            u = g1 + g2
            jaccard = o.sum() / u.sum()
            # try matching complements
            if g1.sum() > size_threshold:
                g1_complement = ~g1
                o = g1_complement * g2
                u = g1_complement + g2
                jaccard_c = o.sum() / u.sum()
            elif g2.sum() > size_threshold:
                g2 = ~g2
                o = g1 * g2
                u = g1 + g2
                jaccard_c = o.sum() / u.sum()
            else:
                jaccard_c = 0
            jaccard = max(jaccard, jaccard_c)
            results[i, j] = jaccard
            results[j, i] = jaccard

    results = pd.DataFrame(data=results, index=genes, columns=genes)
    if verbose:
        print(
            "\tJaccard similarities for {} features computed in {:.2f} s.".format(
                binarized_data.shape[1], time() - t0
            ),
            file=sys.stdout,
        )
    return results


def get_similarity_corr(df, verbose=True):
    t0 = time()
    corr = df.corr()  # .applymap(abs)
    corr = corr[corr > 0]  # to consider only direct correlations
    corr = corr.fillna(0)
    if verbose:
        print(
            "\tPearson's r similarities for {} features computed in {:.2f} s.".format(
                df.shape[1], time() - t0
            ),
            file=sys.stdout,
        )
    return corr

######## Make biclusters #########


def cluster_samples(data, min_n_samples=5, seed=0, method="kmeans"):
    # identify identify bicluster and backgound groups using 2-means
    max_n_iter = max(max(data.shape), 500)
    if method == "kmeans" or method == "Jenks":
        labels = (
            KMeans(
                n_clusters=2,
                random_state=seed,
                init="random",
                n_init=10,
                max_iter=max_n_iter,
            )
            .fit(data)
            .labels_
        )
    elif method == "ward":
        labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(data).labels_
    # elif method == "HC_ward":
    #        model = Ward(n_clusters=2).fit(data).labels_
    elif method == "GMM":
        labels = GaussianMixture(
            n_components=2,
            init_params="kmeans",
            max_iter=max_n_iter,
            n_init=5,
            covariance_type="spherical",
            random_state=seed,
        ).fit_predict(data)
    ndx0 = np.where(labels == 0)[0]
    ndx1 = np.where(labels == 1)[0]
    if min(len(ndx1), len(ndx0)) < min_n_samples:
        return {}
    if len(ndx1) > len(ndx0):
        samples = ndx0
    else:
        samples = ndx1

    bicluster = {"sample_indexes": set(samples), "n_samples": len(samples)}
    return bicluster


def modules2biclusters(
    modules,
    data_to_cluster,
    method="kmeans",
    min_n_samples=5,
    min_n_genes=2,
    seed=0,
    verbose=True,
):
    """Identifies optimal sample set for each module: 
    splits samples into two sets in a subspace of each module
    """
    t0 = time()
    biclusters = {}
    wrong_sample_number = 0
    low_SNR = 0
    i = 0

    for mid in range(0, len(modules)):
        genes = modules[mid]
        if len(genes) >= min_n_genes:
            # cluster samples in a space of selected genes
            data = data_to_cluster.loc[genes, :].T
            bicluster = cluster_samples(
                data, min_n_samples=min_n_samples, seed=seed, method=method
            )
            if len(bicluster) > 0:
                bicluster["id"] = i
                bicluster["genes"] = set(genes)
                bicluster["n_genes"] = len(bicluster["genes"])
                biclusters[i] = bicluster
                i += 1

    if verbose:
        print(
            "time:\tIdentified optimal sample sets for %s modules in %s s."
            % (len(modules), round(time() - t0, 2))
        )
        print(
            "Passed biclusters (>=%s genes, >= samples %s): %s"
            % (min_n_genes, min_n_samples, i - 1),
            file=sys.stdout,
        )
        print()

    return biclusters


def update_bicluster_data(bicluster, data):
    """ distinguishes up- and down-regulated gene
    adds "samples" and "gene_indexes"
    calculates average z-score
    bicluster must contain "sample_indexes" and "genes"
    data must contain all features, not just binarized"""

    # add "samples" and "gene_indexes"
    sample_names = data.columns.values
    gene_names = data.index.values

    bic_samples = sample_names[list(bicluster["sample_indexes"])]
    bic_genes = list(bicluster["genes"])
    bg_samples = [x for x in sample_names if not x in bic_samples]
    bicluster["samples"] = set(bic_samples)
    bicluster["gene_indexes"] = set(
        [np.where(gene_names == x)[0][0] for x in bicluster["genes"]]
    )

    # distinguish up- and down-regulated features
    m_bic = data.loc[bic_genes, bic_samples].mean(axis=1)
    m_bg = data.loc[bic_genes, bg_samples].mean(axis=1)
    genes_up = m_bic[m_bic >= m_bg].index.values
    genes_down = m_bic[m_bic < m_bg].index.values
    bicluster["genes_up"] = set(genes_up)
    bicluster["genes_down"] = set(genes_down)

    genes_up = m_bic[m_bic >= m_bg].index.values
    genes_down = m_bic[m_bic < m_bg].index.values

    # calculate average z-score for each sample
    if min(len(genes_up), len(genes_down)) > 0:  # take direction into account
        avg_zscore = (
            data.loc[genes_up, :].sum() - data.loc[genes_down, :].sum()
        ) / bicluster["n_genes"]
    else:
        avg_zscore = data.loc[list(bicluster["genes"]), :].mean()

    # compute SNR for average z-score for this bicluster
    bicluster["SNR"] = np.abs(
        calc_SNR(avg_zscore[bic_samples], avg_zscore[bg_samples], pd_mode=True)
    )
    return bicluster


def merge_biclusters(
    biclusters, data, J=0.8, min_n_samples=5, seed=42, method="kmeans", verbose=True
):
    #  bicluaters -> binary -> jaccard sim
    binary_representation = {}
    N = data.shape[1]
    for i in biclusters.keys():
        b = np.zeros(N)
        s_ndx = list(biclusters[i]["sample_indexes"])
        b[s_ndx] = 1
        binary_representation[i] = b
    binary_representation = pd.DataFrame.from_dict(binary_representation)
    binary_representation.index = data.columns.values
    bic_similarity = get_similarity_jaccard(binary_representation, verbose=verbose)
    # bic_similarity[bic_similarity >= J] = 1
    # bic_similarity[bic_similarity < J] = 0
    # find groups of biclusters including the same sample sets
    merged, not_merged, similarity_cutoff = run_Louvain(
        bic_similarity, verbose=False, plot=False, similarity_cutoffs=[J]
    )
    if len(merged) == 0 and verbose:
        print("No biclusters to merge", file=sys.stdout)
        return biclusters

    merged_biclusters = {}
    # add biclusters with no changes
    for bic_id in not_merged:
        merged_biclusters[bic_id] = biclusters[bic_id]

    # merge biclusters with overlapping sample sets
    for bic_group in merged:
        bic_group = sorted(bic_group)
        if verbose:
            print("merged biclustres", bic_group, file=sys.stderr)
        new_bicluster = biclusters[bic_group[0]]
        # update genes
        for bic_id in bic_group[1:]:
            bic2 = biclusters[bic_id]
            new_bicluster["genes"] = new_bicluster["genes"] | bic2["genes"]
            new_bicluster["n_genes"] = len(new_bicluster["genes"])
        # update sample set for new bicluster
        # cluster samples in a space of selected genes
        new_bicluster.update(
            cluster_samples(
                data.loc[list(new_bicluster["genes"]), :].T,
                min_n_samples=min_n_samples,
                seed=seed,
                method=method,
            )
        )
        new_bicluster["n_samples"] = len(new_bicluster["sample_indexes"])
        merged_biclusters[bic_group[0]] = new_bicluster
    return merged_biclusters


def make_biclusters(
    feature_clusters,
    binarized_data,
    data,
    merge=1,
    min_n_samples=10,
    min_n_genes=2,
    method="kmeans",
    seed=42,
    cluster_binary=False,
    verbose=True,
):

    biclusters = []

    if cluster_binary:
        data_to_cluster = binarized_data.loc[:, :].T  # binarized expressions
    else:
        data_to_cluster = data.loc[binarized_data.columns.values, :]  # z-scores

    if len(feature_clusters) == 0:
        print("No biclusters found.", file=sys.stderr)
    else:
        biclusters = modules2biclusters(
            feature_clusters,
            data_to_cluster,
            method=method,
            min_n_samples=min_n_samples,
            min_n_genes=min_n_genes,
            verbose=False,
            seed=seed,
        )

        ### merge biclusters with highly similar sample sets
        if merge < 1.0:
            biclusters = merge_biclusters(
                biclusters,
                data,
                method=method,
                J=merge,
                min_n_samples=min_n_samples,
                seed=seed,
                verbose=verbose,
            )

        for i in list(biclusters.keys()):
            biclusters[i] = update_bicluster_data(biclusters[i], data)

    biclusters = pd.DataFrame.from_dict(biclusters).T
    # add direction
    biclusters["direction"] = "BOTH"
    biclusters.loc[
        biclusters["n_genes"] == biclusters["genes_up"].apply(len), "direction"
    ] = "UP"
    biclusters.loc[
        biclusters["n_genes"] == biclusters["genes_down"].apply(len), "direction"
    ] = "DOWN"

    # add p-value for bicluster SNR (computed for avg. zscores)
    # use the same distribution as for single features
    # biclusters["e_pval"] = biclusters.apply(lambda row: calc_e_pval(row["SNR"], row["n_samples"], null_distribution),axis=1)

    # sort and reindex
    biclusters = biclusters.sort_values(by=["SNR", "n_genes"], ascending=[False, False])
    biclusters.index = range(0, biclusters.shape[0])

    biclusters = biclusters.loc[
        :,
        [
            "SNR",
            "n_genes",
            "n_samples",
            "genes",
            "samples",
            "direction",
            "genes_up",
            "genes_down",
            "gene_indexes",
            "sample_indexes",
        ],
    ]

    return biclusters