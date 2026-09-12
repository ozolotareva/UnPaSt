import numpy as np
import pandas as pd
import pytest

from unpast.misc.eval.metrics import calc_metrics
from unpast.tests.test_utils import _hash_table


@pytest.mark.parametrize("n_genes_samples", [(10, 10), (8, 10)])
def test_reproducible(n_genes_samples):
    """Test metrics calculations give exactly the same results as now."""
    true_bics = pd.DataFrame(
        {
            "genes": [{"g1", "g2", "g6"}, {"g1", "g5", "g7"}, {"g1", "g4"}],
            "samples": [
                {"s1", "s2", "s6"},
                {"s2", "s4", "s5"},
                {"s3", "s4", "s5", "s7"},
            ],
        }
    )

    pred_bics = pd.DataFrame(
        {
            "genes": [
                {"g1", "g2", "g6"},
                {"g1", "g2", "g3"},
                {"g1", "g5"},
                {"g6", "g7"},
            ],
            "samples": [
                {"s1", "s2", "s6"},
                {"s3", "s4", "s5"},
                {"s2", "s3", "s7"},
                {"s1", "s2"},
            ],
            "SNR": [3.5, 2.5, 1.5, 0.5],
        }
    )

    n_genes, n_samples = n_genes_samples
    data = pd.DataFrame(
        np.ones((n_genes, n_samples)),
        columns=[f"s{i}" for i in range(n_samples)],
        index=[f"g{i}" for i in range(n_genes)],
    )

    pred_bics["n_genes"] = pred_bics["genes"].apply(len)
    pred_bics["n_samples"] = pred_bics["samples"].apply(len)
    true_bics["n_genes"] = true_bics["genes"].apply(len)
    true_bics["n_samples"] = true_bics["samples"].apply(len)

    metrics = calc_metrics(true_bics, pred_bics, data)
    assert metrics["wARIs"] == 0.3

    # smoke test
    metrics = calc_metrics(true_bics, pred_bics, data.iloc[:8, :8])
    assert metrics["wARIs"] == 0.0  # correspondence removed by pval thresholding


def _gen_random_biclusters(rand, cols, inds, n_bics):
    gene_weights = np.power(0.9, np.arange(len(inds)))
    gene_weights /= gene_weights.sum()
    sample_weights = np.power(0.9, np.arange(len(cols)))
    sample_weights /= sample_weights.sum()

    bics = {}
    for i in range(n_bics):
        n_g = rand.randint(5, 15)
        n_s = rand.randint(5, 15)

        genes = set(rand.choice(inds, size=n_g, replace=False, p=gene_weights))
        samples = set(rand.choice(cols, size=n_s, replace=False, p=sample_weights))
        bics[f"bic_{i}"] = {"genes": genes, "samples": samples}

    return pd.DataFrame.from_dict(bics).T


def test_reproducible_big_random():
    rand = np.random.RandomState(42)
    n_genes = 50
    n_samples = 50
    data = pd.DataFrame(
        rand.rand(n_genes, n_samples),
        columns=[f"s_{i}" for i in range(n_samples)],
        index=[f"g_{i}" for i in range(n_genes)],
    )

    n_true_bics = 5
    n_pred_bics = 20

    repeated_metrics = []

    for i in range(10):
        # use weights to force bics to be often the same
        true_bics = _gen_random_biclusters(
            rand, data.columns.values, data.index.values, n_true_bics
        )
        pred_bics = _gen_random_biclusters(
            rand, data.columns.values, data.index.values, n_pred_bics
        )

        # todo: avoid requiring extra columns for the metrics calculations
        pred_bics["n_genes"] = pred_bics["genes"].apply(len)
        pred_bics["n_samples"] = pred_bics["samples"].apply(len)
        pred_bics["SNR"] = range(len(pred_bics))
        true_bics["n_genes"] = true_bics["genes"].apply(len)
        true_bics["n_samples"] = true_bics["samples"].apply(len)

        metrics = calc_metrics(true_bics, pred_bics, data)
        repeated_metrics.append(metrics)

    metrics_df = pd.DataFrame(repeated_metrics)
    assert _hash_table(metrics_df.drop(columns=["AP_ARI"])) == 17285711652143097685
    assert _hash_table(metrics_df) == 6537791270467914040

    # different extra calculations:
    params_metrics = {}
    for measure in ["ARI", "Jaccard"]:
        for adjust_pvals in ["B", "BH", False]:
            params_metrics[f"{measure}_{adjust_pvals}"] = calc_metrics(
                true_bics,
                pred_bics,
                data,
                matching_measure=measure,
                adjust_pvals=adjust_pvals,
            )
    metrics_df_params = pd.DataFrame(params_metrics)
    assert _hash_table(metrics_df_params.drop(index=["AP_ARI"])) == 892002411010055165
    assert _hash_table(metrics_df_params) == 17996364079568771923


def test_calc_metrics_empty_data():
    exprs = pd.DataFrame(
        np.random.rand(10, 10),
        index=[f"g{i}" for i in range(10)],
        columns=[f"s{i}" for i in range(10)],
    )
    true_bics = pd.DataFrame(
        {
            "genes": [{"g1", "g2"}, {"g3", "g4"}],
            "samples": [{"s1", "s2"}, {"s3", "s4"}],
        }
    )
    pred_bics = pd.DataFrame(
        {
            "genes": [{"g1", "g2"}, {"g3", "g9"}],
            "samples": [{"s1", "s2"}, {"s3", "s9"}],
            "SNR": [2.0, 1.5],
        }
    )

    # todo: avoid requiring extra columns for the metrics calculations
    pred_bics["n_genes"] = pred_bics["genes"].apply(len)
    pred_bics["n_samples"] = pred_bics["samples"].apply(len)
    pred_bics["SNR"] = range(len(pred_bics))
    true_bics["n_genes"] = true_bics["genes"].apply(len)
    true_bics["n_samples"] = true_bics["samples"].apply(len)

    # empty data
    empty_true_bics = true_bics.copy()[:0]
    empty_pred_bics = pred_bics.copy()[:0]

    metrics_11 = calc_metrics(true_bics, pred_bics, exprs)

    metrics_10 = calc_metrics(true_bics, empty_pred_bics, exprs)
    assert metrics_10["FDR_bic"] == 1.0
    assert all(v == 0.0 for (k, v) in metrics_10.items() if k != "FDR_bic")
    assert metrics_10.keys() == metrics_11.keys(), "Metrics keys differ in edge case"

    with pytest.raises(ValueError):
        calc_metrics(empty_true_bics, pred_bics, exprs)

    with pytest.raises(ValueError):
        calc_metrics(empty_true_bics, empty_pred_bics, exprs)

    # metrics_01 = calc_metrics(empty_true_bics, pred_bics, exprs)
    # assert metrics_01["wARIs"] == 1.0

    # metrics_00 = calc_metrics(empty_true_bics, empty_pred_bics, exprs)
    # assert metrics_00["wARIs"] == 0.0

    # # has same keys as full metrics
    # for m in metrics_01, metrics_10, metrics_00:
    #     assert m.keys() == metrics_11.keys()
