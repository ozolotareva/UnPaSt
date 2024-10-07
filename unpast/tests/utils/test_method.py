import numpy as np
import warnings
from unpast.utils.method import  get_trend, calc_SNR


def test_get_trend_single_point():
    sizes = [10]
    thresholds = [2.5]
    min_snr = get_trend(sizes, thresholds, plot=False, verbose=False)
    assert min_snr(10) == 2.5


def test_get_trend_multiple_points():
    sizes = [10, 20, 30, 40, 50]
    thresholds = [2.5, 3.0, 3.5, 4.0, 4.5]
    min_snr = get_trend(sizes, thresholds, plot=False, verbose=False)
    assert np.allclose(min_snr(sizes), thresholds)


def test_get_trend_noisy():
    sizes = [1] * 100 + [2] * 100
    thresholds = np.linspace(0, 1, 200)
    min_snr = get_trend(sizes, thresholds, plot=False, verbose=False)
    # 0.1 tolerance, as set of used points may be not very big
    assert np.allclose(min_snr([1, 1.5, 2]), [0.25, 0.5, 0.75], atol=0.1)


def test_calc_SNR():
    val1 = calc_SNR([0, 1, 2], [1, 2, 3])
    val2 = calc_SNR([0, 1, 2], [1, 2, 3], pd_mode=True)
    np.testing.assert_almost_equal(val1, -0.6123724356957945)
    np.testing.assert_almost_equal(val2, -0.5)

    # assert not failing
    assert calc_SNR([0, 0, 0], [1, 1, 1]) == float("-inf")
    assert calc_SNR([0, 0, 0], [1, 1, 1], True) == float("-inf")
    assert calc_SNR([1, 1, 1], [0, 0, 0]) == float("+inf")

    # sends warning "overflow encountered" 
    # big numbers
    big_nums = [1e307, 1e307, 1e307]
    small_nums = [1e-150, 0, 0]
    assert np.std(big_nums) < float("+inf")
    assert np.std(small_nums) > 0
    assert calc_SNR(big_nums, small_nums) in [float("inf"), float("+inf")]

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # zero std
        assert calc_SNR([1, 1, 1], [0, 0, 0]) in [float("inf"), float("+inf")]
        assert calc_SNR([0, 0, 0], [1, 1, 1]) in [float("inf"), float("-inf")]



# def test_zscore():
#     # Test case 1: Basic functionality
#     df = pd.DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6],
#         'C': [7, 8, 9]
#     })
#     result = zscore(df)
#     assert result.shape == df.shape
#     assert np.allclose(result.mean(), 0, atol=1e-7)
#     assert np.allclose(result.std(), 1, atol=1e-7)

#     # Test case 2: Handling zero variance
#     df_zero_var = pd.DataFrame({
#         'A': [1, 1, 1],
#         'B': [2, 3, 4],
#         'C': [5, 6, 7]
#     })
#     result_zero_var = zscore(df_zero_var)
#     assert result_zero_var.shape == df_zero_var.shape
#     assert np.allclose(result_zero_var.loc['A'], 0)

# def test_prepare_input_matrix():
#     # Test case 1: Basic functionality
#     df = pd.DataFrame({
#         'A': [1, 2, 3, 4, 5],
#         'B': [2, 3, 4, 5, 6],
#         'C': [3, 4, 5, 6, 7]
#     })
#     result = prepare_input_matrix(df)
#     assert result.shape == df.shape
#     assert np.allclose(result.mean(), 0, atol=1e-7)
#     assert np.allclose(result.std(), 1, atol=1e-7)

#     # Test case 2: Handling zero variance
#     df_zero_var = pd.DataFrame({
#         'A': [1, 1, 1, 1, 1],
#         'B': [2, 3, 4, 5, 6],
#         'C': [3, 4, 5, 6, 7]
#     })
#     result_zero_var = prepare_input_matrix(df_zero_var)
#     assert result_zero_var.shape == (2, 5)  # One row should be dropped

#     # Test case 3: Handling missing values
#     df_missing = pd.DataFrame({
#         'A': [1, 2, np.nan, 4, 5],
#         'B': [2, 3, 4, np.nan, 6],
#         'C': [3, 4, 5, 6, 7]
#     })
#     result_missing = prepare_input_matrix(df_missing, min_n_samples=4)
#     assert result_missing.shape == (2, 5)  # One row should be dropped

#     # Test case 4: Ceiling
#     df_ceiling = pd.DataFrame({
#         'A': [1, 2, 3, 4, 5],
#         'B': [2, 3, 4, 5, 6],
#         'C': [3, 4, 5, 6, 7]
#     })
#     result_ceiling = prepare_input_matrix(df_ceiling, ceiling=2)
#     assert result_ceiling.max().max() <= 2
#     assert result_ceiling.min().min() >= -2

#     # Test case 5: Non-standardized input
#     df_non_std = pd.DataFrame({
#         'A': [10, 20, 30, 40, 50],
#         'B': [20, 30, 40, 50, 60],
#         'C': [30, 40, 50, 60, 70]
#     })
#     result_non_std = prepare_input_matrix(df_non_std)
#     assert np.allclose(result_non_std.mean(), 0, atol=1e-7)
#     assert np.allclose(result_non_std.std(), 1, atol=1e-7)
