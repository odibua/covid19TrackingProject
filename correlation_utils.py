# --------------------------
# Standard Python Imports
# --------------------------
from typing import Dict, List, Tuple, Union

# --------------------------
# Third Party Imports
# --------------------------
import dcor
import numpy as np
from pingouin import distance_corr
from scipy import stats
from scipy.spatial import distance


# --------------------------
# covid19Tracking Imports
# --------------------------


def calc_filtered_dist_corr(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, float]:
    N, n, frac = 1000,  Y.shape[0], 1.0
    indices_list, dist_corr_list = [], []
    for idx in range(N):
        indices = np.random.choice(n, size=int(frac * n), replace=True)
        indices_list.append(indices)
    for indices in indices_list:
        dist_corr = dcor.distance_correlation(X[indices], Y[indices]) # distance_corr(X, Y, seed=9)
        dist_corr_list.append(dist_corr)
    low_dist_corr, up_dist_corr = np.percentile(dist_corr_list, 2.5), np.percentile(dist_corr_list, 97.5)
    dist_corr = dcor.distance_correlation(X, Y)
    # pval = dcor.independence.distance_covariance_test(X, Y, exponent=1.0, num_resamples=100)
    # dist_corr, pval = distance_corr(X, Y, n_boot=100)
    # print(f'dco pval: {dist_corr} {pval}')
    # import ipdb
    # ipdb.set_trace()
    return dist_corr, low_dist_corr, up_dist_corr
    # if dist_corr > 0.4:
    # return dist_corr


def calc_filtered_spearman_corr(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float, float]:
    N, n, frac = 1000,  Y.shape[0], 1.0
    indices_list, spearman_corr_list = [], []
    for idx in range(N):
        indices = np.random.choice(n, size=int(frac * n), replace=True)
        indices_list.append(indices)

    for indices in indices_list:
        spearman_corr, p_val = stats.spearmanr(Y[indices], X[indices])
        spearman_corr_list.append(spearman_corr)

    spearman_corr, p_val = stats.spearmanr(Y, X)
    low_spearman_corr, up_spearman_corr = np.percentile(spearman_corr_list, 2.5), np.percentile(spearman_corr_list, 97.5)
    return spearman_corr, low_spearman_corr, up_spearman_corr
    # if abs(spearman_corr) >= 0.4 and p_val <= 0.05:
    #     return spearman_corr, p_val
    # return None, None


def populate_spearman_corr_dict(corr_dict: Dict[str, List[Union[str, float]]], y_key: str, x_key: str, state: str, county: str, n: int,
                                X: np.ndarray, Y: np.ndarray) -> None:
    spearman_corr, low_spearman_corr, up_spearman_corr = calc_filtered_spearman_corr(X=X, Y=Y)
    if spearman_corr is not None:
        corr_dict['corr'].append(spearman_corr)
        corr_dict['Y'].append(y_key)
        corr_dict['X'].append(x_key)
        # corr_dict['p_val'].append(p_val)
        corr_dict['state'].append(state)
        corr_dict['county'].append(county)
        corr_dict['n'].append(n)
        corr_dict['low_corr'].append(low_spearman_corr)
        corr_dict['up_corr'].append(up_spearman_corr )


def populate_dist_corr_dict(corr_dict: Dict[str, Union[str, List[Union[str, float]]]], y_key: str, x_key: str, state: str, county: str, n: int,
                            X: np.ndarray, Y: np.ndarray) -> None:
    dist_corr, low_dist_corr, up_dist_corr = calc_filtered_dist_corr(X=X, Y=Y)
    if dist_corr is not None:
        corr_dict['corr'].append(dist_corr)
        corr_dict['Y'].append(y_key)
        corr_dict['X'].append(x_key)
        corr_dict['state'].append(state)
        corr_dict['county'].append(county)
        corr_dict['n'].append(n)
        corr_dict['low_corr'].append(low_dist_corr)
        corr_dict['up_corr'].append(up_dist_corr)
        # corr_dict['p_val'].append(p_val)

