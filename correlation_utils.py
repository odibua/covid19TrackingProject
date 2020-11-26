# --------------------------
# Standard Python Imports
# --------------------------
from typing import Dict, List, Tuple, Union

# --------------------------
# Third Party Imports
# --------------------------
import dcor
import numpy as np
from scipy import stats
from scipy.spatial import distance


# --------------------------
# covid19Tracking Imports
# --------------------------


def calc_filtered_dist_corr(X: np.ndarray, Y: np.ndarray) -> Union[float, None]:
    dist_corr = dcor.distance_correlation(x=X, y=Y)
    if dist_corr > 0.4:
        return dist_corr
    return None


def calc_filtered_spearman_corr(X: np.ndarray, Y: np.ndarray) -> Tuple[Union[float, None], Union[float, None]]:
    spearman_corr, p_val = stats.spearmanr(Y, X)
    if abs(spearman_corr) >= 0.4 and p_val <= 0.05:
        return spearman_corr, p_val
    return None, None


def populate_spearman_corr_dict(corr_dict: Dict[str, List[Union[str, float]]], y_key: str, x_key: str, state: str, county: str, n: int,
                                X: np.ndarray, Y: np.ndarray) -> None:
    spearman_corr, p_val = calc_filtered_spearman_corr(X=X, Y=Y)
    if spearman_corr is not None:
        corr_dict['corr'].append(spearman_corr)
        corr_dict['Y'].append(y_key)
        corr_dict['X'].append(x_key)
        corr_dict['p_val'].append(p_val)
        corr_dict['state'].append(state)
        corr_dict['county'].append(county)
        corr_dict['n'].append(n)


def populate_dist_corr_dict(corr_dict: Dict[str, Union[str, List[Union[str, float]]]], y_key: str, x_key: str, state: str, county: str, n: int,
                            X: np.ndarray, Y: np.ndarray) -> None:
    dist_corr = calc_filtered_dist_corr(X=X, Y=Y)
    if dist_corr is not None:
        corr_dict['corr'].append(dist_corr)
        corr_dict['Y'].append(y_key)
        corr_dict['X'].append(x_key)
        corr_dict['state'].append(state)
        corr_dict['county'].append(county)
        corr_dict['n'].append(n)
