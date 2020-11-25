# --------------------------
# Standard Python Imports
# --------------------------
from typing import Dict, Tuple, Union

# --------------------------
# Third Party Imports
# --------------------------
import numpy as np
from scipy import stats


# --------------------------
# covid19Tracking Imports
# --------------------------


def calc_filtered_spearman_corr(X: np.ndarray, Y: np.ndarray) -> Tuple[Union[float, None], Union[float, None]]:
    spearman_corr, p_val = stats.spearmanr(Y, X)
    if abs(spearman_corr) >= 0.4 and p_val <= 0.05:
        return spearman_corr, p_val
    return None, None


def populate_spearman_corr_dict(corr_dict: Dict[str, Union[str, float]], y_key: str, x_key: str, state: str, county: str, n: int,
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