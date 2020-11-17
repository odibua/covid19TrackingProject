# --------------------------
# Standard Python Imports
# --------------------------
import collections
from typing import Any, Callable, Dict, List, Tuple, Union

# --------------------------
# Third Party Imports
# --------------------------
import numpy as np
import statsmodels.api as sm

# --------------------------
# covid19Tracking Imports
# --------------------------


def construct_graph_from_features(X: np.ndarray) -> Dict[int, List[int]]:
    index_list = list(range(X.shape[1]))

    index_graph = collections.defaultdict(list)
    for idx in index_list:
        index_graph[idx].extend([idx2 for idx2 in index_list if idx2 != idx])

    return index_graph


def get_best_subset(X: np.ndarray, Y: np.ndarray, fitted_model: sm.OLS.fit, delta: float, tol: float) -> Tuple[sm.OLS.fit, List[int]]:
    idx0 = 0
    X_use = X[:, idx0].reshape((-1, 1))
    model = sm.OLS(Y, X_use)
    results = model.fit()
    Y_pred = results.fittedvalues
    nrmse = np.mean(np.sum(np.sqrt(((Y-Y_pred)/Y)**2)))
    full_subset = list(range(X.shape[1]))
    best_subset = [idx0]
    full_subset.remove(idx0)

    delta_nrmse = float("inf")
    while len(best_subset) < X.shape[1] and delta_nrmse < tol:
        delta_nrmse_list = []
        for idx in full_subset:
            X_use = X[:, best_subset + [idx]]
            model = sm.OLS(Y, X_use)
            results = model.fit()
            Y_pred = results.fittedvalues
            new_nrmse = np.mean(np.sum(np.sqrt(((Y - Y_pred) / Y) ** 2)))
            delta_nrmse_list.append(new_nrmse / nrmse)

        idx = np.argmin(delta_nrmse_list)
        delta_nrmse = delta_nrmse[idx]
        next_idx = full_subset[idx]
        best_subset.append(next_idx)
        full_subset.remove(idx)

    pass


def fit_best_subset(X: np.ndarray, Y: np.ndarray, tol: float) -> sm.OLS:
    graph_X = construct_graph_from_features(X)
    model = sm.OLS(Y,X)
    model = model.fit()
    model, best_subset = get_best_subset(graph=graph_X, X=X, Y=Y, best_subset=[0], fitted_model=model, delta=float("inf"), tol=tol)


def call_multilinear_regression(X: np.ndarray, Y: np.ndarray, regression_keys: List[str]) -> Any:
    tol = 0.05
    model = sm.OLS(Y, X)
    results = model.fit()
    print(results.params)
    return results.summary()