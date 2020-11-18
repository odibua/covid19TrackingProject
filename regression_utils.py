# --------------------------
# Standard Python Imports
# --------------------------
import collections
import copy
import os
from os import path
from typing import Any, Callable, Dict, List, Tuple, Union

# --------------------------
# Third Party Imports
# --------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm

# --------------------------
# covid19Tracking Imports
# --------------------------


def calc_metric(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    ss_res = np.sum((Y_pred - Y) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def construct_graph_from_features(X: np.ndarray) -> Dict[int, List[int]]:
    index_list = list(range(X.shape[1]))
    index_graph = collections.defaultdict(list)
    for idx in index_list:
        index_graph[idx].extend([idx2 for idx2 in index_list if idx2 != idx])

    return index_graph


def fit_subset(X: np.ndarray, Y: np.ndarray, feature_indices: List[int]) -> Tuple[float, sm.OLS.fit]:
    model = sm.OLS(Y, X[:, feature_indices])
    fitted_model = model.fit()
    # Y_pred = fitted_model.fittedvalues
    metric = -fitted_model.bic #calc_metric(Y=Y, Y_pred=Y_pred)
    return metric, fitted_model


def fit_subset_sizes(X, Y, subset_size, full_subset, curr_subset, metric_list, fitted_model_list, subsets_list):
    if len(curr_subset) == subset_size:
        metric, fitted_model = fit_subset(X=X, Y=Y, feature_indices=curr_subset)
        metric_list.append(metric)
        fitted_model_list.append(fitted_model)
        subsets_list.append(curr_subset)
        return

    for idx in full_subset:
        next_full_subset = copy.deepcopy(full_subset)
        next_full_subset.remove(idx)
        fit_subset_sizes(X, Y, subset_size, next_full_subset, curr_subset + [idx], metric_list, fitted_model_list, subsets_list)


def get_best_subset(X: np.ndarray, Y: np.ndarray, tol: float = 0.05) -> Tuple[sm.OLS.fit, List[int]]:
    full_subset = list(range(X.shape[1]))
    delta_metric = float("inf")
    metric_list, fitted_model_list, subsets_list = [], [], []
    subset_size = 2
    full_subset.remove(0)

    while subset_size < X.shape[1] and delta_metric > tol:
        fit_subset_sizes(X, Y, subset_size, full_subset, [0], metric_list, fitted_model_list, subsets_list)
        subset_size = subset_size + 1
    best_idx = np.argmax(metric_list)
    best_fitted_model = fitted_model_list[best_idx]
    best_subset = subsets_list[best_idx]
    return best_fitted_model, best_subset


def call_multilinear_regression(X: np.ndarray, Y: np.ndarray) -> Tuple[sm.OLS.fit, List[int]]:
    fitted_model, feature_subset = get_best_subset(X=X, Y=Y)
    return fitted_model, feature_subset


def multilinear_reg(state_name: str, type: str, county_name: str) -> None:
    # Define path and file for training data
    training_csv_path = path.join('states', state_name, 'training_data_csvs')
    regression_results_path = path.join('states', state_name, 'regression_results_csvs')

    if county_name is None:
        training_file = f'{state_name}_training_{type}.csv'
    else:
        training_file = f'{state_name}_{county_name}_training_{type}.csv'

    training_data_df = pd.read_csv(path.join(training_csv_path, training_file), index_col=0)

    # Set Y as mortality rate
    Y = np.array(training_data_df['mortality_rate'])

    # Populate remaining columns with corresponding metadata
    filter_list = ['mortality_rate']
    metadata_keys = training_data_df.keys()
    metadata_keys = [key for key in metadata_keys if key not in filter_list]

    # Construct X
    X = np.zeros((training_data_df.shape[0], training_data_df.shape[1] - len(filter_list) + 1))

    for idx, key in enumerate(metadata_keys):
        X[:, idx + 1] = training_data_df[key].tolist()
    metadata_keys.insert(0, 'constant')

    X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
    X[:, 0] = 1
    fitted_model, feature_subset = call_multilinear_regression(X=X, Y=Y,)
    features = [metadata_keys[idx] for idx in feature_subset]
    stat_table = fitted_model.summary().tables[-1]

    # Calculate variance inflation factor to check for colinearity
    vif_list = []
    for idx in feature_subset:
        Y_feat = X[:, idx]
        feature_indices = [idx2 for idx2 in feature_subset if idx2 != idx]
        _, fitted_feature_model = fit_subset(Y=Y_feat, X=X, feature_indices=feature_indices)
        vif_list.append(1.0 / (1 - fitted_feature_model.rsquared))

    # Regress on features and ca

    regression_info_dict = {'state': state_name, 'county': county_name, 'n': Y.shape[0]}
    regression_info_dict['features'] = features
    regression_info_dict['coef'] = fitted_model.params
    regression_info_dict['lower_coef'] = regression_info_dict['coef'] - 1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['upper_coef'] = regression_info_dict['coef'] + 1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['vif'] = vif_list
    regression_info_dict['std_err'] = np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['R2'] = fitted_model.rsquared
    regression_info_dict['R2 Adjusted'] = fitted_model.rsquared_adj
    regression_info_dict['Durbin-Watson'] = float(stat_table.data[0][-1])
    regression_info_dict['JB'] = float(stat_table.data[-3][-1])
    regression_info_dict['condition_number'] = fitted_model.condition_number

    regression_info_df = pd.DataFrame(regression_info_dict)

    if not os.path.isdir(regression_results_path):
        os.mkdir(regression_results_path)

    regression_results_file = path.join(regression_results_path, f'{type}_linear_regression_results.csv')
    if not os.path.isfile(regression_results_file):
        regression_info_df.to_csv(regression_results_file, index=False)
    else:
        regression_info_df.to_csv(regression_results_file, mode='a', header=False, index=False)
