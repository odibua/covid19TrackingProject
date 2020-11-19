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
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge

# --------------------------
# covid19Tracking Imports
# --------------------------


def calc_metric(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    ss_res = np.sum((Y_pred - Y) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def calc_nrmse(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    return np.mean(np.sqrt(((Y - Y_pred) / Y) ** 2))


def calc_rmse(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    return np.mean(np.sqrt(((Y - Y_pred)) ** 2))


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
    metric = -fitted_model.bic  # calc_metric(Y=Y, Y_pred=Y_pred)
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
        fit_subset_sizes(
            X,
            Y,
            subset_size,
            next_full_subset,
            curr_subset + [idx],
            metric_list,
            fitted_model_list,
            subsets_list)


def get_best_ridge_model(X: np.ndarray, Y: np.ndarray) -> Tuple[float, Ridge.fit]:
    alpha_list = np.linspace(0, 1, 100)
    fitted_model_list, score_list = [], []
    for alpha in alpha_list:
        model = Ridge(alpha=alpha)
        fitted_model = model.fit(X, Y)
        score = model.score(X, Y)
        fitted_model_list.append(fitted_model)
        score_list.append(score)
    max_idx = np.argmax(score_list)
    best_fitted_model = fitted_model_list[max_idx]
    best_score = score_list[max_idx]
    return best_score, best_fitted_model


def get_best_lasso_model(X: np.ndarray, Y: np.ndarray) -> Tuple[float, Ridge.fit]:
    alpha_list = np.linspace(0, 1, 100)
    fitted_model_list, score_list = [], []
    for alpha in alpha_list:
        model = Lasso(alpha=alpha)
        fitted_model = model.fit(X, Y)
        score = model.score(X, Y)
        fitted_model_list.append(fitted_model)
        score_list.append(score)
    max_idx = np.argmax(score_list)
    best_fitted_model = fitted_model_list[max_idx]
    best_score = score_list[max_idx]
    return best_score, best_fitted_model


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


def call_multilinear_ridge_regression(X: np.ndarray, Y: np.ndarray) -> Tuple[float, Ridge.fit]:
    score, fitted_model = get_best_ridge_model(X=X, Y=Y)
    return score, fitted_model


def call_multilinear_lasso_regression(X: np.ndarray, Y: np.ndarray) -> Tuple[float, Ridge.fit]:
    score, fitted_model = get_best_lasso_model(X=X, Y=Y)
    return score, fitted_model

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

    # Get nrmse and rmse
    nrmse = calc_nrmse(Y, fitted_model.fittedvalues)
    rmse = calc_rmse(Y, fitted_model.fittedvalues)

    regression_info_dict = {'state': state_name, 'county': county_name, 'n': Y.shape[0]}
    regression_info_dict['features'] = features
    regression_info_dict['coef'] = fitted_model.params
    regression_info_dict['lower_coef'] = regression_info_dict['coef'] - \
        1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['upper_coef'] = regression_info_dict['coef'] + \
        1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['vif'] = vif_list
    regression_info_dict['std_err'] = np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['R2'] = fitted_model.rsquared
    regression_info_dict['nrmse'] = nrmse
    regression_info_dict['rmse'] = rmse
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


def multilinear_pca_reg(state_name: str, type: str, county_name: str, var_thresh: float = 0.95) -> None:
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

    # Extract columns for PCA (non-time and non-constant)
    pca_idx_start = 2
    X_star = X[:, pca_idx_start:]
    tot_var = 0
    n_components = 1
    while tot_var < var_thresh:
        pca = PCA(n_components=n_components)
        pca.fit(X_star)
        tot_var = sum(pca.explained_variance_ratio_)
        n_components = n_components + 1
    n_components = pca.n_components
    pca_components = pca.components_.T

    # Construct features from PCA
    X_pca = np.zeros((training_data_df.shape[0], n_components + pca_idx_start))
    X_pca[:, 0:pca_idx_start] = X[:, 0:pca_idx_start]
    X_pca[:, pca_idx_start:] = np.matmul(X_star, pca_components)

    feature_subset = list(range(X_pca.shape[1]))
    # import ipdb
    # ipdb.set_trace()
    metric, fitted_model = fit_subset(Y=Y, X=X_pca, feature_indices=feature_subset)

    # Get nrmse and rmse
    nrmse = calc_nrmse(Y, fitted_model.fittedvalues)
    rmse = calc_rmse(Y, fitted_model.fittedvalues)

    # Get coefficients attached to PCA
    coef = fitted_model.params
    pca_lower_weighted_components = copy.deepcopy(pca_components)
    pca_weighted_components = copy.deepcopy(pca_components)
    pca_upper_weighted_components = copy.deepcopy(pca_components)
    std_params = np.sqrt(np.diag(fitted_model.cov_params()))
    for idx in range(pca_components.shape[1]):
        pca_lower_weighted_components[:, idx] = pca_components[:, idx] * (coef[idx + 2] - 1.96 * std_params[idx + 2])
        pca_weighted_components[:, idx] = pca_components[:, idx] * coef[idx + 2]
        pca_upper_weighted_components[:, idx] = pca_components[:, idx] * (coef[idx + 2] + 1.96 * std_params[idx + 2])
    feature_coef = list(coef[0:pca_idx_start]) + list(np.sum(pca_weighted_components, axis=1))
    lower_feature_coef = list(coef[0:pca_idx_start] - 1.96 * std_params[0:pca_idx_start]) + list(np.sum(pca_lower_weighted_components, axis=1))
    upper_feature_coef = list(coef[0:pca_idx_start] + 1.96 * std_params[0:pca_idx_start]) + list(np.sum(pca_upper_weighted_components, axis=1))

    stat_table = fitted_model.summary().tables[-1]
    # Calculate variance inflation factor to check for colinearity
    vif_list = []
    for idx in feature_subset:
        Y_feat = X[:, idx]
        feature_indices = [idx2 for idx2 in feature_subset if idx2 != idx]
        _, fitted_feature_model = fit_subset(Y=Y_feat, X=X, feature_indices=feature_indices)
        vif_list.append(1.0 / (1 - fitted_feature_model.rsquared))


    regression_info_dict = {'state': state_name, 'county': county_name, 'n': Y.shape[0]}
    regression_info_dict['coef'] = fitted_model.params
    regression_info_dict['lower_coef'] = regression_info_dict['coef'] - \
        1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['upper_coef'] = regression_info_dict['coef'] + \
        1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['vif'] = vif_list
    regression_info_dict['std_err'] = np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['R2'] = fitted_model.rsquared
    regression_info_dict['R2 Adjusted'] = fitted_model.rsquared_adj
    regression_info_dict['nrmse'] = nrmse
    regression_info_dict['rmse'] = rmse
    regression_info_dict['Durbin-Watson'] = float(stat_table.data[0][-1])
    regression_info_dict['JB'] = float(stat_table.data[-3][-1])
    regression_info_dict['condition_number'] = fitted_model.condition_number

    regression_info_df = pd.DataFrame(regression_info_dict)

    # Save information about pca regression
    # ipdb.set_trace()
    regression_pca_info_dict = {'state': state_name, 'county': county_name, 'n': Y.shape[0]}
    regression_pca_info_dict['features'] = metadata_keys
    regression_pca_info_dict['coef'] = feature_coef
    regression_pca_info_dict['lower_coef'] = lower_feature_coef
    regression_pca_info_dict['upper_coef'] = upper_feature_coef
    # regression_info_dict['std_err'] = np.sqrt(np.diag(fitted_model.cov_params()))
    regression_pca_info_dict['R2'] = fitted_model.rsquared
    regression_pca_info_dict['R2 Adjusted'] = fitted_model.rsquared_adj
    regression_pca_info_dict['nrmse'] = nrmse
    regression_pca_info_dict['rmse'] = rmse
    regression_pca_info_dict['Durbin-Watson'] = float(stat_table.data[0][-1])
    regression_pca_info_dict['JB'] = float(stat_table.data[-3][-1])
    regression_pca_info_dict['condition_number'] = fitted_model.condition_number

    regression_pca_info_df = pd.DataFrame(regression_pca_info_dict)
    # import ipdb
    # ipdb.set_trace()
    if not os.path.isdir(regression_results_path):
        os.mkdir(regression_results_path)

    regression_pca_results_file = path.join(regression_results_path, f'{type}_pca_comp_linear_regression_results.csv')
    regression_results_file = path.join(regression_results_path, f'{type}_pca_linear_regression_results.csv')

    if not os.path.isfile(regression_pca_results_file):
        regression_pca_info_df.to_csv(regression_pca_results_file, index=False)
    else:
        regression_pca_info_df.to_csv(regression_pca_results_file, mode='a', header=False, index=False)

    if not os.path.isfile(regression_results_file):
        regression_info_df.to_csv(regression_results_file, index=False)
    else:
        regression_info_df.to_csv(regression_results_file, mode='a', header=False, index=False)


def multilinear_ridge_lasso_reg(state_name: str, type: str, county_name: str, regularizer_type: str = 'ridge') -> None:
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

    if regularizer_type == 'ridge':
        score, fitted_model = call_multilinear_ridge_regression(X=X, Y=Y)
        Y_pred = fitted_model.predict(X)
    elif regularizer_type == 'lasso':
        score, fitted_model = call_multilinear_lasso_regression(X=X, Y=Y)
        Y_pred = fitted_model.predict(X)


    feature_subset = list(range(X.shape[1]))

    # Get nrmse and rmse
    nrmse = calc_nrmse(Y, Y_pred)
    rmse = calc_rmse(Y, Y_pred)
    # Calculate variance inflation factor to check for colinearity
    vif_list = []
    for idx in feature_subset:
        Y_feat = X[:, idx]
        feature_indices = [idx2 for idx2 in feature_subset if idx2 != idx]
        _, fitted_feature_model = fit_subset(Y=Y_feat, X=X, feature_indices=feature_indices)
        vif_list.append(1.0 / (1 - fitted_feature_model.rsquared))

    # import ipdb
    # ipdb.set_trace()
    # Regress on features and ca
    regression_info_dict = {'state': state_name, 'county': county_name, 'n': Y.shape[0]}
    regression_info_dict['features'] = metadata_keys
    regression_info_dict['coef'] = list(fitted_model.coef_)
    regression_info_dict['vif'] = vif_list
    regression_info_dict['R2'] = score
    regression_info_dict['nrmse'] = nrmse
    regression_info_dict['rmse'] = rmse
    regression_info_df = pd.DataFrame(regression_info_dict)

    if not os.path.isdir(regression_results_path):
        os.mkdir(regression_results_path)

    regression_results_file = path.join(regression_results_path, f'{type}_linear_{regularizer_type}_regression_results.csv')
    if not os.path.isfile(regression_results_file):
        regression_info_df.to_csv(regression_results_file, index=False)
    else:
        regression_info_df.to_csv(regression_results_file, mode='a', header=False, index=False)
