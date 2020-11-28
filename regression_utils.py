# --------------------------
# Standard Python Imports
# --------------------------
import collections
import copy
import joblib
import multiprocessing
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


def construct_x(X: np.array, metadata_keys: List[str], df: pd.DataFrame,
                metadata_filter: List[str]) -> Tuple[np.array, List[str]]:
    idx_meta = 1
    if len(metadata_filter) == 0:
        for idx, key in enumerate(metadata_keys):
            X[:, idx + 1] = df[key].tolist()
        return X, metadata_keys

    metadata_keys_copy = []
    if len(metadata_filter) > 0:
        for idx, key in enumerate(metadata_keys):
            if key in metadata_filter or key.lower() == 'time':
                X[:, idx_meta] = df[key].tolist()
                metadata_keys_copy.append(key)
                idx_meta = idx_meta + 1
        X = X[:, 0:idx_meta]
    return X, metadata_keys_copy


def calc_metric(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    ss_res = np.sum((Y_pred - Y) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def calc_nrmse(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    return np.mean(np.sqrt(((Y - Y_pred) / Y) ** 2))


def calc_rmse(Y: np.ndarray, Y_pred: np.ndarray) -> float:
    return np.mean(np.sqrt(((Y - Y_pred)) ** 2))


def fit_subset(X: np.ndarray, Y: np.ndarray, feature_indices: List[int]) -> Tuple[float, sm.OLS.fit]:
    model = sm.OLS(Y, X[:, feature_indices])
    fitted_model = model.fit()
    Y_pred = fitted_model.fittedvalues
    metric = -calc_nrmse(Y=Y, Y_pred=Y_pred)
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


def load_training_df(state_name: str, type: str, county_names: List[str], ethnicity_filter_list: List[str]) -> pd.DataFrame:
    # Define path and file for training data
    training_data_all_df = None
    for county_name in  county_names:
        training_csv_path = path.join('states', state_name, 'training_data_csvs')

        if county_name is None:
            training_file = f'{state_name}_training_{type}.csv'
        else:
            training_file = f'{state_name}_{county_name}_training_{type}.csv'

        training_data_df = pd.read_csv(path.join(training_csv_path, training_file), index_col=0)
        training_data_df['state'] = [state_name] * len(training_data_df)
        training_data_df['county'] = [county_name] * len(training_data_df)

        # Filter to specific ethnicities
        if len(ethnicity_filter_list) > 0:
            ethnicity_filter_list = [ethnicity.lower() for ethnicity in ethnicity_filter_list]
            ethnicities = training_data_df['ethnicity'].str.lower().tolist()
            ethnicity_bool = [True if ethnicity.lower() in ethnicity_filter_list else False for ethnicity in ethnicities]
            training_data_df = training_data_df[ethnicity_bool]

        if training_data_all_df is None:
            training_data_all_df = training_data_df
        else:
            training_data_all_df = pd.concat([training_data_all_df, training_data_df])
    return training_data_all_df


def get_best_ridge_model(X: np.ndarray, Y: np.ndarray) -> Tuple[float, Ridge.fit]:
    alpha_list = np.linspace(0, 1, 20)
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
    alpha_list = np.linspace(0, 1, 20)
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
    full_subset.remove(0)
    subset_size = 2

    while subset_size <= X.shape[1] and delta_metric > tol:
        fit_subset_sizes(X, Y, subset_size, full_subset, [0], metric_list, fitted_model_list, subsets_list)
        subset_size = subset_size + 1
    best_idx = np.argmax(metric_list)
    best_fitted_model = fitted_model_list[best_idx]
    best_subset = subsets_list[best_idx]
    return best_fitted_model, best_subset


def get_X(training_data_df: pd.DataFrame, filter_list: List[str], metadata_keys: List[str], metadata_filter: List[str]) -> np.array:
    X = np.zeros((training_data_df.shape[0], training_data_df.shape[1] - len(filter_list) + 1))
    X, metadata_keys = construct_x(X=X, metadata_keys=metadata_keys,
                                   df=training_data_df, metadata_filter=metadata_filter)
    metadata_keys.insert(0, 'constant')

    X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
    X[:, 0] = 1
    return X


def initialize_regression_dict(regression_dict: Dict[str, Union[List[float], List[str]]], nrmse: float, rmse: float, features: List[str], fitted_model: sm.OLS.fit, stat_table) -> None:
    regression_dict['features'] = features
    regression_dict['coef'] = list(fitted_model.params)
    regression_dict['low_coef'] = list(fitted_model.params - \
        1.96 * np.sqrt(np.diag(fitted_model.cov_params())))
    regression_dict['upper_coef'] = fitted_model.params + \
        1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    regression_dict['std_err'] = [np.sqrt(np.diag(fitted_model.cov_params()))] * len(features)
    regression_dict['R2'] = [fitted_model.rsquared] * len(features)
    regression_dict['R2-adj'] = [fitted_model.rsquared_adj] * len(features)
    regression_dict['Durbin-Watson'] = [float(stat_table.data[0][-1])] * len(features)
    regression_dict['JB'] = [float(stat_table.data[-3][-1])] * len(features)
    regression_dict['Condition No.'] = [fitted_model.condition_number] * len(features)
    regression_dict['nrmse'] = [nrmse] * len(features)
    regression_dict['rmse'] = [rmse] * len(features)


def extend_regression_dict(regression_dict: Dict[str, Union[List[float], List[str]]], nrmse: float, rmse: float,
                                   features: List[str], fitted_model: sm.OLS.fit, stat_table) -> None:
    regression_dict['features'].extend(features)
    regression_dict['coef'].extend(list(fitted_model.params))
    regression_dict['low_coef'].extend(list(fitted_model.params - \
                                       1.96 * np.sqrt(np.diag(fitted_model.cov_params()))))
    regression_dict['upper_coef'].extend(fitted_model.params + \
                                    1.96 * np.sqrt(np.diag(fitted_model.cov_params())))
    regression_dict['std_err'].extend([np.sqrt(np.diag(fitted_model.cov_params()))] * len(features))
    regression_dict['R2'].extend([fitted_model.rsquared] * len(features))
    regression_dict['R2-adj'].extend([fitted_model.rsquared_adj] * len(features))
    regression_dict['Durbin-Watson'].extend([float(stat_table.data[0][-1])] * len(features))
    regression_dict['JB'].extend([float(stat_table.data[-3][-1])] * len(features))
    regression_dict['Condition No.'].extend([fitted_model.condition_number] * len(features))
    regression_dict['nrmse'].extend([nrmse] * len(features))
    regression_dict['rmse'].extend([rmse] * len(features))


def call_multilinear_regression(X: np.ndarray, Y: np.ndarray) -> Tuple[sm.OLS.fit, List[int]]:
    # fitted_model, feature_subset = get_best_subset(X=X, Y=Y)
    feature_subset = list(range(X.shape[1]))
    metric, fitted_model = fit_subset(X=X, Y=Y, feature_indices=feature_subset)
    return fitted_model, feature_subset


def call_multilinear_ridge_regression(X: np.ndarray, Y: np.ndarray) -> Tuple[float, Ridge.fit]:
    score, fitted_model = get_best_ridge_model(X=X, Y=Y)
    return score, fitted_model


def call_multilinear_lasso_regression(X: np.ndarray, Y: np.ndarray) -> Tuple[float, Ridge.fit]:
    score, fitted_model = get_best_lasso_model(X=X, Y=Y)
    return score, fitted_model


def save_regression_results(df: pd.DataFrame, pred_df: pd.DataFrame, type: str, state_name: str,
                            county_names: List[str], reg_key: str, regression_type: str, ethnicity_filter_list: List[str]) -> None:
    # Save regression results in relevant directory
    regression_results_path = path.join('states', state_name, 'regression_results_csvs', regression_type)
    predictions_path = path.join(regression_results_path, reg_key)

    if not os.path.isdir(regression_results_path):
        os.makedirs(regression_results_path)

    if not os.path.isdir(predictions_path):
        os.makedirs(predictions_path)

    county_name = 'multiple' if len(county_names) > 1 else county_names[0]
    predictions_file = f'{type}_{state_name}_{reg_key}' if county_name is None else f'{type}_{state_name}_{county_name}_{reg_key}'
    if len(ethnicity_filter_list) == 0:
        results_file = f'{type}_{reg_key}_{regression_type}_results.csv'
        predictions_file = f'{predictions_file}.csv'
    else:
        results_file = f'{type}_{reg_key}_{regression_type}_results'
        for ethnicity in ethnicity_filter_list:
            results_file = f'{results_file}_{ethnicity}'
            predictions_file = f'{predictions_file}_{ethnicity}'
        results_file = f'{results_file}.csv'
        predictions_file = f'{predictions_file}.csv'

    regression_results_file = path.join(regression_results_path, results_file)
    if not os.path.isfile(regression_results_file):
        df.to_csv(regression_results_file, index=False)
    else:
        df.to_csv(regression_results_file, mode='a', header=False, index=False)

    predictions_file = path.join(predictions_path, predictions_file)
    if not os.path.isfile(predictions_file):
        pred_df.to_csv(predictions_file, index=False)
    else:
        pred_df.to_csv(predictions_file, mode='a', header=False, index=False)


def multilinear_reg(state_name: str, county_names: List[str], type: str,
                    ethnicity_filter_list: List[str], reg_key: str, metadata_filter: List[str],
                    bootstrap_bool: bool = True, N: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, sm.OLS]:
    training_data_df = load_training_df(state_name=state_name, county_names=county_names, type=type, ethnicity_filter_list=ethnicity_filter_list)

    # Set Y as mortality rate
    Y = np.array(training_data_df[reg_key])

    # Populate remaining columns with corresponding metadata
    filter_list = [
        'covid_perc',
        'dem_perc',
        'mortality_rate',
        'detrended_mortality_rate',
        'discrepancy',
        'y_pred',
        'ethnicity',
        'state',
        'county']
    metadata_keys = training_data_df.keys()
    metadata_keys = [key for key in metadata_keys if key not in filter_list]

    # Construct X
    X = get_X(training_data_df=training_data_df, filter_list=filter_list, metadata_keys=metadata_keys, metadata_filter=metadata_filter)

    fitted_model, feature_subset = call_multilinear_regression(X=X, Y=Y)
    # Get nrmse and rmse
    nrmse = calc_nrmse(Y, fitted_model.fittedvalues)
    rmse = calc_rmse(Y, fitted_model.fittedvalues)
    features = [metadata_keys[idx] for idx in feature_subset]
    stat_table = fitted_model.summary().tables[-1]

    if len(county_names) == 1:
        county_name = county_names[0]
        regression_info_dict = {'state': state_name, 'county': county_name, 'n': Y.shape[0]}
    else:
        regression_info_dict = {'state': state_name, 'county': [[county_names]], 'n': Y.shape[0]}

    regression_info_dict['features'] = features
    regression_info_dict['coef'] = fitted_model.params
    regression_info_dict['lower_coef'] = regression_info_dict['coef'] - \
        1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['upper_coef'] = regression_info_dict['coef'] + \
        1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    # regression_info_dict['vif'] = vif_list
    regression_info_dict['std_err'] = np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['R2'] = fitted_model.rsquared
    regression_info_dict['R2 Adjusted'] = fitted_model.rsquared_adj
    regression_info_dict['Durbin-Watson'] = float(stat_table.data[0][-1])
    regression_info_dict['JB'] = float(stat_table.data[-3][-1])
    regression_info_dict['condition_number'] = fitted_model.condition_number

    regression_info_dict['nrmse'] = nrmse
    regression_info_dict['rmse'] = rmse

    # Get bootstrap values of nrms and rmse if prescribed
    if bootstrap_bool:
        n = list(range(X.shape[0]))
        frac = 0.5
        nrmse_list, rmse_list, indices_list = [], [], []
        for idx in range(N):
            indices = np.random.choice(n, size=int(frac * len(n)), replace=True)
            indices_list.append(indices)
        regr_results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(joblib.delayed(
            call_multilinear_regression)(X=X[indices, :], Y=Y[indices]) for indices in indices_list)
        fitted_model_list, _ = zip(*regr_results)

        for idx in range(N):
            nrmse_bootstrap = calc_nrmse(Y[indices_list[idx]], fitted_model_list[idx].fittedvalues)
            rmse_bootstrap = calc_rmse(Y[indices_list[idx]], fitted_model_list[idx].fittedvalues)
            nrmse_list.append(nrmse_bootstrap)
            rmse_list.append(rmse_bootstrap)
        low_nrmse, up_nrmse = np.percentile(nrmse_list, 2.5), np.percentile(nrmse_list, 97.5)
        low_rmse, up_rmse = np.percentile(rmse_list, 2.5), np.percentile(rmse_list, 97.5)
        regression_info_dict['low_nrmse'], regression_info_dict['up_nrmse'] = low_nrmse, up_nrmse
        regression_info_dict['low_rmse'], regression_info_dict['up_rmse'] = low_rmse, up_rmse

    regression_info_df = pd.DataFrame(regression_info_dict)

    predictions_df = pd.DataFrame({'time': training_data_df['time'].tolist(
    ), 'y': list(Y), 'y_pred': list(fitted_model.fittedvalues), 'state': training_data_df['state'].tolist(), 'county': training_data_df['county'].tolist()})
    return regression_info_df, predictions_df, fitted_model


def multilinear_pca_reg(state_name: str, type: str, county_name: str,
                        ethnicity_filter_list: List[str], reg_key: str, var_thresh: float = 0.95) -> Tuple[np.array, np.array]:
    # Define path and file for training data
    training_csv_path = path.join('states', state_name, 'training_data_csvs')
    regression_results_path = path.join('states', state_name, 'regression_results_csvs')

    if county_name is None:
        training_file = f'{state_name}_training_{type}.csv'
    else:
        training_file = f'{state_name}_{county_name}_training_{type}.csv'

    training_data_df = pd.read_csv(path.join(training_csv_path, training_file), index_col=0)

    # Filter to specific ethnicities
    if len(ethnicity_filter_list) > 0:
        ethnicity_filter_list = [ethnicity.lower() for ethnicity in ethnicity_filter_list]
        ethnicities = training_data_df['ethnicity'].str.lower().tolist()
        ethnicity_bool = [True if ethnicity.lower() in ethnicity_filter_list else False for ethnicity in ethnicities]
        training_data_df = training_data_df[ethnicity_bool]

    # Set Y as mortality rate
    Y = np.array(training_data_df[reg_key])

    # Populate remaining columns with corresponding metadata
    filter_list = [
        'covid_perc',
        'dem_perc',
        'mortality_rate',
        'detrended_mortality_rate',
        'discrepancy',
        'y_pred',
        'ethnicity']
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
    lower_feature_coef = list(coef[0:pca_idx_start] - 1.96 * std_params[0:pca_idx_start]
                              ) + list(np.sum(pca_lower_weighted_components, axis=1))
    upper_feature_coef = list(coef[0:pca_idx_start] + 1.96 * std_params[0:pca_idx_start]
                              ) + list(np.sum(pca_upper_weighted_components, axis=1))

    stat_table = fitted_model.summary().tables[-1]
    # # Calculate variance inflation factor to check for colinearity
    # vif_list = []
    # for idx in feature_subset:
    #     Y_feat = X[:, idx]
    #     feature_indices = [idx2 for idx2 in feature_subset if idx2 != idx]
    #     _, fitted_feature_model = fit_subset(Y=Y_feat, X=X, feature_indices=feature_indices)
    #     vif_list.append(1.0 / (1 - fitted_feature_model.rsquared))

    regression_info_dict = {'state': state_name, 'county': county_name, 'n': Y.shape[0]}
    regression_info_dict['coef'] = fitted_model.params
    regression_info_dict['lower_coef'] = regression_info_dict['coef'] - \
        1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['upper_coef'] = regression_info_dict['coef'] + \
        1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    # regression_info_dict['vif'] = vif_list
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
    if not os.path.isdir(regression_results_path):
        os.mkdir(regression_results_path)

    if len(ethnicity_filter_list) == 0:
        pca_comp_results_file = f'{type}_{reg_key}_pca_comp_linear_regression_results.csv'
        pca_results_file = f'{type}_pca_linear_regression_results.csv'
    else:
        pca_comp_results_file = f'{type}_{reg_key}_pca_comp_linear_regression_results'
        pca_results_file = f'{type}_{reg_key}_pca_linear_regression_results'
        for ethnicity in ethnicity_filter_list:
            pca_comp_results_file = f'{pca_comp_results_file}_{ethnicity}'
            pca_results_file = f'{pca_results_file}_{ethnicity}'
        pca_comp_results_file = f'{pca_comp_results_file}.csv'
        pca_results_file = f'{pca_results_file}.csv'

    regression_pca_results_file = path.join(regression_results_path, pca_comp_results_file)
    regression_results_file = path.join(regression_results_path, pca_results_file)

    if not os.path.isfile(regression_pca_results_file):
        regression_pca_info_df.to_csv(regression_pca_results_file, index=False)
    else:
        regression_pca_info_df.to_csv(regression_pca_results_file, mode='a', header=False, index=False)

    if not os.path.isfile(regression_results_file):
        regression_info_df.to_csv(regression_results_file, index=False)
    else:
        regression_info_df.to_csv(regression_results_file, mode='a', header=False, index=False)
    return Y, fitted_model.fittedvalues


def multilinear_ridge_lasso_reg(state_name: str, type: str, county_names: List[str], ethnicity_filter_list:
                                List[str], reg_key: str, metadata_filter: List[str],
                                bootstrap_bool: bool = True, N: int = 100, regularizer_type: str = 'ridge') -> Tuple[pd.DataFrame, pd.DataFrame, sm.OLS]:
    training_data_df = load_training_df(state_name=state_name, county_names=county_names, type=type, ethnicity_filter_list=ethnicity_filter_list)

    # Set Y as mortality rate
    Y = np.array(training_data_df[reg_key])

    # Populate remaining columns with corresponding metadata
    filter_list = [
        'covid_perc',
        'dem_perc',
        'mortality_rate',
        'detrended_mortality_rate',
        'discrepancy',
        'y_pred',
        'ethnicity'
        'state',
        'county']
    metadata_keys = training_data_df.keys()
    metadata_keys = [key for key in metadata_keys if key not in filter_list]

    # Construct X
    X = get_X(training_data_df=training_data_df, filter_list=filter_list, metadata_keys=metadata_keys, metadata_filter=metadata_filter)


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

    # Regress on features and ca
    if len(county_names) == 1:
        county_name = county_names[0]
        regression_info_dict = {'state': state_name, 'county': county_name, 'n': Y.shape[0]}
    else:
        regression_info_dict = {'state': state_name, 'county': [county_names], 'n': Y.shape[0]}
    regression_info_dict['features'] = metadata_keys
    regression_info_dict['coef'] = list(fitted_model.coef_)
    # regression_info_dict['vif'] = vif_list
    regression_info_dict['R2'] = score
    regression_info_dict['nrmse'] = nrmse
    regression_info_dict['rmse'] = rmse

    # Get bootstrap values of nrms and rmse if prescribed
    if bootstrap_bool:
        n = list(range(X.shape[0]))
        frac = 0.5
        nrmse_list, rmse_list, indices_list = [], [], []
        for idx in range(N):
            indices = np.random.choice(n, size=int(frac * len(n)), replace=True)
            indices_list.append(indices)
        if regularizer_type == 'ridge':
            regr_results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
                joblib.delayed(call_multilinear_ridge_regression)(X=X[indices, :], Y=Y[indices]) for indices in indices_list)
        elif regularizer_type == 'lasso':
            regr_results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
                joblib.delayed(call_multilinear_lasso_regression)(X=X[indices, :], Y=Y[indices]) for indices in indices_list)

        _, fitted_model_list = zip(*regr_results)

        for idx in range(N):
            X_bs = X[indices_list[idx], :]
            Y_pred_bs = fitted_model_list[idx].predict(X_bs)
            nrmse_bootstrap = calc_nrmse(Y[indices_list[idx]], Y_pred_bs)
            rmse_bootstrap = calc_rmse(Y[indices_list[idx]], Y_pred_bs)
            nrmse_list.append(nrmse_bootstrap)
            rmse_list.append(rmse_bootstrap)
        low_nrmse, up_nrmse = np.percentile(nrmse_list, 2.5), np.percentile(nrmse_list, 97.5)
        low_rmse, up_rmse = np.percentile(rmse_list, 2.5), np.percentile(rmse_list, 97.5)
        regression_info_dict['low_nrmse'], regression_info_dict['up_nrmse'] = low_nrmse, up_nrmse
        regression_info_dict['low_rmse'], regression_info_dict['up_rmse'] = low_rmse, up_rmse

    regression_info_df = pd.DataFrame(regression_info_dict)

    predictions_df = pd.DataFrame({'time': training_data_df['time'].tolist(), 'y': list(Y), 'y_pred': list(Y_pred), 'state': training_data_df['state'].tolist(), 'county': training_data_df['county'].tolist()})
    return regression_info_df, predictions_df, fitted_model
