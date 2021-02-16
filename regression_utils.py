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
from joblib import dump, load
import numpy as np
import pandas as pd
from statsmodels.iolib.smpickle import load_pickle
import statsmodels.api as sm
import statsmodels.regression.linear_model as lm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct, WhiteKernel
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV

# --------------------------
# covid19Tracking Imports
# --------------------------
import managers as managers_lib
import utils_lib


def construct_x(X: np.array, metadata_keys: List[str], df: pd.DataFrame,
                metadata_filter: List[str]) -> Tuple[np.array, List[str]]:
    """
    Create feature array to be regressed on, validated, tested, etc...

    Arguments:
        X: Empty feature array
        metadata_keys: Metadata used for regression
        df: Data frame containing metadata information used to populate X
        metadata_filter: Metadata list that contains information to be included in X

    Returns:
         X: Feature array
         metadata_keys: Metadata used in feature array
    """
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


def fit_subset(X: np.ndarray, Y: np.ndarray,
               feature_indices: List[int], weight_list: List[float]) -> Tuple[float, sm.OLS.fit]:
    """
    Fit Y to X using weighted least squares

    Arguments:
        X: Features to be fit
        Y: Dependent variable
        feature_indices: Indices of features to be fit
        weight_list: Weighting of examples to be fit
    Returns:
        metric: Metric of accuracy of the fitted model
        model: Model that has been fit to data
    """
    if len(weight_list) == 0:
        weight_list = 1.0
    model = lm.WLS(Y, X[:, feature_indices], weights=weight_list)
    fitted_model = model.fit()
    Y_pred = fitted_model.fittedvalues
    metric = -calc_nrmse(Y=Y, Y_pred=Y_pred)
    return metric, fitted_model


def fit_subset_sizes(X, Y, subset_size, full_subset, curr_subset, metric_list, fitted_model_list, subsets_list):
    """
    Fit best subset of features (based on metric)

    Arguments:
        X: Features to be fitted
        Y: Dependent variable used for fitting
        subset_size: Size of subset of features to be fit
        full_subset: Full subset of features to be fit
        curr_subset: Subset being fitted in current recursive call
        metric_list: List containing metric of each fitted model
        fitted_model_list: List of fitted model
        subsets_list: List of feature subsets that will be fitted
    """
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


def load_data_df(state_name: str, type: str, county_names: List[str], ethnicity_filter_list: List[str]) -> pd.DataFrame:
    """
    Load data frame containing relevant information about a state/county

    Arguments:
        state_name: State for which data will be loaded
        type: Type of data loaded ('cases' or 'deaths')
        county_names: List of counties for which data will be loaded
        ethnicity_filter_list: List of ethnicities for which data will be loaded

    Returns:
        data_all_df: Dataframe containing data
    """
    # Define path and file for training data
    data_all_df = None
    if isinstance(county_names, str):
        county_names = [county_names]
    for county_name in county_names:
        try:
            training_csv_path = path.join('states', state_name, 'training_data_csvs')
            if county_name is None or county_name == 'None':
                training_file = f'{state_name}_training_{type}.csv'
            else:
                training_file = f'{state_name}_{county_name}_training_{type}.csv'

            data_df = pd.read_csv(path.join(training_csv_path, training_file), index_col=0)
            data_df['state'] = [state_name] * len(data_df)
            data_df['county'] = [county_name] * len(data_df)

            # Filter to specific ethnicities
            if len(ethnicity_filter_list) > 0:
                ethnicity_filter_list = [ethnicity.lower() for ethnicity in ethnicity_filter_list]
                ethnicities = data_df['ethnicity'].str.lower().tolist()
                ethnicity_bool = [
                    True if ethnicity.lower() in ethnicity_filter_list else False for ethnicity in ethnicities]
                data_df = data_df[ethnicity_bool]

            if data_all_df is None:
                data_all_df = data_df
            else:
                data_all_df = pd.concat([data_all_df, data_df])
        except BaseException:
            pass
    return data_all_df


def get_best_ridge_model(X: np.ndarray, Y: np.ndarray, val_X: np.ndarray, val_Y: np.ndarray,
                         weight_list: List[float]) -> Tuple[float, Ridge.fit]:
    alpha_list = np.linspace(0, 1, 20)
    fitted_model_list, score_list = [], []
    for alpha in alpha_list:
        model = Ridge(alpha=alpha)
        fitted_model = model.fit(X, Y, sample_weight=weight_list)
        if val_X is None:
            score = model.score(X, Y)
        else:
            score = model.score(val_X, val_Y)
        fitted_model_list.append(fitted_model)
        score_list.append(score)
    max_idx = np.argmax(score_list)
    best_fitted_model = fitted_model_list[max_idx]
    best_score = score_list[max_idx]
    return best_score, best_fitted_model


def get_best_lasso_model(X: np.ndarray, Y: np.ndarray, val_X: np.ndarray, val_Y: np.ndarray,
                         weight_list: List[float]) -> Tuple[float, Ridge.fit]:
    alpha_list = np.logspace(-3, 0, 30)
    fitted_model_list, score_list = [], []

    for alpha in alpha_list:
        # print(f'Alpha: {alpha}')
        model = Lasso(alpha=alpha)
        fitted_model = model.fit(X, Y, sample_weight=weight_list)
        if val_X is None:
            Y_pred = model.predict(X)
            rmse = calc_rmse(Y, Y_pred)
            score = model.score(X, Y)
        else:
            Y_pred = model.predict(val_X)
            rmse = calc_rmse(val_Y, Y_pred)
            score = model.score(val_X, val_Y)
        fitted_model_list.append(fitted_model)
        score_list.append(rmse)
        # print(f'rmse: {rmse} \n')

    max_idx = np.argmin(score_list)
    best_fitted_model = fitted_model_list[max_idx]
    best_score = score_list[max_idx]
    return best_score, best_fitted_model


def get_best_gp_model(X: np.ndarray, Y: np.ndarray, val_X: np.ndarray, val_Y: np.ndarray) -> Tuple[float, Ridge.fit]:
    sigma_list = np.logspace(-2, 2, 20)
    fitted_model_list, score_list = [], []
    for sigma in sigma_list:
        print(f'GP fit sigma {sigma}')
        kernel = DotProduct(sigma_0=sigma) + WhiteKernel()  # C(1.0, (1e-3, 1e4)) * RBF(1.0, (1e-3, 1e4))
        # kernel = C(1.0, (1e-3, 1e4)) * RBF(sigma, (1e-3, 1e4))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        model = model.fit(X, Y)
        if val_X is not None:
            score = model.score(val_X, val_Y)
            Y_pred, std_pred = model.predict(val_X, return_std=True)
            # nrmse = calc_nrmse(val_Y, Y_pred)
            rmse = calc_rmse(val_Y, Y_pred)
        else:
            score = model.score(X, Y)
            # Y_pred, std_pred = model.predict(X, return_std=True)
            # nrmse = calc_nrmse(Y, Y_pred)
            # rmse = calc_rmse(Y, Y_pred)
        print(f'Score {rmse} \n')
        fitted_model_list.append(model)
        score_list.append(rmse)

    max_idx = np.argmin(score_list)
    best_fitted_model = fitted_model_list[max_idx]
    best_score = score_list[max_idx]
    return best_score, best_fitted_model


def get_best_mlp_model(X: np.ndarray, Y: np.ndarray, val_X: np.ndarray, val_Y: np.ndarray) -> Tuple[float, Ridge.fit]:
    sigma_list = np.logspace(-1, 1, 20)
    max_iter_list = [500, 1000, 2000, 5000]
    n_hidden_layers_list = [10, 100, 1000]
    fitted_model_list, score_list = [], []
    for sigma in sigma_list:
        for max_iter in [max_iter_list[0]]:
            for n_hidden_layers in [n_hidden_layers_list[0]]:
                print(f'Fitting sigma: {sigma} max_iter: {max_iter} hidden layers: {n_hidden_layers}GP')
                model = MLPRegressor(random_state=1, max_iter=max_iter, alpha=sigma, solver='lbfgs', hidden_layer_sizes=n_hidden_layers)
                model = model.fit(X, Y)

                if val_X is not None:
                    Y_pred = model.predict(val_X)
                    nrmse = calc_nrmse(val_Y, Y_pred)
                    rmse = calc_rmse(val_Y, Y_pred)
                else:
                    Y_pred = model.predict(X)
                    nrmse = calc_nrmse(Y, Y_pred)
                    rmse = calc_rmse(Y, Y_pred)
                print(f'rmse is {rmse} \n')
                fitted_model_list.append(model)
                score_list.append(rmse)

    max_idx = np.argmin(score_list)
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


def get_X(training_data_df: pd.DataFrame, filter_list: List[str], metadata_keys: List[str], metadata_filter: List[str],
          mu_X: np.array = None, std_X: np.array = None, weight_list: List[float] = []) -> Tuple[np.array, List[str], np.array, np.array]:
    X = np.zeros((training_data_df.shape[0], training_data_df.shape[1] - len(filter_list) + 1))
    X, metadata_keys = construct_x(X=X, metadata_keys=metadata_keys,
                                   df=training_data_df, metadata_filter=metadata_filter)
    metadata_keys.insert(0, 'constant')

    if mu_X is None or std_X is None:
        std_X = np.std(X[:, 1:], axis=0)
        mu_X = np.mean(X[:, 1:], axis=0)
        for idx in range(len(std_X)):
            std_X[idx] = 1 if std_X[idx] < 1e-5 else std_X[idx]
            mu_X[idx] = 0 if std_X[idx] == 1 else mu_X[idx]
    X[:, 1:] = (X[:, 1:] - mu_X) / std_X
    X[:, 0] = 1
    return X, metadata_keys, mu_X, std_X


def call_multilinear_regression(X: np.ndarray, Y: np.ndarray,
                                weight_list: List[float] = []) -> Tuple[sm.OLS.fit, List[int]]:
    feature_subset = list(range(X.shape[1]))
    weight_list = list(weight_list)
    metric, fitted_model = fit_subset(X=X, Y=Y, feature_indices=feature_subset, weight_list=weight_list)
    return fitted_model, feature_subset


def call_multilinear_ridge_regression(X: np.ndarray, Y: np.ndarray, val_X: np.ndarray = None,
                                      val_Y: np.ndarray = None, weight_list: List[float] = None) -> Tuple[float, Ridge.fit]:
    weight_list = list(weight_list)
    score, fitted_model = get_best_ridge_model(X=X, Y=Y, val_X=val_X, val_Y=val_Y, weight_list=weight_list)
    return score, fitted_model


def call_multilinear_lasso_regression(X: np.ndarray, Y: np.ndarray, val_X: np.ndarray = None,
                                      val_Y: np.ndarray = None, weight_list: List[float] = None) -> Tuple[float, Lasso.fit]:
    score, fitted_model = get_best_lasso_model(X=X, Y=Y, val_X=val_X, val_Y=val_Y, weight_list=weight_list)
    return score, fitted_model


def save_regression_results(df: pd.DataFrame, pred_df: pd.DataFrame, type: str, state_name: str,
                            county_names: List[str], reg_key: str, regression_type: str, ethnicity_filter_list: List[str],
                            validate_state_name: str, validate_county_names: List[str], val_info_df: pd.DataFrame, val_predictions_df: pd.DataFrame, fitted_model: Union[sm.OLS, Ridge.fit, Lasso.fit]) -> None:

    # Create regression and validation directories
    regression_results_path = path.join('states', state_name, 'regression_results_csvs', regression_type)
    predictions_path = path.join(regression_results_path, reg_key)
    model_path = path.join(regression_results_path, reg_key, 'model')

    utils_lib.create_dir_if_not_exists(regression_results_path)
    utils_lib.create_dir_if_not_exists(predictions_path)
    utils_lib.create_dir_if_not_exists(model_path)

    # Construction prediction and regression file names for training data
    county_name = '_'.join([county_name for county_name in county_names]) if len(county_names) > 1 else county_names[0]
    predictions_file = f'{type}_{state_name}_{reg_key}' if county_name is None else f'{type}_{state_name}_{county_name}_{reg_key}'
    results_file = f'{type}_{reg_key}_{regression_type}_results'
    model_file = f'{type}_{state_name}_{reg_key}' if county_name is None else f'{type}_{state_name}_{county_name}_{reg_key}_models'

    predictions_file = utils_lib.create_files_name_with_ethnicity(
        file=predictions_file, ethnicity_filter_list=ethnicity_filter_list)
    results_file = utils_lib.create_files_name_with_ethnicity(
        file=results_file, ethnicity_filter_list=ethnicity_filter_list)

    # Save results
    utils_lib.save_df_to_path(df=df, path=regression_results_path, file=results_file)
    utils_lib.save_df_to_path(df=pred_df, path=predictions_path, file=predictions_file)

    if regression_type in managers_lib.RegDefinitions.multilinear_list or regression_type == 'gp' or regression_type == 'mlp':
        ext = 'pickle'
        model_file = utils_lib.create_files_name_with_ethnicity(file=model_file,
                                                                ethnicity_filter_list=ethnicity_filter_list, ext=ext)
        if regression_type == 'gp' or regression_type == 'mlp':
            dump(fitted_model, f'{model_path}/{model_file}')
        else:
            fitted_model.save(f'{model_path}/{model_file}')
    elif regression_type in managers_lib.RegDefinitions.multilinear_lasso_list or regression_type in managers_lib.RegDefinitions.multilinear_ridge_list:
        ext = 'sav'
        model_file = utils_lib.create_files_name_with_ethnicity(file=model_file,
                                                                ethnicity_filter_list=ethnicity_filter_list, ext=ext)
        dump(fitted_model, f'{model_path}/{model_file}')

    if len(validate_county_names) > 0 and validate_state_name is not None:
        validation_results_path = path.join('states', validate_state_name, 'val_results_csvs', regression_type)
        val_predictions_path = path.join(validation_results_path, reg_key)
        val_model_path = path.join(validation_results_path, reg_key, 'model')

        utils_lib.create_dir_if_not_exists(validation_results_path)
        utils_lib.create_dir_if_not_exists(val_predictions_path)
        utils_lib.create_dir_if_not_exists(val_model_path)

        val_county_name = '_'.join([county_name for county_name in validate_county_names]) if len(
            validate_county_names) > 1 else validate_county_names[0]
        val_results_file = f'{type}_{reg_key}_{regression_type}_{validate_state_name}' if val_county_name is None else f'{type}_{reg_key}_{regression_type}_{validate_state_name}_{val_county_name}_results'
        val_predictions_file = f'{type}_{validate_state_name}_{reg_key}' if county_name is None else f'{type}_{validate_state_name}_{val_county_name}_{reg_key}'
        val_model_file = f'{type}_{validate_state_name}' if val_county_name is None else f'{type}_{validate_state_name}_{val_county_name}'

        val_results_file = f'{val_results_file}_train_{state_name}_results' if county_name is None else f'{val_results_file}_train_{state_name}_{county_name}_results'
        val_predictions_file = f'{val_predictions_file}_train_{state_name}' if county_name is None else f'{val_predictions_file}_train_{state_name}_{county_name}'
        val_model_file = f'{val_model_file}_train_{state_name}_{reg_key}' if county_name is None else f'{val_model_file}_train_{state_name}_{county_name}_{reg_key}'

        val_predictions_file = utils_lib.create_files_name_with_ethnicity(
            file=val_predictions_file, ethnicity_filter_list=ethnicity_filter_list)
        val_results_file = utils_lib.create_files_name_with_ethnicity(
            file=val_results_file, ethnicity_filter_list=ethnicity_filter_list)
        # val_model_file = utils_lib.create_files_name_with_ethnicity(file=val_model_file, ethnicity_filter_list=ethnicity_filter_list, ext=ext)

        utils_lib.save_df_to_path(df=val_info_df, path=validation_results_path, file=val_results_file)
        utils_lib.save_df_to_path(df=val_predictions_df, path=val_predictions_path, file=val_predictions_file)

        if regression_type in managers_lib.RegDefinitions.multilinear_list or regression_type == 'gp' or regression_type == 'mlp':
            ext = 'pickle'
            val_model_file = utils_lib.create_files_name_with_ethnicity(file=val_model_file,
                                                                        ethnicity_filter_list=ethnicity_filter_list,
                                                                        ext=ext)
            if regression_type == 'gp' or regression_type == 'mlp':
                dump(fitted_model, f'{val_model_path}/{val_model_file}')
            else:
                fitted_model.save(f'{val_model_path}/{val_model_file}')
        elif regression_type in managers_lib.RegDefinitions.multilinear_lasso_list or regression_type in managers_lib.RegDefinitions.multilinear_ridge_list:
            ext = 'sav'
            val_model_file = utils_lib.create_files_name_with_ethnicity(file=val_model_file,
                                                                        ethnicity_filter_list=ethnicity_filter_list,
                                                                        ext=ext)
            dump(fitted_model, f'{val_model_path}/{val_model_file}')


def save_test_results(test_pred_df: pd.DataFrame, type: str,
                      reg_key: str, regression_type: str, ethnicity_filter_list: List[str],
                      state_name: str, county_names: List[str], validate_state_name: str, validate_county_names: List[str],
                      test_state_name: str, test_county_names: List[str], test_info_df: pd.DataFrame) -> None:

    test_results_path = path.join('states', test_state_name, 'test_results_csvs', regression_type)
    test_predictions_path = path.join(test_results_path, reg_key)

    utils_lib.create_dir_if_not_exists(test_results_path)
    utils_lib.create_dir_if_not_exists(test_predictions_path)

    train_county_name = '_'.join([county_name for county_name in county_names]
                                 ) if len(county_names) > 1 else county_names[0]
    test_results_file = f'{type}_{reg_key}_{regression_type}_train_{state_name}' if train_county_name is None else f'{type}_{reg_key}_{regression_type}_train_{state_name}_{train_county_name}'
    test_predictions_file = f'{type}_train_{state_name}' if train_county_name is None else f'{type}_train_{state_name}_{train_county_name}'

    val_county_name = '_'.join([county_name for county_name in validate_county_names]) if len(
        validate_county_names) > 1 else validate_county_names[0]
    test_results_file = f'{test_results_file}_val_{validate_state_name}' if val_county_name is None else f'{test_results_file}_val_{validate_state_name}_{val_county_name}'
    test_predictions_file = f'{test_predictions_file}_val_{validate_state_name}' if val_county_name is None else f'{test_predictions_file}_val_{validate_state_name}_{val_county_name}'

    test_county_name = '_'.join([county_name for county_name in test_county_names]
                                ) if len(test_county_names) > 1 else test_county_names[0]
    test_results_file = f'{test_results_file}_test_{test_state_name}' if test_county_name is None else f'{test_results_file}_test_{test_state_name}_{test_county_name}'
    test_predictions_file = f'{test_predictions_file}_test_{test_state_name}_{reg_key}' if test_county_name is None else f'{test_predictions_file}_test_{test_state_name}_{test_county_name}_{reg_key}'

    test_predictions_file = utils_lib.create_files_name_with_ethnicity(
        file=test_predictions_file, ethnicity_filter_list=ethnicity_filter_list)
    test_results_file = utils_lib.create_files_name_with_ethnicity(
        file=test_results_file, ethnicity_filter_list=ethnicity_filter_list)

    utils_lib.save_df_to_path(df=test_info_df, path=test_results_path, file=test_results_file)
    utils_lib.save_df_to_path(df=test_pred_df, path=test_predictions_path, file=test_predictions_file)


def gp_reg(state_name: str, county_names: List[str], type: str,
           ethnicity_filter_list: List[str], reg_key: str, metadata_filter: List[str], validate_state_name: str, validate_county_names: List[str],
           bootstrap_bool: bool = False, N: int = 3, weight_by_time: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, sm.OLS, pd.DataFrame, pd.DataFrame]:

    training_data_df = load_data_df(state_name=state_name, county_names=county_names,
                                    type=type, ethnicity_filter_list=ethnicity_filter_list)

    if weight_by_time:
        time_arr = np.array(training_data_df['time'].tolist())
        max_time = max(time_arr)
        weight_list = list(np.exp((time_arr - max_time) * 0))
        training_data_df = training_data_df[np.array(weight_list) >= 0.5]
    else:
        weight_list = None

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
        'county',
        'date',
        'POPULATION_GENDER_FEMALE', 'EMPLOYED_POPULATION_CHARACTERISTICS_OVERALL', 'EMPLOYED_POPULATION_CHARACTERISTICS_Private_Wage_Salary', 'EMPLOYED_POPULATION_CHARACTERISTICS_Government', 'EMPLOYED_POPULATION_CHARACTERISTICS_Self Employed', 'EMPLOYED_POPULATION_CHARACTERISTICS_Unpaid Family', 'EMPLOYED_POPULATION_CHARACTERISTICS_Management Business', 'EMPLOYED_POPULATION_CHARACTERISTICS_Service Occupations', 'EMPLOYED_POPULATION_CHARACTERISTICS_Sales', 'EMPLOYED_POPULATION_CHARACTERISTICS_Natural Resources', 'EMPLOYED_POPULATION_CHARACTERISTICS_Production', 'EMPLOYED_POPULATION_CHARACTERISTICS_Agricilture', 'EMPLOYED_POPULATION_CHARACTERISTICS_Construction', 'EMPLOYED_POPULATION_CHARACTERISTICS_Manufacturing', 'EMPLOYED_POPULATION_CHARACTERISTICS_Wholesale_Trade', 'EMPLOYED_POPULATION_CHARACTERISTICS_Retail', 'EMPLOYED_POPULATION_CHARACTERISTICS_Transportation', 'EMPLOYED_POPULATION_CHARACTERISTICS_Information', 'EMPLOYED_POPULATION_CHARACTERISTICS_Finance_and_Insurance', 'EMPLOYED_POPULATION_CHARACTERISTICS_Professional_Scientific_and_Management', 'EMPLOYED_POPULATION_CHARACTERISTICS_Educational_Services_and_Health_Care', 'EMPLOYED_POPULATION_CHARACTERISTICS_Arts_Entertainment_and_Recreation', 'EMPLOYED_POPULATION_CHARACTERISTICS_Other', 'EMPLOYED_POPULATION_CHARACTERISTICS_Public_Administration', 'POPULATION_RACE_BLACK', 'POPULATION_RACE_HISPANIC', 'POPULATION_RACE_WHITE', 'POPULATION_RACE_ASIAN']
    metadata_keys = training_data_df.keys()
    metadata_keys = [key for key in metadata_keys if key not in filter_list]
    val_metadata_keys = copy.deepcopy(metadata_keys)

    # Construct X
    X, metadata_keys, mu_X, std_X = get_X(
        training_data_df=training_data_df, filter_list=filter_list, metadata_keys=metadata_keys, metadata_filter=metadata_filter)

    # Get relevant information for the validation set if needed
    if validate_state_name is not None:
        val_data_df = load_data_df(state_name=validate_state_name, county_names=validate_county_names, type=type,
                                   ethnicity_filter_list=ethnicity_filter_list)

        time_arr = np.array(val_data_df['time'].tolist())
        max_time = max(time_arr)
        val_weight_list = list(np.exp((time_arr - max_time) * 0))
        val_data_df = val_data_df[np.array(val_weight_list) >= 0.5]

        # Set Y as relevant key
        val_Y = np.array(val_data_df[reg_key])
        val_X, val_metadata_keys, mu_X, std_X = get_X(training_data_df=val_data_df, filter_list=filter_list,
                                                      metadata_keys=val_metadata_keys,
                                                      metadata_filter=metadata_filter, mu_X=mu_X, std_X=std_X)
    else:
        val_X, val_Y = None, None

    # kernel = kernel = DotProduct() + WhiteKernel() #C(1.0, (1e-3, 1e4)) * RBF(1.0, (1e-3, 1e4))
    # gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    # gp.fit(X, Y)
    # Y_pred, std_pred = gp.predict(X, return_std=True)
    nrmse, gp = get_best_gp_model(X=X, Y=Y, val_X=val_X, val_Y=val_Y)
    # Get nrmse and rmse
    Y_pred, std_pred = gp.predict(X, return_std=True)
    nrmse = calc_nrmse(Y, Y_pred)
    rmse = calc_rmse(Y, Y_pred)


    if len(county_names) == 1:
        county_name = county_names[0]
        regression_info_dict = {'state': [state_name], 'county': [county_name], 'n': [Y.shape[0]]}
    else:
        regression_info_dict = {'state': [state_name], 'county': [','.join(county_names)], 'n': [Y.shape[0]]}

    regression_info_dict['nrmse'] = [nrmse]
    regression_info_dict['rmse'] = [rmse]

    # Get bootstrap values of nrms and rmse if prescribed
    if bootstrap_bool:
        def bs_gp_fit(X_bs, Y_bs):
            kernel_bs = C(1.0, (1e-3, 1e4)) * RBF(1.0, (1e-3, 1e4))
            gp_bs = GaussianProcessRegressor(kernel=kernel_bs, n_restarts_optimizer=10, normalize_y=True)
            gp_bs.fit(X_bs, Y_bs)
            return gp_bs

        n = list(range(X.shape[0]))
        frac = 0.5
        nrmse_list, rmse_list, indices_list = [], [], []
        for idx in range(N):
            indices = np.random.choice(n, size=int(frac * len(n)), replace=True)
            indices_list.append(indices)

        regr_results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(joblib.delayed(
            bs_gp_fit)(X_bs=X[indices, :], Y_bs=Y[indices]) for indices in indices_list)
        gp_list = regr_results

        for idx in range(N):
            Y_pred_bs = gp_list[idx].predict(X[indices_list[idx], :])
            nrmse_bootstrap = calc_nrmse(Y[indices_list[idx]], Y_pred_bs)
            rmse_bootstrap = calc_rmse(Y[indices_list[idx]], Y_pred_bs)
            nrmse_list.append(nrmse_bootstrap)
            rmse_list.append(rmse_bootstrap)
        low_nrmse, up_nrmse = np.percentile(nrmse_list, 2.5), np.percentile(nrmse_list, 97.5)
        low_rmse, up_rmse = np.percentile(rmse_list, 2.5), np.percentile(rmse_list, 97.5)
        regression_info_dict['low_nrmse'], regression_info_dict['up_nrmse'] = [low_nrmse], [up_nrmse]
        regression_info_dict['low_rmse'], regression_info_dict['up_rmse'] = [low_rmse], [up_rmse]

    regression_info_df = pd.DataFrame(regression_info_dict)
    predictions_df = pd.DataFrame({'time': training_data_df['time'].tolist(
    ), 'date': training_data_df['date'].tolist(), 'y': list(Y), 'y_pred': list(Y_pred), 'std_pred': list(std_pred), 'state': training_data_df['state'].tolist(), 'county': training_data_df['county'].tolist(),
        'ethnicity': training_data_df['ethnicity'].tolist()})

    val_info_df = None
    val_predictions_df = None
    if validate_state_name is not None:
        # val_data_df = load_data_df(state_name=validate_state_name, county_names=validate_county_names, type=type,
        #                            ethnicity_filter_list=ethnicity_filter_list)
        #
        # # Set Y as relevant key
        # val_Y = np.array(val_data_df[reg_key])
        # val_X, val_metadata_keys, mu_X, std_X = get_X(training_data_df=val_data_df, filter_list=filter_list, metadata_keys=val_metadata_keys,
        #                                               metadata_filter=metadata_filter, mu_X=mu_X, std_X=std_X)

        val_Y_pred = gp.predict(val_X)
        val_nrmse = calc_nrmse(val_Y, val_Y_pred)
        val_rmse = calc_rmse(val_Y, val_Y_pred)
        val_info_df = pd.DataFrame({'nrmse': [val_nrmse], 'rmse': [val_rmse], 'state': ','.join(set(val_data_df['state'].tolist())), 'county':
                                    ','.join(set(val_data_df['county'].tolist())), 'train_state': ','.join(set(training_data_df['state'].tolist())), 'train_county':
                                    ','.join(set(training_data_df['county'].tolist())), 'mu_X': [mu_X], 'std_X': [std_X]})
        val_predictions_df = pd.DataFrame({'time': val_data_df['time'].tolist(
        ), 'date': val_data_df['date'].tolist(), 'y_val': list(val_Y), 'y_val_pred': list(val_Y_pred), 'state': val_data_df['state'].tolist(),
            'county': val_data_df['county'].tolist(), 'ethnicity': val_data_df['ethnicity'].tolist()})

    return regression_info_df, predictions_df, gp, val_info_df, val_predictions_df


def mlp_reg(state_name: str, county_names: List[str], type: str,
            ethnicity_filter_list: List[str], reg_key: str, metadata_filter: List[str], validate_state_name: str, validate_county_names: List[str],
            bootstrap_bool: bool = False, N: int = 3, weight_by_time: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, sm.OLS, pd.DataFrame, pd.DataFrame]:

    training_data_df = load_data_df(state_name=state_name, county_names=county_names,
                                    type=type, ethnicity_filter_list=ethnicity_filter_list)

    if weight_by_time:
        time_arr = np.array(training_data_df['time'].tolist())
        max_time = max(time_arr)
        weight_list = list(np.exp((time_arr - max_time) * 0))
        training_data_df = training_data_df[np.array(weight_list) >= 0.5]
    else:
        weight_list = None

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
        'county',
        'date']
    metadata_keys = training_data_df.keys()
    metadata_keys = [key for key in metadata_keys if key not in filter_list]
    val_metadata_keys = copy.deepcopy(metadata_keys)

    # Construct X
    X, metadata_keys, mu_X, std_X = get_X(
        training_data_df=training_data_df, filter_list=filter_list, metadata_keys=metadata_keys, metadata_filter=metadata_filter)

    # Get relevant information for the validation set if needed
    if validate_state_name is not None:
        val_data_df = load_data_df(state_name=validate_state_name, county_names=validate_county_names, type=type,
                                   ethnicity_filter_list=ethnicity_filter_list)

        time_arr = np.array(val_data_df['time'].tolist())
        max_time = max(time_arr)
        val_weight_list = list(np.exp((time_arr - max_time) * 0))
        val_data_df = val_data_df[np.array(val_weight_list) >= 0.5]

        # Set Y as relevant key
        val_Y = np.array(val_data_df[reg_key])
        val_X, val_metadata_keys, mu_X, std_X = get_X(training_data_df=val_data_df, filter_list=filter_list,
                                                      metadata_keys=val_metadata_keys,
                                                      metadata_filter=metadata_filter, mu_X=mu_X, std_X=std_X)
    else:
        val_X, val_Y = None, None
    if val_X is None:
        ln = X.shape[0]
        training_data_df_1 = training_data_df[0:int(0.8 * ln)]
        X1, Y1 = X[0:int(0.8 * ln), :], Y[0:int(0.8 * ln)]
        training_data_df_2 = training_data_df[int(0.8 * ln):]
        X2, Y2 = X[int(0.8 * ln):, :], Y[int(0.8 * ln):]

        X, Y = X2, Y2
        training_data_df = training_data_df_2

    nrmse, mlp = get_best_mlp_model(X=X1, Y=Y1, val_X=val_X, val_Y=val_Y)


    # Get nrmse and rmse
    Y_pred = mlp.predict(X)
    std_pred = np.ones(np.shape(Y_pred))
    nrmse = calc_nrmse(Y, Y_pred)
    rmse = calc_rmse(Y, Y_pred)
    import ipdb
    ipdb.set_trace()
    if len(county_names) == 1:
        county_name = county_names[0]
        regression_info_dict = {'state': [state_name], 'county': [county_name], 'n': [Y.shape[0]]}
    else:
        regression_info_dict = {'state': [state_name], 'county': [','.join(county_names)], 'n': [Y.shape[0]]}

    regression_info_dict['nrmse'] = [nrmse]
    regression_info_dict['rmse'] = [rmse]

    # Get bootstrap values of nrms and rmse if prescribed
    if bootstrap_bool:
        def bs_mlp_fit(X_bs, Y_bs):
            kernel_bs = C(1.0, (1e-3, 1e4)) * RBF(1.0, (1e-3, 1e4))
            mlp_bs = GaussianProcessRegressor(kernel=kernel_bs, n_restarts_optimizer=10, normalize_y=True)
            mlp_bs.fit(X_bs, Y_bs)
            return mlp_bs

        n = list(range(X.shape[0]))
        frac = 0.5
        nrmse_list, rmse_list, indices_list = [], [], []
        for idx in range(N):
            indices = np.random.choice(n, size=int(frac * len(n)), replace=True)
            indices_list.append(indices)

        regr_results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(joblib.delayed(
            bs_mlp_fit)(X_bs=X[indices, :], Y_bs=Y[indices]) for indices in indices_list)
        gp_list = regr_results

        for idx in range(N):
            Y_pred_bs = gp_list[idx].predict(X[indices_list[idx], :])
            nrmse_bootstrap = calc_nrmse(Y[indices_list[idx]], Y_pred_bs)
            rmse_bootstrap = calc_rmse(Y[indices_list[idx]], Y_pred_bs)
            nrmse_list.append(nrmse_bootstrap)
            rmse_list.append(rmse_bootstrap)
        low_nrmse, up_nrmse = np.percentile(nrmse_list, 2.5), np.percentile(nrmse_list, 97.5)
        low_rmse, up_rmse = np.percentile(rmse_list, 2.5), np.percentile(rmse_list, 97.5)
        regression_info_dict['low_nrmse'], regression_info_dict['up_nrmse'] = [low_nrmse], [up_nrmse]
        regression_info_dict['low_rmse'], regression_info_dict['up_rmse'] = [low_rmse], [up_rmse]

    regression_info_df = pd.DataFrame(regression_info_dict)
    predictions_df = pd.DataFrame({'time': training_data_df['time'].tolist(
    ), 'date': training_data_df['date'].tolist(), 'y': list(Y), 'y_pred': list(Y_pred), 'std_pred': list(std_pred), 'state': training_data_df['state'].tolist(), 'county': training_data_df['county'].tolist(),
        'ethnicity': training_data_df['ethnicity'].tolist()})

    val_info_df = None
    val_predictions_df = None
    if validate_state_name is not None:
        # val_data_df = load_data_df(state_name=validate_state_name, county_names=validate_county_names, type=type,
        #                            ethnicity_filter_list=ethnicity_filter_list)
        #
        # # Set Y as relevant key
        # val_Y = np.array(val_data_df[reg_key])
        # val_X, val_metadata_keys, mu_X, std_X = get_X(training_data_df=val_data_df, filter_list=filter_list, metadata_keys=val_metadata_keys,
        #                                               metadata_filter=metadata_filter, mu_X=mu_X, std_X=std_X)

        val_Y_pred = mlp.predict(val_X)
        val_nrmse = calc_nrmse(val_Y, val_Y_pred)
        val_rmse = calc_rmse(val_Y, val_Y_pred)
        val_info_df = pd.DataFrame({'nrmse': [val_nrmse], 'rmse': [val_rmse], 'state': ','.join(set(val_data_df['state'].tolist())), 'county':
                                    ','.join(set(val_data_df['county'].tolist())), 'train_state': ','.join(set(training_data_df['state'].tolist())), 'train_county':
                                    ','.join(set(training_data_df['county'].tolist())), 'mu_X': [mu_X], 'std_X': [std_X]})
        val_predictions_df = pd.DataFrame({'time': val_data_df['time'].tolist(
        ), 'date': val_data_df['date'].tolist(), 'y_val': list(val_Y), 'y_val_pred': list(val_Y_pred), 'state': val_data_df['state'].tolist(),
            'county': val_data_df['county'].tolist(), 'ethnicity': val_data_df['ethnicity'].tolist()})

    return regression_info_df, predictions_df, mlp, val_info_df, val_predictions_df


def multilinear_reg(state_name: str, county_names: List[str], type: str,
                    ethnicity_filter_list: List[str], reg_key: str, metadata_filter: List[str], validate_state_name: str, validate_county_names: List[str],
                    bootstrap_bool: bool = True, N: int = 1000, weight_by_time: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, sm.OLS, pd.DataFrame, pd.DataFrame]:

    training_data_df = load_data_df(state_name=state_name, county_names=county_names,
                                    type=type, ethnicity_filter_list=ethnicity_filter_list)

    if weight_by_time:
        time_arr = np.array(training_data_df['time'].tolist())
        max_time = max(time_arr)
        weight_list = list(np.exp((time_arr - max_time) * 0))
    else:
        weight_list = []

    # Set Y as relevant key
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
        'county',
        'date']
    metadata_keys = training_data_df.keys()
    metadata_keys = [key for key in metadata_keys if key not in filter_list]
    val_metadata_keys = copy.deepcopy(metadata_keys)

    # Construct X
    X, metadata_keys, mu_X, std_X = get_X(
        training_data_df=training_data_df, filter_list=filter_list, metadata_keys=metadata_keys, metadata_filter=metadata_filter)

    fitted_model, feature_subset = call_multilinear_regression(X=X, Y=Y, weight_list=weight_list)
    # Get nrmse and rmse
    nrmse = calc_nrmse(Y, fitted_model.fittedvalues)
    rmse = calc_rmse(Y, fitted_model.fittedvalues)

    features = [metadata_keys[idx] for idx in feature_subset]
    stat_table = fitted_model.summary().tables[-1]

    if len(county_names) == 1:
        county_name = county_names[0]
        regression_info_dict = {'state': state_name, 'county': county_name, 'n': Y.shape[0]}
    else:
        regression_info_dict = {'state': state_name, 'county': ','.join(county_names), 'n': Y.shape[0]}

    regression_info_dict['features'] = features
    regression_info_dict['coef'] = np.abs(fitted_model.params)
    regression_info_dict['lower_coef'] = np.abs(regression_info_dict['coef']) - \
        1.96 * np.sqrt(np.diag(fitted_model.cov_params()))
    regression_info_dict['upper_coef'] = np.abs(regression_info_dict['coef']) + \
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
        frac = 1.0
        nrmse_list, rmse_list, indices_list = [], [], []
        for idx in range(N):
            indices = np.random.choice(n, size=int(frac * len(n)), replace=True)
            indices_list.append(indices)
        if len(weight_list) > 0:
            regr_results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(joblib.delayed(
                call_multilinear_regression)(X=X[indices, :], Y=Y[indices], weight_list=np.array(weight_list)[indices]) for indices in indices_list)
        else:
            regr_results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(joblib.delayed(
                call_multilinear_regression)(X=X[indices, :], Y=Y[indices])
                                                                               for indices in indices_list)
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
    ), 'date': training_data_df['date'].tolist(), 'y': list(Y), 'y_pred': list(fitted_model.fittedvalues), 'state': training_data_df['state'].tolist(), 'county': training_data_df['county'].tolist(),
        'ethnicity': training_data_df['ethnicity'].tolist()})

    val_info_df = None
    val_predictions_df = None
    if validate_state_name is not None:
        val_data_df = load_data_df(state_name=validate_state_name, county_names=validate_county_names, type=type,
                                   ethnicity_filter_list=ethnicity_filter_list)

        # Set Y as relevant key
        val_Y = np.array(val_data_df[reg_key])
        val_X, val_metadata_keys, mu_X, std_X = get_X(training_data_df=val_data_df, filter_list=filter_list, metadata_keys=val_metadata_keys,
                                                      metadata_filter=metadata_filter, mu_X=mu_X, std_X=std_X)

        val_Y_pred = fitted_model.predict(val_X)
        val_nrmse = calc_nrmse(val_Y, val_Y_pred)
        val_rmse = calc_rmse(val_Y, val_Y_pred)
        val_info_df = pd.DataFrame({'nrmse': [val_nrmse], 'rmse': [val_rmse], 'state': ','.join(set(val_data_df['state'].tolist())), 'county':
                                    ','.join(set(val_data_df['county'].tolist())), 'train_state': ','.join(set(training_data_df['state'].tolist())), 'train_county':
                                    ','.join(set(training_data_df['county'].tolist())), 'mu_X': [mu_X], 'std_X': [std_X]})
        val_predictions_df = pd.DataFrame({'time': val_data_df['time'].tolist(
        ), 'date': val_data_df['date'].tolist(), 'y_val': list(val_Y), 'y_val_pred': list(val_Y_pred), 'state': val_data_df['state'].tolist(),
            'county': val_data_df['county'].tolist(), 'ethnicity': val_data_df['ethnicity'].tolist()})

    return regression_info_df, predictions_df, fitted_model, val_info_df, val_predictions_df


def random_forest_reg(state_name: str, county_names: List[str], type: str,
                    ethnicity_filter_list: List[str], reg_key: str, metadata_filter: List[str], validate_state_name: str, validate_county_names: List[str],
                    bootstrap_bool: bool = True, N: int = 1000, weight_by_time: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, sm.OLS, pd.DataFrame, pd.DataFrame]:

    training_data_df = load_data_df(state_name=state_name, county_names=county_names,
                                    type=type, ethnicity_filter_list=ethnicity_filter_list)

    if weight_by_time:
        time_arr = np.array(training_data_df['time'].tolist())
        max_time = max(time_arr)
        weight_list = list(np.exp((time_arr - max_time) * 0))
    else:
        weight_list = []

    # Set Y as relevant key
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
        'county',
        'date']
    metadata_keys = training_data_df.keys()
    metadata_keys = [key for key in metadata_keys if key not in filter_list]
    val_metadata_keys = copy.deepcopy(metadata_keys)

    # Construct X
    X, metadata_keys, mu_X, std_X = get_X(
        training_data_df=training_data_df, filter_list=filter_list, metadata_keys=metadata_keys, metadata_filter=metadata_filter)

    # ###############################
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=3)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=3)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    fitted_model = RandomForestRegressor()
    fitted_model = RandomizedSearchCV(estimator=fitted_model, param_distributions=random_grid, n_iter=100, cv=20, verbose=2,
                                   random_state=42, n_jobs=-1)

    fitted_model.fit(X, Y)
    fitted_model = fitted_model.best_estimator_
    # import ipdb
    # ipdb.set_trace()


    # fitted_model = RandomForestRegressor(random_state=0)
    # fitted_model.fit(X, Y)

    Y_pred = fitted_model.predict(X)

    # get importance
    importance = list(fitted_model.feature_importances_)
    # Get nrmse and rmse
    nrmse = calc_nrmse(Y, Y_pred)
    rmse = calc_rmse(Y, Y_pred)

    if len(county_names) == 1:
        county_name = county_names[0]
        regression_info_dict = {'state': state_name, 'county': county_name, 'n': Y.shape[0]}
    else:
        regression_info_dict = {'state': state_name, 'county': ','.join(county_names), 'n': Y.shape[0]}

    regression_info_dict['features'] = metadata_keys
    regression_info_dict['coef'] = importance

    regression_info_dict['nrmse'] = nrmse
    regression_info_dict['rmse'] = rmse

    # Get bootstrap values of nrms and rmse if prescribed
    if bootstrap_bool:
        n = list(range(X.shape[0]))
        frac = 1.0
        nrmse_list, rmse_list, indices_list, coeff_list = [], [], [], []
        for idx in range(N):
            indices = np.random.choice(n, size=int(frac * len(n)), replace=True)
            indices_list.append(indices)

        fitted_model_bs = RandomForestRegressor(random_state=0).fit
        regr_results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(joblib.delayed(
            fitted_model_bs)(X[indices, :], Y[indices]) for indices in indices_list)

        fitted_model_list = regr_results

        for idx in range(N):
            Y_pred_bs = fitted_model_list[idx].predict(X[indices_list[idx], :])
            coeff_bs = fitted_model_list[idx].feature_importances_
            nrmse_bootstrap = calc_nrmse(Y[indices_list[idx]], Y_pred_bs)
            rmse_bootstrap = calc_rmse(Y[indices_list[idx]], Y_pred_bs)
            nrmse_list.append(nrmse_bootstrap)
            rmse_list.append(rmse_bootstrap)
            coeff_list.append(coeff_bs)
        coeff_list = np.array(coeff_list)
        low_nrmse, up_nrmse = np.percentile(nrmse_list, 2.5), np.percentile(nrmse_list, 97.5)
        low_coeff, up_coeff = np.percentile(coeff_list, 2.5, axis=0), np.percentile(coeff_list, 97.5, axis=0)
        regression_info_dict['low_nrmse'], regression_info_dict['up_nrmse'] = low_nrmse, up_nrmse
        regression_info_dict['low_coef'], regression_info_dict['up_coef'] = low_coeff, up_coeff

    regression_info_df = pd.DataFrame(regression_info_dict)

    predictions_df = pd.DataFrame({'time': training_data_df['time'].tolist(
    ), 'date': training_data_df['date'].tolist(), 'y': list(Y), 'y_pred': list(Y_pred), 'state': training_data_df['state'].tolist(), 'county': training_data_df['county'].tolist(),
        'ethnicity': training_data_df['ethnicity'].tolist()})

    val_info_df = None
    val_predictions_df = None
    if validate_state_name is not None:
        val_data_df = load_data_df(state_name=validate_state_name, county_names=validate_county_names, type=type,
                                   ethnicity_filter_list=ethnicity_filter_list)

        # Set Y as relevant key
        val_Y = np.array(val_data_df[reg_key])
        val_X, val_metadata_keys, mu_X, std_X = get_X(training_data_df=val_data_df, filter_list=filter_list, metadata_keys=val_metadata_keys,
                                                      metadata_filter=metadata_filter, mu_X=mu_X, std_X=std_X)

        val_Y_pred = fitted_model.predict(val_X)
        val_nrmse = calc_nrmse(val_Y, val_Y_pred)
        val_rmse = calc_rmse(val_Y, val_Y_pred)
        val_info_df = pd.DataFrame({'nrmse': [val_nrmse], 'rmse': [val_rmse], 'state': ','.join(set(val_data_df['state'].tolist())), 'county':
                                    ','.join(set(val_data_df['county'].tolist())), 'train_state': ','.join(set(training_data_df['state'].tolist())), 'train_county':
                                    ','.join(set(training_data_df['county'].tolist())), 'mu_X': [mu_X], 'std_X': [std_X]})
        val_predictions_df = pd.DataFrame({'time': val_data_df['time'].tolist(
        ), 'date': val_data_df['date'].tolist(), 'y_val': list(val_Y), 'y_val_pred': list(val_Y_pred), 'state': val_data_df['state'].tolist(),
            'county': val_data_df['county'].tolist(), 'ethnicity': val_data_df['ethnicity'].tolist()})

    return regression_info_df, predictions_df, fitted_model, val_info_df, val_predictions_df


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
                                List[str], reg_key: str, metadata_filter: List[str], validate_state_name: str, validate_county_names: List[str],
                                bootstrap_bool: bool = True, N: int = 100, regularizer_type: str = 'ridge', weight_by_time: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, sm.OLS, pd.DataFrame, pd.DataFrame]:
    training_data_df = load_data_df(state_name=state_name, county_names=county_names,
                                    type=type, ethnicity_filter_list=ethnicity_filter_list)

    if weight_by_time:
        time_arr = np.array(training_data_df['time'].tolist())
        max_time = max(time_arr)
        weight_list = list(np.exp((time_arr - max_time) * 0))
        train_weight_list = list(np.array(weight_list)[np.array(weight_list) >= 0.5])
    else:
        weight_list = None
        train_weight_list = None

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
        'county',
        'date',
        'POPULATION_GENDER_FEMALE', 'EMPLOYED_POPULATION_CHARACTERISTICS_OVERALL',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Private_Wage_Salary', 'EMPLOYED_POPULATION_CHARACTERISTICS_Government',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Self Employed', 'EMPLOYED_POPULATION_CHARACTERISTICS_Unpaid Family',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Management Business',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Service Occupations', 'EMPLOYED_POPULATION_CHARACTERISTICS_Sales',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Natural Resources', 'EMPLOYED_POPULATION_CHARACTERISTICS_Production',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Agricilture', 'EMPLOYED_POPULATION_CHARACTERISTICS_Construction',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Manufacturing', 'EMPLOYED_POPULATION_CHARACTERISTICS_Wholesale_Trade',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Retail', 'EMPLOYED_POPULATION_CHARACTERISTICS_Transportation',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Information', 'EMPLOYED_POPULATION_CHARACTERISTICS_Finance_and_Insurance',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Professional_Scientific_and_Management',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Educational_Services_and_Health_Care',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Arts_Entertainment_and_Recreation',
        'EMPLOYED_POPULATION_CHARACTERISTICS_Other', 'EMPLOYED_POPULATION_CHARACTERISTICS_Public_Administration',
        'POPULATION_RACE_BLACK', 'POPULATION_RACE_HISPANIC', 'POPULATION_RACE_WHITE', 'POPULATION_RACE_ASIAN']

    metadata_keys = training_data_df.keys()
    metadata_keys = [key for key in metadata_keys if key not in filter_list]
    val_metadata_keys = copy.deepcopy(metadata_keys)

    # Construct X

    X, metadata_keys, mu_X, std_X = get_X(
        training_data_df=training_data_df, filter_list=filter_list, metadata_keys=metadata_keys, metadata_filter=metadata_filter)

    # Get relevant information for the validation set if needed
    if validate_state_name is not None:
        val_data_df = load_data_df(state_name=validate_state_name, county_names=validate_county_names, type=type,
                                   ethnicity_filter_list=ethnicity_filter_list)
        time_arr = np.array(val_data_df['time'].tolist())
        max_time = max(time_arr)
        val_weight_list = list(np.exp((time_arr - max_time) * 0))
        val_data_df = val_data_df[np.array(val_weight_list) >= 0.5]
        # Set Y as relevant key
        val_Y = np.array(val_data_df[reg_key])
        val_X, val_metadata_keys, mu_X, std_X = get_X(training_data_df=val_data_df, filter_list=filter_list,
                                                      metadata_keys=val_metadata_keys,
                                                      metadata_filter=metadata_filter, mu_X=mu_X, std_X=std_X)
    else:
        val_X, val_Y = None, None

    if regularizer_type == 'ridge':
        score, fitted_model = call_multilinear_ridge_regression(
            X=X, Y=Y, val_X=val_X, val_Y=val_Y, weight_list=weight_list)
        Y_pred = fitted_model.predict(X)
    elif regularizer_type == 'lasso':
        score, fitted_model = call_multilinear_lasso_regression(
            X=X, Y=Y, val_X=val_X, val_Y=val_Y, weight_list=weight_list)
        Y_pred = fitted_model.predict(X)

    feature_subset = list(range(X.shape[1]))

    # Get nrmse and rmse
    nrmse = calc_nrmse(Y, Y_pred)
    rmse = calc_rmse(Y, Y_pred)

    # Regress on features and ca
    if len(county_names) == 1:
        county_name = county_names[0]
        regression_info_dict = {
            'state': [state_name] *
            len(metadata_keys),
            'county': [county_name] *
            len(metadata_keys),
            'n': [
                Y.shape[0]] *
            len(metadata_keys)}
    else:
        regression_info_dict = {'state': state_name, 'county': ','.join(county_names), 'n': Y.shape[0]}
    regression_info_dict['features'] = metadata_keys
    regression_info_dict['coef'] = list(fitted_model.coef_)
    # regression_info_dict['vif'] = vif_list
    regression_info_dict['R2'] = [score] * len(metadata_keys)
    regression_info_dict['nrmse'] = [nrmse] * len(metadata_keys)
    regression_info_dict['rmse'] = [rmse] * len(metadata_keys)

    # Get bootstrap values of nrms and rmse if prescribed
    if bootstrap_bool:
        n = list(range(X.shape[0]))
        frac = 0.5
        nrmse_list, rmse_list, indices_list, coef_list = [], [], [], []
        for idx in range(N):
            indices = np.random.choice(n, size=int(frac * len(n)), replace=True)
            indices_list.append(indices)

        if regularizer_type == 'ridge':
            regr_results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
                joblib.delayed(call_multilinear_ridge_regression)(X=X[indices, :], Y=Y[indices], weight_list=np.array(weight_list)[indices]) for indices in indices_list)
        elif regularizer_type == 'lasso':
            if weight_list is not None and len(weight_list) != 0:
                regr_results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
                    joblib.delayed(call_multilinear_lasso_regression)(X=X[indices, :], Y=Y[indices], weight_list=np.array(weight_list)[indices]) for indices in indices_list)
            else:
                regr_results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
                    joblib.delayed(call_multilinear_lasso_regression)(X=X[indices, :], Y=Y[indices]) for indices in indices_list)
        _, fitted_model_list = zip(*regr_results)

        for idx in range(N):
            X_bs = X[indices_list[idx], :]
            Y_pred_bs = fitted_model_list[idx].predict(X_bs)
            coef_bs = fitted_model_list[idx].coef_
            nrmse_bootstrap = calc_nrmse(Y[indices_list[idx]], Y_pred_bs)
            rmse_bootstrap = calc_rmse(Y[indices_list[idx]], Y_pred_bs)
            nrmse_list.append(nrmse_bootstrap)
            rmse_list.append(rmse_bootstrap)
            coef_list.append(coef_bs)
        import ipdb
        ipdb.set_trace()
        coef_list = np.array(coef_list)
        low_nrmse, up_nrmse = np.percentile(nrmse_list, 2.5), np.percentile(nrmse_list, 97.5)
        low_rmse, up_rmse = np.percentile(rmse_list, 2.5), np.percentile(rmse_list, 97.5)
        low_coef, up_coef = list(np.percentile(coef_list, 2.5, axis=0)), list(np.percentile(coef_list, 97.5, axis=0))

        regression_info_dict['low_nrmse'], regression_info_dict['up_nrmse'] = [
            low_nrmse] * len(metadata_keys), [up_nrmse] * len(metadata_keys)
        regression_info_dict['low_rmse'], regression_info_dict['up_rmse'] = [
            low_rmse] * len(metadata_keys), [up_rmse] * len(metadata_keys)
        regression_info_dict['low_coef'], regression_info_dict['up_coef'] = low_coef, up_coef

    regression_info_df = pd.DataFrame(regression_info_dict)

    predictions_df = pd.DataFrame({'time': training_data_df['time'].tolist(), 'date': training_data_df['date'].tolist(), 'y': list(Y), 'y_pred': list(Y_pred), 'state': training_data_df['state'].tolist(), 'county': training_data_df['county'].tolist(),
                                   'ethnicity': training_data_df['ethnicity'].tolist()})

    val_info_df = None
    val_predictions_df = None
    if validate_state_name is not None:
        val_Y_pred = fitted_model.predict(val_X)
        val_nrmse = calc_nrmse(val_Y, val_Y_pred)
        val_rmse = calc_rmse(val_Y, val_Y_pred)

        val_info_df = pd.DataFrame({'nrmse': [val_nrmse], 'rmse': [val_rmse], 'state': ','.join(set(val_data_df['state'].tolist())), 'county':
                                    ','.join(set(val_data_df['county'].tolist())), 'train_state': ','.join(set(training_data_df['state'].tolist())), 'train_county':
                                    ','.join(set(training_data_df['county'].tolist())), 'mu_X': [mu_X], 'std_X': [std_X]})
        val_predictions_df = pd.DataFrame({'time': val_data_df['time'].tolist(
        ), 'date': val_data_df['date'].tolist(), 'y_val': list(val_Y), 'y_val_pred': list(val_Y_pred), 'state': val_data_df['state'].tolist(),
            'county': val_data_df['county'].tolist(), 'ethnicity': val_data_df['ethnicity'].tolist()})

    return regression_info_df, predictions_df, fitted_model, val_info_df, val_predictions_df


def test_multilinear_regs(state_name: str, county_names: List[str], type: str,
                          ethnicity_filter_list: List[str], reg_key: str, metadata_filter: List[str], validate_state_name: str, validate_county_names: List[str],
                          test_state_name: str, test_county_names: List[str], regression_type: str,
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_data_df = load_data_df(state_name=test_state_name, county_names=test_county_names,
                                type=type, ethnicity_filter_list=ethnicity_filter_list)

    # Set Y as mortality rate
    test_Y = np.array(test_data_df[reg_key])

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
        'county',
        'date']
    metadata_keys = test_data_df.keys()
    metadata_keys = [key for key in metadata_keys if key not in filter_list]
    metadata_keys = copy.deepcopy(metadata_keys)

    # Load regression model
    # Construction prediction and regression file names for training data
    train_county_name = '_'.join([county_name for county_name in county_names]
                                 ) if len(county_names) > 1 else county_names[0]
    validation_results_path = path.join('states', validate_state_name, 'val_results_csvs', regression_type)
    val_model_path = path.join(validation_results_path, reg_key, 'model')

    val_county_name = '_'.join([county_name for county_name in validate_county_names]) if len(
        validate_county_names) > 1 else validate_county_names[0]
    val_results_file = f'{type}_{reg_key}_{regression_type}_{validate_state_name}' if val_county_name is None else f'{type}_{reg_key}_{regression_type}_{validate_state_name}_{val_county_name}_results'
    val_results_file = f'{val_results_file}_train_{state_name}_results' if train_county_name is None else f'{val_results_file}_train_{state_name}_{train_county_name}_results'
    val_model_file = f'{type}_{validate_state_name}' if val_county_name is None else f'{type}_{validate_state_name}_{val_county_name}'
    val_model_file = f'{val_model_file}_train_{state_name}_{reg_key}' if train_county_name is None else f'{val_model_file}_train_{state_name}_{train_county_name}_{reg_key}'

    val_results_file = utils_lib.create_files_name_with_ethnicity(file=val_results_file,
                                                                  ethnicity_filter_list=ethnicity_filter_list)
    val_df = pd.read_csv(f'{validation_results_path }/{val_results_file}')
    mu_X, std_X = np.array(
        eval(
            val_df['mu_X'][0].replace(
                ' ', ',').replace(
                '\n', ''))), np.array(
                    eval(
                        val_df['std_X'][0].replace(
                            ' ', ',').replace(
                                '\n', '')))

    if regression_type in managers_lib.RegDefinitions.multilinear_list or regression_type == 'gp':
        ext = 'pickle'
        val_model_file = utils_lib.create_files_name_with_ethnicity(file=val_model_file,
                                                                    ethnicity_filter_list=ethnicity_filter_list, ext=ext)
        if regression_type == 'gp':
            model = load(f'{val_model_path}/{val_model_file}')
        else:
            model = load_pickle(f'{val_model_path}/{val_model_file}')
    elif regression_type in managers_lib.RegDefinitions.multilinear_lasso_list or regression_type in managers_lib.RegDefinitions.multilinear_ridge_list:
        ext = 'sav'
        val_model_file = utils_lib.create_files_name_with_ethnicity(file=val_model_file,
                                                                    ethnicity_filter_list=ethnicity_filter_list, ext=ext)
        model = load(f'{val_model_path}/{val_model_file}')
    else:
        raise Exception(f'Model not loaded for test. Regression type {regression_type}, not loaded')

    # Construct X
    test_X, metadata_keys, mu_X, std_X = get_X(training_data_df=test_data_df, filter_list=filter_list,
                                               metadata_keys=metadata_keys, metadata_filter=metadata_filter, mu_X=mu_X, std_X=std_X)

    # Predict on test set and get accuracy
    test_Y_pred = model.predict(test_X)
    test_nrmse = calc_nrmse(test_Y, test_Y_pred)
    test_rmse = calc_rmse(test_Y, test_Y_pred)
    test_info_df = pd.DataFrame({'nrmse': [test_nrmse], 'rmse': [test_rmse], 'state': [test_data_df['state'][0]], 'county':
                                 [test_data_df['county'][0]], 'model': f'{val_model_path}/{val_model_file}', 'mu_X': [mu_X], 'std_X': [std_X]})
    test_predictions_df = pd.DataFrame({'time': test_data_df['time'].tolist(
    ), 'date': test_data_df['date'].tolist(), 'y_test': list(test_Y), 'y_test_pred': list(test_Y_pred),
        'state': test_data_df['state'].tolist(),
        'county': test_data_df['county'].tolist(), 'ethnicity': test_data_df['ethnicity'].tolist()})

    return test_info_df, test_predictions_df
