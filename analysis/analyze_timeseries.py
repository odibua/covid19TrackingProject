# --------------------------
# Standard Python Imports
# --------------------------
import argparse
import collections
import joblib
import multiprocessing
import os
from typing import Callable, Dict, List, Tuple, Union

# --------------------------
# Third Party Imports
# --------------------------
import importlib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# --------------------------
# covid19Tracking Imports
# --------------------------
from analysis.analyze_aggregate import open_csvs
import utils_lib as util_lib
from visualization import timeseries_vis as timeseries_vis_lib


def get_timeseries_counts_df_dict(df_dict: Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]]
                                  ) -> Dict[str, List[Tuple[str, pd.DataFrame]]]:
    """
    Processes input dictionary into a dictionary that contains list of tuples where
    the first element is an identifier that indicates, the second is a dataframe
    containing counts of cases/deaths with different ethnicities

    Arguments:
        df_dict: Input dictionary of the form:
                {'cases': {'counts': [(county/state location, timeseries dataframe with count), ...],
                          'discrepancies': [(county/state location, timeseries dataframe with discrepancy), ...]},
                 'deaths': {'counts': [(county/state location, timeseries dataframe with count), ...],
                          'discrepancies': [(county/state location, timeseries dataframe with discrepancy), ...]}
                }

    Returns:
        time_series_counts_df_dict: Dictionary containing count data frame for cases/deaths
        {'cases': [(county/state location, timeseries dataframe with count), ...]}
    """
    time_series_counts_df_dict = {}
    for key in df_dict.keys():
        count_list = df_dict[key]['counts']
        for identifier_df_tuple in count_list:
            identifier, df = identifier_df_tuple

            date_list = pd.to_datetime(df['date']).tolist()
            date_list = [(date_ - date_list[0]).days for date_ in date_list]
            df['time'] = date_list

            if key not in time_series_counts_df_dict:
                time_series_counts_df_dict[key] = []
            time_series_counts_df_dict[key].append((identifier, df))
    return time_series_counts_df_dict


def get_identifier_demographics_dict(state: str, county: str, date: str) -> Dict[str, float]:
    """
    Get demographic proportions of each ethnicity in a particular state or county using
    the projectors associated with that region

    Arguments:
        state: State for which demographics are being obtained
        county: County for which demographics are being obtained
        date: Valid date used to initialize relevant projector

    Returns:
        demographic_proportion_dict: Dictionary that maps ethnicity to percent of population
                                     in state and/or county that it represents
    """
    if state != county:
        state_county_dir = os.path.join('states', f'{state}', 'counties', f'{county}')
    else:
        state_county_dir = os.path.join('states', f'{state}')
        county = None

    state_county_dir_list = os.listdir(state_county_dir)
    state_county_projector_list = util_lib.filter_projector_module(projector_candidate_list=state_county_dir_list)
    if len(state_county_projector_list) != 1:
        raise ValueError(
            f"ERROR: ONLY ONE PROJECTOR SHOULD BE IMPLEMENTED IN DIRECTORY. Found {len(state_county_projector_list)} for directory {state_county_dir}")

    module_name = util_lib.get_projector_module(state=state, county=county, projector_name=state_county_projector_list[0][0:-3])

    state_county_projector_module = importlib.import_module(module_name)
    projector_class = util_lib.get_class_in_projector_module(module=state_county_projector_module, module_name=module_name)
    projector = projector_class(state=state, county=county, date_string=date)

    demographic_proportion_dict = {}
    total = sum(list(zip(*projector.ethnicity_demographics.items()))[1])
    for key, item in projector.ethnicity_demographics.items():
        demographic_proportion_dict[key] = item/total
    return demographic_proportion_dict


def bootstrap_gp_fit(x: np.ndarray, y: np.ndarray, N: int = 10) -> Dict[str, Union[float, np.ndarray]]:
    """
    Fit gaussian process to x and y data N times. Return the resulting mean predictions and gp parameters/quantities
    as well as their standard deviation

    Arguments:
        x: Inputs to gaussian process
        y: Outputs to which gaussian process will be fit
        N: Number of times GP is fit to data

    Returns:
        bootstrap_dict: Dictionary containing relevant GP parameters
    """
    def _fit_gaussian(x_, y_):
        kernel = C(1.0, (1e-3, 1e4)) * RBF(1.0, (1e-3, 1e4))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        gp.fit(x, y)
        y_pred, sigma = gp.predict(x, return_std=True)

        nrmse = np.sqrt(np.mean((y_pred - np.array(y)) ** 2)) / np.mean(y)

        constant = gp.kernel_.k1.constant_value
        length_scale = gp.kernel_.k2.length_scale
        return y_pred.reshape((1, -1)), sigma.reshape((1, -1)), nrmse, constant, length_scale

    results = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
        joblib.delayed(_fit_gaussian)(x, y) for _ in range(N)
    )
    y_pred_list, sigma_list, nrmse_list, constant_list, length_scale_list = list(zip(*results))
    y_pred_arr, sigma_arr = np.array(y_pred_list), np.array(sigma_list)

    bootstrap_dict = {'mn_length_scale': np.mean(length_scale_list), 'mn_constant': np.mean(constant_list),
                      'mn_nrmse': np.mean(nrmse_list), 'std_length_scale': np.std(length_scale_list), 'std_constant': np.std(constant_list),
                      'std_nrmse': np.std(nrmse_list), 'mn_y_pred': np.mean(y_pred_arr, axis=0)[0], 'mn_sigma': np.mean(sigma_arr, axis=0)[0], 'y': y.ravel(), 'x': x.ravel()}
    return bootstrap_dict


def fit_time_series_counts(df_dict: Dict[str, List[Tuple[str, pd.DataFrame]]], state: str) \
        -> Dict[str, Dict[str, List[Tuple[str, GaussianProcessRegressor]]]]:
    """
    Fit time series data about case/death counts using gaussian process, as well as
    the case/death count assuming no disparity (i.e. assuming people are impacted
    based on population demographic). The returned dictionary contains tuples of dictionaries
    which have bootstrapped gp parameters, predictions, and measures of uncertainty.

    Arguments:
        df_dict: Dictionary containing count data frame for cases/deaths
        {'cases': [(county/state location, timeseries dataframe with count), ...]}
        state: String containing state being analyzed

    Returns:
        time_regression_dict: Nested dictionary that contains dictionaries which in turn
                              tuple of two dictionaries. These contain GP predictions/parameters
                              on real time series data regarding case/death counts and the fit
                              if the case/death counts matched population distribution.
                              {'cases': {'alameda': {'Black': ({'mn_sigma':...,'mn_y_pred':,...}, {'mn_sigma':...,'mn_y_pred':,...})}}}
    """
    time_regression_dict = {}
    for key in df_dict.keys():
        if key not in time_regression_dict.keys():
            time_regression_dict[key] = {}
        for identifier_df_tuple in df_dict[key]:
            identifier, df = identifier_df_tuple

            # Get demographic proportions of each ethnicity in a region
            identifier_demographics_proportions = get_identifier_demographics_dict(state=state, county=identifier,
                                                                                   date=df['date'][0])
            df_filtered = df.filter(items=[key for key in df.keys() if key != 'date' and key != 'time'])
            df_filtered = df_filtered.sum(axis=1)

            if identifier not in time_regression_dict[key].keys():
                time_regression_dict[key][identifier] = {}
            for column in df.keys():
                if column != 'time' and column != 'date':
                    num_bool = df[column].notnull()
                    x_train, y_train_real = df['time'][num_bool].tolist(), df[column][num_bool].tolist()
                    x_train, y_train_real = np.array(x_train).reshape((-1, 1)), np.array(y_train_real).reshape((-1, 1))

                    y_train_ideal = identifier_demographics_proportions[column] * df_filtered.to_numpy()
                    try:
                        bootstrap_dict_real = bootstrap_gp_fit(x=x_train, y=y_train_real)
                        bootstrap_dict_ideal = bootstrap_gp_fit(x=x_train, y=y_train_ideal)

                        time_regression_dict[key][identifier][column] = (bootstrap_dict_real, bootstrap_dict_ideal)
                    except:
                        pass

    return time_regression_dict


def time_series_analysis(csv_df_dict: Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]], state: str) -> None:
    """
    Performs time series analysis of case/death data for a particular state and corresponding counties

    Arguments:
        csv_df_dict: Dictionary of pandas dataframe containing counts and disparity ratios by ethnicity. Keys of
                     a dictionary are cases and deaths
        state: State for which timeseries analysis is performed
    """
    time_series_counts_df_dict = get_timeseries_counts_df_dict(df_dict=csv_df_dict)
    time_regression_dict = fit_time_series_counts(df_dict=time_series_counts_df_dict, state=state)

    timeseries_vis_lib.vis_mean_ci_bar(stats_dict=time_regression_dict)
    pass


def main():
    # Parse state that will be visualized
    parser = argparse.ArgumentParser(description='Add state for which to analyze data')
    parser.add_argument('--state', help='State whose csvs will be analyzed')
    args = parser.parse_args()
    state = args.state

    # Open csvs with pandas data frame and populate with a dictionary
    state_path = os.path.join('states', state, 'csvs')
    csv_path_dict = collections.defaultdict(list)
    for csv_file in os.listdir(state_path):
        if not csv_file.endswith('.csv'):
            raise ValueError(
                f"Non-csv file {csv_file} found in {state_path}. All files in this directory must be a csv file")

        if 'cases' in csv_file.lower():
            csv_path_dict['cases'].append(os.path.join(state_path, csv_file))
        elif 'deaths' in csv_file.lower():
            csv_path_dict['deaths'].append(os.path.join(state_path, csv_file))
        else:
            raise ValueError(
                f"CSV file exists that does not contain 'case' or 'death'. All files in {state_path} must have 'case' or 'death' in filename")
    csv_df_dict = open_csvs(csv_path_dict=csv_path_dict)

    time_series_analysis(csv_df_dict=csv_df_dict, state=state)


if __name__ == "__main__":
    main()
