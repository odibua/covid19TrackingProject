# --------------------------
# Standard Python Imports
# --------------------------
import argparse
import collections
import operator
import os
from typing import Callable, Dict, List, Tuple

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
from utils import filter_projector_module, get_projector_module, get_class_in_projector_module


def get_timeseries_counts_df_dict(df_dict: Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]]
                                  ) -> Dict[str, List[Tuple[str, pd.DataFrame]]]:
    time_series_counts_df_dict = {}
    for key in df_dict.keys():
        count_list = df_dict[key]['counts']
        for identifier_df_tuple in count_list:
            identifier, df = identifier_df_tuple

            date_list = pd.to_datetime(df['date']).tolist()
            date_list = [(date_ - date_list[0]).days for date_ in date_list]
            df['time'] = date_list
            df = df.filter(items=[key for key in df.keys() if key != 'date'])

            if key not in time_series_counts_df_dict:
                time_series_counts_df_dict[key] = []
            time_series_counts_df_dict[key].append((identifier, df))
    return time_series_counts_df_dict


def get_identifier_demographics_dict(state: str, county: str) -> Dict[str, float]:
    if state != county:
        state_county_dir = os.path.join('states', f'{state}', 'counties', f'{county}')
    else:
        state_county_dir = os.path.join('states', f'{state}')

    state_county_dir_list = os.listdir(state_county_dir)
    state_county_projector_list = filter_projector_module(projector_candidate_list=state_county_dir_list)
    if len(state_county_projector_list) != 1:
        raise ValueError(
            f"ERROR: ONLY ONE PROJECTOR SHOULD BE IMPLEMENTED IN DIRECTORY. Found {len(state_county_projector_list)} for directory {state_county_dir}")

    module_name = get_projector_module(state=state, county=county, projector_name=state_county_projector_list[0][0:-3])
    state_county_projector_module = importlib.import_module(module_name)
    projector_class = get_class_in_projector_module(module=state_county_projector_module, module_name=module_name)
    return projector_class.ethnicity_demographics


def fit_time_series_counts(df_dict: Dict[str, List[Tuple[str, pd.DataFrame]]], state: str) \
        -> Dict[str, Dict[str, List[Tuple[str, GaussianProcessRegressor]]]]:
    time_regression_dict = {}
    for key in df_dict.keys():
        if key not in time_regression_dict.keys():
            time_regression_dict[key] = {}
        for identifier_df_tuple in df_dict[key]:
            identifier, df = identifier_df_tuple
            identifier_demographics = get_identifier_demographics_dict(state=state, county=identifier)
            import ipdb
            ipdb.set_trace()
            if identifier not in time_regression_dict[key].keys():
                time_regression_dict[key][identifier] = []
            for column in df.keys():
                if column != 'time':
                    num_bool = df[column].notnull()
                    x_train, y_train = df['time'][num_bool].tolist(), df[column][num_bool].tolist()
                    x_train, y_train = np.array(x_train).reshape((-1, 1)), np.array(y_train).reshape((-1, 1))

                    kernel = C(1.0, (1e-3, 1e4)) * RBF(1.0, (1e-3, 1e4))
                    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True)
                    try:
                        gp.fit(x_train, y_train)
                        y_pred, sigma = gp.predict(x_train, return_std=True)
                        nrmse = np.sqrt(np.mean((y_pred - np.array(y_train)) ** 2)) / np.mean(y_train)
                        time_regression_dict[key][identifier].append((column, y_pred.ravel(), y_train.ravel(), gp.kernel_.k1.constant_value, gp.kernel_.k2.length_scale, nrmse))
                    except:
                        pass

    return time_regression_dict


def time_series_analysis(csv_df_dict: Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]], state: str) -> None:

    time_series_counts_df_dict = get_timeseries_counts_df_dict(df_dict=csv_df_dict)
    time_regression_dict = fit_time_series_counts(df_dict=time_series_counts_df_dict, state=state)
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
