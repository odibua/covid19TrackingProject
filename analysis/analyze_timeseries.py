# --------------------------
# Standard Python Imports
# --------------------------
import argparse
import collections
import operator
import os
from typing import Dict, List, Tuple

# --------------------------
# Third Party Imports
# --------------------------
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# --------------------------
# covid19Tracking Imports
# --------------------------
from analysis.analyze_aggregate import open_csvs


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


def time_series_analysis(csv_df_dict: Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]], state: str) -> None:

    time_series_counts_df_dict = get_timeseries_counts_df_dict(df_dict=csv_df_dict)
    fit_reg_dict = fit_time_series_counts(df_dict=time_series_counts_df_dict)
    import ipdb
    ipdb.set_trace()
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
