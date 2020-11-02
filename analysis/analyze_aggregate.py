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
import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower

# --------------------------
# covid19Tracking Imports
# --------------------------
from visualization import aggregated_vis as agg_vis_lib, visualization as vis_lib


def aggregated_analysis(csv_df_dict: Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]], state: str) -> None:
    """
    Analyze data frames of csvs by calculating the mean of the maximum disparity ratios on different
    days, the confidence interval of this quanitity, and the power with respect to one of this quantity.
    Visualize these quantities.

    Additionally, if applicable, show the counties difference from state (negative to positive). Finally,
    visualize the inconsitencies of ethnicities shown by California vs states. Give bar graph of portion of ethnicities
    reported by counties vs states.

    Arguments:
        csv_df_dict: Dictionary of pandas dataframe containing counts and disparity ratios by ethnicity. Keys of
                     a dictionary are cases and deaths
        state: State for which aggregated analysis is performed and the counties associated
    """
    # Calculate mean and confidence interval of maximum/median disparities, as well as the associated power
    disparity_stats_dict = calc_disparity_stats(df_dict=csv_df_dict)
    disparity_stats_dict = calc_power_disparity(
        stats_dict=disparity_stats_dict,
        stat_key='max_mean',
        std_stat_key='max_std')
    disparity_stats_dict = calc_power_disparity(
        stats_dict=disparity_stats_dict,
        stat_key='median_mean',
        std_stat_key='median_std')
    disparity_stats_dict = calc_power_disparity(stats_dict=disparity_stats_dict, stat_key='mean', std_stat_key='std')
    disparity_stats_dict = calc_power_disparity(
        stats_dict=disparity_stats_dict,
        stat_key='max_min_ratio_mean',
        std_stat_key='max_min_ratio_std')

    # Visualize the mean/median/max and their respective confidences
    agg_vis_lib.vis_mean_ci_bar(stats_dict=disparity_stats_dict, plot_key='mean', std_plot_key='std', state=state)
    agg_vis_lib.vis_mean_ci_bar(
        stats_dict=disparity_stats_dict,
        plot_key='max_mean',
        std_plot_key='max_std',
        state=state)
    agg_vis_lib.vis_mean_ci_bar(
        stats_dict=disparity_stats_dict,
        plot_key='median_mean',
        std_plot_key='median_std',
        state=state)
    agg_vis_lib.vis_mean_ci_bar(
        stats_dict=disparity_stats_dict,
        plot_key='max_min_ratio_mean',
        std_plot_key='max_min_ratio_std',
        state=state)

#
#     # Calculate two closest and two furthest counties from State
#     distance_from_state_dict = calc_nearest_furthest_from_state(df_dict=csv_df_dict)
#
    # Calculate proportion of ethnicities reported by state that each county reports
    # and show proportion of analyzed counties that report each ethnicity
    county_propotion_dict, state_ethnicity_count_dict = get_county_state_key_portions(df_dict=csv_df_dict, state=state)


def calc_max_min_ratio(df: pd.DataFrame) -> Tuple[List[float], str, str]:
    """
    Calculate the absolute max to minimum ratio of non-zero rows in a data frame

    Arguments:
        df: DataFrame

    Return:
        max_min_ratios: List of max and min ratios for each row in the data frame
    """
    max_min_ratios = []
    min_max_names = collections.defaultdict(int)
    first_second_names = collections.defaultdict(int)
    for idx, row in df.iterrows():
        row_names = list(row.keys())
        row_vals = list(row)
        vals, names = list(zip(*[(float(val), name) for val, name in zip(row_vals, row_names) if val != 0]))
        sorted_idx = np.argsort(vals)

        max_min_ratios.append(vals[sorted_idx[-1]] / vals[sorted_idx[0]])
        first_second_names[f'{names[sorted_idx[-1]].split("_")[0]}_{names[sorted_idx[-2]].split("_")[0]}'] = first_second_names[
            f'{names[sorted_idx[-1]].split("_")[0]}_{names[sorted_idx[-2]].split("_")[0]}'] + 1
        min_max_names[f'{names[sorted_idx[-1]].split("_")[0]}_{names[sorted_idx[0]].split("_")[0]}'] = min_max_names[
            f'{names[sorted_idx[-1]].split("_")[0]}_{names[sorted_idx[0]].split("_")[0]}'] + 1

    first_second_names = max(first_second_names.items(), key=operator.itemgetter(1))[0]
    majority_min_max_names = max(min_max_names.items(), key=operator.itemgetter(1))[0]
    return max_min_ratios, majority_min_max_names, first_second_names


def calc_disparity_stats(df_dict: Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]]
                         ) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """
    Calculate the mean maximum disparity per day of each county, along with the standard deviation, and the 95 percent confidence
    interval. Data is calculated from data frames stored in a nested dictionary

    Arguments:
        df_dict: Nested dictionary that maps 'cases' or 'deaths' to nested dictionary containing 'counts'
                 and 'discrepancies' which are mapped to tuples that contain state/county and the associated
                 pandas dataframe.
                 Example:
                    {'cases': { 'counts': [(county_name, pd.DataFrame], ...],
                                'discrepancies': [(county_name, pd.DataFrame), ...]}}

    Returns:
        max_disparity_stats_dict: Nested dictionary that maps 'cases' or 'deaths; to nested dictionary containing
        'mean' and 'std' which all map to list of tuples containing state/county and the associated statistical value
        Example:
            {'cases': { 'mean': [(county_name, mean max discrepancy], ...],
                                'std': [(county_name, std max discrepancy), ...]}}
    """
    disparity_stats_dict = {}
    for key in df_dict.keys():
        discrepancy_list = df_dict[key]['discrepancies']
        for identifier_df_tuple in discrepancy_list:
            identifier, df = identifier_df_tuple
            df = df.filter(items=[key for key in df.keys() if key != 'date'])
            max_min_ratios, min_max_names, first_second_names = calc_max_min_ratio(df=df)

            mean = df.mean(axis=1).mean()
            max_mean = df.max(axis=1).mean()
            median_mean = df.median(axis=1).mean()
            max_min_ratio_mean = np.mean(max_min_ratios)

            std = df.mean(axis=1).std()
            max_std = df.max(axis=1).std()
            median_std = df.median(axis=1).std()
            max_min_ratio_std = np.std(max_min_ratios)

            if key not in disparity_stats_dict.keys():
                disparity_stats_dict[key] = {'mean': [], 'max_mean': [], 'median_mean': [], 'max_min_ratio_mean': [],
                                             'std': [], 'max_std': [], 'median_std': [], 'max_min_ratio_std': [],
                                             'N': [], 'first_second_names': [], 'min_max_names': []}
            disparity_stats_dict[key]['mean'].append((identifier, mean))
            disparity_stats_dict[key]['max_mean'].append((identifier, max_mean))
            disparity_stats_dict[key]['median_mean'].append((identifier, median_mean))
            disparity_stats_dict[key]['max_min_ratio_mean'].append((identifier, max_min_ratio_mean))

            disparity_stats_dict[key]['std'].append((identifier, std))
            disparity_stats_dict[key]['max_std'].append((identifier, max_std))
            disparity_stats_dict[key]['median_std'].append((identifier, median_std))
            disparity_stats_dict[key]['max_min_ratio_std'].append((identifier, max_min_ratio_std))

            disparity_stats_dict[key]['N'].append((identifier, len(df)))
            disparity_stats_dict[key]['min_max_names'].append((identifier, min_max_names))
            disparity_stats_dict[key]['first_second_names'].append((identifier, first_second_names))

    return disparity_stats_dict


def calc_power_disparity(stats_dict: Dict[str, Dict[str, List[Tuple[str, float]]]], stat_key: str, std_stat_key: str,
                         alpha: float = 0.05) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """
    Calculate the power of the mean max disparity stat for each county/state with respect to a disparity
    ratio of one. The dict processed has form:
    Example:
            {'cases': { 'mean': [(county_name, mean max discrepancy], ...],
                                'std': [(county_name, std max discrepancy), ...]}}
    Arguments:
        df_dict: Dictionary (example shown above) of max disparity statistics
        alpha: 95 percent confidence interval

    Returns:
        df_dict: Dictionary that is modified by adding power, as shown below.
        Example:
                {'cases': { 'mean': [(county_name, mean max discrepancy], ...],
                                'std': [(county_name, std max discrepancy), ...],
                                'power':[(county_name, power), ...]}}
    :return:
    """
    for key in stats_dict.keys():
        mean_tuple_list, std_tuple_list = stats_dict[key][stat_key], stats_dict[key][std_stat_key]
        N_tuple_list = stats_dict[key]['N']
        for N_tuple, std_tuple, mean_tuple in zip(N_tuple_list, std_tuple_list, mean_tuple_list):
            identifier, mean = mean_tuple
            _, N = N_tuple
            _, std = std_tuple
            analysis = TTestIndPower()
            if f'{stat_key}_power' not in stats_dict[key].keys():
                stats_dict[key][f'{stat_key}_power'] = []
            power = analysis.solve_power(effect_size=abs((mean - 1.0) / std), nobs1=N, ratio=1.0, alpha=alpha)
            stats_dict[key][f'{stat_key}_power'].append((identifier, power))
    return stats_dict


def get_county_state_key_portions(df_dict: Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]], state: str) \
        -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Calculate the portion of ethnicities reported by the county that are reported by the state and count
    the number of counties using each ethnicity reported by the state

    Arguments:
        df_dict: Dictionary containing data frames corresponding with the relevant state/county
        state: State being analyzed

    Returns:
        county_proportion_dict: Nested dictinary containing state ethnicities as key and state ethnicities reported by a
                                particular county as a value. The highest level of the dictionary are 'cases' and 'deaths'.
        state_ethnicity_count_dict: Nested dictionary containing state ethnicities as key and count of how many counties report
                                    that particular ethnicity as a value. The highest level of the dictionary are 'cases' and 'deaths'.
    """
    county_proportion_dict, state_ethnicity_count_dict = {}, {}
    for key in df_dict.keys():
        identifier_lst, df_lst = list(zip(*df_dict[key]['counts']))
        state_idx = [idx for idx, identifier in enumerate(identifier_lst) if identifier == state][0]
        state_ethnicities = list(df_lst[state_idx].keys())
        state_ethnicities = [state_ethnicity.lower()
                             for state_ethnicity in state_ethnicities if state_ethnicity != 'date']

        if key not in county_proportion_dict.keys():
            county_proportion_dict[key] = collections.defaultdict(float)
        if key not in state_ethnicity_count_dict.keys():
            state_ethnicity_count_dict[key] = collections.defaultdict(int)
        for idx, identifier in enumerate(identifier_lst):
            if identifier != state:
                identifier_ethnicities = list(df_lst[idx].keys())
                cnt = 0
                for ethnicity in identifier_ethnicities:
                    if ethnicity.lower() in state_ethnicities:
                        cnt = cnt + 1
                        state_ethnicity_count_dict[key][ethnicity.lower(
                        )] = state_ethnicity_count_dict[key][ethnicity.lower()] + 1

                county_proportion_dict[key][identifier] = cnt / len(state_ethnicities)

    return county_proportion_dict, state_ethnicity_count_dict


def open_csvs(csv_path_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]]:
    """
    Open the csv files stored in a dictionary and store it in a dictionary and return the pandas
    data frames in a corresponding dictionary. The file names are expected to be of form
    $STATE_$COUNTY_ethnicity_cases(deaths).csv or $STATE_ethnicity_cases(deaths).csv

    Arguments:
        csv_path_dict: Dictionary containing keys that are either 'cases' or 'deaths' that are mapped
                       to list of county and/or state csv dictionaries. eg. {cases: [path_to_csv,...], deaths: [..]}

    Returns:
        csv_df_dict: Nested dictionary that contains keys that are either 'cases' or 'deaths' that are mapped
                     to a nested dictionary of 'counts' or 'discrepancies', which are in turn mapped to a list
                     of county and/or state pandas data frames.
    """
    csv_df_dict = {}
    state = None
    for key in csv_path_dict.keys():
        for csv_path in csv_path_dict[key]:
            csv_path_list = csv_path.split('/')
            if state is None:
                state = csv_path_list[1]
            if key not in csv_df_dict.keys():
                csv_df_dict[key] = collections.defaultdict(list)
            df = pd.read_csv(csv_path)
            df_count, df_discrepancy = vis_lib.split_pandas_by_discrepancy(df=df)

            csv_filename_list = csv_path_list[-1].split('_')

            # Define identifier as county if csv is for county. Otherwise, state
            if csv_filename_list[1] != 'ethnicity':
                identifier = csv_filename_list[1]
            else:
                identifier = state

            # Store identifier and dataframe tuple in dictionary
            csv_df_dict[key]['counts'].append((identifier, df_count))
            csv_df_dict[key]['discrepancies'].append((identifier, df_discrepancy))
    return csv_df_dict


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

    aggregated_analysis(csv_df_dict=csv_df_dict, state=state)


if __name__ == "__main__":
    main()
