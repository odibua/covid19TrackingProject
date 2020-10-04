# --------------------------
# Standard Python Imports
# --------------------------
import argparse
import collections
import os
from typing import Dict, List, Tuple

# --------------------------
# Third Party Imports
# --------------------------
import pandas as pd
# --------------------------
# covid19Tracking Imports
# --------------------------
from visualization import visualization as vis_lib


# def aggregated_analysis(csv_df_dict: Dict[str, pd.DataFrame]) -> None:
#     """
#     Analyze data frames of csvs by calculating the mean of the maximum disparity ratios on different
#     days, the confidence interval of this quanitity, and the power with respect to one of this quantity.
#     Visualize these quantities.
#
#     Additionally, if applicable, show the counties difference from state (negative to positive). Finally,
#     visualize the inconsitencies of ethnicities shown by California vs states. Give bar graph of portion of ethnicities
#     reported by counties vs states.
#
#     Arguments:
#         csv_df_dict: Dictionary of pandas dataframe containing counts and disparity ratios by ethnicity. Keys of
#                      a dictionary are cases and deaths
#     """
#     # Calculate mean and confidence interval of maximum disparities, as well as the associated power
#     max_ci_dict = calc_max_disparity_ci(df_dict=csv_df_dict)
#     power_dict = calc_power_max_disparity(df_dict=max_ci_dict)
#
#     # Visualize the mean, confidence, and power of maximum disparity ratios
#     visualize_max_ci = aggregated_vis.max_disparity_ci(df_dict=max_ci_dict, power_df_dict=power_dict)
#
#     # Calculate two closest and two furthest counties from State
#     distance_from_state_dict = calc_nearest_furthest_from_state(df_dict=csv_df_dict)
#
#     # Calculate proportion of ethnicities reported by state that each county reports
#     # and show proportion of analyzed counties that report each ethnicity
#     county_propotion_dict, state_ethnicity_proportion_dict = get_county_state_irregular_portions(df_dict=csv_df_dict)


def open_csvs(csv_path_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]]:
    """
    Open the csv files stored in a dictionary and store it in a dictionary and return the pandas
    data frames in a corresponding dictionary.

    Arguments:
        csv_path_dict: Dictionary containing keys that are either 'cases' or 'deaths' that are mapped
                       to list of county and/or state csv dictionaries. eg. {cases: [path_to_csv,...], deaths: [..]}

    Returns:
        csv_df_dict: Nested dictionary that contains keys that are either 'cases' or 'deaths' that are mapped
                     to a nested dictionary of 'counts' or 'discrepancies', which are in turn mapped to a list
                     of county and/or state pandas data frames.
    """
    csv_df_dict = {}
    for key in csv_path_dict.keys():
        for csv_path in csv_path_dict[key]:
            if key not in csv_df_dict.keys():
                csv_df_dict[key] = collections.defaultdict(list)
            df = pd.read_csv(csv_path)
            df_count, df_discrepancy = vis_lib.split_pandas_by_discrepancy(df=df)
            csv_df_dict[key]['counts'].append((csv_path, df_count))
            csv_df_dict[key]['duscrepancies'].append((csv_path, df_discrepancy))
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
            raise ValueError(f"Non-csv file {csv_file} found in {state_path}. All files in this directory must be a csv file")

        if 'cases' in csv_file.lower():
            csv_path_dict['cases'].append(os.path.join(state_path, csv_file))
        elif 'deaths' in csv_file.lower():
            csv_path_dict['deaths'].append(os.path.join(state_path, csv_file))
        else:
            raise ValueError(f"CSV file exists that does not contain 'case' or 'death'. All files in {state_path} must have 'case' or 'death' in filename")
    csv_df_dict = open_csvs(csv_path_dict=csv_path_dict)

    # aggregated_analysis(csv_df_dict=csv_df_dict)

if __name__ == "__main__":
    main()




