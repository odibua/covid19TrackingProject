# --------------------------
# Standard Python Imports
# --------------------------
import datetime
import logging
import os
from os import path
import subprocess as cmd
from typing import List, Tuple

# --------------------------
# Third Party Imports
# --------------------------
import bokeh as bokeh
from bokeh.io import output_file
from bokeh.plotting import ColumnDataSource, figure, show
from bokeh.layouts import column, row
from bokeh.models import ColorBar, Label, LabelSet, LinearColorMapper, LogTicker, Slider

import pandas as pd
import yaml
# --------------------------
# covid19Tracking Imports
# --------------------------


def split_pandas_into_case_discrepancy(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    columns = df.columns.tolist()
    discrepancy_columns = [column.split('_')[0] for column in columns if 'discrepancy' in column or column == 'date']
    columns = [column.split('_')[0] for column in columns if 'discrepancy' not in column or column == 'date']


    return df[columns], df[discrepancy_columns]


def visualize_per_county_stats():
    logging.info("Open State Configuration file and get states to be plotted")
    config_path = 'states/states_config.yaml'
    if not path.isfile(config_path):
        raise ValueError(f"states_config.yaml not found in states directory")
    config_file = open(config_path)
    config = yaml.safe_load(config_file)
    state_list = config['STATES']
    cases_csv_filename, deaths_csv_filename = 'ethnicity_cases.csv', 'ethnicity_deaths.csv'

    logging.info("Create output html file")
    output_file(f"covid_ethnic_discrepancy.html")
    county_fig_names = ['Ethnicity Case Count', 'Ethnicity Case Disparity', 'Ethnicity Death Count', 'Ethnicity Death Disparity']
    tab_names = []
    logging.info(f"Get and process covid19 ethnicity data for each state and corresponding counties")
    for state in state_list:
        state_name = state.lower()
        logging.info(f"Processing {state_name}")
        state_county_dir = os.path.join('states', state_name)

        ethnicity_cases_df = pd.read_csv(f"{state_county_dir}/{cases_csv_filename}")
        ethnicity_deaths_df = pd.read_csv(f"{state_county_dir}/{deaths_csv_filename}")

        ethnicity_cases_df, ethnicity_cases_discrepancy_df = split_pandas_into_case_discrepancy(df=ethnicity_cases_df)
        ethnicity_deaths_df, ethnicity_deaths_discrepancy_df = split_pandas_into_case_discrepancy(df=ethnicity_deaths_df)


        logging.info("\n")
        logging.info(f"Processing county level data for {state_name}")
        county_dirs = os.listdir(path.join('states', state_name, 'counties'))
        county_dirs.sort()
        if len(county_dirs) > 0:
            for county in county_dirs:
                pass