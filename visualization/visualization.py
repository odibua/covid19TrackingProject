# --------------------------
# Standard Python Imports
# --------------------------
import datetime
import itertools
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
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import ColorBar, Label, LabelSet, LinearColorMapper, LogTicker, Slider
from bokeh.palettes import Dark2_5 as palette

import pandas as pd
import yaml
# --------------------------
# covid19Tracking Imports
# --------------------------


def convert_date_str_to_datetime(df_list: List[pd.DataFrame]) -> Tuple[pd.DataFrame]:
    new_df_list = []
    for df in df_list:
        df['date'] = pd.to_datetime(df['date'])
        new_df_list.append(df)
    return tuple(new_df_list)


def split_pandas_into_case_discrepancy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = df.columns.tolist()
    discrepancy_columns = [column for column in columns if 'discrepancy' in column or column ==
                           'date' or column == 'Unnamed: 0']
    columns = [column for column in columns if 'discrepancy' not in column or column == 'date']
    try:
        df[columns], df[discrepancy_columns]
    except:
        import ipdb
        ipdb.set_trace()
    return df[columns], df[discrepancy_columns]


def get_plot_figs(fig_names: List[str]) -> Tuple[figure]:
    return tuple([figure(plot_width=1000, title=fig_name, x_axis_type='datetime',
                         tooltips=[("x", "$x"), ("y", "$y")]) for fig_name in fig_names])


def plot_cases_deaths(fig: figure, source_: ColumnDataSource, df: pd.DataFrame):
    colors = itertools.cycle(palette)
    for key in df.keys():
        if key.lower() != 'date' and key != 'Unnamed: 0':
            color = next(colors)
            if 'discrepancy' in key:
                legend_key = key.split('_')[0]
            else:
                legend_key = key
            fig.line(x='date', y=key, source=source_, legend_label=legend_key, color=color)
            fig.circle(x='date', y=key, source=source_, legend_label=legend_key, color=color)


def run_plot_cases_deaths(fig_names: List[str], cases_df: pd.DataFrame,
                          cases_discrepancy_df: pd.DataFrame, deaths_df: pd.DataFrame, deaths_discrepancy_df: pd.DataFrame):
    cases_fig, cases_discrepancy_fig, deaths_fig, deaths_discrepancy_fig = get_plot_figs(fig_names=fig_names)

    cases_source, cases_discrepancy_source = ColumnDataSource(cases_df), ColumnDataSource(cases_discrepancy_df)
    deaths_source, deaths_discrepancy_source = ColumnDataSource(deaths_df), ColumnDataSource(deaths_discrepancy_df)

    plot_cases_deaths(fig=cases_fig, source_=cases_source, df=cases_df)
    plot_cases_deaths(fig=cases_discrepancy_fig, source_=cases_discrepancy_source, df=cases_discrepancy_df)
    plot_cases_deaths(fig=deaths_fig, source_=deaths_source, df=deaths_df)
    plot_cases_deaths(fig=deaths_discrepancy_fig, source_=deaths_discrepancy_source, df=deaths_discrepancy_df)

    layout = column(cases_fig, cases_discrepancy_fig, deaths_fig, deaths_discrepancy_fig)
    return layout


# TODO(odibua@): Make hover tool read dates. Note inconsistent dates so parsing can be fixed
# TODO(odibua@): Log error if change is too large based on dates
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
    fig_names = [
        'Ethnicity Case Count',
        'Ethnicity Case Disparity',
        'Ethnicity Death Count',
        'Ethnicity Death Disparity']
    tab_list, tab_names = [], []
    logging.info(f"Load deaths and cases csvs for relevant states and all counties in config")
    for state in state_list:
        state_name = state.lower()
        logging.info(f"Processing {state_name}")
        state_county_dir = os.path.join('states', state_name)

        ethnicity_cases_df = pd.read_csv(f"{state_county_dir}/{cases_csv_filename}")
        ethnicity_deaths_df = pd.read_csv(f"{state_county_dir}/{deaths_csv_filename}")
        ethnicity_cases_df, ethnicity_deaths_df = convert_date_str_to_datetime(
            df_list=[ethnicity_cases_df, ethnicity_deaths_df])

        ethnicity_cases_df, ethnicity_cases_discrepancy_df = split_pandas_into_case_discrepancy(df=ethnicity_cases_df)
        ethnicity_deaths_df, ethnicity_deaths_discrepancy_df = split_pandas_into_case_discrepancy(
            df=ethnicity_deaths_df)

        state_layout = run_plot_cases_deaths(
            fig_names=fig_names,
            cases_df=ethnicity_cases_df,
            cases_discrepancy_df=ethnicity_cases_discrepancy_df,
            deaths_df=ethnicity_deaths_df,
            deaths_discrepancy_df=ethnicity_deaths_discrepancy_df)
        tab_list.append(Panel(child=state_layout, title=state_name))
        tab_names.append(state_name)


        logging.info("\n")
        logging.info(f"Processing county level data for {state_name}")
        county_dirs = sorted(os.listdir(path.join('states', state_name, 'counties')))

        if len(county_dirs) > 0:
            for county in county_dirs:
                logging.info(f"County {county}")
                state_county_dir = path.join('states', state_name, 'counties', county)

                ethnicity_cases_df = pd.read_csv(f"{state_county_dir}/{cases_csv_filename}")
                ethnicity_deaths_df = pd.read_csv(f"{state_county_dir}/{deaths_csv_filename}")
                ethnicity_cases_df, ethnicity_deaths_df = convert_date_str_to_datetime(
                    df_list=[ethnicity_cases_df, ethnicity_deaths_df])

                ethnicity_cases_df, ethnicity_cases_discrepancy_df = split_pandas_into_case_discrepancy(
                    df=ethnicity_cases_df)
                ethnicity_deaths_df, ethnicity_deaths_discrepancy_df = split_pandas_into_case_discrepancy(
                    df=ethnicity_deaths_df)

                state_county_layout = run_plot_cases_deaths(
                    fig_names=fig_names,
                    cases_df=ethnicity_cases_df,
                    cases_discrepancy_df=ethnicity_cases_discrepancy_df,
                    deaths_df=ethnicity_deaths_df,
                    deaths_discrepancy_df=ethnicity_deaths_discrepancy_df)
                tab_name = f"{state_name}: {county}"
                tab_list.append(Panel(child=state_county_layout, title=tab_name))
                tab_names.append(tab_name)

        tabs = Tabs(tabs=tab_list)
        show(tabs)

if __name__ == '__main__':
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    visualize_per_county_stats()
