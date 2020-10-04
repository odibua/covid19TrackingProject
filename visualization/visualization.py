# --------------------------
# Standard Python Imports
# --------------------------
import itertools
import logging
import os
from os import path
from typing import Dict, List, Tuple

# --------------------------
# Third Party Imports
# --------------------------
from bokeh.io import export_png, export_svgs
from bokeh.io import output_file
from bokeh.plotting import ColumnDataSource, figure, show
from bokeh.layouts import column, row
from bokeh.models.widgets import Panel, Tabs
from bokeh.palettes import Dark2_5 as palette

import pandas as pd
import yaml
# --------------------------
# covid19Tracking Imports
# --------------------------


def get_mean_df(df_list: List[pd.DataFrame]) -> Tuple[float]:
    new_df_list = []
    for df in df_list:
        new_df_list.append(df.mean().mean())
    return tuple(new_df_list)


def get_mean_max_df(df_list: List[pd.DataFrame]) -> Tuple[float]:
    new_df_list = []
    for df in df_list:
        new_df_list.append(df.mean().max())
    return new_df_list


def convert_date_str_to_datetime(df_list: List[pd.DataFrame]) -> Tuple[pd.DataFrame]:
    new_df_list = []
    for df in df_list:
        df['date'] = pd.to_datetime(df['date'])
        new_df_list.append(df)
    return tuple(new_df_list)


def drop_other_columns(df_list: List[pd.DataFrame]) -> Tuple[pd.DataFrame]:
    new_df_list = []
    for df in df_list:
        for key in df.keys():
            if 'other' in key.lower():
                df = df.drop(key, axis=1)
        new_df_list.append(df)
    return tuple(new_df_list)


def split_pandas_by_discrepancy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = df.columns.tolist()
    discrepancy_columns = [column for column in columns if 'discrepancy' in column or column ==
                           'date' or column == 'Unnamed: 0']
    columns = [column for column in columns if 'discrepancy' not in column or column == 'date']

    return df[columns], df[discrepancy_columns]


def get_plot_figs(fig_names: List[str]) -> Tuple[figure]:
    return tuple([figure(plot_width=1000, title=fig_name, x_axis_type='datetime',
                         tooltips=[("x", "$x"), ("y", "$y")]) for fig_name in fig_names])


def get_bar_plot_figs(fig_names: List[str], x_ranges: List[str]) -> Tuple[figure]:
    return tuple([figure(title=fig_name, width=1200, x_range=x_ranges) for fig_name in fig_names])


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


def plot_bar_cases_deaths(fig: figure, dict_: Dict[str, float], ref_name: str):
    label_list, fair_list, ref_list, count_list = [], [], [], []
    for key in dict_.keys():
        if key != ref_name:
            label_list.append(key)
            count_list.append(dict_[key])
        ref_list.append(dict_[ref_name])
        fair_list.append(1)
    fig.vbar(x=label_list, top=count_list, width=.9,
             fill_alpha=.5,
             fill_color='salmon',
             line_alpha=.5,
             line_color='green',
             line_dash='dashed')
    fig.line(x=label_list, color='blue', y=fair_list, width=3, legend_label='No Disparity')
    fig.line(x=label_list, color='red', line_dash='dotted', y=ref_list, width=3, legend_label=ref_name)
    fig.xaxis.axis_label = "County"
    fig.yaxis.axis_label = "Disparity"
    fig.title.text_font_size = '18pt'
    fig.xaxis.axis_label_text_font_size = '16pt'
    fig.yaxis.axis_label_text_font_size = '16pt'
    fig.xaxis.major_label_text_font_size = "14pt"
    fig.yaxis.major_label_text_font_size = "14pt"

    fig.legend.label_text_font_size = '18pt'


def run_bar_plot_cases_deaths(fig_names: List[str], cases_mean_disparity: Dict[str, float],
                              cases_max_disparity: Dict[str, float], deaths_mean_disparity: Dict[str, float], deaths_max_disparity: Dict[str, float], state_name: str):
    case_list_names = [key for key in cases_mean_disparity.keys() if key != state_name]
    death_list_names = [key for key in deaths_mean_disparity.keys() if key != state_name]

    mean_case_disparity_fig, max_case_disparity_fig = get_bar_plot_figs(
        fig_names=fig_names[0:2], x_ranges=case_list_names)
    mean_death_disparity_fig, max_death_disparity_fig = get_bar_plot_figs(
        fig_names=fig_names[2:], x_ranges=death_list_names)

    plot_bar_cases_deaths(fig=mean_case_disparity_fig, dict_=cases_mean_disparity, ref_name=state_name)
    plot_bar_cases_deaths(fig=max_case_disparity_fig, dict_=cases_max_disparity, ref_name=state_name)
    plot_bar_cases_deaths(fig=mean_death_disparity_fig, dict_=deaths_mean_disparity, ref_name=state_name)
    plot_bar_cases_deaths(fig=max_death_disparity_fig, dict_=deaths_max_disparity, ref_name=state_name)
    # layout = column(row(mean_case_disparity_fig, mean_death_disparity_fig),
    #                 row(max_case_disparity_fig, max_death_disparity_fig))
    layout = column(mean_case_disparity_fig, mean_death_disparity_fig,
                    max_case_disparity_fig, max_death_disparity_fig)
    return layout


def visualize_summary_stats():
    logging.info("Open State Configuration file and get states to be plotted")
    config_path = 'states/states_config.yaml'
    if not path.isfile(config_path):
        raise ValueError(f"states_config.yaml not found in states directory")
    config_file = open(config_path)
    config = yaml.safe_load(config_file)
    state_list = config['STATES']
    cases_csv_filename, deaths_csv_filename = 'ethnicity_cases.csv', 'ethnicity_deaths.csv'

    logging.info("Create output html file")
    output_file(f"covid_ethnic_summary_discrepancy.html")
    fig_names = [
        'Average Case Disparity', 'Max Case Disparity', 'Average Death Disparity',
        'Max Death Disparity']
    tab_list, tab_names = [], []
    logging.info(f"Load deaths and cases csvs for relevant states and all counties in config")
    for state in state_list:
        state_name = state.lower()
        logging.info(f"Processing {state_name}")
        state_county_dir = os.path.join('states', state_name)
        case_disparity_mean_dict, death_disparity_mean_dict = {}, {}
        case_disparity_max_dict, death_disparity_max_dict = {}, {}

        ethnicity_cases_df = pd.read_csv(f"{state_county_dir}/{cases_csv_filename}", index_col=False)
        ethnicity_deaths_df = pd.read_csv(f"{state_county_dir}/{deaths_csv_filename}", index_col=False)
        ethnicity_cases_df, ethnicity_deaths_df = convert_date_str_to_datetime(
            df_list=[ethnicity_cases_df, ethnicity_deaths_df])
        ethnicity_cases_df, ethnicity_deaths_df = drop_other_columns(df_list=[ethnicity_cases_df, ethnicity_deaths_df])

        _, ethnicity_cases_discrepancy_df = split_pandas_by_discrepancy(df=ethnicity_cases_df)
        _, ethnicity_deaths_discrepancy_df = split_pandas_by_discrepancy(
            df=ethnicity_deaths_df)

        ethnicity_cases_discrepancy_df = ethnicity_cases_discrepancy_df.drop(columns=['date'])
        ethnicity_deaths_discrepancy_df = ethnicity_deaths_discrepancy_df.drop(columns=['date'])

        case_disparity_mean_dict[state_name], death_disparity_mean_dict[state_name] = get_mean_df(
            df_list=[ethnicity_cases_discrepancy_df, ethnicity_deaths_discrepancy_df])
        case_disparity_max_dict[state_name], death_disparity_max_dict[state_name] = get_mean_max_df(
            df_list=[ethnicity_cases_discrepancy_df, ethnicity_deaths_discrepancy_df])

        logging.info("\n")
        logging.info(f"Processing county level data for {state_name}")
        county_dirs = sorted(os.listdir(path.join('states', state_name, 'counties')))

        if len(county_dirs) > 0:
            for county in county_dirs:
                logging.info(f"County {county}")
                state_county_dir = path.join('states', state_name, 'counties', county)
                try:
                    ethnicity_cases_df = pd.read_csv(f"{state_county_dir}/{cases_csv_filename}", index_col=False)
                    ethnicity_cases_df = convert_date_str_to_datetime(
                        df_list=[ethnicity_cases_df])[0]
                    _, ethnicity_cases_discrepancy_df = split_pandas_by_discrepancy(
                        df=ethnicity_cases_df)

                    ethnicity_cases_discrepancy_df = ethnicity_cases_discrepancy_df.drop(columns=['date'])
                    case_disparity_mean_dict[county] = get_mean_df(
                        df_list=[ethnicity_cases_discrepancy_df])[0]
                    case_disparity_max_dict[county] = get_mean_max_df(
                        df_list=[ethnicity_cases_discrepancy_df])[0]
                except BaseException:
                    pass

                try:
                    ethnicity_deaths_df = pd.read_csv(f"{state_county_dir}/{deaths_csv_filename}", index_col=False)
                    ethnicity_deaths_df = convert_date_str_to_datetime(
                        df_list=[ethnicity_deaths_df])[0]
                    _, ethnicity_deaths_discrepancy_df = split_pandas_by_discrepancy(
                        df=ethnicity_deaths_df)
                    ethnicity_deaths_discrepancy_df = ethnicity_deaths_discrepancy_df.drop(columns=['date'])
                    death_disparity_mean_dict[county] = get_mean_df(
                        df_list=[ethnicity_deaths_discrepancy_df])[0]
                    death_disparity_max_dict[county] = get_mean_max_df(
                        df_list=[ethnicity_deaths_discrepancy_df])[0]
                except BaseException:
                    pass
        state_county_layout = run_bar_plot_cases_deaths(
            fig_names=fig_names,
            cases_mean_disparity=case_disparity_mean_dict,
            cases_max_disparity=case_disparity_max_dict,
            deaths_mean_disparity=death_disparity_mean_dict,
            deaths_max_disparity=death_disparity_max_dict,
            state_name=state_name)
        tab_name = f"{state_name}"
        tab_list.append(Panel(child=state_county_layout, title=tab_name))
        tab_names.append(tab_name)

        tabs = Tabs(tabs=tab_list)
        show(tabs)
        export_svgs(state_county_layout, filename="plot.png")

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
        ethnicity_cases_df, ethnicity_deaths_df = drop_other_columns(df_list=[ethnicity_cases_df, ethnicity_deaths_df])

        ethnicity_cases_df, ethnicity_cases_discrepancy_df = split_pandas_by_discrepancy(df=ethnicity_cases_df)
        ethnicity_deaths_df, ethnicity_deaths_discrepancy_df = split_pandas_by_discrepancy(
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
                try:
                    ethnicity_cases_df = pd.read_csv(f"{state_county_dir}/{cases_csv_filename}")
                    ethnicity_cases_df = convert_date_str_to_datetime(
                        df_list=[ethnicity_cases_df])[0]
                    ethnicity_cases_df, ethnicity_cases_discrepancy_df = split_pandas_by_discrepancy(
                        df=ethnicity_cases_df)
                except BaseException:
                    ethnicity_cases_df, ethnicity_cases_discrepancy_df = {}, {}

                try:
                    ethnicity_deaths_df = pd.read_csv(f"{state_county_dir}/{deaths_csv_filename}")
                    ethnicity_deaths_df = convert_date_str_to_datetime(
                        df_list=[ethnicity_deaths_df])[0]
                    ethnicity_deaths_df, ethnicity_deaths_discrepancy_df = split_pandas_by_discrepancy(
                        df=ethnicity_deaths_df)
                except BaseException:
                    ethnicity_deaths_df, ethnicity_deaths_discrepancy_df = {}, {}

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
    # visualize_summary_stats()
