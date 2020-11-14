# --------------------------
# Standard Python Imports
# --------------------------
import argparse
import datetime
import logging
import os
from os import path
import subprocess as cmd
from typing import List, Tuple

# --------------------------
# Third Party Imports
# --------------------------
import numpy as np
import pandas as pd

# --------------------------
# covid19Tracking Imports
# --------------------------
import regression_utils
import utils_lib


def get_responses_from_config_files_in_dir(config_dir: str) -> Tuple[List[str], List[str], str]:
    """
    Wrapper that lists config files in a directory, and calls a function that gets requests from them

    Arguments:
        config_dir: Config directory

    Returns:
        response_list: Get list of responses texts from requests
        response_names: List of names containing the location and what is being looked for based on the associated config file
        request_type: Return request type
    """
    config_files = os.listdir(config_dir)
    config_files = [config_file for config_file in config_files if config_file.endswith('.yaml')]
    response_list, response_names, request_type = utils_lib.get_yaml_responses(
        config_dir=config_dir, config_file_list=config_files)

    return response_list, response_names, request_type


def scrape_manager(state_name: str, county_name: str = None) -> None:
    """
    Scraping manager that uses the config file associated with a particular state and county to collect raw data
    about COVID cases and deaths. It can be adopted for other purposes.

    Arguments:
        state_name: State to scrape from
        county_name: County to scrape from. Defaults to 0

    Returns:
        None
    """
    logging.info(f"Create raw data and config directory for state: {state_name} county: {county_name}")
    if county_name is None:
        state_config_path = path.join('states', state_name, 'configs')
        raw_data_dir = path.join('states', state_name, 'raw_data')
    else:
        state_config_path = path.join('states', state_name, 'counties', county_name, 'configs')
        raw_data_dir = path.join('states', state_name, 'counties', county_name, 'raw_data')

    logging.info(f"Get responses from text file")
    state_response_list, state_data_type_names, request_type = get_responses_from_config_files_in_dir(
        config_dir=state_config_path)

    if not path.isdir(raw_data_dir):
        os.makedirs(raw_data_dir)
    utils_lib.save_raw_data(
        save_dir=raw_data_dir,
        response_list=state_response_list,
        data_type_names=state_data_type_names,
        request_type=request_type)


def case_parser_manager(state_name: str, county_name: str = None) -> None:
    state_csv_dir = os.path.join('states', state_name, 'csvs')
    if not os.path.isdir(state_csv_dir):
        os.makedirs(state_csv_dir)
    if county_name is None:
        state_county_dir = os.path.join('states', state_name)
        cases_csv_filename = f"{state_name}_ethnicity_cases.csv"
    else:
        state_county_dir = path.join('states', state_name, 'counties', county_name)
        cases_csv_filename = f"{state_name}_{county_name}_ethnicity_cases.csv"

    case_msg = utils_lib.run_ethnicity_to_case_csv(
        state_csv_dir=state_csv_dir, state_county_dir=state_county_dir, state=state_name, county=county_name,
        cases_csv_filename=cases_csv_filename)

    try:
        add_commit_and_push(state_county_dir=state_csv_dir)
    except BaseException:
        pass
    if case_msg is None:
        return
    if len(case_msg) > 0:
        if 'WARNING' in case_msg:
            logging.warning(f"{case_msg}")
        else:
            raise ValueError(f"{case_msg}")


def death_parser_manager(state_name: str, county_name: str = None) -> None:
    state_csv_dir = os.path.join('states', state_name, 'csvs')
    if not os.path.isdir(state_csv_dir):
        os.makedirs(state_csv_dir)
    if county_name is None:
        state_county_dir = os.path.join('states', state_name)
        deaths_csv_filename = f"{state_name}_ethnicity_deaths.csv"
    else:
        state_county_dir = path.join('states', state_name, 'counties', county_name)
        deaths_csv_filename = f"{state_name}_{county_name}_ethnicity_deaths.csv"

    death_msg = utils_lib.run_ethnicity_to_death_csv(
        state_csv_dir=state_csv_dir, state_county_dir=state_county_dir, state=state_name, county=county_name,
        deaths_csv_filename=deaths_csv_filename)

    try:
        add_commit_and_push(state_county_dir=state_csv_dir)
    except BaseException:
        pass
    if death_msg is None:
        return
    if len(death_msg) > 0:
        if 'WARNING' in death_msg:
            logging.warning(f"{death_msg}")
        else:
            raise ValueError(f"{death_msg}")


def metadata_manager(state_name: str, county_name: str = None) -> None:
    """
    Scraping manager that uses the config file associated with a particular state and county to collect raw data
    about COVID cases and deaths. It can be adopted for other purposes.

    Arguments:
        state_name: State to scrape from
        county_name: County to scrape from. Defaults to 0

    Returns:
        None
    """
    logging.info(f"Create raw data and config directory for state: {state_name} county: {county_name}")
    if county_name is None:
        state_config_path = path.join('states', state_name, 'configs')
        state_county_dir = path.join('states', state_name)
    else:
        state_config_path = path.join('states', state_name, 'counties', county_name, 'configs')
        state_county_dir = path.join('states', state_name, 'counties', county_name)

    # Get dataframe of raw metadata for a state and/or county and save the data to
    # a csv file
    metadata_df = utils_lib.get_raw_metadata_from_config_files(config_dir=state_config_path)
    utils_lib.save_data(
        state_name=state_name,
        county_name=county_name,
        data_df=metadata_df,
        data_dir='meta_data_csv',
        data_suffix='metadata')

    # Process raw metadata and save processed medatada and save data to a csv file
    processed_metadata_df = utils_lib.process_raw_metadata(
        raw_metadata_df=metadata_df,
        config_dir=state_config_path,
        state=state_name,
        county=county_name,
        state_county_dir=state_county_dir)
    utils_lib.save_data(
        state_name=state_name,
        county_name=county_name,
        data_df=processed_metadata_df,
        data_dir='processed_meta_data_csv',
        data_suffix='processed_metadata')

    # Aggregate processed metadata based on mapping of regionally defined ethnicities
    # to ACS ethnicities.
    aggregated_processed_metadata_df = utils_lib.aggregate_processed_raw_metadata(
        processed_metadata_df=processed_metadata_df,
        state=state_name,
        county=county_name,
        state_county_dir=state_county_dir)
    utils_lib.save_data(
        state_name=state_name,
        county_name=county_name,
        data_df=aggregated_processed_metadata_df,
        data_dir='aggregated_processed_meta_data_csv',
        data_suffix='aggregated_processed_metadata')


def training_data_manager(state_name: str, type: str, county_name: str = None) -> None:
    logging.info(f"Create raw data and config directory for state: {state_name} county: {county_name}")
    # Define paths and files containing region covid case rates data
    # and metadata
    metadata_path = path.join('states', state_name, 'aggregated_processed_meta_data_csv')
    csv_path = path.join('states', state_name, 'csvs')
    training_csv_path = path.join('states', state_name, 'training_data_csvs')
    if county_name is None:
        metadata_file = f'{state_name}_aggregated_processed_metadata.csv'
        csv_file = f'{state_name}_ethnicity_{type}.csv'
        training_file = f'{state_name}_training_{type}.csv'
    else:
        metadata_file = f'{state_name}_{county_name}_aggregated_processed_metadata.csv'
        csv_file = f'{state_name}_{county_name}_ethnicity_{type}.csv'
        training_file = f'{state_name}_{county_name}_training_{type}.csv'

    # Get earliest date for case files
    csv_file_list = os.listdir(csv_path)
    csv_file_list = [path.join(csv_path, csv_file) for csv_file in csv_file_list if 'case' in csv_file]
    earliest_date = utils_lib.get_earliest_date_string(csv_file_list=csv_file_list)

    # Get rate columns from csv file
    csv_df = pd.read_csv(path.join(csv_path, csv_file))
    columns = csv_df.keys()
    rate_columns = [column for column in columns if 'rates' in column or column == 'date']
    rate_df = csv_df[rate_columns]

    # Add time column that is based on days
    date_df = pd.to_datetime(rate_df['date'])
    rate_df['time'] = (date_df - earliest_date).dt.days

    # Load aggregated metadata data frame
    aggregated_processed_metadata_df = pd.read_csv(path.join(metadata_path, metadata_file), index_col=0)

    # Get columns that have values that unique values for mortality rates
    # and store them in a dictionary along with relevant regional features
    training_data_dict = {'mortality_rate': [], 'time': []}
    for metadata_name in aggregated_processed_metadata_df.keys():
        training_data_dict[metadata_name] = []
    for column in rate_df.keys():
        ethnicity = column.split('_rates')[0]
        if column != 'date' and column != 'time' and ethnicity.lower() != 'other':
            column_df = rate_df[column]
            delta_df = column_df[1:].subtract(column_df[0:-1].tolist())
            change_bool = (delta_df.abs() > 0).tolist()
            change_bool = [True] + change_bool
            column_df = column_df[change_bool]

            training_data_dict['mortality_rate'].extend(column_df.tolist())
            training_data_dict['time'].extend(rate_df['time'][change_bool])

            # Fill in metadata for the region
            for metadata_name in aggregated_processed_metadata_df.keys():
                metadata_vals = aggregated_processed_metadata_df.loc[ethnicity, metadata_name]
                training_data_dict[metadata_name].extend([metadata_vals] * len(column_df.tolist()))
    training_data_df = pd.DataFrame(training_data_dict)

    if not os.path.exists(training_csv_path):
        os.mkdir(training_csv_path)

    training_data_df.to_csv(path.join(training_csv_path, training_file))


def regression_manager(state_name: str, type: str, county_name: str = None, regression_type: str='multilinear') -> None:
    logging.info(f"Create raw data and config directory for state: {state_name} county: {county_name}")
    # Define path and file for training data
    training_csv_path = path.join('states', state_name, 'training_data_csvs')
    if county_name is None:
        training_file = f'{state_name}_training_{type}.csv'
    else:
        training_file = f'{state_name}_{county_name}_training_{type}.csv'

    training_data_df = pd.read_csv(path.join(training_csv_path, training_file), index_col=0)

    # Set Y as mortality rate
    Y = np.array(training_data_df['mortality_rate'])

    # Find range of days that will be used to construct X
    days_range = training_data_df['time'].max() - training_data_df['time'].min() + 1

    # Populate remaining columns with corresponding metadata
    filter_list = ['mortality_rate']
    metadata_keys = training_data_df.keys()
    metadata_keys = [key for key in metadata_keys if key not in filter_list]

    # Construct X
    X = np.zeros((training_data_df.shape[0], training_data_df.shape[1] - len(filter_list)))

    for idx, key in enumerate(metadata_keys):
        X[:, idx] = training_data_df[key].tolist()

    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    regression_info = regression_utils.call_multilinear_regression(X=X, Y=Y)
    print(regression_info)
    print(metadata_keys)


def add_commit_and_push(state_county_dir: str):
    try:
        logging.info("Add, commit, and push updates to raw data")
        dt = datetime.datetime.now() - datetime.timedelta(days=1)
        today = datetime.date(dt.year, dt.month, dt.day)
        today_str = today.isoformat()
        cmd.check_call(["git", "add", f"{state_county_dir}"])
        message = f"Update {state_county_dir} raw covid ethnicity data with data from {today_str}"
        cmd.check_call(["git", "commit", "-m", f"{message}"])
        cmd.check_call(["git", "push"])
    except BaseException:
        pass


def main(state_name: str, county_name: str = None, mode: str = 'scrape'):
    if mode == 'scrape':
        scrape_manager(state_name=state_name, county_name=county_name)
    elif mode == 'project_case':
        case_parser_manager(state_name=state_name, county_name=county_name)
    elif mode == 'project_death':
        death_parser_manager(state_name=state_name, county_name=county_name)
    elif mode == 'scrape_metadata':
        metadata_manager(state_name=state_name, county_name=county_name)
    elif mode == 'create_case_training_data':
        training_data_manager(state_name=state_name, county_name=county_name, type='cases')
    elif mode == 'create_death_training_data':
        training_data_manager(state_name=state_name, county_name=county_name, type='deaths')
    elif mode == 'perform_cases_multilinear_regression':
        regression_manager(state_name=state_name, county_name=county_name, type='cases')
    elif mode == 'perform_deaths_multilinear_regression':
        regression_manager(state_name=state_name, county_name=county_name, type='deaths')


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    parser = argparse.ArgumentParser(description='Process mode')
    parser.add_argument('--mode', help='Mode that will determine which managers run')
    parser.add_argument('--state', help='Mode that will determine which managers run')
    parser.add_argument('--county', help='Mode that will determine which managers run', default=None)
    args = parser.parse_args()
    main(mode=args.mode, state_name=args.state, county_name=args.county)
