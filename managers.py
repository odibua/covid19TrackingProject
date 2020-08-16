# --------------------------
# Standard Python Imports
# --------------------------
import argparse
import logging
import os
from os import path
from typing import List, Tuple

# --------------------------
# Third Party Imports
# --------------------------
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------
import utils


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
    response_list, response_names, request_type = utils.get_yaml_responses(
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
    utils.save_raw_data(
        save_dir=raw_data_dir,
        response_list=state_response_list,
        data_type_names=state_data_type_names,
        request_type=request_type)


def raw_to_ethnicity_case_csv_manager(state_name: str, county_name: str = None) -> None:
    state_csv_dir = os.path.join('states', state_name, 'csvs')
    if not os.path.isdir(state_csv_dir):
        os.makedirs(state_csv_dir)
    if county_name is None:
        state_county_dir = os.path.join('states', state_name)
        cases_csv_filename = f"{state_name}_ethnicity_cases.csv"
    else:
        state_county_dir = path.join('states', state_name, 'counties', county_name)
        cases_csv_filename = f"{state_name}_{county_name}_ethnicity_cases.csv"

    case_msg = utils.run_ethnicity_to_case_csv(
        state_csv_dir=state_csv_dir, state_county_dir=state_county_dir, state=state_name, county=county_name,
        cases_csv_filename=cases_csv_filename)

    if case_msg is None:
        return
    if len(case_msg) > 0:
        raise ValueError(f"{case_msg}")


def raw_to_ethnicity_death_csv_manager(state_name: str, county_name: str = None) -> None:
    state_csv_dir = os.path.join('states', state_name, 'csvs')
    if not os.path.isdir(state_csv_dir):
        os.makedirs(state_csv_dir)
    if county_name is None:
        state_county_dir = os.path.join('states', state_name)
        deaths_csv_filename = f"{state_name}_ethnicity_deaths.csv"
    else:
        state_county_dir = path.join('states', state_name, 'counties', county_name)
        deaths_csv_filename = f"{state_name}_{county_name}_ethnicity_deaths.csv"

    death_msg = utils.run_ethnicity_to_death_csv(
        state_csv_dir=state_csv_dir, state_county_dir=state_county_dir, state=state_name, county=county_name,
        deaths_csv_filename=deaths_csv_filename)

    if death_msg is None:
        return
    if len(death_msg) > 0:
        raise ValueError(f"{death_msg}")


def main(state_name: str, county_name: str = None, mode: str = 'scrape'):
    if mode == 'scrape':
        scrape_manager(state_name=state_name, county_name=county_name)
    elif mode == 'project_case':
        raw_to_ethnicity_case_csv_manager(state_name=state_name, county_name=county_name)
    elif mode == 'project_death':
        raw_to_ethnicity_death_csv_manager(state_name=state_name, county_name=county_name)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    parser = argparse.ArgumentParser(description='Process mode')
    parser.add_argument('--mode', help='Mode that will determine which managers run')
    parser.add_argument('--state', help='Mode that will determine which managers run')
    parser.add_argument('--county', help='Mode that will determine which managers run', default=None)
    args = parser.parse_args()
    main(mode=args.mode, state_name=args.state, county_name=args.county)
