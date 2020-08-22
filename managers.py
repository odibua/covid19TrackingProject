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


def raw_to_ethnicity_csv_manager():
    logging.info("Open State Configuration file and get states to be processed")
    config_path = 'states/states_config.yaml'
    if not path.isfile(config_path):
        raise ValueError(f"states_config.yaml not found in states directory")
    config_file = open(config_path)
    config = yaml.safe_load(config_file)
    state_list = config['STATES']
    cases_csv_filename, deaths_csv_filename = 'ethnicity_cases.csv', 'ethnicity_deaths.csv'

    logging.info(f"Get and process covid19 ethnicity data for each state and corresponding counties")
    for state in state_list:
        failure_list = []
        state_name = state.lower()
        logging.info(f"Processing {state_name}")
        state_county_dir = os.path.join('states', state_name)

        failure_state_county = utils.run_ethnicity_to_csv(
            state_county_dir=state_county_dir, state=state_name, county=None, cases_csv_filename=cases_csv_filename, deaths_csv_filename=deaths_csv_filename)
        failure_list.extend(failure_state_county)

        logging.info("\n")
        logging.info(f"Processing county level data for {state_name}")
        county_dirs = sorted(os.listdir(path.join('states', state_name, 'counties')))
        if len(county_dirs) > 0:
            for county in county_dirs:
                state_county_dir = path.join('states', state_name, 'counties', county)
                failure_state_county = utils.run_ethnicity_to_csv(
                    state_county_dir=state_county_dir, state=state_name, county=county, cases_csv_filename=cases_csv_filename, deaths_csv_filename=deaths_csv_filename)
                failure_list.extend(failure_state_county)
        else:
            raise Warning(f"No county level data exists for {state_name}")
        failure_dir = f"states/{state_name}/failed_text"
        utils.save_errors(save_dir=failure_dir, failure_list=failure_list, mode='project')


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
    except:
        pass


def main(state_name: str, county_name: str = None, mode: str = 'scrape'):
    if mode == 'scrape':
        scrape_manager(state_name=state_name, county_name=county_name)
    elif mode == 'project':
        raw_to_ethnicity_csv_manager()


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    parser = argparse.ArgumentParser(description='Process mode')
    parser.add_argument('--mode', help='Mode that will determine which managers run')
    parser.add_argument('--state', help='Mode that will determine which managers run')
    parser.add_argument('--county', help='Mode that will determine which managers run', default=None)
    args = parser.parse_args()
    main(mode=args.mode, state_name=args.state, county_name=args.county)
