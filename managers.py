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
from celery import Celery
import pandas as pd
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------
import utils

app = Celery()
app.config_from_object('celeryconfig')


def get_responses_from_config_files_in_dir(config_dir: str) -> Tuple[List[str], List[str], List[str], str]:
    config_files = os.listdir(config_dir)
    config_files = [config_file for config_file in config_files if config_file.endswith('.yaml')]
    if len(config_files) > 0:
        response_list, response_names, failed_response_names, request_type = utils.get_yaml_responses(
            config_dir=config_dir, config_file_list=config_files)
    else:
        response_list, response_names, failed_response_names, request_type = None, None, None, None

    return response_list, response_names, failed_response_names, request_type


def scrape_manager():
    logging.info("Open State Configuration file and get states to be scraped")
    config_path = 'states/states_config.yaml'
    if not path.isfile(config_path):
        raise ValueError(f"states_config.yaml not found in states directory")
    config_file = open(config_path)
    config = yaml.safe_load(config_file)
    state_list = config['STATES']

    logging.info(f"Get and process covid19 ethnicity data for each state and corresponding counties")
    for state in state_list:
        state_name = state.lower()
        logging.info(f"Processing {state_name}")

        state_config_path = path.join('states', state_name, 'configs')
        logging.info("Get state level covid19 raw data with ethnicity")
        state_response_list, state_data_type_names, failed_state_data_type_names, request_type = get_responses_from_config_files_in_dir(
            config_dir=state_config_path)
        if state_response_list is None:
            raise Warning(f"No state level config files exist for {state_name}")
        else:
            logging.info("Save state level raw covid 19 data with ethnicity")
            raw_data_dir = path.join('states', state_name, 'raw_data')
            if not path.isdir(raw_data_dir):
                os.makedirs(raw_data_dir)
            utils.save_raw_data(
                save_dir=raw_data_dir,
                response_list=state_response_list,
                data_type_names=state_data_type_names,
                failed_data_type_names=failed_state_data_type_names,
                request_type=request_type)

        logging.info(f"Processing county level data for {state_name}")
        state_county_dirs = os.listdir(path.join('states', state_name, 'counties'))
        if len(state_county_dirs) > 0:
            for state_county_dir in state_county_dirs:
                logging.info(f"Getting and saving raw data for state: {state_name}, county: {state_county_dir}")
                county_response_list, county_data_type_names, failed_county_data_type_names, request_type = get_responses_from_config_files_in_dir(
                    config_dir=path.join('states', state_name, 'counties', state_county_dir, 'configs'))

                if county_response_list is None:
                    raise Warning(f"No county level config files exist for {state_county_dir}")
                else:
                    logging.info("Save county level raw covid 19 data with ethnicity")
                    raw_data_dir = path.join('states', state_name, 'counties', state_county_dir, 'raw_data')
                    if not path.isdir(raw_data_dir):
                        os.makedirs(raw_data_dir)
                    utils.save_raw_data(save_dir=raw_data_dir, response_list=county_response_list,
                                        data_type_names=county_data_type_names,
                                        failed_data_type_names=failed_county_data_type_names, request_type=request_type)
        else:
            raise Warning(f"No county level data exists for {state_name}")


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
        state_name = state.lower()
        logging.info(f"Processing {state_name}")
        state_county_dir = os.path.join('states', state_name)

        utils.run_ethnicity_to_csv(
            state_county_dir=state_county_dir, state=state_name, county=None, cases_csv_filename=cases_csv_filename, deaths_csv_filename=deaths_csv_filename)

        logging.info(f"Processing county level data for {state_name}")
        county_dirs = os.listdir(path.join('states', state_name, 'counties'))
        county_dirs.sort()
        if len(county_dirs) > 0:
            for county in county_dirs:
                state_county_dir = path.join('states', state_name, 'counties', county)
                utils.run_ethnicity_to_csv(
                    state_county_dir=state_county_dir, state=state_name, county=county, cases_csv_filename=cases_csv_filename, deaths_csv_filename=deaths_csv_filename)
        else:
            raise Warning(f"No county level data exists for {state_name}")

# TODO(odibua@): Create and push to new branch based on date to be later merged in


def add_commit_and_push():
    logging.info("Add, commit, and push updates to raw data")
    dt = datetime.datetime.now() - datetime.timedelta(days=1)
    today = datetime.date(dt.year, dt.month, dt.day)
    today_str = today.isoformat()
    cmd.check_call(["git", "add", "states"])
    message = f"Update states and county raw covid ethnicity data with data from {today_str}"
    cmd.check_call(["git", "commit", "-m", f"{message}"])
    cmd.check_call(["git", "push"])


@app.task
def main(mode: str):
    if mode == 'scrape':
        scrape_manager()
        add_commit_and_push()
    elif mode == 'project':
        raw_to_ethnicity_csv_manager()


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    parser = argparse.ArgumentParser(description='Process mode')
    parser.add_argument('--mode', help='Mode that will determine which managers run')
    args = parser.parse_args()
    main(mode=args.mode)
