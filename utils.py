# --------------------------
# Standard Python Imports
# --------------------------
import datetime
import json
import logging
import os
from os import path
from typing import Dict, List, Tuple, Union

# --------------------------
# Third Party Imports
# --------------------------
import pandas as pd
import requests
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------


def parse_responses_with_projectors(state: str, county: str, state_county_dir: str) -> Dict[str, Union[int, float]]:
    logging.info("Get state/county projector")
    state_county_dir_list = os.listdir(state_county_dir)
    state_county_projector_list = [
        state_county_projector for state_county_projector in state_county_dir_list if state_county_projector.find('projector')]
    if len(state_county_projector_list) != 1:
        raise ValueError(
            f"ERROR: ONLY ONE PROJECTOR SHOULD BE IMPLEMENTED IN DIRECTORY. Found {len(state_county_projector_list)} for directory {state_county_dir}")

    logging.info("Create ethnicity cases and deaths csvs if they don't already exist."
                 "Load if they do exist")
    state_county_cases_csv = os.path.join(state_county_dir,"ethnicity_cases.csv")
    state_county_deaths_csv = os.path.join(state_county_dir,"ethnicity_deaths.csv")
    state_county_cases_df, state_county_deaths_df = None, None

    if not os.path.isfile(state_county_cases_csv):
        open(state_county_cases_csv , "w+")
    else:
        state_county_cases_df = pd.read_csv(state_county_cases_csv)
    if not os.path.isfile(state_county_deaths_csv):
        open(state_county_deaths_csv, 'w+')
    else:
        state_county_deaths_df = pd.read_csv(state_county_deaths_csv)

    logging.info(f"Load raw data directories for state: {state}, county: {county}")
    raw_data_dir = os.path.join(state_county_dir, "raw_data")
    raw_data_dates = os.listdir(raw_data_dir)

    if state_county_cases_df is not None:
        raw_data_cases_dates = [raw_data_date for raw_data_date in raw_data_dates if raw_data_date not in state_county_cases_df['Date'].tolist()]
    if state_county_deaths_df is not None:
        raw_data_deaths_dates = [raw_data_date for raw_data_date in raw_data_dates if raw_data_date not in state_county_deaths_df['Date'].tolist()]






def get_yaml_responses(config_dir: str, config_file_list: List[str]) -> Tuple[List[str], List[str], List[str], str]:
    response_list, response_names, failed_response_names = [], [], []
    for config_file in config_file_list:
        config_file_obj = open(path.join(config_dir, config_file))
        response_config = yaml.safe_load(config_file_obj)
        if 'REQUEST' in response_config.keys():
            data_type_name = response_config['NAME'].lower() + '_' + response_config['DATA_TYPE'].lower()
            url = response_config['REQUEST']['URL']
            request_type = response_config['REQUEST']['TYPE']
            if request_type == 'GET':
                headers = response_config['REQUEST']['HEADERS']
                response = requests.get(url=url, headers=headers)
                status_code = response.status_code
            elif request_type == 'POST':
                headers = response_config['REQUEST']['HEADERS']
                payload = response_config['REQUEST']['PAYLOAD']
                response = requests.post(url=url, headers=headers, json=json.loads(json.dumps(payload)))
                status_code = response.status_code
            else:
                raise ValueError(f"Request only implemented for GET or POST types. Got {request_type}")

            if status_code == 200:
                response_list.append(response.text)
                response_names.append(data_type_name)
            else:
                logging.info(f"ERROR: Response for {data_type_name} failed with status {status_code}")
                failed_response_names.append(data_type_name)

            response.close()
    return response_list, response_names, failed_response_names, request_type


def save_raw_data(save_dir: str, response_list: List[str], data_type_names: List[str],
                  failed_data_type_names: List[str], request_type: str):
    dt = datetime.datetime.now() - datetime.timedelta(days=1)
    today = datetime.date(dt.year, dt.month, dt.day)
    today_str = today.isoformat()
    save_dir = f"{save_dir}/{today_str}"
    if not path.isdir(save_dir):
        os.makedirs(save_dir)
    save_dir_files = os.listdir(save_dir)
    if len(save_dir_files) == 0 or 'failed_queries' in save_dir_files:
        for response, data_type_name in zip(response_list, data_type_names):
            if request_type == 'GET':
                save_path = f"{save_dir}/{data_type_name}.html"
            else:
                save_path = f"{save_dir}/{data_type_name}"
            text_file = open(save_path, "w")
            text_file.write(response)
            text_file.close()
            if path.isfile(f"{save_dir}/failed_queries"):
                os.remove(f"{save_dir}/failed_queries")

    failed_save_path = f"{save_dir}/failed_queries"
    if len(failed_data_type_names):
        with open(failed_save_path, 'w') as f:
            for failed_data_type_name in failed_data_type_names:
                f.write(f"{failed_data_type_name}\n")
