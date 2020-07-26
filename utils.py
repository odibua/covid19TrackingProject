# --------------------------
# Standard Python Imports
# --------------------------
import csv
import datetime
import importlib
import inspect
import json
import logging
import os
from os import path
import sys
from typing import Callable, Dict, List, Tuple, Union

# --------------------------
# Third Party Imports
# --------------------------
import pandas as pd
import requests
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------


def get_class_in_projector_module(module: sys.modules) -> Callable:
    """
    Get class in projector module

    Arguments:
        module: Module to import class from

    Return:
        obj: Projector class
    """
    obj_list = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            obj_list.append(obj)

    if len(obj_list) == 0:
        raise ValueError(f"No class found in {module}")
    if len(obj_list) > 1:
        raise ValueError(f"More than one class found {module}. Projector module should only have on projector class")
    return obj_list[0]


def project_cases(state: str, county: str,
                  date_strings: List[str], projector_class: Callable) -> Tuple[List[Dict[str, int]], List[Dict[str, any]]]:
    ethnicity_cases_list, ethnicity_cases_discrepancies_list = [], []
    for date_string in date_strings:
        projector_instance = projector_class(state=state, county=county, date_string=date_string)
        projector_instance.process_raw_data_to_cases()
        ethnicity_cases_list.append(projector_instance.ethnicity_cases)
        ethnicity_cases_discrepancies_list.append(projector_instance.ethnicity_cases_discrepancies)
    return ethnicity_cases_list, ethnicity_cases_discrepancies_list


def project_deaths(state: str, county: str,
                   date_strings: List[str], projector_class: Callable) -> Tuple[List[Dict[str, int]], List[Dict[str, any]]]:
    ethnicity_deaths_list, ethnicity_deaths_discrepancies_list = [], []
    for date_string in date_strings:
        projector_instance = projector_class(state=state, county=county, date_string=date_string)
        projector_instance.process_raw_data_to_deaths()
        ethnicity_deaths_list.append(projector_instance.ethnicity_cases)
        ethnicity_deaths_discrepancies_list.append(projector_instance.ethnicity_cases_discrepancies)
    return ethnicity_deaths_list, ethnicity_deaths_discrepancies_list


def parse_cases_responses_with_projectors(state: str, county: str, state_county_dir: str,
                                          cases_csv_filename: str) -> Tuple[List[Dict[str, int]], List[Dict[str, any]]]:
    logging.info("Get state/county projector")
    state_county_dir_list = os.listdir(state_county_dir)
    state_county_projector_list = [
        state_county_projector for state_county_projector in state_county_dir_list if state_county_projector.find('projector')]
    if len(state_county_projector_list) != 1:
        raise ValueError(
            f"ERROR: ONLY ONE PROJECTOR SHOULD BE IMPLEMENTED IN DIRECTORY. Found {len(state_county_projector_list)} for directory {state_county_dir}")

    logging.info("Create ethnicity cases and deaths csvs if they don't already exist."
                 "Load if they do exist")
    state_county_cases_csv = os.path.join(state_county_dir, cases_csv_filename)
    state_county_cases_df = None

    logging.info(f"Get projector class for state: {state}, county: {county}")
    state_county_projector_module = importlib.import_module(state_county_projector_list[0])
    projector_class = get_class_in_projector_module(module=state_county_projector_module)

    logging.info("Load cases ethnicities csv if it does not exist. Create if it does not.")
    if not os.path.isfile(state_county_cases_csv):
        headers = projector_class.ethnicities + ['date']
        f_obj = open(state_county_cases_csv, 'w+')
        w_obj = csv.writer(f_obj, delimiter=',')
        w_obj.writerow(headers)
        f_obj.close()
    else:
        state_county_cases_df = pd.read_csv(state_county_cases_csv)

    logging.info(f"Load raw data directories for state: {state}, county: {county}")
    raw_data_dir = os.path.join(state_county_dir, "raw_data")

    logging.info(f"Get raw data dates if not already in cases data frames.")
    raw_data_dates = os.listdir(raw_data_dir)
    raw_data_cases_dates = []
    if state_county_cases_df is not None:
        raw_data_cases_dates = [
            raw_data_date for raw_data_date in raw_data_dates if raw_data_date not in state_county_cases_df['Date'].tolist()]

    logging.info(f"Get case per ethnicity and case discrepancies for each ethnicity. Create if it does not.")
    ethnicity_cases_list, ethnicity_cases_discrepancies_list = project_cases(
        state=state, county=county, date_strings=raw_data_cases_dates, projector_class=projector_class)
    return ethnicity_cases_list, ethnicity_cases_discrepancies_list


def parse_deaths_responses_with_projectors(state: str, county: str, state_county_dir: str,
                                           deaths_csv_filename: str) -> Tuple[List[Dict[str, int]], List[Dict[str, any]]]:
    logging.info("Get state/county projector")
    state_county_dir_list = os.listdir(state_county_dir)
    state_county_projector_list = [
        state_county_projector for state_county_projector in state_county_dir_list if state_county_projector.find('projector')]
    if len(state_county_projector_list) != 1:
        raise ValueError(
            f"ERROR: ONLY ONE PROJECTOR SHOULD BE IMPLEMENTED IN DIRECTORY. Found {len(state_county_projector_list)} for directory {state_county_dir}")

    logging.info(f"Get projector class for state: {state}, county: {county}")
    state_county_projector_module = importlib.import_module(state_county_projector_list[0])
    projector_class = get_class_in_projector_module(module=state_county_projector_module)

    logging.info("Create ethnicity cases and deaths csvs if they don't already exist."
                 "Load if they do exist")
    state_county_deaths_csv = os.path.join(state_county_dir, deaths_csv_filename)
    state_county_cases_df, state_county_deaths_df = None, None

    logging.info("Load deaths ethnicities csv if it does not exist. Create if it does not.")
    if not os.path.isfile(state_county_deaths_csv):
        headers = projector_class.ethnicities + ['date']
        f_obj = open(state_county_deaths_csv, 'w+')
        w_obj = csv.writer(f_obj, delimiter=',')
        w_obj.writerow(headers)
        f_obj.close()
    else:
        state_county_deaths_df = pd.read_csv(state_county_deaths_csv)

    logging.info(f"Load raw data directories for state: {state}, county: {county}")
    raw_data_dir = os.path.join(state_county_dir, "raw_data")

    logging.info(f"Get raw data dates if not already in the deathsdata frames.")
    raw_data_dates = os.listdir(raw_data_dir)
    raw_data_deaths_dates = []
    if state_county_deaths_df is not None:
        raw_data_deaths_dates = [
            raw_data_date for raw_data_date in raw_data_dates if raw_data_date not in state_county_deaths_df['Date'].tolist()]

    logging.info(f"Get case per ethnicity and case discrepancies for each ethnicity")
    ethnicity_dates_list, ethnicity_deaths_discrepancies_list = project_cases(
        state=state, county=county, date_strings=raw_data_deaths_dates, projector_class=projector_class)
    return ethnicity_dates_list, ethnicity_deaths_discrepancies_list


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
