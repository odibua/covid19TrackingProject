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


def get_projector_module(state: str, county: str, projector_name: str) -> str:
    if county is None:
        return f"states.{state}.{projector_name}"
    else:
        return f"states.{state}.counties.{county}.{projector_name}"


def filter_dates_from_df(date_list: List[str], df: pd.DataFrame):
    if df is not None:
        return [
            date for date in date_list if date not in df['date'].tolist()]
    else:
        return date_list


def filter_projector_module(projector_candidate_list: List[str]):
    return [
        projector_candidate for projector_candidate in projector_candidate_list if 'projector' in projector_candidate]


def get_class_in_projector_module(module: sys.modules, module_name: str) -> Callable:
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
            if obj.__module__ == module_name:
                obj_list.append(obj)
    if len(obj_list) == 0:
        raise ValueError(f"No class found in {module}")
    if len(obj_list) > 1:
        raise ValueError(f"More than one class found {module}. Projector module should only have on projector class")
    return obj_list[0]


def project_cases(state: str, county: str,
                  date_strings: List[str], projector_class: Callable) -> Tuple[List[Dict[str, int]], List[Dict[str, any]], List[str]]:
    ethnicity_cases_list, ethnicity_cases_discrepancies_list, failed_dates = [], [], []
    for date_string in date_strings:
        try:
            projector_instance = projector_class(state=state, county=county, date_string=date_string)
            projector_instance.process_raw_data_to_cases()
            ethnicity_cases_list.append(projector_instance.ethnicity_cases)
            ethnicity_cases_discrepancies_list.append(projector_instance.ethnicity_cases_discrepancies)
        except BaseException:
            failed_dates.append(date_string)
    return ethnicity_cases_list, ethnicity_cases_discrepancies_list, failed_dates


def project_deaths(state: str, county: str,
                   date_strings: List[str], projector_class: Callable) -> Tuple[List[Dict[str, int]], List[Dict[str, any]], List[str]]:
    ethnicity_deaths_list, ethnicity_deaths_discrepancies_list, failed_dates = [], [], []
    for date_string in date_strings:
        try:
            projector_instance = projector_class(state=state, county=county, date_string=date_string)
            projector_instance.process_raw_data_to_deaths()
            ethnicity_deaths_list.append(projector_instance.ethnicity_deaths)
            ethnicity_deaths_discrepancies_list.append(projector_instance.ethnicity_deaths_discrepancies)
        except BaseException:
            failed_dates.append(date_string)
    return ethnicity_deaths_list, ethnicity_deaths_discrepancies_list, failed_dates


def parse_cases_responses_with_projectors(state: str, county: str, state_county_dir: str,
                                          cases_csv_filename: str) -> Tuple[List[Dict[str, int]], List[Dict[str, any]], List[str]]:
    logging.info("Get state/county projector")
    state_county_dir_list = os.listdir(state_county_dir)
    state_county_projector_list = filter_projector_module(projector_candidate_list=state_county_dir_list)
    if len(state_county_projector_list) != 1:
        raise ValueError(
            f"ERROR: ONLY ONE PROJECTOR SHOULD BE IMPLEMENTED IN DIRECTORY. Found {len(state_county_projector_list)} for directory {state_county_dir}")

    logging.info("Create ethnicity cases and deaths csvs if they don't already exist."
                 "Load if they do exist")
    state_county_cases_csv = os.path.join(state_county_dir, cases_csv_filename)
    state_county_cases_df = None

    logging.info(f"Get projector class for state: {state}, county: {county}")
    module_name = get_projector_module(state=state, county=county, projector_name=state_county_projector_list[0][0:-3])
    state_county_projector_module = importlib.import_module(module_name)

    projector_class = get_class_in_projector_module(module=state_county_projector_module, module_name=module_name)

    logging.info("Load cases ethnicities csv if it does not exist. Create if it does not.")
    if not os.path.isfile(state_county_cases_csv):
        projector_instance = projector_class(state=state, county=county, date_string='2020-07-09')
        headers = projector_instance.ethnicities
        headers = headers + [f"{header}_discrepancy" for header in headers] + ['date']
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
    raw_data_cases_dates = sorted(filter_dates_from_df(date_list=raw_data_dates, df=state_county_cases_df))
    logging.info(f"Get case per ethnicity and case discrepancies for each ethnicity. Create if it does not.")
    ethnicity_cases_list, ethnicity_cases_discrepancies_list, failed_dates = project_cases(
        state=state, county=county, date_strings=raw_data_cases_dates, projector_class=projector_class)
    return ethnicity_cases_list, ethnicity_cases_discrepancies_list, failed_dates


def parse_deaths_responses_with_projectors(state: str, county: str, state_county_dir: str,
                                           deaths_csv_filename: str) -> Tuple[List[Dict[str, int]], List[Dict[str, any]], List[str]]:
    logging.info("Get state/county projector")
    state_county_dir_list = os.listdir(state_county_dir)
    state_county_projector_list = filter_projector_module(projector_candidate_list=state_county_dir_list)
    if len(state_county_projector_list) != 1:
        raise ValueError(
            f"ERROR: ONLY ONE PROJECTOR SHOULD BE IMPLEMENTED IN DIRECTORY. Found {len(state_county_projector_list)} for directory {state_county_dir}")

    logging.info(f"Get projector class for state: {state}, county: {county}")
    module_name = get_projector_module(state=state, county=county, projector_name=state_county_projector_list[0][0:-3])
    state_county_projector_module = importlib.import_module(module_name)
    projector_class = get_class_in_projector_module(module=state_county_projector_module, module_name=module_name)

    logging.info("Create ethnicity cases and deaths csvs if they don't already exist."
                 "Load if they do exist")
    state_county_deaths_csv = os.path.join(state_county_dir, deaths_csv_filename)
    state_county_deaths_df = None

    logging.info("Load deaths ethnicities csv if it does not exist. Create if it does not.")
    if not os.path.isfile(state_county_deaths_csv):
        projector_instance = projector_class(state=state, county=county, date_string='2020-07-09')
        headers = projector_instance.ethnicities
        headers = headers + [f"{header}_discrepancy" for header in headers] + ['date']
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
    raw_data_deaths_dates = sorted(filter_dates_from_df(date_list=raw_data_dates, df=state_county_deaths_df))
    logging.info(f"Get case per ethnicity and case discrepancies for each ethnicity")
    ethnicity_dates_list, ethnicity_deaths_discrepancies_list, failed_dates = project_deaths(
        state=state, county=county, date_strings=raw_data_deaths_dates, projector_class=projector_class)

    return ethnicity_dates_list, ethnicity_deaths_discrepancies_list, failed_dates


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


def run_ethnicity_to_csv(state_county_dir: str, state: str,
                         county: Union[str, None], cases_csv_filename: str, deaths_csv_filename: str):
    logging.info(f"Get state ethnicity cases and deaths counts and discrepancies")
    state_ethnicity_cases_list, state_ethnicity_cases_discrepancies_list, failed_cases_dates_list = parse_cases_responses_with_projectors(
        state=state, county=county, state_county_dir=state_county_dir, cases_csv_filename=cases_csv_filename)
    state_ethnicity_deaths_list, state_ethnicity_deaths_discrepancies_list, failed_deaths_dates_list = parse_deaths_responses_with_projectors(
        state=state, county=county, state_county_dir=state_county_dir, deaths_csv_filename=deaths_csv_filename)

    if len(failed_cases_dates_list) > 0:
        msg = f"ERROR IN CASE PROJECTION FOR STATE: {state} COUNTY: {county} Num Failed Dates:{len(failed_cases_dates_list)} Failed dates: {failed_cases_dates_list}"
        logging.info(msg)
        f_obj = open(f"{state_county_dir}/failed_cases_projection.txt", 'w')
        f_obj.write(msg)
    if len(failed_deaths_dates_list) > 0:
        msg = f"ERROR IN DEATH PROJECTION FOR STATE: {state} COUNTY: {county} Num Failed Dates:{len(failed_deaths_dates_list)} Failed dates: {failed_deaths_dates_list}"
        logging.info(msg)
        f_obj = open(f"{state_county_dir}/failed_deaths_projection.txt", 'w')
        f_obj.write(msg)

    state_ethnicity_cases_df, state_ethnicity_cases_discrepancies_df = pd.DataFrame(
        state_ethnicity_cases_list), pd.DataFrame(state_ethnicity_cases_discrepancies_list)
    state_ethnicity_deaths_df, state_ethnicity_deaths_discrepancies_df = pd.DataFrame(
        state_ethnicity_deaths_list), pd.DataFrame(state_ethnicity_deaths_discrepancies_list)

    try:
        state_ethnicity_full_cases_df = state_ethnicity_cases_df.merge(
            state_ethnicity_cases_discrepancies_df, left_on='date', right_on='date', suffixes=('', '_discrepancy'))
        state_ethnicity_full_cases_df.to_csv(f"{state_county_dir}/{cases_csv_filename}")
    except BaseException:
        pass

    try:
        state_ethnicity_full_deaths_df = state_ethnicity_deaths_df.merge(
            state_ethnicity_deaths_discrepancies_df, left_on='date', right_on='date', suffixes=('', '_discrepancy'))
        state_ethnicity_full_deaths_df.to_csv(f"{state_county_dir}/{deaths_csv_filename}")
    except BaseException:
        pass


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
