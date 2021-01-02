# --------------------------
# Standard Python Imports
# --------------------------
import copy
import datetime
import importlib
import inspect
import json
import logging
import numpy as np
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


def create_dir_if_not_exists(dir: str) -> None:
    if not os.path.isdir(dir):
        os.makedirs(dir)


def create_files_name_with_ethnicity(file: str, ethnicity_filter_list: List[str], ext: str = 'csv') -> str:
    if len(ethnicity_filter_list) == 0:
        file = f'{file}.{ext}'
    else:
        for ethnicity in ethnicity_filter_list:
            file = f'{file}_{ethnicity}'
        file = f'{file}.{ext}'
    return file


def save_df_to_path(df: pd.DataFrame, path: str, file: str) -> None:
    file = os.path.join(path, file)
    if not os.path.isfile(file):
        df.to_csv(file, index=False)
    else:
        df.to_csv(file, mode='a', header=False, index=False)


def check_valid_change(state: str, county: str, date_string: str,
                       dict1: Dict[str, float], dict2: Dict[str, float], type_: str) -> Tuple[bool, str]:
    """
    Throw an error if the projection from raw data to processed data is anomalous either because of size or
    differences in keys

    Arguments:
        state: State for which we are checking valid change
        county: County for which we are checking valid change
        date_string: Date for which validity is checked
        dict1: Dictionary for comparison
        dict2: Dictionary for comparison
        type_:  Type of projection to check

    Returns:
         boolean

    """
    config_dir = f"states/{state}/configs" if county is None else f"states/{state}/counties/{county}/configs"
    config_file = "projector_exceptions.yaml"
    config_file_obj = open(f"{config_dir}/{config_file}")
    exception_config = yaml.safe_load(config_file_obj)

    if type_ == 'case':
        if exception_config['CASE_DATES'] is not None:
            if date_string in exception_config['CASE_DATES']:
                return True, ''
    elif type_ == 'death':
        if exception_config['DEATH_DATES'] is not None:
            if date_string in exception_config['DEATH_DATES']:
                return True, ''

    if dict1 is None or dict2 is None:
        return True, ''

    dict1_not_nan_keys, dict2_not_nan_keys = [], []
    for key in dict1.keys():
        if key != 'date':
            if not np.isnan(dict1[key]):
                dict1_not_nan_keys.append(key)
    for key in dict2.keys():
        if key != 'date':
            if not np.isnan(dict2[key]):
                dict2_not_nan_keys.append(key)
    if len(dict1_not_nan_keys) != len(dict2_not_nan_keys):
        msg = f"ERROR state: {state} county: {county} {dict1} \n != {dict2}"
        return False, msg

    diff_list = []
    for key in dict1.keys():
        if key != 'date' and not np.isnan(dict1[key]) and dict2[key] != 0:
            try:
                diff_list.append((dict1[key] - dict2[key]) / dict2[key])
            except BaseException:
                msg = f"ERROR state: {state} county: {county} with keys. {dict1} \n != {dict2}"
                return False, msg

    if min(diff_list) < 0 and abs(min(diff_list)) > 0.05:
        msg = f"ERROR state: {state} county: {county} Difference {min(diff_list)} is less than 0" \
            f" {dict1} \n != {dict2}"
        return False, msg

    diff_list = [abs(diff) for diff in diff_list]
    if type_ == 'case':
        if max(diff_list) > exception_config['CASE_THRESH']:
            msg = f"ERROR state: {state} county: {county} Max difference {max(diff_list)} is greater than thresh: {exception_config['CASE_THRESH']}" \
                f" {dict1} \n != {dict2}"
            return False, msg
    elif type_ == 'death':
        if max(diff_list) > exception_config['DEATH_THRESH']:
            msg = f"ERROR state: {state} county: {county} Max difference {max(diff_list)} is greater than thresh: {exception_config['DEATH_THRESH']}" \
                f" {dict1} \n != {dict2}"
            return False, msg
    return True, ''


def get_earliest_date_string(csv_file_list: List[str]) -> str:
    min_date = None
    for csv_file in csv_file_list:
        df = pd.read_csv(csv_file)
        df['date'] = pd.to_datetime(df['date'])
        date = df['date'].min()

        if min_date is None:
            min_date = date
        else:
            if date < min_date:
                min_date = date
    return min_date


def get_projector_module(state: str, county: str, projector_name: str) -> str:
    if county is None:
        return f"states.{state}.{projector_name}"
    else:
        return f"states.{state}.counties.{county}.{projector_name}"


def filter_dates_from_df(date_list: List[str], df: pd.DataFrame):
    if df is not None:
        return [
            date for date in date_list if date not in df['date'].tolist()], list(set(df['date'].tolist()))
    else:
        return date_list, []


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


def get_projector_class(state: str, county: str, state_county_dir: str) -> Callable:
    state_county_dir_list = os.listdir(state_county_dir)
    state_county_projector_list = filter_projector_module(projector_candidate_list=state_county_dir_list)
    if len(state_county_projector_list) != 1:
        raise ValueError(
            f"ERROR: ONLY ONE PROJECTOR SHOULD BE IMPLEMENTED IN DIRECTORY. Found {len(state_county_projector_list)} for directory {state_county_dir}")

    logging.info(f"Get projector class for state: {state}, county: {county}")
    module_name = get_projector_module(state=state, county=county, projector_name=state_county_projector_list[0][0:-3])
    state_county_projector_module = importlib.import_module(module_name)
    projector_class = get_class_in_projector_module(module=state_county_projector_module, module_name=module_name)
    return projector_class


def modify_df_with_old_df(old_df: pd.DataFrame, new_df: pd.DataFrame) -> bool:
    old_keys, new_keys = old_df.keys(), new_df.keys()
    if len(old_keys) != len(new_keys):
        return True
    new_list = [True if old_key in new_keys else False for old_key in old_keys]
    N = len(new_list)
    if sum(new_list) == N:
        return False
    else:
        return True


def project_cases(state: str, county: str,
                  date_strings: List[str], most_recent_entry: Dict[str, float], projector_class: Callable) -> Tuple[List[Dict[str, int]], List[Dict[str, float]], List[Dict[str, any]],
                                                                                                                    List[Dict[str, float]], List[Dict[str, float]], str]:
    """
    For a particular state and county, and an associated projector class, process the raw data for any dates not already stored
    in a csv into numerical format.

    Arguments:
        state: State of projector
        county: County of projector
        date_strings: Date string
        most_recent_entry: Most recent entry that will be used for tracking potential large scale changes
        projector_class: Class that is used to perform projection

    Returns:
        ethnicity_cases_list: List of dictionaries containing number of cases per ethnicity
        ethnicity_cases_discrepancies_list: List of dictionaries containing disparity per ethnicity
        msg: Message that is non-empty if an issue occurs rith projecting cases at a particular date
    """
    ethnicity_cases_percentages_list, demographic_percentages_list = [], []
    ethnicity_cases_list, ethnicity_cases_rates_list, ethnicity_cases_discrepancies_list = [], [], []
    date_strings.sort()
    msg = ''
    for date_string in date_strings:
        try:
            projector_instance = projector_class(state=state, county=county, date_string=date_string)
            projector_instance.process_raw_data_to_cases()
            ethnicity_cases = projector_instance.ethnicity_cases
            ethnicity_cases_percentages = projector_instance.ethnicity_cases_percentages
            demographic_percentages = projector_instance.ethnicity_demographics
            ethnicity_cases_rates = projector_instance.ethnicity_cases_rates
            ethnicity_cases_discrepancies = projector_instance.ethnicity_cases_discrepancies

            demographic_percentages['date'] = date_string
            ethnicity_cases_percentages['date'] = date_string

            for ethnicity in demographic_percentages.keys():
                if ethnicity not in ethnicity_cases:
                    ethnicity_cases[ethnicity] = float("nan")
                if ethnicity not in ethnicity_cases_percentages:
                    ethnicity_cases_percentages[ethnicity] = float("nan")
                if ethnicity not in ethnicity_cases_rates:
                    ethnicity_cases_rates[ethnicity] = float("nan")
                if ethnicity not in ethnicity_cases_discrepancies:
                    ethnicity_cases_discrepancies[ethnicity] = float("nan")

            other_key = [key for key in ethnicity_cases_percentages.keys() if 'other' in key.lower()]
            if len(other_key) > 0:
                del ethnicity_cases_percentages[other_key[0]]
            valid_change_bool, msg = check_valid_change(
                state=state, county=county, date_string=date_string, dict1=ethnicity_cases, dict2=most_recent_entry, type_='case')
            if not valid_change_bool:
                msg = f"CASES: {msg}"
                break
            else:
                most_recent_entry = ethnicity_cases
            ethnicity_cases_list.append(ethnicity_cases)
            ethnicity_cases_rates_list.append(ethnicity_cases_rates)
            ethnicity_cases_discrepancies_list.append(ethnicity_cases_discrepancies)
            ethnicity_cases_percentages_list.append(ethnicity_cases_percentages)
            demographic_percentages_list.append(demographic_percentages)
        except Exception as e:
            if not projector_instance.cases_raw_bool:
                msg = f"WARNING CASES: ERROR in projection state: {state} county: {county}, {date_string}"
                pass
            else:
                msg = f'Issue with parsing cases at date: {date_string} with most recent entry {most_recent_entry}'
                break
    return ethnicity_cases_list, ethnicity_cases_rates_list, ethnicity_cases_discrepancies_list, ethnicity_cases_percentages_list, demographic_percentages_list, msg


def project_deaths(state: str, county: str,
                   date_strings: List[str], most_recent_entry: Dict[str, float], projector_class: Callable) -> Tuple[List[Dict[str, int]], List[Dict[str, float]], List[Dict[str, any]],
                                                                                                                     List[Dict[str, float]], List[Dict[str, float]], str]:
    """
    For a particular state and county, and an associated projector class, process the raw data for any dates not already stored
    in a csv into numerical format.

    Arguments:
        state: State of projector
        county: County of projector
        date_strings: Date string
        most_recent_entry: Most recent entry that will be used for tracking potential large scale changes
        projector_class: Class that is used to perform projection

    Returns:
        ethnicity_deaths_list: List of dictionaries containing number of deaths per ethnicity
        ethnicity_deaths_discrepancies_list: List of dictionaries containing disparity per ethnicity
        msg: Message that is non-empty if an issue occurs rith projecting cases at a particular date
    """
    ethnicity_deaths_percentages_list, demographic_percentages_list = [], []
    ethnicity_deaths_list, ethnicity_deaths_rates_list, ethnicity_deaths_discrepancies_list = [], [], []
    date_strings.sort()
    msg = ''
    for date_string in date_strings:
        try:
            projector_instance = projector_class(state=state, county=county, date_string=date_string)
            projector_instance.process_raw_data_to_deaths()
            ethnicity_deaths = projector_instance.ethnicity_deaths
            ethnicity_deaths_percentages = projector_instance.ethnicity_deaths_percentages
            demographic_percentages = projector_instance.ethnicity_demographics
            ethnicity_deaths_rates = projector_instance.ethnicity_deaths_rates
            ethnicity_deaths_discrepancies = projector_instance.ethnicity_deaths_discrepancies

            demographic_percentages['date'] = date_string
            ethnicity_deaths_percentages['date'] = date_string

            for ethnicity in demographic_percentages.keys():
                if ethnicity not in ethnicity_deaths:
                    ethnicity_deaths[ethnicity] = float("nan")
                if ethnicity not in ethnicity_deaths_percentages:
                    ethnicity_deaths_percentages[ethnicity] = float("nan")
                if ethnicity not in ethnicity_deaths_rates:
                    ethnicity_deaths_rates[ethnicity] = float("nan")
                if ethnicity not in ethnicity_deaths_discrepancies:
                    ethnicity_deaths_discrepancies[ethnicity] = float("nan")

            other_key = [key for key in ethnicity_deaths_percentages.keys() if 'other' in key.lower()]
            if len(other_key) > 0:
                del ethnicity_deaths_percentages[other_key[0]]
            valid_change_bool, msg = check_valid_change(
                state=state, county=county, date_string=date_string, dict1=ethnicity_deaths, dict2=most_recent_entry, type_='death')
            if not valid_change_bool:
                msg = f"DEATHS: {msg}"
                break
            else:
                most_recent_entry = ethnicity_deaths
            ethnicity_deaths_list.append(ethnicity_deaths)
            ethnicity_deaths_rates_list.append(ethnicity_deaths_rates)
            ethnicity_deaths_discrepancies_list.append(ethnicity_deaths_discrepancies)
            ethnicity_deaths_percentages_list.append(ethnicity_deaths_percentages)
            demographic_percentages_list.append(demographic_percentages)
        except Exception as e:
            if not projector_instance.deaths_raw_bool:
                msg = f"WARNING DEATHS: ERROR state: {state} county: {county}, {date_string}"
                pass
            else:
                msg = f'Issue with parsing death at date: {date_string} with most recent entry {most_recent_entry}'
                break
    return ethnicity_deaths_list, ethnicity_deaths_rates_list, ethnicity_deaths_discrepancies_list, ethnicity_deaths_percentages_list, demographic_percentages_list, msg


def parse_cases_responses_with_projectors(state: str, county: str, state_csv_dir: str, state_county_dir: str,
                                          cases_csv_filename: str) -> Tuple[List[Dict[str, int]], List[Dict[str, float]], List[Dict[str, any]],
                                                                            List[Dict[str, float]], List[Dict[str, float]], str]:
    """
    Wrapper function that imports projectors for a particular state and county. This projector will be used to parse
    raw data into number of cases per ethnicity for that state/county, and to quantify disparity.

    Arguments:
        state: State for particular projector
        county: County for a particular projector
        state_csv_dir: Directory in which csv files containing cases/death information are stored for a particular state
        case_csv_filename: File name for cases formatted as {state}_{county}_ethnicity_cases.csv

    Returns:
        ethnicity_cases_list: List of dictionaries containing number of cases per ethnicity
        ethnicity_cases_discrepancies_list: List of dictionaries containing disparity per ethnicity
        msg: Message that is non-empty if an issue occurs rith projecting cases at a particular date
    """
    logging.info("Get state/county projector")
    state_county_dir_list = os.listdir(state_county_dir)
    state_county_projector_list = filter_projector_module(projector_candidate_list=state_county_dir_list)
    if len(state_county_projector_list) != 1:
        raise ValueError(
            f"ERROR: ONLY ONE PROJECTOR SHOULD BE IMPLEMENTED IN DIRECTORY. Found {len(state_county_projector_list)} for directory {state_county_dir}")

    logging.info("Create ethnicity cases and deaths csvs if they don't already exist."
                 "Load if they do exist")
    state_county_cases_csv = os.path.join(state_csv_dir, cases_csv_filename)
    state_county_cases_df = None

    logging.info(f"Get projector class for state: {state}, county: {county}")
    module_name = get_projector_module(state=state, county=county, projector_name=state_county_projector_list[0][0:-3])
    state_county_projector_module = importlib.import_module(module_name)

    projector_class = get_class_in_projector_module(module=state_county_projector_module, module_name=module_name)

    logging.info("Load cases ethnicities csv if it does not exist. Create if it does not.")
    if not os.path.isfile(state_county_cases_csv):
        pass
    else:
        state_county_cases_df = pd.read_csv(state_county_cases_csv, header=0)

    logging.info(f"Load raw data directories for state: {state}, county: {county}")
    raw_data_dir = os.path.join(state_county_dir, "raw_data")

    logging.info(f"Get raw data dates if not already in cases data frames.")
    raw_data_dates = os.listdir(raw_data_dir)
    raw_data_cases_dates, raw_data_cases_old_dates = filter_dates_from_df(
        date_list=raw_data_dates, df=state_county_cases_df)
    raw_data_cases_dates.sort()
    raw_data_cases_old_dates.sort()

    logging.info(f"Get case per ethnicity and case discrepancies for each ethnicity. Create if it does not.")
    if state_county_cases_df is not None and len(state_county_cases_df) > 0:
        most_recent_entry = state_county_cases_df[state_county_cases_df.date.eq(
            raw_data_cases_old_dates[-1])].to_dict('record')[0]
        most_recent_entry_copy = copy.deepcopy(most_recent_entry)
        for key in most_recent_entry.keys():
            if 'discrepancy' in key.lower() or 'unnamed' in key.lower() or 'rates' in key.lower() or \
                    'covidperc' in key.lower() or 'demperc' in key.lower() or 'other' in key.lower():
                del most_recent_entry_copy[key]
        most_recent_entry = most_recent_entry_copy
    else:
        most_recent_entry = None
    try:
        ethnicity_cases_list, ethnicity_cases_rates_list, ethnicity_cases_discrepancies_list, ethnicity_cases_percentages_list, demographic_percentages_list, msg = project_cases(
            state=state, county=county, date_strings=raw_data_cases_dates, most_recent_entry=most_recent_entry, projector_class=projector_class)
    except Exception as e:
        raise ValueError(f"{e}")
    return ethnicity_cases_list, ethnicity_cases_rates_list, ethnicity_cases_discrepancies_list, ethnicity_cases_percentages_list, demographic_percentages_list, msg


def parse_deaths_responses_with_projectors(state: str, county: str, state_csv_dir: str, state_county_dir: str,
                                           deaths_csv_filename: str) -> Tuple[List[Dict[str, int]], List[Dict[str, float]], List[Dict[str, any]],
                                                                              List[Dict[str, float]], List[Dict[str, float]], str]:
    """
    Wrapper function that imports projectors for a particular state and county. This projector will be used to parse
    raw data into number of deaths per ethnicity for that state/county, and to quantify disparity.

    Arguments:
        state: State for particular projector
        county: County for a particular projector
        state_csv_dir: Directory in which csv files containing cases/death information are stored for a particular state
        death_csv_filename: File name for cases formatted as {state}_{county}_ethnicity_deaths.csv

    Returns:
        ethnicity_deaths_list: List of dictionaries containing number of deaths per ethnicity
        ethnicity_deaths_discrepancies_list: List of dictionaries containing disparity per ethnicity
        msg: Message that is non-empty if an issue occurs rith projecting deatjs at a particular date
    """
    logging.info("Get state/county projector")
    projector_class = get_projector_class(state=state, county=county, state_county_dir=state_county_dir)

    logging.info("Create ethnicity cases and deaths csvs if they don't already exist."
                 "Load if they do exist")
    state_county_deaths_csv = os.path.join(state_csv_dir, deaths_csv_filename)
    state_county_deaths_df = None

    logging.info("Load deaths ethnicities csv if it does not exist. Create if it does not.")
    if not os.path.isfile(state_county_deaths_csv):
        pass
    else:
        state_county_deaths_df = pd.read_csv(state_county_deaths_csv, header=0)

    logging.info(f"Load raw data directories for state: {state}, county: {county}")
    raw_data_dir = os.path.join(state_county_dir, "raw_data")

    logging.info(f"Get raw data dates if not already in the deathsdata frames.")
    raw_data_dates = os.listdir(raw_data_dir)


    raw_data_deaths_dates, raw_data_deaths_old_dates = filter_dates_from_df(
        date_list=raw_data_dates, df=state_county_deaths_df)

    raw_data_deaths_old_dates.sort()
    raw_data_deaths_dates.sort()
    raw_data_deaths_dates = [date_ for date_ in raw_data_deaths_dates if date_ > raw_data_deaths_old_dates[-1]]

    logging.info(f"Get case per ethnicity and case discrepancies for each ethnicity")
    if state_county_deaths_df is not None and len(state_county_deaths_df) > 0:
        most_recent_entry = state_county_deaths_df[state_county_deaths_df.date.eq(
            raw_data_deaths_old_dates[-1])].to_dict('records')[0]
        most_recent_entry_copy = copy.deepcopy(most_recent_entry)
        for key in most_recent_entry.keys():
            if 'discrepancy' in key.lower() or 'unnamed' in key.lower() or 'rates' in key.lower()\
                    or 'covidperc' in key.lower() or 'demperc' in key.lower() or 'other' in key.lower():
                del most_recent_entry_copy[key]
        most_recent_entry = most_recent_entry_copy
    else:
        most_recent_entry = None
    ethnicity_dates_list, ethnicity_deaths_rates_list, ethnicity_deaths_discrepancies_list, ethnicity_deaths_percentages_list, demographic_percentages_list, msg = project_deaths(
        state=state, county=county, date_strings=raw_data_deaths_dates, most_recent_entry=most_recent_entry, projector_class=projector_class)

    return ethnicity_dates_list, ethnicity_deaths_rates_list, ethnicity_deaths_discrepancies_list, ethnicity_deaths_percentages_list, demographic_percentages_list, msg


def get_yaml_responses(config_dir: str, config_file_list: List[str]) -> Tuple[List[str], List[str], str]:
    """
    Get responses from website based on a config file containing a REQUEST field

    Arguments:
        config_dir: Directory containing config files to be searched
        config_file_list: List of config files contained in the config directory

    Returns:
        response_list: List of valid response texts obtained based on YAML
        response_names: Name of what the response corresponds to. Of format
                        Name and DataType
    """
    response_list, response_names = [], []
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
                msg = f"ERROR: Response for {data_type_name} failed with status {status_code}"
                raise ValueError(f"{msg}")
            response.close()

    return response_list, response_names, request_type


def get_metadata_config(config_dir: str, config_file_list: List[str]) -> Dict:
    for config_file in config_file_list:
        config_file_obj = open(path.join(config_dir, config_file))
        response_config = yaml.safe_load(config_file_obj)
        if 'ROOT' in response_config.keys():
            return response_config


def get_metadata_response(config_dir: str, config_file_list: List[str]) -> Dict[str, Dict[str, float]]:

    metadata_dict = {}
    response_config = get_metadata_config(config_dir=config_dir, config_file_list=config_file_list)

    # Get root and region url components
    url_base = response_config['ROOT']
    url_region = response_config['REGION'][0]
    for region in response_config['REGION'][1:]:
        url_region = url_region + f'&{region}'

    # Construct metadata dictionary using requests to metadata
    # url in ACS survey
    for metadata_name in response_config['METADATA']:
        metadata_dict[metadata_name] = {}
        metadata_fields = response_config['METADATA'][metadata_name]
        for field in metadata_fields:
            url_use = f'{url_base}get={metadata_fields[field]}&{url_region}&key={response_config["API_KEY"]}'
            headers = response_config['HEADERS']
            acs5_response = requests.get(url=url_use, headers=headers)

            headers, vals = eval(acs5_response.text)
            for idx, header_val_tuple in enumerate(zip(headers, vals)):
                if header_val_tuple[0] == metadata_fields[field]:
                    metadata_dict[metadata_name][field] = float(header_val_tuple[1])
    return metadata_dict


def get_raw_metadata_from_config_files(config_dir: str) -> pd.DataFrame:
    config_files = os.listdir(config_dir)
    config_files = [config_file for config_file in config_files if config_file.endswith('.yaml')]

    metadata_dict = get_metadata_response(config_dir=config_dir, config_file_list=config_files)
    return pd.DataFrame(metadata_dict)


def process_raw_metadata(raw_metadata_df: pd.DataFrame, config_dir: str, state: str,
                         county: str, state_county_dir: str) -> pd.DataFrame:
    def _normalize_df_per_1000(df, key):
        # Get relevant projector class and use this to get total population
        # in region per defined ethnicity
        projector_class = get_projector_class(state=state, county=county, state_county_dir=state_county_dir)
        valid_date = os.listdir(path.join(state_county_dir, 'raw_data'))[0]
        projector_class = projector_class(state=state, county=county, date_string=valid_date)
        ethnicity_demograpics_perc_dict = projector_class.acs_ethnicity_demographics
        # Normalize metadata values per 1000 for each ethnicity
        for ethnicity_key in ethnicity_demograpics_perc_dict.keys():
            if ethnicity_key in df.index.tolist():
                metadata_val = df.loc[ethnicity_key, key]
                df.at[ethnicity_key, key] = metadata_val * 1000 / (ethnicity_demograpics_perc_dict[
                    ethnicity_key] * projector_class.total_population)
        return df

    config_files = os.listdir(config_dir)
    config_file_list = [config_file for config_file in config_files if config_file.endswith('.yaml')]
    response_config = get_metadata_config(config_dir=config_dir, config_file_list=config_file_list)

    process_func_dict = response_config['PROCESS_FUNC']
    processed_metadata_df = copy.deepcopy(raw_metadata_df)
    for key in process_func_dict.keys():
        process_func = process_func_dict[key]
        total = raw_metadata_df.loc['Total', key]
        error_bool = [False] * (len(processed_metadata_df) - 1)
        if process_func == 'ratio_wrt_total':
            processed_metadata_df[key] = processed_metadata_df[key] / total
        elif process_func == 'perc_of_total_per_1000':
            # Get percent of total count of particular metadata
            processed_metadata_df[key] = processed_metadata_df[key] * total * 0.01
            processed_metadata_df = _normalize_df_per_1000(df=processed_metadata_df, key=key)
            error_bool = processed_metadata_df[key] > 1000
        elif process_func == 'total_per_1000':
            processed_metadata_df = _normalize_df_per_1000(df=processed_metadata_df, key=key)
            error_bool = processed_metadata_df[key] > 1000
        elif process_func == 'identity':
            pass
        else:
            raise ValueError(f'Processing of metadata function {process_func} not implemented')

        if sum(error_bool) > 1:
            raise ValueError(
                f'Normalized values per 1000 error for {key}. Following values are greater than 1000 {processed_metadata_df[key][error_bool]}')

    processed_metadata_df = processed_metadata_df.drop(['Total'])
    return processed_metadata_df


def aggregate_processed_raw_metadata(processed_metadata_df: pd.DataFrame, state: str,
                                     county: str, state_county_dir: str) -> pd.DataFrame:
    # Get relevant projector class
    projector_class = get_projector_class(state=state, county=county, state_county_dir=state_county_dir)
    valid_date = os.listdir(path.join(state_county_dir, 'raw_data'))[0]
    projector_class = projector_class(state=state, county=county, date_string=valid_date)

    # Construct dictionary that aggregates processed metadata based on ethnic demographics
    # using ACS demographics
    aggregated_processed_metadata_dict = {}
    for metadata_name in processed_metadata_df.keys():
        aggregated_processed_metadata_dict[metadata_name] = {}
        for ethnicity_key in projector_class.ethnicity_demographics.keys():
            if ethnicity_key in projector_class.map_acs_to_region_ethnicities.keys():
                acs_ethnicity_list = projector_class.map_acs_to_region_ethnicities[ethnicity_key]
                acs_ethnicity_demographics = projector_class.acs_ethnicity_demographics
                weighted_sum = sum([acs_ethnicity_demographics[acs_ethnicity]
                                    for acs_ethnicity in acs_ethnicity_list if processed_metadata_df.loc[acs_ethnicity, metadata_name] >= 0])
                weighted_metadata_val = sum([processed_metadata_df.loc[acs_ethnicity, metadata_name] *
                                             acs_ethnicity_demographics[acs_ethnicity] /
                                             weighted_sum for acs_ethnicity in acs_ethnicity_list if processed_metadata_df.loc[acs_ethnicity, metadata_name] >= 0])
                aggregated_processed_metadata_dict[metadata_name][ethnicity_key] = weighted_metadata_val

    aggregated_processed_metadata_df = pd.DataFrame(aggregated_processed_metadata_dict)
    return aggregated_processed_metadata_df


def save_data(state_name: str, county_name: str, data_df: pd.DataFrame, data_dir: str, data_suffix: str) -> None:
    state_dirs = os.listdir(path.join('states', state_name))
    state_dirs = [dir_ for dir_ in state_dirs if os.path.isdir(path.join('states', state_name, dir_))]
    if data_dir not in state_dirs:
        os.mkdir(path.join('states', state_name, data_dir))

    state_metadata_file = f'{state_name}'
    if county_name is not None:
        state_metadata_file = f'{state_metadata_file}_{county_name}_{data_suffix}'
    else:
        state_metadata_file = f'{state_metadata_file}_{data_suffix}'
    data_df.to_csv(path.join('states', state_name, data_dir, f'{state_metadata_file}.csv'))


def run_ethnicity_to_case_csv(state_csv_dir: str, state_county_dir: str, state: str,
                              county: Union[str, None], cases_csv_filename: str) -> str:
    """
    Convert raw ethnicity data for a particular state and county to number of cases.
    Process these to obtain disparity.

    Arguments:
        state_csv_dir: Directory in which csv file will be failed
        state_county_dir: Directory of state/directory that contains relevant raw data to ethnicity projector
        state: State for which projection will be done
        county: County for which projection will be done
        cases_csv_filename: Name of csv of format {state}_{county}_ethnicity_cases.csv

    Returns:
        msg: Non empty message if there is an error in projecting a particular date
    """
    logging.info(f"Get state ethnicity cases counts and discrepancies")
    change_df_key_bool = False
    state_ethnicity_cases_list, state_ethnicity_cases_rates_list, state_ethnicity_cases_discrepancies_list, state_ethnicity_cases_percentages_list, state_demographic_percentages_list, msg = parse_cases_responses_with_projectors(
        state=state, county=county, state_csv_dir=state_csv_dir, state_county_dir=state_county_dir, cases_csv_filename=cases_csv_filename)
    try:
        state_ethnicity_cases_df, state_ethnicity_cases_rates_df, state_ethnicity_cases_discrepancies_df = pd.DataFrame(
            state_ethnicity_cases_list), pd.DataFrame(
            state_ethnicity_cases_rates_list), pd.DataFrame(state_ethnicity_cases_discrepancies_list)
        state_ethnicity_cases_perc_df, state_demographic_perc_df = pd.DataFrame(
            state_ethnicity_cases_percentages_list), pd.DataFrame(state_demographic_percentages_list)


        state_ethnicity_full_cases_df = state_ethnicity_cases_df.merge(
            state_ethnicity_cases_discrepancies_df, left_on='date', right_on='date', suffixes=('', '_discrepancy'))
        state_ethnicity_full_cases_df = state_ethnicity_full_cases_df.merge(
            state_ethnicity_cases_rates_df, left_on='date', right_on='date', suffixes=('', '_rates'))
        state_ethnicity_full_cases_df = state_ethnicity_full_cases_df.merge(
            state_ethnicity_cases_perc_df, left_on='date', right_on='date', suffixes=('', '_covidperc'))
        state_ethnicity_full_cases_df = state_ethnicity_full_cases_df.merge(
            state_demographic_perc_df, left_on='date', right_on='date', suffixes=('', '_demperc'))
        try:
            old_state_county_df = pd.read_csv(f"{state_csv_dir}/{cases_csv_filename}")

            ordered_cols = []
            for col in old_state_county_df.keys():
                if col in state_ethnicity_full_cases_df.keys():
                    ordered_cols.append(col)
            for col in state_ethnicity_full_cases_df.keys():
                if col not in state_ethnicity_full_cases_df.keys():
                    ordered_cols.append(col)
            state_ethnicity_full_cases_df = state_ethnicity_full_cases_df[ordered_cols]

            change_df_key_bool = modify_df_with_old_df(old_df=old_state_county_df, new_df=state_ethnicity_full_cases_df)
            if len(old_state_county_df) > 0 and not change_df_key_bool:
                state_ethnicity_full_cases_df.to_csv(f"{state_csv_dir}/{cases_csv_filename}", mode='a', index=False,
                                                     header=False)
            else:
                if change_df_key_bool:
                    state_ethnicity_full_cases_df = pd.concat(
                        [old_state_county_df, state_ethnicity_full_cases_df], axis=0, ignore_index=True)
                state_ethnicity_full_cases_df.to_csv(f"{state_csv_dir}/{cases_csv_filename}", mode='w', index=False)
        except BaseException:
            if change_df_key_bool:
                state_ethnicity_full_cases_df = pd.concat([old_state_county_df, state_ethnicity_full_cases_df], axis=0,
                                                          ignore_index=True)
            state_ethnicity_full_cases_df.to_csv(f"{state_csv_dir}/{cases_csv_filename}", mode='w', index=False)
    except BaseException:
        pass
    return msg


def run_ethnicity_to_death_csv(state_csv_dir: str, state_county_dir: str, state: str,
                               county: Union[str, None], deaths_csv_filename: str) -> str:
    """
    Convert raw ethnicity data for a particular state and county to number of deaths.
    Process these to obtain disparity.

    Arguments:
        state_csv_dir: Directory in which csv file will be failed
        state_county_dir: Directory of state/directory that contains relevant raw data to ethnicity projector
        state: State for which projection will be done
        county: County for which projection will be done
        deaths_csv_filename: Name of csv of format {state}_{county}_ethnicity_cases.csv

    Returns:
        msg: Non empty message if there is an error in projecting a particular date
    """
    change_df_key_bool = False
    logging.info(f"Get state ethnicity deaths counts and discrepancies")
    state_ethnicity_deaths_list, state_ethnicity_deaths_rates_list, state_ethnicity_deaths_discrepancies_list, state_ethnicity_deaths_percentages_list, state_demographic_percentages_list, msg = parse_deaths_responses_with_projectors(
        state=state, county=county, state_csv_dir=state_csv_dir, state_county_dir=state_county_dir, deaths_csv_filename=deaths_csv_filename)
    try:
        state_ethnicity_deaths_df, state_ethnicity_deaths_rates_df, state_ethnicity_deaths_discrepancies_df = pd.DataFrame(
            state_ethnicity_deaths_list), pd.DataFrame(state_ethnicity_deaths_rates_list), pd.DataFrame(state_ethnicity_deaths_discrepancies_list)
        state_ethnicity_deaths_perc_df, state_demographic_perc_df = pd.DataFrame(
            state_ethnicity_deaths_percentages_list), pd.DataFrame(state_demographic_percentages_list)

        state_ethnicity_full_deaths_df = state_ethnicity_deaths_df.merge(
            state_ethnicity_deaths_discrepancies_df, left_on='date', right_on='date', suffixes=('', '_discrepancy'))
        state_ethnicity_full_deaths_df = state_ethnicity_full_deaths_df.merge(
            state_ethnicity_deaths_rates_df, left_on='date', right_on='date', suffixes=('', '_rates'))
        state_ethnicity_full_deaths_df = state_ethnicity_full_deaths_df.merge(
            state_ethnicity_deaths_perc_df, left_on='date', right_on='date', suffixes=('', '_covidperc'))
        state_ethnicity_full_deaths_df = state_ethnicity_full_deaths_df.merge(
            state_demographic_perc_df, left_on='date', right_on='date', suffixes=('', '_demperc'))
        try:
            old_state_county_df = pd.read_csv(f"{state_csv_dir}/{deaths_csv_filename}")

            ordered_cols = []
            for col in old_state_county_df.keys():
                if col in state_ethnicity_full_deaths_df.keys():
                    ordered_cols.append(col)
            for col in state_ethnicity_full_deaths_df.keys():
                if col not in state_ethnicity_full_deaths_df.keys():
                    ordered_cols.append(col)
            state_ethnicity_full_deaths_df = state_ethnicity_full_deaths_df[ordered_cols]

            change_df_key_bool = modify_df_with_old_df(
                old_df=old_state_county_df, new_df=state_ethnicity_full_deaths_df)
            if len(old_state_county_df) > 0 and not change_df_key_bool:
                state_ethnicity_full_deaths_df.to_csv(f"{state_csv_dir}/{deaths_csv_filename}", mode='a',
                                                      index=False, header=False)
            else:
                if change_df_key_bool:
                    state_ethnicity_full_deaths_df = pd.concat([old_state_county_df, state_ethnicity_full_deaths_df],
                                                               axis=0, ignore_index=True)
                state_ethnicity_full_deaths_df.to_csv(f"{state_csv_dir}/{deaths_csv_filename}", mode='w',
                                                      index=False)
        except BaseException:
            if change_df_key_bool:
                state_ethnicity_full_deaths_df = pd.concat([old_state_county_df, state_ethnicity_full_deaths_df], axis=0,
                                                           ignore_index=True)
            state_ethnicity_full_deaths_df.to_csv(f"{state_csv_dir}/{deaths_csv_filename}", mode='w',
                                                  index=False)
    except BaseException:
        pass

    return msg


def save_raw_data(save_dir: str, response_list: List[str], data_type_names: List[str],
                  request_type: str) -> None:
    """
    Save raw data based on response list and data  type names.

    Arguments:
        save_dir: Directory that raw data will be saved to
        response_list: List of responses to be saved
        data_type_names: Names of data types and
        request_type: Type of request made

    Returns:
        None
    """
    dt = datetime.datetime.now() - datetime.timedelta(days=1)
    today = datetime.date(dt.year, dt.month, dt.day)
    today_str = today.isoformat()
    save_dir = f"{save_dir}/{today_str}"
    if not path.isdir(save_dir):
        os.makedirs(save_dir)
    save_dir_files = os.listdir(save_dir)
    if len(save_dir_files) == 0:
        for response, data_type_name in zip(response_list, data_type_names):
            if request_type == 'GET':
                save_path = f"{save_dir}/{data_type_name}.html"
            else:
                save_path = f"{save_dir}/{data_type_name}"
            text_file = open(save_path, "w")
            text_file.write(response)
            text_file.close()
