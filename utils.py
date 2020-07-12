# --------------------------
# Standard Python Imports
# --------------------------
import datetime
import json
import logging
import os
from os import path
from typing import List, Tuple

# --------------------------
# Third Party Imports
# --------------------------
import requests
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------


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
            os.remove(f"{save_dir}/failed_queries")

    failed_save_path = f"{save_dir}/failed_queries"
    if len(failed_data_type_names):
        with open(failed_save_path, 'w') as f:
            for failed_data_type_name in failed_data_type_names:
                f.write(f"{failed_data_type_name}\n")
