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


def get_yaml_responses(config_dir: str, config_file_list: List[str]) -> Tuple[List[str], List[str], List[str]]:
    response_list, response_names, failed_response_names = [], [], []
    for config_file in config_file_list:
        config_file_obj = open(path.join(config_dir, config_file))
        response_config = yaml.safe_load(config_file_obj)

        status_code = -1
        data_type_name = response_config['NAME'].lower() + '_' + response_config['DATA_TYPE'].lower()
        url = response_config['REQUEST']['URL']
        request_type = response_config['REQUEST']['TYPE']
        if request_type == 'GET':
            response = requests.get(url)
            status_code = response.status_code
        elif request_type == 'POST':
            headers = response_config['REQUEST']['HEADERS']
            payload = response_config['REQUEST']['PAYLOAD']
            response = requests.post(url=url, headers=headers,  json=json.loads(json.dumps(payload)))
            status_code = response.status_code
        else:
            raise ValueError(f"Request only implemented for GET or POST types. Got {request_type}")

        if status_code == 200:
            response_list.append(response)
            response_names.append(data_type_name)
        else:
            logging.info(f"Response for {data_type_name} failed with status {status_code}")
            failed_response_names.append(data_type_name)
    return response_list, response_names, failed_response_names
