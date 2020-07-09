# --------------------------
# Standard Python Imports
# --------------------------
from datetime import datetime
import logging
from lxml import etree
import os

# --------------------------
# Third Party Imports
# --------------------------
from typing import Dict, List
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------
from states.data_projectors import EthnicDataProjector
from states import utils


class CaliforniaEthnicDataProjector(EthnicDataProjector):
    def __init__(self, date_string: str):
        super().__init__()
        self.state = 'california'
        self.raw_data_dir = os.path.join("states", self.state, "raw_data")

        logging.info("Load california html parsing config")
        configs_dir = os.path.join("states", self.state, "configs")
        html_parser_config_file = open(f"{configs_dir}/california_all_html_parse.yaml")
        html_parser_config = yaml.safe_load(html_parser_config_file)

        logging.info("Get and sort californial html parsing dates")
        html_parser_date_strings = html_parser_config["DATES"].keys()
        html_parser_dates = [datetime.strptime(date_string, '%Y-%m-%d') for date_string in html_parser_date_strings]
        html_parser_dates.sort()

        logging.info("Obtain valid map of ethnicities to xpath containing cases or deaths")
        self.valid_date_string = utils.get_valid_date_string(date_list=html_parser_dates, date_string=date_string)
        self.ethnicity_xpath_map = html_parser_config['DATES'][self.valid_date_string]

        logging.info("Load raw html data and convert it to lxml")
        raw_data_file = f"{self.raw_data_dir}/{date_string}/california_all.html"
        raw_data_file_object = open(raw_data_file, 'r')
        raw_data_file_html = raw_data_file_object.read()
        self.raw_data_lxml = etree.HTML(raw_data_file_html)

        logging.info("Define yaml keys to dictionary map")
        self.cases_yaml_keys_dict_keys_map = {'LATINO_CASES': 'latino', 'WHITE_CASES': 'white', 'ASIAN_CASES': 'asian',
                                              'BLACK_CASES': 'black', 'MULTI_RACE_CASES': 'multirace',
                                              'AMERICAN_INDIAN_OR_ALASKA_NATIVE_CASES': 'american_indian_alaska_native',  'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES': 'native_hawaiian_pacific_islander',
                                              'OTHER_CASES': 'other'}
        self.deaths_yaml_keys_dict_keys_map = {'LATINO_DEATHS': 'latino', 'WHITE_DEATHS': 'white', 'ASIAN_DEATHS': 'asian',
                                              'BLACK_DEATHS': 'black', 'MULTI_RACE_DEATHS': 'multirace',
                                              'AMERICAN_INDIAN_OR_ALASKA_NATIVE_DEATHS': 'american_indian_alaska_native',  'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_DEATHS': 'native_hawaiian_pacific_islander',
                                              'OTHER_DEATHS': 'other'}

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ["latino", "white", "asian", "black", "multirace", "american_indian_alaska_native", "native_hawaiian_pacific_islander", "other"]

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in california
        """
        return {'latino': 0.393, 'white': 0.366, 'asian': 0.145, 'black': 0.055, 'multirace': 0.0308, 'american_indian_alaska_native': 0.0035, 'native_hawaiian_pacific_islander': 0.0036, 'other': 0.0025}
