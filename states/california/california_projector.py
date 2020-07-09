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
    def __init__(self, state: str, county: str, raw_data_file: str, date_string: str, config_file_string: str, json: bool = None, lxml: bool = None):
        super().__init__(state=state, county=county, raw_data_file=raw_data_file, date_string=date_string, config_file_string=config_file_string, lxml=lxml, json=json)
        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {'LATINO_CASES': 'latino', 'WHITE_CASES': 'white', 'ASIAN_CASES': 'asian',
                                              'BLACK_CASES': 'black', 'MULTI_RACE_CASES': 'multirace',
                                              'AMERICAN_INDIAN_OR_ALASKA_NATIVE_CASES': 'american_indian_alaska_native', 'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES': 'native_hawaiian_pacific_islander',
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
