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


class SonomaEthnicDataProjector(EthnicDataProjector):
    def __init__(self, state: str, county: str, raw_data_file_list: str, date_string: str, config_file_string_list: str, json: bool = None, lxml: bool = None):
        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {'HISPANIC_CASES': 'hispanic', 'WHITE_CASES': 'white', 'ASIAN_PACIFIC_ISLANDER_CASES': 'asian_pacific_islander', 'NON_HISPANIC_CASES': 'non_hispanic'}
        self.deaths_yaml_keys_dict_keys_map = None

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['hispanic', "white", "asian_pacific_islander", "black", "non_hispanic"]

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in Sonoma County. Obtained from
        census.gov/quickfacts/fact/table/sonomacountycalifornia,CA/PST045219

        """
        return {'hispanic': 0.273, 'white': 0.629, 'asian_pacific_islander': 0.05, 'non_hispanic': 0.048}
