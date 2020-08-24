# --------------------------
# Standard Python Imports
# --------------------------
import json
import logging
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
from states.california.counties.alameda.alameda_projector import AlamedaEthnicDataProjector
from states import utils


class KernEthnicDataProjector(AlamedaEthnicDataProjector):
    def __init__(self, state: str, county: str, date_string: str):
        self.state, self.county = state, county
        logging.info("Initialize kern county raw and config file strings")
        raw_data_dir = os.path.join("states", state, 'counties', county, "raw_data")
        raw_data_cases_file = f"{raw_data_dir}/{date_string}/kern_cases"

        configs_dir = os.path.join("states", state, 'counties', county, "configs")
        cases_config_file_string = f"{configs_dir}/kern_cases_json_parser.yaml"

        logging.info("Load cases and deaths parsing config")
        json_parser_cases_config = self.load_yaml(cases_config_file_string)

        logging.info("Get and sort json parsing dates")
        json_parser_cases_dates = self.get_sorted_dates_from_strings(
            date_string_list=list(json_parser_cases_config["DATES"].keys()))

        logging.info("Obtain valid map of ethnicities to json containing cases or deaths")
        self.date_string = date_string
        self.cases_valid_date_string = utils.get_valid_date_string(
            date_list=json_parser_cases_dates, date_string=date_string)
        self.cases_ethnicity_json_keys_map, self.deaths_yaml_keys_dict_keys_map = json_parser_cases_config[
            'DATES'][self.cases_valid_date_string], None
        self.ethnicity_json_keys_map = self.cases_ethnicity_json_keys_map

        logging.info("Load raw json data")
        try:
            cases_file_obj = open(raw_data_cases_file, 'r')
            self.raw_data_cases_json = json.load(cases_file_obj)
            self.cases_raw_bool = True
        except BaseException:
            pass

        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {
            'BLACK_CASES': 'Black',
            'HISPANIC_CASES': 'Hispanic',
            'ASIAN_CASES': 'Asian',
            'WHITE_CASES': 'White',
            'OTHER_CASES': 'Other',
        }

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['Hispanic', 'Black', 'Asian', 'White', 'Other', 'Unknown']

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in Kern County.

        Obtained from here: https://www.census.gov/quickfacts/kerncountycalifornia

        """
        return {'Hispanic': 0.546, 'Black': 0.063, 'White': 0.328, 'Asian': 0.054, 'Other': 0.061}
