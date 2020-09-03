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


class SanFranciscoEthnicDataProjector(AlamedaEthnicDataProjector, EthnicDataProjector):
    def __init__(self, state: str, county: str, date_string: str):
        self.state, self.county = state, county
        super().__init__(state=state, county=county, date_string=date_string)
        self.cases_raw_bool, self.deaths_raw_bool = False, False
        logging.info("Initialize imperial county raw and config file strings")
        raw_data_dir = os.path.join("states", state, 'counties', county, "raw_data")
        raw_data_cases_file = f"{raw_data_dir}/{date_string}/sanfrancisco_cases"
        raw_data_deaths_file = f"{raw_data_dir}/{date_string}/sanfrancisco_deaths"

        configs_dir = os.path.join("states", state, 'counties', county, "configs")
        cases_config_file_string = f"{configs_dir}/sanfrancisco_cases_json_parser.yaml"
        deaths_config_file_string = f"{configs_dir}/sanfrancisco_deaths_json_parser.yaml"

        logging.info("Load cases and deaths parsing config")
        json_parser_cases_config = self.load_yaml(cases_config_file_string)
        json_parser_deaths_config = self.load_yaml(deaths_config_file_string)

        logging.info("Get and sort json parsing dates")
        json_parser_cases_dates = self.get_sorted_dates_from_strings(
            date_string_list=list(json_parser_cases_config["DATES"].keys()))
        json_parser_deaths_dates = self.get_sorted_dates_from_strings(
            date_string_list=list(json_parser_deaths_config["DATES"].keys()))

        logging.info("Obtain valid map of ethnicities to json containing cases or deaths")
        self.date_string = date_string
        self.cases_valid_date_string = utils.get_valid_date_string(
            date_list=json_parser_cases_dates, date_string=date_string)
        self.deaths_valid_date_string = utils.get_valid_date_string(
            date_list=json_parser_deaths_dates, date_string=date_string)
        self.cases_ethnicity_json_keys_map = json_parser_cases_config['DATES'][self.cases_valid_date_string]
        self.deaths_ethnicity_json_keys_map = json_parser_deaths_config['DATES'][self.deaths_valid_date_string]
        self.ethnicity_json_keys_map = {**self.cases_ethnicity_json_keys_map, **self.deaths_ethnicity_json_keys_map}

        self.cases_yaml_keys_dict_keys_map, self.deaths_yaml_keys_dict_keys_map = None, None
        try:
            logging.info("Load raw cases json data")
            cases_file_obj = open(raw_data_cases_file, 'r')
            self.raw_data_cases_json = json.load(cases_file_obj)
            logging.info("Define yaml keys to dictionary maps for cases")
            self.cases_yaml_keys_dict_keys_map = {
                'NATIVE_AMERICAN_CASES': 'Native American',
                'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES': 'Native Hawaiian/Pacific Islander',
                'MULTI_RACE_CASES': 'Multi-Race',
                'BLACK_CASES': 'Black',
                'ASIAN_CASES': 'Asian',
                'WHITE_CASES': 'White',
                'HISPANIC_CASES': 'Hispanic'
            }
            self.cases_raw_bool = True
        except BaseException:
            pass

        try:
            logging.info("Load raw deaths json data")
            deaths_file_obj = open(raw_data_deaths_file, 'r')
            self.raw_data_deaths_json = json.load(deaths_file_obj)
            logging.info("Define yaml keys to dictionary maps for deaths")
            self.deaths_yaml_keys_dict_keys_map = {
                'WHITE_DEATHS': 'White',
                'HISPANIC_DEATHS': 'Hispanic',
                'ASIAN_DEATHS': 'Asian',
                'BLACK_DEATHS': 'Black',
                'MULTI_RACE_DEATHS': 'Multi-Race'
            }
            self.deaths_raw_bool = True
        except BaseException:
            pass

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['Native American', 'Native Hawaiian/Pacific Islander',
                'Multi-Race', 'Black', 'Asian', 'White', 'Hispanic']

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in San Francisco County

        Obtained from here: https://www.census.gov/quickfacts/sanfranciscoocountycalifornia

        """
        return {'Native American': 0.013, 'Native Hawaiian/Pacific Islander': 0.002,
                'Multi-Race': 0.028, 'Black': 0.134, 'Asian': 0.059, 'White': 0.763, 'Hispanic': 0.185}
