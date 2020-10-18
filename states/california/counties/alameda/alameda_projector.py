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
from states import utils_state_lib


class AlamedaEthnicDataProjector(EthnicDataProjector):
    def __init__(self, state: str, county: str, date_string: str):
        try:
            super().__init__(state=state, county=county)
            self.cases_raw_bool, self.deaths_raw_bool = False, False
            logging.info("Initialize Alameda raw and config file strings")
            raw_data_dir = os.path.join("states", state, 'counties', county, "raw_data")
            raw_data_cases_file, raw_data_cases_file_html = f"{raw_data_dir}/{date_string}/alameda_cases", f"{raw_data_dir}/{date_string}/alameda_cases.html"
            raw_data_deaths_file, raw_data_deaths_file_html = f"{raw_data_dir}/{date_string}/alameda_deaths", f"{raw_data_dir}/{date_string}/alameda_deaths.html"

            configs_dir = os.path.join("states", state, 'counties', county, "configs")
            cases_config_file_string = f"{configs_dir}/alameda_cases_json_parser.yaml"
            deaths_config_file_string = f"{configs_dir}/alameda_deaths_json_parser.yaml"

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
            self.cases_valid_date_string = utils_state_lib.get_valid_date_string(
                date_list=json_parser_cases_dates, date_string=date_string)
            self.deaths_valid_date_string = utils_state_lib.get_valid_date_string(
                date_list=json_parser_deaths_dates, date_string=date_string)
            self.cases_ethnicity_json_keys_map = json_parser_cases_config['DATES'][self.cases_valid_date_string]
            self.deaths_ethnicity_json_keys_map = json_parser_deaths_config['DATES'][self.deaths_valid_date_string]
            self.ethnicity_json_keys_map = {**self.cases_ethnicity_json_keys_map, **self.deaths_ethnicity_json_keys_map}

            logging.info("Load raw json data")
            try:
                cases_file_obj = open(raw_data_cases_file, 'r')
                self.cases_raw_bool = True
            except BaseException:
                try:
                    cases_file_obj = open(raw_data_cases_file_html, 'r')
                    self.cases_raw_bool = True
                except BaseException:
                    pass

            try:
                deaths_file_obj = open(raw_data_deaths_file, 'r')
                self.deaths_raw_bool = True
            except BaseException:
                try:
                    deaths_file_obj = open(raw_data_deaths_file_html, 'r')
                    self.deaths_raw_bool = True
                except BaseException:
                    pass

            self.raw_data_cases_json = json.load(cases_file_obj)
            self.raw_data_deaths_json = json.load(deaths_file_obj)

            logging.info("Define yaml keys to dictionary maps for cases and deaths")
            self.cases_yaml_keys_dict_keys_map = {
                'HISPANIC_LATINO_CASES': 'Hispanic',
                'WHITE_CASES': 'White',
                'ASIAN_CASES': 'Asian',
                'BLACK_CASES': 'Black',
                'PACIFIC_ISLANDER_CASES': 'Pacific Islander',
                'NATIVE_AMERICAN_CASES': 'Native American',
                'MULTI_RACE_CASES': 'Multi-Race'}
            self.deaths_yaml_keys_dict_keys_map = {
                'HISPANIC_LATINO_DEATHS': 'Hispanic',
                'WHITE_DEATHS': 'White',
                'ASIAN_DEATHS': 'Asian',
                'BLACK_DEATHS': 'Black',
                'WHITE_DEATHS': 'White'}
        except Exception:
            pass

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['White', 'Black', 'Native American', 'Asian', 'Pacific Islander', 'Hispanic', 'Multi-Race']

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in Alameda County.

        Obtained from here: https://www.census.gov/quickfacts/alamedacountycalifornia

        """
        return {'White': 0.306, 'Black': 0.110, 'Native American': 0.011, 'Asian': 0.323,
                'Pacific Islander': 0.009, 'Hispanic': 0.223, 'Multi-Race': 0.054}

    def process_raw_data_to_cases(self) -> bool:
        """
        Process raw data to obtain number of covid cases for each ethnicity and define
        totals and percentages
        """
        if self.cases_yaml_keys_dict_keys_map is not None:
            if self.ethnicity_json_keys_map is not None:
                self.ethnicity_cases_dict, self.ethnicity_cases_percentages_dict = self.get_cases_deaths_using_json(
                    raw_data_json=self.raw_data_cases_json, ethnicity_json_keys_map=self.ethnicity_json_keys_map, yaml_keys_dict_keys_map=self.cases_yaml_keys_dict_keys_map, valid_date_string=self.cases_valid_date_string)
                return True
        return False

    def process_raw_data_to_deaths(self) -> bool:
        """
        Process raw data to obtain number of covid deaths for each ethnicity and define
        totals and percentages
        """
        if self.deaths_yaml_keys_dict_keys_map is not None:
            if self.ethnicity_json_keys_map is not None:
                self.ethnicity_deaths_dict, self.ethnicity_deaths_percentages_dict = self.get_cases_deaths_using_json(
                    raw_data_json=self.raw_data_deaths_json, ethnicity_json_keys_map=self.ethnicity_json_keys_map, yaml_keys_dict_keys_map=self.deaths_yaml_keys_dict_keys_map, valid_date_string=self.cases_valid_date_string)
                return True
        return False
