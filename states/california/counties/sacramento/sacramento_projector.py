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
from states.california.counties.alameda.alameda_projector import AlamedaEthnicDataProjector
from states import utils


class SacramentoEthnicDataProjector(AlamedaEthnicDataProjector):
    def __init__(self, state: str, county: str, date_string: str):
        # Initialize relevant variables
        self.state, self.county = state, county
        self.cases_raw_bool, self.deaths_raw_bool = False, False
        logging.info("Initialize imperial county raw and config file strings")

        # Define raw and config files to be loaded
        raw_data_dir = os.path.join("states", state, 'counties', county, "raw_data")
        raw_data_cases_file, raw_data_cases_file_html = f"{raw_data_dir}/{date_string}/sacramento_cases", f"{raw_data_dir}/{date_string}/sacramento_cases.html"
        raw_data_deaths_file, raw_data_deaths_file_html = f"{raw_data_dir}/{date_string}/sacramento_deaths", f"{raw_data_dir}/{date_string}/sacramento_deaths.html"

        configs_dir = os.path.join("states", state, 'counties', county, "configs")
        cases_config_file_string = f"{configs_dir}/sacramento_cases_json_parser.yaml"
        deaths_config_file_string = f"{configs_dir}/sacramento_deaths_json_parser.yaml"

        # Load config files that will be used for parsing
        logging.info("Load cases and deaths parsing config")
        json_parser_cases_config = self.load_yaml(cases_config_file_string)
        json_parser_deaths_config = self.load_yaml(deaths_config_file_string)

        # Get all dates for which parsing currently exists
        logging.info("Get and sort json parsing dates")
        json_parser_cases_dates = self.get_sorted_dates_from_strings(
            date_string_list=list(json_parser_cases_config["DATES"].keys()))
        json_parser_deaths_dates = self.get_sorted_dates_from_strings(
            date_string_list=list(json_parser_deaths_config["DATES"].keys()))

        # Get most recent parsing date with respect to the passed in date_string
        logging.info("Obtain valid map of ethnicities to json containing cases or deaths")
        self.date_string = date_string
        self.cases_valid_date_string = utils.get_valid_date_string(
            date_list=json_parser_cases_dates, date_string=date_string)
        self.deaths_valid_date_string = utils.get_valid_date_string(
            date_list=json_parser_deaths_dates, date_string=date_string)

        # Get JSON keys for the chosen date
        self.cases_ethnicity_json_keys_map = json_parser_cases_config['DATES'][self.cases_valid_date_string]
        self.deaths_ethnicity_json_keys_map = json_parser_deaths_config['DATES'][self.deaths_valid_date_string]
        self.ethnicity_json_keys_map = {**self.cases_ethnicity_json_keys_map, **self.deaths_ethnicity_json_keys_map}

        # Load raw json files for cases and/or deaths depending on whether or not it exists
        logging.info("Load raw json data")
        try:
            cases_file_obj = open(raw_data_cases_file, 'r')
            self.cases_raw_bool = True
        except BaseException:
            try:
                cases_file_obj = open(
                    raw_data_cases_file_html, 'r')
                self.cases_raw_bool = True
            except BaseException:
                pass

        try:
            deaths_file_obj = open(raw_data_deaths_file, 'r')
        except BaseException:
            try:
                deaths_file_obj = open(
                    raw_data_deaths_file_html, 'r')
                self.deaths_raw_bool = True
            except BaseException:
                pass

        try:
            self.raw_data_cases_json = json.load(cases_file_obj)
            self.raw_data_deaths_json = json.load(deaths_file_obj)
        except BaseException:
            pass

        # Define mapping of YAML keys from the JSON parser to the
        # names in this class
        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {
            'WHITE_CASES': 'White',
            'HISPANIC_CASES': 'Hispanic',
            'ASIAN_CASES': 'Asian',
            'BLACK_CASES': 'Black',
            'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES': 'Native Hawaiian/Pacific Islander',
            'AMERICAN_INDIAN_ALASKA_NATIVE_CASES': 'American Indian/Alaska Native'
        }
        self.deaths_yaml_keys_dict_keys_map = {
            'WHITE_DEATHS': 'White',
            'HISPANIC_DEATHS': 'Hispanic',
            'ASIAN_DEATHS': 'Asian',
            'BLACK_DEATHS': 'Black',
            'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_DEATHS': 'Native Hawaiian/Pacific Islander'
        }

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['White', 'Hispanic', 'Asian', 'Black', 'Native Hawaiian/Pacific Islander', 'American Indian/Alaska Native']

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in Sacramento County

        Obtained from here: https://www.census.gov/quickfacts/sacramentocountycalifornia

        """
        return {'White': 0.628, 'Hispanic': 0.236, 'Asian': 0.170, 'Black': 0.109,
                'Native Hawaiian/Pacific Islander': 0.013, 'American Indian/Alaska Native': 0.015}
