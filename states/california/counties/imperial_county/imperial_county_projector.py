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


class ImperialCountyEthnicDataProjector(AlamedaEthnicDataProjector):
    def __init__(self, state: str, county: str, date_string: str):
        self.state, self.county = state, county
        logging.info("Initialize imperial county raw and config file strings")
        raw_data_dir = os.path.join("states", state, 'counties', county, "raw_data")
        raw_data_cases_file, raw_data_cases_file_html = f"{raw_data_dir}/{date_string}/imperial_county_cases", f"{raw_data_dir}/{date_string}/imperial_county_cases.html"
        raw_data_deaths_file, raw_data_deaths_file_html = f"{raw_data_dir}/{date_string}/imperial_county_deaths", f"{raw_data_dir}/{date_string}/imperial_county_deaths.html"

        configs_dir = os.path.join("states", state, 'counties', county, "configs")
        cases_config_file_string = f"{configs_dir}/imperial_county_cases_json_parser.yaml"
        deaths_config_file_string = f"{configs_dir}/imperial_county_deaths_json_parser.yaml"

        logging.info("Load cases and deaths parsing config")
        json_parser_cases_config = self.load_yaml(cases_config_file_string)
        json_parser_deaths_config = self.load_yaml(deaths_config_file_string)

        logging.info("Get and sort json parsing dates")
        json_parser_cases_dates = self.get_sorted_dates_from_strings(
            date_string_list=list(json_parser_cases_config["DATES"].keys()))
        json_parser_deaths_dates = self.get_sorted_dates_from_strings(
            date_string_list=list(json_parser_deaths_config["DATES"].keys()))

        logging.info("Obtain valid map of ethnicities to json containing cases or deaths")
        self.cases_valid_date_string = utils.get_valid_date_string(
            date_list=json_parser_cases_dates, date_string=date_string)
        self.deaths_valid_date_string = utils.get_valid_date_string(
            date_list=json_parser_deaths_dates, date_string=date_string)
        self.cases_ethnicity_json_keys_map = json_parser_cases_config['DATES'][self.cases_valid_date_string]
        self.deaths_ethnicity_json_keys_map = json_parser_deaths_config['DATES'][self.deaths_valid_date_string]
        self.ethnicity_json_keys_map = {**self.cases_ethnicity_json_keys_map, **self.deaths_ethnicity_json_keys_map}

        logging.info("Load raw json data")
        try:
            cases_file_obj, deaths_file_obj = open(raw_data_cases_file, 'r'), open(raw_data_deaths_file, 'r')
        except BaseException:
            cases_file_obj, deaths_file_obj = open(raw_data_cases_file_html, 'r'), open(raw_data_deaths_file_html, 'r')

        self.raw_data_cases_json = json.load(cases_file_obj)
        self.raw_data_deaths_json = json.load(deaths_file_obj)

        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {
            'HISPANIC_LATINO_CASES': 'Hispanic',
            'NON_HISPANIC_LATINO_CASES': 'Non-Hispanic'}
        self.deaths_yaml_keys_dict_keys_map = {
            'HISPANIC_LATINO_DEATHS': 'Hispanic',
            'NON_HISPANIC_LATINO_DEATHS': 'Non-Hispanic'}

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['Hispanic', 'Non-Hispanic', 'Other']

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in Imperial County

        Obtained from here: https://www.census.gov/quickfacts/imperial_countycountycalifornia

        """
        return {'Hispanic': 0.85, 'Non-Hispanic': 0.15, 'Other': 0}
