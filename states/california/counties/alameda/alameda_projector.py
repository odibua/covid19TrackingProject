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
from states import utils


class AlamedaEthnicDataProjector(EthnicDataProjector):
    def __init__(self, state: str, county: str, date_string: str):
        super().__init__(state=state, county=county)
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
        json_parser_cases_dates = self.get_sorted_dates_from_strings(date_string_list=list(json_parser_cases_config["DATES"].keys()))
        json_parser_deaths_dates = self.get_sorted_dates_from_strings(date_string_list=list(json_parser_deaths_config["DATES"].keys()))

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
            cases_file_obj, deaths_file_obj= open(raw_data_cases_file, 'r'), open(raw_data_deaths_file, 'r')
        except:
            cases_file_obj, deaths_file_obj= open(raw_data_cases_file_html, 'r'), open(raw_data_deaths_file_html, 'r')

        self.raw_data_cases_json = json.load(cases_file_obj)
        self.raw_data_deaths_json = json.load(deaths_file_obj)

        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {
            'HISPANIC_LATINO_CASES': 'hispanic',
            'WHITE_CASES': 'white',
            'ASIAN_CASES': 'asian',
            'BLACK_CASES': 'black',
            'PACIFIC_ISLANDER_CASES': 'pacific_islander',
            'NATIVE_AMERICAN_CASES': 'native_american',
            'MULTI_RACE_CASES': 'multirace'}
        self.deaths_yaml_keys_dict_keys_map = {
            'HISPANIC_LATINO_DEATHS': 'hispanic',
            'WHITE_DEATHS': 'white',
            'ASIAN_DEATHS': 'asian',
            'BLACK_DEATHS': 'black',
            'WHITE_DEATHS': 'white'}

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['white', 'black', 'native_american', 'asian', 'pacific_islander', 'hispanic', 'multirace']

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in Alameda County.

        Obtained from here: https://www.census.gov/quickfacts/alamedacountycalifornia

        """
        return {'white': 0.306, 'black': 0.110, 'native_american': 0.011, 'asian': 0.323, 'pacific_islander': 0.009, 'hispanic': 0.223, 'multirace': 0.054}

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
