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
from states import utils_state_lib


class RiverSideEthnicDataProjector(AlamedaEthnicDataProjector):
    def __init__(self, state: str, county: str, date_string: str):
        self.state, self.county = state, county
        self.cases_raw_bool, self.deaths_raw_bool = False, False
        logging.info("Initialize riverside raw and config file strings")
        raw_data_dir = os.path.join("states", state, 'counties', county, "raw_data")
        raw_data_cases_file, raw_data_cases_file_html = f"{raw_data_dir}/{date_string}/riverside_cases", f"{raw_data_dir}/{date_string}/riverside_cases.html"

        configs_dir = os.path.join("states", state, 'counties', county, "configs")
        cases_config_file_string = f"{configs_dir}/riverside_cases_json_parser.yaml"

        logging.info("Load cases and deaths parsing config")
        json_parser_cases_config = self.load_yaml(cases_config_file_string)

        logging.info("Get and sort json parsing dates")
        json_parser_cases_dates = self.get_sorted_dates_from_strings(
            date_string_list=list(json_parser_cases_config["DATES"].keys()))

        logging.info("Obtain valid map of ethnicities to json containing cases or deaths")
        self.date_string = date_string
        self.cases_valid_date_string = utils_state_lib.get_valid_date_string(
            date_list=json_parser_cases_dates, date_string=date_string)
        self.cases_ethnicity_json_keys_map, self.deaths_yaml_keys_dict_keys_map = json_parser_cases_config[
            'DATES'][self.cases_valid_date_string], None
        self.ethnicity_json_keys_map = self.cases_ethnicity_json_keys_map

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
            self.raw_data_cases_json = json.load(cases_file_obj)
        except BaseException:
            pass

        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {
            'HISPANIC_LATINO_CASES': 'Hispanic',
            'MULTI_RACE_CASES': 'Multi-Race',
            'WHITE_CASES': 'White',
            'ASIAN_PACIFIC_ISLANDER_CASES': 'Asian/Pacific Islander',
            'ASIAN_CASES': 'Asian',
            'AMERICAN_INDIAN_ALASKA_NATIVE_CASES': 'American Indian/Alaska Native',
            'BLACK_CASES': 'Black',
            'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES': 'Native Hawaiian/Pacifc Islander'
        }

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['Hispanic', 'Multi-Race', 'White', 'Asian/Pacific Islander', 'Asian',
                'American Indian/Alaska Native', 'Black', 'Native Hawaiian/Pacifc Islander']

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in Imperial County

        Obtained from here: https://www.census.gov/quickfacts/riversidecountycalifornia

        """
        return {'Hispanic': 0.50, 'Multi-Race': 0.036, 'White': 0.341, 'Asian/Pacific Islander': 0.076, 'Asian': 0.072,
                'American Indian/Alaska Native': 0.019, 'Black': 0.073, 'Native Hawaiian/Pacifc Islander': 0.004}

    @property
    def ethnicity_demographics_total(self) -> Dict[str, float]:
        """
        Return dictionary that contains total of each ethnicity population in Imperial County

        Obtained from here: https://www.census.gov/quickfacts/riversidecountycalifornia

        """
        total = 2470546
        return {'Hispanic': int(0.50 * total), 'Multi-Race': int(0.036 * total), 'White': int(0.341 * total), 'Asian/Pacific Islander': int(0.076 * total), 'Asian': int(0.072 * total),
                'American Indian/Alaska Native': int(0.019 * total), 'Black': int(0.073 * total), 'Native Hawaiian/Pacifc Islander': int(0.004 * total)}
