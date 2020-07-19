# --------------------------
# Standard Python Imports
# --------------------------
import json
import logging
import os

# --------------------------
# Third Party Imports
# --------------------------
from typing import Any, Dict, List, Tuple
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------
from states.california.counties.alameda.alameda_projector import AlamedaEthnicDataProjector
from states import utils


class SantaClaraEthnicDataProjector(AlamedaEthnicDataProjector):
    def __init__(self, state: str, county: str, date_string: str):
        self.state, self.county = state, county
        self.total_cases_int, self.total_deaths_int = None, None
        logging.info("Initialize imperial county raw and config file strings")
        raw_data_dir = os.path.join("states", state, 'counties', county, "raw_data")
        raw_data_cases_file, raw_data_totalcases_file = f"{raw_data_dir}/{date_string}/santaclara_cases", f"{raw_data_dir}/{date_string}/santaclara_totalcases"
        raw_data_deaths_file, raw_data_totaldeaths_file = f"{raw_data_dir}/{date_string}/santaclara_deaths", f"{raw_data_dir}/{date_string}/santaclara_totaldeaths"

        configs_dir = os.path.join("states", state, 'counties', county, "configs")
        cases_config_file_string = f"{configs_dir}/santaclara_cases_json_parser.yaml"
        deaths_config_file_string = f"{configs_dir}/santaclara_deaths_json_parser.yaml"
        totalcases_config_file_string = f"{configs_dir}/santaclara_totalcases_json_parser.yaml"
        totaldeaths_config_file_string = f"{configs_dir}/santaclara_totaldeaths_json_parser.yaml"

        logging.info("Load cases and deaths parsing config")
        json_parser_cases_config = self.load_yaml(cases_config_file_string)
        json_parser_deaths_config = self.load_yaml(deaths_config_file_string)
        json_parser_totalcases_config = self.load_yaml(totalcases_config_file_string)
        json_parser_totaldeaths_config = self.load_yaml(totaldeaths_config_file_string)

        logging.info("Get and sort json parsing dates")
        json_parser_cases_dates = self.get_sorted_dates_from_strings(date_string_list=list(json_parser_cases_config["DATES"].keys()))
        json_parser_deaths_dates = self.get_sorted_dates_from_strings(date_string_list=list(json_parser_deaths_config["DATES"].keys()))
        json_parser_totalcases_dates = self.get_sorted_dates_from_strings(date_string_list=list(json_parser_totalcases_config["DATES"].keys()))
        json_parser_totaldeaths_dates = self.get_sorted_dates_from_strings(date_string_list=list(json_parser_totaldeaths_config["DATES"].keys()))

        logging.info("Obtain valid map of ethnicities to json containing cases or deaths")
        self.cases_valid_date_string = utils.get_valid_date_string(
            date_list=json_parser_cases_dates, date_string=date_string)
        self.deaths_valid_date_string = utils.get_valid_date_string(
            date_list=json_parser_deaths_dates, date_string=date_string)
        self.totalcases_valid_date_string = utils.get_valid_date_string(
            date_list=json_parser_totalcases_dates, date_string=date_string)
        self.totaldeaths_valid_date_string = utils.get_valid_date_string(
            date_list=json_parser_totaldeaths_dates, date_string=date_string)

        self.cases_ethnicity_json_keys_map = json_parser_cases_config['DATES'][self.cases_valid_date_string]
        self.deaths_ethnicity_json_keys_map = json_parser_deaths_config['DATES'][self.deaths_valid_date_string]
        self.ethnicity_json_keys_map = {**self.cases_ethnicity_json_keys_map, **self.deaths_ethnicity_json_keys_map}

        self.totalcases_ethnicity_json_keys_map = json_parser_totalcases_config['DATES'][self.totalcases_valid_date_string]
        self.totaldeaths_ethnicity_json_keys_map = json_parser_totaldeaths_config['DATES'][self.totaldeaths_valid_date_string]
        self.totals_json_keys_map = {**self.totalcases_ethnicity_json_keys_map, **self.totaldeaths_ethnicity_json_keys_map}

        logging.info("Load raw json data")
        cases_file_obj, deaths_file_obj= open(raw_data_cases_file, 'r'), open(raw_data_deaths_file, 'r')
        totalcases_file_obj, totaldeaths_file_obj= open(raw_data_totalcases_file, 'r'), open(raw_data_totaldeaths_file, 'r')


        self.raw_data_cases_json, self.raw_data_deaths_json  = json.load(cases_file_obj), json.load(deaths_file_obj)
        self.raw_data_totalcases_json, self.raw_data_totaldeaths_json = json.load(totalcases_file_obj), json.load(totaldeaths_file_obj)

        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {
            'WHITE_CASES': 'White',
            'HISPANIC_CASES': 'Hispanic',
            'ASIAN_CASES': 'Asian',
            'BLACK_CASES': 'Black',
            'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES': 'Native Hawaiian/Pacific Islander',
            'OTHER_CASES': 'Other'
        }
        self.deaths_yaml_keys_dict_keys_map = {
            'WHITE_DEATHS': 'White',
            'HISPANIC_DEATHS': 'Hispanic',
            'ASIAN_DEATHS': 'Asian',
            'BLACK_DEATHS': 'Black',
            'OTHER_DEATHS': 'Other',
            'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_DEATHS': 'Native Hawaiian/Pacific Islander'
        }
        self.totals_cases_yaml_keys_dict_keys_map = {
            'TOTAL_CASES': 'Total Cases',
        }
        self.totals_deaths_yaml_keys_dict_keys_map = {
            'TOTAL_DEATHS': 'Total Deaths'
        }


    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['White', 'Hispanic', 'Asian', 'Black', 'Native Hawaiian/Pacific Islander', 'Other']

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in Sacramento County

        Obtained from here: https://www.census.gov/quickfacts/santaclaracountycalifornia

        """
        return {'White': 0.524, 'Hispanic': 0.250, 'Asian': 0.390, 'Black': 0.028, 'Native Hawaiian/Pacific Islander': 0.005,  'Other': 0.054}

    @property
    def total_cases(self) -> int:
        return self.total_cases_int

    @property
    def total_deaths(self) -> int:
        return self.total_deaths_int

    def process_raw_data_to_cases(self) -> bool:
        """
        Process raw data to obtain number of covid cases for each ethnicity and define
        totals and percentages
        """
        if self.cases_yaml_keys_dict_keys_map is not None:
            if self.ethnicity_json_keys_map is not None:
                self.ethnicity_cases_dict, self.ethnicity_cases_percentages_dict, self.total_cases_int = self.santa_clara_get_cases_deaths_using_json(
                    raw_data_json=self.raw_data_cases_json, total_raw_data_json=self.raw_data_totalcases_json, ethnicity_json_keys_map=self.ethnicity_json_keys_map,  total_ethnicity_json_keys_map=self.totalcases_ethnicity_json_keys_map,
                    yaml_keys_dict_keys_map=self.cases_yaml_keys_dict_keys_map, total_yaml_keys_dict_keys_map=self.totals_cases_yaml_keys_dict_keys_map, valid_date_string=self.cases_valid_date_string, total_valid_date_string=self.totalcases_valid_date_string)
                return True
        return False

    def process_raw_data_to_deaths(self) -> bool:
        """
        Process raw data to obtain number of covid deaths for each ethnicity and define
        totals and percentages
        """
        if self.deaths_yaml_keys_dict_keys_map is not None:
            if self.ethnicity_json_keys_map is not None:
                self.ethnicity_deaths_dict, self.ethnicity_deaths_percentages_dict, self.total_deaths_int = self.santa_clara_get_cases_deaths_using_json(
                    raw_data_json=self.raw_data_deaths_json, total_raw_data_json=self.raw_data_totaldeaths_json, ethnicity_json_keys_map=self.ethnicity_json_keys_map,  total_ethnicity_json_keys_map=self.totaldeaths_ethnicity_json_keys_map,
                    yaml_keys_dict_keys_map=self.deaths_yaml_keys_dict_keys_map, total_yaml_keys_dict_keys_map=self.totals_deaths_yaml_keys_dict_keys_map, valid_date_string=self.deaths_valid_date_string, total_valid_date_string=self.totaldeaths_valid_date_string)
                return True
        return False

    @staticmethod
    def santa_clara_get_cases_deaths_using_json(raw_data_json: Dict[str, Any], total_raw_data_json: Dict[str, Any], ethnicity_json_keys_map: Dict[str, str],
                                    total_ethnicity_json_keys_map: Dict[str, str], yaml_keys_dict_keys_map: Dict[str, str], total_yaml_keys_dict_keys_map: Dict[str, str], valid_date_string: str, total_valid_date_string: str) -> Tuple[
        Dict[str, int], Dict[str, float], int]:
        """
        Get the case information from the raw_data_lxml using the ethnicity_xpath_map and yaml to dict keys mapping

        Arguments:
            raw_data_json: Raw json object
            ethnicity_json_keys_map: Map of ethnicity to JSON keys
            yaml_keys_dict_keys_map: Yaml key to dictionary key map
            valid_date_string: Date from which ethnicity to xpath map is obtained

        Returns:
            Dictionaries that give counts and percentages
        """
        logging.info(f"Use json from {valid_date_string} to construct percentages of cases or deaths dictionary")
        ethnicity_dict, ethnicity_percentages_dict, total_dict = {}, {}, {}
        for key in yaml_keys_dict_keys_map.keys():
            if key in ethnicity_json_keys_map:
                ethnicity_percentages_dict[yaml_keys_dict_keys_map[key]] = float(utils.get_json_element_int(
                    raw_data_json=raw_data_json, ethnicity_json_keys_list=ethnicity_json_keys_map[key]))

        logging.info(f"Use json from {total_valid_date_string} to construct total cases or deaths dictionary")
        key = list(total_yaml_keys_dict_keys_map.keys())[0]
        total = float(utils.get_json_element_int(
                    raw_data_json=total_raw_data_json, ethnicity_json_keys_list=total_ethnicity_json_keys_map[key]))

        logging.info("Get cases or deaths that are each ethnicity based on known ethnicities")
        for key in ethnicity_percentages_dict.keys():
            ethnicity_dict[key] = int(total * float(ethnicity_percentages_dict[key]))

        return ethnicity_dict, ethnicity_percentages_dict, total
