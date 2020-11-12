# --------------------------
# Standard Python Imports
# --------------------------
from abc import ABC, abstractmethod
from datetime import datetime
import logging
from lxml import etree

# --------------------------
# Third Party Imports
# --------------------------
from typing import Any, Dict, List, Tuple
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------
from states import utils_state_lib


class EthnicDataProjector(ABC):
    def __init__(self, state: str, county: str):
        """
        Initialize the parameters necessary for projecting raw data to cases and deaths numbers

        state: State for which projection will be done
        county: County for which projection will be done
        """
        self.state, self.county = state, county
        self.ethnicitiy_json_keys_map, self.ethnicity_xpath_map = None, None
        self.ethnicity_cases_dict, self.ethnicity_cases_percentages_dict = None, None
        self.ethnicity_deaths_dict, self.ethnicity_deaths_percentages_dict = None, None
        self.cases_yaml_keys_dict_keys_map, self.deaths_yaml_keys_dict_keys_map = None, None
        self.cases_raw_bool, self.deaths_raw_bool = False, False

    @property
    @abstractmethod
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return []

    @property
    @abstractmethod
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in california
        """
        return {}

    @property
    @abstractmethod
    def total_population(self) -> int:
        """
        Return total population for a given region
        """
        return None

    @property
    @abstractmethod
    def ethnicity_demographics_pop_perc(self) -> Dict[str, float]:
        """
        Return dictionary that contains total of each ethnicity population in california
        """
        return {}

    @property
    def ethnicity_cases_percentages(self) -> Dict[str, float]:
        """
        Return dictionary of case percentages of ethnicities contained in an area
        """
        dict_with_date = self.ethnicity_cases_percentages_dict
        dict_with_date['date'] = self.date_string
        return dict_with_date

    @property
    def ethnicity_cases(self) -> Dict[str, int]:
        """
        Return dictionary of cases of ethnicities contained in an area
        """
        dict_with_date = self.ethnicity_cases_dict
        dict_with_date['date'] = self.date_string
        return dict_with_date

    @property
    def ethnicity_cases_rates(self) -> Dict[str, int]:
        """
        Return dictionary of cases of ethnicities per 1000
        """
        case_rates_dict = {}
        if self.cases_yaml_keys_dict_keys_map is not None and self.ethnicity_demographics.keys() is not None:
            for key in self.ethnicity_cases.keys():
                if key != 'date':
                    case_rates_dict[key] = self.ethnicity_cases[key] * 1000 / self.ethnicity_demographics_pop_perc[key]
        case_rates_dict['date'] = self.date_string
        return case_rates_dict

    @property
    def ethnicity_cases_discrepancies(self) -> Dict[str, float]:
        """
        Return dictionary of discrepancy for each race contained quantified as ratio between case percentage and population percentage
        in region
        """
        discrepancy_dict = {}
        if self.cases_yaml_keys_dict_keys_map is not None and self.ethnicity_demographics.keys() is not None:
            for key in self.ethnicity_cases_percentages_dict.keys():
                if key != 'date':
                    discrepancy_dict[key] = self.ethnicity_cases_percentages_dict[key] / \
                        self.ethnicity_demographics[key]

        discrepancy_dict['date'] = self.date_string
        return discrepancy_dict

    @property
    def ethnicity_deaths_percentages(self) -> Dict[str, float]:
        """
        Return dictionary of death percentages of ethnicities contained in an area
        """
        dict_with_date = self.ethnicity_deaths_percentages_dict
        dict_with_date['date'] = self.date_string
        return dict_with_date

    @property
    def ethnicity_deaths(self) -> Dict[str, float]:
        """
        Return dictionary of cases of ethnicities contained in an area
        """
        dict_with_date = self.ethnicity_deaths_dict
        dict_with_date['date'] = self.date_string
        return dict_with_date

    @property
    def ethnicity_deaths_rates(self) -> Dict[str, int]:
        """
        Return dictionary of deaths of ethnicities per 1000
        """
        death_rates_dict = {}
        if self.deaths_yaml_keys_dict_keys_map is not None and self.ethnicity_demographics.keys() is not None:
            for key in self.ethnicity_deaths.keys():
                if key != 'date':
                    death_rates_dict[key] = self.ethnicity_deaths[key] * \
                        1000 / self.ethnicity_demographics_pop_perc[key]
        death_rates_dict['date'] = self.date_string
        return death_rates_dict

    @property
    def ethnicity_deaths_discrepancies(self) -> Dict[str, float]:
        """
        Return dictionary of discrepancy for each race contained quantified as ratio between case percentage and population percentage
        in region
        """
        discrepancy_dict = {}
        if self.deaths_yaml_keys_dict_keys_map is not None and self.ethnicity_demographics.keys() is not None:
            for key in self.ethnicity_deaths_percentages_dict.keys():
                if key != 'date':
                    discrepancy_dict[key] = self.ethnicity_deaths_percentages_dict[key] / \
                        self.ethnicity_demographics[key]

        discrepancy_dict['date'] = self.date_string
        return discrepancy_dict

    @abstractmethod
    def process_raw_data_to_cases(self) -> bool:
        """
        Process raw data to obtain number of covid cases for each ethnicity and define
        totals and percentages
        """
        return False

    @abstractmethod
    def process_raw_data_to_deaths(self) -> bool:
        """
        Process raw data to obtain number of covid deaths for each ethnicity and define
        totals and percentages
        """
        return False

    @staticmethod
    def get_cases_deaths_using_lxml(raw_data_lxml: etree.HTML, ethnicity_xpath_map: Dict[str, str],
                                    yaml_keys_dict_keys_map: Dict[str, str], valid_date_string: str) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Get the case information from the raw_data_lxml using the ethnicity_xpath_map and yaml to dict keys mapping

        Arguments:
            raw_data_lxml: Raw lxml object
            ethnicity_xpath_map: Map of ethnicity to xpath
            yaml_keys_dict_keys_map: Yaml key to dictionary key map
            valid_date_string: Date from which ethnicity to xpath map is obtained

        Returns:
            Dictionaries that give counts and percentages
        """
        logging.info(f"Use xpaths from {valid_date_string} to construct cases or deaths dictionary")
        ethnicity_dict, ethnicity_percentages_dict = {}, {}
        for key in yaml_keys_dict_keys_map.keys():
            if key in ethnicity_xpath_map:
                ethnicity_dict[yaml_keys_dict_keys_map[key]] = utils_state_lib.get_element_int(
                    element=raw_data_lxml.xpath(ethnicity_xpath_map[key]))

        logging.info("Get percentage of cases or deaths that are each ethnicity based on known ethnicities")
        total = utils_state_lib.get_total(numerical_dict=ethnicity_dict)
        for key in ethnicity_dict.keys():
            ethnicity_percentages_dict[key] = float(ethnicity_dict[key]) / total
        return ethnicity_dict, ethnicity_percentages_dict

    @staticmethod
    def get_cases_deaths_using_json(raw_data_json: Dict[str, Any], ethnicity_json_keys_map: Dict[str, str],
                                    yaml_keys_dict_keys_map: Dict[str, str], valid_date_string: str) -> Tuple[Dict[str, int], Dict[str, float]]:
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
        logging.info(f"Use json from {valid_date_string} to construct cases or deaths dictionary")
        ethnicity_dict, ethnicity_percentages_dict = {}, {}
        for key in yaml_keys_dict_keys_map.keys():
            if key in ethnicity_json_keys_map:
                ethnicity_dict[yaml_keys_dict_keys_map[key]] = utils_state_lib.get_json_element_int(
                    raw_data_json=raw_data_json, ethnicity_json_keys_list=ethnicity_json_keys_map[key])

        logging.info("Get percentage of cases or deaths that are each ethnicity based on known ethnicities")
        total = utils_state_lib.get_total(numerical_dict=ethnicity_dict)
        for key in ethnicity_dict.keys():
            ethnicity_percentages_dict[key] = round(float(ethnicity_dict[key]) / total, 3)
        return ethnicity_dict, ethnicity_percentages_dict

    @staticmethod
    def load_yaml(yaml_file: str) -> yaml:
        yaml_file_object = open(yaml_file)
        return yaml.safe_load(yaml_file_object)

    @staticmethod
    def get_sorted_dates_from_strings(date_string_list: List[str]):
        dates = sorted([datetime.strptime(date_string, '%Y-%m-%d') for date_string in date_string_list])
        return dates
