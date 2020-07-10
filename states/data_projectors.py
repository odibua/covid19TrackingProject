# --------------------------
# Standard Python Imports
# --------------------------
from abc import ABC, abstractmethod
from datetime import datetime
import logging
from lxml import etree
import os

# --------------------------
# Third Party Imports
# --------------------------
import bs4
import pandas as pd
from typing import Dict, List, Tuple
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------
from states import utils


class EthnicDataProjector(ABC):
    def __init__(self, state: str, county: str, raw_data_file: str, date_string: str, config_file_string: str, json: bool=None, lxml: bool=None):
        """
        Initialize the parameters necessary for projecting raw data to cases and deaths numbers

        state: State for which projection will be done
        county: County for which projection will be done
        raw_data_file: Raw data file from which ethnic data will be parsed
        date_string: Date of concern
        config_file_string: Configuration file string
        json: Boolean to state if parsing will be json or not
        lxml: Boolean to state if lxml will be used for parsing
        """
        self.state, self.county = state, county
        self.ethnicitiy_json_keys_map = None
        self.ethnicity_cases_dict, self.ethnicity_cases_percentages_dict = {}, {}
        self.ethnicity_deaths_dict, self.ethnicity_deaths_percentages_dict = {}, {}
        self.cases_yaml_keys_dict_keys_map, self.deaths_yaml_keys_dict_keys_map = {}, {}

        logging.info("Load parsing config")
        html_parser_config_file = open(config_file_string)
        html_parser_config = yaml.safe_load(html_parser_config_file)

        logging.info("Get and sort html parsing dates")
        html_parser_date_strings = html_parser_config["DATES"].keys()
        html_parser_dates = [datetime.strptime(date_string, '%Y-%m-%d') for date_string in html_parser_date_strings]
        html_parser_dates.sort()

        logging.info("Obtain valid map of ethnicities to xpath containing cases or deaths")
        self.valid_date_string = utils.get_valid_date_string(date_list=html_parser_dates, date_string=date_string)
        if lxml:
            self.ethnicity_xpath_map = html_parser_config['DATES'][self.valid_date_string]
            logging.info("Load raw html data and convert it to lxml")
            raw_data_file_object = open(raw_data_file, 'r')
            raw_data_file_html = raw_data_file_object.read()
            soup = bs4.BeautifulSoup(raw_data_file_html, 'html5lib')
            raw_data_file_html = soup.prettify()
            self.raw_data_lxml = etree.HTML(raw_data_file_html)

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
    def ethnicity_cases_percentages(self) -> Dict[str, float]:
        """
        Return dictionary of case percentages of ethnicities contained in an area
        """
        return self.ethnicity_cases_percentages_dict

    @property
    def ethnicity_cases(self) -> Dict[str, int]:
        """
        Return dictionary of cases of ethnicities contained in an area
        """
        return self.ethnicity_cases_dict

    @property
    def ethnicity_cases_discrepancies(self) -> Dict[str, float]:
        """
        Return dictionary of discrepancy for each race contained quantified as ratio between case percentage and population percentage
        in region
        """
        discrepancy_dict = {}
        if self.ethnicity_cases_percentages_dict.keys() is not None and self.ethnicity_demographics.keys() is not None:
            for key in self.ethnicity_cases_percentages_dict.keys():
                discrepancy_dict[key] = round(self.ethnicity_cases_percentages_dict[key]/self.ethnicity_demographics[key], 3)
        return discrepancy_dict

    @property
    def ethnicity_deaths_percentages(self) -> Dict[str, float]:
        """
        Return dictionary of death percentages of ethnicities contained in an area
        """
        return self.ethnicity_deaths_percentages_dict

    @property
    def ethnicity_deaths(self) -> Dict[str, float]:
        """
        Return dictionary of cases of ethnicities contained in an area
        """
        return self.ethnicity_deaths_dict

    @property
    def ethnicity_deaths_discrepancies(self) -> Dict[str, float]:
        """
        Return dictionary of discrepancy for each race contained quantified as ratio between case percentage and population percentage
        in region
        """
        discrepancy_dict = {}
        if self.ethnicity_deaths_percentages_dict.keys() is not None and self.ethnicity_demographics.keys() is not None:
            for key in self.ethnicity_deaths_percentages_dict.keys():
                discrepancy_dict[key] = round(self.ethnicity_deaths_percentages_dict[key]/self.ethnicity_demographics[key], 3)
        return discrepancy_dict

    def process_raw_data_to_cases(self) -> None:
        """
        Process raw page to obtain number of covid cases for each ethnicity and define
        totals and percentages
        """
        if self.ethnicity_xpath_map is not None:
            self.ethnicity_cases_dict, self.ethnicity_cases_percentages_dict = self.get_cases_deaths_using_lxml(raw_data_lxml=self.raw_data_lxml, ethnicity_xpath_map=self.ethnicity_xpath_map, yaml_keys_dict_keys_map=self.cases_yaml_keys_dict_keys_map, valid_date_string=self.valid_date_string)
        elif self.ethnicitiy_json_keys_map is not None:
            raise ValueError("Json Keys Map not implemented for processing cases")
        return None

    def process_raw_data_to_deaths(self) -> None:
        """
        Process raw page to obtain number of covid deaths for each ethnicity and define
        totals and percentages
        """
        if self.ethnicity_xpath_map is not None:
            self.ethnicity_deaths_dict, self.ethnicity_deaths_percentages_dict = self.get_cases_deaths_using_lxml(raw_data_lxml=self.raw_data_lxml, ethnicity_xpath_map=self.ethnicity_xpath_map, yaml_keys_dict_keys_map=self.deaths_yaml_keys_dict_keys_map, valid_date_string=self.valid_date_string)
        elif self.ethnicitiy_json_keys_map is not None:
            raise ValueError("Json Keys Map not implemented for processing cases")
        return None

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
            ethnicity_dict[yaml_keys_dict_keys_map[key]] = utils.get_element_int(element=raw_data_lxml.xpath(ethnicity_xpath_map[key]))

        logging.info("Get percentage of cases or deaths that are each ethnicity based on known ethnicities")
        total = utils.get_total(numerical_dict=ethnicity_dict)
        for key in ethnicity_dict.keys():
            ethnicity_percentages_dict[key] = round(float(ethnicity_dict[key])/total, 3)

        return ethnicity_dict, ethnicity_percentages_dict
