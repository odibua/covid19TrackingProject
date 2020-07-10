# --------------------------
# Standard Python Imports
# --------------------------
from datetime import datetime
import logging
from lxml import etree
import os

# --------------------------
# Third Party Imports
# --------------------------
import bs4
from typing import Dict, List
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------
from states.data_projectors import EthnicDataProjector
from states import utils


class CaliforniaEthnicDataProjector(EthnicDataProjector):
    def __init__(self, state: str, county: str, raw_data_file: str, date_string: str, config_file_string: str, json: bool = None, lxml: bool = None):
        super().__init__(state=state, county=county)
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

        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {'LATINO_CASES': 'latino', 'WHITE_CASES': 'white', 'ASIAN_CASES': 'asian',
                                              'BLACK_CASES': 'black', 'MULTI_RACE_CASES': 'multirace',
                                              'AMERICAN_INDIAN_OR_ALASKA_NATIVE_CASES': 'american_indian_alaska_native', 'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES': 'native_hawaiian_pacific_islander',
                                              'OTHER_CASES': 'other'}
        self.deaths_yaml_keys_dict_keys_map = {'LATINO_DEATHS': 'latino', 'WHITE_DEATHS': 'white', 'ASIAN_DEATHS': 'asian',
                                              'BLACK_DEATHS': 'black', 'MULTI_RACE_DEATHS': 'multirace',
                                              'AMERICAN_INDIAN_OR_ALASKA_NATIVE_DEATHS': 'american_indian_alaska_native',  'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_DEATHS': 'native_hawaiian_pacific_islander',
                                              'OTHER_DEATHS': 'other'}

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ["latino", "white", "asian", "black", "multirace", "american_indian_alaska_native", "native_hawaiian_pacific_islander", "other"]

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in california
        """
        return {'latino': 0.393, 'white': 0.366, 'asian': 0.145, 'black': 0.055, 'multirace': 0.0308, 'american_indian_alaska_native': 0.0035, 'native_hawaiian_pacific_islander': 0.0036, 'other': 0.0025}

    def process_raw_data_to_cases(self) -> bool:
        """
        Process raw page to obtain number of covid cases for each ethnicity and define
        totals and percentages
        """
        if self.cases_yaml_keys_dict_keys_map is not None:
            if self.ethnicity_xpath_map is not None:
                self.ethnicity_cases_dict, self.ethnicity_cases_percentages_dict = self.get_cases_deaths_using_lxml(raw_data_lxml=self.raw_data_lxml, ethnicity_xpath_map=self.ethnicity_xpath_map, yaml_keys_dict_keys_map=self.cases_yaml_keys_dict_keys_map, valid_date_string=self.valid_date_string)
            elif self.ethnicitiy_json_keys_map is not None:
                raise ValueError("Json Keys Map not implemented for processing cases")
            return True
        return False

    def process_raw_data_to_deaths(self) -> bool:
        """
        Process raw page to obtain number of covid deaths for each ethnicity and define
        totals and percentages
        """
        if self.deaths_yaml_keys_dict_keys_map is not None:
            if self.ethnicity_xpath_map is not None:
                self.ethnicity_deaths_dict, self.ethnicity_deaths_percentages_dict = self.get_cases_deaths_using_lxml(raw_data_lxml=self.raw_data_lxml, ethnicity_xpath_map=self.ethnicity_xpath_map, yaml_keys_dict_keys_map=self.deaths_yaml_keys_dict_keys_map, valid_date_string=self.valid_date_string)
            elif self.ethnicitiy_json_keys_map is not None:
                raise ValueError("Json Keys Map not implemented for processing cases")
            return True
        return False
