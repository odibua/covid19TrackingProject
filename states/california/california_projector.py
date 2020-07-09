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
import pandas as pd
from typing import Dict, List
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------
from states.data_projectors import EthnicDataProjector
from states import utils


class CaliforniaEthnicDataProjector(EthnicDataProjector):
    def __init__(self, date_string: str):
        super().__init__(self)
        self.state = 'california'
        self.raw_data_dir = os.path.join("states", self.state, "raw_data")

        logging.info("Load california html parsing config")
        configs_dir = os.path.join("states", self.state, "configs")
        html_parser_config_file = open(f"{configs_dir}/california_all_html_parse.yaml")
        html_parser_config = yaml.safe_load(html_parser_config_file)

        logging.info("Get and sort californial html parsing dates")
        html_parser_date_strings = self.html_parser_config["DATES"].keys()
        html_parser_dates = [datetime.strptime(date_string, '%Y-%m-%d') for date_string in html_parser_date_strings]
        html_parser_dates.sort()

        logging.info("Obtain valid map of ethnicities to xpath containing cases or deaths")
        self.valid_date_string = utils.get_valid_date_string(date_list=self.html_parser_dates, date_string=date_string)
        self.ethnicity_xpath_map = html_parser_config['DATES'][self.valid_date_string]

        logging.info("Load raw html data and convert it to lxml")
        raw_data_file = f"{self.raw_data_dir}/california_all.html"
        raw_data_file_object = open(raw_data_file, 'r')
        raw_data_file_html = raw_data_file_object.read()
        self.raw_data_lxml = etree.HTML(raw_data_file_html)

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ["latino", "white", "asian", "black", "multi_race", "american_indian_or_alaska_native", "native_hawaiian_pacific_islander", "other"]

    def process_raw_data_to_cases(self) -> Dict[str, int]:
        """
        Process raw page to obtain number of covid cases for each ethnicity
        """
        cases_dict = {}
        logging.info(f"Use xpaths from {self.valid_date_string} to construct California cases dictionary")
        cases_dict['latino'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['LATINO_CASES'])
        cases_dict['white'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['WHITE_CASES'])
        cases_dict['asian'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['ASIAN_CASES'])
        cases_dict['black'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['BLACK_CASES'])
        cases_dict['multirace'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['MULTI_RACE_CASES'])
        cases_dict['american_indian_alaska_natives'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['AMERICAN_INDIAN_OR_ALASKA_NATIVE_CASES'])
        cases_dict['native_hawaiian_pacific_islander'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES'])
        cases_dict['other'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['OTHER_CASES'])

        return cases_dict

    def process_raw_data_to_deaths(self) -> Dict[str, int]:
        """
        Process raw page to obtain number of covid deaths for each ethnicity
        """
        deaths_dict = {}
        logging.info(f"Use xpaths from {self.valid_date_string} to construct California cases dictionary")
        deaths_dict['latino'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['LATINO_DEATHS'])
        deaths_dict['white'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['WHITE_DEATHS'])
        deaths_dict['asian'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['ASIAN_DEATHS'])
        deaths_dict['black'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['BLACK_DEATHS'])
        deaths_dict['multirace'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['MULTI_RACE_DEATHS'])
        deaths_dict['american_indian_alaska_natives'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['AMERICAN_INDIAN_OR_ALASKA_NATIVE_DEATHS'])
        deaths_dict['native_hawaiian_pacific_islander'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['NATIVE_HAWAIIAN_PACIFIC_ISLANDER_DEATHS'])
        deaths_dict['other'] = self.raw_data_lxml.xpath(self.ethnicity_xpath_map['OTHER_DEATHS'])

        return deaths_dict
