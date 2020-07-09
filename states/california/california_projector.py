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
from typing import Dict, List
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------
from states.data_projectors import EthnicDataProjector
from states import utils


class CaliforniaEthnicDataProjector(EthnicDataProjector):
    def __init__(self, date_string: str):
        super().__init__()
        self.state = 'california'
        self.raw_data_dir = os.path.join("states", self.state, "raw_data")

        logging.info("Load california html parsing config")
        configs_dir = os.path.join("states", self.state, "configs")
        html_parser_config_file = open(f"{configs_dir}/california_all_html_parse.yaml")
        html_parser_config = yaml.safe_load(html_parser_config_file)

        logging.info("Get and sort californial html parsing dates")
        html_parser_date_strings = html_parser_config["DATES"].keys()
        html_parser_dates = [datetime.strptime(date_string, '%Y-%m-%d') for date_string in html_parser_date_strings]
        html_parser_dates.sort()

        logging.info("Obtain valid map of ethnicities to xpath containing cases or deaths")
        self.valid_date_string = utils.get_valid_date_string(date_list=html_parser_dates, date_string=date_string)
        self.ethnicity_xpath_map = html_parser_config['DATES'][self.valid_date_string]

        logging.info("Load raw html data and convert it to lxml")
        raw_data_file = f"{self.raw_data_dir}/{date_string}/california_all.html"
        raw_data_file_object = open(raw_data_file, 'r')
        raw_data_file_html = raw_data_file_object.read()
        self.raw_data_lxml = etree.HTML(raw_data_file_html)

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ["latino", "white", "asian", "black", "multirace", "american_indian_alaska_native", "native_hawaiian_pacific_islander", "other"]

    def process_raw_data_to_cases(self) -> None:
        """
        Process raw page to obtain number of covid cases for each ethnicity
        """
        logging.info(f"Use xpaths from {self.valid_date_string} to construct California cases dictionary")
        self.ethnicity_cases_dict['latino'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['LATINO_CASES']))
        self.ethnicity_cases_dict['white'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['WHITE_CASES']))
        self.ethnicity_cases_dict['asian'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['ASIAN_CASES']))
        self.ethnicity_cases_dict['black'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['BLACK_CASES']))
        self.ethnicity_cases_dict['multirace'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['MULTI_RACE_CASES']))
        self.ethnicity_cases_dict['american_indian_alaska_natives'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['AMERICAN_INDIAN_OR_ALASKA_NATIVE_CASES']))
        self.ethnicity_cases_dict['native_hawaiian_pacific_islander'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES']))
        self.ethnicity_cases_dict['other'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['OTHER_CASES']))

        logging.info("Get percentage of cases that are each ethnicity based on known ethnicities")
        total_cases = utils.get_total(numerical_dict=self.ethnicity_cases_dict)
        for key in self.ethnicity_cases_dict.keys():
            self.ethnicity_case_percentages_dict[key] = round(float(self.ethnicity_cases_dict[key])/total_cases, 3)

    def process_raw_data_to_deaths(self) -> None:
        """
        Process raw page to obtain number of covid deaths for each ethnicity
        """
        logging.info(f"Use xpaths from {self.valid_date_string} to construct California cases dictionary")
        self.ethnicity_deaths_dict['latino'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['LATINO_DEATHS']))
        self.ethnicity_deaths_dict['white'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['WHITE_DEATHS']))
        self.ethnicity_deaths_dict['asian'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['ASIAN_DEATHS']))
        self.ethnicity_deaths_dict['black'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['BLACK_DEATHS']))
        self.ethnicity_deaths_dict['multirace'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['MULTI_RACE_DEATHS']))
        self.ethnicity_deaths_dict['american_indian_alaska_natives'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['AMERICAN_INDIAN_OR_ALASKA_NATIVE_DEATHS']))
        self.ethnicity_deaths_dict['native_hawaiian_pacific_islander'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['NATIVE_HAWAIIAN_PACIFIC_ISLANDER_DEATHS']))
        self.ethnicity_deaths_dict['other'] = utils.get_element_int(element=self.raw_data_lxml.xpath(self.ethnicity_xpath_map['OTHER_DEATHS']))

        logging.info("Get percentage of deaths that are each ethnicity based on known ethnicities")
        total_deaths = utils.get_total(numerical_dict=self.ethnicity_deaths_dict)
        for key in self.ethnicity_deaths_dict.keys():
            self.ethnicity_deaths_percentages_dict[key] = round(float(self.ethnicity_deaths_dict[key])/total_deaths, 3)

