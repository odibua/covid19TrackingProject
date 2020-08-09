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
    def __init__(self, state: str, county: str, date_string: str):
        super().__init__(state=state, county=county)
        logging.info("Initialize California raw and config file strings")
        raw_data_dir = os.path.join("states", state, "raw_data")
        raw_data_file = f"{raw_data_dir}/{date_string}/california_all.html"
        configs_dir = os.path.join("states", state, "configs")
        config_file_string = f"{configs_dir}/california_all_html_parse.yaml"

        logging.info("Load parsing config")
        html_parser_config_file = open(config_file_string)
        html_parser_config = yaml.safe_load(html_parser_config_file)

        logging.info("Get and sort html parsing dates")
        html_parser_date_strings = list(html_parser_config["DATES"].keys())
        html_parser_dates = self.get_sorted_dates_from_strings(date_string_list=html_parser_date_strings)

        logging.info("Obtain valid map of ethnicities to xpath containing cases or deaths")
        self.date_string = date_string
        self.valid_date_string = utils.get_valid_date_string(date_list=html_parser_dates, date_string=date_string)
        self.ethnicity_xpath_map = html_parser_config['DATES'][self.valid_date_string]
        logging.info("Load raw html data and convert it to lxml")
        try:
            raw_data_file_object = open(raw_data_file, 'r')
            raw_data_file_html = raw_data_file_object.read()
            soup = bs4.BeautifulSoup(raw_data_file_html, 'html5lib')
            raw_data_file_html = soup.prettify()
            self.raw_data_lxml = etree.HTML(raw_data_file_html)
        except BaseException:
            pass

        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {'LATINO_CASES': 'Hispanic', 'WHITE_CASES': 'White', 'ASIAN_CASES': 'Asian',
                                              'BLACK_CASES': 'Black', 'MULTI_RACE_CASES': 'Multi-Race',
                                              'AMERICAN_INDIAN_OR_ALASKA_NATIVE_CASES': 'American Indian/Alaska Native', 'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES': 'Native Hawaiian/Pacific Islander',
                                              'OTHER_CASES': 'Other'}
        self.deaths_yaml_keys_dict_keys_map = {'LATINO_DEATHS': 'Hispanic', 'WHITE_DEATHS': 'White', 'ASIAN_DEATHS': 'Asian',
                                               'BLACK_DEATHS': 'Black', 'MULTI_RACE_DEATHS': 'Multi-Race',
                                               'AMERICAN_INDIAN_OR_ALASKA_NATIVE_DEATHS': 'American Indian/Alaska Native', 'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_DEATHS': 'Native Hawaiian/Pacific Islander',
                                               'OTHER_DEATHS': 'Other'}

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ["Hispanic", "White", "Asian", "Black", "Multi-Race",
                "American Indian/Alaska Native", "Native Hawaiian/Pacific Islander", "Other"]

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in california
        """
        return {'Hispanic': 0.393, 'White': 0.366, 'Asian': 0.145, 'Black': 0.055, 'Multi-Race': 0.0308,
                'American Indian/Alaska Native': 0.0035, 'Native Hawaiian/Pacific Islander': 0.0036, 'Other': 0.0025}

    def process_raw_data_to_cases(self) -> bool:
        """
        Process raw data to obtain number of covid cases for each ethnicity and define
        totals and percentages
        """
        if self.cases_yaml_keys_dict_keys_map is not None:
            if self.ethnicity_xpath_map is not None:
                self.ethnicity_cases_dict, self.ethnicity_cases_percentages_dict = self.get_cases_deaths_using_lxml(
                    raw_data_lxml=self.raw_data_lxml, ethnicity_xpath_map=self.ethnicity_xpath_map, yaml_keys_dict_keys_map=self.cases_yaml_keys_dict_keys_map, valid_date_string=self.valid_date_string)
            return True
        return False

    def process_raw_data_to_deaths(self) -> bool:
        """
        Process raw data to obtain number of covid deaths for each ethnicity and define
        totals and percentages
        """
        if self.deaths_yaml_keys_dict_keys_map is not None:
            if self.ethnicity_xpath_map is not None:
                self.ethnicity_deaths_dict, self.ethnicity_deaths_percentages_dict = self.get_cases_deaths_using_lxml(
                    raw_data_lxml=self.raw_data_lxml, ethnicity_xpath_map=self.ethnicity_xpath_map, yaml_keys_dict_keys_map=self.deaths_yaml_keys_dict_keys_map, valid_date_string=self.valid_date_string)
            return True
        return False
