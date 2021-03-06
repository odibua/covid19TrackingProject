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
from states.california.california_projector import CaliforniaEthnicDataProjector
from states import utils_state_lib


class LosAngelesEthnicDataProjector(CaliforniaEthnicDataProjector):
    def __init__(self, state: str, county: str, date_string: str):
        super().__init__(state=state, county=county, date_string=date_string)
        self.cases_raw_bool, self.deaths_raw_bool = False, False
        logging.info("Initialize Los Angeles raw and config file strings")
        raw_data_dir = os.path.join("states", state, 'counties', county, "raw_data")
        raw_data_file = f"{raw_data_dir}/{date_string}/losangeles_all.html"
        configs_dir = os.path.join("states", state, 'counties', county, "configs")
        config_file_string = f"{configs_dir}/losangeles_all_html_parse.yaml"

        logging.info("Load parsing config")
        html_parser_config_file = open(config_file_string)
        html_parser_config = yaml.safe_load(html_parser_config_file)

        logging.info("Get and sort html parsing dates")
        html_parser_date_strings = html_parser_config["DATES"].keys()
        html_parser_dates = sorted([datetime.strptime(date_string, '%Y-%m-%d')
                                    for date_string in html_parser_date_strings])

        logging.info("Obtain valid map of ethnicities to xpath containing cases or deaths")
        self.date_string = date_string
        self.valid_date_string = utils_state_lib.get_valid_date_string(
            date_list=html_parser_dates, date_string=date_string)
        self.ethnicity_xpath_map = html_parser_config['DATES'][self.valid_date_string]
        logging.info("Load raw html data and convert it to lxml")
        try:
            raw_data_file_object = open(raw_data_file, 'r')
            raw_data_file_html = raw_data_file_object.read()
            soup = bs4.BeautifulSoup(raw_data_file_html, 'html5lib')
            raw_data_file_html = soup.prettify()
            self.raw_data_lxml = etree.HTML(raw_data_file_html)
            if len(self.raw_data_lxml.text.strip(' ')) == 1:
                self.raw_data_lxml = soup
            self.cases_raw_bool, self.deaths_raw_bool = True, True
        except BaseException:
            pass

        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {'HISPANIC_CASES': 'Hispanic', 'WHITE_CASES': 'White', 'ASIAN_CASES': 'Asian',
                                              'BLACK_CASES': 'Black',
                                              'AMERICAN_INDIAN_ALASKA_NATIVE_CASES': 'American Indian/Alaska Native', 'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES': 'Native Hawaiian/Pacific Islander',
                                              'OTHER_CASES': 'Other'}
        self.deaths_yaml_keys_dict_keys_map = {'HISPANIC_DEATHS': 'Hispanic', 'WHITE_DEATHS': 'White', 'ASIAN_DEATHS': 'Asian',
                                               'BLACK_DEATHS': 'Black',
                                               'AMERICAN_INDIAN_ALASKA_NATIVE_DEATHS': 'American Indian/Alaska Native', 'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_DEATHS': 'Native Hawaiian/Pacific Islander',
                                               'OTHER_DEATHS': 'Other'}

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['Hispanic', "White", "Asian", "Black", "American Indian/Alaska Native",
                'Native Hawaiian/Pacific Islander', "Other"]

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in Los Angeles

        Obtained from here: https://www.census.gov/quickfacts/losangelescountycalifornia
        """
        return {'Hispanic': 0.486, "White": 0.261, 'Asian': 0.154, 'Black': 0.09,
                "American Indian/Alaska Native": 0.014, 'Native Hawaiian/Pacific Islander': 0.004, 'Other': 0.031}

    @property
    def map_acs_to_region_ethnicities(self) -> Dict[str, List[str]]:
        """
        Return dictionary that maps ACS ethnicities to region ethnicities defined by covid
        """
        return {'Hispanic': ['Hispanic'], 'White': ['White'], 'Asian': ['Asian'], 'Black': ['Black'],
                'American Indian/Alaska Native': ['American Indian/Alaska Native'],
                'Native Hawaiian/Pacific Islander': ['Native Hawaiian/Pacific Islander']}

    @property
    def total_population(self) -> int:
        return 10039107

    @property
    def acs_ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains totalof each ethnicity population in Los Angeles

        Obtained from here: https://www.census.gov/quickfacts/losangelescountycalifornia
        """
        return {'Hispanic': 0.486, 'White': 0.261, 'Asian': 0.154, 'Black': 0.09, 'Multi-Race': 0.031,
                'American Indian/Alaska Native': 0.014, 'Native Hawaiian/Pacific Islander': 0.004}
