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


class SonomaEthnicDataProjector(CaliforniaEthnicDataProjector):
    def __init__(self, state: str, county: str, date_string: str):
        super().__init__(state=state, county=county, date_string=date_string)
        logging.info("Initialize Sonoma raw and config file strings")
        raw_data_dir = os.path.join("states", state, 'counties', county, "raw_data")
        raw_data_file = f"{raw_data_dir}/{date_string}/sonoma_all.html"
        configs_dir = os.path.join("states", state, 'counties', county, "configs")
        config_file_string = f"{configs_dir}/sonoma_all_html_parse.yaml"

        logging.info("Load parsing config")
        html_parser_config = self.load_yaml(config_file_string)

        logging.info("Get and sort html parsing dates")
        html_parser_date_strings = list(html_parser_config["DATES"].keys())
        html_parser_dates = self.get_sorted_dates_from_strings(date_string_list=html_parser_date_strings)

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
            self.cases_raw_bool = True
        except BaseException:
            pass

        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {
            'HISPANIC_CASES': 'hispanic',
            'WHITE_CASES': 'white',
            'ASIAN_CASES': 'asian',
            'ASIAN_PACIFIC_ISLANDER_CASES': 'asian_pacific_islander',
            'NON_HISPANIC_CASES': 'non_hispanic',
            'BLACK_CASES': 'black',
            'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES': 'Native Hawaiian/Pacific Islander',
            'AMERICAN_INDIAN_ALASKA_NATIVE_CASES': 'American Indian/Alaska Native',
        }
        self.deaths_yaml_keys_dict_keys_map = None

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['hispanic', "white", "asian_pacific_islander", "black", "non_hispanic",
                "Native Hawaiian/Pacific Islander", 'American Indian/Alaska Native', 'asian']

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in Sonoma County.

        Obtained from here: census.gov/quickfacts/fact/table/sonomacountycalifornia,CA/PST045219
        """
        return {'hispanic': 0.273, 'white': 0.629, 'black': 0.021, 'asian': 0.046,
                'asian_pacific_islander': 0.05, 'non_hispanic': 0.062, "Native Hawaiian/Pacific Islander": 0.004,
                'American Indian/Alaska Native': 0.022}

    @property
    def map_acs_to_region_ethnicities(self) -> Dict[str, List[str]]:
        """
        Return dictionary that maps ACS ethnicities to region ethnicities defined by covid
        """
        return {'hispanic': ['Hispanic'], 'black': ['Black'], 'asian': ['Asian'], 'white': ['White'], 'asian_pacific_islander': ['Asian', 'Native Hawaiian/Pacific Islander'], 'Black': ['black'],
                'non_hispanic': ['Black', 'American Indian/Alaska Native', 'Multi-Race', 'Native Hawaiian/Pacific Islander'], 'American Indian/Alaska Native': ['American Indian/Alaska Native'],
                'Native Hawaiian/Pacific Islander': ['Native Hawaiian/Pacific Islander']}

    @property
    def total_population(self) -> int:
        return 494336

    @property
    def acs_ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains total of each ethnicity population in Sonoma County.

        Obtained from here: census.gov/quickfacts/fact/table/sonomacountycalifornia
        """
        return {'Hispanic': 0.273, 'White': 0.629, 'Asian': 0.046, 'Black': 0.021, 'Multi-Race': 0.04,
                'American Indian/Alaska Native': 0.022, 'Native Hawaiian/Pacific Islander': 0.004}
