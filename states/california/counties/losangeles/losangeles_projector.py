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
from states import utils


class LosAngelesEthnicDataProjector(CaliforniaEthnicDataProjector):
    def __init__(self, state: str, county: str, date_string: str):
        super().__init__(state=state, county=county, date_string=date_string)
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
        self.valid_date_string = utils.get_valid_date_string(date_list=html_parser_dates, date_string=date_string)
        self.ethnicity_xpath_map = html_parser_config['DATES'][self.valid_date_string]
        logging.info("Load raw html data and convert it to lxml")
        raw_data_file_object = open(raw_data_file, 'r')
        raw_data_file_html = raw_data_file_object.read()
        soup = bs4.BeautifulSoup(raw_data_file_html, 'html5lib')
        raw_data_file_html = soup.prettify()
        self.raw_data_lxml = etree.HTML(raw_data_file_html)

        logging.info("Define yaml keys to dictionary maps for cases and deaths")
        self.cases_yaml_keys_dict_keys_map = {'HISPANIC_CASES': 'hispanic', 'WHITE_CASES': 'white', 'ASIAN_CASES': 'asian',
                                              'BLACK_CASES': 'black',
                                              'AMERICAN_INDIAN_OR_ALASKA_NATIVE_CASES': 'american_indian_alaska_native', 'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_CASES': 'native_hawaiian_pacific_islander',
                                              'OTHER_CASES': 'other'}
        self.deaths_yaml_keys_dict_keys_map = {'HISPANIC_DEATHS': 'hispanic', 'WHITE_DEATHS': 'white', 'ASIAN_DEATHS': 'asian',
                                               'BLACK_DEATHS': 'black',
                                               'AMERICAN_INDIAN_OR_ALASKA_NATIVE_DEATHS': 'american_indian_alaska_native', 'NATIVE_HAWAIIAN_PACIFIC_ISLANDER_DEATHS': 'native_hawaiian_pacific_islander',
                                               'OTHER_DEATHS': 'other'}

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ['hispanic', "white", "asian", "black", "american_indian_alaska_native",
                "native_hawaiian_pacific_islander", "other"]

    @property
    def ethnicity_demographics(self) -> Dict[str, float]:
        """
        Return dictionary that contains percentage of each ethnicity population in california
        """
        return {'hispanic': 0.475, 'white': 0.524, 'asian': 0.138, 'black': 0.086,
                'american_indian_alaska_native': 0.005, 'native_hawaiian_pacific_islander': 0.003, 'other': 0.245}
