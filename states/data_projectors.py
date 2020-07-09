# --------------------------
# Standard Python Imports
# --------------------------
from abc import ABC, abstractmethod
import os

# --------------------------
# Third Party Imports
# --------------------------
import pandas as pd
from typing import Dict, List

# --------------------------
# covid19Tracking Imports
# --------------------------


class EthnicDataProjector(ABC):
    def __init__(self):
        self.state = None
        self.county = None
        self.raw_data_dir = None
        self.ethnicity_cases_dict, self.ethnicity_cases_percentages_dict = {}, {}
        self.ethnicity_deaths_dict, self.ethnicity_deaths_percentages_dict = {}, {}

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

    @abstractmethod
    def process_raw_data_to_cases(self) -> None:
        """
        Process raw page to obtain number of covid cases for each ethnicity and define
        totals and percentages
        """
        return None

    @abstractmethod
    def process_raw_data_to_deaths(self) -> None:
        """
        Process raw page to obtain number of covid deaths for each ethnicity and define
        totals and percentages
        """
        return None

    def get_raw_data_dates_from_dir(self) -> List[str]:
        """
        Get list of raw data dates from {state}/{county}/raw_data
        """
        date_list = os.listdir(self.raw_data_dir)
        return date_list

    def get_raw_data_dates_from_processed_csv(self) -> List[str]:
        """
        Get list of raw data dates from csv files contained in
         {state}/{county}/
        """
        if self.raw_data_dir is None:
            raise ValueError("Raw directory not defined")
        processed_data = pd.read_csv(f"{self.raw_data_dir}/ethnicity.csv")
        date_list = processed_data['date'].tolist()
        return date_list
