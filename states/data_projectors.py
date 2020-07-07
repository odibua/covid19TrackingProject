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
        self.ethnicity_cases_list, self.ethnicity_case_percentages_list = [], []
        self.ethnicity_deaths_list, self.ethnicity_deaths_percentages_list = [], []


    @abstractmethod
    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return []

    @property
    def ethnicity_cases_percentages(self) -> List[float]:
        """
        Return list of case percentages of ethnicities contained in an area
        """
        return self.ethnicity_case_percentages_list

    @property
    def ethnicity_cases(self) -> List[float]:
        """
        Return list of cases of ethnicities contained in an area
        """
        return self.ethnicity_cases_list

    @property
    def ethnicity_deaths_percentages(self) -> List[float]:
        """
        Return list of death percentages of ethnicities contained in an area
        """
        return self.ethnicity_deaths_percentages_list

    @property
    def ethnicity_deaths(self) -> List[float]:
        """
        Return list of cases of ethnicities contained in an area
        """
        return self.ethnicity_deaths_list

    @abstractmethod
    def process_raw_data_to_cases(self) -> Dict[str, int]:
        """
        Process raw page to obtain number of covid cases for each ethnicity and note
        totals and percentages
        """
        return {}

    @abstractmethod
    def process_raw_data_to_deaths(self) -> Dict[str, int]:
        """
        Process raw page to obtain number of covid deaths for each ethnicity
        """
        return {}

    def get_raw_data_dates_from_dir(self) -> List[str]:
        """
        Get list of raw data dates from {state}/{county}/raw_data
        """
        if self.county is not None:
            raw_data_dir = f"states/{self.state}/{self.county}/raw_data"
        else:
            raw_data_dir = f"states/{self.state}/raw_data"
        date_list = os.listdir(raw_data_dir)
        return date_list

    def get_raw_data_dates_from_processed_csv(self) -> List[str]:
        """
        Get list of raw data dates from csv files contained in
         {state}/{county}/
        """
        if self.county is not None:
            raw_data_dir = f"states/{self.state}/{self.county}/raw_data"
        else:
            raw_data_dir = f"states/{self.state}/raw_data"

        processed_data = pd.read_csv(f"{raw_data_dir}/ethnicity.csv")
        date_list = processed_data['date'].tolist()
        return date_list
