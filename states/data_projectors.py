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

    @abstractmethod
    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return []

    @abstractmethod
    @property
    def ethnicity_percentages(self) -> List[float]:
        """
        Return list of percentage of ethnicities contained in an area
        """
        return []

    @abstractmethod
    def process_raw_pages_to_cases(self) -> Dict[str, int]:
        """
        Process raw page to obtain number of covid cases for each ethnicity
        """
        return {}

    @abstractmethod
    def process_raw_pages_to_deaths(self) -> Dict[str, int]:
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
