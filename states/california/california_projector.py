# --------------------------
# Standard Python Imports
# --------------------------
import os

# --------------------------
# Third Party Imports
# --------------------------
import pandas as pd
from typing import Dict, List

# --------------------------
# covid19Tracking Imports
# --------------------------
from states.data_projectors import EthnicDataProjector

class CaliforniaEthnicDataProjector(EthnicDataProjector):
    def __init__(self):
        super().__init__(self)
        self.state = 'california'

    @property
    def ethnicities(self) -> List[str]:
        """
        Return list of ethnicities contained in data gathered from pages
        """
        return ["latino", "white", "asian", "black", "multi_race", "american_indian_or_alaska_native", "other"]

    def process_raw_data_to_cases(self, date: str) -> Dict[str, int]:
        """
        Process raw page to obtain number of covid cases for each ethnicity
        """

        return {}

    def process_raw_data_to_deaths(self, date: str) -> Dict[str, int]:
        """
        Process raw page to obtain number of covid deaths for each ethnicity
        """
        return {}
