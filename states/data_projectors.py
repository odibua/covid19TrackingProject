# --------------------------
# Standard Python Imports
# --------------------------
from abc import ABC, abstractmethod

# --------------------------
# Third Party Imports
# --------------------------
from typing import List

# --------------------------
# covid19Tracking Imports
# --------------------------
import utils


class EthnicDataProjector(ABC):
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
    def process_raw_pages_to_cases(self) -> List[float]:
        """
        Process raw page to obtain number of covid cases for each ethnicity
        """
        return []

    @abstractmethod
    def process_raw_pages_to_deaths(self) -> List[float]:
        """
        Process raw page to obtain number of covid deaths for each ethnicity
        """
        return []
