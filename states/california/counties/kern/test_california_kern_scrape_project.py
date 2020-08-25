# --------------------------
# Standard Python Imports
# --------------------------
import pytest
import unittest

# --------------------------
# Third Party Imports
# --------------------------

# --------------------------
# covid19Tracking Imports
# --------------------------
from managers import raw_to_ethnicity_case_csv_manager


@pytest.mark.usefixtures("project_bools")
class TestCaliforniaKernScrapeAndProject(unittest.TestCase):
    def setUp(self):
        self.state_name = 'california'
        self.county_name = 'kern'
        self.state_county_dir = f"states/{self.state_name}/counties/{self.county_name}/raw_data/"

    def test_raw_to_ethnicity_case_manager(self):
        if self.project_case_bool:
            raw_to_ethnicity_case_csv_manager(state_name=self.state_name)
