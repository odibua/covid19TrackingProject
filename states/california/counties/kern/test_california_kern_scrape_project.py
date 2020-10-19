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
from managers import case_parser_manager


@pytest.mark.usefixtures("project_bools")
class TestCaliforniaKernScrapeAndProject(unittest.TestCase):
    def setUp(self):
        self.state_name = 'california'
        self.county_name = 'kern'
        self.state_county_dir = f"states/{self.state_name}/counties/{self.county_name}/raw_data/"

    def test_raw_to_ethnicity_case_manager(self):
        if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
            if len(self.county_arg) == 0 or self.county_arg.lower() == self.county_name.lower():
                if self.project_case_bool:
                    case_parser_manager(state_name=self.state_name, county_name=self.county_name)
