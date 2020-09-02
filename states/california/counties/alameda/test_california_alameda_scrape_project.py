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
from managers import add_commit_and_push, scrape_manager, raw_to_ethnicity_case_csv_manager, raw_to_ethnicity_death_csv_manager


@pytest.mark.usefixtures("project_bools")
class TestCaliforniaAlamedaScrapeAndProject(unittest.TestCase):
    def setUp(self):
        self.state_name = 'california'
        self.county_name = 'alameda'
        self.state_county_dir = f"states/{self.state_name}/counties/{self.county_name}/raw_data/"

    def test_scrape_manager(self):
        if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
            if len(self.county_arg) == 0 or self.county_arg.lower() == self.county_name.lower():
                scrape_manager(state_name=self.state_name, county_name=self.county_name)
                add_commit_and_push(state_county_dir=self.state_county_dir)

    def test_raw_to_ethnicity_case_manager(self):
        if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
            if len(self.county_arg) == 0 or self.county_arg.lower() == self.county_name.lower():
                if self.project_case_bool:
                    raw_to_ethnicity_case_csv_manager(state_name=self.state_name, county_name=self.county_name)

    def test_raw_to_ethnicity_death_manager(self):
        if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
            if len(self.county_arg) == 0 or self.county_arg.lower() == self.county_name.lower():
                if self.project_death_bool:
                    raw_to_ethnicity_death_csv_manager(state_name=self.state_name, county_name=self.county_name)
