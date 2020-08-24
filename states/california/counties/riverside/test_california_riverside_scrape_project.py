# --------------------------
# Standard Python Imports
# --------------------------
import unittest

# --------------------------
# Third Party Imports
# --------------------------

# --------------------------
# covid19Tracking Imports
# --------------------------
from managers import add_commit_and_push, scrape_manager, raw_to_ethnicity_case_csv_manager, raw_to_ethnicity_death_csv_manager


class TestCaliforniaRiversideScrapeAndProject(unittest.TestCase):
    def setUp(self):
        self.state_name = 'california'
        self.county_name = 'riverside'
        self.state_county_dir = f"states/{self.state_name}/counties/{self.county_name}/raw_data/"

    def test_scrape_manager(self):
        scrape_manager(state_name=self.state_name, county_name=self.county_name)
        add_commit_and_push(state_county_dir=self.state_county_dir)

    def test_raw_to_ethnicity_case_manager(self):
        raw_to_ethnicity_case_csv_manager(state_name=self.state_name, county_name=self.county_name)
