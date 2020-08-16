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
from managers import scrape_manager, raw_to_ethnicity_case_csv_manager, raw_to_ethnicity_death_csv_manager


class TestCaliforniaKernScrapeAndProject(unittest.TestCase):
    def setUp(self):
        self.state_name = 'california'
        self.county_name = 'kern'

    def test_scrape_manager(self):
        scrape_manager(state_name=self.state_name, county_name=self.county_name)

    def test_raw_to_ethnicity_case_manager(self):
        raw_to_ethnicity_case_csv_manager(state_name=self.state_name, county_name=self.county_name)

    def test_raw_to_ethnicity_death_manager(self):
        raw_to_ethnicity_death_csv_manager(state_name=self.state_name, county_name=self.county_name)
