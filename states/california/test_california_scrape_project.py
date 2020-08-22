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
from managers import scrape_manager, add_commit_and_push, raw_to_ethnicity_case_csv_manager, raw_to_ethnicity_death_csv_manager


class TestCaliforniaScrapeAndProject(unittest.TestCase):
    def setUp(self):
        self.state_name = 'california'
        self.state_county_dir = 'states/california/raw_data/'

    def test_scrape_manager(self):
        scrape_manager(state_name=self.state_name)

    def test_raw_to_ethnicity_case_manager(self):
        raw_to_ethnicity_case_csv_manager(state_name=self.state_name)

    def test_raw_to_ethnicity_death_manager(self):
        raw_to_ethnicity_death_csv_manager(state_name=self.state_name)
        add_commit_and_push(state_county_dir=self.state_county_dir)
