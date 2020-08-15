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
from managers import scrape_manager


class TestCaliforniaImperialScrapeAndProject(unittest.TestCase):
    def setUp(self):
        self.state_name = 'california'
        self.county_name = 'imperial'

    def test_scrape_manager(self):
        scrape_manager(state_name=self.state_name, county_name=self.county_name)
