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
from managers import scrape_manager, add_commit_and_push, case_parser_manager, death_parser_manager


@pytest.mark.usefixtures("project_bools")
class TestCaliforniaScrapeAndProject(unittest.TestCase):
    def setUp(self):
        self.state_name = ''
        self.state_county_dir = 'states/raw_data/'

    def test_scrape_manager(self):
        if not self.metadata_bool:
            if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
                scrape_manager(state_name=self.state_name)
                add_commit_and_push(state_county_dir=self.state_county_dir)

    def test_raw_to_ethnicity_case_manager(self):
        if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
            if self.project_case_bool:
                case_parser_manager(state_name=self.state_name)

    def test_raw_to_ethnicity_death_manager(self):
        if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
            if self.project_death_bool:
                death_parser_manager(state_name=self.state_name)
