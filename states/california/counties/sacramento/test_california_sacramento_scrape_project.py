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
from managers import add_commit_and_push, scrape_manager, training_data_manager, metadata_manager, case_parser_manager, death_parser_manager


@pytest.mark.usefixtures("project_bools")
class TestCaliforniaSacramentoScrapeAndProject(unittest.TestCase):
    def setUp(self):
        self.state_name = 'california'
        self.county_name = 'sacramento'
        self.state_county_dir = f"states/{self.state_name}/counties/{self.county_name}/raw_data/"

    def test_scrape_manager(self):
        if not self.metadata_bool:
            if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
                if len(self.county_arg) == 0 or self.county_arg.lower() == self.county_name.lower():
                    scrape_manager(state_name=self.state_name, county_name=self.county_name)
                    add_commit_and_push(state_county_dir=self.state_county_dir)

    def test_metadata_manager(self):
        if self.metadata_bool:
            if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
                metadata_manager(state_name=self.state_name, county_name=self.county_name)

    def test_raw_to_ethnicity_case_manager(self):
        if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
            if len(self.county_arg) == 0 or self.county_arg.lower() == self.county_name.lower():
                if self.project_case_bool:
                    case_parser_manager(state_name=self.state_name, county_name=self.county_name)

    def test_raw_to_ethnicity_death_manager(self):
        if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
            if len(self.county_arg) == 0 or self.county_arg.lower() == self.county_name.lower():
                if self.project_death_bool:
                    death_parser_manager(state_name=self.state_name, county_name=self.county_name)

    def test_case_training_data_manager(self):
        if self.train_data_bool:
            if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
                training_data_manager(state_name=self.state_name, county_name=self.county_name, type='cases')

    def test_death_training_data_manager(self):
        if self.train_data_bool:
            if len(self.state_arg) == 0 or self.state_arg.lower() == self.state_name.lower():
                training_data_manager(state_name=self.state_name, county_name=self.county_name, type='deaths')
