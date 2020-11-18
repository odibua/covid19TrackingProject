# content of conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--train_data_bool", action="store_true", help="Boolean that states whether or not to save training data from processed data"
    )
    parser.addoption(
        "--metadata_bool", action="store_true", help="Boolean that states whether or not to scrape and process metadata"
    )
    parser.addoption(
        "--regression_bool", action="store_true", help="Boolean that states whether to perform regression on trained data"
    )
    parser.addoption(
        "--regression_type", default="multilinear", help="Indicate the type of string for a regression"
    )
    parser.addoption(
        "--project_case_bool", action="store_true", help="Boolean that states whether or not to project cases to csvs"
    )
    parser.addoption(
        "--project_death_bool", action="store_true", help="Boolean that states whether or not to project deaths to csvs"
    )
    parser.addoption(
        "--state", default="", help="Indicate whether to run tests only for a particular state"
    )
    parser.addoption(
        "--county", default="", help="Indicate whether to run tests only for a particular state and county"
    )


@pytest.fixture(scope="class")
def project_bools(request):
    request.cls.regression_bool = request.config.getoption("--regression_bool")
    request.cls.regression_type = request.config.getoption("--regression_type")

    request.cls.train_data_bool = request.config.getoption("--train_data_bool")
    request.cls.metadata_bool = request.config.getoption("--metadata_bool")
    request.cls.project_case_bool = request.config.getoption("--project_case_bool")
    request.cls.project_death_bool = request.config.getoption("--project_death_bool")
    request.cls.state_arg = request.config.getoption("--state")
    if len(request.cls.state_arg) > 0:
        request.cls.county_arg = request.config.getoption("--county")
    else:
        request.cls.county_arg = ""
