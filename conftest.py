# content of conftest.py
import pytest


def pytest_addoption(parser):
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
    request.cls.project_case_bool = request.config.getoption("--project_case_bool")
    request.cls.project_death_bool = request.config.getoption("--project_death_bool")
    request.cls.state_arg = request.config.getoption("--state")
    if len(request.cls.state_arg) > 0:
        request.cls.county_arg = request.config.getoption("--county")
    else:
        request.cls.county_arg = ""
