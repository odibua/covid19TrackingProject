#!/usr/bin/env bash
# Run multi-linear regression for filtered races
python managers.py --mode perform_cases_multilinear_regression --state california --county alameda imperial losangeles riverside sacramento sanfrancisco santaclara sonoma --ethnicity_list Black White Asian Hispanic --reg_key discrepancy
python managers.py --mode perform_deaths_multilinear_regression --state california --county alameda imperial losangeles riverside sacramento sanfrancisco santaclara sonoma --ethnicity_list Black White Asian Hispanic --reg_key discrepancy

## Run multi-linear ridge regression for filtered races
python managers.py --mode perform_cases_multilinear_regression --state california --county alameda imperial losangeles riverside sacramento sanfrancisco santaclara sonoma --ethnicity_list Black White Asian Hispanic --reg_key discrepancy --regression_type multilinear_ridge
python managers.py --mode perform_deaths_multilinear_regression --state california --county alameda imperial losangeles riverside sacramento sanfrancisco santaclara sonoma --ethnicity_list Black White Asian Hispanic --reg_key discrepancy --regression_type multilinear_ridge

## Run multi-linear lasso regression for filtered races
python managers.py --mode perform_cases_multilinear_regression --state california --county alameda imperial losangeles riverside sacramento sanfrancisco santaclara sonoma --ethnicity_list Black White Asian Hispanic --reg_key discrepancy --regression_type multilinear_lasso
python managers.py --mode perform_deaths_multilinear_regression --state california --county alameda imperial losangeles riverside sacramento sanfrancisco santaclara sonoma --ethnicity_list Black White Asian Hispanic --reg_key discrepancy --regression_type multilinear_lasso
