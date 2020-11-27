#!/usr/bin/env bash
# Run multi-linear regression for filtered races
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --ethnicity_list Black White Asian Hispanic --reg_key discrepancy --regression_type multilinear_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --ethnicity_list Black White Asian Hispanic --reg_key mortality_rate --regression_type multilinear_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --ethnicity_list Black White Asian Hispanic --reg_key detrended_mortality_rate --regression_type multilinear_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --reg_key discrepancy --regression_type multilinear_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --reg_key mortality_rate --regression_type multilinear_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --reg_key detrended_mortality_rate --regression_type multilinear_distance_corr

# Run multi-linear ridge regression for filtered races
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --ethnicity_list Black White Asian Hispanic --reg_key discrepancy --regression_type multilinear_ridge_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --ethnicity_list Black White Asian Hispanic --reg_key mortality_rate --regression_type multilinear_ridge_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --ethnicity_list Black White Asian Hispanic --reg_key detrended_mortality_rate --regression_type multilinear_ridge_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --reg_key discrepancy --regression_type multilinear_ridge_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --reg_key mortality_rate --regression_type multilinear_ridge_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --reg_key detrended_mortality_rate --regression_type multilinear_ridge_distance_corr


# Run multi-linear lasso regression for filtered races
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --ethnicity_list Black White Asian Hispanic --reg_key discrepancy --regression_type multilinear_lasso_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --ethnicity_list Black White Asian Hispanic --reg_key mortality_rate --regression_type multilinear_lasso_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --ethnicity_list Black White Asian Hispanic --reg_key detrended_mortality_rate --regression_type multilinear_lasso_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --reg_key discrepancy --regression_type multilinear_lasso_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --reg_key mortality_rate --regression_type multilinear_lasso_distance_corr
python managers.py --mode perform_deaths_multilinear_regression --all_counties_bool --state california --reg_key detrended_mortality_rate --regression_type multilinear_lasso_distance_corr

