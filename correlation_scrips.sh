# Run correlations for cases and deaths while filtering for ethnicity
python managers.py --mode perform_case_spearman_corr --state california --all_counties_bool --corr_key mortality_rate --ethnicity_list Black White Asian Hispanic
python managers.py --mode perform_case_spearman_corr --state california --all_counties_bool --corr_key detrended_mortality_rate --ethnicity_list Black White Asian Hispanic
python managers.py --mode perform_case_spearman_corr --state california --all_counties_bool --corr_key discrepancy --ethnicity_list Black White Asian Hispanic
python managers.py --mode perform_case_spearman_corr --state california --all_counties_bool --corr_key covid_perc --ethnicity_list Black White Asian Hispanic

python managers.py --mode perform_death_spearman_corr --state california --all_counties_bool --corr_key mortality_rate --ethnicity_list Black White Asian Hispanic
python managers.py --mode perform_death_spearman_corr --state california --all_counties_bool --corr_key detrended_mortality_rate --ethnicity_list Black White Asian Hispanic
python managers.py --mode perform_death_spearman_corr --state california --all_counties_bool --corr_key discrepancy --ethnicity_list Black White Asian Hispanic
python managers.py --mode perform_death_spearman_corr --state california --all_counties_bool --corr_key covid_perc --ethnicity_list Black White Asian Hispanic

# Run correlations for cases and deaths without filtering for ethnicity
python managers.py --mode perform_case_spearman_corr --state california --all_counties_bool --corr_key mortality_rate
python managers.py --mode perform_case_spearman_corr --state california --all_counties_bool --corr_key detrended_mortality_rate
python managers.py --mode perform_case_spearman_corr --state california --all_counties_bool --corr_key discrepancy
python managers.py --mode perform_case_spearman_corr --state california --all_counties_bool --corr_key covid_perc

python managers.py --mode perform_death_spearman_corr --state california --all_counties_bool --corr_key mortality_rate
python managers.py --mode perform_death_spearman_corr --state california --all_counties_bool --corr_key detrended_mortality_rate
python managers.py --mode perform_death_spearman_corr --state california --all_counties_bool --corr_key discrepancy
python managers.py --mode perform_death_spearman_corr --state california --all_counties_bool --corr_key covid_perc