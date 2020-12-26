# --------------------------
# Standard Python Imports
# --------------------------
import argparse
import copy
import datetime
import logging
import os
from os import path
import subprocess as cmd
from typing import Dict, List, Tuple, Union

# --------------------------
# Third Party Imports
# --------------------------
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# --------------------------
# covid19Tracking Imports
# --------------------------
import correlation_utils
import regression_utils
import utils_lib


class RegDefinitions:
    reg_list = ['multilinear', 'multilinear_ridge', 'multilinear_lasso']
    spearman_reg_list = ['multilinear_spearman', 'multilinear_ridge_spearman', 'multilinear_lasso_spearman']
    dist_reg_list = ['multilinear_distance_corr', 'multilinear_ridge_distance_corr', 'multilinear_lasso_distance_corr']

    multilinear_list = ['multilinear', 'multilinear_spearman', 'multilinear_distance_corr']
    multilinear_ridge_list = ['multilinear_ridge', 'multilinear_ridge_spearman', 'multilinear_ridge_distance_corr']
    multilinear_lasso_list = ['multilinear_lasso', 'multilinear_lasso_spearman', 'multilinear_lasso_distance_corr']


def filter_empty_list(dict: Dict[str, List[Union[float, str]]]) -> None:
    copy_dict = copy.deepcopy(dict)
    for key in copy_dict.keys():
        if len(copy_dict[key]) == 0:
            del dict[key]


def get_metadata_filter(type: str, state_name: str, county_name: str, regression_type: str,
                        reg_key: str, ethnicity_filter_list: List[str]) -> List[str]:
    metadata_filter = []
    ethnicity_filter_list = [ethnicity.lower() for ethnicity in ethnicity_filter_list]
    if regression_type in RegDefinitions.reg_list:
        return metadata_filter

    if regression_type in RegDefinitions.spearman_reg_list:
        corr_type = 'spearman'
    elif regression_type in RegDefinitions.dist_reg_list:
        corr_type = 'distance_corr'
    else:
        raise ValueError(f'{regression_type} not valid')

    correlation_results_path = path.join('states', state_name, 'correlation_results', corr_type)
    if len(ethnicity_filter_list) == 0:
        results_file = f'{type}_{reg_key}_{corr_type}_corr_results.csv'
    else:
        results_file = f'{type}_{reg_key}_{corr_type}_corr_results'
        for ethnicity in ethnicity_filter_list:
            results_file = f'{results_file}_{ethnicity}'
        results_file = f'{results_file}.csv'
    correlation_results_file = path.join(correlation_results_path, results_file)
    correlation_df = pd.read_csv(correlation_results_file)

    state_bool_list = (correlation_df['state'] == state_name).tolist()
    if county_name is None:
        county_bool_list = correlation_df['county'].isna().tolist()
    else:
        county_bool_list = (correlation_df['county'] == county_name).tolist()
    keep_bool = state_bool_list and county_bool_list
    correlation_df = correlation_df[keep_bool]
    metadata_filter = correlation_df['X'].tolist()

    return metadata_filter


def get_responses_from_config_files_in_dir(config_dir: str) -> Tuple[List[str], List[str], str]:
    """
    Wrapper that lists config files in a directory, and calls a function that gets requests from them

    Arguments:
        config_dir: Config directory

    Returns:
        response_list: Get list of responses texts from requests
        response_names: List of names containing the location and what is being looked for based on the associated config file
        request_type: Return request type
    """
    config_files = os.listdir(config_dir)
    config_files = [config_file for config_file in config_files if config_file.endswith('.yaml')]
    response_list, response_names, request_type = utils_lib.get_yaml_responses(
        config_dir=config_dir, config_file_list=config_files)

    return response_list, response_names, request_type


def scrape_manager(state_name: str, county_name: str = None) -> None:
    """
    Scraping manager that uses the config file associated with a particular state and county to collect raw data
    about COVID cases and deaths. It can be adopted for other purposes.

    Arguments:
        state_name: State to scrape from
        county_name: County to scrape from. Defaults to 0

    Returns:
        None
    """
    logging.info(f"Create raw data and config directory for state: {state_name} county: {county_name}")
    if county_name is None:
        state_config_path = path.join('states', state_name, 'configs')
        raw_data_dir = path.join('states', state_name, 'raw_data')
    else:
        state_config_path = path.join('states', state_name, 'counties', county_name, 'configs')
        raw_data_dir = path.join('states', state_name, 'counties', county_name, 'raw_data')

    logging.info(f"Get responses from text file")
    state_response_list, state_data_type_names, request_type = get_responses_from_config_files_in_dir(
        config_dir=state_config_path)

    if not path.isdir(raw_data_dir):
        os.makedirs(raw_data_dir)
    utils_lib.save_raw_data(
        save_dir=raw_data_dir,
        response_list=state_response_list,
        data_type_names=state_data_type_names,
        request_type=request_type)


def case_parser_manager(state_name: str, county_name: str = None) -> None:
    state_csv_dir = os.path.join('states', state_name, 'csvs')
    if not os.path.isdir(state_csv_dir):
        os.makedirs(state_csv_dir)
    if county_name is None:
        state_county_dir = os.path.join('states', state_name)
        cases_csv_filename = f"{state_name}_ethnicity_cases.csv"
    else:
        state_county_dir = path.join('states', state_name, 'counties', county_name)
        cases_csv_filename = f"{state_name}_{county_name}_ethnicity_cases.csv"

    case_msg = utils_lib.run_ethnicity_to_case_csv(
        state_csv_dir=state_csv_dir, state_county_dir=state_county_dir, state=state_name, county=county_name,
        cases_csv_filename=cases_csv_filename)

    try:
        add_commit_and_push(state_county_dir=state_csv_dir)
    except BaseException:
        pass
    if case_msg is None:
        return
    if len(case_msg) > 0:
        if 'WARNING' in case_msg:
            logging.warning(f"{case_msg}")
        else:
            raise ValueError(f"{case_msg}")


def death_parser_manager(state_name: str, county_name: str = None) -> None:
    state_csv_dir = os.path.join('states', state_name, 'csvs')
    if not os.path.isdir(state_csv_dir):
        os.makedirs(state_csv_dir)
    if county_name is None:
        state_county_dir = os.path.join('states', state_name)
        deaths_csv_filename = f"{state_name}_ethnicity_deaths.csv"
    else:
        state_county_dir = path.join('states', state_name, 'counties', county_name)
        deaths_csv_filename = f"{state_name}_{county_name}_ethnicity_deaths.csv"

    death_msg = utils_lib.run_ethnicity_to_death_csv(
        state_csv_dir=state_csv_dir, state_county_dir=state_county_dir, state=state_name, county=county_name,
        deaths_csv_filename=deaths_csv_filename)

    try:
        add_commit_and_push(state_county_dir=state_csv_dir)
    except BaseException:
        pass
    if death_msg is None:
        return
    if len(death_msg) > 0:
        if 'WARNING' in death_msg:
            logging.warning(f"{death_msg}")
        else:
            raise ValueError(f"{death_msg}")


def metadata_manager(state_name: str, county_name: str = None) -> None:
    """
    Scraping manager that uses the config file associated with a particular state and county to collect raw data
    about COVID cases and deaths. It can be adopted for other purposes.

    Arguments:
        state_name: State to scrape from
        county_name: County to scrape from. Defaults to 0

    Returns:
        None
    """
    logging.info(f"Create raw data and config directory for state: {state_name} county: {county_name}")
    if county_name is None:
        state_config_path = path.join('states', state_name, 'configs')
        state_county_dir = path.join('states', state_name)
    else:
        state_config_path = path.join('states', state_name, 'counties', county_name, 'configs')
        state_county_dir = path.join('states', state_name, 'counties', county_name)

    # Get dataframe of raw metadata for a state and/or county and save the data to
    # a csv file
    metadata_df = utils_lib.get_raw_metadata_from_config_files(config_dir=state_config_path)
    utils_lib.save_data(
        state_name=state_name,
        county_name=county_name,
        data_df=metadata_df,
        data_dir='meta_data_csv',
        data_suffix='metadata')

    # Process raw metadata and save processed medatada and save data to a csv file
    processed_metadata_df = utils_lib.process_raw_metadata(
        raw_metadata_df=metadata_df,
        config_dir=state_config_path,
        state=state_name,
        county=county_name,
        state_county_dir=state_county_dir)
    utils_lib.save_data(
        state_name=state_name,
        county_name=county_name,
        data_df=processed_metadata_df,
        data_dir='processed_meta_data_csv',
        data_suffix='processed_metadata')

    # Aggregate processed metadata based on mapping of regionally defined ethnicities
    # to ACS ethnicities.
    aggregated_processed_metadata_df = utils_lib.aggregate_processed_raw_metadata(
        processed_metadata_df=processed_metadata_df,
        state=state_name,
        county=county_name,
        state_county_dir=state_county_dir)
    utils_lib.save_data(
        state_name=state_name,
        county_name=county_name,
        data_df=aggregated_processed_metadata_df,
        data_dir='aggregated_processed_meta_data_csv',
        data_suffix='aggregated_processed_metadata')


def correlation_manager(state_name: str, type: str, key: str, corr_type: str,
                        ethnicity_filter_list: List = [], county_name: str = None) -> None:
    # Define path and file for training data
    training_csv_path = path.join('states', state_name, 'training_data_csvs')
    correlation_results_path = path.join('states', state_name, 'correlation_results', corr_type)

    if county_name is None:
        training_file = f'{state_name}_training_{type}.csv'
    else:
        training_file = f'{state_name}_{county_name}_training_{type}.csv'

    training_data_df = pd.read_csv(path.join(training_csv_path, training_file), index_col=0)

    # Filter to specific ethnicities
    if len(ethnicity_filter_list) > 0:
        ethnicity_filter_list = [ethnicity.lower() for ethnicity in ethnicity_filter_list]
        ethnicities = training_data_df['ethnicity'].str.lower().tolist()
        ethnicity_bool = [True if ethnicity.lower() in ethnicity_filter_list else False for ethnicity in ethnicities]
        training_data_df = training_data_df[ethnicity_bool]
    Y = np.array(training_data_df[key].tolist())

    keys_to_filter = [
        'time',
        'covid_perc',
        'dem_perc',
        'mortality_rate',
        'detrended_mortality_rate',
        'discrepancy',
        'y_pred',
        'ethnicity']
    corr_keys = [feat_key for feat_key in training_data_df.keys() if feat_key not in keys_to_filter]

    corr_dict = {'corr': [], 'Y': [], 'X': [], 'p_val': [], 'state': [], 'county': [], 'n': []}

    for corr_key in corr_keys:
        X = np.array(training_data_df[corr_key].tolist())
        if corr_type == 'spearman':
            correlation_utils.populate_spearman_corr_dict(corr_dict=corr_dict, y_key=key, x_key=corr_key, state=state_name, county=county_name, n=len(Y),
                                                          X=X, Y=Y)
        elif corr_type == 'distance_corr':
            correlation_utils.populate_dist_corr_dict(corr_dict=corr_dict, y_key=key, x_key=corr_key,
                                                      state=state_name, county=county_name, n=len(Y),
                                                      X=X, Y=Y)
    filter_empty_list(dict=corr_dict)
    corr_df = pd.DataFrame(corr_dict)

    if len(corr_df) > 0:
        if len(ethnicity_filter_list) == 0:
            results_file = f'{type}_{key}_{corr_type}_corr_results.csv'
        else:
            results_file = f'{type}_{key}_{corr_type}_corr_results'
            for ethnicity in ethnicity_filter_list:
                results_file = f'{results_file}_{ethnicity}'
            results_file = f'{results_file}.csv'
        correlation_results_file = path.join(correlation_results_path, results_file)
        if not os.path.exists(correlation_results_path):
            os.makedirs(correlation_results_path)

        if not os.path.isfile(correlation_results_file):
            corr_df.to_csv(correlation_results_file, index=False)
        else:
            corr_df.to_csv(correlation_results_file, header=False, mode='a', index=False)


def training_data_manager(state_name: str, type: str, county_name: str = None) -> None:
    logging.info(f"Create raw data and config directory for state: {state_name} county: {county_name}")
    # Define paths and files containing region covid case rates data
    # and metadata
    metadata_path = path.join('states', state_name, 'aggregated_processed_meta_data_csv')
    csv_path = path.join('states', state_name, 'csvs')
    training_csv_path = path.join('states', state_name, 'training_data_csvs')
    if county_name is None:
        metadata_file = f'{state_name}_aggregated_processed_metadata.csv'
        csv_file = f'{state_name}_ethnicity_{type}.csv'
        training_file = f'{state_name}_training_{type}.csv'
    else:
        metadata_file = f'{state_name}_{county_name}_aggregated_processed_metadata.csv'
        csv_file = f'{state_name}_{county_name}_ethnicity_{type}.csv'
        training_file = f'{state_name}_{county_name}_training_{type}.csv'

    # Get earliest date for case files
    csv_file_list = os.listdir(csv_path)
    csv_file_list = [path.join(csv_path, csv_file) for csv_file in csv_file_list if 'case' in csv_file]

    earliest_date = utils_lib.get_earliest_date_string(csv_file_list=csv_file_list)

    # Get rate columns from csv file
    csv_df = pd.read_csv(path.join(csv_path, csv_file))
    columns = csv_df.keys()
    rate_columns = [column for column in columns if 'rates' in column or column == 'date']
    rate_df = csv_df[rate_columns]

    # Get covid percentages
    covid_perc_columns = [column for column in columns if 'covidperc' in column or column == 'date']
    covid_perc_df = csv_df[covid_perc_columns]

    # Get demographic percentages
    dem_perc_columns = [column for column in columns if 'demperc' in column or column == 'date']
    dem_perc_df = csv_df[dem_perc_columns]

    # Get discrepancy
    discrep_columns = [column for column in columns if 'discrepancy' in column or column == 'date']
    discrep_df = csv_df[discrep_columns]

    # Add time column that is based on days
    date_df = pd.to_datetime(rate_df['date'])
    rate_df['time'] = (date_df - earliest_date).dt.days


    # Load aggregated metadata data frame
    aggregated_processed_metadata_df = pd.read_csv(path.join(metadata_path, metadata_file), index_col=0)

    # Get columns that have values that unique values for mortality rates
    # and store them in a dictionary along with relevant regional features
    training_data_dict = {
        'mortality_rate': [],
        'time': [],
        'covid_perc': [],
        'dem_perc': [],
        'discrepancy': [],
        'ethnicity': []}
    for metadata_name in aggregated_processed_metadata_df.keys():
        training_data_dict[metadata_name] = []

    for column in rate_df.keys():
        ethnicity = column.split('_rates')[0]
        if column != 'date' and column != 'time' and ethnicity.lower() != 'other':
            rate_column_df = rate_df[column]
            if sum(pd.isna(rate_column_df)) == len(rate_column_df):
                continue
            demperc_column_df = dem_perc_df[f'{ethnicity}_demperc']
            covidperc_column_df = covid_perc_df[f'{ethnicity}_covidperc']
            discrep_column_df = discrep_df[f'{ethnicity}_discrepancy']

            time_df = rate_df['time'][rate_column_df.notnull()]
            demperc_column_df = demperc_column_df[rate_column_df.notnull()]
            covidperc_column_df = covidperc_column_df[rate_column_df.notnull()]
            discrep_column_df = discrep_column_df[rate_column_df.notnull()]
            rate_column_df = rate_column_df[rate_column_df.notnull()]

            delta_df = rate_column_df[1:].subtract(rate_column_df[0:-1].tolist())

            change_bool = (delta_df.abs() > 0).tolist()
            change_bool = [True] + change_bool
            rate_column_df = rate_column_df[change_bool]
            demperc_column_df = demperc_column_df[change_bool]
            covidperc_column_df = covidperc_column_df[change_bool]
            discrep_column_df = discrep_column_df[change_bool]

            training_data_dict['mortality_rate'].extend(rate_column_df.tolist())
            training_data_dict['covid_perc'].extend(covidperc_column_df.tolist())
            training_data_dict['dem_perc'].extend(demperc_column_df.tolist())
            training_data_dict['discrepancy'].extend(discrep_column_df.tolist())
            training_data_dict['time'].extend(time_df[change_bool])

            # Fill in ethnicity for the region
            ethnicity_list = [ethnicity] * len(time_df[change_bool])
            training_data_dict['ethnicity'].extend(ethnicity_list)

            # Fill in metadata for the region
            for metadata_name in aggregated_processed_metadata_df.keys():
                metadata_vals = aggregated_processed_metadata_df.loc[ethnicity, metadata_name]
                training_data_dict[metadata_name].extend([metadata_vals] * len(rate_column_df.tolist()))


    # Detrend mortality rate
    X = np.zeros((len(training_data_dict['mortality_rate']), 1))
    X[:, 0] = training_data_dict['time']
    Y = np.array(training_data_dict['mortality_rate']).reshape((-1, 1))
    kernel = C(1.0, (1e-3, 1e4)) * RBF(1.0, (1e-3, 1e4))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X, Y)
    y_pred, _ = gp.predict(X, return_std=True)
    detrended_mortality_rate = np.array(training_data_dict['mortality_rate']).reshape((-1, 1)) - y_pred
    training_data_dict['detrended_mortality_rate'] = list(detrended_mortality_rate[:, 0])
    training_data_dict['y_pred'] = list(y_pred[:, 0])

    training_data_df = pd.DataFrame(training_data_dict)

    if not os.path.exists(training_csv_path):
        os.mkdir(training_csv_path)

    training_data_df.to_csv(path.join(training_csv_path, training_file))


def regression_manager(state_name: str, type: str, validate_state_name: str, validate_county_names: List[str], ethnicity_filter_list: List[str], reg_key: str,
                       county_names: Union[str, List[str]] = None, regression_type: str = 'multilinear') -> None:
    if isinstance(county_names, str) or county_names is None:
        county_names = [county_names]
    if len(county_names) == 1:
        metadata_filter = get_metadata_filter(type=type, state_name=state_name, county_name=county_names[0], regression_type=regression_type, reg_key=reg_key,
                                              ethnicity_filter_list=ethnicity_filter_list)
    else:
        metadata_filter = []

    if regression_type in RegDefinitions.multilinear_list:
        regression_results_df, predictions_df, fitted_model, val_info_df, val_predictions_df = regression_utils.multilinear_reg(
            state_name=state_name, type=type, reg_key=reg_key, county_names=county_names, ethnicity_filter_list=ethnicity_filter_list, metadata_filter=metadata_filter,
        validate_state_name=validate_state_name, validate_county_names=validate_county_names)
    elif regression_type in RegDefinitions.multilinear_ridge_list:
        regression_results_df, predictions_df, fitted_model, val_info_df, val_predictions_df = regression_utils.multilinear_ridge_lasso_reg(
            state_name=state_name,
            type=type,
            county_names=county_names,
            reg_key=reg_key,
            regularizer_type='ridge',
            ethnicity_filter_list=ethnicity_filter_list,
            metadata_filter=metadata_filter,
            validate_state_name=validate_state_name,
            validate_county_names=validate_county_names)
    elif regression_type in RegDefinitions.multilinear_lasso_list:
        regression_results_df, predictions_df, fitted_model, val_info_df, val_predictions_df = regression_utils.multilinear_ridge_lasso_reg(
            state_name=state_name,
            type=type,
            county_names=county_names,
            reg_key=reg_key,
            regularizer_type='lasso',
            ethnicity_filter_list=ethnicity_filter_list,
            metadata_filter=metadata_filter,
            validate_state_name=validate_state_name,
            validate_county_names=validate_county_names)
    else:
        raise ValueError(f'{regression_type} regression logic not implemented')

    regression_utils.save_regression_results(
        df=regression_results_df,
        pred_df=predictions_df,
        type=type,
        state_name=state_name,
        county_names=county_names,
        ethnicity_filter_list=ethnicity_filter_list,
        regression_type=regression_type,
        reg_key=reg_key,
        validate_state_name=validate_state_name,
        validate_county_names=validate_county_names,
        val_info_df=val_info_df,
        val_predictions_df=val_predictions_df)


def add_commit_and_push(state_county_dir: str):
    try:
        logging.info("Add, commit, and push updates to raw data")
        dt = datetime.datetime.now() - datetime.timedelta(days=1)
        today = datetime.date(dt.year, dt.month, dt.day)
        today_str = today.isoformat()
        cmd.check_call(["git", "add", f"{state_county_dir}"])
        message = f"Update {state_county_dir} raw covid ethnicity data with data from {today_str}"
        cmd.check_call(["git", "commit", "-m", f"{message}"])
        cmd.check_call(["git", "push"])
    except BaseException:
        pass


def main(state_name: str, regression_type: str, corr_key: str,
         ethnicity_list: List[str], corr_type: str, reg_key: str, validate_state_name: str = 'california', validate_county_names: List[str] = [None],
         county_names: Union[str, List[str]] = None, mode: str = 'scrape'):
    # if isinstance(county_names, str) or county_names is None:
    #     county_name = county_names

    if isinstance(county_names, list):
        if len(county_names) == 1:
            county_name = county_names[0]
    else:
        county_name = county_names

    if mode == 'scrape':
        scrape_manager(state_name=state_name, county_name=county_name)
    elif mode == 'project_case':
        case_parser_manager(state_name=state_name, county_name=county_name)
    elif mode == 'project_death':
        death_parser_manager(state_name=state_name, county_name=county_name)
    elif mode == 'scrape_metadata':
        metadata_manager(state_name=state_name, county_name=county_name)
    elif mode == 'create_case_training_data':
        training_data_manager(state_name=state_name, county_name=county_name, type='cases')
    elif mode == 'create_death_training_data':
        training_data_manager(state_name=state_name, county_name=county_name, type='deaths')
    elif mode == 'perform_case_spearman_corr':
        correlation_manager(
            state_name=state_name,
            county_name=county_name,
            type='cases',
            key=corr_key,
            corr_type=corr_type,
            ethnicity_filter_list=ethnicity_list)
    elif mode == 'perform_death_spearman_corr':
        correlation_manager(
            state_name=state_name,
            county_name=county_name,
            type='deaths',
            key=corr_key,
            corr_type=corr_type,
            ethnicity_filter_list=ethnicity_list)
    elif mode == 'perform_cases_multilinear_regression':
        regression_manager(
            state_name=state_name,
            county_names=county_name,
            validate_state_name=validate_state_name,
            validate_county_names=validate_county_names,
            type='cases',
            reg_key=reg_key,
            regression_type=regression_type,
            ethnicity_filter_list=ethnicity_list)
    elif mode == 'perform_deaths_multilinear_regression':
        regression_manager(
            state_name=state_name,
            county_names=county_name,
            validate_state_name=validate_state_name,
            validate_county_names=validate_county_names,
            type='deaths',
            reg_key=reg_key,
            regression_type=regression_type,
            ethnicity_filter_list=ethnicity_list)


if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    parser = argparse.ArgumentParser(description='Process mode')
    parser.add_argument('--mode', help='Mode that will determine which managers run')
    parser.add_argument('--reg_key', default='mortality_rate', help='Key that will be regressed on')
    parser.add_argument('--regression_type', default='multilinear', help='Mode that will determine which managers run')
    parser.add_argument('--corr_type', default='spearman', help='Mode that will determine which managers run')
    parser.add_argument('--corr_key', default='mortality_rate', help='Key of quantity to be used in correlation')
    parser.add_argument('--state', help='State for which mode will be run')
    parser.add_argument('--val_state', help='State for which mode will be validated')
    parser.add_argument('--county', help='County for which model will be run', nargs='+', default=None)
    parser.add_argument('--val_county', help='County for which model will be validated', nargs='+', default=None)
    parser.add_argument('--state_bool', help='Whether state should be regressed on', action='store_false')
    parser.add_argument(
        '--all_counties_bool',
        action='store_true',
        help='Boolean that states to run mode for state and all counties in state')
    parser.add_argument('--ethnicity_list', default=[], nargs='+', help='List ethnicities to be filtered when performing correlation or doing'
                        'regressions')

    args = parser.parse_args()
    if isinstance(args.county, list):
        if len(args.county) == 1:
            main(
                mode=args.mode,
                state_name=args.state,
                county_names=args.county,
                regression_type=args.regression_type,
                reg_key=args.reg_key,
                corr_type=args.corr_type,
                corr_key=args.corr_key,
                ethnicity_list=args.ethnicity_list)
    else:
        if not args.all_counties_bool:
            main(
                mode=args.mode,
                state_name=args.state,
                county_names=args.county,
                regression_type=args.regression_type,
                reg_key=args.reg_key,
                corr_type=args.corr_type,
                corr_key=args.corr_key,
                ethnicity_list=args.ethnicity_list)
        else:
            if not args.state_bool:
                try:
                    main(
                        mode=args.mode,
                        state_name=args.state,
                        county_names=None,
                        regression_type=args.regression_type,
                        reg_key=args.reg_key,
                        corr_type=args.corr_type,
                        corr_key=args.corr_key,
                        ethnicity_list=args.ethnicity_list)
                except Exception as e:
                    print(f'Exception occured for state: {args.state} and county: {args.county}')
                    print(f'Exception is {e.args}')
                    pass
            county_list = os.listdir(path.join('states', args.state, 'counties'))
            for county in county_list:
                if county != 'kern':
                    try:
                        main(
                            mode=args.mode,
                            state_name=args.state,
                            county_names=county,
                            regression_type=args.regression_type,
                            corr_type=args.corr_type,
                            corr_key=args.corr_key,
                            reg_key=args.reg_key,
                            ethnicity_list=args.ethnicity_list)
                    except Exception as e:
                        print(f'Exception occured for state: {args.state} and county: {county}')
                        print(f'Exception is {e.args}')
                        pass
