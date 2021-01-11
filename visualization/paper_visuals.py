# --------------------------
# Standard Python Imports
# --------------------------
import os

# --------------------------
# Third Party Imports
# --------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List

# --------------------------
# covid19Tracking Imports
# --------------------------

IMAGE_DIR = 'paper_images'
def calc_weighted_mean(arr: np.array, weights: np.array) -> float:
    weighted_arr = np.multiply(arr, weights)
    return np.sum(weighted_arr) / np.sum(weights)


def calc_weighted_std(arr: np.array, weights: np.array, M: int) -> float:
    mean = calc_weighted_mean(arr, weights)
    numerator = np.multiply(weights, (arr - mean) ** 2)
    denominator = ((M - 1) / M) * weights

    std = np.sqrt(sum(numerator) / sum(denominator))
    if std == float("inf"):
        import ipdb
        ipdb.set_trace()
    return std

def get_unique_elements_in_order(list_: List[str], exclude_list: List[str] = []):
    unique_list = []
    for item in list_:
        if item not in unique_list and item not in exclude_list:
            unique_list.append(item)
    return unique_list


def plot_discrep_mortality(state_name: str, type_: str = 'deaths'):
    csv_file_name = f'states/{state_name}/training_data_csvs/{state_name}_training_{type_}.csv'
    data_df = pd.read_csv(csv_file_name)

    symbols = ['o', 'X', '^', 'D']
    ethnicities = ['Black', 'Asian', 'Hispanic', 'White'] #set(data_df['ethnicity'].tolist())
    discrep_list = []
    mortality_list = []
    time_list = []

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    mrk_size = 40
    for idx, ethnicity in enumerate(ethnicities):
        temp_df = data_df[data_df['ethnicity'] == ethnicity]
        time = np.array(pd.to_datetime(temp_df['date']).tolist())
        discrepancy = np.array(temp_df['discrepancy'].tolist())
        mortality = np.array(temp_df['mortality_rate'].tolist())
        ax1.scatter(time, discrepancy, marker=symbols[idx], s=mrk_size)
        ax2.scatter(time, mortality, marker=symbols[idx], s=mrk_size)

    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel('Discrepancy', fontsize=24)
    ax2.set_ylabel('Mortality Rate (per 1000)', fontsize=22)
    ax1.legend(ethnicities, loc='upper left', fontsize=16)
    ax2.legend(ethnicities, loc='upper left', fontsize=16)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    # plt.show()
    plt.gcf()
    plt.savefig(f'{IMAGE_DIR}/timeseries_{type_}_discrepancy.png', format='png')


def plot_correlations(state_name: str, corr_key: str = 'discrepancy', type_: str = 'deaths', delta=0.15):
    discrep_spearman_csv_name = f'states/{state_name}/correlation_results/spearman/{type_}_{corr_key}_spearman_corr_results_black_white_asian_hispanic.csv'
    discrep_distance_corr_csv_name = f'states/{state_name}/correlation_results/distance_corr/{type_}_{corr_key}_distance_corr_corr_results_black_white_asian_hispanic.csv'

    metadata_map = {'HOUSEHOLD_MEDIAN_INCOME_DOLLARS': 'Median Income',
                     'HIGH_SCHOOL_GRADUATE_OR_HIGHER_25_PLUS_TOTAL': 'High School Graduate',
                     'BACHELOR_DEGREE_OR_HIGHER_25_PLUS_TOTAL': 'Bachelor Degree',
                     'PUBLIC_TRANSPORTATION': 'Public Transportation',
                      'CAR_TRUCK_VAN_ALONE': 'Car/Truck Alone', 'CAR_TRUCK_VAN_CARPOOL': 'Car/Truck Carpool'}
    spearman_df = pd.read_csv(discrep_spearman_csv_name)
    distance_corr_df = pd.read_csv(discrep_distance_corr_csv_name)

    metadata_list = spearman_df['X'].tolist()
    unique_metadata_list = []
    for metadata in metadata_list:
        if metadata_map[metadata] not in unique_metadata_list:
            unique_metadata_list.append(metadata_map[metadata])

    county_list = spearman_df[spearman_df['county'] != 'imperial']['county']
    unique_county_list = []
    for county in county_list:
        if county not in unique_county_list:
            unique_county_list.append(county)

    spearman_corr_list = []
    distance_corr_list = []
    patterns = ["|", ".", "*", "x", "o", "O", "+"]
    for county_name in unique_county_list:
        temp_spearman_df = spearman_df[spearman_df['county'] == f'{county_name}']
        temp_distance_corr_df = distance_corr_df[distance_corr_df ['county'] == f'{county_name}']

        spearman_corr = temp_spearman_df['corr'].tolist()
        spearman_corr = [corr for corr in spearman_corr]
        distance_corr = temp_distance_corr_df['corr'].tolist()

        spearman_corr_list.append(spearman_corr)
        distance_corr_list.append(distance_corr)

    spearman_corr = np.vstack(spearman_corr_list).T
    distance_corr = np.vstack(distance_corr_list).T
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)
    ax1.set_ylim((-1.1, 1.1))
    ax2.set_ylim((0, 1.1))
    ind = np.arange(len(unique_metadata_list))

    upper_bound = [0.4] * 2
    lower_bound = [-0.4] * 2
    x_bound = [-0.15, 38 * 0.15]
    for idx in range(len(unique_county_list)):
        rects = ax1.bar(ind + delta * idx, spearman_corr[:, idx], width=delta, hatch=patterns[idx])
        rects2 = ax2.bar(ind + delta * idx, distance_corr[:, idx], width=delta, hatch=patterns[idx])

        for rect, rect2 in zip(rects, rects2):
            height, height2 = rect.get_height(), rect2.get_height()
            if height > 0:
                loc = 1.05 * height
            else:
                loc = max(1.2 * height, -1.1)
            ax1.text(rect.get_x() + rect.get_width() / 2., loc,
                    f'{round(height, 2)}',
                    ha='center', va='bottom')
            ax2.text(rect2.get_x() + rect2.get_width() / 2., 1.05 * height2,
                    f'{round(height2, 2)}',
                    ha='center', va='bottom')

    ax1.set_ylabel('Spearman Correlation (Linear)', fontsize=16)
    ax2.set_ylabel('Distance Correlation (Non-Linear)', fontsize=16)

    ax2.plot(x_bound, upper_bound, '--')
    ax1.get_xaxis().set_visible(False)
    ax2.set_xticks(ind + delta + delta/2.0)
    ax2.set_xticklabels(unique_metadata_list, fontsize=14)
    ax1.legend(unique_county_list, loc='upper left', fontsize=14)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    ax1.plot(x_bound, upper_bound, '--')
    ax1.plot(x_bound, lower_bound, '--')
    plt.gcf()
    plt.savefig(f'{IMAGE_DIR}/corr_{type_}_{corr_key}.png', format='png')


def plot_nrmse(state_name: str, reg_key: str = 'discrepancy', train_counties: List[str] = ['None'], val_counties: List[str] = ['None'], test_counties: List[str] = ['None'],
               reg_type: str = 'multilinear_lasso', type_: str = 'deaths', ml_mode: str = 'train', delta: float = 0.15):
    title_map = {'discrepancy': 'Discrepancy', 'mortality_rate': 'Mortality Rate (per 1000)'}
    if ml_mode == 'train':
        overall_csv = f'states/california/regression_results_csvs/{reg_type}/{type_}_{reg_key}_{reg_type}_results_Black_White_Asian_Hispanic.csv'
        overall_df = pd.read_csv(overall_csv)
        county_list = overall_df['county'].tolist()
        unique_county_list = get_unique_elements_in_order(county_list, ['imperial'])
    elif ml_mode == 'test':
        if type_ == 'deaths':
            unique_county_list = ["sacramento", "alameda", "losangeles", "sanfrancisco", "santaclara"]
        elif type_ == 'cases':
            unique_county_list = ["sonoma", "riverside", "sacramento", "alameda", "losangeles", "sanfrancisco", "santaclara"]
        test_dir = f'states/california/test_results_csvs/{reg_type}/{reg_key}'
        file_name = f'{type_}_train_california_{"_".join(train_counties)}_val_california_{"_".join(val_counties)}_test_california_{"_".join(unique_county_list)}_{reg_key}_Black_White_Asian_Hispanic.csv'
        file_df = pd.read_csv(f'{test_dir}/{file_name}')
    ethnicity_list = ['Black', 'White', 'Asian', 'Hispanic']
    nrmse_list, std_nrmse_list = [], []
    for county in unique_county_list:
        if ml_mode == 'train':
            file = f'states/california/regression_results_csvs/{reg_type}/{reg_key}/{type_}_california_{county}_{reg_key}_Black_White_Asian_Hispanic.csv'
            file_df = pd.read_csv(file)
        ethnicity_nrmse_list, ethnicity_std_nrmse_list = [], []
        for ethnicity in ethnicity_list:
            # Get nrmse of relevant ethnicity
            temp_file_df = file_df[file_df['ethnicity'] == ethnicity]
            temp_file_df = temp_file_df[temp_file_df['county'] == county]
            if len(temp_file_df) == 0:
                temp_file_df = file_df[file_df['ethnicity'] == ethnicity.lower()]
            if ml_mode == 'train':
                temp_nrmse_list = abs((temp_file_df['y'] - temp_file_df['y_pred']) / temp_file_df['y']).tolist()
            elif ml_mode == 'test':
                temp_nrmse_list = abs((temp_file_df['y_test'] - temp_file_df['y_test_pred']) / temp_file_df['y_test']).tolist()
                # temp_nrmse_list = abs((temp_file_df['y_test'] - temp_file_df['y_test_pred']) / temp_file_df['y_test']).tolist()

            # Calculate weight by time
            time = np.array(temp_file_df['time'].tolist())
            weight_arr = np.exp((time - time.max()) * 5e-2)
            M = sum(weight_arr > 1e-7)

            # Calculate weighed mean and standard deviation
            weighted_nrmse_mean = calc_weighted_mean(np.array(temp_nrmse_list), weight_arr)
            weighted_std_nrmse_mean = calc_weighted_std(np.array(temp_nrmse_list), weight_arr, M)
            ethnicity_nrmse_list.append(weighted_nrmse_mean)
            ethnicity_std_nrmse_list.append(weighted_std_nrmse_mean)

        nrmse_list.append(ethnicity_nrmse_list)
        std_nrmse_list.append(ethnicity_std_nrmse_list)

    nrmse_arr = np.vstack(nrmse_list)
    std_nrmse_arr = np.vstack(std_nrmse_list)
    ax1 = plt.subplot(111)
    ind = np.arange(len(unique_county_list))

    upper_bound = [0.4] * 2
    lower_bound = [-0.4] * 2
    x_bound = [-0.15, 38 * 0.15]
    patterns = ["|", ".", "*", "x", "o", "O", "+"]
    # import ipdb
    # ipdb.set_trace()
    for idx in range(len(ethnicity_list)):
        rects = ax1.bar(ind + delta * idx, nrmse_arr[:, idx], yerr=std_nrmse_arr[:, idx], width=delta, hatch=patterns[idx])

        for rect in rects:
            height = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    f'{round(height, 2)}',
                    ha='center', va='bottom')

    ax1.set_ylabel('NRMSE', fontsize=16)
    ax1.set_xticks(ind + delta + delta/2.0)
    ax1.set_xticklabels(unique_county_list, fontsize=14)
    ax1.set_title(f'{title_map[reg_key]}', fontsize=20)
    ax1.set_ylim((0, 1.0))


    ax1.legend(ethnicity_list, loc='upper left', fontsize=14)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.gcf()
    plt.savefig(f'{IMAGE_DIR}/nrmse_{type_}_{ml_mode}_{reg_type}_{reg_key}.png', format='png')
    # plt.show()



def manager(mode: str) -> None:
    if mode == 'plot_cali_stuff':
        plot_discrep_mortality(state_name='california')
    elif mode == 'plot_correlations':
        plot_correlations(state_name='california')
    elif mode == 'plot_train_nrmse':
        plot_nrmse(state_name='california')


if __name__ == "__main__":
    if not os.path.isdir(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    mode = 'plot_train_nrmse'
    # mode = 'plot_correlations'
    # mode = 'plot_cali_stuff'

    manager(mode)
