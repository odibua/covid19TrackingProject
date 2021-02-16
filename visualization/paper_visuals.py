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
    ethnicities = ['Black', 'Asian', 'Hispanic', 'White']  # set(data_df['ethnicity'].tolist())
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

    plt.show()
    # plt.gcf()
    # plt.savefig(f'{IMAGE_DIR}/timeseries_{type_}_discrepancy.png', format='png')


def plot_correlations(state_name: str, corr_key: str = 'mortality_rate', type_: str = 'deaths', delta=0.15):
    discrep_spearman_csv_name = f'states/{state_name}/correlation_results/individual_counties/spearman/{type_}_{corr_key}_spearman_corr_results_black_white_asian_hispanic.csv'

    all_discrep_spearman_csv_name = f'states/{state_name}/correlation_results/california/spearman/{type_}_{corr_key}_spearman_corr_results_black_white_asian_hispanic.csv'


    metadata_map = {'HOUSEHOLD_MEDIAN_INCOME_DOLLARS': 'Median Income',
                    'HIGH_SCHOOL_GRADUATE_OR_HIGHER_25_PLUS_TOTAL': 'High School Graduate',
                    'BACHELOR_DEGREE_OR_HIGHER_25_PLUS_TOTAL': 'Bachelor Degree',
                    'PUBLIC_TRANSPORTATION': 'Public Transportation',
                    'CAR_TRUCK_VAN_ALONE': 'Car/Truck Alone', 'CAR_TRUCK_VAN_CARPOOL': 'Car/Truck Carpool'}
    spearman_df = pd.read_csv(discrep_spearman_csv_name)

    all_spearman_df = pd.read_csv(all_discrep_spearman_csv_name)

    all_spearman_df['county'] = 'california'

    spearman_df = pd.concat([spearman_df, all_spearman_df])

    spearman_df = spearman_df[[True if elem in metadata_map.keys() else False for elem in spearman_df['X'].tolist()]]

    metadata_list = spearman_df['X'].tolist()
    unique_metadata_list = []
    for metadata in metadata_list:
        if metadata in metadata_map.keys():
            if metadata_map[metadata] not in unique_metadata_list:
                unique_metadata_list.append(metadata_map[metadata])

    county_list = spearman_df[spearman_df['county'] != 'imperial']['county']
    unique_county_list = []
    for county in county_list:
        if county not in unique_county_list:
            unique_county_list.append(county)

    spearman_corr_list, upper_spearman_list, lower_spearman_list = [], [], []
    patterns = ["|", ".", "*", "x", "o", "O", "+"]
    for county_name in unique_county_list:
        temp_spearman_df = spearman_df[spearman_df['county'] == f'{county_name}']

        spearman_corr = temp_spearman_df['corr'].tolist()
        low_spearman = temp_spearman_df['low_corr'].tolist()
        up_spearman = temp_spearman_df['up_corr'].tolist()

        spearman_corr = [corr for corr in spearman_corr]


        spearman_corr_list.append(spearman_corr)
        lower_spearman_list.append(low_spearman)
        upper_spearman_list.append(up_spearman)


    spearman_corr = np.vstack(spearman_corr_list).T
    lower_spearman_corr = np.vstack(lower_spearman_list).T
    upper_spearman_corr = np.vstack(upper_spearman_list).T

    ax1 = plt.subplot(111)
    ax1.set_ylim((-1.1, 1.1))
    ind = np.arange(len(unique_metadata_list))

    upper_bound = [0.4] * 2
    lower_bound = [-0.4] * 2
    x_bound = [-0.15, 39 * 0.15]
    for idx in range(len(unique_county_list)):
        ax1.bar(ind + delta * idx, spearman_corr[:, idx], hatch=patterns[idx], width=delta, edgecolor='black', linewidth=2.0)
    for idx in range(len(unique_county_list)):
        low_err = np.abs(lower_spearman_corr[:, idx] - spearman_corr[:, idx])
        up_err = np.abs(upper_spearman_corr[:, idx] - spearman_corr[:, idx])
        ax1.errorbar(ind + delta * idx, spearman_corr[:, idx], np.array([low_err, up_err]), ecolor='black', fmt='o')

    ax1.set_ylabel('Spearman Correlation (Linear)', fontsize=16)
    ax1.set_xticks(ind + delta + delta / 2.0)
    ax1.set_xticklabels(unique_metadata_list, fontsize=14)
    ax1.legend(unique_county_list, loc='upper left', fontsize=14, handlelength=5, handleheight=3)


    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    ax1.plot(x_bound, upper_bound, 'r--', linewidth=3.0)
    ax1.plot(x_bound, lower_bound, 'r--', linewidth=3.0)
    plt.gcf()
    plt.xticks(rotation=0)


    plt.show()
    # plt.savefig(f'{IMAGE_DIR}/corr_{type_}_{corr_key}.png', format='png')


def plot_coefs(state_name: str, reg_key: str = 'mortality_rate', type_: str = 'deaths', delta=0.15):
    multilinear_csv_name = f'states/{state_name}/regression_results_csvs/individual_counties/multilinear/{type_}_{reg_key}_multilinear_results_Black_White_Asian_Hispanic.csv'
    rf_reg_csv_name = f'states/{state_name}/regression_results_csvs/individual_counties/rf_reg/{type_}_{reg_key}_rf_reg_results_Black_White_Asian_Hispanic.csv'

    all_multilinear_csv_name = f'states/{state_name}/regression_results_csvs/california/multilinear/{type_}_{reg_key}_multilinear_results_Black_White_Asian_Hispanic.csv'
    all_rf_reg_csv_name = f'states/{state_name}/regression_results_csvs/california/rf_reg/{type_}_{reg_key}_rf_reg_results_Black_White_Asian_Hispanic.csv'


    metadata_map = {'HOUSEHOLD_MEDIAN_INCOME_DOLLARS': 'Median Income',
                    'HIGH_SCHOOL_GRADUATE_OR_HIGHER_25_PLUS_TOTAL': 'High School Graduate',
                    'BACHELOR_DEGREE_OR_HIGHER_25_PLUS_TOTAL': 'Bachelor Degree',
                    'PUBLIC_TRANSPORTATION': 'Public Transportation',
                    'CAR_TRUCK_VAN_ALONE': 'Car/Truck Alone', 'CAR_TRUCK_VAN_CARPOOL': 'Car/Truck Carpool',
                    'BELOW_POVERTY_LEVEL': 'Poverty Level', 'PERCENT_INSURED': 'Percent Insured'}

    multilinear_df = pd.read_csv(multilinear_csv_name)
    rf_reg_df = pd.read_csv(rf_reg_csv_name)

    all_multilinear_df = pd.read_csv(all_multilinear_csv_name)
    all_rf_reg_df = pd.read_csv(all_rf_reg_csv_name)

    all_multilinear_df['county'] = 'california'
    all_rf_reg_df['county'] = 'california'

    multilinear_df = pd.concat([multilinear_df, all_multilinear_df])
    rf_reg_df = pd.concat([rf_reg_df, all_rf_reg_df])

    multilinear_df_all = multilinear_df[[True if elem not in ['time', 'constant'] else False for elem in multilinear_df['features'].tolist()]]
    rf_reg_df_all = rf_reg_df[[True if elem not in ['time', 'constant'] else False for elem in rf_reg_df['features'].tolist()]]

    multilinear_df = multilinear_df[[True if elem in metadata_map.keys() else False for elem in multilinear_df['features'].tolist()]]
    rf_reg_df = rf_reg_df[[True if elem in metadata_map.keys() else False for elem in rf_reg_df['features'].tolist()]]

    metadata_list = multilinear_df['features'].tolist()
    unique_metadata_list = []
    unique_idx_list = []

    for idx, metadata in enumerate(metadata_list):
        if metadata in metadata_map.keys():
            if metadata_map[metadata] not in unique_metadata_list:
                unique_metadata_list.append(metadata_map[metadata])
                unique_idx_list.append(idx)

    county_list = multilinear_df[multilinear_df['county'] != 'imperial']['county']
    unique_county_list = []
    for county in county_list:
        if county not in unique_county_list:
            unique_county_list.append(county)

    multi_coef_list, upper_multi_coef_list, lower_multi_coef_list = [], [], []
    rf_coef_list, upper_rf_coef_list, lower_rf_coef_list = [], [], []
    patterns = ["|", ".", "*", "x", "o", "O", "+"]
    for county_name in unique_county_list:
        temp_multilinear_df_all = multilinear_df_all[multilinear_df_all['county'] == f'{county_name}']
        temp_rf_reg_df_all = rf_reg_df_all[rf_reg_df_all['county'] == f'{county_name}']

        temp_multilinear_df = multilinear_df[multilinear_df['county'] == f'{county_name}']
        temp_rf_reg_df = rf_reg_df[rf_reg_df['county'] == f'{county_name}']

        max_multi_coef = temp_multilinear_df_all['coef'].max()
        max_rf_coef = temp_rf_reg_df_all['coef'].max()

        multi_coef = (temp_multilinear_df['coef']/max_multi_coef).tolist()
        lower_multi_coef = (temp_multilinear_df['lower_coef']/max_multi_coef).tolist()
        upper_multi_coef = (temp_multilinear_df['upper_coef']/max_multi_coef).tolist()

        rf_reg_coef = (temp_rf_reg_df['coef']/max_rf_coef).tolist()
        lower_rf_reg = (temp_rf_reg_df['low_coef']/max_rf_coef).tolist()
        upper_rf_reg = (temp_rf_reg_df['up_coef']/max_rf_coef).tolist()

        multi_coef_list.append(multi_coef)
        rf_coef_list.append(rf_reg_coef)
        lower_multi_coef_list.append(lower_multi_coef)
        upper_multi_coef_list.append(upper_multi_coef)
        lower_rf_coef_list.append(lower_rf_reg)
        upper_rf_coef_list.append(upper_rf_reg)


    multi_coef = np.vstack(multi_coef_list).T
    rf_reg_coef = np.vstack(rf_coef_list).T
    lower_multi_coef = np.vstack(lower_multi_coef_list).T
    upper_multi_coef = np.vstack(upper_multi_coef_list).T
    lower_rf_reg_coef = np.vstack(lower_rf_coef_list).T
    upper_rf_reg_coef = np.vstack(upper_rf_coef_list).T
    # import ipdb
    # ipdb.set_trace()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)
    ind = np.arange(len(unique_metadata_list))

    upper_bound = [0.4] * 2
    lower_bound = [-0.4] * 2
    x_bound = [-0.15, 52 * 0.15]
    for idx in range(len(unique_county_list)):
        # if max(upper_spearman_corr[:, idx]) > 0.8:
        #     import ipdb
        #     ipdb.set_trace()
        ax1.bar(ind + delta * idx, multi_coef[:, idx], hatch=patterns[idx], width=delta, edgecolor='black', linewidth=2.0)
        # ax1.bar(ind + delta * idx, multi_coef[:, idx], width=delta, hatch=patterns[1], color='m', edgecolor='black')
        # ax1.bar(ind + delta * idx, lower_multi_coef[:, idx], width=delta, color='b', edgecolor='black', linewidth=2.0)

        ax2.bar(ind + delta * idx, rf_reg_coef[:, idx], hatch=patterns[idx], width=delta, edgecolor='black', linewidth=2.0)
        # ax2.bar(ind + delta * idx, rf_reg_coef[:, idx], width=delta, hatch=patterns[1], color='m', edgecolor='black')
        # # ax2.bar(ind + delta * idx, lower_rf_reg_coef[:, idx], width=delta, color='b', edgecolor='black', linewidth=2.0)
        # for rect, rect2 in zip(rects, rects2):
        #     height, height2 = rect.get_height(), rect2.get_height()
        #     loc = 1.05 * height
        #     ax1.text(rect.get_x() + rect.get_width() / 2., loc,
        #              unique_county_list[idx], rotation='vertical',
        #              ha='center', va='bottom')
        #     ax2.text(rect2.get_x() + rect2.get_width() / 2., 1.05 * height2,
        #              unique_county_list[idx],
        #              ha='center', va='bottom', rotation='vertical')
            # ax1.text(rect.get_x() + rect.get_width() / 2., loc,
            #          f'{round(height, 2)}',
            #          ha='center', va='bottom')
            # ax2.text(rect2.get_x() + rect2.get_width() / 2., 1.05 * height2,
            #          f'{round(height2, 2)}',
            #          ha='center', va='bottom')
    # ax1.set_ylim((0, 1.0))
    for idx in range(len(unique_county_list)):
        low_err = np.abs(lower_multi_coef[:, idx] - multi_coef[:, idx])
        up_err = np.abs(upper_multi_coef[:, idx] - multi_coef[:, idx])
        ax1.errorbar(ind + delta * idx, multi_coef[:, idx], np.array([low_err, up_err]), ecolor='black', fmt='o')
        low_err = np.abs(lower_rf_reg_coef[:, idx] - rf_reg_coef[:, idx])
        up_err = np.abs(upper_rf_reg_coef[:, idx] - rf_reg_coef[:, idx])
        ax2.errorbar(ind + delta * idx, rf_reg_coef[:, idx], np.array([low_err, up_err]), ecolor='black', fmt='o')

    ax1.set_ylabel('Multi-linear Coefficients', fontsize=16)
    ax2.set_ylabel('Random Forest Importance', fontsize=16)

    # ax2.plot(x_bound, upper_bound, 'r--', linewidth=3.0)
    ax1.get_xaxis().set_visible(False)
    ax2.set_xticks(ind + delta + delta / 2.0)
    ax2.set_xticklabels(unique_metadata_list, fontsize=14)
    # ax1.legend(unique_county_list, loc='upper left', fontsize=14)
    ax1.legend(unique_county_list, loc='upper left', fontsize=14, handlelength=5, handleheight=3)


    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    ax1.plot(x_bound, upper_bound, 'r--', linewidth=3.0)
    ax2.plot(x_bound, upper_bound, 'r--', linewidth=3.0)

    # ax1.plot(x_bound, lower_bound, 'r--')
    plt.gcf()
    plt.xticks(rotation=10)

    plt.show()


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
            unique_county_list = [
                "sonoma",
                "riverside",
                "sacramento",
                "alameda",
                "losangeles",
                "sanfrancisco",
                "santaclara"]
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
                temp_nrmse_list = abs(
                    (temp_file_df['y_test'] -
                     temp_file_df['y_test_pred']) /
                    temp_file_df['y_test']).tolist()
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
        rects = ax1.bar(ind + delta * idx, nrmse_arr[:, idx],
                        yerr=std_nrmse_arr[:, idx], width=delta, hatch=patterns[idx])

        for rect in rects:
            height = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                     f'{round(height, 2)}',
                     ha='center', va='bottom')

    ax1.set_ylabel('NRMSE', fontsize=16)
    ax1.set_xticks(ind + delta + delta / 2.0)
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
    elif mode == 'plot_coefs':
        plot_coefs(state_name='california')


if __name__ == "__main__":
    if not os.path.isdir(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    # mode = 'plot_train_nrmse'
    # mode = 'plot_correlations'
    # mode = 'plot_cali_stuff'
    mode = 'plot_coefs'

    manager(mode)
