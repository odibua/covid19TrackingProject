# --------------------------
# Standard Python Imports
# --------------------------

# --------------------------
# Third Party Imports
# --------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------
# covid19Tracking Imports
# --------------------------


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

    plt.show()


def plot_correlations(state_name: str, county_name: str = 'santaclara', type_: str = 'deaths'):
    discrep_spearman_csv_name = f'states/{state_name}/correlation_results/spearman/{type_}_discrepancy_spearman_corr_results_black_white_asian_hispanic.csv'
    discrep_distance_corr_csv_name = f'states/{state_name}/correlation_results/distance_corr/{type_}_discrepancy_distance_corr_corr_results_black_white_asian_hispanic.csv'

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
    patterns = ["|", ".", "*", "x", "o", "O"]
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
    delta = 0.15
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


    ax1.plot(x_bound, upper_bound, '--')
    ax1.plot(x_bound, lower_bound, '--')

    ax1.set_ylabel('Spearman Correlation (Linear)', fontsize=16)
    ax2.set_ylabel('Distance Correlation (Non-Linear)', fontsize=16)

    ax2.plot(x_bound, upper_bound, '--')
    ax1.get_xaxis().set_visible(False)
    ax2.set_xticks(ind + delta + delta/2.0)
    ax2.set_xticklabels(unique_metadata_list, fontsize=14)
    ax1.legend(unique_county_list, loc='upper left', fontsize=14)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.show()


def manager(mode: str) -> None:
    if mode == 'plot_cali_stuff':
        plot_discrep_mortality(state_name='california')
    elif mode == 'plot_correlations':
        plot_correlations(state_name='california')


if __name__ == "__main__":
    mode = 'plot_correlations'
    manager(mode)
