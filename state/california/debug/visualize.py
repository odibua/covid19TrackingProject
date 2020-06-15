# --------------------------
# Standard Python Imports
# --------------------------
import datetime
import os
import time

# --------------------------
# Third Party Imports
# --------------------------
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

# --------------------------
# covid19Tracking Imports
# --------------------------

race_ethnic_categories = ["Latino", "White", "Asian", "African American/Black", "Multi-Race", "American Indian or Alaska Native", "Native Hawaiian and other Pacific Islander", "Other", "Unknown"]
age_case_categories = ["0-17 cases", "18-34 cases", "35-49 cases", "50-64 cases", "65-79 cases", "80+ cases"]
age_death_categories = ["0-17 deaths", "18-34 deaths", "35-49 deaths", "50-64 deaths", "65-79 deaths", "80+ deaths"]

pardir = os.pardir
race_ethnicity_df = pd.read_csv(f"{pardir}/data/covid19_race_ethnicity.csv")

# Create directory to save image results
today = datetime.datetime.now()
today_str = today.isoformat()
plot_dir = os.path.join("plots", today_str)
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

# TODO(odibua@): Make this function
# Turn relevant categories to integers
for category in age_case_categories + age_death_categories:
    category_list = race_ethnicity_df[category].tolist()
    category_list = [int(val) if not isinstance(val, str) else int(val.replace(',', '')) for val in category_list]
    race_ethnicity_df[category] = category_list

# TODO(odibua@): Make this function
# Plot relevant categories
for race_ethnicity in race_ethnic_categories:
    df = race_ethnicity_df[race_ethnicity_df['race/ethnicity'] == race_ethnicity]
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 15))
    for age_case in age_case_categories:
        ax1.plot(df["date"], df[age_case], '-o')
    ax1.legend(age_case_categories)
    ax1.title.set_text(f"{race_ethnicity} Cases")

    for age_death in age_death_categories:
        ax2.plot(df["date"], df[age_death], '-o')
    ax2.legend(age_death_categories)
    ax2.title.set_text(f"{race_ethnicity} Deaths")
    race_ethnicity = race_ethnicity.replace('/', ' or ')
    fig.savefig(os.path.join(plot_dir, race_ethnicity + ".png"), format='png')

