# --------------------------
# Standard Python Imports
# --------------------------
import datetime
import json
import os
import time

# --------------------------
# Third Party Imports
# --------------------------
import pandas as pd
import requests
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------

santaclara_raw_data_dir = 'states/california/counties/santaclara/raw_data'
santaclara_config_file = 'states/california/counties/santaclara/configs/santaclara_timeseries_deaths.yaml'

import ipdb
ipdb.set_trace()
raw_data_dates = os.listdir(santaclara_raw_data_dir)
config_file_obj = open(santaclara_config_file)
response_config = yaml.safe_load(config_file_obj)
url = response_config['REQUEST']['URL']
request_type = response_config['REQUEST']['TYPE']

headers = response_config['REQUEST']['HEADERS']
payload = response_config['REQUEST']['PAYLOAD']
response = requests.post(url=url, headers=headers, json=json.loads(json.dumps(payload)))
status_code = response.status_code
response_text = response.text
response_dict = json.loads(response_text)
death_list = response_dict['results'][0]['result']['data']['dsr']['DS'][0]['PH'][0]['DM0']

results_dict = {'date': [], 'deaths': []}
for death in death_list:
    if 'C' in death.keys():
        num_deaths = death['C'][-1]
        time_cs = death['C'][0]
        date = time.strftime('%Y-%m-%d', time.localtime(time_cs * 0.001))
        date = datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=1)
        results_dict['date'].append(date)
        results_dict['deaths'].append(num_deaths)
results_df = pd.DataFrame(results_dict)
import ipdb
ipdb.set_trace()
results_df.to_csv('states/california/counties/santaclara/death_over_time.csv')


