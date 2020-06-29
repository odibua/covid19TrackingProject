# --------------------------
# Standard Python Imports
# --------------------------
import logging
# --------------------------
# Third Party Imports
# --------------------------
import yaml as yaml

# --------------------------
# covid19Tracking Imports
# --------------------------

logging.info("Open State Configuration file")
state_config_file = open('states/states_config.yaml')
state_conig = yaml.safe_load(state_config_file)

logging.info(f"Extract relevant raw data for state, if it exists")
