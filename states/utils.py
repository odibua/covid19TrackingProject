# --------------------------
# Standard Python Imports
# --------------------------
from datetime import datetime
import logging
from lxml import etree
import os

# --------------------------
# Third Party Imports
# --------------------------
import pandas as pd
from typing import Dict, List
import yaml as yaml


def get_valid_date_string(date_list: List[datetime], date_string: str) -> str:
    """
    Get the date in date list that is greater than one element of date_list and less than the next

    Arguments:
        date_list: Sorted list of dates
        date_string: String date to which date list should be compared

    Returns:
        valid_date_string
    """
    logging.info(f"Get correct html parsing based on {date_string}")
    date_format = datetime.strptime(date_string, '%Y-%m-%d')
    for idx, date in enumerate(date_list[0:-1]):
        if date_format >= date_list[idx] and date < date_list[idx + 1]:
            valid_date_string = datetime.strftime(date_list[idx], '%Y-%m-%d')
            break
    return valid_date_string
