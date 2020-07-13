# --------------------------
# Standard Python Imports
# --------------------------
from datetime import datetime
import logging
from lxml import etree

# --------------------------
# Third Party Imports
# --------------------------
from typing import Any, Dict, List, Union


def get_element_int(element: etree.HTML) -> int:
    element = element[0].text
    element = element.replace(',', '')
    element = element.replace('\u200b', '')
    return int(element)


def get_json_element_int(raw_data_json: Dict[str, Any], ethnicity_json_keys_list: List[Union[str, int]]) -> int:
    dict_temp = raw_data_json
    for json_key in ethnicity_json_keys_list:
        dict_temp = dict_temp[json_key]

    return dict_temp

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
    valid_date_string = None
    for idx, date in enumerate(date_list[0:-1]):
        if date_format >= date_list[idx] and date_format < date_list[idx + 1]:
            valid_date_string = datetime.strftime(date_list[idx], '%Y-%m-%d')
            break
    if valid_date_string is None:
        valid_date_string = datetime.strftime(date_list[-1], '%Y-%m-%d')
    return valid_date_string


def get_total(numerical_dict: Dict[str, Union[float, str]]):
    total = 0
    for key in numerical_dict.keys():
        try:
            total = total + float(numerical_dict[key])
        except:
            import ipdb
            ipdb.set_trace()
    return total
