# --------------------------
# Standard Python Imports
# --------------------------
import itertools
import logging
import os
from os import path
from typing import Dict, List, Tuple

# --------------------------
# Third Party Imports
# --------------------------
from bokeh.io import export_png, export_svgs
from bokeh.io import output_file
from bokeh.plotting import ColumnDataSource, figure, show
from bokeh.layouts import column, row
from bokeh.models import Whisker
from bokeh.palettes import Dark2_5 as palette
from bokeh.transform import factor_cmap


import pandas as pd
from scipy import stats
# --------------------------
# covid19Tracking Imports
# --------------------------


def vis_mean_ci_bar(stats_dict: Dict[str, Dict[str, List[Tuple[str, float]]]]) -> None:
    """
    Function that populates the Bokeh objects that will be used to plot relevant quantities from the stats_dict

    Arguments:
        stats_dict: Dictionary containing gaussian fits of interest in a tuple of dictionaries
                    The dictionary has form:
                        {'cases': {'alameda': {'Black': (dict_with_gp_preds 1, dict_with_gp_preds 2) ..} ....},
                         'deaths': {'alameda': ....}}
    """
    for key1 in stats_dict.keys():
        for key2 in stats_dict[key1].keys():
            # Create output file for region to plot line graphs

            #initialize layout to None
            for key3 in stats_dict[key2].keys():
                # Create figures for each ethnicity in line graph.

                # Plot real data, ideal data, fit of both along with
                # uncertainty

            # Display figure
                pass
    pass
