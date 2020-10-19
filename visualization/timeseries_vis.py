# --------------------------
# Standard Python Imports
# --------------------------
import itertools
import logging
import os
from os import path
from typing import Dict, Tuple, Union

# --------------------------
# Third Party Imports
# --------------------------
from bokeh.io import export_png, export_svgs
from bokeh.io import output_file
from bokeh.plotting import ColumnDataSource, figure, show
from bokeh.layouts import column, row
from bokeh.models import Band, Whisker
from bokeh.palettes import Dark2_5 as palette
from bokeh.transform import factor_cmap

import numpy as np
import pandas as pd
from scipy import stats
# --------------------------
# covid19Tracking Imports
# --------------------------


def get_quants_for_plotting(dict: Dict[str, Union[float, np.ndarray]]) -> ColumnDataSource:
    """
    Gets quantities that will be used to form a ColumnDataSource from a dictionary

    Arguments:
        dict: Dictionary that contains quantities of interest to be plotted

    Returns:
        source: Columnar data source
    """
    source = ColumnDataSource({'x': dict['x'], 'y': dict['y'], 'y_pred': dict['mn_y_pred'], 'sigma': dict['mn_sigma'],
                                 'lower': dict['mn_y_pred'] - dict['mn_sigma'], 'upper': dict['mn_y_pred'] + dict['mn_sigma']})
    return source


def vis_mean_ci_bar(stats_dict: Dict[str, Dict[str, Dict[str, Tuple[Dict[str, float]]]]]) -> None:
    """
    # TODO(odibua@): Refactor to make more compact and require an input that just takes care of plotting
    # and not reading a key
    Function that populates the Bokeh objects that will be used to plot relevant quantities from the stats_dict

    Arguments:
        stats_dict: Dictionary containing gaussian fits of interest in a tuple of dictionaries
                    The dictionary has form:
                        {'cases': {'alameda': {'Black': (dict_with_gp_preds 1, dict_with_gp_preds 2) ..} ....},
                         'deaths': {'alameda': ....}}
    """
    color_list = ['red', 'green']
    for key1 in stats_dict.keys():
        for key2 in stats_dict[key1].keys():
            # Create output file for region to plot line graphs
            output_file(f'{key1}_{key2}_gps.html')
            layout, source_dict, idx = None, {}, 0
            for key3 in stats_dict[key1][key2].keys():
                source_real = get_quants_for_plotting(dict=stats_dict[key1][key2][key3][0])
                source_ideal = get_quants_for_plotting(dict=stats_dict[key1][key2][key3][1])

                # Create figures for each ethnicity in line graph.
                fig = figure(title=key3.upper())

                # Plot real data, ideal data, fit of both along with
                # uncertainty
                idx_col = 0
                legend_labels = [('Real Count', 'Predicted Real Count'), ('Unbiased Count', 'Predicted Unbiased Count')]
                for legend_label, source in zip(legend_labels, [source_real, source_ideal]):
                    fig.circle(x='x', y='y', color=color_list[idx_col], source=source_real, legend_label=legend_label[0])

                    fig.circle(x='x', y='y_pred', color=color_list[idx_col], source=source_ideal, legend_label=legend_label[1])
                    band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay',
                                fill_alpha=1.0, line_width=1, line_color=color_list[idx_col])
                    fig.add_layout(band)
                    idx_col = idx_col + 1
                if idx == 0:
                    layout = row(fig)
                    idx = idx + 1
                else:
                    layout = column(fig, layout)
            show(layout)
