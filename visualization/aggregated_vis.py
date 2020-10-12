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


def vis_mean_ci_bar_helper(fig_list: List[figure], source_list: List[ColumnDataSource], plot_key: str, state: str) -> None:
    """
    Add bar and line graphs to figure and display the plot

    Arguments:
        fig_list: List of bokeh figures
        source_list: List of sources that will be used to populate values in figure plots
        plot_key: Key that dictates which quantitiy is being plotted
        state: State for which plotting occurs
    """
    idx, layout = 0, None
    output_file(f'{plot_key}_test_agg.html')
    for fig, source_ in zip(fig_list, source_list):
        n = len(source_.data['vals'])
        fair_list = [1.0 for _ in range(n)]
        identifiers = source_.data['identifiers']
        state_idx = identifiers.index(state)
        fig.vbar(x='identifiers', top='vals', source=source_, width=0.9, line_color='white', fill_alpha=.5,
             fill_color='salmon',
             line_alpha=.5,)
        if state_idx is not None:
            fig.line(x=identifiers, color='red', y=source_.data['vals'][state_idx], width=3, legend_label=state.upper())
        fig.line(x=identifiers, color='blue', y=fair_list, width=3, legend_label='No Disparity'.upper())
        fig.add_layout(
            Whisker(source=source_, base="identifiers", upper="uppers", lower="lowers")
        )
        fig.xgrid.grid_line_color = None

        if idx == 0:
            layout = row(fig)
        else:
            layout = column(fig, layout)
        idx = idx + 1

    show(layout)


def vis_mean_ci_bar(stats_dict: Dict[str, Dict[str, List[Tuple[str, float]]]], plot_key: str, std_plot_key: str, state: str, alpha: float = 0.05) -> None:
    """
    Function that populates the Bokeh objects that will be used to plot relevant quantities from the stats_dict

    Arguments:
        stats_dict: Dictionary containing statistics of interest
        plot_key: String that dictates what statistical quantity is plotted in the bar graphs
        std_plot_key: String that dictates the standard deviation of the quantity to be plotted
        state: State for which statisitcs have been calculated
        alpha: Confidence interval
    """

    z_score = stats.norm.ppf(1 - 0.5 * alpha)
    figure_list, source_list = [], []
    for key in stats_dict.keys():
        identifier_list, val_list = list(zip(*stats_dict[key][plot_key]))
        _, std_list = list(zip(*stats_dict[key][std_plot_key]))
        lower_list = [val - z_score * std for val, std in zip(val_list, std_list)]
        upper_list = [val + z_score * std for val, std in zip(val_list, std_list)]

        source = ColumnDataSource(data=dict(identifiers=identifier_list, vals=val_list, stds=std_list,
                                            lowers=lower_list, uppers=upper_list))
        fig = figure(x_range=identifier_list, plot_height=350, plot_width=1000, toolbar_location=None, title=key.upper(),
                     y_range=(0, max(upper_list) + 0.1 * max(upper_list)))

        source_list.append(source)
        figure_list.append(fig)

    vis_mean_ci_bar_helper(fig_list=figure_list, source_list=source_list, plot_key=plot_key, state=state)

