# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:13:42 2026

@author: lah
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import ipywidgets as widgets

COLOR = {"piezo": "#8B3A1A"}


FIGURE_NAME = "Composite Plate Demo"
LOGOFILE = "usn-logo-purple.png"


class Composite:
    def __init__(self):
        self.n_pillars = 6
        self.compliance = 0.4
        self.strain = 0.0
        self.n_points = 100

        self.fig, self.axes, self.graphs = self.initialise_graphs()
        self.update_pillars()

    @property
    def thickness(self):
        return 1 + self.strain

    @property
    def amplitude(self):
        return -self.compliance * self.strain

    @property
    def y(self):
        return np.linspace(
            -self.thickness,
            self.thickness,
            self.n_points,
        )

    @property
    def shape(self):
        return 0.5 * (1 - np.cos(pi * self.y / self.thickness))

    def initialise_graphs(self):

        fig, axes = plt.subplots(1, 1, figsize=(16, 4))

        x_left = np.empty((self.n_pillars, self.n_points))
        x_right = np.empty((self.n_pillars, self.n_points))

        offset = 0.5
        shape = self.amplitude * self.shape
        for k in range(self.n_pillars):
            x_left[k, :] = -shape + k + 1 - offset
            x_right[k, :] = shape + k + 1 + offset

        graphs = {}

        # Filler
        h = self.thickness
        graphs["filler"] = axes.fill_between(
            [0, self.n_pillars + 1],
            [-h, -h],
            [h, h],
            color="0.9",
        )

        graphs["borders"] = []
        for y_pos in (-h, h):
            graphs["borders"].append(
                axes.axhline(
                    y=y_pos,
                    color="black",
                )
            )

        # Pillars
        graphs["pillars"] = [None] * self.n_pillars

        axes.set(
            xlim=(0, self.n_pillars + 1),
            ylim=(-1.2, 1.2),
        )

        axes.set_axis_off()

        return fig, axes, graphs

    def update_borders(self):
        h = self.thickness
        for g, y in zip(self.graphs["borders"], [-h, h]):
            g.set_ydata([y, y])

    def update_pillars(self):
        offset = 0.3
        shape = -self.amplitude * self.shape
        x_left = np.empty((self.n_pillars, self.n_points))
        x_right = np.empty((self.n_pillars, self.n_points))
        for k in range(self.n_pillars):
            x_left[k, :] = -shape + k + 1 - offset
            x_right[k, :] = shape + k + 1 + offset

        # Pillars
        for k in range(self.n_pillars):
            p = self.graphs["pillars"][k]
            if p is not None:
                p.remove()

        for k in range(self.n_pillars):
            self.graphs["pillars"][k] = self.axes.fill_betweenx(
                self.y,
                x_left[k, :],
                x_right[k, :],
                color=COLOR["piezo"],
            )

    def update_plate(self):
        self.update_filler()
        self.update_borders()
        self.update_pillars()

    def update_filler(self):
        h = self.thickness

        p = self.graphs["filler"]
        if p is not None:
            p.remove()

        self.graphs["filler"] = self.axes.fill_between(
            [0, self.n_pillars + 1],
            [-h, -h],
            [h, h],
            color="0.9",
        )

    def _strain_change_callback(self, change):
        self.strain = float(change["new"])
        self.update.plate()

    # === Interactive widgets ======================================
    def _create_widgets(self):
        """
        Create widgets for interactive operation.

        Returns
        -------
        widget_layout : ipywidgets widget box
            Widget layout for use in Jupyter Notebook
        widget_list : dict of widgets
            Widgets for use in Jupyter Notebook
        """
        title = "Illustration of the Piezo-composite"
        title_widget = widgets.Label(
            title,
            style=dict(font_weight="bold"),
        )

        layout = {
            "continuous_update": True,
            "layout": widgets.Layout(width="50%"),
            "style": {"description_width": "10%"},
        }

        strain_widget = widgets.FloatSlider(
            value=0,
            min=-0.5,
            max=0.5,
            step=0.05,
            readout_format=".2f",
            description="Strain",
            **layout,
        )
        strain_widget.observe(
            self._strain_change_callback,
            names="value",
        )

        widget_layout = widgets.VBox([title_widget, strain_widget])

        widget_list = {"strain": strain_widget}

        return widget_layout, widget_list
