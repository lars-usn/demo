# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:13:42 2026

@author: lah
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import ipywidgets as widgets
import usdemo

FIGURE_NAME = "Composite Plate Demo"


class Composite:
    def __init__(self, create_widgets=False):
        plt.close(FIGURE_NAME)

        self.n_pillars = 5
        self.compliance = 0.0
        self.strain = 0.0
        self.max_strain = 0.3
        self.n_points = 100
        self.deformation_profile = self._create_deformation_profile()
        self.widget_layout = None
        self.widgets = {}

        self.xspan = [0.5, self.n_pillars + 0.5]
        self.yspan = [-0.05, 1 + self.max_strain + 0.05]

        self.fig, self.axes, self.graphs = self.initialise_graphs()
        self.update_plate()

        if create_widgets:
            self.widget_layout, self.widgets = self._create_widgets()

    def _create_deformation_profile(self):
        """Define a deformation profile to be scaled."""
        n = np.linspace(-1, 1, self.n_points)
        end_value = 0.5
        center_value = 1.0
        return (
            end_value + (center_value - end_value) * (np.cos(pi * n) + 1) / 2
        )

    @property
    def thickness(self):
        """Find strained thickness of composite plate."""
        return 1 + self.strain

    @property
    def amplitude(self):
        """Find amplitude of thickness deformation."""
        return -0.5 * self.compliance * self.strain

    @property
    def y(self):
        """Find y-vector for drawing deformation curves."""
        return np.linspace(
            0,
            self.thickness,
            self.n_points,
        )

    @property
    def alpha_filler(self):
        """Define transparency of filler to indicate compliance."""
        return 1.0 - 0.6 * self.compliance

    def initialise_graphs(self):
        """Draw plate and logo, fix axes, define graphs to fill with data."""
        n_col = 4
        drawing_row = ["drawing"] * n_col

        fig, axes = plt.subplot_mosaic(
            [
                drawing_row,
                drawing_row,
                drawing_row,
                ["logo"] + ["."] * (n_col - 1),
            ],
            figsize=(12, 6),
            layout="tight",
            num=FIGURE_NAME,
        )

        usdemo.usn_logo(axes["logo"])

        # Define plate
        ax = axes["drawing"]
        graphs = {
            "filler": None,
            "pillars": [None] * self.n_pillars,
            "electrodes": [],
        }

        h = self.thickness
        graphs["electrodes"] = []
        for y_pos in (0, h):
            graphs["electrodes"].append(
                ax.axhline(
                    y=y_pos,
                    linewidth=6,
                    color=usdemo.COLORS["electrodes"],
                    zorder=3,
                )
            )

        # Fix axes scales
        ax.set(
            xlim=self.xspan,
            ylim=self.yspan,
        )

        ax.set_axis_off()

        return fig, axes, graphs

    def update_electrodes(self):
        """Update electrode positions."""
        h = self.thickness

        for electrode, y in zip(
            self.graphs["electrodes"],
            (0, h),
        ):
            electrode.set_ydata([y, y])

    def update_pillars(self):
        """Update pillar shapes."""
        half_width = 0.3
        deformation = self.amplitude * self.deformation_profile
        centers = np.arange(1, self.n_pillars + 1)[:, None]

        x_left = centers - half_width - deformation
        x_right = centers + half_width + deformation

        for pillar in self.graphs["pillars"]:
            if pillar is not None:
                pillar.remove()

        ax = self.axes["drawing"]

        y = self.y
        for k in range(self.n_pillars):
            self.graphs["pillars"][k] = ax.fill_betweenx(
                y,
                x_left[k],
                x_right[k],
                facecolor=usdemo.COLORS["piezo"],
                edgecolor=usdemo.COLORS["piezo_edge"],
                linewidth=1.0,
                zorder=2,
            )

    def update_filler(self):
        """Update filler position."""
        h = self.thickness

        filler = self.graphs["filler"]
        if filler is not None:
            filler.remove()

        self.graphs["filler"] = self.axes["drawing"].fill_between(
            self.xspan,
            [0, 0],
            [h, h],
            facecolor=usdemo.COLORS["filler"],
            edgecolor=usdemo.COLORS["filler_edge"],
            linewidth=1.0,
            alpha=self.alpha_filler,
            zorder=1,
        )

    def update_plate(self):
        """Update entire plate drawing."""
        self.update_filler()
        self.update_electrodes()
        self.update_pillars()
        self.fig.canvas.draw_idle()

    def _strain_change_callback(self, change):
        self.strain = float(change["new"])
        self.update_plate()

    def _compliance_change_callback(self, change):
        self.compliance = float(change["new"])
        self.update_plate()

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
        title = "Illustration of Piezo-Composite Plate"
        title_widget = widgets.Label(
            title,
            style=dict(font_weight="bold"),
        )

        layout = {
            "continuous_update": True,
            "layout": widgets.Layout(width="90%"),
            "style": {"description_width": "15%"},
        }

        strain_widget = widgets.FloatSlider(
            value=self.strain,
            min=-self.max_strain,
            max=self.max_strain,
            step=0.05,
            readout_format=".2f",
            description="Strain",
            **layout,
        )
        strain_widget.observe(
            self._strain_change_callback,
            names="value",
        )

        compliance_widget = widgets.FloatSlider(
            value=self.compliance,
            min=0.0,
            max=1.0,
            step=0.1,
            readout_format=".1f",
            description="Compliance",
            **layout,
        )
        compliance_widget.observe(
            self._compliance_change_callback,
            names="value",
        )

        widget_layout = widgets.VBox(
            [
                title_widget,
                strain_widget,
                compliance_widget,
            ]
        )

        widget_list = {
            "strain": strain_widget,
            "compliance": compliance_widget,
        }

        return widget_layout, widget_list
