#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:16:45 2026

@author: lars-hoff
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import ipywidgets as widgets
from enum import Enum
import usdemo


class Direction(Enum):
    INCOMING = 0
    REFLECTED = 1
    TRANSMITTED = 2

    @property
    def color(self) -> str:
        color_map = {
            Direction.INCOMING: "#000080",
            Direction.REFLECTED: "#800000",
            Direction.TRANSMITTED: "#004C00",
        }
        return color_map[self]


COLORS = {
    "medium_0": "#CCFFFF",
    "medium_1": "#FFFFCC",
}

FIGURE_NAME = "Reflection - Refraction Demo"


class Reflection:
    def __init__(self, create_widgets=False):
        plt.close(FIGURE_NAME)

        self.arrow_length = 1.0
        self.angle_in = 30
        self.c = [1500, 2000]

        self.fig, self.axes, self.graphs = self.initialise_graphs()
        self.fig.canvas.draw_idle()

        # if create_widgets:
        #     self.widget_layout, self.widgets = self._create_widgets()

    @property
    def angle_transmitted(self):
        return self.c[1] / self.c[0] * self.angle_in

    @property
    def angles(self):
        return (self.angle_in, self.angle_in, self.angle_transmitted)

    @property
    def directions(self):
        return (Direction.INCOMING, Direction.REFLECTED, Direction.TRANSMITTED)

    def arrow_start_end(self, angle, direction):
        length = self.arrow_length
        phi = np.radians(angle)
        x = -length * np.sin(phi)
        y = length * np.cos(phi)

        print(angle)

        if direction == Direction.INCOMING:
            x_start = x
            y_start = y
            x_end = 0
            y_end = 0
        if direction == Direction.REFLECTED:
            x_start = 0
            y_start = 0
            x_end = -x
            y_end = y
        if direction == Direction.TRANSMITTED:
            x_start = 0
            y_start = 0
            x_end = -x
            y_end = -y

        return (x_start, y_start), (x_end, y_end)

    def update_arrows(self):
        """Update arrow drawings with new angles."""

        for arrow, angle, direction in zip(
            self.arrows, self.angles, self.directions
        ):
            start, end = self.arrow_start_end(angle, direction)
            arrow.set_positions(start, end)

        self.annotate_graph()

    def annotate_graph(self):
        for c, y in zip(self.c, (0.5, -0.5)):
            self.axes["drawing"].text(1.0, y, f"$c$ = {c} m/s")

    def initialise_graphs(self):
        """Draw interface and arrows."""
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

        ax = axes["drawing"]
        axmax = 2

        ax.axhspan(
            0,
            axmax,
            facecolor=COLORS["medium_0"],
            alpha=0.5,
            transform=ax.get_yaxis_transform(),
            zorder=0,
        )
        ax.axhspan(
            -axmax,
            0,
            facecolor=COLORS["medium_1"],
            alpha=0.5,
            transform=ax.get_yaxis_transform(),
            zorder=0,
        )

        # Draw arrows
        arrows = []

        for angle, direction in zip(self.angles, self.directions):
            start, end = self.arrow_start_end(angle, direction)
            arrows.append(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="->",
                    color=direction.color,
                    linewidth=4,
                    mutation_scale=20,
                    zorder=5,
                )
            )

        for arrow in arrows:
            ax.add_patch(arrow)

        ax.set(
            xlim=(-axmax, axmax),
            ylim=(-axmax / 2, axmax / 2),
        )
        ax.set_aspect("equal", adjustable="box")

        ax.axhline(y=0, color="black", linewidth=1.0)
        ax.axvline(x=0, color="black", linewidth=1.0)

        graphs = {}
        self.arrows = arrows

        # ax.set_axis_off()

        return fig, axes, graphs
