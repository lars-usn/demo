# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:13:42 2026

@author: lah
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

COLOR = {"piezo": "#8B3A1A"}


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


# ax.set_axis_off()
