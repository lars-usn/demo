#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:16:45 2026

@author: lars-hoff
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
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

    @property
    def angle_radius(self):
        radius_map = {
            Direction.INCOMING: 0.30,
            Direction.REFLECTED: 0.35,
            Direction.TRANSMITTED: 0.30,
        }
        return radius_map[self]

    def start_end(self, x, y):
        if self is Direction.INCOMING:
            return (x, y), (0, 0)

        elif self is Direction.REFLECTED:
            return (0, 0), (-x, y)

        elif self is Direction.TRANSMITTED:
            return (0, 0), (-x, -y)

        raise ValueError(f"Unknown direction: {self}")

    def arc_angles(self, angle):
        if self is Direction.INCOMING:
            return 90, 90 + angle

        elif self is Direction.REFLECTED:
            return 90 - angle, 90

        elif self is Direction.TRANSMITTED:
            return -90, -90 + angle

        raise ValueError(f"Unknown direction: {self}")


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

        self.fig, self.axes = self.initialise_graphs()

        self.angle_artists = []  # All artists with angles: Arcs, labels
        self.parameter_labels = []  # Labels on axis: Sound speeds
        self.widget_layout = None
        self.widgets = None

        self.update_arrows()

        if create_widgets:
            self.widget_layout, self.widgets = self._create_widgets()

    def remove_artists(self, artists):
        """Remove Matplotlib artists and empty the supplied list."""
        for artist in artists:
            artist.remove()

        artists.clear()

    @property
    def angle_transmitted(self):
        """Calculate transmitted angle from Snell's law."""
        sin_angle = self.c[1] / self.c[0] * np.sin(np.radians(self.angle_in))

        if abs(sin_angle) > 1.0 and not np.isclose(abs(sin_angle), 1.0):
            return None

        sin_angle = np.clip(sin_angle, -1.0, 1.0)
        return np.degrees(np.arcsin(sin_angle))

    @property
    def angles(self):
        return (self.angle_in, self.angle_in, self.angle_transmitted)

    @property
    def directions(self):
        return (Direction.INCOMING, Direction.REFLECTED, Direction.TRANSMITTED)

    def arrow_start_end(self, angle, direction):
        """
        Find start and end positions of arrows.

        Parameters
        ----------
        angle : float
            Arrow angle in degrees
        direction : enum Direction
            Incoming, reflected or transmitted wave

        Returns
        -------
        start : tuple (x,y)
            Start coordinates of arrow
        end : tuple (x,y)
            End coordinates of arrow
        """
        length = self.arrow_length
        phi = np.radians(angle)
        x = -length * np.sin(phi)
        y = length * np.cos(phi)

        return direction.start_end(x, y)

    def update_arrows(self):
        """Update arrows, angle markers, and annotations."""

        self.remove_artists(self.angle_artists)
        self.remove_artists(self.parameter_labels)

        for arrow, angle, direction in zip(
            self.arrows,
            self.angles,
            self.directions,
        ):

            if angle is None:
                arrow.set_visible(False)
            else:
                arrow.set_visible(True)
                start, end = self.arrow_start_end(angle, direction)
                arrow.set_positions(start, end)

                self.draw_angle(angle, direction)

        self.annotate_graph()
        self.fig.canvas.draw_idle()

    def annotate_graph(self):
        """Write parameter values on the axes."""

        ax = self.axes["drawing"]

        for k, (c, y) in enumerate(zip(self.c, (0.65, 0.35))):
            label = ax.text(
                0.05,
                y,
                rf"$c_{k+1} = {c:.0f}\, \mathrm{{m/s}}$",
                transform=ax.transAxes,
                ha="left",
                va="center",
            )
            self.parameter_labels.append(label)

        text = self.create_critical_angle_text()
        critical_label = ax.text(
            0.95,
            0.05,
            text,
            transform=ax.transAxes,
            ha="right",
            va="baseline",
        )
        self.parameter_labels.append(critical_label)

    def create_critical_angle_text(self):
        """Create a text block describing the critical angle."""

        text = "Critical angle\n" r"$\sin\theta_c = c_1/c_2$" "\n\n"

        if self.c[1] > self.c[0]:
            theta_crit = np.degrees(np.arcsin(self.c[0] / self.c[1]))

            text += rf"$\theta_c = {theta_crit:.1f}^\circ$"
        else:
            text += "No critical angle"

        if self.angle_in > 0:
            c2_critical = self.c[0] / np.sin(np.radians(self.angle_in))

            text += (
                "\n"
                rf"$c_{{2,\mathrm{{crit}}}}"
                rf" = {c2_critical:.0f}\, \mathrm{{m/s}}$"
            )

        if self.angle_transmitted is None:
            text += "\nTotal internal reflection"
        else:
            text += "\n"

        return text

    def draw_angle(self, angle, direction):
        """
        Draw an arc and its angle value.

        angle : float
            Arrow angle in degrees
        direction : enum Direction
            Incoming, reflected or transmitted wave
        """
        theta_start, theta_end = direction.arc_angles(angle)

        ax = self.axes["drawing"]
        center = (0, 0)
        radius = direction.angle_radius

        arc = Arc(
            center,
            2 * radius,
            2 * radius,
            theta1=theta_start,
            theta2=theta_end,
            color=direction.color,
            linewidth=1.0,
        )
        ax.add_patch(arc)

        theta_label = np.radians((theta_start + theta_end) / 2)

        label = ax.text(
            center[0] + 1.2 * radius * np.cos(theta_label),
            center[1] + 1.2 * radius * np.sin(theta_label),
            rf"${angle:.0f}^\circ$",
            color=direction.color,
            ha="center",
            va="center",
        )

        self.angle_artists.append(arc)
        self.angle_artists.append(label)

        return arc, label

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
            if angle is None:
                start = (0, 0)
                end = (0, 0)
                visible = False
            else:
                start, end = self.arrow_start_end(angle, direction)
                visible = True

            arrows.append(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="->",
                    color=direction.color,
                    linewidth=4,
                    mutation_scale=20,
                    zorder=5,
                    visible=visible,
                )
            )

        for arrow in arrows:
            ax.add_patch(arrow)

        # Set axis limits
        ax.set(
            xlim=(-axmax, axmax),
            ylim=(-axmax / 2, axmax / 2),
        )
        ax.set_aspect("equal", adjustable="box")

        # Draw indicator lines
        ax.axhline(y=0, color="black", linestyle="dotted", linewidth=1.0)
        ax.axvline(x=0, color="black", linestyle="dotted", linewidth=1.0)

        self.arrows = arrows

        ax.set_axis_off()

        return fig, axes

    # === Callbacks ================================================
    def _angle_change_callback(self, change):
        self.angle_in = float(change["new"])
        self.update_arrows()

    def _sound_speed_change_callback(self, change):
        index = change["owner"].medium_index
        self.c[index] = float(change["new"])
        self.update_arrows()

    # === Interactive widgets ======================================
    def _create_widgets(self):
        """
        Create widgets for interactive operation.

        Returns
        -------
        widget_layout : ipywidgets.VBox
            Widget layout for use in a Jupyter Notebook.
        widget_list : dict
            Dictionary containing the interactive widgets.
        """
        title_widget = widgets.Label(
            value="Illustration of Snell's Law",
            style={"font_weight": "bold"},
        )

        slider_options = {
            "continuous_update": True,
            "layout": widgets.Layout(width="70%"),
            "style": {"description_width": "30ch"},
        }

        angle_widget = widgets.FloatSlider(
            value=self.angle_in,
            min=0,
            max=90,
            step=1.0,
            readout_format=".0f",
            description="Angle of incoming wave [°]",
            **slider_options,
        )

        angle_widget.observe(
            self._angle_change_callback,
            names="value",
        )

        sound_speed_widgets = []

        for k in range(2):
            speed_widget = widgets.FloatSlider(
                value=self.c[k],
                min=500,
                max=3000,
                step=10,
                readout_format=".0f",
                description=f"Speed of sound in medium {k + 1} [m/s]",
                **slider_options,
            )
            speed_widget.medium_index = k
            speed_widget.observe(
                self._sound_speed_change_callback,
                names="value",
            )

            sound_speed_widgets.append(speed_widget)

        widget_layout = widgets.VBox(
            [
                title_widget,
                angle_widget,
                *sound_speed_widgets,
            ]
        )

        widget_list = {
            "angle": angle_widget,
            "speed_1": sound_speed_widgets[0],
            "speed_2": sound_speed_widgets[1],
        }

        return widget_layout, widget_list
